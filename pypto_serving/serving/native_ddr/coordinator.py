# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-writer prefix bookkeeping composed with the existing KV allocator.

No HBM pointers, second allocator, device import, or synchronous worker lookup.
All leases below are actual references in the supplied KvCacheManager.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging

from .types import (
    DdrControlRequest, DdrEntryRef, DdrTransferEvent, DdrTransferRequest, KvPageLease, LoadState,
    Operation, PrefixKey, RestoreDecision, RestoreKind, TransferStatus,
)
from .replay import DeliverySender

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduledPrefix:
    request_id: str
    page_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    prompt_tokens: int
    computed_before: int
    scheduled_tokens: int


@dataclass
class _Job:
    request: DdrTransferRequest
    lease: KvPageLease
    state: LoadState = LoadState.RESERVED
    waiters: set[tuple[str, object]] = field(default_factory=set)
    discard: bool = False
    submitted: bool = False


@dataclass
class _Waiter:
    request: object
    jobs: set[str]
    local_ids: tuple[int, ...]


class DdrPrefixCoordinator:
    def __init__(self, manager, *, namespace_digest: bytes, worker_epoch: str, max_entries: int):
        if manager.has_groups or manager.block_size != 128 or not manager.enable_prefix_cache:
            raise ValueError("DDR coordinator requires existing single-pool page128 prefix cache")
        if len(namespace_digest) != 32 or not worker_epoch or max_entries <= 0:
            raise ValueError("explicit compatibility namespace, epoch and bounded capacity required")
        self.manager = manager
        self.namespace = namespace_digest
        self.worker_epoch = worker_epoch
        self.cache_epoch = worker_epoch
        self.max_entries = max_entries
        self.ready: dict[PrefixKey, DdrEntryRef] = {}
        self.loads_by_key: dict[tuple[str, PrefixKey], str] = {}
        self.stores_by_key: dict[PrefixKey, str] = {}
        self.jobs: dict[str, _Job] = {}
        self.waiters: dict[tuple[str, object], _Waiter] = {}
        self.steps: dict[int, tuple[KvPageLease, tuple[ScheduledPrefix, ...]]] = {}
        self.outbound: deque[str] = deque()
        self.active_job: str | None = None
        self._sequence = 0
        self.closed = False
        self.poisoned = False
        self.controls = {}
        self.control_outbound = deque()
        self.transfer_delivery = DeliverySender()
        self.control_delivery = DeliverySender()
        self.retiring = {}
        self.stopped = False
        self.drained = False
        self.trace = deque(maxlen=512)
        self.diagnostic_results = {}
        self._trace_sequence = 0

    def record(self, event, **fields):
        import time
        self._trace_sequence += 1
        item = {"event": event, "sequence": self._trace_sequence, "time_ns": time.monotonic_ns(), **fields}
        self.trace.append(item)
        logger.info("native_ddr_coordinator %s", json.dumps(item, sort_keys=True))

    def _control(self, operation, *, job_id="", entry_ref=None, page_ids=(), diagnostic_label="", fault=""):
        command = DdrControlRequest(self._id(operation), self.worker_epoch, operation, job_id, entry_ref,
                                    tuple(page_ids), diagnostic_label, fault)
        self.controls[command.command_id] = command
        self.control_outbound.append(command.command_id)
        return command

    def next_control(self):
        if not self.control_delivery.available:
            return None
        while self.control_outbound:
            command = self.controls.get(self.control_outbound.popleft())
            if command is not None:
                command = self.control_delivery.stamp(command)
                self.controls[command.command_id] = command
                return command
        return None

    def evict(self, key):
        ref = self.ready.pop(key, None)
        if ref is None:
            return False
        self.retiring[ref.entry_id] = ref
        self._control("evict", entry_ref=ref)
        self.record("evict_lookup_withdrawn", entry=ref.entry_id)
        return True

    def on_control_event(self, event):
        command = self.controls.get(event.command_id)
        if command is None:
            return
        if event.delivery_sequence != command.delivery_sequence or command.delivery_sequence <= 0:
            self.poisoned = True
            raise RuntimeError("native DDR control delivery identity mismatch")
        if command.operation in ("diagnostic", "fault"):
            if event.worker_epoch != command.worker_epoch or event.operation != command.operation:
                raise RuntimeError("diagnostic receipt identity mismatch")
            self.diagnostic_results[event.command_id] = (event.error, event.payload)
            self.control_delivery.consume(event.delivery_sequence)
            del self.controls[event.command_id]
            return
        if ((event.worker_epoch, event.operation, event.entry_ref) !=
                (command.worker_epoch, command.operation, command.entry_ref)
                or event.error or (command.operation in ("evict", "drain") and not event.released)):
            self.poisoned = True
            raise RuntimeError("native DDR control has no matching release receipt")
        if command.operation == "evict":
            self.retiring.pop(command.entry_ref.entry_id, None)
            self.record("worker_slot_released", entry=command.entry_ref.entry_id)
        if command.operation == "drain":
            self.drained = True
            self.record("module_drained_before_executor_close")
        self.control_delivery.consume(event.delivery_sequence)
        del self.controls[event.command_id]

    @staticmethod
    def _request_key(request):
        return (request.request_id, getattr(request, "request_epoch", id(request)))

    def _id(self, prefix):
        self._sequence += 1
        return f"{self.worker_epoch}:{prefix}:{self._sequence}"

    def _lease(self, ids, hashes, purpose, *, already_owned=False):
        ids = tuple(ids)
        if not already_owned:
            self.manager.retain_block_ids(ids)
        return KvPageLease(self._id("lease"), self.cache_epoch, ids, tuple(hashes), purpose)

    def _release(self, lease):
        self.manager.release_blocks_by_ids(list(lease.block_ids))

    def lookup_after_hbm(self, request, local_blocks, max_hit_tokens) -> RestoreDecision:
        if self.closed or self.poisoned:
            return RestoreDecision(RestoreKind.MISS)
        req_key = self._request_key(request)
        if req_key in self.waiters:
            return RestoreDecision(RestoreKind.WAIT, tuple(self.waiters[req_key].jobs))
        hashes = self.manager.compute_block_hashes(request.prompt_token_ids)
        limit = min(max_hit_tokens, max(0, len(request.prompt_token_ids) - 1)) // 128
        keys = []
        for block_hash in hashes[len(local_blocks):limit]:
            key = PrefixKey(self.namespace, block_hash)
            if key not in self.ready and (self.worker_epoch, key) not in self.loads_by_key:
                break
            keys.append(key)
        if not keys:
            self.record("hbm_or_cold", request=request.request_id, hbm_blocks=len(local_blocks))
            return RestoreDecision(RestoreKind.READY if len(local_blocks) >= limit and limit else RestoreKind.MISS,
                                   reuse_limit=len(local_blocks) * 128)
        return self.reserve_restore(request, keys, local_blocks)

    def reserve_restore(self, request, keys, local_blocks) -> RestoreDecision:
        req_key = self._request_key(request)
        fresh = [key for key in keys if (self.worker_epoch, key) not in self.loads_by_key]
        # This is only the minimum one-token suffix headroom. The scheduler
        # checks the actual chunk allocation and can relinquish completed
        # restore pins before retrying; no private suffix/free-page pool exists.
        if self.manager.num_free_blocks < len(fresh) + 1 or len(self.jobs) + len(fresh) > self.max_entries:
            self.record("restore_reservation_no_headroom", request=request.request_id,
                        free_pages=self.manager.num_free_blocks, fresh_blocks=len(fresh), jobs=len(self.jobs))
            return RestoreDecision(RestoreKind.MISS)
        blocks = self.manager.allocate_blocks(len(fresh))
        if blocks is None:
            return RestoreDecision(RestoreKind.MISS)
        fresh_blocks = iter(blocks)
        job_ids = set()
        for key in keys:
            index_key = (self.worker_epoch, key)
            job_id = self.loads_by_key.get(index_key)
            if job_id is None:
                block = next(fresh_blocks)
                lease = self._lease((block.block_id,), (key.block_hash,), "LOAD", already_owned=True)
                job_id = self._id("load")
                transfer = DdrTransferRequest(job_id, self.worker_epoch, Operation.LOAD, (key,),
                                              (self.ready[key],), lease.lease_id, lease.block_ids, None)
                self.jobs[job_id] = _Job(transfer, lease)
                self.loads_by_key[index_key] = job_id
                self.outbound.appendleft(job_id)
            self.jobs[job_id].waiters.add(req_key)
            job_ids.add(job_id)
        # Transfer ownership of the initial lookup refs only on WAIT.
        self.waiters[req_key] = _Waiter(request, job_ids, tuple(b.block_id for b in local_blocks))
        request.scheduling_held = True
        self.record("restore_held", request=request.request_id, jobs=sorted(job_ids),
                    local_pages=list(self.waiters[req_key].local_ids))
        # All canonical targets may already be committed behind a prefix gap.
        # Such a late waiter must not need another event to become runnable.
        self._wake_waiters()
        return RestoreDecision(RestoreKind.WAIT, tuple(sorted(job_ids)))

    def next_transfer(self) -> DdrTransferRequest | None:
        if self.active_job or self.poisoned or self.closed or not self.transfer_delivery.available:
            return None
        while self.outbound:
            job_id = self.outbound.popleft()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            if (job.request.operation == Operation.STORE
                    and len(self.ready) + len(self.retiring) + len(self.stores_by_key) > self.max_entries):
                self.outbound.append(job_id)
                return None
            job.submitted = True
            # LOAD priority can reorder creation IDs. Stamp only the actual
            # outgoing command and retain this immutable stamp for any retry.
            job.request = self.transfer_delivery.stamp(job.request)
            job.state = LoadState.LOADING
            self.active_job = job_id
            self.record("transfer_dispatched", job=job_id, operation=job.request.operation.value,
                        pages=list(job.request.page_ids), delivery_sequence=job.request.delivery_sequence)
            return job.request
        return None

    def retain_step(self, step_id, scheduled_snapshot) -> KvPageLease:
        if step_id in self.steps:
            raise ValueError("duplicate retained step")
        snapshot = tuple(scheduled_snapshot)
        ids = tuple(dict.fromkeys(p for item in snapshot for p in item.page_ids))
        lease = self._lease(ids, (), "STEP")
        self.steps[step_id] = (lease, snapshot)
        return lease

    def on_confirmed_step(self, step_id, scheduled_snapshot, confirmed_tokens):
        step = self.steps.get(step_id)
        if step is None or step[1] != tuple(scheduled_snapshot):
            raise ValueError("confirmation must match the retained immutable snapshot")
        lease, snapshot = step
        try:
            if self.closed or self.poisoned:
                return
            for item in snapshot:
                confirmed = confirmed_tokens.get(item.request_id, 0)
                limit = min(confirmed, item.prompt_tokens,
                            item.computed_before + item.scheduled_tokens) // 128
                # Promote only blocks newly sealed by this confirmed step.
                # A later suffix/decode confirmation must not blindly retry a
                # cancelled/failed STORE of an older immutable prefix.
                start = min(limit, item.computed_before // 128)
                for page, block_hash in zip(item.page_ids[start:limit], item.block_hashes[start:limit]):
                    key = PrefixKey(self.namespace, block_hash)
                    if key in self.ready or key in self.stores_by_key:
                        continue
                    if len(self.stores_by_key) + len(self.ready) >= self.max_entries:
                        victim = next((k for k in self.ready if (self.worker_epoch, k) not in self.loads_by_key), None)
                        if victim is None or not self.evict(victim):
                            continue
                    # get_cached_block references the canonical hash. It must
                    # refer to THIS snapshot page, not a later duplicate page.
                    block = self.manager.get_cached_block(block_hash)
                    if block is None:
                        continue
                    if block.block_id != page:
                        self.manager.release_cached_blocks([block])
                        continue
                    pin = self._lease((page,), (block_hash,), "STORE", already_owned=True)
                    job_id = self._id("store")
                    ref = DdrEntryRef(self.worker_epoch, self._id("entry"), self._sequence)
                    transfer = DdrTransferRequest(job_id, self.worker_epoch, Operation.STORE, (key,),
                                                  (ref,), pin.lease_id, pin.block_ids, step_id)
                    self.jobs[job_id] = _Job(transfer, pin)
                    self.stores_by_key[key] = job_id
                    self.outbound.append(job_id)
                    self.record("store_source_pinned", job=job_id, page=page, step_id=step_id)
        finally:
            del self.steps[step_id]
            self._release(lease)

    def discard_step(self, step_id, *, stopped=False):
        if step_id not in self.steps:
            return
        if not stopped:
            self.poisoned = True
            return
        self._release(self.steps.pop(step_id)[0])

    def on_transfer_event(self, event: DdrTransferEvent):
        job = self.jobs.get(event.job_id)
        if job is None:
            return  # duplicate completed/unknown job owns no local obligation
        req = job.request
        if not job.submitted:
            self.poisoned = True
            return
        if job.state in (LoadState.COMMITTED, LoadState.DISCARD_PENDING):
            return  # terminal replay while waiters still own the loader pin
        if (event.worker_epoch, event.operation, event.entry_refs, event.delivery_sequence) != (
                req.worker_epoch, req.operation, req.entry_refs, req.delivery_sequence):
            self.poisoned = True
            job.state = LoadState.QUARANTINED
            return
        if event.status == TransferStatus.ACCEPTED:
            self.record("transfer_accepted", job=req.job_id)
            return
        self.record("transfer_terminal", job=req.job_id, status=event.status.value,
                    native_completed=event.native_completed, stopped=event.stopped,
                    bytes=event.transferred_bytes)
        if not event.stopped:
            job.state = LoadState.QUARANTINED
            self.poisoned = True
            return
        self.transfer_delivery.consume(event.delivery_sequence)
        success = (event.status == TransferStatus.SUCCEEDED and event.native_completed
                   and event.transferred_bytes == 20971520 and not job.discard)
        key = req.entry_keys[0]
        # Match and settle the exact old job first. Epoch changes must not leak
        # its pin, nor authorize publication into a replacement cache epoch.
        publish = success and req.worker_epoch == self.worker_epoch and not self.closed and not self.poisoned
        if req.operation == Operation.STORE:
            if publish:
                self.ready[key] = event.entry_refs[0]
                self.record("ddr_ready", job=req.job_id, entry=event.entry_refs[0].entry_id)
            elif not self.stopped:
                self.retiring[req.entry_refs[0].entry_id] = req.entry_refs[0]
                self._control("evict", entry_ref=req.entry_refs[0])
            self.stores_by_key.pop(key, None)
            self._release(job.lease)
            self.record("store_source_released", job=req.job_id, page=req.page_ids[0])
        else:
            if publish:
                job.state = LoadState.NATIVE_COMPLETED
                self.manager.cache_block_ids(list(req.page_ids), [key.block_hash], 0, 1)
                job.state = LoadState.COMMITTED
                self.record("restored_page_valid", job=req.job_id, page=req.page_ids[0])
            else:
                job.state = LoadState.DISCARD_PENDING
                if not success and self.ready.get(key) == req.entry_refs[0]:
                    self.ready.pop(key, None)
                self.loads_by_key.pop((req.worker_epoch, key), None)
                if not self.stopped and not success:
                    self.retiring[req.entry_refs[0].entry_id] = req.entry_refs[0]
                    self._control("evict", entry_ref=req.entry_refs[0])
            # A committed target stays canonical across gaps in ordinary HBM
            # lookup until its LAST waiter has acquired a ref or left waiting.
        if self.active_job == req.job_id:
            self.active_job = None
        if req.operation == Operation.STORE:
            del self.jobs[req.job_id]
        self._wake_waiters()
        if req.operation == Operation.LOAD and not job.waiters:
            self._finish_load(req.job_id)

    def _wake_waiters(self):
        for waiter in self.waiters.values():
            if all(self.jobs[j].state in (LoadState.COMMITTED, LoadState.DISCARD_PENDING)
                   for j in waiter.jobs):
                waiter.request.scheduling_held = False

    def resume_request(self, request):
        """Release initial lookup refs before the scheduler redoes ordinary lookup.

        Caller must invoke after_lookup in the same single-writer critical
        section, so the loader pins protect every newly acquired request ref.
        """
        key = self._request_key(request)
        waiter = self.waiters.get(key)
        if waiter is None or request.scheduling_held:
            return
        self.manager.release_blocks_by_ids(list(waiter.local_ids))
        waiter.local_ids = ()
        request.cached_block_ids = []
        request.num_computed_tokens = 0
        request.num_blocks_cached = 0

    def after_lookup(self, request):
        req_key = self._request_key(request)
        waiter = self.waiters.pop(req_key, None)
        if waiter is None:
            return
        for job_id in waiter.jobs:
            job = self.jobs[job_id]
            job.waiters.discard(req_key)
            if not job.waiters and job.state in (LoadState.COMMITTED, LoadState.DISCARD_PENDING):
                self._finish_load(job_id)

    def _finish_load(self, job_id):
        job = self.jobs.pop(job_id)
        index_key = (job.request.worker_epoch, job.request.entry_keys[0])
        if self.loads_by_key.get(index_key) == job_id:
            del self.loads_by_key[index_key]
        self._release(job.lease)
        self.record("load_target_lease_released", job=job_id, pages=list(job.lease.block_ids))
        job.state = LoadState.RELEASED

    def release_completed_restores(self):
        """Relinquish unconsumed restore refs on actual suffix admission failure.

        Only fully settled waiters can be abandoned. Their valid pages remain
        ordinary evictable cache entries; queued/loading/quarantined targets,
        shared pins still needed by held waiters, step and STORE refs survive.
        Requests stay WAITING and may use normal lookup/cold prefill later.
        """
        before = self.manager.num_free_blocks
        self.record("suffix_headroom_pressure", free_pages=before, waiters=len(self.waiters))
        for req_key, waiter in tuple(self.waiters.items()):
            if all(self.jobs[j].state in (LoadState.COMMITTED, LoadState.DISCARD_PENDING)
                   for j in waiter.jobs):
                self.cancel_request(*req_key)
        return self.manager.num_free_blocks > before

    def cancel_request(self, request_id, request_epoch):
        req_key = (request_id, request_epoch)
        waiter = self.waiters.pop(req_key, None)
        if waiter is None:
            return
        self.manager.release_blocks_by_ids(list(waiter.local_ids))
        waiter.request.cached_block_ids = []
        waiter.request.num_computed_tokens = 0
        waiter.request.num_blocks_cached = 0
        waiter.request.scheduling_held = False
        for job_id in waiter.jobs:
            job = self.jobs[job_id]
            job.waiters.discard(req_key)
            if job.waiters:
                continue
            if job.state in (LoadState.COMMITTED, LoadState.DISCARD_PENDING):
                self._finish_load(job_id)
            elif not job.submitted:
                self.loads_by_key.pop((job.request.worker_epoch, job.request.entry_keys[0]), None)
                self._finish_load(job_id)
            else:
                job.discard = True
                self._control("cancel", job_id=job_id)
                # Native-owned until an actual stopped event; a second cancel
                # must not confuse an in-flight loader with a settled discard.

    def close(self):
        self.closed = True
        for request_id, epoch in tuple(self.waiters):
            self.cancel_request(request_id, epoch)
        for job_id, job in tuple(self.jobs.items()):
            job.discard = True
            if not job.submitted:
                self._release(job.lease)
                self.stores_by_key.pop(job.request.entry_keys[0], None)
                self.loads_by_key.pop((job.request.worker_epoch, job.request.entry_keys[0]), None)
                del self.jobs[job_id]
            else:
                self._control("cancel", job_id=job_id)
        for key in tuple(self.ready):
            self.evict(key)
        if self.jobs or self.steps or self.retiring or self.controls:
            raise RuntimeError("DDR close incomplete: native/step obligations retained")

    def on_worker_stopped(self, worker_epoch, proof):
        # A process reap boolean is deliberately insufficient. Only an owner
        # receipt for native work AND argument cleanup may discharge obligations.
        from .worker import NativeStopProof
        if (not isinstance(proof, NativeStopProof)
                or proof.worker_epoch != worker_epoch or not proof.native_drained
                or not proof.arguments_released):
            raise RuntimeError("native stop proof unavailable; pages remain quarantined")
        if worker_epoch != self.worker_epoch:
            return
        self.stopped = True
        for job in tuple(self.jobs.values()):
            req = job.request
            if req.worker_epoch != worker_epoch:
                continue
            self.on_transfer_event(DdrTransferEvent(req.job_id, req.worker_epoch, req.operation,
                TransferStatus.CANCELLED, req.entry_refs, 0, False, True, "owner stopped", req.delivery_sequence))
        for step_id in tuple(self.steps):
            if self.steps[step_id][0].cache_epoch == worker_epoch:
                self.discard_step(step_id, stopped=True)
        self.ready.clear()
        self.retiring.clear()
        self.controls.clear()
        self.control_outbound.clear()
        self.record("owner_drained_arguments_released")
