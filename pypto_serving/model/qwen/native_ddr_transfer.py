# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded same-owner native transfer adapter; no scheduler or prefix lookup policy."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
import uuid
from typing import Any, Callable

from pypto_serving.serving.native_ddr.types import Operation, QwenKvLayout
from pypto_serving.serving.native_ddr.worker import NativeCompletion, NativeSubmissionError
from pypto_serving.serving.native_ddr.managed_pool import (
    BLOCK_BYTES, GUARD_BYTES, CommittedRange, ManagedPoolAccess, ManagedTransferReceipt, PoolAck, prefix_content_id,
)


@dataclass(eq=False)
class DdrSlot:
    slot: int
    generation: int
    tensor: Any = field(repr=False)
    owner: Any = field(repr=False)
    byte_offset: int
    block: Any = field(repr=False)
    committed: CommittedRange | None = None
    active_lease: Any = field(default=None, repr=False)
    import_ack: PoolAck | None = None


class ManagedDdrSlots:
    """Non-owning views of the one UFlow resource, registered before fork.

    The only serving allocation here is the four-byte page-ID control tensor.
    Logical slots hold UFlow allocation metadata; they do not allocate KV bytes.
    """

    def __init__(self, count: int, access: ManagedPoolAccess):
        if type(count) is not int or count <= 0:
            raise ValueError("native DDR requires an explicit finite positive prefork slot count")
        import numpy as np
        import torch

        if not isinstance(access, ManagedPoolAccess) or count * (BLOCK_BYTES + GUARD_BYTES) > access.identity.nbytes:
            raise ValueError('finite prefix access budget does not fit the actual UFlow pool')
        self.capacity = count
        self.access = access
        # NumPy's owner keeps a real buffer export for the storage's entire
        # lifetime, including detached/dtype/slice aliases after logical release.
        # torch.frombuffer alone retains an object but permits mmap.close while
        # such aliases still live. Both conversions below are zero-copy views.
        self.pool_tensor = torch.from_numpy(np.frombuffer(access.buffer, dtype=np.uint8,
                                                        count=access.identity.nbytes))
        if self.pool_tensor.untyped_storage().nbytes() != access.identity.nbytes:
            raise ValueError('whole pool must have one exact non-copying storage span')
        self.page_ids = torch.empty((1,), dtype=torch.int32).share_memory_()
        self.page_ids.zero_()
        self.leases: dict[int, DdrSlot] = {}
        self.busy: DdrSlot | None = None
        self.generation = 0
        self._blocks = {}
        self._offsets = {}
        self.control_metadata_bytes = 4
        self.control_metadata_mapped_bytes = None
        if os.name == 'posix':
            from pypto_serving.serving.native_ddr.managed_process import shared_mapping_size
            self.control_metadata_mapped_bytes = shared_mapping_size(self.page_ids.data_ptr(), 4)
        self.child_baseline = {}
        self.child_registrations = {}

    @property
    def tensors(self):
        return tuple(slot.tensor for slot in self.leases.values())

    @property
    def inherited_host_tensors(self):
        return (self.pool_tensor, self.page_ids)

    def _view(self, offset):
        import torch
        return self.pool_tensor.narrow(0, offset, BLOCK_BYTES).view(torch.bfloat16).reshape(81920, 128)

    def reserve(self, slot, size):
        if type(slot) is not int or not 0 <= slot < self.capacity or size != BLOCK_BYTES:
            raise ValueError("invalid whole-block DDR slot")
        if slot in self.leases:
            raise ValueError("DDR slot already leased")
        self.generation += 1
        block = self._blocks.get(slot)
        if block is None:
            block = self.access.allocate()
            self._blocks[slot] = block
            self._offsets[slot] = block.byte_offset + GUARD_BYTES
        offset = self._offsets[slot]
        lease = DdrSlot(slot, self.generation, self._view(offset), self, offset, block)
        self.leases[slot] = lease
        return lease

    def import_committed(self, slot, committed, *, content_id):
        committed.validate(self.access.identity, content_id=content_id)
        if type(slot) is not int or not 0 <= slot < self.capacity or slot in self.leases:
            raise ValueError('import needs a free logical slot')
        self.generation += 1
        backing = DdrSlot(slot, self.generation, self._view(committed.byte_offset), self,
                          committed.byte_offset, committed.block)
        self.leases[slot] = backing
        # This is an explicit control-plane import, not synthetic local STORE
        # completion. LOAD later obtains and verifies a fresh reader again.
        try:
            lease = self.access.acquire(committed.block, 'r', committed=committed)
        except Exception:
            if not self.access.unknown:
                # An explicit pre-grant rejection owns no native obligation.
                del self.leases[slot]
                backing.tensor = None
            raise
        backing.active_lease = lease
        backing.import_ack = self.access.complete_read(lease, uuid.uuid4().hex)
        backing.active_lease = None
        backing.committed = committed
        self._blocks[slot] = committed.block
        self._offsets[slot] = committed.byte_offset
        return backing

    def begin_native(self, request, backing):
        self.validate(backing)
        if backing.active_lease is not None:
            raise RuntimeError('slot already owns a native range lease')
        if request.operation == Operation.LOAD:
            if backing.committed is None:
                raise ValueError('LOAD requires a current UFlow Committed entry')
            backing.committed.validate(self.access.identity, content_id=prefix_content_id(request.entry_keys[0]))
            lease = self.access.acquire(backing.block, 'r', committed=backing.committed)
        else:
            lease = self.access.acquire(backing.block, 'w', byte_offset=backing.byte_offset)
            backing.committed = None
        if lease.byte_offset != backing.byte_offset:
            self.access.report_unknown(lease, 'native tensor and lease offset disagree')
            raise RuntimeError('native range offset mismatch')
        backing.active_lease = lease

    def finish_native(self, request, backing, operation_id):
        lease = backing.active_lease
        if lease is None:
            raise RuntimeError('native completion lost its retained UFlow lease')
        if request.operation == Operation.STORE:
            committed = self.access.complete_write(lease, prefix_content_id(request.entry_keys[0]), operation_id)
            backing.committed = committed
            ack = None
        else:
            committed = backing.committed
            ack = self.access.complete_read(lease, operation_id)
        receipt = ManagedTransferReceipt(request.operation.value, self.access.identity, lease.lease_id,
            lease.lease_generation, committed, request.page_lease_id, ack)
        receipt.validate(pool=self.access.identity, operation=request.operation.value,
                         page_lease_id=request.page_lease_id, content_id=prefix_content_id(request.entry_keys[0]))
        backing.active_lease = None
        return receipt

    def validate(self, lease):
        if (not isinstance(lease, DdrSlot) or lease.owner is not self
                or self.leases.get(lease.slot) is not lease):
            raise ValueError("stale or foreign DDR allocation")

    def release(self, lease):
        self.validate(lease)
        if self.busy is lease or lease.active_lease is not None:
            raise RuntimeError("pending native transfer retains DDR backing")
        del self.leases[lease.slot]
        lease.tensor = None

    def close_after_owner(self):
        if self.busy is not None or self.leases or self.access.leases or self.access.children or self.access.unknown:
            raise RuntimeError('native/client mapping obligations must drain before unregister')
        self.pool_tensor = None
        self.page_ids = None
        self.access.close()


class NativeDdrRun:
    """Adapt the real DistributedRunHandle property/result into the typed worker protocol."""

    def __init__(self, backend, request, slot, native):
        self.backend = backend
        self.request = request
        self.slot = slot
        self.native = native
        self._result = None
        self._pool_operation_id = uuid.uuid4().hex
        self._pool_receipt = None
        self._native_drained = False
        self._owner_stopped = False
        self._lock = threading.Lock()
        # Explicit strong references survive even if callers drop their copies.
        self._keepalive = (backend.worker, backend.keys, backend.values, slot, backend.slots.page_ids, request)

    def done(self):
        return self._native_drained or self.native.done

    def _drain_native(self):
        if not self._native_drained:
            try:
                self.native.result(timeout=60.0)
            except BaseException as error:
                self.backend.slots.access.report_unknown(self.slot.active_lease,
                                                         f'native result/cleanup outcome unknown: {error}')
                raise
            self._native_drained = True
            self.backend.record('native_result_and_cleanup', job=self.request.job_id)

    def drain_native(self):
        """Physical fence only; never publishes data or discharges a UF lease."""
        with self._lock:
            self._drain_native()

    def release_stopped_owner(self):
        """Called only after actual owner close and inherited-child exit proof."""
        with self._lock:
            if not self._native_drained:
                raise RuntimeError('native arguments have no completion/cleanup proof')
            self.native = None
            self._owner_stopped = True
            # Keep the unresolved host range/view and request, without retaining
            # the stopped device owner or HBM tensor arguments indefinitely.
            self._keepalive = (self.slot, self.request)

    def result(self):
        with self._lock:
            if self._result is not None:
                return self._result
            if self._owner_stopped:
                raise RuntimeError('stopped owner retains an unresolved pool obligation')
            self.backend.record("result_begin", job=self.request.job_id)
            # result includes native completion, runtime staging copy-back and argument cleanup.
            # A timeout or error retains every adapter obligation.
            self._drain_native()
            if self._pool_receipt is None:
                self._pool_receipt = self.backend.slots.finish_native(self.request, self.slot, self._pool_operation_id)
            receipt = self._pool_receipt
            completion = NativeCompletion(self.backend.layout.block_bytes, True, True, receipt)
            if self.backend.completion_filter is not None:
                completion = self.backend.completion_filter(self.request, completion)
            self.backend.slots.busy = None
            self.backend.pending = None
            self._keepalive = None
            self._result = completion
            self.backend.record("finalized", job=self.request.job_id)
            return self._result


class QwenNativeTransferBackend:
    """One in-flight transfer, retained whole allocations, explicit result publication."""

    def __init__(self, worker, programs, slots: ManagedDdrSlots, layout: QwenKvLayout,
                 keys, values, worker_epoch: str, *, worker_id=0, run_config=None,
                 observer: Callable[[dict], None] | None = None):
        import torch

        if type(worker_id) is not int or worker_id != 0 or not worker_epoch:
            raise ValueError("Run02 needs one explicit local worker and epoch")
        shape = (40 * layout.num_pages * 1024, 128)
        if (tuple(keys.shape) != shape or tuple(values.shape) != shape
                or keys.dtype != torch.bfloat16 or values.dtype != torch.bfloat16
                or keys.data_ptr == values.data_ptr):
            raise ValueError("KV allocations do not match the complete Qwen arena")
        inherited = worker._inherited_host_tensors
        if any(not any(t is tensor for t in inherited) for tensor in slots.inherited_host_tensors):
            raise ValueError("DDR backing and page metadata must be explicitly inherited before fork")
        self.worker, self.programs, self.slots, self.layout = worker, programs, slots, layout
        self.keys, self.values, self.worker_epoch = keys, values, worker_epoch
        self.worker_id, self.run_config, self.observer = worker_id, run_config, observer
        self.pending: NativeDdrRun | None = None
        self.ambiguous = None
        self.closed = False
        self.is_managed = True
        self.completion_filter = None  # explicitly armed test receipt filter, after real native cleanup only
        self._pins = []
        self.descriptors = []
        try:
            for tensor in (keys, values):
                token, descriptor = worker.retain_service_tensor(tensor, worker_id=worker_id)
                self._pins.append(token)
                self.descriptors.append(descriptor)
        except BaseException:
            for token in self._pins:
                worker.release_service_tensor(token)
            raise

    def record(self, event, **fields):
        if self.observer is not None:
            self.observer({"event": event, "time_ns": time.monotonic_ns(), "pid": os.getpid(), **fields})

    def reserve_slot(self, slot, size):
        if self.closed:
            raise RuntimeError("transfer backend closed")
        return self.slots.reserve(slot, size)

    def release_slot(self, backing):
        self.slots.release(backing)

    def submit(self, request, backing):
        try:
            request.validate(self.layout)
            self.slots.validate(backing)
            if request.worker_epoch != self.worker_epoch:
                raise ValueError("worker epoch mismatch")
            if self.closed or self.pending is not None or self.ambiguous is not None or self.slots.busy is not None:
                raise ValueError("transfer owner closed or busy")
            self.slots.page_ids[0] = request.page_ids[0]
            if request.operation == Operation.STORE:
                program = self.programs[0].compiled
                args = (self.keys, self.values, self.slots.page_ids, backing.tensor)
            else:
                program = self.programs[1].compiled
                args = (backing.tensor, self.slots.page_ids, self.keys, self.values)
            self.slots.begin_native(request, backing)
        except Exception as exc:
            if self.slots.access.unknown:
                raise NativeSubmissionError(str(exc), stopped=False) from exc
            raise NativeSubmissionError(str(exc), stopped=True) from exc
        self.slots.busy = backing
        try:
            self.record("submit_begin", job=request.job_id, operation=request.operation.value,
                        page=request.page_ids[0], slot=backing.slot)
            native = self.worker.submit(program, *args, config=self.run_config)
        except BaseException:
            # The native boundary may already have side effects. Preserve all references.
            self.ambiguous = (request, backing, args)
            self.slots.access.report_unknown(backing.active_lease, 'native submit outcome unknown')
            raise
        self.pending = NativeDdrRun(self, request, backing, native)
        # These fixed-stack internal IDs are diagnostics, not a completion criterion.
        inner = native._native_handle
        self.record("accepted", job=request.job_id, dispatch_id=native._dispatch_id,
                    native_run_id=inner._run_id if inner is not None else None,
                    native_pending=not inner.done if inner is not None else False)
        return self.pending

    def read_slot(self, backing):
        self.slots.validate(backing)
        if self.slots.busy is backing or backing.committed is None:
            raise RuntimeError("DDR contents are not consumable before native result/finalize")
        self.record("consumer", slot=backing.slot)
        lease = self.slots.access.acquire(backing.block, 'r', committed=backing.committed)
        try:
            result = backing.tensor.clone()
        except BaseException as error:
            self.slots.access.report_unknown(lease, f'diagnostic reader incomplete: {error}')
            raise
        self.slots.access.complete_read(lease, uuid.uuid4().hex)
        return result

    def close(self):
        if self.ambiguous is not None:
            raise RuntimeError("unknown native submission must retain owner obligations")
        if self.pending is not None:
            self.pending.result()
        if self.slots.leases:
            raise RuntimeError("DDR entries still retained by the worker")
        self._release_pins()

    def _release_pins(self):
        if not self.closed:
            while self._pins:
                self.worker.release_service_tensor(self._pins[-1])
                self._pins.pop()
            self.closed = True

    def drain_native_domain(self):
        """Permit owned device shutdown despite an unresolved control ACK.

        No UF range, imported entry or mapping obligation is removed here.
        Ambiguous native submit still requires separate runtime stop evidence.
        """
        if self.ambiguous is not None:
            raise RuntimeError('unknown submit cannot prove native drain')
        if self.pending is not None:
            self.pending.drain_native()
        self._release_pins()

    def owner_stopped_preserving_pool(self):
        if self.pending is not None:
            self.pending.release_stopped_owner()
        self.worker = self.keys = self.values = None
