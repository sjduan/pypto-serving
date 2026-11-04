# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded worker-side residency and native-handle ownership, without a fake backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import (
    DdrEntryRef, DdrTransferEvent, DdrTransferRequest, EntryState,
    NativeDdrPrefixConfig, Operation, QwenKvLayout, TransferStatus,
)


@dataclass(frozen=True)
class NativeStopProof:
    """M3 owner receipt, never synthesized from a parent process reap."""
    worker_epoch: str
    native_drained: bool
    arguments_released: bool


@dataclass(frozen=True)
class NativeCompletion:
    transferred_bytes: int
    native_completed: bool
    stopped: bool


class NativeTransferHandle(Protocol):
    def done(self) -> bool: ...
    def result(self) -> NativeCompletion: ...


class NativeTransferBackend(Protocol):
    """M3 seam: same-owner pack/unpack, prefork backing, real native fence.

    reserve_slot must return a retained owning allocation, not an interior
    pointer. submit must either return the accepted handle or raise
    NativeSubmissionError with its handle/stop evidence. An ordinary exception
    is UNKNOWN, never proof that native submission had no side effects.
    """
    def reserve_slot(self, slot: int, size: int) -> object: ...
    def release_slot(self, backing: object) -> None: ...
    def submit(self, request: DdrTransferRequest, backing: object) -> NativeTransferHandle: ...


class NativeSubmissionError(RuntimeError):
    def __init__(self, message, *, stopped=False, handle=None):
        super().__init__(message)
        self.stopped = stopped
        self.handle = handle


@dataclass
class _Entry:
    ref: DdrEntryRef
    slot: int
    backing: object
    key: object
    state: EntryState = EntryState.STORE_RESERVED
    readers: int = 0


@dataclass
class _Pending:
    request: DdrTransferRequest
    entry: _Entry
    handle: NativeTransferHandle | None = None
    cancelled: bool = False
    quarantined: bool = False


class DdrPrefixWorker:
    def __init__(self, config: NativeDdrPrefixConfig, layout: QwenKvLayout, worker_epoch: str,
                 backend: NativeTransferBackend | None = None, *, namespace_digest: bytes | None = None):
        if not config.enabled:
            raise ValueError("do not construct DDR worker when disabled")
        self.layout = layout
        self.worker_epoch = worker_epoch
        if namespace_digest is not None and (not isinstance(namespace_digest, bytes) or len(namespace_digest) != 32):
            raise ValueError("invalid worker compatibility namespace")
        self.namespace_digest = namespace_digest
        self.backend = backend
        self.capacity = config.capacity_bytes // layout.block_bytes
        self.entries: dict[str, _Entry] = {}
        self.pending: _Pending | None = None
        self._generation = 0
        self._jobs: dict[str, DdrTransferRequest] = {}
        self._terminal_events: dict[str, DdrTransferEvent] = {}
        # Bounded incarnation-local replay ledger. Never forget a receipt and
        # accidentally replay an old native write into a reused HBM page.
        # Exhaustion explicitly rejects NEW jobs (normal prefill fallback);
        # recycling this budget requires a new, genuinely drained owner.
        self._receipt_limit = max(64, min(4096, self.capacity * 4))
        self.closed = False
        self.poisoned = False

    def _event(self, req, status, *, stopped=False, completed=False, size=0, error=None):
        event = DdrTransferEvent(req.job_id, req.worker_epoch, req.operation, status,
                                 req.entry_refs, size, completed, stopped, error)
        if stopped and self._jobs.get(req.job_id) == req:
            self._terminal_events[req.job_id] = event
        return event

    def submit_transfer(self, request: DdrTransferRequest) -> DdrTransferEvent:
        try:
            previous = self._jobs.get(request.job_id)
            if previous is not None:
                if previous != request:
                    return self._event(request, TransferStatus.FAILED, error="job identity replay conflict")
                if request.job_id in self._terminal_events:
                    return self._terminal_events[request.job_id]
                # A duplicate pending command owns the SAME native obligation;
                # REJECTED/stopped=True would prematurely release its source.
                return self._event(request, TransferStatus.FAILED if self.poisoned else TransferStatus.ACCEPTED,
                                   error="native obligation retained" if self.poisoned else None)
            request.validate(self.layout)
            if request.worker_epoch != self.worker_epoch:
                raise ValueError("worker epoch mismatch")
            if self.namespace_digest is not None and any(
                    key.namespace_digest != self.namespace_digest for key in request.entry_keys):
                raise ValueError("worker compatibility namespace mismatch")
            if self.closed or self.poisoned or self.pending is not None:
                raise ValueError("worker closed, quarantined, busy, or duplicate job")
            if len(self._jobs) >= self._receipt_limit:
                raise ValueError("native DDR replay receipt budget exhausted; a drained new owner is required")
            if self.backend is None:
                raise ValueError("native backend unavailable: M3 same-owner adapter not verified")
            ref = request.entry_refs[0]
            if request.operation == Operation.STORE:
                if ref.entry_id in self.entries or ref.slot_generation <= self._generation:
                    raise ValueError("entry identity/generation reused")
                used = {e.slot for e in self.entries.values()}
                slot = next((s for s in range(self.capacity) if s not in used), None)
                if slot is None:
                    raise ValueError("DDR capacity exhausted")
                backing = self.backend.reserve_slot(slot, self.layout.block_bytes)
                entry = _Entry(ref, slot, backing, request.entry_keys[0])
                self.entries[ref.entry_id] = entry
                self._generation = ref.slot_generation
            else:
                entry = self.entries.get(ref.entry_id)
                if entry is None or entry.ref != ref or entry.state != EntryState.READY:
                    raise ValueError("LOAD needs a ready entry of the exact generation")
                if entry.key != request.entry_keys[0]:
                    raise ValueError("LOAD key does not match resident entry")
                entry.readers += 1
        except (ValueError, MemoryError) as exc:
            return self._event(request, TransferStatus.REJECTED, stopped=True, error=str(exc))
        # Publish the obligation BEFORE crossing a potentially side-effectful seam.
        self._jobs[request.job_id] = request
        pending = self.pending = _Pending(request, entry)
        if request.operation == Operation.STORE:
            entry.state = EntryState.OFFLOADING
        try:
            pending.handle = self.backend.submit(request, entry.backing)
            if pending.handle is None:
                raise RuntimeError("backend returned no native handle")
        except Exception as exc:
            if isinstance(exc, NativeSubmissionError):
                pending.handle = exc.handle
            if isinstance(exc, NativeSubmissionError) and exc.stopped and exc.handle is None:
                self._settle(pending, success=False)
                return self._event(request, TransferStatus.FAILED, stopped=True, error=str(exc))
            pending.quarantined = pending.handle is None
            self.poisoned = True
            entry.state = EntryState.QUARANTINED
            return self._event(request, TransferStatus.FAILED, error=f"native submit outcome unknown: {exc}")
        return self._event(request, TransferStatus.ACCEPTED)

    def _settle(self, pending, *, success):
        entry = pending.entry
        if pending.request.operation == Operation.LOAD:
            entry.readers -= 1
        elif success and not pending.cancelled and entry.state != EntryState.EVICT_PENDING:
            entry.state = EntryState.READY
        else:
            entry.state = EntryState.EVICT_PENDING
        if entry.state == EntryState.EVICT_PENDING and not entry.readers:
            # On release failure retain the entry/backing and poison the owner.
            try:
                assert self.backend is not None
                self.backend.release_slot(entry.backing)
            except Exception:
                self.poisoned = True
                entry.state = EntryState.QUARANTINED
                raise
            del self.entries[entry.ref.entry_id]
        self.pending = None

    def poll_completed(self) -> tuple[DdrTransferEvent, ...]:
        pending = self.pending
        if pending is None or pending.quarantined or pending.handle is None:
            return ()
        req = pending.request
        try:
            if not pending.handle.done():
                return ()
            result = pending.handle.result()
            if not result.stopped:
                raise RuntimeError("native completion has no stop proof")
            success = result.native_completed and result.transferred_bytes == self.layout.block_bytes
            self._settle(pending, success=success)
            status = (TransferStatus.CANCELLED if pending.cancelled else
                      TransferStatus.SUCCEEDED if success else TransferStatus.FAILED)
            return (self._event(req, status, stopped=True, completed=result.native_completed,
                                size=result.transferred_bytes,
                                error=None if success else "incomplete native transfer"),)
        except Exception as exc:
            self.poisoned = True
            pending.quarantined = True
            # Even a result()/cleanup exception is not a stop receipt. Preserve
            # all strong references and the queue obligation for owner diagnosis.
            self.pending = pending
            pending.entry.state = EntryState.QUARANTINED
            return (self._event(req, TransferStatus.FAILED, error=str(exc)),)

    def cancel(self, job_id: str):
        if self.pending is not None and self.pending.request.job_id == job_id:
            self.pending.cancelled = True

    def evict(self, ref: DdrEntryRef):
        entry = self.entries.get(ref.entry_id)
        if entry is None:
            return
        if entry.ref != ref:
            raise ValueError("stale eviction generation")
        if entry.state == EntryState.QUARANTINED:
            return
        entry.state = EntryState.EVICT_PENDING
        if self.pending is not None and self.pending.entry is entry and self.pending.request.operation == Operation.STORE:
            self.pending.cancelled = True
        if not entry.readers and (self.pending is None or self.pending.entry is not entry):
            assert self.backend is not None
            try:
                self.backend.release_slot(entry.backing)
            except Exception:
                self.poisoned = True
                entry.state = EntryState.QUARANTINED
                raise
            del self.entries[ref.entry_id]

    def close(self) -> tuple[DdrTransferEvent, ...]:
        self.closed = True
        events = self.poll_completed()
        if self.pending is not None or self.poisoned:
            raise RuntimeError("native DDR close incomplete: retained native/backing obligations")
        for entry in tuple(self.entries.values()):
            self.evict(entry.ref)
        return events
