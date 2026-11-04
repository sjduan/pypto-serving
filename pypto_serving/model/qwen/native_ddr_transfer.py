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
from typing import Any, Callable

from pypto_serving.serving.native_ddr.types import Operation, QwenKvLayout
from pypto_serving.serving.native_ddr.worker import NativeCompletion, NativeSubmissionError


@dataclass(eq=False)
class DdrSlot:
    slot: int
    generation: int
    tensor: Any = field(repr=False)
    owner: Any = field(repr=False)
    native_complete: bool = False


class PreforkDdrSlots:
    """Owning shared allocations created before the chip child forks."""

    def __init__(self, count: int):
        if type(count) is not int or count <= 0:
            raise ValueError("native DDR requires an explicit finite positive prefork slot count")
        import torch
        from pypto_serving.model.common.runner.buffer_set import shared_empty

        self.tensors = tuple(shared_empty((81920, 128), torch.bfloat16) for _ in range(count))
        self.page_ids = shared_empty((1,), torch.int32)
        self.page_ids.zero_()
        self.leases: dict[int, DdrSlot] = {}
        self.busy: DdrSlot | None = None
        self.generation = 0

    @property
    def inherited_host_tensors(self):
        return (*self.tensors, self.page_ids)

    def reserve(self, slot, size):
        if type(slot) is not int or not 0 <= slot < len(self.tensors) or size != 20971520:
            raise ValueError("invalid whole-block DDR slot")
        if slot in self.leases:
            raise ValueError("DDR slot already leased")
        self.generation += 1
        lease = DdrSlot(slot, self.generation, self.tensors[slot], self)
        self.leases[slot] = lease
        return lease

    def validate(self, lease):
        if (not isinstance(lease, DdrSlot) or lease.owner is not self
                or self.leases.get(lease.slot) is not lease):
            raise ValueError("stale or foreign DDR allocation")

    def release(self, lease):
        self.validate(lease)
        if self.busy is lease:
            raise RuntimeError("pending native transfer retains DDR backing")
        del self.leases[lease.slot]


class NativeDdrRun:
    """Adapt the real DistributedRunHandle property/result into the typed worker protocol."""

    def __init__(self, backend, request, slot, native):
        self.backend = backend
        self.request = request
        self.slot = slot
        self.native = native
        self._result = None
        self._lock = threading.Lock()
        # Explicit strong references survive even if callers drop their copies.
        self._keepalive = (backend.worker, backend.keys, backend.values, slot, backend.slots.page_ids, request)

    def done(self):
        return self.native.done

    def result(self):
        with self._lock:
            if self._result is not None:
                return self._result
            self.backend.record("result_begin", job=self.request.job_id)
            # result includes native completion, runtime staging copy-back and argument cleanup.
            # A timeout or error retains every adapter obligation.
            self.native.result(timeout=60.0)
            self.backend.record("native_result_and_cleanup", job=self.request.job_id)
            if self.request.operation == Operation.STORE:
                self.slot.native_complete = True
            self._result = NativeCompletion(self.backend.layout.block_bytes, True, True)
            self.backend.slots.busy = None
            self.backend.pending = None
            self._keepalive = None
            self.backend.record("finalized", job=self.request.job_id)
            return self._result


class QwenNativeTransferBackend:
    """One in-flight transfer, retained whole allocations, explicit result publication."""

    def __init__(self, worker, programs, slots: PreforkDdrSlots, layout: QwenKvLayout,
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
            if request.operation == Operation.LOAD and not backing.native_complete:
                raise ValueError("LOAD backing has no native completion")
        except (ValueError, RuntimeError) as exc:
            raise NativeSubmissionError(str(exc), stopped=True) from exc
        self.slots.page_ids[0] = request.page_ids[0]
        self.slots.busy = backing
        if request.operation == Operation.STORE:
            program = self.programs[0].compiled
            args = (self.keys, self.values, self.slots.page_ids, backing.tensor)
        else:
            program = self.programs[1].compiled
            args = (backing.tensor, self.slots.page_ids, self.keys, self.values)
        self.record("submit_begin", job=request.job_id, operation=request.operation.value,
                    page=request.page_ids[0], slot=backing.slot)
        try:
            native = self.worker.submit(program, *args, config=self.run_config)
        except BaseException:
            # The native boundary may already have side effects. Preserve all references.
            self.ambiguous = (request, backing, args)
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
        if self.slots.busy is backing or not backing.native_complete:
            raise RuntimeError("DDR contents are not consumable before native result/finalize")
        self.record("consumer", slot=backing.slot)
        return backing.tensor.clone()

    def close(self):
        if self.ambiguous is not None:
            raise RuntimeError("unknown native submission must retain owner obligations")
        if self.pending is not None:
            self.pending.result()
        if self.slots.leases:
            raise RuntimeError("DDR entries still retained by the worker")
        if not self.closed:
            for token in self._pins:
                self.worker.release_service_tensor(token)
            self._pins.clear()
            self.closed = True
