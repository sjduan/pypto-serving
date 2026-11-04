# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""M3 real same-owner fixture and production-prefix byte probes; no alternate transfer backend."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import weakref

import numpy as np
import torch

from pypto_serving.config.types import NativeDdrPrefixConfig, PrefillBatch, RuntimeConfig
from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.qwen.native_ddr_kv import compile_programs
from pypto_serving.model.qwen.native_ddr_transfer import PreforkDdrSlots, QwenNativeTransferBackend
from pypto_serving.model.qwen.npu_executor import (
    Qwen314BPyptoExecutor, _find_pypto_lib_qwen14b_dir, _load_pypto_lib_qwen14b_module,
)
from pypto_serving.model.qwen.npu_runner import Qwen314BModelRunner
from pypto_serving.serving.native_ddr.types import (
    DdrEntryRef, Operation, PrefixKey, QwenKvLayout, DdrTransferRequest, TransferStatus,
)
from pypto_serving.serving.native_ddr.worker import DdrPrefixWorker

ROOT = Path(os.environ["UFLOW_RUN02_ROOT"])


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def block_oracle(host_k, host_v, pages, page):
    # Independent CPU oracle uses explicit [L,P,T,H,D] indexing, not device row arithmetic.
    k = host_k.reshape(40, pages, 128, 8, 128)[:, page]
    v = host_v.reshape(40, pages, 128, 8, 128)[:, page]
    return torch.stack((k, v), dim=1).contiguous().reshape(81920, 128)


def byte_equal(actual, expected):
    return bool(torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)))


def pattern(pages, side):
    result = torch.empty((40 * pages * 1024, 128), dtype=torch.bfloat16)
    array = result.view(torch.uint16).numpy().reshape(40, pages, 128, 8, 128)
    index = np.arange(128 * 8 * 128, dtype=np.uint32).reshape(128, 8, 128)
    for layer in range(40):
        for page in range(pages):
            array[layer, page] = (index * 17 + layer * 997 + page * 113 + side * 3109).astype(np.uint16)
    return result


def read_arena(worker, tensor):
    host = torch.empty(tensor.shape, dtype=tensor.dtype)
    worker.copy_from(host.data_ptr(), tensor.data_ptr, tensor.nbytes)
    return host


def read_block(worker, layout, keys, values, page):
    compact = torch.empty((40, 2, 128, 8, 128), dtype=torch.bfloat16)
    for layer in range(40):
        for side, tensor in enumerate((keys, values)):
            view = compact[layer, side]
            worker.copy_from(view.data_ptr(), tensor.data_ptr, layout.page_side_bytes,
                             src_offset=layout.hbm_offset(layer, page))
    return compact.reshape(81920, 128)


def request(epoch, job, operation, page, entry, generation):
    return DdrTransferRequest(job, epoch, operation,
                              (PrefixKey(b"N" * 32, hashlib.sha256(entry.encode()).digest()),),
                              (DdrEntryRef(epoch, entry, generation),), "lease-" + job, (page,),
                              1 if operation == Operation.STORE else None)


def settle(module, backend, trace):
    pending = module.pending.handle
    trace.append({"event": "native_poll_initial", "done": pending.native.done, "time_ns": time.monotonic_ns()})
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        events = module.poll_completed()
        if events:
            event = events[0]
            trace.append({"event": "worker_event", "status": event.status.value,
                          "native_completed": event.native_completed, "stopped": event.stopped,
                          "time_ns": time.monotonic_ns()})
            return event
        time.sleep(0.0005)
    raise TimeoutError("M3 transfer did not settle")


def fixture(output, device=0):
    trace, report = [], {"case": "fixture", "passed": False, "pid": os.getpid()}
    compiler = KernelCompiler(run_config=build_pypto_run_config(
        platform="a2a3", device_ids=(device,), pypto_build_dir=str(ROOT / "build/fixture")))
    kernel_dir = _find_pypto_lib_qwen14b_dir()
    qwen = [
        compiler.compile("prefill_fwd", _load_pypto_lib_qwen14b_module("prefill_fwd", kernel_dir).qwen3_prefill_host),
        compiler.compile("decode_fwd", _load_pypto_lib_qwen14b_module("decode_fwd", kernel_dir).qwen3_decode_host),
        compiler.compile("topk_select_fwd", _load_pypto_lib_qwen14b_module("topk_select", kernel_dir).qwen3_topk_select_host),
    ]
    copies = compile_programs(compiler)
    slots = PreforkDdrSlots(2)
    # Program-only fixture: real production Qwen compiled programs, same runner startup, no fake execution.
    runner = Qwen314BModelRunner(compiled=None, device_id=device)
    runner._compiled = SimpleNamespace(prefill=qwen[0], decode=qwen[1], topk_select=qwen[2],
                                       native_ddr_programs=copies, native_ddr_slots=slots)
    try:
        worker = runner._shared_l3_worker()
        layout = QwenKvLayout(40, 16, 128, 8, 128)
        host_k, host_v = pattern(16, 0), pattern(16, 1)
        keys = worker.alloc_tensor(host_k.shape, host_k.dtype, init=host_k)
        values = worker.alloc_tensor(host_v.shape, host_v.dtype, init=host_v)
        backend = QwenNativeTransferBackend(worker, copies, slots, layout, keys, values, "fixture", observer=trace.append)
        module = DdrPrefixWorker(NativeDdrPrefixConfig(True, 2 * layout.block_bytes), layout, "fixture", backend)
    except BaseException:
        runner.close()
        raise
    report.update(owner_python_id=id(worker), chip_pids=list(worker._w._chip_pids),
                  allocations=[{"base": t.data_ptr, "bytes": t.nbytes, "worker_id": 0} for t in (keys, values)],
                  prefork_host=[{"ptr": t.data_ptr(), "bytes": t.numel() * t.element_size()} for t in slots.inherited_host_tensors],
                  registered_programs=[p.name for p in [*qwen, *copies]], blocks=[], negatives=[], trace=trace)
    try:
        assert byte_equal(read_arena(worker, keys), host_k)
        assert byte_equal(read_arena(worker, values), host_v)
        # Ordinary post-fork private CPU endpoints exercise the runtime's legal POSIX staging.
        private_page = torch.tensor([5], dtype=torch.int32)
        private_out = torch.empty((81920, 128), dtype=torch.bfloat16)
        assert worker._named_host_buffer(private_out.data_ptr(), private_out.numel() * 2) is None
        h = worker.submit(copies[0].compiled, keys, values, private_page, private_out)
        h.result(timeout=60)
        assert byte_equal(private_out, block_oracle(host_k, host_v, 16, 5))
        private_page[0] = 7
        h = worker.submit(copies[1].compiled, private_out, private_page, keys, values)
        h.result(timeout=60)
        source = private_out.reshape(40, 2, 128, 8, 128)
        host_k.reshape(40, 16, 128, 8, 128)[:, 7].copy_(source[:, 0])
        host_v.reshape(40, 16, 128, 8, 128)[:, 7].copy_(source[:, 1])
        assert byte_equal(read_arena(worker, keys), host_k)
        assert byte_equal(read_arena(worker, values), host_v)
        report["postfork_staging_passed"] = True
        unrelated_args = [
            worker.alloc_tensor((16, 152064), torch.float32, init=torch.zeros((16, 152064))),
            worker.alloc_tensor((2,), torch.int32, init=torch.tensor([1, 1], dtype=torch.int32)),
            worker.alloc_tensor((16, 32), torch.float32), worker.alloc_tensor((16, 32), torch.int32),
        ]
        for generation, (src, dst) in enumerate([(5, 2), (1, 6), (0, 15), (15, 0)], 1):
            name = "block-" + str(generation)
            expected = block_oracle(host_k, host_v, 16, src)
            store = request("fixture", "store-" + name, Operation.STORE, src, name, generation)
            before_ids = (keys.data_ptr, values.data_ptr)
            assert module.submit_transfer(store).status == TransferStatus.ACCEPTED
            backing = module.pending.entry.backing
            handle = module.pending.handle
            if generation == 1:
                native_before = not handle.native._native_handle.done if handle.native._native_handle else False
                t0 = time.monotonic_ns()
                unrelated = worker.submit(qwen[2].compiled, *unrelated_args)
                trace.append({"event": "unrelated_accepted", "time_ns": time.monotonic_ns(),
                              "submit_begin_ns": t0, "transfer_native_pending_before": native_before})
                unrelated.result(timeout=60)
                report["unrelated_existing_program_accepted"] = True
            for label, fn in [
                ("pending_backing_release", lambda: backend.release_slot(backing)),
                ("pending_consumer", lambda: backend.read_slot(backing)),
                ("pending_owner_free", lambda: worker.free_tensor(keys)),
            ]:
                try:
                    fn()
                except (ValueError, RuntimeError):
                    report["negatives"].append(label)
                else:
                    raise AssertionError(label + " was not rejected")
            retained = weakref.ref(backing.tensor)
            gc.collect()
            assert retained() is not None and handle._keepalive is not None
            event = settle(module, backend, trace)
            assert event.status == TransferStatus.SUCCEEDED and event.transferred_bytes == layout.block_bytes
            compact = backend.read_slot(backing)
            assert byte_equal(compact, expected)
            load = request("fixture", "load-" + name, Operation.LOAD, dst, name, generation)
            assert module.submit_transfer(load).status == TransferStatus.ACCEPTED
            assert settle(module, backend, trace).status == TransferStatus.SUCCEEDED
            assert before_ids == (keys.data_ptr, values.data_ptr)
            hk = host_k.reshape(40, 16, 128, 8, 128)
            hv = host_v.reshape(40, 16, 128, 8, 128)
            hk[:, dst].copy_(expected.reshape(40, 2, 128, 8, 128)[:, 0])
            hv[:, dst].copy_(expected.reshape(40, 2, 128, 8, 128)[:, 1])
            # Complete-arena comparison includes every untouched page, layer, head and raw NaN payload.
            assert byte_equal(read_arena(worker, keys), host_k)
            assert byte_equal(read_arena(worker, values), host_v)
            report["blocks"].append({"source": src, "target": dst, "bytes": layout.block_bytes,
                                     "compact_sha256": digest(compact), "all_untouched_bytes_equal": True})
            module.evict(store.entry_refs[0])
        def rejects(label, fn):
            accepted_before = sum(t["event"] == "accepted" for t in trace)
            try:
                fn()
            except (ValueError, RuntimeError, TypeError):
                assert sum(t["event"] == "accepted" for t in trace) == accepted_before
                report["negatives"].append(label)
            else:
                raise AssertionError(label + " was accepted")
        rejects("wrong_worker", lambda: QwenNativeTransferBackend(worker, copies, slots, layout, keys, values,
                                                                 "fixture", worker_id=1))
        from pypto.runtime import DeviceTensor
        forged = DeviceTensor(keys.data_ptr, keys.shape, keys.dtype, buffer=keys.buffer)
        rejects("forged_owner_handle", lambda: QwenNativeTransferBackend(worker, copies, slots, layout,
                                                                        forged, values, "fixture"))
        rejects("interior_pointer_copy", lambda: worker.copy_from(private_out.data_ptr(), keys.data_ptr + 2, 2))
        rejects("range_overflow_copy", lambda: worker.copy_from(private_out.data_ptr(), keys.data_ptr, 2,
                                                               src_offset=keys.nbytes))
        rejects("zero_length_copy", lambda: worker.copy_from(private_out.data_ptr(), keys.data_ptr, 0))
        temp = worker.alloc_tensor(keys.shape, keys.dtype)
        worker.free_tensor(temp)
        rejects("released_allocation", lambda: QwenNativeTransferBackend(worker, copies, slots, layout,
                                                                        temp, values, "fixture"))
        invalid = request("fixture", "bad-page", Operation.STORE, 16, "bad", 10)
        assert module.submit_transfer(invalid).status == TransferStatus.REJECTED
        report["negatives"].append("page_out_of_range")
        # Logical cancellation does not release metadata/backing before actual native completion.
        cancelled = request("fixture", "cancel-store", Operation.STORE, 5, "cancelled", 11)
        assert module.submit_transfer(cancelled).status == TransferStatus.ACCEPTED
        ref = weakref.ref(module.pending.entry.backing.tensor)
        module.cancel(cancelled.job_id)
        assert module.pending is not None and ref() is not None
        assert settle(module, backend, trace).status == TransferStatus.CANCELLED
        assert "cancelled" not in module.entries
        report["cancel_drained_before_release"] = True
        # Fault injection is at the host receipt validator AFTER a real accepted native run.
        # Keep the actual handle and completion/cleanup; never submit an invalid device address.
        failed = request("fixture", "bad-receipt-store", Operation.STORE, 1, "bad-receipt", 12)
        assert module.submit_transfer(failed).status == TransferStatus.ACCEPTED
        real = module.pending.handle
        failed_backing = module.pending.entry.backing

        class BadReceipt:
            native = real.native

            def done(self):
                return real.done()

            def result(self):
                completion = real.result()
                trace.append({"event": "injected_short_receipt_after_native_cleanup",
                              "time_ns": time.monotonic_ns()})
                return replace(completion, transferred_bytes=0)

        module.pending.handle = BadReceipt()
        assert slots.busy is failed_backing and real._keepalive is not None
        failed_event = settle(module, backend, trace)
        assert failed_event.status == TransferStatus.FAILED and failed_event.stopped
        assert failed_event.native_completed and failed_event.transferred_bytes == 0
        assert backend.pending is None and real._keepalive is None and slots.busy is None
        assert "bad-receipt" not in module.entries and not slots.leases
        report["postsubmit_failure"] = "injected short host receipt after actual native result/cleanup; no device fault"
        module.close()
        backend.close()
        report["passed"] = True
    finally:
        report["trace"] = trace
        output.write_text(json.dumps(report, indent=2))
        if backend.pending is not None:
            backend.pending.result()
        if module.pending is None and not module.poisoned:
            module.close()
        if not slots.leases:
            backend.close()
        runner.close()
        output.write_text(json.dumps(report, indent=2))
    return report



def production(output, device=0):
    """Complete one real40-layer prefix, then pack and restore one full block on that exact owner."""
    from pypto_serving.config.types import ModelRecord
    from pypto_serving.model.model_loader import ModelLoader

    trace, report = [], {"case": "production", "passed": False, "pid": os.getpid()}
    runtime = RuntimeConfig(page_size=128, max_batch_size=16, max_seq_len=512,
                            kv_dtype="bfloat16", weight_dtype="bfloat16", npu_memory_utilization=0.8,
                            max_num_batched_tokens=512, ring_dep_pool=262144, ring_task_window=262144,
                            ring_heap=1073741824)
    loaded = ModelLoader().load(model_id="Qwen3-14B", model_dir="/data/models/Qwen/Qwen3-14B",
                                runtime_config=runtime)
    model = loaded.runtime_model
    record = ModelRecord(config=loaded.config, runtime=runtime, tokenizer=loaded.tokenizer,
                         layer_specs=loaded.layer_specs, runtime_model=model)
    executor = Qwen314BPyptoExecutor(platform="a2a3", device_ids=(device,), native_ddr_probe_slots=1,
                                    pypto_build_dir=str(ROOT / "build/production"))
    backend = module = None
    try:
        pages = executor.register_model("Qwen3-14B", record)
        runner = executor._runners["Qwen3-14B"]
        worker = runner._shared_l3_worker()
        owner_id = id(worker)
        # Exactly128 tokens complete one reusable prefix page using the production prefill program.
        text = ("A river crosses the center of a large European city. Its museums and public gardens "
                "attract visitors throughout the year. ") * 12
        tokens = loaded.tokenizer.encode(text)[:128]
        assert len(tokens) == 128 and loaded.config.num_hidden_layers == 40 and pages > 6
        batch = PrefillBatch(["m3-production-prefix"], torch.tensor(tokens, dtype=torch.int32), None,
                              [128], [128], [0], [0], prompt_lens=[128], block_ids=[[5]])
        result = executor.run_prefill(model, batch)
        assert bool(torch.isfinite(result.logits).all())
        trace.append({"event": "production_prefix_completed", "time_ns": time.monotonic_ns(),
                      "tokens": len(tokens), "layers": 40})
        backend = runner.create_native_ddr_backend(model, "production", observer=trace.append)
        layout = backend.layout
        original = read_block(worker, layout, backend.keys, backend.values, 5)
        # Neighbor and arena-boundary pages are independently hashed before and after the native scatter.
        sentinels = {p: digest(read_block(worker, layout, backend.keys, backend.values, p))
                     for p in (0, 1, 3, pages - 1)}
        module = DdrPrefixWorker(NativeDdrPrefixConfig(True, layout.block_bytes), layout, "production", backend)
        store = request("production", "production-store", Operation.STORE, 5, "prefix", 1)
        assert module.submit_transfer(store).status == TransferStatus.ACCEPTED
        backing = module.pending.entry.backing
        assert settle(module, backend, trace).status == TransferStatus.SUCCEEDED
        compact = backend.read_slot(backing)
        assert byte_equal(compact, original)
        # Independent synchronous seed reuses source page only after the store fence; destination stays reserved.
        overwrite = torch.empty((128, 8, 128), dtype=torch.bfloat16)
        overwrite.view(torch.uint16).fill_(0xA55A)
        for layer in range(40):
            for tensor in (backend.keys, backend.values):
                worker.copy_to(tensor.data_ptr, overwrite.data_ptr(), layout.page_side_bytes,
                               dst_offset=layout.hbm_offset(layer, 5))
        load = request("production", "production-load", Operation.LOAD, 2, "prefix", 1)
        target_bases = (backend.keys.data_ptr, backend.values.data_ptr)
        assert module.submit_transfer(load).status == TransferStatus.ACCEPTED
        assert settle(module, backend, trace).status == TransferStatus.SUCCEEDED
        restored = read_block(worker, layout, backend.keys, backend.values, 2)
        assert byte_equal(restored, original)
        assert target_bases == (backend.keys.data_ptr, backend.values.data_ptr) and id(runner._shared_l3_worker()) == owner_id
        assert sentinels == {p: digest(read_block(worker, layout, backend.keys, backend.values, p)) for p in sentinels}
        report.update(passed=True, layers=40, pages=pages, block_bytes=layout.block_bytes,
                      prefix_tokens=tokens, source_page=5, restored_page=2, source_reused_after_store=True,
                      owner_python_id=owner_id, chip_pids=list(worker._w._chip_pids),
                      allocations=[{"base": t.data_ptr, "bytes": t.nbytes, "worker_id": 0}
                                   for t in (backend.keys, backend.values)],
                      original_sha256=digest(original), compact_sha256=digest(compact),
                      restored_sha256=digest(restored), sentinels_unchanged=True,
                      per_layer_side_sha256=[
                          [digest(original.reshape(40, 2, 1024, 128)[layer, side]) for side in (0, 1)]
                          for layer in range(40)], trace=trace, model_config=asdict(loaded.config))
        module.evict(store.entry_refs[0])
        module.close()
        backend.close()
    finally:
        report["trace"] = trace
        output.write_text(json.dumps(report, indent=2))
        if backend is not None:
            if backend.pending is not None:
                backend.pending.result()
            if module is not None and module.pending is None and not module.poisoned:
                module.close()
            if not backend.slots.leases:
                backend.close()
        executor.close()
        output.write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["fixture", "production"], default="fixture")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    fn = fixture if args.case == "fixture" else production
    print(json.dumps({"passed": fn(args.output, args.device)["passed"], "output": str(args.output)}))
