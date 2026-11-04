# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU/fake control-plane tests, NOT native transfer/model/NPU verification."""
from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.qwen.native_ddr_kv import NativeDdrUnavailable, require_native_ddr_backend
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.native_ddr.coordinator import DdrPrefixCoordinator, ScheduledPrefix
from pypto_serving.serving.native_ddr.types import (
    DdrControlEvent, DdrEntryRef, DdrTransferEvent, DdrTransferRequest, EntryState, LoadState,
    NativeDdrPrefixConfig, Operation, PrefixKey, QwenKvLayout, RestoreKind, TransferStatus,
)
from pypto_serving.serving.native_ddr.worker import (
    DdrPrefixWorker, NativeCompletion, NativeSubmissionError, NativeStopProof,
)
from pypto_serving.serving.server.ipc import (
    NativeDdrCommand, StepResult, decode_command, decode_native_ddr_event,
    decode_result, encode_command, encode_native_ddr_event, encode_result,
    NativeDdrMessage, decode_native_ddr_message, encode_native_ddr_message,
)
from pypto_serving.serving.sched.scheduler import Request, Scheduler, SchedulerConfig


LAYOUT = QwenKvLayout(40, 16, 128, 8, 128)
CONFIG = NativeDdrPrefixConfig(True, LAYOUT.block_bytes * 2)
KEY = PrefixKey(b'n' * 32, b'h' * 32)
REF = DdrEntryRef("worker1", "entry1", 1)
_wire_sequences = {}


@pytest.fixture(autouse=True)
def reset_delivery_fixture():
    _wire_sequences.clear()


def transfer(operation=Operation.STORE, *, job="job1", page=0, ref=REF, key=KEY):
    sequence = _wire_sequences.setdefault(job, len(_wire_sequences) + 1)
    return DdrTransferRequest(job, "worker1", operation, (key,), (ref,), "lease1", (page,), 1, sequence)


class FakeHandle:
    """Manual test-only completion, never a production backend or native fence."""
    def __init__(self):
        self.ready = False
        self.outcome = NativeCompletion(LAYOUT.block_bytes, True, True)
        self.error = None

    def done(self):
        return self.ready

    def result(self):
        assert self.ready
        if self.error:
            raise self.error
        return self.outcome


class FakeBackend:
    def __init__(self):
        self.slots = {}
        self.submissions = []
        self.handle = None
        self.error = None
        self.release_error = None

    def reserve_slot(self, slot, size):
        assert size == LAYOUT.block_bytes
        assert slot not in self.slots
        backing = object()  # explicit fake, no DDR allocation or data claims
        self.slots[slot] = backing
        return backing

    def release_slot(self, backing):
        if self.release_error:
            raise self.release_error
        slot = next(s for s, b in self.slots.items() if b is backing)
        del self.slots[slot]

    def submit(self, request, backing):
        assert backing in self.slots.values()
        self.submissions.append(request)
        if self.error:
            raise self.error
        self.handle = FakeHandle()
        return self.handle


def worker(backend=None):
    _wire_sequences.clear()
    backend = backend or FakeBackend()
    return DdrPrefixWorker(CONFIG, LAYOUT, "worker1", backend), backend


def complete(module, backend):
    backend.handle.ready = True
    return module.poll_completed()[0]


def test_layout_full_byte_offsets_cpu_oracle():
    assert LAYOUT.page_side_bytes == 262144
    assert LAYOUT.block_bytes == 20971520
    for layer in range(LAYOUT.num_layers):
        for page in (0, LAYOUT.num_pages - 1):
            assert LAYOUT.hbm_offset(layer, page) == (layer * 16 + page) * 262144
        for side in (0, 1):
            assert LAYOUT.ddr_offset(1, layer, side) == 20971520 + (layer * 2 + side) * 262144
    # Independent CPU byte oracle for the proposed compact ordering. This does
    # not invoke, compile, or claim correctness of a PyPTO/native candidate.
    compact = bytearray(LAYOUT.block_bytes)
    for layer in range(40):
        for side in (0, 1):
            start = LAYOUT.ddr_offset(0, layer, side)
            compact[start:start + LAYOUT.page_side_bytes] = bytes([layer * 2 + side]) * LAYOUT.page_side_bytes
    for layer in range(40):
        for side in (0, 1):
            start = (layer * 2 + side) * 128 * 8 * 128 * 2
            assert compact[start:start + 262144] == bytes([layer * 2 + side]) * 262144


@pytest.mark.parametrize("method,args", [("hbm_offset", (40, 0)), ("hbm_offset", (0, 16)),
                                           ("hbm_offset", (0, True)), ("ddr_offset", (-1, 0, 0)),
                                           ("ddr_offset", (0, 0, 2))])
def test_offset_rejects_invalid_ranges(method, args):
    with pytest.raises(ValueError):
        getattr(LAYOUT, method)(*args)


@pytest.mark.parametrize("kwargs", [{"enabled": True}, {"capacity_bytes": -1},
                                   {"enabled": True, "capacity_bytes": 20971519},
                                   {"max_inflight_transfers": 2}])
def test_config_explicit_bounded_capacity(kwargs):
    with pytest.raises(ValueError):
        NativeDdrPrefixConfig(**kwargs)


def test_default_off_has_no_transfer_owner_and_enabled_rejects_unsupported():
    assert RuntimeConfig().native_ddr == NativeDdrPrefixConfig()
    core = ReplicaEngineCore(EngineConfig(), SimpleNamespace())
    assert core._native_ddr is None
    assert core._native_ddr_output_queue is None
    assert core.kv_cache_manager.num_blocks == 0
    with pytest.raises(ValueError, match="only Qwen"):
        ReplicaEngineCore(EngineConfig(executor_cls="MockExecutor", runtime_config=RuntimeConfig(native_ddr=CONFIG)),
                          SimpleNamespace())
    with pytest.raises(NativeDdrUnavailable, match="no CPU/mock fallback"):
        require_native_ddr_backend()


def test_typed_auxiliary_ipc_roundtrip_does_not_corrupt_step_result():
    request = transfer()
    command = NativeDdrCommand(operation="submit", request=request)
    assert decode_command(encode_command(command)).request == request
    event = DdrTransferEvent(request.job_id, request.worker_epoch, request.operation,
                             TransferStatus.ACCEPTED, request.entry_refs, 0, False, False)
    assert decode_native_ddr_event(encode_native_ddr_event(event)) == event
    assert decode_result(encode_result(StepResult(step_id=31, new_tokens={}))).step_id == 31
    with pytest.raises(ValueError, match="bounded"):
        decode_native_ddr_event(b'x' * 65537)


def test_no_backend_is_explicit_rejection_without_claiming_completed():
    module = DdrPrefixWorker(CONFIG, LAYOUT, "worker1")
    event = module.submit_transfer(transfer())
    assert event.status == TransferStatus.REJECTED
    assert not event.native_completed and event.stopped
    assert not module.entries and module.pending is None


@pytest.mark.parametrize("change", [dict(page_ids=(16,)), dict(page_ids=(True,)),
                                    dict(page_ids=(0, 0)), dict(entry_keys=()),
                                    dict(worker_epoch="old"), dict(entry_refs=(replace(REF, slot_generation=0),))])
def test_invalid_mapping_rejected_before_any_native_submit(change):
    module, backend = worker()
    assert module.submit_transfer(replace(transfer(), **change)).status == TransferStatus.REJECTED
    assert backend.submissions == []
    assert backend.slots == {}


def test_store_load_completion_cancel_and_read_during_evict_keep_backing():
    module, backend = worker()
    assert module.submit_transfer(transfer()).status == TransferStatus.ACCEPTED
    assert module.entries[REF.entry_id].state == EntryState.OFFLOADING
    assert module.poll_completed() == ()
    assert module.submit_transfer(transfer(job="busy")).status == TransferStatus.REJECTED
    assert complete(module, backend).status == TransferStatus.SUCCEEDED
    load = transfer(Operation.LOAD, job="load", page=7)
    assert module.submit_transfer(load).status == TransferStatus.ACCEPTED
    backing = module.entries[REF.entry_id].backing
    module.evict(REF)
    assert backing in backend.slots.values()
    module.cancel(load.job_id)
    assert complete(module, backend).status == TransferStatus.CANCELLED
    assert backend.slots == {}
    assert not module.entries
    assert module.close() == ()
    assert module.submit_transfer(transfer(job="closed")).status == TransferStatus.REJECTED


def test_slot_reuse_requires_fresh_generation_and_load_key_identity():
    module, backend = worker()
    module.submit_transfer(transfer())
    complete(module, backend)
    assert module.submit_transfer(transfer(Operation.LOAD, job="wrong-key", key=replace(KEY, block_hash=b'z' * 32))).status == TransferStatus.REJECTED
    module.evict(REF)
    assert module.submit_transfer(transfer(job="reused", ref=replace(REF, entry_id="entry2"))).status == TransferStatus.REJECTED
    fresh = replace(REF, entry_id="entry2", slot_generation=2)
    assert module.submit_transfer(transfer(job="fresh", ref=fresh)).status == TransferStatus.ACCEPTED
    module.cancel("fresh")
    assert complete(module, backend).status == TransferStatus.CANCELLED
    assert not backend.slots


@pytest.mark.parametrize("outcome", ["ordinary-submit", "result", "no-stop", "release"])
def test_unknown_native_or_cleanup_failure_quarantines_not_reuses(outcome):
    module, backend = worker()
    if outcome == "ordinary-submit":
        backend.error = RuntimeError("submit may have accepted")
        event = module.submit_transfer(transfer())
    else:
        module.submit_transfer(transfer())
        if outcome == "result":
            backend.handle.error = RuntimeError("result failed")
        elif outcome == "no-stop":
            backend.handle.outcome = NativeCompletion(LAYOUT.block_bytes, True, False)
        else:
            module.cancel("job1")
            backend.release_error = RuntimeError("release failed")
        event = complete(module, backend)
    assert event.status == TransferStatus.FAILED and not event.stopped
    assert module.poisoned and module.pending is not None
    assert module.entries[REF.entry_id].state == EntryState.QUARANTINED
    assert backend.slots
    with pytest.raises(RuntimeError, match="obligations"):
        module.close()


def test_typed_pre_submit_rejection_and_partial_completion_are_not_ready():
    module, backend = worker()
    backend.error = NativeSubmissionError("rejected before accept", stopped=True)
    assert module.submit_transfer(transfer()).stopped
    assert not backend.slots
    backend.error = None
    fresh = replace(REF, entry_id="next", slot_generation=2)
    module.submit_transfer(transfer(job="partial", ref=fresh))
    backend.handle.outcome = NativeCompletion(262144, True, True)
    assert complete(module, backend).status == TransferStatus.FAILED
    assert not module.entries


def coordinator(num_blocks=12, capacity=8):
    manager = KvCacheManager(num_blocks=num_blocks, block_size=128, enable_prefix_cache=True)
    coord = DdrPrefixCoordinator(manager, namespace_digest=b'n' * 32, worker_epoch="worker1", max_entries=capacity)
    return coord, manager


def request(name="request", length=129):
    return Request(request_id=name, prompt_token_ids=list(range(length)), max_new_tokens=2)


def seed_ready(coord, manager, req):
    key = PrefixKey(coord.namespace, manager.compute_block_hashes(req.prompt_token_ids)[0])
    coord.ready[key] = REF
    return key


def event_for(job, status=TransferStatus.SUCCEEDED, *, stopped=True, completed=True, size=20971520):
    return DdrTransferEvent(job.job_id, job.worker_epoch, job.operation, status, job.entry_refs,
                             size, completed, stopped, delivery_sequence=job.delivery_sequence)


def test_two_waiters_share_one_target_and_loader_pin_survives_until_lookup():
    coord, manager = coordinator()
    first, second = request("a"), request("b")
    key = seed_ready(coord, manager, first)
    assert coord.lookup_after_hbm(first, [], 128).kind == RestoreKind.WAIT
    assert coord.lookup_after_hbm(second, [], 128).kind == RestoreKind.WAIT
    assert len(coord.loads_by_key) == 1 and len(coord.jobs) == 1
    job = coord.next_transfer()
    assert job is not None and job.entry_keys == (key,)
    block = manager.blocks[job.page_ids[0]]
    assert block.block_hash is None and block.ref_cnt == 1
    coord.on_transfer_event(event_for(job, TransferStatus.ACCEPTED, stopped=False, completed=False, size=0))
    assert first.scheduling_held and second.scheduling_held
    coord.on_transfer_event(event_for(job))
    coord.on_transfer_event(event_for(job))  # replay must not duplicate refs
    assert block.block_hash == key.block_hash
    for waiter in (first, second):
        assert not waiter.scheduling_held
        coord.resume_request(waiter)
        blocks = manager.get_computed_blocks(waiter.prompt_token_ids)
        coord.after_lookup(waiter)
        assert [b.block_id for b in blocks] == [block.block_id]
    assert not coord.jobs
    assert block.ref_cnt == 2
    manager.release_blocks_by_ids([block.block_id, block.block_id])
    assert block.ref_cnt == 0


def test_last_waiter_cancel_and_repeated_close_cannot_release_native_target():
    coord, manager = coordinator()
    req = request()
    seed_ready(coord, manager, req)
    coord.lookup_after_hbm(req, [], 128)
    job = coord.next_transfer()
    block = manager.blocks[job.page_ids[0]]
    coord.cancel_request(*coord._request_key(req))
    assert block.ref_cnt == 1 and block.block_hash is None
    for _ in range(2):
        with pytest.raises(RuntimeError, match="obligations"):
            coord.close()
        assert block.ref_cnt == 1
    coord.on_transfer_event(event_for(job))
    assert block.ref_cnt == 0 and block.block_hash is None
    assert not coord.jobs


@pytest.mark.parametrize("case", ["failure", "old-epoch", "wrong-ref", "unknown"])
def test_load_failure_epoch_and_unknown_never_publish_or_free_early(case):
    coord, manager = coordinator()
    req = request()
    key = seed_ready(coord, manager, req)
    coord.lookup_after_hbm(req, [], 128)
    job = coord.next_transfer()
    block = manager.blocks[job.page_ids[0]]
    event = event_for(job)
    if case == "failure":
        event = event_for(job, TransferStatus.FAILED, completed=False, size=0)
    elif case == "old-epoch":
        coord.worker_epoch = "worker2"  # exact old job still must settle its own lease
    elif case == "wrong-ref":
        event = replace(event, entry_refs=(replace(REF, slot_generation=99),))
    else:
        event = replace(event, stopped=False)
    coord.on_transfer_event(event)
    assert block.block_hash is None
    if case in ("wrong-ref", "unknown"):
        assert coord.poisoned and block.ref_cnt == 1
        assert coord.jobs[job.job_id].state == LoadState.QUARANTINED
    else:
        coord.resume_request(req)
        assert manager.get_computed_blocks(req.prompt_token_ids) == []
        coord.after_lookup(req)
        assert block.ref_cnt == 0
        if case == "failure":
            assert key not in coord.ready


def test_headroom_failure_and_one_cancel_leave_other_waiter_usable():
    coord, manager = coordinator(num_blocks=1)
    req = request()
    seed_ready(coord, manager, req)
    assert coord.lookup_after_hbm(req, [], 128).kind == RestoreKind.MISS
    assert not coord.jobs and manager.num_free_blocks == 1
    coord, manager = coordinator()
    first, second = request("a"), request("b")
    seed_ready(coord, manager, first)
    for req in (first, second):
        coord.lookup_after_hbm(req, [], 128)
    job = coord.next_transfer()
    coord.cancel_request(*coord._request_key(first))
    coord.on_transfer_event(event_for(job))
    assert not second.scheduling_held
    coord.cancel_request(*coord._request_key(second))
    assert not coord.jobs


def test_step_snapshot_store_pin_and_unknown_worker_stop():
    coord, manager = coordinator()
    req = request()
    page = manager.allocate_blocks(1)[0]
    hashes = manager.compute_block_hashes(req.prompt_token_ids)
    manager.cache_block_ids([page.block_id], hashes, 0, 1)
    snapshot = (ScheduledPrefix(req.request_id, (page.block_id,), tuple(hashes), 129, 0, 128),)
    coord.retain_step(1, snapshot)
    assert page.ref_cnt == 2
    coord.on_confirmed_step(1, snapshot, {req.request_id: 128})
    assert page.ref_cnt == 2  # normal request + STORE, step released
    job = coord.next_transfer()
    assert job.operation == Operation.STORE
    manager.release_blocks_by_ids([page.block_id])
    assert page.ref_cnt == 1
    coord.on_transfer_event(event_for(job))
    assert page.ref_cnt == 0 and coord.ready
    # Retain and timeout a separate model dispatch. No process boolean can
    # discharge this pin; only the future same-owner drain receipt may do so.
    live_page = manager.allocate_blocks(1)[0]
    lost = (ScheduledPrefix("lost", (live_page.block_id,), (), 1, 0, 1),)
    coord.retain_step(2, lost)
    coord.discard_step(2)
    with pytest.raises(RuntimeError, match="proof"):
        coord.on_worker_stopped("worker1", True)
    assert live_page.ref_cnt == 2
    coord.on_worker_stopped("worker1", NativeStopProof("worker1", True, True))
    assert live_page.ref_cnt == 1
    manager.release_blocks_by_ids([live_page.block_id])


def test_held_waiting_request_does_not_block_unrelated_request():
    coord, manager = coordinator()
    sched = Scheduler(SchedulerConfig(max_num_scheduled_tokens=512), manager)
    sched._native_ddr = coord
    held, unrelated = request("held"), request("unrelated", 3)
    seed_ready(coord, manager, held)
    sched.add_request(held)
    sched.add_request(unrelated)
    output = sched.schedule()
    assert held.scheduling_held
    assert [sr.request.request_id for sr in output.scheduled_requests] == ["unrelated"]
    job = coord.next_transfer()
    coord.on_transfer_event(event_for(job))
    output = sched.schedule()
    restored = next(sr for sr in output.scheduled_requests if sr.request.request_id == "held")
    assert restored.num_computed_tokens == 128
    assert restored.block_ids[0] == job.page_ids[0]
    assert not coord.waiters and not coord.jobs


def test_duplicate_pending_submit_never_claims_stopped_or_submits_twice():
    module, backend = worker()
    req = transfer()
    module.submit_transfer(req)
    replay = module.submit_transfer(req)
    assert replay.status == TransferStatus.ACCEPTED and not replay.stopped
    conflict = module.submit_transfer(replace(req, page_ids=()))
    assert conflict.status == TransferStatus.FAILED and not conflict.stopped
    final = complete(module, backend)
    assert module.submit_transfer(req) == final
    assert len(backend.submissions) == 1


def test_idle_worker_emits_terminal_on_aux_queue_without_a_model_step():
    import queue
    import threading
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    from pypto_serving.serving.server.ipc import ShutdownCommand

    module, backend = worker()
    process = WorkerProcess.__new__(WorkerProcess)
    process._native_ddr_worker = module
    process.input_queue, process.output_queue = queue.Queue(), queue.Queue()
    process.native_ddr_output_queue = queue.Queue()
    thread = threading.Thread(target=process._serial_busy_loop, name="test-host-ddr-idle")
    thread.start()
    try:
        process.input_queue.put(encode_command(NativeDdrCommand(operation="submit", request=transfer())))
        accepted = decode_native_ddr_message(process.native_ddr_output_queue.get(timeout=2)).event
        assert accepted.status == TransferStatus.ACCEPTED
        backend.handle.ready = True
        final = decode_native_ddr_message(process.native_ddr_output_queue.get(timeout=2)).event
        assert final.status == TransferStatus.SUCCEEDED
        assert process.output_queue.empty()
    finally:
        process.input_queue.put(encode_command(ShutdownCommand()))
        thread.join(timeout=2)
        assert not thread.is_alive()
        module.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_spawn_allocates_no_auxiliary_queue_when_disabled(monkeypatch, enabled):
    import queue
    from pypto_serving.serving.server import serving_worker

    sizes = []
    class Context:
        def Queue(self, maxsize=0):
            sizes.append(maxsize)
            return queue.Queue(maxsize)

        def Array(self, kind, count):
            return [0] * count

        def Event(self):
            return object()

        def Process(self, **kwargs):
            return SimpleNamespace(start=lambda: None, **kwargs)

    monkeypatch.setattr(serving_worker.mp, "get_context", lambda method: Context())
    runtime = RuntimeConfig(native_ddr=CONFIG if enabled else NativeDdrPrefixConfig())
    process, *_ = serving_worker.spawn_worker(EngineConfig(runtime_config=runtime))
    assert sizes == ([0, 0, 0, 16] if enabled else [0, 0, 0])
    assert hasattr(process, "native_ddr_output_queue") is enabled


def test_enabled_backend_fails_before_executor_construction():
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    process = WorkerProcess.__new__(WorkerProcess)
    process.config = SimpleNamespace(resolve_runtime_config=lambda: RuntimeConfig(native_ddr=CONFIG))
    with pytest.raises(NativeDdrUnavailable, match="M3 seam unresolved"):
        process.init_device_and_model()
    assert not hasattr(process, "executor")


@pytest.mark.parametrize("flags,valid", [([], False), (["--block-size", "64"], False),
                                       (["--dtype", "float32"], False),
                                       (["--no-enable-prefix-caching"], False),
                                       (["--native-ddr-kv-bytes", "20971520"], True)])
def test_cli_native_config_without_reading_model_or_device(flags, valid):
    from pypto_serving.cli.main import build_parser, _build_native_ddr_config
    args = build_parser().parse_args(["--model", "not-a-model-load", "--enable-native-ddr-kv", *flags])
    if valid:
        assert _build_native_ddr_config(args).enabled
    else:
        with pytest.raises(ValueError):
            _build_native_ddr_config(args)
    disabled = build_parser().parse_args(["--model", "not-a-model-load", "--no-enable-native-ddr-kv"])
    assert _build_native_ddr_config(disabled) == NativeDdrPrefixConfig()


def test_partial_hbm_prefix_lookup_references_have_exact_waiter_ownership():
    coord, manager = coordinator()
    first, second = request("a", 257), request("b", 257)
    hashes = manager.compute_block_hashes(first.prompt_token_ids)
    page = manager.allocate_blocks(1)[0]
    manager.cache_block_ids([page.block_id], hashes, 0, 1)
    manager.release_blocks_by_ids([page.block_id])
    key = PrefixKey(coord.namespace, hashes[1])
    coord.ready[key] = REF
    for req in (first, second):
        local = manager.get_computed_blocks(req.prompt_token_ids)
        assert coord.lookup_after_hbm(req, local, 256).kind == RestoreKind.WAIT
    assert page.ref_cnt == 2
    coord.cancel_request(*coord._request_key(first))
    assert page.ref_cnt == 1
    job = coord.next_transfer()
    coord.on_transfer_event(event_for(job))
    coord.resume_request(second)
    assert page.ref_cnt == 0
    blocks = manager.get_computed_blocks(second.prompt_token_ids)
    coord.after_lookup(second)
    assert len(blocks) == 2 and page.ref_cnt == 1
    manager.release_cached_blocks(blocks)
    assert page.ref_cnt == 0


def test_engine_enabled_snapshot_and_aux_event_hook_only_use_host_queues():
    import asyncio
    import queue

    core = ReplicaEngineCore(EngineConfig(enable_prefix_cache=True, async_scheduling=False,
        runtime_config=RuntimeConfig(page_size=128, max_prefill_tokens_per_request=128)), SimpleNamespace())
    core.kv_cache_manager.initialize(core._runtime, num_blocks=12)
    coord = DdrPrefixCoordinator(core.kv_cache_manager, namespace_digest=b'n' * 32,
                                 worker_epoch="worker1", max_entries=8)
    core._native_ddr = coord
    core.scheduler._native_ddr = coord
    core._input_queue, core._output_queue, core._native_ddr_output_queue = queue.Queue(), queue.Queue(), queue.Queue()
    req = request()
    core.scheduler.add_request(req)
    assert core._try_dispatch_step()
    assert coord.steps and not coord.jobs  # optimistic dispatch cannot STORE
    command = decode_command(core._input_queue.get_nowait())
    # Host protocol fixture: a completed prefill has no final sampled token in
    # this partial chunk, so no tokenizer/model/consumer is exercised.
    core._output_queue.put(encode_result(StepResult(step_id=command.step_id, new_tokens={})))
    assert asyncio.run(core._await_and_apply_oldest())
    assert not coord.steps and coord.jobs
    core._pump_native_ddr()
    transfer_cmd = decode_command(core._input_queue.get_nowait())
    assert isinstance(transfer_cmd, NativeDdrCommand)
    core._native_ddr_output_queue.put(encode_native_ddr_message(
        NativeDdrMessage("transfer", event=event_for(transfer_cmd.request))))
    core._pump_native_ddr()
    assert coord.ready and not coord.jobs
    assert core._output_queue.empty()


def test_unacknowledged_receipts_are_bounded_and_cannot_forget_pending_native_writes():
    from pypto_serving.serving.native_ddr.replay import MAX_UNACKNOWLEDGED
    module, backend = worker()
    first = None
    for number in range(MAX_UNACKNOWLEDGED):
        ref = DdrEntryRef("worker1", f"entry{number}", number + 1)
        req = transfer(job=f"receipt{number}", ref=ref)
        if number == 0:
            first = req
        assert module.submit_transfer(req).status == TransferStatus.ACCEPTED
        complete(module, backend)
        module.evict(ref)
    event = module.submit_transfer(transfer(job="over-budget", ref=replace(REF, slot_generation=10000)))
    assert event.status == TransferStatus.REJECTED
    assert "unacknowledged delivery window full" in event.error
    assert len(module.transfer_replay.records) == MAX_UNACKNOWLEDGED
    assert module.submit_transfer(first).status == TransferStatus.SUCCEEDED
    assert len(backend.submissions) == MAX_UNACKNOWLEDGED

    module, backend = worker()
    for number in range(2):
        req = transfer(job=f"slot{number}", ref=replace(REF, entry_id=str(number), slot_generation=number + 1))
        module.submit_transfer(req)
        complete(module, backend)
    event = module.submit_transfer(transfer(job="over-capacity", ref=replace(REF, slot_generation=3)))
    assert event.status == TransferStatus.REJECTED and "capacity" in event.error
    assert len(backend.slots) == 2
    module.close()


def test_failed_submit_with_accepted_handle_retains_then_settles_actual_handle():
    module, backend = worker()
    handle = FakeHandle()
    backend.error = NativeSubmissionError("failure after acceptance", handle=handle)
    event = module.submit_transfer(transfer())
    assert not event.stopped and module.pending.handle is handle
    assert module.poll_completed() == ()
    handle.ready = True
    event = module.poll_completed()[0]
    assert event.stopped and module.poisoned
    with pytest.raises(RuntimeError, match="obligations"):
        module.close()  # native stopped, owner is still quarantined


@pytest.mark.parametrize("pipeline", [False, True])
def test_worker_without_pending_native_job_keeps_blocking_queue_semantics(pipeline):
    import queue
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    from pypto_serving.serving.server.ipc import ShutdownCommand

    class BlockingOnlyQueue(queue.Queue):
        def get(self, *args, **kwargs):
            assert not args and not kwargs  # no idle periodic query/poll
            cmd = ShutdownCommand()
            return (cmd, None, {}) if pipeline else encode_command(cmd)

    process = WorkerProcess.__new__(WorkerProcess)
    module, _ = worker()
    process._native_ddr_worker = module
    process._poll_native_ddr = lambda: pytest.fail("idle owner must not poll")
    if pipeline:
        output = queue.Queue()
        process._device_execution_loop(BlockingOnlyQueue(), output)
        assert isinstance(output.get_nowait(), ShutdownCommand)
    else:
        process.input_queue = BlockingOnlyQueue()
        process._serial_busy_loop()
    module.close()


def test_quarantined_engine_rejects_before_scheduler_or_model_admission():
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    coord, _ = coordinator()
    coord.poisoned = True
    core._native_ddr = coord
    # No scheduler/worker exists: rejection must precede touching either.
    with pytest.raises(RuntimeError, match="quarantined"):
        core._try_dispatch_step()



def close_coordinator_with_host_receipts(coord):
    """Host-only control fixture: acknowledge actual queued release commands explicitly."""
    try:
        coord.close()
    except RuntimeError:
        assert not coord.jobs and not coord.steps
    while (cmd := coord.next_control()) is not None:
        coord.on_control_event(DdrControlEvent(cmd.command_id, cmd.worker_epoch, cmd.operation,
                                              cmd.entry_ref, cmd.operation == "evict",
                                              delivery_sequence=cmd.delivery_sequence))
    assert not coord.controls and not coord.retiring
    coord.close()

def _seed_full_prefix(coord, manager, req, count):
    keys = [PrefixKey(coord.namespace, block_hash)
            for block_hash in manager.compute_block_hashes(req.prompt_token_ids)[:count]]
    for index, key in enumerate(keys):
        coord.ready[key] = DdrEntryRef(coord.worker_epoch, f"{req.request_id}:{index}", len(coord.ready) + 1)
    return keys


@pytest.mark.parametrize("chunk", [1, 128, 129, 256, 512])
@pytest.mark.parametrize("requests,pages", [(2, 5), (3, 7)])
def test_completed_multi_block_restores_allow_actual_suffix_admission(chunk, requests, pages):
    coord, manager = coordinator(num_blocks=pages)
    sched = Scheduler(SchedulerConfig(max_num_scheduled_tokens=512,
                                      max_prefill_tokens_per_request=chunk), manager)
    sched._native_ddr = coord
    for index in range(requests):
        req = request(str(index), 385)
        req.prompt_token_ids = [token + index * 10000 for token in req.prompt_token_ids]
        _seed_full_prefix(coord, manager, req, 2)
        sched.add_request(req)
    assert not sched.schedule().scheduled_requests
    assert len(coord.jobs) == requests * 2
    transfers = []
    while (job := coord.next_transfer()) is not None:
        transfers.append(job)
        coord.on_transfer_event(event_for(job))
    assert len(transfers) == requests * 2
    assert coord.active_job is None
    # No pending native/model result could unstick this state: schedule itself
    # must drop reclaimable loader pins and make progress, without user abort.
    output = sched.schedule()
    assert output.scheduled_requests
    first = output.scheduled_requests[0]
    assert first.request.request_id == "0"
    assert first.num_computed_tokens == 256
    assert first.num_new_tokens == min(chunk, 129)
    assert len(first.block_ids) == 2 + (first.num_new_tokens + 127) // 128
    for req in list(sched.requests.values()):
        sched.abort_request(req.request_id)
    close_coordinator_with_host_receipts(coord)
    assert not coord.loads_by_key and not coord.waiters and not coord.jobs
    assert manager.num_free_blocks == pages
    assert all(page.ref_cnt == 0 for page in manager.blocks)


@pytest.mark.parametrize("chunk", [129, 256])
def test_restore_capacity_change_can_fall_back_to_cold_without_losing_other_pins(chunk):
    coord, manager = coordinator(num_blocks=4)
    sched = Scheduler(SchedulerConfig(max_num_scheduled_tokens=512,
                                      max_prefill_tokens_per_request=chunk), manager)
    sched._native_ddr = coord
    req = request(length=385)
    _seed_full_prefix(coord, manager, req, 2)
    sched.add_request(req)
    assert not sched.schedule().scheduled_requests
    # Another normal owner consumes capacity AFTER reserve. A restored suffix
    # now needs four pages, while a cold chunk needs only two and can progress.
    other_blocks = manager.allocate_blocks(1)
    assert other_blocks is not None
    other_page = other_blocks[0]
    while (job := coord.next_transfer()) is not None:
        coord.on_transfer_event(event_for(job))
    output = sched.schedule()
    assert len(output.scheduled_requests) == 1
    scheduled = output.scheduled_requests[0]
    assert scheduled.num_computed_tokens == 0
    assert scheduled.num_new_tokens == chunk
    assert len(scheduled.block_ids) == 2
    assert other_page.block_id not in scheduled.block_ids and other_page.ref_cnt == 1
    assert not coord.jobs and not coord.loads_by_key and not coord.waiters
    sched.abort_request(req.request_id)
    manager.release_blocks_by_ids([other_page.block_id])
    close_coordinator_with_host_receipts(coord)
    assert manager.num_free_blocks == 4


def _next_load_for_key(coord, key):
    # CPU fault-injection ordering, independent of production queue preference.
    job_id = coord.loads_by_key[(coord.worker_epoch, key)]
    coord.outbound.remove(job_id)
    coord.outbound.appendleft(job_id)
    job = coord.next_transfer()
    assert job.entry_keys == (key,)
    return job


@pytest.mark.parametrize("completed_index", [0, 1])
@pytest.mark.parametrize("cancel", ["none", "first", "second", "all"])
@pytest.mark.parametrize("failed", [False, True])
def test_partial_multi_block_completion_late_join_and_cancel_balance_refs(completed_index, cancel, failed):
    coord, manager = coordinator()
    first, second = request("first", 257), request("second", 257)
    keys = _seed_full_prefix(coord, manager, first, 2)
    assert coord.lookup_after_hbm(first, [], 256).kind == RestoreKind.WAIT
    original_jobs = set(coord.jobs)
    original_pages = {job.lease.block_ids for job in coord.jobs.values()}
    done = _next_load_for_key(coord, keys[completed_index])
    coord.on_transfer_event(event_for(done))
    coord.on_transfer_event(event_for(done))
    assert first.scheduling_held  # the other block has not completed
    local = manager.get_computed_blocks(second.prompt_token_ids)
    assert len(local) == (1 if completed_index == 0 else 0)
    assert coord.lookup_after_hbm(second, local, 256).kind == RestoreKind.WAIT
    assert set(coord.jobs) == original_jobs
    assert {job.lease.block_ids for job in coord.jobs.values()} == original_pages
    assert len(coord.loads_by_key) == 2
    pending_index = 1 - completed_index
    pending = _next_load_for_key(coord, keys[pending_index])
    assert pending is not None
    # Actual suffix pressure must not drop a partially settled waiter, including
    # the committed block shared with its still-native-owned sibling.
    refs = [page.ref_cnt for page in manager.blocks]
    assert not coord.release_completed_restores()
    assert [page.ref_cnt for page in manager.blocks] == refs
    survivors = []
    for name, req in (("first", first), ("second", second)):
        if cancel in (name, "all"):
            coord.cancel_request(*coord._request_key(req))
        else:
            survivors.append(req)
    assert manager.blocks[pending.page_ids[0]].ref_cnt == 1
    final = event_for(pending, TransferStatus.FAILED, completed=False, size=0) if failed else event_for(pending)
    coord.on_transfer_event(final)
    coord.on_transfer_event(final)
    assert coord.next_transfer() is None  # exactly two copies; never a third target
    acquired = []
    for req in survivors:
        assert not req.scheduling_held
        coord.resume_request(req)
        blocks = manager.get_computed_blocks(req.prompt_token_ids)
        coord.after_lookup(req)
        assert len(blocks) == (pending_index if failed else 2)
        acquired.extend(block.block_id for block in blocks)
    assert not coord.jobs and not coord.loads_by_key and not coord.waiters
    assert sum(page.ref_cnt for page in manager.blocks) == len(acquired)
    manager.release_blocks_by_ids(acquired)
    coord.on_transfer_event(event_for(done))  # replay after final retirement
    coord.on_transfer_event(final)
    close_coordinator_with_host_receipts(coord)
    assert manager.num_free_blocks == 12
    assert all(page.ref_cnt == 0 for page in manager.blocks)


@pytest.mark.parametrize("state", ["reserved", "loading", "quarantined"])
def test_suffix_pressure_cannot_relinquish_unsettled_native_target(state):
    coord, manager = coordinator()
    req = request()
    seed_ready(coord, manager, req)
    coord.lookup_after_hbm(req, [], 128)
    job = None if state == "reserved" else coord.next_transfer()
    if state == "quarantined":
        coord.on_transfer_event(event_for(job, stopped=False))
    assert not coord.release_completed_restores()
    assert len(coord.jobs) == 1 and len(coord.waiters) == 1
    assert sum(page.ref_cnt for page in manager.blocks) == 1
    coord.cancel_request(*coord._request_key(req))
    if job is not None:
        assert sum(page.ref_cnt for page in manager.blocks) == 1
        coord.on_transfer_event(event_for(job))
    close_coordinator_with_host_receipts(coord)
    assert not coord.jobs and not coord.loads_by_key
    assert manager.num_free_blocks == 12


def test_joining_committed_canonical_target_does_not_wait_for_another_event():
    coord, manager = coordinator()
    first, second = request("first"), request("second")
    key = seed_ready(coord, manager, first)
    coord.lookup_after_hbm(first, [], 128)
    job = coord.next_transfer()
    coord.on_transfer_event(event_for(job))
    # Explicit reserve API with a previously acquired incomplete lookup: this
    # joins the committed target and must not install a permanently held waiter.
    assert coord.reserve_restore(second, [key], []).kind == RestoreKind.WAIT
    assert not second.scheduling_held
    assert len(coord.jobs) == 1 and coord.next_transfer() is None
    for req in (first, second):
        coord.resume_request(req)
        blocks = manager.get_computed_blocks(req.prompt_token_ids)
        coord.after_lookup(req)
        assert len(blocks) == 1
        manager.release_cached_blocks(blocks)
    assert not coord.loads_by_key and not coord.jobs
    close_coordinator_with_host_receipts(coord)
    assert manager.num_free_blocks == 12


def test_startup_waits_for_delayed_auxiliary_metadata_and_validates_capacity():
    import asyncio
    import hashlib
    import json
    import queue
    import threading
    from pypto_serving.serving.native_ddr.types import DdrStartup

    core = ReplicaEngineCore(EngineConfig(model_id="qwen", runtime_config=RuntimeConfig(
        page_size=128, native_ddr=CONFIG)), SimpleNamespace())
    core.kv_cache_manager.initialize(core._runtime, num_blocks=16)
    core._native_ddr_output_queue = queue.Queue()
    descriptor = json.dumps({"worker_epoch": "worker1", "scope": "single-loaded-owner-incarnation-only",
                             "model": {"model_id": "qwen"}, "layout": asdict(LAYOUT),
                             "tensor_parallel": 1}).encode()
    info = DdrStartup("worker1", hashlib.sha256(descriptor).digest(), LAYOUT,
                      CONFIG.capacity_bytes, 2, descriptor)
    producer = threading.Timer(.03, lambda: core._native_ddr_output_queue.put(
        encode_native_ddr_message(NativeDdrMessage("startup", startup=info))))
    producer.start()
    try:
        asyncio.run(core._install_native_ddr(SimpleNamespace(is_alive=lambda: True), 16))
    finally:
        producer.join()
    assert core._native_ddr.worker_epoch == "worker1"
    assert core.scheduler._native_ddr is core._native_ddr
    assert core._native_ddr.max_entries == 2
    core._native_ddr_output_queue.put(encode_native_ddr_message(
        NativeDdrMessage("startup", startup=replace(info, max_entries=3))))
    with pytest.raises(RuntimeError, match="mismatch"):
        asyncio.run(core._install_native_ddr(SimpleNamespace(is_alive=lambda: True), 16))


def test_store_waits_for_worker_eviction_release_instead_of_overselling():
    coord, manager = coordinator(capacity=1)
    old_key = seed_ready(coord, manager, request("old"))
    req = Request(request_id="new", prompt_token_ids=[999] + list(range(128)), max_new_tokens=2)
    page = manager.allocate_blocks(1)[0]
    hashes = manager.compute_block_hashes(req.prompt_token_ids)
    manager.cache_block_ids([page.block_id], hashes, 0, 1)
    snapshot = (ScheduledPrefix(req.request_id, (page.block_id,), tuple(hashes), 129, 0, 128),)
    coord.retain_step(1, snapshot)
    coord.on_confirmed_step(1, snapshot, {req.request_id: 128})
    manager.release_blocks_by_ids([page.block_id])
    assert old_key not in coord.ready and coord.retiring and page.ref_cnt == 1
    assert coord.next_transfer() is None
    cmd = coord.next_control()
    assert cmd.operation == "evict"
    coord.on_control_event(DdrControlEvent(cmd.command_id, cmd.worker_epoch, "evict", cmd.entry_ref, True,
                                          delivery_sequence=cmd.delivery_sequence))
    job = coord.next_transfer()
    assert job.operation == Operation.STORE
    coord.on_transfer_event(event_for(job))
    assert page.ref_cnt == 0 and len(coord.ready) == 1
    close_coordinator_with_host_receipts(coord)


def test_eviction_with_pending_load_retains_reader_until_actual_completion():
    coord, manager = coordinator(capacity=2)
    req = request()
    key = seed_ready(coord, manager, req)
    module, backend = worker()
    store = coord.transfer_delivery.stamp(transfer(key=key))
    module.submit_transfer(store)
    coord.transfer_delivery.consume(complete(module, backend).delivery_sequence)
    coord.lookup_after_hbm(req, [], 128)
    load = coord.next_transfer()
    assert module.submit_transfer(load).status == TransferStatus.ACCEPTED
    assert coord.evict(key)
    cmd = coord.next_control()
    module.evict(cmd.entry_ref)
    assert module.entries[REF.entry_id].readers == 1 and backend.slots
    terminal = complete(module, backend)
    assert terminal.status == TransferStatus.SUCCEEDED and not backend.slots
    coord.on_transfer_event(terminal)
    assert coord.retiring  # worker terminal and physical-release receipt are distinct
    coord.on_control_event(DdrControlEvent(cmd.command_id, cmd.worker_epoch, "evict", cmd.entry_ref, True,
                                          delivery_sequence=cmd.delivery_sequence))
    coord.resume_request(req)
    blocks = manager.get_computed_blocks(req.prompt_token_ids)
    coord.after_lookup(req)
    assert len(blocks) == 1 and not coord.jobs
    manager.release_cached_blocks(blocks)
    close_coordinator_with_host_receipts(coord)
    module.close()


@pytest.mark.parametrize("owner_error", [False, True])
def test_stop_proof_follows_strict_owner_close_not_module_drain(owner_error):
    import queue
    from pypto_serving.serving.server.serving_worker import WorkerProcess

    calls = []
    def owner_close(_model):
        calls.append("owner_close")
        if owner_error:
            raise RuntimeError("injected owner cleanup failure")
    process = WorkerProcess.__new__(WorkerProcess)
    process.config = SimpleNamespace(model_id="qwen")
    process._native_ddr_worker = SimpleNamespace(pending=None, poisoned=False, worker_epoch="worker1",
        close=lambda: calls.append("module_close"), entries={})
    process._native_ddr_backend = SimpleNamespace(close=lambda: calls.append("backend_close"))
    process._native_ddr_controls = {}
    process._native_ddr_control_replay = None
    process._native_ddr_drain_commands = {}
    process._native_ddr_stop_sent = False
    process.native_ddr_output_queue = queue.Queue()
    process.executor = SimpleNamespace(close_native_ddr_owner=owner_close,
                                       close=lambda: calls.append("executor_close"))
    if owner_error:
        with pytest.raises(RuntimeError, match="owner cleanup"):
            process.close()
        assert process.native_ddr_output_queue.empty()
    else:
        process.close()
        proof = decode_native_ddr_message(process.native_ddr_output_queue.get_nowait()).proof
        assert proof == NativeStopProof("worker1", True, True)
        assert calls == ["module_close", "backend_close", "owner_close", "executor_close"]


def test_asset_namespace_evidence_tracks_content_and_rejects_escaping_shards(tmp_path):
    import json
    import struct
    from pypto_serving.serving.native_ddr.compatibility import capture_assets

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "weights.safetensors"}}))
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}" + b"payload")
    (tmp_path / "tokenizer.json").write_text('{"vocab":1}')
    first = capture_assets(tmp_path)
    assert "payload not rescanned" in first["files"]["weights.safetensors"]["evidence"]
    (tmp_path / "tokenizer.json").write_text('{"vocab":2}')
    second = capture_assets(tmp_path)
    assert first["files"]["tokenizer.json"]["sha256"] != second["files"]["tokenizer.json"]["sha256"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../escape.safetensors"}}))
    with pytest.raises((ValueError, FileNotFoundError)):
        capture_assets(tmp_path)


def test_confirmed_decode_cannot_blindly_replay_cancelled_prefix_store():
    coord, manager = coordinator()
    req = request()
    page = manager.allocate_blocks(1)[0]
    hashes = tuple(manager.compute_block_hashes(req.prompt_token_ids))
    manager.cache_block_ids([page.block_id], list(hashes), 0, 1)
    first = (ScheduledPrefix(req.request_id, (page.block_id,), hashes, 129, 0, 128),)
    coord.retain_step(1, first)
    coord.on_confirmed_step(1, first, {req.request_id: 128})
    job = coord.next_transfer()
    coord.jobs[job.job_id].discard = True
    coord.on_transfer_event(event_for(job, TransferStatus.CANCELLED))
    later = (ScheduledPrefix(req.request_id, (page.block_id,), hashes, 129, 128, 1),)
    coord.retain_step(2, later)
    coord.on_confirmed_step(2, later, {req.request_id: 129})
    assert not coord.stores_by_key and not coord.jobs
    manager.release_blocks_by_ids([page.block_id])
    close_coordinator_with_host_receipts(coord)


def test_production_worker_rejects_foreign_namespace_before_reserving_or_submitting():
    backend = FakeBackend()
    module = DdrPrefixWorker(CONFIG, LAYOUT, "worker1", backend, namespace_digest=b"q" * 32)
    event = module.submit_transfer(transfer())
    assert event.status == TransferStatus.REJECTED and event.stopped and "namespace" in event.error
    assert not backend.slots and not backend.submissions
    key = PrefixKey(b"q" * 32, KEY.block_hash)
    assert module.submit_transfer(transfer(key=key)).status == TransferStatus.ACCEPTED
    complete(module, backend)
    module.close()


def test_receipt_filter_exception_keeps_real_handle_obligations_until_explicit_result():
    from pypto_serving.model.qwen.native_ddr_transfer import NativeDdrRun
    calls = []
    slot = SimpleNamespace(native_complete=False)
    backend = SimpleNamespace(worker=object(), keys=object(), values=object(),
        slots=SimpleNamespace(page_ids=object(), busy=slot), layout=LAYOUT,
        record=lambda *args, **kwargs: calls.append(args[0]), pending=None)
    def reject(_request, _result):
        raise RuntimeError("injected host receipt exception")
    backend.completion_filter = reject
    native = SimpleNamespace(done=True, result=lambda **kwargs: calls.append("native_result"))
    handle = NativeDdrRun(backend, transfer(), slot, native)
    backend.pending = handle
    with pytest.raises(RuntimeError, match="receipt exception"):
        handle.result()
    assert handle._result is None and handle._keepalive is not None
    assert backend.pending is handle and backend.slots.busy is slot
    backend.completion_filter = None  # explicit host-fixture repair, not production recovery
    assert handle.result() == NativeCompletion(LAYOUT.block_bytes, True, True)
    assert backend.pending is None and backend.slots.busy is None and handle._keepalive is None


def test_same_owner_reuses_slots_for_256_cycles_and_retires_old_load_before_backend():
    from pypto_serving.serving.native_ddr.replay import DeliverySender, REPLAY_WINDOW
    module, backend = worker()
    sender = DeliverySender()

    def run(req):
        req = sender.stamp(req)
        assert module.submit_transfer(req).status == TransferStatus.ACCEPTED
        done = complete(module, backend)
        assert done.status == TransferStatus.SUCCEEDED
        sender.consume(done.delivery_sequence)
        assert len(module.transfer_replay.records) <= REPLAY_WINDOW + 1
        return req, done

    run(transfer(job="persistent-store"))
    old_load, _ = run(transfer(Operation.LOAD, job="persistent-load", page=7))
    # Keep the original exact DDR entry Ready: stale rejection must be due to
    # retirement, not absence/generation validation masking another native write.
    for number in range(256):
        ref = DdrEntryRef("worker1", f"cycle{number}", number + 2)
        run(transfer(job=f"store{number}", ref=ref))
        recent, terminal = run(transfer(Operation.LOAD, job=f"load{number}", ref=ref, page=7))
        module.evict(ref)
    assert REF.entry_id in module.entries and module.entries[REF.entry_id].state == EntryState.READY
    assert module.transfer_replay.retired_through > old_load.delivery_sequence
    calls = len(backend.submissions)
    assert module.submit_transfer(recent) == terminal
    stale = module.submit_transfer(old_load)
    assert stale.status == TransferStatus.REJECTED and stale.stopped and "retired" in stale.error
    assert len(backend.submissions) == calls == 514
    conflict = module.submit_transfer(replace(recent, page_ids=(8,)))
    assert conflict.status == TransferStatus.FAILED and not conflict.stopped
    assert len(backend.slots) == 1 and not module.poisoned
    module.close()


@pytest.mark.parametrize("state", ["pending", "unknown", "protected", "partial", "cancelled"])
def test_acknowledgement_never_retires_unstopped_or_quarantined_native_obligations(state):
    from pypto_serving.serving.native_ddr.replay import DeliverySender
    module, backend = worker()
    sender = DeliverySender()
    req = sender.stamp(transfer())
    if state == "unknown":
        backend.error = RuntimeError("unknown native submission")
    module.submit_transfer(req)
    if state == "protected":
        backend.handle.error = RuntimeError("unknown result cleanup")
        complete(module, backend)
    if state == "partial":
        backend.handle.outcome = NativeCompletion(262144, True, True)
        done = complete(module, backend)
    if state == "cancelled":
        module.cancel(req.job_id)
        done = complete(module, backend)
    fresh = replace(transfer(job="next", ref=replace(REF, entry_id="next", slot_generation=2)),
                    delivery_sequence=2, retire_through=1)
    if state in ("partial", "cancelled"):
        assert done.stopped
        sender.consume(done.delivery_sequence)
        assert module.submit_transfer(fresh).status == TransferStatus.ACCEPTED
        complete(module, backend)
        module.close()
    else:
        event = module.submit_transfer(fresh)
        assert not event.stopped and "retained" in event.error
        assert module.transfer_replay.acknowledged == 0
        assert module.transfer_replay.retired_through == 0
        assert module.pending is not None and backend.slots
        assert len(module.transfer_replay.records) == 1


def _control_process(module, backend):
    import queue
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    process = WorkerProcess.__new__(WorkerProcess)
    process._native_ddr_worker = module
    process._native_ddr_backend = backend
    process._native_ddr_control_replay = None
    process._native_ddr_controls = {}
    process._native_ddr_drain_commands = {}
    process.native_ddr_output_queue = queue.Queue()
    backend.close = lambda: None
    return process


def _control_wire(cmd):
    return decode_command(encode_command(NativeDdrCommand(**asdict(cmd))))


def test_control_receipts_continue_after_4096_and_old_evict_cannot_release_new_generation():
    from pypto_serving.serving.native_ddr.replay import DeliverySender, REPLAY_WINDOW
    from pypto_serving.serving.native_ddr.types import DdrControlRequest
    module, backend = worker()
    data_sender, sender = DeliverySender(), DeliverySender()
    store = data_sender.stamp(transfer())
    module.submit_transfer(store)
    data_sender.consume(complete(module, backend).delivery_sequence)
    process = _control_process(module, backend)

    def invoke(cmd):
        process._handle_native_ddr_command(_control_wire(cmd))
        return decode_native_ddr_message(process.native_ddr_output_queue.get_nowait()).control

    evict = sender.stamp(DdrControlRequest("old-evict", "worker1", "evict", entry_ref=REF))
    sender.consume(invoke(evict).delivery_sequence)
    new_ref = replace(REF, slot_generation=2)
    req = data_sender.stamp(transfer(job="replacement", ref=new_ref))
    module.submit_transfer(req)
    data_sender.consume(complete(module, backend).delivery_sequence)
    for number in range(5000):
        cmd = sender.stamp(DdrControlRequest(f"cancel{number}", "worker1", "cancel", job_id="already-finished"))
        event = invoke(cmd)
        assert event.error is None
        sender.consume(event.delivery_sequence)
        assert len(process._native_ddr_control_replay.records) <= REPLAY_WINDOW + 1
    assert process._native_ddr_control_replay.retired_through > evict.delivery_sequence
    assert invoke(evict).error.startswith("retired")
    assert module.entries[REF.entry_id].ref == new_ref and len(backend.slots) == 1
    assert invoke(cmd) == event
    assert "conflict" in invoke(replace(cmd, job_id="different")).error
    assert module.entries[REF.entry_id].state == EntryState.READY
    drain = sender.stamp(DdrControlRequest("drain", "worker1", "drain"))
    assert invoke(drain).released and not backend.slots and not module.entries
    assert not module.poisoned


def test_pending_control_eviction_blocks_ack_frontier_while_later_cancel_completes():
    from pypto_serving.serving.native_ddr.replay import DeliverySender
    from pypto_serving.serving.native_ddr.types import DdrControlRequest
    module, backend = worker()
    data_sender, sender = DeliverySender(), DeliverySender()
    store = data_sender.stamp(transfer())
    module.submit_transfer(store)
    data_sender.consume(complete(module, backend).delivery_sequence)
    load = data_sender.stamp(transfer(Operation.LOAD, job="pending-load", page=3))
    module.submit_transfer(load)
    process = _control_process(module, backend)
    evict = sender.stamp(DdrControlRequest("pending-evict", "worker1", "evict", entry_ref=REF))
    process._handle_native_ddr_command(_control_wire(evict))
    process._handle_native_ddr_command(_control_wire(evict))
    assert process.native_ddr_output_queue.empty() and len(backend.slots) == 1
    cancel = sender.stamp(DdrControlRequest("cancel", "worker1", "cancel", job_id=load.job_id))
    process._handle_native_ddr_command(_control_wire(cancel))
    response = decode_native_ddr_message(process.native_ddr_output_queue.get_nowait()).control
    sender.consume(response.delivery_sequence)
    assert sender.acknowledged == 0  # completion 2 cannot cross pending command 1
    forged = DdrControlRequest("forged", "worker1", "cancel", delivery_sequence=3, retire_through=2)
    process._handle_native_ddr_command(_control_wire(forged))
    error = decode_native_ddr_message(process.native_ddr_output_queue.get_nowait()).control
    assert "retained" in error.error and module.pending is not None and backend.slots
    backend.handle.ready = True
    process._poll_native_ddr()
    events = [decode_native_ddr_message(process.native_ddr_output_queue.get_nowait()) for _ in range(2)]
    assert events[0].event.status == TransferStatus.CANCELLED and events[0].event.stopped
    assert events[1].control.released
    sender.consume(events[1].control.delivery_sequence)
    assert sender.acknowledged == 2 and not backend.slots and not module.pending
    module.close()


def test_load_priority_stamps_actual_delivery_after_a_later_created_job_passes_store():
    coord, manager = coordinator()
    req = request("cold-store")
    req.prompt_token_ids = [99] + req.prompt_token_ids[1:]
    page = manager.allocate_blocks(1)[0]
    hashes = manager.compute_block_hashes(req.prompt_token_ids)
    manager.cache_block_ids([page.block_id], hashes, 0, 1)
    snapshot = (ScheduledPrefix(req.request_id, (page.block_id,), tuple(hashes), 129, 0, 128),)
    coord.retain_step(1, snapshot)
    coord.on_confirmed_step(1, snapshot, {req.request_id: 128})
    store_id = coord.outbound[0]
    ready_request = request("later-load")
    seed_ready(coord, manager, ready_request)
    coord.lookup_after_hbm(ready_request, [], 128)
    load = coord.next_transfer()
    assert load.operation == Operation.LOAD and load.delivery_sequence == 1
    assert int(load.job_id.rsplit(':', 1)[1]) > int(store_id.rsplit(':', 1)[1])
    coord.on_transfer_event(event_for(load))
    coord.cancel_request(*coord._request_key(ready_request))
    store = coord.next_transfer()
    assert store.job_id == store_id and store.delivery_sequence == 2 and store.retire_through == 1
    coord.on_transfer_event(event_for(store))
    manager.release_blocks_by_ids([page.block_id])
    close_coordinator_with_host_receipts(coord)


def test_sender_throttles_outstanding_commands_and_cannot_ack_through_gaps():
    from pypto_serving.serving.native_ddr.replay import DeliverySender, MAX_UNACKNOWLEDGED
    from pypto_serving.serving.native_ddr.types import DdrControlRequest
    sender = DeliverySender()
    for number in range(MAX_UNACKNOWLEDGED):
        sender.stamp(DdrControlRequest(str(number), "worker1", "cancel"))
    assert not sender.available
    for number in range(2, MAX_UNACKNOWLEDGED + 1):
        sender.consume(number)
    assert sender.acknowledged == 0 and not sender.available
    sender.consume(1)
    assert sender.acknowledged == MAX_UNACKNOWLEDGED and sender.available
    assert not sender._completed


def test_conflicting_invalid_payload_on_pending_sequence_never_claims_stopped():
    module, backend = worker()
    req = transfer()
    module.submit_transfer(req)
    conflict = module.submit_transfer(replace(req, page_ids=(1000,)))
    assert conflict.status == TransferStatus.FAILED and not conflict.stopped
    assert module.pending is not None and len(backend.submissions) == 1
    complete(module, backend)
    module.close()


def test_terminal_but_quarantined_owner_cannot_retire_its_protected_receipt():
    from pypto_serving.serving.native_ddr.replay import DeliverySender
    module, backend = worker()
    sender = DeliverySender()
    handle = FakeHandle()
    backend.error = NativeSubmissionError('ambiguous accepted submission', handle=handle)
    req = sender.stamp(transfer())
    module.submit_transfer(req)
    handle.ready = True
    done = module.poll_completed()[0]
    assert done.stopped and module.poisoned
    sender.consume(done.delivery_sequence)
    fresh = sender.stamp(transfer(job='new', ref=replace(REF, entry_id='new', slot_generation=2)))
    error = module.submit_transfer(fresh)
    assert not error.stopped and 'retained' in error.error
    assert module.transfer_replay.acknowledged == module.transfer_replay.retired_through == 0
    assert module.transfer_replay.records[1].protected
