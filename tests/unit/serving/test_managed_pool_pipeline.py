# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU control integration with fake UF/native responses; no DMA/model proof."""
from dataclasses import asdict, replace
import gc
from types import SimpleNamespace

import pytest

from pypto_serving.model.qwen.native_ddr_transfer import ManagedDdrSlots, NativeDdrRun
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.native_ddr.coordinator import DdrPrefixCoordinator
from pypto_serving.serving.native_ddr.managed_pool import (
    BLOCK_BYTES, GUARD_BYTES, CommittedRange, ManagedPoolAccess, prefix_content_id,
)
from pypto_serving.serving.native_ddr.types import (
    EntryState, LoadState, NativeDdrPrefixConfig, PrefixKey, QwenKvLayout, RestoreKind, TransferStatus,
)
from pypto_serving.serving.native_ddr.worker import DdrPrefixWorker
from pypto_serving.serving.sched.scheduler import Request
from pypto_serving.serving.server.ipc import (
    NativeDdrCommand, decode_command, decode_native_ddr_message, encode_command,
)
from tests.unit.serving.test_managed_pool_protocol import FakePort
from tests.unit.serving.test_native_ddr_host import _control_process


@pytest.fixture
def pipeline():
    port = FakePort()
    access = ManagedPoolAccess(port)
    slots = ManagedDdrSlots(5, access)
    layout = QwenKvLayout(40, 16, 128, 8, 128)
    config = NativeDdrPrefixConfig(True, BLOCK_BYTES * 5)
    manager = KvCacheManager(num_blocks=16, block_size=128, enable_prefix_cache=True)
    coord = DdrPrefixCoordinator(manager, namespace_digest=b'n' * 32, worker_epoch='worker2', max_entries=5,
                                  pool_identity=port.identity)
    backend = SimpleNamespace(is_managed=True, slots=slots, worker=object(), keys=object(), values=object(),
        reserve_slot=slots.reserve, release_slot=slots.release, layout=layout, pending=None,
        record=lambda *args, **kwargs: None, completion_filter=None)
    calls = []
    def submit(request, backing):
        slots.begin_native(request, backing)
        slots.busy = backing
        # This explicitly fake handle has no data plane. Only protocol ordering
        # and parent metadata/refcount publication are under test here.
        native = SimpleNamespace(done=True, result=lambda **_: calls.append('fake-result'))
        backend.pending = NativeDdrRun(backend, request, backing, native)
        return backend.pending
    backend.submit = submit
    module = DdrPrefixWorker(config, layout, 'worker2', backend, namespace_digest=coord.namespace)
    process = _control_process(module, backend)
    context = SimpleNamespace(port=port, access=access, slots=slots, manager=manager, coord=coord,
                               backend=backend, module=module, process=process, calls=calls)
    yield context
    # Fake fixture teardown even for deliberate unknown cases; this is never a
    # production close/clean receipt or evidence of device/pool cleanup.
    for backing in slots.leases.values():
        backing.tensor = None
    slots.pool_tensor = slots.page_ids = None
    gc.collect()
    access.buffer.release()
    port.buffer.close()


def import_prefix(context, length=257, count=2):
    req = Request(request_id='target', prompt_token_ids=list(range(length)), max_new_tokens=2)
    proofs = []
    for index, digest in enumerate(context.manager.compute_block_hashes(req.prompt_token_ids)[:count]):
        allocation = context.port.allocate_range(BLOCK_BYTES + GUARD_BYTES, 65536)
        proof = CommittedRange('pool', 'epoch', 'provider', allocation, allocation.byte_offset + GUARD_BYTES,
            BLOCK_BYTES, index + 1, prefix_content_id(PrefixKey(context.coord.namespace, digest)),
            f'external-op{index}', f'external-writer{index}')
        context.port.committed_ranges[allocation.block_id] = proof
        proofs.append(proof)
    context.coord.prepare_import_prefix(req.prompt_token_ids, proofs)
    while (command := context.coord.next_control()) is not None:
        wire = encode_command(NativeDdrCommand(**asdict(command)))
        context.process._handle_native_ddr_command(decode_command(wire))
        raw = context.process.native_ddr_output_queue.get_nowait()
        assert b'fake-capability' not in raw
        reply = decode_native_ddr_message(raw).control
        assert reply.imported == command.imported
        context.coord.on_control_event(reply)
    assert len(context.coord.ready) == len(context.module.entries) == count
    assert all(entry.state == EntryState.READY for entry in context.module.entries.values())
    return req


def finish_one(context, request):
    assert context.module.submit_transfer(request).status == TransferStatus.ACCEPTED
    context.process._poll_native_ddr()
    raw = context.process.native_ddr_output_queue.get_nowait()
    assert b'fake-capability' not in raw
    return decode_native_ddr_message(raw).event


def test_typed_import_and_two_full_prefix_loads_use_real_manager_refs_and_publish_after_control_receipts(pipeline):
    context = pipeline
    req = import_prefix(context)
    assert context.coord.lookup_after_hbm(req, [], 256).kind == RestoreKind.WAIT
    targets = [job.lease.block_ids[0] for job in context.coord.jobs.values()]
    assert all(context.manager.blocks[page].block_hash is None for page in targets)
    for index in range(2):
        job = context.coord.next_transfer()
        event = finish_one(context, job)
        assert event.pool_receipt is not None and event.stopped
        assert context.manager.blocks[job.page_ids[0]].block_hash is None
        context.coord.on_transfer_event(event)
        assert req.scheduling_held == (index == 0)
    assert len(context.calls) == 2
    context.coord.resume_request(req)
    ordinary = context.manager.get_computed_blocks(req.prompt_token_ids)
    assert len(ordinary) == 2
    context.coord.after_lookup(req)
    context.manager.release_cached_blocks(ordinary)
    assert not context.coord.jobs and not context.access.leases
    assert context.manager.num_free_blocks == 16


@pytest.mark.parametrize('fault', ['missing', 'old-client', 'wrong-page-lease', 'changed-target'])
def test_bad_pool_or_destination_proof_cannot_publish_or_drop_the_hbm_pin(pipeline, fault):
    context = pipeline
    req = import_prefix(context, 129, 1)
    assert context.coord.lookup_after_hbm(req, [], 128).kind == RestoreKind.WAIT
    job = context.coord.next_transfer()
    event = finish_one(context, job)
    receipt = event.pool_receipt
    if fault == 'missing':
        event = replace(event, pool_receipt=None)
    elif fault == 'old-client':
        event = replace(event, pool_receipt=replace(receipt, pool=replace(receipt.pool, client_session='old')))
    elif fault == 'wrong-page-lease':
        event = replace(event, pool_receipt=replace(receipt, page_lease_id='different-HBM-lease'))
    else:
        context.manager.blocks[job.page_ids[0]].block_hash = b'changed-target' * 2
    before = context.manager.blocks[job.page_ids[0]].ref_cnt
    context.coord.on_transfer_event(event)
    assert context.coord.poisoned and req.scheduling_held
    assert context.coord.jobs[job.job_id].state == LoadState.QUARANTINED
    assert context.manager.blocks[job.page_ids[0]].ref_cnt == before > 0
    assert context.coord.transfer_delivery.acknowledged == 0


def test_cancel_of_an_accepted_load_retains_range_and_hbm_until_real_handle_protocol_finishes(pipeline):
    context = pipeline
    req = import_prefix(context, 129, 1)
    context.coord.lookup_after_hbm(req, [], 128)
    job = context.coord.next_transfer()
    assert context.module.submit_transfer(job).status == TransferStatus.ACCEPTED
    context.coord.cancel_request(*context.coord._request_key(req))
    assert context.access.leases and context.module.pending is not None
    assert context.manager.blocks[job.page_ids[0]].ref_cnt == 1
    cancel = context.coord.next_control()
    context.process._handle_native_ddr_command(decode_command(encode_command(NativeDdrCommand(**asdict(cancel)))))
    reply = decode_native_ddr_message(context.process.native_ddr_output_queue.get_nowait()).control
    context.coord.on_control_event(reply)
    assert context.access.leases and context.module.pending is not None
    context.process._poll_native_ddr()
    event = decode_native_ddr_message(context.process.native_ddr_output_queue.get_nowait()).event
    assert event.status == TransferStatus.CANCELLED and event.pool_receipt is not None
    context.coord.on_transfer_event(event)
    assert not context.access.leases and not context.coord.jobs
    assert context.manager.blocks[job.page_ids[0]].ref_cnt == 0
    assert context.manager.blocks[job.page_ids[0]].block_hash is None


def test_pool_generation_change_while_pending_preserves_both_sides_as_unknown(pipeline):
    context = pipeline
    req = import_prefix(context, 129, 1)
    context.coord.lookup_after_hbm(req, [], 128)
    job = context.coord.next_transfer()
    assert context.module.submit_transfer(job).status == TransferStatus.ACCEPTED
    context.port.corrupt = lambda value: value
    context.process._poll_native_ddr()
    event = decode_native_ddr_message(context.process.native_ddr_output_queue.get_nowait()).event
    assert not event.stopped and context.module.poisoned
    context.coord.on_transfer_event(event)
    assert context.coord.poisoned and req.scheduling_held
    assert context.access.unknown and context.access.leases
    assert context.manager.blocks[job.page_ids[0]].ref_cnt == 1
    assert context.manager.blocks[job.page_ids[0]].block_hash is None
