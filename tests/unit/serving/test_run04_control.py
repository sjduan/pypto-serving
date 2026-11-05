# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU mmap and fake control outcomes; no Worker/init/native/model/UF service."""
import copy
import json
import mmap
import queue
from types import SimpleNamespace

import pytest

from pypto_serving.serving.native_ddr.diagnostics import capture_idle_pool
from pypto_serving.serving.native_ddr.types import TransferStatus
from tests.unit.serving.test_native_ddr_host import transfer, worker
from tests.unit.serving.engine.test_worker_shutdown import _core, _FakeProcess, _FakeQueue


@pytest.fixture(autouse=True)
def reset_test_delivery():
    from tests.unit.serving.test_native_ddr_host import _wire_sequences
    _wire_sequences.clear()


def test_sealed_driver_arguments_are_accepted_by_production_cli_without_loading_model():
    from pypto_serving.cli.main import build_parser
    from tests.unit.test_run04_qwen_driver import configuration
    from qwen_contract import cli_arguments, SDK_SHA
    arguments = build_parser().parse_args(cli_arguments(configuration(), '/existing/u.sock', 'normal', 'epoch'))
    assert arguments.native_ddr_pool_endpoint == '/existing/u.sock'
    assert (arguments.native_ddr_pool_id, arguments.native_ddr_pool_epoch) == ('normal', 'epoch')
    assert arguments.native_ddr_sdk_sha256 == SDK_SHA and arguments.native_ddr_kv_bytes == 5 * 20971520
    assert arguments.block_size == 128 and arguments.device == 10


def test_first_nonterminal_submit_failure_survives_duplicate_without_terminal_replay_or_retirement():
    module, backend = worker()
    backend.error = RuntimeError('original submit stage: incompatible native argument')
    request = transfer(job='first-failure')
    event = module.submit_transfer(request)
    pending = module.pending
    assert event.status == TransferStatus.FAILED and not event.stopped
    assert pending.first_failure is event
    assert module.transfer_replay.records[request.delivery_sequence].response is None
    duplicate = module.submit_transfer(request)
    assert duplicate.error == 'native obligation retained'
    assert pending.first_failure is event and 'original submit stage' in pending.first_failure.error
    assert pending.first_failure_detail['exception_type'] == 'RuntimeError'
    assert 'incompatible native argument' in pending.first_failure_detail['traceback']
    assert not module.poll_completed() and module.pending is pending
    with pytest.raises(RuntimeError, match='retained native'):
        module.close()
    assert backend.slots and module.poisoned
    from dataclasses import replace
    next_request = replace(request, job_id='illegal-retirement', delivery_sequence=request.delivery_sequence + 1,
                            retire_through=request.delivery_sequence)
    rejected = module.submit_transfer(next_request)
    assert not rejected.stopped and 'crosses retained' in rejected.error
    assert module.pending is pending and module.transfer_replay.acknowledged == 0


def test_first_result_cleanup_failure_survives_unchanged():
    module, backend = worker()
    request = transfer(job='cleanup-failure')
    module.submit_transfer(request)
    backend.handle.ready = True
    backend.handle.error = RuntimeError('first native cleanup exception')
    event, = module.poll_completed()
    assert not event.stopped and module.pending.first_failure is event
    assert module.poll_completed() == () and backend.slots
    assert module.transfer_replay.records[request.delivery_sequence].response is None


def test_production_emitter_logs_complete_failure_before_transport_and_without_terminal_claim(caplog):
    from pypto_serving.serving.server.ipc import NativeDdrMessage
    from pypto_serving.serving.server.serving_worker import WorkerProcess
    module, backend = worker()
    backend.error = RuntimeError('first submission exception')
    event = module.submit_transfer(transfer(job='full-first-event'))
    process = WorkerProcess.__new__(WorkerProcess)
    process._native_ddr_worker = module
    process._native_ddr_control_replay = None
    process.native_ddr_output_queue = queue.Queue()
    caplog.set_level('INFO')
    process._emit_native_ddr(NativeDdrMessage('transfer', event=event))
    line = next(r.message for r in caplog.records if r.message.startswith('native_ddr_receipt '))
    record = json.loads(line.removeprefix('native_ddr_receipt '))
    assert record['event']['error'] == event.error
    assert record['event']['entry_refs'][0]['entry_id'] == event.entry_refs[0].entry_id
    assert record['event']['delivery_sequence'] == event.delivery_sequence
    assert record['event']['stopped'] is False and record['event']['native_completed'] is False
    assert record['first_nonterminal_failure']['stage'] == 'native-submit'
    assert 'RuntimeError: first submission exception' in record['first_nonterminal_failure']['traceback']
    assert record['pending_owner']['quarantined'] is True
    assert module.transfer_replay.records[event.delivery_sequence].response is None
    assert process.native_ddr_output_queue.qsize() == 1


@pytest.mark.parametrize('queue_present', [False, True])
def test_managed_client_timeout_retains_process_and_queues_without_term_or_kill(queue_present):
    process, queue = _FakeProcess(dies_after=None), _FakeQueue() if queue_present else None
    core = _core(process, queue)
    core._runtime = SimpleNamespace(native_ddr=SimpleNamespace(enabled=True))
    output, profile = object(), object()
    core._output_queue, core._profile_output_queue = output, profile
    assert core._shutdown_worker(timeout=.01) is False
    assert core._worker_process is process and core._input_queue is queue
    assert core._output_queue is output and core._profile_output_queue is profile
    assert process.signals == [] and process.joins == [.01]
    if queue_present:
        assert len(queue.sent) == 1


def test_managed_client_real_cooperative_reap_retires_parent_handles():
    process, queue = _FakeProcess(dies_after=0), _FakeQueue()
    core = _core(process, queue)
    core._runtime = SimpleNamespace(native_ddr=SimpleNamespace(enabled=True))
    assert core._shutdown_worker(timeout=.01) is True
    assert core._worker_process is core._input_queue is core._output_queue is None
    assert process.signals == [] and len(queue.sent) == 1


@pytest.fixture
def pool():
    # Real anonymous CPU mmap, explicitly synthetic SDK snapshots. No API port
    # may mutate allocation/leases during this guard-only physical observation.
    size = 65536
    buffer = mmap.mmap(-1, size)
    identity = SimpleNamespace(pool_id='pool', pool_epoch='epoch', client_session='client',
                               nbytes=size, device=1, inode=2)
    from dataclasses import make_dataclass
    Identity = make_dataclass('CpuGuardIdentity', [(key, type(value)) for key, value in vars(identity).items()])
    identity = Identity(**vars(identity))
    row = {'pool_id': 'pool', 'pool_epoch': 'epoch', 'nbytes': size, 'frozen': False,
           'canonical': {'dev': 1, 'inode': 2}, 'ranges': {'leases': []},
           'clients': [{'client_session': 'client', 'state': 'Mapped'}]}
    snapshots = []
    def snapshot():
        snapshots.append(1)
        return {'pools': [copy.deepcopy(row)]}
    access = SimpleNamespace(identity=identity, buffer=buffer, leases={}, unknown={}, closed=False,
                             children={(99, 1234): object()}, port=SimpleNamespace(client=SimpleNamespace(snapshot=snapshot)))
    slots = SimpleNamespace(access=access, busy=None, child_registrations={99: 1234}, child_helpers={}, _blocks={})
    yield slots, row, snapshots
    buffer.close()


def test_idle_guard_observer_reads_exact_existing_mmap_without_acquire_commit_or_write(pool, tmp_path):
    slots, _, snapshots = pool
    slots.access.buffer[123] = 89
    result = capture_idle_pool(slots, tmp_path, 'r04-cpu')
    assert result['bytes'] == 65536 and result['child_registry_matches_sdk'] is True
    assert (tmp_path / 'r04-cpu-pool.bin').read_bytes()[123] == 89
    assert snapshots == [1, 1] and slots.access.buffer[123] == 89
    assert 'not Committed prefix consumption' in result['scope']


@pytest.mark.parametrize('fault', ['busy', 'unknown', 'lease', 'child', 'competing-client', 'physical', 'remote-lease'])
def test_guard_observer_refuses_live_or_mismatched_obligations_before_read(pool, tmp_path, fault):
    slots, row, _ = pool
    if fault == 'busy':
        slots.busy = object()
    elif fault == 'unknown':
        slots.access.unknown['lost'] = object()
    elif fault == 'lease':
        slots.access.leases['pending'] = object()
    elif fault == 'child':
        slots.child_registrations.clear()
    elif fault == 'competing-client':
        row['clients'].append({'client_session': 'other', 'state': 'Mapped'})
    elif fault == 'physical':
        row['canonical']['inode'] = 3
    else:
        row['ranges']['leases'].append({'state': 'Writing'})
    with pytest.raises(RuntimeError):
        capture_idle_pool(slots, tmp_path, 'r04-rejected')
    assert not (tmp_path / 'r04-rejected-pool.bin').exists()
