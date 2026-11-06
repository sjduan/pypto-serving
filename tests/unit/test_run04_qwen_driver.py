# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synthetic metadata/bytes and import/spawn tripwires; never native/model evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples/run04'))
import qwen_contract as contract
import qwen_evidence as evidence


def configuration():
    components = {name: {'sha256': 'b' * 64} for name in contract.DEPLOYMENT}
    for name in ('serving', 'qwen_lib'):
        components[name] = {'source_files': 1, 'source_manifest_sha256': 'b' * 64}
    components['native_backend'] = dict.fromkeys(('host_path', 'aicpu_path', 'aicore_path'), 'b' * 64)
    return {'schema': 1, 'pins': contract.PINS, 'sdk_sha256': contract.SDK_SHA,
            'cases': list(contract.CASES), 'rounds': 2, 'slots': 5, 'device': 10, 'utilization': .8,
            'model_dir': '/data/models/Qwen/Qwen3-14B', 'assets': dict.fromkeys(contract.ASSETS, 'a' * 64),
            'expected_deployment': components}


def recorded_snapshot():
    return json.loads((ROOT / 'tests/unit/serving/fixtures/existing_quarantine_snapshot.json').read_text())


def metadata():
    """Independent hand-written two-block control history, no driver invocation."""
    pool = {'pool_id': 'normal', 'pool_epoch': 'epoch', 'provider_session': 'provider',
            'client_session': 'client', 'device': 25, 'inode': 100, 'nbytes': 134217728,
            'resource_id': '1', 'placement_id': '2', 'binding_id': '3', 'access_generation': 5,
            'simpler_owner': 'owner', 'simpler_buffer_id': 1, 'simpler_generation': 1,
            'resource_generation': 1, 'provider_kind': 'posix_shm_simpler'}
    allocations = [{'offset': i * 23068672, 'nbytes': 23068672, 'block_id': i + 1, 'generation': 1}
                   for i in range(2)]
    birth = {'pid': 99, 'starttime_ticks': 1234, 'uid': 0, 'gid': 0,
             'pid_namespace_dev': 4, 'pid_namespace_ino': 123}
    guard = {'identity': pool, 'bytes': 134217728, 'child_registry_matches_sdk': True,
             'registered_children': [[99, 1234]], 'registered_child_births': [copy.deepcopy(birth)],
             'write_ranges': [[a['offset'] + 2097152, 20971520]
                                                                   for a in allocations],
             'snapshot': {'pool_epoch': 'epoch', 'ranges': {'leases': [], 'allocations': allocations},
                          'clients': [{'client_session': 'client', 'state': 'Mapped', 'closing': False,
                              'birth': {**birth, 'pid': 90},
                              'children': [{'birth': copy.deepcopy(birth), 'unknown': False}]}]}}
    report = {'startup': {'worker_epoch': 'worker', 'pool_identity': pool, 'namespace_digest': 'a' * 64}, 'actual_pages': 16,
              'schedule_events': [], 'requests': []}
    round_ = {'number': 0, 'interface': 'direct', 'prompt_tokens': 257, 'blocks': [],
              'pool_before': guard, 'pool_after': copy.deepcopy(guard),
              'sentinel_before': [{'sha256': 'unchanged'}], 'sentinel_after': [{'sha256': 'unchanged'}],
              'cold_request': 'cold', 'hbm_request': 'warm', 'restore_request': 'restore'}
    for request, count in [('cold', 257), ('warm', 1), ('restore', 1)]:
        report['schedule_events'].append({'requests': [
            {'request_id': request, 'is_prefill': True, 'tokens': count},
            {'request_id': request, 'is_prefill': False, 'tokens': 1}]})
    for i in range(2):
        ref = {'entry_id': f'entry{i}', 'worker_epoch': 'worker', 'slot_generation': i + 1}
        content = {'pool_id': 'normal', 'pool_epoch': 'epoch', 'provider_session': 'provider',
                   'nbytes': 20971520, 'byte_offset': allocations[i]['offset'] + 2097152,
                   'content_version': 1, 'writer_lease_id': f'w{i}',
                   'content_id': 'qwen3-prefix-v1:' + 'a' * 64 + ':' + str(i + 1) * 64,
                   'operation_id': f'commit{i}', 'block': {'pool_id': 'normal', 'pool_epoch': 'epoch',
                       'block_id': i + 1, 'generation': 1, 'byte_offset': allocations[i]['offset'], 'nbytes': 23068672}}
        terminals = []
        for op, page, lease, ns in [('STORE', i, f'w{i}', 10), ('LOAD', i + 4, f'r{i}', 50)]:
            receipt = {'pool': pool, 'operation': op, 'page_lease_id': op + str(i),
                       'lease_id': lease, 'lease_generation': 2, 'content': content, 'release_ack': None}
            if op == 'LOAD':
                receipt['release_ack'] = {'pool': pool, 'operation': 'release', 'applied': True,
                    'lease_id': lease, 'lease_generation': 2, 'operation_id': 'read-ack'}
            terminals.append({'operation': op, 'entry_refs': [ref], 'worker_epoch': 'worker',
                'status': 'SUCCEEDED', 'native_completed': True, 'stopped': True, 'bytes': 20971520,
                'error': None, 'pool_receipt': receipt, 'page_lease_id': op + str(i),
                'pages': [page], 'time_ns': ns, 'job': op + str(i), 'event': 'transfer_terminal',
                'sequence': i * 2 + (1 if op == 'STORE' else 2),
                'delivery_sequence': i * 2 + (1 if op == 'STORE' else 2)})
        overwrite = f'overwrite{i}'
        report['requests'].append({'request_id': overwrite, 'prompt_token_ids': list(range(127)),
                                    'begin_ns': 20, 'completed_ns': 40})
        report['schedule_events'].append({'requests': [
            {'request_id': overwrite, 'is_prefill': True, 'computed_before': 0, 'tokens': 127, 'pages': [i]},
            {'request_id': overwrite, 'is_prefill': False, 'computed_before': 127, 'tokens': 1, 'pages': [i]}]})
        round_['blocks'].append({'index': i, 'token_start': i * 128, 'tokens': 128, 'entry_ref': ref,
            'store': terminals[0], 'load': terminals[1], 'source_page': i, 'target_page': i + 4,
            'source': {'page': i}, 'overwritten': {'page': i}, 'restored': {'page': i + 4},
            'overwrite_request': overwrite,
            'committed': {'pool_identity': pool, 'committed': content, 'generation': i + 1, 'entry_id': ref['entry_id']},
            'valid': {'event': 'restored_page_valid', 'job': 'LOAD' + str(i), 'time_ns': 60,
                      'sequence': 5 + i, 'page': i + 4}})
    report['coordinator_trace'] = [copy.deepcopy(event) for block in round_['blocks']
                                   for event in (block['store'], block['load'], block['valid'])]
    return round_, report


def test_plan_has_real_managed_cli_but_no_execution_claim():
    plan = contract.plan(configuration(), recorded_snapshot())
    assert plan['status'] == 'NOT-RUN' and plan['device_admission'] is False
    assert all(case['status'] == 'NOT-RUN' for case in plan['cases'])
    assert plan['argv'][plan['argv'].index('--native-ddr-kv-bytes') + 1] == str(5 * 20971520)
    for flag in ('--native-ddr-pool-endpoint', '--native-ddr-pool-id', '--native-ddr-pool-epoch', '--native-ddr-sdk-sha256'):
        assert flag in plan['argv']


@pytest.mark.parametrize('field,value', [('rounds', 1), ('slots', 16), ('device', 0), ('sdk_sha256', 'c' * 64)])
def test_configuration_cannot_silently_change_identity_or_capacity(field, value):
    config = configuration()
    config[field] = value
    with pytest.raises(ValueError):
        contract.validate_config(config)


def test_plan_rejects_insufficient_catalog_history_without_connecting():
    snapshot = recorded_snapshot()
    for i in range(4):
        released = copy.deepcopy(snapshot['pools'][0])
        released.update(pool_id=str(i), state='Released', clients=[], ranges={'leases': []})
        snapshot['pools'].append(released)
    with pytest.raises(ValueError, match='history cap'):
        contract.plan(configuration(), snapshot)


@pytest.mark.parametrize('length,index,start,tokens', [(256, 1, 128, 128), (255, 1, 128, 128),
    (257, 1, 128, 127), (257, 0, 128, 128), (128, 0, 0, 128), (257, 2, 256, 128)])
def test_last_token_partial_tail_and_noncontiguous_blocks_are_not_promoted(length, index, start, tokens):
    with pytest.raises(ValueError, match='partial tail'):
        evidence.eligible_block(length, index, start, tokens)


def test_two_contiguous_blocks_have_complete_host_order_without_native_acceptance():
    round_, report = metadata()
    result = evidence.check_round(round_, report, '.', inspect_bytes=False)
    assert result['status'] == 'HOST-METADATA-CHECKED' and result['native_acceptance'] is False


@pytest.mark.parametrize('fault', ['missing-block', 'old-worker', 'wrong-token-start', 'early-valid',
    'old-reader', 'changed-content', 'same-target', 'partial-overwrite', 'wrong-source', 'guard-mask', 'child-registry'])
def test_missing_or_stale_stages_cannot_be_accepted(fault):
    round_, report = metadata()
    block = round_['blocks'][1]
    if fault == 'missing-block':
        round_['blocks'].pop()
    elif fault == 'old-worker':
        block['entry_ref']['worker_epoch'] = 'old-worker'
    elif fault == 'wrong-token-start':
        block['token_start'] = 256
    elif fault == 'early-valid':
        block['valid']['time_ns'] = 3
    elif fault == 'old-reader':
        block['load']['pool_receipt']['release_ack']['lease_generation'] = 1
    elif fault == 'changed-content':
        block['load']['pool_receipt']['content'] = {**block['load']['pool_receipt']['content'], 'content_version': 2}
    elif fault == 'same-target':
        block['target_page'] = block['source_page']
    elif fault == 'partial-overwrite':
        report['schedule_events'][-1]['requests'].pop()
    elif fault == 'wrong-source':
        report['schedule_events'][-1]['requests'][0]['pages'] = [8]
    elif fault == 'guard-mask':
        round_['pool_after']['write_ranges'][0][0] = 0
    else:
        round_['pool_after']['child_registry_matches_sdk'] = False
    with pytest.raises(ValueError):
        evidence.check_round(round_, report, '.', inspect_bytes=False)


def test_missing_raw_bytes_are_not_a_pass():
    round_, report = metadata()
    with pytest.raises((KeyError, ValueError)):
        evidence.check_round(round_, report, '.', inspect_bytes=True)


@pytest.mark.parametrize('fault', [None, 'missing-submit', 'failed-native', 'late-cleanup', 'duplicate-complete'])
def test_native_correlation_requires_real_submit_outcome_and_cleanup_order(fault):
    round_, report = metadata()
    report['rounds'] = [round_]
    lines = []
    run = 0
    for block in round_['blocks']:
        for terminal in (block['store'], block['load']):
            run += 1
            terminal['time_ns'] = 50
            lines.append('native_ddr_trace ' + json.dumps({'event': 'accepted', 'job': terminal['job'],
                'native_run_id': run, 'time_ns': 11}))
            lines.append('native_ddr_trace ' + json.dumps({'event': 'native_result_and_cleanup',
                'job': terminal['job'], 'time_ns': 70 if fault == 'late-cleanup' else 40}))
            if fault != 'missing-submit':
                lines.append(f'name=node.submit ts=10 run_id={run} ')
            complete = f'name=node.complete ts=20 run_id={run} outcome={1 if fault == "failed-native" else 0}'
            lines.append(complete)
            if fault == 'duplicate-complete':
                lines.append(complete)
    if fault is None:
        evidence.correlate_native(report, '\n'.join(lines))
    else:
        with pytest.raises(ValueError):
            evidence.correlate_native(report, '\n'.join(lines))


@pytest.mark.parametrize('fault', [None, 'unstopped', 'early-valid', 'store-published', 'new-native-run', 'missing-raw'])
def test_cancel_failure_and_retired_replay_evidence_are_checked_against_trace(fault):
    trace = []
    def row(job, status, bytes_):
        event = {'event': 'transfer_terminal', 'job': job, 'status': status, 'stopped': True,
                 'native_completed': True, 'bytes': bytes_}
        trace.append(event)
        return {'terminal': event}
    one = {**row('one', 'SUCCEEDED', 20971520), 'all_waiters': False,
           'pending_target_retained': True, 'early_valid': False}
    all_ = {**row('all', 'CANCELLED', 20971520), 'all_waiters': True,
            'pending_target_retained': True, 'early_valid': False}
    store = {**row('store', 'CANCELLED', 20971520), 'source_pin_released_after_stop': True,
             'ddr_ready_not_published': True}
    failed_load = {**row('failed-load', 'FAILED', 0), 'layer': 'short host receipt after actual native result/cleanup'}
    failed_store = {**row('failed-store', 'FAILED', 0), 'layer': 'short host receipt after actual native result/cleanup'}
    stale = {'receipt': {'error': 'retired delivery sequence; replay rejected before effects'},
             'native_runs_before_after': [20, 20]}
    report = {'coordinator_trace': trace, 'load_cancellation': [one, all_], 'store_cancellation': store,
              'postsubmit_load_failure': failed_load, 'postsubmit_store_failure': failed_store,
              'stale_original_transfer': stale}
    raw = 'native_ddr_receipt ' + json.dumps(stale['receipt'])
    if fault == 'unstopped':
        store['terminal']['stopped'] = False
    elif fault == 'early-valid':
        trace.append({'event': 'restored_page_valid', 'job': 'all'})
    elif fault == 'store-published':
        trace.append({'event': 'ddr_ready', 'job': 'store'})
    elif fault == 'new-native-run':
        stale['native_runs_before_after'][1] += 1
    elif fault == 'missing-raw':
        raw = ''
    if fault is None:
        evidence.check_controls(report, raw)
    else:
        with pytest.raises(ValueError):
            evidence.check_controls(report, raw)


def test_complete_synthetic_bytes_and_guards_still_make_no_native_acceptance(tmp_path):
    round_, report = metadata()
    def capture(name, size, *, overwritten=False):
        path = tmp_path / (name + '.bin')
        with path.open('wb') as stream:
            stream.truncate(size)
        if overwritten:
            with path.open('r+b') as stream:
                for offset in range(0, size, 2048):
                    stream.seek(offset)
                    stream.write(b'\x01')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda: stream.read(1 << 20), b''):
                digest.update(part)
        return {'file': str(path), 'bytes': size, 'sha256': digest.hexdigest()}
    for i, block in enumerate(round_['blocks']):
        for name in ('source', 'overwritten', 'committed', 'restored'):
            block[name].update(capture(f'{i}-{name}', 20971520, overwritten=name == 'overwritten'))
    for name in ('pool_before', 'pool_after'):
        round_[name].update(capture(name, 134217728))
    result = evidence.check_round(round_, report, tmp_path, inspect_bytes=True)
    assert result['status'] == 'HOST-EVIDENCE-CHECKED' and result['native_acceptance'] is False
    assert len(result['block_checks']) == 2
    # A single guard-byte change must fail even though all four block files
    # continue to compare correctly. Rehash simulates honest changed evidence.
    path = Path(round_['pool_after']['file'])
    with path.open('r+b') as stream:
        stream.write(b'\xff')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    round_['pool_after']['sha256'] = digest.hexdigest()
    with pytest.raises(ValueError, match='guard'):
        evidence.check_round(round_, report, tmp_path, inspect_bytes=True)


def test_plan_and_held_run_tripwire_no_runtime_import_rpc_spawn_or_device_probe(tmp_path):
    config, snap = tmp_path / 'config.json', tmp_path / 'snapshot.json'
    contract.save(config, configuration())
    contract.save(snap, recorded_snapshot())
    hold = tmp_path / 'hold.json'
    contract.save(hold, {'state': 'paused_by_latest_explicit_human', 'recorded_at_utc': '2026-11-05T14:44:27Z'})
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','numpy','pypto','simpler','uflow','acl','pypto_serving','r_probe','qwen_scenario'}:
            raise AssertionError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
def audit(event, args):
    if event in {'subprocess.Popen','os.system','socket.__new__','socket.connect','ctypes.dlopen','os.fork'}:
        raise AssertionError('forbidden execution: ' + event)
    if event == 'open' and str(args[0]).endswith('.safetensors'):
        raise AssertionError('model payload opened')
sys.addaudithook(audit)
import qwen_driver
qwen_driver.main(['plan','--config',sys.argv[2],'--snapshot',sys.argv[3],'--output',sys.argv[4]])
try:
    qwen_driver.main(['check','--config',sys.argv[2],'--report',sys.argv[4],
                     '--descriptor',sys.argv[2],'--log',sys.argv[5],'--output',sys.argv[6]])
except ValueError:
    pass
else:
    raise AssertionError('plan accepted as actual execution')
try:
    qwen_driver.main(['run','--config',sys.argv[2],'--hold',sys.argv[5],'--output',sys.argv[6]])
except PermissionError:
    print('HOST-ONLY tripwire passed; device results NOT-RUN')
else:
    raise AssertionError('HOLD bypassed')
'''
    result = subprocess.run([sys.executable, '-c', script, str(ROOT / 'examples/run04'),
        str(config), str(snap), str(tmp_path / 'plan.json'), str(hold), str(tmp_path / 'execution.json')],
        text=True, capture_output=True, check=True)
    assert 'tripwire passed' in result.stdout and not (tmp_path / 'execution.json').exists()


@pytest.mark.parametrize('release', [None, {'state': 'released_by_explicit_human',
    'supersedes': 'serving-ab-npu-hold-20261104'}, {'state': 'released_by_explicit_human',
    'supersedes': contract.HOLD_ID, 'new_worker_init_native_submit_model_tests_device_reservations_allowed': True,
    'source': {'message_id': 'historical', 'thread_id': 'dev'}, 'recorded_at_utc': '2026-11-05T12:00:00Z'}])
def test_historical_release_cannot_resume_current_hold(release):
    with pytest.raises(PermissionError):
        contract.require_release({'state': 'paused_by_latest_explicit_human',
                                   'recorded_at_utc': '2026-11-05T14:44:27Z'}, release)


def startup_fixture():
    config = configuration()
    pool = metadata()[1]['startup']['pool_identity']
    descriptor = {**pool, 'canonical': {'dev': 25, 'inode': 100}}
    layout = {'num_layers': 40, 'num_pages': 16, 'page_size': 128}
    compat = {'scope': 'managed-pool-compatible-data-local-owner', 'worker_epoch': 'worker',
        'deployment': config['expected_deployment'], 'assets': {'files': {name:
            {'sha256': value, 'evidence': 'full-asset-content-at-startup'} for name, value in config['assets'].items()}},
        'layout': layout, 'model': {'model_id': 'Qwen3-14B'}, 'tokenizer_type': 'Qwen2TokenizerFast',
        'adapter': None, 'position': 'absolute-token-offset/loaded-config-RoPE', 'tensor_parallel': 1}
    data = {'schema': 1, 'model': compat['model'], 'asset_content': config['assets'],
            'tokenizer_type': compat['tokenizer_type'], 'adapter': None, 'position': compat['position'],
            'layout': {k: v for k, v in layout.items() if k != 'num_pages'}, 'tensor_parallel': 1,
            'deployment': config['expected_deployment']}
    startup = {'max_entries': 5, 'capacity_bytes': 5 * 20971520, 'layout': layout,
               'worker_epoch': 'worker', 'compatibility_json': compat, 'pool_identity': pool,
               'namespace_digest': contract.digest(data)}
    return config, startup, descriptor


def test_loaded_startup_binds_full_assets_deployment_pool_and_namespace():
    config, startup, descriptor = startup_fixture()
    contract.check_startup(config, startup, descriptor)


@pytest.mark.parametrize('fault', ['asset-payload', 'header-only', 'binary', 'pool', 'epoch', 'namespace'])
def test_loaded_compatibility_refuses_changed_or_weaker_evidence(fault):
    config, startup, descriptor = startup_fixture()
    if fault in ('asset-payload', 'header-only'):
        asset = startup['compatibility_json']['assets']['files']['config.json']
        asset['sha256' if fault == 'asset-payload' else 'evidence'] = 'changed'
    elif fault == 'binary':
        startup['compatibility_json']['deployment'] = copy.deepcopy(config['expected_deployment'])
        startup['compatibility_json']['deployment']['native_backend']['host_path'] = 'c' * 64
    elif fault == 'pool':
        startup['pool_identity']['inode'] += 1
    elif fault == 'epoch':
        startup['compatibility_json']['worker_epoch'] = 'old-worker'
    else:
        startup['namespace_digest'] = 'd' * 64
    with pytest.raises(ValueError):
        contract.check_startup(config, startup, descriptor)
