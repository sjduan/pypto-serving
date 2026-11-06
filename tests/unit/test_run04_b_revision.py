# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""B admission and full offline entry adversaries. All native/model data is synthetic."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys

import pytest

from tests.unit.test_run04_qwen_driver import configuration, metadata, startup_fixture
import qwen_contract as contract
import qwen_driver as driver
from b_session import check_b_catalog

B_ID = {'hostname': 'liteserver-560t-1.novalocal', 'boot_id': 'fc0237c2-a3d2-40a1-b54a-9e8d554c0387',
        'pid_namespace': 'pid:[4026531836]'}
B_BUDGET = {'task_total_cap_bytes': 268435456, 'retained_A_charge_floor_bytes': 134217728,
            'B_Catalog_hard_budget_cap_bytes': 134217728, 'B_normal_pool_bytes': 134217728,
            'concurrent_new_B_pools_max': 1}


def b_configuration():
    return {**configuration(), 'schema': 2, 'target_host': 'serving-b-sj', 'container': 'openeuler-2403-DS',
            'budget': copy.deepcopy(B_BUDGET), 'expected_execution_identity': dict(B_ID), 'pool_history_cap': 4}


def b_release():
    return {'allowed_hosts': ['serving-b-sj'], 'forbidden_hosts': ['serving-a-sj'],
            'container': 'openeuler-2403-DS', 'A_device_and_remote_work_allowed': False,
            'expected_B_execution_identity': dict(B_ID), 'budget': copy.deepcopy(B_BUDGET),
            'original_A_quarantine': {'retained_charged_bytes': 134217728, 'fresh_A_status_not_queried': True}}


def test_B_plan_uses_no_A_snapshot_and_keeps_historical_charge_floor():
    value = contract.plan(b_configuration())
    assert value['status'] == 'NOT-RUN' and value['B_Catalog_cap_bytes'] == 134217728
    assert 'historical responsibility' in value['domain']
    assert len(value['cases']) == 4 and value['layout']['global_cap_bytes'] == 268435456
    contract.require_B_execution(b_configuration(), b_release(), B_ID)


def test_first_error_record_survives_unexpected_rich_report_value(tmp_path):
    path = tmp_path / 'first-error.json'
    driver.record_first_failure(path, {'error': 'original submit exception and traceback', 'unsupported': object()})
    result = contract.load(path)
    assert result['original_error'] == 'original submit exception and traceback'
    assert result['rich_report_saved'] is False and 'TypeError' in result['report_serialization_error']


def test_B_plan_and_wrong_actual_host_run_tripwire_before_any_runtime_or_service(tmp_path):
    contract.save(tmp_path / 'config.json', b_configuration())
    release = {**b_release(), 'state': 'released_by_explicit_human', 'supersedes': contract.HOLD_ID,
               'recorded_at_utc': '2026-11-06T01:26:43Z',
               'new_worker_init_native_submit_model_tests_device_reservations_allowed': True,
               'source': {'message_id': 'cpu-only-explicit-test-fixture', 'thread_id': 'cpu'}}
    contract.save(tmp_path / 'release.json', release)
    contract.save(tmp_path / 'hold.json', {'state': 'paused_by_latest_explicit_human',
                                          'recorded_at_utc': '2026-11-05T14:44:27Z'})
    script = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class Guard:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'pypto_serving','uflow','torch','numpy','pypto','simpler','acl','r_probe','qwen_scenario'}:
            raise AssertionError('runtime imported before B admission: ' + name)
sys.meta_path.insert(0, Guard())
def audit(event, args):
    if event in {'subprocess.Popen','socket.__new__','socket.connect','ctypes.dlopen','os.fork'}:
        raise AssertionError('execution before B admission: ' + event)
sys.addaudithook(audit)
import qwen_driver
root = Path(sys.argv[2])
common = ['--config',str(root/'config.json'),'--output',str(root/'plan.json')]
qwen_driver.main(['plan',*common])
qwen_driver.execution_identity = lambda: {'hostname':'A','boot_id':'other','pid_namespace':'pid:[1]'}
try:
    qwen_driver.main(['run',*common,'--hold',str(root/'hold.json'),'--release',str(root/'release.json')])
except PermissionError:
    print('B plan pure; wrong actual host rejected before runtime/RPC/spawn')
else:
    raise AssertionError('configuration label authorized wrong machine')
'''
    source = Path(__file__).resolve().parents[2] / 'examples/run04'
    result = subprocess.run([sys.executable, '-c', script, str(source), str(tmp_path)],
                            text=True, capture_output=True, check=True)
    assert 'wrong actual host rejected' in result.stdout


@pytest.mark.parametrize('fault', ['A-host', 'boot', 'namespace', 'container', 'missing-floor', 'wrong-budget',
    'wrong-authority', 'host-label-only', 'old-schema'])
def test_B_only_release_cannot_authorize_other_machine_or_budget(fault):
    config, release, actual = b_configuration(), b_release(), dict(B_ID)
    if fault == 'A-host':
        actual['hostname'] = 'A'
    elif fault == 'boot':
        actual['boot_id'] = 'different-boot'
    elif fault == 'namespace':
        actual['pid_namespace'] = 'pid:[999]'
    elif fault == 'container':
        release['container'] = 'other'
    elif fault == 'missing-floor':
        config['budget']['retained_A_charge_floor_bytes'] = 0
    elif fault == 'wrong-budget':
        release['budget']['B_Catalog_hard_budget_cap_bytes'] *= 2
    elif fault == 'wrong-authority':
        release['allowed_hosts'] = ['serving-a-sj', 'serving-b-sj']
    elif fault == 'host-label-only':
        config['expected_execution_identity']['boot_id'] = 'automatic-new-expectation'
    else:
        config = configuration()
    with pytest.raises((ValueError, PermissionError)):
        contract.require_B_execution(config, release, actual)


@pytest.mark.parametrize('fault', [None, 'two-live-pools', 'wrong-cap', 'reserved', 'unknown', 'other-provider'])
def test_B_accounting_is_single_local_authority_plus_A_retained_floor(fault):
    pool = {'pool_id': 'B-local', 'provider_session': 'B-provider', 'state': 'Allocated',
            'nbytes': 134217728, 'frozen': False, 'ranges': {'leases': []}, 'clients': []}
    state = {'budget_bytes': 134217728, 'charged_bytes': 134217728,
             'creation_reserved_bytes': 0, 'pools': [pool]}
    if fault == 'two-live-pools':
        state['pools'].append({**pool, 'pool_id': 'another'})
    elif fault == 'wrong-cap':
        state['budget_bytes'] *= 2
    elif fault == 'reserved':
        state['creation_reserved_bytes'] = 1
    elif fault == 'unknown':
        pool['frozen'] = True
    elif fault == 'other-provider':
        pool['provider_session'] = 'other'
    if fault is None:
        result = check_b_catalog(state, 'B-local', 'B-provider')
        assert result['task_responsibility_bytes'] == 268435456 and not result['A_fresh_status_queried']
        pool['state'], state['charged_bytes'] = 'Released', 0
        assert check_b_catalog(state, None, 'B-provider')['task_responsibility_bytes'] == 134217728
    else:
        with pytest.raises(ValueError):
            check_b_catalog(state, 'B-local', 'B-provider')


def rewrite(value, replacements):
    if isinstance(value, dict):
        return {k: rewrite(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(v, replacements) for v in value]
    return replacements.get(value, value) if isinstance(value, str) else value


@pytest.fixture(scope='module')
def complete_synthetic_bundle():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config, startup, descriptor = startup_fixture()
        first, report = metadata()
        names = ['cold', 'warm', 'restore', 'overwrite0', 'overwrite1', 'STORE0', 'STORE1', 'LOAD0', 'LOAD1', 'entry0', 'entry1']
        renames = {name: name + '-r1' for name in names}
        second = rewrite(copy.deepcopy(first), renames)
        second.update(number=1, interface='http')
        report['requests'] += rewrite(copy.deepcopy(report['requests']), renames)
        next(a for a in report['requests'] if a['request_id'] == 'cold-r1')['prompt'] = 'synthetic HTTP prefix'
        report['schedule_events'] += rewrite(copy.deepcopy(report['schedule_events']), renames)
        for block in second['blocks']:
            for terminal in (block['store'], block['load']):
                terminal['delivery_sequence'] += 10
                terminal['sequence'] += 10
        def capture(name, size, overwritten=False):
            path = root / (name + '.bin')
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
        for index in range(2):
            for name in ('source', 'overwritten', 'committed', 'restored'):
                row = capture(f'{index}-{name}', 20971520, name == 'overwritten')
                for round_ in (first, second):
                    round_['blocks'][index][name].update(row)
        for name in ('pool_before', 'pool_after'):
            row = capture(name, 134217728)
            first[name].update(row)
            second[name].update(row)
        report.update(mode='execution', case='normal-two-rounds-and-negatives', startup=startup,
                      rounds=[first, second], stop_ok=True,
                      shutdown_report={'cleanup_error': None, 'worker_reaped': True, 'native_ddr_obligations': 0})
        report['coordinator_trace'] = [copy.deepcopy(event) for r in (first, second) for b in r['blocks']
                                       for event in (b['store'], b['load'], b['valid'])]
        report['coordinator_trace'].append({'event': 'owner_drained_arguments_released'})
        report['requests'] += [{'request_id': 'cold', 'prompt': 'synthetic direct prefix'},
                               {'request_id': 'cold-r1', 'prompt': 'synthetic HTTP prefix'}]
        trace = report['coordinator_trace']
        def row(job, status, size):
            event = {'event': 'transfer_terminal', 'job': job, 'status': status,
                     'stopped': True, 'native_completed': True, 'bytes': size}
            trace.append(event)
            return {'terminal': copy.deepcopy(event)}
        report['load_cancellation'] = [{**row('cancel-one', 'SUCCEEDED', 20971520), 'all_waiters': False,
            'pending_target_retained': True, 'early_valid': False},
            {**row('cancel-all', 'CANCELLED', 20971520), 'all_waiters': True,
             'pending_target_retained': True, 'early_valid': False}]
        report['store_cancellation'] = {**row('cancel-store', 'CANCELLED', 20971520),
            'source_pin_released_after_stop': True, 'ddr_ready_not_published': True}
        for op in ('load', 'store'):
            report[f'postsubmit_{op}_failure'] = {**row('failed-' + op, 'FAILED', 0),
                'layer': 'short host receipt after actual native result/cleanup'}
        stale = {'kind': 'transfer', 'identity': 'retired-original', 'status': 'REJECTED',
                 'error': 'retired delivery sequence; replay rejected before effects'}
        report['stale_original_transfer'] = {'receipt': stale, 'native_runs_before_after': [20, 20]}
        report['http_pool_hit'] = {'request_id': 'restore-r1', 'status': 200,
            'payload': {'prompt': 'synthetic HTTP prefix'}, 'body': {'id': 'restore-r1',
                'usage': {'prompt_tokens': 257, 'completion_tokens': 2}, 'choices': [{'text': 'synthetic output'}]}}
        # Bind every content ID to the recomputed loaded namespace used by the
        # actual check entry. No model/native outcome is inferred from this data.
        report = json.loads(json.dumps(report).replace('qwen3-prefix-v1:' + 'a' * 64,
                                                      'qwen3-prefix-v1:' + startup['namespace_digest']))
        records = []
        for original in report['coordinator_trace']:
            records.append(('native_ddr_coordinator ', copy.deepcopy(original)))
        run = 0
        for round_ in report['rounds']:
            for block in round_['blocks']:
                for terminal in (block['store'], block['load']):
                    run += 1
                    event = {'job_id': terminal['job'], 'worker_epoch': terminal['worker_epoch'],
                        'operation': terminal['operation'], 'status': terminal['status'],
                        'entry_refs': terminal['entry_refs'], 'transferred_bytes': terminal['bytes'],
                        'native_completed': True, 'stopped': True, 'error': None,
                        'delivery_sequence': terminal['delivery_sequence'], 'pool_receipt': terminal['pool_receipt']}
                    parts = terminal['pool_receipt']['content']['content_id'].split(':')
                    request = {'job_id': terminal['job'], 'worker_epoch': terminal['worker_epoch'],
                        'operation': terminal['operation'], 'entry_refs': terminal['entry_refs'],
                        'page_ids': terminal['pages'], 'page_lease_id': terminal['page_lease_id'],
                        'delivery_sequence': terminal['delivery_sequence'], 'retire_through': 0,
                        'source_step_id': 1 if terminal['operation'] == 'STORE' else None,
                        'entry_keys': [{'namespace_digest': parts[1], 'block_hash': parts[2]}]}
                    ns = terminal['time_ns']
                    records.append(('native_ddr_receipt ', {'kind': 'transfer', 'identity': terminal['job'],
                        'status': 'SUCCEEDED', 'delivery_sequence': terminal['delivery_sequence'], 'error': None,
                        'event': event, 'request': request, 'time_ns': ns - 1}))
                    records.extend([('native_ddr_trace ', {'event': 'accepted', 'job': terminal['job'],
                        'native_run_id': run, 'time_ns': ns - 4}), ('native_ddr_trace ', {
                            'event': 'native_result_and_cleanup', 'job': terminal['job'], 'time_ns': ns - 2}),
                        ('', f'name=node.submit ts={ns - 4} run_id={run}'),
                        ('', f'name=node.complete ts={ns - 3} run_id={run} outcome=0')])
        records.append(('native_ddr_receipt ', stale))
        contract.save(root / 'config.json', config)
        contract.save(root / 'descriptor.json', descriptor)
        yield root, report, records


@pytest.mark.parametrize('fault', [None, 'summary-page-lease', 'producer-page-lease', 'raw-coordinator-conflict',
    'original-trace-conflict', 'missing-producer', 'duplicate-producer', 'duplicate-coordinator',
    'stale-range-generation', 'substituted-allocation', 'missing-publication', 'producer-namespace', 'late-producer'])
def test_full_check_entry_binds_normal_summaries_to_original_typed_raw_events(complete_synthetic_bundle, tmp_path, fault):
    root, original, original_records = complete_synthetic_bundle
    report, records = copy.deepcopy(original), copy.deepcopy(original_records)
    block = report['rounds'][0]['blocks'][0]
    producer = next(value for marker, value in records if marker == 'native_ddr_receipt ' and value['identity'] == 'LOAD0')
    coordinator = next(value for marker, value in records if marker == 'native_ddr_coordinator '
                       and value.get('job') == 'LOAD0' and value['event'] == 'transfer_terminal')
    if fault == 'summary-page-lease':
        block['load']['page_lease_id'] = block['load']['pool_receipt']['page_lease_id'] = 'old-page-lease'
    elif fault == 'producer-page-lease':
        producer['request']['page_lease_id'] = producer['event']['pool_receipt']['page_lease_id'] = 'old-page-lease'
    elif fault == 'raw-coordinator-conflict':
        coordinator['page_lease_id'] = 'conflicting-raw-lease'
    elif fault == 'original-trace-conflict':
        next(e for e in report['coordinator_trace'] if e.get('job') == 'LOAD0')['pages'] = [15]
    elif fault == 'missing-producer':
        records = [(m, v) for m, v in records if v is not producer]
    elif fault == 'duplicate-producer':
        records.append(('native_ddr_receipt ', copy.deepcopy(producer)))
    elif fault == 'duplicate-coordinator':
        records.append(('native_ddr_coordinator ', copy.deepcopy(coordinator)))
    elif fault == 'stale-range-generation':
        producer['event']['pool_receipt']['lease_generation'] += 1
    elif fault == 'substituted-allocation':
        block['load']['pool_receipt']['content']['block']['generation'] += 1
    elif fault == 'missing-publication':
        report['coordinator_trace'] = [e for e in report['coordinator_trace']
                                       if not(e['event'] == 'restored_page_valid' and e['job'] == 'LOAD0')]
    elif fault == 'producer-namespace':
        producer['request']['entry_keys'][0]['namespace_digest'] = 'f' * 64
    elif fault == 'late-producer':
        producer['time_ns'] = 999
    contract.save(tmp_path / 'report.json', report)
    (tmp_path / 'raw.log').write_text('\n'.join(m + (json.dumps(v) if m else v) for m, v in records))
    argv = ['check', '--config', str(root / 'config.json'), '--report', str(tmp_path / 'report.json'),
            '--descriptor', str(root / 'descriptor.json'), '--log', str(tmp_path / 'raw.log'),
            '--output', str(tmp_path / 'result.json')]
    if fault is None:
        assert driver.main(argv) == 0
        value = contract.load(tmp_path / 'result.json')
        assert value['status'] == 'HOST-EVIDENCE-CHECKED' and value['D1_acceptance'] is False
    else:
        with pytest.raises(ValueError):
            driver.main(argv)
        assert not (tmp_path / 'result.json').exists()
