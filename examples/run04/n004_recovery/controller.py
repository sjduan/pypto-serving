# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fresh identity/resource gates and one original n004 client cooperative close, never create a pool."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time

import actions
import driver

BASE = Path('/workspace/uflow/runs/d1-managed-ddr-20261105/serving')
SOURCE = BASE / 'a003/src/serving'
STATE = BASE / 'a002/n001/state'
PID, BIRTH = 1022467, 484127941
CHILDREN = {1024040: 484128722, 1024041: 484128722}
OWNERS = {899046: 483907077, 899051: 483907079, 899069: 483907103}
PYTHON_SHA = '1195c76f45c7cf16bf776a572cc85e074669c6f1a17d80d83fa8b7a7e3f80a87'
SDK_SHA = '366f3ea883cec08efd62f23f9ebf2afa8546c02a777ff25fdf753fee1debff0e'
CORE_SHA = '9d7c187e37d9d20a898c3e4342f4d4fd31cd805aed18427059d0a2b9b68b5169'
SOURCE_HASHES = {
    str(SOURCE / 'examples/run04/pool_native_probe.py'): 'f3628b7cdcac40acdf29b171be8cfb2b5e8b0971d85fd3fe3e7032c2be8e66cd',
    str(SOURCE / 'pypto_serving/serving/native_ddr/managed_process.py'): 'f9ba14e4f88bbd808998a784560740b569262222435754b57c4c3cd268c8399e',
    str(SOURCE / 'pypto_serving/model/qwen/native_ddr_transfer.py'): 'db0bb2bd24974ae42a646b9df0f9403a4fdbb9f5239ef99d85f43c4e7f8ac101',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_process(pid):
    proc = Path('/proc', str(pid))
    result = {'pid': pid, 'birth': actions.birth(pid), 'exe': os.readlink(proc / 'exe'),
        'exe_sha256': digest(proc / 'exe'), 'cmdline': (proc / 'cmdline').read_bytes().decode().split('\0')[:-1],
        'status': (proc / 'status').read_text(), 'maps': (proc / 'maps').read_text(), 'fds': []}
    for fd in (proc / 'fd').iterdir():
        try:
            info = fd.stat()
            result['fds'].append({'fd': int(fd.name), 'target': os.readlink(fd), 'dev': info.st_dev,
                'inode': info.st_ino, 'mode': info.st_mode, 'rdev': info.st_rdev})
        except FileNotFoundError:
            continue
    assert actions.birth(pid) == result['birth']
    return result


def setup_service(root):
    from existing_pool import ExistingPoolService, retained_pool, PROVIDER, FENCE_SHA
    from pool_host_probe import save
    from uflow.managed_pool import _exchange
    # This recovery deliberately attaches to the existing Catalog. Its ordinary
    # constructor rightly rejects the second Unknown, so do not invoke create.
    service = ExistingPoolService.__new__(ExistingPoolService)
    service.state_root, service.evidence = STATE, root / 'catalog'
    service.evidence.mkdir()
    service.endpoint, service._exchange, service._save = str(STATE.parent / 'u.sock'), _exchange, save
    initial = service.snapshot('fresh-before-recovery')
    baseline = json.loads((BASE / 'a003/retained-baseline.json').read_text())
    service.expected = retained_pool(baseline)
    assert retained_pool(initial) == service.expected
    service.starts = OWNERS.copy()
    service._admin, service.normal = None, None
    assert {p: actions.birth(p) for p in OWNERS} == OWNERS
    os.environ['UFLOW_POOL_TEST_FENCE'] = str(BASE / 'a002/api_fence.so')
    os.environ['UFLOW_POOL_TEST_SYMBOLS'] = str(BASE / 'a002/src/pool/examples/managed_pool/blocked-symbols.json')
    os.environ['UFLOW_POOL_TEST_INVENTORY_ROOT'] = str(root / 'inventories')
    assert digest(BASE / 'a002/api_fence.so') == FENCE_SHA
    service.verify_owners('fresh-before')
    loaded = json.loads((STATE / 'state.json').read_text())['provider_loaded']
    names = json.loads(Path(os.environ['UFLOW_POOL_TEST_SYMBOLS']).read_text())
    assert loaded['provider_session'] == PROVIDER and loaded['pid'] == 899051 and loaded['parent_pid'] == 899046
    assert set(loaded['test_fence_bindings']) == set(names) and len(names) == 79
    assert set(loaded['test_fence_bindings'].values()) == {str(BASE / 'a002/api_fence.so')}
    assert loaded['simpler_revision'] == '508becb93e182a3035a14e3f9226929dbe994346'
    assert loaded['simpler_sha256'] == 'ee1e5b64d5f836950892fc3d6e986a3eda8d106b7c92f27f8420589184073f96'
    assert loaded['task_interface_sha256'] == 'd7945a85cc49aedeed14006f1c7d2077807f6fb5e9565fe21cfa6cb9e694769d'
    save(root / 'provider-loaded.json', loaded)
    assert initial['charged_bytes'] == initial['budget_bytes'] == 268435456
    assert initial['creation_reserved_bytes'] == 0 and len(initial['pools']) == 3
    active = [p for p in initial['pools'] if p['state'] != 'Released']
    assert {p['pool_id'] for p in active} == {service.expected['pool_id'], actions.POOL_ID}
    current = next(p for p in active if p['pool_id'] == actions.POOL_ID)
    descriptor = json.loads((BASE / 'a003/n004/catalog/normal-descriptor.json').read_text())
    assert all(current[k] == v for k, v in descriptor.items()) and current['frozen']
    assert current['nbytes'] == 134217728 and not current['ranges']['leases']
    assert len(current['clients']) == 1
    client = current['clients'][0]
    assert client['client_session'] == actions.CLIENT_ID and client['state'] == 'Unknown' and not client['children']
    service.normal = descriptor
    return service, descriptor, initial


def target_gate(root, descriptor):
    from host_guard import inventory, _validated_calibration
    assert actions.birth(PID) == BIRTH and driver.tracer(PID) == 0
    assert actions.descendants(PID) == CHILDREN
    results = {}
    for pid, expected in {PID: BIRTH, **CHILDREN}.items():
        value = read_process(pid)
        assert value['birth'] == expected and value['exe_sha256'] == PYTHON_SHA
        driver.save(root / f'process-{pid}.json', value)
        results[pid] = value
    target = results[PID]
    assert target['cmdline'][0] == driver.PYTHON and target['cmdline'][1:3] == ['-u', str(SOURCE / 'examples/run04/pool_native_probe.py')]
    assert target['cmdline'][-2:] == ['--device', '10']
    libpython = sorted({s.split(None, 5)[5] for s in target['maps'].splitlines() if '/libpython3.11.so' in s})
    assert libpython == ['/usr/local/python3.11.14/lib/libpython3.11.so.1.0']
    driver.save(root / 'libpython.json', {'path': libpython[0], 'sha256': digest(libpython[0]), 'mapped_in_original_target': True})
    for name, expected in SOURCE_HASHES.items():
        assert digest(name) == expected, name
    assert digest(BASE / 'a003/src/pool/sdk/python/uflow/managed_pool.py') == SDK_SHA
    assert digest('/workspace/pypto/.venv/lib/python3.11/site-packages/pypto/pypto_core.cpython-311-aarch64-linux-gnu.so') == CORE_SHA
    physical = descriptor['canonical']['dev'], descriptor['canonical']['inode']
    for pid in (PID, 1024041):
        value = actions.observation(physical, pid)
        assert len(value['maps']) == 1
        parts = value['maps'][0].split()
        low, high = (int(x, 16) for x in parts[0].split('-'))
        assert high - low == descriptor['nbytes'] and parts[1] == 'rw-s' and int(parts[2], 16) == 0
    tracker = results[1024040]
    assert tracker['cmdline'] == [driver.PYTHON, '-B', '-s', '-c', 'from multiprocessing.resource_tracker import main;main(25)']
    assert int(re.search(r'^PPid:\s+(\d+)', tracker['status'], re.M)[1]) == PID
    assert not actions.observation(physical, 1024040)['maps'] and not actions.observation(physical, 1024040)['fds']
    assert {x['fd'] for x in tracker['fds']} == {0, 1, 2, 25}
    assert tracker['fds'] and all(x['target'] == '/dev/null' or x['target'].endswith('/n004/native.log')
        or x['fd'] == 25 and x['target'].startswith('pipe:[') for x in tracker['fds'])
    helper_inventory = inventory(1024040)
    driver.save(root / 'tracker-inventory.json', helper_inventory)
    calibration = helper_inventory['device_calibration']
    assert not calibration['errors']['before'] and not calibration['errors']['after']
    identity, errors = _validated_calibration(calibration['before'])
    assert not errors
    majors = set(identity['ascend_character_drivers'])
    assert all(not (stat.S_ISCHR(fd['mode']) and str(fd['rdev'][0]) in majors)
        and not fd['target'].startswith(('/dev/davinci', '/dev/devmm', '/dev/hisi', '/dev/ascend'))
        for fd in helper_inventory['fds'].values())
    assert not helper_inventory['loaded_hashes'] and not helper_inventory['LD_PRELOAD']
    return physical


def run(root, shared, execute):
    root.mkdir()
    for name in ('logs', 'inventories'):
        (root / name).mkdir()
    sys.path[:0] = [str(SOURCE / 'examples/run04'), str(SOURCE / 'examples/run02')]
    from r_guard import snapshot
    service, descriptor, before = setup_service(root)
    physical = target_gate(root, descriptor)
    lockpath = Path('/workspace/uflow/locks/run02-npu-10.lock')
    with lockpath.open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource = snapshot(root, 'recovery-before', 10)
        driver.save(root / 'resource-before.json', resource)
        assert resource['status']['health'] == 'OK' and not resource['fds']['denied']
        assert {x['pid'] for x in resource['fds']['users']} <= {PID, 1024041}
        # The original context is deliberately still alive. Reject other SMI
        # occupants on this board; do not require idle/HBM-free for recovery.
        board = resource['status']['board']
        occupants = []
        for line in resource['npu'].splitlines():
            match = re.fullmatch(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|.*', line)
            if match and int(match[1]) == board:
                occupants.append(int(match[3]))
        assert occupants and set(occupants) <= {PID, 1024041}, occupants
        sock.bind(('127.0.0.1', 18892))
        driver.save(root / 'reservation.json', {'controller_pid': os.getpid(), 'lock_path': str(lockpath),
            'port': 18892, 'allowed_live_original_context': {str(p): CHILDREN.get(p, BIRTH) for p in (PID, 1024041)},
            'actual_SMI_occupants': occupants, 'scope': 'advisory held throughout real recovery; not idle device claim'})
        cfg = {'mode': 'n004', 'pid': PID, 'birth': BIRTH, 'children': CHILDREN,
            'mapping_children': {1024041: CHILDREN[1024041]}, 'tracker_pid': 1024040, 'tracker_birth': CHILDREN[1024040],
            'pool_id': descriptor['pool_id'], 'pool_epoch': descriptor['pool_epoch'],
            'provider_session': descriptor['provider_session'], 'canonical': descriptor['canonical'],
            'client_id': actions.CLIENT_ID, 'physical': physical, 'SDK_sha256': SDK_SHA, 'core_sha256': CORE_SHA,
            'source_hashes': SOURCE_HASHES, 'target_source': str(SOURCE / 'examples/run04/pool_native_probe.py'),
            'expected_error': 'native child has no whole shared RW mapping of the UFlow resource',
            'allowed_unwound_sources': list(SOURCE_HASHES), 'original_pool': service.expected,
            'unregister_operation': 'n004-recovery-r001-unregister', 'unmap_operation': 'n004-recovery-r001-unmap'}
        driver.save(root / 'config.json', cfg)
        driver.save(root / 'fresh-gates-passed.json', {'passed': True, 'execute_requested': execute,
            'original_born_client': [PID, BIRTH], 'charged_bytes': before['charged_bytes'], 'new_pool_create_calls': 0})
        if not execute:
            print('Fresh read-only recovery gates passed; no attach or signals')
            return
        assert driver.install(root, shared), 'install failed; preserve original PID/context'
        assert driver.request_cleanup(root), 'cleanup failed; preserve original PID/context and fixed operation IDs'
        deadline = time.monotonic() + 30
        while actions.birth(PID) == BIRTH:
            assert time.monotonic() < deadline, 'success receipt but original PID still alive; retain pool'
            time.sleep(.1)
        assert all(actions.birth(p) != b for p, b in CHILDREN.items())
        receipt = json.loads((root / 'cleanup-success.json').read_text())
        assert receipt['pid'] == PID and receipt['birth'] == BIRTH and receipt['mode'] == 'n004'
        assert receipt['worker_close']['close_complete'] and not receipt['unmapped_observation']['maps']
        assert not receipt['unmapped_observation']['fds']
        # Read the existing capability only after the original peer has proven
        # real drain/unmap/ACK and exited. Never serialize/pass it to children.
        service._admin = (STATE / 'admin.token').read_text()
        released = service.destroy('after-n004-recovery-destroy')
        after = snapshot(root, 'recovery-after', 10)
        driver.save(root / 'resource-after.json', after)
        assert after['status']['health'] == 'OK' and not after['fds']['denied'] and not after['fds']['users']
        assert f'No running processes found in NPU {board}' in after['npu']
        driver.save(root / 'recovery-result.json', {'passed': True, 'original_client_actual_close_ACK_normal_exit': True,
            'original_chip_helper_births_gone': True, 'new_pool_real_UF_destroy': released,
            'old_quarantine_unchanged_and_still_charged_bytes': 134217728, 'new_pool_create_calls': 0,
            'source_runtime_unchanged': True, 'scope': 'n004 recovery only; S1 native/model not yet passed'})
        print('n004 original-client recovery and real UF destroy passed; retained old charge 128MiB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.root, args.shared, args.execute)
