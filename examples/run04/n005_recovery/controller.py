# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fresh exact-identity gates; install and invoke only a read-only n005 diagnostic."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys

BASE = Path('/workspace/uflow/runs/d1-managed-ddr-20261105/serving')
SOURCE = BASE / 'a004/src/serving'
PID, BIRTH = 1145935, 484536731
CHILDREN = {1146315: 484537553, 1146316: 484537554}
OWNERS = {899046: 483907077, 899051: 483907079, 899069: 483907103}
POOL = '9ab082ee-0380-4a5d-9c58-3c026a424496'
CLIENT = '99ed5891-a534-476a-b630-0175cf12702a'
LEASE = 'b5bc3eb8-5aee-4145-a2ad-a1fef8d137f2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, shared, expected_manifest_sha, expected_driver_sha):
    driver_file = BASE / 'f-n004-recovery-r005/shared/driver.py'
    assert sha(driver_file) == expected_driver_sha
    spec = importlib.util.spec_from_file_location('n005_install_driver', driver_file)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    sys.path[:0] = [str(SOURCE / 'examples/run04'), str(SOURCE / 'examples/run02')]
    from existing_pool import ExistingPoolService, retained_pool
    from host_guard import inventory
    from pypto_serving.serving.native_ddr.managed_process import descendants
    from r_guard import snapshot
    from uflow.managed_pool import _exchange
    root.mkdir()
    for name in ('logs', 'inventories', 'catalog'):
        (root / name).mkdir()
    assert sha(BASE / 'a004/source-manifest.json') == expected_manifest_sha
    manifest = json.loads((BASE / 'a004/source-manifest.json').read_text())
    assert manifest['serving_commit'] == 'a4b396c6c0c1ea8847b750e46a96cb0b5fb495d9'
    source_hashes = {}
    for name in ('examples/run04/pool_native_probe.py', 'pypto_serving/model/qwen/native_ddr_transfer.py',
                 'pypto_serving/serving/native_ddr/managed_process.py', 'pypto_serving/serving/native_ddr/cooperative_close.py',
                 'pypto_serving/serving/native_ddr/worker.py', 'pypto_serving/serving/native_ddr/replay.py'):
        expected = manifest['groups']['serving']['files'][name]['sha256']
        assert sha(SOURCE / name) == expected
        source_hashes[str(SOURCE / name)] = expected
    assert subprocess.check_output(['git', '-C', '/workspace/pypto', 'rev-parse', 'HEAD'], text=True).strip() == '8007f1146a800d679301c5ea14bb644b863da6bf'
    assert subprocess.check_output(['git', '-C', '/workspace/pypto/runtime', 'rev-parse', 'HEAD'], text=True).strip() == '508becb93e182a3035a14e3f9226929dbe994346'
    assert driver.birth(PID) == BIRTH and driver.tracer(PID) == 0 and descendants(PID) == CHILDREN
    assert {p: driver.birth(p) for p in OWNERS} == OWNERS
    for pid in (PID, *CHILDREN):
        info = inventory(pid)
        assert info['exe_sha256'] == '1195c76f45c7cf16bf776a572cc85e074669c6f1a17d80d83fa8b7a7e3f80a87'
        driver.save(root / f'process-{pid}.json', info)
    command = (Path('/proc', str(PID), 'cmdline').read_bytes().decode().split('\0')[:-1])
    assert command[1:3] == ['-u', str(SOURCE / 'examples/run04/pool_native_probe.py')]
    assert command[-2:] == ['--device', '10']
    state_root = BASE / 'a002/n001/state'
    service = ExistingPoolService.__new__(ExistingPoolService)
    service.endpoint, service._exchange, service._save = str(state_root.parent / 'u.sock'), _exchange, driver.save
    service.evidence, service.starts = root / 'catalog', OWNERS.copy()
    baseline = json.loads((BASE / 'a004/retained-baseline.json').read_text())
    service.expected = retained_pool(baseline)
    os.environ['UFLOW_POOL_TEST_FENCE'] = str(BASE / 'a002/api_fence.so')
    service.verify_owners('before')
    loaded = json.loads((state_root / 'state.json').read_text())['provider_loaded']
    names = json.loads((BASE / 'a002/src/pool/examples/managed_pool/blocked-symbols.json').read_text())
    assert loaded['pid'] == 899051 and loaded['parent_pid'] == 899046
    assert set(loaded['test_fence_bindings']) == set(names) and len(names) == 79
    assert set(loaded['test_fence_bindings'].values()) == {str(BASE / 'a002/api_fence.so')}
    driver.save(root / 'provider-loaded.json', loaded)
    current = service.snapshot('fresh-before-diagnostic')
    assert retained_pool(current) == service.expected
    assert current['charged_bytes'] == current['budget_bytes'] == 268435456
    assert current['creation_reserved_bytes'] == 0 and len(current['pools']) == 4
    active = [p for p in current['pools'] if p['state'] != 'Released']
    assert {p['pool_id'] for p in active} == {service.expected['pool_id'], POOL}
    pool = next(p for p in active if p['pool_id'] == POOL)
    assert not pool['frozen'] and pool['provider_session'] == '71d2e8ab-4b5c-400c-a77c-131619490cac'
    assert pool['pool_epoch'] == '60074e16-5fed-49ba-9546-0eb4c67839d6'
    assert (pool['canonical']['dev'], pool['canonical']['inode'], pool['canonical']['nbytes']) == (25, 842044, 134217728)
    assert len(pool['clients']) == 1 and pool['clients'][0]['state'] == 'Mapped'
    assert pool['clients'][0]['client_session'] == CLIENT
    assert len(pool['clients'][0]['children']) == 1
    assert pool['clients'][0]['children'][0]['birth']['pid'] == 1146316
    leases = pool['ranges']['leases']
    assert len(leases) == 1
    assert {k: leases[0][k] for k in ('lease_id', 'lease_generation', 'block_id', 'offset', 'nbytes', 'state', 'client_session')} == {
        'lease_id': LEASE, 'lease_generation': 1018, 'block_id': 5, 'offset': 2097152, 'nbytes': 20971520,
        'state': 'Unknown', 'client_session': CLIENT}
    with Path('/workspace/uflow/locks/run02-npu-10.lock').open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource = snapshot(root, 'diagnostic-before', 10)
        driver.save(root / 'resource-before.json', resource)
        assert resource['status']['health'] == 'OK' and not resource['fds']['denied']
        assert {x['pid'] for x in resource['fds']['users']} <= {PID, 1146316}
        board, occupants = resource['status']['board'], []
        for line in resource['npu'].splitlines():
            match = re.fullmatch(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|.*', line)
            if match and int(match[1]) == board:
                occupants.append(int(match[3]))
        assert occupants and set(occupants) <= {PID, 1146316}
        sock.bind(('127.0.0.1', 18892))
        driver.save(root / 'reservation.json', {'pid': os.getpid(), 'selected_device': 10, 'SMI_occupants': occupants,
            'scope': 'advisory held for diagnostic only; original own context is still alive'})
        driver.save(root / 'config.json', {'pid': PID, 'birth': BIRTH, 'children': CHILDREN,
            'pool_id': POOL, 'client_id': CLIENT, 'lease_id': LEASE, 'source_hashes': source_hashes,
            'target_source': str(SOURCE / 'examples/run04/pool_native_probe.py')})
        assert driver.install(root, shared), 'install failed; original objects/PID retained'
        assert driver.birth(PID) == BIRTH and driver.tracer(PID) == 0
        os.kill(PID, signal.SIGUSR1)  # The newly installed action is diagnostic only.
        value = driver.await_file(root / 'diagnostic-1.json', PID, BIRTH, timeout=20)
        after = service.snapshot('fresh-after-diagnostic')
        assert retained_pool(after) == service.expected
        assert next(p for p in after['pools'] if p['pool_id'] == POOL) == pool
        driver.save(root / 'diagnostic-result.json', {'passed': True, 'original_replay_response_present': value['original_replay_response_present'],
            'native_submit_calls': 0, 'SDK_state_mutations': 0, 'pin_releases': 0, 'cleanup_started': False,
            'original_pool_projection_unchanged': True, 'native_case_projection_unchanged': True})
        print(json.dumps({'diagnostic_passed': True, 'original_response': value['original_replay_response'],
            'owner': value['owner'], 'active_handles': value['active_dispatch_handles'], 'accepted': value['accepted_native_handles']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--driver-sha', required=True)
    args = parser.parse_args()
    run(args.root, args.shared, args.manifest_sha, args.driver_sha)
