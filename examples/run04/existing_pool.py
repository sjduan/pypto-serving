# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Controller-only bounded continuation on the existing Catalog, preserving its Unknown pool."""
import copy
import json
import os
from pathlib import Path

POOL_BYTES = 134217728
POOL_ID = '9f1d7e43-7fa5-456d-b03e-014d32115ad3'
EPOCH = '611b006a-cb89-481d-9c66-3c732fb795db'
PROVIDER = '71d2e8ab-4b5c-400c-a77c-131619490cac'
CLIENT = '2de47d55-5a1a-471a-9fce-508ee0f58f41'
OWNER_PIDS = (899046, 899051, 899069)
SERVICE_ROOT = Path('/workspace/uflow/runs/d1-managed-ddr-20261105/serving/a002')
STATE_ROOT = SERVICE_ROOT / 'n001/state'
FENCE_SHA = 'babe1a791fa82551a5cfdcec83eceeb104ca3c5d6ff1b81cacdc0e42e48c9c9e'
DAEMON_SHA = 'd4a84f6be5923fb57b8180f0aec67288b75060c0fe41993a517f59efff4ad09b'
PYTHON_SHA = '1195c76f45c7cf16bf776a572cc85e074669c6f1a17d80d83fa8b7a7e3f80a87'


def retained_pool(snapshot):
    pools = [p for p in snapshot['pools'] if p['pool_id'] == POOL_ID]
    if len(pools) != 1:
        raise ValueError('exact original quarantine pool is missing or repeated')
    pool = pools[0]
    if ((pool['pool_epoch'], pool['provider_session'], pool['state'], pool['frozen'], pool['nbytes'])
            != (EPOCH, PROVIDER, 'Allocated', True, POOL_BYTES)):
        raise ValueError('original quarantine identity/state/capacity changed')
    if (len(pool['clients']) != 1 or pool['clients'][0]['client_session'] != CLIENT
            or pool['clients'][0]['state'] != 'Unknown' or pool['clients'][0]['children']
            or pool['ranges']['leases']):
        raise ValueError('original quarantine access projection changed')
    return pool


def check_catalog(snapshot, expected, normal_id=None):
    """Reject hidden charge, changed quarantine, a second live pool or new Unknown."""
    if json.dumps(retained_pool(snapshot), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('original quarantine projection is not the sealed observation')
    if any(type(snapshot[k]) is not int for k in ('budget_bytes', 'charged_bytes', 'creation_reserved_bytes')):
        raise ValueError('pool budget counters must be exact integers')
    if snapshot['budget_bytes'] != 2 * POOL_BYTES or snapshot['creation_reserved_bytes'] != 0:
        raise ValueError('same Catalog budget/reservation differs from the continuation contract')
    if len(snapshot['pools']) > 8:
        raise ValueError('bounded retained pool history exceeded')
    for pool in snapshot['pools']:
        if pool['pool_id'] != POOL_ID and pool['state'] == 'Released' and (
                any(c['state'] != 'Unmapped' or c['children'] for c in pool['clients'])
                or pool['ranges']['leases']):
            raise ValueError('released record has unexpected remaining access obligations')
    active = [p for p in snapshot['pools'] if p['state'] != 'Released']
    allowed = {POOL_ID} if normal_id is None else {POOL_ID, normal_id}
    if {p['pool_id'] for p in active} != allowed or len(active) != len(allowed):
        raise ValueError('unexpected additional live pool; no fallback authority allowed')
    expected_charge = len(allowed) * POOL_BYTES
    if snapshot['charged_bytes'] != expected_charge or sum(p['nbytes'] for p in active) != expected_charge:
        raise ValueError('physical pool charge does not include quarantine and normal pool')
    if normal_id is not None:
        normal = next(p for p in active if p['pool_id'] == normal_id)
        if (normal['provider_session'] != PROVIDER or normal['state'] != 'Allocated'
                or normal['nbytes'] != POOL_BYTES or normal['frozen']
                or any(c['state'] == 'Unknown' or any(x['unknown'] for x in c['children'])
                       for c in normal['clients'])
                or any(l['state'] == 'Unknown' for l in normal['ranges']['leases'])):
            raise ValueError('new normal pool is frozen/Unknown; stop without another pool')
        if (normal['resource_id'] == expected['resource_id']
                or (normal['canonical']['dev'], normal['canonical']['inode'])
                == (expected['canonical']['dev'], expected['canonical']['inode'])):
            raise ValueError('normal pool must not alias the quarantined physical resource')
    return {'quarantine': copy.deepcopy(expected), 'normal_pool_id': normal_id,
            'charged_bytes': snapshot['charged_bytes'], 'reserved_bytes': 0,
            'budget_bytes': snapshot['budget_bytes'], 'revision': snapshot['revision']}


def process_start(pid):
    return int((Path('/proc') / str(pid) / 'stat').read_text().rpartition(')')[2].split()[19])


class ExistingPoolService:
    """Never launches, shuts down or replaces the retained daemon/provider."""

    def __init__(self, state_root, baseline, evidence):
        from uflow.managed_pool import _exchange
        from pool_host_probe import save
        self.state_root = Path(state_root)
        if self.state_root != STATE_ROOT:
            raise ValueError('only the explicitly retained existing service may be used')
        self.evidence = Path(evidence)
        self.endpoint = str(self.state_root.parent / 'u.sock')
        self.expected = copy.deepcopy(retained_pool(json.loads(Path(baseline).read_text())))
        self._exchange = _exchange
        self._save = save
        self.normal = None
        self.starts = None
        self._admin = None
        self.verify_owners('before')
        initial = self.snapshot('before-create')
        check_catalog(initial, self.expected)
        loaded = json.loads((self.state_root / 'state.json').read_text())['provider_loaded']
        names = json.loads((SERVICE_ROOT / 'src/pool/examples/managed_pool/blocked-symbols.json').read_text())
        bindings = loaded['test_fence_bindings']
        if (loaded['provider_session'] != PROVIDER or loaded['pid'] != OWNER_PIDS[1]
                or loaded['parent_pid'] != OWNER_PIDS[0] or set(bindings) != set(names) or len(bindings) != 79
                or set(bindings.values()) != {str(SERVICE_ROOT / 'api_fence.so')}
                or loaded['simpler_revision'] != '508becb93e182a3035a14e3f9226929dbe994346'
                or loaded['simpler_sha256'] != 'ee1e5b64d5f836950892fc3d6e986a3eda8d106b7c92f27f8420589184073f96'
                or loaded['task_interface_sha256'] != 'd7945a85cc49aedeed14006f1c7d2077807f6fb5e9565fe21cfa6cb9e694769d'):
            raise ValueError('existing provider source/module/79-fence bindings changed')
        self._save(self.evidence / 'existing-provider-loaded.json', loaded)
        if len(initial['pools']) >= 8:
            raise ValueError('no bounded pool record remains; do not create another Catalog')
        # Capability stays in this controller only; never serialized or passed to a child.
        self._admin = (self.state_root / 'admin.token').read_text()

    def verify_owners(self, label):
        from host_guard import inventory, judge, sha
        from pool_host_probe import save
        values = {}
        starts = {pid: process_start(pid) for pid in OWNER_PIDS}
        cutoff = self.expected['clients'][0]['birth']['starttime_ticks']
        if any(value > cutoff for value in starts.values()) or self.starts is not None and starts != self.starts:
            raise ValueError('retained owner process birth changed or PID was reused')
        fence = str(SERVICE_ROOT / 'api_fence.so')
        if os.environ.get('UFLOW_POOL_TEST_FENCE') != fence or sha(fence) != FENCE_SHA:
            raise ValueError('host verification must use the original loaded fence path/hash')
        for pid, role in zip(OWNER_PIDS, ('daemon', 'provider', 'tracker')):
            value = inventory(pid)
            save(self.evidence / f'{label}-{role}-{pid}.json', value)
            judge(value, role)
            if value['exe_sha256'] != (DAEMON_SHA if role == 'daemon' else PYTHON_SHA):
                raise ValueError('retained owner executable changed')
            if value['LD_PRELOAD'] != fence:
                raise ValueError('retained owner lost original fence')
            values[role] = value
        if (values['daemon']['exe'] != str(SERVICE_ROOT / 'uf-daemon')
                or values['provider']['children'] != [OWNER_PIDS[2]]):
            raise ValueError('retained owner tree/executable path differs from original evidence')
        if self.starts is None:
            self.starts = starts
        save(self.evidence / f'{label}-owner-births.json', starts)
        return values

    def snapshot(self, label):
        state, fds = self._exchange(self.endpoint, 'GetManagedPoolSnapshot',
            {'pool_id': POOL_ID, 'pool_epoch': EPOCH, 'provider_session': PROVIDER}, timeout=5.0)
        if fds:
            raise ValueError('metadata-only snapshot unexpectedly returned access FDs')
        self._save(self.evidence / f'{label}-snapshot.json', state)
        return state

    def create(self):
        from uflow.managed_pool import create_pool
        if self.normal is not None:
            raise RuntimeError('one normal pool at a time')
        check_catalog(self.snapshot('pre-create-recheck'), self.expected)
        descriptor = create_pool(self.endpoint, admin_token=self._admin)
        self.normal = descriptor
        if (descriptor['pool_id'] == POOL_ID or descriptor['provider_session'] != PROVIDER
                or descriptor['pool_epoch'] == EPOCH):
            raise ValueError('normal pool identity is not a separate range resource of the same provider')
        self._save(self.evidence / 'normal-descriptor.json', descriptor)
        check_catalog(self.snapshot('after-create'), self.expected, descriptor['pool_id'])
        return descriptor

    def current(self, label):
        if self.normal is None:
            raise RuntimeError('normal pool has not been created')
        snapshot = self.snapshot(label)
        check_catalog(snapshot, self.expected, self.normal['pool_id'])
        pool = next(p for p in snapshot['pools'] if p['pool_id'] == self.normal['pool_id'])
        if any(pool[key] != value for key, value in self.normal.items()):
            raise ValueError('normal pool descriptor changed during the case')
        return snapshot, pool

    def destroy(self, label='after-destroy'):
        from uflow.managed_pool import destroy_pool
        _, pool = self.current('before-' + label)
        if (any(c['state'] != 'Unmapped' or c['children'] for c in pool['clients'])
                or pool['ranges']['leases']):
            raise RuntimeError('normal client/range/child obligations remain; do not release')
        descriptor = self.normal
        result = destroy_pool(self.endpoint, descriptor, admin_token=self._admin)
        self._save(self.evidence / 'normal-provider-release.json', result)
        after = self.snapshot(label)
        check_catalog(after, self.expected)
        name = bytes.fromhex(descriptor['canonical']['body_hex']).decode()
        if Path(name).name != name or (Path('/dev/shm') / name).exists():
            raise RuntimeError('normal owner name still exists after actual UF release')
        physical = (descriptor['canonical']['dev'], descriptor['canonical']['inode'])
        # Actual physical FD release is scoped to the retained owners; the
        # consumer/reader independently prove their own unmap and PID exit.
        for pid in OWNER_PIDS:
            for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
                try:
                    info = fd.stat()
                except FileNotFoundError:
                    continue
                if (info.st_dev, info.st_ino) == physical:
                    raise RuntimeError('normal pool FD remains in an existing service owner')
        self.verify_owners('after')
        self.normal = None
        return {'provider_release': result, 'normal_shm_name_absent': name,
                'normal_physical_owner_FDs_absent': True, 'charged_bytes_after': after['charged_bytes'],
                'retained_owner_pids': list(OWNER_PIDS), 'retained_owner_births': self.starts,
                'scope': 'normal pool released; original Unknown remains charged; service not shut down'}
