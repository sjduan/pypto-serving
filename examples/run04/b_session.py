# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One B-local UFlow authority. A's historical responsibility is never queried."""
import json
import os
from pathlib import Path

POOL_BYTES = 134217728


def check_b_catalog(snapshot, normal_id=None, provider_session=None):
    for key in ('budget_bytes', 'charged_bytes', 'creation_reserved_bytes'):
        if type(snapshot[key]) is not int:
            raise ValueError('B budget counters must be exact integers')
    if (snapshot['budget_bytes'] != POOL_BYTES or snapshot['creation_reserved_bytes'] != 0
            or snapshot['charged_bytes'] != (POOL_BYTES if normal_id else 0)):
        raise ValueError('B Catalog cap/charge/reservation differs from the 128MiB hard limit')
    pools = snapshot['pools']
    if len(pools) > 4 or len({p['pool_id'] for p in pools}) != len(pools):
        raise ValueError('B pool history must contain at most four distinct records')
    live = [p for p in pools if p['state'] != 'Released']
    if [p['pool_id'] for p in live] != ([] if normal_id is None else [normal_id]):
        raise ValueError('unlisted live B pool or missing current resource')
    for pool in pools:
        if (pool['nbytes'] != POOL_BYTES or pool['frozen']
                or provider_session is not None and pool['provider_session'] != provider_session
                or pool['ranges']['leases'] and any(l['state'] == 'Unknown' for l in pool['ranges']['leases'])
                or any(c['state'] == 'Unknown' or any(x['unknown'] for x in c['children']) for c in pool['clients'])):
            raise ValueError('B resource identity/Unknown obligation changed; retain the original domain')
        if pool['state'] == 'Released' and (pool['ranges']['leases'] or any(
                c['state'] != 'Unmapped' or c['children'] for c in pool['clients'])):
            raise ValueError('released B history still has client/child/range obligations')
    return {'B_charged_bytes': snapshot['charged_bytes'], 'B_reserved_bytes': 0,
            'retained_A_charge_floor_bytes': POOL_BYTES,
            'task_responsibility_bytes': POOL_BYTES + snapshot['charged_bytes'],
            'task_total_cap_bytes': 2 * POOL_BYTES, 'same_B_Catalog': True,
            'A_fresh_status_queried': False}


class BPoolSession:
    """Controller keeps the private admin capability; never passes it to consumers."""
    def __init__(self, root, daemon, host_env):
        import subprocess
        import sys
        import time
        from host_guard import sha
        from existing_pool import DAEMON_SHA, FENCE_SHA
        from pool_host_probe import save
        self.root, self.daemon_path = Path(root), Path(daemon).resolve()
        self.root.mkdir()
        (self.root / 'inventories').mkdir()
        self.endpoint, self.state = str(self.root / 'u.sock'), self.root / 'state'
        if len(self.endpoint.encode()) >= 108:
            raise ValueError('B control socket path exceeds Linux bound')
        self.env, self.save, self.counter = host_env, save, 0
        self.normal = self.anchor = None
        self.owners = {}
        if sha(self.daemon_path) != DAEMON_SHA or sha(self.daemon_path.parent / 'api_fence.so') != FENCE_SHA:
            raise ValueError('B daemon/fence bytes differ from accepted P1 binary closure')
        self.log = (self.root / 'daemon.log').open('xb')
        self.command = [str(self.daemon_path), '--host-pool', '--socket', self.endpoint,
            '--pool-state-root', str(self.state), '--pool-provider-python', sys.executable,
            '--pool-budget-bytes', str(POOL_BYTES)]
        self.save(self.root / 'command.json', {'argv': self.command, 'admin_capability_in_argv_env': False})
        self.daemon = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT, env=host_env)
        deadline = time.monotonic() + 40
        while not Path(self.endpoint).exists() or not (self.state / 'state.json').exists():
            if self.daemon.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('B service startup failed; preserve original state/processes')
            time.sleep(.05)
        initial = json.loads((self.state / 'state.json').read_text())
        check_b_catalog(initial)
        self.loaded = initial['provider_loaded']
        self.provider_session = self.loaded['provider_session']
        if self.loaded['parent_pid'] != self.daemon.pid:
            raise ValueError('B provider is not a child of this single daemon')
        # Private capability remains in this object only. No copying/sync/logging.
        self._admin = (self.state / 'admin.token').read_text()
        self.verify_owners('start')

    def verify_owners(self, label):
        from host_guard import inventory, judge
        from existing_pool import DAEMON_SHA, PYTHON_SHA
        from pypto_serving.serving.native_ddr.managed_process import process_birth, descendants
        fence = str(self.daemon_path.parent / 'api_fence.so')
        current = {self.daemon.pid: process_birth(self.daemon.pid),
                   self.loaded['pid']: process_birth(self.loaded['pid'])}
        current.update(descendants(self.loaded['pid']))
        if any(value is None or pid in self.owners and self.owners[pid] != value for pid, value in current.items()):
            raise RuntimeError('B service owner birth changed')
        if self.anchor is not None and self.owners and current != self.owners:
            raise RuntimeError('B service owner tree changed after first allocation')
        self.owners = current
        bindings = self.loaded['test_fence_bindings']
        names = json.loads((self.daemon_path.parent / 'src/pool/examples/managed_pool/blocked-symbols.json').read_text())
        if (set(bindings) != set(names) or len(bindings) != 79 or set(bindings.values()) != {fence}
                or json.loads((self.state / 'state.json').read_text())['provider_loaded'] != self.loaded):
            raise ValueError('B provider loaded/fence/source identity changed')
        for pid in current:
            role = 'daemon' if pid == self.daemon.pid else 'provider' if pid == self.loaded['pid'] else 'tracker'
            value = inventory(pid)
            self.save(self.root / f'{label}-{role}-{pid}.json', value)
            judge(value, role)
            if value['exe_sha256'] != (DAEMON_SHA if role == 'daemon' else PYTHON_SHA) or value['LD_PRELOAD'] != fence:
                raise ValueError('B service executable/fence differs from pinned assembly')
        self.save(self.root / f'{label}-births.json', current)

    def snapshot(self, label):
        self.counter += 1
        if self.anchor is None:
            value = json.loads((self.state / 'state.json').read_text())
        else:
            from uflow.managed_pool import _exchange
            value, fds = _exchange(self.endpoint, 'GetManagedPoolSnapshot',
                {k: self.anchor[k] for k in ('pool_id', 'pool_epoch', 'provider_session')}, timeout=5)
            if fds:
                raise RuntimeError('B metadata snapshot returned unexpected FDs')
        self.save(self.root / f'{self.counter:03d}-{label}.json', {'snapshot': value})
        accounting = check_b_catalog(value, self.normal['pool_id'] if self.normal else None, self.provider_session)
        self.save(self.root / f'{self.counter:03d}-{label}-responsibility.json', accounting)
        return value

    def create(self):
        from uflow.managed_pool import create_pool, ManagedPoolError, ManagedPoolUnknown
        if self.normal is not None:
            raise RuntimeError('only one normal B pool at a time')
        self.snapshot('before-create')
        descriptor = create_pool(self.endpoint, admin_token=self._admin)
        if descriptor['provider_session'] != self.provider_session:
            raise ValueError('B allocation changed provider incarnation')
        self.normal = descriptor
        first = self.anchor is None
        if first:
            self.anchor = descriptor
            # The first allocation may start the provider resource tracker.
            self.owners = {}
        self.snapshot('after-create')
        self.verify_owners(f'created-{self.counter:03d}')
        if first:
            try:
                create_pool(self.endpoint, admin_token=self._admin)
            except ManagedPoolUnknown:
                raise RuntimeError('B second allocation outcome Unknown; preserve domain')
            except ManagedPoolError as error:
                if not error.response or error.response.get('code') != 'Rejected':
                    raise
                self.save(self.root / 'actual-hard-cap-rejection.json', {'error': str(error),
                    'response': error.response, 'scope': 'real Catalog rejected second 128MiB allocation'})
                self.snapshot('after-real-budget-rejection')
            else:
                raise RuntimeError('B Catalog accepted two physical pools despite its hard cap')
        return descriptor

    def current(self, label):
        value = self.snapshot(label)
        pool = next(p for p in value['pools'] if self.normal and p['pool_id'] == self.normal['pool_id'])
        if any(pool[key] != value for key, value in self.normal.items()):
            raise ValueError('B current physical descriptor changed')
        return value, pool

    def destroy(self):
        from uflow.managed_pool import destroy_pool
        _, pool = self.current('before-destroy')
        if pool['ranges']['leases'] or any(c['state'] != 'Unmapped' or c['children'] for c in pool['clients']):
            raise RuntimeError('B client/child/range obligations remain; no budget release')
        descriptor = self.normal
        result = destroy_pool(self.endpoint, descriptor, admin_token=self._admin)
        self.normal = None
        after = self.snapshot('after-destroy')
        name = bytes.fromhex(descriptor['canonical']['body_hex']).decode()
        if Path(name).name != name or (Path('/dev/shm') / name).exists():
            raise RuntimeError('B provider name remains after real Destroy')
        for pid in self.owners:
            for fd in (Path('/proc') / str(pid) / 'fd').iterdir():
                try:
                    info = fd.stat()
                except FileNotFoundError:
                    continue
                if (info.st_dev, info.st_ino) == (descriptor['canonical']['dev'], descriptor['canonical']['inode']):
                    raise RuntimeError('B physical pool FD remains in original service owner')
        self.verify_owners(f'released-{self.counter:03d}')
        return {'provider_release': result, 'B_charged_after': after['charged_bytes'],
                'task_retained_A_floor_after': POOL_BYTES, 'shm_name_absent': name,
                'B_owner_pool_FDs_absent': True, 'service_shutdown': False}

    def shutdown(self):
        import time
        from uflow.managed_pool import shutdown_pool_service
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        self.snapshot('before-service-shutdown')
        result = shutdown_pool_service(self.endpoint, admin_token=self._admin)
        if self.daemon.wait(timeout=25) != 0:
            raise RuntimeError('B normal service shutdown failed')
        self.log.close()
        deadline = time.monotonic() + 20
        while True:
            for pid in self.owners:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            if not any(process_birth(pid) == birth for pid, birth in self.owners.items()):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('B original owner birth remains after normal shutdown')
            time.sleep(.05)
        clean = json.loads((self.state / 'clean.json').read_text())
        if (clean['charged_bytes'] or clean['creation_reserved_bytes'] or not clean['normal_shutdown']
                or not clean['provider_exited'] or Path(self.endpoint).exists()):
            raise RuntimeError('B final clean accounting/socket/provider exit was not proven')
        return {'shutdown': result, 'clean': clean, 'B_owner_births_gone': self.owners,
                'A_retained_floor_bytes': POOL_BYTES, 'A_queried': False}
