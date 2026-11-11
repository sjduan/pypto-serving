# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real accepted P1 host assembly: controller/admin and separate non-admin consumers.

No native owner/model is created. Host processes alone use the pinned test fence.
Every diagnostic pool byte is covered by an actual R lease during capture.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from types import SimpleNamespace


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(buffer):
    return hashlib.sha256(buffer).hexdigest()


def inventory(root, role, label, pid=None):
    from host_guard import save_guard
    os.environ['UFLOW_POOL_TEST_INVENTORY_ROOT'] = str(root / 'inventories')
    proof = save_guard(os.getpid() if pid is None else pid, role, label)
    actual = json.loads(Path(proof['inventory_file']).read_text())
    return {**proof, 'children': actual['children'], 'scope': 'actual fenced host FD/maps inventory'}


def packet(process, timeout=30):
    with selectors.DefaultSelector() as poll:
        poll.register(process.stdout, selectors.EVENT_READ)
        if not poll.select(timeout):
            raise TimeoutError('owned host consumer did not return its bounded receipt')
    raw = process.stdout.readline(65537)
    if not raw or len(raw.encode()) > 65536:
        raise ValueError('missing/oversized consumer control receipt')
    return json.loads(raw)


def capture(client, ranges, path=None):
    held = []
    try:
        for item in ranges:
            held.append(client.acquire_range(item['block'], 'r', item['block'].generation,
                byte_offset=item['offset'], nbytes=item['nbytes']))
        actual = digest(client.view.buffer)
        if path is not None:
            with Path(path).open('xb') as stream:
                stream.write(client.view.buffer)
        return actual
    finally:
        for lease in held:
            client.release(lease)


def consumer(args):
    from uflow.managed_pool import ManagedPoolClient, PoolBlock
    from pypto_serving.serving.native_ddr.managed_pool import BLOCK_BYTES, GUARD_BYTES
    from pypto_serving.serving.native_ddr.managed_pool_sdk import SDK_SHA256, register_managed_pool
    root = args.root
    root.mkdir()
    (root / 'inventories').mkdir()
    if args.role == 'reader':
        # Receives only public descriptor/range metadata, never the admin token.
        expected = json.loads(args.expected.read_text())
        client = ManagedPoolClient.register(args.endpoint, args.pool_id, args.epoch, access='r')
        ranges = [{**item, 'block': PoolBlock.from_dict(item['block'])} for item in expected['ranges']]
        actual = capture(client, ranges)
        assert actual == expected['pool_sha256']
        proof = inventory(root, 'client', 'independent-read-only')
        client.begin_unregister()
        client.view.close()
        ack = client.ack_unmapped(native_drained=True)
        result = {'pool_sha256': actual, 'registered': client.view.descriptor,
                  'unmap_ack': ack, 'inventory': proof, 'native_jobs_started': 0}
        save(root / 'result.json', result)
        print(json.dumps({'done': True, 'pid': os.getpid()}), flush=True)
        return
    import torch
    from pypto_serving.model.qwen.native_ddr_transfer import ManagedDdrSlots
    from verify_bytes import verify_guards
    config = SimpleNamespace(pool_endpoint=args.endpoint, pool_id=args.pool_id, pool_epoch=args.epoch,
                               sdk_sha256=SDK_SHA256)
    access = register_managed_pool(config)
    client, slots = access.port.client, ManagedDdrSlots(5, access)
    assert slots.control_metadata_bytes == 4 and slots.control_metadata_mapped_bytes >= 4
    backings = [slots.reserve(index, BLOCK_BYTES) for index in range(5)]
    assert [item.byte_offset for item in backings] == [n << 20 for n in (2, 24, 46, 68, 90)]
    ranges = []
    for index, backing in enumerate(backings):
        raw = access.port._raw_block(backing.block)
        guard = client.acquire_range(raw, 'w', raw.generation, nbytes=GUARD_BYTES)
        client.view.buffer[raw.offset:raw.offset + GUARD_BYTES] = bytes([130 + index]) * GUARD_BYTES
        client.commit_write(guard, f'host-only-guard-{index}', GUARD_BYTES)
        ranges.append({'block': raw, 'offset': 0, 'nbytes': GUARD_BYTES})
        lease = access.acquire(backing.block, 'w', byte_offset=backing.byte_offset)
        backing.tensor.view(torch.uint8).fill_(0)
        access.complete_write(lease, f'host-only-initial-payload-{index}', f'initialize-{index}')
        ranges.append({'block': raw, 'offset': GUARD_BYTES, 'nbytes': BLOCK_BYTES})
    tail = client.allocate_range(18 << 20)
    assert tail.offset == 110 << 20
    lease = client.acquire_range(tail, 'w', tail.generation)
    client.view.buffer[tail.offset:tail.offset + tail.nbytes] = b'\x8c' * tail.nbytes
    client.commit_write(lease, 'host-only-tail', tail.nbytes)
    ranges.append({'block': tail, 'offset': 0, 'nbytes': tail.nbytes})
    before = root / 'pool-before.bin'
    capture(client, ranges, before)
    committed = []
    for index, backing in enumerate(backings):
        lease = access.acquire(backing.block, 'w', byte_offset=backing.byte_offset)
        backing.tensor.view(torch.uint8).fill_(20 + index)
        proof = access.complete_write(lease, f'host-only-final-payload-{index}', f'write-{index}')
        committed.append(asdict(proof))
    after = root / 'pool-after.bin'
    actual = capture(client, ranges, after)
    import mmap
    with before.open('rb') as a, after.open('rb') as b:
        with mmap.mmap(a.fileno(), 0, access=mmap.ACCESS_READ) as left:
            with mmap.mmap(b.fileno(), 0, access=mmap.ACCESS_READ) as right:
                guards = verify_guards(left, right, [(item.byte_offset, BLOCK_BYTES) for item in backings])
    save(root / 'guards.json', guards)
    proof = inventory(root, 'client', 'five-tensor-views')
    held = backings[0].tensor.view(torch.uint8).flatten()[64:192].view(torch.int16)
    for backing in backings:
        slots.release(backing)
    try:
        slots.close_after_owner()
    except BufferError:
        assert not client.view.closed and not access.closed
        blocked = inventory(root, 'client', 'live-alias-close-blocked')
        print(json.dumps({'blocked': True, 'pid': os.getpid(), 'inventory': blocked}), flush=True)
    else:
        raise AssertionError('live alias did not retain the real registered mmap/FD')
    if sys.stdin.readline().strip() != 'release-alias':
        raise RuntimeError('controller did not acknowledge the held-alias observation')
    del held
    slots.close_after_owner()
    assert client.view.closed and access.closed
    slots.close_after_owner()
    result = {'registered': asdict(access.identity), 'committed': committed,
              'ranges': [{**item, 'block': asdict(item['block'])} for item in ranges],
              'pool_sha256': actual, 'guard_bytes': guards['guard_bytes_compared'],
              'control_metadata_bytes': slots.control_metadata_bytes,
              'control_metadata_mapped_bytes': slots.control_metadata_mapped_bytes,
              'alias_close_blocked': True, 'unmapped_after_alias_release': True,
              'native_jobs_started': 0, 'inventory': proof}
    save(root / 'result.json', result)
    print(json.dumps({'done': True, 'pid': os.getpid()}), flush=True)


def controller(args):
    from uflow.managed_pool import create_pool, destroy_pool, shutdown_pool_service
    root = args.root
    root.mkdir(parents=False, exist_ok=False)
    (root / 'inventories').mkdir()
    # Keep the provider's eventual resource tracker under this owned controller.
    import ctypes
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('owned host controller could not become subreaper')
    endpoint, state = str(root / 'u.sock'), root / 'state'
    assert len(endpoint.encode()) < 108
    inventory(root, 'client', 'controller-start')
    daemon_log = (root / 'daemon.log').open('xb')
    daemon = subprocess.Popen([str(args.daemon), '--host-pool', '--socket', endpoint,
        '--pool-state-root', str(state), '--pool-provider-python', sys.executable],
        stdout=daemon_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 30
    while not Path(endpoint).exists():
        if daemon.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('new host daemon startup failed; preserve daemon log and state')
        time.sleep(.05)
    loaded = json.loads((state / 'state.json').read_text())['provider_loaded']
    provider = inventory(root, 'provider', 'provider-start', loaded['pid'])
    daemon_proof = inventory(root, 'daemon', 'daemon-start', daemon.pid)
    token = (state / 'admin.token').read_text()
    descriptor = create_pool(endpoint, admin_token=token)
    provider = inventory(root, 'provider', 'provider-pool-created', loaded['pid'])
    save(root / 'public-descriptor.json', descriptor)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--role', 'writer',
               '--root', str(root / 'writer'), '--endpoint', endpoint,
               '--pool-id', descriptor['pool_id'], '--epoch', descriptor['pool_epoch']]
    # The admin token is neither in this argv nor passed through the environment.
    with (root / 'writer.stderr').open('xb') as log:
        writer = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=log, text=True, bufsize=1)
        blocked = packet(writer, 90)
        assert blocked['blocked'] is True
        current = json.loads((state / 'state.json').read_text())
        pool = next(row for row in current['pools'] if row['pool_id'] == descriptor['pool_id'])
        assert current['charged_bytes'] == 128 << 20 and pool['frozen'] is True
        assert any(row['state'] == 'Unknown' and row['closing'] for row in pool['clients'])
        save(root / 'blocked-observation.json', {'charged_bytes': current['charged_bytes'],
            'frozen': pool['frozen'], 'clients': pool['clients'], 'writer': blocked})
        writer.stdin.write('release-alias\n')
        writer.stdin.flush()
        assert packet(writer)['done'] is True
        assert writer.wait(timeout=15) == 0
        writer.stdin.close()
        writer.stdout.close()
    current = json.loads((state / 'state.json').read_text())
    assert current['charged_bytes'] == 128 << 20
    assert not next(row for row in current['pools'] if row['pool_id'] == descriptor['pool_id'])['frozen']
    reader_command = [item if item != 'writer' else 'reader' for item in command]
    reader_command[reader_command.index('--root') + 1] = str(root / 'reader')
    reader_command += ['--expected', str(root / 'writer/result.json')]
    with (root / 'reader.stderr').open('xb') as log:
        reader = subprocess.Popen(reader_command, stdout=subprocess.PIPE, stderr=log, text=True)
        assert packet(reader, 60)['done'] is True
        assert reader.wait(timeout=15) == 0
        reader.stdout.close()
    released = destroy_pool(endpoint, descriptor, admin_token=token)
    shutdown = shutdown_pool_service(endpoint, admin_token=token)
    assert daemon.wait(timeout=20) == 0
    daemon_log.close()
    clean = json.loads((state / 'clean.json').read_text())
    assert clean['charged_bytes'] == clean['creation_reserved_bytes'] == 0
    assert clean['normal_shutdown'] is True and clean['provider_exited'] is True
    writer_result = json.loads((root / 'writer/result.json').read_text())
    reader_result = json.loads((root / 'reader/result.json').read_text())
    owned = {daemon.pid, writer.pid, reader.pid, loaded['pid'], *provider['children'],
             *writer_result['inventory']['children'], *reader_result['inventory']['children']}
    deadline = time.monotonic() + 10
    while any(Path('/proc', str(pid)).exists() for pid in owned):
        for pid in owned:
            try:
                os.waitpid(pid, os.WNOHANG)  # only reaps owned/adopted children; sends no signal
            except ChildProcessError:
                pass
        if time.monotonic() > deadline:
            raise RuntimeError('owned host PID has not exited; no clean postcheck')
        time.sleep(.05)
    assert not Path(endpoint).exists()
    shm_name = bytes.fromhex(descriptor['canonical']['body_hex']).decode()
    assert Path(shm_name).name == shm_name and not (Path('/dev/shm') / shm_name).exists()
    result = {'scope': 'real P1 SDK/provider and serving host views only; no native/model execution',
              'passed': True, 'descriptor': descriptor, 'provider_loaded': loaded,
              'daemon_inventory': daemon_proof, 'provider_inventory': provider,
              'consumer_pids': [writer.pid, reader.pid], 'provider_release': released,
              'shutdown': shutdown, 'clean': clean, 'native_jobs_started': 0,
              'owned_pids_absent': sorted(owned), 'provider_shm_name_absent': shm_name}
    save(root / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('controller', 'writer', 'reader'), default='controller')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--daemon', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--pool-id')
    parser.add_argument('--epoch')
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args()
    if args.role == 'controller':
        if args.daemon is None:
            parser.error('controller requires an exact pinned daemon')
        controller(args)
    else:
        consumer(args)


if __name__ == '__main__':
    main()
