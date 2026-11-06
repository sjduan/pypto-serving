# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual common UF/SDK independent-process host proof, never native/model evidence."""
from dataclasses import asdict
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from pypto_serving.serving.native_ddr.managed_pool import (
    BLOCK_BYTES, GUARD_BYTES, CommittedRange, PoolBlock, StaleCommittedRange, prefix_content_id,
)
from pypto_serving.serving.native_ddr.managed_pool_sdk import SDK_SHA256, register_managed_pool
from pypto_serving.serving.native_ddr.shared_directory import DirectoryClient
from pypto_serving.serving.native_ddr.shared_pool import load_bindings
from pypto_serving.serving.native_ddr.types import NativeDdrPrefixConfig, PrefixKey


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def key(number):
    return PrefixKey(b'host-only-not-a-loaded-model!!!!!'[:32], bytes([number]) * 32)


def open_access(args):
    manifest = json.loads(Path(args.manifest).read_text())
    descriptor = manifest['pool_descriptor']
    config = NativeDdrPrefixConfig(True, 4 * BLOCK_BYTES, pool_endpoint=manifest['endpoint'],
        pool_id=descriptor['pool_id'], pool_epoch=descriptor['pool_epoch'], sdk_sha256=SDK_SHA256,
        shared_directory_endpoint=args.endpoint, shared_directory_epoch=args.epoch, shared_slots_path=args.bindings)
    access = register_managed_pool(config)
    save(Path(args.root) / (args.role + f'-{os.getpid()}-registered.json'), {'identity': asdict(access.identity)})
    try:
        load_bindings(args.bindings, access)
    except BaseException:
        access.close()
        raise
    return access


@contextmanager
def retained_access(args):
    access = open_access(args)
    access.host_intents = []
    try:
        yield access
    finally:
        if not access.closed:
            if access.leases or access.children or access.unknown:
                save(Path(args.root) / (f'retained-{os.getpid()}.json'), {
                    'client_session': access.identity.client_session, 'traceback': traceback.format_exc(),
                    'range_leases': list(access.leases), 'unknown': list(access.unknown),
                    'action': 'retain original process/mapping; no PID death or timeout cleanup'})
                while True:
                    time.sleep(30)
            for k, owner in access.host_intents:
                DirectoryClient(args.endpoint, args.epoch).cancel(k, owner)
            access.close()


def write_prefix(access, directory, index, byte):
    k, owner = key(index), f'host-{os.getpid()}-{index}-{uuid.uuid4().hex}'
    intent = directory.claim(k, owner)
    assert intent['state'] == 'Intent'
    access.host_intents.append((k, owner))
    block = PoolBlock(**intent['block'])
    lease = access.acquire(block, 'w')
    data = bytes([byte]) * BLOCK_BYTES
    access.buffer[lease.byte_offset:lease.byte_offset + BLOCK_BYTES] = data
    committed = access.complete_write(lease, prefix_content_id(k), uuid.uuid4().hex)
    # The raw snapshot/SDK commit is real, yet discovery remains empty until publish.
    assert directory.discover([k]) == ()
    assert directory.publish(k, owner, committed) == committed
    access.host_intents.remove((k, owner))
    return {'key': [k.namespace_digest.hex(), k.block_hash.hex()], 'content': asdict(committed),
            'sha256': hashlib.sha256(data).hexdigest(), 'byte': byte,
            'original_peer_query': access.port.client.query_operation(committed.operation_id)}


def guards(access, bindings):
    rows = []
    for block in bindings:
        # This is an actual SDK R grant for only the assigned 2MiB guard.
        lease = access.port.acquire_range(block, 'r', block.generation, 0, GUARD_BYTES)
        access.leases[lease.lease_id] = lease
        access.port.revalidate(lease)
        data = bytes(access.buffer[block.byte_offset:block.byte_offset + GUARD_BYTES])
        access.port.revalidate(lease)
        ack = access.complete_read(lease, uuid.uuid4().hex)
        rows.append({'block': asdict(block), 'sha256': hashlib.sha256(data).hexdigest(),
                     'reader_lease': lease.lease_id, 'release_ack': asdict(ack)})
    return rows


def negatives(access, directory, old, root):
    """Real SDK negative scenarios; no forced crash or Unknown injection."""
    k, owner = key(4), f'half-{os.getpid()}'
    block = PoolBlock(**directory.claim(k, owner)['block'])
    access.host_intents.append((k, owner))
    lease = access.acquire(block, 'w')
    middle = BLOCK_BYTES // 2
    access.buffer[lease.byte_offset:lease.byte_offset + middle] = bytes([174]) * middle
    writing_snapshot = access.port.client.snapshot()
    assert directory.discover([k]) == ()
    uncommitted = CommittedRange(access.identity.pool_id, access.identity.pool_epoch,
        access.identity.provider_session, block, lease.byte_offset, BLOCK_BYTES, 1,
        prefix_content_id(k), 'not-a-committed-operation', lease.lease_id)
    try:
        directory.publish(k, owner, uncommitted)
    except ValueError as error:
        half_rejection = str(error)
    else:
        raise AssertionError('half-written data was published')
    try:
        directory.cancel(k, owner)
    except ValueError as error:
        inflight_rejection = str(error)
    else:
        raise AssertionError('metadata intent was cancelled while its actual W lease remained active')
    access.buffer[lease.byte_offset + middle:lease.byte_offset + BLOCK_BYTES] = bytes([174]) * (BLOCK_BYTES - middle)
    committed = access.complete_write(lease, prefix_content_id(k), uuid.uuid4().hex)
    directory.publish(k, owner, committed)
    access.host_intents.remove((k, owner))

    cancel_key, cancel_owner = key(5), f'cancel-{os.getpid()}'
    cancel_block = PoolBlock(**directory.claim(cancel_key, cancel_owner)['block'])
    access.host_intents.append((cancel_key, cancel_owner))
    cancelled = access.acquire(cancel_block, 'w')
    operation = uuid.uuid4().hex
    # No byte was written under this W grant. Only the original SDK can release
    # that exact lease with write_not_started, followed by its exact ACK.
    try:
        ack = access.port.release(cancelled, operation)
        ack.validate(access.identity, 'release', operation, cancelled)
    except Exception as error:
        access.report_unknown(cancelled, str(error))
        raise
    del access.leases[cancelled.lease_id]
    directory.cancel(cancel_key, cancel_owner)
    access.host_intents.remove((cancel_key, cancel_owner))
    assert directory.discover([cancel_key]) == ()
    fresh = write_prefix(access, directory, 6, 176)
    assert fresh['content']['block']['block_id'] == old.block.block_id
    stale_releases = []
    original_release = access.port.release
    def observe_release(lease, operation):
        result = original_release(lease, operation)
        stale_releases.append({'original_R_grant': asdict(lease), 'original_SDK_release_ACK': asdict(result)})
        return result
    access.port.release = observe_release
    try:
        access.acquire(old.block, 'r', committed=old, reject_stale=True)
    except StaleCommittedRange as error:
        stale_rejection = str(error)
    else:
        raise AssertionError('a cached old content version admitted native access')
    finally:
        access.port.release = original_release
    assert len(stale_releases) == 1
    released = stale_releases[0]
    assert released['original_R_grant']['content_version'] == fresh['content']['content_version']
    assert released['original_SDK_release_ACK']['applied']
    assert released['original_SDK_release_ACK']['lease_id'] == released['original_R_grant']['lease_id']
    assert not access.leases and not access.unknown
    assert directory.discover([PrefixKey(b'incompatible-namespace!!!!!!!!!!'[:32], key(6).block_hash)]) == ()
    intents = []
    for n in range(7, 11):
        candidate = key(n), f'pressure-{n}'
        assert directory.claim(*candidate)['state'] == 'Intent'
        intents.append(candidate)
        access.host_intents.append(candidate)
    assert directory.claim(key(11), 'one-too-many')['state'] == 'Busy'
    pressure_snapshot = directory.call('snapshot')
    for candidate in intents:
        directory.cancel(*candidate)
        access.host_intents.remove(candidate)
    save(root / 'real-negative-controls.json', {'scope': 'actual common SDK/directory host; no native/model proof',
        'half_write_snapshot': writing_snapshot, 'half_write_publication_rejected': half_rejection,
        'inflight_intent_cancel_rejected': inflight_rejection, 'not_started_W_actual_release_ACK': asdict(ack),
        'old_committed_version_rejected_after_real_current_R_release': stale_rejection,
        'stale_current_R_grant_and_exact_original_release_ACK': stale_releases,
        'cached_old_content': asdict(old), 'four_intent_pressure_snapshot': pressure_snapshot,
        'pressure_four_intents_then_Busy': True, 'incompatible_namespace_missed': True,
        'current_fresh_content': fresh})


def role(args):
    with retained_access(args) as access:
        run_role(args, access)


def run_role(args, access):
    directory = DirectoryClient(args.endpoint, args.epoch)
    result = {'role': args.role, 'pid': os.getpid(), 'client_session': access.identity.client_session,
              'native_or_model_started': False}
    if args.role == 'producer':
        bindings = load_bindings(args.bindings, access)
        result['guards_before'] = guards(access, bindings)
        result['prefixes'] = [write_prefix(access, directory, n, 170 + n) for n in (1, 2)]
        result['guards_after'] = guards(access, bindings)
        assert [r['sha256'] for r in result['guards_before']] == [r['sha256'] for r in result['guards_after']]
        access.close()
        result['sdk_unmapped'] = access.closed
        save(Path(args.root) / 'producer-result.json', result)
        return
    content = directory.discover([key(1)])[0]
    lease = access.acquire(content.block, 'r', committed=content, reject_stale=True)
    access.port.revalidate(lease)
    data = bytes(access.buffer[content.byte_offset:content.byte_offset + BLOCK_BYTES])
    assert data == bytes([171]) * BLOCK_BYTES
    result.update(lease=lease.lease_id, version=lease.content_version, sha256=hashlib.sha256(data).hexdigest())
    print(json.dumps(result), flush=True)
    try:
        if sys.stdin.readline().strip() != 'release':
            raise RuntimeError('reader controller did not request normal release')
    finally:
        access.complete_read(lease, uuid.uuid4().hex)
        access.close()
        save(Path(args.root) / (f'reader-{os.getpid()}-closed.json'), {'client_session': result['client_session'],
             'sdk_unmapped': access.closed, 'read_completed_before_release': True})


def controller(args):
    root = Path(args.root)
    root.mkdir()
    manifest = json.loads(Path(args.manifest).read_text())
    identity = {'hostname': os.uname().nodename,
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'pid_namespace': os.readlink('/proc/self/ns/pid'), 'mount_namespace': os.readlink('/proc/self/ns/mnt')}
    assert identity == manifest['host']
    save(root / 'host-identity.json', identity)
    sys.path.insert(0, manifest['helper_root'])
    from resources import host_job
    from session_protocol import request, pool_record, check_retired
    common = ['--root', str(root), '--manifest', args.manifest, '--bindings', args.bindings,
              '--endpoint', args.endpoint, '--epoch', args.epoch]
    with host_job(manifest):
        log = (root / 'directory.log').open('xb')
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('directory_service.py')),
            '--root', str(root / 'directory'), '--endpoint', args.endpoint, '--epoch', args.epoch,
            '--bindings', args.bindings, '--pool-endpoint', manifest['endpoint']], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while not (root / 'directory/ready.json').exists():
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('metadata directory startup failed; original log retained')
            time.sleep(.02)
        readers, reader_logs = [], []
        access = None
        try:
            with (root / 'producer.log').open('xb') as output:
                producer = subprocess.Popen([sys.executable, '-B', __file__, '--role', 'producer', *common],
                                            stdout=output, stderr=subprocess.STDOUT)
                assert producer.wait(timeout=60) == 0
            producer_result = json.loads((root / 'producer-result.json').read_text())
            assert producer_result['sdk_unmapped']  # actual producer is already gone before consumer registration
            from pypto_serving.serving.native_ddr.managed_pool_sdk import accepted_sdk
            first_op = producer_result['prefixes'][0]['content']['operation_id']
            cross_peer_query = accepted_sdk().query_pool_operation(manifest['endpoint'], first_op)
            assert producer_result['prefixes'][0]['original_peer_query']['state'] == 'Completed'
            assert cross_peer_query == {'state': 'NotFound', 'response': None}
            for index in range(2):
                output = (root / f'reader-{index}.stderr').open('xb')
                reader_logs.append(output)
                readers.append(subprocess.Popen([sys.executable, '-B', __file__, '--role', 'reader', *common],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=output, text=True))
            ready = [json.loads(reader.stdout.readline()) for reader in readers]
            assert len({r['client_session'] for r in ready}) == 2 and ready[0]['version'] == ready[1]['version']
            access = open_access(args)
            access.host_intents = []
            first = DirectoryClient(args.endpoint, args.epoch).discover([key(1)])[0]
            pool = pool_record(access.port.client.snapshot(), manifest['pool_descriptor'])
            leases = [l for l in pool['ranges']['leases'] if l['lease_id'] in [r['lease'] for r in ready]]
            assert len(leases) == 2 and all(l['mode'] == 'r' for l in leases)
            try:
                access.acquire(first.block, 'w')
            except ValueError as error:
                conflict = str(error)
            else:
                raise AssertionError('UF admitted an overlapping writer while two real readers held the range')
            assert not access.unknown and not access.leases
            third = write_prefix(access, DirectoryClient(args.endpoint, args.epoch), 3, 173)
            save(root / 'concurrent-real-authority.json', {'two_same_version_readers': ready,
                 'snapshot_with_two_real_readers': pool, 'overlapping_writer_rejected': conflict,
                 'different_range_write_committed_while_readers_live': third, 'producer_exited_before_readers': True,
                 'different_peer_cannot_query_original_operation': cross_peer_query,
                 'directory_independent_SDK_reader_confirmations': DirectoryClient(args.endpoint, args.epoch).call('snapshot')})
            for reader in readers:
                reader.communicate('release\n', timeout=30)
                assert reader.returncode == 0
            before = guards(access, load_bindings(args.bindings, access))
            from types import SimpleNamespace
            from pypto_serving.serving.native_ddr.diagnostics import capture_shared_guards
            observed = SimpleNamespace(access=access, bindings=load_bindings(args.bindings, access), busy=None,
                shared_directory=DirectoryClient(args.endpoint, args.epoch), child_registrations={})
            production_before = capture_shared_guards(observed, root, 'r06-host-before')
            negatives(access, DirectoryClient(args.endpoint, args.epoch), first, root)
            after = guards(access, load_bindings(args.bindings, access))
            production_after = capture_shared_guards(observed, root, 'r06-host-after')
            assert [x['sha256'] for x in before] == [x['sha256'] for x in after]
            assert [x['sha256'] for x in production_before['guards']] == [x['sha256'] for x in production_after['guards']]
            save(root / 'final-real-guard-readers.json', {'before': before, 'after': after,
                 'same_complete_four_assigned_2MiB_guards': True,
                 'production_shared_capture_before': production_before, 'production_shared_capture_after': production_after})
        finally:
            for reader in readers:
                if reader.poll() is None:
                    reader.communicate('release\n', timeout=30)
                assert reader.returncode == 0
            for stream in reader_logs:
                stream.close()
            if access is not None:
                if access.leases or access.unknown:
                    save(root / 'controller-retained.json', {'pid': os.getpid(), 'leases': list(access.leases),
                         'unknown': list(access.unknown), 'no_force_cleanup': True})
                    while True:
                        time.sleep(30)
                for k, owner in access.host_intents:
                    DirectoryClient(args.endpoint, args.epoch).cancel(k, owner)
                access.close()
            save(root / 'directory/stop.json', {'epoch': args.epoch})
            assert child.wait(timeout=20) == 0
            log.close()
        state = request(manifest, 'describe')
        pool = pool_record(state['snapshot'], manifest['pool_descriptor'])
        sessions = [json.loads(p.read_text())['identity']['client_session'] for p in root.glob('*-registered.json')]
        check_retired(pool, sessions)
        result = {'passed': True, 'scope': 'real common UF/SDK multi-process host only; no native/model proof',
            'client_sessions': sessions, 'ordinary_clients_unmapped': True, 'metadata_directory_normally_closed': True,
            'common_pool_still_allocated': True, 'B_charge': state['snapshot']['charged_bytes'],
            'A_historical_retained': 134217728, 'A_contacted': False, 'line_retired': False}
        save(root / 'result.json', result)
        print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=('controller', 'producer', 'reader'), default='controller')
    for name in ('root', 'manifest', 'bindings', 'endpoint', 'epoch'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    (controller if args.role == 'controller' else role)(args)


if __name__ == '__main__':
    main()
