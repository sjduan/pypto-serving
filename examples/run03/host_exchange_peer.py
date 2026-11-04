# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent real SDK peers; host fixtures, no model/native Ready evidence."""
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'pypto', 'simpler', '_task_interface'):
            raise RuntimeError('host-only peer forbids ' + fullname)


if __name__ == '__main__':
    sys.meta_path.insert(0, NoRuntime())

from examples.run03.host_record_probe import fixture_compatibility, fixture_destination, fixture_record
from pypto_serving.serving.native_ddr.shared_compatibility import BLOCK_BYTES
from pypto_serving.serving.native_ddr.shared_exchange import E10Adapter, ExchangeFailure, HostStagingOwner, stage_block
from pypto_serving.serving.native_ddr.shared_prefix_record import (
    ExternalBackingIdentity, SharedPrefixRecord, SourceBinding,
)
from examples.run03.host_exchange_lifecycle import close_stage_on_exit, require_revoked_acquire

CHUNK = bytes(range(256)) * 256


def emit(**value):
    print(json.dumps(value), flush=True)


def receive():
    line = sys.stdin.buffer.readline(65537)
    if not line or len(line) > 65536:
        raise RuntimeError("missing bounded parent command")
    return json.loads(line)


def expected_digest():
    h = hashlib.sha256()
    for _ in range(BLOCK_BYTES // len(CHUNK)):
        h.update(CHUNK)
    return h.hexdigest()


def exact_pattern(data):
    assert len(data) == BLOCK_BYTES
    with memoryview(data) as view:
        assert all(view[i:i+len(CHUNK)] == CHUNK for i in range(0, len(view), len(CHUNK)))


def own_fds():
    result = {}
    for fd in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        assert not target.startswith(('/dev/davinci', '/dev/devmm', '/dev/hisi_hdc', '/dev/ascend'))
        result[fd.name] = target
    return result


def sdk_identity():
    import uflow
    import uflow.external_ddr as sdk

    package, module = Path(uflow.__file__).resolve(), Path(sdk.__file__).resolve()
    assert module == Path('/workspace/uflow/runs/host-exchange/sdk/uflow/external_ddr.py')
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    assert digest == os.environ['UFLOW_H2_SDK_SHA256']
    return {'package': str(package), 'module': str(module), 'sha256': digest,
            'fixed_source_revision': '4fba229373f0b27f0d1c0f6a7aab405542c8fdda'}


def require_peer_namespace(endpoint):
    from examples.run03.host_exchange_namespace import private_propagation

    outer_fd, root_fd = int(os.environ['UFLOW_H2_OUTER_FD']), int(os.environ['UFLOW_H2_ROOT_FD'])
    outer_pid, inner_pid = int(os.environ['UFLOW_H2_OUTER_PID']), int(os.environ['UFLOW_H2_INNER_PID'])
    anchor, current = os.fstat(outer_fd), os.stat('/proc/self/ns/mnt')
    outer, inner = os.stat(f'/proc/{outer_pid}/ns/mnt'), os.stat(f'/proc/{inner_pid}/ns/mnt')
    assert os.getppid() == inner_pid and endpoint == '/workspace/uflow/runs/host-exchange/u.sock'
    assert (outer.st_dev, outer.st_ino) == (anchor.st_dev, anchor.st_ino)
    assert (current.st_dev, current.st_ino) == (inner.st_dev, inner.st_ino) != (anchor.st_dev, anchor.st_ino)
    pinned = os.fstat(root_fd)
    mounted = Path('/workspace/uflow').stat()
    assert (pinned.st_dev, pinned.st_ino) == (mounted.st_dev, mounted.st_ino)
    private_propagation()


def producer(endpoint):
    from uflow import ExternalDdrClient

    client = ExternalDdrClient(endpoint)
    fd = os.memfd_create('run03-immutable-source', os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    handle = None
    try:
        os.ftruncate(fd, 3 * BLOCK_BYTES)
        for base, chunk in ((0, b'L' * len(CHUNK)), (BLOCK_BYTES, CHUNK), (2 * BLOCK_BYTES, b'R' * len(CHUNK))):
            for offset in range(0, BLOCK_BYTES, len(chunk)):
                assert os.pwrite(fd, chunk, base + offset) == len(chunk)
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
        handle = client.register(fd, name='run03-host-prefix', owner_name='run03-producer', size_bytes=3*BLOCK_BYTES)
        source = SourceBinding(ExternalBackingIdentity(**asdict(handle)), 'producer-host-fixture-epoch',
                               'host-fixture-entry', 1, BLOCK_BYTES)
        record = replace(fixture_record(), payload_sha256=expected_digest(), source=source)
        emit(event='registered', pid=os.getpid(), record=record.to_dict(), authority_source=asdict(source),
             seals=fcntl.fcntl(fd, fcntl.F_GET_SEALS), fds=own_fds(), snapshot=client.snapshot(), sdk=sdk_identity(),
             evidence='real E10 registration; compatibility/entry/worker identities remain host fixtures')
        revoked = False
        while True:
            request = receive()
            if request['op'] == 'verify-and-revoke':
                hashes = []
                for base, chunk in ((0, b'L' * len(CHUNK)), (BLOCK_BYTES, CHUNK), (2*BLOCK_BYTES, b'R' * len(CHUNK))):
                    h = hashlib.sha256()
                    for offset in range(0, BLOCK_BYTES, len(chunk)):
                        raw = os.pread(fd, len(chunk), base + offset)
                        assert raw == chunk
                        h.update(raw)
                    hashes.append(h.hexdigest())
                client.revoke(handle)
                revoked = True
                emit(event='revoked', all_three_source_blocks_exact=True, source_sha256=hashes,
                     snapshot=client.snapshot(), fds=own_fds())
            elif request['op'] == 'exit':
                assert revoked
                emit(event='exiting', fds=own_fds())
                return
            else:
                raise ValueError('unsupported producer operation')
    finally:
        # The normal scenario explicitly revokes first. On failure, preserve a
        # best-effort revoke result rather than declaring an unknown domain clean.
        if handle is not None:
            client.revoke(handle)
        os.close(fd)


def consumer(endpoint, output):
    from uflow import ExternalDdrError

    request = receive()
    record = SharedPrefixRecord.from_dict(request['record'])
    # Separate parent-authenticated producer context, never copied out of record.
    source = SourceBinding.from_dict(request['authority_source'])
    destination = fixture_destination()
    current = [destination]
    adapter, staging = E10Adapter(endpoint), HostStagingOwner()
    report = {'pid': os.getpid(), 'passed': False, 'cases': {}, 'snapshots': {},
              'compatibility': 'H1 historical-assets/synthetic-deployment host fixture only',
              'native_ready': False, 'npu_run': False, 'initial_fds': own_fds(), 'sdk': sdk_identity()}
    stage, unexpected_views, failed_resources = None, [], []
    report.update(source_record=record.to_dict(), authority_source=asdict(source), destination=asdict(destination))

    def copy(rec=record, src=source, **extra):
        return stage_block(rec, compatibility=fixture_compatibility(), current_source=lambda: src,
                           destination=destination, current_destination=lambda: current[0],
                           expected_chain_hash=fixture_record().chain_hash, expected_token_start=128,
                           expected_payload_sha256=extra.pop('digest', expected_digest()),
                           staging=staging, adapter=adapter, **extra)

    def storage_hash():
        return hashlib.sha256(staging._storage).hexdigest()

    def refs(label):
        snap = adapter.client.snapshot()
        report['snapshots'][label] = snap
        return snap

    def prewrite(name, action):
        before = storage_hash()
        try:
            action()
        except (ValueError, ExchangeFailure) as error:
            if isinstance(error, ExchangeFailure):
                if not error.resources.closed:
                    failed_resources.append(error.resources)
                assert error.resources.closed and not error.resources.unresolved_acquire
            assert storage_hash() == before and staging.guards_ok() and not staging.busy
            assert refs(name)['active_views'] == 0
            report['cases'][name] = {'rejected': type(error).__name__, 'whole_target_unchanged': True,
                                      'guards_unchanged': True, 'real_views_released': True}
        else:
            raise AssertionError(name + ' unexpectedly accepted')

    try:
        assert record.source.worker_epoch != destination.worker_epoch
        prewrite('stale-source', lambda: copy(replace(record, source=replace(source, entry_generation=2))))
        current[0] = replace(destination, page_lease_id='new-page-lease')
        prewrite('stale-destination', copy)
        current[0] = destination
        # SDK acquire omits each of these fields. The REAL response must expose
        # the mismatch before any target write; no fake broker or SDK patch.
        for field in ('placement_id', 'binding_id', 'owner_lease_id'):
            fake = replace(source, backing=replace(source.backing, **{field: getattr(source.backing, field)+1}))
            prewrite('response-' + field, lambda fake=fake: copy(replace(record, source=fake), fake))
        wrong = hashlib.sha256(b'wrong independent expected bytes').hexdigest()
        prewrite('bad-digest', lambda: copy(replace(record, payload_sha256=wrong), digest=wrong))
        prewrite('host-cancel-before-write', lambda: copy(cancelled=lambda: True))
        try:
            replace(source, offset_bytes=3*BLOCK_BYTES)
        except ValueError:
            report['cases']['range-bounds'] = {'rejected_by_H1': True, 'target_unchanged': True}
        else:
            raise AssertionError('out-of-backing range accepted')
        stage = copy()
        report['stage_lease_generation'] = stage.lease.generation
        report['source_view_fds'] = own_fds()
        retained = stage.resources.views[0]
        retained.verify(source)
        try:
            retained._view._mapping[BLOCK_BYTES] = 0
        except TypeError:
            report['readonly_mapping'] = True
        else:
            raise AssertionError('SDK mapping writable')
        raw = retained.read()
        exact_pattern(raw)
        assert hashlib.sha256(raw).hexdigest() == expected_digest()
        del raw
        raw = stage.read()
        exact_pattern(raw)
        del raw
        assert not stage.native_ready and staging.guards_ok()
        try:
            copy()
        except RuntimeError as error:
            report['cases']['slot-busy'] = {'rejected': str(error), 'real_slot_held': True}
        else:
            raise AssertionError('busy stage reused')
        snapshot = refs('retained-after-root-release')
        assert snapshot['active_views'] == 1 and snapshot['committed_bytes'] == 3*BLOCK_BYTES
        emit(event='held', snapshot=snapshot, generation=stage.lease.generation, fds=own_fds())
        assert receive()['op'] == 'revoked'
        raw = retained.read()
        exact_pattern(raw)
        del raw
        require_revoked_acquire(adapter.acquire, source,
                                exchange_code=adapter.client._exchange.__func__.__code__,
                                error_type=ExternalDdrError, report=report, unexpected_views=unexpected_views)
        assert refs('after-revoke-old-view-readable')['active_views'] == 1
        emit(event='old-view-live-after-revoke')
        assert receive()['op'] == 'owner-exited'
        raw = retained.read()
        exact_pattern(raw)
        del raw
        report['old_view_live_after_owner_exit'] = True
        report['staging_sha256'] = hashlib.sha256(stage.read()).hexdigest()
        stage.close()
        assert not staging.busy
        try:
            stage.read()
        except RuntimeError:
            report['old_stage_closed'] = True
        else:
            raise AssertionError('old stage readable')
        lease = staging.reserve()
        assert lease.generation != stage.lease.generation
        report['next_generation'] = lease.generation
        staging._release(lease)
        final = refs('final')
        for field in ('committed_bytes', 'backing_fds', 'session_pidfds', 'owner_leases',
                      'active_views', 'pending_handoffs', 'in_flight'):
            assert final[field] == 0, (field, final[field])
        report.update(passed=True, final_fds=own_fds(), target_guards_exact=staging.guards_ok(),
                      whole_block_bytes=BLOCK_BYTES, real_nonzero_source_offset=BLOCK_BYTES,
                      verified_cpu_copied_bytes=staging.verified_copied_bytes,
                      buffer_bound='one20MiB slot+128B guards, one20MiB read/hash temporary; no tensor/runtime')
        assert report['staging_sha256'] == expected_digest()
    except BaseException as error:
        if isinstance(error, ExchangeFailure) and all(item is not error.resources for item in failed_resources):
            failed_resources.append(error.resources)
        report['primary_error'] = traceback.format_exc()
        raise
    finally:
        # SDK release is bounded by its configured socket timeout. Confirm it
        # explicitly on assertion/handshake failures too, not only the happy path.
        cleanup = close_stage_on_exit(stage, report)
        for resources in failed_resources:
            note = {}
            proxy = SimpleNamespace(resources=resources, lease=resources.lease, close=resources.close)
            close_stage_on_exit(proxy, note)
            report.setdefault('failed_prepare_cleanups', []).append(note)
            if not note['stage_exit_cleanup']['confirmed']:
                report['passed'] = False
        for view in unexpected_views[:]:
            try:
                view.release()
                unexpected_views.remove(view)
            except BaseException as error:
                report.setdefault('unresolved_unexpected_views', []).append({
                    'error': repr(error), 'view_id': getattr(getattr(view, '_view', None), 'view_id', None),
                    'lease_id': getattr(getattr(view, '_view', None), 'lease_id', None)})
                report['passed'] = False
        output.write_text(json.dumps(report, indent=2)+'\n')
        unresolved_prepare = any(not item.closed for item in failed_resources)
        if (not cleanup['confirmed'] or unexpected_views or unresolved_prepare) and sys.exc_info()[0] is None:
            raise RuntimeError('consumer cleanup unresolved; retained identities in report')
    emit(event='consumer-done', passed=True, snapshot=final)


if __name__ == '__main__':
    require_peer_namespace(sys.argv[2])
    if sys.argv[1] == 'producer':
        producer(sys.argv[2])
    elif sys.argv[1] == 'consumer':
        consumer(sys.argv[2], Path(sys.argv[3]))
    else:
        raise SystemExit('producer|consumer required')
