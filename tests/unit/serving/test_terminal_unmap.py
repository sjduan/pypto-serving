# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real CPU mmap + actual accepted SDK methods, simulated authority/native proofs; no UF/NPU run."""
import copy
import os
from types import SimpleNamespace

import pytest

sdk = pytest.importorskip('uflow.managed_pool')
from pypto_serving.serving.native_ddr.terminal_unmap import terminal_unmap, retire_pins_after_quiescence


@pytest.fixture
def context(tmp_path):
    fd = os.open(tmp_path / 'cpu.bin', os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, 65536)
    actual = os.fstat(fd)
    client = sdk.ManagedPoolClient()
    descriptor = {'pool_id': 'pool', 'pool_epoch': 'epoch', 'provider_session': 'provider', 'client_session': 'client',
        'access_generation': 1, 'access': 'rw', 'nbytes': 65536, 'canonical': {'dev': actual.st_dev, 'inode': actual.st_ino}}
    client._descriptor, client._registered, client._stopping, client._unmapped = descriptor, True, False, False
    client._children, client._leases = {}, {'writer': 'Unknown'}
    client.view = sdk.ManagedPoolView(client, descriptor, fd)
    root = memoryview(client.view.buffer)
    pool = {k: descriptor[k] for k in ('pool_id', 'pool_epoch', 'provider_session', 'nbytes', 'canonical')}
    pool.update(clients=[{'client_session': 'client', 'access_generation': 1, 'state': 'Mapped', 'children': []}],
        ranges={'leases': [{'lease_id': 'writer', 'client_session': 'client', 'block_id': 5, 'offset': 128,
                            'nbytes': 4096, 'mode': 'w', 'state': 'Unknown'}], 'allocations': [{'block_id': 5,
            'segments': [{'offset': 128, 'nbytes': 4096, 'state': 'Committed', 'content_version': 3, 'content_id': 'old'}]}]})
    calls, expected = [], {'lease_id': 'writer', 'block_id': 5, 'offset': 128, 'nbytes': 4096}

    def rpc(op, values=None, operation_id=None):
        calls.append((op, operation_id))
        if op == 'GetManagedPoolSnapshot':
            return {'pools': [copy.deepcopy(pool)]}
        if op == 'BeginPoolUnregister':
            assert client._leases == {'writer': 'Unknown'}
            return {'stopping': True}
        if op == 'ReportPoolUnknown':
            return {'unknown_retained': True}
        assert op == 'AckPoolUnmapped' and values == {'unmapped': True, 'fd_closed': True, 'native_drained': True}
        assert client.view.closed
        pool['clients'][0]['state'] = 'Unmapped'
        pool['ranges']['leases'] = []
        pool['ranges']['allocations'][0]['segments'][0].update(state='Invalid', content_version=0, content_id='')
        return {'unmapped': True, 'unknown_content_invalid_not_committed': True}

    client._call = rpc
    owner = SimpleNamespace(_close_complete=True, _active_dispatch_handles=set(), _service_tokens=set(), _inherited_host_tensors=())
    c = SimpleNamespace(client=client, root=root, owner=owner, pool=pool, expected=expected, calls=calls,
                        state={}, extra=None, helpers_stopped=False)

    def proof():
        assert client.view.closed
        with pytest.raises(OSError):
            os.fstat(fd)
        return {'actual_mmap_and_FD_closed': True}

    def helpers():
        c.helpers_stopped = True
        return {'CPU_only_no_native_helper': True}

    c.run = lambda: terminal_unmap(client, root, owner, expected, release_exports=lambda: None,
        prove_unmapped=proof, stop_helpers=helpers, unregister_operation='fixed-unregister', unmap_operation='fixed-unmap', state=c.state)
    yield c
    if c.extra is not None:
        c.extra.release()
    root.release()
    if not client.view.closed:
        client.view.close()


def test_real_sdk_terminal_ack_closes_unknown_to_invalid_without_commit(context):
    c = context
    c.run()
    assert c.client.view.closed and c.client._leases == {'writer': 'Closed'} and c.helpers_stopped
    assert c.pool['ranges']['allocations'][0]['segments'][0] == {
        'offset': 128, 'nbytes': 4096, 'state': 'Invalid', 'content_version': 0, 'content_id': ''}
    assert [name for name, _ in c.calls].count('AckPoolUnmapped') == 1
    assert not {'CommitPoolWrite', 'ReleasePoolLease'}.intersection(name for name, _ in c.calls)


@pytest.mark.parametrize('debt', ['owner-open', 'active-native', 'pins', 'inherited-export', 'Active-lease', 'foreign-client'])
def test_undrained_or_foreign_debt_does_not_close_or_ack(context, debt):
    c = context
    if debt == 'owner-open': c.owner._close_complete = False
    elif debt == 'active-native': c.owner._active_dispatch_handles.add('pending')
    elif debt == 'pins': c.owner._service_tokens.add('still-retained')
    elif debt == 'inherited-export': c.owner._inherited_host_tensors = ('still-live',)
    elif debt == 'Active-lease': c.pool['ranges']['leases'][0]['state'] = 'Writing'
    elif debt == 'foreign-client': c.pool['ranges']['leases'][0]['client_session'] = 'other'
    with pytest.raises(RuntimeError): c.run()
    assert not c.client.view.closed and c.client._leases == {'writer': 'Unknown'}
    assert not {'BeginPoolUnregister', 'AckPoolUnmapped'}.intersection(name for name, _ in c.calls)


def test_actual_exported_mmap_blocks_ack_and_preserves_unknown(context):
    c = context
    c.extra = memoryview(c.client.view.buffer)[128:4096]
    with pytest.raises(BufferError): c.run()
    assert not c.client.view.closed and c.client._leases == {'writer': 'Unknown'} and not c.helpers_stopped
    assert not any(name == 'AckPoolUnmapped' for name, _ in c.calls)
    c.extra.release()
    c.extra = None
    c.run()  # Fixed original operation IDs, SDK performs its own real terminal state transition.
    assert c.state['ack_done'] and c.client._leases == {'writer': 'Closed'}


def test_no_quiescence_or_failed_pin_release_never_closes_owner():
    released = []
    worker = SimpleNamespace(_service_tokens={'pin'}, _w=SimpleNamespace(_service_buffers={'pin': object()}))
    backend = SimpleNamespace(_pins=['pin'], _release_pins=lambda: released.append('called'))
    with pytest.raises(RuntimeError): retire_pins_after_quiescence(worker, backend, {'current_access_ended': False})
    assert not released and worker._service_tokens == {'pin'}
    def fail(): raise RuntimeError('actual release failed')
    backend._release_pins = fail
    with pytest.raises(RuntimeError, match='release failed'):
        retire_pins_after_quiescence(worker, backend, {'current_access_ended': True})
    assert worker._service_tokens == {'pin'} and backend._pins == ['pin']
