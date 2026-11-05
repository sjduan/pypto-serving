# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real accepted SDK types/methods with simulated RPC, NOT real daemon evidence."""
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sdk = pytest.importorskip('uflow.managed_pool')

from pypto_serving.serving.native_ddr.managed_pool import BLOCK_BYTES, GUARD_BYTES, ManagedPoolAccess, PoolOutcomeUnknown
from pypto_serving.serving.native_ddr.managed_pool_sdk import SdkManagedPoolPort
from tests.unit.serving.test_managed_pool_protocol import FakePort, sdk_descriptor


@pytest.fixture
def fixture():
    fake = FakePort()
    descriptor = sdk_descriptor()
    client = sdk.ManagedPoolClient()
    client.endpoint, client.timeout = 'simulated-transport-only', 1
    client._descriptor = descriptor
    client._registered, client._stopping, client._unmapped = True, False, False
    client._children, client._child_operations, client._leases = {}, {}, {}
    client._ack_operation_id = None
    client.view = SimpleNamespace(descriptor=descriptor, buffer=fake.buffer, closed=False)
    calls, live = [], {'mode': 'w', 'version': 0, 'content': '', 'lost': False, 'reject': False}
    block = sdk.PoolBlock('pool', 'epoch', 'provider', 1, 1, 0, BLOCK_BYTES + GUARD_BYTES)

    def grant(mode):
        return sdk.PoolLease('pool', 'epoch', 'provider', 1, 'real-sdk-lease', 17, 'client', 1, 1,
            GUARD_BYTES, BLOCK_BYTES, mode, 0 if mode == 'w' else live['version'],
            '' if mode == 'w' else live['content'])

    def snapshot():
        pool = {name: descriptor[name] for name in sdk.POOL_FIELDS}
        raw = asdict(grant(live['mode']))
        for name in ('pool_id', 'pool_epoch', 'provider_session', 'access_generation'):
            raw.pop(name)
        raw['state'] = 'Writing' if live['mode'] == 'w' else 'Reading'
        pool.update(frozen=False, clients=[{'client_session': 'client', 'access_generation': 1,
            'state': 'Mapped', 'closing': False, 'access': 'rw'}], ranges={'allocations': [{
                'block_id': 1, 'generation': 1, 'offset': 0, 'nbytes': block.nbytes,
                'segments': [{'offset': 0, 'nbytes': GUARD_BYTES, 'state': 'Uncommitted',
                              'content_version': 0, 'content_id': ''},
                    {'offset': GUARD_BYTES, 'nbytes': BLOCK_BYTES,
                     'state': 'Uncommitted' if live['mode'] == 'w' else 'Committed',
                     'content_version': 0 if live['mode'] == 'w' else live['version'],
                     'content_id': '' if live['mode'] == 'w' else live['content']}]}], 'leases': [raw]})
        return {'schema': 1, 'pools': [pool]}

    def call(op, values=None, operation_id=None):
        calls.append((op, values, operation_id))
        if op == 'AllocatePoolRange':
            return asdict(block)
        if op == 'AcquirePoolRange':
            if live['reject']:
                raise sdk.ManagedPoolError('overlapping range rejected', operation_id=operation_id, client=client,
                    response={'schema': 1, 'status': 'error', 'operation_id': operation_id,
                              'code': 'Rejected', 'detail': 'overlapping range rejected'})
            live['mode'] = values['mode']
            return asdict(grant(values['mode']))
        if op == 'GetManagedPoolSnapshot':
            return snapshot()
        if op == 'CommitPoolWrite':
            live['version'], live['content'] = 7, values['content_id']
            if live['lost']:
                raise sdk.ManagedPoolUnknown('genuinely simulated lost reply', operation_id=operation_id, client=client)
            return {'committed': True, 'content_version': 7, 'content_id': values['content_id'],
                    'lease_id': values['lease']['lease_id']}
        if op == 'ReleasePoolLease':
            return {'released': True, 'lease_id': values['lease']['lease_id']}
        return {'ReportPoolUnknown': {'unknown_retained': True}, 'AttachPoolChild': {'attached': True},
                'DetachPoolChild': {'detached': True, 'native_drain_attested': True}}[op]

    client._call = call
    port = SdkManagedPoolPort(client)
    access = ManagedPoolAccess(port)
    context = SimpleNamespace(client=client, port=port, access=access, calls=calls, live=live, snapshot=snapshot)
    yield context
    access.buffer.release()
    fake.buffer.close()


def test_actual_sdk_write_zero_version_then_commit_assigned_version_and_read_release(fixture):
    c = fixture
    block = c.access.allocate()
    writer = c.access.acquire(block, 'w')
    assert writer.content_version == 0
    content = c.access.complete_write(writer, 'model-prefix', 'commit-operation')
    assert content.content_version == 7 and content.operation_id == 'commit-operation'
    reader = c.access.acquire(block, 'r', committed=content)
    assert reader.content_version == 7
    # Disjoint Uncommitted guard content in the SAME allocation must not cause
    # read coverage validation to widen to the whole block/pool.
    ack = c.access.complete_read(reader, 'release-operation')
    assert ack.lease_generation == 17 and ack.operation_id == 'release-operation'
    assert not c.access.leases and not c.port._leases and not c.port._operations
    assert c.client._leases[writer.lease_id] == 'Closed'


@pytest.mark.parametrize('query_state', ['Completed', 'Pending', 'wrong-operation'])
def test_actual_sdk_unknown_preserves_client_operation_and_requires_exact_query_envelope(fixture, query_state):
    c = fixture
    writer = c.access.acquire(c.access.allocate(), 'w')
    c.live['lost'] = True
    def exchange(endpoint, op, values, **kwargs):
        assert op == 'QueryPoolOperation' and values['target_operation_id'] == 'lost-commit'
        assert kwargs['client'] is c.client
        if query_state == 'Pending':
            response = {'schema': 1, 'status': 'error', 'operation_id': 'lost-commit',
                        'code': 'Unknown', 'detail': 'not terminal'}
            return {'state': 'Pending', 'response': response}, []
        response = {'schema': 1, 'status': 'ok',
            'operation_id': 'other' if query_state == 'wrong-operation' else 'lost-commit',
            'result': {'committed': True, 'content_version': 7, 'content_id': 'prefix', 'lease_id': writer.lease_id}}
        return {'state': 'Completed', 'response': response}, []
    with patch.object(sdk, '_exchange', side_effect=exchange):
        if query_state == 'Completed':
            result = c.access.complete_write(writer, 'prefix', 'lost-commit')
            assert result.operation_id == 'lost-commit' and not c.access.unknown
        else:
            with pytest.raises(PoolOutcomeUnknown):
                c.access.complete_write(writer, 'prefix', 'lost-commit')
            assert c.access.unknown and c.access.leases and c.port._leases
            pending = c.port._operations['lost-commit']
            assert pending['error'].client is c.client and pending['error'].operation_id == 'lost-commit'
    assert sum(op == 'CommitPoolWrite' for op, _, _ in c.calls) == 1


def test_exact_sdk_rejection_before_native_does_not_invent_an_unknown_grant(fixture):
    c = fixture
    block = c.access.allocate()
    c.live['reject'] = True
    with pytest.raises(ValueError, match='overlapping'):
        c.access.acquire(block, 'w')
    assert not c.access.leases and not c.access.unknown and not c.port._operations


def test_lost_acquire_reply_followed_by_exact_queried_rejection_has_no_granted_obligation(fixture):
    c = fixture
    block = c.access.allocate()
    observed = {}
    def lost(op, values=None, operation_id=None):
        assert op == 'AcquirePoolRange'
        observed['operation'] = operation_id
        raise sdk.ManagedPoolUnknown('lost rejection reply', operation_id=operation_id, client=c.client)
    def exchange(endpoint, op, values, **kwargs):
        target = values['target_operation_id']
        assert target == observed['operation']
        return {'state': 'Rejected', 'response': {'schema': 1, 'status': 'error', 'operation_id': target,
                    'code': 'Rejected', 'detail': 'overlapping range'}}, []
    c.client._call = lost
    with patch.object(sdk, '_exchange', side_effect=exchange), pytest.raises(ValueError, match='overlapping'):
        c.access.acquire(block, 'w')
    assert not c.access.leases and not c.access.unknown and not c.port._operations


def test_child_control_uses_full_saved_birth_and_explicit_native_drain(fixture):
    c = fixture
    birth = {'pid': 123, 'uid': 0, 'gid': 0, 'starttime_ticks': 77,
             'pid_namespace_dev': 4, 'pid_namespace_ino': 100}
    with patch.object(sdk, 'process_birth', return_value=birth):
        c.access.attach_child(123, 77)
    c.access.detach_stopped_child(123, 77, owner_drained=True, process_gone=True)
    attach, detach = [values for op, values, _ in c.calls if op in ('AttachPoolChild', 'DetachPoolChild')]
    assert attach['birth'] is birth and detach['birth'] is birth and detach['native_drained'] is True
    assert not c.client._children and not c.port._children and not c.access.children


@pytest.mark.parametrize('fault', ['client-generation', 'allocation-generation', 'writing-version'])
def test_boolean_snapshot_identity_or_version_cannot_discharge_the_held_writer(fixture, fault):
    c = fixture
    writer = c.access.acquire(c.access.allocate(), 'w')
    original = c.client._call
    def corrupt(op, values=None, operation_id=None):
        result = original(op, values, operation_id)
        if op == 'GetManagedPoolSnapshot':
            pool = result['pools'][0]
            if fault == 'client-generation':
                pool['clients'][0]['access_generation'] = True
            elif fault == 'allocation-generation':
                pool['ranges']['allocations'][0]['generation'] = True
            else:
                pool['ranges']['leases'][0]['content_version'] = False
        return result
    c.client._call = corrupt
    with pytest.raises(ValueError):
        c.access.complete_write(writer, 'prefix', 'must-not-commit')
    assert c.access.unknown and c.access.leases and c.port._leases
    assert not any(op == 'CommitPoolWrite' for op, _, _ in c.calls)
