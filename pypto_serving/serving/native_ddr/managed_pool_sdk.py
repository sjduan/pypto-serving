# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Normalize accepted P1 6c885833 control records; UFlow remains sole authority."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import uuid

from .managed_pool import (
    CommittedRange, ManagedPoolAccess, PoolAck, PoolBlock, PoolIdentity, PoolLease, PoolOutcomeUnknown,
    interval, positive,
)

SDK_SHA256 = '366f3ea883cec08efd62f23f9ebf2afa8546c02a777ff25fdf753fee1debff0e'
SDK_FILES = {'managed_pool.py': SDK_SHA256,
             '_managed_pool_wire.py': 'c61fdf8f9c19261b9386ae5c64e1ac8e2b81cf41238ee9c8f3c24c7cab59015e',
             '__init__.py': 'bb529fe27ed36eb0a4e5ced2e739411e270bcf13383001b601ca517064f6f328'}
_STARTUP_OBLIGATIONS = {}


class _QueriedRejection(ValueError):
    pass


def accepted_sdk():
    module = importlib.import_module('uflow.managed_pool')
    directory = Path(module.__file__).resolve(strict=True).parent
    for name, expected in SDK_FILES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'managed control SDK differs from accepted P1: {name}')
    return module


def exact(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('unexpected accepted SDK response fields')
    return value


def identity(value):
    # The accepted SDK validates all canonical fields before FD mapping. Repeat
    # its pinned schema check here, before converting numeric resource IDs.
    accepted_sdk()._pool_descriptor(value, registered=True)
    canonical = value['canonical']
    return PoolIdentity(value['pool_id'], value['pool_epoch'], value['provider_session'], value['client_session'],
        value['nbytes'], canonical['dev'], canonical['inode'], str(value['resource_id']), str(value['placement_id']),
        str(value['binding_id']), value['access_generation'], canonical['owner_instance_id'], canonical['buffer_id'],
        canonical['generation'], value['resource_generation'])


class SdkManagedPoolPort:
    """Retain original SDK objects, pending requests and full child identities."""
    def __init__(self, client):
        self.sdk = accepted_sdk()
        self.client = client
        self.identity = identity(client.view.descriptor)
        self.buffer = client.view.buffer
        self._blocks, self._leases, self._operations, self._children = {}, {}, {}, {}
        self._unregister_operation = uuid.uuid4().hex

    def _block(self, raw):
        pool = self.identity
        if type(raw) is not self.sdk.PoolBlock or (raw.pool_id, raw.pool_epoch, raw.provider_session) != (
                pool.pool_id, pool.pool_epoch, pool.provider_session):
            raise ValueError('SDK allocation changed pool/provider incarnation')
        value = PoolBlock(raw.pool_id, raw.pool_epoch, raw.block_id, raw.generation, raw.offset, raw.nbytes)
        value.validate(pool)
        return value

    def _raw_block(self, value):
        value.validate(self.identity)
        return self._blocks.get((value.block_id, value.generation)) or self.sdk.PoolBlock(
            value.pool_id, value.pool_epoch, self.identity.provider_session, value.block_id,
            value.generation, value.byte_offset, value.nbytes)

    def _lease(self, raw, allocation):
        pool = self.identity
        if type(raw) is not self.sdk.PoolLease or (
                raw.pool_id, raw.pool_epoch, raw.provider_session, raw.client_session, raw.access_generation) != (
                pool.pool_id, pool.pool_epoch, pool.provider_session, pool.client_session, pool.access_generation):
            raise ValueError('SDK range lease changed registered access identity')
        value = PoolLease(pool, allocation, raw.lease_id, raw.lease_generation, raw.offset, raw.nbytes,
            raw.mode, 'Writing' if raw.mode == 'w' else 'Committed', raw.content_version,
            raw.client_session, raw.lease_id, raw.content_id)
        value.validate(pool, allocation, raw.mode, raw.offset, raw.nbytes)
        return value

    def _retained(self, value):
        retained = self._leases.get(value.lease_id)
        if retained is None or retained[0] != value:
            raise ValueError('SDK operation has no exact retained lease')
        return retained[1]

    def _queried_value(self, operation_id):
        result = self.client.query_operation(operation_id)
        exact(result, ('state', 'response'))
        if result['state'] == 'Rejected':
            response = exact(result['response'], ('schema', 'status', 'operation_id', 'code', 'detail'))
            if (type(response['schema']) is not int or response['schema'] != 1
                    or response['status'] != 'error' or response['code'] != 'Rejected'
                    or response['operation_id'] != operation_id):
                raise ValueError('queried rejection changed the original operation')
            raise _QueriedRejection(response['detail'])
        if result['state'] != 'Completed':
            return None
        response = exact(result['response'], ('schema', 'status', 'operation_id', 'result'))
        if (type(response['schema']) is not int or response['schema'] != 1 or response['status'] != 'ok'
                or response['operation_id'] != operation_id):
            raise ValueError('query returned a different or unconfirmed original operation')
        return response['result']

    def _invoke(self, kind, operation_id, arguments, call):
        pending = self._operations.get(operation_id)
        if pending is None:
            pending = self._operations[operation_id] = {'kind': kind, 'arguments': arguments}
        elif (pending['kind'], pending['arguments']) != (kind, arguments):
            raise ValueError('operation ID reused with a different local request')
        try:
            try:
                raw = call()
            except self.sdk.ManagedPoolUnknown as error:
                pending['error'] = error  # includes the actual SDK client and original operation ID
                if error.client is not self.client or error.operation_id != operation_id:
                    raise ValueError('unknown SDK outcome lost its original client/operation identity') from error
                raw = self._queried_value(operation_id)
                if raw is None:
                    raise PoolOutcomeUnknown('same-operation query has no confirmed terminal result') from error
            except self.sdk.ManagedPoolError as error:
                if (error.client is not self.client or error.operation_id != operation_id or not error.response
                        or error.response.get('code') != 'Rejected' or error.response.get('status') != 'error'):
                    raise PoolOutcomeUnknown('SDK rejection has no exact request identity') from error
                del self._operations[operation_id]
                raise ValueError(str(error)) from error  # confirmed rejection; no grant/native action
            pending['raw'] = raw
            result = self._decode(kind, operation_id, arguments, raw)
        except _QueriedRejection as error:
            del self._operations[operation_id]
            raise ValueError(str(error)) from error
        except ValueError:
            if operation_id not in self._operations:
                raise  # only the confirmed SDK rejection above discharges this request
            raise PoolOutcomeUnknown('SDK result cannot be bound to the retained request') from None
        except PoolOutcomeUnknown:
            raise
        except Exception as error:
            pending['error'] = error
            raise PoolOutcomeUnknown('SDK operation outcome remains retained') from error
        del self._operations[operation_id]
        return result

    def _decode(self, kind, operation_id, arguments, raw):
        if kind == 'allocate':
            original = self.sdk.PoolBlock.from_dict(raw) if type(raw) is dict else raw
            value = self._block(original)
            if value.nbytes != arguments[0] or value.byte_offset % arguments[1]:
                raise ValueError('allocation changed requested size/alignment')
            self._blocks[value.block_id, value.generation] = original
            return value
        if kind == 'acquire':
            original = self.sdk.PoolLease.from_dict(raw) if type(raw) is dict else raw
            allocation, mode, generation, offset, nbytes = arguments
            value = self._lease(original, allocation)
            value.validate(self.identity, allocation, mode, allocation.byte_offset + offset, nbytes)
            if value.lease_id in self._leases or value.block.generation != generation:
                raise ValueError('SDK reused a live lease or allocation generation')
            self._leases[value.lease_id] = value, original
            return value
        if kind == 'commit':
            value, content_id, nbytes = arguments
            exact(raw, ('committed', 'content_version', 'content_id', 'lease_id'))
            positive(raw['content_version'], 'committed content version')
            if raw['committed'] is not True or (raw['lease_id'], raw['content_id']) != (value.lease_id, content_id):
                raise ValueError('commit ACK changed completed writer/content')
            result = CommittedRange(self.identity.pool_id, self.identity.pool_epoch, self.identity.provider_session,
                value.block, value.byte_offset, nbytes, raw['content_version'], content_id, operation_id, value.lease_id)
            result.validate(self.identity, content_id=content_id)
            del self._leases[value.lease_id]
            return result
        if kind == 'release':
            value = arguments[0]
            exact(raw, ('released', 'lease_id'))
            if raw['released'] is not True or raw['lease_id'] != value.lease_id:
                raise ValueError('release ACK changed held lease')
            del self._leases[value.lease_id]
            return PoolAck(operation_id, 'release', self.identity, value.lease_id, value.lease_generation, True)
        expected = {'attach': {'attached': True}, 'detach': {'detached': True, 'native_drain_attested': True},
                    'unregister': {'stopping': True}}
        if kind in expected:
            exact(raw, expected[kind])
            if any(raw[key] is not flag for key, flag in expected[kind].items()):
                raise ValueError('child/unregister ACK unconfirmed')
            pid, birth = arguments if kind != 'unregister' else (0, 0)
            return PoolAck(operation_id, kind + '_child' if kind != 'unregister' else kind,
                           self.identity, applied=True, child_pid=pid, child_birth=birth)
        if kind == 'unmapped':
            exact(raw, ('unmapped', 'unknown_content_invalid_not_committed'))
            if raw['unmapped'] is not True or type(raw['unknown_content_invalid_not_committed']) is not bool:
                raise ValueError('unmap ACK unconfirmed')
            return PoolAck(operation_id, 'unmapped', self.identity, applied=True)
        raise ValueError('unsupported local control operation')

    def _admit(self):
        if self._operations:
            raise PoolOutcomeUnknown('a retained SDK operation prevents new admission')

    def allocate_range(self, nbytes, alignment):
        self._admit()
        operation = uuid.uuid4().hex
        return self._invoke('allocate', operation, (nbytes, alignment),
            lambda: self.client.allocate_range(nbytes, alignment, operation_id=operation))

    def acquire_range(self, allocation, mode, generation, byte_offset, nbytes):
        self._admit()
        raw_block = self._raw_block(allocation)
        operation = uuid.uuid4().hex
        return self._invoke('acquire', operation, (allocation, mode, generation, byte_offset, nbytes),
            lambda: self.client.acquire_range(raw_block, mode, generation, byte_offset=byte_offset,
                                              nbytes=nbytes, operation_id=operation))

    def revalidate(self, value):
        raw = self._retained(value)
        snapshot = self.client.snapshot()
        pools = [row for row in snapshot['pools'] if row['pool_id'] == self.identity.pool_id]
        if len(pools) != 1 or pools[0]['state'] != 'Allocated' or pools[0]['frozen'] is not False:
            raise ValueError('managed pool no longer allocated/admitting')
        pool = pools[0]
        descriptor = {key: pool[key] for key in self.sdk.POOL_FIELDS}
        descriptor.update(client_session=self.identity.client_session, access_generation=self.identity.access_generation,
                          access='rw')
        if identity(descriptor) != self.identity:
            raise ValueError('pool snapshot changed canonical resource identity')
        clients = [row for row in pool['clients'] if row['client_session'] == self.identity.client_session]
        if (len(clients) != 1 or clients[0]['access_generation'] != self.identity.access_generation
                or type(clients[0]['access_generation']) is not int
                or clients[0]['state'] != 'Mapped' or clients[0]['closing'] is not False or clients[0]['access'] != 'rw'):
            raise ValueError('client incarnation no longer mapped/admitting')
        allocations = [row for row in pool['ranges']['allocations'] if row['block_id'] == value.block.block_id]
        leases = [row for row in pool['ranges']['leases'] if row['lease_id'] == value.lease_id]
        if len(allocations) != 1 or len(leases) != 1:
            raise ValueError('current snapshot has no unique retained allocation/lease')
        allocation, current = allocations[0], dict(leases[0])
        exact(allocation, ('block_id', 'generation', 'offset', 'nbytes', 'segments'))
        actual_block = self.sdk.PoolBlock(self.identity.pool_id, self.identity.pool_epoch, self.identity.provider_session,
            allocation['block_id'], allocation['generation'], allocation['offset'], allocation['nbytes'])
        if actual_block != self._raw_block(value.block):
            raise ValueError('allocation incarnation changed while native was pending')
        if current.pop('state') != ('Writing' if value.mode == 'w' else 'Reading'):
            raise ValueError('range lease is no longer active')
        current.update(pool_id=self.identity.pool_id, pool_epoch=self.identity.pool_epoch,
                       provider_session=self.identity.provider_session, access_generation=self.identity.access_generation)
        if self.sdk.PoolLease.from_dict(current) != raw:
            raise ValueError('snapshot no longer holds the exact registered range lease')
        # P1 acquisition records a live writer without changing old content
        # segments. That exact Writing lease authorizes overwrite; only readers
        # depend on the previous Committed coverage/version/content identity.
        if value.mode == 'w':
            return
        end, cursor = value.byte_offset + value.nbytes, value.byte_offset
        for segment in allocation['segments']:
            exact(segment, ('offset', 'nbytes', 'state', 'content_version', 'content_id'))
            interval(segment['offset'], segment['nbytes'], self.identity.nbytes)
            if type(segment['content_version']) is not int or segment['content_version'] < 0:
                raise ValueError('range snapshot has an invalid content version')
            start, stop = segment['offset'], segment['offset'] + segment['nbytes']
            if stop <= cursor or start >= end:
                continue
            if (start > cursor or segment['state'] != 'Committed'
                    or (segment['content_version'], segment['content_id']) != (value.content_version, value.content_id)):
                raise ValueError('current range content no longer covers the retained version')
            cursor = min(end, stop)
        if cursor != end:
            raise ValueError('range snapshot has a content gap')

    def commit_write(self, value, content_id, nbytes, operation_id):
        original = self._retained(value)
        return self._invoke('commit', operation_id, (value, content_id, nbytes),
            lambda: self.client.commit_write(original, content_id, nbytes, operation_id=operation_id))

    def release(self, value, operation_id):
        original = self._retained(value)
        return self._invoke('release', operation_id, (value,),
            lambda: self.client.release(original, operation_id=operation_id))

    def query_operation(self, operation_id):
        pending = self._operations[operation_id]
        raw = self._queried_value(operation_id)
        if raw is None:
            return None
        result = self._decode(pending['kind'], operation_id, pending['arguments'], raw)
        del self._operations[operation_id]
        return result

    def report_unknown(self, value, reason):
        raw = self._leases.get(getattr(value, 'lease_id', ''), (None, None))[1]
        bounded = str(reason).encode('utf-8')[:220].decode('utf-8', errors='ignore') or 'unknown serving obligation'
        self.client.report_unknown(raw, reason=bounded, operation_id=uuid.uuid4().hex)

    def attach_child(self, pid, birth):
        raw = self.sdk.process_birth(pid)
        if raw['pid'] != pid or raw['starttime_ticks'] != birth:
            raise ValueError('native child birth identity changed')
        self._children[pid, birth] = raw
        operation = uuid.uuid4().hex
        return self._invoke('attach', operation, (pid, birth),
            lambda: self.client.attach_child(pid, raw, operation_id=operation))

    def detach_child(self, pid, birth):
        raw = self._children[pid, birth]
        operation = uuid.uuid4().hex
        ack = self._invoke('detach', operation, (pid, birth),
            lambda: self.client.detach_child(pid, raw, native_drained=True, operation_id=operation))
        del self._children[pid, birth]
        return ack

    def begin_unregister(self):
        return self._invoke('unregister', self._unregister_operation, (),
            lambda: self.client.begin_unregister(operation_id=self._unregister_operation))

    def close_view(self):
        if self._leases or self._children or any(item['kind'] != 'unmapped' for item in self._operations.values()):
            raise RuntimeError('retained SDK obligations prevent access-view close')
        self.client.view.close()

    def ack_unmapped(self, operation_id):
        # ManagedPoolAccess calls this only with no range/child/unknown refs,
        # after caller-owned native drain and a successful real view/FD close.
        return self._invoke('unmapped', operation_id, (),
            lambda: self.client.ack_unmapped(native_drained=True, operation_id=operation_id))


def register_managed_pool(config):
    """Only register access. Retain uncertain startup clients; never create/destroy."""
    if config.sdk_sha256 != SDK_SHA256:
        raise ValueError('configured SDK is not the manager-accepted P1 module')
    if getattr(config, 'shared_slots_path', ''):
        from .shared_pool import partition_metadata
        document, shared_blocks, manifest = partition_metadata(config.shared_slots_path)
        if ((document['pool']['pool_id'], document['pool']['pool_epoch']) != (config.pool_id, config.pool_epoch)
                or config.pool_endpoint != manifest['endpoint']):
            raise ValueError('shared partition/configuration identity differs before SDK registration')
    module, operation = accepted_sdk(), uuid.uuid4().hex
    if getattr(config, 'shared_slots_path', ''):
        from .shared_directory import DirectoryAuthority
        if module.describe_pool(config.pool_endpoint, config.pool_id, config.pool_epoch) != document['pool']:
            raise ValueError('actual pool differs from sealed partition before SDK registration')
        snapshot = module._exchange(config.pool_endpoint, 'GetManagedPoolSnapshot',
            {k: document['pool'][k] for k in ('pool_id', 'pool_epoch', 'provider_session')})[0]
        DirectoryAuthority(document['pool'], shared_blocks, lambda: snapshot, lambda *args: None)
    try:
        client = module.ManagedPoolClient.register(config.pool_endpoint, config.pool_id, config.pool_epoch,
                                                   access='rw', operation_id=operation)
    except module.ManagedPoolError as error:
        if error.client is not None:
            _STARTUP_OBLIGATIONS[operation] = error.client, error
        raise
    try:
        port = SdkManagedPoolPort(client)
        if (port.identity.pool_id, port.identity.pool_epoch) != (config.pool_id, config.pool_epoch):
            raise ValueError('registered pool differs from configuration')
        if getattr(config, 'shared_slots_path', ''):
            from .shared_pool import register_partition_client
            register_partition_client(config.shared_slots_path, port.identity.client_session)
        return ManagedPoolAccess(port, max_registered_children=3 if getattr(config, 'shared_slots_path', '') else None)
    except BaseException as error:
        try:
            client.begin_unregister()
            client.view.close()
            client.ack_unmapped(native_drained=True)
        except BaseException as close_error:
            _STARTUP_OBLIGATIONS[operation] = client, error, close_error
        raise
