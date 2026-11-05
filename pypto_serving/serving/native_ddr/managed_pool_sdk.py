# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One SDK normalization seam; no SDK/provider protocol leaks into scheduling.

The public v1 method surface is frozen by the D1 contract. Concrete SDK record
bindings are checked here against the manager-accepted P1 pin before real use.
Unrecognized records fail closed rather than inventing allocation/stop proofs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
import hashlib
import importlib
from pathlib import Path

from .managed_pool import (
    CommittedRange, ManagedPoolAccess, PoolAck, PoolBlock, PoolIdentity, PoolLease, PoolOutcomeUnknown,
)


def record(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError('accepted managed SDK must expose a typed record or mapping')


def construct(cls, value, **nested):
    data = record(value)
    allowed = {item.name for item in fields(cls)}
    if data.keys() - allowed:
        raise ValueError(f'unsupported {cls.__name__} record fields: {sorted(data.keys() - allowed)}')
    for name, convert in nested.items():
        if name in data:
            data[name] = convert(data[name])
    return cls(**data)


def identity(value):
    return construct(PoolIdentity, value)


def block(value):
    return construct(PoolBlock, value)


def lease(value):
    return construct(PoolLease, value, pool=identity, block=block)


def committed(value):
    return construct(CommittedRange, value, block=block)


def acknowledgement(value):
    return construct(PoolAck, value, pool=identity)


class SdkManagedPoolPort:
    """Retain SDK-issued objects; Rust/UFlow alone decides grants and versions."""
    def __init__(self, client):
        self.client = client
        self.identity = identity(client.view.descriptor)
        self.buffer = client.view.buffer
        self._blocks = {}
        self._leases = {}
        self._operations = {}
        self._unparsed_grants = []
        self._unresolved_acquires = []

    def allocate_range(self, nbytes, alignment):
        raw = self.client.allocate_range(nbytes, alignment)
        value = block(raw)
        value.validate(self.identity)
        self._blocks[(value.block_id, value.generation)] = raw
        return value

    def acquire_range(self, allocation, mode, expected_generation, byte_offset, nbytes):
        if self._unparsed_grants or self._unresolved_acquires:
            raise PoolOutcomeUnknown('an undecodable grant still retains the pool access')
        allocation.validate(self.identity)
        raw_block = self._blocks.get((allocation.block_id, allocation.generation), asdict(allocation))
        pending = (raw_block, mode, expected_generation, byte_offset, nbytes)
        self._unresolved_acquires.append(pending)
        try:
            raw = self.client.acquire_range(raw_block, mode, expected_generation,
                                            byte_offset=byte_offset, nbytes=nbytes)
        except Exception as error:
            # The accepted SDK's explicit rejection types will be bound at P1
            # integration. Until then an exception cannot prove no remote grant.
            raise PoolOutcomeUnknown('SDK range acquisition has no terminal grant/rejection record') from error
        try:
            value = lease(raw)
            if value.lease_id in self._leases:
                raise ValueError('SDK reused a live range lease identity')
        except Exception as error:
            self._unparsed_grants.append(raw)
            raise PoolOutcomeUnknown('SDK range grant cannot be normalized safely') from error
        self._leases[value.lease_id] = (value, raw)
        self._unresolved_acquires.remove(pending)
        return value

    def _retained(self, value):
        retained = self._leases.get(value.lease_id)
        if retained is None or retained[0] != value:
            raise ValueError('SDK operation has no exact retained lease record')
        return retained[1]

    def revalidate(self, value):
        self._retained(value)
        snapshot = record(self.client.snapshot())
        current = [lease(item) for item in snapshot['leases']
                   if record(item).get('lease_id') == value.lease_id]
        if len(current) != 1 or current[0] != value:
            raise ValueError('current UFlow snapshot no longer holds the exact range lease')

    def commit_write(self, value, content_id, nbytes, operation_id):
        self._operations[operation_id] = ('commit', value, content_id)
        raw = self.client.commit_write(self._retained(value), content_id, nbytes,
                                       operation_id=operation_id)
        result = committed(raw)
        self._validate_commit(value, result, content_id, operation_id)
        del self._leases[value.lease_id]
        del self._operations[operation_id]
        return result

    def _validate_commit(self, value, result, content_id, operation_id):
        result.validate(self.identity, content_id=content_id)
        if (result.operation_id, result.writer_lease_id, result.block, result.byte_offset,
                result.content_version) != (operation_id, value.lease_id, value.block,
                                            value.byte_offset, value.content_version):
            raise ValueError('SDK commit acknowledgement identity mismatch')

    def release(self, value, operation_id):
        self._operations[operation_id] = ('release', value, None)
        raw = self.client.release(self._retained(value), operation_id=operation_id)
        result = acknowledgement(raw)
        result.validate(self.identity, 'release', operation_id, value)
        del self._leases[value.lease_id]
        del self._operations[operation_id]
        return result

    def query_operation(self, operation_id):
        raw = self.client.query_operation(operation_id)
        if raw is None:
            return None
        data = record(raw)
        operation = self._operations.get(operation_id)
        if operation is None:
            raise ValueError('queried operation has no retained local request identity')
        kind, value, expected_content = operation
        if kind == 'commit' and 'content_id' in data:
            result = committed(raw)
            saved = self._leases.get(result.writer_lease_id)
            if saved is None:
                raise ValueError('queried commit has no retained original writer')
            self._validate_commit(saved[0], result, expected_content, operation_id)
            del self._leases[result.writer_lease_id]
            del self._operations[operation_id]
            return result
        result = acknowledgement(raw)
        result.validate(self.identity, 'unmapped' if kind == 'unmapped' else 'release', operation_id, value)
        if value is not None:
            del self._leases[result.lease_id]
        del self._operations[operation_id]
        return result

    def report_unknown(self, value, reason):
        retained = self._leases.get(getattr(value, 'lease_id', ''))
        self.client.report_unknown(retained[1] if retained else None, reason)

    def attach_child(self, pid, birth):
        return acknowledgement(self.client.attach_child(pid, birth))

    def detach_child(self, pid, birth):
        return acknowledgement(self.client.detach_child(pid, birth))

    def begin_unregister(self):
        self.client.begin_unregister()

    def close_view(self):
        if self._leases or self._unparsed_grants or self._unresolved_acquires:
            raise RuntimeError('retained SDK leases prevent access-view close')
        self.client.view.close()

    def ack_unmapped(self, operation_id):
        self._operations[operation_id] = ('unmapped', None, None)
        ack = acknowledgement(self.client.ack_unmapped(operation_id=operation_id))
        ack.validate(self.identity, 'unmapped', operation_id)
        del self._operations[operation_id]
        return ack


def register_managed_pool(config):
    """Register access only. No create/destroy/provider allocator API is used."""
    module = importlib.import_module('uflow.managed_pool')
    path = Path(module.__file__).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != config.sdk_sha256:
        raise ValueError('managed pool SDK differs from the accepted pinned module')
    client = module.ManagedPoolClient.register(endpoint=config.pool_endpoint, pool_id=config.pool_id,
                                              expected_epoch=config.pool_epoch, access='rw')
    try:
        port = SdkManagedPoolPort(client)
        if (port.identity.pool_id, port.identity.pool_epoch) != (config.pool_id, config.pool_epoch):
            raise ValueError('registered access belongs to an unexpected managed pool')
        return ManagedPoolAccess(port)
    except BaseException:
        # No tensor, child, range grant or native job exists at this boundary.
        # Use the SDK's explicit close path; any failure remains a startup error.
        client.begin_unregister()
        client.view.close()
        client.ack_unmapped()
        raise
