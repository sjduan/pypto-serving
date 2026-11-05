# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Serving's typed consumer boundary. UFlow owns every allocation and range lock.

No SDK, tensor, runtime or device import occurs here. A port normalizes the
accepted v1 SDK; it does not implement an allocator, directory or lock manager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
import uuid

BLOCK_BYTES = 20971520
ALIGNMENT = 65536
GUARD_BYTES = 2097152
U64_MAX = (1 << 64) - 1


def positive(value, name):
    if type(value) is not int or not 0 < value <= U64_MAX:
        raise ValueError(f'{name} must be a positive u64')


def opaque(value, name):
    if not isinstance(value, str) or not value or len(value) > 512 or '\x00' in value:
        raise ValueError(f'invalid {name}')


def interval(offset, size, capacity):
    if type(offset) is not int or not 0 <= offset <= U64_MAX:
        raise ValueError('invalid absolute range offset')
    positive(size, 'range length')
    if offset + size > capacity or offset + size > U64_MAX:
        raise ValueError('range escapes its physical resource')


@dataclass(frozen=True)
class PoolIdentity:
    pool_id: str
    pool_epoch: str
    provider_session: str
    client_session: str
    nbytes: int
    device: int
    inode: int
    resource_id: str
    placement_id: str
    binding_id: str
    access_generation: int
    simpler_owner: str
    simpler_buffer_id: int
    simpler_generation: int
    resource_generation: int
    provider_kind: str = 'posix_shm_simpler'

    def __post_init__(self):
        for name in ('pool_id', 'pool_epoch', 'provider_session', 'client_session',
                     'resource_id', 'placement_id', 'binding_id', 'simpler_owner'):
            opaque(getattr(self, name), name)
        for name in ('nbytes', 'inode', 'access_generation', 'simpler_buffer_id',
                     'simpler_generation', 'resource_generation'):
            positive(getattr(self, name), name)
        if type(self.device) is not int or not 0 <= self.device <= U64_MAX or self.provider_kind != 'posix_shm_simpler':
            raise ValueError('unsupported managed pool physical identity')


@dataclass(frozen=True)
class PoolBlock:
    pool_id: str
    pool_epoch: str
    block_id: int
    generation: int
    byte_offset: int
    nbytes: int

    def validate(self, pool):
        if (self.pool_id, self.pool_epoch) != (pool.pool_id, pool.pool_epoch):
            raise ValueError('pool allocation identity mismatch')
        positive(self.block_id, 'block ID')
        positive(self.generation, 'allocation generation')
        interval(self.byte_offset, self.nbytes, pool.nbytes)


@dataclass(frozen=True)
class PoolLease:
    pool: PoolIdentity
    block: PoolBlock
    lease_id: str
    lease_generation: int
    byte_offset: int
    nbytes: int
    mode: str
    state: str
    content_version: int
    holder_session: str
    token: str = field(repr=False)
    content_id: str = ''

    def validate(self, pool, block, mode, offset, size):
        if self.pool != pool or self.block != block:
            raise ValueError('range lease pool/client/allocation identity mismatch')
        block.validate(pool)
        opaque(self.lease_id, 'range lease ID')
        positive(self.lease_generation, 'lease generation')
        positive(self.content_version, 'content version')
        opaque(self.token, 'range capability token')
        if self.holder_session != pool.client_session:
            raise ValueError('range lease holder is not the registered client session')
        interval(self.byte_offset, self.nbytes, pool.nbytes)
        if not block.byte_offset <= self.byte_offset or self.byte_offset + self.nbytes > block.byte_offset + block.nbytes:
            raise ValueError('range lease escapes allocation')
        if (self.mode, self.byte_offset, self.nbytes) != (mode, offset, size):
            raise ValueError('range lease widened or changed mode')
        if self.state != ('Writing' if mode == 'w' else 'Committed'):
            raise ValueError('range is not Writing/Committed as required')
        if mode == 'r':
            opaque(self.content_id, 'committed content identity')


@dataclass(frozen=True)
class CommittedRange:
    pool_id: str
    pool_epoch: str
    provider_session: str
    block: PoolBlock
    byte_offset: int
    nbytes: int
    content_version: int
    content_id: str
    operation_id: str
    writer_lease_id: str

    def validate(self, pool, *, content_id):
        if (self.pool_id, self.pool_epoch, self.provider_session) != (
                pool.pool_id, pool.pool_epoch, pool.provider_session):
            raise ValueError('committed resource incarnation mismatch')
        self.block.validate(pool)
        interval(self.byte_offset, self.nbytes, pool.nbytes)
        if (self.byte_offset < self.block.byte_offset
                or self.byte_offset + self.nbytes > self.block.byte_offset + self.block.nbytes
                or self.byte_offset % ALIGNMENT or self.nbytes != BLOCK_BYTES):
            raise ValueError('committed entry is not a complete aligned Qwen block')
        positive(self.content_version, 'committed content version')
        opaque(self.content_id, 'committed content identity')
        opaque(self.operation_id, 'commit operation')
        opaque(self.writer_lease_id, 'writer lease')
        if self.content_id != content_id:
            raise ValueError('loaded compatibility/prefix content mismatch')


@dataclass(frozen=True)
class PoolAck:
    operation_id: str
    operation: str
    pool: PoolIdentity
    lease_id: str = ''
    lease_generation: int = 0
    applied: bool = False
    child_pid: int = 0
    child_birth: int = 0

    def validate(self, pool, operation, operation_id, lease=None):
        if (self.pool, self.operation, self.operation_id) != (pool, operation, operation_id) or self.applied is not True:
            raise ValueError('old or ambiguous pool control acknowledgement')
        if lease is not None and (self.lease_id, self.lease_generation) != (lease.lease_id, lease.lease_generation):
            raise ValueError('acknowledgement belongs to another lease generation')


class ManagedPoolPort(Protocol):
    """Normalized accepted SDK contract, never a second resource authority."""
    identity: PoolIdentity
    buffer: object

    def allocate_range(self, nbytes: int, alignment: int) -> PoolBlock: ...
    def acquire_range(self, block, mode, expected_generation, byte_offset, nbytes) -> PoolLease: ...
    def revalidate(self, lease: PoolLease) -> None: ...
    def commit_write(self, lease, content_id, nbytes, operation_id) -> CommittedRange: ...
    def release(self, lease, operation_id) -> PoolAck: ...
    def query_operation(self, operation_id) -> CommittedRange | PoolAck | None: ...
    def report_unknown(self, lease, reason) -> None: ...
    def attach_child(self, pid, birth) -> PoolAck: ...
    def detach_child(self, pid, birth) -> PoolAck: ...
    def begin_unregister(self) -> None: ...
    def close_view(self) -> None: ...
    def ack_unmapped(self, operation_id) -> PoolAck: ...


class PoolOutcomeUnknown(RuntimeError):
    pass


@dataclass(frozen=True)
class ManagedTransferReceipt:
    operation: str
    pool: PoolIdentity
    lease_id: str
    lease_generation: int
    content: CommittedRange
    page_lease_id: str
    release_ack: PoolAck | None = None

    def validate(self, *, pool, operation, page_lease_id, content_id):
        if (self.pool, self.operation, self.page_lease_id) != (pool, operation, page_lease_id):
            raise ValueError('managed transfer completion belongs to another pool/page lease')
        opaque(self.lease_id, 'native range lease')
        positive(self.lease_generation, 'native lease generation')
        self.content.validate(pool, content_id=content_id)
        if operation == 'STORE':
            if self.release_ack is not None or self.content.writer_lease_id != self.lease_id:
                raise ValueError('STORE completion has no matching committed writer')
        elif operation == 'LOAD':
            ack = self.release_ack
            if (not isinstance(ack, PoolAck) or ack.pool != pool or ack.operation != 'release'
                    or ack.applied is not True or (ack.lease_id, ack.lease_generation) != (
                        self.lease_id, self.lease_generation)):
                raise ValueError('LOAD completion has no matching released reader')
            opaque(ack.operation_id, 'reader release operation')
        else:
            raise ValueError('invalid managed native operation')


def prefix_content_id(key):
    return 'qwen3-prefix-v1:' + key.namespace_digest.hex() + ':' + key.block_hash.hex()


class ManagedPoolAccess:
    """Local references to UFlow promises; no client-side lock or budget decisions."""
    def __init__(self, port: ManagedPoolPort):
        self.port = port
        self.identity = port.identity
        if not isinstance(self.identity, PoolIdentity):
            raise ValueError('normalized managed pool identity required')
        view = memoryview(port.buffer)
        if view.readonly or not view.c_contiguous or view.nbytes != self.identity.nbytes:
            view.release()
            raise ValueError('registered view must be the whole RW MAP_SHARED pool')
        self.buffer = view
        self.leases = {}
        self.children = {}
        self.unknown = {}
        self.closed = False
        self.unregistering = False
        self._unmap_operation = uuid.uuid4().hex

    def _admit(self):
        if self.closed or self.unregistering or self.unknown:
            raise RuntimeError('managed pool access stopped or has unknown obligations')

    def allocate(self):
        self._admit()
        block = self.port.allocate_range(BLOCK_BYTES + GUARD_BYTES, ALIGNMENT)
        block.validate(self.identity)
        if block.nbytes != BLOCK_BYTES + GUARD_BYTES:
            raise ValueError('UFlow allocation did not preserve the requested guard interval')
        return block

    def acquire(self, block, mode, *, committed=None, byte_offset=None):
        self._admit()
        block.validate(self.identity)
        if mode not in ('r', 'w'):
            raise ValueError('invalid managed access mode')
        offset = committed.byte_offset if committed is not None else (
            block.byte_offset + GUARD_BYTES if byte_offset is None else byte_offset)
        size = BLOCK_BYTES
        interval(offset, size, self.identity.nbytes)
        if offset < block.byte_offset or offset + size > block.byte_offset + block.nbytes or offset % ALIGNMENT:
            raise ValueError('native range escapes its allocation or alignment')
        try:
            lease = self.port.acquire_range(block, mode, block.generation, offset - block.byte_offset, size)
        except (PoolOutcomeUnknown, TimeoutError, ConnectionError) as error:
            # A lost or undecodable grant is not a rejected lock request. The
            # SDK port retains its original reply/call for later reconciliation.
            self.report_unknown(None, f'range acquisition outcome unknown: {error}')
            raise
        # Retain even a malformed/ambiguous SDK answer. Do not silently lose a
        # lock granted remotely because local validation discovered a mismatch.
        identity = getattr(lease, 'lease_id', uuid.uuid4().hex)
        self.leases[identity] = lease
        try:
            lease.validate(self.identity, block, mode, offset, size)
            if committed is not None:
                committed.validate(self.identity, content_id=committed.content_id)
                if (lease.content_version, lease.content_id) != (committed.content_version, committed.content_id):
                    raise ValueError('committed version changed before native admission')
        except Exception as error:
            self.report_unknown(lease, str(error))
            raise
        return lease

    def report_unknown(self, lease, reason):
        key = getattr(lease, 'lease_id', 'unidentified-range')
        self.unknown[key] = (lease, reason)
        try:
            self.port.report_unknown(lease, reason)
        except Exception:
            # Failure to report cannot make the local unknown obligation vanish.
            pass

    def _operation(self, operation_id, call):
        try:
            return call()
        except TimeoutError:
            result = self.port.query_operation(operation_id)
            if result is None:
                raise PoolOutcomeUnknown('control timeout has no same-operation terminal receipt') from None
            return result

    def complete_write(self, lease, content_id, operation_id):
        if self.leases.get(lease.lease_id) is not lease or lease.mode != 'w':
            raise ValueError('write completion has no exact retained range lease')
        try:
            self.port.revalidate(lease)
            committed = self._operation(operation_id, lambda: self.port.commit_write(
                lease, content_id, BLOCK_BYTES, operation_id))
            if not isinstance(committed, CommittedRange):
                raise ValueError('commit has no typed committed range receipt')
            committed.validate(self.identity, content_id=content_id)
            if (committed.block, committed.byte_offset, committed.content_version,
                    committed.operation_id, committed.writer_lease_id) != (
                    lease.block, lease.byte_offset, lease.content_version, operation_id, lease.lease_id):
                raise ValueError('commit receipt does not match the exact completed write')
        except Exception as error:
            self.report_unknown(lease, str(error))
            raise
        del self.leases[lease.lease_id]
        return committed

    def complete_read(self, lease, operation_id):
        if self.leases.get(lease.lease_id) is not lease or lease.mode != 'r':
            raise ValueError('read completion has no exact retained range lease')
        try:
            self.port.revalidate(lease)
            ack = self._operation(operation_id, lambda: self.port.release(lease, operation_id))
            if not isinstance(ack, PoolAck):
                raise ValueError('read release has no typed acknowledgement')
            ack.validate(self.identity, 'release', operation_id, lease)
        except Exception as error:
            self.report_unknown(lease, str(error))
            raise
        del self.leases[lease.lease_id]
        return ack

    def attach_child(self, pid, birth):
        self._admit()
        positive(pid, 'child pid')
        positive(birth, 'child birth')
        key = (pid, birth)
        if key in self.children:
            return
        self.children[key] = None
        try:
            ack = self.port.attach_child(pid, birth)
            if (not isinstance(ack, PoolAck) or ack.pool != self.identity or ack.applied is not True
                    or (ack.operation, ack.child_pid, ack.child_birth) != ('attach_child', pid, birth)):
                raise ValueError('child attach receipt mismatch')
            opaque(ack.operation_id, 'child attach operation')
            self.children[key] = ack
        except Exception as error:
            self.report_unknown(None, f'child attach outcome unknown: {error}')
            raise

    def detach_stopped_child(self, pid, birth, *, owner_drained, process_gone):
        key = (pid, birth)
        if key not in self.children:
            raise ValueError('no matching inherited child mapping obligation')
        if owner_drained is not True or process_gone is not True:
            raise RuntimeError('native drain and original child exit must both be proven')
        try:
            ack = self.port.detach_child(pid, birth)
            if (not isinstance(ack, PoolAck) or ack.pool != self.identity or ack.applied is not True
                    or (ack.operation, ack.child_pid, ack.child_birth) != ('detach_child', pid, birth)):
                raise ValueError('child detach receipt mismatch')
            opaque(ack.operation_id, 'child detach operation')
        except Exception as error:
            self.report_unknown(None, f'child detach outcome unknown: {error}')
            raise
        del self.children[key]

    def close(self):
        if self.closed:
            return
        if self.leases or self.children or self.unknown:
            raise RuntimeError('pool client still owns range/child/unknown obligations')
        self.unregistering = True
        self.port.begin_unregister()
        # Derived tensor views must already be gone. SDK close must propagate
        # BufferError; no unmap acknowledgement follows a failed local close.
        self.buffer.release()
        self.port.close_view()
        operation_id = self._unmap_operation
        ack = self._operation(operation_id, lambda: self.port.ack_unmapped(operation_id))
        if not isinstance(ack, PoolAck):
            raise PoolOutcomeUnknown('unmap acknowledgement missing')
        ack.validate(self.identity, 'unmapped', operation_id)
        self.closed = True
