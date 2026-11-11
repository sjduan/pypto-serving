# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit host-only E10 read/copy/lease handoff. Never a native LOAD or Ready.

The caller supplies independent CURRENT source and destination authorities and
trusted compatibility. CPU staging is a separate resource, not an NPU page.
There is one preallocated block and one bounded full-block validation buffer.
No tensor/runtime or SDK is imported merely by importing this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

from .shared_compatibility import BLOCK_BYTES
from .shared_prefix_record import ExternalBackingIdentity, validate_candidate

E10_SDK_REVISION = "4fba229373f0b27f0d1c0f6a7aab405542c8fdda"
GUARD_BYTES = 64


class StageCancelled(RuntimeError):
    """A cooperative host cancellation, not evidence of native cancellation."""


class AcquireRejected(RuntimeError):
    """Adapter explicitly proved rejection before any resource obligation.

    The fixed E10 SDK does not expose this distinction for all transport errors,
    so its adapter never invents this classification from an exception string.
    """


class ExchangeFailure(RuntimeError):
    """Keep the actual retryable resources, including ambiguous obligations."""

    def __init__(self, cause, resources):
        super().__init__(str(cause))
        self.cause, self.resources = cause, resources


@dataclass(eq=False)
class StagingLease:
    owner: "HostStagingOwner"
    generation: int
    valid_copy: bool = False
    released: bool = False


class HostStagingOwner:
    """One actual preallocated CPU slot. Reserve/release never reallocates it.

    Intended for a single host caller; not a concurrent allocator or authority.
    No mutable view is exported to a caller after stage_block returns.
    """

    def __init__(self):
        self._storage = bytearray(BLOCK_BYTES + 2 * GUARD_BYTES)
        self._storage[:GUARD_BYTES] = b"L" * GUARD_BYTES
        self._storage[-GUARD_BYTES:] = b"R" * GUARD_BYTES
        self._active = None
        self._generation = 0
        self.copy_attempts = 0
        self.verified_copied_bytes = 0
        self.partial_write_unknown = False

    @property
    def busy(self):
        return self._active is not None

    @property
    def capacity_bytes(self):
        return BLOCK_BYTES

    def reserve(self):
        if self.busy:
            raise RuntimeError("host staging slot is busy")
        self._generation += 1
        self._active = StagingLease(self, self._generation)
        return self._active

    def check(self, lease):
        if (type(lease) is not StagingLease or lease.owner is not self
                or self._active is not lease or lease.released
                or lease.generation != self._generation):
            raise RuntimeError("stale host staging lease")

    def guards_ok(self):
        return (self._storage[:GUARD_BYTES] == b"L" * GUARD_BYTES
                and self._storage[-GUARD_BYTES:] == b"R" * GUARD_BYTES)

    def _release(self, lease):
        self.check(lease)
        lease.valid_copy = False
        lease.released = True
        self._active = None

    def read(self, lease):
        self.check(lease)
        if not lease.valid_copy:
            raise RuntimeError("invalid staging bytes")
        with memoryview(self._storage) as whole:
            return bytes(whole[GUARD_BYTES:-GUARD_BYTES])

    def _copy(self, lease, payload, copy_into=None, cancelled=None):
        self.check(lease)
        if cancelled is not None and cancelled():
            raise StageCancelled("cancelled before CPU copy")
        if not self.guards_ok():
            raise RuntimeError("staging guards corrupt before copy")
        self.copy_attempts += 1
        try:
            with memoryview(self._storage) as whole, memoryview(payload) as source:
                target = whole[GUARD_BYTES:-GUARD_BYTES]
                try:
                    if copy_into is None:
                        target[:] = source
                    else:
                        # Trusted host fixture hook; a failure can partially write.
                        copy_into(target, source)
                    if target != source or not self.guards_ok():
                        raise RuntimeError("CPU staging bytes or guards differ")
                finally:
                    target.release()
            if cancelled is not None and cancelled():
                raise StageCancelled("cancelled after CPU copy; bytes are invalid")
        except BaseException:
            lease.valid_copy = False
            self.partial_write_unknown = True
            raise
        self.verified_copied_bytes += BLOCK_BYTES
        lease.valid_copy = True


class HeldResources:
    """A failed/unknown release never returns this slot to its owner."""

    def __init__(self, lease):
        self.lease = lease
        self.views = []
        self.unresolved_acquire = False
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.lease.valid_copy = False
        if self.unresolved_acquire:
            raise RuntimeError("acquire outcome unknown; external resolution required")
        while self.views:
            self.views[0].release()
            self.views.pop(0)  # Only a confirmed SDK release removes an obligation.
        self.lease.owner._release(self.lease)
        self.closed = True


class StagedBlock:
    """Verified CPU bytes plus strong source/staging references, explicitly non-Ready."""

    native_ready = False

    def __init__(self, resources, candidate, digest):
        self.resources, self.candidate, self.payload_sha256 = resources, candidate, digest
        self._closing = False

    @property
    def lease(self):
        return self.resources.lease

    def read(self):
        if self._closing or self.resources.closed:
            raise RuntimeError("stage is closed or release is unresolved")
        return self.lease.owner.read(self.lease)

    def close(self):
        self._closing = True
        self.resources.close()


class E10ReadView:
    """The only SDK-private coupling, pinned to E10_SDK_REVISION.

    E10 acquire sends fewer fields than its response. Compare ALL thirteen
    returned backing fields, including placement/binding/owner lease, and the
    actual range/read-only mmap before accepting bytes. Structure drift fails
    closed; no SDK/public protocol changes are made here.
    """

    def __init__(self, view):
        self._view = view

    def verify(self, source):
        actual = ExternalBackingIdentity.from_dict(asdict(self._view._identity.backing))
        if actual != source.backing:
            raise ValueError("E10 response complete backing tuple differs")
        if (type(self._view.allowed_offset_bytes) is not int
                or type(self._view.allowed_bytes) is not int
                or self._view.allowed_offset_bytes != source.offset_bytes
                or self._view.allowed_bytes != source.nbytes):
            raise ValueError("E10 response range differs")
        with memoryview(self._view._mapping) as mapping:
            if not mapping.readonly or len(mapping) != actual.size_bytes:
                raise ValueError("E10 mapping is not complete and read-only")

    def read(self):
        return self._view.read()

    def retain(self):
        return E10ReadView(self._view.retain())

    def release(self):
        self._view.release()


class E10Adapter:
    def __init__(self, control_socket):
        # Ordinary uflow package surface, no tensors. Deliberately lazy.
        from uflow import ExternalBackingHandle, ExternalDdrClient

        self._handle_type = ExternalBackingHandle
        self.client = ExternalDdrClient(control_socket)

    def acquire(self, source):
        handle = self._handle_type(**asdict(source.backing))
        return E10ReadView(self.client.acquire(handle, offset_bytes=source.offset_bytes, nbytes=source.nbytes))


def stage_block(record, *, compatibility, current_source, destination, current_destination,
                expected_chain_hash, expected_token_start, expected_payload_sha256,
                staging, adapter, cancelled=None, copy_into=None):
    """Acquire/retain/hash, recheck current leases, then perform one real CPU copy.

    The source/destination callables are independent trusted authorities, never
    derived from the untrusted record. This function does not authenticate that
    caller or bind host attestations to loaded replicas. No scheduler is used.
    On ExchangeFailure inspect resources and retry close; never drop unknowns.
    """
    def validate():
        return validate_candidate(
            record, expected_compatibility=compatibility, expected_source=current_source(),
            destination=destination, expected_destination=current_destination(),
            expected_chain_hash=expected_chain_hash, expected_token_start=expected_token_start,
            expected_payload_sha256=expected_payload_sha256,
        )

    candidate = validate()
    resources = HeldResources(staging.reserve())
    try:
        try:
            root = adapter.acquire(candidate.record.source)
        except AcquireRejected:
            raise
        except BaseException:
            resources.unresolved_acquire = True
            raise
        resources.views.append(root)
        root.verify(candidate.record.source)
        payload = root.read()
        if type(payload) is not bytes or len(payload) != BLOCK_BYTES:
            raise ValueError("source read is not one complete CPU block")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_payload_sha256:
            raise ValueError("actual complete payload digest differs")
        try:
            retained = root.retain()
        except BaseException:
            # Retain uses the same provisional handoff as acquire. A transport
            # exception does not prove its remote obligation was rolled back.
            resources.unresolved_acquire = True
            raise
        resources.views.append(retained)
        retained.verify(candidate.record.source)
        root.release()
        resources.views.remove(root)
        validate()  # Reacquire current independent source and destination bindings.
        staging.check(resources.lease)
        staging._copy(resources.lease, payload, copy_into, cancelled)
        return StagedBlock(resources, candidate, digest)
    except BaseException as cause:
        try:
            resources.close()
        except BaseException:
            pass  # The exception retains the actual handle and unreleased slot.
        raise ExchangeFailure(cause, resources) from cause
