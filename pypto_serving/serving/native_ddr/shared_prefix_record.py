# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict source records and independent local destination bindings.

All results are metadata candidates only. E10's live acquire/retain/revoke and
the local page authority remain responsible for permissions and liveness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .shared_compatibility import (
    BLOCK_BYTES, MAX_INTEGER, SharedCompatibility, _digest_bytes,
    _integer, _keys, _sha256, _text, canonical_json, shared_layout, strict_json,
)


@dataclass(frozen=True)
class ExternalBackingIdentity:
    """Lossless thirteen-field E10 ExternalBackingHandle, without an SDK import."""
    host_id: str
    boot_id: str
    broker_instance: str
    owner_incarnation: str
    owner_name: str
    pool_generation: int
    resource_generation: int
    object_id: int
    placement_id: int
    binding_id: int
    owner_lease_id: int
    size_bytes: int
    purpose: str = "immutable"

    def __post_init__(self):
        for name in ("host_id", "boot_id", "broker_instance", "owner_incarnation", "owner_name"):
            _text(getattr(self, name))
        for name in ("pool_generation", "resource_generation", "object_id", "placement_id",
                     "binding_id", "owner_lease_id", "size_bytes"):
            _integer(getattr(self, name), 1)
        if self.purpose != "immutable":
            raise ValueError("only immutable source backing is supported")

    @classmethod
    def from_dict(cls, value):
        _keys(value, cls.__dataclass_fields__)
        return cls(**value)


@dataclass(frozen=True)
class SourceBinding:
    backing: ExternalBackingIdentity
    worker_epoch: str
    entry_id: str
    entry_generation: int
    offset_bytes: int
    nbytes: int = BLOCK_BYTES

    def __post_init__(self):
        if type(self.backing) is not ExternalBackingIdentity:
            raise ValueError("complete E10 backing identity required")
        _text(self.worker_epoch)
        _text(self.entry_id)
        _integer(self.entry_generation, 1)
        _integer(self.offset_bytes)
        _integer(self.nbytes, BLOCK_BYTES, BLOCK_BYTES)
        if self.offset_bytes % BLOCK_BYTES or self.offset_bytes > self.backing.size_bytes - self.nbytes:
            raise ValueError("source must contain one aligned complete block")

    @classmethod
    def from_dict(cls, value):
        _keys(value, cls.__dataclass_fields__)
        return cls(**{**value, "backing": ExternalBackingIdentity.from_dict(value["backing"])})


@dataclass(frozen=True)
class DestinationBinding:
    """Independent local arena/page lease, to compare against current authority.

    num_pages is local capacity, not part of the shared data ID. Epochs and
    page leases are never rewritten from the source or omitted on comparison.
    """
    worker_epoch: str
    arena_id: str
    cache_epoch: str
    num_pages: int
    page_id: int
    page_lease_id: str
    num_layers: int = 40
    page_size: int = 128
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"
    layout_version: str = "qwen3-lpthd-bf16-v1"

    def __post_init__(self):
        for name in ("worker_epoch", "arena_id", "cache_epoch", "page_lease_id"):
            _text(getattr(self, name))
        _integer(self.num_pages, 1, MAX_INTEGER // BLOCK_BYTES)
        _integer(self.page_id, 0, self.num_pages - 1)
        layout = shared_layout()
        for name in ("num_layers", "page_size", "num_kv_heads", "head_dim"):
            _integer(getattr(self, name), layout[name], layout[name])
        if self.dtype != layout["dtype"] or self.layout_version != layout["layout_version"]:
            raise ValueError("unsupported destination KV layout")

    @classmethod
    def from_dict(cls, value):
        _keys(value, cls.__dataclass_fields__)
        return cls(**value)


@dataclass(frozen=True)
class SharedPrefixRecord:
    compatibility_id: str
    chain_hash: bytes
    token_start: int
    token_end: int
    payload_sha256: str
    source: SourceBinding
    schema: int = 1

    def __post_init__(self):
        _integer(self.schema, 1, 1)
        _sha256(self.compatibility_id)
        _digest_bytes(self.chain_hash)
        _sha256(self.payload_sha256)
        _integer(self.token_start)
        _integer(self.token_end, 128)
        if self.token_start % 128 or self.token_end - self.token_start != 128:
            raise ValueError("one sealed 128-token block at an absolute aligned position required")
        if type(self.source) is not SourceBinding:
            raise ValueError("complete source binding required")

    def to_dict(self):
        value = asdict(self)
        value["chain_hash"] = self.chain_hash.hex()
        return value

    @classmethod
    def from_dict(cls, value):
        _keys(value, cls.__dataclass_fields__)
        return cls(**{**value, "chain_hash": bytes.fromhex(_sha256(value["chain_hash"])),
                      "source": SourceBinding.from_dict(value["source"])})


def encode_record(record):
    if type(record) is not SharedPrefixRecord:
        raise ValueError("expected one shared prefix record")
    return canonical_json(record.to_dict())


def decode_record(data):
    """Decode only; even a valid record grants no authority or Ready state."""
    return SharedPrefixRecord.from_dict(strict_json(data))


def encode_destination(binding):
    if type(binding) is not DestinationBinding:
        raise ValueError("expected a destination binding")
    return canonical_json(asdict(binding))


def decode_destination(data):
    return DestinationBinding.from_dict(strict_json(data))


@dataclass(frozen=True)
class MetadataCandidate:
    """A candidate for a later authority-checked Acquire, not permission to read."""
    record: SharedPrefixRecord
    destination: DestinationBinding


def validate_candidate(record, *, expected_compatibility, expected_source, destination,
                       expected_destination, expected_chain_hash, expected_token_start,
                       expected_payload_sha256):
    """Side-effect-free comparison against caller's CURRENT trusted bindings.

    Current source must be obtained from E10/owner authority and current target
    from the local page-lease authority. Equality cannot establish either is
    still alive; H2/production must acquire and hold the actual resources.
    The full serving-provided chained hash is compared, never recomputed here.
    """
    if type(record) is not SharedPrefixRecord or type(expected_compatibility) is not SharedCompatibility:
        raise ValueError("validated record and trusted compatibility required")
    if type(expected_source) is not SourceBinding:
        raise ValueError("current complete source identity required")
    if type(destination) is not DestinationBinding or type(expected_destination) is not DestinationBinding:
        raise ValueError("current independent destination binding required")
    _digest_bytes(expected_chain_hash)
    _integer(expected_token_start)
    _sha256(expected_payload_sha256)
    if record.compatibility_id != expected_compatibility.data_id:
        raise ValueError("incompatible shared data identity")
    if record.source != expected_source:
        raise ValueError("stale source backing, entry or range")
    if destination != expected_destination:
        raise ValueError("stale destination epoch, arena, page or lease")
    if (record.chain_hash != expected_chain_hash or record.token_start != expected_token_start
            or record.payload_sha256 != expected_payload_sha256):
        raise ValueError("prefix chain, absolute position or payload identity mismatch")
    if record.token_end > expected_compatibility.max_position_embeddings:
        raise ValueError("prefix exceeds compatible position limit")
    return MetadataCandidate(record, destination)
