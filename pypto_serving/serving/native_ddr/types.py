# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Address-free, immutable Run02 host contracts. Importing this module allocates no pool."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Operation(str, Enum):
    STORE = "STORE"
    LOAD = "LOAD"


class TransferStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EntryState(str, Enum):
    ABSENT = "ABSENT"
    STORE_RESERVED = "STORE_RESERVED"
    OFFLOADING = "OFFLOADING"
    READY = "READY"
    EVICT_PENDING = "EVICT_PENDING"
    QUARANTINED = "QUARANTINED"


class LoadState(str, Enum):
    RESERVED = "RESERVED"
    LOADING = "LOADING"
    NATIVE_COMPLETED = "NATIVE_COMPLETED"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"
    DISCARD_PENDING = "DISCARD_PENDING"
    QUARANTINED = "QUARANTINED"


class RestoreKind(str, Enum):
    MISS = "MISS"
    WAIT = "WAIT"
    READY = "READY"


@dataclass(frozen=True)
class NativeDdrPrefixConfig:
    enabled: bool = False
    capacity_bytes: int = 0
    max_inflight_transfers: int = 1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("native DDR enabled must be boolean")
        if type(self.capacity_bytes) is not int or self.capacity_bytes < 0:
            raise ValueError("native DDR capacity must be a nonnegative integer")
        if type(self.max_inflight_transfers) is not int or self.max_inflight_transfers != 1:
            raise ValueError("Run02 supports max_inflight_transfers=1 only")
        if self.enabled and self.capacity_bytes < 20971520:
            raise ValueError("native DDR needs explicit capacity for at least one complete block (20971520)")

    def validate_runtime(self, *, executor_cls, tp, page_size, dtype, grouped=False, prefix_cache=True):
        if not self.enabled:
            return
        if (executor_cls != "PyptoQwen14BExecutor" or tp != 1 or page_size != 128
                or dtype != "bfloat16" or grouped or not prefix_cache):
            raise ValueError("native DDR supports only Qwen TP1 BF16 page128 single-pool prefix caching")


@dataclass(frozen=True)
class QwenKvLayout:
    num_layers: int
    num_pages: int
    page_size: int
    num_kv_heads: int
    head_dim: int
    dtype: str = "bfloat16"
    layout_version: str = "qwen3-lpthd-bf16-v1"

    def __post_init__(self):
        if any(type(i) is not int for i in (self.num_layers, self.num_pages, self.page_size,
                                            self.num_kv_heads, self.head_dim)):
            raise ValueError("Qwen KV dimensions must be integers")
        if (self.num_layers, self.page_size, self.num_kv_heads, self.head_dim, self.dtype,
                self.layout_version) != (40, 128, 8, 128, "bfloat16", "qwen3-lpthd-bf16-v1"):
            raise ValueError("unsupported Qwen KV layout")
        if type(self.num_pages) is not int or self.num_pages <= 0:
            raise ValueError("num_pages must be positive")

    @property
    def page_side_bytes(self):
        return self.page_size * self.num_kv_heads * self.head_dim * 2

    @property
    def block_bytes(self):
        return 2 * self.num_layers * self.page_side_bytes

    def hbm_offset(self, layer: int, page: int):
        if (type(layer) is not int or type(page) is not int
                or not 0 <= layer < self.num_layers or not 0 <= page < self.num_pages):
            raise ValueError("HBM layer/page out of range")
        return (layer * self.num_pages + page) * self.page_side_bytes

    def ddr_offset(self, slot: int, layer: int, side: int):
        if (any(type(i) is not int for i in (slot, layer, side))
                or slot < 0 or not 0 <= layer < self.num_layers or side not in (0, 1)):
            raise ValueError("DDR slot/layer/side out of range")
        return slot * self.block_bytes + (layer * 2 + side) * self.page_side_bytes


@dataclass(frozen=True)
class PrefixKey:
    namespace_digest: bytes
    block_hash: bytes

    def __post_init__(self):
        if (not isinstance(self.namespace_digest, bytes) or not isinstance(self.block_hash, bytes)
                or len(self.namespace_digest) != 32 or len(self.block_hash) != 32):
            raise ValueError("prefix identity requires two SHA256 digests")


@dataclass(frozen=True)
class DdrEntryRef:
    worker_epoch: str
    entry_id: str
    slot_generation: int


@dataclass(frozen=True)
class KvPageLease:
    lease_id: str
    cache_epoch: str
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    purpose: str


@dataclass(frozen=True)
class DdrTransferRequest:
    job_id: str
    worker_epoch: str
    operation: Operation
    entry_keys: tuple[PrefixKey, ...]
    entry_refs: tuple[DdrEntryRef, ...]
    page_lease_id: str
    page_ids: tuple[int, ...]
    source_step_id: int | None
    delivery_sequence: int = 0
    retire_through: int = field(default=0, compare=False)

    def validate(self, layout: QwenKvLayout):
        n = len(self.page_ids)
        if (not self.job_id or not self.page_lease_id or not self.worker_epoch or n != 1
                or len(self.entry_keys) != n or len(self.entry_refs) != n):
            raise ValueError("Run02 transfers require one complete one-to-one block and lease identity")
        if not isinstance(self.operation, Operation):
            raise ValueError("invalid transfer operation")
        if self.operation == Operation.STORE and (type(self.source_step_id) is not int or self.source_step_id <= 0):
            raise ValueError("STORE requires a confirmed source step identity")
        if len(set(self.page_ids)) != n or len(set(self.entry_keys)) != n or len(set(self.entry_refs)) != n:
            raise ValueError("duplicate transfer mapping")
        for page, ref in zip(self.page_ids, self.entry_refs):
            if type(page) is not int or not 0 <= page < layout.num_pages:
                raise ValueError("transfer page out of range")
            if (ref.worker_epoch != self.worker_epoch or type(ref.slot_generation) is not int
                    or ref.slot_generation <= 0 or not ref.entry_id):
                raise ValueError("entry epoch/generation mismatch")


@dataclass(frozen=True)
class DdrTransferEvent:
    job_id: str
    worker_epoch: str
    operation: Operation
    status: TransferStatus
    entry_refs: tuple[DdrEntryRef, ...]
    transferred_bytes: int
    native_completed: bool
    stopped: bool
    error: str | None = None
    delivery_sequence: int = 0


@dataclass(frozen=True)
class RestoreDecision:
    kind: RestoreKind
    job_ids: tuple[str, ...] = ()
    reuse_limit: int = 0


@dataclass(frozen=True)
class DdrStartup:
    """Owner-created, address-free incarnation handshake, published before readiness."""
    worker_epoch: str
    namespace_digest: bytes
    layout: QwenKvLayout
    capacity_bytes: int
    max_entries: int
    compatibility_json: bytes


@dataclass(frozen=True)
class DdrControlEvent:
    command_id: str
    worker_epoch: str
    operation: str
    entry_ref: DdrEntryRef | None = None
    released: bool = False
    error: str | None = None
    payload: bytes = b""
    delivery_sequence: int = 0


@dataclass(frozen=True)
class DdrControlRequest:
    command_id: str
    worker_epoch: str
    operation: str
    job_id: str = ""
    entry_ref: DdrEntryRef | None = None
    page_ids: tuple[int, ...] = ()
    diagnostic_label: str = ""
    fault: str = ""
    delivery_sequence: int = 0
    retire_through: int = field(default=0, compare=False)
