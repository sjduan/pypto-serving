# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lazy, bit-preserving Qwen prefix programs. Production native-DDR assembly stays fail-closed."""
from pypto_serving.serving.native_ddr.types import QwenKvLayout


class NativeDdrUnavailable(RuntimeError):
    pass


def require_native_ddr_backend():
    raise NativeDdrUnavailable(
        "native DDR M3 seam unresolved: production startup namespace, page leases and completion "
        "publication are not assembled; no CPU/mock fallback"
    )


def layout_from_model(model, runtime, num_pages):
    return QwenKvLayout(model.num_hidden_layers, num_pages, runtime.page_size,
                        model.num_key_value_heads, model.head_dim, runtime.kv_dtype)


def qwen3_prefix_pack_host(layout: QwenKvLayout | None = None):
    """Construct a full-block copy with a runtime arena row dimension."""
    import pypto.language as pl

    rows = pl.dynamic("QWEN_PREFIX_KV_ROWS")

    @pl.jit
    def prefix_pack(
        keys: pl.Tensor[[rows, 128], pl.BF16],
        values: pl.Tensor[[rows, 128], pl.BF16],
        page_ids: pl.Tensor[[1], pl.INT32],
        compact: pl.Out[pl.Tensor[[81920, 128], pl.BF16]],
    ) -> pl.Tensor[[81920, 128], pl.BF16]:
        pages = pl.tensor.dim(keys, 0) // 40960
        for layer in pl.parallel(40):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefix_pack"):
                page = pl.cast(pl.tensor.read(page_ids, [0]), pl.INDEX)
                for tile in pl.range(64):
                    hbm_row = (layer * pages + page) * 1024 + tile * 16
                    k_row = layer * 2048 + tile * 16
                    v_row = k_row + 1024
                    k = pl.slice(keys, [16, 128], [hbm_row, 0])
                    v = pl.slice(values, [16, 128], [hbm_row, 0])
                    compact = pl.assemble(compact, k, [k_row, 0])
                    compact = pl.assemble(compact, v, [v_row, 0])
        return compact

    @pl.jit.host
    def qwen3_prefix_pack_host(
        keys: pl.Tensor[[rows, 128], pl.BF16],
        values: pl.Tensor[[rows, 128], pl.BF16],
        page_ids: pl.Tensor[[1], pl.INT32],
        compact: pl.Out[pl.Tensor[[81920, 128], pl.BF16]],
    ) -> pl.Tensor[[81920, 128], pl.BF16]:
        return prefix_pack(keys, values, page_ids, compact)

    return qwen3_prefix_pack_host


def qwen3_prefix_unpack_host(layout: QwenKvLayout | None = None):
    """Construct a resident InOut scatter into the original owning KV allocations."""
    import pypto.language as pl

    rows = pl.dynamic("QWEN_PREFIX_KV_ROWS")

    @pl.jit
    def prefix_unpack(
        compact: pl.Tensor[[81920, 128], pl.BF16],
        page_ids: pl.Tensor[[1], pl.INT32],
        keys: pl.InOut[pl.Tensor[[rows, 128], pl.BF16]],
        values: pl.InOut[pl.Tensor[[rows, 128], pl.BF16]],
    ) -> tuple[pl.Tensor[[rows, 128], pl.BF16], pl.Tensor[[rows, 128], pl.BF16]]:
        pages = pl.tensor.dim(keys, 0) // 40960
        for layer in pl.parallel(40):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefix_unpack"):
                page = pl.cast(pl.tensor.read(page_ids, [0]), pl.INDEX)
                for tile in pl.range(64):
                    hbm_row = (layer * pages + page) * 1024 + tile * 16
                    k_row = layer * 2048 + tile * 16
                    v_row = k_row + 1024
                    k = pl.slice(compact, [16, 128], [k_row, 0])
                    v = pl.slice(compact, [16, 128], [v_row, 0])
                    keys = pl.assemble(keys, k, [hbm_row, 0])
                    values = pl.assemble(values, v, [hbm_row, 0])
        return keys, values

    @pl.jit.host
    def qwen3_prefix_unpack_host(
        compact: pl.Tensor[[81920, 128], pl.BF16],
        page_ids: pl.Tensor[[1], pl.INT32],
        keys: pl.InOut[pl.Tensor[[rows, 128], pl.BF16]],
        values: pl.InOut[pl.Tensor[[rows, 128], pl.BF16]],
    ) -> tuple[pl.Tensor[[rows, 128], pl.BF16], pl.Tensor[[rows, 128], pl.BF16]]:
        return prefix_unpack(compact, page_ids, keys, values)

    return qwen3_prefix_unpack_host


def compile_programs(compiler, layout=None):
    """Compile both programs before creating their owning DistributedWorker."""
    return (
        compiler.compile("qwen3_prefix_pack_host", qwen3_prefix_pack_host(layout)),
        compiler.compile("qwen3_prefix_unpack_host", qwen3_prefix_unpack_host(layout)),
    )
