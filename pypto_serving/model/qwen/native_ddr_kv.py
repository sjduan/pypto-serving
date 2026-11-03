# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen layout and lazy pure-copy program construction, NOT a validated backend.

Construction/compilation is explicit. Importing this file never imports PyPTO,
creates a device context, allocates backing, or registers a native program.
M3 must validate these candidate programs on the pinned compiler before wiring
same-owner startup registration, prefork shared slots, and real native handles.
"""
from pypto_serving.serving.native_ddr.types import QwenKvLayout


class NativeDdrUnavailable(RuntimeError):
    pass


def require_native_ddr_backend():
    raise NativeDdrUnavailable(
        "native DDR M3 seam unresolved: same-owner startup registration, prefork shared backing, "
        "resident InOut unpack and native completion/cleanup must be verified; no CPU/mock fallback"
    )


def layout_from_model(model, runtime, num_pages):
    return QwenKvLayout(model.num_hidden_layers, num_pages, runtime.page_size,
                        model.num_key_value_heads, model.head_dim, runtime.kv_dtype)


def qwen3_prefix_pack_host(layout: QwenKvLayout):
    """Construct the candidate HOST entry; do not compile or execute it."""
    import pypto.language as pl

    rows = layout.num_layers * layout.num_pages * 1024
    pages = layout.num_pages

    @pl.jit
    def qwen3_prefix_pack_host(
        keys: pl.Tensor[[rows, 128], pl.BF16],
        values: pl.Tensor[[rows, 128], pl.BF16],
        page_ids: pl.Tensor[[1], pl.INT32],
        compact: pl.Out[pl.Tensor[[81920, 128], pl.BF16]],
    ) -> pl.Tensor[[81920, 128], pl.BF16]:
        for layer in pl.range(40):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefix_pack"):
                page = pl.cast(pl.read(page_ids, [0]), pl.INDEX)
                for tile in pl.range(64):
                    hbm_row = (layer * pages + page) * 1024 + tile * 16
                    k_row = layer * 2048 + tile * 16
                    v_row = k_row + 1024
                    k = pl.load(keys, [hbm_row, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
                    v = pl.load(values, [hbm_row, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
                    pl.store(k, [k_row, 0], compact)
                    pl.store(v, [v_row, 0], compact)
        return compact

    return qwen3_prefix_pack_host


def qwen3_prefix_unpack_host(layout: QwenKvLayout):
    """Construct resident InOut scatter; never substitute a second KV output."""
    import pypto.language as pl

    rows = layout.num_layers * layout.num_pages * 1024
    pages = layout.num_pages

    @pl.jit
    def qwen3_prefix_unpack_host(
        compact: pl.Tensor[[81920, 128], pl.BF16],
        page_ids: pl.Tensor[[1], pl.INT32],
        keys: pl.InOut[pl.Tensor[[rows, 128], pl.BF16]],
        values: pl.InOut[pl.Tensor[[rows, 128], pl.BF16]],
    ) -> tuple[pl.Tensor[[rows, 128], pl.BF16], pl.Tensor[[rows, 128], pl.BF16]]:
        for layer in pl.range(40):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefix_unpack"):
                page = pl.cast(pl.read(page_ids, [0]), pl.INDEX)
                for tile in pl.range(64):
                    hbm_row = (layer * pages + page) * 1024 + tile * 16
                    k_row = layer * 2048 + tile * 16
                    v_row = k_row + 1024
                    k = pl.load(compact, [k_row, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
                    v = pl.load(compact, [v_row, 0], [16, 128], target_memory=pl.MemorySpace.Vec)
                    pl.store(k, [hbm_row, 0], keys)
                    pl.store(v, [hbm_row, 0], values)
        return keys, values

    return qwen3_prefix_unpack_host


def compile_programs(compiler, layout):
    """Explicit KernelCompiler adapter. Not called by this host-only delivery."""
    return (
        compiler.compile("qwen3_prefix_pack_host", qwen3_prefix_pack_host(layout)),
        compiler.compile("qwen3_prefix_unpack_host", qwen3_prefix_unpack_host(layout)),
    )
