# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual fixed argument validator and Buffer views; CPU only, no runtime Worker or UF service."""
import os
import threading
from types import SimpleNamespace

import pytest

from pypto_serving.model.qwen.native_ddr_transfer import ManagedDdrSlots
from pypto_serving.serving.native_ddr.managed_pool import BLOCK_BYTES, ManagedPoolAccess
from tests.unit.serving.test_managed_pool_protocol import FakePort

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Linux fixed Simpler/PyPTO metadata surface')


def test_actual_frozen_validator_accepts_existing_buffer_view_without_torch_shm_copy():
    runtime = pytest.importorskip('pypto.runtime.distributed_runner')
    pytest.importorskip('simpler.buffer')

    class CpuSpanMetadataOnly:
        # Reuse the exact frozen metadata naming helpers; no DistributedWorker
        # is constructed, no chip or transport is created.
        _named_buffer_for_span = runtime.DistributedWorker._named_buffer_for_span
        _named_host_buffer = runtime.DistributedWorker._named_host_buffer

    port = FakePort()
    access = ManagedPoolAccess(port)
    slots = ManagedDdrSlots(2, access)
    owner = CpuSpanMetadataOnly()
    owner._named_host_buffers = {}
    owner._named_host_buffer_mu = threading.Lock()
    owner._buffer_owner_id, owner._buffer_id_seq = None, 0
    start = slots.pool_tensor.data_ptr()
    owner._inherited_host_spans = (((start, start + access.identity.nbytes), False),)
    first, second = slots.reserve(0, BLOCK_BYTES), slots.reserve(1, BLOCK_BYTES)
    try:
        assert not first.tensor.is_shared() and not second.tensor.is_shared()
        info = SimpleNamespace(name='external-UFlow-view')
        with pytest.raises(TypeError, match='host torch.Tensor'):
            runtime.DistributedWorker._validate_prepared_dispatch_arg(None, info, first.tensor)
        a, b = slots.native_argument(owner, first), slots.native_argument(owner, second)
        runtime.DistributedWorker._validate_prepared_dispatch_arg(None, info, a)
        runtime.DistributedWorker._validate_prepared_dispatch_arg(None, info, b)
        assert len(owner._named_host_buffers) == 1
        assert a.buffer.identity == b.buffer.identity
        assert a.buffer.body == b.buffer.body and int.from_bytes(a.buffer.body, 'little') == start
        assert a.byte_offset == first.byte_offset and b.byte_offset == second.byte_offset
        assert a.byte_offset != b.byte_offset and a.buffer.nbytes == access.identity.nbytes
        assert slots.pool_tensor.data_ptr() == start  # No replacement shared allocation.
        slots.release(first)
        slots.release(second)
        del a, b
        owner._named_host_buffers.clear()
        slots.close_after_owner()
        assert access.closed and port.buffer.closed
    finally:
        for backing in slots.leases.values():
            backing.tensor = None
        slots.pool_tensor = slots.page_ids = None
        access.buffer.release()
        if not port.buffer.closed:
            port.buffer.close()
