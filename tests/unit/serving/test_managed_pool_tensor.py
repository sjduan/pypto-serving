# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real CPU tensors over a fake port: alias/layout proof, NOT UF/native proof."""
import ctypes
import os
import unittest

from tests.unit.serving.test_managed_pool_protocol import FakePort
from pypto_serving.model.qwen.native_ddr_transfer import ManagedDdrSlots
from pypto_serving.serving.native_ddr.managed_pool import BLOCK_BYTES, ManagedPoolAccess


class ManagedTensorTests(unittest.TestCase):
    def test_whole_span_and_five_absolute_views_alias_the_original_mapping_without_a_kv_copy(self):
        import torch
        port = FakePort()
        access = ManagedPoolAccess(port)
        slots = ManagedDdrSlots(5, access)
        base = ctypes.addressof(ctypes.c_char.from_buffer(port.buffer))
        self.assertEqual(slots.pool_tensor.data_ptr(), base)
        self.assertEqual(slots.pool_tensor.untyped_storage().nbytes(), port.identity.nbytes)
        self.assertEqual(slots.control_metadata_bytes, 4)
        if os.name == 'posix':
            self.assertGreaterEqual(slots.control_metadata_mapped_bytes, 4)
        else:
            self.assertIsNone(slots.control_metadata_mapped_bytes)  # Linux VMA proof intentionally unavailable
        self.assertEqual(slots.inherited_host_tensors, (slots.pool_tensor, slots.page_ids))
        self.assertFalse(slots.leases)  # no eager UF allocation for imports
        backings = [slots.reserve(index, BLOCK_BYTES) for index in range(5)]
        for index, backing in enumerate(backings):
            self.assertEqual(backing.tensor.data_ptr(), base + ((2 + index * 22) << 20))
            self.assertEqual(backing.tensor.shape, (81920, 128))
            self.assertEqual(backing.tensor.dtype, torch.bfloat16)
            self.assertEqual(backing.tensor.untyped_storage().data_ptr(), base)
        selected = backings[2]
        offset = selected.byte_offset
        port.buffer[offset - 64:offset] = b'G' * 64
        port.buffer[offset + BLOCK_BYTES:offset + BLOCK_BYTES + 64] = b'H' * 64
        selected.tensor.view(torch.uint8).fill_(0x3A)
        self.assertEqual(port.buffer[offset:offset + BLOCK_BYTES], b'\x3a' * BLOCK_BYTES)
        self.assertEqual(port.buffer[offset - 64:offset], b'G' * 64)
        self.assertEqual(port.buffer[offset + BLOCK_BYTES:offset + BLOCK_BYTES + 64], b'H' * 64)
        port.buffer[offset:offset + 16] = b'Z' * 16
        self.assertEqual(selected.tensor.view(torch.uint8).flatten()[:16].tolist(), [ord('Z')] * 16)
        self.assertFalse(access.leases or port.calls)  # alias probe has no fake native/commit
        for backing in backings:
            slots.release(backing)
        del selected, backing, backings
        slots.close_after_owner()
        self.assertTrue(access.closed and port.buffer.closed)

    def test_derived_tensor_alias_retains_real_buffer_export_until_retry_can_unmap_once(self):
        import torch
        aliases = (
            ('flat', lambda tensor: tensor.view(-1)),
            ('dtype-slice', lambda tensor: tensor.view(torch.uint8).flatten()[64:192].view(torch.int16)),
            ('detached', lambda tensor: tensor.detach()),
        )
        for name, make_alias in aliases:
            with self.subTest(alias=name):
                port = FakePort()
                access = ManagedPoolAccess(port)
                slots = ManagedDdrSlots(1, access)
                backing = slots.reserve(0, BLOCK_BYTES)
                held = make_alias(backing.tensor)
                self.assertEqual(held.untyped_storage().data_ptr(), slots.pool_tensor.data_ptr())
                slots.release(backing)
                self.assertIsNone(backing.tensor)
                self.assertFalse(slots.leases)
                try:
                    # The CPU tensor alias owns a real mmap buffer export. No
                    # GC, refcount guess or fake control response establishes it.
                    with self.assertRaises(BufferError):
                        slots.close_after_owner()
                    self.assertFalse(port.buffer.closed or access.closed)
                    self.assertTrue(access.unregistering)
                    self.assertFalse(any(call[0] == 'unmapped' for call in port.calls))
                    self.assertIsNone(slots.pool_tensor)
                    self.assertGreater(held.numel(), 0)  # metadata only, no unsafe read if regression fails
                finally:
                    # Drop the last storage alias explicitly before retry. Never
                    # dereference its data following an unexpectedly early unmap.
                    del held
                    slots.close_after_owner()
                self.assertTrue(port.buffer.closed and access.closed)
                self.assertEqual(sum(call[0] == 'unmapped' for call in port.calls), 1)
                slots.close_after_owner()
                self.assertEqual(sum(call[0] == 'unmapped' for call in port.calls), 1)
