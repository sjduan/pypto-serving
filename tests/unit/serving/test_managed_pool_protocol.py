# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stdlib consumer tests with an explicit fake port, NOT UFlow/native evidence."""
from dataclasses import replace
import mmap
import subprocess
import sys
import unittest

from pypto_serving.serving.native_ddr.managed_pool import (
    BLOCK_BYTES, GUARD_BYTES, CommittedRange, ManagedPoolAccess, PoolAck, PoolBlock,
    PoolIdentity, PoolLease, PoolOutcomeUnknown,
)


class FakePort:
    """Test-only control responses; anonymous host mmap is not a managed pool."""
    def __init__(self):
        self.identity = PoolIdentity('pool', 'epoch', 'provider', 'client', 134217728, 1, 2,
                                     'resource', 'placement', 'binding', 1)
        self.buffer = mmap.mmap(-1, self.identity.nbytes)
        self.calls, self.results = [], {}
        self.cursor = self.serial = 0
        self.corrupt = None
        self.lose_commit = False
        self.query_missing = False

    def allocate_range(self, nbytes, alignment):
        assert self.cursor % alignment == 0
        self.serial += 1
        block = PoolBlock('pool', 'epoch', self.serial, 1, self.cursor, nbytes)
        self.cursor += nbytes
        return block

    def acquire_range(self, block, mode, expected_generation, byte_offset, nbytes):
        assert expected_generation == block.generation
        self.serial += 1
        lease = PoolLease(self.identity, block, f'lease{self.serial}', 1,
                          block.byte_offset + byte_offset, nbytes, mode,
                          'Writing' if mode == 'w' else 'Committed', 1, '' if mode == 'w' else 'content')
        return self.corrupt(lease) if self.corrupt else lease

    def revalidate(self, lease):
        self.calls.append(('revalidate', lease.lease_id))
        if self.corrupt:
            raise ValueError('generation changed while native was pending')

    def commit_write(self, lease, content_id, nbytes, operation_id):
        self.calls.append(('commit', operation_id))
        result = CommittedRange('pool', 'epoch', 'provider', lease.block, lease.byte_offset,
                                nbytes, lease.content_version, content_id, operation_id, lease.lease_id)
        self.results[operation_id] = result
        if self.lose_commit:
            raise TimeoutError('lost commit ACK')
        return result

    def release(self, lease, operation_id):
        return PoolAck(operation_id, 'release', self.identity, lease.lease_id, lease.lease_generation, True)

    def query_operation(self, operation_id):
        self.calls.append(('query', operation_id))
        return None if self.query_missing else self.results.get(operation_id)

    def report_unknown(self, lease, reason):
        self.calls.append(('unknown', getattr(lease, 'lease_id', None), reason))

    def attach_child(self, pid, birth):
        return PoolAck('attach', 'attach_child', self.identity, applied=True, child_pid=pid, child_birth=birth)

    def detach_child(self, pid, birth):
        return PoolAck('detach', 'detach_child', self.identity, applied=True, child_pid=pid, child_birth=birth)

    def begin_unregister(self):
        self.calls.append(('unregister',))

    def close_view(self):
        self.calls.append(('close_view',))
        self.buffer.close()

    def ack_unmapped(self, operation_id):
        self.calls.append(('unmapped', operation_id))
        return PoolAck(operation_id, 'unmapped', self.identity, applied=True)


class ManagedPoolProtocolTests(unittest.TestCase):
    def setUp(self):
        self.port = FakePort()
        self.access = ManagedPoolAccess(self.port)

    def tearDown(self):
        # Explicit fake fixture teardown, never proof of a remote/native stop.
        self.access.buffer.release()
        if not self.port.buffer.closed:
            self.port.buffer.close()

    def committed(self):
        block = self.access.allocate()
        lease = self.access.acquire(block, 'w')
        return self.access.complete_write(lease, 'content', 'commit1')

    def test_five_guarded_ranges_use_the_one_pool_and_writes_only_commit_after_revalidation(self):
        blocks = [self.access.allocate() for _ in range(5)]
        self.assertEqual([b.byte_offset + GUARD_BYTES for b in blocks], [n << 20 for n in (2, 24, 46, 68, 90)])
        self.assertFalse(self.port.calls)
        lease = self.access.acquire(blocks[0], 'w')
        self.assertEqual(lease.nbytes, BLOCK_BYTES)
        self.assertFalse(self.port.calls)
        committed = self.access.complete_write(lease, 'content', 'write1')
        self.assertEqual(committed.writer_lease_id, lease.lease_id)
        self.assertEqual(self.port.calls[:2], [('revalidate', lease.lease_id), ('commit', 'write1')])
        self.assertFalse(self.access.leases)
        self.assertEqual(self.access.identity.nbytes, 134217728)

    def test_read_requires_current_committed_version_and_cannot_widen_its_lock(self):
        committed = self.committed()
        lease = self.access.acquire(committed.block, 'r', committed=committed)
        self.assertEqual((lease.byte_offset, lease.nbytes), (committed.byte_offset, BLOCK_BYTES))
        self.access.complete_read(lease, 'release1')
        self.assertFalse(self.access.leases)

    def test_changed_content_version_does_not_admit_native_or_drop_the_remote_lease(self):
        committed = self.committed()
        self.port.corrupt = lambda lease: replace(lease, content_version=2)
        with self.assertRaisesRegex(ValueError, 'version changed'):
            self.access.acquire(committed.block, 'r', committed=committed)
        self.assertTrue(self.access.unknown and self.access.leases)
        with self.assertRaisesRegex(RuntimeError, 'obligations'):
            self.access.close()

    def test_old_client_session_cannot_admit_a_range(self):
        block = self.access.allocate()
        self.port.corrupt = lambda lease: replace(lease, pool=replace(lease.pool, client_session='old'))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.access.acquire(block, 'w')
        self.assertEqual(len(self.access.leases), 1)
        self.assertTrue(self.access.unknown)

    def test_lost_commit_ack_queries_the_same_operation_without_repeating_native_or_commit(self):
        self.port.lose_commit = True
        committed = self.committed()
        self.assertEqual(committed.operation_id, 'commit1')
        self.assertEqual([call[0] for call in self.port.calls], ['revalidate', 'commit', 'query'])
        self.assertFalse(self.access.leases or self.access.unknown)

    def test_missing_query_result_retains_unknown_write_and_cannot_unregister(self):
        self.port.lose_commit = self.port.query_missing = True
        with self.assertRaises(PoolOutcomeUnknown):
            self.committed()
        self.assertTrue(self.access.leases and self.access.unknown)
        with self.assertRaisesRegex(RuntimeError, 'obligations'):
            self.access.close()
        self.assertNotIn(('unregister',), self.port.calls)

    def test_generation_change_after_native_completion_prevents_control_commit(self):
        lease = self.access.acquire(self.access.allocate(), 'w')
        self.port.corrupt = lambda item: item
        with self.assertRaisesRegex(ValueError, 'generation changed'):
            self.access.complete_write(lease, 'content', 'write1')
        self.assertFalse(any(c[0] == 'commit' for c in self.port.calls))
        self.assertTrue(self.access.unknown)

    def test_wrong_release_ack_is_not_a_stop_or_unlock_proof(self):
        committed = self.committed()
        lease = self.access.acquire(committed.block, 'r', committed=committed)
        self.port.release = lambda current, operation: PoolAck(operation, 'release', self.port.identity,
                                                              current.lease_id, 99, True)
        with self.assertRaisesRegex(ValueError, 'lease generation'):
            self.access.complete_read(lease, 'release1')
        self.assertIn(lease.lease_id, self.access.leases)

    def test_child_mapping_needs_native_drain_and_actual_original_process_exit(self):
        self.access.attach_child(1234, 77)
        with self.assertRaises(RuntimeError):
            self.access.detach_stopped_child(1234, 77, owner_drained=True, process_gone=False)
        with self.assertRaises(RuntimeError):
            self.access.close()
        self.access.detach_stopped_child(1234, 77, owner_drained=True, process_gone=True)
        self.access.close()
        self.assertTrue(self.access.closed)

    def test_held_derived_view_blocks_unmap_ack_and_retry_preserves_the_closing_client(self):
        held = memoryview(self.port.buffer)
        with self.assertRaises(BufferError):
            self.access.close()
        self.assertTrue(self.access.unregistering)
        self.assertFalse(self.access.closed)
        self.assertFalse(any(c[0] == 'unmapped' for c in self.port.calls))
        held.release()
        self.access.close()
        self.assertTrue(self.access.closed)

    def test_bool_or_wrong_generation_cannot_masquerade_as_valid_control_proof(self):
        with self.assertRaises(ValueError):
            replace(self.port.identity, access_generation=True)
        ack = PoolAck('op', 'release', self.port.identity, applied=1)
        with self.assertRaises(ValueError):
            ack.validate(self.port.identity, 'release', 'op')

    def test_control_boundary_imports_no_sdk_tensor_runtime_or_device_module(self):
        code = '''import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','torch_npu','pypto','simpler','uflow','acl'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
import pypto_serving.serving.native_ddr.managed_pool
'''
        subprocess.run([sys.executable, '-c', code], check=True, timeout=15)
