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
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from pypto_serving.serving.native_ddr.managed_pool import (
    BLOCK_BYTES, GUARD_BYTES, CommittedRange, ManagedPoolAccess, PoolAck, PoolBlock,
    PoolIdentity, PoolLease, PoolOutcomeUnknown,
)


class FakePort:
    """Test-only control responses; anonymous host mmap is not a managed pool."""
    def __init__(self):
        self.identity = PoolIdentity('pool', 'epoch', 'provider', 'client', 134217728, 1, 2,
                                     'resource', 'placement', 'binding', 1, 'simpler-owner', 1, 1, 1)
        self.buffer = mmap.mmap(-1, self.identity.nbytes)
        self.calls, self.results = [], {}
        self.committed_ranges = {}
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
        previous = self.committed_ranges.get(block.block_id)
        if mode == 'r' and previous is None:
            raise ValueError('fake range has no committed content')
        version = (previous.content_version if mode == 'r' else previous.content_version + 1) if previous else 1
        lease = PoolLease(self.identity, block, f'lease{self.serial}', 1,
                          block.byte_offset + byte_offset, nbytes, mode,
                          'Writing' if mode == 'w' else 'Committed', version,
                          self.identity.client_session, f'fake-capability{self.serial}',
                          '' if mode == 'w' else previous.content_id)
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
        self.committed_ranges[lease.block.block_id] = result
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


class ManagedNativeOrderingTests(unittest.TestCase):
    """Fake native handle proves call order/retention only, never DMA or KV bytes."""
    def setUp(self):
        from types import SimpleNamespace
        from pypto_serving.model.qwen.native_ddr_transfer import ManagedDdrSlots
        from pypto_serving.serving.native_ddr.types import (
            DdrEntryRef, DdrTransferRequest, Operation, PrefixKey, QwenKvLayout,
        )
        self.port = FakePort()
        self.access = ManagedPoolAccess(self.port)
        self.slots = ManagedDdrSlots.__new__(ManagedDdrSlots)
        self.slots.capacity = 5
        self.slots.access = self.access
        self.slots._blocks, self.slots._offsets, self.slots.leases = {}, {}, {}
        self.slots.generation, self.slots.busy = 0, None
        self.slots.page_ids = object()
        self.slots._view = lambda offset: object()  # explicit metadata fixture, no tensor claim
        self.key = PrefixKey(b'm' * 32, b'p' * 32)
        self.request = DdrTransferRequest('job', 'worker', Operation.STORE, (self.key,),
            (DdrEntryRef('worker', 'entry', 1),), 'destination-page-lease', (0,), 1, 1)
        self.backend = SimpleNamespace(slots=self.slots, worker=object(), keys=object(), values=object(),
            layout=QwenKvLayout(40, 16, 128, 8, 128), completion_filter=None, pending=None,
            record=lambda *args, **kwargs: None)

    def tearDown(self):
        self.access.buffer.release()
        self.port.buffer.close()

    def run_fixture(self, request, backing):
        from types import SimpleNamespace
        from pypto_serving.model.qwen.native_ddr_transfer import NativeDdrRun
        self.slots.begin_native(request, backing)
        self.slots.busy = backing
        self.native_calls = []
        def result(**kwargs):
            self.native_calls.append('result-and-cleanup')
        native = SimpleNamespace(done=True, result=result)
        run = NativeDdrRun(self.backend, request, backing, native)
        self.backend.pending = run
        return run

    def test_store_control_commit_follows_fake_native_result_and_preserves_its_exact_receipt(self):
        backing = self.slots.reserve(0, BLOCK_BYTES)
        run = self.run_fixture(self.request, backing)
        self.assertIsNone(backing.committed)
        self.assertFalse(self.port.calls or self.native_calls)
        result = run.result()
        self.assertEqual(self.native_calls, ['result-and-cleanup'])
        self.assertEqual([c[0] for c in self.port.calls], ['revalidate', 'commit'])
        self.assertEqual(result.pool_receipt.content, backing.committed)
        self.assertEqual(result.pool_receipt.page_lease_id, self.request.page_lease_id)
        self.assertIsNone(run._keepalive)
        self.assertIsNone(self.slots.busy)

    def test_commit_ack_unknown_keeps_the_real_handle_slot_view_and_pool_lease_referenced(self):
        self.port.lose_commit = self.port.query_missing = True
        backing = self.slots.reserve(0, BLOCK_BYTES)
        run = self.run_fixture(self.request, backing)
        with self.assertRaises(PoolOutcomeUnknown):
            run.result()
        self.assertIsNotNone(run._keepalive)
        self.assertIs(self.backend.pending, run)
        self.assertIs(self.slots.busy, backing)
        self.assertTrue(self.access.leases and self.access.unknown)

    def test_imported_committed_entry_load_needs_no_local_store_flag_and_reacquires_a_current_reader(self):
        from pypto_serving.serving.native_ddr.managed_pool import prefix_content_id
        from pypto_serving.serving.native_ddr.types import Operation
        block = self.port.allocate_range(BLOCK_BYTES + GUARD_BYTES, 65536)
        imported = CommittedRange('pool', 'epoch', 'provider', block, block.byte_offset + GUARD_BYTES,
            BLOCK_BYTES, 7, prefix_content_id(self.key), 'external-commit', 'external-writer-lease')
        self.port.committed_ranges[block.block_id] = imported
        backing = self.slots.import_committed(0, imported, content_id=prefix_content_id(self.key))
        self.assertIs(backing.committed, imported)
        self.assertFalse(hasattr(backing, 'native_complete'))
        self.assertFalse(self.access.leases)
        request = replace(self.request, operation=Operation.LOAD, source_step_id=None)
        run = self.run_fixture(request, backing)
        self.assertEqual(backing.active_lease.content_version, 7)
        done = run.result()
        self.assertEqual(done.pool_receipt.content, imported)
        self.assertEqual(done.pool_receipt.release_ack.lease_id, done.pool_receipt.lease_id)
        self.assertFalse(self.access.leases)

    def test_explicit_pregrant_import_rejection_does_not_leave_a_hidden_logical_slot(self):
        block = self.port.allocate_range(BLOCK_BYTES + GUARD_BYTES, 65536)
        uncommitted = CommittedRange('pool', 'epoch', 'provider', block, block.byte_offset + GUARD_BYTES,
            BLOCK_BYTES, 1, 'content', 'external-commit', 'external-writer')
        with self.assertRaisesRegex(ValueError, 'no committed content'):
            self.slots.import_committed(0, uncommitted, content_id='content')
        self.assertFalse(self.access.leases or self.access.unknown or self.slots.leases)

    def test_receipt_filter_exception_does_not_repeat_successful_pool_commit_on_explicit_retry(self):
        backing = self.slots.reserve(0, BLOCK_BYTES)
        run = self.run_fixture(self.request, backing)
        def reject(*_):
            raise RuntimeError('injected receipt observer error')
        self.backend.completion_filter = reject
        with self.assertRaisesRegex(RuntimeError, 'observer error'):
            run.result()
        self.assertIsNotNone(run._keepalive)
        self.assertIsNotNone(run._pool_receipt)
        self.backend.completion_filter = None
        done = run.result()
        self.assertEqual(done.pool_receipt, run._pool_receipt)
        self.assertEqual(sum(c[0] == 'commit' for c in self.port.calls), 1)
        self.assertEqual(self.native_calls, ['result-and-cleanup'])
        self.assertIsNone(run._keepalive)

    def test_control_unknown_can_drain_device_without_publishing_or_unmapping_the_pool(self):
        from pypto_serving.model.qwen.native_ddr_transfer import QwenNativeTransferBackend
        self.port.lose_commit = self.port.query_missing = True
        backing = self.slots.reserve(0, BLOCK_BYTES)
        run = self.run_fixture(self.request, backing)
        with self.assertRaises(PoolOutcomeUnknown):
            run.result()
        self.backend.ambiguous = None
        self.backend.closed = False
        self.backend._pins = []
        self.backend._release_pins = lambda: QwenNativeTransferBackend._release_pins(self.backend)
        QwenNativeTransferBackend.drain_native_domain(self.backend)
        self.assertEqual(self.native_calls, ['result-and-cleanup'])
        # This call represents the runner's separate, successful owner close
        # plus kernel child-exit proof. It does not simulate that physical work.
        QwenNativeTransferBackend.owner_stopped_preserving_pool(self.backend)
        self.assertIsNone(self.backend.worker)
        self.assertIsNone(self.backend.keys)
        self.assertIsNone(run.native)
        self.assertEqual(run._keepalive, (backing, self.request))
        self.assertTrue(self.access.unknown and self.access.leases)
        self.assertIsNone(backing.committed)
        with self.assertRaisesRegex(RuntimeError, 'unresolved pool'):
            run.result()
        with self.assertRaisesRegex(RuntimeError, 'obligations'):
            self.access.close()
        self.assertFalse(any(call[0] == 'unmapped' for call in self.port.calls))

    def test_native_result_failure_cannot_be_converted_to_a_physical_stop(self):
        backing = self.slots.reserve(0, BLOCK_BYTES)
        run = self.run_fixture(self.request, backing)
        def fail(**_):
            raise TimeoutError('native cleanup unknown')
        run.native.result = fail
        with self.assertRaises(TimeoutError):
            run.drain_native()
        with self.assertRaisesRegex(RuntimeError, 'no completion'):
            run.release_stopped_owner()
        self.assertIsNotNone(run._keepalive)
        self.assertTrue(self.access.leases and self.access.unknown)


class ManagedCompatibilityTests(unittest.TestCase):
    def test_full_asset_capture_and_postload_metadata_recheck_are_distinct(self):
        from pypto_serving.serving.native_ddr.compatibility import capture_full_assets, verify_loaded_assets
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'shard.safetensors'}}))
            raw = struct.pack('<Q', 2) + b'{}' + b'fake-weight-payload'
            (root / 'shard.safetensors').write_bytes(raw)
            assets = capture_full_assets(root)
            self.assertEqual(assets['files']['shard.safetensors']['sha256'], hashlib.sha256(raw).hexdigest())
            verify_loaded_assets(root, assets)
            (root / 'shard.safetensors').write_bytes(raw + b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_loaded_assets(root, assets)

    def test_data_compatibility_is_independent_of_epoch_and_local_hbm_page_count(self):
        from pypto_serving.serving.native_ddr.compatibility import data_compatibility_digest
        descriptor = {'worker_epoch': 'owner1', 'model': {'rope_theta': 1000000},
                      'assets': {'files': {'asset': {'evidence': 'full-asset-content-at-startup', 'sha256': 'a' * 64}}},
                      'layout': {'num_layers': 40, 'num_pages': 16, 'page_size': 128, 'dtype': 'bfloat16'},
                      'tokenizer_type': 'test-tokenizer', 'adapter': None, 'position': 'absolute',
                      'tensor_parallel': 1, 'deployment': {'test-only': 'no-runtime-import'}}
        expected = data_compatibility_digest(descriptor)
        descriptor['worker_epoch'] = 'owner2'
        descriptor['layout']['num_pages'] = 618
        self.assertEqual(data_compatibility_digest(descriptor), expected)
        descriptor['model']['rope_theta'] = 2000000
        self.assertNotEqual(data_compatibility_digest(descriptor), expected)
        descriptor['assets']['files']['asset']['evidence'] = 'header-only'
        with self.assertRaisesRegex(ValueError, 'complete'):
            data_compatibility_digest(descriptor)

    def test_sdk_normalization_rejects_unknown_record_fields_and_bool_generations(self):
        from pypto_serving.serving.native_ddr.managed_pool_sdk import identity
        from dataclasses import asdict
        port = FakePort()
        try:
            data = asdict(port.identity)
            self.assertEqual(identity(data), port.identity)
            with self.assertRaises(ValueError):
                identity({**data, 'private_address': 12345})
            with self.assertRaises(ValueError):
                identity({**data, 'simpler_generation': True})
        finally:
            port.buffer.close()

    def test_unparseable_sdk_grant_is_retained_and_blocks_close_and_new_native_admission(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        from pypto_serving.serving.native_ddr.managed_pool_sdk import SdkManagedPoolPort
        fake = FakePort()
        raw = {'unexpected-grant-format': 'must-retain'}
        client = SimpleNamespace(view=SimpleNamespace(descriptor=asdict(fake.identity), buffer=fake.buffer),
            acquire_range=lambda *args, **kwargs: raw, report_unknown=lambda *_: None)
        port = SdkManagedPoolPort(client)
        access = ManagedPoolAccess(port)
        block = fake.allocate_range(BLOCK_BYTES + GUARD_BYTES, 65536)
        try:
            with self.assertRaises(PoolOutcomeUnknown):
                access.acquire(block, 'w')
            self.assertIs(port._unparsed_grants[0], raw)
            self.assertTrue(access.unknown)
            with self.assertRaisesRegex(RuntimeError, 'obligations'):
                access.acquire(block, 'w')
            with self.assertRaisesRegex(RuntimeError, 'retained SDK'):
                port.close_view()
            with self.assertRaisesRegex(RuntimeError, 'obligations'):
                access.close()
        finally:
            access.buffer.release()
            fake.buffer.close()

    def test_acquire_timeout_keeps_the_exact_unresolved_sdk_request_and_blocks_unmap(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        from pypto_serving.serving.native_ddr.managed_pool_sdk import SdkManagedPoolPort
        fake = FakePort()
        def lose_reply(*_, **__):
            raise TimeoutError('grant reply lost')
        client = SimpleNamespace(view=SimpleNamespace(descriptor=asdict(fake.identity), buffer=fake.buffer),
                                  acquire_range=lose_reply, report_unknown=lambda *_: None)
        port = SdkManagedPoolPort(client)
        access = ManagedPoolAccess(port)
        block = fake.allocate_range(BLOCK_BYTES + GUARD_BYTES, 65536)
        try:
            with self.assertRaises(PoolOutcomeUnknown):
                access.acquire(block, 'w')
            self.assertEqual(port._unresolved_acquires, [(asdict(block), 'w', 1, GUARD_BYTES, BLOCK_BYTES)])
            self.assertTrue(access.unknown)
            with self.assertRaisesRegex(RuntimeError, 'retained SDK'):
                port.close_view()
        finally:
            access.buffer.release()
            fake.buffer.close()

    def test_process_map_proof_requires_the_actual_full_shared_rw_backing(self):
        from pypto_serving.serving.native_ddr.managed_process import (
            has_pool_mapping, process_birth, shared_mapping_size,
        )
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '123').mkdir()
            tail = ['S'] + ['0'] * 18 + ['77']
            (root / '123/stat').write_text('123 (name with ) and spaces) ' + ' '.join(tail))
            self.assertEqual(process_birth(123, proc=root), 77)
            resource = SimpleNamespace(device=1, inode=2, nbytes=134217728)
            (root / '123/maps').write_text('10000000-18000000 rw-s 00000000 00:01 2 /fake-test-only\n')
            with patch('os.major', return_value=0, create=True), patch('os.minor', return_value=1, create=True):
                self.assertTrue(has_pool_mapping(123, resource, proc=root))
            self.assertEqual(shared_mapping_size(0x10000000, 4, maps=root / '123/maps'), 134217728)
            (root / '123/maps').write_text('10000000-18000000 rw-p 00000000 00:01 2 /fake-test-only\n')
            with patch('os.major', return_value=0, create=True), patch('os.minor', return_value=1, create=True):
                self.assertFalse(has_pool_mapping(123, resource, proc=root))
            with self.assertRaisesRegex(RuntimeError, 'not a shared'):
                shared_mapping_size(0x10000000, 4, maps=root / '123/maps')


class ManagedPrefixImportTests(unittest.TestCase):
    """Typed metadata only; no direct ready-table seeding or model/DMA claim."""
    def setUp(self):
        from types import SimpleNamespace
        from pypto_serving.serving.native_ddr.coordinator import DdrPrefixCoordinator
        from pypto_serving.serving.native_ddr.types import PrefixKey
        from pypto_serving.serving.native_ddr.managed_pool import prefix_content_id
        self.port = FakePort()
        self.hashes = [bytes([n]) * 32 for n in (1, 2, 3)]
        self.manager = SimpleNamespace(has_groups=False, block_size=128, enable_prefix_cache=True,
            compute_block_hashes=lambda tokens: self.hashes[:len(tokens) // 128])
        self.coord = DdrPrefixCoordinator(self.manager, namespace_digest=b'n' * 32, worker_epoch='worker2',
                                          max_entries=5, pool_identity=self.port.identity)
        self.proofs = []
        for n, digest in enumerate(self.hashes):
            allocation = self.port.allocate_range(BLOCK_BYTES + GUARD_BYTES, 65536)
            self.proofs.append(CommittedRange('pool', 'epoch', 'provider', allocation,
                allocation.byte_offset + GUARD_BYTES, BLOCK_BYTES, n + 1,
                prefix_content_id(PrefixKey(self.coord.namespace, digest)), f'writer-op{n}', f'writer{n}'))

    def tearDown(self):
        self.port.buffer.close()

    def acknowledge(self, command, *, pool=None):
        from pypto_serving.serving.native_ddr.types import DdrControlEvent
        ack = PoolAck('reader-release', 'release', pool or self.port.identity, 'current-reader', 1, True)
        self.coord.on_control_event(DdrControlEvent(command.command_id, command.worker_epoch, 'import',
            command.entry_ref, delivery_sequence=command.delivery_sequence, imported=command.imported, pool_ack=ack))

    def test_complete_prefix_preserves_final_token_and_publishes_only_after_each_explicit_ack(self):
        with self.assertRaisesRegex(ValueError, 'final-token'):
            self.coord.prepare_import_prefix(list(range(128)), self.proofs[:1])
        with self.assertRaisesRegex(ValueError, 'final-token'):
            self.coord.prepare_import_prefix(list(range(256)), self.proofs[:2])
        commands = self.coord.prepare_import_prefix(list(range(257)), self.proofs[:2])
        self.assertEqual(len(commands), 2)
        self.assertFalse(self.coord.ready)
        first = self.coord.next_control()
        self.assertEqual(first.imported, self.proofs[0])
        self.acknowledge(first)
        self.assertEqual(len(self.coord.ready), 1)
        second = self.coord.next_control()
        self.assertEqual(second.imported, self.proofs[1])
        self.assertGreater(second.entry_ref.slot_generation, first.entry_ref.slot_generation)
        self.acknowledge(second)
        self.assertEqual(list(self.coord.managed_ready.values()), self.proofs[:2])
        self.assertFalse(self.coord.pending_imports or self.coord.controls)

    def test_import_cannot_skip_token0_or_a_middle_gap(self):
        for values in (self.proofs[1:2], [self.proofs[0], self.proofs[2]]):
            with self.assertRaisesRegex(ValueError, 'compatibility/prefix'):
                self.coord.prepare_import_prefix(list(range(385)), values)
            self.assertFalse(self.coord.ready or self.coord.controls)

    def test_previous_client_incarnation_ack_never_publishes_or_retires_the_import(self):
        self.coord.prepare_import_prefix(list(range(129)), self.proofs[:1])
        command = self.coord.next_control()
        with self.assertRaisesRegex(RuntimeError, 'current UFlow'):
            self.acknowledge(command, pool=replace(self.port.identity, client_session='old-client'))
        self.assertTrue(self.coord.poisoned)
        self.assertFalse(self.coord.ready)
        self.assertIn(command.command_id, self.coord.pending_imports)
        self.assertEqual(self.coord.control_delivery.acknowledged, 0)
