# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stdlib host fault fixtures; real SDK/memfd peers are a separate Linux test."""
from dataclasses import replace
import hashlib
import mmap
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from examples.run03.host_record_probe import fixture_compatibility, fixture_destination, fixture_record
from pypto_serving.serving.native_ddr.shared_compatibility import BLOCK_BYTES
from pypto_serving.serving.native_ddr.shared_exchange import (
    AcquireRejected, E10ReadView, ExchangeFailure, GUARD_BYTES, HostStagingOwner, StageCancelled, stage_block,
)

PAYLOAD = bytes(range(256)) * (BLOCK_BYTES // 256)
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class FakeView:
    def __init__(self, state):
        self.state, self.released = state, False
        self.fail_release = False
        state.views.append(self)

    def verify(self, source):
        if self.state.bad_identity:
            raise ValueError("wrong source response")

    def read(self):
        if self.released:
            raise RuntimeError("closed")
        if self.state.on_read:
            self.state.on_read()
        return self.state.payload

    def retain(self):
        if self.state.fail_retain:
            raise OSError("retain final confirmation lost")
        return FakeView(self.state)

    def release(self):
        if self.fail_release:
            raise OSError("release acknowledgement lost")
        self.released = True


class FakeAdapter:
    def __init__(self):
        self.payload, self.views = PAYLOAD, []
        self.bad_identity = self.fail_retain = False
        self.on_read = None
        self.acquire_error = None
        self.calls = 0

    def acquire(self, source):
        self.calls += 1
        if self.acquire_error:
            raise self.acquire_error
        return FakeView(self)


class ExchangeTests(unittest.TestCase):
    def setUp(self):
        self.record = replace(fixture_record(), payload_sha256=DIGEST)
        self.destination = fixture_destination()
        self.current_destination = self.destination
        self.current_source = self.record.source
        self.owner = HostStagingOwner()
        self.adapter = FakeAdapter()

    def stage(self, **overrides):
        args = dict(compatibility=fixture_compatibility(), current_source=lambda: self.current_source,
                    destination=self.destination, current_destination=lambda: self.current_destination,
                    expected_chain_hash=self.record.chain_hash, expected_token_start=self.record.token_start,
                    expected_payload_sha256=DIGEST, staging=self.owner, adapter=self.adapter)
        args.update(overrides)
        return stage_block(self.record, **args)

    def assert_prewrite_failure(self, **overrides):
        before = bytes(self.owner._storage)
        with self.assertRaises(ExchangeFailure) as caught:
            self.stage(**overrides)
        self.assertEqual(bytes(self.owner._storage), before)
        self.assertTrue(self.owner.guards_ok())
        return caught.exception

    def test_full_copy_is_explicitly_non_ready_and_owns_retained_view(self):
        stage = self.stage()
        self.assertFalse(stage.native_ready)
        self.assertEqual(stage.read(), PAYLOAD)
        self.assertEqual(self.owner.capacity_bytes, BLOCK_BYTES)
        self.assertEqual(self.owner.verified_copied_bytes, BLOCK_BYTES)
        self.assertTrue(self.adapter.views[0].released)
        self.assertFalse(self.adapter.views[1].released)
        self.assertTrue(self.owner.busy)
        stage.close()
        self.assertTrue(all(view.released for view in self.adapter.views))
        self.assertFalse(self.owner.busy)
        with self.assertRaises(RuntimeError):
            stage.read()

    def test_busy_slot_rejects_without_another_acquire(self):
        stage = self.stage()
        calls = self.adapter.calls
        with self.assertRaises(RuntimeError):
            self.stage()
        self.assertEqual(self.adapter.calls, calls)
        stage.close()

    def test_generation_changes_and_old_lease_cannot_read_or_release_new_slot(self):
        stage = self.stage()
        old = stage.lease
        stage.close()
        new = self.stage()
        self.assertNotEqual(new.lease.generation, old.generation)
        with self.assertRaises(RuntimeError):
            self.owner.read(old)
        with self.assertRaises(RuntimeError):
            self.owner._release(old)
        new.close()

    def test_bad_digest_is_prewrite_and_releases_real_fixture_obligations(self):
        self.adapter.payload = b"x" * BLOCK_BYTES
        self.assert_prewrite_failure()
        self.assertFalse(self.owner.busy)
        self.assertTrue(all(view.released for view in self.adapter.views))

    def test_truncated_read_is_prewrite(self):
        self.adapter.payload = PAYLOAD[:-1]
        self.assert_prewrite_failure()
        self.assertFalse(self.owner.busy)

    def test_bad_response_identity_is_prewrite(self):
        self.adapter.bad_identity = True
        self.assert_prewrite_failure()
        self.assertTrue(self.adapter.views[0].released)

    def test_destination_invalidated_during_preparation_is_prewrite(self):
        self.adapter.on_read = lambda: setattr(self, "current_destination", replace(self.destination, page_lease_id="new"))
        self.assert_prewrite_failure()
        self.assertFalse(self.owner.busy)

    def test_source_invalidated_during_preparation_is_prewrite(self):
        self.adapter.on_read = lambda: setattr(self, "current_source", replace(self.record.source, entry_generation=999))
        self.assert_prewrite_failure()

    def test_reused_staging_generation_cannot_be_written_or_released_by_old_prepare(self):
        new_lease = []
        def reuse():
            self.owner._release(self.owner._active)
            new_lease.append(self.owner.reserve())
        self.adapter.on_read = reuse
        failure = self.assert_prewrite_failure()
        self.assertFalse(failure.resources.closed)
        self.assertTrue(self.owner.busy)
        self.assertIs(self.owner._active, new_lease[0])
        self.owner._release(new_lease[0])

    def test_old_destination_or_source_refused_before_reservation(self):
        for field, value in (("worker_epoch", "new-worker"), ("arena_id", "new-arena"),
                             ("cache_epoch", "new-cache"), ("page_lease_id", "new-lease")):
            self.current_destination = replace(self.destination, **{field: value})
            with self.assertRaises(ValueError):
                self.stage()
            self.assertFalse(self.owner.busy)
        self.current_destination = self.destination
        self.current_source = replace(self.record.source, entry_generation=999)
        with self.assertRaises(ValueError):
            self.stage()
        self.assertEqual(self.adapter.calls, 0)

    def test_cancel_before_copy_keeps_target_and_guards(self):
        failure = self.assert_prewrite_failure(cancelled=lambda: True)
        self.assertIsInstance(failure.cause, StageCancelled)
        self.assertFalse(self.owner.busy)

    def test_cancel_after_copy_invalidates_instead_of_publishing_ready(self):
        calls = []
        def cancelled():
            calls.append(True)
            return len(calls) > 1
        with self.assertRaises(ExchangeFailure) as caught:
            self.stage(cancelled=cancelled)
        self.assertIsInstance(caught.exception.cause, StageCancelled)
        self.assertFalse(caught.exception.resources.lease.valid_copy)
        self.assertEqual(self.owner.verified_copied_bytes, 0)
        self.assertTrue(self.owner.guards_ok())

    def test_cpu_write_failure_is_partial_invalid_not_unchanged(self):
        def partial(target, source):
            target[:17] = source[:17]
            raise OSError("injected CPU write failure")
        with self.assertRaises(ExchangeFailure) as caught:
            self.stage(copy_into=partial)
        self.assertFalse(caught.exception.resources.lease.valid_copy)
        self.assertTrue(self.owner.partial_write_unknown)
        self.assertEqual(self.owner._storage[GUARD_BYTES:GUARD_BYTES+17], PAYLOAD[:17])
        self.assertTrue(self.owner.guards_ok())
        self.assertFalse(self.owner.busy)

    def test_release_unknown_preserves_slot_and_handle_until_original_retry(self):
        stage = self.stage()
        view = self.adapter.views[-1]
        view.fail_release = True
        with self.assertRaises(OSError):
            stage.close()
        self.assertTrue(self.owner.busy)
        self.assertIs(stage.resources.views[0], view)
        with self.assertRaises(RuntimeError):
            stage.read()
        view.fail_release = False
        stage.close()
        stage.close()
        self.assertFalse(self.owner.busy)

    def test_root_release_unknown_during_prepare_keeps_both_views_and_slot(self):
        self.adapter.on_read = lambda: setattr(self.adapter.views[0], "fail_release", True)
        failure = self.assert_prewrite_failure()
        self.assertEqual(len(failure.resources.views), 2)
        self.assertTrue(self.owner.busy)
        self.adapter.views[0].fail_release = False
        failure.resources.close()
        self.assertFalse(self.owner.busy)

    def test_ambiguous_acquire_and_retain_do_not_fake_clean(self):
        for operation in ("acquire", "retain"):
            self.setUp()
            if operation == "acquire":
                self.adapter.acquire_error = OSError("lost final confirmation")
            else:
                self.adapter.fail_retain = True
            failure = self.assert_prewrite_failure()
            self.assertTrue(failure.resources.unresolved_acquire)
            with self.assertRaises(RuntimeError):
                failure.resources.close()
            self.assertTrue(self.owner.busy)

    def test_explicit_preobligation_rejection_can_release_local_reservation(self):
        self.adapter.acquire_error = AcquireRejected("proven pre-obligation rejection")
        failure = self.assert_prewrite_failure()
        self.assertTrue(failure.resources.closed)
        self.assertFalse(self.owner.busy)

    def test_import_does_not_import_sdk_or_tensor_runtime(self):
        program = """
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','torch_npu','pypto','simpler','uflow','_task_interface'):
   raise AssertionError(name)
sys.meta_path.insert(0,Block())
import pypto_serving.serving.native_ddr.shared_exchange
"""
        subprocess.run([sys.executable, "-c", program], check=True)


class PinnedSdkAdapterTests(unittest.TestCase):
    def test_all_thirteen_returned_fields_range_and_readonly_are_checked(self):
        source = fixture_record().source
        with tempfile.TemporaryFile() as file:
            file.truncate(source.backing.size_bytes)
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                raw = SimpleNamespace(_identity=SimpleNamespace(backing=source.backing), _mapping=mapping,
                                      allowed_offset_bytes=source.offset_bytes, allowed_bytes=BLOCK_BYTES)
                adapter = E10ReadView(raw)
                adapter.verify(source)
                for field in source.backing.__dataclass_fields__:
                    value = getattr(source.backing, field)
                    changed = value + 1 if type(value) is int else value + "-wrong"
                    if field == "purpose":
                        # H1 rejects any non-immutable source before an acquire.
                        with self.assertRaises(ValueError):
                            replace(source.backing, purpose=changed)
                        continue
                    raw._identity.backing = replace(source.backing, **{field: changed})
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        adapter.verify(source)
                raw._identity.backing = source.backing
                raw.allowed_bytes = BLOCK_BYTES - 1
                with self.assertRaises(ValueError):
                    adapter.verify(source)
                raw.allowed_bytes = BLOCK_BYTES
                raw.allowed_offset_bytes = 0
                with self.assertRaises(ValueError):
                    adapter.verify(source)
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as writable:
                raw._mapping, raw.allowed_offset_bytes = writable, source.offset_bytes
                with self.assertRaises(ValueError):
                    E10ReadView(raw).verify(source)


if __name__ == "__main__":
    unittest.main()
