# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Portable owned-process lifecycle tests; no real namespace, broker or NPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

from examples.run03.host_exchange_lifecycle import (
    OuterProtection, close_stage_on_exit, require_revoked_acquire, supervise_owned,
)
from pypto_serving.serving.native_ddr.shared_exchange import HeldResources, HostStagingOwner, StagedBlock


class OwnedProcessTests(unittest.TestCase):
    def child(self, directory, *, ignore_stop=False):
        code = """
from pathlib import Path
import sys
root=Path(sys.argv[1]); (root/'ready').write_text('ready')
command=sys.stdin.buffer.readline()
assert command==b'STOP\\n',command
(root/'stop-observed').write_text('normal stop received')
if sys.argv[2]=='ignore':
 assert sys.stdin.buffer.readline()==b'EXIT\\n'
(root/'drained').write_text('owned cleanup finished')
"""
        proc = subprocess.Popen([sys.executable, '-u', '-c', code, str(directory),
                                 'ignore' if ignore_stop else 'normal'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=os.name == 'posix')
        deadline = time.monotonic()+5
        while not (directory/'ready').exists():
            if proc.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('lifecycle fixture failed startup')
            time.sleep(.01)
        return proc

    def test_timeout_requests_normal_cleanup_then_reaps_child_not_auto_stopped_by_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proc = self.child(root)
            try:
                self.assertIsNone(proc.poll())
                report = supervise_owned(proc, timeout=.05, cleanup_grace=5)
                self.assertTrue(report['timed_out'])
                self.assertTrue(report['stop_requested'])
                self.assertTrue(report['reaped'])
                self.assertEqual(proc.returncode, 0)
                self.assertFalse(report['passed'], 'timed-out experiment is not a pass')
                self.assertTrue((root/'stop-observed').exists())
                self.assertTrue((root/'drained').exists())
            finally:
                proc.stdin.close()
                proc.stdout.close()
                proc.stderr.close()

    def test_uncooperative_child_is_reported_alive_without_kill_or_fake_reap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            proc = self.child(root, ignore_stop=True)
            try:
                report = supervise_owned(proc, timeout=.05, cleanup_grace=.05)
                self.assertFalse(report['reaped'])
                self.assertFalse(report['passed'])
                self.assertEqual(report['unresolved_child_pid'], proc.pid)
                self.assertIn('remaining_owned', report)
                self.assertIsNone(proc.poll())
                self.assertTrue((root/'stop-observed').exists())
                self.assertFalse((root/'drained').exists())
            finally:
                # The test owns this fixture and explicitly releases its second
                # cooperative gate. Production never silently kills a survivor.
                proc.stdin.write(b'EXIT\n')
                proc.stdin.flush()
                self.assertEqual(proc.wait(timeout=5), 0)
                proc.stdin.close()
                proc.stdout.close()
                proc.stderr.close()

    def test_normal_child_exit_requires_no_stop(self):
        proc = subprocess.Popen([sys.executable, '-c', 'pass'], stdin=subprocess.PIPE)
        try:
            report = supervise_owned(proc, timeout=5, cleanup_grace=1)
            self.assertTrue(report['passed'])
            self.assertFalse(report['stop_requested'])
            self.assertTrue(report['reaped'])
        finally:
            proc.stdin.close()


class OuterProtectionTests(unittest.TestCase):
    def fixture(self, folder, report):
        root = Path(folder)
        (root/'workspace').mkdir()
        (root/'namespace').write_text('namespace fixture')
        (root/'mountinfo').write_bytes(b'original mounts\n')
        return OuterProtection(report, root/'workspace', root/'mountinfo', root/'namespace')

    def test_timeout_still_samples_final_real_file_identities_and_mount_content(self):
        with tempfile.TemporaryDirectory() as folder:
            report = {}
            guard = self.fixture(folder, report)
            with self.assertRaisesRegex(TimeoutError, 'primary timeout'):
                with guard:
                    raise TimeoutError('primary timeout')
            proof = report['outer_protection']
            self.assertTrue(proof['unchanged'])
            self.assertEqual(proof['before_inode'], proof['after_inode'])
            self.assertEqual(proof['before_namespace'], proof['after_namespace'])
            self.assertEqual(guard.before_mounts, guard.after_mounts)

    def test_changed_mount_content_is_recorded_without_masking_primary_error(self):
        with tempfile.TemporaryDirectory() as folder:
            report = {}
            with self.assertRaisesRegex(ValueError, 'original body failure'):
                with self.fixture(folder, report):
                    (Path(folder)/'mountinfo').write_bytes(b'changed mounts\n')
                    raise ValueError('original body failure')
            self.assertFalse(report['outer_protection']['unchanged'])
            self.assertFalse(report['outer_protection']['mountinfo_unchanged'])
            self.assertIn('after_namespace', report['outer_protection'])

    def test_changed_mount_content_on_success_is_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            report = {}
            with self.assertRaises(RuntimeError):
                with self.fixture(folder, report):
                    (Path(folder)/'mountinfo').write_bytes(b'changed mounts\n')
            self.assertFalse(report['outer_protection']['unchanged'])


class View:
    def __init__(self, fail=False):
        self.fail, self.released = fail, False
        self._view = SimpleNamespace(view_id=17, lease_id=23)

    def release(self):
        if self.fail:
            raise OSError('release RPC outcome unknown')
        self.released = True


class ConsumerCleanupTests(unittest.TestCase):
    def stage(self, fail=False):
        owner = HostStagingOwner()
        resources = HeldResources(owner.reserve())
        view = View(fail)
        resources.views.append(view)
        return StagedBlock(resources, None, 'host fixture'), owner, view

    def fail_consumer(self, stage, report):
        try:
            raise AssertionError('original post-acquire byte assertion')
        finally:
            close_stage_on_exit(stage, report)

    def test_post_acquire_assertion_explicitly_releases_view_and_slot(self):
        stage, owner, view = self.stage()
        report = {'passed': False, 'primary_error': 'original post-acquire byte assertion'}
        with self.assertRaisesRegex(AssertionError, 'original post-acquire'):
            self.fail_consumer(stage, report)
        self.assertTrue(view.released)
        self.assertFalse(owner.busy)
        self.assertTrue(report['stage_exit_cleanup']['confirmed'])
        self.assertEqual(report['primary_error'], 'original post-acquire byte assertion')

    def test_unknown_cleanup_preserves_original_error_view_identity_and_busy_slot(self):
        stage, owner, view = self.stage(fail=True)
        report = {'passed': False, 'primary_error': 'original post-acquire byte assertion'}
        try:
            with self.assertRaisesRegex(AssertionError, 'original post-acquire'):
                self.fail_consumer(stage, report)
            proof = report['stage_exit_cleanup']
            self.assertFalse(proof['confirmed'])
            self.assertTrue(owner.busy)
            self.assertIs(stage.resources.views[0], view)
            self.assertEqual(proof['view_identities'], [{'view_id': 17, 'lease_id': 23}])
            self.assertIn('unresolved_stage', report)
            self.assertEqual(report['primary_error'], 'original post-acquire byte assertion')
        finally:
            view.fail = False
            stage.close()


class BrokerError(RuntimeError):
    pass


class SdkFixture:
    """Error-origin fixture, not a fake broker result presented as Linux evidence."""
    def __init__(self, mode):
        self.mode = mode

    def _exchange(self, op):
        received_fds, mapping = [], None
        if self.mode == 'reject':
            response = {'status': 'error', 'detail': 'backing is not accepting imports'}
            raise BrokerError(response['detail'])
        if self.mode == 'protocol':
            response = {'status': 'ok'}
            raise BrokerError('broker did not commit')
        if self.mode == 'same-text-without-response':
            response = None
            raise BrokerError('backing is not accepting imports')
        if self.mode == 'timeout':
            response = None
            raise TimeoutError('broker timeout')
        if self.mode == 'connection':
            response = None
            raise ConnectionRefusedError('broker not reachable')
        if self.mode == 'other-rejection':
            response = {'status': 'error', 'detail': 'authority quarantined; state unknown'}
            raise BrokerError(response['detail'])
        return received_fds, mapping

    def acquire(self, source):
        return self._exchange('AcquireExternalRead')


class RevocationClassificationTests(unittest.TestCase):
    def run_case(self, mode):
        report = {}
        adapter = SdkFixture(mode)
        require_revoked_acquire(adapter.acquire, None, exchange_code=SdkFixture._exchange.__code__,
                                error_type=BrokerError, report=report, unexpected_views=[])
        return report

    def test_only_actual_non_ok_revocation_response_counts_as_proof(self):
        report = self.run_case('reject')
        self.assertEqual(report['new_acquire_after_revoke']['outcome'], 'explicit_broker_revocation_rejection')

    def test_transport_protocol_same_text_and_other_rejections_are_failures(self):
        for mode, kind in [('protocol', BrokerError), ('same-text-without-response', BrokerError),
                           ('timeout', TimeoutError), ('connection', ConnectionRefusedError),
                           ('other-rejection', BrokerError)]:
            with self.subTest(mode=mode), self.assertRaises(kind):
                self.run_case(mode)

    def test_unexpected_successful_view_is_kept_for_explicit_failure_cleanup(self):
        view, pending, report = View(), [], {}
        with self.assertRaises(AssertionError):
            require_revoked_acquire(lambda source: view, None, exchange_code=SdkFixture._exchange.__code__,
                                    error_type=BrokerError, report=report, unexpected_views=pending)
        self.assertIs(pending[0], view)
        self.assertFalse(view.released)
        self.assertEqual(report['new_acquire_after_revoke']['outcome'], 'unexpected_live_view')
        view.release()


if __name__ == '__main__':
    unittest.main()
