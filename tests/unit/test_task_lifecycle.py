# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real POSIX long-run/failed-close and private SHM reclamation, including retained failures."""
import array
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from pypto_serving.serving.native_ddr.closure_report import classify

SOURCE = Path(__file__).resolve().parents[2]
MODULES = SOURCE / 'pypto_serving/serving/native_ddr'
sys.path.insert(0, str(MODULES))
import task_supervisor as supervisor
import task_lifecycle as lifecycle
import task_resources as resources
PROOFS = []


def wait_file(path, seconds=4):
    end = time.monotonic() + seconds
    while not path.exists():
        if time.monotonic() > end:
            raise TimeoutError(str(path))
        time.sleep(.01)


def actor(mode, root):
    from multiprocessing import resource_tracker
    from multiprocessing.shared_memory import SharedMemory
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == 'early-unregistered':
        import _posixshmem
        name = '/uf-unregistered-' + os.urandom(16).hex()
        fd = _posixshmem.shm_open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        os.ftruncate(fd, 4096)
        (root / 'name.json').write_text(json.dumps({'name': name[1:]}))
        os._exit(4)  # Deliberately before mmap/constructor/ready/registration.
    if mode in ('foreign-attach', 'foreign-refuse'):
        foreign = json.loads((root / 'foreign.json').read_bytes())
        shm = SharedMemory(name=foreign['name'], create=False)
        resource_tracker.unregister(shm._name, 'shared_memory')
        if mode == 'foreign-refuse':
            try:
                lifecycle.exchange('private-created', {'path': '/dev/shm/' + shm.name, 'exclusive_create': True,
                    'original_fd': shm._fd, 'kind': 'UF-provider-or-shared'}, fd=shm._fd)
            except RuntimeError:
                pass
            else:
                raise AssertionError('foreign shared allocation wrongly admitted')
        shm.close(); return
    shm = lifecycle.create_private_shared_memory(4096)
    # The external registry now owns post-death cleanup. Avoid the CPU fixture's
    # name-only resource_tracker racing the intentional inode-drift test.
    resource_tracker.unregister(shm._name, 'shared_memory')
    p = Path('/dev/shm') / shm.name
    s = p.stat()
    (root / 'name.json').write_text(json.dumps({'name': shm.name, 'dev': s.st_dev, 'ino': s.st_ino}))
    shm.buf[:] = b'X' * 4096
    if mode == 'forged-close':
        original = os.environ[lifecycle.ENV]
        context = json.loads(original)
        for key in ('attempt', 'leader_birth', 'monitor_birth'):
            bad = {**context, key: 'wrong' if key == 'attempt' else context[key] + 1}
            os.environ[lifecycle.ENV] = json.dumps(bad)
            try:
                lifecycle.exchange('closing', {'reason': 'forged'})
            except RuntimeError:
                continue
            raise AssertionError('forged close identity accepted')
        os.environ[lifecycle.ENV] = original
    if mode in ('normal-long', 'forged-close'):
        time.sleep(.65); shm.close(); return
    if mode == 'inode-drift':
        p.unlink()
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.write(fd, b'foreign replacement must survive'); os.close(fd)
    if mode == 'uid-drift':
        os.chown(p, 65534, -1)
    if mode == 'broken-notice':
        Path(json.loads(os.environ[lifecycle.ENV])['socket']).unlink()
    if mode == 'foreign-holder':
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as channel:
            channel.connect(str(root / 'holder.sock'))
            channel.sendmsg([b'FD'], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [shm._fd]))])
            assert channel.recv(16) == b'held'
    from pypto_serving.serving.native_ddr.cooperative_close import CooperativePoolClose
    recovery = CooperativePoolClose(root / 'close-channel')
    def blocked_close():
        (root / 'blocked-close.json').write_text(json.dumps({'monotonic_ns': time.monotonic_ns()}))
        while True:
            time.sleep(.05)
    recovery._close = blocked_close  # Controlled fixture; real production retain_failure/notice.
    try:
        raise RuntimeError('known CPU failure before a guarded close that never returns')
    except RuntimeError as error:
        recovery.retain_failure(error, wait_for_request=False)




def holder(root):
    import mmap
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as channel:
        channel.bind(str(root / 'holder.sock')); channel.listen(1)
        conn, _ = channel.accept()
        with conn:
            _, controls, _, _ = conn.recvmsg(16, socket.CMSG_SPACE(4))
            values = array.array('i'); values.frombytes(controls[0][2]); fd = values[0]
            mapping = mmap.mmap(fd, 4096)
            conn.send(b'held')
            (root / 'held.json').write_text(json.dumps(supervisor.observation(os.getpid())))
            while True:
                time.sleep(.1)


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux POSIX/pidfd required')
class TaskLifecycleTests(unittest.TestCase):
    def run_case(self, mode, *, cleanup_seconds=5):
        with tempfile.TemporaryDirectory(prefix='uf-life-') as temp:
            root = Path(temp); other = None; foreign = None
            try:
                if mode == 'foreign-holder':
                    other = subprocess.Popen([sys.executable, '-B', '-s', str(Path(__file__).resolve()), '--holder', str(root)])
                    other_birth = supervisor.birth(other.pid); wait_file(root / 'holder.sock')
                if mode in ('foreign-attach', 'foreign-refuse'):
                    p = Path('/dev/shm') / ('uf-foreign-' + os.urandom(16).hex())
                    fd = os.open(p, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600); os.ftruncate(fd, 4096)
                    foreign = (p, fd)
                    (root / 'foreign.json').write_text(json.dumps({'name': p.name}))
                start = time.monotonic()
                result = supervisor.run_external([sys.executable, '-B', '-s', str(Path(__file__).resolve()), '--actor', mode, str(root)],
                    root / 'monitor', run_seconds=120, grace_seconds=.2, close_seconds=.15, term_seconds=.15,
                    kill_seconds=.8, private_shm=True, private_cleanup_seconds=cleanup_seconds)
                proof = {'case': mode, 'result': result, 'wall_seconds': time.monotonic() - start,
                         'log': (root / 'monitor/task.log').read_text(errors='replace')}
                PROOFS.append(proof)
                self.assertTrue(result['process_tree_empty']); self.assertFalse(result['remaining'])
                self.assertFalse(result['SDK_ACK_inferred_from_death'])
                for pid, born in result['owned_births'].items():
                    self.assertNotEqual(supervisor.birth(int(pid)), born)
                if (root / 'blocked-close.json').exists():
                    failed = json.loads((root / 'blocked-close.json').read_bytes())['monotonic_ns']
                    term = next(e['monotonic_ns'] for e in result['events'] if e.get('signal') == 'SIGTERM' and e.get('sent'))
                    proof['failure_to_TERM_seconds'] = (term - failed) / 1e9
                    self.assertLess(proof['failure_to_TERM_seconds'], 1)
                    self.assertEqual(result['stop_reason'], 'task-closing-notice')
                    self.assertEqual(result['actual_task_exit_code'], -9)
                if other is not None:
                    self.assertEqual(supervisor.birth(other.pid), other_birth)
                    proof['unrelated_holder_still_alive'] = True
                if (root / 'name.json').exists():
                    name = json.loads((root / 'name.json').read_bytes())
                    proof['name'] = name; proof['name_exists_after_product_reaper'] = (Path('/dev/shm') / name['name']).exists()
                if foreign is not None:
                    self.assertTrue(foreign[0].exists())
                assessment = classify(result, {'physical_closed': True}, {'status': 'ddr-unknown'})
                self.assertTrue(assessment['logical_DDR_budget_retained'])
                self.assertFalse(assessment['SDK_ACK_inferred_from_death'])
                if not result['private_resources']['cleanup_complete']:
                    self.assertEqual(assessment['status'], 'cleanup-failed')
                proof['synthetic_DDRUnknown_assessment'] = assessment['status']
                return proof
            finally:
                if other is not None:
                    supervisor.pidfd_signal(other.pid, other_birth, signal.SIGTERM, expected_uid=os.geteuid()); other.wait(timeout=3)
                if foreign is not None:
                    os.close(foreign[1]); foreign[0].unlink()
                if (root / 'name.json').exists():
                    # Test fixtures deliberately retained by negative assertions,
                    # cleaned by their explicit original namespace after proof.
                    name = json.loads((root / 'name.json').read_bytes())['name']; path = Path('/dev/shm') / name
                    if path.exists():
                        self.assertTrue(name.startswith(('uf-task-', 'uf-unregistered-')))
                        path.unlink()

    def test_long_run_failure_notifies_before_blocking_close_and_auto_reclaims_real_SHM(self):
        proof = self.run_case('blocked-close')
        self.assertTrue(proof['result']['private_resources']['cleanup_complete'])
        self.assertFalse(proof['name_exists_after_product_reaper'])
        self.assertEqual(proof['result']['private_resources']['registered'][0]['status'], 'automatically-unlinked')

    def test_normal_computation_keeps_long_run_deadline_and_normal_cleanup(self):
        proof = self.run_case('normal-long')
        self.assertEqual(proof['result']['actual_task_exit_code'], 0)
        self.assertTrue(proof['result']['private_resources']['cleanup_complete'])
        self.assertGreater(proof['wall_seconds'], .65)
        self.assertFalse(any(e.get('signal') in ('SIGTERM', 'SIGKILL') and e.get('sent') for e in proof['result']['events']))

    def test_real_unrelated_FD_and_VMA_holder_prevents_unlink(self):
        proof = self.run_case('foreign-holder')
        self.assertFalse(proof['result']['private_resources']['cleanup_complete'])
        self.assertTrue(proof['name_exists_after_product_reaper'])
        self.assertTrue(proof['unrelated_holder_still_alive'])

    def test_inode_replacement_is_retained_and_never_unlinked(self):
        proof = self.run_case('inode-drift')
        self.assertFalse(proof['result']['private_resources']['cleanup_complete'])
        self.assertTrue(proof['name_exists_after_product_reaper'])

    @unittest.skipUnless(hasattr(os, 'geteuid') and os.geteuid() == 0, 'real UID change requires task-root')
    def test_UID_drift_of_private_object_is_retained(self):
        proof = self.run_case('uid-drift')
        self.assertFalse(proof['result']['private_resources']['cleanup_complete'])
        self.assertTrue(proof['name_exists_after_product_reaper'])

    def test_preexisting_foreign_shared_memory_attach_never_becomes_private(self):
        proof = self.run_case('foreign-attach')
        self.assertTrue(proof['result']['private_resources']['cleanup_complete'])
        self.assertFalse(proof['result']['private_resources']['registered'])

    def test_foreign_shared_registration_is_rejected_without_unlink(self):
        proof = self.run_case('foreign-refuse')
        self.assertFalse(proof['result']['private_resources']['cleanup_complete'])
        self.assertFalse(proof['result']['private_resources']['registered'])

    def test_close_notice_with_wrong_attempt_parent_or_leader_identity_cannot_shorten_compute(self):
        proof = self.run_case('forged-close')
        self.assertEqual(proof['result']['actual_task_exit_code'], 0)
        self.assertTrue(proof['result']['private_resources']['cleanup_complete'])
        self.assertGreater(proof['wall_seconds'], .65)
        self.assertFalse(any(e['event'] == 'task-closing-notice' for e in proof['result']['events']))

    def test_broken_notice_channel_exits_actor_before_blocked_close_without_long_run_wait(self):
        proof = self.run_case('broken-notice')
        self.assertEqual(proof['result']['actual_task_exit_code'], 70)
        self.assertTrue(proof['result']['private_resources']['cleanup_complete'])
        self.assertFalse(proof['name_exists_after_product_reaper'])
        self.assertLess(proof['wall_seconds'], 8)
        self.assertFalse(any(e.get('reason') == 'run-deadline' for e in proof['result']['events']))




    def test_death_before_mmap_registration_or_ready_retains_unproven_creation(self):
        proof = self.run_case('early-unregistered')
        self.assertEqual(proof['result']['actual_task_exit_code'], 4)
        self.assertFalse(proof['result']['private_resources']['cleanup_complete'])
        self.assertTrue(proof['name_exists_after_product_reaper'])

    def test_real_monotonic_observation_deadline_refuses_reclaim(self):
        proof = self.run_case('blocked-close', cleanup_seconds=.000001)
        self.assertFalse(proof['result']['private_resources']['cleanup_complete'])
        self.assertTrue(proof['name_exists_after_product_reaper'])

    @unittest.skipUnless(hasattr(os, 'geteuid') and os.geteuid() == 0, 'real denied proc observation requires task-root')
    def test_real_nonroot_reaper_permission_denial_cannot_unlink_private_root_object(self):
        with tempfile.TemporaryDirectory(prefix='uf-denied-') as temp:
            root = Path(temp)
            path = Path('/dev/shm') / ('uf-denied-' + os.urandom(16).hex())
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600); os.ftruncate(fd, 4096)
            try:
                info = os.fstat(fd)
                config = {'seconds': 2, 'errors': [], 'tree_empty': True, 'owned': {}, 'unproven': [],
                          'monitor': {'pid': os.getpid()}, 'pins': [fd], 'registered': [{
                              'path': str(path), 'dev': info.st_dev, 'ino': info.st_ino,
                              'uid': 0, 'bytes': 4096, 'pin_fd': fd}]}
                (root / 'config.json').write_text(json.dumps(config))
                process = subprocess.run([sys.executable, '-B', '-s', str(Path(__file__).resolve()),
                    '--denied-probe', str(root / 'config.json')], pass_fds=(fd,), capture_output=True, timeout=5)
                self.assertEqual(process.returncode, 0, process.stderr.decode())
                result = json.loads(process.stdout)
                self.assertFalse(result['cleanup_complete'])
                self.assertTrue(any('PermissionError' in e for e in result['errors']))
                self.assertTrue(path.exists())
                PROOFS.append({'case': 'actual nonroot /proc permission denial', 'result': result})
            finally:
                os.close(fd); path.unlink()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--actor':
        actor(sys.argv[2], Path(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == '--holder':
        holder(Path(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == '--denied-probe':
        config = json.loads(Path(sys.argv[2]).read_bytes())
        os.setgroups([]); os.setgid(65534); os.setuid(65534)
        print(json.dumps(resources.reclaim(config)))
    else:
        unittest.main()
