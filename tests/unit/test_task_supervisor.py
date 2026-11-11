# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real Linux POSIX subtree/signal/reap tests; no SDK, Worker, UF or device import."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SOURCE = Path(__file__).resolve().parents[2]
MODULE = SOURCE / 'pypto_serving/serving/native_ddr/task_supervisor.py'
spec = importlib.util.spec_from_file_location('task_supervisor_tested', MODULE)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
PROOFS = []


def ready(path):
    Path(path).write_text(json.dumps({'pid': os.getpid(), 'birth': supervisor.birth(os.getpid()), 'signal': 'SIGUSR1'}))


def actor(mode, directory):
    root = Path(directory)
    if mode == 'normal':
        return
    if mode == 'failure':
        raise RuntimeError('controlled CPU actor failure')
    stop = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: stop.set() if mode == 'graceful' else None)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready(root / 'ready.json')
    if mode == 'orphan':
        if os.fork():
            os._exit(7)
        os.setsid()
        if os.fork():
            os._exit(8)
        ready(root / 'orphan.json')
    if mode == 'late-fork':
        def on_term(*_):
            if os.fork() == 0:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                ready(root / 'late-child.json')
                while True:
                    time.sleep(.05)
            raise SystemExit(9)
        signal.signal(signal.SIGTERM, on_term)
    if mode == 'log-failure':
        event = root / 'monitor/events.jsonl'
        event.rename(event.with_name('events-before-failure.jsonl'))
        event.mkdir()
        if os.fork() == 0:
            ready(root / 'log-child.json')
    while not stop.wait(.02):
        pass
    (root / 'graceful-stop-completed.json').write_text(json.dumps({'normal_CPU_close': True, 'SDK_ACK': False}))


def argv(mode, root):
    return [sys.executable, '-B', '-s', str(Path(__file__).resolve()), '--actor', mode, str(root)]


def config(mode, root, monitor):
    return {'argv': argv(mode, root), 'root': str(monitor), 'run_seconds': .5, 'grace_seconds': .15,
            'term_seconds': .15, 'kill_seconds': .6, 'max_processes': 32,
            'graceful_ready': str(root / 'ready.json'), 'launcher_pid': os.getpid(),
            'launcher_birth': supervisor.birth(os.getpid()), 'cgroup_parent': None, 'memory_bytes': None, 'file_limit': 256}


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'), 'Linux pidfd/POSIX required')
class SupervisorProcessTests(unittest.TestCase):
    def run_case(self, mode):
        with tempfile.TemporaryDirectory(prefix='uf-sup-') as temp:
            root = Path(temp)
            result = supervisor.run_external(argv(mode, root), root / 'monitor', run_seconds=.5,
                grace_seconds=.15, term_seconds=.15, kill_seconds=.6, graceful_ready=root / 'ready.json')
            PROOFS.append({'case': mode, 'result': result,
                           'task_log': (root / 'monitor/task.log').read_text(errors='replace'),
                           'supervisor_log': (root / 'monitor/supervisor.log').read_text(errors='replace')})
            self.assertTrue(result['process_tree_empty'])
            self.assertEqual(result['supervisor_actual_exit_code'], 0)
            self.assertLess(result['elapsed_seconds'], 2.5)
            self.assertFalse(result['SDK_ACK_inferred_from_death'] or result['logical_DDR_closed'])
            self.assertFalse(result['physical_device_reusable'])
            self.assertTrue(result['physical_resource_audit_required'])
            self.assertIsNone(supervisor.birth(result['supervisor_pid']))
            for pid, born in result['owned_births'].items():
                self.assertNotEqual(supervisor.birth(int(pid)), born)
            return result

    def test_normal_task_actual_wait0_without_any_signal(self):
        result = self.run_case('normal')
        self.assertEqual(result['actual_task_exit_code'], 0)
        self.assertFalse([e for e in result['events'] if e['event'] == 'signal'])

    def test_failed_task_keeps_actual_exit1_without_fake_success(self):
        result = self.run_case('failure')
        self.assertEqual(result['actual_task_exit_code'], 1)

    def test_cooperative_channel_is_exact_identity_and_stops_without_TERM(self):
        result = self.run_case('graceful')
        self.assertEqual(result['actual_task_exit_code'], 0)
        self.assertEqual(result['phase'], 'grace')
        self.assertTrue(any(e['event'] == 'cooperative-stop' and e['sent'] for e in result['events']))
        self.assertFalse([e for e in result['events'] if e['event'] == 'signal'])

    def test_ignore_TERM_requires_actual_pidfd_KILL_and_wait_minus9(self):
        result = self.run_case('ignore-term')
        self.assertEqual(result['actual_task_exit_code'], -signal.SIGKILL)
        names = [e['signal'] for e in result['events'] if e['event'] == 'signal' and e['sent']]
        self.assertIn('SIGTERM', names); self.assertIn('SIGKILL', names)

    def test_double_fork_setsid_orphan_is_adopted_and_reaped(self):
        result = self.run_case('orphan')
        self.assertEqual(result['actual_task_exit_code'], 7)
        self.assertGreaterEqual(len(result['owned_births']), 2)
        self.assertTrue(any(c == -signal.SIGKILL for c in result['actual_wait_statuses'].values()))

    def test_new_descendant_created_during_TERM_does_not_escape(self):
        result = self.run_case('late-fork')
        self.assertEqual(result['actual_task_exit_code'], 9)
        self.assertTrue(any(c == -signal.SIGKILL for c in result['actual_wait_statuses'].values()))

    def test_same_name_unrelated_process_is_never_in_signalled_boundary(self):
        foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(15)'])
        born = supervisor.birth(foreign.pid)
        try:
            result = self.run_case('ignore-term')
            self.assertIsNone(foreign.poll())
            PROOFS.append({'case': 'unrelated-same-name-alive', 'pid': foreign.pid, 'birth': born,
                           'unaffected_after_owned_subtree_KILL': True})
            self.assertNotIn(foreign.pid, result['owned_births'])
            self.assertFalse(any(e.get('pid') == foreign.pid for e in result['events'] if e['event'] == 'signal'))
        finally:
            supervisor.pidfd_signal(foreign.pid, born, signal.SIGTERM, expected_uid=os.geteuid())
            foreign.wait(timeout=3)

    def test_wrong_birth_refuses_signal_and_does_not_touch_process(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(15)'])
        born = supervisor.birth(child.pid)
        try:
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                supervisor.pidfd_signal(child.pid, born + 1, signal.SIGKILL, expected_uid=os.geteuid())
            self.assertIsNone(child.poll())
            PROOFS.append({'case': 'wrong-birth-signal-refused', 'pid': child.pid, 'birth': born,
                           'wrong_expected_birth': born + 1, 'actual_process_still_alive': True})
        finally:
            supervisor.pidfd_signal(child.pid, born, signal.SIGTERM, expected_uid=os.geteuid())
            child.wait(timeout=3)

    def test_logging_failure_after_creation_still_bounds_and_reaps_exact_subtree(self):
        with tempfile.TemporaryDirectory(prefix='uf-sup-') as temp:
            root = Path(temp); monitor = root / 'monitor'; monitor.mkdir()
            (monitor / 'config.json').write_text(json.dumps(config('log-failure', root, monitor)))
            with (root / 'supervisor.log').open('wb') as log:
                process = subprocess.Popen([sys.executable, '-B', '-s', str(MODULE), '--config',
                                            str(monitor / 'config.json')], stdout=log, stderr=log)
                self.assertEqual(process.wait(timeout=5), 2)
            result = json.loads((monitor / 'result.json').read_bytes())
            PROOFS.append({'case': 'logging-failure-after-task-start', 'result': result,
                           'supervisor_log': (root / 'supervisor.log').read_text(errors='replace')})
            self.assertTrue(result['process_tree_empty']); self.assertTrue(result['evidence_errors'])
            self.assertLess(result['elapsed_seconds'], 2.5)
            for pid, born in result['owned_births'].items():
                self.assertNotEqual(supervisor.birth(int(pid)), born)

    def test_launcher_death_does_not_kill_monitor_before_it_reaps_orphans(self):
        libc = ctypes.CDLL(None, use_errno=True)
        previous = ctypes.c_int()
        self.assertEqual(libc.prctl(37, ctypes.byref(previous), 0, 0, 0), 0)
        self.assertEqual(libc.prctl(36, 1, 0, 0, 0), 0)
        try:
            with tempfile.TemporaryDirectory(prefix='uf-sup-') as temp:
                root = Path(temp)
                with (root / 'launcher.log').open('wb') as log:
                    parent = subprocess.Popen([sys.executable, '-B', '-s', str(Path(__file__).resolve()),
                                               '--dying-launcher', str(root)], stdout=log, stderr=log)
                    self.assertEqual(parent.wait(timeout=6), 3)
                start = json.loads((root / 'monitor/supervisor-start.json').read_bytes())
                deadline = time.monotonic() + 5
                status = None
                while time.monotonic() < deadline:
                    pid, raw = os.waitpid(start['pid'], os.WNOHANG)
                    if pid:
                        status = os.waitstatus_to_exitcode(raw); break
                    time.sleep(.02)
                self.assertEqual(status, 0)
                result = json.loads((root / 'monitor/result.json').read_bytes())
                PROOFS.append({'case': 'launcher-death', 'launcher_actual_exit': 3,
                               'monitor_actual_wait_exit': status, 'result': result})
                self.assertTrue(result['process_tree_empty'])
                self.assertEqual(result['stop_reason'], 'launcher-gone')
                self.assertFalse(result['logical_DDR_closed'])
        finally:
            self.assertEqual(libc.prctl(36, previous.value, 0, 0, 0), 0)

    def test_cgroup_observation_error_reaps_real_child_but_quarantines_boundary(self):
        with tempfile.TemporaryDirectory(prefix='uf-sup-') as temp:
            root = Path(temp); monitor = root / 'monitor'; monitor.mkdir()
            (monitor / 'config.json').write_text(json.dumps(config('ignore-term', root, monitor)))
            with (root / 'monitor.log').open('wb') as log:
                child = subprocess.Popen([sys.executable, '-B', '-s', str(Path(__file__).resolve()),
                                          '--failing-cgroup', str(root)], stdout=log, stderr=log)
                self.assertEqual(child.wait(timeout=5), 2)
            result = json.loads((monitor / 'result.json').read_bytes())
            PROOFS.append({'case': 'unobservable-cgroup', 'result': result})
            self.assertFalse(result['process_tree_empty'])
            self.assertTrue(result['isolation_error'])
            self.assertEqual(result['actual_task_exit_code'], -9)
            self.assertFalse(result['remaining'])
            self.assertLess(result['elapsed_seconds'], 3)
            for pid, born in result['owned_births'].items():
                self.assertNotEqual(supervisor.birth(int(pid)), born)


class SupervisorSpecTests(unittest.TestCase):
    def test_nonfinite_boolean_unbounded_config_is_rejected_before_any_Popen(self):
        sample = {'argv': ['test'], 'root': '/tmp/test', 'run_seconds': 1, 'grace_seconds': 1,
                  'term_seconds': 1, 'kill_seconds': 1, 'max_processes': 32, 'graceful_ready': None,
                  'launcher_pid': 1, 'launcher_birth': 1, 'cgroup_parent': None, 'memory_bytes': None, 'file_limit': 256}
        for key, value in [('run_seconds', True), ('term_seconds', float('inf')), ('kill_seconds', -1),
                           ('max_processes', 257), ('file_limit', 100000), ('argv', 'pkill anything')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                supervisor.bounded_spec({**sample, key: value})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--actor':
        actor(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == '--failing-cgroup':
        sys.path.insert(0, str(MODULE.parent))
        root = Path(sys.argv[2])
        class UnobservableCgroup:
            path, reason = None, 'controlled cgroup observation error'
            def __init__(self, *_args, **_kwargs):
                pass
            def attach(self, _pid):
                pass
            def members(self):
                if (root / 'ready.json').exists():
                    raise PermissionError('controlled cgroup observer denied after real child admission')
                return []
        supervisor.TaskCgroup = UnobservableCgroup
        result = supervisor.OwnedSupervisor(json.loads((root / 'monitor/config.json').read_bytes())).run()
        raise SystemExit(0 if result['process_tree_empty'] else 2)
    elif len(sys.argv) > 1 and sys.argv[1] == '--dying-launcher':
        root = Path(sys.argv[2])
        def leave():
            deadline = time.monotonic() + 4
            while not (root / 'ready.json').exists():
                if time.monotonic() > deadline:
                    os._exit(4)
                time.sleep(.01)
            os._exit(3)
        threading.Thread(target=leave, daemon=True).start()
        supervisor.run_external(argv('ignore-term', root), root / 'monitor', run_seconds=2,
            grace_seconds=.1, term_seconds=.1, kill_seconds=.6, graceful_ready=root / 'ready.json')
    else:
        unittest.main()
