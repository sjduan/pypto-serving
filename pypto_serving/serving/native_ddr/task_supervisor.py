# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Linux task-only external deadlines: cooperative stop, pidfd TERM/KILL, then real reap.

Run in a dedicated supervisor process, never a native-owning Python thread.
The only children it creates are a gated task entry and that entry's descendants.
It exports physical process evidence, never an SDK unmap/native-drain ACK.
"""
import sys

# Literal script entry otherwise lets the sibling types.py shadow the stdlib
# before argparse/enum load. Preserve standalone use and Python 3.10 support.
if __name__ == '__main__' and not __package__:
    _entry_directory = sys.path.pop(0)
    try:
        __import__('types')
    finally:
        sys.path.insert(0, _entry_directory)

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
import uuid


def observation(pid):
    base = Path('/proc', str(pid))
    try:
        first = (base / 'stat').read_text().rpartition(')')[2].split()
        uid = base.stat().st_uid
        result = {'pid': pid, 'birth': int(first[19]), 'ppid': int(first[1]),
                  'pgrp': int(first[2]), 'session': int(first[3]), 'state': first[0], 'uid': uid}
        last = (base / 'stat').read_text().rpartition(')')[2].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if result['birth'] != int(last[19]):
        raise RuntimeError('process identity changed during observation')
    return result


def birth(pid):
    value = observation(pid)
    return None if value is None else value['birth']


def children(pid):
    try:
        tasks = list(Path('/proc', str(pid), 'task').iterdir())
    except (FileNotFoundError, ProcessLookupError):
        return []
    if len(tasks) > 4096:
        raise RuntimeError('owned task thread count exceeds bounded observation')
    found = set()
    for task in tasks:
        try:
            found.update(int(p) for p in (task / 'children').read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(found)


def delegated_parent():
    """Discover only our own nonroot v2 membership; never alter the mount/root."""
    rows = Path('/proc/self/cgroup').read_text().splitlines()
    unified = [row[3:] for row in rows if row.startswith('0::')]
    if len(unified) != 1 or unified[0] == '/':
        return None
    mount = Path('/sys/fs/cgroup').resolve()
    path = (mount / unified[0].lstrip('/')).resolve()
    if path == mount or not path.is_relative_to(mount):
        return None
    if (path / 'cgroup.controllers').is_file() and os.access(path, os.W_OK):
        return str(path)
    return None


def pidfd_signal(pid, expected_birth, signum, *, expected_uid):
    """No name/pgrp signal, PID-reuse window or replacement process is accepted."""
    before = observation(pid)
    if before is None:
        return False
    if before['birth'] != expected_birth or before['uid'] != expected_uid:
        raise RuntimeError('refuse signal after process identity/UID drift')
    try:
        handle = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        now = observation(pid)
        if now is None:
            return False
        if (now['birth'], now['uid']) != (expected_birth, expected_uid):
            raise RuntimeError('refuse signal after pidfd identity drift')
        signal.pidfd_send_signal(handle, signum)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(handle)


class TaskCgroup:
    """Optional, explicitly delegated nonroot v2 subtree; never mounts/enables root controllers."""
    def __init__(self, parent, *, max_processes, memory_bytes):
        self.path = None
        self.reason = 'no explicit delegated cgroup parent; use dedicated subreaper/session'
        parent = delegated_parent() if parent is None else parent
        if parent is None:
            return
        parent = Path(parent)
        mount = Path('/sys/fs/cgroup').resolve()
        resolved = parent.resolve()
        if resolved == mount or not resolved.is_relative_to(mount) or parent.is_symlink():
            raise ValueError('only an explicit nonroot delegated cgroup-v2 parent is allowed')
        if (not (resolved / 'cgroup.controllers').is_file()
                or os.getpid() not in [int(p) for p in (resolved / 'cgroup.procs').read_text().split()]):
            raise ValueError('delegated cgroup parent does not contain this supervisor')
        candidate = resolved / ('uflow-task-' + uuid.uuid4().hex)
        try:
            candidate.mkdir()
        except (PermissionError, OSError) as error:
            self.reason = 'delegated subtree unavailable: ' + repr(error)
            return
        try:
            if (candidate / 'cgroup.procs').read_text().strip():
                raise RuntimeError('new task cgroup is not empty')
            if not (candidate / 'cgroup.kill').is_file():
                raise RuntimeError('task cgroup has no atomic descendant kill interface')
            # pids.max counts threads as well as process leaders. Keep a
            # separate bounded thread allowance for native runtime pools.
            (candidate / 'pids.max').write_text(str(min(4096, max_processes * 16)))
            if memory_bytes is not None:
                (candidate / 'memory.max').write_text(str(memory_bytes))
            self.path, self.reason = candidate, None
        except BaseException:
            candidate.rmdir()
            raise

    def attach(self, pid):
        if self.path is not None:
            (self.path / 'cgroup.procs').write_text(str(pid))
            if pid not in self.members():
                raise RuntimeError('gated task was not admitted to its own cgroup')

    def members(self):
        return [] if self.path is None else [int(p) for p in (self.path / 'cgroup.procs').read_text().split()]

    def kill(self):
        if self.path is not None:
            (self.path / 'cgroup.kill').write_text('1')

    def close(self):
        if self.path is not None:
            if self.members() or 'populated 1' in (self.path / 'cgroup.events').read_text():
                raise RuntimeError('task cgroup is still populated; keep it quarantined')
            self.path.rmdir()


def bounded_spec(value):
    required = {'argv', 'root', 'run_seconds', 'grace_seconds', 'term_seconds', 'kill_seconds',
                'max_processes', 'graceful_ready', 'launcher_pid', 'launcher_birth', 'cgroup_parent',
                'memory_bytes', 'file_limit'}
    optional = {'close_seconds': 10, 'private_shm': False, 'private_cleanup_seconds': 15}
    if type(value) is not dict or not required <= set(value) or set(value) - required - set(optional):
        raise ValueError('exact supervisor configuration required')
    value = {**optional, **value}
    if (type(value['argv']) is not list or not value['argv']
            or any(type(s) is not str or not s or '\0' in s for s in value['argv'])):
        raise ValueError('literal finite argv required')
    for key in ('run_seconds', 'grace_seconds', 'term_seconds', 'kill_seconds'):
        n = value[key]
        if type(n) not in (int, float) or not math.isfinite(n) or not 0 < n <= 14400:
            raise ValueError('finite positive monotonic deadline required')
    for key in ('max_processes', 'file_limit', 'launcher_pid', 'launcher_birth'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('positive non-bool process/resource bound required')
    if value['max_processes'] > 256 or not 32 <= value['file_limit'] <= 4096:
        raise ValueError('task process/FD bound exceeds supported scope')
    if value['memory_bytes'] is not None and (type(value['memory_bytes']) is not int or value['memory_bytes'] <= 0):
        raise ValueError('optional private cgroup memory bound must be positive')
    if type(value['private_shm']) is not bool:
        raise ValueError('explicit private SHM observation policy required')
    for key in ('close_seconds', 'private_cleanup_seconds'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not 0 < value[key] <= 60:
            raise ValueError('finite positive close/reclaim cap up to60 seconds required')
    return value


class OwnedSupervisor:
    def __init__(self, config):
        self.spec = bounded_spec(config)
        self.root = Path(config['root']).resolve()
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise ValueError('same-UID owned supervisor directory required')
        self.uid = os.geteuid()
        self.events, self.evidence_errors, self.known, self.reaped = [], [], {}, {}
        self.stop_reason, self.signal_requested = None, False
        self.phase, self.cgroup = 'running', None
        self.closing_notice = None

    def event(self, event, **values):
        if len(self.events) >= 2048:
            return
        row = {'event': event, 'monotonic_ns': time.monotonic_ns(), **values}
        self.events.append(row)
        if not self.evidence_errors:
            try:
                with (self.root / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n'); stream.flush()
            except OSError as error:
                self.evidence_errors.append(repr(error))
                self.stop_reason = self.stop_reason or 'evidence-I/O-failed'

    def refresh(self):
        # Dedicated supervisor starts with zero children and creates exactly one
        # gated entry. Every adopted child is therefore from this owned subtree,
        # including double-fork/setsid grandchildren. No /proc name scan is used.
        todo, seen = [os.getpid()], set()
        while todo:
            parent = todo.pop()
            for pid in children(parent):
                if pid in seen:
                    continue
                seen.add(pid)
                if len(seen) > 4096:
                    raise RuntimeError('owned descendant observation exceeds hard safety bound')
                row = observation(pid)
                if row is None:
                    continue
                if row['uid'] != self.uid:
                    raise RuntimeError('owned child UID drift; cannot claim isolation')
                old = self.known.get(pid)
                if old is not None and old['birth'] != row['birth']:
                    raise RuntimeError('owned PID identity was reused')
                if old is None:
                    self.known[pid] = row
                    self.event('owned-birth', **row)
                todo.append(pid)
        if self.cgroup is not None:
            for pid in self.cgroup.members():
                row = observation(pid)
                if row is None:
                    continue
                if row['uid'] != self.uid:
                    raise RuntimeError('private task cgroup member UID changed')
                old = self.known.get(pid)
                if old is not None and old['birth'] != row['birth']:
                    raise RuntimeError('cgroup member identity drift')
                if old is None:
                    self.known[pid] = row
                    self.event('owned-cgroup-birth', **row)
        alive = {p: r for p, r in self.known.items() if birth(p) == r['birth']}
        if len(alive) > self.spec['max_processes'] and self.stop_reason is None:
            self.stop_reason = 'task-process-limit'
        return alive

    def reap(self, child):
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            code = os.waitstatus_to_exitcode(status)
            self.reaped[pid] = code
            if pid == child.pid:
                child.returncode = code
            self.event('actual-wait', pid=pid, exit_code=code)

    def send_all(self, alive, signum):
        for pid, row in sorted(alive.items(), reverse=True):
            try:
                sent = pidfd_signal(pid, row['birth'], signum, expected_uid=self.uid)
            except (OSError, RuntimeError) as error:
                self.event('signal-refused', pid=pid, birth=row['birth'], error=repr(error))
                self.isolation_error = repr(error)
                continue
            self.event('signal', pid=pid, birth=row['birth'], signal=signal.Signals(signum).name, sent=sent)

    def isolation_failed(self, error):
        self.isolation_error = repr(error)
        self.stop_reason = self.stop_reason or 'descendant-observation-failed'
        self.event('isolation-error', error=repr(error))

    def verified_known(self):
        alive = {}
        for pid, row in self.known.items():
            try:
                if birth(pid) == row['birth']:
                    alive[pid] = row
            except (OSError, RuntimeError, ValueError) as error:
                self.isolation_failed(error)
        return alive

    def cgroup_empty(self):
        try:
            return not self.cgroup.members()
        except (OSError, RuntimeError, ValueError) as error:
            self.isolation_failed(error)
            return False

    def notified_closing(self, pid, born):
        if self.closing_notice is None:
            self.closing_notice = time.monotonic()
            self.event('task-closing-notice', pid=pid, birth=born)

    def cooperative(self, child, born):
        path = self.spec['graceful_ready']
        if path is None:
            self.event('graceful-unavailable', reason='no installed task close-channel proof')
            return
        try:
            p = Path(path); info = p.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != self.uid or info.st_size > 65536:
                raise ValueError('invalid cooperative ready receipt')
            ready = json.loads(p.read_bytes())
            if ready['pid'] != child.pid or ready['birth'] != born or ready['signal'] != 'SIGUSR1':
                raise ValueError('close-channel identity/signal mismatch')
            sent = pidfd_signal(child.pid, born, signal.SIGUSR1, expected_uid=self.uid)
            self.event('cooperative-stop', pid=child.pid, birth=born, sent=sent)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.event('graceful-unavailable', error=repr(error))

    def run(self):
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise RuntimeError('Linux pidfd support is required; no racy os.kill fallback')
        if children(os.getpid()):
            raise RuntimeError('supervisor must be a dedicated process without preexisting children')
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')
        if birth(self.spec['launcher_pid']) != self.spec['launcher_birth'] or os.getppid() != self.spec['launcher_pid']:
            raise RuntimeError('launcher identity changed before task creation')
        self.cgroup = TaskCgroup(self.spec['cgroup_parent'], max_processes=self.spec['max_processes'],
                                 memory_bytes=self.spec['memory_bytes'])
        self.isolation_error = None
        # Install monitor handlers before creating the gated task, including
        # the small admission window. Only the actor ever imports native code.
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'signal_requested', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'signal_requested', True))
        self.event('boundary-before-child', mode='delegated-cgroup' if self.cgroup.path else 'dedicated-subreaper-session',
                   cgroup=None if self.cgroup.path is None else str(self.cgroup.path), cgroup_reason=self.cgroup.reason,
                   max_processes=self.spec['max_processes'], file_limit=self.spec['file_limit'],
                   task_thread_limit=min(4096, self.spec['max_processes'] * 16), file_bytes_limit=1 << 30,
                   root_RLIMIT_NPROC_enforced=self.uid != 0, no_native_cleanup_in_supervisor=True)
        from task_resources import LifecycleRegistry
        lifecycle = LifecycleRegistry(self.root, private_shm=self.spec['private_shm'], event=self.event)
        # Log open/intent failure before fork cannot leave an untracked child.
        with (self.root / 'task.log').open('xb') as log, (self.root / 'launch-once.json').open('x') as intent:
            json.dump({'argv': self.spec['argv'], 'supervisor_pid': os.getpid(), 'supervisor_birth': birth(os.getpid())}, intent)
            intent.flush(); os.fsync(intent.fileno())
            read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
            try:
                argv = [sys.executable, '-B', '-s', str(Path(__file__).resolve()), '--entry', str(read_fd),
                        str(self.spec['max_processes']), str(self.spec['file_limit']), '--', *self.spec['argv']]
                child = subprocess.Popen(argv, pass_fds=(read_fd,), start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                os.close(read_fd); read_fd = None
                born = birth(child.pid)
                row = observation(child.pid)
                if row is None or row['birth'] != born or row['session'] != child.pid:
                    raise RuntimeError('gated task session identity is not established')
                self.known[child.pid] = row
                self.cgroup.attach(child.pid)
                self.event('task-start', pid=child.pid, birth=born, session=row['session'])
                context = json.dumps(lifecycle.context(row)).encode()
                if len(context) > 4096:
                    raise RuntimeError('task lifecycle gate context exceeds bound')
                os.write(write_fd, context + b'\n')
            except (OSError, RuntimeError, ValueError) as error:
                # Close the gate, then still enter the bounded reap path.
                # A partially admitted child must not escape on setup failure.
                if 'child' not in locals():
                    raise
                self.isolation_failed(error)
                born = self.known.get(child.pid, {}).get('birth')
            finally:
                if read_fd is not None:
                    os.close(read_fd)
                os.close(write_fd)
            start = time.monotonic()
            deadline = start + self.spec['run_seconds']
            empty_observations = 0
            while True:
                self.reap(child)
                try:
                    alive = self.refresh()
                except (OSError, RuntimeError, ValueError) as error:
                    self.isolation_failed(error)
                    alive = self.verified_known()
                now = time.monotonic()
                try:
                    lifecycle.service(self.known, self.notified_closing)
                except (OSError, RuntimeError, ValueError) as error:
                    lifecycle.error(repr(error))
                    self.stop_reason = self.stop_reason or 'lifecycle-observation-failed'
                if not alive and child.returncode is not None and self.cgroup_empty():
                    empty_observations += 1
                    if empty_observations >= 2:
                        break
                else:
                    empty_observations = 0
                try:
                    parent_gone = birth(self.spec['launcher_pid']) != self.spec['launcher_birth']
                    requested = (self.root / 'stop.request').exists()
                except (OSError, RuntimeError, ValueError) as error:
                    self.isolation_failed(error)
                    parent_gone, requested = True, True
                if self.closing_notice is not None and self.phase in ('running', 'grace'):
                    self.phase = 'failure-close'
                    deadline = min(deadline, self.closing_notice + self.spec['close_seconds'])
                    self.stop_reason = self.stop_reason or 'task-closing-notice'
                    self.event('phase', phase='failure-close', reason=self.stop_reason,
                               close_seconds=self.spec['close_seconds'])
                if self.phase == 'running' and (now >= deadline or parent_gone or self.signal_requested
                                                 or requested or self.stop_reason or child.returncode is not None and alive):
                    self.stop_reason = self.stop_reason or ('launcher-gone' if parent_gone else 'external-stop' if self.signal_requested or requested
                                                           else 'leader-exited-with-descendants' if child.returncode is not None else 'run-deadline')
                    self.phase = 'grace'
                    self.event('phase', phase='grace', reason=self.stop_reason)
                    self.cooperative(child, born)
                    deadline = now + self.spec['grace_seconds']
                elif self.phase in ('grace', 'failure-close') and now >= deadline:
                    self.phase = 'term'; self.send_all(alive, signal.SIGTERM)
                    deadline = now + self.spec['term_seconds']
                elif self.phase == 'term' and now >= deadline:
                    self.phase = 'kill'; self.send_all(alive, signal.SIGKILL)
                    if self.cgroup.path is not None:
                        try:
                            self.cgroup.kill(); self.event('private-cgroup-kill')
                        except (OSError, RuntimeError) as error:
                            self.isolation_failed(error)
                    deadline = now + self.spec['kill_seconds']
                elif self.phase == 'kill':
                    # Newly forked/adopted task descendants are still in this
                    # dedicated boundary; pidfd each exact birth, never pkill.
                    unsignalled = {p: r for p, r in alive.items() if r['state'] != 'Z'}
                    self.send_all(unsignalled, signal.SIGKILL)
                    if now >= deadline:
                        break
                time.sleep(.02)
            self.reap(child)
            try:
                remaining = self.refresh()
            except (OSError, RuntimeError, ValueError) as error:
                self.isolation_failed(error)
                remaining = self.verified_known()
            clear = not self.isolation_error and not remaining and child.returncode is not None and self.cgroup_empty()
            if clear:
                try:
                    self.cgroup.close()
                except (OSError, RuntimeError) as error:
                    self.isolation_failed(error)
                    clear = False
            try:
                private_resources = lifecycle.reclaim({p: r['birth'] for p, r in self.known.items()},
                    self.spec['private_cleanup_seconds'], clear)
            except (OSError, RuntimeError, ValueError) as error:
                private_resources = {'enabled': self.spec['private_shm'], 'cleanup_complete': False, 'error': repr(error)}
            finally:
                lifecycle.close()
            self.reap(child)
            if children(os.getpid()):
                self.isolation_failed(RuntimeError('private resource auditor has not been reaped'))
                clear = False
            result = {'schema': 1, 'supervisor_pid': os.getpid(), 'supervisor_birth': birth(os.getpid()),
                      'task_pid': child.pid, 'task_birth': born, 'actual_task_exit_code': child.returncode,
                      'actual_wait_statuses': self.reaped, 'owned_births': {p: r['birth'] for p, r in self.known.items()},
                      'remaining': remaining, 'process_tree_empty': clear, 'phase': self.phase, 'stop_reason': self.stop_reason,
                      'elapsed_seconds': time.monotonic() - start, 'evidence_errors': self.evidence_errors,
                      'isolation_error': self.isolation_error,
                      'events': self.events, 'logical_DDR_closed': False, 'SDK_ACK_inferred_from_death': False,
                      'physical_device_reusable': False, 'physical_resource_audit_required': True,
                      'status': 'processes-closed-pending-resource-audit' if clear else 'device-quarantined',
                      'cgroup_used': self.cgroup.path is not None}
            result['private_resources'] = private_resources
            result['task_resources_closed'] = clear and private_resources['cleanup_complete']
            try:
                with (self.root / 'result.json').open('x') as stream:
                    json.dump(result, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
            except OSError as error:
                result['evidence_errors'].append(repr(error))
            return result


def run_external(argv, root, *, run_seconds, grace_seconds=10, term_seconds=10, kill_seconds=5,
                 graceful_ready=None, max_processes=64, cgroup_parent=None, memory_bytes=None, file_limit=1024,
                 close_seconds=10, private_shm=False, private_cleanup_seconds=15):
    """Controller wrapper: the separate monitor never owns a native handle."""
    root = Path(root); root.mkdir()
    config = {'argv': list(argv), 'root': str(root.resolve()), 'run_seconds': run_seconds,
              'grace_seconds': grace_seconds, 'term_seconds': term_seconds, 'kill_seconds': kill_seconds,
              'max_processes': max_processes, 'graceful_ready': None if graceful_ready is None else str(graceful_ready),
              'launcher_pid': os.getpid(), 'launcher_birth': birth(os.getpid()), 'cgroup_parent': cgroup_parent,
              'memory_bytes': memory_bytes, 'file_limit': file_limit, 'close_seconds': close_seconds,
              'private_shm': private_shm, 'private_cleanup_seconds': private_cleanup_seconds}
    bounded_spec(config)
    with (root / 'config.json').open('x') as stream:
        json.dump(config, stream, indent=2)
    with (root / 'supervisor.log').open('xb') as log:
        monitor = subprocess.Popen([sys.executable, '-B', '-s', str(Path(__file__).resolve()),
                                    '--config', str(root / 'config.json')], stdout=log, stderr=subprocess.STDOUT)
        monitor_birth = birth(monitor.pid)
        receipt_error = None
        try:
            with (root / 'supervisor-start.json').open('x') as stream:
                json.dump({'pid': monitor.pid, 'birth': monitor_birth}, stream)
        except OSError as error:
            receipt_error = repr(error)
            pidfd_signal(monitor.pid, monitor_birth, signal.SIGTERM, expected_uid=os.geteuid())
        try:
            code = monitor.wait(timeout=run_seconds + max(grace_seconds, close_seconds) + term_seconds + kill_seconds
                                + private_cleanup_seconds + 18)
        except subprocess.TimeoutExpired:
            # Do not hang the launcher or pronounce a wedged kernel safe.
            raise RuntimeError('external monitor exceeded bound; task device must remain quarantined') from None
    result = json.loads((root / 'result.json').read_bytes()) if (root / 'result.json').exists() else None
    if code != 0 or result is None:
        raise RuntimeError('supervisor failed; inspect exact task boundary and keep device quarantined')
    result['supervisor_actual_exit_code'] = code
    result['supervisor_pid'], result['supervisor_birth'] = monitor.pid, monitor_birth
    if receipt_error:
        result['evidence_errors'].append(receipt_error)
    return result


def entry(fd, max_processes, file_limit, argv):
    import resource
    data = os.read(fd, 8192)
    if not data.endswith(b'\n'):
        raise RuntimeError('task gate closed before lifecycle boundary was established')
    context = json.loads(data)
    if (os.getpid(), birth(os.getpid()), os.getppid()) != (context['leader_pid'], context['leader_birth'], context['monitor_pid']):
        raise RuntimeError('task lifecycle gate identity drift')
    os.close(fd)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    _, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    limit = min(1 << 30, hard) if hard != resource.RLIM_INFINITY else 1 << 30
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    limit = min(file_limit, hard) if hard != resource.RLIM_INFINITY else file_limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    threads = min(4096, max_processes * 16)
    limit = min(threads, hard) if hard != resource.RLIM_INFINITY else threads
    resource.setrlimit(resource.RLIMIT_NPROC, (limit, limit))
    os.environ['UFLOW_TASK_LIFECYCLE'] = json.dumps(context)
    # Static Python launcher, no live attach/function replacement or native
    # teardown in the supervisor. General non-Python commands keep their argv.
    if Path(argv[0]).resolve() == Path(sys.executable).resolve():
        index = 1
        while index < len(argv) and argv[index] in ('-B', '-s', '-u'):
            index += 1
        if index < len(argv) and argv[index].endswith('.py') and Path(argv[index]).is_file():
            argv = [*argv[:index], str(Path(__file__).with_name('task_lifecycle.py')),
                    '--task-script', *argv[index:]]
        elif context['private_shm']:
            raise RuntimeError('private resource observation requires a literal Python file entry')
    elif context['private_shm']:
        raise RuntimeError('private resource observation requires the same Python interpreter')
    os.execvp(argv[0], argv)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--entry':
        entry(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[6:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = OwnedSupervisor(json.loads(args.config.read_bytes())).run()
    print(json.dumps({'process_tree_empty': result['process_tree_empty'], 'status': result['status']}))
    raise SystemExit(0 if result['process_tree_empty'] and not result['evidence_errors'] else 2)


if __name__ == '__main__':
    main()
