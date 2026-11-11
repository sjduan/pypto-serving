# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-pinned task-private SHM registry and bounded independent post-reap reclamation.

Only exclusive creations observed by the static launcher are eligible. Naming
conventions alone confer no authority. UF/provider/shared/unknown allocations
are never unlinked. Missing creation evidence is a retained cleanup failure.
"""
import array
import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid


def identity(pid):
    try:
        p = Path('/proc', str(pid))
        born = int((p / 'stat').read_text().rpartition(')')[2].split()[19])
        return {'pid': pid, 'birth': born, 'uid': p.stat().st_uid}
    except (FileNotFoundError, ProcessLookupError):
        return None


class LifecycleRegistry:
    def __init__(self, root, *, private_shm, event):
        self.root, self.private, self.event = Path(root), private_shm, event
        self.attempt, self.monitor = uuid.uuid4().hex, identity(os.getpid())
        self.leader, self.records, self.errors, self.created = None, {}, [], {}
        self.watch = None
        self.path = self.root / 'lifecycle.sock'
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.listener.bind(str(self.path)); self.listener.listen(16); self.listener.setblocking(False)
        self.socket_inode = self.path.lstat().st_ino
        self.baseline = {}
        if private_shm:
            libc = ctypes.CDLL(None, use_errno=True)
            watch = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if watch < 0:
                raise OSError(ctypes.get_errno(), 'inotify_init1')
            self.watch = watch
            if libc.inotify_add_watch(watch, b'/dev/shm', 0x100 | 0x80 | 0x200 | 0x40) < 0:
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
            with os.scandir('/dev/shm') as iterator:
                paths = [Path(p.path) for p in itertools.islice(iterator, 4097)]
            if len(paths) > 4096:
                raise RuntimeError('SHM baseline exceeds bounded inventory')
            for p in paths:
                try:
                    s = p.lstat(); self.baseline[p.name] = (s.st_dev, s.st_ino)
                except FileNotFoundError:
                    continue
            self.poll_creations()

    def error(self, message):
        if len(self.errors) < 64 and message not in self.errors:
            self.errors.append(message)

    def context(self, leader):
        self.leader = leader
        return {'attempt': self.attempt, 'socket': str(self.path), 'monitor_pid': self.monitor['pid'],
                'monitor_birth': self.monitor['birth'], 'leader_pid': leader['pid'],
                'leader_birth': leader['birth'], 'private_shm': self.private}

    def poll_creations(self):
        if self.watch is None:
            return
        # One finite read per tick; kernel queue overflow or excess observations
        # is a failure, never an assumption that nothing was created.
        try:
            data = os.read(self.watch, 262144)
        except BlockingIOError:
            return
        offset = 0
        while offset < len(data):
            _, mask, _, size = struct.unpack_from('iIII', data, offset); offset += 16
            name = data[offset:offset + size].split(b'\0', 1)[0].decode(); offset += size
            if mask & 0x4000:
                self.error('inotify queue overflow; creation provenance incomplete')
            if mask & (0x100 | 0x80) and name:
                if len(self.created) >= 4096:
                    self.error('SHM creation observation bound exceeded'); break
                try:
                    s = (Path('/dev/shm') / name).lstat()
                    self.created[(name, s.st_dev, s.st_ino)] = {'path': '/dev/shm/' + name,
                        'dev': s.st_dev, 'ino': s.st_ino, 'uid': s.st_uid}
                except FileNotFoundError:
                    continue

    def register(self, message, fd, peer):
        if not self.private or fd is None:
            raise RuntimeError('private registration requires an enabled pre-fork inventory and real FD')
        value = message['value']; p = Path(value['path'])
        if (p.parent != Path('/dev/shm') or p.name in self.baseline or p.is_symlink()
                or value.get('exclusive_create') is not True or type(value.get('original_fd')) is not int):
            raise RuntimeError('not a new exclusive task-private SHM creation')
        from task_lifecycle import WORKER_SHA256
        from multiprocessing import shared_memory
        own_sha = hashlib.sha256(Path(__file__).with_name('task_lifecycle.py').read_bytes()).hexdigest()
        shared_sha = hashlib.sha256(Path(shared_memory.__file__).read_bytes()).hexdigest()
        expected = {'explicit-task-private': own_sha, 'frozen-simpler-mailbox': WORKER_SHA256}
        if (value.get('kind') not in expected or value.get('source_sha256') != expected[value['kind']]
                or value.get('shared_source_sha256') != shared_sha):
            raise RuntimeError('unproven exclusive creation source')
        if value['kind'] == 'explicit-task-private':
            if not re.fullmatch('uf-task-[0-9a-f]{32}', p.name):
                raise RuntimeError('explicit private allocation incarnation mismatch')
        else:
            token = value.get('private_namespace')
            if type(token) is not str or not re.fullmatch('[0-9a-f]{32}', token) or not re.fullmatch(
                    'sp-' + token[:8] + '-(?:chip|sub|next)-[0-9]+', p.name):
                raise RuntimeError('frozen private mailbox namespace mismatch')
        s, path_stat = os.fstat(fd), p.lstat()
        live = Path('/proc', str(peer[0]), 'fd', str(value['original_fd'])).stat()
        if (not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or s.st_uid != peer[1]
                or (s.st_dev, s.st_ino) != (path_stat.st_dev, path_stat.st_ino)
                or (s.st_dev, s.st_ino) != (live.st_dev, live.st_ino)):
            raise RuntimeError('private FD/path/owned process UID or inode mismatch')
        key = (s.st_dev, s.st_ino)
        if key in self.records:
            return False
        if len(self.records) >= 64:
            raise RuntimeError('private SHM registration bound exceeded')
        record = {'path': str(p), 'dev': s.st_dev, 'ino': s.st_ino, 'uid': s.st_uid, 'bytes': s.st_size,
                  'creator': {k: message[k] for k in ('pid', 'birth')}, 'origin': value,
                  'original_fdinfo': Path('/proc', str(peer[0]), 'fdinfo', str(value['original_fd'])).read_text(),
                  'pin_fd': fd, 'attempt': self.attempt}
        self.records[key] = record
        self.event('private-SHM-registered', path=str(p), dev=s.st_dev, ino=s.st_ino,
                   creator_pid=peer[0], creator_birth=message['birth'], origin=value['kind'])
        return True

    def service(self, known, closing):
        self.poll_creations()
        for _ in range(8):
            try:
                conn, _ = self.listener.accept()
            except BlockingIOError:
                break
            passed, retained = [], False
            actor = None
            try:
                conn.settimeout(.01)
                peer = struct.unpack('3i', conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                body, control, flags, _ = conn.recvmsg(8192, socket.CMSG_SPACE(16))
                for level, kind, data in control:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        a = array.array('i'); a.frombytes(data[:len(data) - len(data) % a.itemsize]); passed.extend(a)
                if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
                    raise RuntimeError('truncated task lifecycle packet')
                message = json.loads(body)
                actor = known.get(peer[0]); now = identity(peer[0])
                if (actor is None or now is None or now['birth'] != actor['birth'] or peer[1] != self.monitor['uid']
                        or message['attempt'] != self.attempt or message['monitor_birth'] != self.monitor['birth']
                        or (message['leader_pid'], message['leader_birth']) != (self.leader['pid'], self.leader['birth'])
                        or (message['pid'], message['birth']) != (peer[0], actor['birth'])):
                    actor = None
                    raise RuntimeError('unowned/replayed lifecycle identity')
                if message['kind'] == 'closing' and not passed:
                    closing(peer[0], actor['birth'])
                elif message['kind'] == 'private-created' and len(passed) == 1:
                    retained = self.register(message, passed[0], peer)
                elif message['kind'] != 'bootstrap-ready' or passed:
                    raise RuntimeError('unsupported lifecycle message')
                conn.send(json.dumps({'accepted': True}).encode())
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                if actor is not None:
                    self.error(repr(error))
                self.event('lifecycle-refused', error=repr(error))
                try:
                    conn.send(json.dumps({'accepted': False}).encode())
                except OSError:
                    pass
            finally:
                if not retained:
                    for fd in passed:
                        os.close(fd)
                conn.close()

    def reclaim(self, owned, seconds, tree_empty):
        self.poll_creations()
        if not self.private:
            return {'enabled': False, 'cleanup_complete': True, 'registered': []}
        records = list(self.records.values())
        unproven = [r for key, r in self.created.items() if (key[1], key[2]) not in self.records
                    and self.baseline.get(key[0]) != (key[1], key[2]) and r['uid'] == self.monitor['uid']]
        config = {'registered': records, 'unproven': unproven, 'errors': self.errors,
                  'owned': owned, 'tree_empty': tree_empty, 'monitor': self.monitor, 'seconds': seconds,
                  'attempt': self.attempt, 'pins': [r['pin_fd'] for r in records]}
        path = self.root / 'private-reclaim-config.json'
        with path.open('x') as f:
            json.dump(config, f, indent=2)
        from task_supervisor import pidfd_signal, birth
        with (self.root / 'private-reaper.log').open('xb') as log:
            child = subprocess.Popen([sys.executable, '-B', '-s', str(Path(__file__).resolve()), str(path)],
                pass_fds=tuple(config['pins']), stdout=log, stderr=log)
            born = birth(child.pid)
            try:
                code = child.wait(timeout=seconds + 1)
            except subprocess.TimeoutExpired:
                import signal
                pidfd_signal(child.pid, born, signal.SIGKILL, expected_uid=os.geteuid())
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                return {'enabled': True, 'cleanup_complete': False, 'error': 'private audit/reclaim deadline',
                        'reaper_pid': child.pid, 'reaper_birth': born, 'registered': records}
        result_path = self.root / 'private-reclaim-result.json'
        if code != 0 or not result_path.exists():
            return {'enabled': True, 'cleanup_complete': False, 'error': 'private audit/reclaim failed', 'registered': records}
        return json.loads(result_path.read_bytes())

    def close(self):
        for row in self.records.values():
            os.close(row['pin_fd'])
        if self.watch is not None:
            os.close(self.watch)
        self.listener.close()
        if self.path.exists() and self.path.lstat().st_ino == self.socket_inode:
            self.path.unlink()


def reclaim(config):
    """No native cleanup; bounded metadata/reference audit, exact unlink only after reaping."""
    end = time.monotonic() + config['seconds']
    errors = list(config['errors']); results = []
    def bounded():
        if time.monotonic() >= end:
            raise TimeoutError('private kernel observation deadline')
    try:
        if not config['tree_empty'] or any((identity(int(pid)) or {}).get('birth') == born for pid, born in config['owned'].items()):
            raise RuntimeError('original owned tree not proven gone')
        for r in config['unproven']:
            bounded()
            p = Path(r['path'])
            if p.exists():
                errors.append('unregistered creation retained: ' + str(p))
        keys = {(r['dev'], r['ino']) for r in config['registered']}
        holders = {key: [] for key in keys}
        with os.scandir('/proc') as iterator:
            processes = [Path(p.path) for p in itertools.islice((p for p in iterator if p.name.isdecimal()), 8193)]
        if len(processes) > 8192:
            raise RuntimeError('process reference inventory bound')
        for proc in processes if keys else []:
            bounded()
            try:
                with (proc / 'maps').open('rb') as stream:
                    maps = stream.read((64 << 20) + 1)
                if len(maps) > 64 << 20:
                    raise RuntimeError('process maps inventory bound')
                for line in maps.decode().splitlines():
                    fields = line.split(None, 5)
                    major, minor = (int(s, 16) for s in fields[3].split(':'))
                    key = (os.makedev(major, minor), int(fields[4]))
                    if key in keys:
                        holders[key].append({'pid': int(proc.name), 'map': line})
                with os.scandir(proc / 'fd') as iterator:
                    fds = [Path(p.path) for p in itertools.islice(iterator, 4097)]
                if len(fds) > 4096:
                    raise RuntimeError('process FD inventory bound')
                for fd in fds:
                    bounded()
                    if int(proc.name) in (os.getpid(), config['monitor']['pid']) and int(fd.name) in config['pins']:
                        continue
                    try:
                        s = fd.stat(); key = (s.st_dev, s.st_ino)
                        if key in keys:
                            holders[key].append({'pid': int(proc.name), 'fd': int(fd.name)})
                    except (FileNotFoundError, ProcessLookupError):
                        continue
            except (FileNotFoundError, ProcessLookupError):
                continue
        for r in config['registered']:
            bounded(); p = Path(r['path']); row = {'path': str(p), 'dev': r['dev'], 'ino': r['ino'],
                                                  'holders': holders[(r['dev'], r['ino'])]}
            results.append(row)
            if row['holders']:
                row['status'] = 'external-reference-retained'; errors.append(str(row)); continue
            if not p.exists():
                row['status'] = 'already-absent'; continue
            s, pin = p.lstat(), os.fstat(r['pin_fd'])
            if (p.is_symlink() or not stat.S_ISREG(s.st_mode) or s.st_uid != r['uid'] or s.st_nlink != 1
                    or (s.st_dev, s.st_ino, s.st_size) != (r['dev'], r['ino'], r['bytes'])
                    or (pin.st_dev, pin.st_ino, pin.st_uid, pin.st_nlink) != (r['dev'], r['ino'], r['uid'], 1)):
                row['status'] = 'identity-drift-retained'; errors.append(str(row)); continue
            if errors:
                row['status'] = 'referenced-or-unobservable-retained'; errors.append(str(row)); continue
            # Recheck the exact pinned object and original owned births immediately
            # before unlink. Trusted same-UID task, never a malicious-root sandbox.
            if any((identity(int(pid)) or {}).get('birth') == born for pid, born in config['owned'].items()):
                raise RuntimeError('owned birth unexpectedly still live')
            current = p.lstat()
            if (current.st_dev, current.st_ino, current.st_uid) != (r['dev'], r['ino'], r['uid']):
                raise RuntimeError('private object changed at unlink boundary')
            bounded(); p.unlink()
            if os.fstat(r['pin_fd']).st_nlink != 0 or p.exists():
                raise RuntimeError('private unlink not proven')
            row['status'] = 'automatically-unlinked'
        return {'enabled': True, 'cleanup_complete': not errors, 'registered': results, 'errors': errors,
                'observation_seconds': config['seconds'], 'SDK_or_managed_DDR_mutation': False}
    except (OSError, RuntimeError, ValueError) as error:
        return {'enabled': True, 'cleanup_complete': False, 'registered': results, 'errors': [*errors, repr(error)],
                'SDK_or_managed_DDR_mutation': False}


if __name__ == '__main__':
    path = Path(sys.argv[1])
    result = reclaim(json.loads(path.read_bytes()))
    with path.with_name('private-reclaim-result.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'cleanup_complete': result['cleanup_complete']}))
