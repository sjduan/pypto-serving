# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded Linux proof of this client's inherited mappings and original child exit."""
import hashlib
import os
from pathlib import Path
import re
import stat

PINNED_PYTHON = '/workspace/pypto/.venv/bin/python'
PINNED_PYTHON_SHA256 = '1195c76f45c7cf16bf776a572cc85e074669c6f1a17d80d83fa8b7a7e3f80a87'


def process_birth(pid, *, proc=Path('/proc')):
    try:
        text = (proc / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # Process comm may contain spaces and parentheses. Fields after the final
    # ')' begin at stat field 3; starttime is field 22.
    tail = text.rpartition(')')[2].split()
    if len(tail) < 20:
        raise RuntimeError('cannot establish process birth identity')
    return int(tail[19])


def descendants(pid=None, *, proc=Path('/proc'), limit=64, depth=4):
    root = os.getpid() if pid is None else pid
    found, todo = {}, [(root, 0)]
    while todo:
        current, level = todo.pop()
        try:
            children = (proc / str(current) / 'task' / str(current) / 'children').read_text().split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if children and level >= depth:
            raise RuntimeError('owned native child tree exceeds declared depth bound')
        for child in map(int, children):
            birth = process_birth(child, proc=proc)
            if birth is None:
                raise RuntimeError('child exited before its inherited mapping was registered')
            if child in found:
                raise RuntimeError('child tree identity repeated')
            found[child] = birth
            if len(found) > limit:
                raise RuntimeError('owned native child count exceeds declared bound')
            todo.append((child, level + 1))
    return found


def has_pool_mapping(pid, identity, *, proc=Path('/proc')):
    # Physical backing, shared/RW flags, offset and full capacity are checked.
    # No parent or provider VA is sent to UFlow or compared across processes.
    expected_device = (os.major(identity.device), os.minor(identity.device))
    for line in (proc / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 5:
            continue
        addresses, access, offset, device, inode = fields[:5]
        major, minor = (int(part, 16) for part in device.split(':'))
        if (major, minor) != expected_device or int(inode) != identity.inode:
            continue
        start, end = (int(part, 16) for part in addresses.split('-'))
        if access == 'rw-s' and int(offset, 16) == 0 and end - start >= identity.nbytes:
            return True
    return False


def shared_mapping_size(address, nbytes, *, maps=Path('/proc/self/maps')):
    """Actual VMA length of the small control tensor, separately from KV pool."""
    for line in maps.read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 5:
            continue
        start, end = (int(part, 16) for part in fields[0].split('-'))
        if start <= address and address + nbytes <= end:
            if fields[1] != 'rw-s':
                raise RuntimeError('control metadata is not a shared RW mapping')
            return end - start
    raise RuntimeError('control metadata has no complete shared VMA')


def shared_mapping_resource(address, nbytes, *, maps=Path('/proc/self/maps')):
    shared_mapping_size(address, nbytes, maps=maps)
    for line in maps.read_text().splitlines():
        fields = line.split(None, 5)
        start, end = (int(part, 16) for part in fields[0].split('-'))
        if start <= address and address + nbytes <= end:
            major, minor = (int(part, 16) for part in fields[3].split(':'))
            return os.makedev(major, minor), int(fields[4])
    raise RuntimeError('control metadata physical resource disappeared')


def tracker_observation(pid, *, proc=Path('/proc')):
    """Bounded actual process evidence; never infer a role just from a process name."""
    base = proc / str(pid)
    before = process_birth(pid, proc=proc)
    status = (base / 'status').read_text()
    parent = int(re.search(r'^PPid:\s+(\d+)', status, re.M)[1])
    exe = base / 'exe'
    fds = []
    for fd in (base / 'fd').iterdir():
        info = fd.stat()
        fds.append({'fd': int(fd.name), 'target': os.readlink(fd), 'dev': info.st_dev,
                    'inode': info.st_ino, 'mode': info.st_mode, 'rdev': info.st_rdev})
    parent_stdio = {str(n): os.readlink(proc / str(parent) / 'fd' / str(n)) for n in range(3)}
    return {'pid': pid, 'birth': before, 'birth_after': process_birth(pid, proc=proc), 'ppid': parent,
        'exe': str(exe.resolve()), 'exe_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
        'cmdline': (base / 'cmdline').read_bytes().decode().split('\0')[:-1],
        'maps': (base / 'maps').read_text(), 'fds': fds, 'parent_stdio': parent_stdio}


def validate_tracker(value, pid, birth, parent, identity, control_resources=()):
    """The only non-holder exception: exact pinned, direct-child standard-library tracker."""
    if (value['pid'], value['birth'], value['birth_after'], value['ppid']) != (pid, birth, birth, parent):
        raise RuntimeError('resource tracker birth/direct-parent identity differs')
    if (value['exe'], value['exe_sha256']) != (PINNED_PYTHON, PINNED_PYTHON_SHA256):
        raise RuntimeError('resource tracker executable is not the pinned Python')
    argv = value['cmdline']
    if len(argv) != 5 or argv[:4] != [PINNED_PYTHON, '-B', '-s', '-c']:
        raise RuntimeError('unmapped child is not the exact standard-library tracker')
    match = re.fullmatch(r'from multiprocessing\.resource_tracker import main;main\(([0-9]+)\)', argv[4])
    if match is None or int(match[1]) < 3:
        raise RuntimeError('resource tracker command is not the fixed stdlib entry point')
    pipe_fd = int(match[1])
    if {x['fd'] for x in value['fds']} != {0, 1, 2, pipe_fd} or len(value['fds']) != 4:
        raise RuntimeError('resource tracker inherited unexpected control or pool descriptors')
    forbidden = {(identity.device, identity.inode), *map(tuple, control_resources)}
    for fd in value['fds']:
        if (fd['dev'], fd['inode']) in forbidden:
            raise RuntimeError('resource tracker holds a pool/control FD')
        if stat.S_ISCHR(fd['mode']) and fd['target'] != '/dev/null':
            raise RuntimeError('resource tracker holds a device FD')
        if fd['fd'] == pipe_fd:
            if not re.fullmatch(r'pipe:\[[0-9]+\]', fd['target']):
                raise RuntimeError('resource tracker control is not its stdlib pipe')
        elif fd['target'] != value['parent_stdio'][str(fd['fd'])]:
            raise RuntimeError('resource tracker stdio differs from its original parent')
    for line in value['maps'].splitlines():
        fields = line.split(None, 5)
        major, minor = (int(x, 16) for x in fields[3].split(':'))
        physical = (os.makedev(major, minor), int(fields[4]))
        if physical in forbidden or fields[1].endswith('s') and 'w' in fields[1]:
            raise RuntimeError('resource tracker holds a pool/control/shared writable VMA')
        if len(fields) == 6 and any(x in fields[5].lower() for x in (
                'ascend', '_task_interface', 'pypto_core', 'libhost_runtime', 'libacl', 'libaicpu', 'libaicore')):
            raise RuntimeError('resource tracker loaded a native/device module')
    return {'pid': pid, 'birth': birth, 'role': 'stdlib-resource-tracker',
            'pool_control_FD_VMA_absent': True, 'native_device_module_FD_absent': True}


def attach_new_children(access, baseline, *, helpers=None, control_resources=(), proc=Path('/proc')):
    current = descendants(proc=proc)
    born = {pid: birth for pid, birth in current.items() if baseline.get(pid) != birth}
    if not born:
        raise RuntimeError('native owner did not expose any inherited child mapping')
    try:
        registered = {}
        for pid, birth in born.items():
            if not has_pool_mapping(pid, access.identity, proc=proc):
                if helpers is None:
                    raise RuntimeError('unmapped child has no explicit helper lifecycle registry')
                observed = tracker_observation(pid, proc=proc)
                proof = validate_tracker(observed, pid, birth, os.getpid(), access.identity, control_resources)
                helpers[pid] = {**proof, 'observation': observed, 'control_resources': control_resources}
                continue
            access.attach_child(pid, birth)
            registered[pid] = birth
        if not registered or descendants(proc=proc) != current:
            raise RuntimeError('native mapping holder missing or child tree changed during registration')
    except Exception as error:
        access.report_unknown(None, f'inherited child registration incomplete: {error}')
        raise
    return registered


def stop_resource_trackers(helpers, *, proc=Path('/proc')):
    """After worker/control close only: normal stdlib pipe-close and waitpid, no signals."""
    if not helpers:
        return
    from multiprocessing import resource_tracker
    tracker = resource_tracker._resource_tracker
    for pid, proof in tuple(helpers.items()):
        if process_birth(pid, proc=proc) == proof['birth']:
            if tracker._pid != pid:
                raise RuntimeError('cannot normally reap a foreign resource tracker')
            tracker._stop()
        if process_birth(pid, proc=proc) == proof['birth']:
            raise RuntimeError('standard-library tracker did not exit after normal stop')
        del helpers[pid]


def detach_closed_children(access, registered, baseline, *, helpers=None, proc=Path('/proc')):
    remaining = descendants(proc=proc)
    helpers = {} if helpers is None else helpers
    if any(baseline.get(pid) != birth and (pid not in helpers or helpers[pid]['birth'] != birth)
           for pid, birth in remaining.items()):
        raise RuntimeError('owner close left inherited descendants alive')
    for pid in helpers:
        if pid in remaining:
            validate_tracker(tracker_observation(pid, proc=proc), pid, helpers[pid]['birth'], os.getpid(),
                             access.identity, helpers[pid].get('control_resources', ()))
    for pid, birth in tuple(registered.items()):
        if process_birth(pid, proc=proc) == birth:
            raise RuntimeError('original native child still lives after owner close')
        access.detach_stopped_child(pid, birth, owner_drained=True, process_gone=True)
        del registered[pid]
