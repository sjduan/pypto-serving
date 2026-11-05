# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded Linux proof of this client's inherited mappings and original child exit."""
import os
from pathlib import Path


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


def attach_new_children(access, baseline, *, proc=Path('/proc')):
    current = descendants(proc=proc)
    born = {pid: birth for pid, birth in current.items() if baseline.get(pid) != birth}
    if not born:
        raise RuntimeError('native owner did not expose any inherited child mapping')
    try:
        for pid, birth in born.items():
            if not has_pool_mapping(pid, access.identity, proc=proc):
                raise RuntimeError('native child has no whole shared RW mapping of the UFlow resource')
            access.attach_child(pid, birth)
    except Exception as error:
        access.report_unknown(None, f'inherited child registration incomplete: {error}')
        raise
    return born


def detach_closed_children(access, registered, baseline, *, proc=Path('/proc')):
    remaining = descendants(proc=proc)
    if any(baseline.get(pid) != birth for pid, birth in remaining.items()):
        raise RuntimeError('owner close left inherited descendants alive')
    for pid, birth in tuple(registered.items()):
        if process_birth(pid, proc=proc) == birth:
            raise RuntimeError('original native child still lives after owner close')
        access.detach_stopped_child(pid, birth, owner_drained=True, process_gone=True)
        del registered[pid]
