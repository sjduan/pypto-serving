# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test-only deployment-bound FD policy; inventory is always persisted first."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat

SHIM_SHA = '7f1d647e5fdbbba97b79c21271445383a207ac369fe44bc1025af4041fce420e'
DAEMON_SHA = 'be1f0c9d6fc28c5f7b83242836a021a6be584bb8958c1cce8b1a9d00fc5a9404'
CONTROLS = {'/dev/davinci_manager': ([235, 0], 3), '/dev/hisi_hdc': ([509, 0], 1)}
KEY_APIS = ('uf_acl_backend_init', 'aclInit', 'aclrtSetDevice', 'aclrtMalloc')
REAL_CLEANUP = 'uf_acl_backend_finalize'
PRISTINE_BLOCKED = {'aclInit', 'aclrtSetDevice', 'aclrtResetDevice', 'aclFinalize',
                    'aclrtDestroyEvent', 'aclrtDestroyStream', 'aclrtFreeHost', 'aclrtFreePhysical',
                    'aclrtUnmapMem', 'aclrtReleaseMemAddress', 'aclrtFree'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attempt_token(token):
    if not isinstance(token, str) or not re.fullmatch(r'attempt[2-9][0-9]*-[a-z][a-z0-9-]{0,40}', token):
        raise ValueError('strict named attempt token required')
    return token


def inventory(pid):
    result = {}
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            target, info = os.readlink(fd), fd.stat()
        except FileNotFoundError:
            continue
        char = stat.S_ISCHR(info.st_mode)
        result[fd.name] = {'target': target, 'is_char': char,
                           'rdev': [os.major(info.st_rdev), os.minor(info.st_rdev)] if char else None}
    return result


def deployment(pid, shim, fence):
    lines = Path(f'/proc/{pid}/maps').read_text().splitlines()
    paths = {line.split()[-1] for line in lines if '/' in line}
    expected = {'shim': str(Path(shim).resolve()), 'fence': str(Path(fence).resolve())}
    hashes = {}
    for key, path in expected.items():
        if path not in paths:
            raise RuntimeError('missing actual loaded '+key)
        hashes[key] = sha(path)
    for path in sorted(paths):
        if '/Ascend/' in path:
            hashes['cann:'+path] = sha(path)
    if not any(key.endswith('/libascendcl.so') for key in hashes):
        raise RuntimeError('actual CANN mapping missing')
    return hashes


def check_bindings(bindings, fence):
    if not set(KEY_APIS) <= set(bindings) or any(value != str(Path(fence).resolve()) for value in bindings.values()):
        raise RuntimeError('fence symbol binding mismatch')


def check_pristine_bindings(blocked, bindings, fence, shim, exported_uf):
    if (set(name for name in blocked if name.startswith('uf_acl_')) != set(exported_uf)-{REAL_CLEANUP}
            or REAL_CLEANUP in blocked or not PRISTINE_BLOCKED <= set(blocked)
            or set(bindings) != set(blocked)|{REAL_CLEANUP}):
        raise RuntimeError('only the named real pristine cleanup may be excluded')
    check_bindings({name: bindings[name] for name in blocked}, fence)
    if bindings[REAL_CLEANUP] != str(Path(shim).resolve()):
        raise RuntimeError('pristine finalizer must bind to the original actual shim')


def judge(fds, role, loaded=None, baseline=None):
    counts = Counter()
    for entry in fds.values():
        target = entry['target']
        if not target.startswith('/dev/'):
            continue
        if target in ('/dev/null', '/dev/zero', '/dev/random', '/dev/urandom', '/dev/tty') or re.fullmatch(r'/dev/pts/[0-9]+', target):
            continue
        if role != 'daemon' or target not in CONTROLS:
            raise RuntimeError('forbidden device FD '+repr(entry))
        rdev, _ = CONTROLS[target]
        if not entry['is_char'] or entry['rdev'] != rdev:
            raise RuntimeError('wrong control FD type/rdev '+repr(entry))
        counts[target] += 1
    if role == 'daemon':
        if counts != Counter({name: count for name, (_, count) in CONTROLS.items()}):
            raise RuntimeError('control FD count mismatch '+repr(counts))
        if not loaded or not baseline or loaded != baseline or loaded.get('shim') != SHIM_SHA or not loaded.get('fence'):
            raise RuntimeError('actual deployment/fence hash mismatch')


def record_and_check(report, label, pid, role, output, shim=None, fence=None, baseline=None):
    fds = inventory(pid)
    report.setdefault('owned_fds', {})[label] = fds
    output.write_text(json.dumps(report, indent=2)+'\n')
    loaded = deployment(pid, shim, fence) if role == 'daemon' else None
    report.setdefault('loaded_deployments', {})[label] = loaded
    output.write_text(json.dumps(report, indent=2)+'\n')
    judge(fds, role, loaded, baseline)
    return fds
