# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build and prove a test-only exit86 API fence against the actual deployment."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

from examples.run03.host_exchange_guard import (
    REAL_CLEANUP, check_pristine_bindings, deployment, inventory, judge, sha,
)


class DlInfo(ctypes.Structure):
    _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]


class Status(ctypes.Structure):
    _fields_ = [('code', ctypes.c_int32), ('message', ctypes.c_char * 256)]


def probe(shim, fence, symbols, positive):
    before = inventory('self')
    ctypes.CDLL(str(shim), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    handle = ctypes.CDLL(None)
    handle.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    bindings = {}
    for name in [*symbols, REAL_CLEANUP]:
        info = DlInfo()
        assert handle.dladdr(ctypes.cast(getattr(handle, name), ctypes.c_void_p), ctypes.byref(info))
        bindings[name] = str(Path(os.fsdecode(info.filename)).resolve())
    report = {'pid': os.getpid(), 'before_fds': before, 'after_fds': inventory('self'),
              'loaded': deployment('self', shim, fence), 'bindings': bindings}
    print(json.dumps(report), flush=True)  # Evidence survives even failed policy checks or exit86.
    exported = [name for name in symbols if name.startswith('uf_acl_')]+[REAL_CLEANUP]
    check_pristine_bindings(symbols, bindings, fence, shim, exported)
    judge(report['after_fds'], 'daemon', report['loaded'], report['loaded'])
    if positive:
        # Only call after dladdr proves the exact test fence owns this symbol.
        handle.uf_acl_backend_init(None, None)
        raise AssertionError('fence returned')
    before_cleanup = inventory('self')
    finalizer = getattr(handle, REAL_CLEANUP)
    finalizer.argtypes, finalizer.restype = [ctypes.POINTER(Status)], ctypes.c_int
    report['allowed_real_cleanup'] = {REAL_CLEANUP: bindings[REAL_CLEANUP]}
    report['pristine_real_results'] = []
    for _ in range(3):
        status = Status()
        result = finalizer(ctypes.byref(status))
        row = {'return': result, 'status': status.code, 'message': bytes(status.message).decode()}
        report['pristine_real_results'].append(row)
        assert row == {'return': 0, 'status': 0, 'message': 'ok'}, row
    report['after_real_cleanup_fds'] = inventory('self')
    assert report['after_real_cleanup_fds'] == before_cleanup
    print(json.dumps(report), flush=True)


def prepare(output, shim):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = Path(__file__).with_name('host_exchange_api_fence.c')
    exports = subprocess.check_output(['nm', '-D', '--defined-only', str(shim)], text=True)
    uf = {line.split()[-1] for line in exports.splitlines() if line.split()[-1].startswith('uf_acl_')}
    symbols = json.loads(Path(__file__).with_name('host_exchange_blocked_symbols.json').read_text())
    assert uf and uf == {name for name in symbols if name.startswith('uf_acl_')}|{REAL_CLEANUP}
    assert REAL_CLEANUP not in symbols
    text = source.read_text()
    assert all('void '+name+'(void)' in text for name in symbols)
    fence = output/'api_fence.so'
    compile_command = ['cc', '-shared', '-fPIC', '-O2', '-o', str(fence), str(source)]
    subprocess.run(compile_command, check=True, capture_output=True)
    fence_exports = subprocess.check_output(['nm', '-D', '--defined-only', str(fence)], text=True)
    assert {line.split()[-1] for line in fence_exports.splitlines()} == set(symbols)
    dynamic = subprocess.check_output(['readelf', '-d', str(fence)], text=True)
    assert all('libc.so' in line for line in dynamic.splitlines() if '(NEEDED)' in line)
    (output/'dynamic.txt').write_text(dynamic)
    (output/'blocked-symbols.json').write_text(json.dumps(symbols, indent=2)+'\n')
    env = dict(os.environ, LD_PRELOAD=str(fence),
               LD_LIBRARY_PATH=str(shim.parent)+':/usr/local/Ascend/cann-9.0.0/lib64')
    reports = {}
    for mode in ('loader', 'positive'):
        command = [sys.executable, '-m', 'examples.run03.host_exchange_fence', '--probe',
                   '--shim', str(shim), '--fence', str(fence), '--symbols', str(output/'blocked-symbols.json')]
        if mode == 'positive':
            command.append('--positive')
        run = subprocess.run(command, env=env, capture_output=True, timeout=30)
        (output/(mode+'.stdout')).write_bytes(run.stdout)
        (output/(mode+'.stderr')).write_bytes(run.stderr)
        assert run.returncode == (86 if mode == 'positive' else 0), (mode, run.returncode, run.stderr)
        assert run.stderr == (b'H2_FORBIDDEN_API:uf_acl_backend_init\n' if mode == 'positive' else b'')
        reports[mode] = json.loads(run.stdout.splitlines()[-1])
    result = {'loaded': reports['loader']['loaded'], 'fence_sha256': sha(fence),
              'source_sha256': sha(source), 'exported_uf_symbols': sorted(uf),
              'blocked_symbols': symbols, 'compile_command': compile_command,
              'all_bindings_verified': True, 'positive_exit': 86, 'passed': True,
              'allowed_real_cleanup': reports['loader']['allowed_real_cleanup'],
              'pristine_real_results': reports['loader']['pristine_real_results'],
              'pristine_whole_lifetime_condition': 'new process; every other UF entry and device-init/alloc/reset/cleanup API fenced'}
    (output/'baseline.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--positive', action='store_true')
    for name in ('shim', 'fence', 'symbols'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if not args.probe:
        raise SystemExit('probe only; prepare is invoked by guarded outer')
    probe(args.shim, args.fence, json.loads(args.symbols.read_text()), args.positive)
