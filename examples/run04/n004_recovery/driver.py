# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One-shot hash-checked pending-callback installer; never calls close inside stopped GDB threads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

TOOL = Path('/workspace/uflow/runs/d1-managed-ddr-20261105/serving/manager-n004-recovery-c001/tool')
GDB_SHA = '6b98af7449692db116b67aaa6bc128f7afcaf76ff7a0263969b30209d48c5b79'
PYTHON = '/workspace/pypto/.venv/bin/python'


def save(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def birth(pid):
    try:
        return int(Path('/proc', str(pid), 'stat').read_text().rpartition(')')[2].split()[19])
    except FileNotFoundError:
        return None


def tracer(pid):
    return int(next(s.split()[1] for s in Path('/proc', str(pid), 'status').read_text().splitlines()
                    if s.startswith('TracerPid:')))


def await_file(path, pid, expected_birth, timeout=30):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert birth(pid) == expected_birth, 'original process unexpectedly exited/reused'
        assert time.monotonic() < deadline, str(path)
        time.sleep(.05)
    return json.loads(path.read_text())


def install(root, shared):
    cfg = json.loads((root / 'config.json').read_text())
    pid, expected = cfg['pid'], cfg['birth']
    assert birth(pid) == expected and tracer(pid) == 0
    tool = TOOL / 'usr/libexec/gdb'
    assert hashlib.sha256(tool.read_bytes()).hexdigest() == GDB_SHA
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (shared / 'payload.py', shared / 'actions.py', root / 'config.json')}
    # Catch even loader/open/syntax errors so the pending callback always
    # returns normally to the original failure pause. No close action here.
    loader = 'try:\n'
    loader += ' import hashlib as _ufload_hashlib\n'
    loader += ' from pathlib import Path as _ufload_Path\n'
    loader += ' _ufload_checks=' + repr(files) + '\n'
    loader += ' for _ufload_name,_ufload_digest in _ufload_checks.items():\n  assert _ufload_hashlib.sha256(_ufload_Path(_ufload_name).read_bytes()).hexdigest()==_ufload_digest\n'
    loader += ' _ufload_ns={"__name__":"uflow_task_recovery","__file__":' + repr(str(shared / 'payload.py'))
    loader += ',"_UFLOW_RECOVERY_ROOT":' + repr(str(root)) + '}\n'
    loader += ' globals()["_UFLOW_N004_RECOVERY_NS"]=_ufload_ns\n'
    loader += ' exec(compile(_ufload_Path(' + repr(str(shared / 'payload.py')) + ').read_bytes(),'
    loader += repr(str(shared / 'payload.py')) + ',"exec"),_ufload_ns)\n'
    loader += 'except BaseException as _ufload_error:\n'
    loader += ' try:\n  import json as _ufload_json,os as _ufload_os,traceback as _ufload_traceback\n'
    loader += '  from pathlib import Path as _ufload_Path\n'
    loader += '  _ufload_Path(' + repr(str(root / 'callback-loader-failed.json')) + ').write_text(_ufload_json.dumps({"pid":_ufload_os.getpid(),"error":repr(_ufload_error),"traceback":_ufload_traceback.format_exc(),"error_swallowed":True}))\n'
    loader += ' except BaseException:\n  pass\n'
    page = os.sysconf('SC_PAGESIZE')
    assert len(loader.encode()) + 1 < page
    save(root / 'installation-inputs.json', {'pid': pid, 'birth': expected, 'files': files,
        'gdb_sha256': GDB_SHA, 'loader_bytes': len(loader.encode()) + 1, 'CPU_bootstrap_mapping_bytes': page,
        'scope': 'pending callback installs only; no direct GIL/PyRun/Worker close in stopped debugger'})
    commands = ['set pagination off', 'set confirm off', 'set auto-solib-add off',
        'set auto-load python-scripts off', 'set debuginfod enabled off', 'set sysroot /', 'python',
        'import gdb', 'try:', f' gdb.execute("attach {pid}")',
        ' gdb.execute("sharedlibrary libpython3.11")', ' gdb.execute("sharedlibrary libc.so.6")',
        f' matches=[t for t in gdb.selected_inferior().threads() if t.ptid[1]=={pid}]',
        ' assert len(matches)==1', ' matches[0].switch()',
        ' gdb.execute("info threads")', ' gdb.execute("set scheduler-locking on")',
        f' ptr=gdb.parse_and_eval("(void *) mmap((void *)0, {page}, 3, 34, -1, 0)")',
        ' assert int(ptr) not in (0,-1,2**64-1)',
        ' gdb.execute("set $ufbuf = " + str(int(ptr)))',
        ' gdb.execute(' + repr('set {char[' + str(len(loader.encode()) + 1) + ']} $ufbuf = ' + json.dumps(loader)) + ')',
        ' queued=int(gdb.parse_and_eval("(int) Py_AddPendingCall((int (*)(void *)) PyRun_SimpleString, (void *)$ufbuf)"))',
        ' gdb.write("UFLOW_ENQUEUED=%d\\n" % queued)', ' assert queued==0', 'finally:',
        ' if gdb.selected_inferior().pid:', '  gdb.execute("detach")',
        ' gdb.write("UFLOW_DETACHED\\n")', 'end', 'quit']
    command_file = root / 'gdb.commands'
    command_file.write_text('\n'.join(commands) + '\n')
    env = {**os.environ, 'LD_LIBRARY_PATH': str(TOOL / 'usr/lib64')}
    with (root / 'gdb.log').open('xb') as output:
        proc = subprocess.Popen([str(tool), '--data-directory=' + str(TOOL / 'usr/share/gdb'),
            '-q', '-nx', '-nh', '-batch', '-x', str(command_file)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            code = proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            # Interrupt our debugger, not the native process, so its finally
            # block can detach. Never kill/reset the target or blindly retry.
            proc.send_signal(signal.SIGINT)
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                save(root / 'gdb-timeout-retained.json', {'gdb_pid': proc.pid, 'target_pid': pid,
                    'target_birth': birth(pid), 'TracerPid': tracer(pid), 'no_target_kill': True})
                raise RuntimeError('debugger still active; retain scene and report') from None
    log = (root / 'gdb.log').read_text(errors='replace')
    save(root / 'gdb-exit.json', {'pid': proc.pid, 'exit_code': code, 'TracerPid': tracer(pid),
                                'target_birth': birth(pid), 'target_pid': pid})
    assert code == 0 and 'UFLOW_ENQUEUED=0' in log and 'UFLOW_DETACHED' in log
    assert birth(pid) == expected and tracer(pid) == 0
    os.kill(pid, signal.SIGINT)  # Wake the existing caught-KeyboardInterrupt pause; no cleanup yet.
    deadline = time.monotonic() + 20
    while not (root / 'handler-installed.json').exists() and not (root / 'install-failed.json').exists() and not (root / 'callback-loader-failed.json').exists():
        assert birth(pid) == expected and time.monotonic() < deadline
        time.sleep(.05)
    return (root / 'handler-installed.json').exists()


def request_cleanup(root):
    cfg = json.loads((root / 'config.json').read_text())
    installed = json.loads((root / 'handler-installed.json').read_text())
    assert installed['pid'] == cfg['pid'] and installed['birth'] == cfg['birth']
    assert birth(cfg['pid']) == cfg['birth'] and tracer(cfg['pid']) == 0
    os.kill(cfg['pid'], signal.SIGUSR1)
    deadline = time.monotonic() + 90
    while not (root / 'cleanup-success.json').exists() and not (root / 'cleanup-failed-1.json').exists():
        assert time.monotonic() < deadline
        if birth(cfg['pid']) != cfg['birth']:
            assert (root / 'cleanup-success.json').exists(), 'unexpected process death without receipt'
        time.sleep(.05)
    return (root / 'cleanup-success.json').exists()


def fixtures(root, shared):
    root.mkdir()
    summaries = []
    for name, install_fault, close_fault in [('install-failure', True, False),
                                             ('close-failure', False, True), ('success', False, False)]:
        case = root / name
        case.mkdir()
        cfg = {'mode': 'cpu-fixture', 'inject_install_failure': install_fault, 'inject_close_failure': close_fault}
        (case / 'config.json').write_text(json.dumps(cfg))
        with (case / 'parent.log').open('xb') as log:
            child = subprocess.Popen([PYTHON, '-B', '-s', str(shared / 'fixture_parent.py'), str(case)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'})
        ready = await_file(case / 'ready.json', child.pid, birth(child.pid))
        await_file(case / 'pause-ready.json', child.pid, birth(child.pid))
        cfg.update(pid=child.pid, birth=birth(child.pid), physical=ready['physical'],
            tracker_pid=ready['tracker_pid'], tracker_birth=birth(ready['tracker_pid']), mapping_children={},
            target_source=str(shared / 'fixture_parent.py'), expected_error='CPU-only original consumer',
            allowed_unwound_sources=[str(shared / 'fixture_parent.py')],
            source_hashes={str(shared / 'fixture_parent.py'): hashlib.sha256((shared / 'fixture_parent.py').read_bytes()).hexdigest()},
            unregister_operation='fixture-unregister-' + name, unmap_operation='fixture-unmap-' + name)
        (case / 'config.json').write_text(json.dumps(cfg, indent=2))
        installed = install(case, shared)
        assert installed is not install_fault
        success = False
        if installed:
            success = request_cleanup(case)
            assert success is not close_fault
        if install_fault or close_fault:
            assert birth(child.pid) == cfg['birth']
            if close_fault:
                failed = json.loads((case / 'cleanup-failed-1.json').read_text())
                assert failed['stage'] == 'actual-view-close' and 'BufferError' in failed['error']
                assert not failed['ACK_done'] and not failed['view_closed']
            else:
                assert json.loads((case / 'install-failed.json').read_text())['pending_callback_error_swallowed']
            # Fixture-only preinstalled handler performs real final CPU cleanup.
            os.kill(child.pid, signal.SIGTERM)
            assert child.wait(timeout=15) == 0
            assert json.loads((case / 'fixture-graceful-stop.json').read_text())['payload_ACK_count'] == 0
        else:
            assert child.wait(timeout=15) == 0
            receipt = json.loads((case / 'cleanup-success.json').read_text())
            assert receipt['mode'] == 'cpu-fixture' and receipt['snapshot']['ack_count'] == 1
        assert birth(child.pid) is None and birth(cfg['tracker_pid']) is None
        summaries.append({'case': name, 'install': installed, 'cleanup_success': success,
            'failure_kept_original_process_alive': install_fault or close_fault,
            'fixture_final_actual_mmap_close_and_parent_helper_exit': True, 'UFlow_device_operations': 0})
    save(root / 'fixtures-result.json', {'passed': True, 'same_real_payload_tested': True,
                                        'scope': 'CPU fixture only', 'cases': summaries})
    print(json.dumps({'CPU_fixtures_passed': True, 'cases': summaries}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['fixtures'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    args = parser.parse_args()
    fixtures(args.root, args.shared)
