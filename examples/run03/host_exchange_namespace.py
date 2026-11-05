# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Task-bounded private namespace wrapper, derived from the reviewed E10 guard.

Never invokes a daemon/bootstrap before verifying the child namespace and the
pinned task directory. The real outer authority is hidden, not recovered.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import time
import traceback

from examples.run03.host_exchange_fence import prepare
from examples.run03.host_exchange_guard import DAEMON_SHA, SHIM_SHA, attempt_token, record_and_check
from examples.run03.host_exchange_lifecycle import OuterProtection, StopReader, owned_tree, supervise_owned

TASK = Path('/workspace/uflow/runs/run03-host-exchange-20261105')
RUN = Path('/workspace/uflow/runs/host-exchange')
AUTHORITY = Path('/workspace/uflow/authority/external-ddr-v0')


def identity(path):
    st = path.stat(follow_symlinks=False)
    return st.st_dev, st.st_ino


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_propagation():
    mounts = Path('/proc/self/mountinfo').read_text()
    if any(field.startswith(('shared:', 'master:', 'propagate_from:'))
           for line in mounts.splitlines() for field in line.split(' - ', 1)[0].split()[6:]):
        raise RuntimeError('mount propagation is not private')


def require_isolated(outer_fd, outer_pid):
    if type(outer_fd) is not int or type(outer_pid) is not int or outer_fd < 0 or outer_pid <= 0:
        raise RuntimeError('outer namespace descriptor and live parent required')
    if not os.readlink(f'/proc/self/fd/{outer_fd}').startswith('mnt:['):
        raise RuntimeError('outer mount namespace descriptor required')
    anchor = os.fstat(outer_fd)
    outer, current = os.stat(f'/proc/{outer_pid}/ns/mnt'), os.stat('/proc/self/ns/mnt')
    if (os.getppid() != outer_pid or (outer.st_dev, outer.st_ino) != (anchor.st_dev, anchor.st_ino)
            or (current.st_dev, current.st_ino) == (anchor.st_dev, anchor.st_ino)):
        raise RuntimeError('mount namespace was not isolated')
    private_propagation()


def control(endpoint, command):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(10)
        sock.connect(endpoint)
        sock.sendall((command+'\n').encode())
        data = bytearray()
        while not data.endswith(b'\n'):
            part = sock.recv(65536)
            if not part or len(data)+len(part) > 1048576:
                raise RuntimeError('incomplete bounded daemon response')
            data.extend(part)
        return data.decode()


def prior_clean():
    previous = json.loads((TASK/'results/namespace.json').read_text())
    assert previous['outer_protection']['unchanged'] and previous['child_supervision']['reaped']
    assert identity(Path('/workspace/uflow')) == tuple(previous['outer_protection']['after_inode'])
    assert os.stat('/proc/self/ns/mnt').st_ino == previous['outer_protection']['after_namespace'][1]
    roots = [TASK/'sandbox/nsroot']
    roots += sorted(path/'nsroot' for path in (TASK/'sandbox').iterdir() if path.name.startswith('attempt'))
    proofs = []
    for root in roots:
        fixture = root/'runs/host-exchange'
        result = json.loads((fixture/'result.json').read_text())
        assert not any(key.startswith('unresolved') for key in result)
        assert all(item['exit_code'] is not None and item['pid_absent'] for item in result.get('owned_process_exit', []))
        pids = [result.get('daemon_pid')] + [result.get('peers', {}).get(key) for key in ('producer_pid', 'consumer_pid')]
        assert all(not Path(f'/proc/{pid}').exists() for pid in pids if pid)
        assert not (fixture/'u.sock').exists() and not (fixture/'u.sock.external-ddr').exists()
        authority = root/'authority/external-ddr-v0'
        state = json.loads((authority/'state').read_text())
        assert state['mode'] == 'clean' and state['closure'] == 'normal' and state['pending'] is None
        with (authority/'authority.lock').open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        proofs.append({'root': str(root), 'state': state, 'pids_absent': pids, 'sockets_absent': True,
                       'lock_reacquired_released': True})
    return proofs


class Peer:
    def __init__(self, role, endpoint, env, fds, stop):
        self.stop = stop
        self.log = (RUN / f'{role}.log').open('x')
        command = [str(RUN/'venv/bin/python'), '-u', '-m', 'examples.run03.host_exchange_peer', role, endpoint]
        if role == 'consumer':
            command.append(str(RUN/'consumer-result.json'))
        try:
            self.proc = subprocess.Popen(command, env=dict(env, UFLOW_H2_PEER_ROLE=role), cwd=RUN/'serving', pass_fds=fds,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                         text=True, bufsize=1)
        except BaseException:
            self.log.close()
            raise

    def send(self, **message):
        self.stop.check()
        self.proc.stdin.write(json.dumps(message)+'\n')
        self.proc.stdin.flush()

    def receive(self, expected):
        deadline = time.monotonic()+30
        while True:
            self.stop.check()
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{self.proc.pid} did not emit {expected}')
            if self.proc.stdout in select.select([self.proc.stdout, self.stop.fd], [], [], .1)[0]:
                break
        row = self.proc.stdout.readline()
        if not row:
            raise RuntimeError(f'peer exited before {expected}: {self.proc.poll()}')
        result = json.loads(row)
        assert result['event'] == expected, result
        return result


def inner(args):
    stop = StopReader(args.control_fd)
    stop.check()
    # Exact E10 T3 order: namespace/parent first, then canonical inode, then mount.
    require_isolated(args.outer_fd, args.outer_pid)
    nsroot = args.nsroot.resolve(strict=True)
    if nsroot != args.nsroot or nsroot != TASK/'sandbox'/attempt_token(args.attempt)/'nsroot':
        raise RuntimeError('not the canonical task-owned namespace root')
    pinned = os.fstat(args.root_fd)
    if identity(nsroot) != (pinned.st_dev, pinned.st_ino):
        raise RuntimeError('task root fd/inode changed')
    authority_parent = identity(nsroot/'authority')
    subprocess.run(['mount', '--bind', str(nsroot), '/workspace/uflow'], check=True)
    assert identity(Path('/workspace/uflow')) == (pinned.st_dev, pinned.st_ino)
    assert identity(AUTHORITY.parent) == authority_parent
    assert not AUTHORITY.exists()
    assert sha(RUN/'uf-daemon') == args.daemon_sha and sha(RUN/'shim/libuf_acl_shim.so') == args.shim_sha
    env = dict(os.environ, UF_EXTERNAL_DDR_HOST_ID='run03-isolated-host',
               UF_EXTERNAL_DDR_OWNER_MAX_BYTES=str(96*1024*1024),
               UF_EXTERNAL_DDR_TOTAL_MAX_BYTES=str(256*1024*1024),
               UF_HBM_STARTUP_PROBE_BYTES='0', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               PYTHONPATH=f'{RUN}/serving:{RUN}/sdk', TMPDIR=str(RUN/'tmp'),
               LD_LIBRARY_PATH=f'{RUN}/shim:/usr/local/Ascend/cann-9.0.0/lib64',
               LD_PRELOAD=str(RUN/'api_fence.so'),
               UFLOW_H2_OUTER_FD=str(args.outer_fd), UFLOW_H2_ROOT_FD=str(args.root_fd),
               UFLOW_H2_OUTER_PID=str(args.outer_pid), UFLOW_H2_INNER_PID=str(os.getpid()))
    env['UFLOW_H2_SDK_SHA256'] = sha(RUN/'sdk/uflow/external_ddr.py')
    endpoint, binary = str(RUN/'u.sock'), str(RUN/'uf-daemon')
    baseline = json.loads((RUN/'fence-baseline.json').read_text())
    assert sha(RUN/'api_fence.so') == baseline['fence_sha256']
    report = {'passed': False, 'attempt': args.attempt, 'namespace_isolated': True, 'authority_absent_before_bootstrap': True,
              'outer_namespace_inode': os.fstat(args.outer_fd).st_ino,
              'inner_namespace_inode': os.stat('/proc/self/ns/mnt').st_ino,
              'mounted_inode': identity(Path('/workspace/uflow')), 'peers': {}, 'owned_fds': {}}
    def process_fds(pid, label, role='daemon'):
        return record_and_check(report, label, pid, role, RUN/'result.json',
                                RUN/'shim/libuf_acl_shim.so', RUN/'api_fence.so', baseline['loaded'])

    bootstrap = subprocess.run([binary, '--socket', endpoint, '--initialize-external-ddr-authority'],
                               env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    (RUN/'bootstrap.log').write_bytes(bootstrap.stdout)
    assert bootstrap.returncode == 0, bootstrap.stdout
    report['bootstrap_exit'] = bootstrap.returncode
    peers, daemon = [], None
    try:
        stop.check()
        with (RUN/'daemon.log').open('x') as log:
            daemon = subprocess.Popen([binary, '--socket', endpoint, '--execution-mode', 'simulated',
                                       '--startup-probe-bytes', '0'], env=env,
                                      stdout=log, stderr=subprocess.STDOUT)
            report['daemon_pid'] = daemon.pid
            deadline = time.monotonic()+20
            while not Path(endpoint+'.external-ddr').exists():
                stop.check()
                assert daemon.poll() is None and time.monotonic() < deadline
                time.sleep(.02)
            report['owned_fds']['daemon-before'] = process_fds(daemon.pid, 'daemon-before')
            report['daemon_exe'] = os.readlink(f'/proc/{daemon.pid}/exe')
            assert report['daemon_exe'] == binary
            assert sha(Path(f'/proc/{daemon.pid}/exe')) == DAEMON_SHA
            daemon_env = Path(f'/proc/{daemon.pid}/environ').read_bytes().split(b'\0')
            assert ('LD_PRELOAD='+str(RUN/'api_fence.so')).encode() in daemon_env
            report['daemon_preload_verified'] = True
            report['loaded_shim_and_cann'] = [line for line in Path(f'/proc/{daemon.pid}/maps').read_text().splitlines()
                                               if 'libuf_acl_shim' in line or 'libascendcl' in line]
            assert any(str(RUN/'shim/libuf_acl_shim.so') in line for line in report['loaded_shim_and_cann'])
            report['loaded_shim_sha256'] = sha(RUN/'shim/libuf_acl_shim.so')
            assert report['loaded_shim_sha256'] == args.shim_sha
            producer = Peer('producer', endpoint, env, (args.outer_fd, args.root_fd), stop)
            peers.append(producer)
            registered = producer.receive('registered')
            consumer = Peer('consumer', endpoint, env, (args.outer_fd, args.root_fd), stop)
            peers.append(consumer)
            consumer.send(op='begin', record=registered['record'], authority_source=registered['authority_source'])
            held = consumer.receive('held')
            assert registered['pid'] != consumer.proc.pid
            report['owned_fds']['producer-held'] = process_fds(producer.proc.pid, 'producer-held', 'producer')
            report['owned_fds']['consumer-held'] = process_fds(consumer.proc.pid, 'consumer-held', 'consumer')
            report['owned_fds']['daemon-held'] = process_fds(daemon.pid, 'daemon-held')
            producer.send(op='verify-and-revoke')
            revoked = producer.receive('revoked')
            consumer.send(op='revoked')
            consumer.receive('old-view-live-after-revoke')
            producer.send(op='exit')
            producer.receive('exiting')
            assert producer.proc.wait(timeout=15) == 0
            consumer.send(op='owner-exited')
            done = consumer.receive('consumer-done')
            assert consumer.proc.wait(timeout=15) == 0
            report['peers'] = {'registered': registered, 'held': held, 'revoked': revoked, 'done': done,
                               'producer_pid': producer.proc.pid, 'consumer_pid': consumer.proc.pid}
            report['stats_before_shutdown'] = control(endpoint, 'op=GetStats')
            fields = dict(part.split('=', 1) for part in report['stats_before_shutdown'].strip().split(';') if '=' in part)
            assert int(fields['external_mapped_bytes']) > 0 and int(fields['external_copied_bytes']) == 0
            report['mapped_is_not_copied'] = True
            report['owned_fds']['daemon-final'] = process_fds(daemon.pid, 'daemon-final')
            assert 'status=ok' in control(endpoint, 'op=ShutdownDaemon')
            assert daemon.wait(timeout=30) == 0
        assert 'H2_FORBIDDEN_API' not in (RUN/'daemon.log').read_text()
        assert 'H2_FORBIDDEN_API' not in (RUN/'bootstrap.log').read_text()
        assert 'external DDR authority drained and clean' in (RUN/'daemon.log').read_text()
        assert not Path(endpoint).exists() and not Path(endpoint+'.external-ddr').exists()
        assert all(not Path(f'/proc/{peer.proc.pid}').exists() for peer in peers)
        assert not Path(f'/proc/{daemon.pid}').exists()
        report.update(passed=True, clean_log=True, owned_processes_exited=True, sockets_removed=True)
    except BaseException:
        report['primary_error'] = traceback.format_exc()
        raise
    finally:
        for peer in reversed(peers):
            if peer.proc.poll() is None:
                try:
                    peer.proc.stdin.close()  # consumer explicitly closes stage on EOF
                    peer.proc.wait(timeout=15)
                except BaseException as cleanup_error:
                    report.setdefault('unresolved_peers', []).append({'pid': peer.proc.pid,
                                                                      'error': repr(cleanup_error),
                                                                      'identity': owned_tree(peer.proc.pid)})
            peer.log.close()
        if daemon is not None and daemon.poll() is None:
            try:
                report['cleanup_shutdown'] = control(endpoint, 'op=ShutdownDaemon')
                daemon.wait(timeout=30)
            except Exception as error:
                report['unresolved_daemon'] = {'pid': daemon.pid, 'error': repr(error),
                                               'identity': owned_tree(daemon.pid)}
        consumer_report = RUN/'consumer-result.json'
        report['owned_process_exit'] = [
            {'pid': process.pid, 'exit_code': process.poll(), 'pid_absent': not Path(f'/proc/{process.pid}').exists()}
            for process in [*(peer.proc for peer in peers), *([daemon] if daemon is not None else [])]]
        if consumer_report.exists():
            try:
                consumer_cleanup = json.loads(consumer_report.read_text())
                report['consumer_exit_cleanup'] = {key: consumer_cleanup[key] for key in (
                    'stage_exit_cleanup', 'unresolved_stage', 'failed_prepare_cleanups',
                    'unresolved_unexpected_views', 'primary_error') if key in consumer_cleanup}
            except BaseException as error:
                report['consumer_cleanup_read_error'] = repr(error)
        (RUN/'result.json').write_text(json.dumps(report, indent=2)+'\n')


def outer(token):
    token = attempt_token(token)
    logdir, resultdir = TASK/'logs'/token, TASK/'results'/token
    logdir.mkdir(exist_ok=False)
    resultdir.mkdir(exist_ok=False)
    result = {'passed': False, 'outer_pid': os.getpid()}
    protection = OuterProtection(result, Path('/workspace/uflow'), Path('/proc/self/mountinfo'),
                                  Path('/proc/self/ns/mnt'))
    fd = nsfd = proc = None
    try:
        with protection:
            if TASK.resolve(strict=True) != TASK:
                raise RuntimeError('task root is not canonical')
            result['prior_clean_proofs'] = prior_clean()
            source, nsroot = Path(__file__).resolve(), TASK/'sandbox'/token/'nsroot'
            assert sha(TASK/'build/bin/uf-daemon') == DAEMON_SHA
            assert sha(TASK/'build/shim/lib/libuf_acl_shim.so') == SHIM_SHA
            result['fence_baseline'] = prepare(TASK/'build'/token, TASK/'build/shim/lib/libuf_acl_shim.so')
            nsroot.mkdir(mode=0o700, parents=True, exist_ok=False)
            (nsroot/'authority').mkdir(mode=0o700)
            fixture = nsroot/'runs/host-exchange'
            fixture.mkdir(mode=0o700, parents=True)
            for name in ('shim', 'tmp'):
                (fixture/name).mkdir(mode=0o700)
            shutil.copy2(TASK/'build'/token/'api_fence.so', fixture/'api_fence.so')
            shutil.copy2(TASK/'build'/token/'baseline.json', fixture/'fence-baseline.json')
            shutil.copy2(TASK/'build/bin/uf-daemon', fixture/'uf-daemon')
            shutil.copy2(TASK/'build/shim/lib/libuf_acl_shim.so', fixture/'shim/libuf_acl_shim.so')
            shutil.copytree(TASK/'src/data-service/sdk/python/uflow', fixture/'sdk/uflow',
                            ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(TASK/'src/serving', fixture/'serving', symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '.git'))
            subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(fixture/'venv')], check=True)
            fd = os.open(nsroot, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            nsfd = os.open('/proc/self/ns/mnt', os.O_RDONLY | os.O_CLOEXEC)
            result.update(pinned_root=identity(nsroot), outer_namespace_inode=os.fstat(nsfd).st_ino)
            command = ['unshare', '--mount', '--propagation', 'private', sys.executable, str(source),
                       '--inner', '--attempt', token, '--control-fd', '0', '--nsroot', str(nsroot), '--root-fd', str(fd),
                       '--outer-fd', str(nsfd), '--outer-pid', str(os.getpid()),
                       '--daemon-sha', sha(fixture/'uf-daemon'), '--shim-sha', sha(fixture/'shim/libuf_acl_shim.so')]
            with (logdir/'namespace-child.log').open('x') as log:
                proc = subprocess.Popen(command, pass_fds=(fd, nsfd), stdin=subprocess.PIPE,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['child_supervision'] = supervise_owned(proc, timeout=240, cleanup_grace=120)
            assert result['child_supervision']['passed'], result['child_supervision']
            inner_result = json.loads((fixture/'result.json').read_text())
            assert inner_result['passed']
            with (nsroot/'authority/external-ddr-v0/authority.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            result.update(private_lock_reacquired_released=True, passed=True)
    except BaseException:
        result.update(passed=False, primary_error=traceback.format_exc())
        raise
    finally:
        if proc is not None and proc.poll() is None and 'child_supervision' not in result:
            result['exception_cleanup'] = supervise_owned(proc, timeout=0, cleanup_grace=120)
        fixture_report = TASK/'sandbox'/token/'nsroot/runs/host-exchange/result.json'
        if fixture_report.exists():
            try:
                result['inner_report'] = json.loads(fixture_report.read_text())
            except BaseException as error:
                result['inner_report_read_error'] = repr(error)
        if protection.before_mounts is not None:
            protection.capture_after()  # Also after any exception-path child cleanup.
            if not protection.report.get('unchanged'):
                result['passed'] = False
        for anchor in (fd, nsfd):
            if anchor is not None:
                os.close(anchor)
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError as error:
                result['control_close_error'] = repr(error)
        (resultdir/'namespace.json').write_text(json.dumps(result, indent=2)+'\n')
        if protection.before_mounts is not None:
            (resultdir/'mountinfo.before').write_bytes(protection.before_mounts)
        if protection.after_mounts is not None:
            (resultdir/'mountinfo.after').write_bytes(protection.after_mounts)
    if not result['passed']:
        raise RuntimeError('outer workflow not clean; inspect namespace.json')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inner', action='store_true')
    parser.add_argument('--attempt', required=True, type=attempt_token)
    parser.add_argument('--nsroot', type=Path)
    parser.add_argument('--control-fd', type=int, default=0)
    for flag in ('root-fd', 'outer-fd', 'outer-pid'):
        parser.add_argument('--'+flag, type=int)
    for flag in ('daemon-sha', 'shim-sha'):
        parser.add_argument('--'+flag)
    arguments = parser.parse_args()
    inner(arguments) if arguments.inner else outer(arguments.attempt)
