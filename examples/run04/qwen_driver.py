# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run04 plan/check (stdlib only), or explicitly admitted future device execution."""
from __future__ import annotations

import argparse
from pathlib import Path

from qwen_contract import (CASES, cli_arguments, execution_identity, load, plan,
                           require_B_execution, require_release, save, validate_config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'check', 'run', 'consumer'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--hold', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--existing-state', type=Path)
    parser.add_argument('--daemon', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--descriptor', type=Path)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--endpoint')
    parser.add_argument('--listen-fd', type=int)
    args = parser.parse_args(argv)
    config = validate_config(load(args.config))
    if args.mode == 'plan':
        if args.snapshot is None and config['schema'] == 1:
            parser.error('plan requires a sealed --snapshot; it never connects to the service')
        save(args.output, plan(config, load(args.snapshot) if args.snapshot else None))
    elif args.mode == 'check':
        if any(v is None for v in (args.report, args.descriptor, args.log)):
            parser.error('check requires --report, --descriptor and --log')
        from qwen_evidence import check
        save(args.output, check(config, load(args.report), load(args.descriptor),
                                args.report.parent, args.log.read_text()))
    else:
        # This precedes runtime imports, root creation, sockets, SDK and Worker.
        if args.hold is None:
            raise PermissionError('current resource hold record is required')
        require_release(load(args.hold), load(args.release) if args.release else None)
        require_B_execution(config, load(args.release), execution_identity())
        if args.mode == 'run':
            if args.root is None or args.daemon is None:
                parser.error('B run requires new --root and exact pinned --daemon')
            plan(config)
            b_controller(args, config)
        else:
            if any(v is None for v in (args.case, args.endpoint, args.descriptor, args.listen_fd)):
                parser.error('consumer requires --case, --endpoint, --descriptor, --listen-fd')
            consumer(args, config)
    return 0


def b_controller(args, config):
    """One new B service with a real 128MiB hard cap across all four cases."""
    import ctypes
    import fcntl
    import os
    import socket
    import subprocess
    import sys
    import time
    import traceback
    from b_session import BPoolSession
    from pool_native_probe import role_environments
    from pypto_serving.serving.native_ddr.managed_process import descendants, process_birth
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run02'))
    from r_guard import snapshot
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise RuntimeError('B controller cannot become a child subreaper')
    root = args.root.resolve()
    root.mkdir()
    for name in ('logs', 'inventories'):
        (root / name).mkdir()
    host_env, native_env = role_environments(os.environ, args.daemon.resolve().parent, root / 'inventories')
    for name in ('UFLOW_POOL_TEST_FENCE', 'UFLOW_POOL_TEST_SYMBOLS', 'UFLOW_POOL_TEST_INVENTORY_ROOT'):
        os.environ[name] = host_env[name]
    save(root / 'actual-B-identity.json', execution_identity())
    common = ['--config', str(args.config.resolve()), '--hold', str(args.hold.resolve()),
              '--release', str(args.release.resolve())]
    lock_path = Path('/workspace/uflow/locks/run02-npu-10.lock')
    def resource_gate(path, label):
        value = snapshot(path, label, 10)
        save(path / f'resource-{label}.json', value)
        status = value['status']
        if (status['health'] != 'OK' or value['fds']['users'] or value['fds']['denied']
                or status['hbm_total_mib'] - status['hbm_used_mib'] < 40000
                or f"No running processes found in NPU {status['board']}" not in value['npu']
                or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') or os.environ.get('ASCEND_VISIBLE_DEVICES')):
            raise RuntimeError('fresh B10 health/HBM/full FD gate failed; no other host/card')
        return value
    with lock_path.open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resource_gate(root, 'before')
        sock.bind(('127.0.0.1', 18892))
        sock.listen(128)
        save(root / 'reservation.json', {'owner_pid': os.getpid(), 'birth': process_birth(os.getpid()),
             'physical_device': 10, 'lock': str(lock_path), 'port': 18892, 'host': 'serving-b-sj'})
        service = BPoolSession(root / 'service', args.daemon, host_env)
        try:
            for case in CASES:
                require_release(load(args.hold), load(args.release))
                require_B_execution(config, load(args.release), execution_identity())
                case_root = root / case
                case_root.mkdir()
                for name in ('catalog', 'logs', 'results'):
                    (case_root / name).mkdir()
                resource_gate(case_root, 'before')
                descriptor = service.create()
                descriptor_path = case_root / 'catalog/normal-descriptor.json'
                save(descriptor_path, descriptor)
                command = [sys.executable, '-u', str(Path(__file__).resolve()), 'consumer', *common,
                    '--case', case, '--endpoint', service.endpoint, '--descriptor', str(descriptor_path),
                    '--listen-fd', str(sock.fileno()), '--output', str(case_root / 'results/probe.json')]
                save(case_root / 'command.json', {'argv': command, 'same_B_service_owners': service.owners,
                                                 'admin_capability_passed': False})
                with (case_root / 'logs/probe.log').open('xb') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                              pass_fds=(sock.fileno(),), env=native_env)
                    birth = process_birth(child.pid)
                    owned = {} if birth is None else {child.pid: birth}
                    deadline = time.monotonic() + 7200
                    while child.poll() is None:
                        current = {pid: birth for pid, birth in descendants().items() if pid not in service.owners}
                        if any(pid in owned and owned[pid] != birth for pid, birth in current.items()):
                            raise RuntimeError('B consumer descendant birth changed')
                        owned.update(current)
                        if (case_root / 'results/retained-live-failure.json').exists() or time.monotonic() > deadline:
                            save(case_root / 'retained-new-case.json', {'pid': child.pid, 'owned_births': owned,
                                'reason': 'live native/client responsibility retained; no signals or next case'})
                            raise RuntimeError('B consumer retained live obligations; stop entire matrix')
                        time.sleep(.2)
                    code = child.wait()
                save(case_root / 'consumer-exit.json', {'pid': child.pid, 'exit_code': code})
                report = load(case_root / 'results/probe.json') if code == 0 else None
                if case == CASES[0] and code == 0:
                    reader_command = [sys.executable, '-u', str(Path(__file__).with_name('pool_host_probe.py')),
                        '--role', 'reader', '--root', str(case_root / 'reader'), '--endpoint', service.endpoint,
                        '--pool-id', descriptor['pool_id'], '--epoch', descriptor['pool_epoch'],
                        '--expected', str(case_root / 'results/probe.json')]
                    with (case_root / 'logs/reader.log').open('xb') as reader_log:
                        reader = subprocess.Popen(reader_command, stdout=reader_log, stderr=subprocess.STDOUT, env=host_env)
                        reader_birth = process_birth(reader.pid)
                        if reader_birth is not None:
                            owned[reader.pid] = reader_birth
                        if reader.wait(timeout=60) != 0:
                            raise RuntimeError('B independent post-native reader failed; pool retained')
                # Only actual UF checks/close/unlink can release the current pool.
                # Process exit never authorizes Destroy on a mapped/Unknown client.
                cleanup = service.destroy()
                save(case_root / 'normal-cleanup.json', cleanup)
                deadline = time.monotonic() + 20
                while True:
                    for pid in owned:
                        try:
                            os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            pass
                    if not any(process_birth(pid) == birth for pid, birth in owned.items()):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError('original B consumer birth remains')
                    time.sleep(.05)
                save(case_root / 'normal-processes-gone.json', {'owned_births': owned,
                    'all_observed_births_absent': True, 'consumer_wait_reaped': True})
                if code != 0:
                    raise RuntimeError('B case failed; normally released pool, no next case')
                if case == CASES[0]:
                    if not (report['passed'] and report['native_owner_closed'] and report['client_unmapped']):
                        raise RuntimeError('B small native gate incomplete')
                elif case == CASES[1]:
                    from qwen_evidence import check
                    save(case_root / 'offline-check.json', check(config, report, descriptor, case_root / 'results',
                                                               (case_root / 'logs/probe.log').read_text()))
                else:
                    pending = report['native_pending_stop']
                    if (pending['operation'] != case.removeprefix('stop-').upper()
                            or pending['raw_native_pending_at_stop'] is not True
                            or pending['owner_stop_proven'] is not True or not report['stop_ok']):
                        raise RuntimeError('B real pending stop not proven')
                resource_gate(case_root, 'after')
                save(case_root / 'result.json', {'status': 'CAPTURED-PENDING-REVIEW', 'case': case,
                    'B_pool_released': True, 'A_retained_floor_bytes': 134217728, 'D1_acceptance': False})
            shutdown = service.shutdown()
            save(root / 'B-service-normal-shutdown.json', shutdown)
        except BaseException:
            save(root / 'first-controller-failure.json', {'traceback': traceback.format_exc(),
                'service_owners': service.owners, 'current_descriptor': service.normal,
                'scope': 'original B domain retained if live pool/Unknown; never create replacement'})
            if service.normal is None and service.daemon.poll() is None:
                save(root / 'no-live-pool-failure-shutdown.json', service.shutdown())
            raise
    resource_gate(root, 'after')
    with lock_path.open('a+') as lock, socket.socket() as port:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
        port.bind(('127.0.0.1', 18892))
    save(args.output, {'status': 'CAPTURED-PENDING-REVIEW', 'cases': list(CASES), 'D1_acceptance': False,
         'B_service_normal_shutdown': True, 'B_charge_after': 0, 'A_retained_floor_bytes': 134217728,
         'A_queried': False, 'I1_D2_started': False, 'lock_reacquired_released': True, 'port_rebindable': True})


def controller(args, config):
    """Same original Catalog, four sequential pools; never retry a failed case."""
    import ctypes
    import fcntl
    import os
    import socket
    import subprocess
    import sys
    import time
    from existing_pool import ExistingPoolService, OWNER_PIDS, SERVICE_ROOT
    from pool_native_probe import role_environments
    from pypto_serving.serving.native_ddr.managed_process import descendants, process_birth

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER')

    root = args.root.resolve()
    root.mkdir()
    native_script = Path(__file__).with_name('pool_native_probe.py')
    common = ['--config', str(args.config.resolve()), '--hold', str(args.hold.resolve()),
              '--release', str(args.release.resolve())]
    command = [sys.executable, '-u', str(native_script), '--existing-state', str(args.existing_state),
               '--retained-baseline', str(args.snapshot), '--device', str(config['device']),
               '--root', str(root / CASES[0])]
    with (root / 'small-native.log').open('xb') as log:
        # Existing fixture already enforces fresh device guard/reservation and
        # real normal cleanup. Its copy_to overwrite is NOT model byte evidence.
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    fixture = load(root / CASES[0] / 'result.json')
    if fixture.get('passed') is not True:
        raise RuntimeError('small native gate did not complete; no Qwen case may start')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run02'))
    from r_guard import snapshot
    for case in CASES[1:]:
        require_release(load(args.hold), load(args.release))
        case_root = root / case
        case_root.mkdir()
        for name in ('catalog', 'logs', 'inventories', 'results'):
            (case_root / name).mkdir()
        host_env, native_env = role_environments(os.environ, SERVICE_ROOT, case_root / 'inventories')
        for name in ('UFLOW_POOL_TEST_FENCE', 'UFLOW_POOL_TEST_SYMBOLS', 'UFLOW_POOL_TEST_INVENTORY_ROOT'):
            os.environ[name] = host_env[name]
        service = ExistingPoolService(args.existing_state, args.snapshot, case_root / 'catalog')
        lock_path = Path(f'/workspace/uflow/locks/run02-npu-{config["device"]}.lock')
        with lock_path.open('a+') as lock, socket.socket() as sock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            before = snapshot(case_root, 'before', config['device'])
            save(case_root / 'resource-before.json', before)
            if (before['status']['health'] != 'OK' or before['fds']['users'] or before['fds']['denied']
                    or before['status']['hbm_total_mib'] - before['status']['hbm_used_mib'] < 40000
                    or f"No running processes found in NPU {before['status']['board']}" not in before['npu']
                    or os.environ.get('ASCEND_RT_VISIBLE_DEVICES') or os.environ.get('ASCEND_VISIBLE_DEVICES')):
                raise RuntimeError('fresh A10 resource gate failed; no alternative card/authority')
            sock.bind(('127.0.0.1', 18892))
            sock.listen(128)
            save(case_root / 'reservation.json', {'physical_device': config['device'], 'owner_pid': os.getpid(),
                'owner_birth': process_birth(os.getpid()), 'lock_path': str(lock_path), 'port': 18892,
                'scope': 'advisory lock plus fresh complete device FD visibility; one A10 only'})
            descriptor = service.create()
            descriptor_path = case_root / 'catalog/normal-descriptor.json'
            command = [sys.executable, '-u', str(Path(__file__).resolve()), 'consumer', *common,
                '--case', case, '--endpoint', service.endpoint, '--descriptor', str(descriptor_path),
                '--listen-fd', str(sock.fileno()), '--output', str(case_root / 'results/probe.json')]
            save(case_root / 'command.json', {'argv': command, 'admin_capability_passed': False,
                                             'descriptor': descriptor, 'retained_owners': list(OWNER_PIDS)})
            with (case_root / 'logs/probe.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(sock.fileno(),), env=native_env)
                birth = process_birth(child.pid)
                owned = {} if birth is None else {child.pid: birth}
                deadline = time.monotonic() + 7200
                while child.poll() is None:
                    current = descendants()
                    if any(pid in owned and owned[pid] != birth for pid, birth in current.items()):
                        raise RuntimeError('case descendant PID identity changed')
                    owned.update(current)
                    if (case_root / 'results/retained-live-failure.json').exists() or time.monotonic() > deadline:
                        save(case_root / 'retained-new-case.json', {'pid': child.pid,
                            'reason': 'live process/native access retained; no force termination or next pool',
                            'snapshot': service.snapshot('retained-live-failure')})
                        raise RuntimeError('model case retained live obligations; stop whole matrix')
                    time.sleep(.2)
                code = child.wait()
            save(case_root / 'consumer-exit.json', {'pid': child.pid, 'exit_code': code})
            # Even failed requests may have closed normally. Only UF may release
            # the physical resource; a death/exit code is never an unmap proof.
            cleanup = service.destroy()
            save(case_root / 'normal-cleanup.json', cleanup)
            if set(owned).intersection(OWNER_PIDS):
                raise RuntimeError('retained service owners must never be case-owned PIDs')
            deadline = time.monotonic() + 20
            while True:
                for pid in owned:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                remaining = {pid: birth for pid, birth in owned.items() if process_birth(pid) == birth}
                if not remaining:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError('normal case process birth still exists; no subsequent case')
                time.sleep(.05)
            save(case_root / 'normal-processes-gone.json', {'owned_births': owned,
                'all_observed_births_absent': True, 'consumer_wait_reaped': True,
                'retained_owners_excluded': list(OWNER_PIDS)})
            if code != 0:
                raise RuntimeError('failed model case was normally released; do not continue matrix')
            report = load(case_root / 'results/probe.json')
            if case == CASES[1]:
                from qwen_evidence import check
                checked = check(config, report, descriptor, case_root / 'results',
                                (case_root / 'logs/probe.log').read_text())
                save(case_root / 'offline-check.json', checked)
            else:
                pending = report['native_pending_stop']
                if (pending['operation'] != case.removeprefix('stop-').upper()
                        or pending['raw_native_pending_at_stop'] is not True
                        or pending['owner_stop_proven'] is not True or not report['stop_ok']):
                    raise RuntimeError('real pending stop not proven')
            save(case_root / 'result.json', {'status': 'CAPTURED-PENDING-REVIEW', 'case': case,
                                             'normal_pool_released': True, 'D1_acceptance': False})
        after = snapshot(case_root, 'after', config['device'])
        if after['fds']['users'] or after['fds']['denied'] or after['status']['health'] != 'OK':
            raise RuntimeError('device/FD normal release was not proven')
        save(case_root / 'resource-after.json', after)
        with lock_path.open('a+') as lock, socket.socket() as port:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            port.bind(('127.0.0.1', 18892))
    save(args.output, {'status': 'CAPTURED-PENDING-REVIEW', 'cases': list(CASES),
                       'D1_acceptance': False, 'quarantine_charged_bytes': 134217728,
                       'service_shutdown': False, 'no_I1_D2': True})


def consumer(args, config):
    import asyncio
    import logging
    import multiprocessing
    import os
    import sys
    import time
    import traceback
    from types import SimpleNamespace
    from qwen_contract import check_startup
    if os.environ.get('LD_PRELOAD') or any(k.startswith('UFLOW_POOL_TEST_') for k in os.environ):
        raise RuntimeError('host fence must never reach model/native worker')
    if args.case == CASES[0]:
        import pool_native_probe as probe
        descriptor = load(args.descriptor)
        native_root = args.output.parent / 'native'
        native_args = SimpleNamespace(root=native_root, endpoint=args.endpoint, device=config['device'],
                                      pool_id=descriptor['pool_id'], epoch=descriptor['pool_epoch'])
        try:
            probe.consumer(native_args)
            save(args.output, load(native_root / 'result.json'))
        except BaseException as error:
            if probe._ACCESS_ATTEMPTED and probe._ACTIVE_ACCESS is not None and not probe._ACTIVE_ACCESS.closed:
                save(args.output.parent / 'retained-live-failure.json', {'pid': os.getpid(), 'error': repr(error),
                    'scope': 'original small-native owner/SDK responsibility retained'})
                probe._RECOVERY.retain_failure(error)
            raise
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run02'))
    from r_probe import exercise, native_pending_stop
    from qwen_scenario import normal
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)d %(levelname)s %(message)s')
    os.environ['PYPTO_NATIVE_DDR_POOL_EVIDENCE'] = '1'
    # Production worker installs its static recovery action before registration.
    os.environ['PYPTO_PROG_BUILD_DIR'] = str(args.output.parent / 'build')
    descriptor = load(args.descriptor)
    run_args = SimpleNamespace(device=config['device'], native_enabled=True, utilization=config['utilization'],
                               listen_fd=args.listen_fd, output=args.output, case=args.case)
    report = {'mode': 'execution', 'passed': False, 'case': args.case, 'device': config['device'],
              'pid': os.getpid(), 'D1_acceptance': False}
    async def stop_scenario(engine, core, collect, collect_trace, arguments, result, cold, warm):
        await native_pending_stop(engine, core, collect, arguments, result)
    try:
        asyncio.run(exercise(run_args, report,
            cli_arguments=cli_arguments(config, args.endpoint, descriptor['pool_id'], descriptor['pool_epoch']),
            startup_check=lambda startup: check_startup(config, startup, descriptor),
            scenario=normal if args.case == CASES[1] else stop_scenario, minimum_new_tokens=2))
    except BaseException as error:
        report['passed'], report['error'] = False, traceback.format_exc()
        # Legacy report writer is mutable progress; the first failure is sealed
        # separately before another assertion or shutdown-report interpretation.
        save(args.output.parent / 'first-failure.json', report)
        live = [p for p in multiprocessing.active_children() if p.is_alive()]
        if live:
            save(args.output.parent / 'retained-live-failure.json', {'pid': os.getpid(),
                'children': [p.pid for p in live], 'error': repr(error),
                'scope': 'keep original traceback/engine/native references; no forced atexit teardown'})
            while True:
                # Hold strong traceback references, without adding any recovery
                # command or arbitrary exec service. Original worker channel only.
                _retained_error = error
                time.sleep(30)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
