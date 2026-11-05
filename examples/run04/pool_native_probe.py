# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-card UF pack/unpack fixture; full Qwen validation is a separate case."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from pool_host_probe import capture, inventory, save

_ACCESS_ATTEMPTED = False
_ACTIVE_ACCESS = None
_RECOVERY = None


def role_environments(parent, daemon_root, inventory_root):
    """Host test fence belongs only to the daemon/provider and RO host reader."""
    public = {k: v for k, v in parent.items()
              if not k.startswith('UFLOW_POOL_TEST_') and k != 'LD_PRELOAD'}
    host = {**public, 'LD_PRELOAD': str(daemon_root / 'api_fence.so'),
            'UFLOW_POOL_TEST_FENCE': str(daemon_root / 'api_fence.so'),
            'UFLOW_POOL_TEST_SYMBOLS': str(daemon_root / 'src/pool/examples/managed_pool/blocked-symbols.json'),
            'UFLOW_POOL_TEST_INVENTORY_ROOT': str(inventory_root)}
    return host, public


def probe_request(sender, operation, key, ref, page):
    from pypto_serving.serving.native_ddr.types import DdrTransferRequest, Operation
    return sender.stamp(DdrTransferRequest(operation.value, ref.worker_epoch, operation, (key,), (ref,),
        'page-lease-' + operation.value, (page,), 1 if operation == Operation.STORE else None))


def runtime_view_report(worker, slots, backing):
    from simpler.buffer import BackendKind
    whole = worker._named_host_buffer(slots.pool_tensor.data_ptr(), slots.access.identity.nbytes)
    part = worker._named_host_buffer(backing.tensor.data_ptr(), backing.tensor.numel() * 2)
    assert whole is not None and part is not None
    buffer, offset, readonly = part
    buffer.to_descriptor()  # Validate the actual fixed ABI, not a hand-built wire descriptor.
    assert buffer is whole[0] and whole[1] == 0 and not readonly
    assert buffer.backend_kind == BackendKind.FORK_SHM and buffer.nbytes == slots.access.identity.nbytes
    assert offset == backing.byte_offset and backing.tensor.data_ptr() == slots.pool_tensor.data_ptr() + offset
    assert int.from_bytes(buffer.body, 'little') == slots.pool_tensor.data_ptr()
    return {'kind': 'actual-runtime-FORK_SHM-view-of-UFlow-MAP_SHARED', 'byte_offset': offset,
            'range_bytes': backing.tensor.numel() * 2, 'registered_span_bytes': buffer.nbytes,
            'one_runtime_buffer_for_whole_span': True, 'body_matches_local_pool_base': True,
            'runtime_owner_instance_id': bytes(buffer.identity.owner_instance_id).hex(),
            'runtime_buffer_id': int(buffer.identity.buffer_id), 'runtime_generation': int(buffer.identity.generation),
            'runtime_backend_kind': int(buffer.backend_kind), 'runtime_body_sha256': hashlib.sha256(buffer.body).hexdigest()}


def consumer(args):
    global _ACCESS_ATTEMPTED, _ACTIVE_ACCESS, _RECOVERY
    assert not os.environ.get('LD_PRELOAD'), 'host fence must not reach native roles'
    assert not any(k.startswith('UFLOW_POOL_TEST_') for k in os.environ)
    import numpy as np
    import torch
    from pypto.runtime import DistributedWorker
    from pypto_serving.model.common.compiler.compiler import KernelCompiler
    from pypto_serving.model.common.executor.utils import build_pypto_run_config
    from pypto_serving.model.qwen.native_ddr_kv import compile_programs
    from pypto_serving.model.qwen.native_ddr_transfer import ManagedDdrSlots, QwenNativeTransferBackend
    from pypto_serving.serving.native_ddr.managed_pool import BLOCK_BYTES, GUARD_BYTES
    from pypto_serving.serving.native_ddr.managed_pool_sdk import SDK_SHA256, register_managed_pool
    from pypto_serving.serving.native_ddr.managed_process import (
        attach_new_children, descendants, detach_closed_children,
    )
    from pypto_serving.serving.native_ddr.types import (
        DdrEntryRef, NativeDdrPrefixConfig, Operation, PrefixKey, QwenKvLayout, TransferStatus,
    )
    from pypto_serving.serving.native_ddr.replay import DeliverySender
    from pypto_serving.serving.native_ddr.worker import DdrPrefixWorker
    from verify_bytes import verify_guards

    root = args.root
    root.mkdir()
    run_config = build_pypto_run_config(platform='a2a3', device_ids=(args.device,),
                                        pypto_build_dir=str(root / 'build'))
    compiler = KernelCompiler(run_config=run_config)
    programs = compile_programs(compiler)
    # Finish the fixed runtime's compile/assemble stage before registering a
    # client. A missing compiler must not strand a mapped client on exit. The
    # real Worker still registers the UF view before its subsequent fork/init.
    from pypto.runtime.distributed_runner import _assemble_chip_callables
    for program in programs:
        _assemble_chip_callables(program.compiled)
    save(root / 'compiled-before-client.json', {'programs': [p.name for p in programs],
        'client_registered': False, 'native_owner_created': False, 'native_jobs_submitted': 0})
    config = SimpleNamespace(pool_endpoint=args.endpoint, pool_id=args.pool_id,
                             pool_epoch=args.epoch, sdk_sha256=SDK_SHA256)
    from pypto_serving.serving.native_ddr.cooperative_close import CooperativePoolClose
    _RECOVERY = CooperativePoolClose(root / 'cooperative-close')
    _ACCESS_ATTEMPTED = True
    access = register_managed_pool(config)
    _ACTIVE_ACCESS = access
    _RECOVERY.bind_access(access)
    slots = ManagedDdrSlots(1, access)
    _RECOVERY.slots = slots
    client = access.port.client
    backing = slots.reserve(0, BLOCK_BYTES)
    raw_block = access.port._raw_block(backing.block)
    guard = client.acquire_range(raw_block, 'w', raw_block.generation, nbytes=GUARD_BYTES)
    client.view.buffer[:GUARD_BYTES] = b'\x93' * GUARD_BYTES
    client.commit_write(guard, 'native-probe-guard', GUARD_BYTES)
    lease = access.acquire(backing.block, 'w', byte_offset=backing.byte_offset)
    backing.tensor.view(torch.uint8).fill_(0)
    access.complete_write(lease, 'native-probe-initial', 'initial-native-probe')
    tail = client.allocate_range((106 << 20))
    assert tail.offset == 22 << 20
    tail_lease = client.acquire_range(tail, 'w', tail.generation)
    client.view.buffer[tail.offset:] = b'\x94' * tail.nbytes
    client.commit_write(tail_lease, 'native-probe-tail', tail.nbytes)
    ranges = [{'block': raw_block, 'offset': 0, 'nbytes': GUARD_BYTES},
              {'block': raw_block, 'offset': GUARD_BYTES, 'nbytes': BLOCK_BYTES},
              {'block': tail, 'offset': 0, 'nbytes': tail.nbytes}]
    capture(client, ranges, root / 'pool-before.bin')
    slots.release(backing)
    baseline = descendants()
    worker = DistributedWorker([p.compiled for p in programs],
        inherited_host_tensors=slots.inherited_host_tensors, startup_timeout_s=60)
    _RECOVERY.worker = worker
    registered = attach_new_children(access, baseline, helpers=slots.child_helpers,
                                     control_resources=slots.control_resources)
    registered_pids = tuple(registered)
    helper_evidence = dict(slots.child_helpers)
    save(root / 'children.json', {'baseline': baseline, 'registered': registered,
        'helpers': helper_evidence, 'identity': asdict(access.identity), 'native_role_fence': None})
    from host_guard import inventory as process_inventory
    save(root / 'native-process-inventories.json', {
        'scope': 'native roles may own the selected device; host zero-Ascend-FD judge is not applied',
        'parent': process_inventory(os.getpid()),
        'children': [process_inventory(pid) for pid in registered]})
    layout = QwenKvLayout(40, 4, 128, 8, 128)
    arrays = []
    for side in range(2):
        data = torch.empty((40 * 4 * 1024, 128), dtype=torch.bfloat16)
        view = data.view(torch.uint16).numpy().reshape(40, 4, 128, 8, 128)
        index = np.arange(128 * 8 * 128, dtype=np.uint32).reshape(128, 8, 128)
        for layer in range(40):
            for page in range(4):
                view[layer, page] = (index * 17 + layer * 997 + page * 113 + side * 3109).astype(np.uint16)
        arrays.append(data)
    tensors = [worker.alloc_tensor(a.shape, a.dtype, init=a) for a in arrays]
    trace = []
    backend = QwenNativeTransferBackend(worker, programs, slots, layout, *tensors,
                                        'native-probe', observer=trace.append)
    module = DdrPrefixWorker(NativeDdrPrefixConfig(True, BLOCK_BYTES), layout, 'native-probe', backend)
    ref = DdrEntryRef('native-probe', 'one-complete-block', 1)
    key = PrefixKey(b'N' * 32, hashlib.sha256(b'native-probe-prefix').digest())
    sender = DeliverySender()

    def transfer(transfer_module, operation, page):
        request = probe_request(sender, operation, key, ref, page)
        submitted = transfer_module.submit_transfer(request)
        save(root / f'{operation.value.lower()}-submit-event.json', asdict(submitted))
        save(root / f'{operation.value.lower()}-submit-trace.json', trace)
        assert submitted.status == TransferStatus.ACCEPTED, submitted.error
        save(root / f'{operation.value.lower()}-accepted.json', {
            'entry_state': transfer_module.pending.entry.state.value,
            'native_pending': not transfer_module.pending.handle.native.done,
            'actual_native_range': runtime_view_report(transfer_module.backend.worker, slots,
                                                        transfer_module.pending.entry.backing),
            'snapshot': client.snapshot(), 'request': {
                'job_id': request.job_id, 'operation': request.operation.value,
                'page_ids': request.page_ids, 'page_lease_id': request.page_lease_id,
                'entry_refs': [asdict(item) for item in request.entry_refs],
                'delivery_sequence': request.delivery_sequence, 'retire_through': request.retire_through,
                'namespace': key.namespace_digest.hex(), 'block_hash': key.block_hash.hex()}})
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            events = transfer_module.poll_completed()
            if events:
                event = events[0]
                save(root / f'{operation.value.lower()}-completed.json', asdict(event))
                assert event.status == TransferStatus.SUCCEEDED, event
                assert event.native_completed and event.stopped and event.transferred_bytes == BLOCK_BYTES
                assert event.pool_receipt is not None
                sender.consume(event.delivery_sequence)
                return
            time.sleep(.001)
        raise TimeoutError('real native transfer has no bounded completion receipt')

    expected = torch.stack([a.reshape(40, 4, 128, 8, 128)[:, 1] for a in arrays], dim=1).contiguous()
    expected_bytes = expected.view(torch.uint8).numpy().tobytes()
    transfer(module, Operation.STORE, 1)
    actual = backend.read_slot(module.entries[ref.entry_id].backing)
    assert actual.view(torch.uint8).numpy().tobytes() == expected_bytes
    del actual
    # Probe-only synchronous overwrite; full Qwen validation must use real other model computation.
    for data, tensor in zip(arrays, tensors):
        data.reshape(40, 4, 128, 8, 128)[:, 1].view(torch.uint8).fill_(0xA5)
        worker.copy_to(tensor.data_ptr, data.data_ptr(), tensor.nbytes)
    transfer(module, Operation.LOAD, 2)
    for side, (data, tensor) in enumerate(zip(arrays, tensors)):
        data.reshape(40, 4, 128, 8, 128)[:, 2].copy_(expected[:, side])
        observed = torch.empty_like(data)
        worker.copy_from(observed.data_ptr(), tensor.data_ptr, tensor.nbytes)
        assert torch.equal(observed.view(torch.uint8), data.view(torch.uint8))
    pool_sha = capture(client, ranges, root / 'pool-after.bin')
    import mmap
    with (root / 'pool-before.bin').open('rb') as left, (root / 'pool-after.bin').open('rb') as right:
        with mmap.mmap(left.fileno(), 0, access=mmap.ACCESS_READ) as a:
            with mmap.mmap(right.fileno(), 0, access=mmap.ACCESS_READ) as b:
                guards = verify_guards(a, b, [(GUARD_BYTES, BLOCK_BYTES)])
    save(root / 'guards.json', guards)
    module.close()
    backend.close()
    for tensor in tensors:
        worker.free_tensor(tensor)
    worker.close()
    detach_closed_children(access, registered, baseline, helpers=slots.child_helpers)
    del module, backend, worker
    slots.close_after_owner()
    _RECOVERY.restore_after_normal_close()
    assert access.closed and not access.children and not access.leases and not access.unknown
    result = {'passed': True, 'scope': 'real pack/unpack fixture, not full Qwen or actual-compute overwrite',
              'registered': asdict(access.identity), 'ranges': [{**r, 'block': asdict(r['block'])} for r in ranges],
              'pool_sha256': pool_sha, 'payload_bytes': len(expected_bytes),
              'expected_payload_sha256': hashlib.sha256(expected_bytes).hexdigest(),
              'whole_four_page_arenas_equal': True, 'source_page': 1, 'target_page': 2,
              'guard_bytes_compared': guards['guard_bytes_compared'], 'native_trace': trace,
              'original_native_children': registered_pids, 'consumer_child_baseline': baseline,
              'stdlib_helpers_recorded_and_normally_reaped': helper_evidence,
              'native_owner_closed': True, 'children_detached': True, 'client_unmapped': True}
    save(root / 'result.json', result)


def controller(args):
    if args.existing_state is not None:
        return shared_controller(args)
    import fcntl
    from uflow.managed_pool import create_pool, destroy_pool, shutdown_pool_service
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run02'))
    from r_guard import snapshot
    import ctypes
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    root = args.root
    root.mkdir()
    (root / 'logs').mkdir()
    (root / 'inventories').mkdir()
    daemon_root = args.daemon.parent
    host_env, native_env = role_environments(os.environ, daemon_root, root / 'inventories')
    # Local controller resolves only public host-guard configuration. It does
    # not preload the fence and it never imports native/model libraries.
    for name in ('UFLOW_POOL_TEST_FENCE', 'UFLOW_POOL_TEST_SYMBOLS', 'UFLOW_POOL_TEST_INVENTORY_ROOT'):
        os.environ[name] = host_env[name]
    endpoint, state = str(root / 'u.sock'), root / 'state'
    assert len(endpoint.encode()) < 108
    lock_path = Path(f'/workspace/uflow/locks/run02-npu-{args.device}.lock')
    with lock_path.open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(root, 'before', args.device)
        save(root / 'resource-before.json', before)
        status = before['status']
        assert status['health'] == 'OK' and status['hbm_total_mib'] - status['hbm_used_mib'] >= 40000
        assert not before['fds']['users'] and not before['fds']['denied']
        assert f"No running processes found in NPU {status['board']}" in before['npu']
        assert not os.environ.get('ASCEND_RT_VISIBLE_DEVICES') and not os.environ.get('ASCEND_VISIBLE_DEVICES')
        sock.bind(('127.0.0.1', 18892))
        save(root / 'reservation.json', {'physical_device': args.device, 'lock_path': str(lock_path),
                                        'owner_pid': os.getpid(), 'scope': 'advisory task lock, fresh full FD/health check'})
        daemon_log = (root / 'daemon.log').open('xb')
        daemon = subprocess.Popen([str(args.daemon), '--host-pool', '--socket', endpoint,
            '--pool-state-root', str(state), '--pool-provider-python', sys.executable],
            stdout=daemon_log, stderr=subprocess.STDOUT, env=host_env)
        deadline = time.monotonic() + 30
        while not Path(endpoint).exists():
            if daemon.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('host daemon startup failed; original evidence retained')
            time.sleep(.05)
        loaded = json.loads((state / 'state.json').read_text())['provider_loaded']
        inventory(root, 'provider', 'provider-start', loaded['pid'])
        inventory(root, 'daemon', 'daemon-start', daemon.pid)
        token = (state / 'admin.token').read_text()
        descriptor = create_pool(endpoint, admin_token=token)
        provider = inventory(root, 'provider', 'provider-pool-created', loaded['pid'])
        save(root / 'public-descriptor.json', descriptor)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--role', 'native',
            '--root', str(root / 'native'), '--endpoint', endpoint, '--pool-id', descriptor['pool_id'],
            '--epoch', descriptor['pool_epoch'], '--device', str(args.device)]
        with (root / 'native.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=native_env)
            try:
                code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                child.send_signal(signal.SIGINT)
                code = child.wait(timeout=120)
        save(root / 'native-exit.json', {'pid': child.pid, 'exit_code': code, 'command': command})
        if code:
            failed = json.loads((state / 'state.json').read_text())
            failed_pool = next(row for row in failed['pools'] if row['pool_id'] == descriptor['pool_id'])
            if not failed_pool['clients'] and not failed_pool['ranges']['leases']:
                # This is a genuine no-access-granted case. The real UFlow
                # destroy checks and provider close/unlink remain authoritative.
                released = destroy_pool(endpoint, descriptor, admin_token=token)
                stopped = shutdown_pool_service(endpoint, admin_token=token)
                assert daemon.wait(timeout=20) == 0
                daemon_log.close()
                save(root / 'pre-client-failure-cleanup.json', {'native_exit': code,
                    'basis': 'UFlow has no registered clients or range grants, not child death',
                    'provider_release': released, 'shutdown': stopped,
                    'clean': json.loads((state / 'clean.json').read_text())})
                raise RuntimeError('probe failed before client registration; real UF pool destroy completed')
            save(root / 'retained-failure.json', {'reason': 'native child failed; no process-death cleanup claim',
                'daemon_pid': daemon.pid, 'provider_pid': loaded['pid'], 'endpoint': endpoint,
                'state_retained': True, 'device_resources': snapshot(root, 'failure', args.device)})
            raise RuntimeError('native probe failed; UF obligations and original domain retained')
        current = json.loads((state / 'state.json').read_text())
        assert current['charged_bytes'] == 128 << 20
        pool = next(row for row in current['pools'] if row['pool_id'] == descriptor['pool_id'])
        assert not pool['frozen']
        assert all(row['state'] == 'Unmapped' and not row['children'] for row in pool['clients'])
        assert not pool['ranges']['leases']
        save(root / 'after-client.json', current)
        reader_command = [sys.executable, '-u', str(Path(__file__).with_name('pool_host_probe.py')),
            '--role', 'reader', '--root', str(root / 'reader'), '--endpoint', endpoint,
            '--pool-id', descriptor['pool_id'], '--epoch', descriptor['pool_epoch'],
            '--expected', str(root / 'native/result.json')]
        with (root / 'reader.log').open('xb') as log:
            reader = subprocess.Popen(reader_command, stdout=log, stderr=subprocess.STDOUT, env=host_env)
            assert reader.wait(timeout=60) == 0
        released = destroy_pool(endpoint, descriptor, admin_token=token)
        shutdown = shutdown_pool_service(endpoint, admin_token=token)
        assert daemon.wait(timeout=20) == 0
        daemon_log.close()
        clean = json.loads((state / 'clean.json').read_text())
        assert clean['normal_shutdown'] and clean['provider_exited'] and clean['charged_bytes'] == 0
        owned = {daemon.pid, loaded['pid'], child.pid, reader.pid, *provider['children']}
        limit = time.monotonic() + 15
        while any(Path('/proc', str(pid)).exists() for pid in owned):
            for pid in owned:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            assert time.monotonic() < limit, 'owned host process still alive'
            time.sleep(.05)
        assert not Path(endpoint).exists()
        shm_name = bytes.fromhex(descriptor['canonical']['body_hex']).decode()
        assert not (Path('/dev/shm') / shm_name).exists()
    after = snapshot(root, 'after', args.device)
    assert not after['fds']['users'] and not after['fds']['denied'] and after['status']['health'] == 'OK'
    with lock_path.open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
        sock.bind(('127.0.0.1', 18892))
    save(root / 'result.json', {'passed': True, 'scope': 'real UF/Simpler A native fixture, no full Qwen run',
        'descriptor': descriptor, 'provider_loaded': loaded, 'provider_release': released, 'shutdown': shutdown,
        'clean': clean, 'owned_pids_absent': sorted(owned), 'resource_after': after,
        'lock_reacquired_and_released': True, 'port_rebindable': True, 'shm_name_absent': shm_name})


def shared_controller(args):
    """Create/destroy only one new pool in the retained Catalog; never stop its service."""
    import ctypes
    import fcntl
    from existing_pool import ExistingPoolService, OWNER_PIDS, SERVICE_ROOT
    from pypto_serving.serving.native_ddr.managed_process import descendants
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'run02'))
    from r_guard import snapshot
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    root = args.root
    root.mkdir()
    for name in ('logs', 'inventories', 'catalog'):
        (root / name).mkdir()
    host_env, native_env = role_environments(os.environ, SERVICE_ROOT, root / 'inventories')
    for name in ('UFLOW_POOL_TEST_FENCE', 'UFLOW_POOL_TEST_SYMBOLS', 'UFLOW_POOL_TEST_INVENTORY_ROOT'):
        os.environ[name] = host_env[name]
    service = ExistingPoolService(args.existing_state, args.retained_baseline, root / 'catalog')
    lock_path = Path(f'/workspace/uflow/locks/run02-npu-{args.device}.lock')
    with lock_path.open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = snapshot(root, 'before', args.device)
        save(root / 'resource-before.json', before)
        status = before['status']
        assert status['health'] == 'OK' and status['hbm_total_mib'] - status['hbm_used_mib'] >= 40000
        assert not before['fds']['users'] and not before['fds']['denied']
        assert f"No running processes found in NPU {status['board']}" in before['npu']
        assert not os.environ.get('ASCEND_RT_VISIBLE_DEVICES') and not os.environ.get('ASCEND_VISIBLE_DEVICES')
        sock.bind(('127.0.0.1', 18892))
        save(root / 'reservation.json', {'physical_device': args.device, 'owner_pid': os.getpid(),
            'lock_path': str(lock_path), 'scope': 'advisory with fresh health/full FD visibility'})
        descriptor = service.create()
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--role', 'native',
            '--root', str(root / 'native'), '--endpoint', service.endpoint,
            '--pool-id', descriptor['pool_id'], '--epoch', descriptor['pool_epoch'], '--device', str(args.device)]
        with (root / 'native.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=native_env)
            deadline = time.monotonic() + 1800
            while child.poll() is None:
                if (root / 'native/retained-live-failure.json').exists() or time.monotonic() > deadline:
                    current = service.snapshot('retained-live-child')
                    save(root / 'retained-new-case.json', {'child_pid': child.pid, 'command': command,
                        'reason': 'live native access retained for diagnosis; no unmap/death cleanup inference',
                        'snapshot': current, 'retained_existing_owners': list(OWNER_PIDS)})
                    raise RuntimeError('new normal case failed; native process remains alive, no third pool')
                time.sleep(.2)
            code = child.wait()
        save(root / 'native-exit.json', {'pid': child.pid, 'exit_code': code, 'command': command})
        if code:
            _, pool = service.current('failed-child')
            if (any(c['state'] != 'Unmapped' or c['children'] for c in pool['clients'])
                    or pool['ranges']['leases']):
                raise RuntimeError('new case retained access/Unknown; stop without another pool')
            cleanup = service.destroy('failed-case-pool-released')
            save(root / 'failed-case-cleanup.json', cleanup)
            raise RuntimeError('case failed but UF really released its new unreferenced pool; service retained')
        native = json.loads((root / 'native/result.json').read_text())
        assert native['passed'] and native['native_owner_closed'] and native['client_unmapped']
        _, pool = service.current('after-native-client')
        assert all(c['state'] == 'Unmapped' and not c['children'] for c in pool['clients'])
        assert not pool['ranges']['leases']
        reader_command = [sys.executable, '-u', str(Path(__file__).with_name('pool_host_probe.py')),
            '--role', 'reader', '--root', str(root / 'reader'), '--endpoint', service.endpoint,
            '--pool-id', descriptor['pool_id'], '--epoch', descriptor['pool_epoch'],
            '--expected', str(root / 'native/result.json')]
        with (root / 'reader.log').open('xb') as log:
            reader = subprocess.Popen(reader_command, stdout=log, stderr=subprocess.STDOUT, env=host_env)
            assert reader.wait(timeout=60) == 0
        cleanup = service.destroy()
        owned = {child.pid, reader.pid, *native['original_native_children'],
                 *map(int, native['consumer_child_baseline']), *descendants()}
        assert not owned.intersection(OWNER_PIDS)
        limit = time.monotonic() + 20
        while any(Path('/proc', str(pid)).exists() for pid in owned):
            for pid in owned:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            assert time.monotonic() < limit, 'normal case PID still alive; old owners are excluded'
            time.sleep(.05)
    after = snapshot(root, 'after', args.device)
    assert after['status']['health'] == 'OK' and not after['fds']['users'] and not after['fds']['denied']
    with lock_path.open('a+') as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
        sock.bind(('127.0.0.1', 18892))
    save(root / 'result.json', {'passed': True, 'scope': 'same-Catalog normal UF native fixture, not full Qwen',
        'normal_descriptor': descriptor, 'normal_cleanup': cleanup, 'normal_owned_pids_absent': sorted(owned),
        'retained_existing_owner_pids': list(OWNER_PIDS), 'service_shutdown': False,
        'resource_after': after, 'lock_reacquired_and_released': True, 'port_rebindable': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['controller', 'native'], default='controller')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--daemon', type=Path)
    parser.add_argument('--existing-state', type=Path)
    parser.add_argument('--retained-baseline', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--endpoint')
    parser.add_argument('--pool-id')
    parser.add_argument('--epoch')
    args = parser.parse_args()
    if args.role == 'controller':
        assert (args.daemon is not None or args.existing_state is not None) and 0 <= args.device < 16
        if args.existing_state is not None:
            assert args.daemon is None and args.retained_baseline is not None
        controller(args)
    else:
        try:
            consumer(args)
        except BaseException as error:
            if not _ACCESS_ATTEMPTED or _ACTIVE_ACCESS is not None and _ACTIVE_ACCESS.closed:
                raise
            # Keep the original client PID, exception frames and all native
            # references alive on a failed probe. Parent stops the case; it
            # cannot substitute process death for a cooperative unmap receipt.
            import traceback
            traceback.print_exc()
            args.root.mkdir(exist_ok=True)
            save(args.root / 'retained-live-failure.json', {'pid': os.getpid(), 'error': repr(error),
                'scope': 'failed live client retained; no unmap ACK, release, retry or cleanup claim'})
            if _RECOVERY is None:
                raise RuntimeError('client registration attempted without preinstalled close channel') from error
            _RECOVERY.retain_failure(error)


if __name__ == '__main__':
    main()
