# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One original-client recovery, with an explicitly separate CPU fixture backend."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REAL_PID = 1022467
REAL_BIRTH = 484127941
POOL_ID = '5deb1686-4448-415b-b357-ded8d7d61aa1'
CLIENT_ID = '85595fa7-743e-4428-82dd-f450b02dc527'


def birth(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return int(text.rpartition(')')[2].split()[19])


def observation(physical, pid=None):
    """All VMA fragments and FDs of the actual backing, not just a whole-map heuristic."""
    pid = os.getpid() if pid is None else pid
    base = Path('/proc', str(pid))
    maps, fds = [], []
    for line in (base / 'maps').read_text().splitlines():
        fields = line.split(None, 5)
        major, minor = (int(part, 16) for part in fields[3].split(':'))
        if (major, minor, int(fields[4])) == (os.major(physical[0]), os.minor(physical[0]), physical[1]):
            maps.append(line)
    for path in (base / 'fd').iterdir():
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if (info.st_dev, info.st_ino) == tuple(physical):
            fds.append({'fd': int(path.name), 'target': os.readlink(path)})
    return {'pid': pid, 'maps': maps, 'fds': fds}


def descendants(pid):
    result, queue = {}, [(pid, 0)]
    while queue:
        current, depth = queue.pop()
        path = Path('/proc', str(current), 'task', str(current), 'children')
        try:
            children = [int(s) for s in path.read_text().split()]
        except FileNotFoundError:
            continue
        assert depth < 5 and len(result) + len(children) <= 32
        for child in children:
            result[child] = birth(child)
            queue.append((child, depth + 1))
    return result


def _error_and_frames(cfg):
    candidates = []
    active = set()
    for frame in sys._current_frames().values():
        while frame is not None:
            active.add(id(frame))
            if (frame.f_code.co_filename == cfg['target_source']
                    and frame.f_code.co_name in ('main', '<module>') and 'error' in frame.f_locals):
                candidates.append(frame.f_locals['error'])
            frame = frame.f_back
    assert len(candidates) == 1, 'original retained exception must be unique'
    error, frames = candidates[0], []
    assert isinstance(error, RuntimeError) and cfg['expected_error'] in str(error)
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if id(frame) not in active:
            assert frame.f_code.co_filename in cfg['allowed_unwound_sources']
            frames.append(frame)
        tb = tb.tb_next
    consumer = [f for f in frames if f.f_code.co_name == 'consumer' and f.f_code.co_filename == cfg['target_source']]
    assert len(consumer) == 1 and frames
    return error, frames, consumer[0]


def prepare(ctx, cfg):
    assert os.getpid() == cfg['pid'] and birth(os.getpid()) == cfg['birth']
    if cfg['mode'] == 'n004':
        assert (os.getpid(), cfg['birth']) == (REAL_PID, REAL_BIRTH)
        assert cfg['pool_id'] == POOL_ID and cfg['client_id'] == CLIENT_ID
    else:
        assert cfg['mode'] == 'cpu-fixture' and os.getpid() != REAL_PID
        assert 'fixture_parent.py' in cfg['target_source']
    for name, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    error, frames, consumer = _error_and_frames(cfg)
    # Preserve objects before any subsequent validation can fail. Never clear
    # an original frame until the actual worker's close has completed.
    ctx.update(error=error, frames=frames, worker=consumer.f_locals['worker'],
               slots=consumer.f_locals['slots'], access=consumer.f_locals['access'])
    ctx['client'] = ctx['access'].port.client
    ctx['original_frames'] = [{'function': f.f_code.co_name, 'file': f.f_code.co_filename,
                               'line': f.f_lineno} for f in frames]
    if cfg['mode'] == 'n004':
        assert sys.modules['__main__']._ACTIVE_ACCESS is ctx['access']
        assert type(ctx['worker']).__module__ == 'pypto.runtime.distributed_runner'
        assert type(ctx['worker']).__name__ == 'DistributedWorker'
        assert type(ctx['client']).__module__ == 'uflow.managed_pool'
        assert type(ctx['client']).__name__ == 'ManagedPoolClient'
        assert not any(name in consumer.f_locals for name in ('backend', 'module', 'tensors', 'arrays'))
        worker = ctx['worker']
        assert worker._next_dispatch_id == 1 and not worker._active_dispatch_handles
        assert not worker._service_tokens and not worker._device_buffers
        assert not worker._close_complete
        descriptor = ctx['client'].view.descriptor
        assert descriptor['pool_id'] == POOL_ID and descriptor['client_session'] == CLIENT_ID
        assert descriptor['pool_epoch'] == cfg['pool_epoch'] and descriptor['provider_session'] == cfg['provider_session']
        assert descriptor['canonical'] == cfg['canonical']
        assert not ctx['access'].children and not ctx['access'].leases and ctx['access'].unknown
        assert not ctx['access'].port._children and not ctx['access'].port._leases
        assert not ctx['client']._children and all(x == 'Closed' for x in ctx['client']._leases.values())
        assert not ctx['slots'].leases and ctx['slots'].busy is None
        assert hashlib.sha256(Path(sys.modules['uflow.managed_pool'].__file__).read_bytes()).hexdigest() == cfg['SDK_sha256']
        assert hashlib.sha256(Path(sys.modules['pypto.pypto_core'].__file__).read_bytes()).hexdigest() == cfg['core_sha256']
        current = descendants(os.getpid())
        assert current == {int(k): v for k, v in cfg['children'].items()}
    else:
        main = sys.modules['__main__']
        ctx['fixture_control'] = main._FIXTURE_CONTROL
    if cfg.get('inject_install_failure'):
        assert cfg['mode'] == 'cpu-fixture'
        raise RuntimeError('injected install failure; no handler or cleanup was run')
    return {'pid': os.getpid(), 'birth': birth(os.getpid()), 'mode': cfg['mode'],
            'original_frames': ctx['original_frames'], 'objects_preserved': True}


def close_worker(ctx, cfg):
    ctx['worker'].close()
    assert ctx['worker']._close_complete
    assert not ctx['worker']._active_dispatch_handles
    assert not ctx['worker']._inherited_host_tensors
    for pid, expected in cfg['mapping_children'].items():
        assert birth(int(pid)) != expected, 'original mapping child still lives after Worker.close'
    remaining = descendants(os.getpid())
    assert set(remaining) <= {cfg['tracker_pid']}, 'unexpected child remains after native close'
    for pid in remaining:
        assert birth(pid) == cfg['tracker_birth']
        view = observation(cfg['physical'], pid)
        assert not view['maps'] and not view['fds'], 'unmapped helper unexpectedly holds pool'
    ctx['worker_close_receipt'] = {'returned_normally': True, 'close_complete': True,
        'mapping_children_original_births_gone': cfg['mapping_children'], 'remaining_children': remaining,
        'time_ns': time.monotonic_ns(), 'scope': 'actual Worker.close' if cfg['mode'] == 'n004' else 'CPU toy Worker only'}


def release_exports(ctx):
    # The executing main/handler frames are never in this allowlisted set.
    for frame in ctx['frames']:
        frame.clear()
    ctx['frames'] = []
    ctx['error'].__traceback__ = None
    ctx['slots'].pool_tensor = None
    ctx['slots'].page_ids = None
    ctx['slots'] = None
    # Worker.close has returned and its original mapping children are gone.
    # Retire the closed owner, including its prepared control tensor references.
    ctx['worker'] = None


def verify_unmapped(ctx, cfg):
    proof = observation(cfg['physical'])
    assert not proof['maps'] and not proof['fds'], 'actual pool VMA/FD still present'
    assert ctx['client'].view.closed
    ctx['unmapped_observation'] = proof
    return proof


def stop_tracker(ctx, cfg):
    from multiprocessing import resource_tracker
    if cfg['mode'] == 'cpu-fixture':
        ctx['fixture_control'].close()
        ctx['fixture_control'].unlink()
        ctx['fixture_control'] = None
    tracker = resource_tracker._resource_tracker
    assert tracker._pid == cfg['tracker_pid'] and birth(tracker._pid) == cfg['tracker_birth']
    tracker._stop()  # Normal stdlib pipe-close/waitpid path, never a kill signal.
    assert birth(cfg['tracker_pid']) != cfg['tracker_birth']
    assert not descendants(os.getpid())
    return {'original_tracker_pid': cfg['tracker_pid'], 'original_birth_gone': True,
            'method': 'stdlib resource_tracker._stop pipe close and wait', 'time_ns': time.monotonic_ns()}


def verify_snapshot(ctx, cfg):
    result = ctx['client'].snapshot()
    if cfg['mode'] == 'cpu-fixture':
        assert result['ack_count'] == 1 and ctx['client'].view.closed
        return result
    assert result['charged_bytes'] == result['budget_bytes'] == 268435456
    assert result['creation_reserved_bytes'] == 0
    original = next(p for p in result['pools'] if p['pool_id'] == cfg['original_pool']['pool_id'])
    assert json.dumps(original, sort_keys=True) == json.dumps(cfg['original_pool'], sort_keys=True)
    current = next(p for p in result['pools'] if p['pool_id'] == POOL_ID)
    assert not current['frozen'] and not current['ranges']['leases']
    client = next(c for c in current['clients'] if c['client_session'] == CLIENT_ID)
    assert client['state'] == 'Unmapped' and not client['children']
    return result
