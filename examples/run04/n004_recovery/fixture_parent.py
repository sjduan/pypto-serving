# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only real mmap/export fixture for the exact recovery payload; zero UFlow/device imports."""
from types import SimpleNamespace
import json
import mmap
from multiprocessing import resource_tracker, shared_memory
import os
from pathlib import Path
import signal
import sys

ROOT = Path(sys.argv[1])
CFG = json.loads((ROOT / 'config.json').read_text())
_FIXTURE_CONTROL = None
_FIXTURE_EXTRA_EXPORT = None
_FIXTURE_ACCESS = None
_FIXTURE_SLOTS = None
_FIXTURE_WORKER = None
_FIXTURE_ERROR = None


def write(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


class ToyWorker:
    def __init__(self, tensor):
        self._close_complete = False
        self._active_dispatch_handles = set()
        self._inherited_host_tensors = (tensor,)

    def close(self):
        self._inherited_host_tensors = ()
        self._close_complete = True


class ActualView:
    def __init__(self, fd, mapping):
        self.fd, self.mapping, self.fd_closed = fd, mapping, False

    @property
    def closed(self):
        return self.mapping.closed and self.fd_closed

    def close(self):
        self.mapping.close()  # Real exported-memory BufferError, not a mocked failure.
        if not self.fd_closed:
            os.close(self.fd)
            self.fd_closed = True


class CpuClient:
    def __init__(self, view):
        self.view, self.ack_count = view, 0

    def begin_unregister(self, *, operation_id):
        return {'stopping': True, 'operation_id': operation_id, 'CPU_fixture_only': True}

    def ack_unmapped(self, *, native_drained, operation_id):
        assert self.view.closed and native_drained is True and _FIXTURE_WORKER._close_complete
        self.ack_count += 1
        return {'unmapped': True, 'operation_id': operation_id, 'CPU_fixture_only': True}

    def snapshot(self):
        return {'ack_count': self.ack_count, 'UF_operations': 0, 'view_closed': self.view.closed}

    def query_operation(self, operation_id):
        return {'scope': 'CPU fixture only; no UFlow operation', 'operation_id': operation_id}


def consumer():
    global _FIXTURE_CONTROL, _FIXTURE_EXTRA_EXPORT, _FIXTURE_ACCESS, _FIXTURE_SLOTS, _FIXTURE_WORKER
    fd = os.open(ROOT / 'backing.bin', os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, 65536)
    mapping = mmap.mmap(fd, 65536, flags=mmap.MAP_SHARED)
    info = os.fstat(fd)
    slots = SimpleNamespace(pool_tensor=memoryview(mapping), page_ids=bytearray(4))
    worker = ToyWorker(slots.pool_tensor)
    access = SimpleNamespace(buffer=memoryview(mapping), port=SimpleNamespace(client=CpuClient(ActualView(fd, mapping))))
    _FIXTURE_WORKER, _FIXTURE_SLOTS, _FIXTURE_ACCESS = worker, slots, access
    _FIXTURE_CONTROL = shared_memory.SharedMemory(create=True, size=64)
    derived = memoryview(mapping)[128:4096]
    derived[:] = b'A' * len(derived)
    if CFG.get('inject_close_failure'):
        _FIXTURE_EXTRA_EXPORT = memoryview(mapping)[4096:8192]
    write('ready.json', {'pid': os.getpid(), 'physical': [info.st_dev, info.st_ino],
                        'tracker_pid': resource_tracker._resource_tracker._pid, 'UF_device_operations': 0})
    raise RuntimeError('CPU-only original consumer retained a real exported mmap')


def abort_fixture(signum, frame):
    del signum, frame
    global _FIXTURE_EXTRA_EXPORT
    try:
        _FIXTURE_WORKER.close()
        _FIXTURE_SLOTS.pool_tensor = None
        _FIXTURE_SLOTS.page_ids = None
        if _FIXTURE_EXTRA_EXPORT is not None:
            _FIXTURE_EXTRA_EXPORT.release()
            _FIXTURE_EXTRA_EXPORT = None
        target = _FIXTURE_ERROR.__traceback__
        while target is not None:
            if target.tb_frame.f_code.co_name == 'consumer':
                target.tb_frame.clear()
            target = target.tb_next
        _FIXTURE_ERROR.__traceback__ = None
        _FIXTURE_ACCESS.buffer.release()
        _FIXTURE_ACCESS.port.client.view.close()
        _FIXTURE_CONTROL.close()
        _FIXTURE_CONTROL.unlink()
        resource_tracker._resource_tracker._stop()
        write('fixture-graceful-stop.json', {'pid': os.getpid(), 'actual_mmap_closed': True,
            'payload_ACK_count': _FIXTURE_ACCESS.port.client.ack_count, 'scope': 'CPU fixture graceful final cleanup, no UFlow/device'})
    except BaseException as stop_error:
        write('fixture-stop-failed.json', {'error': repr(stop_error)})
        return
    raise SystemExit(0)


signal.signal(signal.SIGTERM, abort_fixture)  # This fixture-only control is a cooperative close, not SIGKILL.
try:
    consumer()
except BaseException as error:
    _FIXTURE_ERROR = error
    write('pause-ready.json', {'pid': os.getpid(), 'error': repr(error), 'consumer_unwound': True})
    while True:
        try:
            signal.pause()
        except KeyboardInterrupt:
            continue
