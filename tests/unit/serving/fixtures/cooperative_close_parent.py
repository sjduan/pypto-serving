# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only real mmap fixture for the production fixed close channel; no real SDK/device calls."""
import json
import mmap
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

from pypto_serving.serving.native_ddr.cooperative_close import CooperativePoolClose

ROOT = Path(sys.argv[1])
MODE = sys.argv[2]
EXTRA = None
RECOVERY = CooperativePoolClose(ROOT / 'channel')


class View:
    descriptor = {'client_session': 'cpu-client'}

    def __init__(self, fd):
        self.fd, self.mapping, self.closed = fd, mmap.mmap(fd, 65536), False

    def close(self):
        self.mapping.close()
        os.close(self.fd)
        self.closed = True


class Client:
    def __init__(self, fd):
        self.view, self._leases, self._children = View(fd), {}, {}
        self.ack_count = 0

    def begin_unregister(self, *, operation_id):
        return {'CPU_only': True, 'operation_id': operation_id}

    def ack_unmapped(self, *, operation_id, native_drained):
        assert self.view.closed and native_drained
        self.ack_count += 1
        return {'CPU_only': True, 'operation_id': operation_id, 'ack_count': self.ack_count}

    def query_operation(self, op):
        return {'CPU_only': True, 'operation_id': op, 'ack_count': self.ack_count}

    def snapshot(self):
        return {'CPU_only': True, 'pools': [{'pool_id': 'cpu-pool', 'ranges': {'leases': []},
            'clients': [{'client_session': 'cpu-client', 'state': 'Unmapped', 'children': []}]}]}


def consumer():
    global EXTRA
    assert (ROOT / 'channel/installed-before-registration.json').exists()
    fd = os.open(ROOT / 'backing.bin', os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.ftruncate(fd, 65536)
    actual = os.fstat(fd)
    client = Client(fd)
    access = SimpleNamespace(identity=SimpleNamespace(device=actual.st_dev, inode=actual.st_ino, pool_id='cpu-pool'),
        buffer=memoryview(client.view.mapping), port=SimpleNamespace(client=client, _leases={}), leases={}, children={})
    RECOVERY.bind_access(access)
    derived = memoryview(client.view.mapping)[128:4096]
    derived[:] = b'Z' * len(derived)
    if MODE == 'close-failure':
        EXTRA = memoryview(client.view.mapping)[4096:8192]
    raise RuntimeError('CPU fixture original consumer failure')


def stop_failed_fixture(signum, current_frame):
    del signum, current_frame
    global EXTRA
    if EXTRA is not None:
        EXTRA.release()
        EXTRA = None
    RECOVERY._release_unwound_exports()
    RECOVERY.access.buffer.release()
    RECOVERY.access.port.client.view.close()
    (ROOT / 'fixture-final-close.json').write_text(json.dumps({'actual_mmap_closed': True,
        'ACK_count': RECOVERY.access.port.client.ack_count, 'CPU_only': True}))
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop_failed_fixture)
try:
    consumer()
except BaseException as error:
    RECOVERY.retain_failure(error)
