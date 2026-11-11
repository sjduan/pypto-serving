# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Separate CPU processes use the production fixed handler and actual exported mmap."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Linux signals and physical mmap identity')


def await_json(path, child):
    deadline = time.monotonic() + 10
    while not path.exists():
        assert child.poll() is None, 'CPU fixture died before its proof receipt'
        assert time.monotonic() < deadline, str(path)
        time.sleep(.02)
    return json.loads(path.read_text())


@pytest.mark.parametrize('mode', ['success', 'close-failure'])
def test_real_mmap_lifetime_and_failure_keepalive(tmp_path, mode):
    script = Path(__file__).with_name('fixtures') / 'cooperative_close_parent.py'
    with (tmp_path / 'parent.log').open('wb') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-s', str(script), str(tmp_path), mode],
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    retained = await_json(tmp_path / 'channel/retained-failure.json', child)
    assert retained['static_close_channel_ready'] and retained['pid'] == child.pid
    assert (tmp_path / 'channel/installed-before-registration.json').exists()
    child.send_signal(signal.SIGUSR1)
    if mode == 'success':
        success = await_json(tmp_path / 'channel/close-success.json', child)
        assert success['actual_SDK_view_closed'] and success['ACK']['ack_count'] == 1
        assert not success['pool_no_VMA_FD']['maps'] and not success['pool_no_VMA_FD']['fds']
    else:
        failure = await_json(tmp_path / 'channel/close-failed-1.json', child)
        assert failure['stage'] == 'actual-mmap-FD-close' and 'BufferError' in failure['error']
        assert failure['original_context_retained'] and not failure['ACK_done'] and child.poll() is None
        child.send_signal(signal.SIGTERM)  # CPU fixture's fixed final actual close; never a native target.
        final = await_json(tmp_path / 'fixture-final-close.json', child)
        assert final['actual_mmap_closed'] and final['ACK_count'] == 0
    assert child.wait(timeout=10) == 0
