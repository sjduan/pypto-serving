# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU/fake startup tests. No executor/model construction, device or native code.

The external-driver test invokes this file, not an NPU entry point. Its exit is
observed by the parent, and its final JSON is written only after asyncio.run
has joined the owned default executor. A timeout is a FAILURE, never a pass.
"""
from __future__ import annotations

import asyncio
import importlib.abc
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest


class _DeviceImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {"torch_npu", "pypto", "acl", "runtime"}:
            raise ImportError(f"device module forbidden in host test: {fullname}")
        return None


sys.meta_path.insert(0, _DeviceImportGuard())
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore
from pypto_serving.serving.engine.lifecycle import wait_worker_ready
from pypto_serving.serving.server.ipc import ShutdownCommand, decode_command


class _Event:
    def __init__(self, ready=False):
        self.ready = ready
        self.checks = 0

    def is_set(self):
        self.checks += 1
        return self.ready

    def wait(self, *args, **kwargs):
        raise AssertionError("a host startup must NEVER start a blocking Event.wait")


class _Process:
    pid = 123456  # identity of a fake, never used for OS signalling
    exitcode = None

    def __init__(self, *, alive=True, reaps=True, delay=0, error=None):
        self.alive, self.reaps, self.delay, self.error = alive, reaps, delay, error
        self.joins = 0
        self.signals = []
        self.startup_status = bytearray(4097)

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        assert timeout is not None
        self.joins += 1
        time.sleep(self.delay)
        if self.error:
            raise self.error
        if self.reaps:
            self.alive = False
            self.exitcode = 0

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")


class _Core(ReplicaEngineCore):
    """Keep the production start/stop; the serving loop is explicitly host-only."""
    async def _engine_loop(self):
        while self._running:
            await asyncio.sleep(0.001)


def _core(*, ready=False, capacity=0, process=None, timeout=0.08):
    event = _Event(ready)
    process = process or _Process()
    if ready:
        process.startup_status[0] = 1
    input_q = queue.Queue()
    factory = lambda config: (process, input_q, queue.Queue(), queue.Queue(), event, [capacity])
    core = _Core(EngineConfig(), tokenizer=SimpleNamespace(), worker_factory=factory)
    core._init_timeout = timeout
    return core, process, event, input_q


def _assert_owned_tasks_done(core):
    assert core._stop_task.done()
    assert core._loop_task is None
    assert not core.is_ready()
    assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]


def test_ready_requires_reliable_success_and_positive_capacity():
    async def check():
        core, process, _, messages = _core(ready=True, capacity=8)
        await core.start()
        assert core.is_ready()
        assert await core.stop()
        assert await core.stop()
        assert process.joins == 1
        assert isinstance(decode_command(messages.get_nowait()), ShutdownCommand)
        assert messages.empty()
        assert core.shutdown_report["native_children_stopped"] == "unknown"
        _assert_owned_tasks_done(core)
    asyncio.run(check())


@pytest.mark.parametrize("mode", ["zero", "init-error-ready", "exited", "never-ready", "pending-outcome"])
def test_startup_failure_never_admits_and_retains_original_error(tmp_path, mode):
    async def check():
        core, process, event, _ = _core(ready=True, capacity=8)
        if mode == "zero":
            core._worker_factory = lambda config: (process, queue.Queue(), queue.Queue(), queue.Queue(), event, [0])
        elif mode == "init-error-ready":
            error = b"original initialization failure after capacity publication"
            process.startup_status[1:1 + len(error)] = error
            process.startup_status[0] = 2
        elif mode == "exited":
            process.alive, process.exitcode = False, 7
        elif mode == "never-ready":
            event.ready = False
        else:
            process.startup_status[0] = 0
        with pytest.raises(RuntimeError) as exc:
            await core.start()
        assert core.startup_error
        assert not core.is_ready()
        assert process.joins == 1
        assert await core.stop()
        if mode == "init-error-ready":
            assert "original initialization failure" in str(exc.value)
        _assert_owned_tasks_done(core)
        result = {"passed": False, "case": mode, "error": core.startup_error,
                  "shutdown_report": core.shutdown_report, "host_only": True}
        target = tmp_path / (mode + ".json")
        target.write_text(json.dumps(result), encoding="utf-8")
        assert json.loads(target.read_text())["passed"] is False
    before = set(threading.enumerate())
    asyncio.run(check())
    assert set(threading.enumerate()) == before


@pytest.mark.parametrize("outer_timeout", [False, True])
def test_cancel_after_wait_entered_and_second_cancel_joins_cleanup(outer_timeout):
    async def check():
        core, process, event, _ = _core(process=_Process(delay=0.08), timeout=60)
        task = asyncio.create_task(asyncio.wait_for(core.start(), 0.03) if outer_timeout else core.start())
        while not event.checks:
            await asyncio.sleep(0.001)
        assert core._worker_process is process
        if not outer_timeout:
            task.cancel()
            while core._stop_task is None:
                await asyncio.sleep(0.001)
            task.cancel()  # cleanup must still retain its sole owner
        with pytest.raises(TimeoutError if outer_timeout else asyncio.CancelledError):
            await task
        assert process.joins == 1
        assert not process.is_alive()
        assert await core.stop()
        _assert_owned_tasks_done(core)
    before = set(threading.enumerate())
    asyncio.run(check())
    assert set(threading.enumerate()) == before


def test_stop_caller_double_cancel_cannot_abandon_cleanup_owner():
    async def check():
        core, process, _, _ = _core(ready=True, capacity=8, process=_Process(delay=0.08))
        await core.start()
        task = asyncio.create_task(core.stop())
        while not process.joins:
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.joins == 1
        assert not process.is_alive()
        assert await core.stop()
        _assert_owned_tasks_done(core)
    before = set(threading.enumerate())
    asyncio.run(check())
    assert set(threading.enumerate()) == before


def test_stop_false_is_sticky_and_unknown_is_not_global_release():
    async def check():
        core, process, _, _ = _core(process=_Process(reaps=False), timeout=0.001)
        with pytest.raises(RuntimeError, match="initialize"):
            await core.start()
        assert not await core.stop()
        assert not await core.stop()
        assert process.joins == 3
        assert process.signals == ["terminate", "kill"]  # fake methods only
        assert core.shutdown_report["worker_reaped"] is False
        assert core.shutdown_report["native_children_stopped"] == "unknown"
        _assert_owned_tasks_done(core)
    asyncio.run(check())


def test_cleanup_exception_does_not_replace_startup_error():
    class FailingCleanup(_Core):
        def _shutdown_worker(self, *, timeout):
            raise RuntimeError("cleanup-original")

    async def check():
        core, _, _, _ = _core(timeout=0.001)
        core.__class__ = FailingCleanup  # test fixture only; never a production adapter
        with pytest.raises(RuntimeError, match="initialize"):
            await core.start()
        assert "cleanup-original" in core.shutdown_report["cleanup_error"]
        assert "initialize" in core.startup_error
        for _ in range(2):
            with pytest.raises(RuntimeError, match="cleanup-original"):
                await core.stop()
        _assert_owned_tasks_done(core)
    asyncio.run(check())


def test_waiter_never_creates_a_default_executor():
    async def check():
        loop = asyncio.get_running_loop()
        with pytest.raises(RuntimeError, match="initialize"):
            await wait_worker_ready(_Process(), _Event(), [0], 0.001)
        assert loop._default_executor is None
    asyncio.run(check())


def test_external_cpu_driver_really_exits_and_joins_owned_threads(tmp_path):
    target = tmp_path / "driver-final.json"
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(target)],
                            capture_output=True, text=True, timeout=30, check=False,
                            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[4])})
    assert result.returncode == 0, result.stdout + result.stderr
    final = json.loads(target.read_text())
    assert final["stage"] == "after-asyncio-run-before-driver-exit"
    assert final["remaining_owned_threads"] == []
    assert final["business_passed"] is False
    assert final["shutdown_report"]["native_children_stopped"] == "unknown"


def test_real_entry_host_fixture_records_failure_even_after_capacity_was_written(monkeypatch):
    from pypto_serving.serving.server import serving_worker
    import signal

    calls = []
    class FakeWorker:
        def __init__(self, *args):
            self.executor = SimpleNamespace(cache_group_sizes=lambda model: {})

        def init_device_and_model(self):
            return 8

        def close(self):
            calls.append("closed")

    def fail_after_capacity():
        raise RuntimeError("original-host-init-error-after-capacity")

    monkeypatch.setattr(serving_worker, "WorkerProcess", FakeWorker)
    monkeypatch.setattr(serving_worker, "freeze_gc_heap", fail_after_capacity)
    monkeypatch.setattr(serving_worker.logging, "basicConfig", lambda **kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    event = _Event()
    event.set = lambda: setattr(event, "ready", True)
    process = _Process()
    capacity = [0]
    serving_worker._worker_entry(EngineConfig(), queue.Queue(), queue.Queue(), event, capacity,
                                 startup_status=process.startup_status)
    assert calls == ["closed"] and capacity == [8] and event.ready
    assert process.startup_status[0] == 2
    with pytest.raises(RuntimeError, match="original-host-init-error-after-capacity"):
        asyncio.run(wait_worker_ready(process, event, capacity, 1))


def test_stopped_core_cannot_spawn_again_using_an_old_completed_stop_owner():
    async def check():
        core, process, _, _ = _core(ready=True, capacity=8)
        await core.start()
        assert await core.stop()
        with pytest.raises(RuntimeError, match="fresh core"):
            await core.start()
        assert not process.alive and process.joins == 1
    asyncio.run(check())


if __name__ == "__main__":
    before = set(threading.enumerate())
    report = {}

    async def driver():
        core, _, event, _ = _core(process=_Process(delay=0.05), timeout=60)
        task = asyncio.create_task(core.start())
        while not event.checks:
            await asyncio.sleep(0.001)
        task.cancel()
        while core._stop_task is None:
            await asyncio.sleep(0.001)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            report["error"] = core.startup_error
        else:
            raise AssertionError("cancelled startup was silently reported successful")
        _assert_owned_tasks_done(core)
        report["shutdown_report"] = core.shutdown_report

    asyncio.run(driver())
    report.update(stage="after-asyncio-run-before-driver-exit", business_passed=False,
                  remaining_owned_threads=[t.name for t in set(threading.enumerate()) - before])
    Path(sys.argv[1]).write_text(json.dumps(report), encoding="utf-8")
    assert not report["remaining_owned_threads"]
