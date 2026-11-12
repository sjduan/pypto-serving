# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent transfer-process mappings must retire before source leases."""

from dataclasses import dataclass
import ctypes
import os
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from pypto_serving.pd import transfer


@dataclass(frozen=True)
class Export:
    device_id: int = 4
    descriptor: bytes = b"backing-identity"
    token: int = 1
    shareable_handle: int = 42
    mapping_bytes: int = 4096


@pytest.fixture(autouse=True)
def buffer_sdk(monkeypatch):
    class ExportError(RuntimeError):
        def __init__(self, exported):
            self.exported = exported
    class ImportError(RuntimeError):
        def __init__(self, mapping):
            self.mapping = mapping
    sdk = ModuleType("simpler.buffer_export")
    sdk.BufferExport = Export
    sdk.BufferExportError, sdk.BufferImportError = ExportError, ImportError
    monkeypatch.setitem(sys.modules, "simpler.buffer_export", sdk)
    return sdk


@pytest.mark.parametrize("failure", [None, "import", "unmap"])
def test_imported_address_and_retirement_order(monkeypatch, buffer_sdk, failure):
    events = []
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "16666")
    monkeypatch.setenv("HCCL_HOST_SOCKET_PORT_RANGE", "16666")
    class Context:
        def __init__(self, device):
            assert device == 4
            assert os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] == "auto"
            assert os.environ["HCCL_HOST_SOCKET_PORT_RANGE"] == "auto"
        def bind(self): pass
        def close(self): events.append("context-close")
    class Mapping:
        base = 8192
        descriptor = SimpleNamespace(nbytes=1024)
        def __init__(self, exported):
            assert exported.shareable_handle == 42
            events.append("import")
            self.close_failed = False
            if failure == "import":
                raise buffer_sdk.BufferImportError(self)
        def close(self):
            if failure == "unmap" and not self.close_failed:
                self.close_failed = True
                events.append("unmap-failed")
                raise RuntimeError("unmap failed")
            events.append("unmap")
    class Provider:
        def register(self, boot, rank, name, address, extent):
            assert address == 8192 and extent == 1024
            events.append("register")
            return transfer.Region(boot, rank, name, extent, {"address": address})
        def release(self): events.append("unregister-disconnect")
    monkeypatch.setattr(transfer, "DeviceContext", Context)
    monkeypatch.setattr(transfer, "configure_profiler", lambda *a, **kw: None)
    buffer_sdk.BufferMapping = Mapping
    worker = transfer.TransferWorker("boot", 0, "host")
    thread = threading.Thread(target=transfer._serve, args=(worker.child, "boot", 0, 4, "host",
                              "fake", lambda *a: Provider(), worker.profile_config), daemon=True)
    thread.start()
    worker.process = SimpleNamespace(join=thread.join, is_alive=thread.is_alive, exitcode=0)
    runtime = SimpleNamespace(export_buffer=lambda tensor, **kw: Export(),
                              release_export=lambda exported: events.append("source-release"))
    try:
        if failure == "import":
            with pytest.raises(transfer.TransferError) as error:
                worker.register(runtime, SimpleNamespace(nbytes=1024), "cache")
            assert not error.value.uncertain and not worker.uncertain
            assert events == ["import"]
        else:
            region = worker.register(runtime, SimpleNamespace(nbytes=1024), "cache")
            assert region.handle["address"] == 8192
        with pytest.raises(RuntimeError, match="registered"):
            worker.close()
        if failure == "unmap":
            with pytest.raises(transfer.TransferError) as error:
                worker.release()
            assert not error.value.uncertain and not worker.uncertain
            assert worker.retained and "source-release" not in events
        worker.release()
        expected = ["import"] + ([] if failure == "import" else ["register"])
        if failure == "unmap":
            expected += ["unregister-disconnect", "unmap-failed"]
        assert events == expected + ["unregister-disconnect", "unmap", "source-release"]
        worker.close()
        assert events[-2:] == ["unregister-disconnect", "context-close"]
        assert not thread.is_alive()
    finally:
        worker.parent.close()


def test_unconfirmed_export_remains_reachable_until_release(buffer_sdk):
    exported = Export()
    worker = transfer.TransferWorker("boot", 0, "host")
    commands, releases = [], []
    def export(tensor, **kwargs):
        raise buffer_sdk.BufferExportError(exported)
    def release(value):
        releases.append(value)
        if len(releases) == 1:
            raise TimeoutError("retain confirmation lost")
    worker.call = lambda operation, **kwargs: commands.append(operation)
    runtime = SimpleNamespace(export_buffer=export, release_export=release)
    try:
        with pytest.raises(buffer_sdk.BufferExportError):
            worker.register(runtime, SimpleNamespace(nbytes=1024), "cache")
        assert not commands and worker.retained == [(runtime, exported)]
        with pytest.raises(RuntimeError, match="registered"):
            worker.close()
        with pytest.raises(TimeoutError):
            worker.release()
        assert worker.retained == [(runtime, exported)]
        worker.release()
        assert not worker.retained and releases == [exported, exported]
        worker.close()
    finally:
        worker.child.close()
        worker.parent.close()


@pytest.mark.parametrize("operation", ["register", "release"])
def test_unknown_mapping_operation_retains_source_lease(operation):
    events = []
    worker = transfer.TransferWorker("boot", 0, "host")
    exported = Export()
    runtime = SimpleNamespace(export_buffer=lambda tensor, **kw: exported,
                              release_export=lambda export: events.append(export))
    def call(op, **kwargs):
        if op == operation:
            raise transfer.TransferError("native completion uncertain", uncertain=True)
        return transfer.Region("boot", 0, "cache", 1024, {})
    worker.call = call
    try:
        if operation == "register":
            with pytest.raises(transfer.TransferError):
                worker.register(runtime, SimpleNamespace(nbytes=1024), "cache")
        else:
            worker.register(runtime, SimpleNamespace(nbytes=1024), "cache")
            with pytest.raises(transfer.TransferError):
                worker.release()
        assert worker.retained == [(runtime, exported)]
        assert not events
        with pytest.raises(RuntimeError, match="registered"):
            worker.close()
    finally:
        worker.child.close()
        worker.parent.close()


def test_only_transfer_storage_is_exportable():
    from pypto_serving.pd.adapter import DSparkExtension
    assert DSparkExtension.allocation_options("cache") == {"exportable": True}
    assert DSparkExtension.allocation_options("prefill_state", "hidden") == {"exportable": True}
    assert DSparkExtension.allocation_options("prefill_state", "positions") == {}
    assert DSparkExtension.allocation_options("decode", "hidden") == {}


def test_transfer_channels_start_after_compute_owners(monkeypatch):
    from pypto_serving.pd import worker as module
    events = []
    class Transfer:
        retained = ()
        def __init__(self, *args): events.append("channel")
        def start(self, device): events.append(device)
        def close(self): events.append("close")
    monkeypatch.setattr(module, "TransferWorker", Transfer)
    model = SimpleNamespace(ranks=1, worker=lambda: SimpleNamespace(dc=SimpleNamespace(device_ids=[4])))
    settings = SimpleNamespace(boot="boot", hostname="host", provider="fake")
    runtime = module.WorkerRuntime(settings, model)
    assert not events and not runtime.owners
    runtime.worker_ready()
    assert events == ["channel", 4]
    runtime.close()
    assert events[-1] == "close"


def test_probe_checks_payload_and_guards_without_cache_storage():
    backing = ctypes.create_string_buffer(192)
    def malloc(pointer, size, policy):
        assert size == 192
        pointer._obj.value = ctypes.addressof(backing)
        return 0
    def copy(destination, capacity, source, size, direction):
        ctypes.memmove(destination, source, size)
        return 0
    context = transfer.DeviceContext.__new__(transfer.DeviceContext)
    context.probe = ctypes.c_void_p()
    context.acl = SimpleNamespace(aclrtMalloc=malloc, aclrtMemcpy=copy)
    address = context.probe_address()
    with pytest.raises(RuntimeError, match="payload/guard"):
        context.check_probe()
    ctypes.memmove(address + 64, transfer.PROBE_BYTES, 64)
    context.check_probe()
    ctypes.memset(address + 128, 0, 1)
    with pytest.raises(RuntimeError, match="payload/guard"):
        context.check_probe()


def test_preflight_writes_all_edges_and_retains_unknown_failure():
    from concurrent.futures import Future
    from pypto_serving.pd.worker import WorkerRuntime
    calls = []
    def write(operation, **payload):
        calls.append(payload["copies"][0])
    class Pool:
        def submit(self, action, *args, **kwargs):
            future = Future()
            try:
                future.set_result(action(*args, **kwargs))
            except Exception as error:
                future.set_exception(error)
            return future
    runtime = WorkerRuntime(SimpleNamespace(), SimpleNamespace(ranks=2))
    runtime.pool.shutdown()
    runtime.pool = Pool()
    runtime.owners = (SimpleNamespace(call=write), SimpleNamespace(call=write))
    runtime.peers["peer"] = ()
    edges = [(0, 0), (0, 1), (1, 0), (1, 1)]
    runtime.preflight(dict(peer="peer", mapping=edges))
    assert [(c.source_rank, c.destination_rank) for c in calls] == edges
    assert all((c.component, c.source_offset, c.destination_offset, c.length) ==
               (transfer.PROBE_COMPONENT, 0, 64, 64) for c in calls)
    def fail(*args, **kwargs):
        raise transfer.TransferError("native channel failed", uncertain=True)
    runtime.owners[0].call = fail
    with pytest.raises(transfer.TransferError):
        runtime.preflight(dict(peer="peer", mapping=edges))
    assert runtime.uncertain and not runtime.checking
