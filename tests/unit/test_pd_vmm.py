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


def test_imported_address_and_retirement_order(monkeypatch):
    events = []
    class Context:
        def __init__(self, device): assert device == 4
        def bind(self): pass
        def close(self): events.append("context-close")
    class Mapping:
        base = 8192
        descriptor = SimpleNamespace(nbytes=1024)
        def __init__(self, exported):
            assert exported.shareable_handle == 42
            events.append("import")
        def close(self): events.append("unmap")
    class Provider:
        def register(self, boot, rank, name, address, extent):
            assert address == 8192 and extent == 1024
            events.append("register")
            return transfer.Region(boot, rank, name, extent, {"address": address})
        def release(self): events.append("unregister-disconnect")
    monkeypatch.setattr(transfer, "DeviceContext", Context)
    monkeypatch.setattr(transfer, "configure_profiler", lambda *a, **kw: None)
    sdk = ModuleType("simpler.buffer_export")
    sdk.BufferExport, sdk.BufferMapping = Export, Mapping
    monkeypatch.setitem(sys.modules, "simpler.buffer_export", sdk)
    worker = transfer.TransferWorker("boot", 0, "host")
    thread = threading.Thread(target=transfer._serve, args=(worker.child, "boot", 0, 4, "host",
                              "fake", lambda *a: Provider(), worker.profile_config), daemon=True)
    thread.start()
    worker.process = SimpleNamespace(join=thread.join, is_alive=thread.is_alive, exitcode=0)
    runtime = SimpleNamespace(export_buffer=lambda tensor, **kw: Export(),
                              release_export=lambda exported: events.append("source-release"))
    try:
        region = worker.register(runtime, SimpleNamespace(nbytes=1024), "cache")
        assert region.handle["address"] == 8192
        with pytest.raises(RuntimeError, match="registered"):
            worker.close()
        worker.release()
        assert events == ["import", "register", "unregister-disconnect", "unmap", "source-release"]
        worker.close()
        assert events[-2:] == ["unregister-disconnect", "context-close"]
        assert not thread.is_alive()
    finally:
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
