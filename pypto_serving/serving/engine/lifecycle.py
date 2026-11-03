# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only startup waiting. No blocking Event.wait or executor ownership."""
from __future__ import annotations

import asyncio


async def wait_worker_ready(process, ready_event, cache_capacity, timeout: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        status = getattr(process, "startup_status", None)
        if status is not None and status[0] == 2:
            error = bytes(status[1:]).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            raise RuntimeError(f"Worker initialization failed: {error}")
        if not process.is_alive():
            raise RuntimeError(f"Worker exited during initialization (exitcode={process.exitcode})")
        if ready_event.is_set():
            # The worker also sets ready on an exception. Capacity is committed
            # only after initialization; zero must never admit serving.
            if cache_capacity[0] <= 0:
                raise RuntimeError("Worker initialization failed: invalid KV cache capacity")
            if status is None or status[0] == 1:
                return
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RuntimeError(f"Worker failed to initialize within {timeout:g}s timeout")
        await asyncio.sleep(min(0.02, remaining))


async def finish_cleanup(task: asyncio.Task):
    """Keep ownership through repeated cancellation until bounded cleanup ends."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
