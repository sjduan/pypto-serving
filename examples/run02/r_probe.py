# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real CLI-built engine and HTTP R probes; no replacement allocator or executor."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import socket
import time
import traceback

from m2_baseline import LONG_PROMPT, SHORT_PROMPT, record_processes, validate_output, write_json


async def exercise(args, report):
    import httpx
    import uvicorn
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
    from pypto_serving.serving.server.server import create_serving_app

    cli = ["--model", "/data/models/Qwen/Qwen3-14B", "--backend", "npu", "--platform", "a2a3",
           "--device", str(args.device), "--max-model-len", "512", "--max-num-seqs", "16",
           "--max-num-batched-tokens", "512", "--long-prefill-token-threshold", "128",
           "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--block-size", "128",
           "--ring-dep-pool", "262144", "--ring-task-window", "262144", "--ring-heap", "1073741824",
           "--npu-memory-utilization", str(args.utilization)]
    cli += ["--enable-native-ddr-kv", "--native-ddr-kv-bytes", str(16 * 20971520)] if args.native_enabled else [
        "--no-enable-native-ddr-kv"]
    config = build_serving_engine_config(build_parser().parse_args(cli))
    report.update(cli_arguments=cli, engine_config=asdict(config), requests=[], coordinator_trace=[])
    engine = AsyncLLMEngine(config, load_tokenizer(config.model_dir))
    core = engine._cores[0]
    generate = GenerateConfig(max_new_tokens=4, temperature=0.0, top_p=1.0, top_k=None, seed=0, stream=True)
    server = server_task = None
    last_sequence = 0

    def collect_trace():
        nonlocal last_sequence
        coord = core._native_ddr
        if coord is not None:
            fresh = [e for e in coord.trace if e["sequence"] > last_sequence]
            report["coordinator_trace"].extend(fresh)
            if fresh:
                last_sequence = fresh[-1]["sequence"]

    async def collect(name, prompt, generation=generate):
        request_id = engine.generate_request_id()
        record = {"name": name, "request_id": request_id, "prompt": prompt,
                  "prompt_token_ids": engine.tokenizer.encode(prompt), "generate_config": asdict(generation),
                  "outputs": []}
        report["requests"].append(record)
        async for output in engine.add_request(request_id, prompt, generation):
            record["outputs"].append(asdict(output))
        ids = validate_output(record)
        assert engine.tokenizer.decode(ids) == record["outputs"][-1]["text"]
        collect_trace()
        write_json(args.output, report)
        return record

    try:
        await engine.start()
        report["actual_pages"] = core.kv_cache_manager.num_blocks
        report["processes_loaded"] = record_processes()
        if args.native_enabled:
            report["startup"] = asdict(core._native_ddr_startup)
            report["startup"]["namespace_digest"] = core._native_ddr_startup.namespace_digest.hex()
            report["startup"]["compatibility_json"] = json.loads(core._native_ddr_startup.compatibility_json)
            assert core._native_ddr is core.scheduler._native_ddr and core._native_ddr.max_entries == 16
        else:
            assert core._native_ddr is None and core._native_ddr_output_queue is None
            assert not hasattr(core._worker_process, "native_ddr_output_queue")
        await collect("short-cold", SHORT_PROMPT)
        cold = await collect("prefix-cold", LONG_PROMPT)
        warm = await collect("prefix-hbm", LONG_PROMPT)
        await asyncio.gather(collect("batch-France", SHORT_PROMPT), collect("batch-Japan", "The capital of Japan is"))
        app = create_serving_app(engine, config.model_id, replace(generate, stream=False))
        sock = socket.socket(fileno=args.listen_fd)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=18892, lifespan="off", log_level="info"))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            if server_task.done():
                await server_task
                raise RuntimeError("HTTP exited before readiness")
            await asyncio.sleep(.02)
        assert server.started
        payload = {"model": config.model_id, "prompt": SHORT_PROMPT, "max_tokens": 4,
                   "temperature": 0.0, "top_p": 1.0, "seed": 0, "stream": False}
        async with httpx.AsyncClient(trust_env=False, timeout=180) as client:
            response = await client.post("http://127.0.0.1:18892/v1/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            assert body["choices"][0]["text"].strip() and body["usage"]["completion_tokens"] == 4
            assert body["usage"]["prompt_tokens"] == len(engine.tokenizer.encode(SHORT_PROMPT))
            report["http"] = {"status": response.status_code, "payload": payload, "body": body}
        if args.case == "matrix":
            await matrix(engine, core, collect, collect_trace, args, report, cold, warm)
        collect_trace()
        report["passed"] = True
    finally:
        if server is not None:
            server.should_exit = True
        if server_task is not None:
            await server_task
        collect_trace()
        report["stop_ok"] = await engine.stop()
        report["shutdown_report"] = core.shutdown_report
        report["metrics_final"] = engine.metrics.snapshot()
        collect_trace()
        write_json(args.output, report)
    assert report["stop_ok"]
    return report


async def matrix(engine, core, collect, collect_trace, args, report, cold, warm):
    raise NotImplementedError("full R matrix is not implemented yet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["smoke", "matrix"], required=True)
    parser.add_argument("--native-enabled", type=int, choices=[0, 1], required=True)
    parser.add_argument("--utilization", type=float, default=.80)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--listen-fd", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"passed": False, "case": args.case, "native_enabled": bool(args.native_enabled),
              "utilization": args.utilization, "device": args.device, "pid": os.getpid()}
    try:
        asyncio.run(exercise(args, report))
    except BaseException:
        report["passed"] = False
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
