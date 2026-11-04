# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real single-replica serving baseline; no alternate executor or KV offload implementation.

Scheduler instrumentation is test-only: it records and returns the original schedule unchanged.
The HTTP app, engine, worker and Qwen executor are the production implementations.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, replace
import fcntl
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

from identity import artifact
from worker_probe import device_users, selected_chip_status

SHORT_PROMPT = "The capital of France is"
LONG_PROMPT = "Paris is the capital of France. " * 25
M1_SHA = "ae0666ce937bafb456c50606705ec28c3d8ad489"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def descendants(pid):
    found = []
    children = Path(f"/proc/{pid}/task/{pid}/children")
    if children.exists():
        for child in map(int, children.read_text().split()):
            found.append(child)
            found.extend(descendants(child))
    return found


def unix_sockets(root):
    return [line for line in Path("/proc/net/unix").read_text().splitlines() if str(root) in line]


def record_processes():
    result = {}
    for pid in [os.getpid(), *descendants(os.getpid())]:
        try:
            result[str(pid)] = {
                "cmdline": Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " "),
                "maps": [line for line in Path(f"/proc/{pid}/maps").read_text().splitlines()
                         if any(name in line for name in
                                ("libhost_runtime", "pypto_core", "_task_interface", "libascendcl"))],
            }
        except FileNotFoundError:
            continue
    return result


def validate_output(record):
    final = record["outputs"][-1]
    if not final["finished"] or final["finish_reason"] == "error":
        raise AssertionError(f"No normal completion: {record['name']}")
    if len(final["token_ids"]) < 8 or not final["text"].strip():
        raise AssertionError(f"Trivial output: {record['name']}")
    return final["token_ids"]


async def exercise(args, root, result, sock):
    import httpx
    import pypto
    import pypto.pypto_core
    import pypto_serving
    import torch
    import uvicorn
    from pypto_serving.cli.main import build_parser, build_serving_engine_config
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.model.tokenizer import load_tokenizer
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
    from pypto_serving.serving.server.server import create_serving_app

    cli_args = ["--model", "/data/models/Qwen/Qwen3-14B", "--backend", "npu",
                "--platform", "a2a3", "--device", str(args.device), "--max-model-len", "512",
                "--max-num-seqs", "16", "--max-num-batched-tokens", "512",
                "--long-prefill-token-threshold", "128", "--dtype", "bfloat16",
                "--kv-cache-dtype", "bfloat16", "--block-size", "128",
                "--ring-dep-pool", "262144", "--ring-task-window", "262144",
                "--ring-heap", "1073741824", "--npu-memory-utilization", "0.80"]
    config = build_serving_engine_config(build_parser().parse_args(cli_args))
    generate = GenerateConfig(max_new_tokens=32, temperature=0.0, top_p=1.0,
                              top_k=None, seed=0, stream=True)
    result.update(cli_arguments=cli_args, engine_config=asdict(config), generate_config=asdict(generate),
                  prompt_mode="raw completion; no chat template or thinking-mode transformation",
                  offload="native DDR disabled; original HBM prefix caching remains enabled",
                  imports={"serving": artifact(pypto_serving.__file__),
                           "pypto": artifact(pypto.__file__), "core": artifact(pypto.pypto_core.__file__)},
                  torch_version=torch.__version__, python_version=sys.version)
    serving = Path(os.environ["PYPTO_SERVING_HOME"])
    if not Path(pypto_serving.__file__).resolve().is_relative_to(serving):
        raise RuntimeError("Wrong serving import source")
    engine = AsyncLLMEngine(config=config, tokenizer=load_tokenizer(config.model_dir))
    events = []
    original_schedule = engine.scheduler.schedule

    def observe_schedule(*call_args, **kwargs):
        output = original_schedule(*call_args, **kwargs)
        if output.scheduled_requests:
            events.append({"time": time.monotonic(), "requests": [
                {"request_id": sr.request.request_id, "is_prefill": sr.is_prefill,
                 "num_new_tokens": sr.num_new_tokens} for sr in output.scheduled_requests]})
        return output

    engine.scheduler.schedule = observe_schedule
    result["requests"] = []
    server = None
    server_task = None

    async def collect(name, prompt, generation=generate):
        request_id = engine.generate_request_id()
        record = {"name": name, "request_id": request_id, "prompt": prompt,
                  "prompt_token_ids": engine.tokenizer.encode(prompt),
                  "generate_config": asdict(generation), "outputs": [],
                  "metrics_before": engine.metrics.snapshot()}
        result["requests"].append(record)
        start = time.monotonic()
        async for output in engine.add_request(request_id, prompt, generation):
            record["outputs"].append({"elapsed_s": time.monotonic() - start, **asdict(output)})
        record["metrics_after"] = engine.metrics.snapshot()
        record["elapsed_s"] = time.monotonic() - start
        write_json(args.output, result)
        print(json.dumps({"request": name, "final": record["outputs"][-1]}), flush=True)
        validate_output(record)
        return record

    try:
        await engine.start()
        result["processes_loaded"] = record_processes()
        from simpler_setup.runtime_builder import RuntimeBuilder
        backend = RuntimeBuilder("a2a3").get_binaries("tensormap_and_ringbuffer", build=False)
        result["expected_native_backend"] = artifact(backend.host_path)
        if not any(str(backend.host_path) in line
                   for proc in result["processes_loaded"].values() for line in proc["maps"]):
            raise RuntimeError("Expected native child backend was not mapped")
        short = [await collect(f"short-cold-{i}", SHORT_PROMPT) for i in range(3)]
        # Every short prompt is below a complete cache page, so repeats recompute prefill.
        long = [await collect(f"long-{'cold' if i == 0 else 'warm'}-{i}", LONG_PROMPT) for i in range(3)]
        if not 128 < len(long[0]["prompt_token_ids"]) < 480:
            raise AssertionError("Long prompt does not cross a complete prefix page")
        batch = await asyncio.gather(*(collect(f"batch-{i}", SHORT_PROMPT) for i in range(4)))
        # Serve the same production API over a real exclusively bound TCP socket.
        app = create_serving_app(engine, config.model_id, replace(generate, stream=False))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port,
                                               lifespan="off", log_level="info"))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        for _ in range(200):
            if server.started:
                break
            if server_task.done():
                await server_task
                raise RuntimeError("HTTP server exited before becoming ready")
            await asyncio.sleep(0.05)
        if not server.started:
            raise RuntimeError("HTTP server startup timed out")
        payload = {"model": config.model_id, "prompt": SHORT_PROMPT,
                   "max_tokens": 32, "temperature": 0.0, "top_p": 1.0, "top_k": None,
                   "seed": 0, "stream": False}
        async with httpx.AsyncClient(trust_env=False, timeout=1200) as client:
            response = await client.post(f"http://127.0.0.1:{args.port}/v1/completions", json=payload)
            result["http"] = {"url": str(response.url), "payload": payload,
                              "status": response.status_code, "raw_body": response.text}
            response.raise_for_status()
            body = response.json()
            if body["choices"][0]["text"] != short[0]["outputs"][-1]["text"]:
                raise AssertionError("HTTP response differs from the token-recorded engine baseline")
            if body["usage"]["completion_tokens"] != 32:
                raise AssertionError("HTTP request did not execute the expected decode length")
        # Cancel after actual output, then complete a fresh request to prove continued usability.
        cancel_id = engine.generate_request_id()
        stream = engine.add_request(cancel_id, SHORT_PROMPT, replace(generate, max_new_tokens=128))
        try:
            async for output in stream:
                if output.token_id is not None:
                    result["cancel"] = {"request_id": cancel_id, "observed": asdict(output)}
                    await engine.abort_request(cancel_id)
                    break
        finally:
            await stream.aclose()
        recovery = await collect("after-cancel", SHORT_PROMPT)
        same_short = [*short, *batch, recovery]
        if any(validate_output(r) != validate_output(short[0]) for r in same_short):
            raise AssertionError("Greedy short repeat/batch/cancel baseline is not token-exact")
        if any(validate_output(r) != validate_output(long[0]) for r in long):
            raise AssertionError("Greedy cold/warm prefix baseline is not token-exact")
        counters = Counter(sr["request_id"] for event in events for sr in event["requests"] if sr["is_prefill"])
        if counters[long[0]["request_id"]] < 2:
            raise AssertionError("Chunked long prefill was not actually scheduled")
        if max(len(event["requests"]) for event in events) < 2:
            raise AssertionError("Concurrent requests were not actually co-batched")
        cold_metrics = long[0]["metrics_after"]["replicas"][0]["counters"]
        warm_metrics = long[-1]["metrics_after"]["replicas"][0]["counters"]
        if warm_metrics["prefix_cache_hits"] <= cold_metrics["prefix_cache_hits"]:
            raise AssertionError("Repeated long prefix did not actually hit HBM cache")
        result["comparison"] = {"short_repeat_batch_and_after_cancel": "exact_full_token_ids",
                                "long_cold_warm": "exact_full_token_ids",
                                "http": "exact_text_and_completion_count",
                                "tolerance": "none; greedy observations, not a universal determinism guarantee"}
        result["passed"] = True
    finally:
        if server is not None:
            server.should_exit = True
        try:
            if server_task is not None:
                await server_task
        finally:
            result["metrics_final"] = engine.metrics.snapshot()
            result["schedule_events"] = events
            result["processes_before_stop"] = record_processes()
            result["engine_stop_ok"] = await engine.stop()
            result["descendants_after_stop"] = descendants(os.getpid())
            # Python's resource tracker stays until this driver closes its pipe at exit;
            # it owns no NPU FD and is not a serving/native worker. Verify its exit externally.
            remaining = record_processes()
            result["auxiliary_trackers_until_driver_exit"] = [
                pid for pid in result["descendants_after_stop"]
                if "multiprocessing.resource_tracker" in remaining.get(str(pid), {}).get("cmdline", "")]
            result["native_children_remaining"] = [
                pid for pid in result["descendants_after_stop"]
                if pid not in result["auxiliary_trackers_until_driver_exit"]]
            write_json(args.output, result)
        if not result["engine_stop_ok"] or result["native_children_remaining"]:
            raise RuntimeError("Engine/native children were not reaped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--port", type=int, default=18892)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    root = Path(os.environ["UFLOW_RUN02_ROOT"]).resolve()
    if not root.is_relative_to("/workspace/uflow") or not args.output.resolve().is_relative_to(root):
        raise ValueError("All outputs must belong to the dedicated task root")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"passed": False, "device": args.device, "port": args.port, "pid": os.getpid(),
              "m1_reviewed_sha": M1_SHA, "root": str(root), "command": sys.argv,
              "host": os.uname().nodename, "unix_sockets_before": unix_sockets(root)}
    locks = Path("/workspace/uflow/locks")
    locks.mkdir(exist_ok=True)
    with (locks / f"run02-npu-{args.device}.lock").open("a+") as lock, socket.socket() as sock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Binding and retaining the socket reserves the real port without a TOCTOU port check.
        sock.bind(("127.0.0.1", args.port))
        sock.listen(128)
        before = subprocess.check_output(["npu-smi", "info"], text=True)
        (args.output.parent / "npu-before.txt").write_text(before)
        status = selected_chip_status(before, args.device)
        result["device_before"] = status
        result["device_users_before"] = device_users(args.device)
        if (status["health"] != "OK" or status["hbm_total_mib"] - status["hbm_used_mib"] < 40000
                or result["device_users_before"] or os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
                or os.environ.get("ASCEND_VISIBLE_DEVICES")
                or f"No running processes found in NPU {status['board']}" not in before):
            raise RuntimeError("Selected physical device is busy, remapped, unhealthy or lacks HBM")
        try:
            asyncio.run(exercise(args, root, result, sock))
        except BaseException:
            result["passed"] = False
            result["error"] = traceback.format_exc()
            raise
        finally:
            after = subprocess.check_output(["npu-smi", "info"], text=True)
            (args.output.parent / "npu-after.txt").write_text(after)
            result["device_after"] = selected_chip_status(after, args.device)
            result["device_users_after"] = device_users(args.device)
            result["unix_sockets_after"] = unix_sockets(root)
            write_json(args.output, result)
        if result["device_users_after"] or result["unix_sockets_after"] or result["device_after"]["health"] != "OK":
            raise RuntimeError("Device/socket cleanup or health check failed")
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
