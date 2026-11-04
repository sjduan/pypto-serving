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
    report.update(cli_arguments=cli, engine_config=asdict(config), requests=[], coordinator_trace=[], schedule_events=[])
    if args.native_enabled:
        diagnostic_root = args.output.parent / "diagnostics"
        diagnostic_root.mkdir()
        os.environ["PYPTO_NATIVE_DDR_DIAGNOSTICS_DIR"] = str(diagnostic_root)
    engine = AsyncLLMEngine(config, load_tokenizer(config.model_dir))
    core = engine._cores[0]
    # Observation only, identical in both arms: call the original scheduler once,
    # preserve its output/state, and copy metadata. Reclamation uses its public allocator below.
    original_schedule = core.scheduler.schedule
    def observe_schedule(*call_args, **kwargs):
        output = original_schedule(*call_args, **kwargs)
        if output.scheduled_requests:
            report["schedule_events"].append({"time_ns": time.monotonic_ns(), "requests": [{
                "request_id": sr.request.request_id, "pages": list(sr.block_ids),
                "computed_before": sr.num_computed_tokens, "tokens": sr.num_new_tokens,
                "is_prefill": sr.is_prefill,
            } for sr in output.scheduled_requests]})
        return output
    core.scheduler.schedule = observe_schedule
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
        def prefill_count(record):
            return sum(sr["tokens"] for e in report["schedule_events"] for sr in e["requests"]
                       if sr["request_id"] == record["request_id"] and sr["is_prefill"])
        assert prefill_count(cold) == len(cold["prompt_token_ids"])
        assert prefill_count(warm) == len(warm["prompt_token_ids"]) - 128
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
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.serving.native_ddr.types import PrefixKey

    manager, coord = core.kv_cache_manager, core._native_ddr
    held = set()
    async def wait_for(predicate, timeout=90):
        deadline = time.monotonic() + timeout
        while not predicate():
            if core._loop_task.done():
                await core._loop_task
                raise RuntimeError("engine stopped during R matrix")
            if time.monotonic() >= deadline:
                raise TimeoutError("R matrix condition did not settle")
            await asyncio.sleep(.001)
        collect_trace()

    async def idle():
        await wait_for(lambda: not core._batch_queue and (coord is None or not coord.jobs and not coord.controls))

    async def inspect(pages, ref, label):
        await idle()
        for attempt in range(100):
            command = coord._control("diagnostic", entry_ref=ref, page_ids=pages,
                                      diagnostic_label=f"{label}-{attempt}")
            await wait_for(lambda: command.command_id in coord.diagnostic_results)
            error, payload = coord.diagnostic_results.pop(command.command_id)
            if error is None:
                return json.loads(payload)
            if "already idle" not in error:
                raise RuntimeError(error)
            await asyncio.sleep(.01)
        raise TimeoutError("native owner did not become statically inspectable")

    def claim_free():
        blocks = manager.allocate_blocks(manager.num_free_blocks)
        assert blocks is not None
        held.update(b.block_id for b in blocks)
        return blocks

    def release(ids):
        ids = list(ids)
        assert set(ids) <= held
        manager.release_blocks_by_ids(ids)
        held.difference_update(ids)

    def scheduled(record):
        return [sr for event in report["schedule_events"] for sr in event["requests"]
                if sr["request_id"] == record["request_id"]]

    await idle()
    block_hash = manager.compute_block_hashes(cold["prompt_token_ids"])[0]
    block = manager.get_cached_block(block_hash)
    assert block is not None
    source = block.block_id
    manager.release_cached_blocks([block])
    key = PrefixKey(coord.namespace, block_hash) if coord else None
    ref = coord.ready[key] if coord else None
    before = await inspect((source,), ref, "original") if coord else None
    if before:
        original = Path(before["pages"][0]["file"]).read_bytes()
        assert original == Path(before["ddr"]["file"]).read_bytes() and len(original) == 20971520
    other_ids = engine.tokenizer.encode(("A river crosses a Japanese city, where museums and gardens welcome "
        "visitors throughout the year. The national capital has busy railway stations and public parks. ") * 20)[:127]
    other_prompt = engine.tokenizer.decode(other_ids)
    assert len(engine.tokenizer.encode(other_prompt)) == 127
    other_generation = GenerateConfig(max_new_tokens=2, min_tokens=2, temperature=0.0, top_p=1.0,
                                      top_k=None, seed=0, stream=True)
    try:
        assert manager.num_free_blocks == manager.num_blocks
        claim_free()
        assert source in held and manager.num_free_blocks == 0
        release((source,))
        report["allocator_pressure"] = {"actual_pages": manager.num_blocks, "reserved_pages": sorted(held),
                                         "free_for_other_model": source, "reported_capacity_unchanged": True}
        overwritten = await collect("real-other-owner-of-source", other_prompt, other_generation)
        steps = scheduled(overwritten)
        assert all(sr["pages"][0] == source for sr in steps)
        assert sum(sr["tokens"] for sr in steps if sr["is_prefill"]) == 127
        assert any(not sr["is_prefill"] and sr["computed_before"] == 127 and sr["tokens"] == 1 for sr in steps)
        after_other = await inspect((source,), None, "overwritten") if coord else None
        if after_other:
            assert Path(after_other["pages"][0]["file"]).read_bytes() != original
            assert all(a != b for a, b in zip(before["pages"][0]["layer_side_sha256"],
                                               after_other["pages"][0]["layer_side_sha256"]))
        # The genuine allocator reserves the overwritten source again. Release
        # three different real pages for one canonical target and suffix/unrelated work.
        claim = manager.allocate_blocks(1)
        assert claim is not None and [b.block_id for b in claim] == [source]
        held.add(source)
        release(sorted(held - {source})[:3])
        sentinels = sorted(held - {source})[-2:]
        sentinel_before = await inspect(tuple(sentinels), None, "sentinel-before") if coord else None
        trace_start = len(report["coordinator_trace"])
        a, b, unrelated = await asyncio.gather(collect("restore-a", LONG_PROMPT), collect("restore-b", LONG_PROMPT),
            collect("unrelated-during-restore", SHORT_PROMPT))
        await idle()
        expected = len(cold["prompt_token_ids"]) - (128 if coord else 0)
        assert all(sum(sr["tokens"] for sr in scheduled(record) if sr["is_prefill"]) == expected for record in (a, b))
        proof = {"source_page": source, "other_request": overwritten["request_id"],
                 "source_written_positions": 128, "source_layer_sides": 80,
                 "restore_requests": [a["request_id"], b["request_id"]], "prefill_tokens_each": expected,
                 "unrelated_request": unrelated["request_id"], "ordinary_allocator_only": True}
        if coord:
            events = report["coordinator_trace"][trace_start:]
            loads = [e for e in events if e["event"] == "transfer_dispatched" and e["operation"] == "LOAD"]
            assert len(loads) == 1
            waits = [e for e in events if e["event"] == "restore_held"]
            assert len(waits) == 2 and waits[0]["jobs"] == waits[1]["jobs"]
            target = loads[0]["pages"][0]
            assert target != source
            restored = await inspect((target,), ref, "restored")
            assert Path(restored["pages"][0]["file"]).read_bytes() == original
            sentinel_after = await inspect(tuple(sentinels), None, "sentinel-after")
            assert [p["sha256"] for p in sentinel_before["pages"]] == [p["sha256"] for p in sentinel_after["pages"]]
            proof.update(target_page=target, original_sha256=before["pages"][0]["sha256"],
                         overwritten_sha256=after_other["pages"][0]["sha256"],
                         restored_sha256=restored["pages"][0]["sha256"], all_bytes_equal=True,
                         untouched_sentinels_equal=True, canonical_load_count=1,
                         held_begin_ns=min(e["time_ns"] for e in waits),
                         valid_ns=next(e["time_ns"] for e in events if e["event"] == "restored_page_valid"))
        report["source_reuse_and_restore"] = proof
    finally:
        if held:
            release(tuple(held))


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
