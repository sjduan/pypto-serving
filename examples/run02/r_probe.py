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

    async def collect(name, prompt, generation=generate, *, allow_abort=False):
        request_id = engine.generate_request_id()
        record = {"name": name, "request_id": request_id, "prompt": prompt,
                  "prompt_token_ids": engine.tokenizer.encode(prompt), "generate_config": asdict(generation),
                  "outputs": [], "begin_ns": time.monotonic_ns()}
        report["requests"].append(record)
        async for output in engine.add_request(request_id, prompt, generation):
            record["outputs"].append({"time_ns": time.monotonic_ns(), **asdict(output)})
        record["completed_ns"] = time.monotonic_ns()
        if allow_abort and record["outputs"][-1]["finish_reason"] == "FINISHED_ABORTED":
            final = record["outputs"][-1]
            emitted = [o["token_id"] for o in record["outputs"] if o["token_id"] is not None]
            assert list(final["token_ids"]) == emitted and final["completion_tokens"] == len(emitted)
            assert final["prompt_tokens"] == len(record["prompt_token_ids"])
            assert "".join(o["text_delta"] for o in record["outputs"]) == final["text"]
            assert engine.tokenizer.decode(emitted) == final["text"]
            collect_trace()
            return record
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
        elif args.case == "ledger":
            from r_replay_probe import ledger
            await ledger(engine, core, collect, collect_trace, args, report)
        elif args.case in ("stop-load", "stop-store"):
            await native_pending_stop(engine, core, collect, args, report)
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


async def native_pending_stop(engine, core, collect, args, report):
    """Passively observe the real backend accept trace; never hold a host completion event."""
    import re
    coord = core._native_ddr
    assert coord is not None
    deadline = time.monotonic() + 30
    while coord.jobs or coord.controls or core._batch_queue:
        if time.monotonic() > deadline:
            raise TimeoutError("stop probe did not reach its initial idle state")
        await asyncio.sleep(.001)
    held = set()
    log_path = args.output.parent.parent / "logs/probe.log"
    with log_path.open() as observer:
        observer.seek(0, 2)
        try:
            if args.case == "stop-load":
                blocks = core.kv_cache_manager.allocate_blocks(core.kv_cache_manager.num_free_blocks)
                held = {b.block_id for b in blocks}
                free = sorted(held)[:6]
                core.kv_cache_manager.release_blocks_by_ids(free)
                held.difference_update(free)
                prompt = LONG_PROMPT
            else:
                prompt = "A native accepted stop observation: " + LONG_PROMPT
            task = asyncio.create_task(collect("native-pending-stop-client", prompt, allow_abort=True))
            accepted = None
            deadline = time.monotonic() + 30
            while accepted is None:
                line = observer.readline()
                if line:
                    if "native_ddr_trace {" in line:
                        item = json.loads(line.split("native_ddr_trace ", 1)[1])
                        if (item["event"] == "accepted" and item["native_pending"]
                                and coord.active_job == item["job"]):
                            accepted = item
                else:
                    if time.monotonic() > deadline or task.done():
                        raise TimeoutError("real native acceptance window was not observed")
                    # Yield for actual model/IPC progress; no delay or override is
                    # introduced into the producer, native handle or aux consumer.
                    await asyncio.sleep(0)
            invoke_ns = time.monotonic_ns()
            job = coord.jobs[accepted["job"]]
            page = job.request.page_ids[0]
            report["native_pending_stop"] = {"operation": job.request.operation.value,
                "job": accepted["job"], "native_run_id": accepted["native_run_id"],
                "accepted_ns": accepted["time_ns"], "invoke_ns": invoke_ns,
                "retained_ref_cnt": core.kv_cache_manager.blocks[page].ref_cnt,
                "observation": "passive backend native_pending=true trace; no event withholding"}
            assert await engine.stop()
            await task
            lines = log_path.read_text().splitlines()
            complete = [s for s in lines if "name=node.complete " in s and
                        re.search(r"\brun_id=" + str(accepted["native_run_id"]) + r"\b", s)]
            assert len(complete) == 1
            complete_ns = int(re.search(r"\bts=(\d+)", complete[0])[1])
            report["native_pending_stop"].update(native_complete_ns=complete_ns, owner_stop_proven=coord.stopped,
                raw_native_pending_at_stop=accepted["time_ns"] <= invoke_ns < complete_ns)
            assert accepted["time_ns"] <= invoke_ns < complete_ns, "native observation window was already completed"
            assert coord.stopped and not coord.jobs and not coord.steps
        finally:
            if held:
                core.kv_cache_manager.release_blocks_by_ids(list(held))
    assert all(b.ref_cnt == 0 for b in core.kv_cache_manager.blocks)
    report["native_pending_stop"]["all_page_refs_released_after_proof"] = True


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

    # Preserve exactly the first local prefix reference while the same allocator
    # reclaims its successors. No cache hash or allocator internals are edited.
    prefix_ids = engine.tokenizer.encode((LONG_PROMPT + " ") * 4)[:385]
    prefix_prompt = engine.tokenizer.decode(prefix_ids)
    assert len(engine.tokenizer.encode(prefix_prompt)) == 385
    seed = await collect("partial-seed", prefix_prompt)
    await idle()
    hashes = manager.compute_block_hashes(prefix_ids)[:3]
    keys = [PrefixKey(coord.namespace, h) for h in hashes] if coord else []
    if coord:
        assert all(k in coord.ready for k in keys)
    local = manager.get_computed_blocks(prefix_ids)
    assert len(local) == 3
    first_pin = local[0]
    manager.release_cached_blocks(local[1:])
    snapshots = []
    try:
        if coord:
            for i in (1, 2):
                snapshots.append(await inspect((local[i].block_id,), coord.ready[keys[i]], f"partial-source-{i}"))
        claim_free()
        release(sorted(held)[:6])
        start = len(report["coordinator_trace"])
        partial = await collect("partial-hbm-ddr", prefix_prompt)
        await idle()
        partial_tokens = sum(s["tokens"] for s in scheduled(partial) if s["is_prefill"])
        assert partial_tokens == (1 if coord else 257)
        details = {"local_prefix_page": first_pin.block_id, "prefill_tokens": partial_tokens}
        if coord:
            events = report["coordinator_trace"][start:]
            loads = [e for e in events if e["event"] == "transfer_dispatched" and e["operation"] == "LOAD"]
            assert len(loads) == 2
            restored_hashes = []
            for i in (1, 2):
                cached = manager.get_cached_block(hashes[i])
                assert cached is not None
                page_id = cached.block_id
                manager.release_cached_blocks([cached])
                restored = await inspect((page_id,), None, f"partial-restored-{i}")
                assert Path(restored["pages"][0]["file"]).read_bytes() == Path(snapshots[i - 1]["ddr"]["file"]).read_bytes()
                restored_hashes.append(restored["pages"][0]["sha256"])
            details.update(canonical_loads=2, full_block_bytes_equal=True, restored_sha256=restored_hashes)
        report["partial_contiguous_successors"] = details
    finally:
        if held:
            release(tuple(held))
        manager.release_cached_blocks([first_pin])

    await idle()
    if coord:
        assert coord.evict(keys[1])
        await idle()
        assert keys[0] in coord.ready and keys[1] not in coord.ready and keys[2] in coord.ready
    try:
        claim_free()
        release(sorted(held)[:6])
        start = len(report["coordinator_trace"])
        gap = await collect("gap-stops-at-missing-middle", prefix_prompt)
        await idle()
        count = sum(s["tokens"] for s in scheduled(gap) if s["is_prefill"])
        assert count == (257 if coord else 385)
        details = {"prefill_tokens": count, "later_block_was_not_skipped": True}
        if coord:
            events = report["coordinator_trace"][start:]
            assert len([e for e in events if e["event"] == "transfer_dispatched" and e["operation"] == "LOAD"]) == 1
            details["canonical_loads"] = 1
        report["gap_boundary"] = details
    finally:
        if held:
            release(tuple(held))

    # Exercise real finite slots through normal requests; new immutable prefixes
    # cause release-acknowledged FIFO retirement before their STORE can be admitted.
    filled = []
    for number in range(15):
        filled.append(await collect(f"pool-fill-{number}", f"Geography report {number}: " + LONG_PROMPT))
        await idle()
        if coord:
            assert len(coord.ready) <= coord.max_entries and not coord.retiring
    report["capacity"] = {"actual_hbm_pages": manager.num_blocks, "new_prefixes": len(filled)}
    if coord:
        assert len(coord.ready) == coord.max_entries
        releases = [e for e in report["coordinator_trace"] if e["event"] == "worker_slot_released"]
        assert len(releases) >= 2
        report["capacity"].update(ready_entries=len(coord.ready), slots=coord.max_entries,
                                  real_release_receipts=len(releases))

    # Ordinary client cancellation after actual output is paired in both arms.
    from dataclasses import replace
    active = asyncio.create_task(collect("abort-after-output", SHORT_PROMPT,
        replace(other_generation, max_new_tokens=32), allow_abort=True))
    await wait_for(lambda: any(r["name"] == "abort-after-output" and any(
        o["token_id"] is not None for o in r["outputs"]) for r in report["requests"]))
    record = next(r for r in report["requests"] if r["name"] == "abort-after-output")
    await engine.abort_request(record["request_id"])
    aborted = await active
    assert aborted["outputs"][-1]["finish_reason"] == "FINISHED_ABORTED"
    await collect("after-client-abort", SHORT_PROMPT)
    await idle()

    if coord is None:
        report["disabled_resources_absent"] = core._native_ddr_output_queue is None
        return

    # These are real engine waiters and real accepted native LOAD obligations.
    # Cancellation is logical; release is checked only after the actual stopped event.
    cancel_prompt = filled[-1]["prompt"]
    cancel_hash = manager.compute_block_hashes(engine.tokenizer.encode(cancel_prompt))[0]
    cancel_key = PrefixKey(coord.namespace, cancel_hash)
    report["load_cancellation"] = []
    for all_waiters in (False, True):
        await idle()
        assert cancel_key in coord.ready
        try:
            claim_free()
            release(sorted(held)[:6])
            label = "all" if all_waiters else "one"
            a_task = asyncio.create_task(collect(f"cancel-{label}-a", cancel_prompt, allow_abort=True))
            b_task = asyncio.create_task(collect(f"cancel-{label}-b", cancel_prompt, allow_abort=True))
            await wait_for(lambda: len(coord.waiters) == 2 and coord.active_job is not None
                           and coord.jobs[coord.active_job].request.operation.value == "LOAD")
            job_id = coord.active_job
            target = coord.jobs[job_id].request.page_ids[0]
            assert manager.blocks[target].ref_cnt >= 1 and manager.blocks[target].block_hash is None
            names = [f"cancel-{label}-a", f"cancel-{label}-b"]
            ids = [next(r["request_id"] for r in report["requests"] if r["name"] == n) for n in names]
            sent_ns = time.monotonic_ns()
            await engine.abort_request(ids[0])
            if all_waiters:
                await engine.abort_request(ids[1])
            assert manager.blocks[target].ref_cnt >= 1
            a_result, b_result = await asyncio.gather(a_task, b_task)
            assert a_result["outputs"][-1]["finish_reason"] == "FINISHED_ABORTED"
            if all_waiters:
                assert b_result["outputs"][-1]["finish_reason"] == "FINISHED_ABORTED"
            await idle()
            terminal = next(e for e in report["coordinator_trace"]
                            if e["event"] == "transfer_terminal" and e["job"] == job_id)
            assert terminal["stopped"] and terminal["native_completed"]
            if all_waiters:
                assert manager.get_cached_block(cancel_hash) is None and cancel_key not in coord.ready
                assert not any(e["event"] == "restored_page_valid" and e["job"] == job_id
                               for e in report["coordinator_trace"])
            else:
                assert cancel_key in coord.ready
            report["load_cancellation"].append({"all_waiters": all_waiters, "job": job_id,
                "target": target, "cancel_sent_ns": sent_ns, "terminal": terminal,
                "pending_target_retained": True, "early_valid": False})
        finally:
            if held:
                release(tuple(held))
    await collect("after-all-waiters-cancel", cancel_prompt)
    await idle()

    store_prompt = "A separate cancelled cache report: " + LONG_PROMPT
    store_task = asyncio.create_task(collect("cancel-pending-store", store_prompt))
    await wait_for(lambda: coord.active_job is not None
                   and coord.jobs[coord.active_job].request.operation.value == "STORE")
    job_id = coord.active_job
    job = coord.jobs[job_id]
    page = job.request.page_ids[0]
    assert manager.blocks[page].ref_cnt >= 1
    job.discard = True
    coord._control("cancel", job_id=job_id)
    await store_task
    await idle()
    assert job.request.entry_keys[0] not in coord.ready and manager.blocks[page].ref_cnt == 0
    terminal = next(e for e in report["coordinator_trace"] if e["event"] == "transfer_terminal" and e["job"] == job_id)
    assert terminal["stopped"] and terminal["native_completed"]
    report["store_cancellation"] = {"job": job_id, "source_page": page, "terminal": terminal,
                                    "source_pin_released_after_stop": True, "ddr_ready_not_published": True}

    async def arm_fault(kind):
        await idle()
        cmd = coord._control("fault", fault=kind)
        await wait_for(lambda: cmd.command_id in coord.diagnostic_results)
        error, payload = coord.diagnostic_results.pop(cmd.command_id)
        if error:
            raise RuntimeError(error)
        assert json.loads(payload)["armed"] == kind

    # Real headroom exhaustion, then release genuine allocator reservations.
    # The request safely cold-prefills instead of reserving a target with no suffix room.
    try:
        await idle()
        claim_free()
        release(sorted(held)[:1])
        headroom_sequence = coord._trace_sequence
        headroom_task = asyncio.create_task(collect("headroom-safe-cold-fallback", cancel_prompt))
        await wait_for(lambda: any(e["event"] == "restore_reservation_no_headroom" and e["free_pages"] == 1
                                  and e["sequence"] > headroom_sequence
                                  for e in coord.trace))
        release(sorted(held)[:3])
        headroom_result = await headroom_task
        await idle()
        assert sum(s["tokens"] for s in scheduled(headroom_result) if s["is_prefill"]) == len(headroom_result["prompt_token_ids"])
        report["headroom_fallback"] = {"actual_free_pages": 1, "fresh_target_plus_suffix_required": 2,
                                       "cold_prefill_completed": True, "capacity_unchanged": True}
    finally:
        if held:
            release(tuple(held))

    await arm_fault("short_load_receipt")
    try:
        claim_free()
        release(sorted(held)[:6])
        start = len(report["coordinator_trace"])
        failed_load = await collect("short-receipt-load-fallback", cancel_prompt)
        await idle()
        events = report["coordinator_trace"][start:]
        failure = next(e for e in events if e["event"] == "transfer_terminal" and e["status"] == "FAILED")
        assert failure["native_completed"] and failure["stopped"] and failure["bytes"] == 0
        assert not any(e["event"] == "restored_page_valid" and e["job"] == failure["job"] for e in events)
        assert sum(s["tokens"] for s in scheduled(failed_load) if s["is_prefill"]) == len(failed_load["prompt_token_ids"])
        report["postsubmit_load_failure"] = {"layer": "short host receipt after actual native result/cleanup",
            "terminal": failure, "invalid_target_not_consumed": True, "real_cold_fallback_completed": True}
    finally:
        if held:
            release(tuple(held))
    await arm_fault("short_store_receipt")
    start = len(report["coordinator_trace"])
    failed_store = await collect("short-receipt-store", "A failed cache receipt report: " + LONG_PROMPT)
    await idle()
    events = report["coordinator_trace"][start:]
    failure = next(e for e in events if e["event"] == "transfer_terminal" and e["status"] == "FAILED")
    assert failure["native_completed"] and failure["stopped"] and failure["bytes"] == 0
    assert not any(e["event"] == "ddr_ready" and e["job"] == failure["job"] for e in events)
    report["postsubmit_store_failure"] = {"layer": "short host receipt after actual native result/cleanup",
                                         "terminal": failure, "request_completed": failed_store["request_id"]}

    # Both native operation types get a real pending shutdown across the paired
    # main/pressure configurations. Idle shutdown is covered by smoke/disabled.
    stop_operation = "LOAD" if args.utilization < .7 else "STORE"
    try:
        if stop_operation == "LOAD":
            claim_free()
            release(sorted(held)[:6])
            stop_prompt = cancel_prompt
        else:
            stop_prompt = "A report for pending owner shutdown: " + LONG_PROMPT
        pending_client = asyncio.create_task(collect("pending-stop-client", stop_prompt, allow_abort=True))
        await wait_for(lambda: coord.active_job is not None
                       and coord.jobs[coord.active_job].request.operation.value == stop_operation)
        job_id = coord.active_job
        page = coord.jobs[job_id].request.page_ids[0]
        report["pending_stop"] = {"operation": stop_operation, "job": job_id, "page": page,
                                   "invoke_ns": time.monotonic_ns(), "retained_ref_cnt": manager.blocks[page].ref_cnt}
        assert manager.blocks[page].ref_cnt >= 1
        assert await engine.stop()
        await pending_client
        assert coord.stopped and not coord.jobs and not coord.steps
        report["pending_stop"]["owner_stop_proven"] = True
    finally:
        if held:
            release(tuple(held))
    assert all(b.ref_cnt == 0 for b in manager.blocks)
    report["pending_stop"]["all_page_refs_released_after_proof"] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["smoke", "matrix", "ledger", "stop-load", "stop-store"], required=True)
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
