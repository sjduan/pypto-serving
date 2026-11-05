# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-only scenario. Imported only after explicit current resource admission."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

async def normal(engine, core, collect, collect_trace, args, report, cold, warm):
    import httpx
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.serving.native_ddr.types import PrefixKey
    from r_probe import matrix

    manager, coord = core.kv_cache_manager, core._native_ddr
    held, delivered = set(), []
    original_next = coord.next_transfer
    def observe_transfer():
        item = original_next()
        if item is not None:
            delivered.append(item)
        return item
    coord.next_transfer = observe_transfer

    async def wait_for(predicate, timeout=90):
        deadline = time.monotonic() + timeout
        while not predicate():
            if core._loop_task.done():
                await core._loop_task
                raise RuntimeError('engine stopped before evidence settled')
            if time.monotonic() > deadline:
                raise TimeoutError('Run04 condition did not settle')
            await asyncio.sleep(.001)
        collect_trace()

    async def idle():
        await wait_for(lambda: not coord.jobs and not coord.controls and not core._batch_queue)

    async def inspect(pages, ref, label):
        await idle()
        for retry in range(100):
            cmd = coord._control('diagnostic', entry_ref=ref, page_ids=pages,
                                 diagnostic_label=f'r04-{label}-{retry}')
            await wait_for(lambda: cmd.command_id in coord.diagnostic_results)
            error, payload = coord.diagnostic_results.pop(cmd.command_id)
            if error is None:
                return json.loads(payload)
            if 'already idle' not in error:
                raise RuntimeError(error)
            await asyncio.sleep(.01)
        raise TimeoutError('owner did not become statically inspectable')

    def claim_free():
        blocks = manager.allocate_blocks(manager.num_free_blocks)
        assert blocks is not None
        held.update(block.block_id for block in blocks)

    def release(ids):
        ids = list(ids)
        assert set(ids) <= held
        manager.release_blocks_by_ids(ids)
        held.difference_update(ids)

    def steps(request_id):
        return [s for event in report['schedule_events'] for s in event['requests'] if s['request_id'] == request_id]

    def token_prompt(label, size):
        ids = engine.tokenizer.encode((label + ' Rivers flow through cities with museums and gardens. ') * 100)[:size]
        text = engine.tokenizer.decode(ids)
        assert engine.tokenizer.encode(text) == ids and len(ids) == size
        return text

    generation = GenerateConfig(max_new_tokens=2, min_tokens=2, temperature=0.0,
                                top_p=1.0, top_k=None, seed=0, stream=True)
    report['rounds'] = []
    for number in range(2):
        await idle()
        for key in tuple(coord.ready):
            coord.evict(key)
        await idle()
        pool_before = await inspect((0,), None, f'{number}-before')
        prompt = token_prompt(f'Round {number} contiguous prefix report.', 257)
        seed = await collect(f'r04-{number}-double-miss', prompt, generation)
        await idle()
        hbm = await collect(f'r04-{number}-hbm-hit', prompt, generation)
        await idle()
        assert sum(s['tokens'] for s in steps(seed['request_id']) if s['is_prefill']) == 257
        assert sum(s['tokens'] for s in steps(hbm['request_id']) if s['is_prefill']) == 1
        keys = [PrefixKey(coord.namespace, value) for value in manager.compute_block_hashes(seed['prompt_token_ids'])[:2]]
        assert len(keys) == 2 and all(key in coord.ready for key in keys)
        sources, refs, captures = [], [], []
        for index, key in enumerate(keys):
            block = manager.get_cached_block(key.block_hash)
            assert block is not None
            sources.append(block.block_id)
            manager.release_cached_blocks([block])
            refs.append(coord.ready[key])
            captures.append(await inspect((sources[-1],), refs[-1], f'{number}-{index}-source'))
        assert len(set(sources)) == 2
        overwritten = []
        try:
            assert manager.num_free_blocks == manager.num_blocks
            claim_free()
            for index, source in enumerate(sources):
                release((source,))
                other = await collect(f'r04-{number}-{index}-real-model-overwrite',
                    token_prompt(f'Different calculation {number} {index}.', 127), generation)
                await idle()
                schedule = steps(other['request_id'])
                assert all(s['pages'] == [source] for s in schedule)
                assert sum(s['tokens'] for s in schedule if s['is_prefill']) == 127
                assert any(not s['is_prefill'] and s['computed_before'] == 127 and s['tokens'] == 1 for s in schedule)
                overwritten.append({'request_id': other['request_id'], 'schedule': schedule,
                                    'capture': await inspect((source,), None, f'{number}-{index}-overwrite')})
                allocated = manager.allocate_blocks(1)
                assert allocated is not None and [b.block_id for b in allocated] == [source]
                held.add(source)
            # Sources remain pinned away from LOAD. All destinations come from
            # the genuine original arena; capacity and cache hashes are untouched.
            release(sorted(held - set(sources))[:5])
            sentinels = sorted(held - set(sources))[-2:]
            assert len(sentinels) == 2
            sentinel_before = await inspect(tuple(sentinels), None, f'{number}-sentinels-before')
            collect_trace()
            trace_start = len(report['coordinator_trace'])
            if number == 0:
                restored_request = await collect(f'r04-{number}-pool-hit-direct', prompt, generation)
                restore_id = restored_request['request_id']
                interface = 'direct'
            else:
                start = len(report['schedule_events'])
                payload = {'model': engine.config.model_id if hasattr(engine, 'config') else core.config.model_id,
                           'prompt': prompt, 'max_tokens': 2, 'temperature': 0.0,
                           'top_p': 1.0, 'seed': 0, 'stream': False}
                async with httpx.AsyncClient(trust_env=False, timeout=180) as client:
                    response = await client.post('http://127.0.0.1:18892/v1/completions', json=payload)
                    response.raise_for_status()
                    body = response.json()
                assert body['choices'][0]['text'].strip() and body['usage']['prompt_tokens'] == 257
                assert body['usage']['completion_tokens'] == 2
                ids = {s['request_id'] for e in report['schedule_events'][start:] for s in e['requests']}
                assert len(ids) == 1
                restore_id = ids.pop()
                assert body['id'] == restore_id
                report['http_pool_hit'] = {'request_id': restore_id, 'payload': payload, 'body': body,
                                           'status': response.status_code}
                interface = 'http'
            await idle()
            assert sum(s['tokens'] for s in steps(restore_id) if s['is_prefill']) == 1
            assert any(not s['is_prefill'] for s in steps(restore_id))
            events = report['coordinator_trace'][trace_start:]
            loads = [e for e in events if e['event'] == 'transfer_dispatched' and e['operation'] == 'LOAD']
            assert len(loads) == 2
            blocks = []
            for index, (source, ref, original) in enumerate(zip(sources, refs, captures)):
                terminals = [e for e in events if e['event'] == 'transfer_terminal' and e['operation'] == 'LOAD'
                             and e['entry_refs'][0] == asdict(ref)]
                assert len(terminals) == 1
                target = terminals[0]['pages'][0]
                assert target not in sources
                restored = await inspect((target,), ref, f'{number}-{index}-restored')
                stores = [e for e in report['coordinator_trace'] if e['event'] == 'transfer_terminal'
                          and e['operation'] == 'STORE' and e['entry_refs'][0] == asdict(ref)]
                assert len(stores) == 1
                blocks.append({'index': index, 'token_start': index * 128, 'tokens': 128,
                    'source_page': source, 'target_page': target, 'entry_ref': asdict(ref),
                    'source': original['pages'][0], 'committed': original['ddr'],
                    'overwritten': overwritten[index]['capture']['pages'][0],
                    'overwrite_request': overwritten[index]['request_id'], 'restored': restored['pages'][0],
                    'store': stores[0], 'load': terminals[0],
                    'valid': next(e for e in events if e['event'] == 'restored_page_valid'
                                  and e['job'] == terminals[0]['job'])})
            sentinel_after = await inspect(tuple(sentinels), None, f'{number}-sentinels-after')
            assert [p['sha256'] for p in sentinel_before['pages']] == [p['sha256'] for p in sentinel_after['pages']]
            after = await inspect((blocks[0]['target_page'],), None, f'{number}-after')
            report['rounds'].append({'number': number, 'interface': interface, 'prompt_tokens': 257,
                'cold_request': seed['request_id'], 'hbm_request': hbm['request_id'],
                'restore_request': restore_id, 'blocks': blocks, 'pool_before': pool_before['pool'],
                'pool_after': after['pool'], 'sentinel_before': sentinel_before['pages'],
                'sentinel_after': sentinel_after['pages']})
            from qwen_evidence import check_round
            check_round(report['rounds'][-1], report, Path(args.output).parent, inspect_bytes=True)
        finally:
            if held:
                release(tuple(held))

    # Reuse accepted finite pressure/cancel/short-receipt controls, not the old
    # 48-cycle/4160-control matrix. It uses this same managed production engine.
    await collect('refresh-legacy-prefix', cold['prompt'])
    await matrix(engine, core, collect, collect_trace, args, report, cold, warm, stop_at_end=False)
    await idle()
    await replay_retired(core, delivered[0], args, idle, report)
    coord.next_transfer = original_next


async def replay_retired(core, request, args, idle, report):
    """An actual old envelope, rejected by the original worker before new effects."""
    from pypto_serving.serving.server.ipc import NativeDdrCommand, encode_command
    await idle()
    log = args.output.parent.parent / 'logs/probe.log'
    def accepted(text):
        return [json.loads(line.split('native_ddr_trace ', 1)[1])['native_run_id']
                for line in text.splitlines() if 'native_ddr_trace {' in line
                and json.loads(line.split('native_ddr_trace ', 1)[1])['event'] == 'accepted']
    before = accepted(log.read_text())
    with log.open() as observer:
        observer.seek(0, 2)
        core._input_queue.put(encode_command(NativeDdrCommand(operation='submit', request=request)))
        deadline = time.monotonic() + 10
        while True:
            line = observer.readline()
            if 'native_ddr_receipt {' in line:
                item = json.loads(line.split('native_ddr_receipt ', 1)[1])
                if item['identity'] == request.job_id and item['delivery_sequence'] == request.delivery_sequence:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError('retired original envelope rejection not observed')
            if not line:
                await asyncio.sleep(.001)
    assert item['error'].startswith('retired delivery sequence')
    assert accepted(log.read_text()) == before
    report['stale_original_transfer'] = {'receipt': item, 'native_runs_before_after': [len(before), len(before)],
                                          'scope': 'real old envelope; no synthetic native completion'}
