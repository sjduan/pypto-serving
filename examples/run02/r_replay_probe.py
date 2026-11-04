# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Same real engine/owner across 97 native transfers and retired-envelope replay."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

from m2_baseline import LONG_PROMPT, write_json


async def ledger(engine, core, collect, collect_trace, args, report):
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.serving.native_ddr.types import Operation, PrefixKey
    from pypto_serving.serving.server.ipc import NativeDdrCommand, encode_command

    coord, manager = core._native_ddr, core.kv_cache_manager
    assert coord is not None and args.native_enabled
    held, delivered, controls = set(), [], []
    next_transfer, next_control = coord.next_transfer, coord.next_control
    def observe_transfer():
        item = next_transfer()
        if item is not None:
            delivered.append(item)
        return item
    def observe_control():
        item = next_control()
        if item is not None:
            controls.append(item)
        return item
    coord.next_transfer, coord.next_control = observe_transfer, observe_control
    evidence = report['ledger'] = {'cycles': [], 'owner_epoch': coord.worker_epoch,
                                  'scope': 'real allocator pressure reservations; no capacity/hash mutation'}

    async def wait_for(predicate):
        deadline = time.monotonic() + 90
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError('ledger condition did not settle')
            await asyncio.sleep(.001)
        collect_trace()

    async def idle():
        await wait_for(lambda: not coord.jobs and not coord.controls and not core._batch_queue)

    async def inspect(pages, ref, label):
        await idle()
        for retry in range(100):
            cmd = coord._control('diagnostic', entry_ref=ref, page_ids=pages,
                                 diagnostic_label=f'{label}-{retry}')
            await wait_for(lambda: cmd.command_id in coord.diagnostic_results)
            error, payload = coord.diagnostic_results.pop(cmd.command_id)
            if error is None:
                return json.loads(payload)
            if 'already idle' not in error:
                raise RuntimeError(error)
            await asyncio.sleep(.01)
        raise TimeoutError('owner not idle for static readback')

    def claim_free():
        blocks = manager.allocate_blocks(manager.num_free_blocks)
        assert blocks is not None
        held.update(b.block_id for b in blocks)

    def release(ids):
        manager.release_blocks_by_ids(list(ids))
        held.difference_update(ids)

    def scheduled(record):
        return [sr for e in report['schedule_events'] for sr in e['requests']
                if sr['request_id'] == record['request_id']]

    def other_prompt(label):
        ids = engine.tokenizer.encode((label + ' A river crosses a city with museums, parks and busy stations. ') * 30)[:127]
        text = engine.tokenizer.decode(ids)
        assert len(engine.tokenizer.encode(text)) == 127
        return text

    other_generation = GenerateConfig(max_new_tokens=2, min_tokens=2, temperature=0.0,
                                      top_p=1.0, top_k=None, seed=0, stream=True)
    log_path = args.output.parent.parent / 'logs/probe.log'

    async def replay(command, kind, identity, sequence):
        await idle()
        def native_runs(text):
            events = [json.loads(line.split('native_ddr_trace ', 1)[1]) for line in text.splitlines()
                      if 'native_ddr_trace {' in line]
            return [item['native_run_id'] for item in events if item['event'] == 'accepted']
        before = log_path.read_text()
        native_before = native_runs(before)
        with log_path.open() as observer:
            observer.seek(0, 2)
            core._input_queue.put(encode_command(command))
            deadline = time.monotonic() + 10
            while True:
                line = observer.readline()
                if 'native_ddr_receipt {' in line:
                    item = json.loads(line.split('native_ddr_receipt ', 1)[1])
                    if (item['kind'], item['identity'], item['delivery_sequence']) == (kind, identity, sequence):
                        break
                if time.monotonic() > deadline:
                    raise TimeoutError('stale envelope receipt missing')
                if not line:
                    await asyncio.sleep(.001)
        assert item['error'].startswith('retired delivery sequence')
        after = log_path.read_text()
        assert native_runs(after) == native_before
        return {'receipt': item, 'new_native_runs': 0, 'native_count_before': len(native_before)}

    try:
        await idle()
        for key in tuple(coord.ready):
            coord.evict(key)
        await idle()
        first_load = first_ref = first_original = old_evict = None
        for number in range(48):
            assert coord.worker_epoch == evidence['owner_epoch']
            prompt = f'Ledger cycle {number:03d}: ' + LONG_PROMPT
            cold = await collect(f'ledger-{number}-cold', prompt)
            await idle()
            key = PrefixKey(coord.namespace, manager.compute_block_hashes(cold['prompt_token_ids'])[0])
            ref = coord.ready[key]
            block = manager.get_cached_block(key.block_hash)
            assert block is not None
            source = block.block_id
            manager.release_cached_blocks([block])
            original = await inspect((source,), ref, f'cycle-{number}-original')
            raw = Path(original['pages'][0]['file']).read_bytes()
            assert len(raw) == 20971520 and raw == Path(original['ddr']['file']).read_bytes()
            assert manager.num_free_blocks == manager.num_blocks
            claim_free()
            assert source in held
            release((source,))
            other = await collect(f'ledger-{number}-overwrite', other_prompt(f'Overwrite {number}'), other_generation)
            steps = scheduled(other)
            assert all(sr['pages'][0] == source for sr in steps)
            assert sum(sr['tokens'] for sr in steps if sr['is_prefill']) == 127
            assert any(not sr['is_prefill'] and sr['computed_before'] == 127 for sr in steps)
            overwritten = await inspect((source,), None, f'cycle-{number}-overwritten')
            assert all(a != b for a, b in zip(original['pages'][0]['layer_side_sha256'],
                                             overwritten['pages'][0]['layer_side_sha256']))
            block = manager.allocate_blocks(1)
            assert block is not None and block[0].block_id == source
            held.add(source)
            release(sorted(held - {source})[:3])
            sentinel_ids = tuple(sorted(held - {source})[-2:])
            sentinel_before = await inspect(sentinel_ids, None, f'cycle-{number}-sentinel-before') if number in (0, 32, 47) else None
            restore = await collect(f'ledger-{number}-restore', prompt)
            await idle()
            load = delivered[-1]
            assert load.operation == Operation.LOAD and load.entry_refs == (ref,)
            target = load.page_ids[0]
            assert target != source
            suffix = sum(sr['tokens'] for sr in scheduled(restore) if sr['is_prefill'])
            assert suffix == len(cold['prompt_token_ids']) - 128
            restored = await inspect((target,), ref, f'cycle-{number}-restored')
            assert raw == Path(restored['pages'][0]['file']).read_bytes() == Path(restored['ddr']['file']).read_bytes()
            assert original['pages'][0]['layer_side_sha256'] == restored['pages'][0]['layer_side_sha256']
            if sentinel_before:
                after = await inspect(sentinel_ids, None, f'cycle-{number}-sentinel-after')
                assert [p['sha256'] for p in sentinel_before['pages']] == [p['sha256'] for p in after['pages']]
            row = {'cycle': number, 'source': source, 'target': target, 'suffix_tokens': suffix,
                   'load': asdict(load), 'original': original, 'overwritten': overwritten, 'restored': restored,
                   'all_bytes_equal': True, 'all_80_sides_overwritten': True,
                   'sentinels_unchanged': True if sentinel_before else 'sampled at cycles0/32/47'}
            # bytes fields in typed PrefixKey are recorded as hexadecimal.
            row['load'] = {**row['load'], 'entry_keys': [
                {'namespace_digest': k.namespace_digest.hex(), 'block_hash': k.block_hash.hex()} for k in load.entry_keys]}
            evidence['cycles'].append(row)
            if number == 0:
                first_load, first_ref, first_original = load, ref, original
            elif number < 47:
                coord.evict(key)
                await idle()
                if old_evict is None:
                    old_evict = (next(c for c in reversed(controls) if c.operation == 'evict'), original['ddr']['slot'])
            release(tuple(held))
            write_json(args.output, report)

        # Exercise continuing typed control traffic on the same actual owner.
        for _ in range(130):
            batch = [coord._control('cancel', job_id='already-finished') for _ in range(32)]
            await wait_for(lambda: all(cmd.command_id not in coord.controls for cmd in batch))
        evidence['additional_real_controls'] = 4160
        await idle()
        assert first_load is not None and first_ref in coord.ready.values()
        claim_free()
        target = first_load.page_ids[0]
        release((target,))
        other = await collect('reuse-old-load-target', other_prompt('Retired replay target'), other_generation)
        assert all(sr['pages'][0] == target for sr in scheduled(other))
        reclaimed = manager.allocate_blocks(1)
        assert reclaimed is not None and reclaimed[0].block_id == target
        held.add(target)
        before = await inspect((target,), first_ref, 'replay-target-before')
        assert before['transfer_replay']['retired_through'] >= first_load.delivery_sequence
        assert before['pages'][0]['sha256'] != first_original['pages'][0]['sha256']
        stale = await replay(NativeDdrCommand(operation='submit', request=first_load), 'transfer',
                             first_load.job_id, first_load.delivery_sequence)
        after = await inspect((target,), first_ref, 'replay-target-after')
        assert Path(before['pages'][0]['file']).read_bytes() == Path(after['pages'][0]['file']).read_bytes()
        evidence['retired_load_replay'] = {**stale, 'old_request': asdict(first_load),
            'target_reused_by_request': other['request_id'], 'target_bytes_unchanged': True,
            'original_exact_ddr_entry_still_ready': first_ref in coord.ready.values()}
        evidence['retired_load_replay']['old_request']['entry_keys'] = evidence['cycles'][0]['load']['entry_keys']
        old, old_slot = old_evict
        last = evidence['cycles'][-1]
        assert old_slot == last['restored']['ddr']['slot'] and old.entry_ref != ref
        assert after['control_replay']['retired_through'] >= old.delivery_sequence
        stale = await replay(NativeDdrCommand(**asdict(old)), 'control', old.command_id, old.delivery_sequence)
        final = await inspect((target,), ref, 'new-slot-after-old-evict')
        assert final['ddr']['sha256'] == last['original']['ddr']['sha256']
        evidence['retired_control_replay'] = {**stale, 'old_ref': asdict(old.entry_ref),
            'replacement_ref': asdict(ref), 'same_slot': old_slot, 'new_entry_still_ready': ref in coord.ready.values()}
        evidence['final_bookkeeping'] = {'transfer': final['transfer_replay'], 'control': final['control_replay']}
        assert final['transfer_replay']['high_water'] >= 97
        assert final['control_replay']['high_water'] > 4096
        assert final['transfer_replay']['records'] <= 80 and final['control_replay']['records'] <= 80
    finally:
        if held:
            release(tuple(held))
        coord.next_transfer, coord.next_control = next_transfer, next_control
    assert not coord.jobs and not coord.controls and manager.num_free_blocks == manager.num_blocks
    evidence['all_page_refs_released'] = True
