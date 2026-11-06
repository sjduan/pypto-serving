# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Transport-independent real scenario sequence; fake transports label CPU orchestration only."""
from __future__ import annotations

import statistics
from scenario_contract import MEASURED, WARMUP, PAYLOAD_BYTES, plan


def check_request(row, *, prefill, loads, reset=True):
    if (type(row) is not dict or row['prefill_tokens'] != prefill or len(row['loads']) != loads
            or not row['schedule'] or row['first_token_ns'] < row['begin_ns']
            or row['completed_ns'] < row['first_token_ns']):
        raise ValueError('request/prefill/native/timing evidence differs')
    if reset:
        proof = row['reset_proof']
        if not proof or proof['original_allocator'] is not True or proof['all_claimed_hashes_invalidated'] is not True:
            raise ValueError('sample lacks a true original-allocator cold reset')
        if proof['time_ns'] >= row['begin_ns'] or proof['actor_pid'] != row['actor_pid']:
            raise ValueError('sample reset does not precede its exact actor request')
    if loads and row['stores']:
        raise ValueError('shared consumer STOREd the supposedly cold imported prefix')
    jobs = set()
    for event in row['loads']:
        receipt = event['pool_receipt']
        if (event['status'] != 'SUCCEEDED' or event['native_completed'] is not True
                or event['stopped'] is not True or event['bytes'] != PAYLOAD_BYTES
                or not event['page_lease_id'] or event['delivery_sequence'] <= 0
                or event['job'] in jobs or receipt['operation'] != 'LOAD'
                or receipt['content']['nbytes'] != PAYLOAD_BYTES):
            raise ValueError('LOAD lacks exact stopped typed native/UF receipt')
        jobs.add(event['job'])
        if not any(x['event'] == 'restored_page_valid' and x['job'] == event['job']
                   for x in row['coordinator_trace']):
            raise ValueError('LOAD destination was never legally published')
    return row


def samples(cold, shared):
    if len(cold) != MEASURED or len(shared) != MEASURED:
        raise ValueError('paired measurements require twenty samples each')
    def ttft(r):return (r['first_token_ns'] - r['begin_ns']) / 1e6
    for a, b in zip(cold, shared):
        if (a['prompt_token_ids'] != b['prompt_token_ids'] or a['generate_config'] != b['generate_config']
                or a['interface'] != 'direct' or b['interface'] != 'direct'):
            raise ValueError('paired workload/sampling/interface differs')
    def summary(values):
        values=sorted(values)
        return {'median':statistics.median(values),'p90':values[(9*len(values)+9)//10-1]}
    return {'cold_TTFT_ms':summary([ttft(r) for r in cold]),
        'shared_TTFT_ms':summary([ttft(r) for r in shared]),
        'paired_shared_minus_cold_ms':[ttft(b)-ttft(a) for a,b in zip(cold,shared)],
        'paired_difference_ms':summary([ttft(b)-ttft(a) for a,b in zip(cold,shared)])}


async def orchestrate(config, transport):
    prepared=plan(config)  # All metadata preflight precedes any transport entry.
    report={'scope':'real-runtime-captured-pending-review' if transport.real_runtime else 'CPU-ORCHESTRATION-ONLY',
            'model_acceptance':False,'plan':prepared,'baseline':{},'shared':{},'functional':[],
            'M1_line_retired':False,'pool_destroyed':False}
    prompt=None
    try:
        await transport.admit(config,prepared)
        # Baseline has no SDK client, and is truly stopped before another model
        # is loaded on that same physical device. Never co-load two14B weights.
        for index in range(config['replicas']):
            name=await transport.start_model(index,shared=False)
            candidate=await transport.call(name,'prompt')
            if prompt is None:prompt=candidate
            if candidate!=prompt:raise ValueError('fixed tokenizer workload differs between processes')
            rows=[]
            for number in range(WARMUP+MEASURED):
                await transport.call(name,'reset')
                row=await transport.measured_request(name,prompt['text'])
                check_request(row,prefill=257,loads=0)
                row['sample_phase']='warmup' if number<WARMUP else 'measured'
                rows.append(row)
            report['baseline'][str(index)]=rows
            await transport.stop_model(name)
        await transport.start_directory()
        producer=await transport.start_model(0,shared=True)
        peers=[await transport.start_model(i,shared=True) for i in range(1,config['replicas'])]
        seed=await transport.call(producer,'request',text=prompt['text'])
        if len(seed['stores'])!=2 or seed['prefill_tokens']!=257:
            raise ValueError('producer did not really compute and STORE two contiguous prefix blocks')
        original=await transport.call(producer,'capture',text=prompt['text'],label='source',guards=True)
        original['prompt_ids']=prompt['ids']
        sources=[x['page'] for x in original['blocks']]
        if len(set(sources))!=2:raise ValueError('two independent original source pages required')
        for peer in peers:
            await transport.verify_namespace(producer,peer)
            for interface in ('direct','direct','http','http'):
                await transport.call(peer,'reset',avoid=sources,probe_sentinels=True)
                row=await transport.call(peer,'request',text=prompt['text'],interface=interface)
                check_request(row,prefill=1,loads=2)
                target=await transport.call(peer,'capture',text=prompt['text'],label='before-exit',guards=False)
                report['functional'].append(await transport.check_bytes(original,None,target,row))
        if config['replicas']==4 and config['router'] is not None:
            report['router']=await transport.router_check([producer,*peers],prompt['text'])
        await transport.pin_original(original)
        overwritten=await transport.call(producer,'overwrite',original=original)
        await transport.check_overwrites(original,overwritten)
        report['producer_stop']=await transport.stop_model(producer)
        await transport.release_original()
        # Only now can subsequent restores establish producer-exit independence.
        for peer in peers:
            for interface in ('direct','direct','http','http'):
                await transport.call(peer,'reset',avoid=sources,probe_sentinels=True)
                row=await transport.call(peer,'request',text=prompt['text'],interface=interface)
                check_request(row,prefill=1,loads=2)
                target=await transport.call(peer,'capture',text=prompt['text'],label='after-exit',guards=False)
                report['functional'].append(await transport.check_bytes(original,overwritten,target,row))
            rows=[]
            for number in range(WARMUP+MEASURED):
                await transport.call(peer,'reset',avoid=sources)
                row=await transport.measured_request(peer,prompt['text'])
                check_request(row,prefill=1,loads=2)
                row['sample_phase']='warmup' if number<WARMUP else 'measured'
                rows.append(row)
            # Original local HBM priority: no reset, and no native LOAD at all.
            hbm=await transport.call(peer,'request',text=prompt['text'])
            check_request(hbm,prefill=1,loads=0,reset=False)
            final=await transport.call(peer,'capture',text=prompt['text'],label='final',guards=True)
            await transport.check_guards(original,final)
            index=transport.model_index(peer)
            report['shared'][str(index)]={'samples':rows,'HBM_priority':hbm,
                'metrics':samples(report['baseline'][str(index)][WARMUP:],rows[WARMUP:])}
        # Run unrelated double misses last: bounded replacement may evict old
        # canonical slots and must not contaminate later shared-hit measurements.
        for peer in peers:
            await transport.call(peer,'reset')
            different=await transport.call(peer,'prompt',label='A separate namespace workload '+peer)
            row=await transport.call(peer,'request',text=different['text'])
            check_request(row,prefill=257,loads=0)
            report['shared'][str(transport.model_index(peer))]['double_miss']=row
            await transport.stop_model(peer)
        await transport.stop_directory()
        report['cleanup']=await transport.verify_cleanup()
        report['status']='CAPTURED-PENDING-REVIEW' if transport.real_runtime else 'CPU-ORCHESTRATION-ONLY'
        if transport.real_runtime:await transport.publish_closed(report)
        return report
    except BaseException as error:
        # The transport owns real process/native obligations, and retains them.
        # Do not treat a Python exception or a timeout as a drain/SDK ACK.
        await transport.retain(error,report)
        raise
