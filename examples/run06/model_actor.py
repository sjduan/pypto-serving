# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Qwen TP1 actor. Imported runtime only in explicit future device execution."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

from scenario_contract import cli, save, validate


class ModelActor:
    def __init__(self, config, index, shared, root):
        self.config, self.index, self.shared, self.root = config, index, shared, Path(root)
        self.engine = self.server = self.server_task = None
        self.held, self.schedule, self.records = set(), [], []
        self.counter = 0
        self.hbm_guard_before = []

    async def start(self):
        validate(self.config)
        from scenario_admission import validate_release
        validate_release(self.config)
        if (os.environ.get('LD_PRELOAD') or any(k.startswith('UFLOW_POOL_TEST_') for k in os.environ)
                or any(os.environ.get(k) for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES'))):
            raise RuntimeError('native actor inherited host fence')
        admission=json.loads((Path(self.config['root'])/'parent-admission.json').read_text())
        from scenario_contract import plan
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        if (admission['parent_pid']!=os.getppid() or process_birth(os.getppid())!=admission['parent_birth']
                or admission['config_sha256']!=plan(self.config)['config_sha256']):
            raise PermissionError('original admitted controller birth/configuration differs')
        import uvicorn
        from pypto_serving.cli.main import build_parser, build_serving_engine_config
        from pypto_serving.config.types import GenerateConfig
        from pypto_serving.model.tokenizer import load_tokenizer
        from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
        from pypto_serving.serving.server.server import create_serving_app
        import socket
        listener=socket.socket(fileno=int(sys.argv[5]))
        if listener.getsockname()!=('127.0.0.1',self.config['devices'][self.index]['port']):
            raise PermissionError('original reserved listener FD/port differs')
        self.root.mkdir()
        os.environ['PYPTO_NATIVE_DDR_DIAGNOSTICS_DIR'] = str(self.root / 'diagnostics')
        (self.root / 'diagnostics').mkdir()
        os.environ['PYPTO_NATIVE_DDR_POOL_EVIDENCE'] = '1'
        for key, name in (('PYPTO_PROG_BUILD_DIR', 'build'), ('PYPTO_CACHE_DIR', 'cache'),
                          ('ASCEND_PROCESS_LOG_PATH', 'ascend'), ('ASCEND_WORK_PATH', 'ascend')):
            path = self.root / name;path.mkdir(exist_ok=True);os.environ[key] = str(path)
        self.device = self.config['devices'][self.index]
        arguments = cli(self.config, self.device, shared=self.shared)
        engine_config = build_serving_engine_config(build_parser().parse_args(arguments))
        self.engine_config = engine_config
        self.engine = AsyncLLMEngine(engine_config, load_tokenizer(engine_config.model_dir))
        self.core = self.engine._cores[0]
        original = self.core.scheduler.schedule
        def observe(*args, **kwargs):
            value = original(*args, **kwargs)
            for sr in value.scheduled_requests:
                self.schedule.append({'request_id': sr.request.request_id, 'pages': list(sr.block_ids),
                    'computed_before': sr.num_computed_tokens, 'tokens': sr.num_new_tokens,
                    'is_prefill': sr.is_prefill, 'time_ns': time.monotonic_ns()})
            return value
        self.core.scheduler.schedule = observe
        self.generation = GenerateConfig(max_new_tokens=2, min_tokens=2, temperature=0.0,
                                         top_p=1.0, top_k=None, seed=0, stream=True)
        await self.engine.start()
        self.coord = self.core._native_ddr
        self.manager = self.core.kv_cache_manager
        self.server = uvicorn.Server(uvicorn.Config(create_serving_app(self.engine, engine_config.model_id,
            self.generation), host='127.0.0.1', port=self.device['port'], lifespan='off', log_level='info'))
        self.server_task = asyncio.create_task(self.server.serve(sockets=[listener]))
        await self.wait(lambda: self.server.started)
        startup = asdict(self.core._native_ddr_startup) if self.shared else None
        if startup:
            startup['namespace_digest'] = self.core._native_ddr_startup.namespace_digest.hex()
            startup['compatibility_json'] = json.loads(self.core._native_ddr_startup.compatibility_json)
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        from pypto_serving.serving.native_ddr.managed_process import descendants
        return {'descendant_births': descendants(), 'pid': os.getpid(), 'birth': process_birth(os.getpid()), 'device': self.device,
            'worker_pid': self.core._worker_process.pid, 'startup': startup, 'shared': self.shared,
            'argv': arguments, 'num_pages': self.manager.num_blocks}

    async def wait(self, predicate, timeout=900):
        deadline = time.monotonic() + timeout
        while not predicate():
            if self.core._loop_task is not None and self.core._loop_task.done():
                await self.core._loop_task
                raise RuntimeError('model loop stopped before evidence settled')
            if time.monotonic() > deadline:
                raise TimeoutError('retain original native actor; no second command or forced cleanup')
            await asyncio.sleep(.005)

    async def idle(self):
        await self.wait(lambda: not self.core._batch_queue and
            (self.coord is None or not (self.coord.jobs or self.coord.controls or self.coord.steps)))

    def prompt(self, label='Shared contiguous prefix data report.', size=257):
        ids = self.engine.tokenizer.encode((label + ' Rivers flow through cities with museums and gardens. ') * 100)[:size]
        text = self.engine.tokenizer.decode(ids)
        if self.engine.tokenizer.encode(text) != ids or len(ids) != size:
            raise ValueError('fixed workload does not roundtrip to exact token count')
        return {'text': text, 'ids': ids}

    async def reset(self, avoid=(), probe_sentinels=False):
        await self.idle()
        if self.held:
            self.manager.release_blocks_by_ids(sorted(self.held));self.held.clear()
        if self.manager.num_free_blocks != self.manager.num_blocks:
            raise RuntimeError('active HBM allocations prevent a true cold reset')
        allocated = self.manager.allocate_blocks(self.manager.num_free_blocks)
        if allocated is None or len(allocated) != self.manager.num_blocks:
            raise RuntimeError('original allocator did not claim every free page')
        ids = [b.block_id for b in allocated]
        if any(b.block_hash is not None for b in allocated):
            raise RuntimeError('original allocator retained a cached prefix under cold reset')
        if any(type(n) is not int or n < 0 for n in avoid):raise ValueError('foreign source page-number schema differs')
        self.held = set(avoid) & set(ids)  # Other arenas may have a different P; unrepresentable IDs cannot collide.
        available=[n for n in ids if n not in self.held]
        self.sentinel_pages = available[3:5] if probe_sentinels else []
        self.held.update(self.sentinel_pages)
        if len(ids) - len(self.held) < 5:
            raise ValueError('illegal original-arena target exclusions')
        self.manager.release_blocks_by_ids([n for n in ids if n not in self.held])
        proof = {'actor_pid': os.getpid(), 'original_allocator': True, 'all_pages_claimed': len(ids),
            'all_claimed_hashes_invalidated': True, 'excluded_pages_held': sorted(self.held),
            'free_after': self.manager.num_free_blocks, 'time_ns': time.monotonic_ns()}
        self.hbm_guard_before = (await self.inspect_hbm(self.sentinel_pages,'hbm-guards-before'))['pages'] if self.sentinel_pages else []
        proof['sentinel_pages_held']=self.sentinel_pages
        self.last_reset = proof
        return proof

    async def request(self, text, interface='direct'):
        import httpx
        from m2_baseline import validate_output
        await self.idle()
        start = time.monotonic_ns();schedule_start = len(self.schedule)
        usage_before=self.usage()
        first_sequence = self.coord._trace_sequence if self.coord is not None else 0
        outputs = []
        if interface == 'direct':
            request_id = self.engine.generate_request_id()
            async for output in self.engine.add_request(request_id, text, self.generation):
                outputs.append({'time_ns': time.monotonic_ns(), **asdict(output)})
            record = {'name': request_id, 'generate_config': asdict(self.generation),
                      'prompt_token_ids': self.engine.tokenizer.encode(text), 'outputs': outputs}
            ids = validate_output(record)
            if self.engine.tokenizer.decode(ids) != outputs[-1]['text']:
                raise ValueError('decoded output differs from emitted token IDs')
            first_token_ns = next(x['time_ns'] for x in outputs if x['token_id'] is not None)
            body = None
        elif interface == 'http':
            chunks = []
            first_token_ns = None
            async with httpx.AsyncClient(trust_env=False, timeout=90) as client:
                async with client.stream('POST', f"http://127.0.0.1:{self.device['port']}/v1/completions", json={
                    'model': self.engine_config.model_id, 'prompt': text, 'max_tokens': 2,
                    'temperature': 0.0, 'top_p': 1.0, 'seed': 0, 'stream': True,
                    'stream_options': {'include_usage': True}}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith('data: ') or line == 'data: [DONE]':continue
                        part = json.loads(line[6:]);now = time.monotonic_ns()
                        chunks.append({'time_ns': now, 'body': part})
                        if first_token_ns is None and any(c.get('text') for c in part.get('choices', [])):
                            first_token_ns = now
            if not chunks or first_token_ns is None:raise ValueError('HTTP stream emitted no text')
            ids = {x['body']['id'] for x in chunks};usages=[x['body']['usage'] for x in chunks if x['body'].get('usage')]
            if len(ids) != 1 or len(usages) != 1 or usages[0]['completion_tokens'] != 2:
                raise ValueError('HTTP request identity/terminal token accounting differs')
            if usages[0]['prompt_tokens'] != len(self.engine.tokenizer.encode(text)):
                raise ValueError('HTTP prompt token accounting differs')
            request_id = ids.pop();body={'stream_chunks': chunks, 'usage': usages[0]}
        else:
            raise ValueError('unsupported request interface')
        await self.idle()
        steps = self.schedule[schedule_start:]
        if {s['request_id'] for s in steps} != {request_id}:
            raise ValueError('request/scheduler correlation is not unique')
        trace = [x for x in self.coord.trace if x['sequence'] > first_sequence] if self.coord else []
        row = {'request_id': request_id, 'interface': interface, 'prompt': text,
            'prompt_token_ids': self.engine.tokenizer.encode(text), 'generate_config': asdict(self.generation),
            'usage_before': usage_before, 'usage_after': self.usage(),
            'native_max_inflight_config': 1 if self.shared else 0,
            'outputs': outputs, 'body': body, 'begin_ns': start, 'first_token_ns': first_token_ns,
            'completed_ns': time.monotonic_ns(), 'schedule': steps, 'coordinator_trace': trace,
            'prefill_tokens': sum(s['tokens'] for s in steps if s['is_prefill']),
            'loads': [x for x in trace if x['event'] == 'transfer_terminal' and x.get('operation') == 'LOAD'],
            'stores': [x for x in trace if x['event'] == 'transfer_terminal' and x.get('operation') == 'STORE'],
            'reset_proof': getattr(self, 'last_reset', None),
            'TTFT_kind': 'first received nonempty HTTP text' if interface == 'http' else 'first direct token', 'actor_pid': os.getpid(), 'device': self.device}
        self.last_reset = None
        self.records.append(row)
        return row

    def usage(self):
        from pypto_serving.serving.native_ddr.managed_process import descendants
        rows={}
        for pid in [os.getpid(),*descendants()]:
            try:rows[str(pid)]={'stat':Path(f'/proc/{pid}/stat').read_text(),
                               'status':Path(f'/proc/{pid}/status').read_text()}
            except (FileNotFoundError,ProcessLookupError):continue
        return {'time_ns':time.monotonic_ns(),'owned_processes':rows,
                'coordinator_jobs':len(self.coord.jobs) if self.coord else 0}

    async def route_observation(self,request_id,begin_ns,end_ns):
        await self.idle()
        return {'request_id':request_id,'actor_pid':os.getpid(),
            'schedule':[x for x in self.schedule if x['request_id']==request_id],
            'coordinator_trace':[x for x in self.coord.trace if begin_ns<=x['time_ns']<=end_ns]}

    async def inspect_hbm(self,pages,label):
        await self.idle();self.counter+=1
        command=self.coord._control('diagnostic',page_ids=tuple(pages),diagnostic_label=f'm1-{label}-{self.counter}')
        await self.wait(lambda:command.command_id in self.coord.diagnostic_results)
        error,payload=self.coord.diagnostic_results.pop(command.command_id)
        if error:raise RuntimeError(error)
        return json.loads(payload)

    async def capture(self, text, label, guards=False):
        from pypto_serving.serving.native_ddr.types import PrefixKey
        await self.idle()
        hashes = self.manager.compute_block_hashes(self.engine.tokenizer.encode(text))[:2]
        captures = []
        for i, value in enumerate(hashes):
            key = PrefixKey(self.coord.namespace, value);ref = self.coord.ready[key]
            block = self.manager.get_cached_block(value)
            if block is None:raise RuntimeError('no legal cached original-arena page to inspect')
            page = block.block_id;self.manager.release_cached_blocks([block])
            self.counter += 1
            diagnostic_label = ('r06-' if guards and i == 0 else 'm1-') + f'{label}-{self.counter}'
            cmd = self.coord._control('diagnostic', entry_ref=ref, page_ids=(page,), diagnostic_label=diagnostic_label)
            await self.wait(lambda: cmd.command_id in self.coord.diagnostic_results)
            error, payload = self.coord.diagnostic_results.pop(cmd.command_id)
            if error:raise RuntimeError(error)
            capture = json.loads(payload)
            captures.append({'key': [key.namespace_digest.hex(), key.block_hash.hex()], 'entry_ref': asdict(ref),
                'page': page, 'native_actor': os.getpid(), 'device': self.device, 'capture': capture})
        result={'blocks':captures,'num_pages':self.manager.num_blocks}
        if self.hbm_guard_before:
            result['hbm_guard_before']=self.hbm_guard_before
            result['hbm_guard_after']=(await self.inspect_hbm(self.sentinel_pages,'hbm-guards-after'))['pages']
        return result

    async def overwrite(self, original):
        from qwen_scenario import overwrite_candidates
        await self.reset()
        allocated = self.manager.allocate_blocks(self.manager.num_free_blocks)
        if allocated is None:raise RuntimeError('source overwrite could not claim original arena')
        self.held.update(b.block_id for b in allocated)
        proofs = []
        for index, item in enumerate(original['blocks']):
            source = item['page']
            self.manager.release_blocks_by_ids([source]);self.held.remove(source)
            tokens = original['prompt_ids'][index * 128:(index + 1) * 128]
            for attempt, (text, _) in enumerate(overwrite_candidates(self.engine.tokenizer, tokens)):
                if attempt >= 2:raise RuntimeError('bounded overwrite attempts exhausted; keep original domain')
                row = await self.request(text)
                if any(s['pages'] != [source] for s in row['schedule']):
                    raise RuntimeError('overwrite computation used another page')
                first = next(o['token_id'] for o in row['outputs'] if o['token_id'] is not None)
                if first != tokens[127]:break
            else:raise RuntimeError('no bounded real overwrite candidate')
            self.counter += 1
            cmd = self.coord._control('diagnostic', page_ids=(source,), diagnostic_label=f'm1-overwrite-{self.counter}')
            await self.wait(lambda: cmd.command_id in self.coord.diagnostic_results)
            error, payload = self.coord.diagnostic_results.pop(cmd.command_id)
            if error:raise RuntimeError(error)
            proofs.append({'request': row, 'capture': json.loads(payload)})
            held = self.manager.allocate_blocks(1)
            if held is None or [b.block_id for b in held] != [source]:raise RuntimeError('source was not truly reusable')
            self.held.add(source)
        return {'overwrites': proofs, 'num_pages': self.manager.num_blocks}

    async def stop(self):
        await self.idle()
        if self.held:
            self.manager.release_blocks_by_ids(sorted(self.held));self.held.clear()
        self.server.should_exit = True
        await self.server_task
        worker_pid = self.core._worker_process.pid
        ok = await self.engine.stop()
        if not ok:raise RuntimeError('normal model stop retained obligations')
        proof=None
        if self.shared:
            proof={'worker_epoch':self.coord.worker_epoch,'module_drained':self.coord.drained,
                   'owner_drained_arguments_released':self.coord.stopped,
                   'jobs':len(self.coord.jobs),'steps':len(self.coord.steps),
                   'controls':len(self.coord.controls),'retiring':len(self.coord.retiring)}
            if (not proof['module_drained'] or not proof['owner_drained_arguments_released']
                    or any(proof[k] for k in ('jobs','steps','controls','retiring'))):
                raise RuntimeError('actual native owner/argument closure absent')
        return {'stop_ok': ok, 'shutdown_report': self.core.shutdown_report,
                'native_stop_proof':proof,
                'worker_pid': worker_pid, 'actor_pid': os.getpid(), 'shared': self.shared}


async def main():
    config = json.loads(Path(sys.argv[1]).read_text());index=int(sys.argv[2]);shared=sys.argv[3]=='shared'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(process)d %(levelname)s %(message)s')
    actor = ModelActor(config, index, shared, sys.argv[4])
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'run02'), str(Path(__file__).resolve().parents[1] / 'run04')]
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line or len(line)>65536:raise RuntimeError('bounded controller channel lost; retain actor')
            from scenario_wire import command
            request=command(line);op=request['op'];values=request['values']
            if op not in {'start','prompt','reset','request','capture','overwrite','route_observation','stop'}:
                raise ValueError('unsupported scenario command')
            method=getattr(actor,op)
            result=method(**values)
            if hasattr(result,'__await__'):result=await result
            path=actor.root / f"result-{request['id']}.json"
            save(path,result)
            protocol.write(json.dumps({'id':request['id'],'status':'ok','result_path':str(path)})+'\n')
            if op=='stop':return
    except BaseException:
        # Original engine/native traceback references stay alive on any failure.
        # No kill, PID-death ACK, replacement actor, or forced Python teardown.
        actor.root.mkdir(exist_ok=True,parents=True)
        save(actor.root/'retained-failure.json',{'pid':os.getpid(),'error':traceback.format_exc()})
        while True:await asyncio.sleep(30)


if __name__ == '__main__':
    asyncio.run(main())
