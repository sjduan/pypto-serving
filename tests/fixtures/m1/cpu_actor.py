# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only independent protocol peer; real ModelActor request/stop, no model/UF/device."""
import ast
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'examples/run06')]
from actor_control import serve_actor
from model_actor import ModelActor
from pypto_serving.config.types import GenerateConfig


@dataclass
class Output:
    finished: bool
    finish_reason: str | None
    token_ids: tuple[int, ...]
    token_id: int | None
    text: str
    text_delta: str
    completion_tokens: int
    prompt_tokens: int = 257


async def main():
    validator = ROOT / 'examples/run02/m2_baseline.py'
    function = next(n for n in ast.parse(validator.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_output')
    module = ModuleType('m2_baseline')
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(validator), 'exec'), module.__dict__)
    sys.modules['m2_baseline'] = module
    actor = ModelActor({}, 0, False, sys.argv[1])
    actor.root.mkdir()
    actor.coord = None
    actor.device = {'CPU_fixture': True}
    actor.generation = GenerateConfig(max_new_tokens=2, min_tokens=2, temperature=0.0, seed=0, stream=True)
    async def idle():pass
    actor.idle = idle
    actor.usage = lambda: {'CPU_fixture': True}
    async def stream(request_id, text, generation):
        actor.schedule.append({'request_id': request_id, 'is_prefill': True, 'tokens': 257})
        yield Output(False, None, (10,), 10, 'A', 'A', 1)
        yield Output(True, 'FINISHED_LENGTH', (10, 11), 12, 'AB', 'B', 2)
    worker = None
    async def start():
        nonlocal worker
        worker = subprocess.Popen([sys.executable, '-B', '-c', 'import sys; assert sys.stdin.readline()=="stop\\n"'], stdin=subprocess.PIPE)
        actor.core = SimpleNamespace(_worker_process=worker, scheduler=SimpleNamespace(has_work=lambda: sys.argv[2] == 'unknown'),
                                     _batch_queue=[], shutdown_report={})
        actor.manager = SimpleNamespace(release_blocks_by_ids=lambda values: None)
        actor.server = SimpleNamespace(should_exit=False)
        actor.http_gate = SimpleNamespace(active=0)
        async def server():
            while not actor.server.should_exit:await asyncio.sleep(.005)
        actor.server_task = asyncio.create_task(server())
        actor.start_complete = True
        from pypto_serving.serving.native_ddr.managed_process import descendants, process_birth
        return {'pid': os.getpid(), 'birth': process_birth(os.getpid()), 'worker_pid': worker.pid,
                'descendant_births': descendants(), 'private_shared_names': [], 'shared': False, 'startup': None}
    async def stop(*, cooperative_only):
        assert cooperative_only is True
        with (actor.root / 'actual-engine-stop.txt').open('x') as file:file.write('cooperative_only=True')
        if sys.argv[2] == 'stop-fails':return False
        worker.stdin.write(b'stop\n');worker.stdin.flush()
        code = await asyncio.to_thread(worker.wait, 10)
        actor.core.shutdown_report = {'worker_reaped': code == 0, 'cleanup_error': None,
                                     'worker_pid': worker.pid, 'native_ddr_obligations': 0}
        return code == 0
    actor.start = start
    actor.engine = SimpleNamespace(generate_request_id=lambda: 'CPU-request', add_request=stream, stop=stop,
        tokenizer=SimpleNamespace(encode=lambda text: list(range(257)), decode=lambda ids: 'AB'))
    task = asyncio.create_task(serve_actor(actor, sys.stdin, sys.stdout))
    while not task.done() and not Path(sys.argv[3]).exists():await asyncio.sleep(.01)
    if not task.done():
        # Test-only release of CPU fixtures, independent of the product channel.
        # Never used with an engine/model, SDK client or actual device worker.
        if worker is not None and worker.poll() is None:
            worker.stdin.write(b'stop\n');worker.stdin.flush()
            assert await asyncio.to_thread(worker.wait, 10) == 0
        if actor.server is not None:actor.server.should_exit = True
        task.cancel()
        await asyncio.sleep(.01)
        if not task.done():task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    else:
        await task


if __name__ == '__main__':asyncio.run(main())
