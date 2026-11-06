# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Original actor channel: immediate error replies, then one cooperative stop only."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import traceback

from scenario_contract import save
from scenario_wire import command


class RetainedHTTPGate:
    """Count complete ASGI response lifetimes, including streaming bodies."""
    def __init__(self, app, actor):
        self.app, self.actor, self.active = app, actor, 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        if self.actor.retained:
            from starlette.responses import JSONResponse
            return await JSONResponse({'error': 'original actor retained; generation admission closed'},
                                      status_code=503)(scope, receive, send)
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1


class ActorControl:
    def __init__(self, actor):
        self.actor = actor
        self.highest_id = 0
        self.failures = []  # Keep exceptions and their original traceback/object references alive.
        self.stop_attempted = False

    def retain(self, error):
        self.actor.retained = True
        self.failures.append(error)
        path = self.actor.root / 'retained-failure.json'
        if not path.exists():
            try:
                self.actor.root.mkdir(exist_ok=True, parents=True)
                save(path, {'pid': os.getpid(), 'error': repr(error),
                            'traceback': ''.join(traceback.format_exception(error)),
                            'action': 'original actor retained; only original cooperative stop permitted'})
            except Exception as evidence_error:
                self.failures.append(evidence_error)  # Evidence I/O must not delay the error reply.

    async def dispatch(self, raw):
        request = None
        try:
            request = command(raw)
            number, op = request['id'], request['op']
            if number <= self.highest_id:
                raise ValueError('duplicate or stale actor command; no operation re-executed')
            self.highest_id = number
            if self.actor.retained and op != 'stop':
                raise RuntimeError('retained actor accepts only stop')
            if op == 'stop':
                if self.stop_attempted:
                    raise RuntimeError('original stop already attempted; retain uncertain outcome')
                self.stop_attempted = True
                self.actor.retained = True  # Close HTTP admission before checking quiescence.
            result = getattr(self.actor, op)(**request['values'])
            if inspect.isawaitable(result):
                result = await result
            path = self.actor.root / f'result-{number}.json'
            save(path, result)
            return {'id': number, 'status': 'ok', 'result_path': str(path)}, op == 'stop'
        except BaseException as error:
            self.retain(error)
            return {'id': request['id'] if request else None, 'status': 'error',
                    'error': repr(error)[:2048], 'retained': True,
                    'failure_path': str(self.actor.root / 'retained-failure.json')}, False


async def serve_actor(actor, reader, writer):
    control = ActorControl(actor)
    while True:
        # Bounded readline prevents allocating unbounded controller metadata.
        try:
            line = await asyncio.to_thread(reader.readline, 65537)
        except BaseException as error:
            control.retain(error)
            await asyncio.Event().wait()
        if not line:
            control.retain(RuntimeError('original controller channel lost; retain owner'))
            await asyncio.Event().wait()
        response, closed = await control.dispatch(line)
        try:
            writer.write(json.dumps(response) + '\n')
            writer.flush()
        except BaseException as error:
            control.retain(error)
            await asyncio.Event().wait()
        if closed:
            return
