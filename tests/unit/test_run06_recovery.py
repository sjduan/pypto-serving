# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real CPU processes/UNIX sockets and actual actor/transport methods; no devices or UF."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from contextlib import ExitStack, suppress

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples/run06'))
from actor_control import ActorControl
from model_actor import ModelActor
from recovery_control import RecoveryControl
from scenario_runtime import ActorCommandError, ProcessTransport


def test_actor_failure_keeps_original_exception_and_rejects_duplicate_or_new_work(tmp_path):
    async def scenario():
        error = ValueError('original validator object')
        calls = []
        async def request(text):
            calls.append(text)
            raise error
        async def stop():
            calls.append('stop')
            raise RuntimeError('actual stop not proved')
        actor = SimpleNamespace(root=tmp_path, retained=False, request=request, stop=stop)
        control = ActorControl(actor)
        def packet(number, op, values=None):return json.dumps({'id': number, 'op': op, 'values': values or {}}) + '\n'
        reply, closed = await control.dispatch(packet(1, 'request', {'text': 'first'}))
        assert reply['status'] == 'error' and reply['id'] == 1 and not closed
        assert control.failures[0] is error and error.__traceback__ is not None
        for raw in (packet(1, 'request', {'text': 'duplicate'}), packet(2, 'reset'), packet(3, 'request', {'text': 'new'})):
            assert (await control.dispatch(raw))[0]['status'] == 'error'
        assert (await control.dispatch(packet(4, 'stop')))[0]['status'] == 'error'
        assert (await control.dispatch(packet(5, 'stop')))[0]['status'] == 'error'
        assert calls == ['first', 'stop'] and actor.retained
    asyncio.run(scenario())


@pytest.mark.skipif(os.name != 'posix', reason='Linux owned-birth/shared-memory and independent process contract')
@pytest.mark.parametrize('mode', ['normal', 'unknown', 'stop-fails'])
def test_real_actor_validator_error_replies_immediately_and_only_original_stop_remains(tmp_path, mode):
    async def scenario():
        root = tmp_path / 'actor'
        release = tmp_path / 'cpu-test-release'
        process = await asyncio.create_subprocess_exec(sys.executable, '-B', str(ROOT / 'tests/fixtures/m1/cpu_actor.py'),
            str(root), mode, str(release), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        transport = ProcessTransport.__new__(ProcessTransport)
        transport.root = tmp_path;transport.sequence = 0
        transport.models = {'actor': process}
        transport.manifest = {'coordination_root': str(tmp_path)}
        async def rpc(op, **values):return await asyncio.wait_for(transport.rpc(process, 'actor', op, values), 10)
        try:
            await rpc('start')
            with pytest.raises(ActorCommandError, match='Stream/final token data mismatch'):
                await asyncio.wait_for(transport.measured_request('actor', 'CPU prompt'), 10)
            # Use the original fixed resources.measurement_window above, and a
            # different OS process below. The failed RPC must have released the
            # real kernel FLOCK before this recovery/stop work begins.
            child = await asyncio.create_subprocess_exec(sys.executable, '-B', '-c',
                'import fcntl,sys; f=open(sys.argv[1],"a+"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',
                str(tmp_path/'measurement.lock'))
            assert await asyncio.wait_for(child.wait(), 5) == 0
            for op, values in [('request', {'text': 'again'}), ('reset', {}), ('prompt', {})]:
                with pytest.raises(ActorCommandError, match='only stop'):await rpc(op, **values)
            if mode == 'normal':
                result = await rpc('stop')
                assert result['stop_ok'] is True and result['shutdown_report']['worker_reaped'] is True
                assert await asyncio.wait_for(process.wait(), 10) == 0
            else:
                with pytest.raises(ActorCommandError, match='in-flight/Unknown' if mode == 'unknown' else 'retained obligations'):
                    await rpc('stop')
                assert process.returncode is None
                with pytest.raises(ActorCommandError, match='already attempted'):await rpc('stop')
            assert (root / 'actual-engine-stop.txt').exists() is (mode != 'unknown')
            failure = json.loads((root / 'retained-failure.json').read_text())
            assert 'validate_output' in failure['traceback']
        finally:
            release.touch()
            process.stdin.close()
            assert await asyncio.wait_for(process.wait(), 15) == 0, (await process.stderr.read()).decode()
    asyncio.run(scenario())


@pytest.mark.skipif(os.name != 'posix', reason='Linux original controller SO_PEERCRED endpoint')
@pytest.mark.parametrize('failure', [False, True])
def test_recovery_endpoint_binds_original_identity_and_duplicate_never_reexecutes(failure):
    async def scenario():
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        with tempfile.TemporaryDirectory(prefix='m1-recovery-') as directory:
            root = Path(directory);claim = root / 'two.started.json'
            binding = {'uid': os.getuid(), 'controller_pid': os.getpid(), 'controller_birth': process_birth(os.getpid()),
                'claim_path': str(claim), 'config_sha256': 'c'*64, 'source_commit': 'a'*40,
                'source_manifest_sha256': 'b'*64, 'public_manifest_sha256': 'd'*64, 'envelope_sha256': 'e'*64}
            calls = []
            gate = asyncio.Event()
            async def stop_owned():
                calls.append('actual-stop-owner')
                await gate.wait()
                if failure:raise RuntimeError('Unknown retains original process/SDK obligation')
                return {'CPU_mock_API_closed': True}
            endpoint = RecoveryControl(root / 'r.sock', binding, stop_owned)
            await endpoint.install()
            async def send(packet):
                reader, writer = await asyncio.open_unix_connection(str(endpoint.path))
                writer.write((json.dumps(packet) + '\n').encode());await writer.drain()
                answer = json.loads(await asyncio.wait_for(reader.readline(), 5))
                writer.close()
                with suppress(ConnectionError):await writer.wait_closed()
                return answer
            try:
                assert (await send({'op': 'stop-owned', 'identity': None, 'operation_id': None}))['status'] == 'rejected'
                claim.write_text(json.dumps({'identity': {'phase': 'two'}, 'config_sha256': 'c'*64}))
                endpoint.activate(claim, {'phase': 'two'}, 'c'*64)
                packet = {'op': 'stop-owned', 'identity': endpoint.identity, 'operation_id': endpoint.operation_id}
                for field in binding:
                    wrong = json.loads(json.dumps(packet));wrong['identity'][field] = 'wrong'
                    assert (await send(wrong))['status'] == 'rejected'
                for wrong in ({**packet, 'pid': os.getpid()}, {**packet, 'op': 'exec'},
                              {**packet, 'operation_id': 'f'*64}):
                    assert (await send(wrong))['status'] == 'rejected'
                original=claim.read_bytes();claim.write_bytes(original+b'\n')
                assert (await send(packet))['status']=='rejected'
                claim.write_bytes(original)
                endpoint.binding['uid']+=1
                assert (await send(packet))['status']=='rejected'  # Actual SO_PEERCRED mismatch.
                endpoint.binding['uid']-=1
                assert calls == []
                reader,writer=await asyncio.open_unix_connection(str(endpoint.path))
                writer.write((json.dumps(packet)+'\n').encode());await writer.drain()
                writer.close();await writer.wait_closed()  # A disconnected requester cannot cancel the owner.
                first = asyncio.create_task(send(packet));second = asyncio.create_task(send(packet))
                while not calls:await asyncio.sleep(.001)
                assert calls == ['actual-stop-owner']
                gate.set()
                expected = 'retained' if failure else 'closed'
                assert [r['status'] for r in await asyncio.gather(first, second)] == [expected, expected]
                assert (await send(packet))['status'] == expected and len(calls) == 1
                assert endpoint.completed.is_set() is not failure
                receipt = json.loads(endpoint.path.with_suffix('.result.json').read_text())
                assert receipt['campaign_success'] is False and receipt['four_eligible'] is False
                assert not (root / 'two.closed.json').exists()
                if failure:assert endpoint.failures and endpoint.path.exists()
            finally:
                await endpoint.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('state', ['incomplete', 'inflight', 'unknown', 'http-inflight'])
def test_actual_model_stop_refuses_uncertain_or_incomplete_engine_before_stop_api(tmp_path, state):
    actor = ModelActor({}, 0, False, tmp_path)
    actor.start_complete = state != 'incomplete'
    actor.http_gate = SimpleNamespace(active=int(state == 'http-inflight'))
    actor.engine = SimpleNamespace(stop=lambda **kwargs: pytest.fail('unsafe engine stop invoked'))
    actor.core = SimpleNamespace(_worker_process=SimpleNamespace(pid=123), _batch_queue=[],
                                scheduler=SimpleNamespace(has_work=lambda: state == 'inflight'))
    actor.coord = SimpleNamespace(jobs={}, controls={}, steps={}, retiring={}, poisoned=True) if state == 'unknown' else None
    with pytest.raises(RuntimeError, match='incomplete startup|in-flight/Unknown'):
        asyncio.run(actor.stop())
    assert actor.retained


def test_http_gate_tracks_stream_body_and_rejects_new_admission_after_retain():
    from actor_control import RetainedHTTPGate
    async def scenario():
        actor = SimpleNamespace(retained=False)
        entered, finish = asyncio.Event(), asyncio.Event()
        calls=[]
        async def app(scope,receive,send):
            calls.append('body-start');entered.set()
            await finish.wait()
            calls.append('body-final')
        gate=RetainedHTTPGate(app,actor)
        async def send(message):calls.append(message.get('status'))
        pending=asyncio.create_task(gate({'type':'http'},None,send))
        await entered.wait();assert gate.active==1
        actor.retained=True
        await gate({'type':'http'},None,send)
        assert gate.active==1 and 503 in calls and calls.count('body-start')==1
        finish.set();await pending
        assert gate.active==0 and calls[-1]=='body-final'
    asyncio.run(scenario())


@pytest.mark.skipif(os.name != 'posix', reason='Linux independent original actor/Worker and process birth proof')
@pytest.mark.parametrize('sdk_unknown', [False, True])
@pytest.mark.parametrize('late_tracker', [False, True])
def test_actual_stop_owned_closes_original_cpu_workers_then_requires_mock_sdk_unmap_before_release(tmp_path,monkeypatch,sdk_unknown,late_tracker):
    """Exercise real stop_model/verify_cleanup, not a fake transport success flag."""
    from types import ModuleType
    from pypto_serving.serving.native_ddr.managed_process import process_birth
    async def scenario():
        root=tmp_path/'cold0';release=tmp_path/'cpu-test-release'
        process=await asyncio.create_subprocess_exec(sys.executable,'-B',str(ROOT/'tests/fixtures/m1/cpu_actor.py'),
            str(root),'late-tracker' if late_tracker else 'normal',str(release),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        transport=ProcessTransport.__new__(ProcessTransport)
        transport.root=tmp_path;transport.sequence=0;transport.models={'cold0':process}
        transport.started={};transport.stopped={};transport.sessions=['mock-session']
        transport.manifest={'pool_descriptor':{'CPU_fixture':True}}
        transport.config={'devices':[{'physical':0}]};transport.commands=[]
        transport.owned_births={process.pid:process_birth(process.pid)}
        transport.observer=transport.directory=None;transport.reserved=True
        transport.held=ExitStack();closed=[]
        transport.held.callback(lambda:closed.append('actual-lock-stack-closed'))
        listener=SimpleNamespace(close=lambda:closed.append('listener-closed'))
        transport.listeners={'cold0':listener};transport.logs={}
        sdk=ModuleType('session_protocol');operations=[]
        def request(manifest,op,**values):
            operations.append((op,values))
            if op=='describe':return {'snapshot':{'CPU_mock_catalog':True}}
            assert op=='reserve-devices' and values=={'line':'M1','reservations':[]}
            return {'actual_CPU_mock_API_ACK':True}
        sdk.request=request;sdk.pool_record=lambda snapshot,descriptor:snapshot
        def check_retired(pool,sessions):
            assert sessions==['mock-session']
            if sdk_unknown:raise RuntimeError('CPU mock SDK client Unknown; no release')
        sdk.check_retired=check_retired
        guard=ModuleType('r_guard')
        guard.snapshot=lambda *args:{'fds':{'users':[],'denied':[]},'status':{'board':0,'chip':0,'health':'OK'},
                                    'npu':'Process id\n| No running processes found in NPU 0 |'}
        monkeypatch.setitem(sys.modules,'session_protocol',sdk);monkeypatch.setitem(sys.modules,'r_guard',guard)
        try:
            start=await transport.call('cold0','start');transport.started['cold0']=start
            transport.owned_births.update({int(pid):birth for pid,birth in start['descendant_births'].items()})
            tail_alive_after_actor_exit=[]
            if late_tracker:
                from pypto_serving.serving.native_ddr import managed_process
                tail=set(map(int,start['descendant_births']))-{start['worker_pid']}
                assert len(tail)==1
                def observe_birth(pid):
                    value=process_birth(pid)
                    if pid in tail and value is not None and process_birth(process.pid)!=start['birth']:
                        tail_alive_after_actor_exit.append(pid)
                    return value
                monkeypatch.setattr(managed_process,'process_birth',observe_birth)
            with pytest.raises(ActorCommandError):await transport.call('cold0','request',text='CPU prompt')
            if sdk_unknown:
                with pytest.raises(RuntimeError,match='SDK client Unknown'):await transport.stop_owned()
                assert transport.reserved and 'actual-lock-stack-closed' not in closed
                assert all(op!='reserve-devices' for op,_ in operations)
            else:
                result=await transport.stop_owned()
                assert result['reservations_released'] is True and not transport.reserved
                assert all(result['original_births_gone'].values())
                assert closed==['listener-closed','actual-lock-stack-closed']
                assert operations[-1]==('reserve-devices',{'line':'M1','reservations':[]})
            assert await asyncio.wait_for(process.wait(),10)==0
            if late_tracker:assert tail_alive_after_actor_exit
            assert not (tmp_path/'two.closed.json').exists()
        finally:
            release.touch();process.stdin.close()
            assert await asyncio.wait_for(process.wait(),15)==0,(await process.stderr.read()).decode()
    asyncio.run(scenario())
