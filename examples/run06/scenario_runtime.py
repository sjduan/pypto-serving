# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual future process transport. Never imported by plan-only or CPU fake orchestration."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
import json
import os
import re
from pathlib import Path
import socket
import sys
import time
import traceback

from scenario_contract import digest, save, validate
from scenario_admission import observe, phase_ledger, validate_admission, validate_release, verify_source_closure


class ActorCommandError(RuntimeError):
    """The original actor failed a command and retained its actual owner."""


class ProcessTransport:
    real_runtime=True

    def __init__(self, config, config_path):
        self.config,self.config_path=config,str(Path(config_path).resolve())
        self.root=Path(config['root']);self.models={};self.started={};self.stopped={};self.sessions=[]
        self.held=ExitStack();self.sequence=0;self.directory=self.observer=None
        self.logs={};self.listeners={};self.owned_births={};self.commands=[];self.reserved=False
        self.manifest=validate(config)[2]
        self.recovery = None
        self.observer_pinned = self.directory_ready = False
        self.failures = []
        self.recovery_stopping = False

    async def install_recovery(self, prepared):
        # Pure release/source metadata only. Install before admit, claim or actor creation.
        from recovery_control import RecoveryControl
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        envelope = validate_release(self.config)
        self.ledger = phase_ledger(self.config, envelope)
        self.phase = 'two' if self.config['replicas'] == 2 else 'four'
        source = json.loads(Path(self.config['source_manifest']).read_text())
        binding = {'uid': os.getuid(), 'controller_pid': os.getpid(), 'controller_birth': process_birth(os.getpid()),
            'config_sha256': prepared['config_sha256'], 'source_commit': source['serving'],
            'source_manifest_sha256': self.config['source_manifest_sha256'],
            'public_manifest_sha256': self.config['manifest_sha256'],
            'envelope_sha256': self.config['operation_allowance']['sha256'],
            'claim_path': str(self.ledger.root / (self.phase + '.started.json')), 'root': str(self.root)}
        self.recovery = RecoveryControl(str(self.root) + '.recovery.sock', binding, self.stop_owned)
        await self.recovery.install()

    async def admit(self, config, prepared):
        envelope=validate_release(config)
        self.ledger=phase_ledger(config,envelope);self.phase='two' if config['replicas']==2 else 'four'
        self.ledger.prior(self.phase)
        import fcntl
        self.root.mkdir();(self.root/'logs').mkdir()
        sys.path[:0]=[self.manifest['helper_root'],str(Path(__file__).resolve().parents[1]/'run02'),
                       str(Path(__file__).resolve().parents[1]/'run04')]
        from resources import host_job
        with host_job(self.manifest):
            source=await asyncio.to_thread(verify_source_closure,config)
        fresh=await asyncio.to_thread(observe,config,self.root)
        validate_admission(config,fresh);save(self.root/'admission-before-locks.json',fresh)
        for d in config['devices']:
            stream=self.held.enter_context(Path(f"/workspace/uflow/locks/run02-npu-{d['physical']}.lock").open('a+'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked=await asyncio.to_thread(observe,config,self.root);validate_admission(config,locked)
        # Exclusive immutable marker precedes reservations, SDK registration and
        # Worker creation. Failure keeps this claim; changing the run root cannot retry it.
        self.ledger.claim(self.phase,locked['operation_records'],prepared['config_sha256'])
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        from session_protocol import request
        self.parent_birth=process_birth(os.getpid())
        reservations=[{'physical_device':d['physical'],'board':d['board'],'chip':d['chip'],'bdf':d['bdf'],
            'execution_domain':d['bdf'],'lock_path':f"/workspace/uflow/locks/run02-npu-{d['physical']}.lock",
            'lock_holder_pid':os.getpid(),'lock_holder_birth':self.parent_birth,
            'evidence_path':str(self.root/'admission-before-locks.json')} for d in config['devices']]
        self.reserved=True  # A lost metadata response is not a confirmed release.
        acknowledgement=request(self.manifest,'reserve-devices',line='M1',reservations=reservations)
        self.admission={'parent_pid':os.getpid(),'parent_birth':self.parent_birth,
            'config_sha256':prepared['config_sha256'],'source':source['serving'],'reservations':reservations,
            'snapshot':locked,'pool_admin':False,'actual_reservation_ACK':acknowledgement}
        save(self.root/'parent-admission.json',self.admission)

    async def spawn(self, name, script, args, *, host=False, pass_fds=()):
        env_path=self.config['host_env' if host else 'native_env']
        logfile=(self.root/'logs'/f'{name}.log').open('xb');self.logs[name]=logfile
        command=['bash','--noprofile','--norc','-c','source "$1"; shift; exec "$@"','m1-env',env_path,
                 'python','-u','-B',str(Path(__file__).with_name(script)),*map(str,args)]
        child=await asyncio.create_subprocess_exec(*command,stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,stderr=logfile,pass_fds=pass_fds,limit=65537)
        self.commands.append({'name':name,'argv':command,'pid':child.pid,'host_fence_only':host})
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        self.owned_births[child.pid]=process_birth(child.pid)
        return child

    async def start_model(self,index,shared):
        # Each physical domain must still be genuinely idle before a new engine.
        # Other owned model cards may be live, so inspect only this target here.
        from r_guard import snapshot
        from scenario_contract import MIN_FREE_HBM_MIB
        d=self.config['devices'][index]
        fresh=await asyncio.to_thread(snapshot,self.root,'model-start-'+str(self.sequence),d['physical'])
        s=fresh['status']
        bdf=re.search(rf'\|\s*{s["chip"]}\s+{d["physical"]}\s*\|\s*([0-9A-Fa-f:.]+)\s*\|',fresh['npu'])
        if (not bdf or bdf[1]!=d['bdf'] or s['board']!=d['board'] or s['chip']!=d['chip']
                or s['health']!='OK' or s['hbm_total_mib']-s['hbm_used_mib']<MIN_FREE_HBM_MIB
                or fresh['fds']['users'] or fresh['fds']['denied']):
            raise PermissionError('target physical model domain no longer idle/admissible')
        name=('shared' if shared else 'cold')+str(index)
        listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',d['port']));listener.listen(128)
        self.listeners[name]=listener
        child=await self.spawn(name,'model_actor.py',[self.config_path,index,'shared' if shared else 'cold',
            self.root/name,listener.fileno()],pass_fds=(listener.fileno(),))
        self.models[name]=child
        from resources import host_job
        with host_job(self.manifest):
            result=await self.call(name,'start')
        self.started[name]=result
        for pid,birth in result['descendant_births'].items():self.owned_births[int(pid)]=birth
        if shared:
            startup=result['startup'];compat=startup['compatibility_json']
            if startup['capacity_bytes']!=83886080 or startup['max_entries']!=4:
                raise RuntimeError('model startup differs from shared four logical slots')
            for n,h in self.config['model_assets'].items():
                if compat['assets']['files'][n]['sha256']!=h:raise RuntimeError('loaded asset content differs')
            self.verify_loaded_closure(compat)
            self.sessions.append(startup['pool_identity']['client_session'])
        return name

    def verify_loaded_closure(self,compat):
        source=json.loads(Path(self.config['source_manifest']).read_text())
        deployment=compat['deployment']
        for component,group,prefix in (('serving','serving','pypto_serving/'),('qwen_lib','lib','models/qwen3_14b/')):
            entries={name[len(prefix):]:row['sha256'] for name,row in source['groups'][group].items()
                     if name.startswith(prefix) and name.endswith('.py')}
            encoded=json.dumps(entries,sort_keys=True,separators=(',',':')).encode()
            if (deployment[component]['source_files']!=len(entries)
                    or deployment[component]['source_manifest_sha256']!=digest(encoded)):
                raise RuntimeError('actual imported model source closure differs from sealed source manifest')
        expected=self.config['device_release']['runtime_files']
        for component,key in (('pypto.pypto_core','pypto_core'),('_task_interface','task_interface'),('ptoas','ptoas')):
            if deployment[component]['sha256']!=expected[key]['sha256']:
                raise RuntimeError('actual imported runtime/compiler binary differs')
        for component,key in (('host_path','native_host'),('aicpu_path','native_aicpu'),('aicore_path','native_aicore')):
            if deployment['native_backend'][component]!=expected[key]['sha256']:
                raise RuntimeError('actual loaded native backend differs')
        if (deployment['uflow.managed_pool']['sha256']!='366f3ea883cec08efd62f23f9ebf2afa8546c02a777ff25fdf753fee1debff0e'
                or deployment['uflow._managed_pool_wire']['sha256']!='c61fdf8f9c19261b9386ae5c64e1ac8e2b81cf41238ee9c8f3c24c7cab59015e'):
            raise RuntimeError('actual loaded UF SDK differs from accepted production')

    async def rpc(self, child, name, op, values):
        self.sequence+=1;number=self.sequence
        packet={'id':number,'op':op,'values':values}
        data=json.dumps(packet).encode()+b'\n'
        if len(data)>65536:raise ValueError('scenario metadata command exceeds bound')
        child.stdin.write(data);await child.stdin.drain()
        raw=await asyncio.wait_for(child.stdout.readline(),timeout=1000)
        if not raw or len(raw)>65536:raise RuntimeError('original actor channel missing/oversized; retain')
        from scenario_wire import _unique
        response=json.loads(raw,object_pairs_hook=_unique)
        if type(response.get('id')) is not int or response.get('id')!=number:
            raise RuntimeError('actor response identity differs')
        if response.get('status')=='error':
            if (set(response)!={'id','status','error','retained','failure_path'} or response['retained'] is not True
                    or type(response['error']) is not str or len(response['error'])>2048
                    or Path(response['failure_path']).resolve()!=(self.root/name/'retained-failure.json').resolve()):
                raise RuntimeError('actor error response schema/owned failure path differs')
            save(self.root/f'command-{number}.json',{'actor':name,'request':packet,'response':response,
                 'response_time_ns':time.monotonic_ns(),'actual_actor_error':True})
            raise ActorCommandError(f'{name} command {number} {op}: {response["error"]}')
        if response.get('status')!='ok' or set(response)!={'id','status','result_path'}:
            raise RuntimeError('actor success response schema differs')
        path=Path(response['result_path'])
        if not path.resolve().is_relative_to((self.root/name).resolve()) or path.stat().st_size>16<<20:
            raise ValueError('actor evidence escaped owned root/bound')
        value=json.loads(path.read_text())
        save(self.root/f'command-{number}.json',{'actor':name,'request':packet,'response':response,
            'evidence_sha256':digest(path.read_bytes()),'response_time_ns':time.monotonic_ns()})
        return value

    async def call(self,name,op,**values):
        value=await self.rpc(self.models[name],name,op,values)
        if op=='request':
            from qwen_evidence import correlate_native,bind_round_event
            raw=(self.root/'logs'/f'{name}.log').read_text(errors='replace')
            proof={'coordinator_trace':value['coordinator_trace']}
            correlate_native(proof,raw)
            for event in value['coordinator_trace']:
                if event['event']=='transfer_terminal':bind_round_event(event,proof,raw)
        return value

    def model_index(self,name):return int(name.removeprefix('shared').removeprefix('cold'))

    async def measured_request(self,name,text):
        from resources import measurement_window
        row=None
        try:
            with measurement_window(self.manifest) as window:
                before={'loadavg':Path('/proc/loadavg').read_text(),'time_ns':time.monotonic_ns()}
                row=await self.call(name,'request',text=text,interface='direct')
                row['measurement_context']={'window':window,'before':before,
                    'after':{'loadavg':Path('/proc/loadavg').read_text(),'time_ns':time.monotonic_ns()},
                    'reservation_state':json.loads(Path(self.manifest['coordination']['physical_reservations_file']).read_text()),
                    'contention':'unclassified; raw co-tenant context retained, no samples excluded'}
        except BaseException:
            if row is not None:save(self.root/f'contended-sample-{self.sequence}.json',row)
            raise
        return row

    async def verify_namespace(self,producer,peer):
        a=self.started[producer]['startup'];b=self.started[peer]['startup']
        if (a['namespace_digest']!=b['namespace_digest'] or a['worker_epoch']==b['worker_epoch']
                or a['pool_identity']['client_session']==b['pool_identity']['client_session']):
            raise RuntimeError('independent model compatibility/owner boundary differs')

    async def start_directory(self):
        c=self.config;self.directory=await self.spawn('directory','directory_service.py',[
            '--root',self.root/'directory','--endpoint',self.root/'d.sock','--epoch','m1-'+self.root.name,
            '--bindings',c['bindings'],'--pool-endpoint',self.manifest['endpoint']],host=True)
        await self.wait_file(self.root/'directory/ready.json',self.directory)
        reg=list(self.root.glob('directory-*-registered.json'))
        if len(reg)!=1:raise RuntimeError('directory original SDK identity missing')
        self.sessions.append(json.loads(reg[0].read_text())['identity']['client_session'])
        self.directory_ready=True

    async def wait_file(self,path,child):
        deadline=time.monotonic()+60
        while not path.exists():
            if child.returncode is not None or time.monotonic()>deadline:raise RuntimeError('owned process readiness missing')
            await asyncio.sleep(.02)
        return json.loads(path.read_text())

    async def pin_original(self,original):
        self.observer=await self.spawn('observer','prefix_observer.py',[self.config_path,self.root/'observer'],host=True)
        registered=await self.wait_file(self.root/'observer/registered.json',self.observer)
        self.sessions.append(registered['identity']['client_session'])
        contents=[b['capture']['ddr']['committed'] for b in original['blocks']]
        self.observer.stdin.write((json.dumps({'op':'pin','contents':contents})+'\n').encode());await self.observer.stdin.drain()
        answer=json.loads(await asyncio.wait_for(self.observer.stdout.readline(),60))
        if answer.get('status')!='ok':raise RuntimeError('original observer pins unconfirmed')
        pins=json.loads(Path(answer['result_path']).read_text())
        if len(pins['held_R_grants'])!=2 or pins['payload_bytes_read']!=0:raise RuntimeError('observer is not exact real R metadata')
        self.observer_pinned=True

    async def release_original(self):
        if not self.observer_pinned:raise RuntimeError('observer original pins unconfirmed; retain')
        self.observer.stdin.write(b'{"op":"release"}\n');await self.observer.stdin.drain()
        answer=json.loads(await asyncio.wait_for(self.observer.stdout.readline(),60))
        if answer.get('status')!='ok':raise RuntimeError('observer original release outcome unknown')
        closed=json.loads(Path(answer['result_path']).read_text())
        if not closed['SDK_client_unmapped'] or len(closed['actual_release_ACKs'])!=2:raise RuntimeError('observer not drained')
        if await asyncio.wait_for(self.observer.wait(),30)!=0:raise RuntimeError('observer abnormal exit')
        self.observer=None
        self.observer_pinned=False

    def capture_bytes(self,row):
        path=Path(row['file'])
        if not path.resolve().is_relative_to(self.root.resolve()):raise ValueError('capture escaped task root')
        raw=path.read_bytes()
        if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:raise ValueError('capture full bytes/seal differ')
        return raw

    async def check_overwrites(self,original,overwritten):
        if len(original['blocks'])!=2 or len(overwritten['overwrites'])!=2:raise ValueError('two overwrite proofs required')
        for src,over in zip(original['blocks'],overwritten['overwrites']):
            if len(src['capture']['pages'])!=1:raise ValueError('source capture is not unique')
            raw=self.capture_bytes(src['capture']['pages'][0]);ddr=self.capture_bytes(src['capture']['ddr'])
            changed=self.capture_bytes(over['capture']['pages'][0])
            # Only the row-change oracle is needed here; target byte evidence
            # is checked independently after actual consumer LOAD below.
            if len(changed)!=20971520 or raw!=ddr:raise ValueError('source/DDR full bytes differ')
            stride=2048
            if any(raw[i:i+stride]==changed[i:i+stride] for i in range(0,len(raw),stride)):
                raise ValueError('not all128 rows on all80 sides were model-overwritten')

    async def check_bytes(self,original,overwritten,target,row):
        from verify_bytes import verify_roundtrip
        if len(original['blocks'])!=len(target['blocks']) or len(target['blocks'])!=2:raise ValueError('two complete prefix blocks required')
        before=target.get('hbm_guard_before',[]);after=target.get('hbm_guard_after',[])
        if len(before)!=2 or len(after)!=2:raise ValueError('two real held HBM sentinel captures required')
        targets={b['page'] for b in target['blocks']}
        for x,y in zip(before,after):
            if x['page']!=y['page'] or x['page'] in targets or self.capture_bytes(x)!=self.capture_bytes(y):
                raise ValueError('consumer HBM neighbor/sentinel bytes changed')
        checks=[]
        for index,(src,dst) in enumerate(zip(original['blocks'],target['blocks'])):
            if src['key']!=dst['key'] or src['device']['physical']==dst['device']['physical']:
                raise ValueError('consumer key/device independence differs')
            raw=self.capture_bytes(src['capture']['pages'][0]);committed=self.capture_bytes(src['capture']['ddr'])
            restored=self.capture_bytes(dst['capture']['pages'][0]);current=self.capture_bytes(dst['capture']['ddr'])
            if not raw==committed==restored==current or src['page']==dst['page']:
                raise ValueError('full source/DDR/distinct-target payload differs')
            terminals=[e for e in row['loads'] if e['pages']==[dst['page']]]
            if len(terminals)!=1 or terminals[0]['pool_receipt']['content']!=src['capture']['ddr']['committed']:
                raise ValueError('target actual LOAD did not consume original exact committed range')
            if overwritten is not None:
                changed=self.capture_bytes(overwritten['overwrites'][index]['capture']['pages'][0])
                if not (0<=src['page']<original['num_pages'] and 0<=dst['page']<target['num_pages']):
                    raise ValueError('source/target page is not legal in its independent original arena')
                checked=verify_roundtrip(raw,changed,committed,restored,num_pages=max(original['num_pages'],target['num_pages']),
                    source_page=src['page'],target_page=dst['page'])
                checked['oracle_index_upper_bound']=checked.pop('num_pages')
                checked['source_num_pages']=original['num_pages'];checked['target_num_pages']=target['num_pages']
                checked['arenas']='independent owner incarnations; separate legality checks above'
            else:checked={'all_20971520_bytes_equal':True,'producer_exit_not_yet_proved':True}
            checks.append(checked)
        return {'request_id':row['request_id'],'interface':row['interface'],'blocks':checks,'producer_exited':overwritten is not None}

    async def check_guards(self,original,target):
        a=original['blocks'][0]['capture']['shared_guards'];b=target['blocks'][0]['capture']['shared_guards']
        if len(a['guards'])!=4 or len(b['guards'])!=4:raise ValueError('four assigned guard readers required')
        for x,y in zip(a['guards'],b['guards']):
            if x['block']!=y['block']:raise ValueError('guard allocation generation differs')
            if self.capture_bytes(x)!=self.capture_bytes(y):raise ValueError('assigned full leased guard bytes changed')
            if not x['release_ack']['applied'] or not y['release_ack']['applied']:raise ValueError('guard release ACK missing')

    async def router_check(self,names,text):
        import httpx
        import secrets
        import uvicorn
        from pypto_serving.router.app import create_router_app
        from pypto_serving.router.config import ReplicaSpec,RouterConfig
        from pypto_serving.router.http import SESSION_HEADER
        if len(names)!=4:raise ValueError('four real live services required before router validation')
        replicas=tuple(ReplicaSpec(name,'127.0.0.1',self.config['devices'][self.model_index(name)]['port']) for name in names)
        app=create_router_app(RouterConfig(replicas=replicas,hosts=(),initial_replicas=0,replace_dead=False),
                              admin_token=secrets.token_hex(32))
        listener=socket.socket();listener.bind(('127.0.0.1',18929));listener.listen(128)
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=18929,log_level='info'))
        task=asyncio.create_task(server.serve(sockets=[listener]));rows=[];session='m1-functional-affinity'
        try:
            deadline=time.monotonic()+30
            while not server.started:
                if task.done():await task
                if time.monotonic()>deadline:raise TimeoutError('ordinary router readiness missing')
                await asyncio.sleep(.01)
            await app.state.router.health.probe_once()
            async with httpx.AsyncClient(trust_env=False,timeout=90) as client:
                for replica in replicas:
                    response=await client.get(replica.base_url+'/health');response.raise_for_status()
                for _ in range(2):
                    begin=time.monotonic_ns()
                    response=await client.post('http://127.0.0.1:18929/v1/completions',headers={SESSION_HEADER:session},
                        json={'model':Path(self.config['model']).name,'prompt':text,'max_tokens':2,
                              'temperature':0.0,'top_p':1.0,'seed':0,'stream':False})
                    response.raise_for_status();end=time.monotonic_ns();body=response.json()
                    if body['usage']['prompt_tokens']!=257 or body['usage']['completion_tokens']!=2:
                        raise ValueError('router output token accounting differs')
                    pin=app.state.router.sessions.lookup(session)
                    if pin not in names:raise ValueError('ordinary affinity selected an unknown replica')
                    observations=[await self.call(n,'route_observation',request_id=body['id'],begin_ns=begin,end_ns=end) for n in names]
                    actual=[o for o in observations if o['schedule']]
                    if len(actual)!=1 or actual[0]['actor_pid']!=self.started[pin]['pid']:
                        raise ValueError('router/body/request/actor schedule correlation differs')
                    rows.append({'body':body,'headers':dict(response.headers),'pin':pin,'observations':observations,
                                 'begin_ns':begin,'completed_ns':end})
            if rows[0]['pin']!=rows[1]['pin']:raise ValueError('ordinary session affinity was not preserved')
            return {'requests':rows,'live_external_replica_names':names,'router_launches_or_owns_models':False,
                    'model_startups':{n:self.started[n] for n in names}}
        finally:
            server.should_exit=True
            await task
            listener.close()

    async def stop_model(self,name):
        if name not in self.started:raise RuntimeError('original model startup incomplete; retain')
        result=await self.call(name,'stop')
        start=self.started[name];report=result['shutdown_report']
        if (result['stop_ok'] is not True or report['cleanup_error'] is not None
                or report['worker_reaped'] is not True or report['native_ddr_obligations']!=0
                or result['actor_pid']!=start['pid'] or result['worker_pid']!=start['worker_pid']
                or report['worker_pid']!=start['worker_pid'] or result['shared'] is not start['shared']):
            raise RuntimeError('actual native model stop not proved')
        if start['shared']:
            proof=result['native_stop_proof']
            if (proof['worker_epoch']!=start['startup']['worker_epoch']
                    or proof['module_drained'] is not True or proof['owner_drained_arguments_released'] is not True
                    or any(type(proof[k]) is not int or proof[k]!=0 for k in ('jobs','steps','controls','retiring'))):
                raise RuntimeError('original native drain/owner argument proof absent')
        elif result['native_stop_proof'] is not None:raise RuntimeError('unexpected native proof for native-off actor')
        for pid,birth in result['descendant_births_before'].items():
            if str(pid) in start['descendant_births'] and start['descendant_births'][str(pid)]!=birth:
                raise RuntimeError('original descendant birth changed before stop')
            self.owned_births[int(pid)]=birth
        if await asyncio.wait_for(self.models[name].wait(),30)!=0:raise RuntimeError('model actor abnormal exit')
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        births={**start['descendant_births'],**result['descendant_births_before']}
        if process_birth(start['pid'])==start['birth'] or any(process_birth(int(pid))==birth for pid,birth in births.items()):
            raise RuntimeError('normal model stop left an original descendant birth')
        from stop_evidence import check_shared_names_gone,check_device_closed
        check_shared_names_gone(start['private_shared_names']+result['private_shared_names_before'])
        from r_guard import snapshot
        after=snapshot(self.root,'stopped-'+name,self.config['devices'][self.model_index(name)]['physical'])
        check_device_closed(after)
        save(self.root/(name+'-device-stopped.json'),after)
        from session_protocol import request,pool_record,check_retired
        if self.started[name]['shared']:
            state=request(self.manifest,'describe');pool=pool_record(state['snapshot'],self.manifest['pool_descriptor'])
            session=self.started[name]['startup']['pool_identity']['client_session'];check_retired(pool,[session])
            save(self.root/(name+'-actual-SDK-unmapped.json'),{'session':session,'pool':pool,'worker_stop':result})
        self.models.pop(name)
        if not getattr(self,'recovery_stopping',False):self.listeners.pop(name).close()
        self.stopped[name]=result
        return result

    async def stop_directory(self):
        if not self.directory_ready:raise RuntimeError('directory startup incomplete; retain')
        save(self.root/'directory/stop.json',{'epoch':'m1-'+self.root.name})
        if await asyncio.wait_for(self.directory.wait(),30)!=0:raise RuntimeError('directory normal close failed')
        if (self.root/'d.sock').exists():raise RuntimeError('own metadata endpoint remains')
        closed=json.loads((self.root/'directory/closed.json').read_text())
        if closed['SDK_client_unmapped'] is not True or closed['socket_absent'] is not True:
            raise RuntimeError('actual directory SDK/socket closure absent')
        self.directory=None
        self.directory_ready=False

    async def verify_cleanup(self):
        if (self.models or self.observer or self.directory
                or self.listeners and not getattr(self,'recovery_stopping',False)):
            raise RuntimeError('original owned actors/listeners not closed')
        from session_protocol import request,pool_record,check_retired
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        state=request(self.manifest,'describe');pool=pool_record(state['snapshot'],self.manifest['pool_descriptor'])
        check_retired(pool,self.sessions)
        if any(process_birth(pid)==birth for pid,birth in self.owned_births.items()):raise RuntimeError('owned actor birth remains')
        save(self.root/'commands.json',self.commands)
        result={'all_own_sessions_actual_Unmapped':self.sessions,'actual_common_snapshot':state['snapshot'],
            'model_startups':self.started,'model_stops':self.stopped,
            'original_births_gone':{str(pid):process_birth(pid)!=birth for pid,birth in self.owned_births.items()},
            'directory_socket_absent':not (self.root/'d.sock').exists(),'listeners_closed':not self.listeners,'devices':[],
            'M1_line_retired':False,'common_pool_finalized':False,'B_retained_bytes':134217728,'A_historical_bytes':134217728}
        from r_guard import snapshot
        from stop_evidence import check_device_closed
        for device in self.config['devices']:
            final=snapshot(self.root,'final-'+str(device['physical']),device['physical'])
            check_device_closed(final)
            save(self.root/('final-device-'+str(device['physical'])+'.json'),final)
            result['devices'].append(final)
        acknowledgement=request(self.manifest,'reserve-devices',line='M1',reservations=[])
        save(self.root/'reservation-release-ACK.json',acknowledgement)
        self.reserved=False
        for listener in self.listeners.values():listener.close()
        self.listeners.clear()
        self.held.close()
        result['listeners_closed']=True
        result['reservations_released']=True
        return result

    async def stop_owned(self):
        # The endpoint can select only these retained original handles, never a caller PID.
        self.recovery_stopping=True
        for name in list(self.models):
            await self.stop_model(name)
        if self.observer is not None:await self.release_original()
        if self.directory is not None:await self.stop_directory()
        result=await self.verify_cleanup()
        save(self.root/'recovery-cleanup.json',result)
        for stream in self.logs.values():stream.close()
        return result

    async def publish_closed(self,report):
        report['phase_identity']=self.ledger.identity(self.phase)
        self.ledger.close(self.phase,report,self.root/'phase-evidence.json')
        await self.recovery.close()

    async def retain(self,error,report):
        self.failures.append(error)  # Retain the original scenario exception and traceback.
        if not (self.models or self.directory or self.observer or self.reserved):
            self.held.close()
            if self.recovery is not None:await self.recovery.close()
            raise error
        try:
            save(self.root/'retained-controller.json',{'pid':os.getpid(),'error':repr(error),
                'traceback':''.join(traceback.format_exception(error)),
                'report':report,'commands':self.commands,'owned_births':self.owned_births,
                'action':'original domain/locks/processes retained; no force kill/new pool/false SDK ACK'})
            if self.recovery is None:raise RuntimeError('recovery was not installed before admission')
            self.recovery.activate(self.ledger.root/(self.phase+'.started.json'),self.ledger.identity(self.phase),
                                   report['plan']['config_sha256'])
        except BaseException as binding_error:
            self.failures.append(binding_error)
            # Preserve all actual handles even when evidence storage or claim
            # binding fails; no exception here may abandon a retained owner.
            await asyncio.Event().wait()  # Unknown claim cannot close or abandon its original owner.
        await self.recovery.completed.wait()
        await self.recovery.close()
        raise error  # Recovery closes obligations; the failed campaign stays failed.
