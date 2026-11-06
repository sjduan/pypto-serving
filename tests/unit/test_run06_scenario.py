# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU metadata/fake transport orchestration only. Never actual SDK/model/native evidence."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys

import pytest

RUN06=Path(__file__).resolve().parents[2]/'examples/run06'
# tests/unit/test_run06_scenario.py -> repository root is parents[2].
sys.path.insert(0,str(RUN06))
from scenario_contract import operation_budget,plan
from scenario_controller import orchestrate,check_request,samples
from scenario_admission import validate_admission,observe
from mutation_envelope import CAPS, STAGES, PhaseLedger, bound, digest, validate_envelope

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/m1'


@pytest.fixture
def config(tmp_path):
    mf=tmp_path/'manifest.json';mf.write_bytes((FIXTURE/'public-manifest.json').read_bytes())
    doc=json.loads((FIXTURE/'slots.json').read_text());doc['session_manifest']=str(mf)
    doc['source_manifest_sha256']=hashlib.sha256(mf.read_bytes()).hexdigest()
    bindings=tmp_path/'slots.json';bindings.write_text(json.dumps(doc))
    assets={'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json',
        *(f'model-{i:05d}-of-00008.safetensors' for i in range(1,9))}
    return {'schema':1,'bindings':str(bindings),'manifest_sha256':doc['source_manifest_sha256'],
        'root':'/workspace/uflow/runs/phasef-parallel-20261106/m1/device-test','replicas':2,
        'devices':[{'physical':12,'board':6,'chip':0,'bdf':'0000:85:00.0','port':18920},
                   {'physical':14,'board':7,'chip':0,'bdf':'0000:81:00.0','port':18921}],
        'model':'/data/models/Qwen/Qwen3-14B','model_assets':{n:'0'*64 for n in assets},
        'source_manifest':'/not-opened-in-plan/source-manifest.json','source_manifest_sha256':'1'*64,
        'native_env':'/not-opened-in-plan/native-env.sh','native_env_sha256':'2'*64,
        'host_env':'/not-opened-in-plan/host-env.sh','host_env_sha256':'3'*64,
        'operation_allowance':None,'device_release':None,'router':None}


class FakeTransport:
    real_runtime=False
    def __init__(self,fail=None):self.events=[];self.clock=100;self.reset={};self.fail=fail;self.retained=False
    async def admit(self,config,prepared):self.events.append(('admit',));self.config=config
    async def start_model(self,index,shared):
        name=('shared' if shared else 'cold')+str(index);self.events.append(('start',name));return name
    def model_index(self,name):return int(name.removeprefix('shared').removeprefix('cold'))
    async def stop_model(self,name):self.events.append(('stop',name));return {'CPU_only':True}
    async def start_directory(self):self.events.append(('directory-start',))
    async def stop_directory(self):self.events.append(('directory-stop',))
    async def verify_cleanup(self):self.events.append(('cleanup',));return {'CPU_only':True}
    async def verify_namespace(self,*args):self.events.append(('namespace',*args))
    async def pin_original(self,original):self.events.append(('pin-original',))
    async def release_original(self):self.events.append(('release-original',))
    async def check_overwrites(self,*args):self.events.append(('overwrite-oracle',))
    async def check_bytes(self,*args):return {'CPU_only':True}
    async def check_guards(self,*args):self.events.append(('guard-oracle',))
    async def router_check(self,names,text):self.events.append(('router',len(names)));return {'CPU_only':True}
    async def retain(self,error,report):self.retained=True;self.events.append(('retained',type(error).__name__))
    async def measured_request(self,name,text):return await self.call(name,'request',text=text)
    async def call(self,name,op,**values):
        self.events.append((op,name))
        if self.fail==(name,op):raise TimeoutError('synthetic pending native owner; CPU only')
        self.clock+=100
        if op=='prompt':return {'text':values.get('label','fixed257'),'ids':list(range(257))}
        if op=='reset':
            self.reset[name]={'time_ns':self.clock,'actor_pid':name,'original_allocator':True,
                              'all_claimed_hashes_invalidated':True};return self.reset[name]
        if op=='capture':return {'blocks':[{'page':0},{'page':1}],'num_pages':16}
        if op=='overwrite':return {'CPU_only':True}
        if op=='request':
            reset=self.reset.pop(name,None);shared=name.startswith('shared');different=values['text']!='fixed257'
            producer=name=='shared0';loads=2 if shared and not producer and reset and not different else 0
            events=[]
            for index in range(loads):
                event={'job':str(self.clock)+str(index),'status':'SUCCEEDED','native_completed':True,'stopped':True,
                    'bytes':20971520,'page_lease_id':'fake-page','delivery_sequence':self.clock+index,
                    'pool_receipt':{'operation':'LOAD','content':{'nbytes':20971520}}}
                events.append(event)
            return {'request_id':str(self.clock),'actor_pid':name,'reset_proof':reset,'prefill_tokens':257 if not shared or producer or different else 1,
                'loads':events,'stores':[{},{}] if producer else [],'schedule':[{'request_id':str(self.clock)}],
                'coordinator_trace':[{'event':'restored_page_valid','job':e['job']} for e in events],
                'begin_ns':self.clock,'first_token_ns':self.clock+1,'completed_ns':self.clock+2,
                'prompt_token_ids':list(range(257)),'generate_config':{'max_new_tokens':2,'seed':0},
                'interface':values.get('interface','direct')}
        raise AssertionError(op)


@pytest.mark.parametrize('replicas',[2,4])
def test_whole_sequence_with_every_sample_reset_and_post_exit_HTTP(config,replicas):
    if replicas==4:
        config['replicas']=4;config['devices'] += [
            {'physical':8,'board':4,'chip':0,'bdf':'0000:8D:00.0','port':18922},
            {'physical':9,'board':4,'chip':1,'bdf':'0000:8F:00.0','port':18923}]
        config['router']={'port':18929}
    fake=FakeTransport();result=asyncio.run(orchestrate(config,fake))
    assert result['scope']==result['status']=='CPU-ORCHESTRATION-ONLY' and result['model_acceptance'] is False
    assert len(result['baseline'])==replicas and all(len(v)==23 for v in result['baseline'].values())
    assert len(result['shared'])==replicas-1 and all(len(v['samples'])==23 for v in result['shared'].values())
    assert all(len(v['metrics']['paired_shared_minus_cold_ms'])==20 for v in result['shared'].values())
    stop=fake.events.index(('stop','shared0'));release=fake.events.index(('release-original',))
    assert fake.events.index(('pin-original',))<stop<release
    assert any(event==('request','shared1') for event in fake.events[release+1:])
    assert fake.events[-1]==('cleanup',) and not fake.retained
    assert len(result['functional'])==8*(replicas-1)
    if replicas==4:assert ('router',4) in fake.events[:stop]


def test_inflight_transport_failure_never_restarts_or_fabricates_cleanup(config):
    fake=FakeTransport(fail=('shared1','request'))
    with pytest.raises(TimeoutError):asyncio.run(orchestrate(config,fake))
    assert fake.retained and ('cleanup',) not in fake.events and ('stop','shared0') not in fake.events
    assert fake.events.count(('start','shared1'))==1


def test_bad_partition_is_rejected_before_any_transport_entry(config):
    path=Path(config['bindings']);doc=json.loads(path.read_text());doc['blocks'][0]['block_id']=1
    path.write_text(json.dumps(doc));fake=FakeTransport()
    with pytest.raises(ValueError):asyncio.run(orchestrate(config,fake))
    assert fake.events==[]


def test_plan_never_reads_model_or_source_or_opens_network(config,monkeypatch):
    import socket,subprocess
    def forbidden(*a,**k):raise AssertionError('plan entered network/process/device path')
    monkeypatch.setattr(socket,'socket',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    result=plan(config)
    assert result['status']=='NOT-RUN' and result['runtime_start_permitted_by_this_plan'] is False
    assert result['fixed_workload']['measured_each_arm']==20
    with pytest.raises(PermissionError):observe(config,Path(config['root']))


def released_config(config,tmp_path):
    p=plan(config);m=json.loads(Path(json.loads(Path(config['bindings']).read_text())['session_manifest']).read_text())
    source=tmp_path/'source.json';source.write_text(json.dumps({'serving':'f'*40}))
    config['source_manifest']=str(source);config['source_manifest_sha256']=digest(source.read_bytes())
    env={'schema':2,'decision_id':'CPU-METADATA-FIXTURE-NOT-A-GRANT','resource_group':'phasef-parallel-20261106',
        **{k:p['pool'][k] for k in ('pool_id','pool_epoch','provider_session')},
        'common_manifest_sha256':config['manifest_sha256'],'counter_anchor':312,
        'admission_limit':4096,'cleanup_limit':8192,'terminal_recovery_reserve':256,
        'participants':{line:{'source_sha':('f' if line=='M1' else 'e')*40,'total':CAPS[line],
            'stages':[{'id':s,'max':cap} for s,cap in STAGES[line]]} for line in CAPS}}
    path=tmp_path/'envelope.json';path.write_text(json.dumps(env))
    config['operation_allowance']={'path':str(path),'sha256':digest(path.read_bytes())}
    config['device_release']={'schema':1,'device_stage_allowed':True,'source_manifest_sha256':config['source_manifest_sha256'],
        'manifest_sha256':config['manifest_sha256'],'envelope_sha256':config['operation_allowance']['sha256']}
    state={'host':m['host'],'operation_records':312,'unknown_or_frozen':False,'M1_retired':False,
        'own_active_obligations':False,
        'budget_bytes':134217728,'charged_bytes':134217728,'creation_reserved_bytes':0,'pool_descriptor':p['pool'],
        'other_reservations':[],'devices':{str(d['physical']):{**d,'health':'OK','hbm_free_mib':60000,
            'fd_users':[],'fd_denied':[],'driver_processes_present':False} for d in config['devices']}}
    return env,state


@pytest.mark.parametrize('fault',['unknown','headroom','schema-bool','physical-alias','occupied','frozen','own-open'])
def test_future_admission_fails_before_process_or_SDK_mutation(config,tmp_path,monkeypatch,fault):
    env,state=released_config(config,tmp_path)
    import scenario_admission
    ledger=PhaseLedger(tmp_path/'ledger','0'*64,env,plan(config)['pool'],config['source_manifest_sha256'])
    monkeypatch.setattr(scenario_admission,'phase_ledger',lambda *_:ledger)
    if fault=='unknown':config['operation_allowance']=None
    elif fault=='headroom':state['operation_records']=4000
    elif fault=='schema-bool':config['device_release']['schema']=True
    elif fault=='physical-alias':state['devices']['12']['bdf']='0000:81:00.0'
    elif fault=='occupied':state['devices']['12']['fd_users']=[{'pid':123}]
    elif fault=='frozen':state['unknown_or_frozen']=True
    elif fault=='own-open':state['own_active_obligations']=True
    with pytest.raises(PermissionError):validate_admission(config,state)


@pytest.mark.parametrize('vector',json.loads((FIXTURE/'mutation-vectors.json').read_text()))
def test_v2_manager_reference_equivalence(vector):
    assert bound(vector['anchor'],vector['observed'],vector['line'],vector['phase'])==vector['expected']


@pytest.mark.parametrize('args',[(324,324,'M1','two'),(312,311,'M1','two'),
    (312,3201,'M1','two'),(True,312,'M1','two'),(312,True,'M1','two'),(312,312,'M1','skip')])
def test_v2_invalid_counters_and_phase_rejected(args):
    with pytest.raises(ValueError):bound(*args)


def test_other_line_twelve_spent_is_not_double_counted_and_zero_is_not_drain():
    assert bound(312,324,'O1','calibrate')['worst']==4084
    result=bound(312,3200,'M1','two')
    assert result['other_remaining_upper']==0 and result['not_a_resource_or_lease_grant'] is True


@pytest.mark.parametrize('fault',['schema','cap','stage-order','stage-bool','source','manifest','provider','extra','reserve'])
def test_v2_exact_immutable_envelope(config,tmp_path,fault):
    env,_=released_config(config,tmp_path);pool=plan(config)['pool']
    validate_envelope(env,pool,config['manifest_sha256'],'f'*40)
    if fault=='schema':env['schema']=True
    elif fault=='cap':env['participants']['M1']['total']=885
    elif fault=='stage-order':env['participants']['M1']['stages'].reverse()
    elif fault=='stage-bool':env['participants']['M1']['stages'][0]['max']=True
    elif fault=='source':env['participants']['M1']['source_sha']='a'*40
    elif fault=='manifest':env['common_manifest_sha256']='a'*64
    elif fault=='provider':env['provider_session']='other'
    elif fault=='extra':env['own_completed_upper']=248
    elif fault=='reserve':env['terminal_recovery_reserve']=128
    with pytest.raises(ValueError):validate_envelope(env,pool,config['manifest_sha256'],'f'*40)


def test_v2_admission_corrected_interleave_and_once_only_claim(config,tmp_path,monkeypatch):
    import scenario_admission
    env,state=released_config(config,tmp_path)
    ledger=PhaseLedger(tmp_path/'ledger','0'*64,env,plan(config)['pool'],config['source_manifest_sha256'])
    monkeypatch.setattr(scenario_admission,'phase_ledger',lambda *_:ledger)
    state['operation_records']=324
    assert validate_admission(config,state)['normal_records_bound']==248
    ledger.claim('two',324,'c'*64)
    with pytest.raises(ValueError,match='already claimed'):ledger.claim('two',324,'d'*64)
    with pytest.raises(OSError):ledger.check('four',324)
    with pytest.raises(PermissionError):validate_admission(config,state)


@pytest.mark.parametrize('scope',['CPU-ORCHESTRATION-ONLY','NOT-RUN','real-runtime-captured-pending-review'])
def test_v2_fake_host_or_shallow_report_cannot_advance(config,tmp_path,scope):
    env,_=released_config(config,tmp_path)
    ledger=PhaseLedger(tmp_path/'ledger','0'*64,env,plan(config)['pool'],config['source_manifest_sha256'])
    ledger.claim('two',312,'c'*64)
    report={'phase_identity':ledger.identity('two'),'scope':scope,'status':'CAPTURED-PENDING-REVIEW',
        'M1_line_retired':False,'pool_destroyed':False,'plan':plan(config),'cleanup':{'CPU_only':True}}
    with pytest.raises((ValueError,KeyError)):ledger.close('two',report,tmp_path/'fake.json')
    assert not (ledger.root/'two.closed.json').exists()


def closure_metadata_fixture(config,env):
    """Synthetic typed metadata ONLY, to exercise the validator without any runtime."""
    report=asyncio.run(orchestrate(config,FakeTransport()))
    report['scope']='real-runtime-captured-pending-review';report['status']='CAPTURED-PENDING-REVIEW'
    snapshot=json.loads((FIXTURE/'uf-snapshot.json').read_text());pool=snapshot['pools'][0]
    sessions=[row['client_session'] for row in pool['clients'][:4]]
    starts={};stops={};births={}
    for index,name in enumerate(['cold0','cold1','shared0','shared1']):
        pid=7000+index;worker=8000+index;births[str(pid)]=births[str(worker)]=True
        startup={'worker_epoch':'test-'+name,'pool_identity':{'client_session':sessions[index-2]}} if index>=2 else None
        starts[name]={'pid':pid,'worker_pid':worker,'birth':'CPU-only-birth','descendant_births':{str(worker):'CPU-only-child'},
            'startup':startup}
        stops[name]={'actor_pid':pid,'worker_pid':worker,'stop_ok':True,
            'shutdown_report':{'cleanup_error':None,'worker_reaped':True},
            'native_stop_proof':{'worker_epoch':'test-'+name,'module_drained':True,
                'owner_drained_arguments_released':True,'jobs':0,'steps':0,'controls':0,'retiring':0}}
    report['cleanup']={'model_startups':starts,'model_stops':stops,'all_own_sessions_actual_Unmapped':sessions,
        'actual_common_snapshot':snapshot,'original_births_gone':births,'reservations_released':True,
        'directory_socket_absent':True,'listeners_closed':True,
        'devices':[{'fds':{'users':[],'denied':[]},'status':{'health':'OK'}} for _ in range(2)]}
    return report


@pytest.mark.parametrize('fault',[None,'source','native','worker','child','lease','fd','birth','counter','config','host-scope'])
def test_v2_full_closure_validation_and_phase_progression(config,tmp_path,fault):
    env,_=released_config(config,tmp_path)
    ledger=PhaseLedger(tmp_path/'ledger','0'*64,env,plan(config)['pool'],config['source_manifest_sha256'])
    ledger.claim('two',312,plan(config)['config_sha256'])
    report=closure_metadata_fixture(config,env);report['phase_identity']=ledger.identity('two')
    cleanup=report['cleanup'];pool=cleanup['actual_common_snapshot']['pools'][0]
    if fault=='source':report['phase_identity']['source_sha']='a'*40
    elif fault=='native':cleanup['model_stops']['shared0']['native_stop_proof']['module_drained']=False
    elif fault=='worker':cleanup['model_stops']['cold0']['shutdown_report']['worker_reaped']=False
    elif fault=='child':pool['clients'][0]['children']=[{'unknown':False}]
    elif fault=='lease':pool['ranges']['leases']=[{'client_session':cleanup['all_own_sessions_actual_Unmapped'][0]}]
    elif fault=='fd':cleanup['devices'][0]['fds']['users']=[{'pid':123}]
    elif fault=='birth':cleanup['original_births_gone'].pop('8000')
    elif fault=='counter':cleanup['actual_common_snapshot']['operation_records']=311
    elif fault=='config':report['plan']['config_sha256']='a'*64
    elif fault=='host-scope':report['scope']='CPU-ORCHESTRATION-ONLY'
    if fault:
        with pytest.raises((ValueError,KeyError)):ledger.close('two',report,tmp_path/'report.json')
        assert not (ledger.root/'two.closed.json').exists()
    else:
        ledger.close('two',report,tmp_path/'report.json')
        result=ledger.check('four',324)
        assert result['own_completed_upper']==248 and result['own_remaining_upper']==636
        ledger.claim('four',324,'c'*64)
        with pytest.raises(ValueError,match='already claimed'):ledger.check('four',324)
        # Corrupting a once-sealed original report cannot confer C=248 later.
        (tmp_path/'report.json').write_text('{}')
        with pytest.raises(ValueError,match='hash'):ledger.prior('done')


def test_missing_reset_or_accepted_only_output_cannot_pass():
    row={'prefill_tokens':1,'loads':[],'schedule':[{}],'begin_ns':1,'first_token_ns':2,'completed_ns':3,'reset_proof':None}
    with pytest.raises(ValueError):check_request(row,prefill=1,loads=2)
    with pytest.raises(ValueError):check_request(row,prefill=1,loads=0)
    with pytest.raises(ValueError):samples([],[])


def test_sdk_operation_budget_counts_cleanup_against_admission_gate():
    a,b=operation_budget(2),operation_budget(4)
    assert a['required_shared_record_allowance']==a['normal_records_bound']
    assert a['normal_records_bound']+b['normal_records_bound']==884
    assert 312+2888+884<4096
    assert a['normal_records_bound']<b['normal_records_bound']<4096
    assert a['admission_stops_at_ALL_record_count']==4096 and b['cleanup_stops_at_ALL_record_count']==8192
    assert 'Release' in ' '.join(b['new_journal_records_by_path'])


@pytest.mark.parametrize('packet',[
    {'id':True,'op':'start','values':{}},
    {'id':1,'op':'exec','values':{}},
    {'id':1,'op':'request','values':{}},
    {'id':1,'op':'capture','values':{'text':'x','label':'../../escape'}},
    {'id':1,'op':'reset','values':{'avoid':[True]}},
    {'id':1,'op':'start','values':{'arbitrary_command':'no'}},
    {'id':1,'op':'overwrite','values':{'original':{'blocks':[],'prompt_ids':[]}}}])
def test_actor_wire_rejects_missing_confused_or_unsafe_metadata(packet):
    from scenario_wire import command
    with pytest.raises(ValueError):command(json.dumps(packet)+'\n')


def test_actor_start_without_release_does_not_import_runtime_or_register(config,tmp_path,monkeypatch):
    from model_actor import ModelActor
    from pypto_serving.serving.native_ddr import managed_pool_sdk
    def forbidden(*args,**kwargs):raise AssertionError('host-only actor entered SDK')
    monkeypatch.setattr(managed_pool_sdk,'accepted_sdk',forbidden)
    with pytest.raises(PermissionError):asyncio.run(ModelActor(config,0,True,tmp_path/'not-created').start())
    assert not (tmp_path/'not-created').exists()


def test_cancelled_inflight_transport_preserves_owner_without_cleanup(config):
    class Cancelled(FakeTransport):
        async def call(self,name,op,**values):
            if name=='shared1' and op=='request':raise asyncio.CancelledError()
            return await super().call(name,op,**values)
    transport=Cancelled()
    with pytest.raises(asyncio.CancelledError):asyncio.run(orchestrate(config,transport))
    assert transport.retained and ('cleanup',) not in transport.events
