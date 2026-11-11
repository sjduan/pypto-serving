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


def test_synthetic_manifest_v2_cannot_reuse_v1_runtime_envelope(config,tmp_path):
    from scenario_admission import validate_release
    released_config(config,tmp_path)
    validate_release(config)
    bindings=Path(config['bindings']);document=json.loads(bindings.read_text())
    mf=Path(document['session_manifest']);manifest=json.loads(mf.read_text())
    manifest['manifest_version']=2
    manifest['controller_birth']={'pid':90000001,'starttime_ticks':90000002}
    mf.write_text(json.dumps(manifest));new_sha=digest(mf.read_bytes())
    document['source_manifest_sha256']=new_sha;bindings.write_text(json.dumps(document))
    config['manifest_sha256']=new_sha;config['device_release']['manifest_sha256']=new_sha
    # The new metadata itself is accepted, but the original immutable grant
    # remains bound to the old public manifest and cannot authorize this domain.
    assert plan(config)['status']=='NOT-RUN'
    with pytest.raises(PermissionError,match='immutable shared mutation envelope'):validate_release(config)


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


def m1_only_config(config,tmp_path):
    env,state=released_config(config,tmp_path)
    env['schema']=3;env['participants'].pop('O1')
    path=Path(config['operation_allowance']['path']);path.write_text(json.dumps(env))
    config['operation_allowance']['sha256']=digest(path.read_bytes())
    config['device_release']['envelope_sha256']=config['operation_allowance']['sha256']
    return env,state


def test_v3_m1_only_release_keeps_actual_cap_and_finalizer_reserve(config,tmp_path,monkeypatch):
    import scenario_admission
    from scenario_admission import validate_release
    env,state=m1_only_config(config,tmp_path)
    assert validate_release(config)==env
    ledger=PhaseLedger(tmp_path/'ledger','3'*64,env,plan(config)['pool'],config['source_manifest_sha256'])
    monkeypatch.setattr(scenario_admission,'phase_ledger',lambda *_:ledger)
    assert validate_admission(config,state)['normal_records_bound']==248
    budget=ledger.claim('two',312,plan(config)['config_sha256'])
    assert budget=={'participants':['M1'],'own_completed_upper':0,'own_remaining_upper':884,
                    'finalizer_records_reserved':2,'worst':1198,'strict_admission_spare':2897,
                    'not_a_resource_or_lease_grant':True}
    assert ledger.identity('two')['schema']==3
    assert CAPS=={'O1':2888,'M1':884}  # Schema2 tables and reference vectors remain unchanged.
    with pytest.raises(ValueError,match='already claimed'):ledger.claim('two',312,'a'*64)
    with pytest.raises(OSError):ledger.check('four',312)


@pytest.mark.parametrize('fault',['schema-bool','schema-float','extra-O1','missing-M1','cap','stage-order',
    'stage-bool','extra-stage','source','manifest','provider','epoch','pool','anchor-bool','headroom',
    'admission','cleanup','terminal','extra-field','missing-field','extra-participant-field'])
def test_v3_exact_m1_only_envelope(config,tmp_path,fault):
    env,_=m1_only_config(config,tmp_path);pool=plan(config)['pool']
    validate_envelope(env,pool,config['manifest_sha256'],'f'*40)
    if fault=='schema-bool':env['schema']=True
    elif fault=='schema-float':env['schema']=3.0
    elif fault=='extra-O1':env['participants']['O1']={'source_sha':'e'*40,'total':2888,'stages':[]}
    elif fault=='missing-M1':env['participants']={}
    elif fault=='cap':env['participants']['M1']['total']=886
    elif fault=='stage-order':env['participants']['M1']['stages'].reverse()
    elif fault=='stage-bool':env['participants']['M1']['stages'][0]['max']=True
    elif fault=='extra-stage':env['participants']['M1']['stages'].append({'id':'finalize','max':2})
    elif fault=='source':env['participants']['M1']['source_sha']='a'*40
    elif fault=='manifest':env['common_manifest_sha256']='a'*64
    elif fault=='provider':env['provider_session']='different'
    elif fault=='epoch':env['pool_epoch']='different'
    elif fault=='pool':env['pool_id']='different'
    elif fault=='anchor-bool':env['counter_anchor']=True
    elif fault=='headroom':env['counter_anchor']=3210
    elif fault=='admission':env['admission_limit']=4097
    elif fault=='cleanup':env['cleanup_limit']=8193
    elif fault=='terminal':env['terminal_recovery_reserve']=255
    elif fault=='extra-field':env['finalizer']=0
    elif fault=='missing-field':env.pop('terminal_recovery_reserve')
    elif fault=='extra-participant-field':env['participants']['M1']['other_spent']=0
    with pytest.raises((ValueError,KeyError)):validate_envelope(env,pool,config['manifest_sha256'],'f'*40)


@pytest.mark.parametrize('args',[(312,311,'M1','two'),(312,313,'M1','two'),(312,561,'M1','four'),
    (312,1197,'M1','done'),(312,312,'O1','calibrate'),(312,312,'M1','skip'),
    (True,312,'M1','two'),(312,True,'M1','two'),(3210,3210,'M1','two')])
def test_v3_no_other_participant_spend_or_unproven_counter_credit(args):
    with pytest.raises(ValueError):bound(*args,schema=3)


def test_v3_strict_boundary_reserves_finalizer_without_adding_to_normal_cap():
    assert bound(3209,3209,'M1','two',schema=3)['worst']==4095
    assert bound(3209,3457,'M1','four',schema=3)['own_remaining_upper']==636
    assert bound(3209,4093,'M1','done',schema=3)['finalizer_records_reserved']==2
    for schema in (True,3.0,4):
        with pytest.raises(ValueError):bound(312,312,'M1','two',schema=schema)


@pytest.mark.parametrize('fault',[None,'over-cap','legacy-identity','counter-regressed','open-child','host-closure'])
def test_v3_real_prior_phase_evidence_still_required(config,tmp_path,fault):
    env,_=m1_only_config(config,tmp_path)
    ledger=PhaseLedger(tmp_path/'ledger','3'*64,env,plan(config)['pool'],config['source_manifest_sha256'])
    ledger.claim('two',312,plan(config)['config_sha256'])
    report=closure_metadata_fixture(config,env);report['phase_identity']=ledger.identity('two')
    report['cleanup']['actual_common_snapshot']['operation_records']=336
    if fault=='over-cap':report['cleanup']['actual_common_snapshot']['operation_records']=561
    elif fault=='legacy-identity':report['phase_identity']['schema']=2
    elif fault=='counter-regressed':report['cleanup']['actual_common_snapshot']['operation_records']=311
    elif fault=='open-child':report['cleanup']['actual_common_snapshot']['pools'][0]['clients'][0]['children']=[{'open':True}]
    elif fault=='host-closure':report['scope']='CPU-ORCHESTRATION-ONLY'
    if fault:
        with pytest.raises((ValueError,KeyError)):ledger.close('two',report,tmp_path/'report.json')
        assert not (ledger.root/'two.closed.json').exists()
    else:
        ledger.close('two',report,tmp_path/'report.json')
        assert ledger.check('four',336)['worst']==974
        with pytest.raises(ValueError,match='regressed'):ledger.check('four',335)
        ledger.claim('four',336,'c'*64)
        with pytest.raises(ValueError,match='already claimed'):ledger.check('four',336)


def test_v3_duplicate_json_fields_fail_before_release(config,tmp_path):
    from scenario_admission import validate_release
    m1_only_config(config,tmp_path)
    path=Path(config['operation_allowance']['path'])
    path.write_text(path.read_text().replace('"schema": 3','"schema": 2, "schema": 3'))
    config['operation_allowance']['sha256']=digest(path.read_bytes())
    config['device_release']['envelope_sha256']=config['operation_allowance']['sha256']
    with pytest.raises(PermissionError,match='immutable'):validate_release(config)


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


@pytest.mark.parametrize('mismatched_stream', [False, True])
def test_actual_actor_direct_request_passes_complete_record_to_real_output_validator(config,tmp_path,monkeypatch,mismatched_stream):
    """CPU stream fixture exercises the real actor and Run02 validator contract."""
    import ast
    from dataclasses import dataclass
    from types import ModuleType,SimpleNamespace
    from model_actor import ModelActor
    from pypto_serving.config.types import GenerateConfig
    # Execute the actual pure validator definition without the harness's Linux
    # fcntl import. Its source/body is unchanged on Windows and Linux.
    validator_path=RUN06.parent/'run02/m2_baseline.py'
    tree=ast.parse(validator_path.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_output')
    module=ModuleType('m2_baseline')
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(validator_path),'exec'),module.__dict__)
    monkeypatch.setitem(sys.modules,'m2_baseline',module)
    actor=ModelActor(config,0,False,tmp_path/'not-started')
    actor.generation=GenerateConfig(max_new_tokens=2,min_tokens=2,temperature=0.0,seed=0,stream=True)
    actor.coord=None;actor.device=config['devices'][0]
    async def idle():pass
    actor.idle=idle;actor.usage=lambda:{'CPU_fixture_only':True}
    @dataclass
    class Output:
        finished:bool
        finish_reason:str|None
        token_ids:tuple[int,...]
        token_id:int|None
        text:str
        text_delta:str
        completion_tokens:int
        prompt_tokens:int=257
    async def stream(request_id,text,generation):
        actor.schedule.append({'request_id':request_id,'is_prefill':True,'tokens':257})
        yield Output(False,None,(10,),10,'A','A',1)
        yield Output(True,'FINISHED_LENGTH',(10,11),12 if mismatched_stream else 11,'AB','B',2)
    actor.engine=SimpleNamespace(generate_request_id=lambda:'CPU-only-request',add_request=stream,
        tokenizer=SimpleNamespace(encode=lambda text:list(range(257)),decode=lambda ids:'AB'))
    if mismatched_stream:
        with pytest.raises(AssertionError,match='Stream/final token data mismatch: CPU-only-request'):
            asyncio.run(actor.request('CPU-only prompt'))
    else:
        row=asyncio.run(actor.request('CPU-only prompt'))
        assert row['generate_config']['max_new_tokens']==2 and row['prefill_tokens']==257
        assert row['outputs'][-1]['token_ids']==(10,11) and row['loads']==row['stores']==[]


def test_cancelled_inflight_transport_preserves_owner_without_cleanup(config):
    class Cancelled(FakeTransport):
        async def call(self,name,op,**values):
            if name=='shared1' and op=='request':raise asyncio.CancelledError()
            return await super().call(name,op,**values)
    transport=Cancelled()
    with pytest.raises(asyncio.CancelledError):asyncio.run(orchestrate(config,transport))
    assert transport.retained and ('cleanup',) not in transport.events


@pytest.mark.parametrize('fault', [None, 'final', 'done', 'identity', 'usage'])
def test_actual_actor_http_with_actual_serving_api_binds_generation_stream_and_final(config,tmp_path,monkeypatch,fault):
    import httpx
    from dataclasses import asdict
    from types import SimpleNamespace
    from model_actor import ModelActor
    from pypto_serving.config.types import GenerateConfig
    from pypto_serving.serving.engine.async_engine import TokenOutput
    from pypto_serving.serving.server.server import create_serving_app
    actor=ModelActor(config,0,False,tmp_path)
    actor.coord=None;actor.device=config['devices'][0]
    actor.generation=GenerateConfig(max_new_tokens=2,min_tokens=2,temperature=0.0,seed=0,stream=True,ignore_eos=True)
    actor.engine_config=SimpleNamespace(model_id='CPU-model')
    actual=[]
    async def idle():pass
    actor.idle=idle;actor.usage=lambda:{'CPU_fixture_only':True}
    async def stream(request_id,text,generation):
        actual.append(asdict(generation))
        actor.schedule.append({'request_id':request_id,'is_prefill':True,'tokens':257})
        yield TokenOutput(token_id=10,text='A',token_ids=(10,),prompt_tokens=257,completion_tokens=1)
        yield TokenOutput(token_id=11,text='AB',token_ids=(10,11),finished=True,
                          finish_reason='FINISHED_LENGTH',prompt_tokens=257,completion_tokens=2)
    actor.engine=SimpleNamespace(add_request=stream,tokenizer=SimpleNamespace(encode=lambda text:list(range(257))))
    app=create_serving_app(actor.engine,'CPU-model',actor.generation)
    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self,request):
            response=await httpx.ASGITransport(app=app).handle_async_request(request)
            data=await response.aread()
            if fault=='done':data=data.replace(b'data: [DONE]\n\n',b'')
            if fault=='final':data=data.replace(b'"finish_reason": "length"',b'"finish_reason": null')
            if fault=='identity':
                parts=data.split(b'\n\n');row=json.loads(parts[0][6:]);row['id']='another-request'
                parts[0]=b'data: '+json.dumps(row).encode();data=b'\n\n'.join(parts)
            if fault=='usage':data=data.replace(b'"completion_tokens": 2',b'"completion_tokens": 1')
            return httpx.Response(response.status_code,headers=response.headers,content=data)
    client=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kwargs:client(transport=Transport(),**kwargs))
    if fault:
        with pytest.raises(ValueError,match='HTTP'):asyncio.run(actor.request('CPU prompt','http'))
    else:
        row=asyncio.run(actor.request('CPU prompt','http'))
        assert row['request_id'].startswith('cmpl-') and row['body']['usage']['completion_tokens']==2
        assert actual==[row['generate_config']]==[asdict(actor.generation)]
