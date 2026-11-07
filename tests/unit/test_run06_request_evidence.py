# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-request CPU metadata fixtures, never native or SDK success evidence."""
import asyncio
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'examples/run06'),str(ROOT/'examples/run04')]
from request_evidence import correlate_request
from scenario_runtime import ProcessTransport


def fixture(operation):
    key={'namespace_digest':'a'*64,'block_hash':'b'*64}
    terminal={'event':'transfer_terminal','job':'CPU-transfer','worker_epoch':'CPU-worker',
        'operation':operation,'entry_refs':[{'entry_id':'CPU-entry','slot_generation':1}],
        'status':'SUCCEEDED','native_completed':True,'stopped':True,'bytes':20971520,'error':None,
        'pool_receipt':{'operation':operation,'content':{'nbytes':20971520,
            'content_id':'qwen3-prefix-v1:'+'a'*64+':'+'b'*64}},
        'pages':[4],'page_lease_id':'CPU-page','delivery_sequence':1,'time_ns':50}
    request={'job_id':'CPU-transfer','worker_epoch':'CPU-worker','operation':operation,
        'entry_refs':terminal['entry_refs'],'page_ids':[4],'page_lease_id':'CPU-page','delivery_sequence':1,
        'entry_keys':[key],'retire_through':0,'source_step_id':2 if operation=='STORE' else None}
    event={'job_id':'CPU-transfer','worker_epoch':'CPU-worker','operation':operation,
        'status':'SUCCEEDED','entry_refs':terminal['entry_refs'],'transferred_bytes':20971520,
        'native_completed':True,'stopped':True,'error':None,'delivery_sequence':1,'pool_receipt':terminal['pool_receipt']}
    producer={'kind':'transfer','identity':'CPU-transfer','status':'SUCCEEDED','event':event,
        'error':None,'delivery_sequence':1,'time_ns':40,'request':request}
    row={'coordinator_trace':[terminal],'loads':[terminal] if operation=='LOAD' else [],
         'stores':[terminal] if operation=='STORE' else []}
    return row,producer,{'namespace_digest':'a'*64}


def test_actual_transport_native_off_request_does_not_require_run04_rounds(tmp_path):
    transport=ProcessTransport.__new__(ProcessTransport);transport.root=tmp_path
    (tmp_path/'logs').mkdir();(tmp_path/'logs/cold0.log').write_text('ordinary Qwen computation log; no DDR transfer\n')
    transport.started={'cold0':{'startup':None}}
    row={'coordinator_trace':[],'loads':[],'stores':[]}
    async def rpc(*args):return copy.deepcopy(row)
    transport.rpc=rpc;transport.models={'cold0':object()}
    result=asyncio.run(transport.call('cold0','request',text='CPU-only'))
    assert result=={**row,'native_log_bindings':[]}


@pytest.mark.parametrize('operation',['STORE','LOAD'])
@pytest.mark.parametrize('fault',[None,'missing-submit','failed-native','duplicate-complete','late-cleanup',
    'wrong-key','wrong-page-lease','wrong-delivery','missing-receipt','duplicate-receipt','wrong-namespace',
    'native-off-transfer','mismatched-summary','duplicate-terminal'])
def test_single_request_binds_actual_raw_terminal_and_run_without_fabricated_rounds(operation,fault):
    row,producer,startup=fixture(operation)
    if fault=='wrong-key':producer['request']['entry_keys'][0]['block_hash']='c'*64
    elif fault=='wrong-page-lease':producer['request']['page_lease_id']='wrong'
    elif fault=='wrong-delivery':producer['request']['delivery_sequence']=2
    elif fault=='wrong-namespace':startup['namespace_digest']='c'*64
    elif fault=='native-off-transfer':startup=None
    elif fault=='mismatched-summary':row['loads' if operation=='LOAD' else 'stores']=[]
    elif fault=='duplicate-terminal':row['coordinator_trace'].append(copy.deepcopy(row['coordinator_trace'][0]))
    terminal=row['coordinator_trace'][0]
    lines=['native_ddr_coordinator '+json.dumps(terminal),
           'native_ddr_trace '+json.dumps({'event':'accepted','job':'CPU-transfer','native_run_id':7,'time_ns':10}),
           'native_ddr_trace '+json.dumps({'event':'native_result_and_cleanup','job':'CPU-transfer',
                                          'time_ns':60 if fault=='late-cleanup' else 30})]
    if fault!='missing-submit':lines.append('name=node.submit ts=15 run_id=7 ')
    lines.append('name=node.complete ts=25 run_id=7 outcome='+('1' if fault=='failed-native' else '0'))
    if fault=='duplicate-complete':lines.append(lines[-1])
    if fault!='missing-receipt':lines.append('native_ddr_receipt '+json.dumps(producer))
    if fault=='duplicate-receipt':lines.append(lines[-1])
    if fault:
        with pytest.raises(ValueError):correlate_request(row,startup,'\n'.join(lines))
    else:
        result=correlate_request(row,startup,'\n'.join(lines))
        assert result==[{'job':'CPU-transfer','operation':operation,'native_run_id':7,
                        'submit_ns':15,'complete_ns':25,'cleanup_ns':30,'terminal_ns':50}]
