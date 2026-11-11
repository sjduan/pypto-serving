# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bind per-request transfer terminals without inventing Run04 paired rounds."""
import re

from qwen_evidence import bind_round_event,log_records,require


def correlate_request(row,startup,raw_log):
    trace=row['coordinator_trace']
    terminals=[event for event in trace if event['event']=='transfer_terminal']
    require(row['loads']==[e for e in terminals if e['operation']=='LOAD']
            and row['stores']==[e for e in terminals if e['operation']=='STORE'],
            'request transfer summaries differ from original coordinator events')
    require(len({e['job'] for e in terminals})==len(terminals),'duplicate request transfer terminal')
    require(not terminals or startup is not None,'native-off request unexpectedly transferred DDR')
    report={'coordinator_trace':trace,'startup':startup}
    events=log_records(raw_log,'native_ddr_trace ')
    bindings=[]
    for terminal in terminals:
        require(terminal['operation'] in ('STORE','LOAD') and terminal['status']=='SUCCEEDED'
                and terminal['stopped'] is True and terminal['native_completed'] is True,
                'request did not complete a successful stopped transfer')
        bind_round_event(terminal,report,raw_log)
        accepted=[e for e in events if e['event']=='accepted' and e['job']==terminal['job']]
        cleaned=[e for e in events if e['event']=='native_result_and_cleanup' and e['job']==terminal['job']]
        require(len(accepted)==len(cleaned)==1,'missing unique actual native accept/result-cleanup trace')
        run=accepted[0]['native_run_id'];require(type(run) is int,'actual native run identity missing')
        def matching(name):
            return [line for line in raw_log.splitlines() if 'name='+name+' ' in line
                    and re.search(r'\brun_id='+str(run)+r'\b',line)]
        submits,completes=matching('node.submit'),matching('node.complete')
        require(len(submits)==len(completes)==1 and re.search(r'\boutcome=0\b',completes[0]),
                'actual native submit/successful completion not uniquely correlated')
        begin=int(re.search(r'\bts=(\d+)',submits[0])[1]);end=int(re.search(r'\bts=(\d+)',completes[0])[1])
        require(begin<=end<=cleaned[0]['time_ns']<=terminal['time_ns'],
                'native/SDK/parent terminal order differs')
        bindings.append({'job':terminal['job'],'operation':terminal['operation'],'native_run_id':run,
                         'submit_ns':begin,'complete_ns':end,'cleanup_ns':cleaned[0]['time_ns'],
                         'terminal_ns':terminal['time_ns']})
    return bindings
