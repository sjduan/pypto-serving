# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail-closed future device admission. Current host-only runs never call observe()."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scenario_contract import MIN_FREE_HBM_MIB, digest, validate
from mutation_envelope import PhaseLedger, check_sessions, sealed, validate_envelope


def validate_release(config):
    document,_,_,_=validate(config)
    release=config['device_release'];allowance=config['operation_allowance']
    if (type(release) is not dict or release.get('device_stage_allowed') is not True
            or type(release.get('schema')) is not int or release['schema']!=1
            or release.get('source_manifest_sha256')!=config['source_manifest_sha256']
            or release.get('manifest_sha256')!=config['manifest_sha256']):
        raise PermissionError('host-only: explicit device-stage identity/budget release absent')
    try:
        envelope=sealed(allowance)
        source=sealed({'path':config['source_manifest'],'sha256':config['source_manifest_sha256']})
        validate_envelope(envelope,document['pool'],config['manifest_sha256'],source['serving'])
        if release.get('envelope_sha256')!=allowance['sha256']:
            raise ValueError('device release must pin the same immutable envelope')
    except (ValueError,KeyError,TypeError,OSError) as error:
        raise PermissionError('unknown/drifted immutable shared mutation envelope') from error
    return envelope


def phase_ledger(config,envelope=None):
    return PhaseLedger('/workspace/uflow/runs/phasef-parallel-20261106/m1/campaigns',
        config['operation_allowance']['sha256'],envelope or validate_release(config),
        validate(config)[0]['pool'],config['source_manifest_sha256'])


def validate_admission(config, observed):
    document,_,manifest,budget=validate(config)
    envelope=validate_release(config)
    count=observed['operation_records']
    try:
        phase_ledger(config,envelope).check('two' if config['replicas']==2 else 'four',count)
    except (ValueError,KeyError,TypeError,OSError) as error:
        raise PermissionError('shared phase/counter/real prior closure gate refused') from error
    if observed.get('own_active_obligations') is not False:
        raise PermissionError('own prior native/client/child/view/range obligations not closed')
    if observed['host']!=manifest['host'] or observed['unknown_or_frozen'] or observed['M1_retired']:
        raise PermissionError('actual host/domain/line authority differs')
    if (observed['budget_bytes'],observed['charged_bytes'],observed['creation_reserved_bytes'])!=(134217728,134217728,0):
        raise PermissionError('single original common physical budget differs')
    if observed['pool_descriptor']!=document['pool']:
        raise PermissionError('actual canonical pool/provider incarnation differs')
    selected={d['physical']:d for d in config['devices']}
    for physical in selected:
        actual=observed['devices'][str(physical)];candidate=selected[physical]
        if (any(actual[k]!=candidate[k] for k in ('physical','board','chip','bdf')) or actual['health']!='OK'
                or actual['hbm_free_mib']<MIN_FREE_HBM_MIB or actual['fd_users'] or actual['fd_denied']
                or actual['driver_processes_present']):
            raise PermissionError('selected physical Qwen domain is occupied/unhealthy/under-budget')
    if any(r['physical_device'] in selected or r['execution_domain'] in {d['bdf'] for d in selected.values()}
           for r in observed['other_reservations']):
        raise PermissionError('actual O1/foreign physical execution domain is reserved')
    return budget


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def verify_source_closure(config):
    for name in ('source_manifest','native_env','host_env'):
        if not str(Path(config[name])).startswith('/workspace/uflow/runs/phasef-parallel-20261106/m1/'):
            raise PermissionError('source/environment path escaped owned B M1 root')
        if file_hash(config[name])!=config[name+'_sha256']:raise PermissionError('source/environment seal differs')
    path=Path(config['source_manifest']);manifest=json.loads(path.read_text());root=path.parent/'src/serving'
    for group,files in manifest['groups'].items():
        if group not in ('serving','lib'):raise ValueError('unexpected source closure group')
        base=root if group=='serving' else root/'pypto-lib'
        for name,row in files.items():
            p=base/name
            if not p.resolve().is_relative_to(base.resolve()) and row['mode']!='120000':
                raise ValueError('source escaped its bounded deployed tree')
            raw=os.readlink(p).encode() if row['mode']=='120000' else p.read_bytes()
            if len(raw)!=row['bytes'] or digest(raw)!=row['sha256']:raise PermissionError('actual deployed source differs')
    for name,expected in config['model_assets'].items():
        if file_hash(Path(config['model'])/name)!=expected:raise PermissionError('full model asset hash differs')
    for path,revision in (('/workspace/pypto','8007f1146a800d679301c5ea14bb644b863da6bf'),
                          ('/workspace/pypto/runtime','508becb93e182a3035a14e3f9226929dbe994346')):
        actual=subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip()
        if actual!=revision:raise PermissionError('frozen runtime revision differs')
        subprocess.run(['git','-C',path,'diff','--exit-code','HEAD','--','python'],check=True,capture_output=True)
    runtime_files=config['device_release'].get('runtime_files')
    required={'pypto_core','task_interface','native_host','native_aicpu','native_aicore','ptoas','shim'}
    if type(runtime_files) is not dict or set(runtime_files)!=required:raise PermissionError('binary closure incomplete')
    for row in runtime_files.values():
        if set(row)!={'path','sha256'} or file_hash(row['path'])!=row['sha256']:
            raise PermissionError('frozen actual binary identity differs')
    return manifest


def observe(config, root):
    # No model/runtime imports. All commands are fixed, read-only observations.
    _,_,manifest,_=validate(config)
    validate_release(config)  # Pure gate precedes device queries, helper RPC, or any mutation.
    if os.environ.get('LD_PRELOAD') or any(os.environ.get(k) for k in ('ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES')):
        raise PermissionError('controller has host fence or uncalibrated logical device mask')
    sys.path[:0]=[manifest['helper_root'],str(Path(__file__).resolve().parents[1]/'run02')]
    from session_protocol import request,pool_record,check_retired
    from r_guard import snapshot
    state=request(manifest,'describe');s=state['snapshot'];pool=pool_record(s,manifest['pool_descriptor'])
    # The shared service's complete M1 registration history is authoritative;
    # checking just this process's in-memory clients would miss an older attempt.
    check_retired(pool,state['lines']['M1']['clients'])
    check_sessions(s,manifest['pool_descriptor'],state['lines']['M1']['clients'])
    host={'hostname':os.uname().nodename,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'pid_namespace':os.readlink('/proc/self/ns/pid'),'mount_namespace':os.readlink('/proc/self/ns/mnt')}
    mapping=subprocess.check_output(['npu-smi','info','-m'],text=True,timeout=15)
    selected={d['physical']:d for d in config['devices']};devices={}
    for physical in selected:
        fresh=snapshot(root,'admit-'+str(physical)+'-'+str(time.monotonic_ns()),physical)
        status=fresh['status']
        pattern=rf'\|\s*{status["chip"]}\s+{physical}\s*\|\s*([0-9A-Fa-f:.]+)\s*\|'
        found=re.search(pattern,fresh['npu'])
        if not found:raise PermissionError('actual BDF unavailable')
        mapped=any(re.match(rf'\s*{status["board"]}\s+{status["chip"]}\s+{physical}\s+{physical}\s+',line)
                   for line in mapping.splitlines())
        if not mapped:raise PermissionError('physical/logical/board/chip mapping changed')
        processes=False;table=False
        for line in fresh['npu'].splitlines():
            if 'Process id' in line:table=True
            if table and re.match(rf'\|\s*{status["board"]}\s+{status["chip"]}\s*\|\s*\d+',line):processes=True
        devices[str(physical)]={'physical':physical,'board':status['board'],'chip':status['chip'],'bdf':found[1],
            'health':status['health'],'hbm_free_mib':status['hbm_total_mib']-status['hbm_used_mib'],
            'fd_users':fresh['fds']['users'],'fd_denied':fresh['fds']['denied'],'driver_processes_present':processes}
    reservations=json.loads(Path(manifest['coordination']['physical_reservations_file']).read_text())['reservations']
    return {'host':host,'devices':devices,'operation_records':s['operation_records'],'pool_descriptor':manifest['pool_descriptor'],
        'budget_bytes':s['budget_bytes'],'charged_bytes':s['charged_bytes'],'creation_reserved_bytes':s['creation_reserved_bytes'],
        'unknown_or_frozen':pool['frozen'],'M1_retired':state['lines']['M1']['retired'],
        'own_active_obligations':any(r['line']=='M1' for r in reservations),
        'other_reservations':[r for r in reservations if r['line']!='M1'],'raw_common_state':state}
