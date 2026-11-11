# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-fenced ordinary SDK observer pins two real ranges during genuine source overwrite."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from dataclasses import asdict

from scenario_contract import save, validate


def main():
    config=json.loads(Path(sys.argv[1]).read_text());document,blocks,manifest,_=validate(config)
    root=Path(sys.argv[2]);root.mkdir()
    from pypto_serving.serving.native_ddr.managed_pool_sdk import register_managed_pool, SDK_SHA256
    from pypto_serving.serving.native_ddr.managed_pool import CommittedRange, PoolBlock
    from pypto_serving.serving.native_ddr.types import NativeDdrPrefixConfig
    p=document['pool']
    access=register_managed_pool(NativeDdrPrefixConfig(True,4*20971520,pool_endpoint=manifest['endpoint'],
        pool_id=p['pool_id'],pool_epoch=p['pool_epoch'],sdk_sha256=SDK_SHA256,
        shared_directory_endpoint=config['root']+'/d.sock',shared_directory_epoch='m1-'+Path(config['root']).name,
        shared_slots_path=config['bindings']))
    held=[]
    save(root/'registered.json',{'identity':asdict(access.identity),'pid':os.getpid()})
    try:
        for line in sys.stdin:
            if len(line)>65536:raise ValueError('oversized observer metadata command')
            row=json.loads(line)
            if row['op']=='pin' and not held:
                contents=row['contents']
                if len(contents)!=2:raise ValueError('exactly two original prefix ranges required')
                for raw in contents:
                    content=CommittedRange(**{**raw,'block':PoolBlock(**raw['block'])})
                    if content.block not in blocks:raise ValueError('observer range is outside exact M1 partition')
                    held.append(access.acquire(content.block,'r',committed=content,reject_stale=True))
                value={'pid':os.getpid(),'held_R_grants':[asdict(x) for x in held],'payload_bytes_read':0}
            elif row['op']=='release' and len(held)==2:
                value={'actual_release_ACKs':[asdict(access.complete_read(x,uuid.uuid4().hex)) for x in held]}
                held.clear();access.close();value['SDK_client_unmapped']=access.closed
            else:raise ValueError('observer phase/order mismatch')
            path=root/('pins.json' if row['op']=='pin' else 'closed.json');save(path,value)
            print(json.dumps({'status':'ok','result_path':str(path)}),flush=True)
            if row['op']=='release':return
        raise RuntimeError('observer channel lost; retain original SDK owner')
    except BaseException:
        save(root/'retained-failure.json',{'pid':os.getpid(),'error':traceback.format_exc(),
            'retained_lease_ids':list(access.leases),'unknown':list(access.unknown)})
        while True:time.sleep(30)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        from pypto_serving.serving.native_ddr import managed_pool_sdk
        if not managed_pool_sdk._STARTUP_OBLIGATIONS:raise
        root=Path(sys.argv[2]);root.mkdir(exist_ok=True,parents=True)
        save(root/'retained-startup.json',{'pid':os.getpid(),'error':repr(error),
             'original_startup_operation_ids':list(managed_pool_sdk._STARTUP_OBLIGATIONS)})
        while True:time.sleep(30)
