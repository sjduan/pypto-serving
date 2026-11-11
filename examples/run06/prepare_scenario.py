# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate sealed local-plan and B-runtime inputs; never grants execution or imports runtime."""
import argparse
import json
from pathlib import Path

from scenario_contract import digest,plan,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicas',type=int,choices=(2,4),required=True)
    parser.add_argument('--root',required=True,help='New future evidence root under the owned B m1 path.')
    args=parser.parse_args();d=args.deployment.resolve();out=args.output.resolve();out.mkdir()
    review=json.loads((d/'package-review.json').read_text());remote=review['remote_root']
    mf=(d/'common-manifest.json').read_bytes()
    # A new local copy preserves exact original bytes; no public manifest rewrite.
    with (out/'public-manifest.json').open('xb') as stream:stream.write(mf)
    document=json.loads((d/'shared-slots.json').read_text());document['session_manifest']=str(out/'public-manifest.json')
    save(out/'local-bindings.json',document)
    raw_assets=json.loads(args.assets.read_text());assets=raw_assets.get('assets',raw_assets)
    placements=[{'physical':12,'board':6,'chip':0,'bdf':'0000:85:00.0','port':18920},
        {'physical':14,'board':7,'chip':0,'bdf':'0000:81:00.0','port':18921},
        {'physical':8,'board':4,'chip':0,'bdf':'0000:8D:00.0','port':18922},
        {'physical':9,'board':4,'chip':1,'bdf':'0000:8F:00.0','port':18923}]
    config={'schema':1,'bindings':str(out/'local-bindings.json'),'manifest_sha256':digest(mf),
        'root':args.root,'replicas':args.replicas,'devices':placements[:args.replicas],
        'model':'/data/models/Qwen/Qwen3-14B','model_assets':assets,
        'source_manifest':remote+'/source-manifest.json','source_manifest_sha256':digest((d/'source-manifest.json').read_bytes()),
        'native_env':remote+'/native-env.sh','native_env_sha256':digest((d/'native-env.sh').read_bytes()),
        'host_env':remote+'/host-env.sh','host_env_sha256':digest((d/'host-env.sh').read_bytes()),
        'operation_allowance':None,'device_release':None,'router':{'port':18929} if args.replicas==4 else None}
    prepared=plan(config);save(out/'local-plan-config.json',config);save(out/'plan.json',prepared)
    config['bindings']=remote+'/shared-slots.json';save(out/'runtime-config.NOT-RELEASED.json',config)
    save(out/'preparation.json',{'status':'NOT-RUN','source':review['serving'],
        'candidates_are_not_reservations':True,'runtime_release_or_operation_grant_written':False,
        'local_plan_scope':'recorded exact public metadata only; no B state/device/source/model scan'})


if __name__=='__main__':main()
