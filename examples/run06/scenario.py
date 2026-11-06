# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run06 pure plan, or explicitly released future independent-model scenario."""
import argparse
import asyncio
import json
from pathlib import Path

from scenario_contract import plan,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text());prepared=plan(config)
    if args.mode=='plan':
        save(args.output,prepared)
        return 0
    # Runtime transport import is after complete pure metadata validation. With
    # absent release/headroom, execution fails before any device query or SDK mutation.
    if config['device_release'] is None or config['operation_allowance'] is None:
        raise PermissionError('host-only: device stage/shared journal headroom not released')
    from scenario_controller import orchestrate
    from scenario_runtime import ProcessTransport
    result=asyncio.run(orchestrate(config,ProcessTransport(config,args.config)))
    save(args.output,result)
    return 0


if __name__=='__main__':raise SystemExit(main())
