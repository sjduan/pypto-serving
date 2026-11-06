# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One explicitly owned metadata directory; never a UF pool owner/admin."""
import argparse
import json
import os
from pathlib import Path

from pypto_serving.serving.native_ddr.shared_directory import DirectoryServer, SharedPrefixDirectory
from pypto_serving.serving.native_ddr.shared_pool import sdk_authority


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'endpoint', 'epoch', 'bindings', 'pool-endpoint'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    root, endpoint = Path(args.root).resolve(), Path(args.endpoint).resolve()
    if endpoint.parent != root.parent or len(str(endpoint).encode()) >= 108:
        raise ValueError('directory socket must be a bounded path in its own task root')
    root.mkdir()
    directory = SharedPrefixDirectory(args.epoch, sdk_authority(args.pool_endpoint, args.bindings))
    with DirectoryServer(str(endpoint), directory) as server:
        server.timeout = .2
        save(root / 'ready.json', {'pid': os.getpid(), 'epoch': args.epoch, 'endpoint': str(endpoint),
             'description': directory.dispatch({'epoch': args.epoch, 'op': 'describe'}), 'UF_admin': False})
        while not (root / 'stop.json').exists():
            server.handle_request()
        stop = json.loads((root / 'stop.json').read_text())
        if stop != {'epoch': args.epoch} or directory.intents:
            save(root / 'retained-metadata.json', {'epoch': args.epoch, 'intent_count': len(directory.intents)})
            raise RuntimeError('metadata intents remain; no false clean directory receipt')
    endpoint.unlink()
    save(root / 'closed.json', {'pid': os.getpid(), 'epoch': args.epoch, 'socket_absent': not endpoint.exists(),
                              'pool_not_destroyed': True})


if __name__ == '__main__':
    main()
