# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual owned-birth shared-memory inventory and driver-table closure guards."""
import os
from pathlib import Path
import re


def shared_names(births, common=None):
    from pypto_serving.serving.native_ddr.managed_process import process_birth
    found = {}
    for pid, birth in births.items():
        if process_birth(int(pid)) != birth:
            raise RuntimeError('owned process birth changed before shared-memory inventory')
        for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) != 6 or not fields[5].startswith('/dev/shm/'):
                continue
            path = fields[5].removesuffix(' (deleted)')
            if Path(path).parent != Path('/dev/shm'):
                raise RuntimeError('unexpected shared-memory name shape')
            major, minor = (int(n, 16) for n in fields[3].split(':'))
            device, inode = os.makedev(major, minor), int(fields[4])
            if common is not None and (device, inode) == (common.device, common.inode):
                continue  # The shared UF backing remains owned by the common provider.
            found[(device, inode, path)] = {'device': device, 'inode': inode, 'path': path}
        for fd in Path(f'/proc/{pid}/fd').iterdir():
            try:
                path = os.readlink(fd).removesuffix(' (deleted)')
                if not path.startswith('/dev/shm/'):
                    continue
                info = fd.stat()
            except FileNotFoundError:
                continue
            if Path(path).parent != Path('/dev/shm'):
                raise RuntimeError('unexpected shared-memory FD name shape')
            device, inode = info.st_dev, info.st_ino
            if common is None or (device, inode) != (common.device, common.inode):
                found[(device, inode, path)] = {'device': device, 'inode': inode, 'path': path}
        if process_birth(int(pid)) != birth:
            raise RuntimeError('owned process birth changed during shared-memory inventory')
    return list(found.values())


def check_shared_names_gone(resources):
    for row in resources:
        path = Path(row['path'])
        if path.parent != Path('/dev/shm'):
            raise ValueError('shared-memory evidence escaped bounded name directory')
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if (info.st_dev, info.st_ino) == (row['device'], row['inode']):
            raise RuntimeError('original private shared-memory name remains')


def check_device_closed(snapshot):
    status = snapshot['status']
    table = False
    for line in snapshot['npu'].splitlines():
        if 'Process id' in line:
            table = True
        if table and re.match(rf'\|\s*{status["board"]}\s+{status["chip"]}\s*\|\s*\d+', line):
            raise RuntimeError('original selected chip still has a driver process')
    if not table or snapshot['fds']['users'] or snapshot['fds']['denied'] or status['health'] != 'OK':
        raise RuntimeError('actual device FD/driver/health closure unproved')
