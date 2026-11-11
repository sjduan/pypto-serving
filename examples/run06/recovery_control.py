# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preinstalled same-UID, original-controller, once-only stop-owned endpoint.

Recovery is separate from campaign success. No incoming code, PID, argv or new
claim is ever used to select resources or invoke an operation.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import socket
import stat
import struct
import traceback

from scenario_contract import digest, save
from scenario_wire import _unique


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class RecoveryControl:
    def __init__(self, path, binding, stop_owned):
        self.path = Path(path)
        self.binding = json.loads(canonical(binding))
        self.stop_owned = stop_owned
        self.server = self.socket_identity = self.identity = self.operation_id = None
        self.stop_task = None
        self.failures = []
        self.completed = asyncio.Event()

    async def install(self):
        from pypto_serving.serving.native_ddr.managed_process import process_birth
        if (len(str(self.path).encode()) >= 108 or self.path.exists()
                or self.binding['uid'] != os.getuid() or self.binding['controller_pid'] != os.getpid()
                or self.binding['controller_birth'] != process_birth(os.getpid())):
            raise PermissionError('recovery endpoint original owner/path differs')
        self.server = await asyncio.start_unix_server(self.handle, path=str(self.path), limit=8193)
        os.chmod(self.path, 0o600)
        info = self.path.lstat()
        self.socket_identity = (info.st_dev, info.st_ino, info.st_uid)
        save(self.path.with_suffix('.installed.json'), {'binding': self.binding, 'socket': str(self.path),
             'active': False, 'operation': 'stop-owned', 'installed_before_admission': True})

    def activate(self, claim_path, claim_identity, config_sha256):
        """Bind the authoritative original once-claim after entering retain."""
        path = Path(claim_path)
        if str(path) != self.binding['claim_path']:
            raise PermissionError('recovery claim path differs')
        raw = path.read_bytes()
        claim = json.loads(raw, object_pairs_hook=_unique)
        if (canonical(claim['identity']) != canonical(claim_identity)
                or claim['config_sha256'] != config_sha256 or config_sha256 != self.binding['config_sha256']):
            raise PermissionError('recovery original claim/source/configuration differs')
        identity = {**self.binding, 'claim_sha256': digest(raw), 'claim_identity': claim_identity}
        operation_id = digest(canonical(identity).encode())
        save(self.path.with_suffix('.retained.json'), {'identity': identity, 'op': 'stop-owned',
             'operation_id': operation_id, 'socket': str(self.path), 'recovery_is_campaign_success': False})
        self.identity, self.operation_id = identity, operation_id

    async def _stop_once(self):
        try:
            result = await self.stop_owned()
            receipt = {'status': 'closed', 'identity': self.identity, 'operation_id': self.operation_id,
                       'cleanup': result, 'campaign_success': False, 'four_eligible': False}
        except BaseException as error:
            self.failures.append(error)
            receipt = {'status': 'retained', 'identity': self.identity, 'operation_id': self.operation_id,
                       'error': repr(error), 'traceback': ''.join(traceback.format_exception(error)),
                       'campaign_success': False, 'four_eligible': False}
        save(self.path.with_suffix('.result.json'), receipt)
        return receipt

    async def handle(self, reader, writer):
        receipt = None
        try:
            from pypto_serving.serving.native_ddr.managed_process import process_birth
            peer = writer.get_extra_info('socket').getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            _, uid, _ = struct.unpack('3i', peer)
            if uid != self.binding['uid'] or process_birth(os.getpid()) != self.binding['controller_birth']:
                raise PermissionError('recovery peer UID/original controller birth differs')
            raw = await asyncio.wait_for(reader.readline(), 5)
            if not raw.endswith(b'\n') or len(raw) > 8192:
                raise ValueError('bounded complete recovery frame required')
            request = json.loads(raw, object_pairs_hook=_unique)
            expected = {'op': 'stop-owned', 'identity': self.identity, 'operation_id': self.operation_id}
            if self.identity is None or canonical(request) != canonical(expected):
                raise PermissionError('inactive recovery or exact original recovery identity differs')
            if digest(Path(self.binding['claim_path']).read_bytes()) != self.identity['claim_sha256']:
                raise PermissionError('original retained claim changed; no recovery operation')
            # A duplicate joins the sole owner; a lost requester cannot cancel it.
            if self.stop_task is None:
                self.stop_task = asyncio.create_task(self._stop_once())
            receipt = await asyncio.shield(self.stop_task)
            answer = {'status': receipt['status'], 'operation_id': self.operation_id,
                      'receipt_path': str(self.path.with_suffix('.result.json'))}
        except Exception as error:
            answer = {'status': 'rejected', 'error': repr(error)[:1024]}
        try:
            writer.write((json.dumps(answer) + '\n').encode())
            await writer.drain()
        except ConnectionError:
            pass  # The shielded original cleanup owner and receipt still exist.
        finally:
            writer.close()
            try:
                with suppress(ConnectionError):await writer.wait_closed()
            finally:
                if receipt is not None and receipt['status'] == 'closed':
                    self.completed.set()

    async def close(self):
        if self.server is None:
            return
        self.server.close()
        await self.server.wait_closed()
        info = self.path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or (info.st_dev, info.st_ino, info.st_uid) != self.socket_identity:
            raise PermissionError('recovery endpoint replaced; no unlink')
        self.path.unlink()
        self.server = None
