# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preinstalled, main-thread-only fixed close action for a retained failed pool client."""
import json
import os
from pathlib import Path
import signal
import sys
import traceback
import uuid

from .managed_process import descendants, detach_closed_children, process_birth, stop_resource_trackers


def physical_holders(pid, identity):
    base = Path('/proc', str(pid))
    maps, fds = [], []
    for line in (base / 'maps').read_text().splitlines():
        fields = line.split(None, 5)
        major, minor = (int(x, 16) for x in fields[3].split(':'))
        if (os.makedev(major, minor), int(fields[4])) == (identity.device, identity.inode):
            maps.append(line)
    for fd in (base / 'fd').iterdir():
        try:
            info = fd.stat()
        except FileNotFoundError:
            continue
        if (info.st_dev, info.st_ino) == (identity.device, identity.inode):
            fds.append(int(fd.name))
    return {'pid': pid, 'maps': maps, 'fds': fds}


class CooperativePoolClose:
    """Signals carry no code or arguments; close is accepted only after a failure pause."""

    def __init__(self, evidence):
        if os.name != 'posix' or signal.getsignal(signal.SIGUSR1) != signal.SIG_DFL:
            raise RuntimeError('fixed pool recovery requires the unclaimed Linux main-thread SIGUSR1')
        self.root = Path(evidence)
        self.root.mkdir(parents=True, exist_ok=False)
        self.pid, self.birth = os.getpid(), process_birth(os.getpid())
        self.access = self.slots = self.worker = self.error = None
        self.frames, self.children = [], {}
        self.paused = self.busy = False
        self.stage, self.operation, self.attempt = 'pre-registration', None, 0
        self.unregister_id, self.unmap_id = uuid.uuid4().hex, uuid.uuid4().hex
        self.unregister_done = self.ack_done = False
        signal.signal(signal.SIGUSR1, self._request)
        self._save('installed-before-registration.json', {'signal': 'SIGUSR1', 'accepts_code_or_arguments': False})

    def _save(self, name, value):
        with (self.root / name).open('x') as stream:
            json.dump({'pid': self.pid, 'birth': self.birth, **value}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

    def bind_access(self, access):
        if self.access is not None or self.paused:
            raise RuntimeError('recovery access identity is immutable')
        self.access = access
        access.cooperative_close = self

    def restore_after_normal_close(self):
        if self.access is None or not self.access.closed or self.paused:
            raise RuntimeError('normal recovery-channel retirement needs actual access close')
        if signal.getsignal(signal.SIGUSR1) == self._request:
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)

    def retain_failure(self, error):
        """Called by the outer exception boundary after native/consumer frames unwind."""
        self.error, self.paused = error, True
        active = set()
        for frame in sys._current_frames().values():
            while frame is not None:
                active.add(id(frame))
                frame = frame.f_back
        allowed = Path(__file__).resolve().parents[3]
        tb = error.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            if id(frame) not in active and Path(frame.f_code.co_filename).resolve().is_relative_to(allowed):
                self.frames.append(frame)
            tb = tb.tb_next
        self.children = descendants()
        self._save('retained-failure.json', {'error': repr(error), 'traceback': ''.join(traceback.format_exception(error)),
            'fixed_unregister_operation': self.unregister_id, 'fixed_unmap_operation': self.unmap_id,
            'children': self.children, 'static_close_channel_ready': True})
        while True:
            try:
                signal.pause()
            except KeyboardInterrupt:
                continue

    def _release_unwound_exports(self):
        for frame in self.frames:
            frame.clear()
        self.frames = []
        self.error.__traceback__ = None
        if self.slots is not None:
            self.slots.pool_tensor = self.slots.page_ids = None

    def _close(self):
        access, slots, owner = self.access, self.slots, self.worker
        if access is None:
            raise RuntimeError('registration outcome needs original SDK reconciliation; no access object')
        client = access.port.client
        self.stage = 'native-owner-close'
        if owner is not None:
            owner.close()
            if not owner._close_complete or owner._active_dispatch_handles or owner._inherited_host_tensors:
                raise RuntimeError('native close is not complete')
        remaining = descendants()
        helpers = {} if slots is None else slots.child_helpers
        if any(pid not in helpers or helpers[pid]['birth'] != born for pid, born in remaining.items()):
            raise RuntimeError('unproven child remains after actual native close')
        if slots is not None:
            detach_closed_children(access, slots.child_registrations, slots.child_baseline, helpers=helpers)
        if (access.leases or access.children or access.port._leases or client._children
                or any(state != 'Closed' for state in client._leases.values())
                or slots is not None and (slots.leases or slots.busy is not None)):
            raise RuntimeError('active or ambiguous range/child obligations require reconciliation')
        self.stage = 'release-unwound-exports'
        self._release_unwound_exports()
        self.stage, self.operation = 'begin-unregister', self.unregister_id
        if not self.unregister_done:
            client.begin_unregister(operation_id=self.unregister_id)
            self.unregister_done = True
        self.operation, self.stage = None, 'actual-mmap-FD-close'
        access.buffer.release()
        client.view.close()
        proof = physical_holders(self.pid, access.identity)
        if not client.view.closed or proof['maps'] or proof['fds']:
            raise RuntimeError('pool VMA/FD still exists after actual SDK close')
        self.stage = 'normal-helper-stop'
        stop_resource_trackers(helpers)
        if descendants():
            raise RuntimeError('owned descendants remain after normal tracker stop')
        self.stage, self.operation = 'original-session-unmap-ACK', self.unmap_id
        if not self.ack_done:
            ack = client.ack_unmapped(native_drained=True, operation_id=self.unmap_id)
            self.ack_done = True
        else:
            ack = client.query_operation(self.unmap_id)
        self.operation, self.stage = None, 'fresh-Unmapped-snapshot'
        snapshot = client.snapshot()
        pool = next(p for p in snapshot['pools'] if p['pool_id'] == access.identity.pool_id)
        peer = next(c for c in pool['clients'] if c['client_session'] == client.view.descriptor['client_session'])
        if peer['state'] != 'Unmapped' or peer['children'] or pool['ranges']['leases']:
            raise RuntimeError('fresh Catalog does not confirm original client Unmapped')
        self._save('close-success.json', {'actual_native_close_complete': owner is None or owner._close_complete,
            'actual_SDK_view_closed': client.view.closed, 'pool_no_VMA_FD': proof, 'ACK': ack,
            'snapshot': snapshot, 'operation_id': self.unmap_id, 'scope': 'original peer; pool still UF-owned'})

    def _request(self, signum, current_frame):
        del current_frame
        if signum != signal.SIGUSR1 or not self.paused or self.busy:
            return
        if (os.getpid(), process_birth(os.getpid())) != (self.pid, self.birth):
            return  # Forked descendants cannot acknowledge the parent's mapping.
        self.busy, self.attempt = True, self.attempt + 1
        succeeded = False
        try:
            self._close()
            succeeded = True
        except BaseException as error:
            query = None
            if self.operation is not None and self.access is not None:
                try:
                    query = self.access.port.client.query_operation(self.operation)
                except BaseException as query_error:
                    query = {'error': repr(query_error)}
            try:
                self._save(f'close-failed-{self.attempt}.json', {'stage': self.stage, 'error': repr(error),
                    'traceback': traceback.format_exc(), 'operation_id': self.operation, 'query': query,
                    'original_context_retained': True, 'ACK_done': self.ack_done})
            except BaseException:
                pass  # Evidence I/O failure must not turn a retained client into a dead peer.
        finally:
            self.busy = False
        if succeeded:
            raise SystemExit(0)
