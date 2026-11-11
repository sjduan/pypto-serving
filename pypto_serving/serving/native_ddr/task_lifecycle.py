# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static task bootstrap: authenticated close notices and observational SHM creation receipts.

This is part of the normal Python launcher, installed before the task runs. It
does not attach to a live interpreter or replace any frozen runtime function.
Python audit hooks observe trusted cooperative code; they are not a sandbox.
"""
import array
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import struct
import sys

ENV = 'UFLOW_TASK_LIFECYCLE'
WORKER_SHA256 = 'f28dbd43ff747080eec171d774e0934a3b19bbf7b153c7a0f5880117d4144a1f'
_installed = False


def birth(pid):
    return int(Path('/proc', str(pid), 'stat').read_text().rpartition(')')[2].split()[19])


def exchange(kind, value=None, fd=None):
    """Finite transport; peer credentials and exact monitor birth authenticate the parent."""
    raw = os.environ.get(ENV)
    if raw is None:
        return None
    context = json.loads(raw)
    if birth(context['monitor_pid']) != context['monitor_birth']:
        raise RuntimeError('task monitor birth changed')
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as channel:
        channel.settimeout(.75)
        channel.connect(context['socket'])
        peer = struct.unpack('3i', channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if peer[:2] != (context['monitor_pid'], os.geteuid()):
            raise RuntimeError('task monitor peer identity changed')
        message = {'attempt': context['attempt'], 'leader_pid': context['leader_pid'],
                   'leader_birth': context['leader_birth'], 'monitor_birth': context['monitor_birth'],
                   'pid': os.getpid(), 'birth': birth(os.getpid()), 'kind': kind, 'value': value}
        control = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))]
        channel.sendmsg([json.dumps(message).encode()], control)
        response = json.loads(channel.recv(4096))
        if response != {'accepted': True}:
            raise RuntimeError('task lifecycle registration refused: ' + repr(response))
        return response


def begin_close(reason='failure'):
    """Call BEFORE any potentially blocking guarded/native close, including exception logging."""
    try:
        return exchange('closing', {'reason': reason})
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        # A known failure must not enter a possibly blocked destructor after
        # losing its short-deadline channel. Exit this actor without native
        # finally: the external monitor observes death and reaps descendants,
        # with logical DDR still Unknown. Never fall back to the long run cap.
        try:
            os.write(2, ('task close notification failed: ' + repr(error) + '\n').encode()[:1024])
        except OSError:
            pass
        os._exit(70)


def create_private_shared_memory(size):
    """Explicit task-private CPU buffer; never a shared UF/provider allocation."""
    from multiprocessing.shared_memory import SharedMemory
    if not os.environ.get(ENV):
        raise RuntimeError('private task allocation requires the established external monitor')
    return SharedMemory(create=True, size=size, name='uf-task-' + os.urandom(16).hex())


def install_observer():
    """Record exclusive creation with a real FD before mmap/constructor can return."""
    global _installed
    if _installed:
        return
    context = json.loads(os.environ[ENV])
    if not context['private_shm']:
        _installed = True
        return
    import multiprocessing.shared_memory as shared
    shared_source = str(Path(shared.__file__).resolve())
    shared_sha = hashlib.sha256(Path(shared_source).read_bytes()).hexdigest()
    own_source = str(Path(__file__).resolve())
    worker_source = Path('/workspace/pypto/runtime/python/simpler/worker.py')
    allowed_worker = str(worker_source.resolve()) if worker_source.is_file() and hashlib.sha256(
        worker_source.read_bytes()).hexdigest() == WORKER_SHA256 else None

    def audit(event, args):
        if event != 'mmap.__new__' or type(args[0]) is not int or args[0] < 0:
            return
        fd = args[0]
        try:
            target = os.readlink('/proc/self/fd/' + str(fd))
        except OSError:
            return
        if not target.startswith('/dev/shm/') or target.endswith(' (deleted)'):
            return
        frame, shared_frame, creator = sys._getframe().f_back, None, None
        for _ in range(64):
            if frame is None:
                break
            filename = str(Path(frame.f_code.co_filename).resolve())
            if (filename == shared_source and frame.f_code.co_name == '__init__'
                    and frame.f_locals.get('create') is True):
                shared_frame = frame
            if shared_frame is not None:
                if filename == own_source and frame.f_code.co_name == 'create_private_shared_memory':
                    creator = {'kind': 'explicit-task-private', 'source_sha256': hashlib.sha256(Path(own_source).read_bytes()).hexdigest()}
                    break
                if filename == allowed_worker and frame.f_code.co_name == '_init_hierarchical':
                    token = getattr(frame.f_locals.get('self'), '_shm_token', None)
                    if type(token) is str and re.fullmatch('[0-9a-f]{32}', token) and re.fullmatch(
                            'sp-' + token[:8] + '-(?:chip|sub|next)-[0-9]+', Path(target).name):
                        creator = {'kind': 'frozen-simpler-mailbox', 'source_sha256': WORKER_SHA256, 'private_namespace': token}
                    break
            frame = frame.f_back
        if creator is not None:
            try:
                exchange('private-created', {**creator, 'shared_source_sha256': shared_sha,
                         'path': target, 'original_fd': fd, 'exclusive_create': True}, fd=fd)
            except (OSError, ValueError, RuntimeError, KeyError):
                begin_close('private-registration-failed-before-constructor-unwind')
                raise

    sys.addaudithook(audit)
    _installed = True


def main():
    if len(sys.argv) < 3 or sys.argv[1] != '--task-script':
        raise RuntimeError('static task bootstrap requires a literal Python file')
    script, args = sys.argv[2], sys.argv[3:]
    install_observer()
    exchange('bootstrap-ready')
    sys.argv = [script, *args]
    sys.path[0] = str(Path(script).resolve().parent)
    try:
        runpy.run_path(script, run_name='__main__')
    except BaseException:
        begin_close('task-exception')
        raise


if __name__ == '__main__':
    main()
