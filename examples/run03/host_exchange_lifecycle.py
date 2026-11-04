# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded cooperative host-fixture cleanup; no signals to a process group."""
import os
from pathlib import Path
import select
import subprocess
import traceback


class OuterProtection:
    """Always sample the REAL outer identities/mountinfo, even on an exception."""
    def __init__(self, report, workspace, mountinfo, namespace):
        self.report = report.setdefault('outer_protection', {})
        self.workspace, self.mountinfo, self.namespace = workspace, mountinfo, namespace
        self.before_mounts = self.after_mounts = None

    def capture(self):
        workspace, namespace = self.workspace.stat(), self.namespace.stat()
        return ((workspace.st_dev, workspace.st_ino), (namespace.st_dev, namespace.st_ino),
                self.mountinfo.read_bytes())

    def __enter__(self):
        self.before_workspace, self.before_namespace, self.before_mounts = self.capture()
        self.report.update(before_inode=self.before_workspace, before_namespace=self.before_namespace)
        return self

    def __exit__(self, kind, error, tb):
        self.capture_after()
        if not self.report['unchanged'] and kind is None:
            raise RuntimeError('outer namespace/mount protection failed; preserve evidence')
        return False  # Never replace or suppress a primary experiment error.

    def capture_after(self):
        try:
            workspace, namespace, self.after_mounts = self.capture()
            self.report.update(after_inode=workspace, after_namespace=namespace,
                               mountinfo_unchanged=self.before_mounts == self.after_mounts,
                               inode_unchanged=workspace == self.before_workspace,
                               namespace_unchanged=namespace == self.before_namespace)
            self.report['unchanged'] = (self.report['mountinfo_unchanged'] and self.report['inode_unchanged']
                                        and self.report['namespace_unchanged'])
        except BaseException as capture_error:
            self.report.update(unchanged=False, capture_error=repr(capture_error))


class StopRequested(RuntimeError):
    pass


class StopReader:
    """Linux inner's dedicated parent pipe; poll only at owned workflow waits."""
    def __init__(self, fd):
        self.fd = fd

    def check(self):
        if select.select([self.fd], [], [], 0)[0]:
            command = os.read(self.fd, 64)
            raise StopRequested('outer requested normal cleanup' if command == b'STOP\n'
                                else 'outer control closed or invalid; normal cleanup required')


def owned_tree(pid, limit=64):
    """Only follow this live child's /proc children, never enumerate global /proc."""
    pending, result = [(pid, os.getpid())], []
    while pending and len(result) < limit:
        current, parent = pending.pop()
        proc = Path(f'/proc/{current}')
        if not proc.exists():
            continue
        try:
            stat = (proc/'stat').read_text()
            fields = stat.rsplit(') ', 1)[1].split()
            if int(fields[1]) != parent:
                continue
            children = (proc/f'task/{current}/children').read_text().split()
            fds = {}
            for fd in (proc/'fd').iterdir():
                try:
                    fds[fd.name] = os.readlink(fd)
                except FileNotFoundError:
                    pass
            result.append({'pid': current, 'ppid': parent, 'start_ticks': fields[19],
                           'cmdline': (proc/'cmdline').read_text().replace('\0', ' '), 'fds': fds})
            pending.extend((int(child), current) for child in children)
        except (FileNotFoundError, ProcessLookupError, PermissionError) as error:
            result.append({'pid': current, 'identity_error': repr(error)})
    return {'processes': result, 'truncated': bool(pending), 'scope': 'owned child descendants only'}


def supervise_owned(proc, *, timeout, cleanup_grace):
    """wait(timeout) alone leaves a child alive. Request cleanup, then bounded reap.

    STOP travels only to this exact Popen stdin pipe. A survivor is reported,
    never killed/reset or declared clean. Timeout remains a failed experiment
    even if the cooperative cleanup eventually exits zero.
    """
    report = {'pid': proc.pid, 'timed_out': False, 'stop_requested': False, 'reaped': False,
              'initial_identity': owned_tree(proc.pid)}
    try:
        report['exit_code'] = proc.wait(timeout=timeout)
        report['reaped'] = True
    except BaseException as error:
        report['wait_error'] = repr(error)
        report['wait_traceback'] = traceback.format_exc()
        report['timed_out'] = isinstance(error, subprocess.TimeoutExpired)
        try:
            proc.stdin.write(b'STOP\n')
            proc.stdin.flush()
            report['stop_requested'] = True
        except (OSError, ValueError) as stop_error:
            report['stop_error'] = repr(stop_error)
        try:
            report['exit_code'] = proc.wait(timeout=cleanup_grace)
            report['reaped'] = True
        except BaseException as cleanup_error:
            report['cleanup_grace_expired'] = isinstance(cleanup_error, subprocess.TimeoutExpired)
            report['cleanup_wait_error'] = repr(cleanup_error)
    if not report['reaped']:
        report['remaining_owned'] = owned_tree(proc.pid)
        report['unresolved_child_pid'] = proc.pid
    report['passed'] = report['reaped'] and report.get('exit_code') == 0 and 'wait_error' not in report
    return report


def close_stage_on_exit(stage, report):
    """One bounded SDK close attempt. Preserve the primary error in report."""
    if stage is None:
        result = {'attempted': False, 'confirmed': True, 'scope': 'no stage created'}
    else:
        resources = stage.resources
        result = {'attempted': True, 'confirmed': False, 'generation': stage.lease.generation}
        try:
            stage.close()
            if not resources.closed or resources.views or not stage.lease.released:
                raise RuntimeError('stage close returned without confirmed resource release')
            result['confirmed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            result['traceback'] = traceback.format_exc()
        result.update(slot_busy=stage.lease.owner.busy, pending_views=len(resources.views),
                      unknown_acquire_or_retain=resources.unresolved_acquire,
                      view_identities=[{
                          'view_id': getattr(getattr(view, '_view', None), 'view_id', None),
                          'lease_id': getattr(getattr(view, '_view', None), 'lease_id', None),
                      } for view in resources.views])
    report['stage_exit_cleanup'] = result
    if not result['confirmed']:
        report['passed'] = False
        report['unresolved_stage'] = result
    return result


def explicit_revocation_rejection(error, exchange_code, error_type):
    """Pinned SDK test-only coupling: require the actual non-ok response frame.

    ExternalDdrError also represents malformed replies and lost commits. Type or
    text alone is insufficient. Inspect the fixed SDK's exact _exchange frame;
    no SDK/broker patch is installed. Any changed/missing structure is unknown.
    """
    if type(error) is not error_type:
        return False
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is exchange_code:
            values = tb.tb_frame.f_locals
            response = values.get('response')
            return (values.get('op') == 'AcquireExternalRead' and type(response) is dict
                    and response.get('status') == 'error'
                    and response.get('detail') == 'backing is not accepting imports'
                    and type(values.get('received_fds')) is list and not values['received_fds']
                    and 'mapping' in values and values['mapping'] is None
                    and str(error) == response['detail'])
        tb = tb.tb_next
    return False


def require_revoked_acquire(acquire, source, *, exchange_code, error_type, report, unexpected_views):
    try:
        view = acquire(source)
    except Exception as error:
        detail = {'error_type': type(error).__name__, 'detail': str(error)}
        if not explicit_revocation_rejection(error, exchange_code, error_type):
            report['new_acquire_after_revoke'] = {**detail, 'outcome': 'unknown_failure_not_revocation_proof'}
            raise  # Preserve the actual transport/protocol/other error.
        report['new_acquire_after_revoke'] = {**detail, 'outcome': 'explicit_broker_revocation_rejection'}
    else:
        unexpected_views.append(view)  # The consumer's finally must release it.
        report['new_acquire_after_revoke'] = {'outcome': 'unexpected_live_view', 'release_pending': True}
        raise AssertionError('revoked backing accepted new acquire')
