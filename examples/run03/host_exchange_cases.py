# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Declared host lifecycle cases and durable peer evidence, never recovery."""
import json
import os
import stat

from examples.run03.host_exchange_guard import sha

from examples.run03.host_exchange_guard import attempt_token

EXPOSURE_FIELDS = ('committed_bytes', 'backing_fds', 'owner_leases', 'active_views',
                   'pending_handoffs', 'in_flight')
ALL_FIELDS = (*EXPOSURE_FIELDS, 'session_pidfds')


def case_for_token(token):
    attempt_token(token)
    for case in ('normal-clean', 'owner-exit-negative'):
        if token.endswith('-'+case):
            return case
    raise ValueError('explicit lifecycle case required in attempt token')


def declaration(case):
    if case not in ('normal-clean', 'owner-exit-negative'):
        raise ValueError('unsupported declared case')
    return {'case': case, 'expected_authority_mode': 'clean' if case == 'normal-clean' else 'dirty',
            'requires_final_zero': case == 'normal-clean', 'recovery_supported': False}


def require_zero(snapshot, fields=EXPOSURE_FIELDS):
    for field in fields:
        if type(snapshot.get(field)) is not int or snapshot[field] != 0:
            raise RuntimeError('exposure not closed: '+field+'='+repr(snapshot.get(field)))


def permit_owner_exit(case, *, revoked, exposure_snapshot=None):
    declaration(case)
    if not revoked:
        raise RuntimeError('owner exit before confirmed revoke')
    if case == 'normal-clean':
        if exposure_snapshot is None:
            raise RuntimeError('owner must stay alive until explicit exposure-close proof')
        require_zero(exposure_snapshot)


def persist_event(output, journal, report, role, expected, raw):
    # Raw line survives JSON parsing, event mismatch, or any later assertion.
    row = {'role': role, 'expected': expected, 'raw': raw}
    with journal.open('a') as stream:
        stream.write(json.dumps(row)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    report.setdefault('peer_events', []).append(row)
    output.write_text(json.dumps(report, indent=2)+'\n')


def domain_snapshot(root):
    """Hash only this exact old task authority plus its two socket identities."""
    authority = root/'authority/external-ddr-v0'
    paths = [authority, *sorted(authority.iterdir()),
             root/'runs/host-exchange/u.sock', root/'runs/host-exchange/u.sock.external-ddr']
    result = {}
    for path in paths:
        relative = str(path.relative_to(root))
        try:
            info = path.lstat()
        except FileNotFoundError:
            result[relative] = None
            continue
        result[relative] = {'dev': info.st_dev, 'inode': info.st_ino, 'mode': info.st_mode,
                            'sha256': sha(path) if stat.S_ISREG(info.st_mode) else None}
    return result


def explicit_unknown(error, exchange_code, error_type, operation):
    """Require the pinned real SDK response frame, not same-text transport errors."""
    details = {'AcquireExternalRead': 'unresolved exposure, authority admission disabled',
               'GetExternalDdrSnapshot': 'unresolved exposure, external DDR state unknown'}
    if type(error) is not error_type or operation not in details:
        return None
    tb = error.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code is exchange_code:
            values = tb.tb_frame.f_locals
            response = values.get('response')
            if (values.get('op') == operation and type(response) is dict and response.get('status') == 'error'
                    and response.get('detail') == details[operation] and str(error) == details[operation]
                    and type(values.get('received_fds')) is list and not values['received_fds']
                    and 'mapping' in values and values['mapping'] is None):
                return {'op': operation, 'response': response, 'received_fds': [], 'mapping': None,
                        'error_type': type(error).__name__, 'outcome': 'explicit_unknown_exposure_rejection'}
        tb = tb.tb_next
    return None
