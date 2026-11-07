# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Immutable schema2/shared and schema3/M1-only phase budgets. No SDK/device imports."""
from __future__ import annotations

import json
from pathlib import Path
import re

from scenario_contract import digest, save

CAPS = {'O1': 2888, 'M1': 884}
STAGES = {'O1': [('calibrate', 606), ('run', 2074), ('profile', 206), ('finalize', 2)],
          'M1': [('two', 248), ('four', 636)]}
PROGRESS = {'O1': {'calibrate': 0, 'run': 606, 'profile': 2680, 'finalize': 2886},
            'M1': {'two': 0, 'four': 248, 'done': 884}}
FIELDS = {'schema', 'decision_id', 'resource_group', 'pool_id', 'pool_epoch', 'provider_session',
          'common_manifest_sha256', 'counter_anchor', 'admission_limit', 'cleanup_limit',
          'terminal_recovery_reserve', 'participants'}


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative strict integer required')
    return value


def bound(anchor, observed, line, phase, *, schema=2):
    """Arithmetic only: the caller must prove the prior real phases closed."""
    anchor, observed = integer(anchor), integer(observed)
    if type(schema) is not int or schema not in (2,3):raise ValueError('strict envelope schema2 or3 required')
    if schema==3:
        if line!='M1' or phase not in PROGRESS['M1']:raise ValueError('schema3 admits only ordered M1 phases')
        completed=PROGRESS['M1'][phase];remaining=CAPS['M1']-completed
        if observed<anchor or observed-anchor>completed:
            raise ValueError('M1-only counter exceeds proven prior phase cap or regressed')
        # Finalizer is outside the normal M1 stages; retain its two actual
        # future mutations without inventing an absent O1 participant/spend.
        worst=observed+remaining+2
        if anchor+CAPS['M1']+2>=4096 or worst>=4096 or worst+256>=8192:
            raise ValueError('M1-only envelope lacks normal/terminal/finalizer headroom')
        return {'participants':['M1'],'own_completed_upper':completed,'own_remaining_upper':remaining,
                'finalizer_records_reserved':2,'worst':worst,'strict_admission_spare':4095-worst,
                'not_a_resource_or_lease_grant':True}
    if anchor + sum(CAPS.values()) >= 4096 or anchor + sum(CAPS.values()) + 256 >= 8192:
        raise ValueError('initial immutable envelope does not fit')
    if line not in PROGRESS or phase not in PROGRESS[line]:
        raise ValueError('invalid ordered phase')
    completed = PROGRESS[line][phase]
    remaining = CAPS[line] - completed
    other = 'M1' if line == 'O1' else 'O1'
    if observed < anchor:
        raise ValueError('original Catalog counter regressed')
    lower = max(0, observed - anchor - completed)
    if lower > CAPS[other]:
        raise ValueError('unexpected extra records exceed other normal cap')
    other_remaining = CAPS[other] - lower
    worst = observed + remaining + other_remaining
    if worst >= 4096 or worst + 256 >= 8192:
        raise ValueError('insufficient admission/terminal cleanup headroom')
    return {'other_spent_lower': lower, 'other_remaining_upper': other_remaining,
            'own_completed_upper': completed, 'own_remaining_upper': remaining,
            'worst': worst, 'strict_admission_spare': 4095 - worst,
            'not_a_resource_or_lease_grant': True}


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def sealed(reference):
    if type(reference) is not dict or set(reference) != {'path', 'sha256'}:
        raise ValueError('exact immutable file reference required')
    if type(reference['path']) is not str or not Path(reference['path']).is_absolute():
        raise ValueError('absolute immutable file path required')
    if digest(Path(reference['path']).read_bytes()) != reference['sha256']:
        raise ValueError('immutable evidence hash differs')
    return read(reference['path'])


def validate_envelope(value, pool, manifest_sha, source_sha):
    if type(value) is not dict or set(value) != FIELDS:
        raise ValueError('exact immutable envelope fields required')
    if type(value['schema']) is not int or value['schema'] not in (2,3):
        raise ValueError('schema2 or schema3 strict integer required')
    if type(value['decision_id']) is not str or not value['decision_id'].strip():
        raise ValueError('manager decision identity required')
    if value['resource_group'] != 'phasef-parallel-20261106':
        raise ValueError('resource group differs')
    for key in ('pool_id', 'pool_epoch', 'provider_session'):
        if value[key] != pool[key]:raise ValueError('original pool/provider identity differs')
    if value['common_manifest_sha256'] != manifest_sha:
        raise ValueError('original public manifest differs')
    for key, expected in [('admission_limit', 4096), ('cleanup_limit', 8192), ('terminal_recovery_reserve', 256)]:
        if integer(value[key]) != expected:raise ValueError('fixed Catalog limits/reserve differ')
    participants = value['participants']
    expected_lines=set(CAPS) if value['schema']==2 else {'M1'}
    if type(participants) is not dict or set(participants) != expected_lines:
        raise ValueError('exact final source participants required for envelope schema')
    for line in expected_lines:
        row = participants[line]
        if type(row) is not dict or set(row) != {'source_sha', 'total', 'stages'}:
            raise ValueError('exact participant fields required')
        if type(row['source_sha']) is not str or not re.fullmatch('[0-9a-f]{40}', row['source_sha']):
            raise ValueError('final Git source SHA required')
        if integer(row['total']) != CAPS[line]:raise ValueError('fixed participant cap differs')
        expected = [{'id': name, 'max': cap} for name, cap in STAGES[line]]
        if type(row['stages']) is not list or len(row['stages']) != len(expected):
            raise ValueError('ordered stage cap table differs')
        for stage, want in zip(row['stages'], expected):
            if type(stage) is not dict or set(stage) != {'id', 'max'} or stage['id'] != want['id']:
                raise ValueError('ordered exact stage fields differ')
            if integer(stage['max']) != want['max']:raise ValueError('fixed stage cap differs')
    if participants['M1']['source_sha'] != source_sha:
        raise ValueError('final M1 source differs from deployment')
    bound(value['counter_anchor'], value['counter_anchor'], 'M1', 'two',schema=value['schema'])
    return value


def check_sessions(snapshot, pool, sessions):
    if len(snapshot['pools']) != 1:raise ValueError('original single pool required')
    actual = snapshot['pools'][0]
    if any(actual.get(key) != value for key, value in pool.items()) or actual['state'] != 'Allocated' or actual['frozen']:
        raise ValueError('original active pool drifted/Unknown')
    clients = {row['client_session']: row for row in actual['clients']}
    if len(clients) != len(actual['clients']):raise ValueError('duplicate client identity')
    for session in sessions:
        row = clients[session]
        if row['state'] != 'Unmapped' or row['children']:
            raise ValueError('own actual client/child/view obligation remains')
    if any(row['client_session'] in sessions for row in actual['ranges']['leases']):
        raise ValueError('own actual range obligation remains')


class PhaseLedger:
    """Append-only start/closed evidence, never a mutable quota or activation file.

    Paths are fixed by the envelope digest, not a caller supplied previous receipt.
    A failed/incomplete claim permanently blocks this campaign's next attempt.
    """
    def __init__(self, root, envelope_sha, envelope, pool, source_manifest_sha):
        self.root = Path(root) / envelope_sha
        self.envelope_sha, self.envelope, self.pool = envelope_sha, envelope, pool
        self.source_manifest_sha = source_manifest_sha

    def identity(self, phase):
        return {'schema': self.envelope['schema'], 'phase': phase, 'envelope_sha256': self.envelope_sha,
                'source_sha': self.envelope['participants']['M1']['source_sha'],
                'source_manifest_sha256': self.source_manifest_sha,
                'common_manifest_sha256': self.envelope['common_manifest_sha256'], 'pool': self.pool}

    def budget(self,observed,phase):
        return bound(self.envelope['counter_anchor'],observed,'M1',phase,schema=self.envelope['schema'])

    def other_cap(self):
        return CAPS['O1'] if self.envelope['schema']==2 else 0

    def prior(self, phase):
        if phase not in ('two', 'four', 'done'):raise ValueError('unknown M1 phase')
        expected = {'two': [], 'four': ['two'], 'done': ['two', 'four']}[phase]
        previous_end = self.envelope['counter_anchor']
        for prior in expected:
            start_path = self.root / (prior + '.started.json')
            start = read(start_path)
            closed = read(self.root / (prior + '.closed.json'))
            if (set(closed) != {'identity', 'start_sha256', 'evidence'} or closed['identity'] != self.identity(prior)
                    or closed['start_sha256'] != digest(start_path.read_bytes())
                    or set(start) != {'identity', 'observed_records', 'config_sha256', 'budget'}
                    or start['identity'] != self.identity(prior)):
                raise ValueError('prior phase/start/source/envelope identity differs')
            report = sealed(closed['evidence'])
            if start['config_sha256'] != report['plan']['config_sha256']:
                raise ValueError('original admitted phase configuration differs')
            end = self.validate_closed_report(prior, report)
            begin = integer(start['observed_records'])
            if start['budget'] != self.budget(begin,prior):
                raise ValueError('original phase budget differs from immutable envelope')
            if begin < previous_end or end < begin or end - begin > dict(STAGES['M1'])[prior] + self.other_cap():
                raise ValueError('phase counter chronology/cap differs')
            previous_end = end
        if phase != 'done' and any((self.root / (phase + '.' + suffix + '.json')).exists() for suffix in ('started', 'closed')):
            raise ValueError('phase already claimed; no retry/new attempt after failure')
        if phase == 'two' and any((self.root / ('four.' + suffix + '.json')).exists() for suffix in ('started', 'closed')):
            raise ValueError('out-of-order phase history')
        return previous_end

    def validate_closed_report(self, phase, report):
        n = 2 if phase == 'two' else 4
        if (report.get('phase_identity') != self.identity(phase)
                or report.get('scope') != 'real-runtime-captured-pending-review'
                or report.get('status') != 'CAPTURED-PENDING-REVIEW'
                or report.get('M1_line_retired') is not False or report.get('pool_destroyed') is not False
                or report['plan']['pool'] != self.pool or report['plan']['budget']['replicas'] != n
                or report['plan']['source_manifest_sha256'] != self.source_manifest_sha):
            raise ValueError('prior phase is not real original-source model evidence')
        cleanup = report['cleanup']
        starts, stops = cleanup['model_startups'], cleanup['model_stops']
        names = {prefix + str(i) for prefix in ('cold', 'shared') for i in range(n)}
        if set(starts) != names or set(stops) != names:raise ValueError('full independent model lifecycle absent')
        for name in names:
            start, stop = starts[name], stops[name]
            if (type(start['pid']) is not int or not start['birth'] or not start['descendant_births']
                    or stop['actor_pid'] != start['pid'] or stop['worker_pid'] != start['worker_pid']
                    or stop['stop_ok'] is not True or stop['shutdown_report']['cleanup_error'] is not None
                    or stop['shutdown_report']['worker_reaped'] is not True):
                raise ValueError('original real Worker stop/reap evidence absent')
            if name.startswith('shared'):
                proof = stop['native_stop_proof']
                if (proof['worker_epoch'] != start['startup']['worker_epoch']
                        or any(proof[k] is not True for k in ('module_drained', 'owner_drained_arguments_released'))
                        or any(integer(proof[k]) != 0 for k in ('jobs', 'steps', 'controls', 'retiring'))):
                    raise ValueError('original native drain/argument proof absent')
        sessions = cleanup['all_own_sessions_actual_Unmapped']
        if len(sessions) != n + 2 or len(set(sessions)) != n + 2:
            raise ValueError('complete actual model/directory/observer sessions absent')
        for name in names:
            if name.startswith('shared') and starts[name]['startup']['pool_identity']['client_session'] not in sessions:
                raise ValueError('real model SDK session not closed')
        check_sessions(cleanup['actual_common_snapshot'], self.pool, sessions)
        if (not cleanup['original_births_gone'] or cleanup['reservations_released'] is not True
                or cleanup['directory_socket_absent'] is not True or cleanup['listeners_closed'] is not True):
            raise ValueError('original process/view/reservation/endpoint cleanup absent')
        if any(value is not True for value in cleanup['original_births_gone'].values()):
            raise ValueError('original process birth remains')
        required_births = {str(start['pid']) for start in starts.values()}
        required_births.update(str(pid) for start in starts.values() for pid in start['descendant_births'])
        if not required_births.issubset(cleanup['original_births_gone']):
            raise ValueError('original actor/native child births omitted')
        devices = cleanup['devices']
        if len(devices) != n or any(d['fds']['users'] or d['fds']['denied'] or d['status']['health'] != 'OK' for d in devices):
            raise ValueError('actual device FD/health closure absent')
        if len(report['functional']) != 8 * (n - 1) or len(report['baseline']) != n or len(report['shared']) != n - 1:
            raise ValueError('complete original model campaign evidence absent')
        from scenario_controller import check_request, samples
        for rows in report['baseline'].values():
            if len(rows) != 23:raise ValueError('baseline sample count differs')
            for row in rows:check_request(row, prefill=257, loads=0)
        for index, value in report['shared'].items():
            rows = value['samples']
            if len(rows) != 23:raise ValueError('shared sample count differs')
            for row in rows:check_request(row, prefill=1, loads=2)
            samples(report['baseline'][index][3:], rows[3:])
        return integer(cleanup['actual_common_snapshot']['operation_records'])

    def check(self, phase, observed):
        if integer(observed) < self.prior(phase):raise ValueError('counter regressed from prior actual closure')
        return self.budget(observed,phase)

    def claim(self, phase, observed, config_sha):
        result = self.check(phase, observed)
        self.root.mkdir(parents=True, exist_ok=True)
        save(self.root / (phase + '.started.json'), {'identity': self.identity(phase),
             'observed_records': observed, 'config_sha256': config_sha, 'budget': result})
        return result

    def close(self, phase, report, evidence_path):
        end = self.validate_closed_report(phase, report)
        start = self.root / (phase + '.started.json')
        begin = integer(read(start)['observed_records'])
        claim = read(start)
        if claim['identity'] != self.identity(phase) or claim['config_sha256'] != report['plan']['config_sha256']:
            raise ValueError('original phase claim/source/configuration differs')
        if end < begin or end - begin > dict(STAGES['M1'])[phase] + self.other_cap():
            raise ValueError('actual campaign exceeds fixed mutation envelope')
        self.budget(end,'four' if phase == 'two' else 'done')
        save(evidence_path, report)
        save(self.root / (phase + '.closed.json'), {'identity': self.identity(phase),
             'start_sha256': digest(start.read_bytes()),
             'evidence': {'path': str(Path(evidence_path).resolve()), 'sha256': digest(Path(evidence_path).read_bytes())}})
