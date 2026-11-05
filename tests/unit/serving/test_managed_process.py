# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual recorded n004 tracker and strict negative role/physical-resource replay."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.serving.native_ddr import managed_process as process

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='Linux proc/device identities')
RECORD = json.loads((Path(__file__).with_name('fixtures') / 'n004_resource_tracker.json').read_text())
IDENTITY = SimpleNamespace(device=RECORD['physical'][0], inode=RECORD['physical'][1])


def validate(value):
    return process.validate_tracker(value, 1024040, 484128722, RECORD['parent'], IDENTITY)


def test_recorded_actual_tracker_is_not_a_mapping_holder():
    result = validate(RECORD['observation'])
    assert result['role'] == 'stdlib-resource-tracker'
    assert result['pool_control_FD_VMA_absent'] and result['native_device_module_FD_absent']


@pytest.mark.parametrize('field,value', [
    ('exe', '/tmp/python'), ('exe_sha256', '0' * 64), ('pid', 1), ('ppid', 1),
    ('birth', 1), ('birth_after', 1), ('cmdline', ['/workspace/pypto/.venv/bin/python', '-c', 'fake']),
])
def test_forged_identity_rejected(field, value):
    observed = copy.deepcopy(RECORD['observation'])
    observed[field] = value
    with pytest.raises(RuntimeError):
        validate(observed)


@pytest.mark.parametrize('command', [
    'from multiprocessing.resource_tracker import main;main(25);evil()',
    'from multiprocessing.resource_tracker import main;main(2)',
    'from multiprocessing.resource_tracker import main;main(26)',
    'from multiprocessing.resource_tracker import main;main(25) ',
])
def test_forged_command_rejected(command):
    observed = copy.deepcopy(RECORD['observation'])
    observed['cmdline'][-1] = command
    with pytest.raises(RuntimeError):
        validate(observed)


@pytest.mark.parametrize('kind', ['extra-fd', 'pool-fd', 'device-fd', 'stdio', 'pipe', 'pool-vma',
                                       'partial-pool-vma', 'shared-control', 'native-module', 'ascend-module'])
def test_unmapped_helper_must_have_no_resource_or_device_access(kind):
    value = copy.deepcopy(RECORD['observation'])
    pipe = next(fd for fd in value['fds'] if fd['fd'] == 25)
    if kind == 'extra-fd':
        value['fds'].append({**pipe, 'fd': 27})
    elif kind == 'pool-fd':
        pipe.update(dev=25, inode=842024)
    elif kind == 'device-fd':
        pipe.update(mode=0o20666, target='/tmp/disguised-device-alias')
    elif kind == 'stdio':
        next(fd for fd in value['fds'] if fd['fd'] == 1)['target'] = '/tmp/foreign.log'
    elif kind == 'pipe':
        pipe['target'] = '/dev/shm/control'
    elif kind in ('pool-vma', 'partial-pool-vma'):
        value['maps'] += '1000-2000 rw-s 00001000 00:19 842024 /dev/shm/renamed\n'
    elif kind == 'shared-control':
        value['maps'] += '1000-2000 rw-s 00000000 00:19 123 /dev/shm/control\n'
    elif kind == 'native-module':
        value['maps'] += '1000-2000 r-xp 00000000 08:10 123 /tmp/pypto_core.so\n'
    elif kind == 'ascend-module':
        value['maps'] += '1000-2000 r-xp 00000000 08:10 123 /usr/local/Ascend/libruntime.so\n'
    with pytest.raises(RuntimeError):
        validate(value)


def test_explicit_control_resource_fd_is_rejected():
    value = copy.deepcopy(RECORD['observation'])
    pipe = next(fd for fd in value['fds'] if fd['fd'] == 25)
    with pytest.raises(RuntimeError, match='pool/control FD'):
        process.validate_tracker(value, 1024040, 484128722, RECORD['parent'], IDENTITY,
                                 [(pipe['dev'], pipe['inode'])])


def test_unknown_unmapped_descendant_still_poisoned_not_skipped(monkeypatch):
    unknown = []
    access = SimpleNamespace(identity=IDENTITY, report_unknown=lambda *a: unknown.append(a))
    monkeypatch.setattr(process, 'descendants', lambda **k: {1024040: 484128722})
    monkeypatch.setattr(process, 'has_pool_mapping', lambda *a, **k: False)
    value = copy.deepcopy(RECORD['observation'])
    value['cmdline'] = ['sleep', '100']
    monkeypatch.setattr(process, 'tracker_observation', lambda *a, **k: value)
    monkeypatch.setattr(process.os, 'getpid', lambda: RECORD['parent'])
    with pytest.raises(RuntimeError):
        process.attach_new_children(access, {}, helpers={})
    assert len(unknown) == 1


def test_helper_separate_and_real_mapped_holder_keeps_registration(monkeypatch):
    current = {1024040: 484128722, 1024041: 484128722}
    attached, helpers = [], {}
    monkeypatch.setattr(process, 'descendants', lambda **k: current)
    monkeypatch.setattr(process, 'has_pool_mapping', lambda pid, *a, **k: pid == 1024041)
    monkeypatch.setattr(process, 'tracker_observation', lambda *a, **k: RECORD['observation'])
    monkeypatch.setattr(process.os, 'getpid', lambda: RECORD['parent'])
    access = SimpleNamespace(identity=IDENTITY, attach_child=lambda *a: attached.append(a),
        report_unknown=lambda *a: pytest.fail('valid separate helper was poisoned'))
    assert process.attach_new_children(access, {}, helpers=helpers) == {1024041: 484128722}
    assert attached == [(1024041, 484128722)] and set(helpers) == {1024040}


def test_helper_only_without_native_mapping_holder_is_rejected(monkeypatch):
    monkeypatch.setattr(process, 'descendants', lambda **k: {1024040: 484128722})
    monkeypatch.setattr(process, 'has_pool_mapping', lambda *a, **k: False)
    monkeypatch.setattr(process, 'tracker_observation', lambda *a, **k: RECORD['observation'])
    monkeypatch.setattr(process.os, 'getpid', lambda: RECORD['parent'])
    errors = []
    access = SimpleNamespace(identity=IDENTITY, report_unknown=lambda *a: errors.append(a))
    with pytest.raises(RuntimeError, match='mapping holder missing'):
        process.attach_new_children(access, {}, helpers={})
    assert errors
