# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU mmap/fake-authority guard invariants; never actual UF/native evidence."""
import copy
from types import SimpleNamespace

import pytest

from pypto_serving.serving.native_ddr.diagnostics import capture_shared_guards
from pypto_serving.serving.native_ddr.managed_pool import ManagedPoolAccess
from tests.unit.serving.test_managed_pool_protocol import FakePort
from tests.unit.serving.test_shared_prefix_directory import SyntheticAuthority


@pytest.mark.parametrize('fault', [None, 'missing-child', 'wrong-namespace', 'unknown-child', 'closing',
                                 'foreign-backing', 'boolean-birth'])
def test_shared_guard_readers_keep_full_own_sdk_child_checks_with_other_mapped_clients(tmp_path, fault):
    port = FakePort()
    access = ManagedPoolAccess(port)
    uf = SyntheticAuthority()
    child = {'pid': 123, 'starttime_ticks': 20, 'uid': 0, 'gid': 0,
             'pid_namespace_dev': 1, 'pid_namespace_ino': 2}
    owner = {**child, 'pid': 42, 'starttime_ticks': 10}
    current = {'client_session': port.identity.client_session, 'state': 'Mapped', 'closing': False,
        'access': 'rw', 'access_generation': port.identity.access_generation, 'birth': owner,
        'children': [{'birth': copy.deepcopy(child), 'unknown': False}]}
    uf.pool['clients'] = [current, {**copy.deepcopy(current), 'client_session': 'peer', 'children': []}]
    port._children = {(123, 20): child}
    port.client = SimpleNamespace(_children={123: child}, snapshot=uf.snapshot)
    access.children = {(123, 20): object()}
    for block in uf.blocks:
        port.committed_ranges[block.block_id] = SimpleNamespace(content_version=1, content_id='guard-content')
    slots = SimpleNamespace(access=access, bindings=uf.blocks, shared_directory=object(),
                            busy=None, child_registrations={123: 20})
    if fault == 'missing-child':
        current['children'] = []
    elif fault == 'wrong-namespace':
        current['children'][0]['birth']['pid_namespace_ino'] = 3
    elif fault == 'unknown-child':
        current['children'][0]['unknown'] = True
    elif fault == 'closing':
        current['closing'] = True
    elif fault == 'foreign-backing':
        uf.pool['canonical']['inode'] = 99
    elif fault == 'boolean-birth':
        current['children'][0]['birth']['pid_namespace_dev'] = True
    try:
        if fault is None:
            result = capture_shared_guards(slots, tmp_path, 'r06-test')
            assert len(result['guards']) == 4 and all(row['bytes'] == 2097152 for row in result['guards'])
            assert result['client_before'] == result['client_after'] == current
            assert all(row['release_ack']['applied'] is True for row in result['guards'])
            assert not access.leases and not access.unknown
        else:
            with pytest.raises(RuntimeError):
                capture_shared_guards(slots, tmp_path, 'r06-test')
            assert not list(tmp_path.glob('*.bin')) and not access.leases and not access.unknown
    finally:
        access.children.clear()
        port._children.clear()
        port.client._children.clear()
        access.close()
