# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Sealed real P1 metadata replay; no live daemon, RPC, FD, native or model run."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

sdk = pytest.importorskip('uflow.managed_pool')

from pypto_serving.serving.native_ddr.managed_pool_sdk import SdkManagedPoolPort

RECORDED = json.loads((Path(__file__).with_name('fixtures') / 'p1_n003_ranges.json').read_text())
assert RECORDED['P1_candidate'] == '6c88583388aa7262a318cf23550e9eae9bf793b5'
assert RECORDED['source_sha256'] == '1b415a3ca474f00ecc6d9d4a9950209ee9db4d6c2494fc7c6abdb09d9543ca88'


def replay(mode):
    name = 'ab-held-before-c' if mode == 'w' else 'ab-still-held-after-c-actual-write-read'
    snapshot = copy.deepcopy(RECORDED['snapshots'][name])
    pool = snapshot['pools'][0]
    held = next(item for item in pool['ranges']['leases'] if item['mode'] == mode)
    client = next(item for item in pool['clients'] if item['client_session'] == held['client_session'])
    descriptor = {key: pool[key] for key in sdk.POOL_FIELDS}
    descriptor.update(client_session=client['client_session'], access_generation=client['access_generation'],
                      access=client['access'])
    port = SdkManagedPoolPort(SimpleNamespace(view=SimpleNamespace(descriptor=descriptor, buffer=None),
                                              snapshot=lambda: snapshot))
    allocation = next(item for item in pool['ranges']['allocations'] if item['block_id'] == held['block_id'])
    raw_block = sdk.PoolBlock(pool['pool_id'], pool['pool_epoch'], pool['provider_session'],
        allocation['block_id'], allocation['generation'], allocation['offset'], allocation['nbytes'])
    raw_lease = {key: value for key, value in held.items() if key != 'state'}
    raw_lease.update(pool_id=pool['pool_id'], pool_epoch=pool['pool_epoch'], provider_session=pool['provider_session'],
                     access_generation=client['access_generation'])
    original = sdk.PoolLease.from_dict(raw_lease)
    value = port._lease(original, port._block(raw_block))
    port._blocks[value.block.block_id, value.block.generation] = raw_block
    port._leases[value.lease_id] = value, original
    return port, value, pool, allocation, held, client


@pytest.mark.parametrize('old_content', ['recorded-Uncommitted', 'Committed', 'Invalid', 'disjoint-change'])
def test_actual_recorded_writing_lease_authorizes_rewrite_independent_of_old_content(old_content):
    port, value, pool, allocation, held, client = replay('w')
    assert held['state'] == 'Writing' and value.content_version == 0
    assert all(item['state'] == 'Uncommitted' for item in allocation['segments'])
    if old_content in ('Committed', 'Invalid'):
        for segment in allocation['segments']:
            segment.update(state=old_content, content_version=9 if old_content == 'Committed' else 0,
                           content_id='previous-committed-prefix' if old_content == 'Committed' else '')
    elif old_content == 'disjoint-change':
        begin, end = value.byte_offset, value.byte_offset + value.nbytes
        # Calibrate a new content version strictly AFTER this writer's half-open
        # interval in the same allocation; the exact active lease is unchanged.
        assert end < allocation['offset'] + allocation['nbytes']
        allocation['segments'] = [
            {'offset': begin, 'nbytes': value.nbytes, 'state': 'Uncommitted', 'content_version': 0, 'content_id': ''},
            {'offset': end, 'nbytes': allocation['offset'] + allocation['nbytes'] - end,
             'state': 'Committed', 'content_version': 19, 'content_id': 'another-range'}]
    port.revalidate(value)
    assert port._leases[value.lease_id][0] is value


@pytest.mark.parametrize('fault', ['Unknown', 'Closed', 'missing', 'client-generation', 'allocation-generation',
                                  'lease-generation', 'holder', 'widened-range', 'pool-epoch'])
def test_writer_still_requires_full_current_live_identity_before_ignoring_old_content(fault):
    port, value, pool, allocation, held, client = replay('w')
    if fault in ('Unknown', 'Closed'):
        held['state'] = fault
    elif fault == 'missing':
        pool['ranges']['leases'].remove(held)
    elif fault == 'client-generation':
        client['access_generation'] += 1
    elif fault == 'allocation-generation':
        allocation['generation'] += 1
    elif fault == 'lease-generation':
        held['lease_generation'] += 1
    elif fault == 'holder':
        held['client_session'] = 'different-client'
    elif fault == 'widened-range':
        held['nbytes'] += 65536
    else:
        pool['pool_epoch'] = 'previous-pool-incarnation'
    with pytest.raises(ValueError):
        port.revalidate(value)
    assert value.lease_id in port._leases


def test_actual_recorded_read_still_requires_complete_committed_content():
    port, value, pool, allocation, held, client = replay('r')
    assert held['state'] == 'Reading' and value.content_version > 0
    port.revalidate(value)


@pytest.mark.parametrize('fault', ['version', 'content-id', 'gap', 'Invalid', 'Reading-lease-Unknown'])
def test_recorded_read_rejects_content_version_identity_or_coverage_changes(fault):
    port, value, pool, allocation, held, client = replay('r')
    segment = next(item for item in allocation['segments']
                   if item['offset'] <= value.byte_offset < item['offset'] + item['nbytes'])
    if fault == 'version':
        segment['content_version'] += 1
    elif fault == 'content-id':
        segment['content_id'] = 'stale-prefix'
    elif fault == 'gap':
        allocation['segments'].remove(segment)
    elif fault == 'Invalid':
        segment['state'] = 'Invalid'
    else:
        held['state'] = 'Unknown'
    with pytest.raises(ValueError):
        port.revalidate(value)
    assert value.lease_id in port._leases
