# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recorded public metadata only; invalid partitions never reach an SDK mutation."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pypto_serving.serving.native_ddr.shared_pool import bindings_document, validate_partition, register_partition_client
from pypto_serving.serving.native_ddr import managed_pool_sdk as sdk

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures/m1'


def inputs():
    return (json.loads((FIXTURE / 'slots.json').read_text()),
            json.loads((FIXTURE / 'public-manifest.json').read_text()))


def test_valid_recorded_partition_is_exactly_M1():
    document, manifest = inputs()
    assert [b.block_id for b in validate_partition(document, manifest)] == [2, 3, 4, 5]


@pytest.mark.parametrize('fault', ['O1', 'duplicate', 'missing', 'generation', 'offset', 'bounds', 'group',
    'partition', 'bool-schema', 'bool-block', 'bool-manifest', 'bool-index', 'line', 'guard', 'payload',
    'partition-indices', 'pool-bool', 'overlap'])
def test_bad_recorded_partition_fails_before_any_SDK_or_socket_entry(tmp_path, monkeypatch, fault):
    document, manifest = inputs()
    if fault == 'O1':
        raw = manifest['allocations'][0]['block']
        document['blocks'][0].update(block_id=raw['block_id'], byte_offset=raw['offset'])
    elif fault == 'duplicate': document['blocks'][0] = copy.deepcopy(document['blocks'][1])
    elif fault == 'missing': document['blocks'].pop()
    elif fault == 'generation': document['blocks'][0]['generation'] += 1
    elif fault == 'offset': document['blocks'][0]['byte_offset'] += 65536
    elif fault == 'bounds': document['blocks'][0]['nbytes'] += 65536
    elif fault == 'group': document['resource_group'] = 'wrong-group'
    elif fault == 'partition': document['partition'] = 'O1'
    elif fault == 'bool-schema': document['schema'] = True
    elif fault == 'bool-block': document['blocks'][0]['generation'] = True
    elif fault == 'bool-manifest': manifest['schema'] = True
    elif fault == 'bool-index': manifest['allocations'][1]['index'] = True
    elif fault == 'line': manifest['allocations'][1]['line'] = 'O1'
    elif fault == 'guard': manifest['allocations'][1]['guard']['nbytes'] -= 65536
    elif fault == 'payload': manifest['allocations'][1]['payload']['offset'] += 65536
    elif fault == 'partition-indices': manifest['partitions']['M1'] = [0, 2, 3, 4]
    elif fault == 'pool-bool': document['pool']['schema'] = True
    elif fault == 'overlap': manifest['allocations'][1]['block']['offset'] = 0
    mf = tmp_path / 'manifest.json';mf.write_text(json.dumps(manifest))
    document['source_manifest_sha256'] = hashlib.sha256(mf.read_bytes()).hexdigest()
    document['session_manifest'] = str(mf)
    path = tmp_path / 'slots.json';path.write_text(json.dumps(document))
    entries = []
    def forbidden(*args, **kwargs):
        entries.append('unexpected SDK/socket')
        raise AssertionError('invalid metadata reached SDK/socket')
    monkeypatch.setattr(sdk, 'accepted_sdk', forbidden)
    import socket
    monkeypatch.setattr(socket, 'socket', forbidden)
    config = SimpleNamespace(shared_slots_path=str(path), sdk_sha256=sdk.SDK_SHA256,
                             pool_id=document['pool']['pool_id'], pool_epoch=document['pool']['pool_epoch'], pool_endpoint=manifest['endpoint'])
    with pytest.raises(ValueError):bindings_document(path)
    with pytest.raises(ValueError):sdk.register_managed_pool(config)
    with pytest.raises(ValueError):register_partition_client(path, 'not-registered')
    assert entries == []
