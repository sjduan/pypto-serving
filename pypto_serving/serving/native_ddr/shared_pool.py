# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordinary accepted-SDK validation; no create/destroy/admin capability."""
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import socket
import uuid

from .managed_pool import BLOCK_BYTES, GUARD_BYTES, PoolBlock, prefix_content_id
from .shared_directory import DirectoryAuthority


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate metadata key')
        result[key] = value
    return result


def _shape(value, keys):
    if type(value) is not dict or set(value) != set(keys.split()):
        raise ValueError('shared partition metadata schema differs')


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('shared partition requires strict integer metadata')


def _text(value):
    if type(value) is not str or not value or len(value) > 4096:
        raise ValueError('shared partition requires bounded text metadata')


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('shared partition requires a SHA256 seal')


def validate_partition(value, manifest):
    """Pure metadata preflight. This grants no SDK registration or range access."""
    _shape(value, 'schema resource_group partition pool blocks source_manifest_sha256 session_manifest')
    if type(value['schema']) is not int or value['schema'] != 1 or value['partition'] != 'M1':
        raise ValueError('only the sealed M1 schema/partition is supported')
    _text(value['resource_group'])
    _text(value['session_manifest'])
    _digest(value['source_manifest_sha256'])
    _shape(manifest, 'allocations bootstrap_client bootstrap_unmapped capacity container control_endpoint '
        'controller_birth coordination coordination_root endpoint frozen_inputs group helper_root host host_alias '
        'manifest_version owner_births owner_root partitions pool_descriptor protocol provider_loaded schema state')
    if (type(manifest['schema']) is not int or manifest['schema'] != 1
            or type(manifest['manifest_version']) is not int or manifest['manifest_version'] != 1
            or manifest['state'] != 'ready' or manifest['group'] != 'phasef-parallel-20261106'
            or manifest['group'] != value['resource_group']
            or manifest['host_alias'] != 'serving-b-sj' or manifest['container'] != 'openeuler-2403-DS'):
        raise ValueError('sealed common manifest group/schema/B domain differs')
    _shape(manifest['partitions'], 'M1 O1')
    for name, expected in (('M1', [1, 2, 3, 4]), ('O1', [0])):
        part = manifest['partitions'][name]
        if type(part) is not list or part != expected or any(type(i) is not int for i in part):
            raise ValueError('common M1/O1 allocation partition differs')
    expected_capacity = {'A_queried': False, 'A_retained_historical_bytes': 134217728,
        'B_budget_bytes': 134217728, 'B_charged_bytes': 134217728, 'B_creation_reserved_bytes': 0,
        'allocated_range_bytes': 115343360, 'task_total_responsibility_bytes': 268435456,
        'unassigned_still_charged_bytes': 18874368}
    if json.dumps(manifest['capacity'], sort_keys=True) != json.dumps(expected_capacity, sort_keys=True):
        raise ValueError('common physical budget/retained responsibility differs')
    _shape(manifest['host'], 'boot_id hostname mount_namespace pid_namespace')
    for name in ('boot_id', 'hostname', 'mount_namespace', 'pid_namespace'):
        _text(manifest['host'][name])
    for name in ('endpoint', 'control_endpoint', 'helper_root', 'owner_root', 'coordination_root'):
        _text(manifest[name])
    descriptor = manifest['pool_descriptor']
    _shape(descriptor, 'binding_id canonical nbytes placement_id pool_epoch pool_id provider_kind provider_session '
        'resource_generation resource_id schema state')
    if (type(descriptor['schema']) is not int or descriptor['schema'] != 1 or descriptor['state'] != 'Allocated'
            or descriptor['provider_kind'] != 'posix_shm_simpler'):
        raise ValueError('common pool schema/state/provider differs')
    for name in ('binding_id', 'placement_id', 'resource_generation', 'resource_id', 'nbytes'):
        _integer(descriptor[name], 1)
    for name in ('pool_id', 'pool_epoch', 'provider_session'):
        _text(descriptor[name])
    canonical = descriptor['canonical']
    _shape(canonical, 'access address_space backend_kind body_hex buffer_id dev generation inode kind '
        'native_descriptor nbytes owner_instance_id owner_worker_path_id')
    for name in ('access', 'address_space', 'backend_kind', 'buffer_id', 'dev', 'generation', 'inode',
                 'nbytes', 'owner_worker_path_id'):
        _integer(canonical[name])
    for name in ('body_hex', 'kind', 'native_descriptor', 'owner_instance_id'):
        _text(canonical[name])
    for name in ('buffer_id', 'dev', 'generation', 'inode', 'nbytes'):
        _integer(canonical[name], 1)
    if (canonical['access'] != 2 or canonical['address_space'] != 0 or canonical['backend_kind'] != 1
            or canonical['kind'] != 'posix_shm_simpler' or canonical['native_descriptor'] != 'validated-Simpler-Buffer-ABI-v1'):
        raise ValueError('canonical managed backing ABI differs')
    if descriptor['nbytes'] != 134217728 or canonical['nbytes'] != descriptor['nbytes']:
        raise ValueError('common backing must remain the original 128MiB pool')
    # JSON equality alone accepts True == 1: compare the canonical typed encoding.
    if json.dumps(value['pool'], sort_keys=True) != json.dumps(descriptor, sort_keys=True):
        raise ValueError('partition pool differs from its sealed public manifest')
    rows = manifest['allocations']
    if type(rows) is not list or len(rows) != 5:
        raise ValueError('five actual common allocations are required')
    selected, intervals, identities = [], [], set()
    for index, row in enumerate(rows):
        _shape(row, 'block guard index line payload')
        if type(row['index']) is not int or row['index'] != index or row['line'] != ('O1' if index == 0 else 'M1'):
            raise ValueError('allocation index/line differs from the sealed partition')
        block = row['block']
        _shape(block, 'block_id generation nbytes offset pool_epoch pool_id provider_session')
        for name in ('block_id', 'generation', 'nbytes'):
            _integer(block[name], 1)
        _integer(block['offset'])
        if (any(block[k] != descriptor[k] for k in ('pool_id', 'pool_epoch', 'provider_session'))
                or block['nbytes'] != BLOCK_BYTES + GUARD_BYTES or block['offset'] % 65536
                or block['offset'] + block['nbytes'] > descriptor['nbytes'] or block['block_id'] in identities):
            raise ValueError('allocation identity/generation/bounds differ')
        identities.add(block['block_id'])
        intervals.append((block['offset'], block['offset'] + block['nbytes']))
        for name, relative, size in (('guard', 0, GUARD_BYTES), ('payload', GUARD_BYTES, BLOCK_BYTES)):
            region = row[name]
            _shape(region, 'byte_offset_in_block byte_value initial_commit nbytes offset sha256')
            for field in ('byte_offset_in_block', 'byte_value', 'nbytes', 'offset'):
                _integer(region[field])
            _digest(region['sha256'])
            commit = region['initial_commit']
            _shape(commit, 'committed content_id content_version lease_id')
            _integer(commit['content_version'], 1)
            _text(commit['content_id'])
            _text(commit['lease_id'])
            if (commit['committed'] is not True or region['byte_offset_in_block'] != relative
                    or region['offset'] != block['offset'] + relative or region['nbytes'] != size
                    or region['byte_value'] > 255):
                raise ValueError('actual guard/payload layout differs')
        if index in manifest['partitions']['M1']:
            selected.append(PoolBlock(block['pool_id'], block['pool_epoch'], block['block_id'], block['generation'],
                                      block['offset'], block['nbytes']))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError('common allocation bounds overlap')
    if type(value['blocks']) is not list or len(value['blocks']) != 4:
        raise ValueError('four exact M1 allocations are required')
    for block in value['blocks']:
        _shape(block, 'pool_id pool_epoch block_id generation byte_offset nbytes')
        for name in ('block_id', 'generation', 'nbytes'):
            _integer(block[name], 1)
        _integer(block['byte_offset'])
    if value['blocks'] != [asdict(b) for b in selected]:
        raise ValueError('document blocks are not the exact ordered public M1 partition')
    return tuple(selected)


def partition_metadata(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_object)
    # Check these fields before following any path from unvalidated metadata.
    _shape(value, 'schema resource_group partition pool blocks source_manifest_sha256 session_manifest')
    _text(value['session_manifest'])
    _digest(value['source_manifest_sha256'])
    raw = Path(value['session_manifest']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['source_manifest_sha256']:
        raise ValueError('external session public manifest changed')
    manifest = json.loads(raw, object_pairs_hook=_object)
    return value, validate_partition(value, manifest), manifest


def bindings_document(path):
    document, blocks, _ = partition_metadata(path)
    return document, blocks


def register_partition_client(path, session):
    """Register one actual ordinary SDK session with the external lifecycle owner.

    This bounded metadata call grants no access and never retires a live client.
    The controller, after all model/host consumers close, retires the whole line.
    """
    document, _ = bindings_document(path)
    raw = Path(document['session_manifest']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != document['source_manifest_sha256']:
        raise ValueError('external session public manifest changed')
    manifest = json.loads(raw)
    if (manifest['group'] != document['resource_group'] or manifest['pool_descriptor'] != document['pool']):
        raise ValueError('external lifecycle owner describes a different resource group/pool')
    data = json.dumps({'schema': 1, 'group': manifest['group'], 'op': 'register-client',
                       'line': document['partition'], 'client_session': session}).encode() + b'\n'
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(35)
        sock.connect(manifest['control_endpoint'])
        sock.sendall(data)
        with sock.makefile('rb') as stream:
            answer = stream.readline(65537)
    if len(answer) > 65536 or not answer.endswith(b'\n'):
        raise ValueError('external session response exceeds its bounded metadata protocol')
    value = json.loads(answer)
    if value.get('status') != 'ok':
        raise ValueError('external owner did not register the actual ordinary SDK session')
    return value['result']


def load_bindings(path, access):
    document, blocks = bindings_document(path)
    expected = document['pool']
    actual = access.port.client.view.descriptor
    from .managed_pool_sdk import accepted_sdk
    sdk = accepted_sdk()
    sdk._pool_descriptor(expected)
    if any(actual[k] != expected[k] for k in expected):
        raise ValueError('registered access differs from the sealed common pool descriptor')
    DirectoryAuthority(expected, blocks, access.port.client.snapshot,
                       lambda key, value: None)
    return blocks


def sdk_authority(endpoint, document, access):
    from .managed_pool_sdk import accepted_sdk
    expected, blocks = bindings_document(document)
    sdk = accepted_sdk()
    descriptor = expected['pool']
    fresh = sdk.describe_pool(endpoint, descriptor['pool_id'], descriptor['pool_epoch'])
    if fresh != descriptor:
        raise ValueError('common pool description changed before directory startup')
    def confirm(key, content):
        # Operation IDs are peer/PID scoped in the accepted UF journal. A
        # different directory PID must not pretend that a NotFound query is a
        # missing commit. Independently acquire the current real reader instead.
        content.validate(access.identity, content_id=prefix_content_id(key))
        lease = access.acquire(content.block, 'r', committed=content, reject_stale=True)
        access.port.revalidate(lease)
        ack = access.complete_read(lease, uuid.uuid4().hex)
        return {'lease_id': lease.lease_id, 'lease_generation': lease.lease_generation,
                'byte_offset': lease.byte_offset, 'nbytes': lease.nbytes,
                'content_version': lease.content_version, 'content_id': lease.content_id,
                'release_ack': asdict(ack), 'payload_read_or_transferred': False}
    return DirectoryAuthority(descriptor, blocks,
        lambda: sdk._exchange(endpoint, 'GetManagedPoolSnapshot',
                             {k: descriptor[k] for k in ('pool_id', 'pool_epoch', 'provider_session')})[0],
        confirm)
