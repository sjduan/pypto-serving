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
from pathlib import Path
import socket

from .managed_pool import PoolBlock
from .shared_directory import DirectoryAuthority


def bindings_document(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (set(value) != {'schema', 'resource_group', 'partition', 'pool', 'blocks', 'source_manifest_sha256', 'session_manifest'}
            or value['schema'] != 1 or not value['partition'] or not value['resource_group']
            or len(value['source_manifest_sha256']) != 64 or len(value['blocks']) != 4):
        raise ValueError('explicit M1 partition from the common public manifest is required')
    return value, tuple(PoolBlock(**b) for b in value['blocks'])


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
                       lambda operation: sdk.query_pool_operation(access.port.client.endpoint, operation))
    return blocks


def sdk_authority(endpoint, document):
    from .managed_pool_sdk import accepted_sdk
    sdk = accepted_sdk()
    expected, blocks = bindings_document(document)
    descriptor = expected['pool']
    fresh = sdk.describe_pool(endpoint, descriptor['pool_id'], descriptor['pool_epoch'])
    if fresh != descriptor:
        raise ValueError('common pool description changed before directory startup')
    return DirectoryAuthority(descriptor, blocks,
        lambda: sdk._exchange(endpoint, 'GetManagedPoolSnapshot',
                             {k: descriptor[k] for k in ('pool_id', 'pool_epoch', 'provider_session')})[0],
        lambda operation: sdk.query_pool_operation(endpoint, operation))
