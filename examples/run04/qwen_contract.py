# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stdlib-only Run04 configuration, admission and offline evidence contract."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from existing_pool import POOL_BYTES, check_catalog, retained_pool

BLOCK_BYTES = 20971520
SDK_SHA = '366f3ea883cec08efd62f23f9ebf2afa8546c02a777ff25fdf753fee1debff0e'
PINS = {'checkpoint': '7696877eeba446be18fb90a38eac3eb78bf83ea5',
        'P1_guard': '9d07e4b4d951bf5dadd49421b0fe7de9c5884594',
        'P1_runtime': '6c88583388aa7262a318cf23550e9eae9bf793b5',
        'lib': 'd77ac175065d06f8afe401ab230370dfb7703d3b',
        'pypto': '8007f1146a800d679301c5ea14bb644b863da6bf',
        'simpler': '508becb93e182a3035a14e3f9226929dbe994346',
        'ptoas': '0.70', 'ptoisa': '217bb98952a8f9004b95fe938da97d8d5f68f702'}
ASSETS = {'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
          *(f'model-{i:05d}-of-00008.safetensors' for i in range(1, 9))}
DEPLOYMENT = {'serving', 'qwen_lib', 'pypto', 'pypto.pypto_core', 'pypto.runtime', 'simpler',
              '_task_interface', 'uflow', 'uflow.managed_pool', 'uflow._managed_pool_wire',
              'native_backend', 'ptoas'}
CASES = ('small-native', 'normal-two-rounds-and-negatives', 'stop-load', 'stop-store')
HOLD_ID = 'serving-ab-npu-hold-20261105'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(config):
    if (config.get('schema') not in (1, 2) or config.get('pins') != PINS or config.get('sdk_sha256') != SDK_SHA
            or config.get('cases') != list(CASES) or config.get('rounds') != 2
            or config.get('slots') != 5 or config.get('device') != 10
            or config.get('utilization') != .8 or config.get('model_dir') != '/data/models/Qwen/Qwen3-14B'):
        raise ValueError('configuration differs from sealed single-card Run04 contract')
    if config['schema'] == 2:
        validate_B_domain(config)
    files = config['assets']
    if set(files) != ASSETS or any(not re.fullmatch('[0-9a-f]{64}', value) for value in files.values()):
        raise ValueError('exactly twelve complete model asset hashes are required')
    deployment = config['expected_deployment']
    if set(deployment) != DEPLOYMENT:
        raise ValueError('incomplete source/binary/loaded deployment manifest')
    for name, row in deployment.items():
        hashes = ([row['source_manifest_sha256']] if name in ('serving', 'qwen_lib') else
                  list(row.values()) if name == 'native_backend' else [row['sha256']])
        if any(not isinstance(h, str) or not re.fullmatch('[0-9a-f]{64}', h) for h in hashes):
            raise ValueError('invalid deployment content digest')
    if set(deployment['native_backend']) != {'host_path', 'aicpu_path', 'aicore_path'}:
        raise ValueError('native binary identity missing')
    return config


def validate_B_domain(config):
    if (config.get('target_host') != 'serving-b-sj' or config.get('container') != 'openeuler-2403-DS'
            or config.get('budget') != {'task_total_cap_bytes': 268435456,
                'retained_A_charge_floor_bytes': 134217728, 'B_Catalog_hard_budget_cap_bytes': 134217728,
                'B_normal_pool_bytes': 134217728, 'concurrent_new_B_pools_max': 1}
            or config.get('pool_history_cap') != 4):
        raise ValueError('B-only domain must retain A responsibility and a real 128MiB B hard cap')
    identity = config.get('expected_execution_identity', {})
    if set(identity) != {'hostname', 'boot_id', 'pid_namespace'} or any(not isinstance(s, str) or not s
            for s in identity.values()):
        raise ValueError('sealed B hostname/boot/pid namespace required')


def cli_arguments(config, endpoint, pool_id, pool_epoch):
    """One production CLI; endpoint/identity come from the actual new UF descriptor."""
    validate_config(config)
    if not all(isinstance(s, str) and s for s in (endpoint, pool_id, pool_epoch)):
        raise ValueError('explicit UFlow registration identity required')
    return ['--model', config['model_dir'], '--backend', 'npu', '--platform', 'a2a3',
            '--device', str(config['device']), '--max-model-len', '512', '--max-num-seqs', '16',
            '--max-num-batched-tokens', '512', '--long-prefill-token-threshold', '128',
            '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16', '--block-size', '128',
            '--ring-dep-pool', '262144', '--ring-task-window', '262144', '--ring-heap', '1073741824',
            '--npu-memory-utilization', str(config['utilization']), '--enable-native-ddr-kv',
            '--native-ddr-kv-bytes', str(5 * BLOCK_BYTES), '--native-ddr-pool-endpoint', endpoint,
            '--native-ddr-pool-id', pool_id, '--native-ddr-pool-epoch', pool_epoch,
            '--native-ddr-sdk-sha256', SDK_SHA]


def plan(config, snapshot=None):
    validate_config(config)
    if config['schema'] == 1:
        check_catalog(snapshot, retained_pool(snapshot))
        if len(snapshot['pools']) + len(CASES) > 8:
            raise ValueError('four sequential normal pools would exceed retained Catalog history cap eight')
    elif snapshot is not None:
        from b_session import check_b_catalog
        check_b_catalog(snapshot, normal_id=None)
    return {'schema': 1, 'mode': 'plan', 'status': 'NOT-RUN', 'config_sha256': digest(config),
            'runtime_imported': False, 'device_admission': False, 'D1_acceptance': False,
            'layout': {'pool_bytes': POOL_BYTES, 'quarantine_charged_bytes': POOL_BYTES,
                       'global_cap_bytes': 2 * POOL_BYTES, 'slot_bytes': BLOCK_BYTES,
                       'guard_bytes_per_allocation': 2097152, 'slots': 5,
                       'token_starts': [0, 128], 'tokens_per_block': 128, 'layer_sides': 80},
            'domain': ('new single B Catalog, A retained floor is historical responsibility only'
                       if config['schema'] == 2 else 'original retained A Catalog (historical plan only)'),
            'B_Catalog_cap_bytes': 134217728 if config['schema'] == 2 else None,
            'argv': cli_arguments(config, '<same-Catalog-endpoint>', '<normal-pool-id>', '<normal-pool-epoch>'),
            'cases': [{'name': name, 'status': 'NOT-RUN', 'depends_on': list(CASES[:index]),
                       'expected': ['catalog/normal-descriptor.json', 'resource-before.json',
                                    'logs/probe.log', 'result.json', 'normal-cleanup.json']}
                      for index, name in enumerate(CASES)],
            'normal_round_evidence': ['startup.compatibility_json', 'schedule_events', 'coordinator_trace',
                                      'rounds[0..1].blocks[0..1].{source,overwritten,committed,restored}',
                                      'rounds[0..1].pool_before/after', 'http', 'shutdown_report'],
            'pending': ['same-whole-Buffer native parameter validation', 'actual source/binary/loaded identity',
                        '12 complete assets at model startup', 'two token-0 contiguous block model rounds',
                        'HBM/pool/double miss; direct+HTTP; all bytes/guards; suffix+decode',
                        'real cancel/stale/short receipt paths', 'passively observed native pending LOAD/STORE stop',
                        'actual detach/unmap/release, old quarantine remains charged'],
            'stop_policy': 'identity/owner/pin/Unknown mismatch stops; no other daemon/card/pool retry'}


def require_release(hold, release):
    """Fail closed BEFORE importing any execution module, reserving or spawning."""
    if (hold.get('state') != 'paused_by_latest_explicit_human' or release is None
            or release.get('state') != 'released_by_explicit_human'
            or release.get('supersedes') != HOLD_ID
            or release.get('new_worker_init_native_submit_model_tests_device_reservations_allowed') is not True
            or not release.get('source', {}).get('message_id') or not release['source'].get('thread_id')):
        raise PermissionError('A/B device HOLD; a new explicit human environment-available release is required')
    if datetime.fromisoformat(release['recorded_at_utc'].replace('Z', '+00:00')) <= datetime.fromisoformat(
            hold['recorded_at_utc'].replace('Z', '+00:00')):
        raise PermissionError('historical release cannot supersede the current hold')


def require_B_execution(config, release, actual):
    validate_config(config)
    if config['schema'] != 2:
        raise PermissionError('historical A configuration cannot execute under B-only release')
    expected = release.get('expected_B_execution_identity', {})
    identity = {k: expected.get(k) for k in ('hostname', 'boot_id', 'pid_namespace')}
    if (release.get('allowed_hosts') != ['serving-b-sj'] or release.get('forbidden_hosts') != ['serving-a-sj']
            or release.get('A_device_and_remote_work_allowed') is not False
            or release.get('container') != config['container'] or release.get('budget') != config['budget']
            or config['expected_execution_identity'] != identity or actual != identity
            or release.get('original_A_quarantine', {}).get('retained_charged_bytes') != 134217728
            or release.get('original_A_quarantine', {}).get('fresh_A_status_not_queried') is not True):
        raise PermissionError('actual machine/namespace or B-only release/budget differs; do not use A')


def execution_identity():
    """Local OS metadata only; called in run/consumer before any execution imports."""
    import os
    import socket
    return {'hostname': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'pid_namespace': os.readlink('/proc/self/ns/pid')}


def check_startup(config, startup, descriptor):
    validate_config(config)
    compat = startup['compatibility_json']
    if (startup['max_entries'] != 5 or startup['capacity_bytes'] != 5 * BLOCK_BYTES
            or startup['layout']['page_size'] != 128 or startup['layout']['num_layers'] != 40
            or compat['scope'] != 'managed-pool-compatible-data-local-owner'
            or compat['worker_epoch'] != startup['worker_epoch']
            or compat['deployment'] != config['expected_deployment']):
        raise ValueError('actual loaded source/binary/owner compatibility differs from sealed config')
    files = compat['assets']['files']
    if (set(files) != ASSETS or any(row.get('sha256') != config['assets'][name]
            or row.get('evidence') != 'full-asset-content-at-startup' for name, row in files.items())):
        raise ValueError('model startup did not fully verify the twelve sealed assets')
    pool = startup['pool_identity']
    if any(pool[key] != descriptor[key] for key in ('pool_id', 'pool_epoch', 'provider_session', 'nbytes')):
        raise ValueError('loaded owner mapped a different UFlow resource')
    if (pool['device'], pool['inode']) != (descriptor['canonical']['dev'], descriptor['canonical']['inode']):
        raise ValueError('loaded physical UFlow mapping changed')
    # Recompute the production content namespace without importing production.
    layout = {key: value for key, value in compat['layout'].items() if key != 'num_pages'}
    data = {'schema': 1, 'model': compat['model'], 'asset_content': config['assets'],
            'tokenizer_type': compat['tokenizer_type'], 'adapter': compat['adapter'],
            'position': compat['position'], 'layout': layout, 'tensor_parallel': compat['tensor_parallel'],
            'deployment': compat['deployment']}
    if digest(data) != startup['namespace_digest']:
        raise ValueError('startup namespace does not bind the actual loaded compatibility')
