# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure Run06 scenario schema and finite operation-record budget. No runtime imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from pypto_serving.serving.native_ddr.shared_pool import partition_metadata

WARMUP = 3
MEASURED = 20
PROMPT_TOKENS = 257
PAYLOAD_BYTES = 20971520
MIN_FREE_HBM_MIB = 40000  # Accepted D1 Qwen admission; never the O1 4GiB microbenchmark gate.
SDK = '366f3ea883cec08efd62f23f9ebf2afa8546c02a777ff25fdf753fee1debff0e'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def operation_budget(replicas):
    if type(replicas) is not int or replicas not in (2, 4):
        raise ValueError('two or four independent TP1 replicas required')
    peers = replicas - 1
    requests = peers * (8 + WARMUP + MEASURED)
    rows = {
        'RegisterPoolClient_and_AckPoolMapping': 2 * (replicas + 2),
        'AttachPoolChild_and_DetachPoolChild_total_max3_each_model': 6 * replicas,
        'BeginPoolUnregister_and_AckPoolUnmapped': 2 * (replicas + 2),
        'two_LOAD_Acquire_Release_per_shared_request': 4 * requests,
        'first_two_import_R_Acquire_Release_per_consumer': 4 * peers,
        'source2_plus_overwrite_max4_plus_doublemiss2_each_STORE_and_directory_confirm': 4 * (6 + 2 * peers),
        'source_and_8functional_plus_final_DDR_diagnostic_R_Acquire_Release': 4 + 36 * peers,
        'one_four_guard_capture_per_model_R_Acquire_Release': 8 * replicas,
        'observer_two_R_pins_Acquire_Release': 4,
        'optional_router_two_request_worstcase_LOADs': 8 if replicas==4 else 0,
    }
    total = sum(rows.values())
    return {'schema': 1, 'replicas': replicas, 'shared_requests': requests,
        'cold_native_disabled_requests': replicas * (WARMUP + MEASURED),
        'new_journal_records_by_path': rows, 'normal_records_bound': total,
        'normal_cleanup_already_included': True, 'terminal_recovery_headroom_at8192': 128,
        'required_shared_record_allowance': total,
        'first_failure': 'stop all new admissions, retain actual owner/child/lease; no new-ID retry or replacement',
        'admission_stops_at_ALL_record_count': 4096, 'cleanup_stops_at_ALL_record_count': 8192,
        'readonly_zero_new_records': ['GetManagedPoolSnapshot', 'DescribeManagedPool', 'QueryPoolOperation',
                                     'SdkManagedPoolPort.revalidate uses snapshots'],
        'no_allocate_create_destroy_shutdown': True,
        'source_paths': ['sdk/python/uflow/managed_pool.py:255-304,328-483',
                         'crates/uf-daemon/src/pool_service.rs:324-363'],
        'count_semantics': 'All mutation records count toward BOTH bounds, including rejected operations. '
                           'Known idempotent same-operation replay is not a new record. Never retry with new ID.',
        'unknown_or_exhausted_allowance': 'STOP before any SDK mutation; never reduce samples or replace Catalog'}


def validate(config):
    required = {'schema', 'bindings', 'manifest_sha256', 'root', 'replicas', 'devices', 'model',
        'model_assets', 'source_manifest', 'source_manifest_sha256', 'native_env', 'native_env_sha256',
        'host_env', 'host_env_sha256', 'operation_allowance', 'device_release', 'router'}
    if type(config) is not dict or set(config) != required or type(config['schema']) is not int or config['schema'] != 1:
        raise ValueError('exact scenario schema1 required')
    budget = operation_budget(config['replicas'])
    document, blocks, manifest = partition_metadata(config['bindings'])
    if config['manifest_sha256'] != document['source_manifest_sha256']:
        raise ValueError('scenario and partition manifest seals disagree')
    root = PurePosixPath(config['root'])
    if (not root.is_absolute() or '..' in root.parts or
            not str(root).startswith('/workspace/uflow/runs/phasef-parallel-20261106/m1/')):
        raise ValueError('new owned B M1 evidence root required')
    devices = config['devices']
    if type(devices) is not list or len(devices) != config['replicas']:
        raise ValueError('one independent physical placement per actor')
    for d in devices:
        if (type(d) is not dict or set(d) != {'physical', 'board', 'chip', 'bdf', 'port'}
                or any(type(d[k]) is not int for k in ('physical', 'board', 'chip', 'port'))
                or d['physical'] not in range(16) or d['physical'] in (10, 11)
                or d['board'] not in range(8) or d['chip'] not in (0, 1)
                or not isinstance(d['bdf'], str) or not d['bdf'] or d['port'] not in range(18920, 18924)):
            raise ValueError('complete candidate physical/board/chip/BDF/port required; O1 domain excluded')
    if (len({d['physical'] for d in devices}) != len(devices) or len({d['port'] for d in devices}) != len(devices)
            or len({d['bdf'] for d in devices}) != len(devices)):
        raise ValueError('replica physical placement/port/domain duplicated')
    if config['model'] != '/data/models/Qwen/Qwen3-14B':
        raise ValueError('existing fixed Qwen3-14B assets required')
    assets = {'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
              *(f'model-{i:05d}-of-00008.safetensors' for i in range(1, 9))}
    if type(config['model_assets']) is not dict or set(config['model_assets']) != assets:
        raise ValueError('all twelve complete asset hashes required')
    hashes = list(config['model_assets'].values()) + [config[k] for k in
        ('source_manifest_sha256', 'native_env_sha256', 'host_env_sha256')]
    if any(type(h) is not str or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes):
        raise ValueError('complete source/environment/asset SHA256 closure required')
    if config['router'] is not None and (config['replicas'] != 4 or config['router'] != {'port':18929}):
        raise ValueError('ordinary static four-service router must use its reserved loopback18929')
    return document, blocks, manifest, budget


def cli(config, device, *, shared):
    document, _, manifest, _ = validate(config)
    args = ['--model', config['model'], '--platform', 'a2a3', '--backend', 'npu', '--device', str(device['physical']),
        '--host', '127.0.0.1', '--port', str(device['port']), '--max-model-len', '512', '--max-num-seqs', '16',
        '--max-num-batched-tokens', '512', '--long-prefill-token-threshold', '128', '--dtype', 'bfloat16',
        '--kv-cache-dtype', 'bfloat16', '--block-size', '128', '--ring-dep-pool', '262144',
        '--ring-task-window', '262144', '--ring-heap', '1073741824', '--npu-memory-utilization', '0.8']
    if not shared:
        return args + ['--no-enable-native-ddr-kv']
    descriptor = document['pool']
    return args + ['--enable-native-ddr-kv', '--native-ddr-kv-bytes', str(4 * PAYLOAD_BYTES),
        '--native-ddr-pool-endpoint', manifest['endpoint'], '--native-ddr-pool-id', descriptor['pool_id'],
        '--native-ddr-pool-epoch', descriptor['pool_epoch'], '--native-ddr-sdk-sha256', SDK,
        '--shared-ddr-directory', config['root'] + '/d.sock', '--shared-ddr-directory-epoch', 'm1-' + Path(config['root']).name,
        '--shared-ddr-slots', config['bindings']]


def plan(config):
    document, blocks, manifest, budget = validate(config)
    return {'schema': 1, 'status': 'NOT-RUN', 'scope': 'pure schema/plan/operation-budget; no SDK/device/runtime',
        'config_sha256': digest(json.dumps(config, sort_keys=True).encode()), 'budget': budget,
        'source_manifest_sha256':config['source_manifest_sha256'],
        'placements': config['devices'], 'manifest_sha256': document['source_manifest_sha256'],
        'shared_blocks': [b.block_id for b in blocks], 'pool': document['pool'],
        'minimum_free_HBM_MiB_per_Qwen': MIN_FREE_HBM_MIB,
        'steps': ['preflight source/assets/identity/budget BEFORE model/Worker/SDK mutation',
                  'native-off baseline per physical actor:23 requests each with allocator cold reset',
                  'one metadata directory, producer and independent never-STORE consumers',
                  'producer two contiguous token0 STOREs then source/DDR captures',
                  'each consumer2direct+2HTTP, fresh cold HBM and two actual LOADs each',
                  'observer two true DDR R pins; original allocator source overwrite, all80sides/128rows',
                  'producer normal engine stop and true UF unmap; observer releases original R pins',
                  'consumers2direct+2HTTP after producer exit, independent full-byte restores',
                  'shared3warmup+20measured per consumer: every sample fresh cold HBM, fresh native LOADs',
                  'HBM priority, different doublemiss, optional ordinary four-service router and affinity',
                  'normal all-model/native/SDK/child drain, directory close; keep common owner and do not line-retire'],
        'fixed_workload': {'prompt_tokens': 257, 'prefix_blocks': 2, 'tokens_per_block': 128,
            'bytes_per_block': PAYLOAD_BYTES, 'layer_sides': 80, 'max_new_tokens': 2, 'seed': 0,
            'warmup_each_arm': WARMUP, 'measured_each_arm': MEASURED},
        'runtime_start_permitted_by_this_plan': False, 'M1_acceptance': False,
        'resource_group': manifest['group']}
