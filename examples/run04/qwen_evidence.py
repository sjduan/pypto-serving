# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offline structural/byte checker. Results remain subject to capture-origin review."""
import hashlib
import json
from pathlib import Path
import re

from qwen_contract import BLOCK_BYTES, check_startup
from verify_bytes import map_capture, verify_guards, verify_roundtrip


def require(value, message):
    if not value:
        raise ValueError(message)


def eligible_block(prompt_tokens, index, token_start, tokens):
    # The last prompt token must be computed; a full final prompt page is not
    # reusable as a full prefix. Never round a partial tail up to 128.
    require(all(type(v) is int for v in (prompt_tokens, index, token_start, tokens))
            and prompt_tokens > 0 and 0 <= index < (prompt_tokens - 1) // 128
            and token_start == index * 128 and tokens == 128,
            'partial tail/last prompt token cannot be promoted to a prefix block')


def steps(report, request):
    return [s for e in report['schedule_events'] for s in e['requests'] if s['request_id'] == request]


def capture_path(root, row):
    original = Path(row['file'])
    path = original if original.is_file() else Path(root) / 'diagnostics' / original.name
    require(path.is_file(), f'missing raw capture: {row["file"]}')
    require(path.stat().st_size == row['bytes'], 'raw capture has wrong length')
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    require(digest.hexdigest() == row['sha256'], 'raw capture content hash mismatch')
    return path


def log_records(raw, marker):
    return [json.loads(line.split(marker, 1)[1]) for line in raw.splitlines() if marker + '{' in line]


def bind_round_event(summary, report, raw_log=None):
    """Bind full original records, never a projection of a self-consistent summary."""
    event = summary['event']
    originals = [row for row in report['coordinator_trace']
                 if row['event'] == event and row.get('job') == summary['job']]
    require(originals == [summary], 'normal summary differs from unique original coordinator event')
    if raw_log is None:
        return
    coordinators = log_records(raw_log, 'native_ddr_coordinator ')
    require([row for row in coordinators if row['event'] == event
             and row.get('job') == summary['job']] == [summary],
            'normal summary differs from unique raw coordinator log')
    if event != 'transfer_terminal':
        return
    producers = [row for row in log_records(raw_log, 'native_ddr_receipt ')
                 if row['kind'] == 'transfer' and row['identity'] == summary['job']
                 and row['status'] != 'ACCEPTED']
    stale = report.get('stale_original_transfer', {}).get('receipt')
    if stale in producers:
        # The final negative deliberately resends one original normal envelope.
        # Its unique, fully bound rejection is not a second native completion.
        event = {'job_id': summary['job'], 'worker_epoch': summary['worker_epoch'],
            'operation': summary['operation'], 'status': 'REJECTED', 'entry_refs': summary['entry_refs'],
            'transferred_bytes': 0, 'native_completed': False, 'stopped': True,
            'error': 'retired delivery sequence; replay rejected before effects',
            'delivery_sequence': summary['delivery_sequence'], 'pool_receipt': None}
        accepted = log_records(raw_log, 'native_ddr_trace ')
        before = sum(r['event'] == 'accepted' and r['time_ns'] < stale['time_ns'] for r in accepted)
        total = sum(r['event'] == 'accepted' for r in accepted)
        require(producers.count(stale) == 1 and stale['status'] == 'REJECTED'
                and stale['event'] == event and stale['error'] == event['error']
                and stale['delivery_sequence'] == summary['delivery_sequence']
                and stale['request'] is None and stale['pending_owner'] is None
                and stale['first_nonterminal_failure'] is None
                and stale['transfer_replay']['retired_through'] >= summary['delivery_sequence']
                and stale['time_ns'] > summary['time_ns']
                and report['stale_original_transfer']['native_runs_before_after'] == [before, total]
                and before == total > 0,
                'retired-envelope rejection differs from the unique original negative without native effects')
        producers.remove(stale)
    require(len(producers) == 1, 'normal terminal has missing/duplicate/conflicting producer receipts')
    producer = producers[0]
    expected = {'job_id': summary['job'], 'worker_epoch': summary['worker_epoch'],
        'operation': summary['operation'], 'status': summary['status'], 'entry_refs': summary['entry_refs'],
        'transferred_bytes': summary['bytes'], 'native_completed': summary['native_completed'],
        'stopped': summary['stopped'], 'error': summary['error'],
        'delivery_sequence': summary['delivery_sequence'], 'pool_receipt': summary['pool_receipt']}
    require(producer['event'] == expected and producer['error'] == summary['error']
            and producer['delivery_sequence'] == summary['delivery_sequence']
            and producer['status'] == summary['status'] and producer['time_ns'] <= summary['time_ns'],
            'producer typed event differs from coordinator terminal identity/content/order')
    request = producer['request']
    for key, value in {'job_id': summary['job'], 'worker_epoch': summary['worker_epoch'],
            'operation': summary['operation'], 'entry_refs': summary['entry_refs'],
            'page_ids': summary['pages'], 'page_lease_id': summary['page_lease_id'],
            'delivery_sequence': summary['delivery_sequence']}.items():
        require(request[key] == value, 'producer original request differs from normal terminal lease/pages')
    keys = request['entry_keys']
    require(len(keys) == 1 and keys[0]['namespace_digest'] == report['startup']['namespace_digest']
            and re.fullmatch('[0-9a-f]{64}', keys[0]['block_hash'])
            and summary['pool_receipt']['content']['content_id'] ==
                'qwen3-prefix-v1:' + keys[0]['namespace_digest'] + ':' + keys[0]['block_hash']
            and type(request['delivery_sequence']) is int and request['delivery_sequence'] > 0
            and type(request['retire_through']) is int and 0 <= request['retire_through'] < request['delivery_sequence'],
            'producer content namespace or delivery generation is stale')
    require((request['operation'] == 'STORE' and type(request['source_step_id']) is int
             and request['source_step_id'] > 0) or
            (request['operation'] == 'LOAD' and request['source_step_id'] is None),
            'producer request has no valid source step identity')


def check_round(round_, report, root, *, inspect_bytes, raw_log=None):
    require(round_['number'] in (0, 1) and round_['prompt_tokens'] == 257, 'wrong normal round/prefix length')
    require(round_['interface'] == ('direct' if round_['number'] == 0 else 'http'), 'wrong request interface')
    require(len(round_['blocks']) == 2, 'token-zero contiguous two-block evidence is incomplete')
    identity = report['startup']['pool_identity']
    for pool in (round_['pool_before'], round_['pool_after']):
        require(pool['identity'] == identity and pool['bytes'] == 134217728,
                'guard observation belongs to a different physical pool/client')
        require(pool['child_registry_matches_sdk'] is True and pool['registered_children'],
                'actual child registry must match SDK inherited mapping obligations')
        snapshot = pool['snapshot']
        clients = [c for c in snapshot['clients'] if c['client_session'] == identity['client_session']]
        require(len(clients) == 1 and clients[0]['state'] == 'Mapped' and clients[0]['closing'] is False,
                'guard capture lacks unique current non-closing SDK client')
        births = {b['pid']: b for b in pool['registered_child_births']}
        require(len(births) == len(pool['registered_children']) == len(clients[0]['children'])
                and sorted([[b['pid'], b['starttime_ticks']] for b in births.values()])
                    == sorted(pool['registered_children'])
                and all(c['unknown'] is False and c['birth'] == births.get(c['birth']['pid'])
                    and all(c['birth'][k] == clients[0]['birth'][k]
                            for k in ('uid', 'gid', 'pid_namespace_dev', 'pid_namespace_ino'))
                    for c in clients[0]['children']), 'guard child evidence differs from actual SDK snapshot')
        allocations = pool['snapshot']['ranges']['allocations']
        require(pool['snapshot']['ranges']['leases'] == []
                and pool['write_ranges'] == [[a['offset'] + 2097152, BLOCK_BYTES] for a in allocations]
                and all(a['nbytes'] == BLOCK_BYTES + 2097152 for a in allocations)
                and 0 < len(allocations) <= 5, 'guard mask differs from actual finite UF allocations')
    require(round_['pool_before']['snapshot']['pool_epoch'] == round_['pool_after']['snapshot']['pool_epoch'],
            'guard observation crosses a pool epoch')
    require(round_['sentinel_before'] and [p['sha256'] for p in round_['sentinel_before']]
            == [p['sha256'] for p in round_['sentinel_after']], 'unrelated held HBM pages changed')
    for label, count in (('cold_request', 257), ('hbm_request', 1), ('restore_request', 1)):
        scheduled = steps(report, round_[label])
        require(sum(s['tokens'] for s in scheduled if s['is_prefill']) == count,
                'cold/HBM/pool prefill work does not match contiguous prefix reuse')
        require(any(not s['is_prefill'] for s in scheduled), 'actual suffix/decode schedule is missing')
    results = []
    targets, sources = set(), {b['source_page'] for b in round_['blocks']}
    require(len(sources) == 2, 'two source blocks alias')
    for index, block in enumerate(round_['blocks']):
        eligible_block(round_['prompt_tokens'], block['index'], block['token_start'], block['tokens'])
        require(block['index'] == index, 'prefix blocks are missing/reordered')
        ref, store, load = block['entry_ref'], block['store'], block['load']
        require(ref['worker_epoch'] == report['startup']['worker_epoch'], 'stale worker epoch')
        for operation, terminal in (('STORE', store), ('LOAD', load)):
            bind_round_event(terminal, report, raw_log)
            require(terminal['operation'] == operation and terminal['entry_refs'] == [ref]
                    and terminal['worker_epoch'] == ref['worker_epoch'] and terminal['status'] == 'SUCCEEDED'
                    and terminal['native_completed'] is True and terminal['stopped'] is True
                    and terminal['bytes'] == BLOCK_BYTES and terminal['error'] is None,
                    'incomplete or mismatched native terminal event')
            receipt = terminal['pool_receipt']
            require(receipt['pool'] == identity and receipt['operation'] == operation
                    and receipt['page_lease_id'] == terminal['page_lease_id']
                    and receipt['lease_id'] and type(receipt['lease_generation']) is int
                    and receipt['lease_generation'] > 0, 'stale pool/page/range lease receipt')
        committed = block['committed']['committed']
        require(set(committed) == {'pool_id', 'pool_epoch', 'provider_session', 'block', 'byte_offset',
                'nbytes', 'content_version', 'content_id', 'operation_id', 'writer_lease_id'},
                'complete typed committed allocation/content identity is required')
        allocation = committed['block']
        require(allocation['pool_id'] == identity['pool_id'] and allocation['pool_epoch'] == identity['pool_epoch']
                and type(allocation['generation']) is int and allocation['generation'] > 0
                and committed['operation_id'] and any(
                    (a['block_id'], a['generation'], a['offset'], a['nbytes']) == (
                        allocation['block_id'], allocation['generation'], allocation['byte_offset'], allocation['nbytes'])
                    for a in round_['pool_after']['snapshot']['ranges']['allocations'])
                and committed['byte_offset'] == allocation['byte_offset'] + 2097152,
                'Committed allocation differs from real pool range generation/layout')
        require(store['pool_receipt']['content'] == committed == load['pool_receipt']['content'],
                'LOAD changed the Committed content identity/version/allocation')
        require(committed['pool_id'] == identity['pool_id'] and committed['pool_epoch'] == identity['pool_epoch']
                and committed['provider_session'] == identity['provider_session']
                and committed['nbytes'] == BLOCK_BYTES and committed['content_version'] > 0
                and committed['byte_offset'] % 65536 == 0
                and committed['writer_lease_id'] == store['pool_receipt']['lease_id']
                and store['pool_receipt']['release_ack'] is None, 'STORE has no exact committed writer receipt')
        ack = load['pool_receipt']['release_ack']
        require(ack is not None and ack['applied'] is True and ack['operation'] == 'release'
                and ack['pool'] == identity and ack['lease_id'] == load['pool_receipt']['lease_id']
                and ack['lease_generation'] == load['pool_receipt']['lease_generation']
                and ack['operation_id'] and ack['lease_id'] != committed['writer_lease_id'],
                'LOAD reader was not actually acknowledged/released')
        require(block['committed']['pool_identity'] == identity
                and block['committed']['generation'] == ref['slot_generation']
                and block['committed']['entry_id'] == ref['entry_id'], 'captured DDR entry names another lease')
        source, target = block['source_page'], block['target_page']
        require(store['pages'] == [source] and load['pages'] == [target]
                and target not in sources and target not in targets
                and block['source']['page'] == source and block['overwritten']['page'] == source
                and block['restored']['page'] == target, 'LOAD destination is not a distinct original-arena page')
        targets.add(target)
        overwrite = next(r for r in report['requests'] if r['request_id'] == block['overwrite_request'])
        original_request = next(r for r in report['requests'] if r['request_id'] == round_['cold_request'])
        original_tokens = original_request['prompt_token_ids'][index * 128:(index + 1) * 128]
        emitted = [o['token_id'] for o in overwrite['outputs'] if o.get('token_id') is not None]
        require(len(original_tokens) == 128 and emitted and all(a != b for a, b in zip(
                    overwrite['prompt_token_ids'] + emitted[:1], original_tokens)),
                'overwrite stimulus reuses original token values; all rows must have distinct actual model inputs')
        schedule = steps(report, block['overwrite_request'])
        require(len(overwrite['prompt_token_ids']) == 127 and all(s['pages'] == [source] for s in schedule)
                and {position for s in schedule for position in range(s['computed_before'],
                     s['computed_before'] + s['tokens'])} == set(range(128)),
                'ordinary allocator/model did not overwrite all 128 source positions')
        require(store['time_ns'] < overwrite['begin_ns'] < overwrite['completed_ns'] < load['time_ns']
                and load['time_ns'] <= block['valid']['time_ns']
                and block['valid']['event'] == 'restored_page_valid' and block['valid']['job'] == load['job']
                and block['valid']['page'] == target, 'STORE/real model overwrite/LOAD/publication order is invalid')
        bind_round_event(block['valid'], report, raw_log)
        if inspect_bytes:
            from contextlib import ExitStack
            with ExitStack() as stack:
                identities = {}
                values = [map_capture(stack, capture_path(root, block[name]), BLOCK_BYTES, identities)
                          for name in ('source', 'overwritten', 'committed', 'restored')]
                result = verify_roundtrip(*values, num_pages=report['actual_pages'],
                                          source_page=source, target_page=target)
                results.append(result)
    if inspect_bytes:
        from contextlib import ExitStack
        with ExitStack() as stack:
            identities = {}
            before, after = [map_capture(stack, capture_path(root, round_[name]), 134217728, identities)
                             for name in ('pool_before', 'pool_after')]
            verify_guards(before, after, round_['pool_after']['write_ranges'])
    return {'status': 'HOST-EVIDENCE-CHECKED' if inspect_bytes else 'HOST-METADATA-CHECKED',
            'native_acceptance': False, 'block_checks': results}


def correlate_native(report, raw_log):
    events = [json.loads(line.split('native_ddr_trace ', 1)[1]) for line in raw_log.splitlines()
              if 'native_ddr_trace {' in line]
    for round_ in report['rounds']:
        for block in round_['blocks']:
            for terminal in (block['store'], block['load']):
                accepted = [e for e in events if e['event'] == 'accepted' and e['job'] == terminal['job']]
                cleaned = [e for e in events if e['event'] == 'native_result_and_cleanup' and e['job'] == terminal['job']]
                require(len(accepted) == len(cleaned) == 1, 'missing actual native accept/result-cleanup trace')
                run = accepted[0]['native_run_id']
                require(type(run) is int, 'native trace has no actual run identity')
                lines = [line for line in raw_log.splitlines() if 'name=node.complete ' in line
                         and re.search(r'\brun_id=' + str(run) + r'\b', line)]
                submissions = [line for line in raw_log.splitlines() if 'name=node.submit ' in line
                               and re.search(r'\brun_id=' + str(run) + r'\b', line)]
                require(len(lines) == len(submissions) == 1 and re.search(r'\boutcome=0\b', lines[0]),
                        'raw native submit/successful completion is not uniquely correlated')
                ns = int(re.search(r'\bts=(\d+)', lines[0])[1])
                submit_ns = int(re.search(r'\bts=(\d+)', submissions[0])[1])
                require(submit_ns <= ns <= cleaned[0]['time_ns'] <= terminal['time_ns'],
                        'native/SDK/parent completion order differs from trace')


def check_controls(report, raw_log):
    trace = report['coordinator_trace']
    def terminal(row, status, bytes_):
        value = row['terminal']
        allowed = (status,) if isinstance(status, str) else status
        require(value['status'] in allowed and value['stopped'] is True and value['native_completed'] is True
                and value['bytes'] == bytes_, 'negative scenario lacks an actual stopped native receipt')
        require([e for e in trace if e['event'] == 'transfer_terminal' and e['job'] == value['job']] == [value],
                'negative scenario terminal differs from the actual coordinator trace')
        return value['job']
    cancellations = report['load_cancellation']
    require([r['all_waiters'] for r in cancellations] == [False, True], 'both LOAD cancellation cohorts are required')
    for row in cancellations:
        job = terminal(row, ('CANCELLED', 'SUCCEEDED') if row['all_waiters'] else 'SUCCEEDED', BLOCK_BYTES)
        require(row['pending_target_retained'] is True and row['early_valid'] is False,
                'cancellation did not retain the pending destination pin')
        if row['all_waiters']:
            require(not any(e['event'] == 'restored_page_valid' and e['job'] == job for e in trace),
                    'cancelled LOAD destination was published')
    row = report['store_cancellation']
    # A cancellation may race with physical completion. The parent must discard
    # either stopped outcome, without publishing content or releasing pins early.
    job = terminal(row, ('CANCELLED', 'SUCCEEDED'), BLOCK_BYTES)
    require(row['source_pin_released_after_stop'] is True and row['ddr_ready_not_published'] is True
            and not any(e['event'] == 'ddr_ready' and e['job'] == job for e in trace),
            'cancelled STORE was committed as a usable entry')
    for name, forbidden in (('postsubmit_load_failure', 'restored_page_valid'),
                            ('postsubmit_store_failure', 'ddr_ready')):
        row = report[name]
        job = terminal(row, 'FAILED', 0)
        require(row['layer'] == 'short host receipt after actual native result/cleanup'
                and not any(e['event'] == forbidden and e['job'] == job for e in trace),
                'short receipt was misrepresented or published')
    stale = report['stale_original_transfer']
    require(stale['receipt']['error'].startswith('retired delivery sequence')
            and len(stale['native_runs_before_after']) == 2
            and stale['native_runs_before_after'][0] == stale['native_runs_before_after'][1] > 0,
            'old envelope was not rejected before effects')
    receipts = [json.loads(line.split('native_ddr_receipt ', 1)[1]) for line in raw_log.splitlines()
                if 'native_ddr_receipt {' in line]
    require(stale['receipt'] in receipts, 'stale receipt has no matching actual worker log')


def check(config, report, descriptor, root, raw_log):
    require(report.get('mode') == 'execution' and report.get('case') == 'normal-two-rounds-and-negatives',
            'plan/replay metadata cannot be accepted as execution evidence')
    check_startup(config, report['startup'], descriptor)
    require([r['number'] for r in report['rounds']] == [0, 1], 'both normal rounds are required')
    result = [check_round(r, report, root, inspect_bytes=True, raw_log=raw_log) for r in report['rounds']]
    correlate_native(report, raw_log)
    for key in ('load_cancellation', 'store_cancellation', 'postsubmit_load_failure',
                'postsubmit_store_failure', 'stale_original_transfer', 'http_pool_hit'):
        require(report.get(key), f'missing real control scenario: {key}')
    check_controls(report, raw_log)
    http = report['http_pool_hit']
    seed = next(r for r in report['requests'] if r['request_id'] == report['rounds'][1]['cold_request'])
    require(http['request_id'] == report['rounds'][1]['restore_request'] == http['body']['id']
            and http['status'] == 200 and http['payload']['prompt'] == seed['prompt']
            and http['body']['usage']['prompt_tokens'] == 257
            and http['body']['usage']['completion_tokens'] == 2
            and http['body']['choices'][0]['text'].strip(), 'HTTP evidence is not the actual two-block restore')
    require(report.get('stop_ok') is True and report['shutdown_report']['cleanup_error'] is None
            and report['shutdown_report']['worker_reaped'] is True
            and report['shutdown_report']['native_ddr_obligations'] == 0
            and any(e['event'] == 'owner_drained_arguments_released' for e in report['coordinator_trace']),
            'normal managed owner shutdown was not proven')
    return {'status': 'HOST-EVIDENCE-CHECKED', 'D1_acceptance': False, 'rounds': result,
            'scope': 'offline correlation/bytes only; capture origin and external resource cleanup require review'}
