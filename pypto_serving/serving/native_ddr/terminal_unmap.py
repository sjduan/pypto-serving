# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit terminal unmap of failed content after actual owner stop; never ordinary success/release."""


def retire_pins_after_quiescence(worker, backend, proof):
    if proof.get('current_access_ended') is not True:
        raise RuntimeError('native access has not been proven quiescent')
    if set(backend._pins) != worker._service_tokens:
        raise RuntimeError('service retain ownership differs from the failed backend')
    backend._release_pins()  # The actual release_service_tensor protocol, not a set/list clear.
    if backend._pins or worker._service_tokens or worker._w._service_buffers:
        raise RuntimeError('service retain release is incomplete')


def unknown_writer_pool(client, snapshot, expected):
    descriptor = client.view.descriptor
    matches = [p for p in snapshot['pools'] if p['pool_id'] == descriptor['pool_id']]
    if len(matches) != 1:
        raise RuntimeError('terminal pool identity is not unique')
    pool = matches[0]
    if any(pool[k] != descriptor[k] for k in ('pool_epoch', 'provider_session', 'canonical', 'nbytes')):
        raise RuntimeError('terminal pool physical incarnation changed')
    peers = pool['clients']
    if (len(peers) != 1 or peers[0]['client_session'] != descriptor['client_session']
            or peers[0]['access_generation'] != descriptor['access_generation'] or peers[0]['children']
            or peers[0]['state'] not in ('Mapped', 'Unknown')):
        raise RuntimeError('terminal access has foreign or remaining child obligations')
    leases = pool['ranges']['leases']
    if (len(leases) != 1 or any(leases[0].get(k) != v for k, v in expected.items())
            or leases[0]['state'] != 'Unknown' or leases[0]['mode'] != 'w'
            or leases[0]['client_session'] != descriptor['client_session']):
        raise RuntimeError('terminal path admits only the exact original Unknown writer, never Active')
    if (client._children or any(v not in ('Unknown', 'Closed') for v in client._leases.values())
            or client._leases.get(expected['lease_id']) != 'Unknown'):
        raise RuntimeError('local original SDK has Active or foreign obligations')
    return pool


def invalid_writer_pool(client, snapshot, expected):
    descriptor = client.view.descriptor
    pool = next(p for p in snapshot['pools'] if p['pool_id'] == descriptor['pool_id'])
    if any(pool[k] != descriptor[k] for k in ('pool_epoch', 'provider_session', 'canonical', 'nbytes')):
        raise RuntimeError('terminal ACK changed the physical pool incarnation')
    peer = next(c for c in pool['clients'] if c['client_session'] == descriptor['client_session'])
    if peer['state'] != 'Unmapped' or peer['children'] or pool['ranges']['leases']:
        raise RuntimeError('original session is not terminally unmapped')
    block = next(b for b in pool['ranges']['allocations'] if b['block_id'] == expected['block_id'])
    cursor, end = expected['offset'], expected['offset'] + expected['nbytes']
    for segment in block['segments']:
        start, stop = segment['offset'], segment['offset'] + segment['nbytes']
        if stop <= cursor or start >= end:
            continue
        if (start > cursor or segment['state'] != 'Invalid' or segment['content_version'] != 0 or segment['content_id']):
            raise RuntimeError('failed writer was not invalidated; never accept Committed content')
        cursor = min(end, stop)
    if cursor != end:
        raise RuntimeError('terminal Invalid coverage has a gap')
    if any(value != 'Closed' for value in client._leases.values()):
        raise RuntimeError('original SDK has not retired its lease states after real ACK')
    return pool


def terminal_unmap(client, root_view, owner, expected, *, release_exports, prove_unmapped,
                   stop_helpers, unregister_operation, unmap_operation, state):
    """Original SDK performs the real Unknown-to-Invalid protocol; flags/debt are never hand-cleared."""
    if (not owner._close_complete or owner._active_dispatch_handles or owner._service_tokens
            or owner._inherited_host_tensors):
        raise RuntimeError('actual native owner close must complete before terminal unmap')
    if not state.get('unregister_done'):
        state['stage'] = 'verify-only-original-Unknown'
        state['before_snapshot'] = client.snapshot()
        unknown_writer_pool(client, state['before_snapshot'], expected)
    state['stage'] = 'release-stopped-owner-exports'
    if not state.get('exports_released'):
        release_exports()
        state['exports_released'] = True
    if not state.get('unregister_done'):
        state['stage'], state['operation_id'] = 'begin-original-unregister', unregister_operation
        state['unregister'] = client.begin_unregister(operation_id=unregister_operation)
        state['unregister_done'] = True
    state['stage'], state['operation_id'] = 'actual-mmap-FD-close', None
    root_view.release()
    client.view.close()  # Real BufferError must stop the protocol before any unmap ACK.
    state['unmapped_observation'] = prove_unmapped()
    state['stage'] = 'normal-helper-stop'
    if not state.get('helpers_stopped'):
        state['helpers'] = stop_helpers()
        state['helpers_stopped'] = True
    if not state.get('ack_done'):
        state['stage'], state['operation_id'] = 'original-session-terminal-unmap-ACK', unmap_operation
        state['ACK'] = client.ack_unmapped(native_drained=True, operation_id=unmap_operation)
        state['ack_done'] = True
    if state['ACK'].get('unknown_content_invalid_not_committed') is not True:
        raise RuntimeError('terminal ACK has no Unknown writer invalidation receipt')
    state['stage'], state['operation_id'] = 'fresh-Invalid-not-Committed', None
    state['after_snapshot'] = client.snapshot()
    invalid_writer_pool(client, state['after_snapshot'], expected)
    return state
