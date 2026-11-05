# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only original n005 object diagnosis; preserve every tensor, frame, pin and obligation."""
from dataclasses import fields, is_dataclass
import hashlib
import inspect
from pathlib import Path
import sys


def serial(value):
    if is_dataclass(value):
        # Do not deepcopy native descriptor fields: nanobind canonical
        # identities are deliberately non-pickleable and must stay untouched.
        return {field.name: serial(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, bytes):
        return {'hex': value.hex()}
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [serial(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {'type': type(value).__module__ + '.' + type(value).__name__, 'object_id': hex(id(value))}


def handle(value):
    if value is None:
        return None
    result = {'type': type(value).__module__ + '.' + type(value).__name__, 'object_id': hex(id(value))}
    fields = getattr(value, '__dict__', {})
    for name in ('_dispatch_id', '_run_id', '_result_called', '_result_cached', '_closed', '_released',
                 '_error', '_done', '_finished', '_cleanup_done', '_state'):
        if name in fields:
            result[name] = serial(fields[name])
    result['fields'] = {k: serial(v) for k, v in fields.items()
                        if any(s in k for s in ('fence', 'event', 'done', 'native_handle', 'retired'))}
    return result


def prepare(ctx, cfg):
    from pypto_serving.serving.native_ddr.managed_process import process_birth, descendants
    import os
    assert (os.getpid(), process_birth(os.getpid())) == (1145935, 484536731)
    assert (cfg['pid'], cfg['birth']) == (1145935, 484536731)
    for name, digest in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    original = sys.modules['__main__']._RECOVERY
    assert type(original).__name__ == 'CooperativePoolClose' and original.paused and not original.busy
    frames = list(original.frames)
    consumers = [f for f in frames if f.f_code.co_name == 'consumer' and f.f_code.co_filename == cfg['target_source']]
    transfers = [f for f in frames if f.f_code.co_name == 'transfer' and f.f_code.co_filename == cfg['target_source']]
    assert len(consumers) == len(transfers) == 1
    consumer = consumers[0]
    module = consumer.f_locals['module']
    worker, slots, access = (consumer.f_locals[n] for n in ('worker', 'slots', 'access'))
    backend = consumer.f_locals['backend']
    assert module.backend is backend and backend.worker is worker and backend.slots is slots
    assert original.worker is worker and original.slots is slots and original.access is access
    assert sys.modules['__main__']._ACTIVE_ACCESS is access
    assert descendants() == {int(p): b for p, b in cfg['children'].items()}
    assert access.identity.pool_id == cfg['pool_id'] and access.identity.client_session == cfg['client_id']
    assert access.identity.device == 25 and access.identity.inode == 842044
    ctx.update(original=original, frames=frames, module=module, backend=backend,
               worker=worker, slots=slots, access=access, client=access.port.client)
    return {'pid': os.getpid(), 'birth': process_birth(os.getpid()), 'prepared_only': True,
            'original_objects_preserved': True, 'actions': 'diagnostic only; no native/SDK state mutation'}


def diagnose(ctx, cfg):
    worker, backend, module, slots, access = (ctx[n] for n in ('worker', 'backend', 'module', 'slots', 'access'))
    record = module.transfer_replay.records[1]
    ambiguous = backend.ambiguous
    assert ambiguous is not None and len(ambiguous) == 3
    request, backing, args = ambiguous
    assert module.pending.request is request and module.pending.entry.backing is backing
    assert backing is slots.busy and backing.active_lease.lease_id == cfg['lease_id']
    assert args[-1] is backing.tensor
    owner_fields = ('_next_dispatch_id', '_persistent', '_persistent_error', '_closed', '_close_complete',
                    '_closing', '_accepting_dispatches')
    frames = [{'slot_id': f.slot_id, 'in_use': f.in_use, 'handle': handle(f.handle),
        'tensor_names': list(f.tensors), 'keepalive_count': len(f.keepalive), 'cleanup_count': len(f.cleanup)}
        for f in worker._dispatch_frames]
    active = list(worker._active_dispatch_handles)
    accepted = worker._accepted_native_handles()  # Read-only set snapshot under the actual runtime lock.
    low = worker._w
    assert isinstance(low._accepted_run_handles, set)
    with low._hierarchical_start_cv:
        low_handles = list(low._accepted_run_handles)
    raw_args = []
    for arg in args:
        value = {'type': type(arg).__module__ + '.' + type(arg).__name__, 'object_id': hex(id(arg)),
                 'shape': list(arg.shape) if hasattr(arg, 'shape') else None,
                 'dtype': str(arg.dtype) if hasattr(arg, 'dtype') else None}
        if hasattr(arg, 'is_shared'):
            value['torch_is_shared'] = arg.is_shared()
            value['data_ptr'] = arg.data_ptr()
        raw_args.append(value)
    state = worker._states[backend.programs[0].compiled]
    supplemental = []
    for info, arg in zip(state['param_infos'], args):
        try:
            worker._validate_prepared_dispatch_arg(info, arg)
            supplemental.append({'parameter': info.name, 'accepted': True})
        except BaseException as error:
            supplemental.append({'parameter': info.name, 'accepted': False, 'error': repr(error)})
    import pypto.runtime.distributed_runner as runtime
    source = Path(runtime.__file__)
    methods = {}
    for name in ('_validate_prepared_dispatch_arg', '_submit_compiled', '_release_unpublished_dispatch_frame',
                 '_submit_native_dispatch', '_accepted_native_handles', 'release_service_tensor', 'close'):
        method = getattr(type(worker), name)
        methods[name] = {'line': inspect.getsourcelines(method)[1], 'source': inspect.getsource(method)}
    result = {'pid': cfg['pid'], 'birth': cfg['birth'], 'original_replay_response': serial(record.response),
        'original_replay_response_present': record.response is not None, 'original_replay_identity': serial(record.identity),
        'original_replay_protected': record.protected, 'original_replay_fields': serial(vars(record)),
        'module_pending_handle': handle(module.pending.handle), 'module_pending_quarantined': module.pending.quarantined,
        'module_poisoned': module.poisoned, 'entry_state': module.pending.entry.state.value,
        'original_ambiguous_request': serial(request), 'backing_active_lease': serial(backing.active_lease),
        'original_ambiguous_args': raw_args, 'compact_is_original_backing_tensor': True,
        'compact_byte_offset': backing.byte_offset, 'actual_compact_bytes': backing.tensor.numel() * backing.tensor.element_size(),
        'owner': {name: serial(getattr(worker, name)) for name in owner_fields}, 'dispatch_frames': frames,
        'active_dispatch_handles': [handle(h) for h in active],
        'accepted_native_handles': None if accepted is None else [handle(h) for h in accepted],
        'underlying_accepted_native_handles': [handle(h) for h in low_handles],
        'underlying_runtime_metadata': {k: serial(v) for k, v in vars(low).items()
            if any(word in k for word in ('fence', 'pending', 'handle', 'service', 'run_id'))},
        'service_tokens': sorted(worker._service_tokens), 'backend_pins': list(backend._pins),
        'access_unknown': serial(access.unknown), 'access_leases': serial(access.leases),
        'SDK_lease_states': dict(ctx['client']._leases), 'port_leases': serial(access.port._leases),
        'new_supplemental_original_argument_validation': supplemental,
        'supplemental_is_new_call_not_original_trace': True,
        'actual_runtime_source': {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'methods': methods},
        'retained_frames': [{'path': f.f_code.co_filename, 'name': f.f_code.co_name, 'line': f.f_lineno} for f in ctx['frames']],
        'scope': 'original object read-only diagnostic; no new submit/SDK mutation/pin release/owner close/frame clear'}
    ctx['diagnosis'] = result
    return result


def quiescence(ctx):
    """Current access proof, under the actual admission/lifecycle locks; no historical-count claim."""
    worker, low = ctx['worker'], ctx['worker']._w
    assert ctx['original'].paused and not ctx['original'].busy and ctx['module'].poisoned
    empty = ('_accepted_run_handles', '_abandoned_run_handles', '_chip_runs', '_chip_run_touched_identities',
        '_chip_source_uses', '_pending_remote_buffer_frees', '_pending_remote_import_releases',
        '_pending_remote_import_release_states', '_chip_service_factories', '_lease_depth', '_run_finalization_depth')
    clear = ('_ordered_cleanup_error', '_worker_host_mapped_cleanup_error', '_delegated_session_fatal',
             '_building_run_resources', '_region_access_service')
    with worker._dispatch_submit_mu, low._submit_mu.exclusive(), low._hierarchical_start_cv:
        values = {name: serial(getattr(low, name)) for name in (*empty, *clear)}
        assert not worker._active_dispatch_handles
        assert all(not f.in_use and f.handle is None and not f.tensors and not f.keepalive and not f.cleanup
                   for f in worker._dispatch_frames)
        for name in empty:
            assert not getattr(low, name), f'current native/cleanup obligation: {name}'
        for name in clear:
            assert getattr(low, name) is None, f'current native poison/accessor: {name}'
        assert low._active_ops == 0 and not low._source_context_failed
        assert low._worker_host_mapped_cleanup_state == (None, ())
        values.update(active_operations=low._active_ops, source_context_failed=low._source_context_failed,
                      mapped_cleanup_state=serial(low._worker_host_mapped_cleanup_state))
    import simpler.worker as source
    values['actual_Simpler_source'] = {'path': source.__file__, 'sha256': hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        'acceptance_rule': inspect.getsource(type(low)._submit_l3_locked),
        'abandonment_publication_rule': inspect.getsource(type(low)._publish_abandoned_run)}
    values.update(current_access_ended=True, scope='current static pause: no accepted/active/pending/building/abandoned/debt/poison/accessor',
                  historical_native_submit_count_claimed=False, callable_registrations_retained=serial(low._live_handles))
    return values


def close_device(ctx, cfg, save):
    """Release only the original n005 device owner. Pool Unknown/SDK obligations remain untouched."""
    from pypto_serving.serving.native_ddr.managed_process import process_birth, descendants, detach_closed_children
    proof = quiescence(ctx)
    save('native-current-quiescence.json', proof)
    ctx['stage'] = 'release-service-pins'
    assert set(ctx['backend']._pins) == ctx['worker']._service_tokens and len(ctx['backend']._pins) == 2
    pins = list(ctx['backend']._pins)
    ctx['backend']._release_pins()  # Actual release_service_tensor calls; ambiguous is deliberately retained.
    assert not ctx['worker']._service_tokens and not ctx['backend']._pins and not ctx['worker']._w._service_buffers
    save('pins-actually-released.json', {'pins': pins, 'runtime_service_tokens_empty': True,
        'underlying_service_buffers_empty': True, 'business_ambiguous_retained': ctx['backend'].ambiguous is not None})
    ctx['stage'] = 'actual-worker-close'
    ctx['worker'].close()
    assert ctx['worker']._close_complete and not ctx['worker']._active_dispatch_handles
    assert not ctx['worker']._inherited_host_tensors
    assert process_birth(1146316) != 484537554
    remaining = descendants(cfg['pid'])
    assert set(remaining) <= {1146315}
    save('device-close-success.json', {'pid': cfg['pid'], 'birth': cfg['birth'],
        'actual_Worker_close_returned': True, 'actual_close_complete': True,
        'original_chip_pid': 1146316, 'original_chip_birth_gone': True, 'remaining_children': remaining,
        'SDK_state_mutated': False, 'unmap_ACK_done': False, 'pool_destroyed': False,
        'scope': 'normal close of existing device owner under A/B hold; pool/client Unknown responsibility remains'})
    ctx['device_closed'] = True
    ctx['stage'] = 'detach-actually-stopped-child'
    slots = ctx['slots']
    detach_closed_children(ctx['access'], slots.child_registrations, slots.child_baseline, helpers=slots.child_helpers)
    save('device-stage-complete.json', {'normal_Worker_close': True, 'original_mapping_child_detached': True,
        'original_writer_unknown_retained': True, 'unmap_ACK_done': False, 'pool_destroyed': False,
        'scope': 'existing device resource released; further host terminal-unmap still required'})
