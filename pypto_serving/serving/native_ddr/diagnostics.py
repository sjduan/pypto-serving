# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Opt-in static byte evidence. No writes, device waits or inference-path fallback."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


def capture_idle_pool(slots, root, label):
    """Guard-only physical observation, never a prefix read/commit authority.

    Unlike entry readback (which uses an SDK R lease), guards include unallocated
    and Invalid bytes. Observe the existing registered map only while idle and
    reject competing clients/leases. Do not allocate, initialize, or consume it.
    """
    from dataclasses import asdict
    access = slots.access
    if slots.busy is not None or access.leases or access.unknown or access.closed:
        raise RuntimeError('pool guard observation requires already idle managed access')
    if set(slots.child_registrations.items()) != set(access.children):
        raise RuntimeError('pool guard observation lost the actual inherited child registry')
    retained = getattr(access.port, '_children', None)
    sdk_retained = getattr(access.port.client, '_children', None)
    birth_fields = {'pid', 'starttime_ticks', 'uid', 'gid', 'pid_namespace_dev', 'pid_namespace_ino'}
    if (type(retained) is not dict or type(sdk_retained) is not dict or set(retained) != set(access.children)
            or sdk_retained != {pid: birth for (pid, _), birth in retained.items()}):
        raise RuntimeError('original SDK child birth registry differs from serving obligations')
    def birth_identity(birth):
        if (type(birth) is not dict or set(birth) != birth_fields
                or any(type(v) is not int or v < 0 for v in birth.values())
                or any(birth[k] <= 0 for k in ('pid', 'starttime_ticks', 'pid_namespace_dev', 'pid_namespace_ino'))):
            raise RuntimeError('invalid original SDK process birth schema')
        return birth['pid'], birth['starttime_ticks']
    if any(birth_identity(birth) != key for key, birth in retained.items()):
        raise RuntimeError('original SDK child birth key differs from retained registry')
    def snapshot():
        value = access.port.client.snapshot()
        pools = [p for p in value['pools'] if p['pool_id'] == access.identity.pool_id]
        if len(pools) != 1:
            raise ValueError('pool guard observation lost its unique resource')
        pool = pools[0]
        if (pool['pool_epoch'] != access.identity.pool_epoch or pool['frozen']
                or pool['ranges']['leases'] or pool['nbytes'] != access.identity.nbytes
                or (pool['canonical']['dev'], pool['canonical']['inode']) != (
                    access.identity.device, access.identity.inode)
                or len([c for c in pool['clients'] if c['state'] != 'Unmapped']) != 1
                or not any(c['client_session'] == access.identity.client_session and c['state'] == 'Mapped'
                           for c in pool['clients'])):
            raise RuntimeError('pool guard observation has competing/unknown physical access')
        clients = [c for c in pool['clients'] if c['client_session'] == access.identity.client_session]
        if (len(clients) != 1 or clients[0]['closing'] is not False or clients[0]['access'] != 'rw'
                or clients[0]['access_generation'] != access.identity.access_generation):
            raise RuntimeError('current SDK client is closing or changed registered ownership')
        client = clients[0]
        birth_identity(client['birth'])
        children = {}
        for child in client['children']:
            if set(child) != {'birth', 'unknown'} or child['unknown'] is not False:
                raise RuntimeError('actual SDK child is Unknown or has an unexpected schema')
            key = birth_identity(child['birth'])
            if key in children or any(child['birth'][k] != client['birth'][k]
                    for k in ('uid', 'gid', 'pid_namespace_dev', 'pid_namespace_ino')):
                raise RuntimeError('duplicate child or foreign process namespace/credentials')
            children[key] = child['birth']
        if children != retained:
            raise RuntimeError('actual public SDK children/births differ from original retained registry')
        return pool
    before = snapshot()
    path = Path(root) / (label + '-pool.bin')
    digest = hashlib.sha256()
    with path.open('xb') as stream:
        for start in range(0, access.identity.nbytes, 1 << 20):
            data = bytes(access.buffer[start:start + (1 << 20)])
            stream.write(data)
            digest.update(data)
    after = snapshot()
    if before != after:
        raise RuntimeError('pool state changed during idle guard observation')
    return {'file': str(path), 'bytes': access.identity.nbytes, 'sha256': digest.hexdigest(),
            'identity': asdict(access.identity), 'snapshot': after,
            'write_ranges': [[b.byte_offset + 2097152, 20971520] for b in slots._blocks.values()],
            'scope': 'idle guard-only physical observation; not Committed prefix consumption',
            'child_registry_matches_sdk': set(slots.child_registrations.items()) == set(access.children),
            'registered_children': [list(key) for key in slots.child_registrations.items()],
            'registered_child_births': [dict(birth) for birth in retained.values()],
            'helper_pids': list(slots.child_helpers)}


def capture_shared_guards(slots, root, label):
    """Read only assigned committed guards under actual SDK readers.

    Peer model mappings and disjoint payload operations may remain live. This
    does not relax the Run04 single-client full-pool observation above, consume
    a peer payload, or claim a lease for unallocated bytes.
    """
    from dataclasses import asdict
    import uuid
    from .managed_pool import GUARD_BYTES
    access = slots.access
    if (slots.bindings is None or slots.shared_directory is None or slots.busy is not None
            or access.leases or access.unknown or access.closed):
        raise RuntimeError('shared guard capture requires idle exact preallocated access')
    retained = access.port._children
    birth_fields = {'pid', 'starttime_ticks', 'uid', 'gid', 'pid_namespace_dev', 'pid_namespace_ino'}
    def checked_birth(birth):
        if (type(birth) is not dict or set(birth) != birth_fields
                or any(type(v) is not int or v < 0 for v in birth.values())
                or any(birth[k] <= 0 for k in ('pid', 'starttime_ticks', 'pid_namespace_dev', 'pid_namespace_ino'))):
            raise RuntimeError('shared guard original SDK birth schema is invalid')
        return birth['pid'], birth['starttime_ticks']
    if type(retained) is not dict or any(checked_birth(birth) != key for key, birth in retained.items()):
        raise RuntimeError('shared guard retained birth keys disagree with their full identities')
    if (set(slots.child_registrations.items()) != set(access.children) or set(retained) != set(access.children)
            or access.port.client._children != {pid: birth for (pid, _), birth in retained.items()}):
        raise RuntimeError('shared guard capture lost its original inherited child registries')
    def own_client():
        snapshot = access.port.client.snapshot()
        pools = [p for p in snapshot['pools'] if p['pool_id'] == access.identity.pool_id]
        if len(pools) != 1:
            raise RuntimeError('shared guard lost the unique actual pool')
        pool = pools[0]
        if (pool['pool_epoch'] != access.identity.pool_epoch or pool['provider_session'] != access.identity.provider_session
                or pool['state'] != 'Allocated' or pool['frozen']
                or (pool['canonical']['dev'], pool['canonical']['inode'], pool['nbytes']) != (
                    access.identity.device, access.identity.inode, access.identity.nbytes)):
            raise RuntimeError('shared guard pool/backing incarnation changed')
        clients = [c for c in pool['clients'] if c['client_session'] == access.identity.client_session]
        if (len(clients) != 1 or clients[0]['state'] != 'Mapped' or clients[0]['closing']
                or clients[0]['access'] != 'rw' or clients[0]['access_generation'] != access.identity.access_generation):
            raise RuntimeError('shared guard owner is not the original mapped/admitting client')
        client = clients[0]
        checked_birth(client['birth'])
        children = {}
        for child in client['children']:
            if set(child) != {'birth', 'unknown'}:
                raise RuntimeError('shared guard child schema changed')
            birth = child['birth']
            key = checked_birth(birth)
            if (child['unknown'] is not False or key in children
                    or any(birth[k] != client['birth'][k] for k in ('uid', 'gid', 'pid_namespace_dev', 'pid_namespace_ino'))):
                raise RuntimeError('shared guard child is Unknown/duplicated or outside the original namespace')
            children[key] = birth
        if children != retained:
            raise RuntimeError('shared guard public SDK child births differ from retained originals')
        return client
    before = own_client()
    rows = []
    for block in slots.bindings:
        lease = access.port.acquire_range(block, 'r', block.generation, 0, GUARD_BYTES)
        access.leases[lease.lease_id] = lease
        try:
            lease.validate(access.identity, block, 'r', block.byte_offset, GUARD_BYTES)
            access.port.revalidate(lease)
            data = bytes(access.buffer[block.byte_offset:block.byte_offset + GUARD_BYTES])
            access.port.revalidate(lease)
            ack = access.complete_read(lease, uuid.uuid4().hex)
        except Exception as error:
            access.report_unknown(lease, f'shared guard reader outcome retained: {error}')
            raise
        path = Path(root) / f'{label}-guard-{block.block_id}.bin'
        with path.open('xb') as stream:
            stream.write(data)
        rows.append({'block': asdict(block), 'file': str(path), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'lease_id': lease.lease_id,
            'lease_generation': lease.lease_generation, 'content_version': lease.content_version,
            'release_ack': asdict(ack)})
    after = own_client()
    if before != after:
        raise RuntimeError('own mapped child/birth obligations changed while capturing shared guards')
    return {'scope': 'assigned committed guard bytes with actual SDK R/revalidate/release; peer payloads excluded',
            'identity': asdict(access.identity), 'client_before': before, 'client_after': after,
            'registered_child_births': list(retained.values()), 'guards': rows}


def inspect_idle_pages(backend, module, command):
    import torch
    from .types import EntryState

    root = os.environ.get("PYPTO_NATIVE_DDR_DIAGNOSTICS_DIR")
    if not root:
        raise ValueError("native DDR static diagnostics are disabled")
    root = Path(root).resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", command.diagnostic_label):
        raise ValueError("invalid diagnostic label")
    if (not 0 < len(command.page_ids) <= 4
            or any(type(p) is not int or not 0 <= p < backend.layout.num_pages for p in command.page_ids)):
        raise ValueError("diagnostics require one to four legal pages")
    # Reject instead of introducing a barrier or draining unrelated model handles.
    if (module.pending is not None or backend.pending is not None
            or backend.worker._active_dispatch_handles or module.poisoned):
        raise RuntimeError("diagnostics require an already idle native owner")
    layout = backend.layout
    result = {"pages": [], "scope": "static public copy_from readback; no target writes",
              "transfer_replay": module.transfer_replay.stats()}
    for page in command.page_ids:
        compact = torch.empty((40, 2, 128, 8, 128), dtype=torch.bfloat16)
        for layer in range(40):
            for side, tensor in enumerate((backend.keys, backend.values)):
                backend.worker.copy_from(compact[layer, side].data_ptr(), tensor.data_ptr,
                                          layout.page_side_bytes, src_offset=layout.hbm_offset(layer, page))
        raw = compact.view(torch.uint8).numpy().tobytes()
        path = root / (command.diagnostic_label + f"-hbm-{page}.bin")
        with path.open("xb") as stream:
            stream.write(raw)
        layer_bytes = layout.page_side_bytes
        result["pages"].append({"page": page, "file": str(path), "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                                "layer_side_sha256": [hashlib.sha256(raw[i:i + layer_bytes]).hexdigest()
                                                       for i in range(0, len(raw), layer_bytes)]})
    if command.entry_ref is not None:
        entry = module.entries.get(command.entry_ref.entry_id)
        if entry is None or entry.ref != command.entry_ref or entry.state != EntryState.READY:
            raise ValueError("diagnostic DDR reference is not ready")
        raw = backend.read_slot(entry.backing).view(torch.uint8).numpy().tobytes()
        path = root / (command.diagnostic_label + "-ddr.bin")
        with path.open("xb") as stream:
            stream.write(raw)
        result["ddr"] = {"entry_id": entry.ref.entry_id, "slot": entry.slot,
                         "generation": entry.ref.slot_generation, "file": str(path), "bytes": len(raw),
                         "sha256": hashlib.sha256(raw).hexdigest()}
        if getattr(backend, 'is_managed', False):
            from dataclasses import asdict
            result['ddr'].update(byte_offset=entry.backing.byte_offset,
                                 committed=asdict(entry.backing.committed),
                                 pool_identity=asdict(backend.slots.access.identity))
    if (command.diagnostic_label.startswith('r04-')
            and os.environ.get('PYPTO_NATIVE_DDR_POOL_EVIDENCE') == '1'):
        result['pool'] = capture_idle_pool(backend.slots, root, command.diagnostic_label)
    if (command.diagnostic_label.startswith('r06-') and backend.slots.shared_directory is not None
            and os.environ.get('PYPTO_NATIVE_DDR_POOL_EVIDENCE') == '1'):
        result['shared_guards'] = capture_shared_guards(backend.slots, root, command.diagnostic_label)
    return json.dumps(result, sort_keys=True).encode()

def arm_short_receipt(backend, module, kind):
    """Inject host receipt validation failure AFTER a genuine successful native result."""
    from dataclasses import replace
    if not os.environ.get("PYPTO_NATIVE_DDR_DIAGNOSTICS_DIR"):
        raise ValueError("native DDR test faults are disabled")
    if kind not in ("short_load_receipt", "short_store_receipt"):
        raise ValueError("unsupported diagnostic fault")
    if module.pending is not None or backend.worker._active_dispatch_handles:
        raise RuntimeError("fault arming requires an already idle owner")
    operation = "LOAD" if kind == "short_load_receipt" else "STORE"
    armed = True
    def shorten(request, completion):
        nonlocal armed
        if armed and request.operation.value == operation:
            armed = False
            backend.record("injected_short_host_receipt_after_native_cleanup", job=request.job_id,
                           actual_bytes=completion.transferred_bytes, reported_bytes=0)
            return replace(completion, transferred_bytes=0)
        return completion
    backend.completion_filter = shorten
