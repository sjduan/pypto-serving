# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded business metadata; UFlow alone grants mappings, leases and versions.

No tensor, runtime or SDK import happens here. A directory never transfers KV
payloads or frees a pool. The injected authority reads the original SDK records.
Readers still acquire/revalidate a real range lease before any native access.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import socket
import socketserver
import threading

from .managed_pool import BLOCK_BYTES, GUARD_BYTES, CommittedRange, PoolBlock, opaque, prefix_content_id
from .types import PrefixKey

MAX_FRAME = 65536


def key_wire(key):
    if not isinstance(key, PrefixKey) or len(key.namespace_digest) != 32 or len(key.block_hash) != 32:
        raise ValueError('a full compatible prefix key is required')
    return [key.namespace_digest.hex(), key.block_hash.hex()]


def key_from_wire(value):
    if type(value) is not list or len(value) != 2 or any(type(x) is not str or len(x) != 64 for x in value):
        raise ValueError('invalid shared prefix key')
    return PrefixKey(*(bytes.fromhex(x) for x in value))


def committed_from_wire(value):
    return CommittedRange(**{**value, 'block': PoolBlock(**value['block'])})


class DirectoryAuthority:
    def __init__(self, descriptor, blocks, snapshot, confirm):
        if not 1 <= len(blocks) <= 4 or len(set(blocks)) != len(blocks):
            raise ValueError('shared directory requires one to four actual preallocated blocks')
        self.descriptor, self.blocks, self.snapshot, self.confirm = descriptor, tuple(blocks), snapshot, confirm
        intervals = sorted((b.byte_offset, b.byte_offset + b.nbytes) for b in blocks)
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('shared allocation intervals overlap')
        self.current()

    def current(self):
        snapshot = self.snapshot()
        if (any(type(snapshot[k]) is not int for k in ('budget_bytes', 'charged_bytes', 'creation_reserved_bytes'))
                or snapshot['budget_bytes'] != 134217728 or snapshot['charged_bytes'] != 134217728
                or snapshot['creation_reserved_bytes'] != 0 or len(snapshot['pools']) != 1):
            raise ValueError('shared pool budget/charge differs from the single 128MiB authority')
        pools = [p for p in snapshot['pools'] if p['pool_id'] == self.descriptor['pool_id']]
        if len(pools) != 1:
            raise ValueError('shared pool identity missing or duplicated')
        pool = pools[0]
        for name in ('pool_id', 'pool_epoch', 'provider_session', 'canonical', 'resource_id',
                     'resource_generation', 'placement_id', 'binding_id', 'nbytes', 'provider_kind'):
            if pool[name] != self.descriptor[name]:
                raise ValueError('shared backing/provider incarnation changed')
        if (pool['state'] != 'Allocated' or pool['frozen'] or any(c['state'] == 'Unknown'
                or any(x['unknown'] for x in c['children']) for c in pool['clients'])
                or any(l['state'] == 'Unknown' for l in pool['ranges']['leases'])):
            raise ValueError('shared pool has unresolved mapping/native obligations')
        for block in self.blocks:
            rows = [a for a in pool['ranges']['allocations'] if a['block_id'] == block.block_id]
            if (block.pool_id != pool['pool_id'] or block.pool_epoch != pool['pool_epoch']
                    or block.nbytes != BLOCK_BYTES + GUARD_BYTES or len(rows) != 1
                    or any(rows[0][a] != b for a, b in (
                        ('generation', block.generation), ('offset', block.byte_offset), ('nbytes', block.nbytes)))):
                raise ValueError('directory partition is not an actual current UFlow allocation')
        return pool

    @staticmethod
    def idle(pool, block):
        begin, end = block.byte_offset + GUARD_BYTES, block.byte_offset + block.nbytes
        return not any(l['offset'] < end and begin < l['offset'] + l['nbytes']
                       for l in pool['ranges']['leases'])

    def valid(self, pool, key, value):
        if (value.block not in self.blocks or value.pool_id != pool['pool_id']
                or value.pool_epoch != pool['pool_epoch'] or value.provider_session != pool['provider_session']
                or value.byte_offset != value.block.byte_offset + GUARD_BYTES or value.nbytes != BLOCK_BYTES
                or value.content_id != prefix_content_id(key) or type(value.content_version) is not int
                or value.content_version <= 0):
            return False
        block = next(a for a in pool['ranges']['allocations'] if a['block_id'] == value.block.block_id)
        return any(s['state'] == 'Committed' and s['offset'] == value.byte_offset
                   and s['nbytes'] == value.nbytes and s['content_id'] == value.content_id
                   and s['content_version'] == value.content_version for s in block['segments'])

    def committed(self, key, value):
        opaque(value.operation_id, 'original writer operation provenance')
        opaque(value.writer_lease_id, 'original writer lease provenance')
        pool = self.current()
        if not self.valid(pool, key, value):
            raise ValueError('publication is not the current exact UFlow committed content')
        confirmation = self.confirm(key, value)
        if not self.valid(self.current(), key, value):
            raise ValueError('committed version changed after the independent SDK reader confirmation')
        return confirmation


class SharedPrefixDirectory:
    """One bounded single-writer index. Intents are never range permissions."""
    def __init__(self, epoch, authority):
        opaque(epoch, 'directory epoch')
        self.epoch, self.authority = epoch, authority
        self.records, self.intents = {}, {}
        self._lock = threading.RLock()
        self.revision = 0

    def discover(self, keys):
        if not 1 <= len(keys) <= len(self.authority.blocks):
            raise ValueError('discovery must be a bounded contiguous token0 prefix chain')
        with self._lock:
            pool = self.authority.current()
            result = []
            for key in keys:
                value = self.records.get(key)
                if value is None or not self.authority.valid(pool, key, value['content']):
                    break
                result.append(asdict(value['content']))
            return result

    def claim(self, key, owner):
        opaque(owner, 'publication intent owner')
        with self._lock:
            pool = self.authority.current()
            value = self.records.get(key)
            if value is not None and self.authority.valid(pool, key, value['content']):
                return {'state': 'Committed', 'content': asdict(value['content'])}
            if key in self.intents:
                intent = self.intents[key]
                if intent['owner'] != owner:
                    return {'state': 'Busy'}
                return {'state': 'Intent', 'block': asdict(intent['block']), 'owner': owner}
            used = {r['block'] for r in self.intents.values()}
            occupied = {r['content'].block for r in self.records.values()}
            candidates = [b for b in self.authority.blocks if b not in used and b not in occupied]
            candidates += [r['content'].block for r in sorted(self.records.values(), key=lambda r: r['revision'])
                           if r['content'].block not in used]
            block = next((b for b in candidates if self.authority.idle(pool, b)), None)
            if block is None:
                return {'state': 'Busy'}
            self.records = {k: r for k, r in self.records.items() if r['content'].block != block}
            self.intents[key] = {'owner': owner, 'block': block}
            return {'state': 'Intent', 'block': asdict(block), 'owner': owner}

    def publish(self, key, owner, content):
        with self._lock:
            existing = self.records.get(key)
            if existing is not None and existing['owner'] == owner and existing['content'] == content:
                self.authority.committed(key, content)
                return asdict(content)
            intent = self.intents.get(key)
            if intent is None or (intent['owner'], intent['block']) != (owner, content.block):
                raise ValueError('publication has no exact key/owner/allocation intent')
            confirmation = self.authority.committed(key, content)
            self.revision += 1
            self.records[key] = {'owner': owner, 'content': content, 'revision': self.revision,
                                 'SDK_confirmation': confirmation}
            del self.intents[key]
            return asdict(content)

    def cancel(self, key, owner):
        with self._lock:
            intent = self.intents.get(key)
            if intent is None:
                return {'cancelled': True}
            if intent['owner'] != owner or not self.authority.idle(self.authority.current(), intent['block']):
                raise ValueError('intent has another owner or actual in-flight UFlow range obligations')
            del self.intents[key]
            return {'cancelled': True}

    def dispatch(self, request):
        if type(request) is not dict or request.get('epoch') != self.epoch:
            raise ValueError('directory incarnation mismatch')
        op = request.get('op')
        if op == 'describe' and set(request) == {'epoch', 'op'}:
            with self._lock:
                self.authority.current()
                return {'epoch': self.epoch, 'capacity': len(self.authority.blocks),
                        'records': len(self.records), 'intents': len(self.intents), 'revision': self.revision,
                        'descriptor': self.authority.descriptor, 'blocks': [asdict(b) for b in self.authority.blocks]}
        if op == 'snapshot' and set(request) == {'epoch', 'op'}:
            with self._lock:
                self.authority.current()
                return {'epoch': self.epoch, 'revision': self.revision,
                    'records': [{'key': key_wire(k), 'content': asdict(r['content']),
                                 'SDK_confirmation': r['SDK_confirmation']} for k, r in self.records.items()],
                    'intents': [{'key': key_wire(k), 'owner': r['owner'], 'block': asdict(r['block'])}
                                for k, r in self.intents.items()]}
        if op == 'discover' and set(request) == {'epoch', 'op', 'keys'}:
            return self.discover([key_from_wire(k) for k in request['keys']])
        if op in ('claim', 'cancel') and set(request) == {'epoch', 'op', 'key', 'owner'}:
            return getattr(self, op)(key_from_wire(request['key']), request['owner'])
        if op == 'publish' and set(request) == {'epoch', 'op', 'key', 'owner', 'content'}:
            return self.publish(key_from_wire(request['key']), request['owner'], committed_from_wire(request['content']))
        raise ValueError('unsupported directory metadata request')


class DirectoryClient:
    def __init__(self, endpoint, epoch):
        opaque(endpoint, 'directory endpoint')
        opaque(epoch, 'directory epoch')
        self.endpoint, self.epoch = endpoint, epoch

    def call(self, op, **values):
        data = json.dumps({'epoch': self.epoch, 'op': op, **values}, separators=(',', ':')).encode() + b'\n'
        if len(data) > MAX_FRAME:
            raise ValueError('directory metadata frame exceeds bound')
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(5)
            sock.connect(self.endpoint)
            sock.sendall(data)
            with sock.makefile('rb') as stream:
                raw = stream.readline(MAX_FRAME + 1)
        if len(raw) > MAX_FRAME or not raw.endswith(b'\n'):
            raise ValueError('directory response is not a bounded complete frame')
        result = json.loads(raw)
        if result.get('error'):
            raise ValueError(result['error'])
        if set(result) != {'result'}:
            raise ValueError('invalid directory response')
        return result['result']

    def discover(self, keys):
        return tuple(committed_from_wire(v) for v in self.call('discover', keys=[key_wire(k) for k in keys]))

    def claim(self, key, owner):
        return self.call('claim', key=key_wire(key), owner=owner)

    def publish(self, key, owner, content):
        return committed_from_wire(self.call('publish', key=key_wire(key), owner=owner, content=asdict(content)))

    def cancel(self, key, owner):
        return self.call('cancel', key=key_wire(key), owner=owner)


class DirectoryServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Explicitly owned local metadata service; no pool admin/finalize endpoint."""
    daemon_threads = False
    block_on_close = True
    address_family = getattr(socket, 'AF_UNIX', socket.AF_INET)
    request_queue_size = 16

    def __init__(self, endpoint, directory):
        if not hasattr(socket, 'AF_UNIX'):
            raise RuntimeError('the metadata service requires a Unix-domain-socket Python build')
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(5)
                try:
                    raw = self.rfile.readline(MAX_FRAME + 1)
                    if len(raw) > MAX_FRAME or not raw.endswith(b'\n'):
                        raise ValueError('directory request is not a bounded complete frame')
                    response = {'result': directory.dispatch(json.loads(raw))}
                except (ValueError, KeyError, TypeError, OSError) as error:
                    response = {'error': str(error)}
                encoded = json.dumps(response, separators=(',', ':')).encode() + b'\n'
                if len(encoded) > MAX_FRAME:
                    encoded = b'{"error":"directory response exceeds metadata bound"}\n'
                self.wfile.write(encoded)
        self._workers = threading.BoundedSemaphore(8)
        super().__init__(endpoint, Handler)

    def process_request(self, request, client_address):
        if not self._workers.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._workers.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._workers.release()
