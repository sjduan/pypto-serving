# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent-process business-index tests with synthetic authority, not UF/NPU evidence."""
from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import tempfile

import pytest

from pypto_serving.serving.native_ddr.managed_pool import (
    BLOCK_BYTES, GUARD_BYTES, CommittedRange, ManagedPoolAccess, PoolBlock, StaleCommittedRange, prefix_content_id,
)
from pypto_serving.serving.native_ddr.shared_directory import (
    DirectoryAuthority, DirectoryClient, DirectoryServer, SharedPrefixDirectory, key_wire,
)
from pypto_serving.serving.native_ddr.types import PrefixKey
from tests.unit.serving.test_managed_pool_protocol import FakePort


class SyntheticAuthority:
    def __init__(self):
        self.descriptor = {'pool_id': 'pool', 'pool_epoch': 'epoch', 'provider_session': 'provider',
            'canonical': {'dev': 1, 'inode': 2}, 'resource_id': 1, 'resource_generation': 1,
            'placement_id': 1, 'binding_id': 1, 'nbytes': 134217728, 'provider_kind': 'posix_shm_simpler'}
        self.blocks = tuple(PoolBlock('pool', 'epoch', n + 1, 1, (n + 1) * 23068672, 23068672) for n in range(4))
        self.pool = {**self.descriptor, 'state': 'Allocated', 'frozen': False, 'clients': [],
            'ranges': {'allocations': [{'block_id': b.block_id, 'generation': b.generation,
                'offset': b.byte_offset, 'nbytes': b.nbytes, 'segments': []} for b in self.blocks], 'leases': []}}
        self.operations = {}

    def snapshot(self):
        return copy.deepcopy({'budget_bytes': 134217728, 'charged_bytes': 134217728,
                              'creation_reserved_bytes': 0, 'pools': [self.pool]})

    def authority(self):
        return DirectoryAuthority(self.descriptor, self.blocks, self.snapshot, lambda op: self.operations[op])

    def commit(self, key, block, version=1):
        value = CommittedRange('pool', 'epoch', 'provider', block, block.byte_offset + GUARD_BYTES,
            BLOCK_BYTES, version, prefix_content_id(key), f'op-{block.block_id}-{version}', f'writer-{version}')
        row = next(a for a in self.pool['ranges']['allocations'] if a['block_id'] == block.block_id)
        row['segments'] = [{'state': 'Committed', 'offset': value.byte_offset, 'nbytes': value.nbytes,
                            'content_version': version, 'content_id': value.content_id}]
        self.operations[value.operation_id] = {'state': 'Completed', 'response': {'operation_id': value.operation_id,
            'status': 'ok', 'result': {'committed': True, 'content_version': version,
                                      'content_id': value.content_id, 'lease_id': value.writer_lease_id}}}
        return value


def key(number):
    return PrefixKey(b'n' * 32, bytes([number]) * 32)


def test_commit_is_invisible_until_atomic_validated_publication_and_survives_producer_exit():
    uf = SyntheticAuthority()
    directory = SharedPrefixDirectory('directory-epoch', uf.authority())
    intent = directory.claim(key(1), 'producer-job')
    block = PoolBlock(**intent['block'])
    assert directory.discover([key(1)]) == []
    content = uf.commit(key(1), block)
    assert directory.discover([key(1)]) == []  # a UF commit alone is not business publication
    assert directory.publish(key(1), 'producer-job', content) == asdict(content)
    directory.cancel(key(1), 'producer-job')  # normal producer retirement leaves committed content
    assert directory.discover([key(1)]) == [asdict(content)]
    assert directory.claim(key(1), 'another-producer')['state'] == 'Committed'


@pytest.mark.parametrize('fault', ['half-write', 'version', 'writer', 'receipt', 'range', 'provider', 'allocation'])
def test_forged_or_half_written_publication_never_becomes_discoverable(fault):
    uf = SyntheticAuthority()
    directory = SharedPrefixDirectory('directory-epoch', uf.authority())
    block = PoolBlock(**directory.claim(key(1), 'owner')['block'])
    content = uf.commit(key(1), block)
    if fault == 'half-write':
        uf.pool['ranges']['allocations'][0]['segments'][0]['state'] = 'Writing'
    elif fault == 'version':
        content = replace(content, content_version=2)
    elif fault == 'writer':
        content = replace(content, writer_lease_id='not-the-original-writer')
    elif fault == 'receipt':
        uf.operations[content.operation_id]['state'] = 'Unknown'
    elif fault == 'range':
        content = replace(content, byte_offset=content.byte_offset + 65536)
    elif fault == 'provider':
        content = replace(content, provider_session='another-provider')
    else:
        content = replace(content, block=replace(block, generation=2))
    with pytest.raises(ValueError):
        directory.publish(key(1), 'owner', content)
    assert directory.discover([key(1)]) == []


def test_intents_are_bounded_and_cannot_recycle_a_real_inflight_range():
    uf = SyntheticAuthority()
    directory = SharedPrefixDirectory('directory-epoch', uf.authority())
    for n in range(4):
        assert directory.claim(key(n), f'owner{n}')['state'] == 'Intent'
    assert directory.claim(key(5), 'pressure')['state'] == 'Busy'
    block = directory.intents[key(0)]['block']
    uf.pool['ranges']['leases'] = [{'offset': block.byte_offset + GUARD_BYTES, 'nbytes': BLOCK_BYTES,
                                  'mode': 'w', 'state': 'Writing'}]
    with pytest.raises(ValueError):
        directory.cancel(key(0), 'owner0')
    assert len(directory.intents) == 4
    uf.pool['ranges']['leases'] = []
    directory.cancel(key(0), 'owner0')
    assert directory.claim(key(5), 'pressure')['state'] == 'Intent'


def test_current_version_and_contiguous_token0_chain_are_revalidated_for_each_discovery():
    uf = SyntheticAuthority()
    directory = SharedPrefixDirectory('directory-epoch', uf.authority())
    for n in (1, 2):
        block = PoolBlock(**directory.claim(key(n), f'owner{n}')['block'])
        content = uf.commit(key(n), block)
        directory.publish(key(n), f'owner{n}', content)
    assert len(directory.discover([key(1), key(2)])) == 2
    first = directory.records[key(1)]['content']
    uf.commit(key(9), first.block, version=2)
    assert directory.discover([key(1), key(2)]) == []
    assert directory.discover([PrefixKey(b'x' * 32, key(2).block_hash)]) == []
    uf.pool['frozen'] = True
    with pytest.raises(ValueError):
        directory.discover([key(2)])


@pytest.mark.skipif(not hasattr(socket, 'AF_UNIX'), reason='Windows Python build has no AF_UNIX; run on Linux')
def test_independent_processes_share_one_directory_and_one_canonical_intent():
    with tempfile.TemporaryDirectory(prefix='m1-', dir='/tmp') as short:
        uf = SyntheticAuthority()
        endpoint = str(Path(short) / 'd.sock')
        directory = SharedPrefixDirectory('epoch', uf.authority())
        with DirectoryServer(endpoint, directory) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                code = ('import json,sys;from pypto_serving.serving.native_ddr.shared_directory import DirectoryClient;'
                        'c=DirectoryClient(sys.argv[1],"epoch");'
                        'print(json.dumps(c.call("claim",key=json.loads(sys.argv[2]),owner=sys.argv[3])))')
                children = [subprocess.Popen([sys.executable, '-B', '-c', code, endpoint, json.dumps(key_wire(key(1))),
                                             f'independent-{n}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for n in range(2)]
                rows = []
                for child in children:
                    output, error = child.communicate(timeout=15)
                    assert child.returncode == 0, error.decode()
                    rows.append(json.loads(output))
                assert sorted(row['state'] for row in rows) == ['Busy', 'Intent']
                description = DirectoryClient(endpoint, 'epoch').call('describe')
                assert description['intents'] == 1 and description['records'] == 0
                with pytest.raises(ValueError, match='incarnation'):
                    DirectoryClient(endpoint, 'stale-epoch').call('describe')
            finally:
                server.shutdown()
                thread.join()
        Path(endpoint).unlink()


def test_shared_stale_version_releases_the_actual_current_reader_before_rejection():
    port = FakePort()
    access = ManagedPoolAccess(port)
    block = access.allocate()
    writer = access.acquire(block, 'w')
    first = access.complete_write(writer, prefix_content_id(key(1)), 'commit1')
    writer = access.acquire(block, 'w')
    current = access.complete_write(writer, prefix_content_id(key(2)), 'commit2')
    assert current.content_version == first.content_version + 1
    released = []
    original_release = port.release
    def release(lease, operation):
        released.append(lease)
        return original_release(lease, operation)
    port.release = release
    with pytest.raises(StaleCommittedRange):
        access.acquire(block, 'r', committed=first, reject_stale=True)
    assert not access.leases and not access.unknown
    assert len(released) == 1 and released[0].content_version == current.content_version
    access.close()


def test_preallocated_reservation_never_calls_the_old_lazy_allocator():
    from tests.unit.serving.test_managed_pool_protocol import ManagedNativeOrderingTests
    fixture = ManagedNativeOrderingTests()
    fixture.setUp()
    try:
        fixture.slots.capacity = 4
        fixture.slots.bindings = SyntheticAuthority().blocks
        def forbidden():
            raise AssertionError('shared replicas must not allocate another set of ranges')
        fixture.access.allocate = forbidden
        backing = fixture.slots.reserve(2, BLOCK_BYTES)
        assert backing.block == fixture.slots.bindings[2]
        assert backing.byte_offset == backing.block.byte_offset + GUARD_BYTES
        fixture.slots.release(backing)
        assert not fixture.slots.leases and not fixture.access.leases
    finally:
        fixture.tearDown()


def test_hbm_hit_has_priority_and_never_queries_the_shared_directory():
    from tests.unit.serving.test_native_ddr_host import coordinator, request
    coord, manager = coordinator(capacity=4)
    req = request('hbm', 257)
    pages = manager.allocate_blocks(2)
    hashes = manager.compute_block_hashes(req.prompt_token_ids)
    manager.cache_block_ids([p.block_id for p in pages], hashes, 0, 2)
    manager.release_blocks_by_ids([p.block_id for p in pages])
    class ForbiddenDirectory:
        def discover(self, keys):
            raise AssertionError('HBM hit must not need shared metadata or an SDK lease')
    coord.shared_directory = ForbiddenDirectory()
    cached = manager.get_computed_blocks(req.prompt_token_ids)
    assert len(cached) == 2
    assert coord.lookup_after_hbm(req, cached, 256).kind == 'READY'
    manager.release_cached_blocks(cached)


def test_cold_discovery_queues_original_import_controls_before_any_target_is_valid():
    from types import SimpleNamespace
    from pypto_serving.serving.native_ddr.managed_pool import PoolAck
    from pypto_serving.serving.native_ddr.types import DdrControlEvent
    from tests.unit.serving.test_native_ddr_host import coordinator, request
    coord, manager = coordinator(capacity=4)
    port = FakePort()
    coord.pool_identity = port.identity
    req = request('cold-peer', 257)
    proofs = []
    for n, digest in enumerate(manager.compute_block_hashes(req.prompt_token_ids)):
        block = PoolBlock('pool', 'epoch', n + 1, 1, n * 23068672, 23068672)
        proofs.append(CommittedRange('pool', 'epoch', 'provider', block, block.byte_offset + GUARD_BYTES,
            BLOCK_BYTES, n + 1, prefix_content_id(PrefixKey(coord.namespace, digest)), f'op{n}', f'w{n}'))
    coord.shared_directory = SimpleNamespace(discover=lambda keys: tuple(proofs))
    try:
        assert coord.lookup_after_hbm(req, [], 256).kind == 'WAIT'
        assert not coord.ready and not coord.jobs and len(coord.pending_imports) == 2
        while (command := coord.next_control()) is not None:
            ack = PoolAck('release-op', 'release', port.identity, 'current-reader', 1, True)
            event = DdrControlEvent(command.command_id, command.worker_epoch, 'import', command.entry_ref,
                delivery_sequence=command.delivery_sequence, imported=command.imported, pool_ack=ack)
            coord.on_control_event(event)
        assert len(coord.ready) == 2 and not coord.pending_imports
        assert coord.lookup_after_hbm(req, [], 256).kind == 'WAIT'
        assert len(coord.jobs) == 2 and all(manager.blocks[j.lease.block_ids[0]].block_hash is None
                                          for j in coord.jobs.values())
    finally:
        port.buffer.close()
