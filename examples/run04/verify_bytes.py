# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Offline D1 raw-byte oracle. Never imports SDK/tensor/device or executes a transfer.

Inputs are compact [40 layers, K/V, 128 tokens, 8 heads, 128 dims] BF16 captures
or complete pool snapshots. Capture origin, actual allocator/model computation,
native/lease ordering, resource reservation and lifecycle need separate evidence.
Successful byte comparisons alone cannot establish S1/D1 acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import mmap
import os
from pathlib import Path

TOKEN_BYTES = 8 * 128 * 2
SIDE_BYTES = 128 * TOKEN_BYTES
BLOCK_BYTES = 40 * 2 * SIDE_BYTES
ALIGNMENT = 65536
MAX_POOL_BYTES = 256 << 20
CHUNK_BYTES = 1 << 20
SCOPE = 'Offline raw bytes only; runtime/capture origin/allocator/UF/native/model/lifecycle not verified'


def bounded_integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer in [{lower}, {upper}]')


def require_equal(left, right, begin, end, label):
    for start in range(begin, end, CHUNK_BYTES):
        stop = min(start + CHUNK_BYTES, end)
        a, b = left[start:stop], right[start:stop]
        if a != b:
            offset = start + next(index for index, (x, y) in enumerate(zip(a, b)) if x != y)
            raise ValueError(f'{label}: first changed byte at absolute offset {offset}')


def verify_roundtrip(source, overwritten, committed, restored, *, num_pages, source_page, target_page):
    """Check every byte and all 128 token rows on each of 80 K/V sides."""
    bounded_integer(num_pages, 2, (1 << 32) - 1, 'actual HBM page count')
    for name, page in (('source page', source_page), ('target page', target_page)):
        bounded_integer(page, 0, num_pages - 1, name)
    if source_page == target_page:
        raise ValueError('restored target must be a different original-arena page')
    captures = {'source': source, 'overwritten': overwritten, 'committed': committed, 'restored': restored}
    if any(len(value) != BLOCK_BYTES for value in captures.values()):
        raise ValueError('each compact capture must contain exactly 20971520 bytes')
    require_equal(source, committed, 0, BLOCK_BYTES, 'committed pool content')
    require_equal(source, restored, 0, BLOCK_BYTES, 'restored HBM content')
    sides = []
    for layer in range(40):
        for side in range(2):
            begin = (layer * 2 + side) * SIDE_BYTES
            changed = sum(source[offset:offset + TOKEN_BYTES] != overwritten[offset:offset + TOKEN_BYTES]
                          for offset in range(begin, begin + SIDE_BYTES, TOKEN_BYTES))
            if changed != 128:
                raise ValueError(f'layer {layer} side {side}: only {changed}/128 source token rows changed')
            sides.append({'layer': layer, 'side': 'K' if side == 0 else 'V', 'changed_token_rows': changed,
                          'source_sha256': hashlib.sha256(source[begin:begin + SIDE_BYTES]).hexdigest(),
                          'overwritten_sha256': hashlib.sha256(overwritten[begin:begin + SIDE_BYTES]).hexdigest()})
    return {'scope': SCOPE, 'byte_checks_passed': True, 'block_bytes': BLOCK_BYTES,
            'num_pages': num_pages, 'source_page': source_page, 'target_page': target_page,
            'sha256': {name: hashlib.sha256(value).hexdigest() for name, value in captures.items()},
            'sides': sides}


def verify_guards(before, after, written_ranges):
    """Compare the full complement of granted writes, including first/last bytes."""
    capacity = len(before)
    bounded_integer(capacity, BLOCK_BYTES + ALIGNMENT, MAX_POOL_BYTES, 'pool capacity')
    if len(after) != capacity or not 1 <= len(written_ranges) <= 5:
        raise ValueError('guard snapshots must have equal capacity and one to five granted writes')
    ordered = []
    for offset, nbytes in written_ranges:
        bounded_integer(offset, 0, capacity - BLOCK_BYTES, 'write offset')
        if type(nbytes) is not int or nbytes != BLOCK_BYTES or offset % ALIGNMENT:
            raise ValueError('written range must be one aligned complete Qwen block')
        ordered.append((offset, offset + nbytes))
    ordered.sort()
    guards, cursor = [], 0
    for begin, end in ordered:
        if begin < cursor:
            raise ValueError('overlapping write masks cannot qualify as guard evidence')
        if begin > cursor:
            guards.append((cursor, begin))
        cursor = end
    if cursor < capacity:
        guards.append((cursor, capacity))
    total = sum(end - begin for begin, end in guards)
    if total < ALIGNMENT:
        raise ValueError('guard evidence needs at least one full alignment interval')
    for begin, end in guards:
        require_equal(before, after, begin, end, 'pool guard')
    return {'scope': SCOPE, 'byte_checks_passed': True, 'pool_bytes': capacity,
            'written_ranges': [list(item) for item in sorted(written_ranges)],
            'guard_intervals': [list(item) for item in guards], 'guard_bytes_compared': total,
            'before_sha256': hashlib.sha256(before).hexdigest(),
            'after_sha256': hashlib.sha256(after).hexdigest()}


def map_capture(stack, path, expected_bytes, identities):
    stream = stack.enter_context(Path(path).open('rb'))
    info = os.fstat(stream.fileno())
    if info.st_size != expected_bytes:
        raise ValueError(f'{path}: capture size does not match {expected_bytes} bytes')
    physical = (info.st_dev, info.st_ino)
    if physical in identities:
        raise ValueError(f'{path}: each before/after capture must be a distinct file, including hardlinks')
    identities[physical] = {'path': str(Path(path).resolve()), 'bytes': expected_bytes,
                            'device': info.st_dev, 'inode': info.st_ino}
    return stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    trip = commands.add_parser('roundtrip', help='Compare four complete compact block captures.')
    for name in ('source', 'overwritten', 'committed', 'restored'):
        trip.add_argument('--' + name, type=Path, required=True)
    for name in ('num-pages', 'source-page', 'target-page'):
        trip.add_argument('--' + name, type=int, required=True)
    guards = commands.add_parser('guards', help='Check all bytes outside exact granted write ranges.')
    guards.add_argument('--before', type=Path, required=True)
    guards.add_argument('--after', type=Path, required=True)
    guards.add_argument('--pool-bytes', type=int, default=128 << 20)
    guards.add_argument('--written-range', type=int, nargs=2, action='append', required=True,
                        metavar=('ABSOLUTE_OFFSET', 'NBYTES'))
    for command in (trip, guards):
        command.add_argument('--output', type=Path, help='New report file; existing evidence is never overwritten.')
    args = parser.parse_args(argv)
    report, code = {'scope': SCOPE, 'byte_checks_passed': False}, 1
    identities = {}
    try:
        with ExitStack() as stack:
            if args.command == 'roundtrip':
                buffers = [map_capture(stack, getattr(args, name), BLOCK_BYTES, identities)
                           for name in ('source', 'overwritten', 'committed', 'restored')]
                report = verify_roundtrip(*buffers, num_pages=args.num_pages,
                                          source_page=args.source_page, target_page=args.target_page)
            else:
                bounded_integer(args.pool_bytes, BLOCK_BYTES + ALIGNMENT, MAX_POOL_BYTES, 'pool capacity')
                before, after = [map_capture(stack, path, args.pool_bytes, identities)
                                 for path in (args.before, args.after)]
                report = verify_guards(before, after, args.written_range)
        code = 0
    except (OSError, ValueError) as error:
        report['error'] = str(error)
    report['input_files'] = list(identities.values())
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output is not None:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
