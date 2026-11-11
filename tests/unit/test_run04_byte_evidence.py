# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synthetic raw-byte oracle tests; never native/model/capture-origin evidence."""
import importlib.util
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'examples/run04/verify_bytes.py'
SPEC = importlib.util.spec_from_file_location('run04_verify_bytes', SCRIPT)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class RoundtripEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = bytes(ORACLE.BLOCK_BYTES)
        cls.overwritten = bytearray(ORACLE.BLOCK_BYTES)
        for offset in range(0, ORACLE.BLOCK_BYTES, ORACLE.TOKEN_BYTES):
            cls.overwritten[offset] = 1

    def verify(self, *, overwritten=None, committed=None, restored=None, **pages):
        return ORACLE.verify_roundtrip(self.source, self.overwritten if overwritten is None else overwritten,
            self.source if committed is None else committed, self.source if restored is None else restored,
            **({'num_pages': 16, 'source_page': 7, 'target_page': 0} | pages))

    def test_complete_20mib_80_sides_and_every_token_row_are_checked(self):
        result = self.verify()
        self.assertTrue(result['byte_checks_passed'])
        self.assertEqual(len(result['sides']), 80)
        self.assertEqual(sum(row['changed_token_rows'] for row in result['sides']), 80 * 128)
        self.assertEqual(result['sha256']['source'], result['sha256']['committed'])
        self.assertEqual(result['sha256']['source'], result['sha256']['restored'])
        self.assertNotEqual(result['sha256']['source'], result['sha256']['overwritten'])
        self.assertIn('not verified', result['scope'])

    def test_corruption_at_last_byte_is_rejected_in_both_pool_and_restored_capture(self):
        wrong = bytearray(self.source)
        wrong[-1] = 1
        for name in ('committed', 'restored'):
            with self.subTest(capture=name), self.assertRaisesRegex(ValueError, str(ORACLE.BLOCK_BYTES - 1)):
                self.verify(**{name: wrong})

    def test_one_unchanged_token_in_last_layer_value_side_cannot_qualify_as_full_source_overwrite(self):
        wrong = bytearray(self.overwritten)
        wrong[-ORACLE.TOKEN_BYTES] = 0
        with self.assertRaisesRegex(ValueError, 'layer 39 side 1: only 127/128'):
            self.verify(overwritten=wrong)

    def test_partial_snapshot_cannot_pass_a_full_page_check(self):
        with self.assertRaisesRegex(ValueError, 'exactly 20971520'):
            self.verify(restored=self.source[:-1])

    def test_same_page_out_of_range_or_bool_page_is_rejected(self):
        for arguments in ({'target_page': 7}, {'source_page': -1}, {'target_page': 16},
                          {'source_page': True}, {'num_pages': True}, {'num_pages': 1}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.verify(**arguments)

    def test_real_file_cli_records_success_and_preserves_existing_reports(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original, overwritten = root / 'source.bin', root / 'overwritten.bin'
            committed, restored = root / 'committed.bin', root / 'restored.bin'
            original.write_bytes(self.source)
            overwritten.write_bytes(self.overwritten)
            committed.write_bytes(self.source)
            restored.write_bytes(self.source)
            output = root / 'report.json'
            args = ['roundtrip', '--source', str(original), '--overwritten', str(overwritten),
                    '--committed', str(committed), '--restored', str(restored),
                    '--num-pages', '16', '--source-page', '7', '--target-page', '0', '--output', str(output)]
            self.assertEqual(ORACLE.main(args), 0)
            result = json.loads(output.read_text())
            self.assertTrue(result['byte_checks_passed'])
            self.assertEqual(len(result['input_files']), 4)
            saved = output.read_bytes()
            with self.assertRaises(FileExistsError):
                ORACLE.main(args)
            self.assertEqual(output.read_bytes(), saved)

    def test_cli_rejects_reusing_the_same_capture_or_a_hardlink_as_a_second_observation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original, alias = root / 'source.bin', root / 'hardlink.bin'
            original.write_bytes(self.source)
            os.link(original, alias)
            for index, second in enumerate((original, alias)):
                with self.subTest(second=second):
                    output = root / f'failed-{index}.json'
                    args = ['roundtrip', '--source', str(original), '--overwritten', str(second),
                            '--committed', str(original), '--restored', str(original),
                            '--num-pages', '16', '--source-page', '7', '--target-page', '0',
                            '--output', str(output)]
                    self.assertEqual(ORACLE.main(args), 1)
                    result = json.loads(output.read_text())
                    self.assertFalse(result['byte_checks_passed'])
                    self.assertIn('distinct file', result['error'])

    def test_cli_records_failure_in_a_new_file_when_capture_is_truncated(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            truncated = root / 'short.bin'
            truncated.write_bytes(b'not-a-block')
            output = root / 'failed.json'
            args = ['roundtrip', '--source', str(truncated), '--overwritten', str(truncated),
                    '--committed', str(truncated), '--restored', str(truncated),
                    '--num-pages', '16', '--source-page', '7', '--target-page', '0', '--output', str(output)]
            self.assertEqual(ORACLE.main(args), 1)
            result = json.loads(output.read_text())
            self.assertFalse(result['byte_checks_passed'])
            self.assertIn('size does not match', result['error'])


class GuardEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.before = mmap.mmap(-1, 128 << 20)
        self.after = mmap.mmap(-1, 128 << 20)
        self.ranges = [(start << 20, ORACLE.BLOCK_BYTES) for start in (2, 24, 46, 68, 90)]
        for offset, _ in self.ranges:
            self.after[offset] = 1  # explicit synthetic writes only in granted payloads

    def tearDown(self):
        self.before.close()
        self.after.close()

    def test_every_guard_byte_is_compared_including_gaps_and_tail(self):
        result = ORACLE.verify_guards(self.before, self.after, self.ranges)
        self.assertTrue(result['byte_checks_passed'])
        self.assertEqual(result['guard_bytes_compared'], 28 << 20)
        self.assertEqual(result['guard_intervals'][0], [0, 2 << 20])
        self.assertEqual(result['guard_intervals'][-1], [110 << 20, 128 << 20])

    def test_first_middle_gap_and_final_pool_byte_corruption_are_all_detected(self):
        for offset in (0, 45 << 20, (128 << 20) - 1):
            with self.subTest(offset=offset):
                self.after[offset] = 1
                with self.assertRaisesRegex(ValueError, str(offset)):
                    ORACLE.verify_guards(self.before, self.after, self.ranges)
                self.after[offset] = 0

    def test_overlap_alignment_widened_or_invalid_write_masks_cannot_hide_guard_corruption(self):
        for ranges in ([(0, ORACLE.BLOCK_BYTES), (65536, ORACLE.BLOCK_BYTES)],
                       [(1, ORACLE.BLOCK_BYTES)], [(True, ORACLE.BLOCK_BYTES)],
                       [(0, True)], [(0, ORACLE.BLOCK_BYTES + 65536)],
                       [(-65536, ORACLE.BLOCK_BYTES)], [(120 << 20, ORACLE.BLOCK_BYTES)], [],
                       self.ranges + self.ranges):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                ORACLE.verify_guards(self.before, self.after, ranges)

    def test_truncated_pool_snapshot_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'equal capacity'):
            ORACLE.verify_guards(self.before, b'incomplete', self.ranges)

    def test_guard_cli_uses_actual_snapshot_files_and_records_a_later_guard_failure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            before, after = root / 'before.bin', root / 'after.bin'
            capacity = ORACLE.BLOCK_BYTES + 2 * ORACLE.ALIGNMENT
            for path in (before, after):
                with path.open('wb') as stream:
                    stream.truncate(capacity)
            with after.open('r+b') as stream:
                stream.seek(ORACLE.ALIGNMENT)
                stream.write(b'payload-write')
            args = ['guards', '--before', str(before), '--after', str(after),
                    '--pool-bytes', str(capacity), '--written-range', str(ORACLE.ALIGNMENT),
                    str(ORACLE.BLOCK_BYTES), '--output', str(root / 'good.json')]
            self.assertEqual(ORACLE.main(args), 0)
            report = json.loads((root / 'good.json').read_text())
            self.assertEqual(report['guard_bytes_compared'], 2 * ORACLE.ALIGNMENT)
            self.assertEqual(len(report['input_files']), 2)
            with after.open('r+b') as stream:
                stream.write(b'bad-guard')
            args[-1] = str(root / 'failed.json')
            self.assertEqual(ORACLE.main(args), 1)
            report = json.loads((root / 'failed.json').read_text())
            self.assertFalse(report['byte_checks_passed'])
            self.assertIn('offset 0', report['error'])


class ImportBoundaryTests(unittest.TestCase):
    def test_offline_cli_imports_no_tensor_sdk_or_device_runtime_in_a_fresh_process(self):
        code = '''import importlib.abc, runpy, sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'numpy','torch','torch_npu','pypto','simpler','uflow','acl'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
        subprocess.run([sys.executable, '-c', code, str(SCRIPT)], check=True, timeout=15,
                       stdout=subprocess.DEVNULL)
