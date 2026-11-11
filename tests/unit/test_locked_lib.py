# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise source overlays against real isolated Git indexes; no network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('locked_source', Path(__file__).resolve().parents[2] / 'tools/prepare_uflow_lib.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LockedSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / 'checkout'
        self.checkout.mkdir()
        self.git('init', '-q')
        self.git('config', 'core.autocrlf', 'false')
        (self.checkout / 'old.txt').write_bytes(b'old\n')
        (self.checkout / 'untouched.txt').write_bytes(b'keep\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()
        (self.checkout / 'old.txt').write_bytes(b'new\n')
        (self.checkout / 'added.txt').write_bytes(b'added\n')
        self.git('add', '.')
        patch = self.git('diff', '--cached', '--binary', '--full-index').encode()
        tree = self.git('write-tree').strip()
        files = {n: {'mode': '100644', 'sha256': hashlib.sha256((self.checkout / n).read_bytes()).hexdigest()}
                 for n in ['old.txt', 'added.txt']}
        self.git('read-tree', '--reset', '-u', 'HEAD')
        self.patch = self.root / 'change.patch'
        self.patch.write_bytes(patch)
        self.lock = self.root / 'lock.json'
        self.lock.write_text(json.dumps({'schema': 1, 'sources': {'fixture': {'base': self.base,
            'accepted_source_commit': None, 'accepted_tree': tree, 'files': files,
            'patch': 'change.patch', 'patch_sha256': hashlib.sha256(patch).hexdigest()}}}))

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.checkout), *args], check=True,
                              capture_output=True, text=True).stdout

    def test_apply_matches_exact_tree_and_repeats_without_a_commit(self):
        self.assertEqual(module.apply(self.lock, 'fixture', self.checkout), 'applied')
        self.assertEqual(module.apply(self.lock, 'fixture', self.checkout, True), 'verified')
        self.assertEqual(module.apply(self.lock, 'fixture', self.checkout), 'verified')
        self.assertEqual(self.git('rev-parse', 'HEAD').strip(), self.base)

    def test_check_missing_overlay_keeps_source_unchanged(self):
        with self.assertRaisesRegex(ValueError, 'missing or modified'):
            module.apply(self.lock, 'fixture', self.checkout, True)
        self.assertEqual((self.checkout / 'old.txt').read_bytes(), b'old\n')
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_unrelated_dirty_file_is_preserved(self):
        (self.checkout / 'untouched.txt').write_bytes(b'user edit\n')
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            module.apply(self.lock, 'fixture', self.checkout)
        self.assertEqual((self.checkout / 'untouched.txt').read_bytes(), b'user edit\n')

    def test_tampered_patch_is_rejected_before_applying(self):
        self.patch.write_bytes(self.patch.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            module.apply(self.lock, 'fixture', self.checkout)
        self.assertEqual(self.git('status', '--porcelain'), '')


if __name__ == '__main__':
    unittest.main()
