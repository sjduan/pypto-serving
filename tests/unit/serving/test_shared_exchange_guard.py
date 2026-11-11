# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Portable policy fixtures; actual loader/fence proof is a separate Linux run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from examples.run03.host_exchange_guard import (
    SHIM_SHA, attempt_token, check_bindings, judge, record_and_check,
)


def controls():
    return {str(i): {'target': '/dev/davinci_manager' if i < 3 else '/dev/hisi_hdc',
                     'is_char': True, 'rdev': [235, 0] if i < 3 else [509, 0]} for i in range(4)}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.loaded = {'shim': SHIM_SHA, 'fence': 'test-fence-hash', 'cann:/actual/libascendcl.so': 'cann-hash'}

    def test_exact_controls_only_for_daemon(self):
        judge(controls(), 'daemon', self.loaded, self.loaded)
        for role in ('producer', 'consumer'):
            with self.subTest(role=role), self.assertRaises(RuntimeError):
                judge(controls(), role)
            judge({}, role)

    def test_numbered_devmm_unknown_rejected(self):
        for target in ('/dev/davinci0', '/dev/devmm_svm', '/dev/ascend_unknown', '/dev/unexpected'):
            fds = controls()
            fds['4'] = {'target': target, 'is_char': True, 'rdev': [1, 2]}
            with self.subTest(target=target), self.assertRaises(RuntimeError):
                judge(fds, 'daemon', self.loaded, self.loaded)

    def test_count_type_and_rdev_rejected(self):
        cases = []
        missing = controls()
        del missing['0']
        cases.append(missing)
        extra = controls()
        extra['4'] = extra['0']
        cases.append(extra)
        for field, value in (('is_char', False), ('rdev', [235, 1])):
            wrong = controls()
            wrong['0'][field] = value
            cases.append(wrong)
        for fds in cases:
            with self.subTest(fds=fds), self.assertRaises(RuntimeError):
                judge(fds, 'daemon', self.loaded, self.loaded)

    def test_wrong_actual_shim_cann_fence_hash_rejected(self):
        for key in self.loaded:
            wrong = dict(self.loaded, **{key: 'wrong-hash'})
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                judge(controls(), 'daemon', wrong, self.loaded)
        wrong = dict(self.loaded, shim='wrong-same-baseline')
        with self.assertRaises(RuntimeError):
            judge(controls(), 'daemon', wrong, wrong)

    def test_missing_fence_rejected(self):
        wrong = copy.copy(self.loaded)
        del wrong['fence']
        with self.assertRaises(RuntimeError):
            judge(controls(), 'daemon', wrong, wrong)

    def test_binding_missing_or_wrong_rejected(self):
        path = str(Path('fixture-fence.so').resolve())
        good = dict.fromkeys(('uf_acl_backend_init', 'aclInit', 'aclrtSetDevice', 'aclrtMalloc'), path)
        check_bindings(good, path)
        for bad in ({}, dict(good, aclInit='real-cann.so')):
            with self.assertRaises(RuntimeError):
                check_bindings(bad, path)

    def test_complete_inventory_persisted_before_policy_failure(self):
        fds = controls()
        fds['7'] = {'target': '/dev/davinci3', 'is_char': True, 'rdev': [236, 3]}
        with tempfile.TemporaryDirectory() as directory:
            output, report = Path(directory)/'failure.json', {}
            with patch('examples.run03.host_exchange_guard.inventory', return_value=fds), \
                    patch('examples.run03.host_exchange_guard.deployment', return_value=self.loaded), \
                    self.assertRaises(RuntimeError):
                record_and_check(report, 'failed', 123, 'daemon', output, 'shim', 'fence', self.loaded)
            self.assertEqual(json.loads(output.read_text())['owned_fds']['failed'], fds)

    def test_inventory_preserved_when_fence_mapping_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'failure.json'
            with patch('examples.run03.host_exchange_guard.inventory', return_value=controls()), \
                    patch('examples.run03.host_exchange_guard.deployment', side_effect=RuntimeError('missing fence')), \
                    self.assertRaises(RuntimeError):
                record_and_check({}, 'failed', 123, 'daemon', output, 'shim', 'fence', self.loaded)
            self.assertEqual(json.loads(output.read_text())['owned_fds']['failed'], controls())

    def test_attempt_token_is_bounded_not_a_root(self):
        self.assertEqual(attempt_token('attempt2-api-fenced'), 'attempt2-api-fenced')
        for token in ('', '../attempt2-a', 'attempt2-a/x', 'attempt1-a', 'attempt2-'+('a'*50)):
            with self.subTest(token=token), self.assertRaises(ValueError):
                attempt_token(token)


if __name__ == '__main__':
    unittest.main()
