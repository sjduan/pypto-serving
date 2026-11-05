# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Recorded Catalog metadata only; no UFlow/device/client operations or cleanup claims."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'examples/run04/existing_pool.py'
spec = importlib.util.spec_from_file_location('existing_pool_probe', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RECORDED = json.loads((Path(__file__).with_name('fixtures') / 'existing_quarantine_snapshot.json').read_text())


def two_pools():
    snapshot = copy.deepcopy(RECORDED)
    normal = copy.deepcopy(snapshot['pools'][0])
    normal.update(pool_id='new-normal-pool', pool_epoch='new-epoch', resource_id=normal['resource_id'] + 1,
                  frozen=False, clients=[], ranges={'allocations': [], 'leases': []})
    normal['canonical']['inode'] += 1
    snapshot['pools'].append(normal)
    snapshot['charged_bytes'] = 2 * module.POOL_BYTES
    snapshot['revision'] += 5
    return snapshot, normal


class ExistingCatalogTests(unittest.TestCase):
    def setUp(self):
        self.expected = copy.deepcopy(module.retained_pool(RECORDED))

    def test_exact_recorded_unknown_remains_a_full_charge(self):
        result = module.check_catalog(RECORDED, self.expected)
        self.assertEqual(result['charged_bytes'], module.POOL_BYTES)
        self.assertEqual(result['quarantine']['clients'][0]['state'], 'Unknown')

    def test_separate_normal_pool_shares_budget_without_hiding_quarantine(self):
        snapshot, normal = two_pools()
        result = module.check_catalog(snapshot, self.expected, normal['pool_id'])
        self.assertEqual(result['charged_bytes'], 268435456)
        self.assertEqual(result['quarantine'], self.expected)

    def test_legitimate_control_history_growth_is_not_original_pool_mutation(self):
        snapshot = copy.deepcopy(RECORDED)
        snapshot['revision'] += 10
        snapshot['control_counts']['GetManagedPoolSnapshot'] += 20
        module.check_catalog(snapshot, self.expected)

    def test_normal_released_record_does_not_make_total_budget_zero(self):
        snapshot, normal = two_pools()
        normal['state'] = 'Released'
        snapshot['charged_bytes'] = module.POOL_BYTES
        module.check_catalog(snapshot, self.expected)

    def test_original_identity_and_typed_projection_changes_are_rejected(self):
        for field, value in [('pool_epoch', 'other'), ('provider_session', 'other'),
                             ('frozen', False), ('nbytes', 1), ('resource_id', True)]:
            with self.subTest(field=field):
                snapshot = copy.deepcopy(RECORDED)
                snapshot['pools'][0][field] = value
                with self.assertRaises(ValueError):
                    module.check_catalog(snapshot, self.expected)

    def test_original_unknown_cannot_be_relabelled_clean(self):
        snapshot = copy.deepcopy(RECORDED)
        snapshot['pools'][0]['clients'][0]['state'] = 'Unmapped'
        with self.assertRaises(ValueError):
            module.check_catalog(snapshot, self.expected)

    def test_wrong_total_reservation_or_budget_is_rejected(self):
        for field, value in [('charged_bytes', 0), ('budget_bytes', 3 * module.POOL_BYTES),
                             ('creation_reserved_bytes', 1), ('charged_bytes', float(module.POOL_BYTES))]:
            with self.subTest(field=field, value=value):
                snapshot = copy.deepcopy(RECORDED)
                snapshot[field] = value
                with self.assertRaises(ValueError):
                    module.check_catalog(snapshot, self.expected)

    def test_second_pool_must_be_explicitly_owned_and_physically_distinct(self):
        snapshot, normal = two_pools()
        with self.assertRaises(ValueError):
            module.check_catalog(snapshot, self.expected)
        normal['canonical']['inode'] = self.expected['canonical']['inode']
        with self.assertRaises(ValueError):
            module.check_catalog(snapshot, self.expected, normal['pool_id'])

    def test_new_unknown_or_frozen_pool_stops_continuation(self):
        for fault in ['frozen', 'client', 'lease']:
            with self.subTest(fault=fault):
                snapshot, normal = two_pools()
                if fault == 'frozen':
                    normal['frozen'] = True
                elif fault == 'client':
                    normal['clients'] = [{'state': 'Unknown', 'children': []}]
                else:
                    normal['ranges']['leases'] = [{'state': 'Unknown'}]
                with self.assertRaises(ValueError):
                    module.check_catalog(snapshot, self.expected, normal['pool_id'])

    def test_fake_released_record_with_access_is_rejected(self):
        snapshot, normal = two_pools()
        normal['state'] = 'Released'
        normal['clients'] = [{'state': 'Unknown', 'children': []}]
        snapshot['charged_bytes'] = module.POOL_BYTES
        with self.assertRaises(ValueError):
            module.check_catalog(snapshot, self.expected)


if __name__ == '__main__':
    unittest.main()
