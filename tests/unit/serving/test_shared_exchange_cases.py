# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Portable declared-case ordering, raw-error origin and evidence-retention tests."""
import json
from pathlib import Path
import tempfile
import unittest

from examples.run03.host_exchange_cases import (
    ALL_FIELDS, EXPOSURE_FIELDS, case_for_token, declaration, domain_snapshot,
    explicit_unknown, permit_owner_exit, persist_event, require_zero,
)
from examples.run03.host_exchange_guard import PRISTINE_BLOCKED, REAL_CLEANUP, check_pristine_bindings


class WireError(RuntimeError):
    pass


def exchange(operation, detail, status='error'):
    op, response, received_fds, mapping = operation, {'status': status, 'detail': detail}, [], None
    raise WireError(detail)


class CaseTests(unittest.TestCase):
    def test_normal_owner_cannot_exit_with_open_exposure(self):
        with self.assertRaises(RuntimeError):
            permit_owner_exit('normal-clean', revoked=True)
        nonzero = dict.fromkeys(EXPOSURE_FIELDS, 0)
        nonzero['active_views'] = 1
        with self.assertRaises(RuntimeError):
            permit_owner_exit('normal-clean', revoked=True, exposure_snapshot=nonzero)

    def test_normal_close_proof_while_owner_alive_precedes_exit(self):
        snap = dict.fromkeys(ALL_FIELDS, 0)
        snap['session_pidfds'] = 2
        permit_owner_exit('normal-clean', revoked=True, exposure_snapshot=snap)
        with self.assertRaises(RuntimeError):
            require_zero(snap, ALL_FIELDS)

    def test_negative_is_declared_dirty_without_zero_claim(self):
        result = declaration('owner-exit-negative')
        self.assertEqual(result['expected_authority_mode'], 'dirty')
        self.assertFalse(result['requires_final_zero'])
        self.assertFalse(result['recovery_supported'])
        permit_owner_exit('owner-exit-negative', revoked=True)
        with self.assertRaises(RuntimeError):
            permit_owner_exit('owner-exit-negative', revoked=False)

    def test_case_required_before_fixture_start(self):
        self.assertEqual(case_for_token('attempt3-normal-clean'), 'normal-clean')
        self.assertEqual(case_for_token('attempt4-owner-exit-negative'), 'owner-exit-negative')
        with self.assertRaises(ValueError):
            case_for_token('attempt3-ambiguous')

    def test_early_raw_event_survives_later_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = json.dumps({'event': 'revoked', 'source_sha256': ['left', 'payload', 'right']})
            try:
                persist_event(root/'report.json', root/'events.jsonl', {}, 'producer', 'revoked', raw)
                raise RuntimeError('later consumer failure')
            except RuntimeError:
                pass
            self.assertEqual(json.loads((root/'events.jsonl').read_text())['raw'], raw)
            self.assertEqual(json.loads((root/'report.json').read_text())['peer_events'][0]['raw'], raw)

    def test_malformed_raw_event_also_survives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            persist_event(root/'r.json', root/'j.jsonl', {}, 'consumer', 'held', '{broken')
            with self.assertRaises(json.JSONDecodeError):
                json.loads('{broken')
            self.assertIn('{broken', (root/'j.jsonl').read_text())

    def test_old_domain_snapshot_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority = root/'authority/external-ddr-v0'
            authority.mkdir(parents=True)
            (authority/'state').write_text('dirty')
            before = domain_snapshot(root)
            self.assertEqual(before, domain_snapshot(root))
            (authority/'state').write_text('fake-clean')
            self.assertNotEqual(before, domain_snapshot(root))

    def bindings(self):
        blocked = PRISTINE_BLOCKED|{'aclrtMalloc', 'uf_acl_backend_init', 'uf_acl_d2d'}
        fence, shim = str(Path('fence.so').resolve()), str(Path('real-shim.so').resolve())
        bound = dict.fromkeys(blocked, fence)
        bound[REAL_CLEANUP] = shim
        return blocked, bound, fence, shim

    def test_only_original_finalizer_may_be_unfenced(self):
        blocked, bound, fence, shim = self.bindings()
        exports = {'uf_acl_backend_init', 'uf_acl_d2d', REAL_CLEANUP}
        check_pristine_bindings(blocked, bound, fence, shim, exports)
        with self.assertRaises(RuntimeError):
            check_pristine_bindings(blocked-{'uf_acl_backend_init'}, bound, fence, shim, exports)
        with self.assertRaises(RuntimeError):
            check_pristine_bindings(blocked-{'aclrtFreePhysical'}, bound, fence, shim, exports)

    def test_fake_or_fenced_finalizer_binding_rejected(self):
        blocked, bound, fence, shim = self.bindings()
        for target in (fence, 'stub.so'):
            bound[REAL_CLEANUP] = target
            with self.assertRaises(RuntimeError):
                check_pristine_bindings(blocked, bound, fence, shim, {'uf_acl_backend_init', 'uf_acl_d2d', REAL_CLEANUP})

    def test_expected_unknown_requires_exact_real_exchange_frame(self):
        detail = 'unresolved exposure, authority admission disabled'
        try:
            exchange('AcquireExternalRead', detail)
        except WireError as error:
            proof = explicit_unknown(error, exchange.__code__, WireError, 'AcquireExternalRead')
        self.assertEqual(proof['response'], {'status': 'error', 'detail': detail})

    def test_network_protocol_or_same_text_cannot_be_expected_unknown(self):
        detail = 'unresolved exposure, authority admission disabled'
        for error in (WireError(detail), OSError(detail)):
            self.assertIsNone(explicit_unknown(error, exchange.__code__, WireError, 'AcquireExternalRead'))
        try:
            exchange('AcquireExternalRead', detail, status='ok')
        except WireError as error:
            self.assertIsNone(explicit_unknown(error, exchange.__code__, WireError, 'AcquireExternalRead'))


if __name__ == '__main__':
    unittest.main()
