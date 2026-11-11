# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stdlib-only protocol checks: no real backing, E10 authority, runtime or NPU."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

from examples.run03.host_record_probe import (
    fixture_compatibility, fixture_descriptor, fixture_destination, fixture_record,
)
from pypto_serving.serving.native_ddr.shared_compatibility import (
    BLOCK_BYTES, EVIDENCE_LEVEL, MAX_INTEGER, MAX_RECORD_BYTES, SharedCompatibility,
    TrustedDeploymentAttestation, build_shared_compatibility, canonical_json, descriptor_digest,
)
from pypto_serving.serving.native_ddr.shared_prefix_record import (
    ExternalBackingIdentity,
    decode_destination, decode_record, encode_destination, encode_record, validate_candidate,
)

ROOT = Path(__file__).resolve().parents[3]
DIFFERENT_SHA = hashlib.sha256(b"different-data").hexdigest()


def attest(descriptor, reference="HOST-FIXTURE/trusted-context"):
    return TrustedDeploymentAttestation(EVIDENCE_LEVEL, reference, descriptor_digest(descriptor))


def compatibility(descriptor):
    return build_shared_compatibility(descriptor, trusted_attestation=attest(descriptor))


class SharedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.descriptor = fixture_descriptor()
        self.compatibility = fixture_compatibility()

    def test_asset_order_and_attestation_reference_do_not_change_data_id(self):
        reordered = deepcopy(self.descriptor)
        reordered["assets"].reverse()
        other = build_shared_compatibility(reordered, trusted_attestation=attest(reordered, "OTHER-HISTORICAL-REF"))
        self.assertEqual(self.compatibility.data_id, other.data_id)
        self.assertEqual(self.compatibility.descriptor_json, other.descriptor_json)

    def test_explicit_full_attestation_is_required_even_for_direct_construction(self):
        for missing in (None, True, {}, "same-model"):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                build_shared_compatibility(self.descriptor, trusted_attestation=missing)
            with self.assertRaises(ValueError):
                SharedCompatibility(canonical_json(self.descriptor), missing)
        for level in ("header-stat", "full-small-asset-content", "unknown", ""):
            with self.subTest(level=level), self.assertRaises(ValueError):
                TrustedDeploymentAttestation(level, "source", self.compatibility.data_id)

    def test_old_attestation_rejects_changes_to_every_identity_component(self):
        variants = []
        for index, asset in enumerate(self.descriptor["assets"]):
            changed = deepcopy(self.descriptor)
            changed["assets"][index]["sha256"] = DIFFERENT_SHA
            if asset["name"] == "config.json":
                changed["model_config_sha256"] = DIFFERENT_SHA
            variants.append((asset["name"], changed))
        for component in self.descriptor["deployment"]:
            changed = deepcopy(self.descriptor)
            changed["deployment"][component] = DIFFERENT_SHA
            variants.append((component, changed))
        for section, field, value in (
                ("rope", "theta", 500000),
                ("position", "max_position_embeddings", 8192),
                ("tokenizer", "implementation", "DifferentTokenizer")):
            changed = deepcopy(self.descriptor)
            changed[section][field] = value
            variants.append((section, changed))
        for label, changed in variants:
            with self.subTest(component=label):
                with self.assertRaises(ValueError):
                    build_shared_compatibility(changed, trusted_attestation=attest(self.descriptor))
                changed_identity = compatibility(changed)
                self.assertNotEqual(changed_identity.data_id, self.compatibility.data_id)

    def test_unknown_or_unsupported_identity_is_rejected(self):
        variants = []
        for field, value in (
                ("schema", 2), ("schema", True), ("tensor_parallel", 2), ("tensor_parallel", True),
                ("model", "Qwen3-8B"), ("dtype", "float16"), ("adapter", "lora"),
                ("quantization", "int8"), ("worker_epoch", "must-not-enter-shared-data"),
                ("num_pages", 16), ("assets_path", "/local/model")):
            changed = deepcopy(self.descriptor)
            changed[field] = value
            variants.append((field, changed))
        for section, field, value in (
                ("rope", "scaling", "dynamic"),
                ("rope", "theta", 0),
                ("position", "scheme", "unknown"),
                ("position", "max_position_embeddings", True),
                ("tokenizer", "implementation", "unknown"),
                ("layout", "num_layers", 39),
                ("layout", "num_layers", True),
                ("layout", "sides", ["V", "K"]),
                ("layout", "ddr_order", "different"),
                ("layout", "layout_version", "new-version")):
            changed = deepcopy(self.descriptor)
            changed[section][field] = value
            variants.append((section + "." + field, changed))
        for label, changed in variants:
            with self.subTest(field=label), self.assertRaises(ValueError):
                compatibility(changed)
        for field in self.descriptor:
            changed = deepcopy(self.descriptor)
            del changed[field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                compatibility(changed)

    def test_complete_named_manifest_and_config_binding(self):
        for mutate in (
                lambda d: d["assets"].pop(),
                lambda d: d["assets"].append(dict(d["assets"][0])),
                lambda d: d["assets"].__setitem__(0, dict(d["assets"][1])),
                lambda d: d["assets"][0].update(name="../model.safetensors"),
                lambda d: d["assets"][0].update(header_sha256=DIFFERENT_SHA),
                lambda d: d["assets"][0].update(sha256="0" * 63),
                lambda d: d["assets"][0].update(sha256="G" * 64),
                lambda d: d["assets"][0].update(sha256="a" * 64, size_bytes=True),
                lambda d: d["assets"][0].update(size_bytes=-1),
                lambda d: d["assets"][0].update(size_bytes=MAX_INTEGER + 1),
                lambda d: d.update(model_config_sha256=DIFFERENT_SHA),
                lambda d: d["deployment"].pop("native_backend"),
                lambda d: d["deployment"].update(unknown="a" * 64)):
            changed = deepcopy(self.descriptor)
            mutate(changed)
            with self.subTest(descriptor=changed), self.assertRaises(ValueError):
                compatibility(changed)
        sized = deepcopy(self.descriptor)
        sized["assets"][0]["size_bytes"] = 728
        self.assertEqual(compatibility(sized).to_dict()["assets"][0]["size_bytes"], 728)

    def test_identity_is_immutable_and_has_no_local_paths_or_epochs(self):
        exported = self.compatibility.to_dict()
        exported["assets"][0]["sha256"] = DIFFERENT_SHA
        self.assertEqual(self.compatibility.data_id, fixture_compatibility().data_id)
        text = self.compatibility.descriptor_json.decode()
        for excluded in ("worker_epoch", "num_pages", "mtime", "inode", "resolved", "evidence_reference"):
            self.assertNotIn(excluded, text)
        self.assertNotIn("Ready", self.compatibility.__dataclass_fields__)


class SharedRecordTests(unittest.TestCase):
    def setUp(self):
        self.record = fixture_record()
        self.destination = fixture_destination()

    def validate(self, record=None, destination=None, **changes):
        values = dict(
            expected_compatibility=fixture_compatibility(), expected_source=self.record.source,
            destination=destination or self.destination, expected_destination=self.destination,
            expected_chain_hash=self.record.chain_hash, expected_token_start=self.record.token_start,
            expected_payload_sha256=self.record.payload_sha256,
        )
        values.update(changes)
        return validate_candidate(record or self.record, **values)

    def test_independent_replica_capacity_epoch_and_path_are_not_content_identity(self):
        first = self.validate()
        other_destination = replace(self.destination, worker_epoch="replica-two", arena_id="different-arena",
                                    cache_epoch="different-cache", num_pages=64, page_id=63,
                                    page_lease_id="other-lease")
        second = self.validate(destination=other_destination, expected_destination=other_destination)
        other_source = replace(self.record.source, worker_epoch="other-source-epoch",
                               backing=replace(self.record.source.backing, owner_incarnation="other-source-owner"))
        other_record = replace(self.record, source=other_source)
        third = self.validate(other_record, expected_source=other_source)
        self.assertEqual(first.record.compatibility_id, second.record.compatibility_id)
        self.assertEqual(first.record.compatibility_id, third.record.compatibility_id)
        self.assertEqual(first.record.source, self.record.source)
        self.assertNotEqual(first.record.source.worker_epoch, first.destination.worker_epoch)

    def test_complete_thirteen_field_source_roundtrip(self):
        backing = self.record.source.backing
        self.assertEqual(len(asdict(backing)), 13)
        self.assertEqual(ExternalBackingIdentity.from_dict(asdict(backing)), backing)
        self.assertEqual(decode_record(encode_record(self.record)), self.record)
        self.assertEqual(decode_destination(encode_destination(self.destination)), self.destination)
        self.assertEqual(encode_record(decode_record(encode_record(self.record))), encode_record(self.record))

    def test_each_source_identity_field_must_match_current_authority(self):
        for name, value in asdict(self.record.source.backing).items():
            if name == "purpose":
                with self.assertRaises(ValueError):
                    replace(self.record.source.backing, purpose="staging")
                continue
            changed_value = value + 1 if type(value) is int else value + "-new"
            backing = replace(self.record.source.backing, **{name: changed_value})
            stale = replace(self.record, source=replace(self.record.source, backing=backing))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "stale source"):
                self.validate(stale)
        for name, value in (("worker_epoch", "retired"), ("entry_id", "other-entry"),
                            ("entry_generation", 8), ("offset_bytes", 0)):
            stale = replace(self.record, source=replace(self.record.source, **{name: value}))
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "stale source"):
                self.validate(stale)

    def test_each_destination_identity_field_must_match_current_lease(self):
        for name, value in (("worker_epoch", "retired"), ("arena_id", "other"),
                            ("cache_epoch", "retired-cache"), ("num_pages", 32),
                            ("page_id", 8), ("page_lease_id", "retired-lease")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "stale destination"):
                self.validate(destination=replace(self.destination, **{name: value}))

    def test_same_data_does_not_bypass_old_expected_source_or_page_lease(self):
        current_source = replace(self.record.source, entry_generation=8,
                                 backing=replace(self.record.source.backing, resource_generation=3))
        with self.assertRaisesRegex(ValueError, "stale source"):
            self.validate(expected_source=current_source)
        current_page = replace(self.destination, page_lease_id="fresh-lease")
        with self.assertRaisesRegex(ValueError, "stale destination"):
            self.validate(expected_destination=current_page)

    def test_content_chain_position_and_payload_must_match(self):
        for stale in (
                replace(self.record, compatibility_id=DIFFERENT_SHA),
                replace(self.record, chain_hash=b"z" * 32),
                replace(self.record, token_start=256, token_end=384),
                replace(self.record, payload_sha256=DIFFERENT_SHA)):
            with self.subTest(record=stale), self.assertRaises(ValueError):
                self.validate(stale)
        # Same local block content at another parent prefix must not match.
        with self.assertRaisesRegex(ValueError, "prefix chain"):
            self.validate(expected_chain_hash=hashlib.sha256(b"different-parent").digest())
        with self.assertRaises(ValueError):
            self.validate(replace(self.record, token_start=40960, token_end=41088),
                          expected_token_start=40960)

    def test_aligned_nonzero_offset_and_exact_last_block(self):
        self.assertEqual(self.record.source.offset_bytes, BLOCK_BYTES)
        self.assertEqual(self.record.source.offset_bytes + self.record.source.nbytes,
                         self.record.source.backing.size_bytes)
        self.validate()
        for change in (dict(offset_bytes=1), dict(offset_bytes=-BLOCK_BYTES),
                       dict(offset_bytes=BLOCK_BYTES * 2), dict(offset_bytes=MAX_INTEGER),
                       dict(nbytes=BLOCK_BYTES - 1), dict(nbytes=BLOCK_BYTES + 1), dict(nbytes=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.record.source, **change)

    def test_token_boundaries_and_hash_types(self):
        for change in (
                dict(token_start=1, token_end=129), dict(token_start=0, token_end=256),
                dict(token_start=-128, token_end=0), dict(token_start=True),
                dict(token_end=MAX_INTEGER + 1), dict(chain_hash=b"x" * 31),
                dict(chain_hash="a" * 64), dict(payload_sha256="A" * 64), dict(schema=True)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.record, **change)
        first = replace(self.record, token_start=0, token_end=128)
        self.validate(first, expected_token_start=0)

    def test_destination_range_layout_and_typed_fields(self):
        for change in (
                dict(num_pages=0), dict(num_pages=True), dict(num_pages=MAX_INTEGER),
                dict(page_id=-1), dict(page_id=31), dict(page_id=True),
                dict(num_layers=39), dict(page_size=64), dict(num_kv_heads=4),
                dict(head_dim=True), dict(dtype="float16"), dict(layout_version="v2"),
                dict(worker_epoch=""), dict(page_lease_id="unknown"), dict(arena_id="x\n")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.destination, **change)

    def test_backing_numeric_fields_reject_bool_negative_zero_and_overflow(self):
        for field in ("pool_generation", "resource_generation", "object_id", "placement_id",
                      "binding_id", "owner_lease_id", "size_bytes"):
            for invalid in (True, -1, 0, MAX_INTEGER + 1, "1", 1.0):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    replace(self.record.source.backing, **{field: invalid})
        for field in ("worker_epoch", "entry_id"):
            with self.assertRaises(ValueError):
                replace(self.record.source, **{field: "x" * 257})
        with self.assertRaises(ValueError):
            replace(self.record.source, entry_generation=True)

    def test_validation_is_metadata_only_and_preserves_inputs(self):
        before = encode_record(self.record), encode_destination(self.destination)
        candidate = self.validate()
        self.assertEqual((encode_record(self.record), encode_destination(self.destination)), before)
        self.assertEqual(set(candidate.__dataclass_fields__), {"record", "destination"})
        self.assertFalse(hasattr(candidate, "ready"))
        self.assertFalse(hasattr(candidate, "transfer_request"))


class StrictCodecTests(unittest.TestCase):
    def setUp(self):
        self.record = fixture_record()
        self.wire = encode_record(self.record)

    def test_all_required_fields_and_unknown_fields_at_each_level(self):
        for scope in ("record", "source", "backing"):
            original = self.record.to_dict()
            target = original if scope == "record" else original["source"]
            if scope == "backing":
                target = target["backing"]
            for missing in list(target):
                changed = deepcopy(original)
                child = changed if scope == "record" else changed["source"]
                if scope == "backing":
                    child = child["backing"]
                del child[missing]
                with self.subTest(scope=scope, missing=missing), self.assertRaises(ValueError):
                    decode_record(canonical_json(changed))
            target["extra"] = "not-allowed"
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                decode_record(canonical_json(original))
        target = asdict(fixture_destination())
        for missing in target:
            changed = dict(target)
            del changed[missing]
            with self.assertRaises(ValueError):
                decode_destination(canonical_json(changed))

    def test_duplicate_keys_at_top_and_nested_levels(self):
        for raw in (
                self.wire.replace(b'"schema":1', b'"schema":1,"schema":1'),
                self.wire.replace(b'"object_id":3', b'"object_id":3,"object_id":3'),
                self.wire.replace(b'"worker_epoch":', b'"worker_epoch":"forged","worker_epoch":')):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "duplicate"):
                decode_record(raw)

    def test_bounds_types_encoding_hashes_and_non_json_values(self):
        for raw in (
                b"", b" " * (MAX_RECORD_BYTES + 1), b"\xff", b"\xef\xbb\xbf{}",
                b"[]", b"null", b"{}", b'{"schema":NaN}', b'{"schema":Infinity}',
                self.wire.replace(b'"schema":1', b'"schema":true'),
                self.wire.replace(b'"schema":1', b'"schema":2'),
                self.wire.replace(b'"token_start":128', b'"token_start":128.0'),
                self.wire.replace(b'"entry_generation":7', b'"entry_generation":-1'),
                self.wire.replace(b'"entry_generation":7', b'"entry_generation":9223372036854775808'),
                self.wire.replace(b'"chain_hash":"', b'"chain_hash":"f'),
                self.wire.replace(b'"owner_name":"producer"', b'"owner_name":"\\ud800"'),
                b'{"nested":' + b"[" * 1100 + b"]" * 1100 + b"}",
                b'{"nested":[' + b"1," * 40 + b"1]}",
                b'{"nested":"' + b"x" * 257 + b'"}'):
            with self.subTest(raw=raw[:120]), self.assertRaises(ValueError):
                decode_record(raw)
        for invalid in ("{}", bytearray(self.wire), memoryview(self.wire)):
            with self.assertRaises(ValueError):
                decode_record(invalid)

    def test_missing_and_wrong_types_are_not_coerced(self):
        for field, value in (
                ("source", None), ("chain_hash", True), ("payload_sha256", None),
                ("token_end", "256"), ("token_start", False), ("schema", [1])):
            changed = self.record.to_dict()
            changed[field] = value
            raw = json.dumps(changed).encode()
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode_record(raw)


class IndependentProcessTests(unittest.TestCase):
    def test_original_single_owner_and_full_block_contracts_are_preserved(self):
        from pypto_serving.serving.native_ddr.types import (
            DdrEntryRef, DdrTransferRequest, Operation, PrefixKey, QwenKvLayout,
        )
        layout = QwenKvLayout(40, 31, 128, 8, 128)
        self.assertEqual(layout.block_bytes, BLOCK_BYTES)
        request = DdrTransferRequest(
            "host-contract-check", "owner-one", Operation.STORE,
            (PrefixKey(b"n" * 32, b"h" * 32),),
            (DdrEntryRef("owner-one", "entry", 1),), "lease", (9,), 1,
        )
        request.validate(layout)
        with self.assertRaisesRegex(ValueError, "epoch"):
            replace(request, worker_epoch="other-owner").validate(layout)
        with self.assertRaisesRegex(ValueError, "page"):
            replace(request, page_ids=(31,)).validate(layout)

    def test_real_producer_consumer_normal_import_without_runtime_or_pool(self):
        producer = subprocess.run(
            [sys.executable, "-m", "examples.run03.host_record_probe", "produce"],
            cwd=ROOT, capture_output=True, check=True, timeout=20,
        )
        self.assertEqual(producer.stdout, encode_record(fixture_record()))
        consumer = subprocess.run(
            [sys.executable, "-m", "examples.run03.host_record_probe", "consume"],
            cwd=ROOT, input=producer.stdout, capture_output=True, check=True, timeout=20,
        )
        output = json.loads(consumer.stdout)
        self.assertTrue(output["canonical_roundtrip"])
        self.assertEqual(output["data_id"], fixture_compatibility().data_id)
        self.assertEqual(output["runtime_modules_loaded"], [])
        self.assertFalse(output["real_e10_or_native_executed"])
        self.assertNotEqual(output["source_worker_epoch"], output["destination"]["worker_epoch"])
        tampered = producer.stdout.replace(b'"resource_generation":2', b'"resource_generation":9')
        rejected = subprocess.run(
            [sys.executable, "-m", "examples.run03.host_record_probe", "consume"],
            cwd=ROOT, input=tampered, capture_output=True, timeout=20,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(b"stale source", rejected.stderr)

    def test_fresh_package_import_works_with_runtime_imports_actively_blocked(self):
        code = """
import importlib.abc
import sys
class DenyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torch_npu','pypto','simpler','uflow','_task_interface'}:
            raise RuntimeError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, DenyRuntime())
from pypto_serving.serving.native_ddr.shared_compatibility import BLOCK_BYTES
from pypto_serving.serving.native_ddr.shared_prefix_record import decode_record
assert BLOCK_BYTES == 20971520
print('normal-package-import-with-denied-runtimes: pass')
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                                check=True, timeout=20)
        self.assertIn(b"pass", result.stdout)


if __name__ == "__main__":
    unittest.main()
