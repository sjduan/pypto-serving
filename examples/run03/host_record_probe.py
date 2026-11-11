# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Metadata-only two-process probe; synthetic deployment/backing, no real E10/NPU.

The twelve asset hashes below are a normalized, named historical M2 fixture:
artifacts/run02-m2-baseline-20261104/model-full-hashes.json, accepted by
artifacts/run02-m2-review-20261104/receipt.json (2026-11-04). No weights are read
or rehashed. This does NOT prove the model loaded by any current replica.
Deployment, source/target leases, prefix and payload values are HOST FIXTURES.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sys

from pypto_serving.serving.native_ddr.shared_compatibility import (
    BLOCK_BYTES, EVIDENCE_LEVEL, TrustedDeploymentAttestation,
    build_shared_compatibility, descriptor_digest, shared_layout,
)
from pypto_serving.serving.native_ddr.shared_prefix_record import (
    DestinationBinding, ExternalBackingIdentity, SharedPrefixRecord, SourceBinding,
    decode_record, encode_record, validate_candidate,
)

HISTORICAL_ASSETS = {
    "config.json": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a",
    "model.safetensors.index.json": "62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "model-00001-of-00008.safetensors": "e942bdbdf08857d16a8fef7d1dae9fceabeb4e84def6043485fe2f6f085dab0e",
    "model-00002-of-00008.safetensors": "f7c9c6eee628f5ad831d2d1d292e120505e5fcadeb38f88b4d3c4cb86306ccf9",
    "model-00003-of-00008.safetensors": "dfb8c5df9404b41ad6ae74e8b6b367135f017b4467b884cf71b17c71954f18a9",
    "model-00004-of-00008.safetensors": "eab286fec759e3e59ab228621aefa0fef14ed56039e06f959e67257d5af7604d",
    "model-00005-of-00008.safetensors": "97f0dc2992e59da95c466eff6f4fd0c8335843bbc36ed5c913a6f5150748c0e6",
    "model-00006-of-00008.safetensors": "9e8e76a013cd5e253865b792991e0b410f869b136b3c500079b531b09198e99e",
    "model-00007-of-00008.safetensors": "0aee70ee6e91dc00d818804fb47f124d13ee4ad5b4a64553e09dbf9391cd5750",
    "model-00008-of-00008.safetensors": "0d6b92296e326d39bbbaeb32c3ec454ac606da843d4c8ffa8edf010b62b8c9e0",
}


def fixture_descriptor():
    """Synthetic host descriptor with named historical full-content asset input."""
    return {
        "schema": 1, "model": "Qwen3-14B", "tensor_parallel": 1, "dtype": "bfloat16",
        "adapter": "none", "quantization": "bf16",
        "assets": [{"name": name, "sha256": digest} for name, digest in sorted(HISTORICAL_ASSETS.items())],
        "model_config_sha256": HISTORICAL_ASSETS["config.json"],
        "rope": {"theta": 1000000, "scaling": "none"},
        "position": {"scheme": "absolute-token-offset/loaded-config-RoPE", "max_position_embeddings": 40960},
        "tokenizer": {"implementation": "Qwen2TokenizerFast"},
        "layout": shared_layout(),
        "deployment": {name: hashlib.sha256(("host-fixture:" + name).encode()).hexdigest()
                       for name in ("serving", "qwen_lib", "pypto", "simpler", "ptoas",
                                    "pto_isa", "native_backend")},
    }


def fixture_compatibility():
    descriptor = fixture_descriptor()
    attestation = TrustedDeploymentAttestation(
        EVIDENCE_LEVEL, "HOST-FIXTURE/M2-historical-assets+synthetic-deployment",
        descriptor_digest(descriptor),
    )
    return build_shared_compatibility(descriptor, trusted_attestation=attestation)


def fixture_record():
    source = SourceBinding(
        ExternalBackingIdentity("host-fixture", "boot-fixture", "broker-fixture", "producer-incarnation",
                                "producer", 1, 2, 3, 4, 5, 6, BLOCK_BYTES * 2),
        "producer-worker-epoch", "entry-one", 7, BLOCK_BYTES,
    )
    return SharedPrefixRecord(
        fixture_compatibility().data_id, hashlib.sha256(b"serving-provided-chain/HOST-FIXTURE").digest(),
        128, 256, hashlib.sha256(b"metadata-only-payload/HOST-FIXTURE").hexdigest(), source,
    )


def fixture_destination():
    return DestinationBinding("consumer-worker-epoch", "consumer-arena", "consumer-cache", 31, 9, "page-lease")


def check_host_imports():
    forbidden = ("torch", "torch_npu", "pypto", "simpler", "uflow", "_task_interface")
    loaded = sorted(name for name in sys.modules
                    if any(name == root or name.startswith(root + ".") for root in forbidden))
    if loaded:
        raise RuntimeError("runtime modules unexpectedly loaded: " + repr(loaded))
    return loaded


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("produce", "consume"):
        raise SystemExit("usage: python -m examples.run03.host_record_probe produce|consume")
    expected = fixture_record()
    if sys.argv[1] == "produce":
        check_host_imports()
        sys.stdout.buffer.write(encode_record(expected))
        return
    raw = sys.stdin.buffer.read(65537)
    record = decode_record(raw)
    current_destination = fixture_destination()
    candidate = validate_candidate(
        record, expected_compatibility=fixture_compatibility(), expected_source=expected.source,
        destination=current_destination, expected_destination=current_destination,
        expected_chain_hash=expected.chain_hash, expected_token_start=expected.token_start,
        expected_payload_sha256=expected.payload_sha256,
    )
    if encode_record(record) != raw:
        raise ValueError("probe requires canonical producer encoding")
    print(json.dumps({
        "kind": "host-metadata-candidate-only", "data_id": candidate.record.compatibility_id,
        "source_worker_epoch": candidate.record.source.worker_epoch,
        "destination": asdict(candidate.destination), "canonical_roundtrip": True,
        "runtime_modules_loaded": check_host_imports(), "real_e10_or_native_executed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
