# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Portable host-only content identity; never a loaded-model or Ready proof.

The trusted attestation is supplied by a deployment authority out of band.
Validating its structure is not authentication of an arbitrary caller. Production
integration must bind actual loaded assets/configuration/binaries to this identity.
No file is opened, weight hashed, runtime imported or allocation made here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

MAX_RECORD_BYTES = 65536
MAX_INTEGER = (1 << 63) - 1
BLOCK_BYTES = 20971520
ASSET_NAMES = frozenset({
    "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    *(f"model-{index:05d}-of-00008.safetensors" for index in range(1, 9)),
})
DEPLOYMENT_COMPONENTS = frozenset({
    "serving", "qwen_lib", "pypto", "simpler", "ptoas", "pto_isa", "native_backend",
})
EVIDENCE_LEVEL = "trusted-full-assets-and-deployment-v1"


def shared_layout():
    """Return a fresh descriptor so callers cannot mutate schema constants."""
    return {
        "num_layers": 40, "page_size": 128, "num_kv_heads": 8, "head_dim": 128,
        "dtype": "bfloat16", "layout_version": "qwen3-lpthd-bf16-v1",
        "ddr_order": "block-layer-kv-token-head-dim-v1", "sides": ["K", "V"],
    }


def _keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("missing or unknown fields")
    return value


def _integer(value, minimum=0, maximum=MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer out of range")
    return value


def _text(value):
    if (type(value) is not str or not 1 <= len(value) <= 256
            or not value.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in value)
            or value.lower() in ("unknown", "none", "null")):
        raise ValueError("invalid or unknown identity string")
    return value


def _sha256(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA256")
    return value


def _digest_bytes(value):
    if type(value) is not bytes or len(value) != 32:
        raise ValueError("expected a complete 32-byte chained hash")
    return value


def _bounded_tree(value, depth=0, budget=None):
    if budget is None:
        budget = [1024]
    budget[0] -= 1
    if depth > 12 or budget[0] < 0:
        raise ValueError("metadata tree exceeds bounds")
    if type(value) is dict:
        if len(value) > 32:
            raise ValueError("too many fields")
        for key, item in value.items():
            if type(key) is not str or len(key) > 128:
                raise ValueError("invalid field name")
            _bounded_tree(item, depth + 1, budget)
    elif type(value) is list:
        if len(value) > 32:
            raise ValueError("too many entries")
        for item in value:
            _bounded_tree(item, depth + 1, budget)
    elif type(value) is str:
        if len(value) > 256 or any(0xD800 <= ord(c) <= 0xDFFF for c in value):
            raise ValueError("invalid string")
    elif type(value) is int:
        _integer(value)
    elif value is not None:
        # No floats/NaN/infinity, boolean-as-integer, tuples or custom objects.
        raise ValueError("unsupported metadata value")


def canonical_json(value):
    """Bound and encode the small, integer-only protocol tree."""
    _bounded_tree(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("metadata exceeds 64KiB")
    return encoded


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def strict_json(data):
    """Decode bounded UTF-8 JSON without duplicate keys or lossy conversions."""
    if type(data) is not bytes or not 0 < len(data) <= MAX_RECORD_BYTES:
        raise ValueError("expected bounded JSON bytes")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
        _bounded_tree(value)
        return value
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError("invalid bounded UTF-8 JSON") from error


def descriptor_digest(descriptor):
    return hashlib.sha256(_validate_descriptor(descriptor)).hexdigest()


def _validate_descriptor(value):
    _keys(value, {
        "schema", "model", "tensor_parallel", "dtype", "adapter", "quantization",
        "assets", "model_config_sha256", "rope", "position", "tokenizer", "layout", "deployment",
    })
    _integer(value["schema"], 1, 1)
    _integer(value["tensor_parallel"], 1, 1)
    if (value["model"], value["dtype"], value["adapter"], value["quantization"]) != (
            "Qwen3-14B", "bfloat16", "none", "bf16"):
        raise ValueError("unsupported model, dtype, adapter or quantization")
    layout = value["layout"]
    expected_layout = shared_layout()
    _keys(layout, expected_layout)
    for name in ("num_layers", "page_size", "num_kv_heads", "head_dim"):
        _integer(layout[name], expected_layout[name], expected_layout[name])
    if layout != expected_layout:
        raise ValueError("unsupported shared KV layout")
    assets = value["assets"]
    if type(assets) is not list or len(assets) != len(ASSET_NAMES):
        raise ValueError("complete twelve-file content manifest required")
    seen = set()
    for asset in assets:
        if type(asset) is not dict:
            raise ValueError("invalid asset manifest entry")
        expected = {"name", "sha256", "size_bytes"} if "size_bytes" in asset else {"name", "sha256"}
        _keys(asset, expected)
        if type(asset["name"]) is not str or asset["name"] not in ASSET_NAMES or asset["name"] in seen:
            raise ValueError("unexpected or duplicate asset")
        seen.add(asset["name"])
        _sha256(asset["sha256"])
        if "size_bytes" in asset:
            _integer(asset["size_bytes"], 1)
    if seen != ASSET_NAMES:
        raise ValueError("incomplete content manifest")
    config_hash = next(asset["sha256"] for asset in assets if asset["name"] == "config.json")
    if _sha256(value["model_config_sha256"]) != config_hash:
        raise ValueError("model configuration does not match trusted asset manifest")
    _keys(value["rope"], {"theta", "scaling"})
    _integer(value["rope"]["theta"], 1)
    if value["rope"]["scaling"] != "none":
        raise ValueError("only unscaled RoPE is supported")
    _keys(value["position"], {"scheme", "max_position_embeddings"})
    if value["position"]["scheme"] != "absolute-token-offset/loaded-config-RoPE":
        raise ValueError("unsupported position semantics")
    _integer(value["position"]["max_position_embeddings"], 128)
    _keys(value["tokenizer"], {"implementation"})
    _text(value["tokenizer"]["implementation"])
    _keys(value["deployment"], DEPLOYMENT_COMPONENTS)
    for digest in value["deployment"].values():
        _sha256(digest)
    # Asset order is not content identity; normalize it before hashing.
    normalized = strict_json(canonical_json(value))
    normalized["assets"].sort(key=lambda asset: asset["name"])
    return canonical_json(normalized)


@dataclass(frozen=True)
class TrustedDeploymentAttestation:
    """Out-of-band authority input, not a signature or a fresh loaded-model proof.

    descriptor_sha256 pins the complete configuration, full twelve-file hashes
    and actual deployment component identities. References are audit text only.
    H1 does not authenticate the caller supplying this object.
    """
    evidence_level: str
    evidence_reference: str
    descriptor_sha256: str

    def __post_init__(self):
        if self.evidence_level != EVIDENCE_LEVEL:
            raise ValueError("full content/deployment evidence required")
        _text(self.evidence_reference)
        _sha256(self.descriptor_sha256)


@dataclass(frozen=True)
class SharedCompatibility:
    """Immutable, structurally validated metadata; no runtime readiness state."""
    descriptor_json: bytes
    trusted_attestation: TrustedDeploymentAttestation

    def __post_init__(self):
        canonical = _validate_descriptor(strict_json(self.descriptor_json))
        if type(self.trusted_attestation) is not TrustedDeploymentAttestation:
            raise ValueError("explicit trusted deployment attestation required")
        if hashlib.sha256(canonical).hexdigest() != self.trusted_attestation.descriptor_sha256:
            raise ValueError("descriptor differs from trusted deployment attestation")
        object.__setattr__(self, "descriptor_json", canonical)

    @property
    def data_id(self):
        return hashlib.sha256(self.descriptor_json).hexdigest()

    def to_dict(self):
        return strict_json(self.descriptor_json)

    @property
    def max_position_embeddings(self):
        return self.to_dict()["position"]["max_position_embeddings"]


def build_shared_compatibility(descriptor, *, trusted_attestation):
    """Require explicit trusted context. A wire digest alone is insufficient.

    Supplying fabricated attestations can only produce host metadata, never
    authenticated E10 permissions, a model Ready transition or a transfer.
    """
    if type(trusted_attestation) is not TrustedDeploymentAttestation:
        raise ValueError("explicit trusted deployment attestation required")
    canonical = _validate_descriptor(descriptor)
    if hashlib.sha256(canonical).hexdigest() != trusted_attestation.descriptor_sha256:
        raise ValueError("descriptor differs from trusted deployment attestation")
    return SharedCompatibility(canonical, trusted_attestation)
