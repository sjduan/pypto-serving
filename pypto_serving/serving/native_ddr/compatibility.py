# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Incarnation-local compatibility from actual loaded assets and deployment files.

Weight payloads are not rescanned. Header content and file metadata are explicit
evidence levels, never advertised as fresh full-weight hashes or cross-restart identity.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import struct


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            result.update(part)
    return result.hexdigest()


def capture_assets(model_dir):
    root = Path(model_dir).resolve(strict=True)
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())
    shards = sorted(set(index["weight_map"].values()))
    names = set(shards)
    names.update(p.name for p in root.iterdir()
                 if p.is_file() and (p.suffix in (".json", ".model", ".txt")
                                    or p.name.startswith(("tokenizer", "vocab", "merges"))))
    files = {}
    for name in sorted(names):
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("model asset escapes its model directory")
        stat = path.stat()
        row = {"bytes": stat.st_size, "device": stat.st_dev, "inode": stat.st_ino,
               "mtime_ns": stat.st_mtime_ns}
        if name in shards:
            with path.open("rb") as stream:
                length_raw = stream.read(8)
                if len(length_raw) != 8:
                    raise ValueError("truncated safetensors header")
                length = struct.unpack("<Q", length_raw)[0]
                if not 0 < length <= min(stat.st_size - 8, 16 << 20):
                    raise ValueError("invalid safetensors header length")
                header = stream.read(length)
                if len(header) != length:
                    raise ValueError("truncated safetensors header")
            row["header_sha256"] = hashlib.sha256(header).hexdigest()
            row["evidence"] = "header-content+file-stat; payload not rescanned"
        else:
            row["sha256"] = digest_file(path)
            row["evidence"] = "full-small-asset-content"
        files[name] = row
    return {"resolved": str(root), "files": files}


def build_compatibility(loaded, assets, worker_epoch, layout):
    # Hash the actual imported sources/binaries, independently of Git availability
    # in an installed deployment. No management or previous experiment path is used.
    import pypto_serving
    from pypto_serving.model.qwen.npu_executor import _find_pypto_lib_qwen14b_dir

    source_roots = {"serving": Path(pypto_serving.__file__).resolve().parent,
                    "qwen_lib": _find_pypto_lib_qwen14b_dir()}
    components = {}
    for name, root in source_roots.items():
        entries = {p.relative_to(root).as_posix(): digest_file(p)
                   for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
        components[name] = {"source_manifest_sha256": hashlib.sha256(canonical_json(entries)).hexdigest(),
                            "source_files": len(entries)}
    for name in ("pypto", "pypto.pypto_core", "pypto.runtime", "simpler", "_task_interface"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        components[name] = {"sha256": digest_file(path),
                            "build_commit": getattr(module, "__build_commit__", None)}
    from simpler_setup.runtime_builder import RuntimeBuilder
    binaries = RuntimeBuilder("a2a3").get_binaries("tensormap_and_ringbuffer", build=False)
    components["native_backend"] = {name: digest_file(getattr(binaries, name))
                                     for name in ("host_path", "aicpu_path", "aicore_path")}
    import shutil
    import subprocess
    ptoas = shutil.which("ptoas")
    components["ptoas"] = {"sha256": digest_file(ptoas),
                           "version": subprocess.check_output([ptoas, "--version"], text=True, timeout=10).strip()}
    descriptor = {"schema": 1, "scope": "single-loaded-owner-incarnation-only",
                  "worker_epoch": worker_epoch, "model": asdict(loaded.config),
                  "assets": assets, "tokenizer_type": type(loaded.tokenizer).__qualname__,
                  "adapter": None, "position": "absolute-token-offset/loaded-config-RoPE",
                  "layout": asdict(layout), "tensor_parallel": 1, "deployment": components}
    encoded = canonical_json(descriptor)
    if len(encoded) > 60000:
        raise ValueError("compatibility descriptor exceeds bounded startup metadata")
    return hashlib.sha256(encoded).digest(), encoded
