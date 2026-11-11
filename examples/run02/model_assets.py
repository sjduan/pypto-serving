# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read-only Qwen asset manifest; optionally expose the existing model mount at its contracted path."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_model(model):
    config = json.loads((model / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 40:
        raise ValueError("Expected the contracted Qwen3-14B configuration")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
    required.update(index["weight_map"].values())
    files = {}
    for name in sorted(required):
        path = model / name
        if not path.resolve().is_relative_to(model.resolve()):
            raise ValueError(f"Asset escapes model root: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    from safetensors import safe_open
    from transformers import AutoTokenizer
    weight_map = index["weight_map"]
    checked_tensors = 0
    for shard in sorted(set(weight_map.values())):
        with safe_open(model / shard, framework="pt", device="cpu") as handle:
            expected = {key for key, value in weight_map.items() if value == shard}
            if set(handle.keys()) != expected:
                raise ValueError(f"Safetensors index/header mismatch: {shard}")
            checked_tensors += len(expected)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tokens = tokenizer.encode("Environment identity audit.", add_special_tokens=False)
    if not tokens:
        raise ValueError("Local tokenizer probe returned no tokens")
    return {"model": str(model), "resolved": str(model.resolve()), "config": config, "files": files,
            "checked_tensor_headers": checked_tensors, "tokenizer_probe_ids": tokens,
            "scope": "read_only_files_headers_and_tokenizer_not_model_inference"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expose-existing-mount", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = Path("/data/models/Qwen/Qwen3-14B")
    source = Path("/models/Qwen/Qwen3-14B")
    if args.expose_existing_mount and not model.exists():
        if model.is_symlink():
            raise ValueError("Refusing to replace an existing dangling model link")
        if not (source / "config.json").is_file():
            raise ValueError("Existing read-only model input is missing")
        model.parent.mkdir(parents=True, exist_ok=True)
        model.symlink_to(source, target_is_directory=True)
    result = inspect_model(model)
    result["existing_mount_same_inode"] = (model / "config.json").samefile(source / "config.json")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"model": result["model"], "resolved": result["resolved"],
                      "files": len(result["files"]), "manifest": str(args.output)}))


if __name__ == "__main__":
    main()
