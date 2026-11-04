# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Audit the reviewed M2 sources and the existing fixed A toolchain without NPU initialization."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from identity import artifact, git
from model_assets import sha256


def main():
    root = Path(os.environ["UFLOW_RUN02_ROOT"])
    serving = Path(os.environ["PYPTO_SERVING_HOME"])
    manifest = json.loads((root / "configs/source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(serving / name) != expected:
            raise ValueError(f"Uploaded source mismatch: {name}")
    pins = {"/workspace/pypto": "8007f1146a800d679301c5ea14bb644b863da6bf",
            "/workspace/pypto/runtime": "508becb93e182a3035a14e3f9226929dbe994346"}
    result = {"source_manifest": artifact(root / "configs/source-manifest.json"),
              "repos": {}, "modules": {}, "native_backend": "tensormap_and_ringbuffer"}
    for repo, pin in pins.items():
        head = git(repo, "rev-parse", "HEAD")
        status = git(repo, "status", "--porcelain", "--untracked-files=no")
        if head != pin or status:
            raise ValueError(f"Frozen source mismatch: {repo} {head} {status}")
        result["repos"][repo] = {"head": head, "tracked_status": status}
    routes = {
        "pypto_serving": serving,
        "pypto_serving.model.qwen.npu_runner": serving,
        "models.qwen3_14b.constants": serving / "pypto-lib",
        "simpler": Path("/workspace/pypto/runtime/python/simpler"),
        "simpler_setup": Path("/workspace/pypto/runtime/simpler_setup"),
    }
    for name in [*routes, "pypto", "pypto.pypto_core", "pypto.runtime", "_task_interface",
                 "torch", "transformers", "safetensors", "fastapi", "httpx", "msgspec", "uvicorn"]:
        module = importlib.import_module(name)
        result["modules"][name] = artifact(module.__file__)
        if name in routes and not Path(module.__file__).resolve().is_relative_to(routes[name]):
            raise ValueError(f"Wrong import route: {name}")
    core = result["modules"]["pypto.pypto_core"]
    if core["sha256"] != "9d7c187e37d9d20a898c3e4342f4d4fd31cd805aed18427059d0a2b9b68b5169":
        raise ValueError("PyPTO native core differs from the preserved M1 source build")
    package = Path(result["modules"]["pypto"]["resolved"]).parent
    checked = []
    for path in package.rglob("*.py"):
        rel = path.relative_to(package)
        source = Path("/workspace/pypto/python/pypto") / rel
        if path.read_bytes() != source.read_bytes():
            raise ValueError(f"PyPTO installed/source mismatch: {rel}")
        checked.append(str(rel))
    result["pypto_python_source_equal"] = checked
    binding = importlib.import_module("_task_interface")
    if binding.__build_commit__ != pins["/workspace/pypto/runtime"]:
        raise ValueError("Native binding stamp mismatch")
    from simpler_setup.pto_isa import read_pto_isa_pin
    from simpler_setup.runtime_builder import RuntimeBuilder
    isa = read_pto_isa_pin()
    if isa != "217bb98952a8f9004b95fe938da97d8d5f68f702":
        raise ValueError("PTO ISA pin changed")
    binaries = RuntimeBuilder("a2a3").get_binaries(result["native_backend"], build=False)
    stamp = binding.__build_commit__ + ":pto-isa=" + isa
    result["backend_inputs"] = {}
    for target, field in (("host", "host_path"), ("aicpu", "aicpu_path"), ("aicore", "aicore_path")):
        path = Path(getattr(binaries, field))
        cache = Path("/workspace/pypto/runtime/build/cache/a2a3/onboard") / result["native_backend"] / target
        actual = (cache / ".git_commit").read_text().strip()
        if actual != stamp or path.read_bytes() != (cache / path.name).read_bytes():
            raise ValueError(f"Native backend stamp/artifact mismatch: {target}")
        result["backend_inputs"][target] = {"stamp": actual, "artifact": artifact(path)}
    ptoas = shutil.which("ptoas")
    version = subprocess.check_output([ptoas, "--version"], text=True).strip()
    if version != "ptoas 0.70":
        raise ValueError(f"Wrong PTOAS: {version}")
    result["ptoas"] = {"version": version, **artifact(ptoas)}
    result["environment"] = {k: v for k, v in os.environ.items()
                             if k.startswith(("PYPTO", "SIMPLER", "ASCEND", "UFLOW", "HF_"))
                             or k in ("PYTHONPATH", "LD_LIBRARY_PATH", "PATH", "TMPDIR")}
    result["scope"] = "host identity only; no model inference or NPU initialization"
    result["passed"] = True
    (root / "results/environment-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"environment_audit": "passed", "source_files": len(manifest["files"]),
                      "native_backend": result["native_backend"], "pypto_python_files": len(checked)}))


if __name__ == "__main__":
    main()
