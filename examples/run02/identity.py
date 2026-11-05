# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail-closed fixed-source, installed-wheel, backend, and executable identity audit (no NPU init)."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import zipfile

from model_assets import sha256
from prepare import PINS


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def artifact(path):
    p = Path(path)
    return {"path": str(p), "resolved": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)}


def inspect_environment(root):
    repos = dict(PINS)
    repos.pop("/workspace/pypto-serving")
    repos.pop("/workspace/pypto-lib")
    repos[str(root / "src/serving")] = PINS["/workspace/pypto-serving"]
    repos[str(root / "src/serving/pypto-lib")] = PINS["/workspace/pypto-lib"]
    result = {"python": sys.executable, "version": sys.version, "host": platform.node(),
              "uname": list(platform.uname()), "root": str(root), "repos": {}, "modules": {}}
    for repo, pin in repos.items():
        head, status = git(repo, "rev-parse", "HEAD"), git(repo, "status", "--porcelain")
        if head != pin or status:
            raise ValueError(f"Wrong or dirty source input: {repo} {head} {status}")
        result["repos"][repo] = {"head": head, "status": status}
    gitlink = git(root / "src/serving", "ls-tree", "HEAD", "pypto-lib")
    if PINS["/workspace/pypto-lib"] not in gitlink:
        raise ValueError("serving submodule pin mismatch")
    result["lib_gitlink"] = gitlink
    for name in ["pypto", "pypto.pypto_core", "pypto.runtime", "simpler", "_task_interface",
                 "pypto_serving", "pypto_serving.model.qwen.npu_runner", "models.qwen3_14b.constants",
                 "simpler_setup", "torch", "transformers", "safetensors"]:
        m = importlib.import_module(name)
        result["modules"][name] = artifact(m.__file__)
    for module, expected in {
        "simpler": Path("/workspace/pypto/runtime/python/simpler"),
        "simpler_setup": Path("/workspace/pypto/runtime/simpler_setup"),
        "pypto_serving": root / "src/serving/pypto_serving",
        "pypto_serving.model.qwen.npu_runner": root / "src/serving/pypto_serving",
        "models.qwen3_14b.constants": root / "src/serving/pypto-lib",
    }.items():
        if not Path(result["modules"][module]["resolved"]).is_relative_to(expected):
            raise ValueError(f"Unexpected import route: {module}")
    pypto = importlib.import_module("pypto")
    package = Path(pypto.__file__).parent
    if not package.is_relative_to(root / "envs/python"):
        raise ValueError("PyPTO came from an old external install")
    checked = []
    for p in sorted(package.rglob("*.py")):
        rel = p.relative_to(package)
        source = Path("/workspace/pypto/python/pypto") / rel
        if p.read_bytes() != source.read_bytes():
            raise ValueError(f"Installed/source Python mismatch: {rel}")
        checked.append(str(rel))
    result["installed_pypto_source_equal"] = checked
    wheel, = (root / "build/wheels").glob("pypto-*.whl")
    result["wheel"] = artifact(wheel)
    core = Path(result["modules"]["pypto.pypto_core"]["path"])
    with zipfile.ZipFile(wheel) as z:
        if core.read_bytes() != z.read("pypto/" + core.name):
            raise ValueError("Loaded PyPTO core differs from built wheel")
    result["core_matches_built_wheel"] = True
    binding = importlib.import_module("_task_interface")
    if binding.__build_commit__ != PINS["/workspace/pypto/runtime"]:
        raise ValueError("Simpler binding stamp mismatch")
    result["simpler_binding_commit"] = binding.__build_commit__
    from simpler_setup.runtime_builder import RuntimeBuilder
    from simpler_setup.pto_isa import read_pto_isa_pin
    builder = RuntimeBuilder("a2a3")
    binaries = builder.get_binaries("host_build_graph", build=False)
    result["backend"] = {name: artifact(path) for name, path in vars(binaries).items() if path is not None}
    stamp = PINS["/workspace/pypto/runtime"] + ":pto-isa=" + read_pto_isa_pin()
    result["backend_build_stamps"] = {}
    for target, field in [("host", "host_path"), ("aicpu", "aicpu_path"), ("aicore", "aicore_path")]:
        cache = Path("/workspace/pypto/runtime/build/cache/a2a3/onboard/host_build_graph") / target
        actual = (cache / ".git_commit").read_text().strip()
        if actual != stamp:
            raise ValueError(f"Backend build stamp mismatch: {target}: {actual}")
        lib = Path(getattr(binaries, field))
        built = cache / lib.name
        if lib.read_bytes() != built.read_bytes():
            raise ValueError(f"Backend installed/cache artifact mismatch: {lib}")
        result["backend_build_stamps"][target] = {"stamp": actual, "built": artifact(built),
                                                 "cache": artifact(cache / "CMakeCache.txt")}
    ptoas = shutil.which("ptoas")
    version = subprocess.check_output([ptoas, "--version"], text=True).strip()
    if version != "ptoas 0.70":
        raise ValueError(f"Wrong PTOAS: {version}")
    result["ptoas"] = {**artifact(ptoas), "version": version}
    # Native CLI --version may exit(0) rather than raise Python SystemExit.
    # Resolve exactly the selected wrapper's package route, but run its CLI only in the subprocess above.
    wrapper = runpy.run_path(ptoas, run_name="run02_ptoas_wrapper_identity")
    wrapper["_add_configured_python_root"](Path(ptoas).resolve())
    wrapper["_disable_editable_import_redirects"]()
    result["ptoas"]["payloads"] = {
        name: artifact(importlib.import_module(name).__file__) for name in ["ptoas", "ptoas._cli", "ptoas._core"]
    }
    wrapper_root = Path(ptoas).resolve().parents[1]
    if not Path(result["ptoas"]["payloads"]["ptoas._core"]["resolved"]).is_relative_to(wrapper_root):
        raise ValueError("PTOAS native core escaped the selected wrapper tree")
    result["maps"] = [s for s in Path("/proc/self/maps").read_text().splitlines()
                      if any(k in s for k in ["pypto_core", "_task_interface", str(wrapper_root)])]
    result["environment"] = {k: os.environ[k] for k in sorted(os.environ)
                             if k.startswith(("UFLOW_", "PYPTO_", "SIMPLER_")) or k in
                             ["PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "TMPDIR", "HF_HOME", "XDG_CACHE_HOME"]}
    result["acceptance_level"] = "host_identity_and_build_only_no_model_or_transfer"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_environment(Path(os.environ["UFLOW_RUN02_ROOT"]).resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"host_identity": "pass", "python_files_equal": len(result["installed_pypto_source_equal"]),
                      "core": result["modules"]["pypto.pypto_core"], "ptoas": result["ptoas"]}))


if __name__ == "__main__":
    main()
