# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare Run02 without changing any shared source, install, or device state."""
import json
import os
from pathlib import Path
import subprocess
import sys

PINS = {
    "/workspace/pypto": "8007f1146a800d679301c5ea14bb644b863da6bf",
    "/workspace/pypto/runtime": "508becb93e182a3035a14e3f9226929dbe994346",
    "/workspace/pypto-serving": "5f548be23d7af120b20a40694601f5bb5b8b7a90",
    "/workspace/pypto-lib": "ec58b8ad7454b321f379260e95a0ae4431a5ae57",
}


def run(*args):
    print("+", *args, flush=True)
    subprocess.run(args, check=True)


def main():
    root = Path(os.environ["UFLOW_RUN02_ROOT"]).resolve()
    if not root.is_relative_to("/workspace/uflow"):
        raise ValueError("Run02 root must be under /workspace/uflow")
    for name in ("scripts", "src", "envs", "cache", "build", "tmp", "logs", "results", "configs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for repo, pin in PINS.items():
        actual = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
        if actual != pin:
            raise RuntimeError(f"{repo}: expected {pin}, got {actual}")
        if subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True):
            raise RuntimeError(f"Shared input is dirty: {repo}")
    serving = root / "src/serving"
    if not serving.exists():
        run("git", "clone", "--shared", "--no-checkout", "/workspace/pypto-serving", str(serving))
        run("git", "-C", str(serving), "checkout", "-b", "run02/t_ad4b1e29", PINS["/workspace/pypto-serving"])
    if not (serving / "pypto-lib/.git").exists():
        run("git", "clone", "--shared", "--no-checkout", "/workspace/pypto-lib", str(serving / "pypto-lib"))
        run("git", "-C", str(serving / "pypto-lib"), "checkout", "--detach", PINS["/workspace/pypto-lib"])
    for repo, pin in [(serving, PINS["/workspace/pypto-serving"]),
                      (serving / "pypto-lib", PINS["/workspace/pypto-lib"])]:
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
        if actual != pin or dirty:
            raise RuntimeError(f"Refusing to overwrite an existing changed Run02 source: {repo}")
    py = root / "envs/python/bin/python"
    if not py.exists():
        run("/workspace/pypto/.venv/bin/python", "-m", "venv", "--system-site-packages", str(py.parents[1]))
    site = root / "envs/python/lib/python3.11/site-packages"
    # Reuse build tools and the stamped Simpler binding, not a copied install.
    # PathFinder finds this venv's freshly built pypto package before editable finders.
    (site / "run02-shared-build-tools.pth").write_text(
        "/workspace/pypto/.venv/lib/python3.11/site-packages\n"
    )
    (root / "results/input-pins.json").write_text(json.dumps(PINS, indent=2) + "\n")
    if "--build" in sys.argv:
        wheels = root / "build/wheels"
        wheels.mkdir(exist_ok=True)
        run(str(py), "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-v",
            "--config-settings=build-dir=" + str(root / "build/pypto"),
            "--config-settings=cmake.build-type=Release", "--wheel-dir", str(wheels), "/workspace/pypto")
        wheel, = wheels.glob("pypto-*.whl")
        run(str(py), "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel))


if __name__ == "__main__":
    main()
