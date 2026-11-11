# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Apply an exact source overlay to a clean, explicitly pinned checkout.

No downloads, builds, commits or devices are started by this helper. Submodule
overlays must be prepared separately using their own locked source entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def locked_source(lock: Path, name: str) -> tuple[dict, Path]:
    document = json.loads(lock.read_bytes())
    if document["schema"] != 1:
        raise ValueError("unsupported source lock")
    entry = document["sources"][name]
    patch = lock.parent / entry["patch"]
    if not patch.resolve().is_relative_to(lock.parent.resolve()):
        raise ValueError("patch escapes source lock directory")
    if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["patch_sha256"]:
        raise ValueError("source patch checksum mismatch")
    for name in entry["files"]:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts or "\\" in name:
            raise ValueError("unsafe overlay path")
    return entry, patch


def verify_files(root: Path, entry: dict) -> bool:
    for name, expected in entry["files"].items():
        path = root / name
        if expected is None:
            if path.exists() or path.is_symlink():
                return False
        elif expected["mode"] == "160000":
            # This is a recorded gitlink, not a network fetch instruction.
            continue
        elif path.is_symlink() or not path.is_file():
            return False
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            return False
    return True


def apply(lock: Path, name: str, checkout: Path, check: bool = False) -> str:
    entry, patch = locked_source(lock.resolve(), name)
    root = checkout.resolve(strict=True)
    head = git(root, "rev-parse", "HEAD")
    if head not in (entry["base"], entry["accepted_source_commit"]):
        raise ValueError("checkout HEAD does not match the locked base/source")
    changed = set(git(root, "diff", "--name-only", "--ignore-submodules=all", "HEAD").splitlines())
    new = set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    expected_paths = set(entry["files"])
    if (changed | new) - expected_paths:
        raise ValueError("unrelated checkout changes; overlay refused")
    tree_matches = git(root, "write-tree") == entry["accepted_tree"]
    unstaged = git(root, "diff", "--name-only", "--ignore-submodules=all")
    if verify_files(root, entry) and tree_matches and not unstaged:
        return "verified"
    if check:
        raise ValueError("locked overlay is missing or modified")
    if head != entry["base"] or changed or new:
        raise ValueError("overlay requires a clean locked base; existing changes preserved")
    git(root, "apply", "--index", "--check", str(patch))
    git(root, "apply", "--index", str(patch))
    if not verify_files(root, entry) or git(root, "write-tree") != entry["accepted_tree"]:
        raise ValueError("applied source differs from accepted file hashes")
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "dependencies/locked-lib.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(apply(args.lock, args.name, args.checkout, args.check))


if __name__ == "__main__":
    main()
