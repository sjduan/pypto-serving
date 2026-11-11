# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-chip Simpler startup/load/close smoke, not a model or HBM transfer experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

from identity import artifact


def selected_chip_status(snapshot, device):
    lines = snapshot.splitlines()
    for index, line in enumerate(lines):
        match = re.fullmatch(
            r"\|\s*(\d+)\s+(\d+)\s*\|\s*[\dA-Fa-f:]+\.\d+\s*\|.*?\s(\d+)\s*/\s*(\d+)\s*\|", line
        )
        if match and int(match[2]) == device:
            prior = lines[index - 1].split("|")
            return {"board": int(prior[1].split()[0]), "chip": int(match[1]),
                    "physical_id": device, "health": prior[2].strip(),
                    "hbm_used_mib": int(match[3]), "hbm_total_mib": int(match[4])}
    raise ValueError("Cannot resolve selected physical chip from this npu-smi format")


def device_users(device):
    target = f"/dev/davinci{device}"
    users = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.readlink(fd) == target:
                        users.append(int(proc.name))
                        break
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(users)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        raise ValueError("A physical chip id is required")
    root = Path(os.environ["UFLOW_RUN02_ROOT"])
    locks = Path("/workspace/uflow/locks")
    locks.mkdir(exist_ok=True)
    # Advisory only: also inspect actual device users and NPU health, do not claim a global reservation.
    with (locks / f"run02-npu-{args.device}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = subprocess.check_output(["npu-smi", "info"], text=True)
        (root / "logs/npu-at-worker-start.txt").write_text(before)
        status = selected_chip_status(before, args.device)
        if status["health"] != "OK" or status["hbm_total_mib"] - status["hbm_used_mib"] < 256:
            raise RuntimeError(f"Selected chip is unhealthy or lacks startup memory: {status}")
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            raise RuntimeError("Resolve the physical/visible device mapping before probing")
        if f"No running processes found in NPU {status['board']}" not in before:
            raise RuntimeError("Selected NPU has running processes; choose another healthy idle chip")
        users = device_users(args.device)
        if users:
            raise RuntimeError(f"Selected chip has existing device FDs: {users}")
        import pypto.runtime
        from simpler import Worker
        from simpler_setup.runtime_builder import RuntimeBuilder
        binaries = RuntimeBuilder("a2a3").get_binaries("host_build_graph", build=False)
        worker = Worker(3, platform="a2a3", runtime="host_build_graph", device_ids=[args.device],
                        num_sub_workers=0, startup_timeout_s=60)
        result = {"device": args.device, "device_status": status, "pid": os.getpid(), "device_users_before": users,
                  "pypto_runtime": artifact(pypto.runtime.__file__),
                  "reservation": str(locks / f"run02-npu-{args.device}.lock"),
                  "reservation_scope": "advisory_run02_only_plus_live_device_checks",
                  "native_inputs": {k: artifact(v) for k, v in vars(binaries).items() if v is not None}}
        children = []
        result["task_sockets_before"] = [s for s in Path("/proc/net/unix").read_text().splitlines()
                                         if str(root) in s]
        try:
            worker.init()
            children = list(worker._chip_pids) + list(worker._sub_pids)
            result["children"] = children
            result["maps"] = {}
            for pid in [os.getpid(), *children]:
                maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
                result["maps"][str(pid)] = [s for s in maps if any(k in s for k in
                    ["libhost_runtime", "_task_interface", "pypto_core", "libascendcl", "libruntime"]) ]
            if not any(str(binaries.host_path) in s for p in children for s in result["maps"][str(p)]):
                raise RuntimeError("Chip child did not map the expected host runtime")
            result["initialized"] = True
        finally:
            worker.close()
            result["children_remaining"] = [p for p in children if Path(f"/proc/{p}").exists()]
            result["teardown_reports"] = str(worker._teardown_reports)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        if result["children_remaining"]:
            raise RuntimeError("Native children still live after close")
        after = subprocess.check_output(["npu-smi", "info"], text=True)
        (root / "logs/npu-after-worker.txt").write_text(after)
        result["device_status_after"] = selected_chip_status(after, args.device)
        result["device_users_after"] = device_users(args.device)
        result["task_sockets_after"] = [s for s in Path("/proc/net/unix").read_text().splitlines()
                                        if str(root) in s]
        result["service_port"] = None
        result["acceptance_level"] = "native_backend_startup_only_no_model_no_copy"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"worker_init_close": "pass", "device": args.device, "children": children,
                          "children_remaining": result["children_remaining"],
                          "scope": "native_backend_startup_only_no_model_no_copy"}))


if __name__ == "__main__":
    main()
