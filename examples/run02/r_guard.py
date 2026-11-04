# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded R parent/subreaper: this parent owns the advisory lock; the driver never reacquires it."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import stat
import subprocess
import sys
import time

from m2_guard import children, descendants, port_rows
from worker_probe import selected_chip_status


def snapshot(root, tag, device):
    text = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    (root / "logs" / f"r-{tag}-npu.txt").write_text(text)
    target = os.stat(f"/dev/davinci{device}").st_rdev
    users, denied = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            denied.append(int(proc.name))
            continue
        for fd in fds:
            try:
                info = fd.stat()
                if stat.S_ISCHR(info.st_mode) and info.st_rdev == target:
                    users.append({"pid": int(proc.name), "stat": (proc / "stat").read_text()})
                    break
            except (FileNotFoundError, ProcessLookupError):
                pass
            except PermissionError:
                denied.append(int(proc.name))
    return {"status": selected_chip_status(text, device), "npu": text,
            "fds": {"users": users, "denied": sorted(set(denied))}, "port18892": port_rows()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["smoke", "matrix"], required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--native-enabled", type=int, choices=[0, 1], required=True)
    parser.add_argument("--utilization", type=float, default=.80)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        raise ValueError("invalid attempt")
    if not 0 <= args.device < 16:
        raise ValueError("invalid physical device")
    root = Path(os.environ["UFLOW_RUN02_ROOT"])
    attempt = root / "attempts" / args.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "logs").mkdir()
    (attempt / "results").mkdir()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    report = {"case": args.case, "owner": os.getpid(), "passed": False,
              "scope": "parent owns advisory Run02 lock and reserved18892; no global allocator",
              "device_reset": False, "deadline_s": 1800, "signals": []}
    try:
        manifest_path = root / "configs/source-manifest.json"
        audit_path = root / "results/environment-audit.json"
        manifest = json.loads(manifest_path.read_text())
        audit = json.loads(audit_path.read_text())
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if not audit["passed"] or audit["source_manifest"]["sha256"] != manifest_sha:
            raise RuntimeError("source/environment identity does not match the current attempt")
        shutil.copy2(manifest_path, attempt / "results/source-manifest.json")
        shutil.copy2(audit_path, attempt / "results/environment-audit.json")
        report["serving_sha"] = manifest["serving"]
        report["source_manifest_sha256"] = manifest_sha
        expected_timeouts = {"SIMPLER_SCHEDULER_TIMEOUT_MS": "20000",
            "SIMPLER_OP_EXECUTE_TIMEOUT_US": "45000000", "SIMPLER_STREAM_SYNC_TIMEOUT_MS": "50000",
            "PYPTO_WORKER_INIT_TIMEOUT": "900"}
        report["native_timeouts"] = {k: os.environ.get(k) for k in expected_timeouts}
        if report["native_timeouts"] != expected_timeouts:
            raise RuntimeError("finite native/startup timeouts changed")
        lockpath = Path(f"/workspace/uflow/locks/run02-npu-{args.device}.lock")
        with lockpath.open("a+") as lock, socket.socket() as sock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report["before"] = snapshot(attempt, "before", args.device)
            report["physical_device"] = args.device
            report["lockpath"] = str(lockpath)
            before = report["before"]
            status = before["status"]
            headroom = 40000
            if (status["health"] != "OK" or status["hbm_total_mib"] - status["hbm_used_mib"] < headroom
                    or before["fds"]["users"] or before["fds"]["denied"]
                    or f"No running processes found in NPU {status['board']}" not in before["npu"]
                    or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")):
                raise RuntimeError("device occupied, unhealthy, remapped or unobservable")
            sock.bind(("127.0.0.1", 18892))
            sock.listen(128)
            output = attempt / "results/probe.json"
            command = [sys.executable, "-u", "examples/run02/r_probe.py",
                       "--case", args.case, "--output", str(output), "--device", str(args.device), "--native-enabled", str(args.native_enabled),
                       "--utilization", str(args.utilization), "--listen-fd", str(sock.fileno())]
            report["command"] = command
            logpath = attempt / "logs/probe.log"
            with logpath.open("w") as log:
                child = subprocess.Popen(command, cwd=os.environ["PYPTO_SERVING_HOME"],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(sock.fileno(),))
                report["child_pid"] = child.pid
                deadline = time.monotonic() + 1800
                maps, identities = {}, {}
                while child.poll() is None and time.monotonic() < deadline:
                    for pid in descendants(child.pid):
                        proc = Path(f"/proc/{pid}")
                        try:
                            identities[str(pid)] = {"stat": (proc / "stat").read_text(),
                                                     "cmdline": (proc / "cmdline").read_text().replace("\0", " ")}
                            selected = [l for l in (proc / "maps").read_text().splitlines()
                                        if any(k in l for k in ("libhost_runtime", "pypto_core", "_task_interface",
                                                               "libascendcl", "libruntime"))]
                            if selected:
                                maps[str(pid)] = selected
                        except (FileNotFoundError, ProcessLookupError):
                            pass
                    time.sleep(1)
                report["timeout"] = child.poll() is None
                if report["timeout"]:
                    child.send_signal(signal.SIGINT)
                    report["signals"].append({"pid": child.pid, "signal": "SIGINT"})
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        report["survived_grace"] = True
                report["exit_code"] = child.poll()
                if child.poll() is not None:
                    child.wait()
            report["loaded_maps"], report["owned_processes"] = maps, identities
            reaped = []
            limit = time.monotonic() + 30
            while True:
                try:
                    pid, status_code = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid:
                    reaped.append({"pid": pid, "status": status_code})
                elif time.monotonic() < limit:
                    time.sleep(.2)
                else:
                    break
            report["adopted_reaped"] = reaped
            report["remaining_adopted"] = children(os.getpid())
            text = logpath.read_text(errors="replace")
            paths = sorted(set(re.findall(r"JSON=(/tmp/[^\s]+)", text)))
            report["remaining_native_bootstrap"] = [p for p in paths if Path(p).exists()]
            report["task_sockets"] = [l for l in Path("/proc/net/unix").read_text().splitlines() if str(root) in l]
            result = json.loads(output.read_text()) if output.exists() else {}
            report["driver_passed"] = result.get("passed", False)
        report["after"] = snapshot(attempt, "after", args.device)
        after = report["after"]
        with lockpath.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 18892))
        report["lock_reacquired_and_released"] = True
        report["port_rebindable"] = True
        report["port_unchanged"] = before["port18892"] == after["port18892"]
        report["clean"] = (report["exit_code"] is not None and not report["remaining_adopted"]
                           and not after["fds"]["users"] and not after["fds"]["denied"]
                           and not report["task_sockets"] and not report["remaining_native_bootstrap"]
                           and report["port_unchanged"] and after["status"]["health"] == "OK")
        report["passed"] = (report["exit_code"] == 0 and report["driver_passed"]
                            and report["clean"] and not report["timeout"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (attempt / "results/external-receipt.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report.get(k) for k in ("passed", "exit_code", "timeout", "clean", "error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
