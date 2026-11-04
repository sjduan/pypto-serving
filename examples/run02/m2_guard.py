# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded external owner supervision for M2; only the baseline driver holds the device lock."""
import ctypes
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time

from worker_probe import selected_chip_status


def device_fds():
    target = os.stat("/dev/davinci0").st_rdev
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
    return {"users": users, "denied": sorted(set(denied))}


def children(pid):
    try:
        return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def descendants(pid):
    result = [pid]
    for current in result:
        result.extend(p for p in children(current) if p not in result)
    return result


def port_rows():
    return {name: [line for line in Path("/proc/net/" + name).read_text().splitlines()[1:]
                   if line.split()[1].split(":")[-1] == f"{18892:04X}"]
            for name in ("tcp", "tcp6")}


def check_free_lock():
    with Path("/workspace/uflow/locks/run02-npu-0.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return True


def check_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 18892))
    return True


def snapshot(root, tag):
    text = subprocess.check_output(["npu-smi", "info"], text=True, timeout=15)
    (root / "logs" / f"m2-{tag}-npu.txt").write_text(text)
    return {"status": selected_chip_status(text, 0), "npu": text, "fds": device_fds(),
            "port18892": port_rows()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="baseline")
    parser.add_argument("--driver", choices=("m2_baseline.py", "m2_precision_probe.py"), default="m2_baseline.py")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        raise ValueError("Invalid attempt name")
    root = Path(os.environ["UFLOW_RUN02_ROOT"]).resolve()
    attempt_root = root / "attempts" / args.attempt
    attempt_root.mkdir(parents=True, exist_ok=False)
    (attempt_root / "results").mkdir()
    (attempt_root / "logs").mkdir()
    report = {"owner": os.getpid(), "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "external owned-process supervision; M2 driver owns the device lock and port",
              "deadline_s": 1800, "reset_called": False, "signals": [], "passed": False}
    receipt = attempt_root / "results/m2-external-receipt.json"
    try:
        if not root.is_relative_to("/workspace/uflow"):
            raise ValueError("Invalid task root")
        if not json.loads((root / "results/environment-audit.json").read_text())["passed"]:
            raise ValueError("Environment audit has not passed")
        if not json.loads((root / "results/model-comparison.json").read_text())["passed"]:
            raise ValueError("Model comparison has not passed")
        expected = {"SIMPLER_SCHEDULER_TIMEOUT_MS": "20000", "SIMPLER_OP_EXECUTE_TIMEOUT_US": "45000000",
                    "SIMPLER_STREAM_SYNC_TIMEOUT_MS": "50000", "PYPTO_WORKER_INIT_TIMEOUT": "900"}
        report["native_timeout_environment"] = {key: os.environ.get(key) for key in expected}
        if report["native_timeout_environment"] != expected:
            raise ValueError("Finite native/startup timeouts changed")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
        report["lock_preflight"] = check_free_lock()
        report["port_preflight"] = check_free_port()
        before = snapshot(attempt_root, "before")
        report["before"] = before
        status = before["status"]
        if (status["health"] != "OK" or status["hbm_total_mib"] - status["hbm_used_mib"] < 40000
                or before["fds"]["users"] or before["fds"]["denied"]
                or f"No running processes found in NPU {status['board']}" not in before["npu"]
                or os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("ASCEND_VISIBLE_DEVICES")):
            raise RuntimeError("Selected device is occupied, remapped, unhealthy, or not fully observable")
        output = attempt_root / "results/m2-baseline.json"
        if output.exists():
            raise ValueError("Refusing to overwrite an earlier M2 result")
        command = [sys.executable, "-u", "examples/run02/" + args.driver,
                   "--device", "0", "--port", "18892", "--output", str(output)]
        report["command"] = command
        owned, maps = {}, {}
        logpath = attempt_root / "logs/m2-baseline.log"
        with logpath.open("w") as log:
            child = subprocess.Popen(command, cwd=os.environ["PYPTO_SERVING_HOME"],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, UFLOW_M2_ATTEMPT_ROOT=str(attempt_root)))
            report["driver_pid"] = child.pid
            deadline = time.monotonic() + 1800
            while child.poll() is None and time.monotonic() < deadline:
                for pid in descendants(child.pid):
                    try:
                        proc = Path(f"/proc/{pid}")
                        owned[str(pid)] = {"stat": (proc / "stat").read_text(),
                                           "cmdline": (proc / "cmdline").read_text().replace("\0", " ")}
                        selected = [line for line in (proc / "maps").read_text().splitlines()
                                    if any(k in line for k in ("libhost_runtime", "_task_interface",
                                                               "pypto_core", "libascendcl", "libruntime"))]
                        if selected:
                            maps[str(pid)] = selected
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                time.sleep(1)
            report["timeout"] = child.poll() is None
            if report["timeout"]:
                # Signal only this new session's driver; its engine owns worker shutdown.
                child.send_signal(signal.SIGINT)
                report["signals"].append({"pid": child.pid, "signal": "SIGINT"})
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    report["driver_survived_grace"] = True
            report["exit_code"] = child.poll()
            if child.poll() is not None:
                child.wait()
        report["owned_processes_observed"] = owned
        report["loaded_maps"] = maps
        reaped = []
        until = time.monotonic() + 30
        while True:
            try:
                pid, status_code = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid:
                reaped.append({"pid": pid, "wait_status": status_code})
            elif time.monotonic() < until:
                time.sleep(0.2)
            else:
                break
        report["adopted_reaped"] = reaped
        report["remaining_adopted"] = children(os.getpid())
        after = snapshot(attempt_root, "after")
        report["after"] = after
        report["task_sockets"] = [line for line in Path("/proc/net/unix").read_text().splitlines()
                                  if str(root) in line]
        report["port_unchanged"] = before["port18892"] == after["port18892"]
        report["port_rebindable"] = check_free_port()
        report["lock_reacquired_and_released"] = check_free_lock()
        text = logpath.read_text(errors="replace")
        paths = sorted(set(re.findall(r"JSON=(/tmp/[^\s]+)", text)))
        report["native_bootstrap_paths"] = [{"path": p, "exists_after": Path(p).exists()} for p in paths]
        report["remaining_native_bootstrap_paths"] = [p for p in paths if Path(p).exists()]
        result = json.loads(output.read_text()) if output.exists() else {}
        report["baseline_passed"] = result.get("passed", False)
        report["engine_stop_ok"] = result.get("engine_stop_ok", False)
        report["clean"] = (report["exit_code"] is not None and not report["remaining_adopted"]
                           and not after["fds"]["users"] and not after["fds"]["denied"]
                           and not report["task_sockets"] and report["port_unchanged"]
                           and not report["remaining_native_bootstrap_paths"] and after["status"]["health"] == "OK")
        report["passed"] = (report["exit_code"] == 0 and not report["timeout"] and report["baseline_passed"]
                            and report["engine_stop_ok"] and report["clean"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report.get(k) for k in ("passed", "exit_code", "timeout", "clean", "error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
