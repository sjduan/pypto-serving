# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read task-owned process/FD state; optionally interrupt only a validated baseline driver."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess

from m2_baseline import descendants, unix_sockets
from worker_probe import device_users, selected_chip_status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=int, required=True)
    parser.add_argument("--interrupt", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["UFLOW_RUN02_ROOT"])
    result = {"driver": args.driver, "processes": {}, "interrupt_sent": False}
    for pid in [args.driver, *descendants(args.driver)]:
        proc = Path(f"/proc/{pid}")
        try:
            result["processes"][str(pid)] = {
                "cmdline": (proc / "cmdline").read_text().replace("\0", " "),
                "status": (proc / "status").read_text(),
                "wchan": (proc / "wchan").read_text(),
                "fds": [os.readlink(fd) for fd in (proc / "fd").iterdir()],
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
    if args.interrupt:
        driver = result["processes"].get(str(args.driver), {})
        if "python scripts/examples/run02/m2_baseline.py --device 0 --port 18892" not in driver.get("cmdline", ""):
            raise RuntimeError("Refusing to signal a process not matching the owned baseline driver")
        if Path(f"/proc/{args.driver}/cwd").resolve() != root:
            raise RuntimeError("Refusing to signal a driver outside this task root")
        os.kill(args.driver, signal.SIGINT)
        result["interrupt_sent"] = True
    snapshot = subprocess.check_output(["npu-smi", "info"], text=True)
    result["device_status"] = selected_chip_status(snapshot, 0)
    result["device_users"] = device_users(0)
    result["task_sockets"] = unix_sockets(root)
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 18892))
            result["port_free"] = True
        except OSError:
            result["port_free"] = False
    with Path("/workspace/uflow/locks/run02-npu-0.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result["advisory_lock_free"] = True
        except BlockingIOError:
            result["advisory_lock_free"] = False
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "processes"}))


if __name__ == "__main__":
    main()
