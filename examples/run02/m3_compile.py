# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Construct/codegen the M3 programs on the fixed toolchain, without a device context."""
import json
import os
from pathlib import Path

from pypto_serving.model.common.compiler.compiler import KernelCompiler
from pypto_serving.model.common.executor.utils import build_pypto_run_config
from pypto_serving.model.qwen.native_ddr_kv import compile_programs


if __name__ == "__main__":
    root = Path(os.environ["UFLOW_RUN02_ROOT"])
    compiler = KernelCompiler(run_config=build_pypto_run_config(
        platform="a2a3", device_ids=(0,), pypto_build_dir=str(root / "build/copy-codegen")))
    programs = compile_programs(compiler)
    report = {"constructed": [p.name for p in programs], "scope": "DSL/codegen only; no owner or NPU"}
    (root / "results/copy-codegen.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
