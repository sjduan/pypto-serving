# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reject the known fixed-width base before loading its Qwen kernels."""
import hashlib
from pathlib import Path

_UNPATCHED = {
    "paged_attention_pypto.py": "4087a49cd0b6c6396d4fef859576a02f8c6b096e9b94f5b55560e1f4b238e843",
    "prefill_fwd.py": "4423da03298fb4ab8be67f105d99f5b1d7da8a2cb1459a52cc837a2b9a635563",
}


def reject_unpatched_uflow_base(kernel_directory: Path) -> None:
    for name, known_base in _UNPATCHED.items():
        path = kernel_directory / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == known_base:
            raise RuntimeError(
                "The UFlow Qwen baseline requires its runtime-worker patch. "
                "Run python tools/prepare_uflow_lib.py pypto-lib <pypto-lib-checkout> "
                "from the serving checkout before loading kernels."
            )
