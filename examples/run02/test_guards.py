# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-only guard tests. Fixtures here are synthetic; actual machine evidence is separate."""
import json
from pathlib import Path

import pytest

from model_assets import inspect_model, sha256
from worker_probe import selected_chip_status


def test_sha256_matches_standard_library(tmp_path):
    import hashlib
    path = tmp_path / "bytes"
    path.write_bytes(bytes(range(256)) * 200)
    assert sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_non_qwen_rejected_before_loading_assets(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "not-qwen"}))
    with pytest.raises(ValueError, match="Qwen3-14B"):
        inspect_model(tmp_path)


def test_index_cannot_escape_model_root(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 40}))
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../outside"}}))
    with pytest.raises(ValueError, match="escapes"):
        inspect_model(model)


def test_unknown_npu_output_is_rejected():
    with pytest.raises(ValueError, match="Cannot resolve"):
        selected_chip_status("unsupported format", 0)


def test_health_and_memory_are_for_selected_physical_chip():
    fixture = "\n".join([
        "| 0 Ascend910 | OK | 170 |",
        "| 0 0 | 0000:9D:00.0 | 0 0 / 0 3144 / 65536 |",
        "| 0 Ascend910 | Warning | 170 |",
        "| 1 1 | 0000:9F:00.0 | 0 0 / 0 60000 / 65536 |",
    ])
    assert selected_chip_status(fixture, 0) == {
        "board": 0, "chip": 0, "physical_id": 0, "health": "OK",
        "hbm_used_mib": 3144, "hbm_total_mib": 65536,
    }
    assert selected_chip_status(fixture, 1)["health"] == "Warning"
    with pytest.raises(ValueError):
        selected_chip_status(fixture, 2)


def test_environment_entrypoint_has_no_shared_output_roots():
    text = Path(__file__).with_name("env_all.sh").read_text()
    assert 'TMPDIR="$UFLOW_RUN02_ROOT/tmp"' in text
    assert 'PYPTO_CACHE_DIR="$UFLOW_RUN02_ROOT/cache/pypto"' in text
    assert 'PYPTO_PROG_BUILD_DIR="$UFLOW_RUN02_ROOT/build/programs"' in text
