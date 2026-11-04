# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Protect M2 data checks while allowing model outputs to vary across independent requests."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def validate(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "examples/run02"))
    spec = importlib.util.spec_from_file_location("run02_baseline_validation", root / "examples/run02/m2_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_output


def record():
    return {
        "name": "normal", "prompt_token_ids": [1], "generate_config": {"max_new_tokens": 2},
        "outputs": [
            {"token_id": 10, "text": "A", "text_delta": "A", "finished": False,
             "finish_reason": "", "prompt_tokens": 1, "completion_tokens": 1, "token_ids": []},
            {"token_id": 11, "text": "AB", "text_delta": "B", "finished": True,
             "finish_reason": "FINISHED_LENGTH", "prompt_tokens": 1, "completion_tokens": 2,
             "token_ids": [10, 11]},
        ],
    }


@pytest.mark.parametrize("sequence_type", [list, tuple])
def test_independent_normal_outputs_can_differ(validate, sequence_type):
    first = record()
    second = deepcopy(first)
    second["outputs"][1].update(token_id=12, text="AC", text_delta="C", token_ids=[10, 12])
    first["outputs"][1]["token_ids"] = sequence_type(first["outputs"][1]["token_ids"])
    second["outputs"][1]["token_ids"] = sequence_type(second["outputs"][1]["token_ids"])
    assert validate(first) == [10, 11]
    assert validate(second) == [10, 12]


@pytest.mark.parametrize("corruption", ["token", "text", "prompt_count", "output_count", "truncated", "invalid_text"])
def test_response_data_corruption_is_rejected(validate, corruption):
    item = record()
    final = item["outputs"][-1]
    if corruption == "token":
        final["token_ids"] = [10, 12]
    elif corruption == "text":
        final["text"] = "AX"
    elif corruption == "prompt_count":
        item["outputs"][0]["prompt_tokens"] = 2
    elif corruption == "output_count":
        final["completion_tokens"] = 3
    elif corruption == "truncated":
        final["token_ids"] = [10]
    elif corruption == "invalid_text":
        final["text"] = "\ufffd"
    with pytest.raises(AssertionError):
        validate(item)
