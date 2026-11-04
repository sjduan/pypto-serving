# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the unchanged M2 matrix with read-only first-divergence logits diagnostics."""
import hashlib
import json
import os
from pathlib import Path

import torch

from pypto_serving.model.qwen.npu_runner import Qwen314BModelRunner

_ORIGINAL_DECODE = Qwen314BModelRunner.run_decode
_ORIGINAL_PREFILL = Qwen314BModelRunner.run_prefill
_TRACE_IDS = {"serving-req-1", "serving-req-2", "serving-req-3"}


def record(phase, batch, logits, sampled):
    ids = batch.request_ids
    if len(ids) != 1 or ids[0] not in _TRACE_IDS:
        return
    length = int(batch.seq_lens[0])
    if length > 17:
        return
    values = logits.detach().cpu().float().contiguous()
    row = values[0]
    top_values, top_ids = torch.topk(row, 16)
    data = {"phase": phase, "request_id": ids[0], "seq_len": length,
            "sampled_id": int(sampled[0, 0]) if sampled is not None else None,
            "cpu_argmax": int(row.argmax()), "finite": bool(torch.isfinite(row).all()),
            "logits_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
            "top_ids": top_ids.tolist(), "top_values": top_values.tolist(),
            "top_margin": float(top_values[0] - top_values[1]),
            "input_ids": batch.token_ids.tolist(),
            "page_ids": [a.page_ids for a in batch.kv_allocations]}
    root = Path(os.environ["UFLOW_M2_ATTEMPT_ROOT"]) / "results"
    with (root / "precision-trace.jsonl").open("a") as output:
        output.write(json.dumps(data) + "\n")
    if phase == "decode" and length == 15:
        torch.save(values, root / (ids[0] + "-seq15-logits.pt"))


def traced_decode(self, model, batch):
    result = _ORIGINAL_DECODE(self, model, batch)
    if len(batch.request_ids) == 1 and batch.request_ids[0] in _TRACE_IDS and int(batch.seq_lens[0]) <= 17:
        device = self._decode_logits_dev_tensor
        if device is None:
            values = result.logits
        else:
            values = torch.empty((1, device.shape[1]), dtype=device.dtype)
            self._shared_l3_worker().copy_from(values.data_ptr(), device.data_ptr,
                                              values.numel() * values.element_size())
        record("decode", batch, values, result.sampled_token_ids)
    return result


def traced_prefill(self, model, batch):
    result = _ORIGINAL_PREFILL(self, model, batch)
    record("prefill", batch, result.logits, result.sampled_token_ids)
    return result


# Spawn reimports this entrypoint, so the worker receives the same observation-only hooks.
Qwen314BModelRunner.run_decode = traced_decode
Qwen314BModelRunner.run_prefill = traced_prefill


if __name__ == "__main__":
    import m2_baseline
    m2_baseline.main()
