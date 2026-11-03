#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# This command is host-only; use worker_probe.py separately with an explicitly selected device.
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$scripts/env_all.sh"
first_path="$PATH"
source "$scripts/env_all.sh"
test "$PATH" = "$first_path"
cd "$UFLOW_RUN02_ROOT"
python "$scripts/prepare.py"
result_dir="$(mktemp -d "$UFLOW_RUN02_ROOT/results/verify-XXXXXX")"
printf 'fresh_result_dir=%s\n' "$result_dir"
python "$scripts/identity.py" --output "$result_dir/identity.json"
test -s "$result_dir/identity.json"
python "$scripts/model_assets.py" --output "$result_dir/model-assets.json"
test -s "$result_dir/model-assets.json"
python -m pytest -q -p no:cacheprovider "$scripts/test_guards.py"
python -m pytest -q -p no:cacheprovider src/serving/tests/unit/cli
