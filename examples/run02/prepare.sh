#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
set -euo pipefail
scripts="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$scripts/env_all.sh"
exec /workspace/pypto/.venv/bin/python "$scripts/prepare.py" "$@"
