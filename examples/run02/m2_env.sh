#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# Reuse the reviewed M1 wheel/venv read-only; all new outputs belong to M2.
export UFLOW_RUN02_ROOT="${UFLOW_RUN02_ROOT:-/workspace/uflow/phase-f/run02/t_cb402b6e}"
source "$(dirname "${BASH_SOURCE[0]}")/env_all.sh" || return 1
export VIRTUAL_ENV=/workspace/uflow/phase-f/run02/t_ad4b1e29/envs/python
export PTOAS_PYTHON="$VIRTUAL_ENV/bin/python"
export PATH="$VIRTUAL_ENV/bin:$PATH"
# Per-dispatch ring sizes are supplied explicitly by CLI/config, not obsolete env defaults.
unset PTO2_RING_HEAP PTO2_RING_TASK_WINDOW PTO2_RING_DEP_POOL
