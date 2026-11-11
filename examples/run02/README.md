# Run02 M1: fixed environment and native loading audit

This is an environment-only entrypoint for Phase F Run02. It does not send model requests,
implement KV offload, or claim M2/M3/M4 acceptance. All commands below run inside
`serving-b-sj` / `openeuler-2403-DS`, not on the host with a similar directory name.
Do not use these scripts to upgrade dependencies or repair a mismatched shared checkout.

## Fixed inputs

| Component | Required identity |
| --- | --- |
| PyPTO, read-only `/workspace/pypto` | `8007f1146a800d679301c5ea14bb644b863da6bf` |
| Simpler, read-only `/workspace/pypto/runtime` | `508becb93e182a3035a14e3f9226929dbe994346` |
| PTOAS | `0.70` |
| serving | `5f548be23d7af120b20a40694601f5bb5b8b7a90` |
| serving's pypto-lib gitlink | `ec58b8ad7454b321f379260e95a0ae4431a5ae57` |

The scripts fail on wrong/dirty source inputs and unexpected import routes. The M1 source
clone remains at the fixed serving commit; later feature work belongs in its own task tree,
with the exact preceding reviewed feature commit, rather than silently advancing this snapshot.
No recursive submodule update is performed. Shared Git objects are reused with dedicated
clones/branches; shared source checkouts, refs, installed packages, and backend builds are not modified.

## Layout and initialization

Default `UFLOW_RUN02_ROOT=/workspace/uflow/phase-f/run02/t_ad4b1e29`:

- `scripts/examples/run02/`: a copy of these versioned entrypoints.
- `src/serving/`: isolated clone and branch `run02/t_ad4b1e29`; `pypto-lib/` is a separate pinned clone.
- `envs/python/`: dedicated Python 3.11 venv, with system packages and explicit read-only reuse of
  `/workspace/pypto/.venv/lib/python3.11/site-packages` for build tools and the stamped Simpler binding.
- `envs/lint/`: optional, separate Ruff tooling; not part of the runtime dependency chain.
- `build/pypto/`, `build/wheels/`, `build/programs/`: CMake, wheel, and future generated-program outputs.
- `cache/`, `tmp/`, `logs/`, `results/`, `configs/`: task-owned output roots.
- `results/verify-XXXXXX/`: a fresh audit directory per verification, never a stale success marker.

Copy this directory under `scripts/examples/run02` in the container, then run:

```bash
cd /workspace/uflow/phase-f/run02/t_ad4b1e29
env -i HOME=/root PATH=/usr/bin:/bin bash scripts/examples/run02/prepare.sh --build
source scripts/examples/run02/env_all.sh
python scripts/examples/run02/model_assets.py --expose-existing-mount --output results/model-assets.json
env -i HOME=/root PATH=/usr/bin:/bin bash scripts/examples/run02/verify.sh
```

`prepare.sh --build` builds a wheel from the unchanged fixed PyPTO source into the task's
CMake directory, then installs it only into the task venv. No dependency installation/upgrade
or network source update is requested by that build. Repeating it uses the same build cache.
`prepare.sh` without `--build` checks/prepares the layout without recompiling.
`env_all.sh` can be sourced repeatedly and does not build or initialize an NPU.
The `/workspace/uflow/env_all.sh` global name is deliberately not created or overwritten.

The model command only exposes the already available `/models/Qwen/Qwen3-14B` input through
an absent leaf symlink at the contracted `/data/models/Qwen/Qwen3-14B`. On the verified B host,
`/data/models` resolves to `/home/data/models`; the container's existing `/models` bind mount
is that same input (config device/inode matched). No weights were copied, modified, or downloaded;
no container was restarted. Existing model paths, including dangling links, are not replaced.
The command hashes all required weight/config/index/tokenizer files, checks safetensors
index/header agreement, and parses the local tokenizer. This is not model inference.

Read-only external inputs also include CANN 9.0.0, the driver, MLIR installation, PTOAS, and
existing stamped Simpler binaries. Simpler's native startup creates transient POSIX shared
memory through its existing `/dev/shm` implementation; these are not new persistent output roots.
The advisory Run02 chip lock is under `/workspace/uflow/locks/`. No old UFlow authority or
quarantine directory is touched.

## Actual loading and build provenance

`identity.py` verifies the installed PyPTO Python files byte-for-byte against the fixed source,
checks that the loaded native core is the same file contained in the locally built wheel,
checks serving/lib/Simpler import paths, records the native binding's build stamp, and compares
the selected `a2a3/onboard/host_build_graph` binaries with their stamped compiler-cache outputs.
The backend cache stamp includes the fixed Simpler commit and its existing PTO-ISA pin
`217bb98952a8f9004b95fe938da97d8d5f68f702`; neither pin is changed.

PTOAS resolves from `/workspace/PTOAS/bin/ptoas` to
`/home/sj/git/PTOAS-0.70/bin/ptoas`. The wrapper, `_cli.py`, and actual
`ptoas/_core.cpython-311-aarch64-linux-gnu.so` are individually recorded and hashed.
The native CLI's `--version` is run only in a subprocess: its native exit can terminate
Python without raising `SystemExit`. A wrapper-version-only log is not a completed identity audit.

The original 346-step core build and subsequent clean-environment incremental rebuild both
succeeded. The new core's SHA256 is
`9d7c187e37d9d20a898c3e4342f4d4fd31cd805aed18427059d0a2b9b68b5169`.
It happens to equal the old installed core's hash; the new build provides the previously
missing same-source provenance without inventing a build-commit attribute.

## Optional single-chip backend startup smoke

After checking current health, HBM, existing users, and reservations, explicitly select a chip:

```bash
source scripts/examples/run02/env_all.sh
python scripts/examples/run02/worker_probe.py --device 0 --output results/worker.json
```

Device 0 is the chip used for this M1 observation, not a permanent reservation or automatic
choice for later tests. The script resolves physical chip/board/health/HBM from the current
`npu-smi` output, rejects remapped visibility, checks live device FDs and the NPU process list,
and holds an advisory Run02 lock. It does not claim that other users honor that advisory lock;
recheck resources before every subsequent run. Missing `task-submit`, `fuser`, and `ss` on this
container were not worked around by killing processes; live `npu-smi`, `/proc` device-FD checks,
and task-local socket observations were used instead. No service port is opened.

The smoke imports the dedicated PyPTO runtime and initializes a real single-chip Simpler L3
worker with `host_build_graph`, records parent/child maps and backend inputs, and closes it.
The observed child mapped the expected fixed `libhost_runtime.so`; normal teardown returned
zero and the child was reaped. This proves startup/loading compatibility only. It does not
submit an application program, copy KV bytes, prove async completion, or execute Qwen prefill/decode.

## Verified results and evidence

M1 runner self-checks (2026-11-03):

- Dedicated core build and clean-shell reinitialization/rebuild: passed.
- Installed PyPTO source equality: 208 Python files; native wheel identity: passed.
- Fixed binding/backend cache stamps and actual child loading: passed.
- Model visibility: 12 required files, including all 8 weight shards; 443 tensor headers;
  tokenizer parsing: passed. Full SHA256 values are in the model manifest.
- Host guard tests: 6 passed. Existing serving CLI unit tests: 21 passed.
- Ruff F checks, project Python headers, English-only checks, and shell syntax checks are
  recorded with the review evidence, rather than treating a zero-file Windows lint as coverage.

Raw evidence is under the container task root's `logs/` and `results/`; the review attachment
`run02-m1-evidence.zip` includes the final evidence index, raw outputs, CMake cache/compile
commands, and the exact entrypoint sources. The index identifies the final fresh verification
directory, worker result, and checksums. Earlier exploratory/incomplete logs are preserved but
not counted as accepted identity evidence. Generated builds/caches/logs/weights are not committed.

M1 still requires same-card F-SE acceptance. M2 is `t_cb402b6e`; it must perform an actual
ordinary Qwen3-14B serving request before KV changes. M3 and final offload/async behavior are
not tested here. No main integration, push, publication, or deployment of a model service is
claimed by this environment-only feature commit.
