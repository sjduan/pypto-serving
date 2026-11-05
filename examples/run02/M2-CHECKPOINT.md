# Run02 M2 checkpoint: NOT accepted; hardware paused

Task: `t_cb402b6e`, runner run169, branch `wt/t_cb402b6e`.
Reviewed predecessor: `ae0666ce937bafb456c50606705ec28c3d8ad489` (M1).
Fixed serving base: `5f548be23d7af120b20a40694601f5bb5b8b7a90`.
Canonical main remains outside this task; no main integration or publication occurred.

## Mandatory pause gate

User pause `t_14428e51` is an additional native parent of this task.
No further A/B deployment, transfer, reservation, request, hardware diagnostic or retry is authorized
until the user explicitly releases that gate and its native dependency is resolved.
Idle hardware is not permission to resume. No monitoring agent, cron or automatic retry was created.
The existing 507018 compatibility failure remains a separate unresolved technical stop after release.

## Work saved, not validated as a working model baseline

- `m2_env.sh`: reuses the reviewed M1 Python wheel/venv read-only. Dedicated serving/lib source and
  all new outputs are under `/workspace/uflow/phase-f/run02/t_cb402b6e` in the B container.
- `m2_baseline.py`: production CLI configuration builder, AsyncLLMEngine/Qwen worker and HTTP app;
  intended short repeats, long prefix/chunked requests, concurrent batch, HTTP and cancellation checks.
  Test-only scheduler observer returns the original schedule unchanged. No alternate executor or
  offload feature is implemented. Request checks were NOT reached, so this is WIP, not accepted code.
- `m2_inspect.py`: task-owned process/FD/lock/port inspection used only for safe pause cleanup.
  It never initializes an NPU or sends a model request. Signal mode checks the driver command and cwd.

The remote serving clone remains at the reviewed M1 SHA; new WIP scripts were run from the task's
`scripts/examples/run02` directory. Do not claim the remote clone already contains the eventual local
WIP feature commit. Fixed PyPTO/Simpler/PTOAS/lib were not edited or upgraded.
The M1 model manifest is reused by name; no new full model hash/inference acceptance was completed.

## Real attempts and failure

Environment: `serving-b-sj`, `openeuler-2403-DS`, physical chip0, TP1, original HBM prefix cache on,
new DDR offload absent/off. Model input `/data/models/Qwen/Qwen3-14B` resolves to the existing
container mount `/models/Qwen/Qwen3-14B` through the already reviewed M1 symlink.

First attempt `logs/baseline-run169.log`: max-num-seqs=4 was rejected before application dispatch:
`topk_select_fwd is compiled for a fixed kernel BATCH of 16, but runtime max_batch_size is 4`.
The original failed result is retained in `results/baseline-run169.json`.
Its resource-tracker child was initially misclassified by the harness as an unreaped native worker;
that bookkeeping was corrected in the second script version, without changing production code.

Second attempt `logs/baseline-run169-b.log`: max-num-seqs=16, max-model-len=512,
max-num-batched-tokens=512, chunk threshold128, BF16 weights/KV, block128, NPU utilization0.80,
ring-dep-pool/task-window262144, ring-heap1073741824. Three application kernels compiled;
the native L3 child initialized and static tensors uploaded. The initial production warmup prefill
(batch16/tokens512) then failed before a user request or response.

Host error: `507018 ACL_ERROR_RT_AICPU_EXCEPTION`, stream44/task3, native child PID3909697,
run_epoch1/generation1/dispatch_id1. Device-classified evidence, not a guessed interpretation:

`logs/ascend/debug/device-0/device-3909697_20261103192024104.log:5`:
`FATAL(code=7): require_sync_start block_num=24 > limit=20 (deadlock guaranteed)`.

The same device log reports `simpler runtime failed with rc=-7` and an orchestration assertion.
This does not establish hardware incapability versus a safe existing launch-configuration solution;
F-SE must decide from the fixed source/actual runtime before any authorized retry. Do not change lib
constants, upgrade fixed dependencies, relax output checks, or move to A without satisfying the pause
and technical gates. No short/long/prefix/batch/HTTP/cancellation request has passed.

## Safe stop and fresh release evidence

After receiving USER_PAUSE, no new requests, deployment or hardware tests were submitted.
Only cleanup and evidence preservation continued. There was no model copy to A and no A session.

Owned identities: driver3776478, serving worker3795018, native child3909697,
Python resource tracker3795009. Their ownership was recorded from the original process tree/logs.
Driver cmdline and cwd were checked before each signal; no process-group or other-user kill occurred.
SIGINT requested normal cancellation. Existing engine logs reported `ReplicaEngineCore stopped` twice.
The already in-flight Simpler fatal finalization itself reported force-reset/quarantine handling and
later `probe confirmed`/generation retirement; no manual device/container reset was requested.
The host-only driver remained alive with the advisory lock/port after native children had exited.
After verifying that only its Python resource tracker remained and it held no device FD, the exact
owned driver received SIGTERM. The tracked SSH command exited143. This is not a normal model-run pass
or proof of successful warmup; interruption prevented a final `baseline-run169-b.json` result.

Fresh final checks, not inferred from old logs:

- Targeted `ps` for all four recorded PIDs returned no rows after driver exit.
- `results/pause-final-release.json`: driver process absent, physical0 health OK,
  selected-device FD users empty, task Unix sockets empty, port18892 bindable,
  original advisory lock `/workspace/uflow/locks/run02-npu-0.lock` acquirable/released.
- Final chip HBM was 60156/65536 MiB, NOT the pre-run 3145 MiB. No attribution or whole-machine idle,
  complete memory-drain or physical recovery acceptance is claimed. Do not interfere with other tests.
- No remaining task submission, pending model request, reservation, monitor or retry loop is known.
  This is task-owned process/FD/lock release evidence, not successful native computation.

Remote evidence remains in the task's `logs/` and `results/`. Local preserved copies are under
`artifacts/logs/`, `artifacts/results/`, `artifacts/run169-pause-preservation.tar` and
`artifacts/pause-owned-driver-stop.txt`. The earlier broad process snapshot is not part of the handoff.

## Verification and next step

Local syntax compilation of both new Python scripts and `git diff --check` passed.
No successful model baseline, suite, functional SE review, main merge or milestone completion occurred.
The request-suite body and HTTP comparisons remain unexercised WIP.
Potential harness lifecycle issue: cancellation of startup's `asyncio.to_thread(ready_event.wait)`
can leave the host driver waiting on its thread/executor after engine shutdown; preserve and review
that failure before using the harness again. Do not treat a timeout or cancellation as completion.

Block on `t_14428e51` now, preserving original M1 prerequisite and downstream M3 gate.
After explicit release, re-read task/claim/worktree/SHA, resolve the fixed-stack technical stop with
F-SE, then recheck resources before any hardware work. Do not automatically rerun the failing command.
Temporary notification-policy=all was refused by the CLI worker mutation guard; no bypass was used.
Manager is concentrating the real pause/blocker notice in the original goal/thread. No Slack receipt
or user-read claim is made by this checkpoint.
