# UFlow shared DDR candidate

This fork delivers the accepted managed DDR connector for multiple independent
serving instances sharing one Simpler-owned DDR allocation. UFlow owns the
physical budget, allocation identity, range leases, committed content and
Unknown outcomes. Serving owns KV placement and prefix-cache policy and wraps
the registered pool view as CPU tensors. There is no second KV payload pool
hidden in the connector.

Prepare the [fixed dependencies](../../dependencies/README.md) before starting a
model. The Lib gitlink points to a public base; the supplied exact overlay
preserves the tested runtime-width adaptation. The known unpatched base is
rejected before Qwen kernel compilation.

## Supported boundaries

| Component | Baseline behavior |
|---|---|
| Managed control | `managed_pool.py` and `managed_pool_sdk.py` normalize the fixed UFlow SDK and retain exact request/lease identities |
| Data view | `model/qwen/native_ddr_transfer.py` wraps the registered original mapping; actual native access still requires the fixed Simpler/A3 runtime |
| Shared replicas | `shared_pool.py` retains one pool identity with explicit partitions and guarded shared prefix metadata |
| Fault handling | Uncertain operation/view/native completion stays retained; restart, process death or a host test is not an ACK |
| mem_service/ub_sim | Implemented in UFlow as a simulated remote execution provider; this connector does not turn a Guest handle into a POSIX tensor |

The existing opt-in configuration and public CLI behavior remain unchanged.
Historical `examples/run04/existing_pool.py` describes an observed pool and its
original seals; its hardcoded paths are evidence, not a fresh deployment recipe.
Generate a new configuration and partition contract for a later device run.

## Bounded task lifecycle

The generic `task_supervisor.py`, `task_lifecycle.py` and `task_resources.py`
come from separately accepted safe-teardown commit `567be28`. Their standalone
entry now loads standard-library `types` before exposing the script directory,
preventing the sibling `types.py` from shadowing it in a clean interpreter.
Ownership, signal, resource and accounting checks remain unchanged.
Its pure physical-versus-logical classifier is available as `closure_report.py`;
generic tests no longer import the O1 experiment directory. The O1 entry and
its three entry-specific CPU tests remain on that separate branch. Use the supervisor for new native
or serving jobs: establish a dedicated task root, bounded run/close/TERM/KILL
budgets, process/memory/file limits and the actual argv, then invoke:

```sh
python -m pypto_serving.serving.native_ddr.task_supervisor --config "$CASE/supervisor.json"
```

The configuration parser documents the exact required fields in
`bounded_spec`; the Linux unit tests provide synthetic examples. Process
ownership is established before the entry gate opens. The independent monitor
accepts failure notices and uses exact task identities/pidfds, waits and reaps.
It reports physical process/resource closure separately from logical DDR/SDK
closure. A wedged kernel remains quarantined; the wrapper performs no GDB
attach, code injection, broad process-name kill or device/server reset.

## Validation boundary

Historical four-card shared-serving validation used Serving `1f96d877` and
UFlow `8eac5c22`. Separate two-CPU-Guest validation used UFlow `16f7ea6d` and
ub_sim `c1e73ed2` and passed bidirectional 20 MiB data checks. An additional
same-Guest import hit GSVA overlap/`KEY_MISMATCH`; full normal drain was not
proven. Wait for ub_sim support before that targeted retest.

This delivery runs host contract/tensor/lifecycle tests only. It does not claim
a new model accuracy run, device test, QEMU boot, four-card regression or
performance result. Four-card regression of the merged candidate is explicitly
deferred to the next A3 task before sixteen-instance expansion.

On this fork, push/PR CI runs host checks. Physical NPU and daily performance
jobs require an explicit manual `run_npu=true`; ordinary PRs do not reserve a
device. Upstream workflow behavior is retained. GitHub Pages publication is
restricted to the upstream repository.
