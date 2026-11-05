Run04 managed Qwen evidence driver
=================================

Current status: host preparation only. All new A/B Worker, native transfer,
model, device reservation and pool cases are NOT-RUN under the 2026-11-05
human resource hold. ``plan`` and ``check`` use only the Python standard
library. Neither imports the production package, loads weights, connects to
UFlow, starts a process, nor reserves a device. A successful offline check
is labelled HOST-EVIDENCE-CHECKED and never grants D1 acceptance.

Inputs and host commands
-----------------------

The sealed configuration has schema 1, the twelve full Qwen3-14B asset
hashes, exact source/binary deployment expectations, frozen dependency pins,
five 20MiB logical slots, A10, and two normal rounds. The snapshot is a
sealed observation after actual n005 release: four pool-history records,
old quarantine charged 128MiB, no normal live pool, total cap 256MiB.
Historical module/binary hashes are expectations to check against the new
loaded owner; they are not a fresh runtime result.

From this source tree::

    python examples/run04/qwen_driver.py plan \
      --config /sealed/config.json --snapshot /sealed/retained-snapshot.json \
      --output /new/plan.json

    python examples/run04/qwen_driver.py check \
      --config /sealed/config.json --report /case/results/probe.json \
      --descriptor /case/catalog/normal-descriptor.json \
      --log /case/logs/probe.log --output /new/offline-check.json

Outputs use exclusive creation. Progress ``results/probe.json`` remains the
existing Run02 mutable report; the first failure and raw logs are separate.
Raw byte files may be reviewed in their original location or in the report's
``diagnostics`` sibling, preserving filenames and hashes. Missing captures,
old leases, incomplete rounds, reordered blocks and early publication fail.

Future execution, only after explicit human environment release
--------------------------------------------------------------

No execution command is authorized by this document. The driver requires
the current hold plus a newer human release record explicitly superseding
``serving-ab-npu-hold-20261105``. An older release or a plan cannot resume it.
The release record must identify the actual human message and chat, state
``released_by_explicit_human``, carry a later UTC timestamp and set the
full device-admission field true. The operator must verify that source;
the JSON is an admission record, not an independent source of permission.

In the existing fixed container, prepare the normal reviewed source/runtime
environment, never the host API fence. Put this new serving source first on
PYTHONPATH, accepted P1 SDK/examples next, and the frozen PyPTO/Simpler paths
after them. Keep PYPTO_LIB_ROOT on the accepted d77 library. Preserve the
reviewed CANN 9 compiler/sdk-view, native library paths and offline tokenizer
settings. Use a fresh task build/cache/tmp/log root, not a prior failure
domain. Before running, reconcile deployed source hashes with the sealed
configuration. Runtime startup independently compares all loaded fingerprints.

The actual executable entry, after that separate release, is::

    python -u examples/run04/qwen_driver.py run \
      --config /sealed/config.json --snapshot /sealed/retained-snapshot.json \
      --hold /current/hold.json --release /new/human-release.json \
      --existing-state /workspace/uflow/runs/d1-managed-ddr-20261105/serving/a002/n001/state \
      --root /workspace/uflow/runs/d1-managed-ddr-20261105/serving/host-prep-20261106/device-r001 \
      --output /new/execution-summary.json

Case sequence and real interfaces
--------------------------------

1. ``small-native`` calls the existing same-Catalog native fixture, including
   compilation/assembly before registration, strict helper classification,
   actual child registry, same-whole-Buffer native parameters, byte guards,
   normal owner close, SDK detach/unmap and UF Destroy. Its synchronous
   ``copy_to`` overwrite is not counted as model-computation evidence.
2. ``normal-two-rounds-and-negatives`` constructs the actual production CLI
   and AsyncLLMEngine through the reused Run02 driver. The startup handshake
   must bind exact loaded source/binaries, twelve complete asset hashes,
   physical UFlow resource and the production content namespace. Each normal
   round uses a different 257-token prompt: cold prefill 257, HBM hit prefill
   1, and two contiguous blocks at token offsets 0 and 128. Ordinary allocator
   reservations force two separate real 127-token prefill plus one decode
   overwrites on the original HBM source pages. Sources remain reserved while
   LOAD restores into different legal original-arena pages. Round zero restores
   via direct engine calls; round one uses the real HTTP completions endpoint.
   Both require suffix/decode, prefill 1, complete 20MiB/80-side/128-row checks
   for each block, held HBM sentinels and full pool-complement guards.
   Existing Run02 bounded pressure, gap, cancellation and short-receipt controls
   run against this same managed engine. One actual retired original envelope
   is replayed without rerunning the 48-cycle/4160-control historical matrix.
3. ``stop-load`` and ``stop-store`` each call the existing passive observer:
   stop must fall between actual native-pending acceptance and raw completion.
   No native event or result is withheld. Normal stopped owner and actual SDK
   unmap/release remain necessary.

One normal pool is created for each consumer session, never per HTTP request.
Each must really release before the next one. Four initial records plus four
new records exhaust the cap of eight. The original Unknown pool, three original
UF owners and their responsibility remain untouched and charged. Any owner,
epoch, pin, budget or resource discrepancy stops without another authority/card.

Failure and evidence boundaries
-------------------------------

Production logs preserve the complete typed submit event before transport and
the first nonterminal exception type/traceback separately from a stopped replay
receipt. A first-failure observation never marks stopped, retires replay,
commits a writer, validates an HBM page or acknowledges unmap. Managed shutdown
timeout retains the original process handle/queues and uses the preinstalled
static worker recovery channel. The driver retains live children and exception
frames on failure; it adds no arbitrary exec endpoint and never force-kills them.

Entry byte readback uses a real SDK R lease on Committed content. Full pool
snapshots are a separate opt-in guard-only physical observation of the existing
mapping, including unallocated guards. They require an already idle native
owner, no local or SDK range obligations, one mapped client, matching inherited
child registry and identical before/after SDK projections. They neither acquire
prefix-consumption authority nor write/initialize/allocate guards.

The checker correlates typed STORE/LOAD receipts, page/range leases, model
schedules, SDK commit/read-release, publication ordering, raw native submit/
outcome-zero completion and native cleanup. External PID/FD/lock/port/provider
release and capture origin still require review. Outputs must be reasonable
and token-consistent; deterministic logits are not an acceptance requirement.
Host fixtures and prior Run02 results do not establish fresh native success.
There is no automatic I1, D2, integration or GitHub push.
