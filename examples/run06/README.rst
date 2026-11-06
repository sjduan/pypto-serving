Run06: independent Qwen replicas sharing one existing DDR pool
=============================================================

Status and scope
----------------

This directory implements the metadata partition preflight and an executable
scenario controller. CPU fake transports test orchestration only. No new
Run06 model/native result is implied by a successful plan or CPU suite.
Current host-only inputs leave ``device_release`` and ``operation_allowance``
null. Neither a plan nor an example supplies execution authority or a device
reservation. Frozen SDK/Lib/PyPTO/Simpler are unchanged.

The only pool is the lifecycle owner's existing 128MiB B pool. Four actual M1
allocations each contain a 2MiB guard and a 20MiB payload. Every replica uses
the same allocations. The metadata directory carries keys/receipts and never
KV payloads. UF remains the lease/version/mapping authority. Ordinary clients
register with the existing external owner; this controller never creates,
destroys or restarts that pool, changes its public manifest, or retires M1.

Partition preflight
-------------------

``shared_pool.partition_metadata`` validates exact schema, public SHA, group,
B domain, full descriptor, M1/O1 indices, allocation line/identity/generation/
offset/size, and complete guard/payload layout. Every normalized block must
match the ordered public M1 list. The production SDK adapter also performs
readonly real descriptor/allocation checks before RegisterPoolClient. A
metadata seal grants no access; consumers still acquire/revalidate real UF
readers. Bad/duplicate/O1/stale/bool-schema configurations fail before any new
SDK mutation or directory startup.

Generating a pure plan
----------------------

Prepare from a sealed task-local deployment and a JSON containing the twelve
previously accepted complete model asset hashes::

  python -B examples/run06/prepare_scenario.py \
    --deployment /path/to/sealed-m1-deployment \
    --assets /path/to/model-assets.json --replicas 2 \
    --root /workspace/uflow/runs/phasef-parallel-20261106/m1/device-two-001 \
    --output /path/to/new-local-plan-two

  python -B examples/run06/scenario.py plan \
    --config /path/to/new-local-plan-two/local-plan-config.json \
    --output /path/to/new-local-plan-two/independent-plan.json

The local bindings copy uses the exact recorded public manifest bytes. The
separate ``runtime-config.NOT-RELEASED.json`` references the B deployment and
has no runtime release or journal allowance. For the four-instance stage use
``--replicas 4`` with another new root. Its future release must identify the
sealed real two-instance result from the same source/physical pool first.
These commands do not scan assets/source on B, import device runtimes, call
SDK/control endpoints, query/reserve cards, or start models.

Future execution and evidence
-----------------------------

Only after the actual resource/headroom gate is released, execute on B under
the sealed native environment::

  python -B examples/run06/scenario.py run \
    --config /path/to/released-runtime-config.json \
    --output /path/to/new-real-device-evidence.json

The release and shared operation allowance are source/public-manifest/pool
bound. All normal cached records from both lines must stay below4096; real
terminal recovery headroom is separately checked against8192. No allowance
is inferred from historical counters. Missing identity, full source/assets/
binary closure, unknown/frozen pool state, retired line, mapping conflicts,
occupied FD/driver domain or insufficient HBM prevents a new model/SDK
mutation. Per-card admission uses full FD/health/HBM and actual physical/
board/chip/BDF mapping plus legal locks and common reservations. Qwen requires
at least40000MiB free, following accepted D1, not the O1 4GiB microbenchmark
budget. The actual KV allocation uses the original0.8 utilization policy.

The baseline loads only one model per physical card at a time, disables DDR,
and performs3warmup+20measured requests after genuine original-allocator
HBM-cold reset for every sample. It normally stops before a shared model is
loaded on that card. Source compilation/startup shares the bounded host-job
slots. Each actor has distinct ports/build/cache/log/diagnostic roots.

Then one independent directory and two/four independent TP1 engines run on
separate physical devices. The producer calculates and STOREs two contiguous
token0 blocks. Each never-STORE consumer does2direct+2HTTP requests from cold
HBM, with two actual LOADs, true257-to1 prefill, suffix/decode, and full80-side
source/DDR/restored payload captures. Two held original-arena HBM sentinel
pages are captured before/after every functional restore. Assigned UF guards
use actual readers, revalidation and release ACKs. A separate host-fenced
ordinary observer holds two original real readers while the producer's
original allocator/model calculation overwrites its source pages. Producer
normal exit and actual SDK unmap precede each consumer's next2direct+2HTTP
restores; the observer is released and normally closed.

Shared performance also uses3warmup+20measured pairs per consumer, resetting
HBM every sample and requiring fresh native LOAD receipts. No second-sample
HBM hit can masquerade as shared DDR timing. Raw per-request outputs, actual
prefill schedules, native typed receipts and original node submit/result
bindings, process CPU/RSS, bounded measurement-window/co-tenant context,
median/p90 and paired differences are retained. HBM priority is tested without
reset; unrelated double misses run last, after all shared measurements, so
finite replacement cannot contaminate them. Optional four-instance routing
uses only existing ordinary static replicas; no cache-aware router or router
owned models are added.

The controller reads/correlates full original actor evidence files. It does
not accept stdout readiness, labels, or an accepted-only native event as a
pass. Normal SDK Unmapped/children/lease state and original process births,
selected FD/health, owned sockets and reservations are checked before normal
release. Failure/cancel/unknown retains the original owner/domain; there is
no automatic new attempt, kill, reset, force-unmap or fake cleanup. The shared
owner stays allocated until both lines genuinely finish.

Finite operation-record budget
------------------------------

With all normal close ACKs included, the two-instance full stage is248 and the
four-instance full stage636 new cached mutations:884 total. This preserves
all samples and guard/ACK evidence. Shared accesses enforce a total lifetime
maximum of3 SDK child-attach admissions. An unexpected fourth actual
inherited holder remains unacknowledged/Unknown and blocks continuation and
ordinary close; it is never classified away. Legacy non-shared behavior is
unchanged. Historical accepted D1 captures had1 mapped child; that history is
supporting topology evidence, not a new M1 result or grant.

The source-derived table is emitted in every plan. Source directories and
observer each cost registration/mapping plus normal unregistration/unmap.
All real native LOAD/import/diagnostic/guard/publication readers and writer
commit paths are counted. Readonly snapshot/describe/query and same-ID replay
cost no new cached record. The fixed normal caps are O1=2888 and M1=884.
At the historical312, normal completion would total4084, distance12 from4096
and strict spare11. That observation is not a fresh activation or permission.

The future manager-issued immutable schema2 envelope binds both final Git
source SHAs, the original manifest/pool/epoch/provider, one fresh counter
anchor A, the ordered stage caps, limits4096/8192 and terminal reserve256.
Its exact participants table uses O1 and M1 keys, each with source_sha, total,
and ordered stages as objects with exactly id and max. O1 stages are
calibrate606/run2074/profile206/finalize2; M1 stages are two248/four636.
Finalizer2 belongs only to O1. No runtime envelope has been issued.

operation_allowance must be exactly an absolute path and sha256 reference to
that immutable file. device_release must also pin envelope_sha256, alongside
the existing source/public-manifest identity and runtime-file hashes. Defaults
remain null. prepare_scenario writes NOT-RELEASED configurations, never grants.

Before either stage, R is a fresh original-Catalog counter while this line has
no client/child/view/range/native obligation. With C derived only from real
previously closed stages, U the current and future own caps, and Q=2888:

::

    other_spent_lower = max(0, R - A - C)
    other_remaining_upper = Q - other_spent_lower
    worst = R + U + other_remaining_upper

Require strict integers, R>=A, lower<=Q, worst<4096 and worst+256<8192.
M1 two has C0/U884; four has C248/U636; done has C884/U0. The ledger validates
the full previous model/native stop, original births, SDK Unmapped/children/
leases, FD/health and normal endpoint/reservation release evidence. CPU,
plan-only and shallow status labels cannot supply C. An arithmetic remaining
zero is no lifecycle completion or Destroy authority.

Once-only started/closed evidence is stored under the fixed M1 campaigns
directory keyed by envelope SHA. Exclusive creation precedes any reservation,
SDK mutation or Worker. An incomplete/failed claim blocks another run root or
attempt; no automatic retry or replacement. Closed receipts seal the original
start and complete real report. These are append-only evidence, not mutable
quota files or an activation service. There is no daemon or polling.

The local CPU suite checks all204 manager reference vectors and counter,
source, fake closure, original native/client/child/lease, duplicate phase and
tampered evidence negatives. This does not establish device/model success.

Retained-owner recovery
-----------------------

New-source controllers install ``<root>.recovery.sock`` before admission or
actor creation. It is mode0600 and checks the actual UNIX peer UID. The
immutable installed descriptor binds the original controller PID/birth,
configuration, source commit/manifest, public manifest, envelope and fixed
claim path. Only after failure does the retained descriptor add the actual
original claim hash/identity and one deterministic operation ID.

The sole accepted request is the exact three-field JSON object ``op``,
``identity``, ``operation_id`` copied from that retained descriptor, with
``op="stop-owned"``. There are no PID, argument, code or retry fields.
Duplicates share one cleanup owner and its actual receipt; a failed stop
cannot be re-executed by a second request or a new operation ID. A changed
claim, wrong identity, incomplete startup, in-flight/Unknown work or missing
proof keeps the original ownership and locks.

Actor command errors return immediately on the original channel, retaining
the exception/traceback while the controller exits its measurement context.
The failed actor rejects further generation/reset/STORE/LOAD work, including
new HTTP admission, but can consume its one original stop command. ModelActor
uses the optional engine ``cooperative_only_stop=True`` constructor policy and
explicit ``stop(cooperative_only=True)``. Timeout returns False and retains
worker handles/queues without terminate/kill. The default engine policy is
unchanged. Startup failure cleanup also inherits the constructor policy.

Recovery sequentially stops original actors and checks actual worker/native
proofs, startup and pre-stop descendants, private shared-memory names, driver
processes/FDs and real SDK unmapping. Original observer R-release and directory
close are required when those owners exist. Only complete actual closure can
issue the real empty reservation ACK and close this controller's locks and
listeners. The separate recovery receipt never writes a normal phase closed
receipt, never establishes two-stage success or four-stage eligibility, and
the failed controller ultimately exits with its original scenario error.
These hooks cannot be injected into older live processes or reopen a failed
claim. New-source native execution still requires its own valid release.
