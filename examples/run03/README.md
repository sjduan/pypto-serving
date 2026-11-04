# Run03: dormant host metadata and explicit E10 staging

This example and the new shared modules are dormant. They do not change Run02
single-owner compatibility, IPC, scheduling, page leases or native transfers.
They require only Python 3.10+ and the standard library.

From the serving repository root:

```console
python -m unittest tests.unit.serving.test_shared_prefix_record -v
python -m examples.run03.host_record_probe produce > record.json
python -m examples.run03.host_record_probe consume < record.json
```

For byte-exact redirection on PowerShell versions that transcode native output,
use Python subprocess pipes (as the unit test does), rather than shell text
redirection. The producer and consumer are separate Python processes.

## API and identities

- `shared_compatibility.build_shared_compatibility(descriptor,
  trusted_attestation=TrustedDeploymentAttestation(...))` returns immutable
  canonical metadata with a stable `data_id`. The attestation must explicitly
  pin the complete normalized descriptor SHA256 at evidence level
  `trusted-full-assets-and-deployment-v1`. It is an out-of-band authority input,
  **not authentication of its caller**, a signature or proof of an actual load.
- The descriptor requires all twelve named Qwen3-14B asset full SHA256 hashes;
  optional positive sizes are included in the identity. Config, unscaled RoPE,
  absolute position semantics, tokenizer implementation, adapter=none, BF16,
  TP1, deployment component hashes and the exact full K/V block layout are
  included. Deployment component hashes represent actual source/binary
  manifests supplied by the trusted deployment authority, not Git labels.
  Unknown fields, schema versions and unsupported layouts/modes are rejected.
- `SourceBinding` retains an `ExternalBackingIdentity` with **all thirteen**
  E10 handle fields, plus source worker epoch, entry ID/generation and range.
  `ExternalBackingIdentity.from_dict(dataclasses.asdict(e10_handle))` copies
  the existing SDK dataclass losslessly without importing the SDK in H1.
  Only immutable backing, one aligned 20MiB block and in-bounds ranges work.
- `SharedPrefixRecord` carries the content ID, serving-provided complete
  32-byte chained prefix hash, aligned absolute token interval of exactly 128
  sealed tokens, payload SHA256 and source binding. It does not compute token
  hashes or maintain a cache directory. The input must already follow
  `kv_cache.NONE_HASH/hash_block_tokens`, including all preceding blocks.
- `DestinationBinding` keeps its own worker epoch, arena/cache epoch, local
  layout and num_pages, page ID and page lease. Capacity, paths and worker
  epochs never enter the shared content ID.
- `encode_record/decode_record` and `encode_destination/decode_destination`
  use bounded UTF-8 canonical JSON. Record limit: 64KiB, at most 32 fields or
  entries per container, twelve asset rows, depth 12 and 1024 total tree nodes.
  Numeric IDs/ranges are non-boolean signed-63-bit nonnegative integers;
  generations and resource IDs are positive. Digests are lowercase SHA256.
- `validate_candidate(record, expected_compatibility=...,
  expected_source=..., destination=..., expected_destination=...,
  expected_chain_hash=..., expected_token_start=...,
  expected_payload_sha256=...)` compares against caller-supplied **current**
  source and destination identities and returns only `MetadataCandidate`.
  Content equality cannot bypass stale source tuples or destination leases.

The caller must obtain current source identity from E10/owner authority and
current destination binding from the local page authority. Metadata equality
is not liveness. H2 must acquire and retain the actual E10 read view while
copying verified bytes; production must bind actual loaded assets/config and
deployment binaries to the trusted identity before publishing shared Ready.
H1 performs none of these authority, allocation, copy, transfer, Ready or
release operations.

## Evidence limits

The example embeds a normalized historical full-hash fixture from
`artifacts/run02-m2-baseline-20261104/model-full-hashes.json`, accepted by
`artifacts/run02-m2-review-20261104/receipt.json` on 2026-11-04. No weights
are read or rehashed. It proves neither a new hash nor current replica loading.

Deployment hashes, backing IDs, source/target leases, chained hash and payload
digest in the example are explicitly **host-only fixtures**. A passing probe
means two-process metadata compatibility and strict stale-identity rejection.
It is not real E10 exchange, a native LOAD, NPU/model consumption, cross-replica
Ready, DMA overlap or performance evidence. Those gates remain pending.

## H2: explicit real E10 acquisition and CPU staging

`shared_exchange.py` adds an opt-in E10 adapter and one preallocated 20MiB CPU
slot with 64-byte guards on each side. The independent trusted H1 context is
validated before acquisition and again before copying. All thirteen returned
E10 backing fields, the exact nonzero range and the read-only mmap are checked.
The adapter deliberately isolates its private `_identity.backing`/`_mapping`
coupling to fixed SDK revision `4fba229373f0b27f0d1c0f6a7aab405542c8fdda`.
SDK imports remain lazy; ordinary metadata/module imports require no SDK or
tensor/runtime packages.

The host snapshot is completely read and hashed before a target write. A
successful `StagedBlock` owns the retained source view and the actual CPU slot
lease; it is explicitly **not native Ready**. `close()` first confirms the SDK
release, then returns the local slot. A failed/unknown release preserves the
handle and busy slot for retry. An ambiguous acquire/retain is conservatively
unresolved; its `ExchangeFailure.resources` must be retained for external
resolution. A mid-copy host exception can partially write invalid target bytes;
it never promises unchanged bytes or native cancellation. Copy metrics count
verified successful CPU copies separately from mapped bytes and failed attempts.

Run both standard-library suites without hardware:

```console
python -m unittest discover -s tests/unit/serving -p "test_shared*.py" -v
```

`host_exchange_namespace.py` is a Linux-only real broker integration driver,
not a general deployment utility. Its task root is fixed to
`/workspace/uflow/runs/run03-host-exchange-20261105`. Provision fixed-version
sources and CPU-built daemon/shim there first. Run only its **outer** mode:

```console
python src/serving/examples/run03/host_exchange_namespace.py
```

It pins the outer mount namespace and new task directory FDs, uses private
`unshare --mount`, and verifies parent/namespace/propagation/canonical inode
before binding that new directory over `/workspace/uflow` in the child only.
Only then can the previously absent private authority be initialized. Peers
also validate inherited namespace/root anchors. The original dirty authority
is never opened for mutation, initialized, recovered or cleared. Insufficient
namespace privileges are a hard boundary, not permission to run unisolated.

The real independent producer creates/seals 60MiB of immutable memfd backing:
20MiB left sentinel, 20MiB deterministic payload, 20MiB right sentinel. The
consumer uses the ordinary `uflow` SDK, acquires the middle block read-only,
retains it and releases the root view, then makes a real bounded CPU copy.
Examples verify complete bytes/guards, response-only placement/binding/lease
mismatches, stale metadata/destination, a bad digest, busy slot and host cancel.
After producer revoke and exit, the old legal view remains readable while new
acquisition fails. All references and budget must drain before normal daemon
shutdown; final socket/PID/FD/lock and unchanged outer-mount evidence are kept.
Broker mapped-byte telemetry is not relabeled as our local CPU copied bytes.

This driver uses `--execution-mode simulated --startup-probe-bytes 0` to disable
HBM startup; its memfd, SCM_RIGHTS, mappings and CPU data are real. It never calls
NPU resource probes, ACL device initialization, PyPTO/Simpler or model/native
execution. Compatibility/deployment/worker/page contexts remain named host
fixtures, not new loaded-model evidence or production sharing. Full NPU/model,
cross-replica Ready, multi-card and performance gates remain NOT-RUN. H2 does
not install a scheduler path, enable shared serving, merge main or push.

The outer supervisor owns a separate inner session and a dedicated STOP pipe.
A 240-second workflow timeout requests normal inner cleanup and allows at most
120 seconds of cleanup grace; it never signals the process group. Inner peer
waits check the pipe and unwind through explicit peer EOF/stage release and
normal broker shutdown. A surviving owned child is recorded with bounded
descendant identities/FDs, not killed or called clean. Real outer inode,
namespace and mountinfo are compared on success, exception and timeout.

The consumer explicitly closes an acquired stage on post-acquire assertions or
handshake failures too. Cleanup errors preserve the original failure and record
unresolved view/lease identities and slot ownership. A failed new acquire after
revoke counts as revocation evidence only when the pinned SDK's actual
`_exchange` error frame contains the expected non-ok broker response with no FD.
Transport, malformed/protocol replies and unknown outcomes are failures. This
additional SDK-private inspection belongs only to the negative test driver;
it does not change the SDK or turn arbitrary exceptions into permission proofs.
