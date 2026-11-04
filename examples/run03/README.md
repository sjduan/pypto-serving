# Run03 H1: host metadata protocol

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
