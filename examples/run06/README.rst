Run06: independent Qwen services sharing one managed DDR pool
===========================================================

This example consumes the common owner public manifest. It does not create a
pool, allocate another set of slots, obtain an admin capability, or finalize
another line. The accepted SDK remains the mapping, range, version and budget
authority. The four M1 slots are actual preallocated 22MiB ranges: a 2MiB guard
and a 20MiB Qwen KV payload each. Each service maps the same 128MiB backing.

The explicitly configured shared directory stores at most four canonical keys
and four publication intents. A real committed SDK operation receipt and current
snapshot are required before publication. Discovering a descriptor grants no
read access: the worker imports it through the original replay/epoch-controlled
interface, obtains a current reader, releases that reader, then reacquires and
revalidates the exact version before native LOAD. Local HBM hits take priority.
Directory misses, pressure and confirmed stale readers fall back safely; an
ambiguous grant/result retains the original client/native obligations.

``shared-slots.json`` is a task-local normalized copy of the common public
manifest. It names the actual pool descriptor, four actual allocation identities,
source manifest SHA256 and public session-manifest path. Startup checks these
against the registered descriptor and real current allocation snapshot. It also
registers the actual ordinary SDK session with the common lifecycle owner.

Explicit optional CLI configuration::

  --enable-native-ddr-kv --native-ddr-kv-bytes 83886080
  --native-ddr-pool-endpoint /exact/common/u.sock
  --native-ddr-pool-id ACTUAL_ID --native-ddr-pool-epoch ACTUAL_EPOCH
  --native-ddr-sdk-sha256 ACCEPTED_SDK_SHA256
  --shared-ddr-directory /exact/m1/d.sock
  --shared-ddr-directory-epoch EXPLICIT_DIRECTORY_EPOCH
  --shared-ddr-slots /exact/m1/shared-slots.json

``directory_service.py`` owns only the bounded metadata service. Its fixed
epoch-matching stop file requires no remaining intents and never destroys UF.
Normal producer retirement leaves valid UF-owned content discoverable.

``host_probe.py`` runs the actual common SDK through independent producer and
reader interpreters. It proves producer exit before consumption, two same-version
read leases, actual overlapping-write rejection and another-range commit while
readers remain live. Its byte patterns are host tests, not model/native evidence.
The common heavy-job lock is held for that host test. Each SDK client is normally
unmapped; the M1 line and shared owner remain live for subsequent model testing.

Native/model processes must never inherit the host API fence. Model evidence
must separately cover two then four actual physical cards, cold target services,
two token-zero prefix blocks, full 80-side bytes, legitimate allocator pages,
direct/HTTP suffix and decode, actual prefill reductions and cooperative stops.
No two-instance or host result constitutes four-model completion. Measurement
uses the common bounded window without a held UF lease or directory lock.
