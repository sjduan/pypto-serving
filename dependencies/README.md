# UFlow candidate dependencies

Initialize the fetchable Lib submodule base, then apply the exact accepted
runtime-worker overlay:

```sh
git -c core.autocrlf=false submodule update --init pypto-lib
python tools/prepare_uflow_lib.py pypto-lib pypto-lib
python tools/prepare_uflow_lib.py pypto-lib pypto-lib --check
```

The gitlink is public `ec58b8ad7454b321f379260e95a0ae4431a5ae57`.
`locked-lib.json` plus `patches/qwen3-runtime-workers.patch` reconstructs the
complete tree of original local commit
`d77ac175065d06f8afe401ab230370dfb7703d3b`. It changes the two accepted Qwen
prefill attention paths and real decode attention to use runtime worker counts
and matching task strides. It preserves BATCH16, barriers, numerical kernels
and buffer capacities. This packaging does not claim new 20/24-core results.

The helper stages the overlay in the dependency index, verifies the complete
tree and file hashes, and refuses unrelated edits or a tampered patch. It is
idempotent. Do not run `git submodule update` again after applying it, because
that would restore the unpatched base. The Qwen source loader rejects the
known unpatched base before compiling kernels.

The matching PyPTO, Simpler, PTOAS and ISA sources are locked and reconstructible
in the [UFlow dependency directory](https://github.com/sjduan/UFlow/tree/main/dependencies).
Use PyPTO `8007f114` and Simpler `508becb9` **source overlays**, not a fetch of
those unpublished SHAs or ub_sim's different vendor Simpler revision. Install
the UFlow SDK from the exact repository commit in `uflow-sdk.json`:

```sh
python -m pip install --no-deps ./UFlow/sdk/python
```

The connector checks the accepted SDK module hashes before registration. A
different package version or matching pool name is not enough. Keep provider
configuration, shared partition seals and SDK provenance together. No admin
capability, model weight or live data belongs in this source lock.
