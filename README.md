# pypto-serving

PyPTO Serving is a small local inference stack for running Qwen3-14B and DeepSeek V4 generation with PyPTO kernels on Ascend NPUs. It includes an installable Python package, model executor integrations and CLI entry points.

## Documentation

The external documentation lives under [`docs/`](docs/index.md). Start with:

- [Installation](docs/get-started/installation.md)
- [Quickstart](docs/get-started/quickstart.md)
- [Online Serving](docs/user-guide/online-serving.md)
- [Heterogeneous Prefill/Decode Deployment](#heterogeneous-prefilldecode-deployment)
- [CLI Reference](docs/cli-reference/pypto-serving.md)
- [Models](docs/models/index.md)

## Heterogeneous Prefill/Decode Deployment

This DeepSeek V4 Flash DSpark K5 deployment uses two ARM64 Ascend A3 hosts
with 16 NPUs each. Both model nodes use the same prepacked W8A8 checkpoint.

| Role | Host | Attention DP | Attention TP | MoE EP | LM-head TP |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill | A | 4 | 4 | 16 | 4 |
| Decode | B | 16 | 1 | 16 | 4 |
| Router | B | - | - | - | - |

Save this configuration as `pd.json` on both hosts. Replace the example IPs
with the reachable addresses of the Prefill and Decode hosts. All three
processes use the same configuration.

```json
{
  "replicas": [
    {"host": "192.0.2.10", "port": 8000, "role": "prefill"},
    {"host": "192.0.2.20", "port": 8000, "role": "decode"}
  ]
}
```

Both model nodes serve HTTP on port `8000` and Decode listens for PD control
on port `18000`. Keep those ports and the Mooncake RPC/RoCE endpoints reachable
on a trusted network. The Router accepts client traffic on port `8110`.

The published image tag `20261125-ca0ed96-lib9b2cc05-arm64` bundles the matching
PyPTO, Simpler, PTOAS and Mooncake runtime. Run the Bash setup below on both model hosts. Set the model and
configuration paths on each host and adjust the host-side Ascend mount paths
for its driver installation. The example uses a 32,768-token prompt-plus-output
limit and a 50 GB DDR prefix-cache pool per model node.

```bash
PD_IMAGE=ghcr.io/sjduan/pypto-serving-pd@sha256:d0d7f49fb37a0bb2fd763f0728e52920f220fe4d85ccbac1b551a3541c00ca20
PD_CONFIG=/absolute/path/to/pd.json
PD_MODEL_DIR=/absolute/path/to/dsv4-flash-dspark-w8a8
docker pull "$PD_IMAGE"

pd_devices=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
for device_index in {0..15}; do
  pd_devices+=(--device "/dev/davinci$device_index")
done
pd_driver_mounts=()
for driver_mount in /etc/ascend_install.info /etc/hccn.conf /usr/local/dcmi \
  /usr/local/Ascend/driver/lib64 /usr/local/Ascend/driver/version.info \
  /usr/local/Ascend/driver/tools/hccn_tool /usr/bin/hccn_tool /usr/local/bin/npu-smi; do
  pd_driver_mounts+=(--mount "type=bind,source=$driver_mount,target=$driver_mount,readonly")
done

start_pd_model() {
  local pd_role="$1"
  shift
  docker run -d --name "dsv4-$pd_role" --network host --ipc host --pid host \
    --privileged --ulimit memlock=-1:-1 "${pd_devices[@]}" "${pd_driver_mounts[@]}" \
    --mount "type=bind,source=$PD_CONFIG,target=/config/pd.json,readonly" \
    --mount "type=bind,source=$PD_MODEL_DIR,target=/models/dsv4-flash-0731-dspark-w8a8,readonly" \
    "$PD_IMAGE" pypto-serving \
    --model /models/dsv4-flash-0731-dspark-w8a8 --served-model-name dsv4 \
    --backend npu --platform a2a3 --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --ep 16 --lm-head-tp 4 --block-size 32 --max-model-len 32768 \
    --long-prefill-token-threshold 8192 --npu-memory-utilization 0.95 \
    --enable-prefix-caching --prefix-cache-retention-interval 8192 --kv-offload-bytes 50000000000 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
    --generate-config '{"max_new_tokens":131072,"temperature":0,"seed":1024}' \
    --async-scheduling --show-startup-logs --pd-config /config/pd.json \
    --pd-role "$pd_role" --host 0.0.0.0 --port 8000 "$@"
}
```

Start Prefill on host A.

```bash
start_pd_model prefill --dp 4 --tp 4 --max-num-seqs 16 --max-num-batched-tokens 32768 \
  --ring-heap 1073741824,1073741824,1073741824,4294967296
```

Start Decode and the Router on host B.

```bash
start_pd_model decode --dp 16 --tp 1 --max-num-seqs 128 --max-num-batched-tokens 768 \
  --ring-heap 1073741824,1073741824,1073741824,1073741824

docker run -d --name dsv4-router --network host \
  --mount "type=bind,source=$PD_CONFIG,target=/config/pd.json,readonly" \
  "$PD_IMAGE" pypto-serving-router --replicas /config/pd.json --host 0.0.0.0 --port 8110
```

On host B, wait for the Router's `/health` to report `"status":"ok"`, then send
a generation request. Check for generated output and the `data: [DONE]`
marker. An SSE error event indicates a failed request.

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8110/health
curl --noproxy '*' -fsSN http://127.0.0.1:8110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":32,"stream":true}'
```

See the [PD guide](https://github.com/sjduan/pypto-serving/blob/ca0ed96cf82541a0aa36ef397fa09e212071ec5e/docs/user-guide/pd-disaggregation.md) for source installs,
network configuration and additional deployment options.

## Layout

```text
pypto_serving/
  cli/                         pypto-serving CLI implementation
  config/                      runtime, generation, and parallel configuration
  serving/                     engine, scheduler, KV cache, HTTP server, workers
  model/                       loading, common runtime, Qwen, and DeepSeek integrations
  worker/                      Simpler worker wrapper for NPU dispatch
  tools/profile/               Chrome-trace profiling support
pypto-lib/                     submodule providing model-specific PyPTO kernels
scripts/
  convert_deepseek_v4_to_w8a8.py  DeepSeek V4 checkpoint converter
tests/                         host-side unit tests and CI NPU accuracy guards
```

## Quick Checks

Initialize the kernel submodule after cloning:

```bash
git submodule update --init --recursive
python -m pip install --no-deps -e .
```

Show CLI help:

```bash
pypto-serving --help
```

## NPU Generation (offline)

Offline generation runs through the same engine as serving (scheduler, worker process, KV cache) from the `pypto-serving` CLI, without opening a port:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --prompt 'Huawei is' \
  --generate-config '{"max_new_tokens": 5}'
```

DeepSeek V4 Flash W8A8 offline generation on eight devices:

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 --ep 8 \
  --max-model-len 512 \
  --long-prefill-token-threshold 2048 \
  --prompt 'Huawei is' \
  --generate-config '{"max_new_tokens": 20}' \
  --no-enable-prefix-caching \
  --num-speculative-tokens 1
```

Repeat `--prompt` for offline continuous batching. Add `--profile` to capture the generation window.

## HTTP Serving (OpenAI-compatible API)

Start the serving server with a multiprocess worker:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

Send a generation request after the server logs `Application startup complete`:

```bash
# Health check
curl --noproxy "*" http://127.0.0.1:8899/health

# Completion
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "temperature": 0.0}'

# Streaming
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Huawei is", "max_tokens": 32, "stream": true}'

# Chat completion
curl --noproxy "*" http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 1+1?"}], "max_tokens": 32}'
```

Requests that omit sampling fields use the server-wide `GenerateConfig`
defaults for every model: `temperature=0.0` (greedy decoding) and `top_p=1.0`.
Override them per request or with `--generate-config` when starting the server.

## Local Monitoring

The serving process exposes cumulative engine metrics at `/metrics` in
Prometheus text format, including `vllm:` families for existing vLLM dashboards,
and at `/metrics/json` as structured JSON. Metrics cover
request latency, token traffic and throughput inputs, scheduler queues, KV cache
usage, prefix-cache hits, request outcomes, and MTP/DSpark draft acceptance,
acceptance length, and speculation fallbacks.

For a self-contained local dashboard with SQLite history, start the companion
tool from the repository root:

```bash
python -m tools.monitor \
  --target http://127.0.0.1:8899 \
  --port 9090
```

Open <http://127.0.0.1:9090>. See
[`tools/monitor/README.md`](tools/monitor/README.md) for retention, database, and
timezone options.

## Notes

- All model/device/runtime options are passed via CLI arguments. Run `pypto-serving --help` for the exact arguments available in the installed package. See `docs/cli-reference/pypto-serving.md` for the documented reference.
- Parallel serving notes live in `docs/user-guide/parallel.md`.
- DeepSeek V4 checkpoint preparation lives in [the MTP model guide](docs/models/deepseek-v4/mtp.md#checkpoint-conversion).
- Generated kernel artifacts are written under `build_output/` and are ignored by git.
- This repository expects PyPTO, CANN, torch, safetensors, transformers, and the local Ascend runtime environment to be available in the active Python environment.
- `pypto-lib/` is not included in the wheel. An editable checkout discovers its kernel submodule automatically; for any other installation, set `PYPTO_LIB_ROOT` to the root of a `pypto-lib` checkout before loading a model.
