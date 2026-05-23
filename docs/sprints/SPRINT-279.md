# Sprint 279 - TP/EP Deployment Defaults And GPU Utilization

Date: 2026-05-23

## Goal

Make the Kubernetes appliance example point at the TP/EP serving path and add
GPU-utilization capture to the sustained HTTP matrix.

## Implementation

Updated `tools/ds4-v100-run-appliance.sh`.

- Added `DS4_V100_ALLOW_NONLOCAL_HOST`.
- Kept loopback bind as the default.
- Requires `DS4_V100_ALLOW_NONLOCAL_HOST=1` before binding to a non-loopback
  address such as `0.0.0.0`.
- Records the bind gate in `startup.env`.

Updated `deploy/v100/ds4-v100-appliance.k8s.yaml`.

- Sets `DS4_V100_SERVE_MODE=tp-ep`.
- Uses the resident TP/EP server binary
  `./tools/ds4-v100-tp-ep-full-layer-smoke`.
- Uses the current production TP/EP pack and contract paths from the V100
  workspace.
- Sets the serving shape to `32` slots / `256K` context /
  `32` active microbatch.
- Uses `/localpool/ds4/workspace` for the writable workspace.
- Uses the existing `llm-models-local` PVC for `/models`.
- Enables non-loopback bind explicitly for the Kubernetes service.

Updated `tools/ds4-v100-tp-ep-http-bench.sh`.

- Samples `nvidia-smi` during the generation POST.
- Writes per-case `gpu_util.csv`.
- Adds `gpu_util_avg`, `gpu_util_max`, and `gpu_mem_used_max_mib` to
  `sustained_http.tsv` and per-case `result.json`.

## Validation

Local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-tp-ep-http-bench.sh
ruby -e 'require "yaml"; YAML.load_stream(File.read("deploy/v100/ds4-v100-appliance.k8s.yaml")); puts "yaml ok"'
kubectl apply --dry-run=client -f deploy/v100/ds4-v100-appliance.k8s.yaml
DS4_V100_SERVE_MODE=tp-ep DS4_V100_ALLOW_NONLOCAL_HOST=1 DS4_V100_HOST=0.0.0.0 ./tools/ds4-v100-run-appliance.sh --check --allow-missing
```

Cluster validation on `llm/llamacpp-build-8gpu`:

```text
ctx=262144
slots=32
active_microbatch=32
serve_mode=tp-ep
tokens_cases=32,64
```

Results:

| Tokens/request | Generated tokens | Continuation tokens | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Decode continuation tok/s | Avg GPU util | Max GPU util | Max GPU mem MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 992 | 745.699174 | 771.902910 | 961.190833 | 1000.254964 | 15.520833 | 38.000000 | 13184 |
| 64 | 2048 | 2016 | 753.708353 | 766.803086 | 976.515582 | 996.465789 | 18.537500 | 40.000000 | 13184 |

Both cases returned `32/32` token match.

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint279-tp-ep-k8s-gpuutil/cluster/
```

Primary files:

- `sustained_http.tsv`
- `sustained_http.json`
- `cases/case_0_ctx262144_s32_tok32/gpu_util.csv`
- `cases/case_1_ctx262144_s32_tok64/gpu_util.csv`
- Per-case `response.json`, `result.json`, `status_before.json`, `metrics.txt`,
  `server.log`, and `server.err`.

## Decision

The TP/EP path now has Kubernetes defaults and HTTP-level GPU-utilization
metrology. The deployment manifest has been validated with client-side
dry-run, but it was not applied because the build pod is currently the active
8-GPU test owner.

The GPU-utilization data confirms that the TP/EP server is doing GPU work
during the request, but utilization remains low for the operational goal:
`38-40%` max and `15-19%` average during these short synthetic requests. The
next useful work should move the serving harness closer to practical continuous
batching and reduce the compose/copy overhead already identified in the
resident loop.

## Next

- Apply the Kubernetes deployment when the 8-GPU build pod is quiesced or when
  scheduling is otherwise explicit.
- Add a continuous request/coalescing serving path so multiple HTTP requests
  can feed a resident 32-slot microbatch instead of only one synthetic
  selected-token batch per POST.
- Keep tracking generated and continuation tok/s separately, with GPU
  utilization, at `32` slots / `256K`.
- Return to compose-copy and fused dense/compose kernel optimization after the
  operational serving loop exposes stable metrology.
