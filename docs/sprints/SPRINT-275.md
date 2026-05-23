# Sprint 275 - TP/EP Sustained Serving Artifact Wrapper

Date: 2026-05-23

## Goal

Create a repeatable sustained-serving artifact wrapper for the resident TP/EP
backend, without returning to PP/layer-split work.

## Rationale

Sprint 274 removed the hot per-token/per-layer scaffold from `--serving-bench`
and produced useful serving-shaped wall metrics. Before integrating the backend
with the HTTP appliance server, we need a stable artifact format that captures
generated and continuation throughput, token-match status, stdout/stderr, and
the exact promoted runtime settings.

This also follows the current steering: finish TP/EP end-to-end operational
serving first, then return to individual kernel optimization and MTP.

## Implementation

Added `tools/ds4-v100-tp-ep-sustained-bench.sh`.

The wrapper runs `tools/ds4-v100-tp-ep-full-layer-smoke` with:

- `--serving-bench`
- `--token-major-all-layers`
- `--all-layers`
- `--shared-expert-bindings`
- `--shared-dense-ops`
- `--overlap-ep-dense`
- `--source-copy-schedule`
- `--skip-self-compose-copy`
- `--multi-copy-streams`
- dense FP16 cache compose enabled

It writes:

- `sustained_decode.tsv`
- `sustained_decode.json`
- `cases/tp-ep-resident/result.json`
- `cases/tp-ep-resident/stdout.log`
- `cases/tp-ep-resident/stderr.log`

The wrapper currently fails closed for any context other than `262144`, because
the underlying resident TP/EP smoke is the 256K target harness.

## Cluster Result

Command shape:

```text
slots=32
ctx=262144
tokens_per_request=32
position=80000
top_k=6
kv_slot=7
```

Topline:

| Metric | Value |
|---|---:|
| Generated tokens | 1024 |
| Continuation tokens | 992 |
| Token match | 32/32 |
| Wall generated tok/s | 749.304439 |
| Wall continuation tok/s | 774.209856 |
| Decode-only generated tok/s | 963.264018 |
| Decode-only continuation tok/s | 1000.823072 |
| Total wall time | 1.366601 s |
| Total decode time | 1.063052 s |

Stage summary from the resident scaffold row:

| Stage | Time |
|---|---:|
| Sum decode | 1063.052269 ms |
| Sum EP | 456.109556 ms |
| Sum compose | 606.643757 ms |
| Compose reduce | 87.618525 ms |
| Compose copy | 455.241728 ms |
| Compose final | 63.783504 ms |

## Evidence

Cluster artifacts are saved under:

```text
logs/from-cluster/sprint275-tp-ep-sustained-bench/cluster/
```

Primary files:

- `sustained_decode.tsv`
- `sustained_decode.json`
- `cases/tp-ep-resident/result.json`
- `cases/tp-ep-resident/stdout.log`
- `cases/tp-ep-resident/stderr.log`

## Decision

Promote the wrapper as the current TP/EP sustained-serving artifact producer.
Do not treat it as the final appliance server. The next sprint should wire the
same resident backend into the operational HTTP sustained-decode path so request
handling, health/status, and serving metrology use the actual appliance surface.

## Next

Build the HTTP TP/EP serving bridge:

- Start a TP/EP-only appliance server path.
- Load the resident TP/EP backend once.
- Admit `32` slots at `256K`.
- Expose status showing TP/EP backend readiness and warmed resident state.
- Serve a sustained-decode request matrix through HTTP.
- Report prompt, generated, continuation, wall, and decode timing separately.
