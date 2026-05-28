# Sprint 459: TP/EP Persistent Graph Cache Telemetry

## Objective

Make persistent CUDA graph replay explainable in HTTP serving by recording
cache hits, cache misses, invalidations, and invalidation reasons, then run a
focused V100 probe against the current TP/EP baseline.

## Rationale

Sprint 458 proved that the target-shape HTTP serving path can capture and
launch graphs for all 43 layers, but the candidate returned wrong tokens. The
next useful question was whether the failure came from stale graph reuse,
per-token recapture, or graph replay semantics inside a single decode step.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add persistent graph cache telemetry to:
  - per-layer token-major lines
  - final token-major scaffold summary
  - `tp_ep_decode_cudagraph_audit`
  - profile `summary.json`
  - HTTP A/B `ab-summary.json` and markdown
- Rebuild on the V100 node.
- Run a focused reduced-shape HTTP A/B at 8 slots / 256K / 3 generated tokens.

## Definition of Done

- Local Python syntax checks pass.
- Remote Python syntax checks pass.
- CUDA smoke binary rebuilds on gpu-01.
- A focused HTTP A/B records persistent graph cache-hit/miss/invalidation
  counters.
- The result has a promote/reject decision and an explicit next action.

## Implementation

Updated:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`

New telemetry fields include:

- `decode_cudagraph_persistent_cache_hits`
- `decode_cudagraph_persistent_cache_misses`
- `decode_cudagraph_persistent_invalidations`
- `decode_cudagraph_persistent_invalidate_layer`
- `decode_cudagraph_persistent_invalidate_slots`
- `decode_cudagraph_persistent_invalidate_position`
- `decode_cudagraph_persistent_invalidate_root_device`
- `decode_cudagraph_persistent_invalidate_root_stream`

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
```

Remote:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Both passed.

Focused V100 artifact:

```text
/localpool/ds4/workspace/logs/s459-graph-cache-s8-t3
```

Shape:

```text
8 requests / 8 slots / 256K context / 3 generated tokens/request
```

| Metric | Control | Persistent graph candidate |
|---|---:|---:|
| readiness | pass | fail |
| response parity | - | 0/8 |
| HTTP 200 | 8 | 8 |
| server generated decode tok/s | 20.032612 | 39.823731 |
| server continuation decode tok/s | 20.024299 | 39.737946 |
| client generated tok/s | 1.941361 | 0.572085 |
| request-window avg GPU util % | 11.325 | 6.181548 |
| min free VRAM MiB | 5092 | 4650 |
| output-head first token | 52762 | 123327 |
| graph capture attempted/succeeded | 0/0 | 43/43 |
| graph replay attempted/succeeded | 0/0 | 43/43 |
| persistent cache hits | 0 | 0 |
| persistent cache misses | 0 | 43 |
| persistent invalidations | 0 | 43 |
| position invalidations | 0 | 43 |
| graph instantiate ms | 0 | 268.522412 |
| graph replay ms | 0 | 201.336832 |
| graph blocker | n/a | none |

## Decision

Do not promote persistent graph HTTP serving.

The telemetry rules out stale persistent cache hits as the immediate
explanation: there were zero cache hits and every layer invalidated on position.
The candidate still changed the first token and failed response parity. That
means the current replay path is semantically unsafe even when it recaptures for
the current position.

It also shows why the current implementation is not the desired final graph
strategy. It gets a server-decode speedup by removing host sync from the layer
step, but it pays `268.5 ms` of graph instantiate time per request window and
does not reuse stable per-layer graphs across token positions.

## Follow-Up

The next graph sprint should not try to promote persistent replay as-is. It
should first split the problem:

- Validate graph-safe event ordering without replay.
- Add a capture/no-replay or replay-parity diagnostic that cannot silently
  return graph-corrupted responses.
- Move position/KV-row/route metadata to device-resident dynamic inputs before
  attempting stable persistent graph reuse.

A second no-replay A/B was started but aborted after another benchmark appeared
on the node, which would have polluted the comparison. Only exact PIDs from this
sprint's second A/B were stopped; the completed `s459-graph-cache-s8-t3` result
is unaffected.
