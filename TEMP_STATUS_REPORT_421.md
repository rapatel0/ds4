# TEMP Status Report 421

Date: 2026-05-27

## Focus

Follow up Sprint 416's positive direct-decode rank-local attention projection
result with the first HTTP serving A/B.

This stayed TP/EP-only. No PP/layer-split work.

## Shape

```text
endpoint=selected-token
requests=8
slots=8
ctx=262144
position=262080
tokens/request=8
hc-current-nccl=on
persistent graph replay=on
defer-nccl-init=on
tp-runtime-scratch=256 MiB
skip unused comp-state=on
model-router routes=on
compact MoE=on
```

The run used the permanent HTTP profile harness from inside the CUDA container
after installing `python3` into the image for the harness process.

## Artifacts

Control:

```text
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/control-selected-slot8-token8-container/
```

Rank-local:

```text
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/ranklocal-selected-slot8-token8-container/
```

## Results

### 8-slot selected-token A/B

| Metric | Control | Rank-local | Delta |
|---|---:|---:|---:|
| HTTP 200 | 8/8 | 8/8 | same |
| First token | 45124 | 45124 | same |
| Client generated tok/s | 22.180780 | 24.225369 | +9.22% |
| Status generated decode tok/s | 88.402819 | 100.059560 | +13.18% |
| Status continuation decode tok/s | 94.811395 | 107.260053 | +13.13% |
| Status generated wall tok/s | 22.504843 | 24.604307 | +9.33% |
| Status continuation wall tok/s | 65.887682 | 72.105327 | +9.44% |
| Scaffold projected slot-step tok/s | 94.272991 | 108.583858 | +15.18% |
| Compressed-KV parsed sum ms | 39.763711 | 34.757769 | -12.59% |
| Avg sampled GPU util | 6.95% | 6.85% | flat |
| Max sampled GPU util | 44% | 41% | flat |
| Min free VRAM | 6886 MiB | 6886 MiB | same |
| NCCL reserve failures | 0 | 0 | same |

### 28-slot selected-token follow-up

Control artifact:

```text
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/control-selected-slot28-token8-container/
```

Control result:

```text
HTTP 200: 28/28
first token: 45124
client generated tok/s: 52.690208
status generated decode tok/s: 129.750653
status continuation decode tok/s: 131.436685
avg sampled GPU util: 16.328125%
max sampled GPU util: 58%
min free VRAM: 4570 MiB
NCCL reserve failures: 0
```

Initial rank-local artifact:

```text
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/ranklocal-selected-slot28-token8-container/
```

Initial rank-local result: inconclusive. The server process was terminated
during readiness before emitting model output:

```text
RuntimeError: server exited rc=-15
server.out: empty
server.err: empty
```

This should be rerun. Do not count it as a model/kernel regression.

Retry rank-local artifact:

```text
/localpool/ds4/workspace/logs/sprint421-ranklocal-http/ranklocal-selected-slot28-token8-retry-container/
```

Retry rank-local result:

```text
HTTP 200: 28/28
first token: 45124
client generated tok/s: 59.592035
status generated decode tok/s: 158.385152
status continuation decode tok/s: 162.101543
avg sampled GPU util: 13.696429%
max sampled GPU util: 52%
min free VRAM: 4570 MiB
NCCL reserve failures: 0
```

## Decision

Rank-local attention projection input survives selected-token HTTP serving
checks at both 8 and 28 slots.

It is still not fully promoted as the production default because these are
selected-token checks. The next promotion step is chat/readiness/parity.

## Notes

- The first-token match is strong evidence that the rank-local path preserves
  the current selected-token behavior for this shape.
- GPU utilization remains low. Rank-local improves latency/throughput by
  removing local-layout copy work, but it does not solve the larger hardware
  utilization problem.
- The HTTP harness should run in a CUDA container with Python available. Host
  Python can drive the harness, but the server binary needs `libcudart.so.12`
  from the CUDA image.

## Next

1. Run chat/readiness/parity at the practical long-context tier.
2. Retry `32` slots / `256K` after expert residency/headroom work.
4. Add a reusable Python-capable CUDA profile container or avoid per-run
   `apt-get install python3`.
5. Start the expert-residency headroom sprint:
   - planner report for full expert residency
   - staged/lazy expert load option
   - admission check for scratch size and target slots
