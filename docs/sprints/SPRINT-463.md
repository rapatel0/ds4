# Sprint 463: TP/EP Startup Metrology and Parallel Expert Load

## Objective

Make the TP/EP serving profile distinguish startup from steady-state serving,
then address the observed serial expert-load pattern without touching the
PP/layer-split path.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add lifecycle and dmon-based GPU telemetry to the profile harness.
- Add an opt-in parallel expert residency loader.
- Validate on the V100 node at the 32-slot / 256K target shape.

## Implementation

Added profile lifecycle events:

```text
process_start
server_spawned
server_ready
requests_start
responses_complete
status_metrics_complete
summary_written
```

Added dmon-based GPU sampling as the default profile sampler and retained the
old query sampler behind:

```text
--gpu-sampler query
```

Added opt-in parallel expert loading:

```text
--parallel-expert-load-gate
DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=1
```

The loader keeps layer order sequential but loads all 8 GPU expert bindings
concurrently inside each layer.

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check -- tools/ds4-v100-tp-ep-full-layer-smoke.cu tools/ds4-v100-run-appliance.sh tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
```

Remote build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Clean 8-slot smoke:

```text
/localpool/ds4/workspace/logs/s460-parallel-expert-load-s8-t3-clean
```

| Metric | Result |
|---|---:|
| HTTP responses | 8/8 |
| readiness elapsed | 53.110685 s |
| request elapsed | 11.508914 s |
| server generated decode tok/s | 20.688739 |
| request-window avg GPU util | 11.867925% |
| min free VRAM | 5092 MiB |
| VRAM failures | 0 |

Target-shape long run:

```text
/localpool/ds4/workspace/logs/s460-parallel-expert-load-s32-t32-long
```

Shape:

```text
32 requests / 32 slots / 256K context / 32 generated tokens
```

| Metric | Result |
|---|---:|
| HTTP responses | 32/32 |
| generated tokens | 1024 |
| readiness elapsed | 106.215634 s |
| request elapsed | 67.038542 s |
| server generated decode tok/s | 35.813083 |
| server continuation decode tok/s | 35.815690 |
| client generated tok/s | 15.274795 |
| full-run avg GPU util | 5.246531% |
| startup avg GPU util | 1.729430% |
| request-window avg GPU util | 12.534426% |
| request-window max GPU util | 32.000000% |
| moving-average peak GPU util | 16.000000% |
| min free VRAM | 1734 MiB |
| VRAM failures | 0 |

## Decision

Keep the parallel expert-load gate as a validated opt-in path. It removes the
old sequential round-robin load behavior and materially improves benchmark
iteration time, but it does not change the steady-state serving bottleneck.

Do not promote it as a throughput optimization.

## Next Action

Continue runtime optimization on TP/EP rank-major execution:

1. Reduce HC-current input/staging movement.
2. Continue routed FFN/EP optimization around the real 32-slot shape.
3. Use the new lifecycle and request-window dmon metrics for every benchmark so
   initialization does not pollute serving throughput conclusions.
