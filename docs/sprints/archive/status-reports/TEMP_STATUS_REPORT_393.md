# TEMP Status Report 393: TP/EP Serving Readiness Gate

Date: 2026-05-25

## Topline

Sprint 393 added a permanent HTTP serving readiness checker:

```text
tools/ds4-v100-http-readiness-check.py
```

This does not claim DS4 is production-complete. It makes the current TP/EP
serving artifact contract executable so future performance changes cannot
silently weaken correctness, KV residency, VRAM admission, prompt soak
coverage, or response metadata.

## Validation

Syntax:

```text
python3 -m py_compile tools/ds4-v100-http-readiness-check.py
```

Sprint 392 control artifact:

| Metric | Value |
|---|---:|
| Ready | `true` |
| Failure count | `0` |
| HTTP 200 | `32` |
| Tokens/request | `32` |
| Slots/context | `32 / 256K` |
| Prompt count | `16` |
| Server decode tok/s | `106.390802` |
| Client generated tok/s | `38.912861` |
| Avg GPU util | `9.772727%` |
| Max GPU util | `52%` |
| First token | `83484` |
| VRAM failures | `0` |
| Min free VRAM | `1746 MiB` |

Sprint 392 E5M2 candidate artifact:

| Metric | Value |
|---|---:|
| Ready | `true` |
| Failure count | `0` |
| HTTP 200 | `32` |
| Tokens/request | `32` |
| Slots/context | `32 / 256K` |
| Prompt count | `16` |
| Server decode tok/s | `106.483285` |
| Client generated tok/s | `39.774181` |
| Avg GPU util | `9.691860%` |
| Max GPU util | `50%` |
| First token | `83484` |
| VRAM failures | `0` |
| Min free VRAM | `1746 MiB` |

Negative fixture:

```text
ready=false
failure_count=2
failed_checks=["token_match", "checksum"]
```

## Current Assessment

The target-shape TP/EP serving harness is operational enough to gate future
performance work: multi-prompt HTTP requests complete, KV/HC state is resident,
typed KV metadata is present, VRAM admission passes, and response checksums are
available.

The performance problem remains unchanged: the current real-router compact-MoE
path is still around `106` server decode tok/s with roughly `10%` average GPU
utilization at `32` requests / `32` slots / `256K` / `32` generated tokens.
That supports the existing throughput thesis that the bottleneck is
steady-state launch/synchronization and GPU0-heavy orchestration, not slot
admission or KV capacity.

## Next

Use this readiness checker with the existing response parity comparator for
the next TP/EP performance sprint. The next implementation should target a
real measured boundary, likely broader route/router/HC-current scheduling or
another default-off serving-shaped kernel boundary, and must attach readiness
plus parity summaries before any promotion.
