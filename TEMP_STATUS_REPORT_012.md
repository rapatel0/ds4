# TEMP Status Report 012

Date: 2026-05-23

## Current Topline

Sprint 198 made CUDA graph replay compatible with the current
`fused6_reduce` routed executor. This is a direct-replay improvement only; it
is not a serving promotion yet.

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Output IDs | Graph evidence |
|---|---:|---:|---:|---|---|
| graph off | `4.475985` | `3.568240` | `16.022442` | `201,200,84921,200,18,90,926,14` | n/a |
| graph on | `4.679220` | `3.780181` | `17.980888` | `201,200,84921,200,18,90,926,14` | `43` captures, `129` launches, `0` failures |

## Interpretation

Graph replay now measures the current fused routed path instead of being
blocked by old executor gating. The `+12.2%` continuation signal is useful, but
Sprint 169 already showed direct replay can improve while served throughput
regresses. Treat this as a candidate, not a default.

## Tensor-Parallel State

TP has been investigated as a bounded implementation direction, not shipped as
production inference. The estimator shows the current layer split moves about
`7.000 MiB` per token at the 16-slot/256K tier, while full TP/EP topologies move
more wire bytes: `75.250 MiB` for TP2/PP1, `112.875 MiB` for TP4/PP1, and
`131.688 MiB` for TP8/PP1. That traffic is only worth paying if the whole layer
becomes native to the topology.

The TP4 primitive tests are correct on the V100 NVLink islands:

| Collective | 16-token decode payload | 1024-token payload | Use |
|---|---:|---:|---|
| root gather/reduce/broadcast | `0.110762 ms` | `3.675847 ms` | correctness floor |
| recursive doubling | `0.133761 ms` | `1.655687 ms` | better batched/prefill primitive |

This does not reject tensor parallelism. It rejects a naive production TP4
decode path built only around full-hidden collectives. TP should be revisited as
a fused full-layer or persistent TP/EP boundary where the denser GEMM shape pays
for the extra collectives.

## Next Gate

Sprint 199 should run a same-binary served 16-slot/256K A/B:

- control: `DS4_V100_TURBOMIND_GRAPH=0`
- candidate: `DS4_V100_TURBOMIND_GRAPH=1`
- both with `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce`
- record prompt, generated, continuation tok/s separately
- require `16/16` token match and graph capture/launch/failure counts

If served mode is positive, graph replay becomes the first practical
execution-boundary optimization candidate. If it regresses again, stop graph
work and implement either a true persistent/tile-level routed FFN executor or a
bounded full-layer TP4/PP1 prototype.
