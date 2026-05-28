# Sprint 442: TP/EP Routed Executor Upper-Bound Probe

## Objective

Stay on the TP/EP path only and determine whether a graph-safe active-route
routed-FFN executor is still a material lever.

Sprint 441 showed that masking inactive compact-copy rows is HTTP-correct but
does not improve serving throughput. The next hypothesis is execution masking:
keep fixed graph-captured host shapes while skipping inactive route rows inside
the routed FFN executor.

## Implementation Plan

1. Inspect the TurboMind SM70 grouped-GEMM scheduler and the TP/EP fixed-capacity
   route-plan call path.
2. Measure the existing upper bound with:
   - fixed-capacity post-attention route planning, graph-safe shape;
   - device-actual route sync, not graph-safe but a useful upper-bound control.
3. If device-actual route sync is materially faster, add a default-off
   graph-safe executor probe that preserves host launch shape and uses
   device-resident route totals/offsets to suppress inactive work internally.
4. If device-actual route sync is flat, stop pursuing executor-row masking and
   move the vision to the next real bottleneck.

## Decision Gate

Promote executor work only if the non-graph-safe actual-route upper bound is at
least 5% faster than fixed-capacity routing at the same 8-slot, 256K TP/EP
configuration.

## Validation

- Build on gpu-01 using sm_70.
- Run 8-slot, 256K, TP/EP direct decode profiles for fixed-capacity and
  actual-route-sync.
- Keep correctness/parity evidence where output-head is enabled.
- Record gate/down/EP timings and the top-line generated decode tok/s.

## Out Of Scope

- PP/layer-split variants.
- MTP.
- Kernel fusion that is not directly needed to prove or reject executor-row
  masking.

## V100 Evidence

HTTP chat A/B artifacts:

```text
/localpool/ds4/workspace/logs/s442-ab
```

Shape:

```text
8 requests / 8 slots / 256K ctx / 2 generated tokens/request
```

Result:

```text
control ready: true
candidate ready: true
response parity: 8/8 matched

server generated decode tok/s:    14.080773 -> 13.885178
server continuation decode tok/s: 14.129698 -> 13.895648
client generated tok/s:            1.205039 ->  1.210999
avg GPU util:                       8.4891% ->  8.5707%
min free VRAM:                      4674 MiB both
```

## Outcome

Reject actual-route-sync as a promotion path.

The HTTP result is parity-clean and readiness-clean, but it is slower on server
decode and continuation throughput. This matches the earlier direct diagnostic:
host/device actual-route synchronization is not the missing performance lever.

## Next

Do not build a host-synchronized actual-route serving path.

If executor-row masking is pursued, it must be a true full-shape executor:

- fixed host-visible graph launch dimensions;
- device-resident active-route masks/totals;
- no host readback or route-count synchronization;
- no selected-token or response-sequence drift.

## Outcome

Added permanent default-off serving/profile/Harness plumbing for:

```text
DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC
--post-attention-device-actual-route-sync
--candidate-post-attention-device-actual-route-sync
```

This mode forces fixed-capacity post-attention route planning, then synchronizes
device route totals to host and launches only actual routed rows. It is not
CUDA-graph safe and is only an upper-bound diagnostic.

## V100 Evidence

HTTP A/B:

```text
artifact: /localpool/ds4/workspace/logs/s442-ab
shape: 8 requests, 8 slots, 256K ctx, 2 generated tokens
control: fixed-capacity route planner + HC-current NCCL
candidate: control + actual-route-sync
readiness: control true, candidate true
parity: 8/8 matched
first token: 72960 both legs
server generated decode tok/s: 14.080773 -> 13.885178
server continuation decode tok/s: 14.129698 -> 13.895648
client generated tok/s: 1.205039 -> 1.210999
avg GPU util: 8.489% -> 8.571%
HC-current input ms: 384.282571 -> 389.666722
```

No stale GPU compute processes remained after the run.

## Decision

Reject actual-route-sync as a promotion path and do not implement the graph-safe
active-route executor next.

The upper bound is flat/slower in the real HTTP serving path. The routed
executor is not the next material bottleneck under this configuration; the
profile points back to HC-current/post-attention staging and route/router upload
work.
