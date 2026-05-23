# TEMP Status Report 013

Date: 2026-05-23

## Sprint 199 Result

The served graph replay gate passed and the V100 production appliance defaults
were promoted for the Sprint 181+ pack.

Default production stack after Sprint 199:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce
DS4_V100_TURBOMIND_GRAPH=1
```

## V100 Served Data

16-slot/256K, 16 requests x 64 generated tokens, per-step async with event
handoff:

| Mode | Prompt tok/s | Generated tok/s | Continuation tok/s | Match | Avg GPU util | Max GPU util |
|---|---:|---:|---:|---:|---:|---:|
| routed executor off, graph off | `15.952247` | `56.719099` | `55.832863` | `16/16` | `33.471%` | `64%` |
| `fused6_reduce`, graph off | `15.391536` | `54.725463` | `53.870377` | `16/16` | `34.098%` | `66%` |
| `fused6_reduce`, graph on | `19.093013` | `67.886268` | `66.825545` | `16/16` | `39.148%` | `69%` |

Graph evidence:

```text
turbomind_graph captured:       43
turbomind_graph launched:       129
turbomind_graph launch_failed:  0
turbomind_graph capture_failed: 0
begin_capture_failed:           0
```

## Interpretation

This is a real production-serving improvement, not just a direct replay result.
The promoted stack is about `+19.7%` continuation tok/s versus the
routed-executor-off production control in this harness.

It still does not realize the throughput vision. The topline is now roughly
`67` aggregate continuation tok/s at 16-slot/256K, while the practical target is
at least several hundred tok/s. The next work should use this promoted baseline
and move to a larger execution shape:

- persistent/tile-level routed FFN kernel, or
- bounded full-layer TP4/EP prototype that makes dense and routed work native
  to the topology.

More wrapper-level graph or staging experiments are unlikely to close the
remaining gap.
