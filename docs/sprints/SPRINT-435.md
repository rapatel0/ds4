# Sprint 435: Static Route-Cap Output Parity

## Objective

Resolve whether Sprint 434's static route-cap speedup is a real serving
candidate by comparing output-head selected tokens, not only graph checksums.

No PP/layer-split variants are in scope.

## V100 Evidence

Logs:

```text
/localpool/ds4/workspace/logs/sprint435-output-head-parity/fullcap-slots8-graph-head.stdout
/localpool/ds4/workspace/logs/sprint435-output-head-parity/cap16-slots8-graph-head.stdout
```

Both runs used:

```text
8 slots / 256K / all 43 layers / persistent graph replay
--post-attention-fixed-capacity-route-plan-gate
--diagnostic-output-head-lazy-gate
```

The cap16 run additionally used:

```text
--post-attention-static-rank-route-cap 16
```

Result:

```text
full cap: projected_slot_step_tok_s=36.846896, first_token=50845, first_logit=18.253084183
cap 16:   projected_slot_step_tok_s=50.408429, first_token=106720, first_logit=17.242784500
```

The cap16 route audit stayed overflow-free through layer 42:

```text
tp_ep_static_route_cap_audit_total layer 42 cap 16 max_actual 9 overflow_ranks 0 PASS
```

## Decision

Reject static route caps as a serving strategy.

The speedup is meaningful, but the selected token changes. This means the
executor cannot safely reduce the host-visible grouped-GEMM row count, even
when the actual route totals fit under the cap.

## Next

Implement the actual-route routed FFN executor with a static graph envelope:

- keep host `total_tokens = route_capacity`;
- keep graph capture and launch dimensions stable;
- pass device route totals or a route mask into a DS4-specific TurboMind/CUTLASS
  path;
- skip inactive rows inside the executor before useful MMA/output contribution;
- require output-head token parity before reading throughput as promotable.
