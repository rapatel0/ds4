# Sprint 437: Compose-Only Static Route Cap

## Objective

Test whether the Sprint 434 speedup came from reducing compact route
transfer/compose bytes while keeping the TurboMind grouped GEMM launch shape
unchanged.

No PP/layer-split variants are in scope.

## Implementation

Added a default-off diagnostic:

```text
--post-attention-static-compose-route-cap N
```

This keeps routed FFN executor rows at full fixed capacity and caps only the
compact route pack/copy envelope used during composition.

## V100 Evidence

Logs:

```text
/localpool/ds4/workspace/logs/sprint436-executor-cap/fullcap-slots8-graph-head.stdout
/localpool/ds4/workspace/logs/sprint437-compose-cap/composecap16-slots8-graph-head.stdout
```

Shape:

```text
8 slots / 256K / 43 layers / persistent graph replay / lazy output head
```

Result:

```text
full cap:       projected_slot_step_tok_s=38.765556, first_token=50845, first_logit=18.315456390
compose cap 16: projected_slot_step_tok_s=49.381819, first_token=164,   first_logit=19.294666290
```

The compose cap audit was overflow-free:

```text
layer 40: cap=16, max_actual=10, overflow_ranks=0, PASS
layer 41: cap=16, max_actual=10, overflow_ranks=0, PASS
layer 42: cap=16, max_actual=10, overflow_ranks=0, PASS
```

## Decision

Reject compose-only static caps.

This confirms that the speedup is mostly in route transfer/compose volume, but
the current compact graph path is semantically sensitive to the fixed segment
shape. Host-side caps are not safe even when the cap covers actual route totals.

## Next

The next optimization must preserve the host-visible graph envelope and move
the active-route decision into the device-side compose path:

- keep copy/segment shape fixed for graph capture;
- use device route indices/counts to avoid useful work on inactive rows;
- do not change TurboMind `total_tokens` or compact segment copy length until a
  token-parity proof exists;
- measure whether internal masking changes useful work, not just host-visible
  byte counts.
