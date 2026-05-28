# Sprint 436: Executor-Only Static Route Cap

## Objective

Test whether Sprint 434 failed because it reduced the whole route envelope, or
because TurboMind grouped GEMM itself is sensitive to `total_tokens`.

No PP/layer-split variants are in scope.

## Implementation

Added a default-off diagnostic:

```text
--post-attention-static-executor-route-cap N
```

Unlike `--post-attention-static-rank-route-cap`, this keeps `r.routes` and the
compose/transfer envelope at full fixed capacity. It only caps the row count
passed to the routed gate/down TurboMind executor.

## V100 Evidence

Logs:

```text
/localpool/ds4/workspace/logs/sprint436-executor-cap/fullcap-slots8-graph-head.stdout
/localpool/ds4/workspace/logs/sprint436-executor-cap/execcap16-slots8-graph-head.stdout
```

Shape:

```text
8 slots / 256K / 43 layers / persistent graph replay / lazy output head
```

Result:

```text
full cap:    projected_slot_step_tok_s=38.765556, first_token=50845, first_logit=18.315456390
exec cap 16: projected_slot_step_tok_s=38.786726, first_token=7518,  first_logit=17.357364655
```

The executor cap preserved the route-transfer envelope:

```text
aggregate_routes=384
ep_return_bytes=5505024
```

## Decision

Reject executor-only static route caps.

This test rules out the simpler explanation that Sprint 434 only failed because
the compact route transfer/compose envelope changed. Even when the envelope
stays full, changing the TurboMind grouped GEMM `total_tokens` shape changes
the selected output token.

## Next

The production route must keep the host-visible TurboMind launch shape fixed:

- `total_tokens = route_capacity`;
- `Ddesc.rows = route_capacity`;
- route-transfer/compose bytes remain fixed for graph capture;
- inactive-route skipping happens inside a DS4-specific full-shape executor,
  not by changing the host launch row count.

The next implementation should add a TurboMind/DS4 internal route mask or a
dedicated routed FFN kernel that receives device route totals while preserving
the captured launch shape.
