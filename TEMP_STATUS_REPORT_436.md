# TEMP Status Report 436

## Focus

TP/EP only. Tested whether capping only the routed FFN executor rows, while
keeping the full graph transfer/compose envelope, is correctness-safe.

## Added

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` now accepts:

```text
--post-attention-static-executor-route-cap N
```

This is diagnostic and default-off.

## V100 Result

Same-binary A/B:

```text
8 slots / 256K / all layers / persistent graph / output-head diagnostic

full cap:    38.765556 tok/s, first_token=50845
exec cap 16: 38.786726 tok/s, first_token=7518
```

The executor-cap run kept `aggregate_routes=384` and
`ep_return_bytes=5505024`, so the failure is not from reducing the peer-copy
or compose envelope.

## Decision

Reject executor-only static caps. TurboMind grouped GEMM `total_tokens` /
`Ddesc.rows` must remain full for correctness.

## Next

Implement the actual-route skip inside a full-shape executor:

- keep host-visible launch dimensions fixed;
- pass device route totals/mask into the executor;
- early-return inactive CTAs/rows internally;
- prove output-head token parity before treating throughput as real.

No GPU jobs were left running after the tests.
