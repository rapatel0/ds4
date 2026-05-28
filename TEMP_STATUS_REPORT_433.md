# TEMP Status Report 433

## Focus

TP/EP only. Tested whether reading actual post-attention route counts from the
GPU can replace fixed-capacity routed FFN execution.

## Added

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` now has:

```text
--post-attention-device-actual-route-sync-gate
```

This is diagnostic only. It synchronizes after the GPU route planner, copies
the eight per-rank `d_route_totals` values to host, and launches routed FFN with
actual route counts. It rejects graph-event mode because that host readback is
not capturable.

## V100 Results

Build passed on `gpu-01` with `sm_70`.

Direct actual-route sync, 8 slots / 256K:

```text
log: /localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-actual-sync-slots8-skipstats.stdout
PASS
projected_slot_step_tok_s: 17.260141
aggregate_routes: 48
ep_return_bytes: 688128
```

Direct current fixed route-plan, same binary:

```text
log: /localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-fixed-current-slots8-skipstats.stdout
PASS
projected_slot_step_tok_s: 17.371569
aggregate_routes: 48
ep_return_bytes: 688128
```

Persistent graph replay current fixed route-plan:

```text
log: /localpool/ds4/workspace/logs/sprint432-device-actual-route-sync/alllayers-fixed-current-slots8-graph.stdout
PASS
projected_slot_step_tok_s: 39.491776
aggregate_routes: 384
ep_return_bytes: 5505024
```

## Interpretation

Host-synchronized actual-route execution is not the serving fix. It proves the
route-count plumbing, but direct mode remains slower than graph replay.

The next optimization must be graph-safe: the routed FFN executor has to keep a
static captured launch while consuming device route totals or masks internally
so inactive fixed-capacity rows stop doing useful work.

No GPU jobs were left running after the probes.
