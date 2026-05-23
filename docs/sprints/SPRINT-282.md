# Sprint 282 - TP/EP Event-Wait Compose Copy

Date: 2026-05-23

## Goal

Reduce TP/EP compose-copy host synchronization by letting destination compose
streams wait on peer-copy events instead of forcing a global host-side copy
stream synchronization before final compose.

## Implementation

Updated `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.

- Added `--copy-event-compose`.
- Added per-source/per-destination copy completion events.
- In the source-scheduled copy path, records a CUDA event after each peer copy.
- Destination streams wait on their incoming copy events before launching final
  compose.
- This keeps correctness dependencies on-device and avoids a host barrier
  between copy submission and final compose launch.

Updated launcher/config/bench wiring.

- Added `DS4_V100_TP_EP_COPY_EVENT_COMPOSE`.
- Promoted the event-wait path to the appliance default.
- Added `--copy-event-compose` and `--no-copy-event-compose` to the HTTP
  matrix driver so A/B runs remain explicit.
- Added the promoted default to the Kubernetes example.

## Validation

Local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
bash -n tools/ds4-v100-tp-ep-http-bench.sh
ruby -e 'require "yaml"; YAML.load_stream(File.read("deploy/v100/ds4-v100-appliance.k8s.yaml")); puts "yaml ok"'
kubectl apply --dry-run=client -f deploy/v100/ds4-v100-appliance.k8s.yaml
git diff --check
```

V100 pod validation:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Same-binary 64-token A/B at `32` slots / `256K` / three generation requests:

| Mode | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Compose ms | Compose copy ms | Compose final ms | Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 752.669235 | 757.403683 | 977.373068 | 3670.569130 | 2734.135125 | 393.399271 | 96/96 |
| copy-event-compose | 771.276064 | 775.670776 | 995.388776 | 3585.726648 | 1741.603085 | 1310.021028 | 96/96 |

Event-wait uplift on the same binary:

```text
wall generated tok/s:      +2.47%
wall continuation tok/s:   +2.41%
decode generated tok/s:    +1.84%
decode continuation tok/s: +1.77%
```

32-token event-wait sanity at `32` slots / `256K` / three generation requests:

| Tokens/request | Wall generated tok/s | Wall continuation tok/s | Decode generated tok/s | Compose ms | Compose copy ms | Match |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 757.105839 | 766.551144 | 982.810892 | 1803.836959 | 891.831766 | 96/96 |

## Evidence

Cluster artifacts:

```text
logs/from-cluster/sprint282-tp-ep-copy-event-compose/
```

Subdirectories:

- `control64/cluster/`
- `event64/cluster/`
- `event32/cluster/`

Each contains `sustained_http.tsv`, `sustained_http.json`, per-request
responses, `status_after.json`, `metrics.txt`, GPU utilization, and server
logs.

## Decision

Promote `copy-event-compose` as the TP/EP appliance default. It is correct in
the serving-shaped HTTP path and improves same-binary 64-token wall throughput
by about `2.5%`.

The stage split should be interpreted carefully: copy wait time moves from the
explicit copy bucket into the final compose bucket because the host no longer
synchronizes all copy streams before launching final compose. Total throughput
and total compose time are the decision metrics.

## Next

- Keep the event-wait path enabled for TP/EP serving matrices.
- Continue optimizing compose movement, likely by reducing the amount of
  copied FP32 contribution data or replacing the staged all-to-all with a
  more direct fused reduction strategy.
- Then add true HTTP request coalescing/admission.
