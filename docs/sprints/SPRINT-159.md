# Sprint 159 - 256K Slot Scaling Evidence

Date: 2026-05-21

## Objective

Find out whether practical `>=256K` serving is currently underbatched before
starting another kernel or topology rewrite.

The production-safe default remains:

```text
ctx = 262144
slots = 16
active_microbatch = 16
```

This sprint adds only an explicit diagnostic override so V100 cluster tests can
try higher 256K slot counts and decide whether the admission table should move.

## Rationale

The continuous 16-slot / 256K soak reached about `63.4` generated tok/s and
`62.4` continuation tok/s with max memory below `24 GiB` on the fullest V100.
That leaves enough VRAM to test whether 24 or 32 active slots increases
throughput. If scaling is positive, the next work should be served batch
formation and denser executor selection. If scaling is flat, the next work
should pivot harder toward TP/EP or a larger persistent routed-FFN boundary.

## Scope

- Add a launcher-only experimental context slot-cap override.
- Keep the default context-aware admission table unchanged.
- Run 256K continuous soaks at 24 and 32 slots if admission and memory allow.
- Record generated and continuation/decode tok/s separately.
- Preserve correctness using the selected-token fixture.

## Non-Goals

- No production default slot-cap change without measured evidence.
- No KV format change.
- No MTP enablement.
- No tensor-parallel scheduler rewrite in this sprint.

## Implementation Plan

1. Add `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP`.
2. Validate the launcher accepts 24/32-slot 256K configs only when the override
   is set.
3. Copy the launcher change to the V100 pod.
4. Run continuous 256K soaks:
   - 24 slots, 64 generated tokens/request, at least 96 requests.
   - 32 slots, 64 generated tokens/request, at least 128 requests.
5. Compare against the current 16-slot sustained baseline:
   - generated: `63.407958` tok/s
   - continuation/decode: `62.417209` tok/s

## Definition Of Done

- Shell syntax check passes for `tools/ds4-v100-run-appliance.sh`.
- Launcher `--check` rejects 24 slots at 256K without the override.
- Launcher `--check` accepts 24 or 32 slots at 256K with the override.
- V100 24-slot continuous soak either passes or records the concrete failure.
- V100 32-slot continuous soak either passes or records the concrete failure.
- Results are copied under `logs/from-cluster/`.
- The decision is recorded in this sprint document and `docs/sprints/VISION.md`.

## Decision Gate

Promote a new 256K slot cap only if:

- selected-token correctness remains exact;
- no GPU crosses the 32 GiB VRAM limit or trips the launcher reserve check;
- continuation/decode tok/s improves materially over the 16-slot sustained
  baseline, not just by run noise.

If 24/32-slot scaling is flat or memory-limited, keep 16 slots as the 256K
default and move to TP/EP or persistent routed-FFN work.

## Results

Validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
```

passed locally.

Launcher admission behaved as intended:

| Check | Result |
|---|---|
| 24 slots / 256K without override | rejected: cap 16 |
| 32 slots / 256K with `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=32` | accepted |

The launcher update was copied to the V100 pod and the cluster check also
accepted 24 slots / 256K only with the override.

Continuous V100 soaks:

| Config | Requests | Generated tok/s | Continuation tok/s | Correctness | Max memory |
|---|---:|---:|---:|---:|---:|
| 16 slots / 256K baseline | 128 | `63.407958` | `62.417209` | `128/128` | `23.9 GiB` |
| 24 slots / 256K | 96 | `65.654665` | `64.628811` | `96/96` | `23.4 GiB` |
| 24 slots / 256K repeat | 120 | `63.761151` | `62.764883` | `120/120` | `23.4 GiB` |
| 32 slots / 256K | 128 | `65.174011` | `64.155667` | `128/128` | `23.4 GiB` |

All runs used:

```text
ctx = 262144
tokens = 64
async_pipeline_mode = per-step
async_event_handoff = true
MTP = off
TurboMind gated SiLU = on
routed executor = off
```

Telemetry:

- 24-slot max sampled GPU utilization reached `89-93%` on the busy stages.
- 32-slot max sampled GPU utilization reached `94-98%` on the busy stages.
- Average utilization remained much lower (`~26-37%` depending on stage),
  which means the layer-split served topology is still not keeping all GPUs
  busy continuously.
- Memory did not grow materially versus 16 slots in this workload, confirming
  the practical limit is not immediate VRAM exhaustion for short prompts.

## Decision

Do **not** raise the 256K production slot cap from 16 yet.

The 24/32-slot runs fit and were correct, but the throughput gain was not
repeatable enough to clear the default-change bar. Keep
`DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP` as a guarded diagnostic tool for future
cluster experiments.

The next material path should not be more admission widening. The evidence now
points back to execution topology:

- fix served batch formation so the routed FFN sees dense multi-slot route
  shapes in the HTTP path, or
- prototype TP/EP/persistent routed-FFN work that creates denser HMMA-heavy
  kernels without relying on more slots at 256K.

Artifacts:

- `logs/from-cluster/continuous-256k-16slot-64tok-128req/`
- `logs/from-cluster/sprint159-256k-24slot-64tok-96req/`
- `logs/from-cluster/sprint159-256k-24slot-64tok-120req/`
- `logs/from-cluster/sprint159-256k-32slot-64tok-128req/`
