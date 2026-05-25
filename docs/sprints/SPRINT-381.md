# Sprint 381: FP8 E5M2 KV Gate

## Overview

Add a default-off FP8 E5M2 KV-cache format gate to the TP/EP appliance and
measure it against the current FP8 E4M3 block-128 typed KV path at the target
`32` slot / `256K` context shape.

This sprint does not change topology, scheduler semantics, or PP/layer-split
code. It keeps the existing block-128 row layout:

```text
1 E8M0 scale byte + 128 FP8 payload bytes
```

The gate changes quant/dequant semantics only:

- control: `DS4_V100_TP_KV_F8_E4M3_B128`
- candidate: `DS4_V100_TP_KV_F8_E5M2_B128`

## Rationale

The active vision points at typed attention/KV traffic and launch fragmentation
as the near-term TP/EP serving bottleneck. E5M2 has the same memory footprint
as E4M3 in this row layout, so it is not a memory-capacity win by itself. Its
value is as a controlled format experiment: better FP8 dynamic range may reduce
clipping risk in long-context KV paths while preserving the appliance's compact
device-resident KV layout.

## Scope

- Add `DS4_V100_TP_KV_F8_E5M2_B128` to the TP runtime dtype enum.
- Generalize typed F8 KV row store/load kernels to dispatch E4M3 or E5M2.
- Add host/device E5M2 quant/dequant helpers.
- Keep row sizing, sharding, allocator layout, and cache indexing unchanged.
- Add `--kv-dtype f8_e5m2_b128` to `tools/ds4-v100-tp-runtime-smoke`.
- Add `--fp8-e5m2-kv-gate` to the full-layer smoke/runtime path.
- Add launcher/profile plumbing:
  `DS4_V100_TP_EP_FP8_E5M2_KV=1` and
  `tools/ds4-v100-tp-ep-profile.py --fp8-e5m2-kv`.

## Out Of Scope

- No PP/layer-split work.
- No TP-sharded expert serving integration.
- No MTP changes.
- No KV allocator redesign.
- No promotion without V100 same-binary parity evidence.

## Definition Of Done

- V100 build succeeds for:
  `tools/ds4-v100-tp-runtime-smoke` and
  `tools/ds4-v100-tp-ep-full-layer-smoke`.
- E5M2 typed KV row smoke passes for `attn`, `attn_raw`, and `indexer` rows:
  zero decoded errors and zero packed byte mismatches.
- E5M2 device store/load roundtrip passes for the same row kinds.
- E4M3 typed/device smoke still passes after shared F8 scale-byte cleanup.
- Direct production-shaped A/B is recorded at `32` slots / `256K`:
  control E4M3 vs candidate E5M2.
- The sprint ends with an explicit promote/reject/diagnostic decision.

## Risks

- E5M2 can preserve first-token parity but drift on continuation tokens because
  generated KV rows feed later decode steps.
- Because E4M3 and E5M2 have the same packed size here, performance wins are
  expected to be small unless numeric range changes reduce downstream work.
- The long-context direct harness has large startup/load time, so short A/B
  measurements are noisy and should not be overinterpreted.

## Outcome

Implemented and validated as a default-off diagnostic gate.

Code changes:

- `DS4_V100_TP_KV_F8_E5M2_B128` runtime dtype.
- Host/device E5M2 quant/dequant helpers.
- Shared F8 typed KV row kernels that dispatch E4M3 or E5M2 from the runtime
  dtype.
- Deterministic E8M0 scale-byte selection from `amax / fp8_max`, which keeps
  E4M3 byte parity and prevents host/device E5M2 scale drift.
- CLI/launcher/profile gates:
  `--kv-dtype f8_e5m2_b128`,
  `--fp8-e5m2-kv-gate`,
  `DS4_V100_TP_EP_FP8_E5M2_KV=1`,
  and `tools/ds4-v100-tp-ep-profile.py --fp8-e5m2-kv`.

V100 evidence:

| Check | Result |
|---|---|
| Build | `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-runtime-smoke tools/ds4-v100-tp-ep-full-layer-smoke` passed |
| E5M2 typed rows | `attn`, `attn_raw`, `indexer`: `bad_values=0`, `byte_mismatches=0` |
| E5M2 device rows | `attn`, `attn_raw`, `indexer`: `bad_values=0`, `max_abs=0` |
| E4M3 regression | `attn` typed/device row still passed with zero errors |
| Direct 1-token A/B | first token `54639` both; decode `67.710842 -> 69.225694` tok/s |
| Direct 4-token A/B | checksum `13373834059` both; first token `98751` both; decode `70.710875 -> 75.787866` tok/s; continuation decode `75.203353 -> 78.105479` tok/s |
| HTTP selected-token 4-token A/B | `32/32` HTTP 200 both; first token `45178` both; client tok/s `17.212677 -> 22.389190`; compressed-KV sum `491.310011 -> 442.415827` ms |
| HTTP memory | both successful HTTP runs reported `32418 MiB` max used; one immediate candidate run after the control failed with CUDA OOM before readiness, so memory margin is still very tight |

Artifacts:

```text
/workspace/logs/sprint381-e5m2-kv/
```

## Decision

Keep E5M2 KV default-off for now.

The gate is correct in the typed-row/device-row tests and promising in short
direct/HTTP measurements, but it is not promoted as the serving default yet
because E5M2 trades mantissa precision for exponent range and the validation
only covers short selected-token runs. The transient HTTP OOM also shows the
current `32` slot / `256K` serving shape has almost no VRAM slack.

Promotion requires a longer deterministic chat or selected-token parity run
with repeated A/B evidence and no startup OOM at the default launcher shape.
