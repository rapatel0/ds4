# Sprint 190 - Attention-Only Single-Slot Scratch

Date: 2026-05-22

## Objective

Separate the Sprint 189 attention scratch win from the broader
`DS4_V100_SINGLE_SLOT_BATCH_SCRATCH` selector by adding an attention-only
single-slot scratch gate.

## Rationale

Sprint 189 found a real filled-context performance signal:

- len-256 continuation: `13.858673` -> `17.503002` tok/s
- len-1024 continuation: `15.228124` -> `16.923312` tok/s

However, the broad single-slot selector controls more than attention. It also
changes HC and FFN scratch behavior. Explicit opt-in matched selected tokens,
but a no-env default-on attempt drifted token IDs. The next step is to isolate
the attention portion with a narrower gate before trying to promote anything.

## Scope

- Add `DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=1`.
- Use that gate only in `execute_attention_output()`.
- Leave `DS4_V100_SINGLE_SLOT_BATCH_SCRATCH=1` unchanged as the broad
  diagnostic flag.
- Validate direct synthetic len-256 and len-1024 on the persistent Sprint 181
  appliance pack.

## Non-Goals

- No default promotion unless no-env validation is token-stable.
- No new attention softmax kernel.
- No online attention promotion.
- No TP/EP topology work.

## Definition of Done

- [x] V100 build passes.
- [x] Attention-only scratch direct synthetic len-256 matches control.
- [x] Attention-only scratch direct synthetic len-1024 matches control.
- [x] Throughput impact is recorded against Sprint 189 control.
- [x] Evidence is archived under `logs/from-cluster/`.
- [x] Sprint outcome and decision are recorded.
- [x] Vision is updated.
- [x] Changes are committed.

## Implementation

Added `DS4_V100_SINGLE_SLOT_ATTN_SCRATCH`.

Behavior:

- `DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=1` enables slot-0 scratch reuse only
  inside `execute_attention_output()`.
- `DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=0` is the rollback.
- No-env default is enabled for attention-only scratch.
- `DS4_V100_SINGLE_SLOT_BATCH_SCRATCH=1` remains the broader diagnostic flag
  that also enables HC/FFN scratch.

Launcher wiring:

- `tools/ds4-v100-run-appliance.sh` defaults
  `DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=1`, validates it as a boolean, records it
  in startup env, and exports it.
- `deploy/v100/ds4-v100-appliance.env.example` documents the default and
  rollback.

## Evidence

Build and launcher syntax:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
bash -n tools/ds4-v100-run-appliance.sh
```

passed.

Direct synthetic len-256 / ctx-262144:

| Mode | Prompt tok/s | Continuation tok/s | Output IDs |
|---|---:|---:|---|
| control | `12.742135` | `11.743995` | `3955, 361` |
| attention-only opt-in | `15.002672` | `15.973668` | `3955, 361` |
| default-on | `14.734542` | `15.874062` | `3955, 361` |
| rollback | `12.529401` | `12.100086` | `3955, 361` |

Direct synthetic len-1024 / ctx-262144:

| Mode | Prompt tok/s | Continuation tok/s | Output IDs |
|---|---:|---:|---|
| control | `15.006616` | `15.144401` | `926, 926` |
| attention-only opt-in | `15.471745` | `15.192258` | `926, 926` |
| default-on | `15.447474` | `15.563907` | `926, 926` |

Evidence:

```text
logs/from-cluster/sprint190-attn-only-scratch/
```

## Decision

Promote attention-only single-slot scratch as the default, with
`DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=0` as the rollback.

Unlike Sprint 189's broad scratch selector, the attention-only default preserved
selected token IDs in both len-256 and len-1024 direct synthetic runs. The
speedup is strongest at shorter filled context and still positive in the
len-1024 default-on run.

This does not realize the full high-throughput serving vision yet, but it is a
safe filled-context attention/KV improvement and should become the new baseline
for the next sprint.
