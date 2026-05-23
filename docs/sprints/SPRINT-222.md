# Sprint 222 - MTP Draft Block Chaining Diagnostic

Date: 2026-05-23
Status: Planned

## Overview

Sprint 221 created the target-model block verification boundary and proved
snapshot/restore determinism on the V100 production appliance pack. That closes
one side of true MTP speculative serving: the target can now verify a forced
one-slot token block through an explicit API.

The other side is still missing. The current MTP forward helper returns only
top-k draft tokens/logits. Internally it computes the next MTP hidden-control
state (`ffn_next_t`), but that state is not exposed to callers. As a result,
serving can ask MTP for one draft after a target token, but cannot chain MTP's
own accepted draft into a multi-token draft block without running the target
model again.

Sprint 222 builds the draft side of the same boundary: expose MTP next-HC,
chain a short MTP-generated draft block in a diagnostic mode, and verify that
block through Sprint 221's target-block verifier. This still does not promote
MTP serving; it creates the first end-to-end shape needed for future target
forward savings.

## Goals

- Extend the MTP forward helper so callers can optionally read the next MTP HC
  state produced by a forward pass.
- Add a guarded diagnostic that:
  - replays a prompt to the target pre-draft state;
  - reads the target output HC and selected target token;
  - runs MTP for a configurable draft block length by feeding each MTP draft
    token and the previous MTP next-HC into the next MTP step;
  - verifies the drafted block through `ds4_v100_replay_verify_token_block()`;
  - reports drafted tokens, accepted prefix, target forwards, effective tokens,
    and speculative saves.
- Run the diagnostic on the V100 production appliance pack and record whether
  real chained MTP drafts are coherent enough to justify a serving integration
  sprint.

## Non-Goals

- No production MTP default change.
- No multi-slot MTP serving.
- No claim that MTP is faster unless `speculative_saves > 0` and throughput
  evidence proves it.
- No target-block optimization or CUDA graph capture in this sprint.
- No changes to PP/TP topology.

## Implementation

1. Extend `tools/ds4-v100-mtp-forward-common.h/.c`:
   - keep the existing `ds4_v100_mtp_forward_run_host()` ABI intact;
   - add a new optional-output wrapper or sibling function that copies
     `ffn_next_t` into a caller-provided `float next_hc[16384]`;
   - preserve existing smoke behavior.
2. Extend `tools/ds4-v100-replay.c` with a diagnostic
   `--mtp-draft-block-smoke N` mode:
   - requires `--mtp-model`, `--slots 1`, and `--active-microbatch 1`;
   - opens the MTP service;
   - replays the prompt and selects the first target token;
   - runs MTP chained draft steps using target embedding/HC for step 0 and
     MTP draft token embedding plus MTP next-HC for later steps;
   - verifies the draft block with the Sprint 221 target block API;
   - reports accepted prefix and accounting.
3. Add local syntax/object validation.
4. Build `tools/ds4-v100-replay` on the V100 pod.
5. Run the diagnostic on the Sprint 181 production appliance pack.
6. Copy logs to `logs/from-cluster/sprint222-mtp-draft-block-smoke/`.
7. Update sprint/status/vision/runbook docs with the decision:
   - continue to target-block optimization if chained drafts have non-trivial
     accepted prefixes;
   - otherwise pause MTP throughput work and pivot back to attention/KV or
     persistent low-bit execution.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-mtp-forward-common.h` | MTP forward next-HC API |
| `tools/ds4-v100-mtp-forward-common.c` | next-HC readback implementation |
| `tools/ds4-v100-replay.c` | draft-block diagnostic |
| `docs/sprints/SPRINT-222.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | vision update |
| `docs/operations/DS4-V100-APPLIANCE.md` | MTP diagnostic/runbook note |
| `logs/from-cluster/sprint222-mtp-draft-block-smoke/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Existing MTP forward ABI remains source-compatible.
- [ ] New MTP forward path can return next-HC.
- [ ] `tools/ds4-v100-replay --mtp-draft-block-smoke N` exists and fails
      closed unless MTP model and one-slot mode are configured.
- [ ] Local validation passes.
- [ ] V100 build passes.
- [ ] V100 draft-block smoke runs on the production appliance pack.
- [ ] Logs are copied to
      `logs/from-cluster/sprint222-mtp-draft-block-smoke/`.
- [ ] Docs state whether chained MTP drafts justify the next MTP optimization
      sprint.
- [ ] Changes are committed with explicit `git add` paths.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
model: /models/DSv4-Flash-256e-fixed.gguf
mtp_model: /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Smoke command:

```text
./tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 262144 \
  --slots 1 \
  --active-microbatch 1 \
  --tokens 8 \
  --mtp-draft-block-smoke 4 \
  --expected-token-hex 3136
```

Expected:

- target first token remains `3136`;
- MTP emits a 4-token draft block without target forwards between draft steps;
- target block verification runs against that draft block;
- report includes draft token IDs, accepted prefix, target forwards, effective
  output tokens, and speculative saves;
- diagnostic does not mutate production serving defaults.

## Risks

- Chained MTP drafts may diverge immediately. If accepted prefix is consistently
  zero, MTP should pause as a throughput lever.
- Reading next-HC to host is diagnostic-only and may be too slow for serving.
  A future serving path should keep this state device-resident.
- The MTP raw attention cache currently resets inside each helper call; if that
  prevents coherent chained drafts, this sprint should record it explicitly and
  stop rather than forcing a misleading integration.

## Security

No external serving exposure. Do not log model weights or full logits.

## Dependencies

- Sprint 221 target block verifier.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- MTP sidecar `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
