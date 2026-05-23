# Sprint 221 - MTP Target Block Verification Primitive

Date: 2026-05-23
Status: Complete

## Overview

Sprint 220 made the warmed `32`-slot/`256K` appliance mode deployable through
the production operator path. The high-throughput serving vision is still not
realized: maximum-context serving is correct and deployable, but still only in
the `~58` generated tok/s band.

The next material lever is MTP only if it can save target-model forwards. Sprint
216 proved the current MTP commit path cannot: it accepts drafts, but still
reports `target_forwards=effective_output_tokens` and `speculative_saves=0`.
The missing runtime boundary is a target-model verification/state-advance
primitive for a one-slot drafted token block.

Sprint 221 builds that primitive without promoting MTP serving by default. The
first implementation may still execute the drafted positions serially inside
the primitive; the value of this sprint is to create and validate the explicit
API boundary, snapshot/rollback behavior, and accounting that later work can
replace with a fused, batched, or graph-captured implementation.

## Goals

- Add a replay-level snapshot wrapper over all eight stage schedulers.
- Add a replay-level target block verification API that:
  - accepts one slot's drafted/forced token block and positions;
  - feeds the block through the target model as one explicit verification
    boundary;
  - returns target outputs per drafted position;
  - reports target forwards, accepted prefix length, effective tokens, and
    speculative saves.
- Add a V100 smoke path proving that:
  - a block verified after a prompt matches the normal greedy replay outputs;
  - restoring the snapshot and verifying the same block again gives the same
    outputs;
  - snapshot bytes and target-forward accounting are reported.
- Keep production MTP serving default-off and unchanged unless this primitive
  proves a real target-forward saving path later.

## Non-Goals

- No production MTP promotion.
- No multi-slot MTP commit.
- No claim of throughput improvement from this sprint alone.
- No TP/PP scheduler integration.
- No new routed-FFN or attention kernels.

## Implementation

1. Extend `ds4_v100_replay.h` / `ds4_v100_replay.c`:
   - introduce an opaque `ds4_v100_replay_snapshot`;
   - add create/restore/free/bytes wrappers over
     `ds4_v100_stage_scheduler_snapshot_*`;
   - add `ds4_v100_replay_verify_token_block()` and a small report struct.
2. Extend `tools/ds4-v100-replay.c` with a diagnostic
   `--target-block-smoke N` mode:
   - run a normal greedy baseline for `N+1` tokens;
   - reset and replay the prompt to the pre-block state;
   - snapshot that state;
   - verify the forced block of baseline tokens and compare outputs against the
     baseline next-token sequence;
   - restore the snapshot and repeat the block to prove rollback determinism;
   - print/report snapshot bytes, target forwards, accepted prefix length, and
     speculative saves.
3. Run local build/syntax validation.
4. Build `tools/ds4-v100-replay` on the V100 pod.
5. Run the target-block smoke on the Sprint 181 production appliance pack.
6. Copy V100 logs to
   `logs/from-cluster/sprint221-mtp-target-block-smoke/`.
7. Update status, vision, and this sprint doc with evidence and decision.

## Files In Scope

| File | Purpose |
|---|---|
| `ds4_v100_replay.h` | public replay snapshot/block verification API |
| `ds4_v100_replay.c` | snapshot wrappers and target block implementation |
| `tools/ds4-v100-replay.c` | diagnostic target-block smoke |
| `docs/sprints/SPRINT-221.md` | plan and execution evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | vision update |
| `docs/operations/DS4-V100-APPLIANCE.md` | MTP/operator note if behavior changes |
| `logs/from-cluster/sprint221-mtp-target-block-smoke/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Replay snapshot create/restore/free/bytes API exists.
- [x] Target block verification API exists and reports accounting.
- [x] `tools/ds4-v100-replay --target-block-smoke N` exists and is guarded to
      one-slot diagnostics.
- [x] Local validation passes.
- [x] V100 build passes.
- [x] V100 target-block smoke passes on the production appliance pack.
- [x] Evidence logs are copied to
      `logs/from-cluster/sprint221-mtp-target-block-smoke/`.
- [x] Docs are updated with whether this primitive is ready for the next MTP
      integration sprint.
- [x] Changes are committed with explicit `git add` paths.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
model: /models/DSv4-Flash-256e-fixed.gguf
```

Smoke command:

```text
./tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 262144 \
  --slots 1 \
  --tokens 8 \
  --target-block-smoke 4
```

Expected:

- baseline first token remains `3136` when paired with the existing fixture;
- first block pass matches baseline outputs for the forced block;
- restore + second block pass matches the first block pass;
- report has `target_forwards=4`, `accepted_prefix_len=4`,
  `effective_output_tokens=4`, and `speculative_saves=0` for the serial first
  implementation;
- snapshot bytes are non-zero.

## Risks

- The first primitive does not improve throughput by itself. It is still useful
  only if it cleanly isolates the boundary that future MTP work can optimize.
- Existing scheduler snapshots require `active_slots=1`; the smoke must fail
  closed for multi-slot configs.
- Snapshot capture copies device state to host and is not a production path.
  Later work must replace it with prefix commit/rollback state management.
- The MTP sidecar may not produce long accepted prefixes on real prompts; this
  sprint does not solve acceptance quality.

## Security

No external serving exposure. The smoke uses local files and does not log model
weights.

## Dependencies

- Sprint 216 MTP accounting evidence.
- Sprint 220 production appliance deployment path.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.

## Execution

Local validation:

```text
git diff --check
make -j8 tools/ds4-v100-replay.o ds4_v100_replay.o
```

The full `tools/ds4-v100-replay` target is CUDA-only on the local Mac and
correctly fails there with `tools/ds4-v100-replay requires a CUDA build`; the
actual binary build was run on the V100 pod.

V100 build:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
command: make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
result: pass
```

V100 target-block smoke:

```text
./tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --ctx 262144 \
  --slots 1 \
  --active-microbatch 1 \
  --tokens 8 \
  --target-block-smoke 4 \
  --expected-token-hex 3136
```

Production TurboMind environment flags matched the Sprint 220 launcher defaults
for the interleaved gated-SiLU appliance pack.

Result:

```text
ds4-v100-replay: target_block_smoke block_tokens=4 baseline_tokens=8 first_token=926 first_hex=3136 snapshot_bytes=30107648 target_forwards=4 accepted_prefix_len=4 target_tokens_verified=4 effective_output_tokens=4 speculative_saves=0 first_verify_ms=232.302 second_verify_ms=231.457 ok
```

Negative guard:

```text
ds4-v100-replay: --target-block-smoke currently requires --slots 1 --active-microbatch 1
negative_rc=2
```

## Evidence

Logs:

```text
logs/from-cluster/sprint221-mtp-target-block-smoke/target-block-smoke.log
logs/from-cluster/sprint221-mtp-target-block-smoke/target-block-negative.log
```

The smoke proves:

- the normal greedy baseline still produces fixture bytes `3136`;
- the first forced target block matches baseline tokens 2-5;
- snapshot restore plus second forced target block matches the first block;
- snapshot capture spans `30107648` bytes of target state;
- the new API reports `target_forwards=4`, `accepted_prefix_len=4`,
  `target_tokens_verified=4`, `effective_output_tokens=4`, and
  `speculative_saves=0`.

## Decision

Ship the replay-level target block verification primitive as the next MTP
foundation. This sprint does not promote MTP as a throughput feature because
the initial implementation still performs serial target forwards inside the
block. It does, however, make the missing boundary concrete and tested:
future work can now replace the serial body with a graph-captured, batched, or
state-advance implementation while preserving the same API and rollback smoke.
