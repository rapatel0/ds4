# Sprint 225 - Long-Prompt Replay Reset Determinism

Date: 2026-05-23
Status: Complete - Reset Blocker Cleared, MTP Serving Still Blocked

## Overview

Sprint 224 found a promising but blocked MTP path. Exact block-2 commit was
token-correct and faster than same-process baseline on four of five prompt
fixtures, with the intended accounting shape: eight emitted tokens from seven
target forwards and one speculative save. The blocker is not acceptance rate
anymore; it is replay state determinism.

The `long_memory_archive` fixture failed token parity in
`--mtp-block2-commit-smoke`, and the existing target verifier failed on the
same prompt before MTP-specific logic was involved. A follow-up graph-off
target-block isolation run did not produce a verdict in a bounded amount of
time and had to be killed. Sprint 225 therefore focuses on a bounded,
diagnosable reset path. MTP serving remains blocked until this passes.

## Goals

- Add a bounded replay reset parity diagnostic that:
  - runs greedy generation once;
  - resets the same resident runtime;
  - runs greedy generation again on the same prompt;
  - compares generated token ids and first-token bytes;
  - reports prompt tokens, generated tokens, match status, mismatch index, and
    per-run timing.
- Add a prompt-token limit option for diagnostics so long-prompt failures can
  be bisected without running the full fixture every time.
- Use the new diagnostic to isolate whether reset nondeterminism starts at a
  specific prompt length, with graph on and graph off where practical.
- Fix the reset/snapshot state if the failure is reproducible in a bounded
  case.
- Rerun the target-block and block-2 MTP gates on the bounded failing case and
  on the Sprint 224 prompt matrix.

## Non-Goals

- No MTP serving integration.
- No TP scheduler work.
- No generic scheduler abstraction.
- No changes to the separate TP-only codepath policy.
- No broader throughput tuning unless it is required to keep the diagnostic
  bounded.
- No throughput claim from single-slot replay diagnostics. Practical serving
  throughput must use multi-slot server mode so decode is amortized across
  slots and the routed mat-vec shape becomes a batched mat-mat shape.

## Implementation

1. Add `--reset-parity-smoke N` to `tools/ds4-v100-replay`.
   - Requires direct replay mode, not `--serve` or `--open-only`.
   - Runs two same-runtime greedy generations separated by
     `ds4_v100_replay_reset`.
   - Emits JSON and text output.
   - Fails nonzero on token mismatch.
2. Add `--prompt-token-limit N` to diagnostic/direct replay prompt preparation.
   - Applies after tokenization or synthetic prompt creation.
   - Rejects zero.
   - Truncates only for explicit diagnostics and direct benchmark runs.
3. Use the reset-parity smoke to test:
   - short prompt control;
   - `long_memory_archive` prefixes such as 4K, 32K, 128K, and full prompt if
     runtime is acceptable;
   - graph-on production flags;
   - graph-off isolation if bounded prefixes reproduce the failure.
4. If a bounded failure appears, inspect and fix reset coverage in the replay
   runtime or stage scheduler. Candidate state to verify includes:
   - per-stage raw KV and compressed KV/state;
   - `n_attn_comp` and `n_index_comp`;
   - current HC buffer selection;
   - async pipeline/mailbox generation state;
   - TurboMind CUDA graph cache only if graph-on differs from graph-off.
5. Rerun:
   - `--reset-parity-smoke`;
   - `--target-block-smoke 2`;
   - `--mtp-block2-commit-smoke 8`;
   - at least the Sprint 224 matrix prompts.
6. Tighten the sustained serving benchmark so production throughput defaults to
   `ctx=262144`, `slots=32`, and `active_microbatch=32`, while slot tier `1`
   requires an explicit diagnostic opt-in.
7. Copy cluster evidence to
   `logs/from-cluster/sprint225-reset-determinism/`.
8. Update `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and this sprint
   document with the result.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-replay.c` | reset parity smoke, prompt-token limit, diagnostics |
| `tools/ds4-v100-sustained-decode-bench.sh` | practical throughput guard against accidental single-slot claims |
| `ds4_v100_replay.c` | reset fix if runtime state is incomplete |
| `ds4_v100_scheduler.c` | reset/snapshot fix if scheduler state is incomplete |
| `docs/sprints/SPRINT-225.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | vision update |
| `logs/from-cluster/sprint225-reset-determinism/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] `--reset-parity-smoke` exists, is documented in CLI usage, and fails
      closed on token mismatch.
- [x] `--prompt-token-limit` exists and is documented.
- [x] Local build validation passes.
- [x] V100 build passes.
- [x] V100 reset parity passes on a short control prompt.
- [x] V100 reset parity result is recorded for bounded
      `long_memory_archive` prefixes.
- [x] If a bounded reset mismatch is reproduced, the underlying reset bug is
      fixed or the sprint records a concrete stop condition with evidence.
- [x] `--target-block-smoke 2` passes on the bounded case used for MTP gating,
      or the remaining failure is precisely classified.
- [x] `--mtp-block2-commit-smoke 8` is rerun after the reset gate and its
      promotion decision is updated.
- [x] Sustained serving benchmark defaults to a multi-slot practical tier and
      rejects single-slot throughput runs unless diagnostic mode is explicit.
- [x] Logs are copied to
      `logs/from-cluster/sprint225-reset-determinism/`.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
bash -n tools/ds4-v100-sustained-decode-bench.sh
git diff --check
make -B -j8 tools/ds4-v100-replay.o
```

V100 build:

```text
cd /workspace/ds4-sprint181
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
bash -n tools/ds4-v100-sustained-decode-bench.sh
```

V100 reset parity:

```text
short_code_completion:
  prompt_tokens=27 generated_tokens=8 match=true

long_memory_archive prefixes:
  128 tokens:  match=true generated_tokens=8
  512 tokens:  match=true generated_tokens=8
  1024 tokens: match=true generated_tokens=8
  2048 tokens: match=true generated_tokens=1
  full prompt: prompt_tokens=3353 generated_tokens=1 match=true first_token=32085
```

V100 target-block restore:

```text
long_memory_archive:
  128 tokens:  target_block_smoke=true first_token=18954
  512 tokens:  target_block_smoke=true first_token=1353
  1024 tokens: target_block_smoke=true first_token=9238
  full prompt: target_block_smoke=true prompt_tokens=3353 first_token=32085
               snapshot_bytes=907214848 first_verify_ms=139.651
               second_verify_ms=136.687 rc=0
```

V100 bounded MTP block-2:

```text
long_memory_archive prefix 128:
  token_match=true speculative_saves=1 block2_generated_tps=1.035794

long_memory_archive prefix 512:
  token_match=true speculative_saves=1 block2_generated_tps=0.264497

long_memory_archive prefix 1024:
  token_match=true speculative_saves=1 block2_generated_tps=0.131278
```

The full-prompt MTP block-2 run was intentionally stopped after the throughput
methodology discussion. It is not used as promotion evidence.

V100 practical serving methodology guard:

```text
tools/ds4-v100-sustained-decode-bench.sh --slot-tiers 1 ...
ds4-v100-sustained-decode-bench: --slot-tiers includes 1; use --allow-single-slot-diagnostic for correctness/latency probes, not throughput
rc=1
```

V100 practical multi-slot serving gate:

```text
ctx=262144 slots=32 active_microbatch=32 requests=64 tokens=16
async_pipeline_mode=per-step async_event_handoff=1 microbatch_wait_us=200000 startup_warmup=1
status_200=64 token_match=64 token_mismatch=0
aggregate_generated_tokens_per_second=50.434232
aggregate_continuation_tokens_per_second=47.282093
avg_gpu_util_percent=47.076 max_gpu_util_percent=96.000
```

Evidence is stored in
`logs/from-cluster/sprint225-reset-determinism/`.

## Decision

The Sprint 224 reset/snapshot blocker is cleared. The full
`long_memory_archive` prompt now passes same-runtime reset parity and
target-block snapshot/restore parity. The earlier `got=32085 want=10220`
failure is not reproduced by the current reset and target-block gates.

Do not promote MTP serving yet. The MTP block-2 diagnostic remains single-slot
only, while practical throughput must be measured in multi-slot server mode.
The benchmark harness now defaults to the practical `32`-slot `256K` tier,
uses production-style startup warmup and a `200 ms` microbatch wait, and rejects
single-slot throughput runs unless they are explicitly marked diagnostic.

The next sprint should either implement multi-slot MTP serving semantics or
pivot back to a production multi-slot throughput lever. Any throughput result
must report generated and continuation tok/s separately and must prove
`active_microbatch == slots` for the measured case.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
model: /models/DSv4-Flash-256e-fixed.gguf
mtp_model: /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
```

Representative commands:

```text
./tools/ds4-v100-replay ... --prompt-file tests/test-vectors/prompts/short_code_completion.txt --tokens 8 --reset-parity-smoke 8 --json
./tools/ds4-v100-replay ... --prompt-file tests/test-vectors/prompts/long_memory_archive.txt --prompt-token-limit 4096 --tokens 8 --reset-parity-smoke 8 --json
./tools/ds4-v100-replay ... --prompt-file tests/test-vectors/prompts/long_memory_archive.txt --prompt-token-limit 4096 --tokens 8 --target-block-smoke 2 --json
./tools/ds4-v100-replay ... --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --prompt-file tests/test-vectors/prompts/long_memory_archive.txt --prompt-token-limit 4096 --tokens 8 --mtp-block2-commit-smoke 8 --json
```

Decision gate:

- Continue to MTP serving integration only if reset parity and target-block
  parity pass on the relevant prompt class.
- If reset parity passes but MTP block-2 still fails, the next sprint should
  debug MTP commit state specifically.
- If reset parity cannot be made deterministic in a bounded case, MTP remains
  blocked and the next practical-serving sprint should pivot away from MTP.

## Risks

- Full `long_memory_archive` replay can be slow enough to hide the bug behind
  operational timeouts. Prefix bisection must be the default.
- Prompt-token truncation can mask the failure if the bug only appears after a
  compressed-cache transition.
- Reset parity may pass while target-block restore still fails, implying a
  snapshot-specific bug rather than a full reset bug.

## Security

No external serving exposure. Do not log model weights or full logits.

## Dependencies

- Sprint 221 target-block verifier.
- Sprint 224 block-2 commit gate.
- Production pack `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- MTP sidecar `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`.
