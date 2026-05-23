# Sprint 217 Intent - 32-Slot 256K Admission Probe

Date: 2026-05-23

## Seed Prompt

Continue the high-throughput practical-serving loop after Sprint 216 rejected
current MTP commit as a target-forward-saving speedup.

## Orientation Summary

- Sprint 215 established the best current practical long-context mode:
  `32` slots at `128K`, `68.403129` continuation tok/s.
- Sprint 215 also showed the maximum-context production mode remains `16`
  slots at `256K`, `61.624766` continuation tok/s.
- `32` slots at `256K` was not actually measured; it failed closed at the
  launcher cap: `DS4_V100_SLOTS=32 exceeds ctx=262144 admission cap 16`.
- Sprint 216 showed MTP is not the next immediate throughput lever:
  one-slot commit accepted `8/15` drafts but still had `speculative_saves=0`.
- The launcher already exposes `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP`, so the
  next practical experiment can test whether the `256K`/`32`-slot cap is
  conservative without changing production defaults first.

## Vision Context

The practical serving target needs more aggregate decode throughput while
keeping at least `128K` context and ideally `256K`. If `32` slots at `256K`
fits in 32GB V100 VRAM with acceptable reserve, it is the most direct
operator-facing improvement available before deeper attention/KV or MTP
runtime work.

## Relevant Code Areas

- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-sustained-decode-bench.sh`
- `tools/ds4-v100-practical-serving-matrix.sh`
- `tools/ds4-v100-plan.c`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `docs/sprints/STATUS.md`
- `docs/sprints/VISION.md`

## Constraints

- Preserve default fail-closed production admission unless V100 evidence passes.
- Use the existing experimental cap override for probing.
- Do not hide OOM or token mismatch failures.
- Record prompt/prefill and continuation/decode tok/s separately.
- Record GPU memory and utilization.
- Promote `32`-slot/`256K` only if correctness passes and worst-GPU headroom is
  acceptable.

## Success Criteria

- A repeatable focused `256K`/`32`-slot admission probe exists.
- The probe runs with production TurboMind/F8 flags and
  `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=32`.
- The V100 result records token match, generated tok/s, continuation tok/s,
  prompt tok/s, GPU utilization, and max memory used.
- If it passes with reserve, update launcher/docs to admit `32` slots at
  `256K`; otherwise keep the cap at `16` and document the blocker.
- Logs are captured under `logs/from-cluster/sprint217-256k-32slot-gate/`.

## Verification Strategy

- First run launcher `--check` with the override to ensure the path is explicit.
- Run one focused sustained decode case at `ctx=262144`, `slots=32`,
  `tokens=64`, `requests=32`, `warmup_requests=0`.
- If the first run passes and memory headroom is reasonable, repeat once or run
  a shorter sanity with the promoted cap.

## Uncertainty Assessment

- Correctness: Medium. This is the same serving path, but wider KV state.
- Scope: Medium. The first run may require only a harness; promotion touches the
  launcher and docs.
- Architecture: Low. This does not introduce a new scheduler.

## Open Questions

- Does `32` slots at `256K` fit in real VRAM with the current quantized/cache
  configuration?
- Is the launcher cap conservative, or does the replay allocator fail/OOM?
- If it fits, does wider concurrency improve aggregate continuation tok/s over
  `32` slots at `128K` and `16` slots at `256K`?
