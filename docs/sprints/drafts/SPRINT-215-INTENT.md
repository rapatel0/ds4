# Sprint 215 Intent - Practical Serving Matrix And MTP Viability

Date: 2026-05-23

## Seed Prompt

Continue looping on sprint planning and execution until the next stage of
high-throughput practical serving vision is realized.

## Orientation Summary

- Current promoted production baseline is graph-backed `fused6_reduce` on the
  persistent TurboMind appliance pack.
- Sprints 199-214 exhausted wrapper-level routed-FFN changes, TP4/TP8
  near-term decode integration, and a SIMT tile-local down/reduce diagnostic.
  Sprint 214 specifically showed that avoiding one global down-route buffer is
  not enough if the down projection leaves the Tensor Core path.
- The user target remains practical serving on the 8x V100 32GB node with
  long context: at least 128K, ideally 256K, and enough slots to raise aggregate
  tok/s.
- Existing runbook says `ctx=131072` admits 32 slots, while `ctx=262144`
  remains capped at 16 slots. The user has repeatedly asked whether 32-slot
  long-context serving and quantized KV are being accounted for.
- MTP exists as verify/commit plumbing, but prior evidence says verify is not a
  speedup and commit remains one-slot. MTP must be treated as a measured
  viability lane, not as shipped throughput.

## Vision Context

The north star is a DS4 V100 appliance that preserves the source quantized model
quality, remains device-resident by default, and reaches practical
high-throughput serving. Sprint 214 narrows the next step: do not keep tuning
six-route reducer wrappers. Move to a production serving matrix that proves the
best deployable long-context operating point and exposes the remaining gap to
MTP or a real Tensor Core fused/persistent routed-FFN implementation.

## Relevant Code Areas

- `tools/ds4-v100-sustained-decode-bench.sh`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-replay.c`
- `ds4_v100_replay.c`
- `ds4_v100_mtp.c`
- `docs/operations/DS4-V100-APPLIANCE.md`
- `docs/sprints/STATUS.md`
- `docs/sprints/VISION.md`
- `logs/from-cluster/`

## Constraints

- Use the localpool-backed V100 build pod and persistent appliance pack:
  `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- Do not add a generic scheduler abstraction.
- Do not touch TP/PP scheduler files unless the sprint explicitly chooses TP,
  which this sprint does not.
- Record prompt/prefill, generated, and continuation/decode tok/s separately.
- Keep model weights out of logs.
- Use explicit git staging.

## Success Criteria

- A repeatable practical-serving matrix exists and is checked into the repo.
- V100 evidence covers at least:
  - base 16-slot/256K production baseline;
  - base 32-slot/128K production mode;
  - a 32-slot/256K admission attempt or explicit fail-closed evidence;
  - MTP verify and one-slot commit viability with clear acceptance/commit
    counters.
- The report identifies the best deployable mode today and the numerical gap to
  the target range.
- The sprint makes a clear decision about the next implementation lever:
  MTP true speculative verifier, continuous batching/admission, attention/KV
  boundary, or Tensor Core fused routed-FFN.

## Verification Strategy

- Shell validation for any new benchmark wrapper.
- V100 build if replay/launcher code changes.
- Sustained decode runs on `llm/llamacpp-build-8gpu`.
- Capture status JSON, sustained decode TSV/JSON, and relevant logs under
  `logs/from-cluster/sprint215-practical-serving-matrix/`.
- Token match must pass for served runs.

## Uncertainty Assessment

- Correctness: Low for benchmark/reporting changes; Medium if MTP commit
  accounting or launcher admission changes are needed.
- Scope: Medium. The sprint should ship measurement and operational clarity,
  not a full speculative decoding rewrite.
- Architecture: Low if no scheduler changes are made.

## Open Questions

- Does the current launcher cap alone block 32-slot/256K, or does runtime VRAM
  fail even with the persistent quantized pack?
- Is MTP commit currently just an observability path that still recomputes the
  target token, or can it produce any real effective-token gain in the current
  runtime?
- Does 32-slot/128K materially improve aggregate continuation throughput versus
  16-slot/256K enough to be the recommended practical mode?

## Actionable Deferred/Follow-Up Items

- Sprint 170 follow-up: true persistent/fused routed-FFN executor remains
  critical, but Sprint 214 rejected the first non-Tensor-Core tile-local
  diagnostic. Do not continue reducer-only variants in this sprint.
- Sprint 212 deferred TP runtime items remain parked; TP4/TP8 decode ownership
  did not clear the near-term gates.
- Sprint 182 decision: attention/KV and host/stage wait are now visible
  bottlenecks at 256K. Include the serving matrix results before selecting that
  as the next coding target.
