# Sprint 008 Merge Notes

## Inputs

- `docs/sprints/drafts/SPRINT-008-INTENT.md`
- `docs/sprints/drafts/SPRINT-008-CLAUDE-DRAFT.md`
- `docs/sprints/drafts/SPRINT-008-CODEX-DRAFT.md`
- `docs/sprints/drafts/SPRINT-008-GEMINI-DRAFT.md`
- `docs/sprints/VISION.md`
- `docs/architecture/DS4-V100-LAYOUT.md`
- `docs/sprints/SPRINT-007-REPORT.md`
- `docs/sprints/SPRINT-007-DEFERRED.md`
- `docs/sprints/SPRINT-007-FOLLOWUPS.md`

## Draft Positions

- Claude proposed a very bounded sprint: automate the existing oracle,
  harden source-layout guards, add F16 KV admission reporting, add a
  conservative device source-format anchor, and keep full prefill/runtime work
  deferred.
- Codex proposed the same core shape, with `tests/ds4_test.c` as the preferred
  oracle harness, V100 context as the KV admission owner, and a synthetic
  `F8_E4M3_B128` CUDA row anchor as the first production-relevant device proof.
- Gemini put more emphasis on prompt prefill and hidden-state transition work,
  but still identified oracle automation, KV budget/admission, and device-side
  source anchors as the right near-term foundation.

## Merge Decision

Sprint 008 will be a bridge sprint named **Source Oracle Harness And V100 KV
Admission Anchors**.

This is narrower than the original vision placeholder "Prefill, KV, And
Compressed Attention." The narrower scope is intentional. Sprint 007 proved a
single selected token, but full V100 prefill/KV execution still lacks:

- repeatable oracle automation;
- guard regression coverage;
- exact F16 KV admission by stage, slot, and context;
- a CUDA source-format anchor against the shared source helpers.

Trying to implement full source-layout V100 prefill before those contracts
exist would mix correctness, memory, and kernel risks in one sprint.

## Chosen Release Gates

- `SHIP`: oracle automation passes the official short vector, guard tests pass,
  V100 F16 KV admission/reporting is exact and tested, MXFP4 parity is hardened,
  and one bounded CUDA source-format anchor passes on `sm_70`.
- `EXTEND`: oracle automation, guard tests, and KV admission ship, but the CUDA
  source-format anchor is blocked by cluster access or CUDA build issues.
- `STOP`: any work requires unlocking normal source-layout serving, adding
  broad production kernels, using host/SSD/offload as a success path, or
  weakening the F16 KV correctness baseline.

## Deferred From Sprint 008

- Full source-layout V100 prefill execution.
- Full compressed attention/indexer device population.
- Production FP8 dense GEMMs and MXFP4 routed expert kernels.
- Server deployment.
- Multi-slot throughput, wavefront scheduling, MTP, and tensor-parallel
  exceptions.

These should move to the next vision milestones after Sprint 008 closes.
