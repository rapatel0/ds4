---
sprint: 020
title: V100 Compressor/Indexer And HC Scheduler Bridge
status: intent
date: 2026-05-18
target_repo: rapatel0/ds4
architecture: ../../architecture/DS4-V100-LAYOUT.md
---

# SPRINT-020 Intent: V100 Compressor/Indexer And HC Scheduler Bridge

## Seed Prompt

Continue from Sprint 019 without shrinking back into tiny primitives. The next
sprint should turn the hidden-vector layer body into a more faithful DS4 layer
slice by binding real compressor/indexer descriptors and wrapping the executor
with HC pre/post state handling.

## Orientation Summary

- Sprint 019 shipped `ds4_v100_layer_execute_decode`, grouped F8 attention
  output, semantic attention over explicit raw/compressed KV, and real
  router-selected FFN.
- The Sprint 019 gate passed on V100 and now reports `ready=false` because full
  43-layer scheduling, selected-token decode, serving, MTP, and throughput
  remain incomplete.
- The two immediate blockers are honest and concrete: the executor receives
  prebuilt raw/compressed KV instead of generating compressed rows from real
  descriptors, and it operates on one hidden vector instead of the DS4
  `[4 x 4096]` HC state.
- `docs/architecture/DS4-V100-LAYOUT.md` remains the topology and dtype anchor:
  layer-sharded baseline, source FP8 dense, source MXFP4 routed experts, F16 KV
  first, no persistent dequantized weights.
- Hardware validation must stay on V100, with one-card iteration before the
  full gate.

## Relevant Codebase Areas

- `ds4_v100_layer_state.*`: add compressor/indexer descriptors and arena spans.
- `ds4_v100_layer_execute.*`: add compressed-row generation, indexer selection,
  and HC scheduler entrypoint.
- `ds4_gpu.h`, `ds4_cuda.cu`: compressor, indexer, attention, and HC CUDA
  helpers.
- `tests/cuda_v100_integrated_layer_smoke.c`: extend or pair with a new HC
  integrated smoke.
- `tools/ds4-v100-gate.sh`: add the Sprint 020 gate target.
- `ds4.c`: CPU reference for compressor/indexer and HC pre/post behavior.

## Sprint Goal

Extend the Sprint 019 layer executor so a representative ratio-4 layer can:

1. bind real attention compressor and ratio-4 indexer descriptors;
2. produce/update compressed attention and indexer rows from the current
   `attn_norm` row;
3. select visible compressed rows through the real ratio-4 indexer path;
4. execute semantic attention using executor-owned raw/compressed/indexed KV;
5. wrap the hidden-vector layer body with DS4 HC attention and FFN pre/post
   scheduling; and
6. pass the V100 gate with readiness still honest.

## Parallel Workstreams

| Lane | Owner Scope | Write Scope | Output |
|---|---|---|---|
| A: descriptor state | Bind compressor/indexer tensors and validate dimensions | `ds4_v100_layer_state.*`, `tests/v100_layer_state_smoke.c` | layer state exposes all tensors needed for ratio-4 compression/indexing |
| B: compressed-row execution | Generate compressed rows, indexer rows, and top-k/masks inside executor | `ds4_v100_layer_execute.*`, CUDA helpers if needed | executor no longer needs caller-supplied compressed KV for layer-2 smoke |
| C: HC scheduler wrapper | Add HC pre/post entrypoint around hidden-vector executor | `ds4_v100_layer_execute.*`, HC tests | `[4 x 4096] -> [4 x 4096]` layer slice on V100 |
| D: gate/evidence | Integrated smoke, gate target, direct V100 logs | `tests/*`, `Makefile`, `tools/ds4-v100-gate.sh`, docs logs | one-card and full-gate hardware evidence |

## Constraints

- No public serving, MTP, or throughput optimization in Sprint 020.
- No tensor-parallel implementation.
- No persistent dequantized weights.
- F16 KV remains the baseline.
- Do not claim selected-token readiness until a real output-head path consumes
  a 43-layer scheduled state.

## Success Criteria

- `ds4_v100_layer_state` binds and validates compressor/indexer descriptors for
  layer 2.
- The executor can generate at least one real compressed attention row and
  ratio-4 indexer row from real source bytes.
- The executor can select compressed rows through the indexer/top-k path, or
  produce an explicit documented blocking reason if the current CUDA indexer
  APIs need shape changes.
- A V100 HC layer smoke consumes `[4 x 4096]` state and returns `[4 x 4096]`
  state.
- The appliance gate includes the Sprint 020 smoke and passes on V100.

## Verification Strategy

- Local:
  - `make ds4_v100_layer_execute.o tests/v100_layer_state_smoke`
  - `make tests/cuda_v100_integrated_layer_smoke.o`
  - any new Sprint 020 smoke object target
  - `git diff --check`
- V100 one-card:
  - `CUDA_VISIBLE_DEVICES=0` for layer 2 smoke iteration.
- V100 full gate:
  - `tools/ds4-v100-gate.sh --model /models/DSv4-Flash-256e-fixed.gguf --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --descriptor-layer 2`

## Uncertainty Assessment

| Area | Level | Notes |
|---|---|---|
| Correctness | High | Compressor recurrence, indexer top-k visibility, and HC post composition are all exact-model semantics. |
| Scope | Medium | The sprint is larger than earlier work but bounded to layer 2 and one-card iteration. |
| Architecture | Medium | The layer-state/executor split is now established; the risk is descriptor coverage and scratch shape. |
| Hardware | Medium | Existing kernels passed individually, but the combined compressor/indexer/HC path will stress scratch and launch sequencing. |

## Open Questions

1. Should the executor produce compressed-row masks or top-k row indices as the
   normalized internal representation for ratio-4 layers?
2. Can the current HC CUDA helpers exactly match CPU `hc_pre_from_state_one`
   and `hc_post_one` for one-row decode, or do they need a wrapper kernel?
3. Does Sprint 020 remove `full_43_layer_scheduler`, or should that stay until
   all 43 layers can be walked?

