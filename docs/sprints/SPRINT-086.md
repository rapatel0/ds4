# Sprint 086: TurboMind Sidecar VRAM Admission

## Status

Complete.

## Overview

Sprint 085 proved that a bounded TurboMind sidecar can execute from persistent
device memory. Sprint 086 adds the next guardrail: an admission report that
combines the normal source pack index with a TurboMind sidecar index and shows
whether a layout fits under the 32 GB V100 budget with reserve, KV, and scratch
allowances.

The report deliberately separates duplicate residency from replacement-style
residency. Duplicate residency means keeping the current source expert bytes
and adding TurboMind expert sidecars. Replacement-style residency estimates the
target appliance layout where source expert payload is not also resident after
the TurboMind pack is selected.

## Goals

1. Add a CPU admission tool for TurboMind sidecar memory accounting.
2. Report source arena bytes, source expert payload bytes, and sidecar bytes
   per GPU.
3. Include explicit reserve, KV, and scratch budgets.
4. Show both duplicate and replacement-style fit status.
5. Validate against the bounded Sprint 085 sidecar on the V100 cluster.

## Non-Goals

- Generating full all-layer sidecars.
- Changing scheduler defaults.
- Compacting source pack shards to remove expert payload.
- Predicting tok/s from the admission report.

## Definition of Done

- [x] `tools/ds4-v100-turbomind-admit` builds locally and on the cluster.
- [x] The tool reads `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`.
- [x] The tool reads `turbomind-pack-index.tsv`.
- [x] The report includes duplicate and replacement-style totals.
- [x] The cluster run uses 32 GiB VRAM, 4 GiB reserve, 1 GiB KV, and 1 GiB
      scratch budgets.
- [x] V100 log is recorded under `logs/from-cluster/`.
- [x] Artifacts are committed.

## Result

`SHIP_SIDECAR_ADMISSION`.

For the bounded layer-0/two-expert sidecar, duplicate residency still fits with
the conservative 6 GiB fixed budget:

```text
gpu  source_arena_gib  tm_sidecar_gib  duplicate_total_gib  duplicate_status
0    20.977            0.025           27.002               OK
```

The same report shows why full production sidecars should be treated as
replacement/admitted artifacts, not silently duplicated: GPU 0 already has
`19.125 GiB` of source expert payload. A full TurboMind sidecar for the same
expert set would likely make duplicate residency exceed 32 GiB, while a
replacement-style layout remains plausible.

## Next Step

Sprint 087 should connect admission to generation/runtime: either generate a
full per-GPU sidecar for one admitted GPU/layer range and run the sidecar smoke
against it, or add scheduler-side flags that only enable TurboMind execution
when the admitted sidecar is present.
