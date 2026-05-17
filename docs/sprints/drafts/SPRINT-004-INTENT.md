# Sprint 004 Intent: Runtime Pack Loading And Device Residency Smoke

## Seed

Plan the next sprint after Sprint 003. The working seed is: move from manifest
and packer artifacts into runtime pack-index/shard loading and the first V100
source-format device-resident path, without claiming decode correctness yet.

## Context

- No `docs/sprints/VISION.md` exists. Prior sprints exist, so planning proceeds
  from the current sprint sequence rather than creating a roadmap first.
- Sprint 002 made `/models/DSv4-Flash-256e-fixed.gguf` recognizable and
  inspectable in the fork, with native source dtypes (`BF16`, `MXFP4`,
  `F8_E4M3_B128`) and source tensor-name binding.
- Sprint 003 added `tools/ds4-v100-pack`, which consumes the Sprint 002
  manifest, computes deterministic `gpuN.weights` shard offsets, validates
  source GGUF byte ranges, writes `pack-index.tsv`, and can emit shard files
  when explicitly requested.
- Runtime decode is still intentionally blocked for the native source model.
  The next step should prove runtime can consume the pack contract and place
  source-faithful shard bytes in device memory before adding source-format math.
- The baseline architecture remains 8-stage layer sharding across V100-SXM2
  32GB GPUs, with pure device-resident weights and only HC relay between stages.

## Recent Sprint Context

- `a197d32 Add DS4 source loader manifest baseline`
  - Source GGUF inspect succeeds.
  - Normal generation exits with the explicit source-kernel guard.
  - `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv` records tensor source
    offsets, dtypes, owner GPUs, runtime layouts, and kernel families.
- `31fc4f1 Add DS4 V100 manifest packer`
  - `tools/ds4-v100-pack` dry-run validates the real GGUF and writes
    `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`.
  - Cluster dry-run confirmed the measured shard plan:
    gpu0 20.98 GiB, gpu1-4 20.02 GiB, gpu5 16.69 GiB, gpu6 16.67 GiB,
    gpu7 11.01 GiB.
  - Full real-model shard emission was deferred because the temporary pod
    filesystem is disposable and the copy writes about 145.42 GiB.

## Vision Context

No vision document — planning from current sprint artifacts and follow-ups.

## Relevant Codebase Areas

- `tools/ds4-v100-pack.c`: manifest reader, shard-offset planner, optional
  shard emitter, pack-index writer.
- `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv`: source-to-owner manifest.
- `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`: derived shard index.
- `ds4.c`: GGUF model loading, source-layout detection, tensor binding, source
  generation guard, engine startup.
- `ds4.h`: public engine options; keep API narrow.
- `ds4_gpu.h`: current single-runtime GPU API and model-map/cache hooks.
- `ds4_cuda.cu`: current CUDA global model cache/range loader, single-device
  tensor allocation helpers, and existing q8/f16-style kernels.
- `/Users/ravi/repos/deepseek/docs/sprints/SPRINT-026-CLUSTER-TESTING.md`:
  cluster interaction rules for the V100 pod.
- `docs/architecture/DS4-V100-LAYOUT.md`: layer ownership, shard sizes, format
  policy, and "no persistent dequantized weight buffers" ground rules.

## Constraints

- Preserve `AGENT.md`: DS4 is a model-specific inference engine, not a generic
  GGUF runner; keep public APIs narrow; do not introduce C++; correctness before
  speed.
- Do not materialize persistent F16 dequantized copies of large source weights.
  The sprint may copy source-faithful packed bytes into device arenas.
- Do not enable generation for the source model unless the sprint includes a
  bounded correctness harness that proves the attempted path. The likely sprint
  should leave the decode guard in place.
- Do not make INT8 the default. INT8 remains a later quality/performance branch.
- Use pure VRAM residency as the target. Host/SSD paths are diagnostic only.
- Full shard emission should use persistent scratch, not an `emptyDir` pod
  workspace.
- Avoid broad MTP, multi-slot scheduling, tensor parallelism, and full decode
  integration until source-format residency and at least one math path are
  coherent.

## Actionable Deferred Items

- From `SPRINT-001-DEFERRED.md`:
  - Loader, packer, and runtime shards: loader/packer are now partly done;
    runtime shard loading is actionable.
  - Per-device CUDA state and HC relay: per-device state is becoming actionable,
    but HC relay and scheduling should probably remain out of Sprint 004 unless
    needed for a residency smoke.
  - Full decode integration: still premature until source-format upload and at
    least one kernel path are verified.
  - TurboMind/tc-grid import: still defer broad import; a small kernel probe may
    be planned only if tied to a specific source-format path.

## Actionable Follow-Ups

- From `SPRINT-003-FOLLOWUPS.md`:
  - Run full real-model shard emission on persistent scratch; Important;
    affected files: `tools/ds4-v100-pack.c`, cluster scratch docs.
  - Add runtime pack index reader; Critical; affected files: `ds4.c`, `ds4.h`,
    future runtime pack files.
  - Wire first source-format GPU upload path; Critical; affected files:
    `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`.
- From `SPRINT-002-FOLLOWUPS.md`:
  - Source FP8/MXFP4 runtime upload and dispatch remains Critical.
  - Source-format correctness harness remains Critical once any math path is
    attempted.

## Success Criteria

This sprint is successful if:

1. The repo has a runtime pack-index reader that validates `pack-index.tsv`
   against the source model metadata/tensor binding.
2. The repo can resolve source-layout tensors to `(owning_gpu, shard_file,
   shard_offset, byte_length, dtype/layout/kernel_family)` without relying on
   GGUF byte offsets for the runtime pack path.
3. On the V100 cluster, full real-model shard emission is run on persistent
   scratch or an explicit STOP is recorded with the concrete storage blocker.
4. On the V100 cluster, a CUDA residency smoke loads source-faithful packed
   shard bytes into per-GPU device arenas without overfilling 32GB GPUs.
5. The source-model generation guard remains in place unless a bounded
   correctness harness is implemented and passes.
6. New validation artifacts record per-GPU shard sizes, checksums or file-size
   facts, and per-GPU VRAM usage after upload.

## Verification Strategy

- Local:
  - `make cpu`
  - `make tools/ds4-v100-pack`
  - unit or smoke tests for pack-index parsing using a small synthetic index
  - `git diff --check`
- Cluster:
  - recreate `llamacpp-build-8gpu` only for validation; delete it after.
  - build the relevant CPU/CUDA targets on the pod.
  - run `tools/ds4-v100-pack --emit-shards` only to persistent scratch, not
    temporary workspace.
  - validate `pack-index.tsv` against generated shard files and source GGUF.
  - run a device-residency smoke that allocates/copies all per-GPU shard arenas,
    prints memory usage, and exits without decode.
- Correctness reference:
  - For this sprint, correctness is structural: tensor identity, dtype/layout,
    byte ranges, owner GPU, shard offsets, file sizes/checksums, and VRAM fit.
  - Numerical decode correctness is a later sprint unless this sprint explicitly
    includes a tiny source-format math harness.

## Uncertainty Assessment

- Correctness uncertainty: Medium — pack/index parsing is straightforward, but
  source-to-runtime tensor identity and shard validation must be exact.
- Scope uncertainty: Medium — full shard emission plus CUDA residency is
  feasible but may expose cluster storage or CUDA state assumptions.
- Architecture uncertainty: High — the current CUDA backend is largely global
  and single-device-oriented, while the target appliance needs per-GPU arenas
  and eventually per-stage execution.

## Open Questions

1. Should Sprint 004 include full real-model shard emission as required work,
   or only keep it as a validation step if persistent scratch is available?
2. Should the first runtime pack path read from emitted `gpuN.weights`, or
   should it also support direct GGUF source offsets for faster iteration?
3. Should Sprint 004 stop at device-resident raw packed bytes, or attempt one
   tiny source-format math path such as BF16 embedding/output or F32 HC control?
4. How much CUDA multi-device refactor is necessary for a residency smoke:
   minimal per-device arenas only, or a broader `ds4_gpu_context[gpu]` split?
5. What is the minimum acceptable validation artifact: file sizes only,
   per-shard SHA-256, per-tensor spot checks, or all of the above?
