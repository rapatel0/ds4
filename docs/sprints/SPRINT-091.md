# Sprint 091: Appliance Directory Launcher Path

## Goal

Make the operator launcher and HTTP smoke use the full appliance directory
created in Sprint 090.

## Implementation Plan

- Add `DS4_V100_APPLIANCE_DIR` to the launcher config contract.
- When set, validate `pack-index.tsv`, `turbomind-pack-index.tsv`, and
  `gpu0.weights` through `gpu7.weights`.
- Launch `tools/ds4-v100-replay --serve --appliance-dir DIR` instead of the
  source pack-index scheduler path.
- Add matching `--appliance-dir` support to the HTTP smoke script.
- Run a one-request HTTP smoke against
  `/workspace/ds4-appliance-full-tm-s090` on the V100 pod.

## Definition Of Done

- [x] Launcher config check accepts the full appliance directory.
- [x] Launcher print-command shows `--appliance-dir`, not source `--index`.
- [x] HTTP smoke returns first token hex `3136` from the full appliance.
- [x] Cluster log is committed.

## Result

Sprint 091 wires the Sprint 090 artifact into the operator-facing launch path.

Changes:

- Added `DS4_V100_APPLIANCE_DIR` to
  `tools/ds4-v100-run-appliance.sh`.
- When set, the launcher validates `pack-index.tsv`,
  `turbomind-pack-index.tsv`, and all eight `gpuN.weights` files.
- The launcher now prints/runs `tools/ds4-v100-replay --appliance-dir DIR`
  instead of the source-layout `--index` path.
- Added `--appliance-dir` to `tools/ds4-v100-appliance-smoke.sh`.
- Documented the new env var in
  `deploy/v100/ds4-v100-appliance.env.example`.

Cluster validation:

```text
ds4-v100-run-appliance: config ok ... appliance_dir=/workspace/ds4-appliance-full-tm-s090 ...
CUDA_VISIBLE_DEVICES=0\,1\,2\,3\,4\,5\,6\,7 ./tools/ds4-v100-replay --serve ... --appliance-dir /workspace/ds4-appliance-full-tm-s090 --max-requests 3
ds4-v100-appliance-smoke: request=1 prompt_tokens=18 generated_tokens=1 first_token=926 first_hex=3136 continuation_ms=0.000 ok
ds4-v100-appliance-smoke: health=ok status=ok requests=1 prompt_tokens=18 generated_tokens=1 first_token=926 first_hex=3136 ok
```

The served response uploaded exactly 8 appliance tensors and reported the full
appliance arena sizes:

```text
uploaded_tensors=8
uploaded_bytes=156142896212
arena_bytes=[22524134668,21494393612,21494393612,21494393612,21494393612,17922654732,17901334540,11817197824]
```

The remaining practical gap is performance optimization from this appliance
path: multi-slot async serving, MTP commit mode against appliance shards, and
then sustained tok/s benchmarking.
