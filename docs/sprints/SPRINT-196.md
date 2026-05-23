# Sprint 196 - TP4 Pairwise All-Reduce Candidate

Date: 2026-05-23
Status: Completed

## Objective

Move beyond the Sprint 195 root-collective floor by implementing and measuring
a parallel four-GPU all-reduce candidate for the DS4 `active_microbatch x
hidden` TP payload.

## Context

Sprint 195 proved that peer-copy collectives are correct on the V100 node, but
the naive root gather/reduce/broadcast reaches only about `27 GB/s` effective
wire bandwidth at larger payloads and costs about `0.11 ms` for the
16-token/4096-hidden decode payload.

That is useful as a floor, not a production collective. Before a full-layer TP4
prototype, we need to know whether a repo-owned parallel exchange/reduce pattern
materially improves the communication envelope on the actual NVLink islands.

## Scope

- Extend `tools/ds4-v100-tp4-collective-smoke.cu` with `--algo root|doubling`.
- Keep `root` as the default for comparability with Sprint 195.
- Add a recursive-doubling all-reduce path:
  - initialize each device output from its local input;
  - exchange full tensors with XOR peers over two phases;
  - reduce received tensors in-place with a CUDA kernel;
  - verify every GPU receives the exact reduced tensor.
- Report the selected algorithm, latency, and effective wire bandwidth.
- Benchmark root versus doubling on at least one four-GPU NVLink island.

## Non-Goals

- No NCCL dependency in this sprint.
- No production runtime TP integration yet.
- No chunked ring or fused layer boundary yet.
- No model-quality benchmark; this sprint measures the collective primitive.

## Definition Of Done

- [x] `--algo root|doubling` parses and rejects invalid values.
- [x] The doubling path verifies correctness on the V100 pod.
- [x] Root and doubling are benchmarked at the DS4 16-token/4096-hidden payload.
- [x] At least one larger payload is benchmarked to estimate bandwidth scaling.
- [x] The TP4 decision is updated based on measured data.
- [x] Vision/status docs are updated.
- [x] Changes are committed.

## Validation

Local static check:

```text
$ git diff --check
```

Local non-CUDA host behavior remains fail-closed:

```text
$ make tools/ds4-v100-tp4-collective-smoke
tools/ds4-v100-tp4-collective-smoke requires a CUDA build
make: *** [tools/ds4-v100-tp4-collective-smoke] Error 2
```

V100 build:

```text
$ make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp4-collective-smoke
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -arch=sm_70 ...
```

Argument guard:

```text
$ ./tools/ds4-v100-tp4-collective-smoke --algo nope ...
invalid --algo value; expected root or doubling
```

## V100 Results

Same-tool sequential A/B on NVLink island `0,1,2,3`:

| Algo | Tokens | Hidden | Bytes/tensor | Avg ms | Min ms | Max ms | Effective wire GB/s | Verify |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| root | 16 | 4096 | 262144 | 0.110762 | 0.107607 | 0.121792 | 14.200 | ok |
| doubling | 16 | 4096 | 262144 | 0.133761 | 0.126406 | 0.164495 | 15.678 | ok |
| root | 64 | 4096 | 1048576 | 0.278300 | 0.270162 | 0.284836 | 22.607 | ok |
| doubling | 64 | 4096 | 1048576 | 0.184181 | 0.178400 | 0.200720 | 45.545 | ok |
| root | 128 | 4096 | 2097152 | 0.505929 | 0.499698 | 0.522185 | 24.871 | ok |
| doubling | 128 | 4096 | 2097152 | 0.275143 | 0.269721 | 0.290990 | 60.976 | ok |
| root | 256 | 4096 | 4194304 | 0.973573 | 0.960720 | 0.981913 | 25.849 | ok |
| doubling | 256 | 4096 | 4194304 | 0.496396 | 0.484756 | 0.518827 | 67.596 | ok |
| root | 512 | 4096 | 8388608 | 1.867614 | 1.850375 | 1.880853 | 26.950 | ok |
| doubling | 512 | 4096 | 8388608 | 0.890541 | 0.883041 | 0.907927 | 75.357 | ok |
| root | 1024 | 4096 | 16777216 | 3.675847 | 3.662383 | 3.690902 | 27.385 | ok |
| doubling | 1024 | 4096 | 16777216 | 1.655687 | 1.648249 | 1.674694 | 81.065 | ok |

16-token cross-island check:

| Devices | Algo | Avg ms | Effective wire GB/s | Verify |
|---|---|---:|---:|---|
| 4,5,6,7 | root | 0.113759 | 13.826 | ok |
| 4,5,6,7 | doubling | 0.130128 | 16.116 | ok |

## Decision

Recursive doubling is a real improvement for large active payloads, but it does
not help the current production-shaped `active_microbatch=16` decode payload.
At 16 tokens, it adds about `0.023 ms` latency versus root. At 64+ tokens it is
substantially better, and by 1024 tokens it cuts latency from `3.676 ms` to
`1.656 ms`.

This means:

- full TP4 is still plausible for dense batched work, prefill-like shapes, or a
  future scheduler that creates larger per-layer payloads;
- full TP4 is not yet attractive for the current 16-slot/256K decode shape
  unless the collective is fused into a larger persistent layer boundary;
- the next production-serving sprint should favor the monolithic routed-FFN /
  persistent boundary path, while preserving the doubling collective as the
  baseline for a later bounded TP4 layer prototype.
