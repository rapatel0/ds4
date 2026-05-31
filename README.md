# DS4 V100 Research Archive

> **Status: abandoned / research archive.** This repository is public for
> reference, but it is not worth actively pursuing for the moment. The V100
> 8-GPU appliance work did not reach a useful serving configuration, and the
> MTP speculative path is numerically wrong. Treat this repo as an engineering
> record, not an active product direction.

## What This Is

This repository was inspired by the original `ds4` project, but it became an
entirely new and different V100-focused experiment. The goal was to see whether
DeepSeek V4 Flash could be generalized into an 8x NVIDIA V100 appliance with
high-throughput serving.

The original DwarfStar4 project direction was centered around a narrow DeepSeek V4 Flash
engine, especially the Metal/macOS path. This repo diverged into a CUDA
appliance effort with new runtime structure, new packing assumptions, a
TurboMind integration, custom V100 kernels, and a long sequence of TP/EP serving
experiments.

## Outcome

We tried two broad ways to map the model onto 8x V100s:

- **LP / layer-parallel:** layers or layer-owned expert blocks are assigned to
  different GPUs. This made the model fit, but decode behaved like a mostly
  serial pipeline with uneven GPU utilization and idle time between handoffs.
- **EP / expert-parallel:** every layer can use all 8 GPUs, with each rank
  owning a slice of the experts. This improved the structure, but the routed MoE
  shape was too sparse at normal serving batch sizes to keep V100s dense.

The best promoted TP/EP reference point reached about `26.8` aggregate decode
tok/s at `32` slots / `256K` context / `64` tokens per request. Profiling showed
the core problem clearly: EP/MoE all-to-all and expert orchestration were about
`65.2%` of decode time. Attention was about `12%`, compose about `11%`, HC about
`8%`, and host sync about `5%`.

At 32 slots with top-6 routing, 192 routed activations are spread over 256
experts. The average expert sees less than one token. That is a hard shape for
the V100: grouped GEMMs are tiny, dispatch/compose/all-to-all overhead is large,
and SM occupancy stays poor even after CUDA graph launch overhead is reduced.
MoE compute density, not just kernel launch overhead, is the structural blocker.

## What Was Added

The repo contains a substantial amount of experimental V100-specific work:

- An 8-GPU TP/EP serving appliance path.
- TurboMind integration for routed expert execution.
- Custom CUDA kernels for the V100 path.
- Int8-oriented computation designed to use V100 FP16 tensor cores where
  practical.
- NCCL transport replacements for hot cross-rank movement.
- CUDA graph capture/replay work for the decode path.
- MTP/speculative-decoding integration attempts.
- Sprint documentation recording what was tried and what was rejected.

The MTP path is specifically **not working**. The integrated draft path runs and
does not corrupt the main model token stream, but deterministic draft acceptance
remained `0/71`, so it provides no useful speedup.

## Research Ideas Left On The Table

There are still ideas that could be pursued, but none looked like a clear 10x
win from the evidence:

- Pack experts onto GPUs based on observed load patterns rather than static
  expert ranges. (hard to do without lots of data and throughput is too slow on this hardware)
- Fuse expert routing / gate-up / activation / down / compose into a single
  kernel to improve expert compute density and reduce orchestration overhead.

Those may be interesting research directions, but they are not enough to justify
continuing this track right now.

## Repository Notes

Useful context lives in:
- `docs/sprints/` - sprint-by-sprint execution history.
- `appliance/`, `engine/`, `kernels/`, `tools/` - the experimental CUDA/TP/EP
  implementation.

The untracked `research/` folder used during development was informational and
is not part of the repo history.

## Acknowledgements

This work was inspired by and benefited from:

- **ds4** - the original DeepSeek V4 Flash-specific engine and model-focused
  direction.
- **llama.cpp / GGML** - GGUF conventions, quantization formats, and the broader
  local-inference engineering foundation.
- **TurboMind** - expert execution ideas and integration points used in the V100
  appliance experiments.
- **CUTLASS** - CUDA GEMM and tensor-core implementation guidance.

This repository is separate experimental work, but those projects shaped the
technical path and deserve explicit credit.
