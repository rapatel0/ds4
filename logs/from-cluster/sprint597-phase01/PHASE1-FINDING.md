# Sprint 597 Phase 1 finding: the promoted EP return crosses SYS, and SYS is the dominant decode cost

Date: 2026-06-11. Pod: llm/llamacpp-build-8gpu on gpu-01 (8x V100-SXM2-32GB,
driver 580.126.20). Rebuilt environment (Sprint 597 bootstrap pack
`/workspace/packs/ds4-appliance-full-tm-gated-s597`).

## Topology ground truth

`nvidia-smi topo -m` (archived: nvidia-smi-topo.txt) confirms the hybrid cube
mesh: every GPU has exactly 4 NVLink peers (2x NV1, 2x NV2) and 3 SYS peers
(PCIe + QPI/UPI across the two NUMA sockets, GPUs 0-3 vs 4-7). 12 of 28
undirected pairs are SYS: (0,5) (0,6) (0,7) (1,4) (1,6) (1,7) (2,4) (2,5)
(2,7) (3,4) (3,5) (3,6) — i.e. 24 of the 56 directed (dst,src) EP-return
copies. Every SYS pair has exactly two one-hop NVLink relay candidates
(see phase1-peer-copy-analysis.txt for the full relay table), so a static
no-SYS one-hop forwarding schedule (B2-C) exists for all 12 pairs.

## Microbench (isolated, one pair at a time)

`tools/s597-peer-copy-microbench.cu` (new standalone tool) times
copy_f32_kernel-style UVA remote loads (dst-side kernel, block=256, peer
access enabled as in engine/tp_runtime.cu) for all 64 (dst,src) pairs at
8/64/192/384/512 KiB. At the promoted EP-return payload (192 routes x 512
f32 = 384 KiB, grid 384 — exactly what the fixed-capacity route plan ships
every layer):

| class | per-copy (burst) | bandwidth |
|---|---|---|
| self | ~2.8 us | ~140 GB/s |
| NV2  | ~13.5 us | ~29 GB/s |
| NV1  | ~22 us | ~17.8 GB/s |
| SYS  | ~46-49 us | ~8.0-8.6 GB/s |

Isolated SYS is only ~2-3.5x slower than NVLink. The danger looked moderate
— until measured in situ.

## In-situ (one steady-state 32-slot replay window, unmodified promoted default)

nsys (cuda-graph-trace=node) over one 8-step serving batch captured exactly
19,264 EP-return copy kernels = 56 pairs x 43 layers x 8 steps
(phase1-nsys-insitu-attribution.txt). Mean per-copy kernel time:

| class | in-situ mean | vs microbench |
|---|---|---|
| NV2 | 11.1 us | matches (~13.5) |
| NV1 | 19.7 us | matches (~22) |
| SYS | **1,990 us** (range 686-3,094 by pair; per-instance min 90 us, max 3,597 us) | **~40x worse** |

The microbench CLASS RANKING holds in situ (NV2 < NV1 << SYS), but under
real concurrency the 24 SYS remote loads run simultaneously and saturate the
PCIe/QPI path: per-copy SYS cost explodes from ~47 us isolated to ~2 ms.

## Cost on the promoted path

Per layer, each dst rank runs its 7 return copies serially on its stream:
NV portion ~61 us, SYS portion ~5.5-6.5 ms. Per-dst serial EP-return cost is
4.9-6.5 ms/layer (worst: dst 2, 6.53 ms). Against the measured full-capture
replay of 10.11 ms/layer-step (Phase 0 Leg A steady state), the SYS remote
loads are ~55-65% of the entire decode step on the promoted default:
~280 ms/step on the worst rank (x43 layers) out of ~435 ms/step total.

**Verdict: the promoted EP return does cross SYS (24/56 directed copies) and
"nothing accounts for them" is confirmed — `peer_copy_sys_bytes=0` while
~0.25 s/step of SYS traffic flows. This is the single largest measured cost
in the decode step.** If B2-C one-hop NVLink relays replaced the 3 SYS
sources per dst (2 NV hops each, ~40 us per relayed pair in-situ NV class),
the EP return would drop from ~5.8 ms to ~0.2 ms/layer, bounding the step
at roughly 435 - ~240 = ~190 ms/step — a ~2.3x decode throughput headroom
(73.6 -> ~170 tok/s decode-domain) before any other B2 stage.

Caveat: per-pair SYS means vary widely (686-3,094 us) and are
congestion-coupled — they should be treated as a distribution, not stable
per-pair constants; scheduling changes (B2-D) will shift them.
