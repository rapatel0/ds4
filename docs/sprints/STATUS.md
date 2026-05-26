# DS4 V100 Appliance Status

Last updated: 2026-05-26

## Topline

Current bottleneck reference:
[`docs/architecture/DS4-V100-TP-EP-BOTTLENECKS.md`](../architecture/DS4-V100-TP-EP-BOTTLENECKS.md)
summarizes the measured bottlenecks, layer-by-layer hot paths, and experiments
already tried.

Latest NCCL status: Sprint 396 added `--algo nccl` to
`tools/ds4-v100-tp8-collective-workbench` and linked it against NCCL
`2.19.3`. All V100 modes passed correctness at `tokens=32` and `tokens=128`.
NCCL is materially faster than the current peer-copy doubling workbench:
`32`-token allreduce improved from `13.365976` to `4.513166` ms (`2.96x`),
`32`-token rs-ag from `31.431235` to `10.282541` ms (`3.06x`),
`128`-token reduce-scatter from `29.035444` to `6.076402` ms (`4.78x`), and
`128`-token allgather from `20.682822` to `6.142763` ms (`3.37x`). This is
strong enough to make serving-path NCCL integration the next TP/EP sprint.

Latest promoted TP/EP default: Sprint 395 promoted
`DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=1` in the launcher/profile path. It
keeps the existing CPU route-plan semantics but uses persistent pinned host
buffers and stream-ordered async H2D uploads for route offsets, route slots,
route weights, and the packed compact-MoE plan. Same-binary V100 HTTP A/B at
`32` requests / `32` slots / `256K` / `position=262080` / `32` generated
tokens preserved `32/32` response parity and readiness, kept summary first
token `83484`, improved server decode from `104.834948` to `107.092211` tok/s,
improved client generated throughput from `37.153198` to `37.239503` tok/s,
reduced route upload from `6.785109` to `4.736281` ms, reduced router D2H from
`1.016605` to `0.562918` ms, and kept `vram_failures=0` with `1746 MiB`
minimum free VRAM. This is a small but valid boundary cleanup; the next focus
returns to NCCL/collective work because average GPU utilization is still only
about `9.3%`.

Latest serving hardening status: Sprint 382 added permanent TP/EP VRAM
admission telemetry for the target `32` slot / `256K` appliance shape. The
launcher now has `DS4_V100_TP_EP_VRAM_REPORT` and
`DS4_V100_TP_EP_VRAM_MIN_FREE_MIB`, defaulting to a `64 MiB` free-memory
reserve. The full-layer smoke emits `tp_ep_vram_plan`, `tp_ep_vram`, and
`tp_ep_vram_summary` rows during resident startup, and the profile harness
captures aggregate VRAM fields in `summary.json`. V100 validation passed:
build succeeded, launcher `--print-command` showed
`--vram-report --vram-min-free-mib 64`, direct `32` slot / `256K` /
`position=262080` validation returned `returncode=0`, first token `54639`,
`66.136824` generated decode tok/s, `vram_failures=0`,
`vram_min_free_mib=1754`, and `vram_max_used_mib=30739`. A high-threshold
negative check with `--vram-min-free-mib 40000` failed cleanly at startup with
`rc=14` and `failures=8`.

Latest baseline status: Sprint 383 made the active-slot matrix VRAM-aware and
reran the target `32` slot / `256K` chat shape with GPU sampling and
`--vram-min-free-mib 64`. The combined V100 matrix is in
`/workspace/logs/sprint383-vram-aware-matrix-combined/`. Active requests
`1,4,8,16,32` all completed after adding inter-case cooldown, with
`vram_failures=0`, `vram_min_free_mib=1754`, and max sampled memory
`32398 MiB`. Client tok/s scaled from `1.321769` to `43.853691`, but server
decode stayed flat at `97.749438`, `97.092452`, `96.330654`, `92.690622`,
and `97.076706` tok/s; average GPU utilization stayed `8.24-9.29%`. The first
no-cooldown matrix failed on the second resident startup with CUDA OOM at
`cudaSetDevice`, and the first retry did the same on the next case, so the
matrix runner now has `--case-cooldown-seconds` for repeated server startups.
This confirms the next bottleneck is steady-state launch/synchronization and
GPU0-heavy orchestration, not active-slot admission or VRAM capacity.

Latest real-router status: Sprint 384 measured the quality-preserving
model-router compact-MoE serving path at the same `32` slot / `256K` shape.
The matrix in `/workspace/logs/sprint384-real-router-matrix/` completed
active requests `1,4,8,16,32` with `vram_failures=0`, `vram_min_free_mib=1754`,
and max sampled memory `32418 MiB`. Server decode was `80.934514`,
`81.231383`, `79.547736`, `76.816196`, and `81.505160` tok/s; `32`-request
client throughput was `38.554075` tok/s. This is slower than the Sprint 383
default/synthetic-route baseline, but it is the correct baseline for
intelligence-preserving DS4 serving. The extra cost shows up in the
HC-current FFN/router stage, around `85-88 ms` per all-layer decode step.

Latest real-router optimization status: Sprint 386 packed the compact-MoE
route plan into one H2D upload per destination GPU. Direct real-router
validation preserved first token `54639`, reduced route upload from Sprint
385's `44.079759` to `10.241125` ms, and improved generated decode from
`68.544741` to `74.838601` tok/s. The serving-shaped HTTP `32` request run
preserved first token `83484`, returned `32/32` HTTP 200 responses, improved
server decode from `85.792845` to `91.778174` tok/s, and reduced route upload
from `38.837019` to `6.796221` ms with `vram_failures=0`. Client aggregate
tok/s moved down from `42.427324` to `40.302457` in this single run, so the
result is a server-side decode/stage win rather than a proven full-stack HTTP
topline win. The remaining measured real-router hot substage is router
dense/select, still flat at about `27.8` ms in the HTTP `32` case.

Latest router-kernel diagnostic: Sprint 387 added a default-off
`--router-cublas-gate` / `DS4_V100_TP_EP_ROUTER_CUBLAS=1` path. Same-binary
direct A/B preserved first token `54639`, reduced router dense/select from
`33.591907` to `18.815270` ms, and improved generated decode from
`76.179292` to `79.718036` tok/s. Same-binary HTTP `32` request chat A/B also
preserved first token `83484` and reduced router dense/select from
`27.752540` to `4.959189` ms, but server decode improved only from
`94.952767` to `95.944290` tok/s while client generated tok/s regressed from
`44.579314` to `41.769369`. Keep cuBLAS router dense diagnostic-only; the
next promotion candidate needs to recover this local win inside a broader
fusion/scheduling boundary.

Latest route-planner diagnostic: Sprint 388 added a default-off
`--gpu-route-plan-gate` / `DS4_V100_TP_EP_GPU_ROUTE_PLAN=1` path that builds
expert offsets, route slots, route weights, and compact compose maps on-device
from GPU-selected experts. It preserves token parity but is rejected as a
default. Direct first token stayed `54639`, but generated decode regressed
from `76.179292` to `65.263520` tok/s and route-plan/upload time rose from
`10.190194` to `20.049537` ms. HTTP `32` request chat first token stayed
`83484`, but client tok/s regressed from `44.579314` to `39.283698` and
server decode from `94.952767` to `87.652515`; route-plan/upload rose from
`6.742906` to `14.474102` ms. The naive GPU planner removes router D2H but
adds P2P replication, small kernels, synchronization, and route-total readback.
Future route-boundary work should fuse planning with expert dispatch/compose
or eliminate per-layer host involvement completely, not move the current CPU
planner structure kernel-for-kernel onto GPUs.

Latest promoted TP/EP default: Sprint 389 promoted
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1` in
`tools/ds4-v100-run-appliance.sh`. This removes host-side diagnostic dense
output statistics from the production compressed/indexer projection path while
leaving an explicit `=0` opt-out; the permanent profile harness is aligned
and now has `--disable-skip-compressed-dense-stats` for control runs.
Same-binary V100 direct A/B at `32` slots / `256K` / `position=262080` /
`32` decode steps preserved first token `98751`,
improved generated decode from `91.869507` to `102.871437` tok/s, and reduced
compressed-KV sum from `3138.980697` to `1798.907552` ms. HTTP chat A/B at
`32` requests / `32` slots / `256K` / `32` generated tokens preserved first
token `83484`; all `32` generated token sequences matched and checksum stayed
`17913667583206000416`. Server decode improved from `89.709430` to
`103.758804` tok/s, client generated tok/s improved from `42.183007` to
`44.592824`, compressed-KV sum dropped from `5063.395601` to `2835.901361` ms,
average GPU utilization moved from `8.621875%` to `9.003289%`, and
`vram_failures=0` with `1746 MiB` minimum free VRAM in both runs.

Latest parity hardening: Sprint 390 added
`tools/ds4-v100-http-response-parity.py` as the standard HTTP artifact
comparator for TP/EP A/B promotions. It parses paired `response-NN.txt` files
with trailing `HTTP_STATUS:NNN` lines, compares HTTP status, generated token
sequence, choice token IDs, selected token, DS4 checksum, and generated text,
then emits JSON and exits non-zero on parity failure. It passes on the Sprint
389 control/candidate HTTP artifacts with `32/32` matched pairs and fails a
mutated-token negative fixture on `generated_token_sequence`. Future HTTP A/B
sprints should use this tool instead of ad hoc JSON snippets.

Latest throughput direction before memory hardening: Sprint 381 implemented
`--fp8-e5m2-kv-gate` as a default-off typed-KV format diagnostic. The row
layout stays block-128 with one E8M0 scale byte plus 128 FP8 payload bytes, so
E5M2 is not a capacity win over E4M3; it tests FP8 exponent/mantissa semantics
inside the existing TP/EP sharded KV path. V100 row validation passed for
`attn`, `attn_raw`, and `indexer` with `bad_values=0` and
`byte_mismatches=0`; E4M3 row regression also passed after the shared scale-byte
cleanup. Direct `32` slot / `256K` / 4-token A/B preserved checksum
`13373834059`, first token `98751`, and improved decode from `70.710875` to
`75.787866` tok/s. HTTP selected-token 4-token A/B returned `32/32` HTTP 200
for both control and candidate, preserved first token `45178`, improved client
tok/s from `17.212677` to `22.389190`, and reduced parsed compressed-KV sum
from `491.310011` to `442.415827` ms. The gate is still not promoted because
E5M2 has lower mantissa precision, validation is short, and one immediate HTTP
candidate run after control failed with CUDA OOM before readiness. The default
remains E4M3 until longer parity/soak and VRAM-margin work are done.

Latest E5M2 KV follow-up: Sprint 391 reran E5M2 at the real `32` slot /
`256K` real-router compact-MoE shape with the promoted skip-dense-stats
default and the permanent response parity comparator. The sprint also fixed a
profile-harness bug where direct-token-major mode did not inherit the promoted
skip-dense-stats default. Direct A/B preserved first token `98751` but moved
generated decode from `103.237368` to `102.152512` tok/s. HTTP chat A/B
preserved first token `83484`, passed `32/32` response parity pairs, improved
server decode from `101.206458` to `107.281060` tok/s, improved client
throughput from `46.115999` to `47.895831` tok/s, and reduced compressed-KV
sum from `2882.657866` to `2678.431998` ms with `vram_failures=0` and
`1746 MiB` minimum free VRAM. Keep E5M2 default-off for now because direct
decode is slightly negative and E5M2 still needs a broader multi-prompt
parity/soak before accepting the precision risk.

Latest multi-prompt soak: Sprint 392 added `--prompt-file` support to
`tools/ds4-v100-tp-ep-profile.py` and committed
`tests/v100_tp_ep_soak_prompts.jsonl` with `16` varied prompts. The profile
summary now records `prompt_file`, `prompt_count`, and `prompt_digest`. V100
multi-prompt E5M2 A/B at `32` requests / `32` slots / `256K` /
`32` generated tokens passed `32/32` HTTP responses and `32/32` permanent
response-parity pairs. E5M2 preserved first token `83484`; server decode was
effectively flat (`106.390802` to `106.483285` tok/s), client throughput moved
from `38.912861` to `39.774181` tok/s, compressed-KV sum moved from
`3343.550356` to `3301.691102` ms, and `vram_failures=0` with `1746 MiB`
minimum free VRAM. Keep E5M2 default-off: the broader prompt soak is
parity-clean, but the performance win is not material and the current E5M2
layout is not a capacity win over E4M3.

Latest serving readiness gate: Sprint 393 added
`tools/ds4-v100-http-readiness-check.py` as the standard one-case artifact
checker for TP/EP serving promotion gates. It validates response files,
`summary.json`, `status.json`, generated-token sequence length, `32` slots,
`256K` context, resident KV/HC metadata, typed DS4 KV gates, compact MoE,
token-match metadata, DS4 checksums, GPU utilization samples, prompt soak
metadata, and VRAM admission. It passes on the Sprint 392 control artifact
with `32/32` HTTP 200, `106.390802` server decode tok/s, `38.912861` client
generated tok/s, `9.772727%` average GPU utilization, first token `83484`,
`vram_failures=0`, and `1746 MiB` minimum free VRAM. It also passes on the
Sprint 392 E5M2 candidate artifact with `106.483285` server decode tok/s and
fails non-zero on a mutated response fixture that sets `token_mismatch=1` and
removes the checksum. Future default promotions should attach both response
parity and readiness summaries.

Latest router-boundary diagnostic: Sprint 394 added default-off
`--router-hash-fast-gate` / `DS4_V100_TP_EP_ROUTER_HASH_FAST=1`. The gate uses
a hash-specific router select kernel that evaluates only the six hash-row
experts instead of computing probabilities for all `256` experts first. It is
correct but not promotable. Same-binary V100 HTTP A/B at the real-router
compact-MoE `32` request / `32` slot / `256K` / `32` generated-token shape
passed `32/32` response parity and readiness on both sides. Server decode moved
from `106.900859` to `107.274556` tok/s and client throughput from
`37.231411` to `38.262372` tok/s, but the targeted router select boundary only
moved from `27.766750` to `27.683134` ms, HC-current FFN/router moved from
`36.211953` to `36.287395` ms, scaffold decode regressed from `289.821429` to
`293.484520` ms, and compressed-KV sum regressed from `3285.935154` to
`3317.395070` ms. Keep the gate diagnostic-only; the router bottleneck is not
primarily the extra non-hash probability work inside the select kernel.

Current active steering source: `TEMP_THROUGHPUT_PROMPT.md`. Sprint 380
implemented S-F `--tp-experts-ab-gate` as a permanent measurement driver,
`tools/ds4-v100-tp-experts-ab.py`, and closed the immediate topology decision:
do not integrate TP-sharded experts into serving yet. The EP8 direct serving
control at the target shape measured `66.569095` direct decode tok/s (`54639`
first token, `18.220610` ms EP, `22.522762` ms compose). TP8 MXFP4 still fails
correctness at `96/192/384` routes with `378153/756305/1512469` NaNs. TP4 is
correct at all three tiers and has compute speedup, but total speedup is only
`1.055x/0.891x/0.927x`; simple output reduction/compose erases the win at the
larger route tiers. A future TP expert sprint must prototype a better fused
TP4 reduction/compose boundary before serving integration.

The near-term
performance queue remains isolated default-off gates, same-binary V100 A/B,
and a strict promote/reject decision per gate. S-B async output and S-A CUDA
graph replay were rejected. S-C batched paged attention row planning was
closed as a diagnostic-only redirect because pending typed-history reloads were
already `0` in the observed compressed/indexer samples. S-D compact MoE is now
promoted for model-router compact compose. S-E fused gated-SiLU is closed
diagnostic-only: the generic epilogue changes tokens, and the new DS4-clamped
ABI needs a resident serving precheck fix or a narrower parity harness before
promotion can be considered. S-F TP-sharded experts is closed measurement-only:
TP8 is still numerically invalid, and TP4 reduction/compose erases the compute
win. S-G E5M2 KV is closed diagnostic-only pending longer parity and VRAM
admission work.

Latest TP/EP format status: Sprint 374 built and ran the V100 workbench for
the Sprint 373 INT8 candidate shapes. The copied tc-grid INT8 kernels are
numerically acceptable but not performance candidates for the BF16 attention
compressor GEMMs at `M=32,K=4096`: for `N=128`, cuBLAS FP16 measured
`0.009250 ms` while best tc-grid INT8 (`v12s_ks8+zero`) measured
`0.042721 ms`; for `N=64`, cuBLAS measured `0.008803 ms` while best tc-grid
INT8 measured `0.036673 ms`. Do not wire tc-grid INT8 into
`attn_compress_{kv,gate}.weight`. The next format/kernel path should either
adapt the vLLM/TurboMind SM70 small-M GEMM registry for this exact shape or
fuse the compressor dense boundary with adjacent state/emit work. Sprint 373
remains useful as the memory audit: scoped INT8+scale would save
`280608768` bytes aggregate, but the measured tc-grid compute path is slower
than the FP16 tensor-op baseline.

Latest TP/EP optimization status: Sprint 372 added an opt-in gate to skip
host-side dense-output statistics in the compressed-KV projection path:
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1`. This is diagnostic
work being removed from the serving path, not a dtype change. Direct
token-major `32` slot / `256K` / `32` step validation kept the same first
token (`98751`), improved scaffold decode from `100.739521` to `117.463961`
tok/s, and reduced parsed compressed-KV time from `3141.768079` to
`1789.795027` ms. Full chat A/B at `32` requests / `32` slots / `256K` /
`position=262080` improved client tok/s from `51.345855` to `58.923892` and
server decode tok/s from `99.748339` to `117.340768`. The gate remains
default-off until chat/token parity is compared deterministically; selected
token parity is clean.

Latest TP/EP metrology status: Sprint 369 added opt-in GPU utilization
sampling to `tools/ds4-v100-tp-ep-profile.py`. Passing
`--gpu-sample-interval-ms N` now writes `gpu_util.csv` and adds aggregate plus
per-GPU utilization/memory fields to `summary.json` for both HTTP serving and
direct token-major profiles; the default `0` keeps the sampler disabled. A
V100 `/v1/chat/completions` smoke at `32` configured slots, `4` active
requests, `4` tokens/request, `256K` context, and `position=100000` returned
`4/4` HTTP 200 with `coalesced_batch_size=4`, server decode `99.340235` tok/s,
average GPU utilization `8.412879%`, and max GPU utilization `39%`. Per-GPU
averages showed GPU0 at `27.090909%` while peers were mostly `3-12%`, giving
the next active-slot/scheduling work concrete imbalance evidence in the normal
profile artifact.

Sprint 370 added `tools/ds4-v100-tp-ep-active-slot-matrix.py`, a reusable
driver that runs the TP/EP profile harness over active-request cases and writes
aggregate `active_slot_matrix.tsv` / `active_slot_matrix.json`. The V100 smoke
matrix used `32` configured slots, `256K` context, `position=100000`,
`2` tokens/request, and active request cases `1,4`. It passed both cases:
`1/1` and `4/4` HTTP 200, with coalesced batch sizes `1` and `4`. Server
decode stayed flat (`101.842964` to `101.159316` tok/s) and average GPU
utilization stayed flat (`8.341667%` to `8.333333%`). This validates the
matrix harness and strengthens the next question: whether the full 1/4/8/16/32
matrix shows any active-slot scaling before we invest in active-slot compaction
or deeper dense/state kernel fusion.

Sprint 371 ran that full matrix at the target long-context chat shape:
`32` configured slots, `256K` context, `position=262080`, and `32`
tokens/request. All cases passed and coalesced correctly:

| Active requests | HTTP 200 | Client tok/s | Server decode tok/s | Avg GPU util |
|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.584552 | 98.230713 | 10.264286% |
| 4 | 4/4 | 6.430512 | 99.991505 | 10.200000% |
| 8 | 8/8 | 12.450978 | 97.865480 | 9.958333% |
| 16 | 16/16 | 24.557272 | 97.446076 | 9.840278% |
| 32 | 32/32 | 50.694229 | 98.768134 | 10.317857% |

Interpretation: client aggregate tok/s scales because the fixed batch cost is
amortized over more active responses, but the underlying server decode rate and
GPU utilization are flat. Active-slot compaction is still useful for low
occupancy, but it will not fix the full 32-slot topline. The next optimization
target should be full-occupancy kernel/state work: compressed/indexer dense
projection, attention projection/state, and GPU0-heavy staging/imbalance.

Latest TP/EP typed-KV serving status: Sprint 347 added a direct non-server
profile mode to the permanent profiler harness:
`tools/ds4-v100-tp-ep-profile.py --run-mode direct-token-major`. It invokes
`tools/ds4-v100-tp-ep-full-layer-smoke` directly with the same resident
32-slot / 256K typed-KV serving flags as the HTTP path, then writes command,
stdout/stderr, summary JSON, and parsed top-kernel TSV artifacts. Direct V100
no-profiler validation passed at `32` slots / `256K` / `2` decode steps with
`64` generated tokens, `83.882587` generated tok/s decode, `91.958152`
continuation tok/s decode, and finite output-head results. Direct windowed
`nvprof` also passed, emitted profiler start/stop markers, and produced usable
kernel rows: TurboMind SM70 FP4 HMMA (`46.028956 ms`, `172` calls), CUTLASS
WMMA FP16 (`14.949741 ms`, `720` calls), dense input fill
(`14.745042 ms`, `128` calls), compressor store (`12.823440 ms`, `124`
calls), and `bf16_dense_kernel` (`7.437483 ms`, `1` call) lead the scoped
window. A broad direct trace also produced rows and confirmed BF16/F8 unpack,
gather, dense-fill, cast, compressor, CUTLASS, and TurboMind kernels are all
active. The current measured bottleneck is not missing tensor-core dispatch;
it is current-HC/input staging and transform fragmentation:
`sum_hc_current_input_ms=622.442653` out of `sum_decode_ms=762.971220` in the
direct no-profiler run. Next work should fuse or bypass this staging path, then
rerun direct profiler plus HTTP serving A/B.

Sprint 348 tested the first bypass attempt for that staging path:
`--tp-hc-current-input-peer-gather-gate` /
`DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER=1`. The gate lets every TP rank
build its own full current vector from all eight current shards and skips the
old GPU0 full-current broadcast. It is correct but rejected for performance.
On the V100 direct 32-slot / 256K / 2-step A/B, control measured
`87.263615` generated tok/s decode, `100.446187` continuation tok/s decode,
`733.409911` sum decode ms, and `596.248809` HC-current ms. Peer gather
measured `67.495350`, `80.223389`, `948.213473`, and `801.525057`
respectively, with finite output head in both cases. The next optimization
should therefore target HC control computation/synchronization or fuse the
split/norm/fill chain, not naive all-rank peer gathering.

Sprint 349 tested the synchronization part of that target. The new
`--tp-hc-current-input-stream-sync-gate` /
`DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1` keeps the layout unchanged but
runs central GPU0 HC-current control kernels on rank 0's stream and uses
stream-scoped barriers where the old path used GPU0 device-wide barriers.
Direct 32-slot / 256K / 2-step A/B improved generated decode throughput from
`74.841520` to `81.190638` tok/s and reduced HC-current time from
`711.608991` to `647.492171` ms. HTTP 32-request / 32-slot / 256K / 2-token
A/B also stayed correct (`32/32` HTTP 200) and improved server generated
throughput from `82.573137` to `83.813937` tok/s and decode throughput from
`97.500352` to `98.859925` tok/s. This gate is now promoted as the launcher
default. The next optimization target remains the HC control/fill chain
itself, because synchronization scope only moves a small part of the serving
topline.

Sprint 350 corrected the interpretation of that hot timer. The direct summary
now splits HC-current substages, and a 32-slot / 256K / 2-step V100 run with
stream sync enabled passed with `92.630324` generated tok/s decode and finite
output head. The measured HC-current substages were only `83.066250` ms total:
seed `2.485326`, attention-HC mix `42.340819`, split `1.245295`, current
gather `6.960973`, FFN/router `1.784974`, and fill/pack `28.248863`. The old
`sum_hc_current_input_ms` field was `557.301289` ms, so that label is broader
than HC-current and includes true-attention/compressed-KV prefix work before
the EP/dense/compose timer begins. The next optimization target should be that
true-attention/compressed-KV prefix, especially compressed projection/store and
dense-fill/WMMA fragmentation, not more HC-current gather work.

Sprint 351 split that true-attention/compressed-KV prefix. The V100 direct
32-slot / 256K / 2-step run passed with finite output head, `83.265760`
generated tok/s decode, and `99.612333` continuation tok/s decode. The old
broad `sum_hc_current_input_ms` was `626.823138` ms, and the measured prefix
stages now account for `626.787298` ms of it: compressed KV projection/store
`228.813152`, attention projection `170.865666`, attention state
`105.654904`, HC-current `85.249101`, raw/window read `34.932798`, and typed
history load `1.271677`. The next optimization target is therefore compressed
KV projection/store fragmentation first, then attention projection/state.

Sprint 352 split compressed-KV internals and corrected the emitted-row test
shape. The old direct default `position=100000` emits zero compressed rows, so
the store path must be tested at an emitting position. A one-token boundary
run at `position=262143` passed with `41` emitted compressed layers,
`81.647302` generated tok/s decode, `391.929670` sum decode ms, and
`129.990107` ms in the pre-EP compressed-KV stage. The internal dominant
costs were indexer dense `36.615896` ms, attention dense `24.659453` ms,
attention state/emit `24.362932` ms, combined input fill `16.776362` ms, and
indexer state/emit `9.007686` ms. Suppressing both compressed and indexer
typed stores was flat: `81.733945` generated tok/s decode and `128.338783`
ms compressed-KV. Typed stores are therefore not the next lever; target
shared/fused compressor-indexer input fill and compressor state/emit work.

Sprint 353 implemented that shared-fill experiment as an opt-in TP/EP gate:
`--true-ds4-compressed-kv-fused-input-fill-gate` /
`--fused-compressed-input-fill`. The fused path reads each ratio-4 rank's
current vector once and writes the five current-derived compressor/indexer
half-input buffers in one kernel. V100 same-binary emitted-row A/B at `32`
slots / `256K` passed with identical output token `54639` and finite output
head. Control measured `79.011931` generated decode tok/s and `130.391665`
ms pre-EP compressed-KV time; fused-fill measured `80.534845` tok/s and
`129.781758` ms. The fused path was selected on all `21` ratio-4 layers, but
the stage reduction was under `1 ms`, so this remains opt-in and is not
promoted as a default. The next lever should be compressor/indexer state/emit
fusion rather than more fill-only work.

Sprint 354 tested the first narrow state/emit fusion:
`--true-ds4-compressed-kv-fused-rope-round-gate` /
`--fused-compressed-rope-round`. It combines compressed-row RoPE and the
following F16-round pass for emitted rows. The V100 same-binary emitted-row
A/B at `32` slots / `256K` passed with identical output token `54639`.
Control measured `79.810167` generated decode tok/s and `130.520098` ms
pre-EP compressed-KV time. Fused RoPE+round selected `41` emitted compressed
layers and measured `79.344207` tok/s and `130.382524` ms. Attention
state/emit improved slightly (`24.680003` to `24.352357` ms) and indexer
state/emit also moved slightly (`9.003129` to `8.880899` ms), but total
decode regressed within noise. This gate remains diagnostic-only; the next
state/emit work should target pooling+normalization or store+pooling rather
than RoPE+round alone.

Sprint 355 tested that larger pooling+normalization boundary with
`--true-ds4-compressed-kv-fused-pool-norm-gate` /
`--fused-compressed-pool-norm`. The fused kernel computes each emitted
compressor row into shared memory, normalizes it in the same block, and writes
only the normalized row. V100 same-binary emitted-row A/B at `32` slots /
`256K` passed with identical output token `54639`. Control measured
`81.189757` generated decode tok/s, `131.016911` ms pre-EP compressed-KV
time, and `130.510967` ms compressed-KV sum. Fused pool+norm selected `41`
emitted layers and measured `81.687107` tok/s, `128.201681` ms pre-EP
compressed-KV time, and `127.736989` ms compressed-KV sum. This is a real
stage-level win but only `+0.61%` topline in one run, so it remains opt-in
pending repeat/combination testing with fused input fill.

Sprint 356 made the compressed-fusion gates reachable from the TP/EP serving
launcher and profile harness. New default-off env vars are
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL`,
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND`, and
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM`; the deploy env
example documents them. The launcher `--print-command` proof includes both
fused input-fill and fused pool-norm flags. V100 direct combined A/B at `32`
slots / `256K` / emitted-row `position=262143` passed with identical output
token `54639`: control measured `80.511365` generated decode tok/s,
`130.812593` ms pre-EP compressed-KV time, and `130.329162` ms compressed-KV
sum; input-fill + pool-norm measured `81.311102` tok/s, `128.988052` ms
pre-EP compressed-KV time, and `128.467170` ms compressed-KV sum. Keep the
gates opt-in pending either repeated direct A/B or an emitted-row HTTP profile
mode that avoids prompt-prefill position ambiguity.

Sprint 357 added that emitted-row HTTP profile mode:
`tools/ds4-v100-tp-ep-profile.py --run-mode http --http-endpoint selected-token`.
The harness now targets `POST /v100/selected-token` without prompt prefill,
passes `DS4_V100_TP_EP_POSITION`, and parses TP/EP compressed-KV timing lines
from server output. V100 selected-token HTTP A/B at `32` slots / `256K` /
`position=262143` returned `32/32` HTTP 200 responses for both variants and
exercised all `41` emitted compressed layers. Control measured
`127.697384` ms compressed-KV sum; fused input-fill + pool-norm measured
`123.651985` ms. Client one-token tok/s was flat (`19.739916` vs
`19.719484`) because this diagnostic endpoint is dominated by HTTP
orchestration. Keep compressed fusions opt-in; the next useful gate should
amortize request overhead with a longer serving A/B or reduce the remaining
state/emit fragmentation in direct decode.

Sprint 358 ran that longer selected-token HTTP A/B. The first attempted shape
(`position=262143`, `32` tokens) correctly failed after one generated token
because the next position is outside `ctx=262144`; the valid amortized run
starts at `position=262112` and generates `32` tokens/request. At `32` slots /
`256K`, all valid variants returned `32/32` HTTP 200 responses and preserved
first token `109328`. Control measured `71.818394` client tok/s,
`3506.921796` ms compressed-KV sum, and `98.772310` scaffold decode tok/s.
Combined input-fill + pool-norm measured `72.297469`, `3509.986423`, and
`98.505291`, so it is not promotable. Pool-norm only measured `73.052883`,
`3474.878472`, and `97.552747`: promising for client wall and compressed-KV
sum, but conflicting with the scaffold decode proxy. Keep all compressed
fusions default-off; next work should be a repeated/direct multi-step A/B for
pool-norm or a deeper state/emit fusion.

Sprint 359 resolved the Sprint 358 client-vs-scaffold disagreement with a
direct non-HTTP multi-step A/B at the same valid long-context window:
`32` slots / `256K` / `position=262112` / `32` decode steps. Control measured
`95.851552` generated decode tok/s, `74.814127` generated wall tok/s,
`3521.094409` ms compressed-KV sum, and first token `98751`. Fused pool-norm
measured `97.619138`, `76.140370`, `3458.469603` ms, and the same first token
with finite output head. This is a clean `+1.84%` decode / `+1.77%` wall win,
so `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM` is now promoted as
the TP/EP launcher default. Fused input-fill and fused RoPE+round remain
default-off diagnostics.

Sprint 360 validated that promotion through the actual TP/EP launcher path.
`tools/ds4-v100-run-appliance.sh --print-command` includes
`--true-ds4-compressed-kv-fused-pool-norm-gate` by default when
`DS4_V100_SERVE_MODE=tp-ep`, without setting the pool-norm env var. A
launcher-started selected-token HTTP run with the required true-attention typed
KV gates returned `32/32` HTTP 200 responses at `32` slots / `256K` /
`position=262112` / `32` tokens/request, measured `73.289956` client generated
tok/s, and logged `187` fused pool-norm compressed rows with `0` fused
input-fill rows. The first selected token was `109328`. The initial bare
launcher attempt returned HTTP 500 because it omitted the full TP/EP
true-attention gate set; this does not affect the default-promotion result.

Sprint 361 reran the launcher path through `/v1/chat/completions`, comparing
an explicit pool-norm-off control against the launcher default. Both runs used
`32` concurrent requests and `8` generated tokens/request at `32` slots /
`256K`; both returned `32/32` HTTP 200 and first token `24893`. Control
measured `24.280060` client generated tok/s with `0` fused pool rows. The
default-pool run measured `24.118711` client generated tok/s with `126` fused
pool rows. This proves the promoted default is active and stable through the
full chat endpoint, but does not show a short-chat topline gain; the result is
`-0.66%` at a shape where tokenization/prefill/HTTP overhead dominate.

Sprint 362 aligned the permanent profile harness with that launcher default.
`tools/ds4-v100-tp-ep-profile.py` now omits
`DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM` by default in HTTP
mode, so `tools/ds4-v100-run-appliance.sh` supplies the production default.
New `--disable-fused-compressed-pool-norm` forces the control behavior for
A/B runs, and it is mutually exclusive with `--fused-compressed-pool-norm`.
V100 selected-token proof returned `1/1` HTTP 200 in both modes: default
command had the pool-norm gate and logged `40` fused pool layers; disabled
command omitted the gate and logged `0` fused pool layers.

Sprint 363 tested the next wider emitted-row fusion,
`--true-ds4-compressed-kv-fused-pool-norm-rope-round-gate`, which combines
compressor pooling, RMSNorm, compressed-row RoPE, F16 rounding, and final row
write into one opt-in kernel. It is correct but not promotable. The V100
direct 32-step A/B at `32` slots / `256K` / `position=262112` preserved first
token `98751` and finite output in both variants, but regressed from
`95.908399` to `95.463298` generated decode tok/s and increased
compressed-KV sum from `3460.932833` to `3470.682826` ms. A one-token
`nvprof-window` boundary run showed lower compressed-KV sum
(`142.456129` to `140.699321` ms) but the full direct decode result is the
decision gate. Keep the fused pool+norm+RoPE+round path diagnostic-only.

Sprint 364 tested direct compressed input fill, bypassing the per-rank
`d_current_full` staging copy and having each rank's half-fill kernels read
`hc->d_attn_normed` directly. The gate is legal and correct but clearly slower.
Same-build one-token emitted-row A/B at `32` slots / `256K` preserved first
token `54639` and finite output, but compressed-KV sum regressed from
`126.724613` to `260.365841` ms. The regression is localized to peer-read
input fill: attention input fill moved from `12.587939` to `84.142732` ms, and
indexer input fill from `3.754212` to `65.145245` ms. Keep direct compressed
input fill diagnostic-only and preserve local per-rank reads.

Sprint 365 tested local fused attention input fill, preserving the staged
per-rank current vector while replacing the two attention compressor half-fill
launches with one two-destination kernel. The gate is correct and selected
`1312` fused attention rows in the full 32-step direct run. Direct A/B at
`32` slots / `256K` / `position=262112` moved generated decode tok/s from
`94.237924` to `94.396298` and compressed-KV sum from `3532.911129` to
`3499.213977` ms. The same selected-token HTTP shape regressed client tok/s
from `72.886325` to `70.674037` and compressed-KV sum from `3493.666516` to
`3506.331429` ms. Keep fused attention input fill diagnostic-only; the next
TP/EP lever should be a larger compressed/indexer dense projection or
attention projection/state boundary.

Sprint 366 promoted compressed dense event waits as the next TP/EP serving
default. Instead of synchronizing rank streams on the host after compressed
attention/indexer input fill, the gate records per-rank CUDA events and makes
the dense stream wait on them. Direct 32-step A/B at `32` slots / `256K` /
`position=262112` preserved token `98751` and improved generated decode
throughput from `96.214306` to `99.093248` tok/s, wall throughput from
`75.215206` to `76.897975` tok/s, and compressed-KV sum from `3431.137744` to
`3127.236790` ms. Selected-token HTTP preserved token `109328`, returned
`32/32` HTTP 200, and improved client throughput from `71.833757` to
`74.432464` tok/s while reducing compressed-KV sum from `3437.636456` to
`3137.755187` ms. `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT`
is now default-on and can be disabled for control runs.

Sprint 367 validated that promoted default through the real chat endpoint at
a decode-heavy long-context shape. The selected-token start position
`262112` is invalid for chat because prompt prefill plus `32` generated tokens
reaches `position=262144`, so the valid run used `position=262080`. At `32`
slots / `256K`, `32` concurrent `/v1/chat/completions` requests, and `32`
generated tokens/request, both variants returned `32/32` HTTP 200, coalesced
batch `32`, generated `1024` tokens, and preserved first token `89340`.
Disabling event waits measured `50.648397` client tok/s, `81.426024` server
wall tok/s, and `96.116667` server decode tok/s. The default event-wait path
measured `52.022782`, `83.891024`, and `99.521680` respectively, with
compressed-KV sum reduced from `5100.469710` to `4681.992882` ms.

Sprint 368 converted that invalid chat boundary from a backend decode failure
into explicit HTTP admission. The TP/EP server now checks
`start_position + prompt_prefill_steps + requested_decode_steps <= 262144`
before slot assignment and GPU decode. The Sprint 367 invalid shape
(`position=262112`, `32` requested tokens, `16` prompt-prefill steps) now
returns HTTP 400 with `context_window_exceeded`, `final_position=262160`, and
no `tp_ep_http_decode_failed` log line. The valid long-context shape
(`position=262080`, `32` concurrent chat requests, `32` generated
tokens/request) still returns `32/32` HTTP 200, coalesces batch `32`, preserves
first token `89340`, and measures `51.069220` client tok/s,
`82.089657` server wall tok/s, and `98.301727` server decode tok/s.

Current TP/EP implementation status: the forward path is TP8/EP8 only, with
PP/layer-split work frozen as a baseline. The resident TP/EP backend keeps the
TP runtime, sharded KV, rank buffers, TurboMind API handles, active MXFP4
expert bindings, dense FP16 cache, shared dense ops, source-scheduled peer
copies, skip-self compose, and multi-copy streams resident across the
token-major all-layer loop. Sprint 275 added a repeatable sustained-serving
artifact wrapper. Sprint 276 added a TP/EP-only resident HTTP harness exposing
`/health`, `/v100/status`, `/metrics`, and `POST /v100/selected-token` without
using the PP replay server. Sprint 277 wired that server into
`tools/ds4-v100-run-appliance.sh` as `DS4_V100_SERVE_MODE=tp-ep`. Sprint 278
added the sustained HTTP matrix driver. Sprint 279 made the Kubernetes example
point at the TP/EP serving path and added GPU-utilization capture around the
HTTP matrix. Sprint 280 extended that harness to resident multi-request
metrology and added cumulative `/v100/status` plus `/metrics` counters. The
current V100 matrix at `32` slots / `256K` with three generation requests per
case reports `751.114404` wall generated tok/s and `760.078310` wall
continuation tok/s for `32` tokens/request, and `762.277426` wall generated
tok/s and `766.925593` wall continuation tok/s for `64` tokens/request.
Sprint 281 then exposed EP/dense/compose stage timing through HTTP artifacts.
The stage-metric rerun reports `742.897231` and `739.612937` wall generated
tok/s for 32-token and 64-token cases. In the 64-token case, EP is
`2663.985462 ms`, compose is `3626.650073 ms`, and compose-copy alone is
`2569.208878 ms`, or `70.8%` of compose time. Both cases return aggregate
`96/96` token match. GPU utilization remains low, peaking at `33-40%`, so the
remaining performance gap is compose-copy movement/synchronization; true HTTP
request coalescing remains the serving-semantics gap. Sprint 282 added and
promoted event-wait compose copy. Same-binary 64-token serving A/B improves
wall generated throughput from `752.669235` to `771.276064` tok/s and wall
continuation throughput from `757.403683` to `775.670776` tok/s, with
aggregate `96/96` token match. The current promoted TP/EP serving default is
therefore `DS4_V100_TP_EP_COPY_EVENT_COMPOSE=1`. Sprint 283 rechecked FP16 EP
return under that event-wait path. It remains rejected: same-binary 64-token
serving A/B regresses from `766.883263` to `635.936079` wall generated tok/s
and from `997.165341` to `793.283316` decode generated tok/s. The diagnostic
toggle exists as `DS4_V100_TP_EP_RETURN_FP16`, but the serving default remains
`0`. Sprint 284 added and promoted compact route-compose. Same-binary
64-token serving A/B improves wall generated throughput from `711.177884` to
`791.453850` tok/s and wall continuation throughput from `719.489689` to
`796.894336` tok/s, with aggregate `96/96` token match. The 32-token compact
sanity run reaches `802.701663` wall generated tok/s and `813.475877` wall
continuation tok/s. The current promoted TP/EP serving defaults are therefore
`DS4_V100_TP_EP_COPY_EVENT_COMPOSE=1`,
`DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1`, and
`DS4_V100_TP_EP_RETURN_FP16=0`. Sprint 285 re-established the promoted normal
launcher topline: at `32` slots / `256K` / three resident requests, the V100
pod reports `771.036527` wall generated tok/s and `781.922821` wall
continuation tok/s for `32` tokens/request, and `794.694599` wall generated
tok/s and `799.391755` wall continuation tok/s for `64` tokens/request. Both
cases return aggregate `96/96` token match. Sprint 286 added true TP/EP HTTP
request coalescing for the selected-token harness. The server now admits
concurrent generation requests into one resident decode batch, reports
`generation_batches` / `coalesced_requests`, and returns per-client responses
with `coalesced_batch_id` and `coalesced_batch_size`. The new serving-shaped
matrix at `32` slots / `256K` with `32` concurrent HTTP requests forms one
`coalesced_batch_size=32` batch in both cases: `32` tokens/request reports
`721.446441` wall generated tok/s and `950.363316` decode generated tok/s,
and `64` tokens/request reports `787.316214` wall generated tok/s and
`1030.972573` decode generated tok/s. Both cases return aggregate `32/32`
token match. This is the current practical-serving semantic baseline; the
next gap is replacing the selected-token harness with the real prompt/token
API and bucketed admission queues. Sprint 287 then added bucketed admission:
mixed concurrent requests with different `max_tokens` are queued into
same-length decode batches instead of rejected. A V100 run with 32 concurrent
requests alternating `32,64` tokens forms two batches, reports
`bucketed_requests=16`, returns `32/32` token match with zero rejected
requests, and reaches `387.877251` wall generated tok/s /
`510.747848` decode generated tok/s over `1536` admitted client tokens. A
uniform 32-request sanity run still forms one full 32-slot batch and reports
`759.490446` wall generated tok/s / `991.405750` decode generated tok/s.
Sprint 288 added a TP/EP diagnostic `POST /v1/completions` endpoint on top of
the same coalesced and bucketed resident decode path. The response is
OpenAI-shaped and carries usage/choices fields, but explicitly marks
`ds4_v100.diagnostic=true` because prompt prefill, tokenizer text output, and
output-head token selection are not yet wired in this TP/EP endpoint. The V100
completion-shaped mixed run at `32` slots / `256K` with 32 concurrent requests
and pattern `32,64` forms two 16-client buckets, returns `32/32` token match,
and reports `384.581100` wall generated tok/s / `505.797315` decode generated
tok/s. The selected-token regression sanity still forms one 32-client batch
and reports `726.823991` wall generated tok/s / `944.195924` decode generated
tok/s. Sprint 289 then added a TP/EP-only vocab-sharded output-head gate. It
loads real `hc_head_fn`, `hc_head_base`, `hc_head_scale`, `output_norm.weight`,
and real BF16 `output.weight` vocab shards across all 8 GPUs. At `32` slots,
the scalar BF16 output projection passes with token `26803`, `2192.810195 ms`
cold projection time, `7.593408 ms` worst per-GPU projection-kernel time, and
`6.070330 ms` host top-1 reduction. The BF16-to-FP16 cuBLAS diagnostic also
passes with the same token, but is slower in this cold gate:
`2217.599099 ms` projection time and `22.116352 ms` worst per-GPU kernel time.
This makes the output-head layout operational, but `/v1/completions` still
needs final HC carried into the output head before it can emit real model text.
Sprint 290 then converted that cold output-head gate into a resident TP/EP
gate and added GPU-side per-shard top-1 reduction. With full-logit host
readback, 32 slots measured `15.980438 ms` total and `2002.448256`
output-head tok/s. After device-side shard top-1, 32 slots measured
`8.528343 ms` total, `7.474198 ms` projection wall time,
`7.427597 ms` worst per-GPU projection-kernel time, `0.211761 ms`
device-top1/readback time, and `3752.194257` output-head tok/s. The 16-slot
and 64-slot resident gates also pass, at `3563.755123` and `3877.433386`
output-head tok/s respectively. Full-logit readback is rejected for serving;
the next gap remains carrying final HC `[slots,4,4096]` through the TP/EP
token-major loop and feeding this resident output-head primitive.
Sprint 291 added a TP/EP-only final-HC carry scaffold behind
`--final-hc-carry-gate`. Each GPU owns `[slots][4][512]` F32, giving a logical
`[slots][4][4096]` output-head input. This is explicitly a proxy carry layout,
not yet true DS4 HC row semantics. The 1-token all-layer V100 gate passes with
`43/43` invocations, `75.554825 ms` summed decode, `2.100054 ms` summed
final-HC carry cost, and `423.533507` decode tok/s. The matching control pass
without HC carry reports `70.923652 ms` summed decode and `451.189400` decode
tok/s. A 4-token carry run passes `172/172` invocations with `8.113938 ms`
summed final-HC carry cost, `712.985252` aggregate decode tok/s, and
`960.823272` continuation decode tok/s. The next gap is replacing the proxy HC
expansion with true DS4 HC row semantics, then feeding the resident output head
from this sharded HC state.
Sprint 292 wired that sharded HC carry into a resident TP/EP output-head
service and exposed diagnostic selected-token metadata through HTTP
completions. The new `--diagnostic-output-head` flag implies HC carry, loads
real output controls plus BF16 vocab shards once, gathers per-rank
`[slots][4][512]` HC shards into logical `[slots][4][4096]`, runs the
vocab-sharded output head, and returns `selected_token` / `selected_logit` in
`ds4_v100`. The launcher now accepts
`DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=1`, and the HTTP bench has
`--diagnostic-output-head`. Direct 32-slot V100 validation reports
`8.903469 ms` output-head time with token `122445`. A full launcher-level
32-concurrent completions run forms one 32-request batch, returns `32/32`
HTTP 200 responses with diagnostic output-head metadata, and reports
`8.586224 ms` output-head time, `7.592902 ms` projection time,
`0.341194 ms` top-1 time, `158.576748` wall generated tok/s, and
`294.331849` decode generated tok/s for the 1-token diagnostic case. This is
still proxy-HC diagnostic serving, not real DeepSeek text serving.
Sprint 293 replaced the arbitrary final proxy HC expansion with an opt-in
TP/EP DS4-style HC final-expand diagnostic path. The new
`--tp-hc-final-expand-gate` / `DS4_V100_TP_EP_HC_FINAL_EXPAND=1` path loads
real `blk.N.hc_ffn_fn`, `blk.N.hc_ffn_base`, and `blk.N.hc_ffn_scale` controls
for all 43 layers, computes HC split from gathered sharded HC on GPU0, and
expands the next sharded `[slots][4][512]` HC state per rank. Direct 32-slot
V100 validation passes with `sum_final_hc_ms=25.407638` and
`8.750574 ms` output-head time. A full launcher-level 32-concurrent
completions run forms one 32-request batch, returns `32/32` HTTP 200 responses
with `proxy_hc=0`, and reports `160.904882` wall generated tok/s /
`271.342877` decode generated tok/s for the 1-token diagnostic case. This is a
better HC semantic bridge than Sprint 292, but it still lacks the full DS4 HC
attention/FFN pre/post sequence, prompt prefill, token feedback, and tokenizer
text output.
Sprint 294 added the next TP/EP HC semantic bridge:
`--tp-hc-current-input-gate` / `DS4_V100_TP_EP_HC_CURRENT_INPUT=1`. The path
loads real per-layer `hc_attn_*` controls, derives a current vector from the
resident sharded HC state, and feeds that vector into the routed expert
activations before the existing TurboMind MXFP4 EP compute. Dense diagnostic
inputs are still bridge inputs, filled by repeat/truncate from the HC-derived
current vector to match the existing diagnostic dense tensor widths. Direct
32-slot / 256K / 1-token all-layer V100 validation passes with
`sum_decode_ms=134.008975`, `238.789977` projected decode tok/s,
`sum_hc_current_input_ms=40.646652`, `sum_final_hc_ms=22.678353`, and
`8.530776 ms` output-head time. The launcher-level `/v1/completions` run with
32 concurrent requests forms one 32-request batch, returns `32/32` HTTP 200
responses, and reports `145.914985` wall generated tok/s /
`225.722945` decode generated tok/s. This is now the best TP/EP prototype
server scaffold, but it is still diagnostic: prompt prefill, tokenizer text,
selected-token feedback, and exact DS4 attention/FFN HC sequencing remain.
Sprint 295 added KV/resident-state guardrails for downstream serving work.
The old resident HTTP path allocated sharded KV but only exercised one
diagnostic KV slot and reset HC state per serving call. The new
`--tp-kv-all-slots-gate` / `DS4_V100_TP_EP_KV_ALL_SLOTS=1` updates and
verifies KV rows for every active slot, and
`--tp-hc-persist-state-gate` / `DS4_V100_TP_EP_HC_PERSIST_STATE=1` prevents
the token-major serving loop from resetting resident HC state on each call.
HTTP `/status`, `/metrics`, and response metadata now report
`kv_runtime_resident`, `kv_all_slots_gate`, and `hc_persist_state_gate`.
Direct 32-slot / 256K validation passes with `243.089283` decode tok/s and
`71.431217` wall tok/s; the wall drop is expected because all-slot KV
write/read verification is outside the timed decode stage. Launcher-level
`/v1/completions` with 32 concurrent requests passes with one 32-request
batch, `32/32` HTTP 200 responses, `58.791255` wall generated tok/s, and
`206.196887` decode generated tok/s. This is a correctness mode, not an
optimized KV path. Real per-client session keys, stable slot ownership,
prefill population, eviction/reset semantics, and token feedback remain.
Sprint 296 added the first TP/EP HTTP session-slot primitive after reviewing
the existing `ds4.c` session timeline model and llama.cpp server slot/cache
semantics. The diagnostic HTTP endpoint now derives a cache key from
`session_id`, `cache_key`, `conversation_id`, or a prompt hash fallback,
assigns stable resident slots with LRU eviction, buckets only requests with the
same `max_tokens` and resident position, rejects duplicate session keys inside
one decode batch, and exposes `/v100/slots` plus cache hit/miss/eviction
metadata in `/v100/status`, `/metrics`, and responses. V100 validation with
`session_id=alpha` shows the first request missing slot `0` at position
`100000 -> 100001`, the second request hitting the same slot at
`100001 -> 100002`, and status reporting `cache_hits=1`, `cache_misses=1`,
`cache_evictions=0`, `next_position=100002`. This is still diagnostic serving,
but the serving cursor is no longer only global scratch state.
Sprint 297 added a prompt-fingerprint cache guard on top of that session
table. A repeated `session_id` now only hits resident KV/HC if the prompt
fingerprint matches; a changed prompt resets the slot to the base position and
records a miss. The V100 smoke shows `alpha/hello` miss then hit on slot `0`
from `100000 -> 100001 -> 100002`, followed by `alpha/goodbye` as
`cache_hit=0`, `cache_prompt_match=0`, `100000 -> 100001`. Status reports
`cache_hits=1`, `cache_misses=2`, `cache_evictions=0`. This is still
string-fingerprint protection, not tokenizer-prefix reuse.
Sprint 298 ran a longer diagnostic `/v1/completions` benchmark after those API
guardrails. With `32` concurrent requests, `32` slots, `256K` context,
diagnostic output head, HC-current input, HC final expand, and persistent HC
state enabled, the `16/32/64` token cases each formed one coalesced 32-request
batch and returned `32/32` HTTP 200 responses. Wall generated throughput was
`194.530928`, `199.286944`, and `200.272837` tok/s respectively; decode
generated throughput was `329.048680`, `340.196025`, and `338.142261` tok/s.
The KV all-slot readback verifier was intentionally off for this throughput
run. GPU utilization remained low at roughly `7.4-8.4%` average and `36-37%`
max.
Sprint 299 added tokenized prompt acceptance and per-session generated-token
timeline tracking to the TP/EP diagnostic completion endpoint, following the
slot/session model in `ds4.c` and llama.cpp. `prompt_tokens:[...]` and numeric
`prompt:[...]` now drive token-sequence prompt fingerprints; resident slots
record prompt token ID count, generated token ID count, and last selected
token; `/v100/slots`, `/v100/status`, and response metadata expose those
counters. The V100 smoke with `session_id=tokalpha` and
`prompt_tokens=[1,2,3,4]` shows first request miss, second request hit, slot
cursor `100000 -> 100001 -> 100002`, slot prompt-token IDs held at `4`,
generated-token IDs advancing from `1` to `2`, status
`cache_hits/cache_misses=1/1`, and `total_generated_tokens=2`. This is still a
diagnostic endpoint: full tokenizer prefill and selected-token feedback into
the next CUDA decode input remain before real DeepSeek text serving.
Sprint 300 added the first selected-token feedback bridge. The TP/EP HTTP path
now loads resident BF16 `token_embd.weight` on GPU0
(`1059061760` bytes) and seeds each rank's layer-0 HC shard from a real token
embedding. On a cache miss, the decode input token is the prompt tail; on a
cache hit, it is the slot's previous selected token. The V100 smoke shows
request 1 using input token `4` and selecting token `77960`; request 2 hits
the same session and uses `77960` as `decode_input_token`, with generated-token
history advancing to `2`. This is request-boundary feedback only. The next
serving gap is a per-step output-head/sample/feed loop for `max_tokens > 1`,
plus tokenizer text I/O and prompt prefill.
Sprint 301 implemented that per-step loop for the diagnostic HTTP endpoint.
For `max_tokens > 1`, the server now runs one TP/EP decode step, runs the
diagnostic output head, feeds the selected token back through the resident
BF16 token embedding seed, and repeats. Session commit appends the full
generated-token sequence. The V100 smoke with `session_id=multi`,
`prompt_tokens=[11,12,13]`, and `max_tokens=3` starts from decode input token
`13`, records `3` generated token IDs, advances the slot cursor
`100000 -> 100003`, and reports `153.126777` wall generated tok/s /
`252.798645` decode generated tok/s for this correctness-oriented single
request. The endpoint still lacks tokenizer text I/O, real prompt prefill,
active-slot-only decode, and MTP.
Sprint 302 added the first diagnostic prompt-prefill bridge on cache misses.
Prompt tokens before the tail now run through TP/EP one-token passes without
output-head selection, updating resident KV/HC state before generation starts
from the final prompt token. The V100 smoke with `session_id=prefill`,
`prompt_tokens=[21,22,23]`, and `max_tokens=2` reports
`prompt_prefill_tokens=2`, `generated_token_ids=2`, slot cursor
`100000 -> 100004`, and `next_position=100004`. The generated section reports
`212.692685` wall tok/s / `351.116767` decode tok/s. This is correctness
prefill, not optimized batched prefill.
Sprint 303 exposed the generated token sequence in the TP/EP diagnostic
completion response. `/v1/completions` now returns
`ds4_v100.generated_token_sequence` and an explicit `slot_position` alias for
the committed resident cursor. The V100 smoke with `session_id=seq`,
`prompt_tokens=[31,32,33]`, and `max_tokens=3` returns
`generated_token_sequence=[127885,57114,78026]`, `generated_token_ids=3`,
`slot_generated_token_ids=3`, `prompt_prefill_tokens=2`,
`slot_position=cache_pos_out=100005`, and `214.100724` wall tok/s /
`353.667490` decode tok/s for the generated section. Tokenized prompt prefill,
multi-token feedback, session persistence, and token-ID output are now wired;
tokenizer text I/O, active-slot-only decode, optimized/batched prefill, exact
DS4 HC parity, and MTP remain.
Sprint 304 added the matching diagnostic `/v1/chat/completions` route and
OpenAI-style chat envelope on top of the same TP/EP resident path. The V100
smoke with `session_id=chatseq`, `prompt_tokens=[41,42,43]`, and
`max_tokens=3` returns `object=chat.completion`,
`choices[0].message.role=assistant`, matching `choices[0].token_ids` and
`ds4_v100.generated_token_sequence` of `[0,57085,104170]`,
`slot_position=cache_pos_out=100005`, and `210.355981` wall tok/s /
`350.653125` decode tok/s. The chat route is still diagnostic because
assistant text remains empty until tokenizer rendering is connected.
Sprint 305 wired tokenizer text I/O into that TP/EP path by linking the
existing `ds4.c` CPU tokenizer in inspect-only mode and adding
`--tokenizer-model` / `DS4_V100_TP_EP_TOKENIZER_MODEL`. Text prompts now
materialize prompt tokens before cache fingerprinting and prefill, and
generated token IDs are decoded into both OpenAI response content and
`ds4_v100.generated_text`. The V100 chat smoke with message content `"Hello"`
reports `tokenizer_ready=1`, `request_prompt_token_ids=5`,
`prompt_prefill_tokens=4`, generated token IDs `[95933,89868]`, decoded text
`ICCungtod`, `slot_position=cache_pos_out=100006`, and `213.595353` wall
tok/s / `350.755948` decode tok/s. The remaining API gaps are full role-aware
chat parsing, streaming, active-slot-only decode, optimized/batched prefill,
exact DS4 HC parity, and MTP.
Sprint 306 benchmarked that tokenizer-enabled chat path with 32 concurrent
text requests. The run formed one full 32-slot coalesced batch at `256K`,
tokenized each request to `7` prompt tokens, ran `6` diagnostic prefill tokens
per request, generated `256` total tokens, and returned `32/32` HTTP 200
responses. Server-side generated-section throughput was `214.155740` wall
tok/s / `355.130754` decode tok/s; client-side effective throughput including
HTTP orchestration was `110.036538` tok/s.
Sprint 307 added the first end-to-end reference-vector parity harness for the
TP/EP server. It parses `tests/test-vectors/official.vec`, drives the live
HTTP endpoint, and compares decoded generated bytes against official selected
token bytes. The first V100 gate on `short_reasoning_plain` fails as expected
for the current bridge path: official expected `16` (`3136` hex), while TP/EP
returned `ICC` (`494343` hex), token ID `95933`. This makes semantic parity,
not endpoint availability, the active production blocker.

Current promoted serving baseline is Sprint 199's graph-backed
`fused6_reduce` production pack at 16-slot/256K: `67.886268` generated tok/s
and `66.825545` continuation tok/s with `16/16` token match. Sprint 200 then
rejected the easy exact six-route kernel cut-ins, and Sprint 201 measured the
first full-layer TP4 boundary envelope. A 43-layer, 4-collective/layer TP4
proxy costs `22-24 ms` at 16 active tokens, or about `655-724 tok/s`
overhead-only before DS4 compute. The same boundary improves to
`1837 tok/s` overhead-only at 64 active tokens and `2509 tok/s` at 128 active
tokens. Sprint 202 then measured real TurboMind MXFP4 routed-FFN TP4 compute,
after fixing a benchmark lifecycle bug where the full reference and shard 0
crossed streams on GPU0 and shared one TurboMind workspace. Corrected
compute-only speedup reaches `2.350x` at 96 routes and `3.636x` at 768 routes,
but conservative copy-inclusive timing regresses to `0.783x` and `0.682x`.
This keeps full-layer TP4/EP viable only if dense and routed work stay inside
the TP boundary; routed-only overlays remain rejected. Sprint 203 then built
that first resident TP4 layer-slice gate. It is correct, but the naive
resident root boundary is still slower than a one-GPU full-width routed-FFN
reference: `0.825x` at `96 routes x 43 layers` and `0.589x` at
`768 routes x 43 layers`. This blocks production TP4 scheduler integration
until a real concurrent collective or fused reduction boundary exists.
Sprint 204 added a concurrent `doubling_async` resident reduction. It improves
the resident slice and is positive at larger 768-route shapes (`1.071x` over
43 layers), but it does not reliably clear the production 96-route decode gate:
the first 43-layer run was `1.006x`, while the longer repeat was `0.896x`.
TP4 should remain a larger-batch/prefill candidate unless the collective gets a
fused/NCCL-grade implementation.
Sprint 205 tested the missing async root variant. It is correct, but slower:
`0.970x` at `96 routes x 4 layers`, `0.866x` at `768 routes x 4 layers`, and
`0.860x` at `96 routes x 43 layers`. This closes the current TP4 production
decode branch; next implementation should pivot to persistent fused routed-FFN
work.
Sprint 208 re-opened topology investigation for the 32-slot target with a
separate TP8 path, not a PP scheduler abstraction. The new TP planner shows
32-slot/256K `PP1/TP8` fits with F8 KV sharding at `26.84 GiB` worst GPU and
fails with replicated KV at `50.63 GiB`, making KV sharding mandatory. The
8-GPU FP16 collective smoke passed at 32/64/128 tokens; recursive doubling beat
root and measured `0.322599 ms`, `0.372364 ms`, and `0.436299 ms`
respectively. The 43-layer, 2-reduction/layer resident-boundary proxy measured
`29.381000 ms` at 32 tokens (`1089.139` overhead-only tok/s), `32.605223 ms`
at 64 tokens (`1962.876` tok/s), and `37.994584 ms` at 128 tokens
(`3368.901` tok/s). This clears the first TP8 investigation gate but does not
prove serving; the next TP sprint should build a bounded one-layer TP8
prototype in new TP-only files with sharded KV ownership inside the boundary.
Sprint 209 built that bounded prototype without touching the PP scheduler. At
32 slots / 256K / ratio-4 F8 KV, each GPU allocates a `169347072` byte KV shard
for the layer instead of a replicated `1.262 GiB` logical KV allocation. The
one-layer TP8 smoke passed correctness at 32/64/128 tokens with total average
latencies of `0.739408 ms`, `0.876011 ms`, and `1.098461 ms`; reduction time
was `0.634680 ms`, `0.718601 ms`, and `0.840586 ms`. This keeps TP8 alive, but
still only as a separate TP branch: the synthetic compute body must be replaced
with a real TP-only DS4 layer slice before any serving integration.
Sprint 210 replaced the synthetic body with a TP-only resident FFN fixture that
uses cuBLAS FP16 Tensor Core GEMMs for column-parallel gate/up, gated SiLU,
row-parallel down, and TP8 hidden reduction. At 32 slots / 256K / ratio-4 F8
KV, the `mid_shard=1024` gate passed correctness at 32/64/128 tokens with
total latencies `0.614750 ms`, `0.709350 ms`, and `0.796927 ms`. The denser
`mid_shard=2048` sweep also passed and reached `62.956` fixture TFLOP/s at 128
tokens. This confirms TP8 can put useful resident GEMM work inside the boundary,
but it is still an FP16 fixture; the next TP sprint should adapt the low-bit
TurboMind MXFP4 expert path to the separate TP8 codepath before any serving
integration.
Sprint 211 did that low-bit TP8 gate and rejected the current TP8 MXFP4 shard
shape. The new separate TP-only `ds4-v100-tp8-turbomind-ffn-smoke` runs through
the public TurboMind ABI with synthetic MXFP4 fixtures, but TP8
`mid_shard=256` produces invalid partial sums: 96/192/384-route runs all fail
correctness with large NaN counts. The compute-only timing is attractive
(`3.927x-4.189x` versus full reference), but the simple output gather/reduce is
already slower than the reference (`0.524x`, `0.368x`, `0.317x` total speedup).
The existing TP4 TurboMind control remains correct at the same route shapes and
shows `2.333x-3.676x` compute speedup, although copy-inclusive timing is still
below `1.0x`. Next work should pivot to TP4/PP1 low-bit layer ownership with a
better reduction boundary, or explicitly design a TP8 MXFP4 shard-256 kernel
before returning to TP8.
Sprint 212 executed that TP4/PP1 low-bit layer-body gate in a separate TP-only
tool. Correctness passes at 96/192/384 routes after fixing two benchmark bugs
(synthetic fixture overflow and a missing `cudaSetDevice()` before TurboMind
calls). Compute-only TP4 speedup is still strong (`2.335x/2.597x/3.707x`), but
the resident root reduction only wins at 96 routes (`1.078x`) and regresses at
192/384 routes (`0.932x/0.967x`). This rejects TP4/PP1 runtime ownership as
the next implementation step; TP should stay parked for prefill/larger-batch
work or until a fused/NCCL-grade collective exists. The next sprint should
return to a monolithic/persistent low-bit routed-FFN executor.
Sprint 213 closed the existing `fused6_split_reduce` materialized reducer
branch. The opt-in path is correct, builds on V100, passes full scheduler smoke
with `tm_layers=43`, and preserves CUDA graph capture (`43` captures, `129`
launches, `0` failures). The focused six-route FFN sequence improved from
`0.1391 ms` atomic to `0.1290 ms` materialized, but served 16-slot/256K
continuation only moved from `60.236036` to `60.655009` tok/s (`+0.7%`), below
the promotion gate. Defaults stay on `fused6_reduce + graph`; the next sprint
should stop reducer-wrapper tuning and build a true tile-local/persistent
routed-FFN workbench.
Sprint 214 built that standalone workbench and rejected the first tile-local
candidate. The tool compares `gated_silu -> down_reduce`,
`gated_silu -> down -> split_reduce`, and a diagnostic raw-MXFP4
down+route-reduce kernel for the exact six-route production decode shape. On
the V100 pod, the finite fixture passed baseline and candidate correctness
(`split_bad=0/4096`, `candidate_bad=0/4096`), but timing was decisively below
the gate: atomic sequence `0.184721 ms`, split sequence `0.165159 ms`,
candidate sequence `0.370901 ms`, and candidate down-only `0.276122 ms`
(`0.445x` versus best baseline). The branch is rejected because replacing the
Tensor Core down GEMM with SIMT F32 accumulation loses more than it saves by
removing the materialized down-route buffer. The next sprint should either move
to practical serving levers such as MTP/continuous batching or build a real
Tensor Core fused/persistent routed-FFN kernel rather than another reducer-only
wrapper.
Sprint 215 added a repeatable practical-serving matrix and ran it against the
persistent production TurboMind pack. The best current practical long-context
mode is now `32` slots at `128K`: `69.488893` generated tok/s and
`68.403129` continuation tok/s with `32/32` token match, `45.88%` average GPU
utilization, and `24124 MiB` max observed memory. The current `16`-slot/`256K`
baseline remains valid at `62.602937` generated tok/s and `61.624766`
continuation tok/s with `16/16` token match. `32` slots at `256K` still fails
closed at the production launcher cap: `DS4_V100_SLOTS=32 exceeds ctx=262144
admission cap 16`. MTP verify is compatible but not a speedup
(`attempted=16`, `accepted=0`, `16.373227` generated tok/s), and one-slot MTP
commit accepted `8/15` drafts but only reached `8.369430` generated tok/s /
`7.846341` continuation tok/s. MTP is therefore not shipped as a speedup; the
next MTP step must be true speculative target verification over drafted tokens.
Sprint 216 built that focused MTP speculative gate and rejected the current
commit path as a throughput feature. The new replay and sustained-bench
accounting reports draft proposals, accepted drafts, target tokens verified,
target forwards, effective output tokens, and speculative saves. On the V100
pod, one-slot `256K` MTP commit again accepted `8/15` drafts, but the decisive
fields were `target_forwards=16`, `effective_output_tokens=16`, and
`speculative_saves=0`; continuation throughput was `4.276211` tok/s versus
`4.644949` for the same one-slot baseline. The API gap is now explicit:
current replay batching is across slots at the same decode step, while MTP
needs a one-slot multi-position target verification/state-advance primitive to
save target forwards. MTP remains default-off for production throughput.
Sprint 217 tested whether the `256K` slot cap was merely conservative and
found a real cold-path failure above 16 active slots. Sprint 218 localized that
failure: without launcher startup warmup, an `18`-slot/`256K` run first reports
NaN HC at `stage=1`, `gpu=1`, `layer=6`, `slot=0`, `token=0`, `position=0`.
Pre-output-head checking shows NaN HC reaches the output stage before logits,
so the output-head fastpath is not the first source. With launcher startup
warmup enabled, the same warmed appliance path passes the full target shape.
Sprint 219 made that warmed-readiness contract explicit in `/v100/status` and
metrics, then ran the longer production gate at `64` requests: `64/64`
matches, max memory `24124 MiB`, max GPU util `88%`, `58.241743` generated
tok/s, `16.380490` prompt tok/s, and `57.331715` continuation tok/s. The
launcher admits `ctx=262144`, `slots=32` only when startup warmup resolves
enabled; the cold `DS4_V100_STARTUP_WARMUP=0` path still fails closed at cap
`16`. Sprint 220 then aligned the operator deployment path with that result:
the env example now defaults to the production appliance dir, `ctx=262144`,
`slots=32`, `active_microbatch=32`, and `DS4_V100_STARTUP_WARMUP=auto`; the
production deployment smoke now accepts `--appliance-dir`, checks warmed
readiness in status and metrics, and passed on the V100 pod with two bounded
generation requests returning `3136`. Sprint 221 added the first explicit
MTP target-block verification primitive. Replay now has all-stage target
snapshot wrappers and a one-slot forced-block verification API, with
`tools/ds4-v100-replay --target-block-smoke N` as the diagnostic gate. The
V100 production-pack smoke passed for a 4-token block: first token bytes
`3136`, snapshot bytes `30107648`, `target_forwards=4`,
`accepted_prefix_len=4`, `target_tokens_verified=4`,
`effective_output_tokens=4`, and `speculative_saves=0`; a negative guard fails
closed for multi-slot use. This is the right MTP boundary, but not yet an MTP
speedup because the block body is still serial target execution. Sprint 222
then connected real chained MTP draft blocks to that boundary. The helper can
now return MTP next-HC, and `--mtp-draft-block-smoke 4` produced a real draft
sequence without target forwards between draft steps. The V100 production-pack
fixture reported `draft_tokens=1,0,1,0`, `target_tokens=1,380,5,380`,
`accepted_prefix_len=1`, `target_forwards=4`, `target_tokens_verified=4`,
`effective_output_tokens=2`, `speculative_saves=0`, `mtp_ms=18.081`, and
`verify_ms=231.553`; the multi-slot guard fails closed. This ships the
diagnostic but does not promote MTP throughput. The next MTP decision requires
a broader acceptance matrix; if accepted prefixes stay low, the practical
serving branch should pivot back to attention/KV or persistent low-bit
execution. Sprint 223 ran that matrix on five prompt fixtures and block sizes
`2,4,8` after fixing real-prompt replay compressed-cache cap sizing. The full
V100 production-pack matrix passed `15/15` cases with average accepted prefix
`1.533`, max accepted prefix `2`, `10/15` cases at accepted prefix `>=2`, and
total speculative saves `4`. The decision is to continue MTP only in the
block-2 shape: block-2 accepted both drafted tokens in `4/5` cases and reported
`speculative_saves=1`, while block-4 and block-8 never accepted more than two
tokens and mostly add verifier work. The next MTP sprint should build and
measure a block-2 exact speculative commit/verify path, not longer draft
blocks. Sprint 224 built that exact block-2 path and measured it on the V100
production pack. The path is token-correct and faster than same-process
baseline on `4/5` prompts: ok-case average `3.663043` block2 tok/s versus
`2.032918` baseline tok/s (`1.801865x`). It reports `target_forwards=7`,
`effective_output_tokens=8`, and `speculative_saves=1` for 8-token runs. It is
not ready for serving integration because `long_memory_archive` failed token
parity at token 1 (`baseline=16`, `got=8773`), and a follow-up
`--target-block-smoke 2` on that same prompt also failed target replay reset
determinism (`got=32085 want=10220`). The next sprint should fix long-prompt
replay reset/snapshot determinism, then rerun the block-2 gate.
Sprint 225 cleared that reset/snapshot blocker and tightened throughput
methodology. `tools/ds4-v100-replay` now has `--reset-parity-smoke` and
`--prompt-token-limit`; full `long_memory_archive` reset parity passes with
`prompt_tokens=3353`, `generated_tokens=1`, `first_token=32085`, and
`match=true`. Full `--target-block-smoke 2` also passes on that prompt with
`snapshot_bytes=907214848`. Bounded MTP block-2 checks pass through the 1024
token prefix with `token_match=true` and `speculative_saves=1`, but the full
single-slot MTP run was stopped after the methodology review and is not
promotion evidence. Practical throughput claims are now guarded:
`tools/ds4-v100-sustained-decode-bench.sh` defaults to `32` slots at `256K`,
`active_microbatch=slots`, startup warmup, `200000 us` microbatch wait, and
per-step async event handoff; slot tier `1` requires
`--allow-single-slot-diagnostic`. The current Sprint 225 practical serving
repeat measured `50.434232` generated tok/s and `47.282093` continuation tok/s
at `32` slots / `256K`, with `64/64` token match, average GPU utilization
`47.076%`, and max GPU utilization `96%`. TP remains prototype-only and is not
operational in production serving.
Sprint 226 implements the hard-cut TP/EP planner contract. The old
`tools/ds4-v100-plan-tp.c` PP-style topology variants are removed; the tool now
plans only `PP=1` (no pipeline), `TP=8`, `EP=8`, with sharded KV. Built on the
V100 pod against `/workspace/packs/ds4-appliance-full-tm-gated-s181`, the real
pack bytes sum to `145.42 GiB`, or `18.18 GiB` per TP rank. The target
`32` slots / `256K` / F8-KV plan fits at `27.00 GiB` per GPU including a
`2.00 GiB` reserve, leaving `5.00 GiB` post-reserve headroom. The planner
reports admission of `126` slots at `128K`, `63` at `256K`, `31` at `512K`,
and `15` at `1M` under current assumptions. It also records the expected
decode traffic shape: `37.625 MiB` hidden collective wire per decode step and
`3.000 MiB` aggregate EP dispatch+return at 32 slots. TP/EP is still not a
serving runtime, but the memory/topology contract is now explicit and PP modes
cannot be selected from the TP planner.
Sprint 227 adds the TP8 collective workbench and characterizes the next
boundary. The new `tools/ds4-v100-tp8-collective-workbench` builds on the V100
pod and supports `allreduce`, `reduce-scatter`, `allgather`, `rs-ag`, and
`ep-reduce` modes. At 32 tokens / hidden 4096 / 43 layers, the two-collective
doubling all-reduce proxy measures `26.904544 ms` (`1189.390` overhead-only
tok/s), while the EP output-reduce proxy measures `27.436756 ms`
(`1166.319` tok/s). Density helps: all-reduce reaches `2119.907` tok/s at 64
tokens and `3332.257` tok/s at 128 tokens; EP reduce reaches `1745.157` and
`3253.920` respectively. The root/direct RS+AG proxy is correct but slower
(`32.361613 ms`, `988.826` tok/s at 32 tokens), so it is not the first runtime
boundary candidate. NVLink status snapshots show all links at `25.781 GB/s`,
but byte counters remain `N/A` in the pod.
Sprint 228 adds the TP/EP pack contract generator. The new
`tools/ds4-v100-tp-ep-pack-contract` reads the production pack metadata and
emits `tp-ep-pack-contract.tsv`, `tp-ep-memory-summary.tsv`, and
`tp-ep-pack-contract.md`. Against
`/workspace/packs/ds4-appliance-full-tm-gated-s181` at `32` slots / `256K` /
F8 KV, it emits `11121` manifest lines: `4096` dense TP rows, `5496`
replicated control/router rows, `688` EP expert rows, and `840` KV/state rows.
The per-GPU contract is balanced at `27.024 GiB` including `1.5 GiB` scratch
and `2.0 GiB` reserve: `1.006 GiB` dense TP, `0.310 GiB` replicated control,
`17.133 GiB` EP experts, `3.396 GiB` KV, and `1.680 GiB` compression state.
This is a contract, not a byte-repacked TP loader; the next step is the
separate TP runtime skeleton.
Sprint 229 adds that first separate TP runtime skeleton in
`ds4_v100_tp_runtime.{h,cu}` plus `tools/ds4-v100-tp-runtime-smoke.cu`.
No PP scheduler files are touched. The V100 smoke opens all eight GPUs,
enables peer access, allocates the target runtime arenas at `32` slots /
`256K` / F8 KV, runs a fixture pass, and closes cleanly. Per GPU it allocates
`524288` hidden bytes, `3646642176` KV bytes, `1803550720` compression-state
bytes, and `1610612736` scratch bytes, for `7061329920` runtime bytes before
weights. The fixture reports `fixture_max_abs=0.000000000`, and `nvidia-smi`
shows `0 MiB` used on all eight GPUs after teardown.
Sprint 230 adds the first bounded dense/KV slice to the separate TP runtime.
The runtime now builds explicit per-layer, per-slot sharded KV offsets and the
smoke tool can write/read deterministic resident rows without touching the PP
scheduler. On the V100 pod at `32` slots / `256K` / F8 KV, the allocation
smoke passes with `3707940864` KV bytes, `1803550720` compression-state bytes,
and `7122628608` total runtime bytes per GPU before weights. The ratio-4
layer-2 slice with indexer KV passes at slot `7`, position `1024`, attn row
`384`, indexer row `256`, attn row bytes `65`, indexer row bytes `17`, and
`max_abs=0.000000000` on all eight GPUs. The ratio-128 layer-3 slice without
indexer KV passes at slot `7`, position `8192`, attn row `192`, attn row
bytes `65`, and `max_abs=0.000000000`. TP/EP is still not a serving runtime;
the next step is a bounded EP routed-expert slice using the real low-bit
expert kernels inside the separate TP runtime model.
Sprint 231 adds that bounded EP routed-expert slice as a new TP/EP-only smoke
tool, `tools/ds4-v100-tp-ep-expert-smoke.cu`. It uses the real TurboMind
MXFP4 grouped gated-SiLU and grouped down ABIs on all eight V100s, with EP8
ownership modeled as `256` global experts and `32` local experts per rank. At
the target `32` slots / `top_k=6`, the smoke reports `192` aggregate routes,
`24` routes per GPU, `6` active local experts per GPU, route imbalance
`1.000000`, `1572864` dispatch bytes, `1572864` return bytes,
`repeat_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`, and `PASS`. Per-rank
latency is skewed: ranks `0-6` are about `0.059 ms`, while rank `7` is
`0.249378 ms`; the 64-slot diagnostic also passes with `384` aggregate routes
and rank `7` at `0.268049 ms`. TP/EP is still not serving, but the EP
low-bit kernel gate is now live; the next step is a one-layer TP/EP
correctness gate that combines dense/KV, routing, EP experts, and reduction.
Sprint 232 adds that one-layer TP/EP fixture gate as
`tools/ds4-v100-tp-ep-layer-smoke.cu`. The tool links the separate TP runtime
with the TurboMind MXFP4 ABI in one process, opens `32` slots / `256K` /
F8-KV runtime arenas, verifies a layer-2 ratio-4 KV slice, and then runs the
EP8 routed expert fixture on all eight V100s. The V100 run passes with
`kv_bytes=3707940864`, `comp_state_bytes=1803550720`, `total=7122628608`
runtime bytes per GPU, KV `max_abs=0`, `192` aggregate routes, `1572864`
dispatch bytes, `1572864` return bytes, route imbalance `1.000000`,
`repeat_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`, and `PASS`. The measured
fixture one-layer envelope is `1.321812 ms`: `1.078032 ms` dense/KV fixture
time plus `0.243780 ms` worst-rank EP time. Rank `7` remains the slow EP rank.
TP/EP is still not serving; the next step is descriptor-driven one-real-layer
TP/EP correctness before scaling to all 43 layers.
Sprint 233 starts that descriptor-driven path with a TP/EP-only contract
smoke, `tools/ds4-v100-tp-ep-layer-descriptor-smoke.c`. Against the real
Sprint 228 production-pack contract on the V100 pod, layer `2` reports
`288` total rows: `112` dense TP rows, `136` replicated control/router rows,
`16` EP expert rows, `16` KV shard rows, and `8` compression-state rows, with
`bad_rows=0`. Every GPU reports `36` rows, `711945176` estimated layer bytes,
`14` dense rows, `17` control rows, `2` expert rows, `2` KV rows, `1`
compression row, and zero ownership mismatches. Expert ownership resolves to
the expected EP8 spans: GPU0 `0..31`, GPU1 `32..63`, through GPU7 `224..255`.
This proves descriptor ownership, not real descriptor-backed execution. The
next step is binding those descriptor rows to actual production-pack byte spans
and feeding descriptor-derived expert pointers into the one-layer TP/EP smoke.
Sprint 234 completes that byte-binding gate. The TP/EP layer smoke now has an
opt-in descriptor-backed expert mode that parses the real
`turbomind-pack-index.tsv`, reads production-packed layer-2
`ffn_gate_up_exps` and `ffn_down_exps` weight/scale bytes from
`gpu0.weights`, copies the selected EP experts to each target V100, and builds
TurboMind pointer tables from descriptor-derived strides and offsets. On the
V100 pod at `32` slots / `256K` / `top_k=6`, descriptor-backed execution
passes with `192` aggregate routes, `641728512` descriptor bytes read,
runtime bytes per GPU `7122628608`, KV `max_abs=0.000000000`,
`worst_ep_ms=0.246647`, `dense_kv_ms=1.121624`, `one_layer_ms=1.368271`,
`repeat_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`, and `PASS`. Same-binary
synthetic regression also passes with `worst_ep_ms=0.247603`. TP/EP is still
not serving; this proves real packed expert byte binding for one layer, while
dense/control/router/attention descriptor execution and full 43-layer decode
remain the next gates.
Sprint 235 adds the first TP/EP-only full-layer scaffold as
`tools/ds4-v100-tp-ep-full-layer-smoke.cu`. It does not touch the PP
scheduler. Against the real Sprint 228 TP/EP contract and the Sprint 181
production pack, layer `2` parses and binds all descriptor families:
`288` rows total, `112` dense rows, `136` control rows, `16` expert rows,
`16` KV rows, and `8` compression-state rows. The V100 run at `32` slots /
`256K` / `top_k=6` device-checks `163102720` dense bytes and `84041408`
control bytes, loads `641728512` routed-expert bytes, reports descriptor
checksum `3434523335`, keeps KV `max_abs=0.000000000`, and passes EP repeat
with `repeat_bad=0`, `repeat_nan=0`. Worst EP time is `0.249378 ms`, dense/KV
fixture time is `0.744619 ms`, and the one-shot descriptor load/check phase is
`2414.124867 ms`. That descriptor time is not a serving metric; it is a
startup/scaffold byte-binding cost. TP/EP is still not serving and not
logits-equivalent, but all layer-2 families now have a concrete TP/EP runtime
binding outside the PP path.
Sprint 236 replaces the first dense checksum stage with real descriptor-backed
low-bit dense computation. The TP/EP-only full-layer smoke now accepts
`--dense-compute-tensor blk.2.attn_q_a.weight`, resolves the real layer-2 F8
contract rows, loads packed F8 E4M3 block-128 shards directly from the
production pack, and expands F8 values inside a CUDA kernel. On the V100 pod at
`32` slots / `256K`, each GPU computes `128` local rows x `4096` columns for
`32` slots. The dense gate loads `4227072` packed bytes, measures
`dense_compute_ms=0.081783`, passes exact repeat (`repeat_bad=0`,
`repeat_nan=0`), and matches the bounded CPU oracle with
`dense_compute_oracle_max_abs=0.000000007`. The Sprint 235 scaffold still
passes in the same run: `288` layer rows, KV `max_abs=0`, EP
`worst_ep_ms=0.242517`, and final `PASS`. TP/EP is still not serving and not
full-layer logits-equivalent, but packed dense bytes now feed real GPU compute
inside the separate TP/EP path.
Sprint 237 broadens that to layer-2 F8 dense coverage. The TP/EP-only
full-layer smoke now supports `--dense-compute-all-f8`, discovers all compatible
layer-2 F8 dense TP tensor groups from the real contract, and executes all
nine groups. The V100 run covers `blk.2.attn_kv_latent.weight`,
`blk.2.attn_output_a.weight`, `blk.2.attn_output_b.weight`,
`blk.2.attn_q_a.weight`, `blk.2.attn_q_b.weight`,
`blk.2.ffn_down_shexp.weight`, `blk.2.ffn_gate_shexp.weight`,
`blk.2.ffn_up_shexp.weight`, and `blk.2.indexer.attn_q_b.weight`. Aggregate
packed bytes loaded are `141606912`, worst dense compute time is `0.654029 ms`,
repeat is exact (`dense_compute_repeat_bad=0`,
`dense_compute_repeat_nan=0`), and worst bounded CPU oracle error is
`0.000000015`. The full scaffold still passes in the same run with `288` layer
rows, KV `max_abs=0`, EP `worst_ep_ms=0.241766`, and final `PASS`. TP/EP is
still not serving and still excludes BF16 compressor/indexer dense math, but
the F8 dense tensor families for layer `2` now execute from packed bytes in
the separate TP/EP path.
Sprint 238 closes the layer-2 dense coverage gap by adding BF16
compressor/indexer coverage to the same separate TP/EP smoke. The new
`--dense-compute-all-bf16` path discovers all compatible BF16 `dense_tp`
tensors, loads production pack bytes, expands BF16 inside CUDA code on the
GPU, and validates repeat plus bounded CPU oracle checks. At `32` slots /
`256K`, all five BF16 groups pass:
`blk.2.attn_compress_gate.weight`, `blk.2.attn_compress_kv.weight`,
`blk.2.indexer.compress_gate.weight`, `blk.2.indexer.compress_kv.weight`, and
`blk.2.indexer.proj.weight`. Aggregate BF16 bytes loaded are `21495808`, worst
BF16 compute time is `0.047206 ms`, repeat is exact, and worst CPU oracle error
is `0.000000119`. The combined `--dense-compute-all` run also preserves all
nine Sprint 237 F8 dense checks with `dense_compute_pass=1`, reports
`bf16_compute_pass=1`, keeps KV `max_abs=0`, measures `worst_ep_ms=0.250368`,
and ends in final `PASS`. TP/EP still is not serving and still needs real
layer dataflow composition, but layer-2 F8 and BF16 dense families now execute
from production bytes in the TP/EP-only codepath.
Sprint 239 adds the first TP/EP-only next-hidden composition gate. The
full-layer smoke now supports `--compose-next-hidden`, materializes route-slot
mapping for the EP schedule, reduces TurboMind routed expert down outputs into
512-wide destination hidden shards, peer-copies those expert contributions
across all eight V100s, and composes resident next-hidden shards from
`blk.2.attn_output_b.weight`, `blk.2.ffn_down_shexp.weight`, returned EP
contributions, and deterministic residual input. The 32-slot/256K V100 run
passes with `ep_contribution_bytes=4194304`, `ep_return_bytes=4194304`,
`attn_dense_ms=0.555213`, `shared_dense_ms=0.153702`,
`compose_ms=3.707477`, non-zero checksum `4112649481`, `finite_bad=0`,
exact repeat, and `compose_pass=1`. The same run preserves combined dense
coverage (`dense_compute_pass=1`, `bf16_compute_pass=1`), KV `max_abs=0`,
EP `worst_ep_ms=0.255590`, and final `PASS`. TP/EP is still not production
serving and still not logits-equivalent, but the separate path now composes a
real resident next-hidden shard from production bytes and explicit EP return.
Sprint 240 adds a resident repeated decode-loop gate to the separate TP/EP
path. The full-layer smoke now supports `--decode-steps N`, loads the two F8
dense composition tensors once, keeps TurboMind EP weights and composition
buffers resident, and repeats the representative layer-2 step without rereading
pack bytes. On the V100 pod at `32` slots / `256K`, MTP off, `50` resident
steps pass with `1600` slot-steps, `total_ms=92.277411`,
`ms_per_step=1.845548`, and `slot_step_tok_s=17339.021356`. Stage timing is
`ep_ms_per_step=0.319095`, `dense_ms_per_step=0.756244`, and
`compose_ms_per_step=0.770121`, with `finite_bad=0` and non-zero checksum
`2382924023`. The same run preserves combined dense coverage
(`dense_compute_pass=1`, `bf16_compute_pass=1`), Sprint 239 composition
(`compose_pass=1`), KV `max_abs=0`, and final `PASS`. This is not end-to-end
generated tok/s; it is a representative layer-loop metric showing the current
TP/EP resident path is dominated by scalar dense kernels plus compose/peer
synchronization rather than EP alone.
Sprint 241 adds and measures an opt-in FP16 EP return path. The implementation
keeps local EP contribution accumulation in FP32, casts each source/destination
return shard to FP16 before peer copy, then expands FP16 back to FP32 while
summing on the destination rank. The FP16 path passes at `32` slots / `256K`
and halves the reported EP return payload from `4194304` bytes to `2097152`
bytes, but it is slower as a standalone optimization. Same-binary 50-step A/B:
FP32 return measures `ms_per_step=1.788149`, `slot_step_tok_s=17895.603225`,
`compose_ms_per_step=0.713836`; FP16 return measures
`ms_per_step=1.937399`, `slot_step_tok_s=16516.992775`,
`compose_ms_per_step=0.859697`. Both pass finite/checksum checks. Decision:
keep FP32 return as default, keep `--ep-return-fp16` as a diagnostic, and only
revisit FP16 return if cast/copy/sum is fused into a larger compose kernel.
Sprint 242 fuses the FP32 EP remote-sum directly into the next-hidden compose
kernel in the separate TP/EP full-layer smoke. The new opt-in
`--fuse-compose-sum` removes the destination `ep_sum` zero kernel and eight
standalone add kernels per destination rank. Same-binary 50-step A/B at
`32` slots / `256K`, MTP off: baseline FP32 return measures
`ms_per_step=1.784008`, `slot_step_tok_s=17937.138290`, and
`compose_ms_per_step=0.713663`; fused compose/sum measures
`ms_per_step=1.641832`, `slot_step_tok_s=19490.418145`, and
`compose_ms_per_step=0.568906`. Both paths preserve checksum `2382924023` and
pass finite validation. Decision: keep FP32 return, move fusion forward, and
continue collapsing TP/EP synchronization boundaries before serving
integration.
Sprint 243 tested a bounded HMMA dense replacement for the two F8 composition
tensors in the separate TP/EP resident loop. The new opt-in
`--dense-hmma-compose` keeps packed F8 bytes resident and decodes tiles into
FP16 WMMA fragments on GPU, but this first implementation is slower. Same
32-slot/256K/50-step A/B with fused compose enabled: scalar dense measures
`ms_per_step=1.620386`, `slot_step_tok_s=19748.386791`, and
`dense_ms_per_step=0.753941`; HMMA dense measures `ms_per_step=3.533215`,
`slot_step_tok_s=9056.907248`, and `dense_ms_per_step=2.667910`. Both pass
finite/repeat checks. Decision: reject this naive HMMA candidate as default;
do not tune it blindly. The next dense optimization should adapt the older
shape-specific F8 HMMA paths or introduce a prepacked/software-pipelined
low-bit dense kernel that amortizes F8 decode.
Sprint 244 measured that dense ceiling directly with an opt-in resident
FP16/cuBLAS path for the two F8 composition tensors. This is diagnostic, not
the final model format: setup expands packed F8 to resident FP16 on device and
the decode loop uses FP16 Tensor Core GEMM with FP32 output. Same
32-slot/256K/50-step fused-compose A/B: scalar dense measures
`ms_per_step=1.685018`, `slot_step_tok_s=18990.892348`, and
`dense_ms_per_step=0.755645`; resident FP16/cuBLAS dense measures
`ms_per_step=1.050770`, `slot_step_tok_s=30453.870979`, and
`dense_ms_per_step=0.175605`. Both pass finite/repeat checks. Decision: dense
is a real removable bottleneck, but expanded FP16 is only a ceiling. The next
production sprint should implement a packed low-bit dense path that preserves
model residency while feeding tensor cores efficiently.
Sprint 245 added the corresponding memory admission gate to the separate
TP/EP pack contract. Against the real production pack at `32` slots / `256K`
/ F8 KV, the base TP/EP plan remains `27.024 GiB` per GPU including the
existing `2.0 GiB` reserve. The contract now reports `0.687 GiB` of cacheable
F8 dense packed bytes per GPU, `1.364 GiB` for the F8-to-FP16 runtime cache,
`0.319 GiB` of BF16 dense shadow bytes, and a practical replace-source total
of `27.701 GiB` per GPU. That leaves `4.299 GiB` physical headroom versus
32 GiB while preserving the source quantized pack as the offline artifact.
Decision: dense FP16 runtime caching is memory-admissible for the target TP/EP
shape if cacheable dense source tensors are not kept twice in VRAM. The next
implementation should add the TP/EP dense-cache loader/runtime path for all
dense tensors and then benchmark the resident all-layer path before returning
to custom packed low-bit dense kernels.
Sprint 246 added that loader as a new TP/EP-only CUDA tool,
`tools/ds4-v100-tp-ep-dense-cache-smoke`. It allocates one FP16 dense cache
arena per GPU, stages source shards through a temporary GPU buffer, converts
`f8_e4m3_b128` and `bf16` dense tensors on device, and validates FP16 cache
checksums/nonfinite counts. The layer-2 subset passes with `112` dense rows,
`0.151901 GiB` source bytes, and `0.281738 GiB` aggregate cache. The full
contract passes with `4096` dense rows, `8.047012 GiB` aggregate source bytes,
and `13.459473 GiB` aggregate FP16 cache. Per GPU, this is `512` rows,
`1.005877 GiB` source, `1.682434 GiB` FP16 cache, `126.250 MiB` max temporary
source staging, and zero nonfinite FP16 values. Decision: the dense cache is
now a real V100 allocation/conversion path. Next wire the cache arena into the
TP/EP resident layer execution path, then benchmark a resident all-layer loop.
Sprint 247 wires that cache into execution for the representative layer-2
TP/EP resident decode loop. The new `--dense-f16-cache-compose` option builds
a layer-local dense FP16 cache from the contract and makes the resident
FP16/cuBLAS dense ops use cache pointers instead of private weight copies.
Same-binary 50-step A/B/C at `32` slots / `256K`, MTP off, fused compose:
scalar dense passes at `1.642514 ms/step` and `19482.326340` slot-step tok/s;
private FP16/cuBLAS passes at `1.056807 ms/step` and `30279.894858`
slot-step tok/s; cache-backed FP16/cuBLAS passes at `1.015128 ms/step` and
`31523.122614` slot-step tok/s. The cache-backed path preserves checksum
`2515001`, emits `dense_f16_cache=1`, and materializes `112` layer-2 dense
rows into `302514176` cache bytes. Decision: dense cache lookup is now wired
into decode execution. Next lift it from the two composition tensors to a
descriptor-selected dense execution table for every layer.
Sprint 248 adds that descriptor-selected dense execution table to the TP/EP
dense-cache workbench. The new `--execute-table` mode groups complete
`dense_tp` contract rows by `(layer, tensor_id)` and runs cache-backed
FP16/cuBLAS GEMMs across all eight GPUs. The layer-2 table passes with `14`
groups, `112` GEMMs per iteration, `1.384323 ms/iteration`, `6.992914`
dense-table TFLOP/s, and zero nonfinite outputs. The all-layer table passes
with `510` transformer-layer groups, `4080` GEMMs per iteration,
`394684006400` FLOPs per iteration, `51.003671 ms/iteration`, `7.738345`
dense-table TFLOP/s, checksum `15841839914005485`, and zero nonfinite outputs.
Decision: hardcoded layer-2 dense selection is no longer the all-layer blocker.
Next compose this dense table with EP routed experts, KV/update, and
layer-to-layer hidden flow in a resident all-layer TP/EP loop.

Current maximum-context production mode is now the Sprint 219 warmed
32-slot/256K appliance result. Sprint 137 adds an explicit
128-slot/32K short-context throughput mode. Sprint 139 raises the best observed
gated-appliance 128-slot/32K run to `60.130047` generated tok/s, while showing
the fixed-shape gate/up probe itself only contributes about `0.1%` end-to-end.
The runtime now reliably coalesces
high-slot concurrent requests into one tensor batch by resolving launcher
`auto` microbatch wait to 200 ms at `active_microbatch >= 16`. The best current
served result is Sprint 146's `61.223893` generated tok/s at 256-slot/16K;
the best 32K result remains `60.130047`, and the current 256K production-auto
repeat remains `43.534061` generated tok/s. Sprint 123 found
correct opt-in shared-FFN fusions up to `43.887206`. Sprint 124 added a
correct opt-in TurboMind route-row reduce path and measured up to `43.822500`.
Sprint 125 added a correct grouped-batch attention output-A probe and measured
up to `43.640921`. Sprint 126 added a default-off production routed-expert
stage profiler and confirmed the current binary still serves at `43.453309`
generated tok/s with `16/16` token match. Sprint 127 added an opt-in
TurboMind gated-SiLU path with interleaved fused gate/up packs. It removed the
standalone SwiGLU bucket from the routed-expert profile and measured
`43.933293` generated tok/s with `16/16` token match. Sprint 128 compacted the
packed TurboMind grouped schedule from 256 experts to at most `total_routes`
groups and promoted that path as the launcher default after same-binary A/B
reached `45.888778` generated tok/s on the existing fused appliance and
`46.394722` on the interleaved gated appliance with route-row-reduce opt-in.
Sprint 129 exposed TurboMind dispatch policy selection, rejected unsafe
`measure` after a full-scheduler measurer fatal, and found safe `reuse`
neutral at `45.813841` vs `45.840691` default. Sprint 130 reran the closest
existing routed-FFN epilogue-fusion analogue on the current fused appliance:
compact control was `45.837745`, while compact plus route-row-reduce was
`45.660765`, so route-row-reduce remains opt-in. Sprint 131 added a correct
opt-in TurboMind indexed-A path that avoids route-expanded FP16 activations for
gate/up GEMMs, but served A/B was only `45.789937` vs `45.663281` control, so
it also remains opt-in. Sprint 132 extended the standalone TurboMind gate/up
benchmark to the production 96-route shape from the served profile; the
interleaved gated path passed at `0.1776 ms` vs `0.2889 ms` for separate
gate+up, a `1.626x` isolated speedup. Sprint 133 corrected that benchmark to
also use the served compact active-expert topology; compact 96-route gated-SiLU
is `0.1740 ms` vs `0.1895 ms` separate gate+up, only `1.089x`. Sprint 134
added a fixed-shape DS4 ABI probe that bypasses generic dispatch and directly
launches the matching SM70 MXFP4 gated kernel; it was bit-identical and exactly
neutral at `0.1746 ms` vs `0.1746 ms` generic gated. Sprint 135 raised the
admitted short-context throughput tier to 32 slots at 128K, while keeping 256K
capped at 16 slots. The 32-slot 128K appliance passed full scheduler smoke and
served correctness, reaching `52.840889` generated tok/s versus `45.780913`
for a same-context 16-slot control. The next target is therefore not dispatch
bypass or gate/up launch fusion; it must change kernel math/dataflow or widen
the served scheduling shape further. Sprint 136 widened the short-context tier
again to 64 slots at 64K, passed full scheduler smoke, and reached `57.322945`
generated tok/s versus `52.884400` for a same-context 32-slot control.
Sprint 137 admitted 128 slots at 32K, passed full scheduler smoke, and reached
`59.598172` generated tok/s versus `57.170428` for a same-context 64-slot
control. The slot-width sweep remains positive but is clearly diminishing.
Sprint 138 widened the standalone TurboMind compact gate/up benchmark defaults
to cover 192/384/768 routed-row shapes. The 768-route compact baseline is
`0.6379 ms` for fused gate_up and `0.6481 ms` for gated-SiLU. Sprint 139 added
a fixed-shape 768-route m128 gated-SiLU probe and wired it into the appliance
under exact production guards. It beat the isolated generic gated path
(`0.5999 ms` vs `0.6480 ms`) and served correctly at `60.130047` generated
tok/s on the 128-slot/32K gated appliance, but same-binary probe-off was
`60.061899`, so the end-to-end gain is only about `0.1%`. Sprint 140 repeated
that fixed-shape strategy for the 768-route down projection. The down probe was
correct and faster in isolation (`0.3026 ms` vs `0.3272 ms`), but served A/B
was slower with it enabled (`60.038469` vs `60.129772`), so it remains opt-in
and default-off. Sprint 141 added an opt-in half2-vectorized variant of the
route-row reduce tail. It passed the full 43-layer 128-slot smoke, but served
A/B stayed neutral: control was `60.108232`, scalar route-row reduce was
`60.112248`, and half2 route-row reduce was `60.104512`, all with `128/128`
token match. Tail-kernel vectorization is therefore not the missing throughput
lever. Sprint 142 moved that idea into the TurboMind down GEMM epilogue for
the exact 768-route high-slot shape. The fused epilogue reduce path passed the
full 43-layer 128-slot smoke and served correctly at `60.041003` generated
tok/s versus `59.987105` same-binary control, so it remains opt-in/off because
the effect is only run-noise positive. Sprint 143 added first-class prefill
versus decode metrics to the soak, sustained decode, and aggregate throughput
harnesses so future A/B runs show prompt replay, continuation decode, and
aggregate generated rates separately. Sprint 144 added explicit SM70 MXFP4
`m64n256` tile probes for the 768-route gate/up and down shapes. Both passed
full 43-layer smoke; the standalone down probe was slightly faster
(`0.2896 ms` vs `0.2936 ms`), but served A/B regressed to `59.791839`
generated tok/s versus `59.993301` control, and gate `m64n256` also regressed
to `59.797232`. The probes remain explicit opt-ins only. Sprint 145 admitted a
guarded 256-slot/16K short-context tier after planner and full-scheduler
validation. It served correctly at `61.065087` generated tok/s and
`57.248519` continuation/decode tok/s with `256/256` token match, but the
decode gain over the 128-slot/16K control was only about 2%, so slot widening
is now a ceiling-expansion tactic rather than the main throughput lever.
Sprint 146 added explicit 1536-route fixed-shape gate/up and down probes for
the 256-slot compact routed shape. The gate probe was correct and slightly
faster in isolation (`0.9435 ms` vs `0.9651 ms` generic gated), but served A/B
was flat to slightly worse (`61.204203` generated tok/s and `57.378940`
continuation/decode tok/s versus `61.223893` and `57.397400` control), so the
1536-route probes remain explicit opt-ins and are not selected by `auto`.
Sprint 147 extended the down-reduce epilogue to the 1536-route shape and
validated it with a full 43-layer 256-slot smoke, but served A/B was deferred
after the strategy pivot to larger fused-kernel work. Sprint 148 tested a real
stage-4 SM70 software-pipeline variant of the fused MXFP4 gate/up+gated-SiLU
kernel. The 768-route `m128_s4` probe improved the isolated gate/up benchmark
(`0.5811 ms` vs `0.6033 ms` for `m128`) and passed full 43-layer smoke, but
served A/B was only run-noise positive (`60.049057` generated and `56.295991`
continuation/decode tok/s versus `59.865668` and `56.124063` control), and the
profile did not show a reliable gate/up bucket reduction. Stage-4 probes remain
explicit opt-ins. Sprint 149 added a TP split benchmark and a P2P reduce-payload
proxy. The 2-way FFN middle-dimension split shows ideal compute speedups of
`1.858x` at 768 routes and `1.468x` at 1536 routes before communication.
Peer-copy measurements show a 12 MiB hidden payload takes about `0.26 ms` over
NV2, `0.52 ms` over NV1, and `1.29-1.31 ms` over SYS, so a TP prototype should
start with NV2 pairs and stay bounded before any scheduler-wide rewrite.
Sprint 150 built that bounded 2-GPU TP proxy. On clean NV2 pairs, the
768-route shape shows about `1.87x` concurrent compute speedup and about
`1.28x` total speedup after conservative input/output copies. The 1536-route
shape is neutral to slower after copies (`0.85-0.94x`), so TP is a candidate
for the 128-slot/32K tier first, not a broad replacement for the 256-slot/16K
ceiling. Sprint 151 added the missing correctness gate: finite MXFP4 fixtures
now compare full one-GPU down output against the sum of the two TP partials.
Both clean NV2 pairs pass at 768 and 1536 routes with `rel ~= 2.46e-04`,
`bad=0`, and max absolute difference `6.1035e-05`. Sprint 152 completed the
fused gate/up software-pipeline sweep by adding 3-stage variants and comparing
2/3/4-stage fixed probes. At 768 routes, `m128`, `m128_s3`, and `m128_s4`
measured `0.5809 ms`, `0.5863 ms`, and `0.5794 ms`; at 1536 routes,
`m128_1536`, `m128_s3_1536`, and `m128_s4_1536` measured `0.8743 ms`,
`0.8821 ms`, and `0.8774 ms`. NCU fixed-probe counters were also neutral, with
identical HMMA instruction counts. Stage-count tuning inside the existing
fused gate/up GEMM is now exhausted as a material lever. Sprint 153 added a
bounded 2-way TP appliance-pack contract and context binding for split
TurboMind expert descriptors. A layer-3, six-expert pack emitted 8 TurboMind
rows across GPU0 and GPU3 and passed partial context binding. The real 2-GPU
NV2 proxy remains positive only at the 768-route shape (`1.157x`
total-with-copy on pair `0,3` in this run) and slower at 1536 routes
(`0.912x`), so TP remains a narrow 128-slot/32K prototype candidate rather
than a broad topology replacement. Sprint 154 closed the served A/B gap for
the largest currently implemented fused routed-FFN boundary: fused gate/up plus
gated-SiLU plus the down-projection route-weighted reduce epilogue. The
128-slot/32K result was flat (`59.509317` generated / `55.789985`
continuation tok/s versus `59.502747` / `55.783825` control), and the
256-slot/16K result was slightly slower (`60.642962` / `56.852777` versus
`60.671924` / `56.879929` control). A synchronized profile kept gate/up at
about `58-61%` and down at about `25-29%` of profiled routed-FFN time, so
epilogue-only fusion is not a material software-pipeline lever. Sprint 155
then implemented a true opt-in stream-per-expert routed-FFN pipeline for the
current non-interleaved fused gate/up pack. The path proved active on V100
(`group_pipeline_calls=6` in a profiled stage smoke), but served throughput
regressed: 128-slot/32K was `59.125703` generated / `55.430346` continuation
tok/s versus `59.394915` / `55.682733` control, and 256-slot/16K was
`60.308689` / `56.539396` versus `60.648138` / `56.857630` control. The flag
remains diagnostic-only; the next material FFN path must remove launches/stream
joins with a persistent fused boundary or continue the bounded 2-way TP
prototype. Sprint 156 retested that path with the exact observed six active
expert groups. The manual six-group diagnostic was slightly positive at
128-slot/32K (`59.645848` generated / `55.917982` continuation tok/s versus
`59.516392` / `55.796618` control) and at 256-slot/16K (`60.675527` /
`56.883307` versus `60.442968` / `56.665283` control), but hardcoding six
groups is not safe for arbitrary serving traffic. A safe auto-group mode was
implemented and passed full scheduler smoke, but its host active-group readback
regressed served throughput. The group pipeline therefore remains diagnostic;
the next material path is still a persistent/larger fused routed-FFN executor.
Sprint 157 added an opt-in CUDA Graph replay probe around the TurboMind
routed-FFN core. It built and passed full 43-layer graph-off and graph-on
scheduler smokes, but served 128-slot/32K capture failed in the current
legacy-default-stream kernel path. Graph-disabled control was `59.607704`
generated / `55.882222` continuation tok/s; graph enabled with stable scratch
was correct but measured `59.450666` / `55.734999` with zero captures, and the
thread-local capture variant measured `59.367233` / `55.656781` with zero
captures. The graph flag remains diagnostic-only; real graph replay would
require threading an explicit stream through the routed-FFN executor.
Sprint 158 added `DS4_V100_TURBOMIND_ROUTED_EXECUTOR` and a guarded fixed96
routed gate_up executor for the 16-slot/256K product shape. Full 43-layer
scheduler smoke proved the intended fused-kernel shape is legal
(`total_routes=96`, six compact active experts, 16 routes per expert) and
selected the fixed gate_up kernel. Served 16-slot/256K A/B was correct but did
not select fixed96 because the HTTP path is currently reaching the routed FFN
as one request at a time (`total_routes=6`). The final guard avoids overhead on
that served shape: control measured `46.113721` generated / `43.231614`
continuation tok/s and guarded fixed96 measured `46.167311` / `43.281854`.
The flag remains explicit opt-in. The next material issue is served batch
formation for `>=256K`, or a topology path that makes the executor dense
without relying on current HTTP coalescing.

The default stack still uses the Sprint 111 fused TurboMind gate/up appliance,
Sprint 115 shared gate/up SwiGLU F8 HMMA, Sprint 116 batched
attention-projection F8 HMMA for active 4/8-slot batches, and Sprint 119
event-ordered handoff for multi-slot per-step serving. Sprint 128 adds compact
TurboMind expert scheduling as a default routed-FFN optimization. Sprint 122
confirms that chunking slots to expose wider batch kernels is slower in the
current topology because it gives up too much stage overlap.

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Sprint 146 256-slot 16K control repeat | 16,384 | 256 | `61.223893` | `57.397400` | 256/256 token match |
| Sprint 146 gate `m128_1536` opt-in | 16,384 | 256 | `61.204203` | `57.378940` | 256/256 token match |
| Sprint 145 256-slot 16K throughput ceiling | 16,384 | 256 | `61.065087` | `57.248519` | 256/256 token match |
| Sprint 156 six-group pipeline diagnostic | 16,384 | 256 | `60.675527` | `56.883307` | 256/256 token match |
| Sprint 156 same-binary control | 16,384 | 256 | `60.442968` | `56.665283` | 256/256 token match |
| Sprint 156 safe auto-group diagnostic | 16,384 | 256 | `60.232265` | `56.467748` | 256/256 token match |
| Sprint 154 256-slot 16K down-reduce control | 16,384 | 256 | `60.671924` | `56.879929` | 256/256 token match |
| Sprint 154 256-slot 16K down-reduce opt-in | 16,384 | 256 | `60.642962` | `56.852777` | 256/256 token match |
| Sprint 155 active group-pipeline opt-in | 16,384 | 256 | `60.308689` | `56.539396` | 256/256 token match |
| Sprint 145 192-slot 16K midpoint | 16,384 | 192 | `60.700926` | `56.907118` | 192/192 token match |
| Sprint 145 128-slot 16K control | 16,384 | 128 | `59.860493` | `56.119213` | 128/128 token match |
| Sprint 139 gated m128 auto probe | 32,768 | 128 | `60.130047` | `56.371919` | 128/128 token match |
| Sprint 157 graph-disabled control | 32,768 | 128 | `59.607704` | `55.882222` | 128/128 token match |
| Sprint 157 graph stable-scratch global capture probe | 32,768 | 128 | `59.450666` | `55.734999` | 128/128 token match, 0 captures |
| Sprint 157 graph stable-scratch thread-local capture probe | 32,768 | 128 | `59.367233` | `55.656781` | 128/128 token match, 0 captures |
| Sprint 156 six-group pipeline diagnostic | 32,768 | 128 | `59.645848` | `55.917982` | 128/128 token match |
| Sprint 156 same-binary control | 32,768 | 128 | `59.516392` | `55.796618` | 128/128 token match |
| Sprint 156 safe auto-group diagnostic | 32,768 | 128 | `58.988662` | `55.301871` | 128/128 token match |
| Sprint 154 128-slot 32K down-reduce opt-in | 32,768 | 128 | `59.509317` | `55.789985` | 128/128 token match |
| Sprint 154 128-slot 32K down-reduce control | 32,768 | 128 | `59.502747` | `55.783825` | 128/128 token match |
| Sprint 148 gate `m128_s4` opt-in | 32,768 | 128 | `60.049057` | `56.295991` | 128/128 token match |
| Sprint 148 same-binary control | 32,768 | 128 | `59.865668` | `56.124063` | 128/128 token match |
| Sprint 140 gated down-probe-off control | 32,768 | 128 | `60.129772` | `56.371661` | 128/128 token match |
| Sprint 141 scalar route-row reduce repeat | 32,768 | 128 | `60.112248` | `56.355232` | 128/128 token match |
| Sprint 141 control repeat | 32,768 | 128 | `60.108232` | `56.351468` | 128/128 token match |
| Sprint 141 half2 route-row reduce | 32,768 | 128 | `60.104512` | `56.347980` | 128/128 token match |
| Sprint 141 indexed-A repeat | 32,768 | 128 | `60.056960` | `56.303400` | 128/128 token match |
| Sprint 142 down-reduce epilogue opt-in | 32,768 | 128 | `60.041003` | `56.288440` | 128/128 token match |
| Sprint 141 route-row reduce earlier repeat | 32,768 | 128 | `60.022743` | `56.271322` | 128/128 token match |
| Sprint 144 control with split metrics | 32,768 | 128 | `59.993301` | `56.243719` | 128/128 token match |
| Sprint 144 gate m64n256 probe | 32,768 | 128 | `59.797232` | `56.059905` | 128/128 token match |
| Sprint 144 down m64n256 probe | 32,768 | 128 | `59.791839` | `56.054849` | 128/128 token match |
| Sprint 140 gated down-probe-auto candidate | 32,768 | 128 | `60.038469` | `56.286064` | 128/128 token match |
| Sprint 142 down-reduce epilogue control | 32,768 | 128 | `59.987105` | `56.237910` | 128/128 token match |
| Sprint 139 gated probe-off control | 32,768 | 128 | `60.061899` | `56.308030` | 128/128 token match |
| Sprint 137 128-slot 32K throughput mode | 32,768 | 128 | `59.598172` | `55.873286` | 128/128 token match |
| Sprint 137 same-context control | 32,768 | 64 | `57.170428` | `53.597276` | 64/64 token match |
| Sprint 136 64-slot 64K throughput mode | 65,536 | 64 | `57.322945` | `53.740261` | 64/64 token match |
| Sprint 136 same-context control | 65,536 | 32 | `52.884400` | `49.579125` | 32/32 token match |
| Sprint 135 32-slot 128K throughput mode | 131,072 | 32 | `52.840889` | `49.538334` | 32/32 token match |
| Sprint 135 same-context control | 131,072 | 16 | `45.780913` | `42.919606` | 16/16 token match |
| Sprint 128 gated compact + route-row-reduce opt-in | 262,144 | 16 | `46.394722` | `43.495052` | 16/16 token match |
| Sprint 128 gated compact opt-in | 262,144 | 16 | `46.328184` | `43.432672` | 16/16 token match |
| Sprint 158 guarded fixed96 served path | 262,144 | 16 | `46.167311` | `43.281854` | 16/16 token match, fixed96 not selected in HTTP path |
| Sprint 158 same-binary control | 262,144 | 16 | `46.113721` | `43.231614` | 16/16 token match |
| Sprint 128 compact launcher default on fused appliance | 262,144 | 16 | `45.888778` | `43.020729` | 16/16 token match |
| Sprint 129 default dispatch control | 262,144 | 16 | `45.840691` | `42.975648` | 16/16 token match |
| Sprint 129 reuse dispatch probe | 262,144 | 16 | `45.813841` | `42.950476` | 16/16 token match |
| Sprint 130 compact fused control repeat | 262,144 | 16 | `45.837745` | `42.972886` | 16/16 token match |
| Sprint 131 compact fused indexed-A opt-in | 262,144 | 16 | `45.789937` | `42.928066` | 16/16 token match |
| Sprint 130 compact fused route-row-reduce repeat | 262,144 | 16 | `45.660765` | `42.806967` | 16/16 token match |
| Sprint 131 compact fused control repeat | 262,144 | 16 | `45.663281` | `42.809326` | 16/16 token match |
| Sprint 128 compact explicit on fused appliance | 262,144 | 16 | `45.747461` | `42.888244` | 16/16 token match |
| Sprint 128 gated compact-off same-binary control | 262,144 | 16 | `43.879880` | `41.137387` | 16/16 token match |
| Sprint 127 interleaved gated-SiLU opt-in | 262,144 | 16 | `43.933293` | `41.187462` | 16/16 token match |
| Sprint 123 best opt-in shared FFN fusion | 262,144 | 16 | `43.887206` | `41.144256` | 16/16 token match |
| Sprint 127 same-binary fused gate/up control | 262,144 | 16 | `43.691032` | `40.960343` | 16/16 token match |
| Sprint 126 no-profile same-binary sanity | 262,144 | 16 | `43.453309` | `40.737477` | 16/16 token match |
| Sprint 124 route-row reduce opt-in | 262,144 | 16 | `43.822500` | `41.083593` | 16/16 token match |
| Sprint 125 output-A rows2 batch opt-in | 262,144 | 16 | `43.640921` | `40.913364` | 16/16 token match |
| Sprint 125 output-A HMMA plus output-B batch opt-in | 262,144 | 16 | `43.245208` | `40.542383` | 16/16 token match |
| Sprint 124 same-binary control repeat | 262,144 | 16 | `43.517862` | `40.797995` | 16/16 token match |
| Sprint 123 shared-down-add plus scalar shared-pair fusion | 262,144 | 16 | `43.812630` | `41.074340` | 16/16 token match |
| Sprint 123 same-binary fused-add control | 262,144 | 16 | `43.070728` | `40.378807` | 16/16 token match |
| Sprint 122 production-auto 16-slot throughput mode | 262,144 | 16 | `43.534061` | `40.813182` | 16/16 token match |
| Sprint 122 best observed 16-slot candidate | 262,144 | 16 | `43.730215` | `40.997076` | 16/16 token match |
| Sprint 121 16-slot throughput mode | 262,144 | 16 | `43.659461` | `40.930745` | 16/16 token match |
| Sprint 121 same-binary 8-slot control | 262,144 | 8 | `34.445844` | `32.292979` | 8/8 token match |
| Sprint 120 current default repeat | 262,144 | 8 | `34.490294` | `32.334651` | 8/8 token match |
| Single scalar fusion opt-in repeat | 262,144 | 8 | `34.689964` | `32.521841` | 8/8 token match |
| Single row-pair fusion opt-in | 262,144 | 8 | `34.380968` | `32.232157` | 8/8 token match |
| Event-ordered handoff default | 262,144 | 8 | `34.433252` | `32.281173` | 8/8 token match |
| Event-ordered handoff default | 1,048,576 | 4 | `21.771077` | `20.410385` | 4/4 token match |
| Batched attention projection F8 HMMA default | 262,144 | 8 | `33.697698` | `31.591592` | 8/8 token match |
| Single-token HMMA opt-in | 262,144 | 8 | `16.083451` | `15.078235` | 8/8 token match |
| Sprint 118 same-binary control | 262,144 | 8 | `33.502249` | `31.408359` | 8/8 token match |
| Per-slot shared pair-SwiGLU opt-in | 262,144 | 8 | `33.562643` | `31.464978` | 8/8 token match |
| Async slot chunk 4 opt-in | 262,144 | 8 | `11.483646` | `10.765918` | 8/8 token match |
| Promoted launcher default repeat | 262,144 | 8 | `33.540586` | `31.444300` | 8/8 token match |
| Pair-SwiGLU F8 HMMA default | 262,144 | 8 | `33.578236` | `31.479596` | 8/8 token match |
| Pair+down F8 HMMA opt-in | 262,144 | 8 | `33.674684` | `31.570016` | 8/8 token match |
| Production fused gate_up appliance | 262,144 | 8 | `33.589285` | `31.489955` | 8/8 token match |
| Shared-down F8 HMMA opt-in | 262,144 | 8 | `33.550415` | `31.453514` | 8/8 token match |
| Direct FFN delta opt-in | 262,144 | 8 | `33.360404` | `31.275379` | 8/8 token match |
| Same-binary separate gate/up control | 262,144 | 8 | `31.312694` | `29.355651` | 8/8 token match |
| Batched attention projection F8 HMMA default | 1,048,576 | 4 | `21.469010` | `20.127197` | 4/4 token match |
| Pair-SwiGLU F8 HMMA default | 1,048,576 | 4 | `21.455638` | `20.114660` | 4/4 token match |
| Production fused gate_up appliance | 1,048,576 | 4 | `21.403909` | `20.066165` | 4/4 token match |
| Shared-down F8 HMMA opt-in | 1,048,576 | 4 | `21.396331` | `20.059061` | 4/4 token match |
| Small-route opt-in | 1,048,576 | 4 | `20.249531` | `18.983935` | 4/4 token match |
| Pair+down F8 HMMA opt-in | 1,048,576 | 4 | `21.370925` | `20.035242` | 4/4 token match |

The last pre-Sprint107 committed baseline was Sprint 104 at `31.451185`
generated tok/s for 8-slot/256K and `20.026385` for 4-slot/1M.

## Recent Experiments

| Sprint | Experiment | Result | Decision |
|---|---|---|---|
| 213 | Routed FFN materialized split-reduce gate | V100 build passed; symbol exported; focused split-reduce correctness passed and improved the focused FFN sequence `0.1391 ms -> 0.1290 ms`; full scheduler smoke passed; served A/B was `60.655009` vs `60.236036` continuation tok/s, `16/16` token match, `43` graph captures and `129` launches with `0` failures | Keep as diagnostic/default-off; do not promote; next sprint should build a tile-local/persistent routed-FFN workbench |
| 212 | TP4/PP1 low-bit layer body pivot | New separate TP-only `tools/ds4-v100-tp4-turbomind-layer-smoke` built on V100; correctness passed at 96/192/384 routes; TP4 compute speedups were `2.335x`, `2.597x`, `3.707x`; resident-reduce total speedups were `1.078x`, `0.932x`, `0.967x` | Do not build TP4/PP1 runtime ownership next; return to monolithic/persistent low-bit routed-FFN or a better collective |
| 211 | TP8 TurboMind MXFP4 expert body | Separate TP-only TP8 low-bit smoke ran the public TurboMind ABI, but `mid_shard=256` failed correctness at 96/192/384 routes with NaNs despite `3.927x-4.189x` compute-only speedup | Reject current TP8 MXFP4 shard shape; pivot to TP4 control or design a shard-256 kernel |
| 210 | TP8 real layer-body fixture | Separate TP-only FP16 Tensor Core layer body passed 32/64/128 token gates with `0.614750/0.709350/0.796927 ms` total latency | Continue TP8 only by replacing the FP16 fixture with real low-bit expert work |
| 209 | TP8 one-layer prototype | Separate TP-only one-layer smoke passed 32/64/128 token gates with sharded 32-slot/256K F8 KV and total latencies `0.739408/0.876011/1.098461 ms` | Continue TP8 in new TP-only files; no PP scheduler integration |
| 208 | Separate TP8 investigation path | TP8 planner/probes showed 32-slot/256K fits with sharded KV at `26.84 GiB` worst GPU and the 43-layer boundary proxy measured `29.381/32.605/37.995 ms` at 32/64/128 tokens | Continue bounded TP8 prototypes in separate files |
| 205 | Async root resident TP4 reduction | V100 build passed; `root_async` correctness passed; speedups were `0.970x` at 96 routes x 4 layers, `0.866x` at 768 routes x 4 layers, and `0.860x` at 96 routes x 43 layers | Reject root_async; pause TP4 production decode branch and pivot to persistent fused routed-FFN |
| 204 | Concurrent resident TP4 reduction | V100 build passed; `doubling_async` correctness passed at 96/768 routes; 43-layer 768-route speedup was `1.071x`, but 43-layer 96-route repeat was `0.896x` | Keep TP4 for larger batched/prefill investigation only; do not integrate production decode scheduler without a fused/NCCL-grade collective |
| 203 | Resident TP4 layer-slice gate | V100 build passed; resident TP4 correctness passed at 6/96/768 routes; 43-layer root speedup was `0.825x` at 96 routes and `0.589x` at 768 routes; hand-rolled doubling was slower than root in 4-layer tests | Do not wire this TP4 boundary into production; next TP work needs a real concurrent collective/fused reduction, otherwise pivot back to persistent fused routed-FFN |
| 202 | TP4 routed-FFN compute envelope | V100 build passed; fixed a GPU0 stream/workspace overlap in the benchmark warmup; real TurboMind MXFP4 TP4 split correctness passed at 6/96/768 routes; corrected compute-only speedup was `2.686x`, `2.350x`, `3.636x`; copy-inclusive speedup was `0.986x`, `0.783x`, `0.682x` | TP4 compute is strong enough for full-layer TP/EP, but routed-only full-hidden copy overlays are rejected |
| 201 | TP4 full-layer boundary proxy | V100 build passed; 16-token/43-layer/4-collective boundary measured `22.113369 ms` root and `24.414061 ms` doubling, both verified; 64-token doubling measured `34.830881 ms`, 128-token doubling measured `51.026125 ms` | Full-layer TP4/EP remains plausible only as a broad topology that keeps dense+routed compute inside the boundary; do not expand routed-only TP overlays |
| 103 | Exact-bit E4M3 F8 decode replacing `ldexpf()` | Improved 8-slot/256K to `30.862791` generated tok/s and 4-slot/1M to `19.733742` | Shipped |
| 104 | Warp-shuffle reductions for hot F8 arena kernels | Improved 8-slot/256K repeat to `31.451185`; 4-slot/1M to `20.026385` | Shipped; current baseline |
| 105 | Extend warp reductions to BF16/F32 matmuls | Correct, but repeat result was inside Sprint 104 band | Rejected and reverted |
| 106 | Warm served `nvprof` profile of Sprint 104 | F8 rows2/grouped rows2 were ~51% GPU time; TurboMind SM70 MXFP4 was ~25%; GPU memcpy traffic was small | Use profile to choose next kernel target |
| 107 | DS4-specific grouped F8 rows2 attention-output-A kernel | Correct and faster for 8-slot/256K; neutral for 4-slot/1M | Shipped |
| 108 | TurboMind small-route count/prefix/scatter fusion | Correct; 8-slot repeat was `31.759013` opt-in vs `31.794180` rollback, while 4-slot/1M was `20.249531` opt-in vs `20.081695` rollback | Kept opt-in |
| 109 | F8 four-output-row CTA probe | Correct; regressed 8-slot/256K to `30.998275` vs `31.380225` control and 4-slot/1M to `19.898462` vs `20.041787` control | Rejected as default; opt-in only |
| 110 | TurboMind fused gate+up grouped-GEMM probe | Correct; `1.504x`, `1.532x`, and `1.462x` faster at 6, 24, and 48 total routes | Proceed to appliance implementation |
| 111 | Production fused TurboMind gate_up appliance | Correct; 8-slot/256K improved to `33.430971` from `31.312694` same-binary separate control | Shipped/default for fused packs |
| 112 | Fused appliance profile plus F8 warp-scale probe | Profile showed F8 row-pair/grouped kernels at `54.58%` GPU time; warp-scale was correct but regressed 8-slot/256K to `29.009399` vs `33.484099` control | Kept opt-in/off |
| 113 | Direct FFN delta accumulation and cached FFN input ptr table | Correct, but `33.360404` vs `33.589285` control at 8-slot/256K | Kept opt-in/off |
| 114 | DS4-shaped shared-down F8 HMMA batch kernel | Correct; `33.550415` vs `33.397763` control at 8-slot/256K and `21.396331` vs `21.365610` at 4-slot/1M | Kept opt-in/off |
| 115 | DS4-shaped shared gate/up SwiGLU F8 HMMA kernel | Correct; `33.578236` vs `33.292541` control at 8-slot/256K and `21.455638` vs `21.430420` at 4-slot/1M | Shipped/default |
| 116 | DS4-shaped attention projection F8 HMMA batch kernel | Correct; `33.697698` vs `33.380614` control at 8-slot/256K and `21.469010` vs `21.333447` at 4-slot/1M | Shipped/default for active 4/8-slot batches |
| 117 | F8 wrapper shape trace and per-slot shared gate/up/SwiGLU fusion | Correct; trace showed the fast path is per-slot stage-pipelined, chunk-4 batching dropped to `11.483646`, and scalar shared-pair fusion reached `33.562643` | Kept opt-in/off; next target should be software-pipelined/Tensor-Core fusion |
| 118 | Single-token HMMA for the hot `4096 x 8192` F8 projection | Correct and traced, but regressed to `16.083451` vs `33.502249` same-binary control | Kept opt-in/off; naive n=1 WMMA is not viable |
| 119 | Event-ordered stage handoff | Correct; `34.433252` vs `33.379839` at 8-slot/256K and `21.771077` vs `21.566859` at 4-slot/1M | Shipped/default as `DS4_V100_ASYNC_EVENT_HANDOFF=auto` |
| 120 | Single shared gate/up/SwiGLU row-pair probe | Correct; `34.380968` row-pair vs `34.490294` default and `34.689964` scalar single-fusion at 8-slot/256K | Kept opt-in/off; row-pair compaction does not beat the default |
| 121 | 16-slot 256K throughput mode | Correct; `43.659461` at 16-slot/256K vs `34.445844` same-binary 8-slot control | Shipped as admitted 256K mode with context-aware launcher guard |
| 122 | 16-slot profile, 16-token HMMA admission, async chunk probes, and rendezvous stabilization | Correct; best `43.730215`, production-auto `43.534061`, one 16-request tensor batch after 200 ms auto wait; chunked tensor scheduling regressed (`28.876459` at chunk 2, `18.447169` at chunk 4, `13.315378` at chunk 16) | Shipped 16-slot auto rendezvous; kept chunk/output-B/shared-down probes opt-in/off |
| 123 | Production-path shared FFN fusion A/B | Correct; scalar shared-pair fusion reached `43.887206`, fused shared-down-add reached `43.539555`, and combined scalar+down-add reached `43.812630` at 16-slot/256K | Kept opt-in/off; launch/epilogue fusion alone is not enough |
| 124 | TurboMind route-row reduce replacing packed output clear plus atomic scatter-add | Correct; first candidate reached `43.822500`, but the repeat was `42.998450` vs `43.517862` control repeat at 16-slot/256K | Kept opt-in/off; routed-FFN tail fusion alone is not enough |
| 125 | Batched grouped attention output-A probe | Correct; output-A rows2 batching reached `43.640921`, rows2 A+B reached `43.619996`, and HMMA A+B reached `43.245208` vs `43.503005` control at 16-slot/256K | Kept opt-in/off; another single projection boundary is too small |
| 126 | Production routed-expert stage profiler | Correct; full 43-layer profile showed fused gate/up at `47.0%`, down at `23.4%`, route build at `16.8%`, and SwiGLU at only `3.2%` of profiled routed-FFN time; no-profile served sanity was `43.453309` generated tok/s | Shipped default-off diagnostic; next target should be TurboMind gated epilogue/interleaved pack or deeper persistent routed-expert pipeline |
| 127 | TurboMind gated-SiLU epilogue with interleaved fused gate/up appliance pack | Correct; standalone grouped test showed `1.47x-1.55x` speedup vs separate gate/up, full 43-layer gated profile removed standalone SwiGLU and dropped profiled routed-FFN total from `28.242 ms` to `26.734 ms`, served A/B was `43.933293` vs `43.691032` control | Keep opt-in/off; format and epilogue fusion are valid, but the next material step is a persistent routed-expert pipeline |
| 128 | TurboMind compact active-expert schedule | Correct; compact schedule passed full 43-layer smokes on both the interleaved gated and existing fused appliances, improved served A/B from `43.879880` to `46.328184` on the gated appliance, and the launcher-default fused appliance reached `45.888778` | Shipped/default as `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`; keep gated-SiLU and route-row-reduce opt-in |
| 129 | TurboMind dispatch policy probe | Correct for `default` and `reuse`; full scheduler `measure` aborted inside TurboMind's measurer, while served `reuse` was `45.813841` vs `45.840691` default | Keep default dispatch; guard unsafe measure/append; move to DS4-specific persistent routed-FFN |
| 130 | Routed FFN software-pipeline targeting | Correctness held; compact fused route-row-reduce repeated at `45.660765` vs `45.837745` control, confirming final scatter/reduce fusion is not the lever | Keep route-row-reduce opt-in; next code should target the packed MXFP4 gate/up mainloop with DS4-specific software pipelining |
| 131 | TurboMind indexed-A routed activation probe | Correct; full 43-layer smokes passed with indexed-A off/on, and served A/B was `45.789937` vs `45.663281` control | Keep indexed-A opt-in; wrapper-level activation compaction is correct but not a promotion-level win |
| 132 | Production-shaped TurboMind gate/up benchmark | Correct; historical 6/24/48-route cases still pass, and the 96-route served-profile case shows gated-SiLU at `0.1776 ms` vs `0.2889 ms` separate gate+up | Use this as the benchmark harness for any lower-level SM70 mainloop probe; no appliance default change |
| 133 | Compact-group gate/up benchmark correction | Correct; at 96 routes, sparse256 gated is `0.2128 ms` while compact gated is `0.1740 ms`, and compact separate gate+up is already `0.1895 ms` | Future probes must beat compact gated, not sparse grouped overhead |
| 134 | Fixed-shape compact gate/up ABI probe | Correct; direct fixed SM70 launch was bit-identical and `0.1746 ms` vs `0.1746 ms` generic gated | Do not promote; generic TurboMind already selects this effective path |
| 135 | 32-slot 128K throughput admission | Correct; full 43-layer smoke passed, and 32-slot 128K served at `52.840889` vs `45.780913` same-context 16-slot control | Ship as explicit short-context throughput mode; test wider short-context admission and lower-level software-pipelined kernels next |
| 136 | 64-slot 64K throughput admission | Correct; full 43-layer smoke passed, and 64-slot 64K served at `57.322945` vs `52.884400` same-context 32-slot control | Ship as explicit short-context throughput mode; diminishing slot-width returns make software-pipelined expert kernels the next major lever |
| 137 | 128-slot 32K throughput admission | Correct; full 43-layer smoke and status/metrics confirmed 128 slots, and served throughput reached `59.598172` vs `57.170428` same-context 64-slot control | Ship as explicit short-context throughput mode; stop treating admission width as the main lever and move to software-pipelined expert kernels |
| 138 | Wide compact TurboMind gate/up benchmark | Correct; default compact benchmark now covers up to 768 routed rows, where fused gate_up is `0.6379 ms` and gated-SiLU is `0.6481 ms` | Use `0.638 ms` as the acceptance target for the next packed MXFP4 software-pipelined kernel probe |
| 139 | Fixed-shape 128-slot gate/up probe | Correct; the 768-route m128 probe measured `0.5999 ms` vs `0.6480 ms` generic gated in isolation, passed full 43-layer 128-slot smoke, and served at `60.130047` vs `60.061899` probe-off | Keep guarded auto selection, but do not treat gate/up-only fusion as the remaining major lever |
| 140 | Fixed-shape 128-slot down probe | Correct; down m128 measured `0.3026 ms` vs `0.3272 ms` generic in isolation and full 43-layer smoke passed, but served A/B was `60.038469` vs `60.129772` down-probe-off | Keep down probe opt-in/off; move to down epilogue plus weighted reduce or a persistent routed-FFN executor |
| 141 | Half2 route-row reduce tail probe | Correct; full 43-layer 128-slot smoke passed, but 128-slot served A/B was neutral: half2 route-row reduce `60.104512`, scalar route-row reduce `60.112248`, control `60.108232` | Keep half2 reduce opt-in/off; separate tail-kernel vectorization is not enough |
| 142 | TurboMind down-epilogue reduce probe | Correct; full 43-layer 128-slot smoke passed and served A/B was `60.041003` vs `59.987105` control | Keep off by default; atomic epilogue fusion proves the integration boundary but is not a material throughput win |
| 143 | Prefill/decode metric split | Correct; V100 one-request smoke reported aggregate prompt `6.841274`, generated `0.760142`, continuation `0.380071`, and response-local prompt/decode rates | Ship benchmark visibility change; no runtime default change |
| 144 | SM70 MXFP4 m64n256 tile probe | Correct; full 43-layer smoke passed, but served 128-slot/32K A/B regressed: control `59.993301`, down `m64n256` `59.791839`, gate `m64n256` `59.797232` | Keep explicit opt-in only; larger routed-FFN executor work is still the next lever |
| 145 | 256-slot 16K short-context admission | Correct; planner worst GPU was `29.07 GiB / 32.00 GiB` including reserve, full 43-layer smoke passed, and served 16K runs reached `59.860493` at 128 slots, `60.700926` at 192 slots, and `61.065087` at 256 slots | Ship guarded 256-slot admission for `ctx <= 16K`; simple slot widening is now mostly exhausted |
| 146 | 1536-route fixed-shape gate/up and down probes | Correct; standalone gate `m128_1536` improved to `0.9435 ms` vs `0.9651 ms`, but served A/B was `61.204203` vs `61.223893` control and continuation/decode was `57.378940` vs `57.397400` | Keep explicit opt-in only; do not select 1536 probes from `auto` |
| 147 | 1536-route down-reduce correctness checkpoint | Correct; the down GEMM route-weighted F32 accumulation epilogue covers 1536 routes and passed full 43-layer 256-slot smoke | Keep opt-in pending served A/B |
| 148 | Stage-4 fused gate/up software-pipeline probe | Correct; isolated `m128_s4` improved the 768-route probe but served A/B was only `60.049057` vs `59.865668`, inside the run band | Keep stage-count variants explicit opt-ins |
| 149 | TP split and P2P topology probe | Correct; ideal 2-way FFN split was positive before communication and NV2 payloads were fast enough for a bounded prototype | Prototype only on clean NV2 pairs |
| 150 | Two-GPU TP split proxy | Correct; 768 routes were positive after copies, but 1536 routes were neutral to slower | Scope TP to 128-slot/32K first |
| 151 | Two-GPU TP correctness gate | Correct; full one-GPU output matched the sum of two TP partials at 768 and 1536 routes on NV2 pairs | Split math accepted; scheduler remains the risk |
| 152 | Fused gate/up stage-count sweep | Correct; 2/3/4-stage variants were neutral at 768 and 1536 routes, and NCU counters were flat | Stop tuning gate/up stage count |
| 153 | Bounded TP pack contract | Correct; `--emit-tp-split` emits split gate/up and down rows and context binding accepts TP descriptors. Real 2-GPU NV2 proxy was `1.157x` at 768 routes and `0.912x` at 1536 routes | Keep TP to a one-layer 128-slot/32K prototype |
| 154 | Fused routed-FFN boundary validation | Correct; down-reduce epilogue served A/B was flat at 128-slot/32K and slightly slower at 256-slot/16K, while profile kept gate/up at `~58-61%` and down at `~25-29%` | Keep down-reduce opt-in; next work must change gate/up+down execution, not only the epilogue |

## Sprint 106 Profile Takeaway

The profile does not point at disk or host RAM as the decode bottleneck.
`cudaMemcpy` API accounting is noisy, but GPU memcpy time was tiny. The main
device buckets were:

- F8 rows2 arena matmul: `38.97%`
- F8 grouped rows2 arena matmul: `12.39%`
- TurboMind SM70 MXFP4 grouped GEMM: `25.42%`

That makes the practical next targets F8 execution shape and TurboMind routed
expert scheduling.

## Current Shipped Change

Sprint 107 adds a guarded DS4-specialized CUDA kernel for the fixed grouped
attention-output-A shape:

- groups: `8`
- rows per group: `1024`
- columns per group: `4096`
- fallback: existing generic grouped rows2 kernel
- rollback knob: `DS4_V100_CUDA_F8_GROUPED_DS4_FAST=0`

Validation already completed on the cluster:

- `cuda_source_dtypes_smoke`: passed
- `cuda_v100_projection_attention_smoke`: passed
- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- `cuda_v100_selected_token_smoke --expected-token-hex 3136`: passed

Throughput completed:

- 8-slot/256K fast: `31.811137` generated tok/s, `8/8`
- 8-slot/256K fast repeat: `31.630774` generated tok/s, `8/8`
- 8-slot/256K rollback: `31.098630` generated tok/s, `8/8`
- 4-slot/1M fast: `20.095510` generated tok/s, `4/4`
- 4-slot/1M rollback: `20.105807` generated tok/s, `4/4`

Remaining optional validation:

- Focused profile to confirm whether grouped F8 kernel time moved.

## Current Opt-In Probe

Sprint 108 adds a guarded small-route TurboMind route builder:

- combines route count, prefix, and scatter into one one-block kernel for the
  production small-route shape;
- is controlled by `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD`;
- remains disabled by default because it did not improve the 8-slot/256K
  practical target.

Validation completed on the cluster:

- `cuda_v100_turbomind_adapter_smoke`: passed
- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- selected-token smoke with small-route on: passed, token id `926`, hex `3136`
- selected-token smoke with small-route off: passed, token id `926`, hex `3136`
- rebuilt default check: `turbomind_small_route_build=0`, selected-token passed

Sprint 109 adds a guarded F8 row4 CTA probe:

- computes four large F8 output rows per CTA for the ungrouped and DS4 grouped
  attention-output paths;
- is controlled by `DS4_V100_CUDA_F8_ROW4`;
- remains disabled by default because it reduced throughput in both measured
  serving tiers.

Validation completed on the cluster:

- `cuda_source_dtypes_smoke`: passed
- `cuda_v100_projection_attention_smoke`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- selected-token smoke with row4 on: passed, token id `926`, hex `3136`

Sprint 110 adds a standalone TurboMind fused gate/up benchmark:

- shape: `K=4096`, `N=2048`, fused `N=4096`, 256 experts;
- route set: six sparse active experts;
- result: fused gate_up is `1.46x-1.53x` faster than separate gate and up
  grouped calls;
- correctness: exact output match for both halves of the fused tensor.

Sprint 111 ships that fused gate/up result into the appliance:

- packer emits `blk.N.ffn_gate_up_exps.weight` with `--fuse-gate-up`;
- runtime defaults to the fused path with `DS4_V100_TURBOMIND_FUSED_GATE_UP=1`;
- selected-token smoke passed with token id `926`, hex `3136`;
- full scheduler smoke passed with `tm_layers=43`;
- 8-slot/256K served A/B improved from `31.312694` to `33.430971`
  generated tok/s;
- 4-slot/1M fused sanity passed at `21.403909` generated tok/s.

Sprint 112 profiles the fused appliance and tests a narrow F8 scale-hoist
variant:

- fused 8-slot/256K profile reached `33.972205` generated tok/s under the
  profiler harness and preserved `8/8` token matches;
- F8 row-pair plus DS4 grouped attention-output kernels were `54.58%` of GPU
  time after Sprint 111;
- warp-broadcast E8M0 scale loading passed source/projection, scheduler, and
  selected-token correctness;
- same-binary 8-slot/256K A/B regressed from `33.484099` to `29.009399`
  generated tok/s, so `DS4_V100_CUDA_F8_WARP_SCALE=0` remains the default.

Sprint 113 tests direct FFN delta accumulation:

- batch scratch now exposes contiguous FFN norm/delta tensors with stable
  per-slot views;
- TurboMind routed FFN wrappers can consume an existing device pointer table;
- TurboMind routed FFN wrappers can accumulate into an existing output tensor;
- selected-token correctness passed with `DS4_V100_FFN_DIRECT_DELTA=1`;
- same-binary 8-slot/256K A/B was `33.360404` generated tok/s with direct delta
  versus `33.589285` control, so `DS4_V100_FFN_DIRECT_DELTA=0` remains the
  default.

Sprint 114 tests a DS4-shaped shared-down F8 HMMA batch kernel:

- the kernel is guarded by `DS4_CUDA_F8_HMMA_SHARED_DOWN=1`;
- it only dispatches for `rows=4096`, `cols=2048`, and `n_tokens=4/8`;
- focused target-shape smoke passed against the existing scalar F8 path;
- full scheduler and selected-token smokes passed with the fused appliance;
- same-binary A/B showed small positive deltas, but not enough to promote:
  `33.550415` vs `33.397763` at 8-slot/256K and `21.396331` vs `21.365610`
  at 4-slot/1M.

Sprint 115 ships a DS4-shaped shared gate/up SwiGLU F8 HMMA batch kernel:

- the kernel is guarded by `DS4_CUDA_F8_HMMA_PAIR_SWIGLU=1`;
- it only dispatches for `rows=2048`, `cols=4096`, and `n_tokens=4/8`;
- focused pair-SwiGLU smoke, full scheduler, and selected-token smokes passed;
- same-binary A/B improved both measured tiers:
  `33.578236` vs `33.292541` at 8-slot/256K and `21.455638` vs `21.430420`
  at 4-slot/1M;
- the combined pair+shared-down HMMA path reached `33.674684` at 8-slot/256K
  but regressed 4-slot/1M to `21.370925`, so only pair-SwiGLU HMMA is default.

Sprint 116 ships a DS4-shaped batched attention projection F8 HMMA path:

- the remaining profile after Sprint 115 still showed ungrouped F8 row-pair
  matmuls as the largest device bucket: `41.65%` GPU time and `12,341` calls;
- the new kernels cover `attn_q_a` (`1024 x 4096`), `attn_kv_latent`
  (`512 x 4096`), and `attn_q_b` (`32768 x 1024`) for active 4/8-slot batches;
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` and
  `DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=1` are now launcher defaults, while
  non-4/8-slot batches stay on the per-slot projection path unless
  `DS4_V100_ENABLE_BATCH_ATTN_PROJ_ANY=1` is set;
- focused CUDA smoke, full scheduler, and selected-token oracle all passed;
- same-binary A/B improved both measured tiers:
  `33.697698` vs `33.380614` at 8-slot/256K and `21.469010` vs `21.333447`
  at 4-slot/1M.

## Next Target

The next target still needs to change a larger execution boundary. Sprints
123-154 show that per-slot shared-FFN fusion, route-row tail fusion, dispatch
bypass, tile probes, fixed-shape 768/1536-route probes, stage-count tuning,
epilogue-only down-reduce fusion, bounded TP probes, and simple slot widening
are correct but too small to close the practical serving gap. Aggregate
throughput is still about `61` tok/s at 256-slot/16K and about `46-71` tok/s at
16-slot/256K depending on async-pipeline era and test harness, far below the
practical serving target. Sprint 194 adds a topology estimator and changes the
TP decision rule: the existing routed-only TP2 overlay should not be expanded,
because at 16-slot/256K it moves about `21.531 MiB` per token without changing
the dense execution shape, versus `7.000 MiB` for the current layer split. Full
TP/EP moves more wire bytes (`112.875 MiB` for TP4/PP1 at the same tier), so it
is only worth implementing if attention, shared FFN, routed experts, and output
ownership all become native to the topology. Sprint 195 then measured the first
TP4 primitive directly: a root gather/reduce/broadcast hidden collective is
correct on the V100 NVLink islands, but costs about `0.11 ms` for the
16-token/4096-hidden decode payload and only reaches about `27 GB/s` effective
wire bandwidth at larger payloads. The next sprint should not build production
TP4 on that root collective. It should either implement a real ring/tree/NCCL
TP4 collective inside one four-GPU NVLink island, or return to the monolithic
routed-FFN kernel that removes the global `mid_half` boundary.
Sprint 196 implemented the repo-owned tree-style option as recursive doubling.
It is correct and materially faster for larger payloads (`1.656 ms` vs
`3.676 ms` root at 1024 x 4096), but it is slower at the actual 16-token decode
payload (`0.134 ms` vs `0.111 ms`). That narrows the next production path:
direct TP4 collectives are not the decode lever unless fused into a larger
persistent boundary. The next sprint should prioritize the monolithic
routed-FFN or persistent layer boundary, while keeping the doubling collective
as the baseline for later TP4/prefill work.
Sprint 197 added the missing liveness contract to the TurboMind profile. The
current `fused6_reduce` production-shaped path now proves compact activation
staging is selected, `down_routes` is elided, and `mid_half` remains
materialized on every routed FFN call. The remaining `mid_half` buffer is only
`24576` bytes per six-route call, so a pure buffer-elision sprint is unlikely to
move serving throughput. The next implementation should target a persistent or
tile-level gate/up+down executor that reduces launch/GEMM boundary cost, or
use the TP4 doubling collective for larger batched/prefill shapes where it
actually wins.
Sprint 198 reopened CUDA graph replay for the current `fused6` /
`fused6_reduce` routed executor path. Direct replay with `fused6_reduce`
matched output IDs and improved continuation throughput from `16.022442` to
`17.980888` tok/s, with `43` graph captures, `129` launches, and `0` failures.
This is not promoted: Sprint 169 showed graph replay can help direct replay
while regressing served throughput. The required next gate is a same-binary
16-slot/256K served A/B before treating graph replay as a practical serving
optimization.
Sprint 199 ran that missing served gate and promoted the combined
`fused6_reduce + graph replay` stack for the Sprint 181+ production V100
appliance pack. Same-binary 16-slot/256K serving improved from `54.725463`
generated / `53.870377` continuation tok/s with `fused6_reduce` graph off to
`67.886268` / `66.825545` with graph on, with `16/16` token match. Against the
routed-executor-off production control (`56.719099` / `55.832863`), the
promoted stack is about `+19.7%` continuation. The launcher and V100 env
template now default to `DS4_V100_TURBOMIND_GATED_SILU=1`,
`DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce`, and
`DS4_V100_TURBOMIND_GRAPH=1` for the production pack.
Sprint 200 added the missing exact six-route focused TurboMind bench and
triggered the persistent-kernel stop condition. The fixed `m16_6` gated-SiLU
probe measured `0.1196 ms` versus `0.0946 ms` for the generic gated-SiLU path,
which confirms the Sprint 199 promoted stack should not use the fixed six-route
gate/up probe. The down-reduce output clear measured only `0.0022 ms`, while
six-route down-reduce with clear measured `0.0650 ms`; a clear-only fusion is
therefore not material. The next sprint should pivot to bounded full-layer
TP4/EP rather than adding another six-route wrapper ABI.
Sprint 201 added the bounded TP4 layer-boundary proxy and measured the full
43-layer communication envelope directly on the V100 pod. The 16-token target
shape costs `22-24 ms` before any DS4 compute, which is acceptable only if TP4
changes the whole layer execution shape. Larger active-token runs improve the
overhead-only envelope to `1837 tok/s` at 64 tokens and `2509 tok/s` at
128 tokens, so TP4 is more promising for high-batch throughput/prefill than
low-batch decode latency.
Sprint 202 measured the matching routed-expert compute side with a four-GPU
TurboMind MXFP4 split. It also fixed a benchmark warmup bug where the full
reference and shard 0 shared the GPU0 workspace on different streams. TP4
compute itself is strong (`2.35x-3.64x` at practical route counts), but
conservative routed-only input/output copies erase it (`0.68x-0.78x`). This
confirms that the next TP sprint must be a full-layer resident boundary or not
TP at all.
Sprint 203 implemented that resident boundary as a benchmark slice. Correctness
passes, but the naive root all-reduce boundary is still slower than the
one-GPU reference (`0.825x` at 96 routes over 43 layers, `0.589x` at 768
routes), and the simple doubling variant is worse in this implementation. TP4
production work should not continue into the scheduler until the collective is
made concurrent/fused; otherwise the next practical serving sprint should
return to a persistent fused routed-FFN executor.
Sprint 204 made that boundary concurrent with per-device async doubling. The
larger 768-route 43-layer shape improved to `1.071x`, but the production
96-route 43-layer repeat was `0.896x`. This keeps TP4 alive for larger
batched/prefill work, not for immediate production decode integration.
Sprint 205 tested async root gather/reduce/broadcast for small-payload decode.
It was correct but slower than the one-GPU reference and slower than
`doubling_async`, with `0.860x` at the 96-route 43-layer gate. The current TP4
decode branch is now blocked; the next sprint should return to persistent fused
routed-FFN work.

The concise current status is also tracked in
`docs/sprints/EXPERIMENT-STATUS.md`.
