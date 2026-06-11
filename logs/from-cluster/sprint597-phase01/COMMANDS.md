# Sprint 597 Phase -1/0/1 command log (pod llamacpp-build-8gpu, gpu-01)

## Phase -1 bootstrap (2026-06-11)

Pod prep (bare nvidia/cuda:12.2.2-devel-ubuntu22.04):

    apt-get update && apt-get install -y cmake git python3 curl numactl pciutils
    apt-get install -y cuda-nsight-systems-12-2   # nsys 2023.2.3 at /usr/local/cuda/bin/nsys

Repo shipped from laptop (excludes .git/build/logs/research/*.gguf):

    tar -C /Users/ravi/repos/ds4 --exclude=.git --exclude=build --exclude=logs \
        --exclude=research --exclude='*.gguf' -czf - . \
      | kubectl -n llm exec -i llamacpp-build-8gpu -- bash -c 'mkdir -p /workspace/ds4 && tar -xzf - -C /workspace/ds4'

FetchContent deps pre-cloned to /workspace/deps (GitHub reachable from pod, retries on DNS flake):
fmt@11.0.2, cutlass@v2.11.0, concurrentqueue@v1.0.4.

Build (see /workspace/s597-build.sh, log /workspace/s597-phase01-artifacts/build.log):

    cd /workspace/ds4
    cmake -S kernels/turbomind/ggml-turbomind -B build/turbomind-v100 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DFETCHCONTENT_SOURCE_DIR_FMT=/workspace/deps/fmt \
      -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/workspace/deps/cutlass \
      -DFETCHCONTENT_SOURCE_DIR_CONCURRENTQUEUE=/workspace/deps/concurrentqueue
    cmake --build build/turbomind-v100 --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j72
    CUDA_ARCH=sm_70 make -j72 tools/ds4-v100-appliance-pack tools/ds4-v100-tp-ep-pack-contract \
      tools/ds4-v100-replay appliance/ds4-v100-tp-ep-appliance

Pack regeneration (s181 production conventions: fused gate_up interleaved; log pack.log):

    cd /workspace/ds4 && ./tools/ds4-v100-appliance-pack \
      --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
      --source /models/DSv4-Flash-256e-fixed.gguf \
      --out-dir /workspace/packs/ds4-appliance-full-tm-gated-s597 \
      --pack-gpu 0 --fuse-gate-up-interleaved \
      --lib build/turbomind-v100/libggml-turbomind.so

Contract regeneration (defaults: ctx=262144 slots=32 kv f8_e4m3_b128, matching production):

    cd /workspace/ds4 && ./tools/ds4-v100-tp-ep-pack-contract \
      --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s597 \
      --out-dir /workspace/s597-contract

Env file: /workspace/s597-phase01-artifacts/s597.env
(APPLIANCE_DIR=/workspace/packs/ds4-appliance-full-tm-gated-s597,
 CONTRACT=/workspace/s597-contract/tp-ep-pack-contract.tsv, ctx=262144 slots=32 tokens=64)

Launcher check + probe smoke:

    cd /workspace/ds4 && ./tools/ds4-v100-run-tp-ep-appliance.sh --env /workspace/s597-phase01-artifacts/s597.env --check
    DS4_V100_TOKENS=2 DS4_V100_MAX_REQUESTS=3 ./tools/ds4-v100-run-tp-ep-appliance.sh --env ...   # then curl /health /v100/status /v100/selected-token
    # result: serving PASS, generated_token_sequence=[48177,3263] (probe-server.log)

## Phase 0

NOTE: bench harness pod copy patched: server-listen wait bumped from 180s to 900s
(sed 's/seq 1 180/seq 1 900/' tools/ds4-v100-tp-ep-http-bench.sh) because cold
weight load (~160 GB) can exceed 180s. No other harness change.

Leg A (promoted full capture, reference shape 32 slots/256K/64tok/128 req concurrent):

    cd /workspace/ds4 && DS4_V100_TP_EP_DECODE_GRAPH_MODE=full ./tools/ds4-v100-tp-ep-http-bench.sh \
      --log-dir /workspace/s597-phase01-artifacts/phase0-full \
      --tokens-cases 64 --requests 128 --concurrent-requests \
      --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s597 \
      --contract /workspace/s597-contract/tp-ep-pack-contract.tsv \
      --turbomind-lib /workspace/ds4/build/turbomind-v100/libggml-turbomind.so \
      --tp-ep-bin ./appliance/ds4-v100-tp-ep-appliance

Leg B (pure eager): same with DS4_V100_TP_EP_DECODE_GRAPH_MODE=eager,
--log-dir /workspace/s597-phase01-artifacts/phase0-eager.

Analysis: python3 /workspace/s597-phase01-artifacts/s597-phase0-analyze.py <case_dir> <label>

### Phase 0 harness deviations (pod copy of tools/ds4-v100-tp-ep-http-bench.sh only)

1. First Leg A attempt FAILED: (a) 128 simultaneous TCP connects vs server
   listen backlog 16 (appliance/http_server.cu:415) -> Errno 110 connect
   timeouts for ~80 requests; (b) generated_text bytes are not valid UTF-8 with
   the regenerated pack -> python decode errors. Server was killed, artifacts
   wiped, harness patched:
   - body.decode("utf-8") -> body.decode("utf-8", "replace") (2 sites)
   - concurrent submission in waves of 32 (one wave = one full 32-slot batch)
   No serving-side/source change. Result: 4 clean coalesced batches of 32.

### Phase 0 results

Leg A (full): 128/128 ok, 4x32 batches.
  all batches:  agg decode-domain 71.39 tok/s, wall 59.26 tok/s
  steady state (batches 2-4): decode-domain 73.59 tok/s, wall 61.47 tok/s
  per-layer-step replay: 10.113 ms steady (11.36 first batch incl capture)
  per-request decode: 64 tok / 27.8 s = 2.30 tok/s vs anchor 2.344 (-1.8%)
Tolerance-gate control artifact: /workspace/s597-phase01-artifacts/phase0-full-control/
  (response-N.txt x128, tools/ds4-v100-http-response-tolerance.py format)

Leg B (eager): 128/128 ok, 4x32 batches.
  steady state (batches 2-4): decode-domain 40.97 tok/s, wall 37.81 tok/s
  eager attribution per layer-step, steady (total 18.165 ms):
    EP 11.136 (61.3%) | attention 3.395 (18.7%) | compose 0.893 (4.9%)
    HC-current 5.552 (30.6%) | final_hc 0.534 (2.9%) | host-sync 0.863 (4.8%)
    (buckets are rank-local elapsed and overlap; do not force-fit to 100%)
  vs Sprint 581 anchors: EP 9.419->11.136 (+18.2%), total 14.445->18.165 (+25.8%)
  -> RE-ANCHOR. HC-current bucket grew 1.096->5.552 (post-MTP churn);
     attention 1.774->3.395. EP share 65.2%->61.3% (still dominant).

## Phase 1

Topology archived: nvidia-smi-topo.txt. 12 of 28 undirected pairs are SYS
(hybrid cube mesh confirmed). Every GPU has exactly 3 SYS peers.

Microbench (NEW standalone tool /workspace/ds4/tools/s597-peer-copy-microbench.cu):

    nvcc -O3 -arch=sm_70 -o /workspace/s597-phase01-artifacts/s597-peer-copy-microbench \
      /workspace/ds4/tools/s597-peer-copy-microbench.cu
    cd /workspace/s597-phase01-artifacts && ./s597-peer-copy-microbench > peer-copy-microbench.tsv

  384 KiB (the promoted fixed-capacity EP-return payload, 192x512 f32):
    self ~140 GB/s; NV2 ~29 GB/s (~13.5us); NV1 ~17.8 GB/s (~22us); SYS ~8.3 GB/s (~46-49us)
  24/56 directed pairs are SYS. Worst-dst serial EP return ~216us/layer -> 9.28ms/step;
  SYS excess ~87.6us/layer (avg dst) -> ~3.76ms/step.

In-situ nsys (s597-nsys-insitu.sh): nsys launch (session) + start/stop around one
32-slot x 8-tok replay batch on the unmodified promoted full default; first attempt
captured nothing because the launcher env file overrides exported DS4_V100_PORT
(driver hit 18200, server on 18080); fixed driver to 18080 and reran.

In-situ nsys result (second attempt, NSYS_INSITU_OK):
  capture = 19,264 EP-return copy kernels = 56 pairs x 43 layers x 8 steps.
  per-copy mean: NV2 11.1us, NV1 19.7us, SYS 1,990us (686-3,094 by pair).
  Ranking matches microbench; SYS magnitude is ~40x the isolated microbench
  value due to 24 concurrent SYS loads saturating PCIe/QPI.
  Per-dst serial EP return: 4.9-6.5 ms/layer (worst dst 2) of the 10.11 ms
  full-capture layer replay -> SYS remote loads are ~55-65% of the decode step
  (~280 ms/step worst rank). See PHASE1-FINDING.md.

  Analysis: python3 s597-nsys-analyze.py nsys-insitu.sqlite nvidia-smi-topo.txt
  (kernels identified by shortName like %copy_f32% AND gridX=384; per-dst
   consecutive groups of 7 map to src order 0..7 minus dst, per
   engine/decode_loop.cu:1176-1195 launch order.)

Artifacts in this directory:
  build.log pack.log probe-server.log s597.env
  phase0-full/ phase0-eager/ (bench logs, responses, server logs, gpu_util)
  phase0-full-control/ (tolerance-gate control, response-N.txt x128)
  nvidia-smi-topo.txt peer-copy-microbench.tsv (+ .cu source in /workspace/ds4/tools/)
  nsys-insitu.nsys-rep nsys-insitu.sqlite nsys-server.log nsys-insitu-run.log
  PHASE1-FINDING.md
Laptop copies: /Users/ravi/repos/ds4/logs/from-cluster/sprint597-phase01/

## Phase 2/3/4 (s597-phase234-artifacts)

Source changes (laptop repo, synced to /workspace/ds4, appliance rebuilt):
  engine/runtime_options.cuh  - Options.ep_stage_profile from
                                DS4_V100_TP_EP_EP_STAGE_PROFILE (default off)
  engine/runtime_profiler.cu  - EP sub-stage profiler: stage enum/names,
                                per-rank device stamp slots + %globaltimer
                                stamp kernel (graph mode), CUDA events (eager),
                                collect/TSV emitter (tp_ep_ep_stage_profile /
                                tp_ep_ep_stage_routes / synthetic ep_window)
  engine/decode_loop.cu       - flag-gated marks at EP sub-stage boundaries,
                                sync_all_prof barrier-site wrappers, collect
                                call sites (replay_cache_hit /
                                first_capture_replay / eager)
  tools/ds4-v100-run-tp-ep-appliance.sh - env plumbing + validation
IMPLEMENTATION NOTE: CUDA rejects cudaEventElapsedTime on events recorded
inside a captured graph (cudaErrorInvalidValue, verified prof-smoke2.log), so
graph-mode timing uses paired 1-thread %globaltimer stamp kernels into
pre-allocated device slots (the graph-compatible equivalent of paired event
records); eager mode uses real cudaEventRecord pairs. No cudaMalloc/D2H in
any captured region.

Verification runs (reference shape, waves of 32):
  p2-flagoff-full: flag OFF, new binary. 128/128. generated_tok_s 60.19,
    decode-domain 71.96 (vs old-binary Phase 0: 59.26/71.39 -> noise band).
    Node counts identical: 2697/layer graph + 115971 (x256), same as Phase 0.
    Tolerance vs phase0-full-control: naive index-paired 0.9375/0.9573 is a
    request->slot assignment artifact (slot outputs are slot-seeded by
    design); slot-indexed comparison = 1.0 selected-token, 1.0 sequence,
    1.0 bit-exact logits -> byte-identity PASS.
  p3-flagon-full: flag ON, table leg (running).

Authority leg (Phase 3a): python3 s597-nsys-stages.py nsys-insitu.sqlite
  (phase3-authority-nsys-stages.txt): per layer-step rank-summed busy:
  ep_return_copy 48.25 ms (74%), other copies 5.16, nccl 3.71,
  dense 2.78, TurboMind expert GEMM 2.39 (3.7%), all else < 1.

### Phase 2/3 verification results (final)

p2-flagoff-full (flag OFF, instrumented binary):
  128/128; wall 60.19 tok/s, decode-domain 71.96; nodes 2697/layer (+115971
  full graph), identical to Phase 0; slot-indexed tolerance vs
  phase0-full-control: 1.0/1.0/1.0 (bit-exact logits) -> byte-identity PASS.
p3-flagon-full (flag ON, table leg):
  128/128; wall 57.84 (-3.9%), decode-domain 70.63 (-1.85% vs flag-off ->
  PASSES the <=3% decode gate); nodes 2985/layer (+288 = exactly the stamp
  kernels; full graph 128355 = +12384 = 288x43); replay_ms 10.24 vs 10.11
  flag-off; coverage residual 0.4% (<=10% gate).
p3-flagon-eager (reconciliation): 128/128; decode-domain 42.33 (flag-off
  41.37, within noise). ep_window 12.15 ms/rank/layer-step reconciles with
  the chrono buckets (ep 10.41 + compose 0.92 = 11.33). compose_copy (the
  eager NCCL-broadcast EP return) = 0.68 ms/layer-step -> the NCCL control.
p3-ramp16 (sub-capacity ramp window, 16 req x 16 tok, 1 wave):
  identical stage profile (ep_window 8.68; copies unchanged) at HALF the
  load (p50 12 routes/rank, 11.2% zero-route ranks) -> envelope cost is
  load-independent, as predicted by the fixed-capacity route plan.

Analyzers: s597-phase3-analyze.py (TSV legs), s597-nsys-stages.py
(authority leg). Laptop copies under logs/from-cluster/sprint597-phase234/.
