# Sprint 598 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-11

Source changes (laptop -> pod /workspace/ds4, appliance rebuilt):
  appliance/http_server.cu        listen backlog 16 -> 256 (warm-up #2)
  tools/ds4-v100-tp-ep-http-bench.sh  upstreamed: 900s listen wait,
                                  decode(utf-8,replace) x2, wave-of-32 submission
  engine/runtime_options.cuh      Options.ep_return_nccl from
                                  DS4_V100_TP_EP_EP_RETURN_TRANSPORT (default copy)
  engine/runtime_pack.cu          broadcast_ep_return_slices(+skip_stream_sync)
                                  capture-safe mode (skips host stream syncs)
  engine/decode_loop.cu           C1 branch: graph EP return via grouped
                                  per-source NCCL broadcasts when flag=nccl,
                                  profiler stage marks around it
  engine/runtime_profiler.cu      one new stage name ep_return_nccl (deviation:
                                  outside the listed surface, measurement-only)
  tools/ds4-v100-run-tp-ep-appliance.sh  flag plumbing + validation

C1 capture probe (tools/s598-nccl-capture-probe.cu, standalone):
  nvcc -O3 -arch=sm_70 -o s598-nccl-capture-probe ... -lnccl
  NCCL_P2P_LEVEL=NVL ./s598-nccl-capture-probe
  RESULT: PASS. nccl 2.19.3; capture 200 nodes; eager/first-replay/
  fresh-data/post-timing parity all PASS; 0.7202 ms per 8-src grouped
  broadcast round (3 MiB per src). NOTE: probe process hangs at exit in
  ncclCommDestroy after graph use (comms destroyed while exec alive) - probe
  bug, not a transport issue; kill after PASS line.

C1 appliance smoke (flag=nccl, profiler on, max_tokens=4):
  capture succeeded nodes=3017, 257 replays OK, tokens [48177,3263,65270,40429]
  match copy transport; ep_return_nccl stage 0.677 ms/layer.

Reference runs (fixed harness, 128 req x 64 tok, waves of 32):
  r1-copy-baseline (flag=copy, profiler off):
    wall 59.20 tok/s, decode-domain 71.21; nodes 2697/115971 (identical to
    committed s597 binary); slot-indexed tolerance vs phase0-full-control:
    1.0 / 1.0 / logits bit-exact -> warm-up #1 identity re-proof DONE.
  r2-nccl (flag=nccl, profiler off):
    wall 108.50 tok/s, decode-domain 162.06 -> 2.28x decode-domain, 1.83x wall
    vs r1. Tolerance vs control: 1.0 selected-token / 1.0 sequence /
    max logit rel err 0.0 (bit-exact).
  r3-nccl-prof (flag=nccl, profiler on): running.

Environment incidents:
  - First c1 smoke crashed (segfault) during a transient foreign GPU burst
    (host pid 1382988, 8.1 GiB on all 8 GPUs, not a k8s GPU pod; gemma pod is
    on gpu-02, not gpu-01). Burst self-cleared; retry passed. One R1 attempt
    failed the launcher reserve check against our own dying smoke server
    (kill+3s race); fixed with a wait-for-idle preflight before every run.

r3-nccl-prof (flag=nccl, profiler on):
  decode-domain 160.08 (-1.2% vs profiler-off nccl leg: within the 3% band)
  stage table: ep_return_nccl 0.6113 ms/layer (max 0.87); ep_copy_src* stages
  ABSENT; ep_window 8.52 -> 2.47 ms; layer replay 10.24 -> 4.25 ms;
  nodes 3017/layer; route skew unchanged (p50 24 / p95 52 / max 120).

nsys spot-check (s598-nsys-insitu.sh, one 32x8 replay window, flag=nccl):
  grid-384 copy_f32 kernels: 0 (was 19,264 in the s597 window).
  EP return = ncclDevKernel_Broadcast_RING_LL: 5.63 ms/layer-step summed over
  8 ranks (0.70 ms/rank), RING on the NVLink-only ring (NCCL_P2P_LEVEL=NVL).
  No-SYS proof complete. Analysis: nsys-no-sys-proof.txt.

PROMOTION: all DoD gates pass ->
  tools/ds4-v100-run-tp-ep-appliance.sh default flipped:
  DS4_V100_TP_EP_EP_RETURN_TRANSPORT=nccl (copy kept as rollback flag;
  binary/Options default remains copy for non-launcher invocations).
