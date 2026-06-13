# Sprint 603 command log (pod llamacpp-build-8gpu, gpu-01) - 2026-06-12

## Setup
- Tree synced from laptop HEAD bb0756ec + s603 working-tree edits (tar pipe,
  s597 convention: exclude .git/build/logs/research/*.gguf).
- New code: DS4_V100_TP_EP_S602_SYNC=join|edges (default join, byte-identical),
  per-point overrides DS4_V100_TP_EP_S602_SYNC_E0/_E1/_E2, edge table in
  docs/sprints/SPRINT-603-REPORT.md Phase A (derived from runtime_pack.cu
  kernel args BEFORE implementation per spec).
- build1: bash /workspace/s597-build.sh > /workspace/s603-artifacts/build1.log (S597_BUILD_OK)
- Runner: /workspace/ds4/tools/s603-run.sh (s602-run clone, ART_DIR=/workspace/s603-artifacts)
- Launcher defaults are the zero-NCCL stack (kernel/relay/batched) since 9b261c9b;
  runs below rely on those defaults, overriding only SYNC/NCCL_PROTO.

## Run log
- phaseA.sh: esmoke (edges bring-up, rc=0) + actl603 (join control: dd 154.84,
  capture 43x971 nodes, ONE event b2 s42 ck-all-32 + token flip slot 0).
- chain1.sh: actl603b/c (both 1.0/1.0 clean; dd 153.81/154.90) then phaseB.sh:
  e-sb-1..3 (Simple stress, edges): 3/3 pairwise BIT-IDENTICAL;
  ge-1..6 (edges census): 0.83 ck-events/run, ge-4 TOKEN events (2, 9 slots)
  -> census FAIL; ge-pairwise: 10/15 identical (all 5 divergent involve ge-4).
- chain2.sh: fb-1..6 (FULL_BARRIER=1): 6/6 BIT-EXACT, dd 71.1-72.8;
  vb-1..3 (edges+E0=join): 0.67 ck/run, 1 token run; vc-1..3 (edges+E1=join):
  0.33 ck/run, 0 token. Rate monotone in speed -> hazard outside rank sync.
- build2.log: + DS4_V100_TP_EP_S602_DENSE_GUARD (bcast-site dense-WAR guard,
  =2 all sites; default 0). S597_BUILD_OK.
- chain3.sh: gd-1..6 (edges+guard census), gd-sb-1..3 (stress), gj-1..3
  (join+guard), d8ep/d8jp (stage tables), d1f603/d8f603 (floors).
- chain3 results: gd census 4 clean / gd-2 ck-only(s21) / gd-6 TOKEN (s1, 19 slots
  + s38 ck) -> bcast dense-guard insufficient; gd-sb stress 2/3 identical (one
  ck-only batch s40); gj (join+guard): gj-1 TOKEN (s42+s52, 15 flips), gj-2 2x
  ck-only, gj-3 clean -> guard does not move the rate, fix FALSIFIED.
- d8ep/d8jp stage means (S=8 prof): edges reclaims ~0.39 ms/layer
  (route_plan_pack 0.798->0.677, prefix_hc_current 0.707->0.544, ep_window
  2.586->2.483).
- Floors (edges): d1f603 step 175.0 ms, d8f603 177.1 ms (join: 186.9/188.7 s602).
- DENSE_GUARD=2 census NOT run (mechanism falsified at =1; deferred).
