#!/bin/bash
# Sprint 604 Phase B: do OTHER sites amplify? (separate carriers / event classes)
set -u
cd /workspace/ds4
ART=/workspace/s604-artifacts
run() { name=$1; shift; env "$@" ./tools/s604-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
# Late-class probe: amplify pre_compose (EP/dense overlap) - does the late-step class widen here?
run b-precompose20 DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=pre_compose
# pre_dense probe (attn/shared GEMM hand-off vs swiglu)
run b-predense20   DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=pre_dense
# attn_out_b probe (second attention GEMM)
run b-aob20        DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_b
# FIX vs the strongest other-site amplifier (whichever fires): fix + pre_compose
run b-fix-precompose20 DS4_V100_TP_EP_DENSE_FIX=1 DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=pre_compose
echo PHASEB_DONE
