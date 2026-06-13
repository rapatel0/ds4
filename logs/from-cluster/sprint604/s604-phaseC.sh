#!/bin/bash
# Sprint 604 Phase C: does DENSE_FIX=1 kill the amplified hazard?
set -u
cd /workspace/ds4
ART=/workspace/s604-artifacts
run() { name=$1; shift; env "$@" ./tools/s604-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
# C1: fix ON, amp ON @ attn_out_a 20us (the config that was ~100% token-corrupt with fix off)
run c1-fix-aoa20  DS4_V100_TP_EP_DENSE_FIX=1 DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
# C2: fix ON, amp ON @ attn_out_a 50us (harshest)
run c2-fix-aoa50  DS4_V100_TP_EP_DENSE_FIX=1 DS4_V100_TP_EP_DENSE_HAZARD_AMP=50 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
# C3: fix ON, amp OFF (byte-clean tolerance baseline check)
run c3-fix-off    DS4_V100_TP_EP_DENSE_FIX=1
echo PHASEC_DONE
