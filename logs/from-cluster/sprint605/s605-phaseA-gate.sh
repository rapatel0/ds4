#!/bin/bash
# Sprint 605 Phase A gate: edges+fix correctness gates BEFORE the soak.
#   - aedctl     : edges+fix, amp OFF -> must be 1.0/1.0 zero events (byte-clean base)
#   - aed-amp20-{1,2,3} : edges+fix under amp@20us @ attn_out_a -> must stay 1.0/1.0 (fix closes window)
#   - aed-precompose20  : edges+fix under amp@20us @ pre_compose -> close the late-step class (followup #3)
# Stack = launcher defaults (kernel/relay/batched) + S602_SYNC=edges + DENSE_FIX=1.
set -u
cd /workspace/ds4
ART=/workspace/s605-artifacts
EDGES="DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1"
run() { name=$1; shift; env "$@" ./tools/s605-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
run aedctl            $EDGES
run aed-amp20-1       $EDGES DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
run aed-amp20-2       $EDGES DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
run aed-amp20-3       $EDGES DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
run aed-precompose20  $EDGES DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=pre_compose
echo PHASEA_GATE_DONE
