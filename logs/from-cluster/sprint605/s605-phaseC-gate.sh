#!/bin/bash
# Sprint 605 Phase C: gate the attn_output gather8 lever on the promoted
# edges+fix base. Correctness FIRST (amplifier + control), then A/B perf.
#   cg-amp20    : gather8 ON + amp 20us @ attn_out_a  -> must stay 1.0/1.0
#                 (gather8 IS at the carrier site; any reorder reopens the hazard)
#   cg-ctl      : gather8 ON, un-amplified            -> must be 1.0/1.0 (byte-identical)
#   cg-off      : gather8 OFF (incumbent A/B arm)      -> reference perf
#   cg-on-{1,2} : gather8 ON, reference perf A/B
# Base = launcher defaults (kernel/relay/batched + edges + DENSE_FIX=1).
set -u
cd /workspace/ds4
ART=/workspace/s605-artifacts
G8="DS4_V100_TP_EP_ATTN_OUT_GATHER8=1"
run() { name=$1; shift; env "$@" ./tools/s605-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
run cg-amp20 $G8 DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
run cg-ctl   $G8
run cg-off   DS4_V100_TP_EP_ATTN_OUT_GATHER8=0
run cg-on-1  $G8
run cg-on-2  $G8
echo PHASEC_GATE_DONE
