#!/bin/bash
# Sprint 604 Phase A: amplifier byte-identity + tuning sweep.
# Default stack = zero-NCCL kernel/relay/batched/join (launcher defaults).
set -u
cd /workspace/ds4
ART=/workspace/s604-artifacts
run() { name=$1; shift; env "$@" ./tools/s604-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
# A0: amp OFF byte-identity (default stack, must be 1.0/1.0 zero events)
run a0ctl
# A1: amplifier at attn_out_a (codex candidate 1, early/step-1 token class), sweep us
run a1-aoa5   DS4_V100_TP_EP_DENSE_HAZARD_AMP=5   DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
run a1-aoa20  DS4_V100_TP_EP_DENSE_HAZARD_AMP=20  DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
run a1-aoa50  DS4_V100_TP_EP_DENSE_HAZARD_AMP=50  DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
echo PHASEA_SWEEP_DONE
