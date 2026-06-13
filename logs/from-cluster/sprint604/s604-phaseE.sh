#!/bin/bash
# Sprint 604 Phase E: fast-stack composition + floors on the clean fix base.
set -u
cd /workspace/ds4
ART=/workspace/s604-artifacts
run() { name=$1; shift; env "$@" ./tools/s604-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
# E1: fix composes with the EDGES fast stack (edges was the known-exposed config) - census x3
for i in 1 2 3; do
  run e-edges-fix-$i DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1
done
# E2: fix composes with edges + amp (does fix hold under edges+amp?)
run e-edges-fix-amp20 DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1 \
    DS4_V100_TP_EP_DENSE_HAZARD_AMP=20 DS4_V100_TP_EP_DENSE_HAZARD_AMP_SITE=attn_out_a
# E3: floors (fix on, join default), SKIP_TOL, S=1 and S=8
SKIP_TOL=1 SLOTS=1 run e-floor-s1 DS4_V100_TP_EP_DENSE_FIX=1
SKIP_TOL=1 SLOTS=8 run e-floor-s8 DS4_V100_TP_EP_DENSE_FIX=1
# E4: floors on edges+fix (the fast base), S=1 and S=8
SKIP_TOL=1 SLOTS=1 run e-floor-edges-s1 DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1
SKIP_TOL=1 SLOTS=8 run e-floor-edges-s8 DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1
echo PHASEE_DONE
