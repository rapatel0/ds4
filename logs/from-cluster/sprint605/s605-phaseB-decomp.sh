#!/bin/bash
# Sprint 605 Phase B: clean-base (edges+fix default) per-layer decomposition.
# S=8 and S=32 FIRST (load-bearing for the MTP math + lever ranking; proven
# slot counts). S=1 LAST behind a timeout (the replay-probe gate loops
# pathologically at S=1 - see deviations; non-critical datapoint).
# Each run gets a hard timeout so one hang cannot block the rest.
set -u
cd /workspace/ds4
ART=/workspace/s605-artifacts
EDGES="DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1"
runf() { name=$1; s=$2; tmo=$3; shift 3; timeout "$tmo" env SLOTS=$s SKIP_TOL=1 $EDGES "$@" ./tools/s605-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
# Floors (profiler off) - load-bearing first
runf b-floor-s8  8  20m
runf b-floor-s32 32 20m
# Profiler-on (stage table + busy/replay split)
runf b-prof-s8  8  25m DS4_V100_TP_EP_EP_STAGE_PROFILE=1
runf b-prof-s32 32 25m DS4_V100_TP_EP_EP_STAGE_PROFILE=1
# S=1 last, short timeout + fewer requests (step floor is per-step; 16 reqs is plenty)
runf b-floor-s1 1  15m REQUESTS=16
runf b-prof-s1  1  18m REQUESTS=16 DS4_V100_TP_EP_EP_STAGE_PROFILE=1
echo PHASEB_DECOMP_DONE
