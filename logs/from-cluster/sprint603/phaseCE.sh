#!/bin/bash
# s603 Phase C stage tables + Phase E step floors.
# SYNCMODE=join|edges selects the candidate config (default edges).
set -u
cd /workspace/ds4
ART=/workspace/s603-artifacts
SYNCMODE="${SYNCMODE:-edges}"
run() { name=$1; shift; env "$@" ./tools/s603-run.sh "$name" > "$ART/$name.out" 2>&1; }
# stage tables at S=8 (profiler on; relative structure)
run d8ep SLOTS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_EP_STAGE_PROFILE=1 DS4_V100_TP_EP_S602_SYNC=edges
echo D8EP_DONE
run d8jp SLOTS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_EP_STAGE_PROFILE=1 DS4_V100_TP_EP_S602_SYNC=join
echo D8JP_DONE
# Phase E floors on the final config
run d1f603 SLOTS=1 REQUESTS=8 SKIP_TOL=1 DS4_V100_TP_EP_S602_SYNC=$SYNCMODE
echo D1F_DONE
run d8f603 SLOTS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_S602_SYNC=$SYNCMODE
echo D8F_DONE
echo PHASECE_DONE
