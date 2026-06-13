#!/bin/bash
# s603 Phase A gates: edges-mode smoke (crash/hang + init echo), then the
# join-default control (byte-identity: node counts + tolerance vs s597).
set -u
cd /workspace/ds4
ART=/workspace/s603-artifacts
run() { name=$1; shift; env "$@" ./tools/s603-run.sh "$name" > "$ART/$name.out" 2>&1; }
run esmoke TOKENS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_S602_SYNC=edges
echo ESMOKE_DONE rc=$?
c=$(ls -d $ART/esmoke/cases/case_0* | head -1)
grep -h "tp_ep_s602_init" $c/server.log
grep -h "tp_ep_decode_cudagraph_capture" $c/server.log | head -4
run actl603
echo ACTL603_DONE rc=$?
c=$(ls -d $ART/actl603/cases/case_0* | head -1)
grep -h "tp_ep_s602_init" $c/server.log
grep -h "tp_ep_decode_cudagraph_capture" $c/server.log | awk -F"nodes\t" "{print \$2}" | awk "{print \$1}" | sort | uniq -c
echo PHASEA_DONE
