#!/bin/bash
set -u
cd /workspace/ds4
ART=/workspace/s603-artifacts
run() { name=$1; shift; env "$@" ./tools/s603-run.sh "$name" > "$ART/$name.out" 2>&1; }
run actl603b
echo ACTL603B_DONE
run actl603c
echo ACTL603C_DONE
bash $ART/phaseB.sh >> $ART/phaseB.out 2>&1
echo CHAIN1_DONE
