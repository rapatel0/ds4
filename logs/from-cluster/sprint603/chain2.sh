#!/bin/bash
# s603 bisect chain: full-barrier LL census x6 (Phase D control),
# then E0=join and E1=join edge variants x3 each.
set -u
cd /workspace/ds4
ART=/workspace/s603-artifacts
run() { name=$1; shift; env "$@" ./tools/s603-run.sh "$name" > "$ART/$name.out" 2>&1; }
for i in 1 2 3 4 5 6; do
  run fb-$i DS4_V100_TP_EP_S602_FULL_BARRIER=1
  echo "FB_${i}_DONE"
done
for i in 1 2 3; do
  run vb-$i DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_SYNC_E0=join
  echo "VB_${i}_DONE"
done
for i in 1 2 3; do
  run vc-$i DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_SYNC_E1=join
  echo "VC_${i}_DONE"
done
echo CHAIN2_DONE
