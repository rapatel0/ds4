#!/bin/bash
# s603 Phase B race gates on SYNC=edges (s602 methodology):
# Simple-stress pairwise x3, then LL census x6 + tolerance + 15 pairwise.
set -u
cd /workspace/ds4
ART=/workspace/s603-artifacts
run() { name=$1; shift; env "$@" ./tools/s603-run.sh "$name" > "$ART/$name.out" 2>&1; }
cmp2() {
  a=$(ls -d $ART/$1/cases/case_0* | head -1); b=$(ls -d $ART/$2/cases/case_0* | head -1)
  echo "=== compare $2 vs $1"; python3 /workspace/s600-artifacts/s600-first-divergence.py "$a" "$b"
}
for i in 1 2 3; do
  run e-sb-$i SKIP_TOL=1 DS4_V100_NCCL_PROTO=Simple DS4_V100_TP_EP_S602_SYNC=edges
  echo "ESB_${i}_DONE"
done
{ cmp2 e-sb-1 e-sb-2; cmp2 e-sb-1 e-sb-3; cmp2 e-sb-2 e-sb-3; } > $ART/e-sb-compare.txt 2>&1
echo ESB_COMPARES_DONE
for i in 1 2 3 4 5 6; do
  run ge-$i DS4_V100_TP_EP_S602_SYNC=edges
  echo "GE_${i}_DONE"
done
for i in 1 2 3 4 5; do
  for j in $(seq $((i+1)) 6); do cmp2 ge-$i ge-$j | grep -E "compare|agreement|histogram"; done
done > $ART/ge-pairwise.txt 2>&1
echo PHASEB_DONE
