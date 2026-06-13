#!/bin/bash
# s603 chain3: dense-guard gates. gd = edges+guard census x6;
# gd-sb = Simple-stress pairwise x3; gj = join+guard census x3;
# then Phase C stage tables + Phase E floors.
set -u
cd /workspace/ds4
ART=/workspace/s603-artifacts
run() { name=$1; shift; env "$@" ./tools/s603-run.sh "$name" > "$ART/$name.out" 2>&1; }
cmp2() {
  a=$(ls -d $ART/$1/cases/case_0* | head -1); b=$(ls -d $ART/$2/cases/case_0* | head -1)
  echo "=== compare $2 vs $1"; python3 /workspace/s600-artifacts/s600-first-divergence.py "$a" "$b"
}
for i in 1 2 3 4 5 6; do
  run gd-$i DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_DENSE_GUARD=1
  echo "GD_${i}_DONE"
done
for i in 1 2 3; do
  run gd-sb-$i SKIP_TOL=1 DS4_V100_NCCL_PROTO=Simple DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_DENSE_GUARD=1
  echo "GDSB_${i}_DONE"
done
{ cmp2 gd-sb-1 gd-sb-2; cmp2 gd-sb-1 gd-sb-3; cmp2 gd-sb-2 gd-sb-3; } > $ART/gd-sb-compare.txt 2>&1
for i in 1 2 3; do
  run gj-$i DS4_V100_TP_EP_S602_DENSE_GUARD=1
  echo "GJ_${i}_DONE"
done
run d8ep SLOTS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_EP_STAGE_PROFILE=1 DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_DENSE_GUARD=1
echo D8EP_DONE
run d8jp SLOTS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_EP_STAGE_PROFILE=1 DS4_V100_TP_EP_S602_DENSE_GUARD=1
echo D8JP_DONE
run d1f603 SLOTS=1 REQUESTS=8 SKIP_TOL=1 DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_DENSE_GUARD=1
echo D1F_DONE
run d8f603 SLOTS=8 REQUESTS=32 SKIP_TOL=1 DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_S602_DENSE_GUARD=1
echo D8F_DONE
echo CHAIN3_DONE
