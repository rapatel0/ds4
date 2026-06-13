#!/bin/bash
# Summarize the Phase D soak: per-arm (fix on/off) event counts + telemetry.
set -u
ART=/workspace/s604-artifacts
echo "run fix sel seq first_ck_hist first_tok_hist"
for f in $(ls -v $ART/soak-on-*.out $ART/soak-off-*.out 2>/dev/null); do
  n=$(basename "$f" .out)
  fix=0; [[ "$n" == soak-on-* ]] && fix=1
  sel=$(grep "selected_token_agreement" "$f" 2>/dev/null | grep -oE "\([0-9]+/[0-9]+\)" | head -1)
  seq=$(grep "sequence_agreement" "$f" 2>/dev/null | grep -oE "\([0-9]+/[0-9]+\)" | head -1)
  ck=$(grep "first_ck_step histogram" "$f" 2>/dev/null | sed 's/first_ck_step histogram: //')
  tok=$(grep "first_tok_step histogram" "$f" 2>/dev/null | sed 's/first_tok_step histogram: //')
  echo "$n fix=$fix sel=$sel seq=$seq ck=$ck tok=$tok"
done
echo "=== ARM TOTALS ==="
for arm in on off; do
  tokev=0; ckev=0; nruns=0
  for f in $(ls -v $ART/soak-$arm-*.out 2>/dev/null); do
    nruns=$((nruns+1))
    # token event = any non-None key in first_tok histogram
    grep -q "first_tok_step histogram:.*[0-9]:" "$f" 2>/dev/null && tokev=$((tokev+1))
    grep -q "first_ck_step histogram:.*[0-9]:" "$f" 2>/dev/null && ckev=$((ckev+1))
  done
  echo "arm=$arm runs=$nruns runs_with_token_event=$tokev runs_with_ck_event=$ckev"
done
echo "=== TELEMETRY (mean per arm) ==="
awk -F'\t' 'NR>1{a[$2"_t"]+=$3; a[$2"_s"]+=$4; a[$2"_p"]+=$6; a[$2"_e"]+=$7; n[$2]++} END{for(k in n) printf "fix=%s n=%d mean_temp=%.1f mean_sm_mhz=%.0f mean_power=%.1f ecc=%.0f\n", k, n[k], a[k"_t"]/n[k], a[k"_s"]/n[k], a[k"_p"]/n[k], a[k"_e"]/n[k]}' $ART/phaseD-telemetry.tsv 2>/dev/null
echo "=== concurrent-proc check (should be 0 every run) ==="
awk -F'\t' 'NR>1 && $8!=0 {print "WARNING foreign proc at "$1": "$8}' $ART/phaseD-telemetry.tsv 2>/dev/null; echo "(no warnings above = clean)"
