#!/bin/bash
# Sprint 604 Phase D: >=50-run soak, ALTERNATING fix-on/fix-off, un-amplified
# reference shape, with per-run pod telemetry. Default stack (kernel/relay/
# batched/join). Telemetry captured right before each run.
set -u
cd /workspace/ds4
ART=/workspace/s604-artifacts
TEL=$ART/phaseD-telemetry.tsv
echo -e "run\tfix\ttemp_c\tsm_mhz\tmem_mhz\tpower_w\tecc_unc\tn_concurrent_proc" > "$TEL"
tel() {
  local run=$1 fix=$2
  local line
  line=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits | awk -F", " '{t+=$1; s+=$2; m+=$3; p+=$4; e+=$5; n++} END{printf "%.0f\t%.0f\t%.0f\t%.1f\t%.0f", t/n, s/n, m/n, p/n, e}')
  local np
  np=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)
  echo -e "${run}\t${fix}\t${line}\t${np}" >> "$TEL"
}
run() { name=$1; fix=$2; shift 2; tel "$name" "$fix"; env "$@" ./tools/s604-run.sh "$name" > "$ART/$name.out" 2>&1; echo "${name}_DONE rc=$?"; }
N=${SOAK_PAIRS:-26}
for i in $(seq 1 $N); do
  run soak-on-$i  1 DS4_V100_TP_EP_DENSE_FIX=1
  run soak-off-$i 0
  echo "SOAK_PAIR_${i}_DONE"
done
echo PHASED_DONE
