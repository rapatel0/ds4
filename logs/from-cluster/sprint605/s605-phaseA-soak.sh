#!/bin/bash
# Sprint 605 Phase A soak: edges+fix, un-amplified reference shape, >=30 runs,
# per-run pod telemetry. Zero token events required to promote.
# Stack = launcher defaults (kernel/relay/batched) + S602_SYNC=edges + DENSE_FIX=1.
set -u
cd /workspace/ds4
ART=/workspace/s605-artifacts
TEL=$ART/phaseA-soak-telemetry.tsv
echo -e "run\ttemp_c\tsm_mhz\tmem_mhz\tpower_w\tecc_unc\tn_proc_rows" > "$TEL"
tel() {
  local run=$1 line np
  line=$(nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits | awk -F", " "{t+=\$1; s+=\$2; m+=\$3; p+=\$4; e+=\$5; n++} END{printf \"%.0f\t%.0f\t%.0f\t%.1f\t%.0f\", t/n, s/n, m/n, p/n, e}")
  np=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed "/^\$/d" | wc -l)
  echo -e "${run}\t${line}\t${np}" >> "$TEL"
}
N=${SOAK_RUNS:-32}
for i in $(seq 1 $N); do
  tel "soak-$i"
  env DS4_V100_TP_EP_S602_SYNC=edges DS4_V100_TP_EP_DENSE_FIX=1 ./tools/s605-run.sh "soak-$i" > "$ART/soak-$i.out" 2>&1
  echo "soak-${i}_DONE rc=$?"
done
echo PHASEA_SOAK_DONE
