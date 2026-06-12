#!/bin/bash
# Sprint 601 run wrapper (pod-side): wait-for-idle + foreign-process
# preflight, reference-shape HTTP bench, slot-indexed tolerance +
# first-divergence localization vs a configurable control run.
#
# usage: [env] s601-run.sh <name>
#   REQUESTS     generation requests (default 128)
#   TOKENS       tokens per request (default 64)
#   SLOTS        active slots (default 32; non-32 skips tolerance)
#   ART_DIR      artifact root (default /workspace/s601-artifacts)
#   CONTROL_DIR  control case dir for tolerance
#                (default /workspace/s597-phase01-artifacts/phase0-full-control)
#   SKIP_TOL=1   skip the tolerance comparison
# All DS4_V100_* / NCCL_* envs pass through to the bench/launcher/appliance.
set -u
name="$1"; shift || true
requests="${REQUESTS:-128}"
tokens="${TOKENS:-64}"
slots="${SLOTS:-32}"
art="${ART_DIR:-/workspace/s601-artifacts}"
log="$art/$name"
ctl="${CONTROL_DIR:-/workspace/s597-phase01-artifacts/phase0-full-control}"
# wait-for-idle preflight (s598/s599/s600 discipline)
ok=0
for i in $(seq 1 180); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>0' | wc -l)
  if [ "$busy" = "0" ]; then ok=1; break; fi
  sleep 5
done
[ "$ok" = "1" ] || { echo "s601-run: GPUs not idle, aborting" >&2; exit 1; }
fp=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^$/d' | wc -l)
[ "$fp" = "0" ] || { echo "s601-run: foreign GPU processes present, aborting" >&2; nvidia-smi; exit 1; }
cd /workspace/ds4
DS4_V100_TP_EP_DECODE_GRAPH_MODE=full ./tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir "$log" \
  --tokens-cases "$tokens" --requests "$requests" --slots "$slots" \
  --concurrent-requests \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s597 \
  --contract /workspace/s597-contract/tp-ep-pack-contract.tsv \
  --turbomind-lib /workspace/ds4/build/turbomind-v100/libggml-turbomind.so \
  --tp-ep-bin ./appliance/ds4-v100-tp-ep-appliance
rc=$?
echo "=== bench rc=$rc"
case_dir=$(ls -d "$log"/cases/case_0* 2>/dev/null | head -1)
if [ -n "$case_dir" ]; then
  echo "=== summary row:"
  tail -1 "$log/sustained_http.tsv"
  if [ "${SKIP_TOL:-0}" != "1" ] && [ "$slots" = "32" ]; then
    echo "=== tolerance vs $ctl:"
    python3 /workspace/s600-artifacts/s600-first-divergence.py "$ctl" "$case_dir"
  fi
  echo "=== s600/s601 flag lines in server.log:"
  grep -h "tp_ep_s601\|tp_ep_s600\|tp_ep_nccl" "$case_dir/server.log" 2>/dev/null | head -20 || true
fi
exit $rc
