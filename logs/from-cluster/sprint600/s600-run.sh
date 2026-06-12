#!/bin/bash
# S600 run wrapper: wait-for-idle preflight + reference-shape bench + slot-indexed tolerance.
# usage: s600-run.sh <name> [REQUESTS=N] -- env assignments via environment of caller
set -u
name="$1"; shift || true
requests="${REQUESTS:-128}"
log="/workspace/s600-artifacts/$name"
ctl="${CONTROL_DIR:-/workspace/s597-phase01-artifacts/phase0-full-control}"
# wait-for-idle preflight (s598/s599 discipline)
ok=0
for i in $(seq 1 180); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>0' | wc -l)
  if [ "$busy" = "0" ]; then ok=1; break; fi
  sleep 5
done
[ "$ok" = "1" ] || { echo "s600-run: GPUs not idle, aborting" >&2; exit 1; }
cd /workspace/ds4
DS4_V100_TP_EP_DECODE_GRAPH_MODE=full ./tools/ds4-v100-tp-ep-http-bench.sh \
  --log-dir "$log" \
  --tokens-cases 64 --requests "$requests" --concurrent-requests \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s597 \
  --contract /workspace/s597-contract/tp-ep-pack-contract.tsv \
  --turbomind-lib /workspace/ds4/build/turbomind-v100/libggml-turbomind.so \
  --tp-ep-bin ./appliance/ds4-v100-tp-ep-appliance
rc=$?
echo "=== bench rc=$rc"
case_dir=$(ls -d "$log"/cases/case_0* 2>/dev/null | head -1)
if [ -n "$case_dir" ]; then
  echo "=== summary row:"; tail -1 "$log/sustained_http.tsv"
  echo "=== tolerance vs $ctl:"
  python3 /workspace/s600-artifacts/s600-first-divergence.py "$ctl" "$case_dir"
  echo "=== s600 probe lines in server.log:"
  grep -h "tp_ep_s600" "$case_dir/server.log" | head -40 || true
fi
exit $rc
