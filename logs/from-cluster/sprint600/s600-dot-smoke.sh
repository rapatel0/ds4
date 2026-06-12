#!/bin/bash
# S600 dot-print smoke: short serve with graph dot dump for one exchange mode.
# usage: s600-dot-smoke.sh <copy|batched|nccl|memcpy2d>
set -u
mode="$1"
out=/workspace/s600-artifacts/dot
mkdir -p "$out"
for i in $(seq 1 180); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>0' | wc -l)
  [ "$busy" = "0" ] && break
  sleep 5
done
cd /workspace/ds4
log=/workspace/s600-artifacts/dot/smoke-$mode.log
DS4_V100_TP_EP_DECODE_GRAPH_MODE=full \
DS4_V100_TP_EP_SWIGLU_EXCHANGE=$mode \
DS4_V100_TP_EP_GRAPH_DOT_DIR=$out \
DS4_V100_TP_EP_GRAPH_DOT_LAYER=2 \
DS4_V100_TOKENS=2 DS4_V100_MAX_REQUESTS=2 \
DS4_V100_LOG_DIR=/workspace/s600-artifacts/dot/launcher-$mode \
./tools/ds4-v100-run-tp-ep-appliance.sh --env /workspace/s597-phase01-artifacts/s597.env > "$log" 2>&1 &
pid=$!
for i in $(seq 1 900); do
  grep -q tp_ep_http_serving "$log" && break
  kill -0 $pid 2>/dev/null || { echo "server died"; tail -5 "$log"; exit 1; }
  sleep 1
done
curl -sf -m 600 -H 'Content-Type: application/json' -d '{"prompt":"hello","tokens":2}' http://127.0.0.1:18080/v100/selected-token > /dev/null
sleep 2
grep "tp_ep_s600_graph_dot" "$log" || echo "no dot line"
kill $pid 2>/dev/null
for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill -9 $pid 2>/dev/null
ls -la "$out"/*.dot 2>/dev/null
