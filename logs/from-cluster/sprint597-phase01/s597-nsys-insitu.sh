#!/bin/bash
# Sprint 597 Phase 1.3: in-situ nsys capture of one steady-state serving window
# on the unmodified promoted full-capture default.
set -x
cd /workspace/ds4
ART=/workspace/s597-phase01-artifacts
SESSION=s597insitu
export DS4_V100_TP_EP_DECODE_GRAPH_MODE=full
export DS4_V100_TOKENS=8
export DS4_V100_MAX_REQUESTS=80
export DS4_V100_PORT=18200
export DS4_V100_LOG_DIR=$ART/nsys-launcher-logs

nsys launch --session-new=$SESSION -t cuda --cuda-graph-trace=node \
  ./tools/ds4-v100-run-tp-ep-appliance.sh --env $ART/s597.env \
  > $ART/nsys-server.log 2>&1 &
LAUNCH_PID=$!

for i in $(seq 1 900); do
  grep -q tp_ep_http_serving $ART/nsys-server.log && break
  sleep 1
done
grep -q tp_ep_http_serving $ART/nsys-server.log || { echo NSYS_SERVER_FAILED; exit 1; }

# Warm batch (graph capture + first replay) outside the profile window.
python3 $ART/s597-nsys-drive.py 18200 32 8 || exit 1

# Profile window: one steady-state replay batch.
nsys start --session=$SESSION -o $ART/nsys-insitu --force-overwrite=true || exit 1
python3 $ART/s597-nsys-drive.py 18200 32 8 || exit 1
nsys stop --session=$SESSION || exit 1

pkill -f "ds4-v100-tp-ep-appliance --serve-http"
sleep 5
pkill -9 -f "ds4-v100-tp-ep-appliance --serve-http" 2>/dev/null
nsys export --type sqlite --force-overwrite=true -o $ART/nsys-insitu.sqlite $ART/nsys-insitu.nsys-rep || exit 1
echo NSYS_INSITU_OK
