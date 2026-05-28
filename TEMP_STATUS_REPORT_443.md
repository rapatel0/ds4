# TEMP Status Report 443: Rank-Major Serving Harness

## Scope

TP/EP only. No PP/layer-split work.

The current target is the rank-major approach from the discussion: stop using
`gather to device 0 -> compute -> redistribute` as a production shape. The first
implementation step was to make the HTTP A/B harness able to test rank-major
serving candidates explicitly.

## Code Changes

- `tools/ds4-v100-tp-ep-profile.py`
  - HTTP mode now wires `DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS`.
  - Summary extraction now includes
    `attention_projection_rank_local_input_gate`.
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`
  - Added control/candidate flags for:
    - rank-local/rank-major attention projection input;
    - rank-major FFN input;
    - rank-major router logits.
  - Added scratch/deferred-NCCL controls:
    - `--tp-runtime-scratch-mib`
    - `--defer-nccl-init`
  - Added those gates to JSON/markdown summaries.

## Intended Rank-Major A/B

Control:

- HC-current NCCL allgather
- post-attention FFN input
- fixed-capacity route plan

Candidate:

- control plus rank-local/rank-major attention-projection input
- rank-major FFN input
- rank-major router logits

Target shape:

- 8 slots
- 256K context
- chat endpoint

## Cluster Evidence

No clean rank-major A/B result yet.

Attempts:

1. `s443-rank-major-http-ab`
   - 8 requests / max 16 / scratch 1536.
   - Interrupted by unrelated root graph jobs using the same GPUs.
2. `s443-rank-major-http-ab-s512`
   - 8 requests / max 16 / scratch 512 / deferred NCCL.
   - Control reached allocation but failed before serving:
     `cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9643: out of memory`.
   - This is expert residency allocation, not rank-major runtime evidence.
3. `s443-rank-major-http-ab-s512-r4`
   - 4 requests / max 12 / scratch 512 / deferred NCCL.
   - Interrupted before readiness by an external `sudo kill` of the managed
     server process. Server logs were empty; no health/status/summary was
     produced.

Related external graph run:

- `/localpool/ds4/workspace/logs/s444-graph-ab-position-key`
- Control completed, candidate failed validation.
- Persistent graph is not ready to stack onto rank-major serving yet.

## Current Decision

The harness patch is valid and syntax-checked, but Sprint 443 does not yet have
a valid performance result. Do not promote or reject rank-major serving from the
interrupted runs.

The next clean run needs an exclusive node window, or the external root cleanup
processes need to be stopped first.

## Next Run

Use the lower-memory shape first:

```bash
cd /localpool/ds4/workspace/ds4-sprint181
ART=/localpool/ds4/workspace/logs/s443-rank-major-http-ab-s512-r4
sudo rm -rf "$ART" && sudo mkdir -p "$ART" && sudo chown -R ubuntu:ubuntu "$ART"
LD_LIBRARY_PATH=/localpool/ds4/cuda-12.2-link/lib64 timeout 1800s \
  python3 tools/ds4-v100-tp-ep-nccl-http-ab.py \
  --artifact-dir "$ART" \
  --ctx 262144 --slots 8 --requests 4 --max-requests 12 \
  --tokens 2 --position 100000 --port-base 18600 \
  --readiness-seconds 600 --request-timeout-seconds 1200 \
  --gpu-sample-interval-ms 500 --http-endpoint chat \
  --control-hc-current-nccl --tp-runtime-scratch-mib 512 --defer-nccl-init \
  --post-attention-ffn-input --post-attention-fixed-capacity-route-plan \
  --candidate-attention-projection-rank-local-input \
  --candidate-routed-ffn-rank-major-input \
  --candidate-model-router-rank-major-logits \
  --candidate-label rank-major-serving-s512-r4 \
  --vram-min-free-mib 64 --nccl-min-free-mib 64 --min-free-mib 64 \
  --max-vram-failures 0 --min-server-decode-tok-s 1 \
  --min-client-generated-tok-s 1 --min-gpu-samples 1 \
  --promotion-min-speedup 1.02
```
