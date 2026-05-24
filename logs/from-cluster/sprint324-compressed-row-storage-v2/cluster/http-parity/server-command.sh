#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ds4-sprint181
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-tp-ep-full-layer-smoke \
  --serve-http \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --tm-index /workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv \
  --lib /workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --top-k 6 --kv-slot 7 --position 100000 \
  --warmup 0 --iters 1 --decode-steps 1 \
  --fuse-compose-sum \
  --dense-f16-cublas-compose --dense-f16-cache-compose \
  --skip-descriptor-checks --skip-predecode-probes \
  --shared-expert-bindings --shared-dense-ops \
  --overlap-ep-dense --source-copy-schedule --skip-self-compose-copy --multi-copy-streams \
  --token-major-all-layers --all-layers \
  --host 127.0.0.1 --port 18328 \
  --microbatch-wait-us 50000 \
  --tokenizer-model /models/DSv4-Flash-256e-fixed.gguf \
  --copy-event-compose \
  --tp-hc-final-expand-gate --tp-hc-current-input-gate --tp-hc-persist-state-gate \
  --diagnostic-output-head \
  --true-ds4-post-attention-ffn-input-gate \
  --true-ds4-indexer-attention-gate \
  --max-requests 2
