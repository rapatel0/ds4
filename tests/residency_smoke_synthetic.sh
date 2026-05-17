#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/ds4-residency-smoke.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

MODEL="$TMP/model.bin"
INDEX="$TMP/pack-index.tsv"
SHARD="$TMP/gpu0.weights"

printf '0123456789abcdefABCDEFGHIJKLMNOPqrstuvwxyzUVWXYZ--++!!??' > "$MODEL"
dd if="$MODEL" of="$SHARD" bs=1 count=48 2>/dev/null

cat > "$INDEX" <<'TSV'
semantic_tensor_id	source_name	source_dtype	source_shape	runtime_layout	owning_gpu	layer_id	kernel_family	source_offset	byte_length	shard_file	shard_offset	scale_offset	checksum
tok	tok	bf16	[4x4]	source_bf16	0	-1	embed	0	16	gpu0.weights	0	-1	pending
ctl	ctl	f32	[4]	source_f32_control	0	0	control	16	16	gpu0.weights	16	-1	pending
lyr	lyr	f8_e4m3_b128	[16x8]	source_f8	0	1	fp8	32	16	gpu0.weights	32	-1	pending
TSV

"$ROOT/tools/ds4-v100-residency-smoke" \
  --model "$MODEL" \
  --index "$INDEX" \
  --provider gguf \
  --report "$TMP/gguf.log"

"$ROOT/tools/ds4-v100-residency-smoke" \
  --model "$MODEL" \
  --index "$INDEX" \
  --provider shard \
  --shard-dir "$TMP" \
  --crosscheck \
  --report "$TMP/shard.log"

grep -q 'result	OK' "$TMP/gguf.log"
grep -q 'result	OK' "$TMP/shard.log"
grep -q 'crosscheck	' "$TMP/shard.log"

echo "residency_smoke_synthetic: ok"
