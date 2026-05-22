#!/usr/bin/env bash
set -eu

: "${DS4_V100_NVPROF_LOG:=logs/v100-appliance/nvprof.log}"
: "${DS4_V100_REPLAY_UNDERLYING_BIN:=./tools/ds4-v100-replay}"

mkdir -p "$(dirname "$DS4_V100_NVPROF_LOG")"
exec nvprof --profile-from-start off \
    --log-file "$DS4_V100_NVPROF_LOG" \
    "$DS4_V100_REPLAY_UNDERLYING_BIN" "$@"
