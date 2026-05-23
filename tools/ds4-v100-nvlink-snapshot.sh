#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 OUT_DIR -- command [args...]" >&2
    exit 2
fi

out_dir="$1"
shift
if [[ "${1:-}" != "--" ]]; then
    echo "usage: $0 OUT_DIR -- command [args...]" >&2
    exit 2
fi
shift
if [[ $# -eq 0 ]]; then
    echo "missing command" >&2
    exit 2
fi

mkdir -p "$out_dir"

snapshot() {
    local label="$1"
    {
        date -Is
        echo "nvidia-smi nvlink --status"
        nvidia-smi nvlink --status || true
        echo
        echo "nvidia-smi nvlink -gt d"
        nvidia-smi nvlink -gt d || true
        echo
        echo "nvidia-smi topo -m"
        nvidia-smi topo -m || true
    } > "$out_dir/nvlink-${label}.txt" 2>&1
}

snapshot before
"$@" | tee "$out_dir/command.log"
status=${PIPESTATUS[0]}
snapshot after
exit "$status"
