#!/usr/bin/env bash
set -eu

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export DS4_V100_SERVE_MODE_LOCK=base
exec "$script_dir/ds4-v100-run-appliance.sh" "$@"
