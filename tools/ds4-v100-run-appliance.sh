#!/usr/bin/env bash
set -eu

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-run-appliance.sh [options]

Compatibility dispatcher. Use one of the explicit launchers for new scripts:
  tools/ds4-v100-run-pp-appliance.sh
  tools/ds4-v100-run-tp-ep-appliance.sh

DS4_V100_SERVE_MODE=tp-ep dispatches to the TP/EP launcher; all other modes
dispatch to the PP/base launcher.
USAGE
}

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

case "${1:-}" in
    --help|-h)
        usage
        exit 0
        ;;
esac

env_file=""
args=("$@")
while [ "$#" -gt 0 ]; do
    case "$1" in
        --env)
            [ "$#" -ge 2 ] || {
                echo "ds4-v100-run-appliance: --env requires a value" >&2
                exit 1
            }
            env_file="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -n "$env_file" ]; then
    [ -f "$env_file" ] || {
        echo "ds4-v100-run-appliance: missing env file $env_file" >&2
        exit 1
    }
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
fi

case "${DS4_V100_SERVE_MODE:-base}" in
    tp-ep)
        exec "$script_dir/ds4-v100-run-tp-ep-appliance.sh" "${args[@]}"
        ;;
    base|"")
        exec "$script_dir/ds4-v100-run-pp-appliance.sh" "${args[@]}"
        ;;
    *)
        echo "ds4-v100-run-appliance: DS4_V100_SERVE_MODE must be base or tp-ep" >&2
        exit 1
        ;;
esac
