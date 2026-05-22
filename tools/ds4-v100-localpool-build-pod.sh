#!/usr/bin/env bash
set -eu

namespace="${DS4_V100_NAMESPACE:-llm}"
pod="${DS4_V100_BUILD_POD:-llamacpp-build-8gpu}"
node_ssh="${DS4_V100_NODE_SSH:-ubuntu@192.168.102.5}"
localpool_root="${DS4_V100_LOCALPOOL_ROOT:-/localpool/ds4}"
manifest="${DS4_V100_BUILD_POD_MANIFEST:-deploy/v100/ds4-v100-build-localpool.pod.yaml}"
timeout="${DS4_V100_BUILD_POD_TIMEOUT:-300s}"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-localpool-build-pod.sh ensure|status|delete

Commands:
  ensure  create /localpool/ds4 on gpu-01, replace the 8-GPU build pod, and wait until ready
  status  print pod state and mounted filesystem sizes
  delete  delete the build pod

Environment:
  DS4_V100_NAMESPACE                 default: llm
  DS4_V100_BUILD_POD                 default: llamacpp-build-8gpu
  DS4_V100_NODE_SSH                  default: ubuntu@192.168.102.5
  DS4_V100_LOCALPOOL_ROOT            default: /localpool/ds4
  DS4_V100_BUILD_POD_MANIFEST        default: deploy/v100/ds4-v100-build-localpool.pod.yaml
  DS4_V100_BUILD_POD_TIMEOUT         default: 300s
USAGE
}

ensure_localpool_dirs() {
    ssh "$node_ssh" "sudo mkdir -p '$localpool_root/workspace' '$localpool_root/packs' '$localpool_root/logs' && sudo chmod 0777 '$localpool_root' '$localpool_root/workspace' '$localpool_root/packs' '$localpool_root/logs'"
}

case "${1:-}" in
    ensure)
        [ -f "$manifest" ] || {
            echo "ds4-v100-localpool-build-pod: missing manifest $manifest" >&2
            exit 1
        }
        ensure_localpool_dirs
        kubectl -n "$namespace" delete pod "$pod" --ignore-not-found=true --wait=true
        kubectl apply -f "$manifest"
        kubectl -n "$namespace" wait --for=condition=Ready "pod/$pod" --timeout="$timeout"
        kubectl -n "$namespace" exec "$pod" -- bash -lc 'df -h /workspace /models; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader'
        ;;
    status)
        kubectl -n "$namespace" get pod "$pod" -o wide
        kubectl -n "$namespace" exec "$pod" -- bash -lc 'df -h /workspace /models; ls -la /workspace | head; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader' || true
        ;;
    delete)
        kubectl -n "$namespace" delete pod "$pod" --ignore-not-found=true --wait=true
        ;;
    --help|-h|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
