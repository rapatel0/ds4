#!/usr/bin/env python3
"""Run a DS4 V100 TP/EP HTTP A/B comparison.

The harness intentionally composes the existing serving profile, readiness, and
response-parity tools. It gives candidate runs a repeatable promotion gate at
the real serving shape instead of relying on ad hoc paired shell commands.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


NCCL_DEFAULT_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
NCCL_NO_SYS_VISIBLE_DEVICES = "0,3,2,1,5,7,6,4"
NCCL_NO_SYS_RING = "0 3 2 1 5 7 6 4"


def run(cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value


def find_one_summary(root: pathlib.Path) -> pathlib.Path:
    matches = sorted(root.rglob("summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one summary.json under {root}, found {len(matches)}")
    return matches[0]


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def ratio(candidate: Any, control: Any) -> float | None:
    cand = numeric(candidate)
    ctrl = numeric(control)
    if cand is None or ctrl is None or ctrl == 0.0:
        return None
    return cand / ctrl


def default_global_lock_file() -> pathlib.Path:
    localpool = pathlib.Path("/localpool/ds4/workspace")
    if localpool.exists():
        return localpool / "ds4-tp-ep-http-ab.lock"
    return pathlib.Path("/tmp/ds4-tp-ep-http-ab.lock")


def acquire_global_lock(path: pathlib.Path, wait: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = path.open("a+", encoding="utf-8")
    flags = fcntl.LOCK_EX
    if not wait:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(lock_fh.fileno(), flags)
    except BlockingIOError as exc:
        lock_fh.seek(0)
        owner = lock_fh.read().strip()
        detail = f"; current owner: {owner}" if owner else ""
        lock_fh.close()
        raise RuntimeError(
            f"could not acquire global TP/EP A/B lock {path}{detail}; "
            "another DS4 benchmark may be running. Check DS4 GPU processes or "
            "pass --wait-global-lock to queue intentionally."
        ) from exc
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"pid={os.getpid()} argv={' '.join(sys.argv)}\n")
    lock_fh.flush()
    return lock_fh


def case_profile_cmd(
    args: argparse.Namespace,
    case: str,
    port: int,
    run_description: str,
    nccl: bool,
    decode_cudagraph: bool,
    decode_cudagraph_output_sync: bool,
    decode_cudagraph_hc_current_sync: bool,
    decode_cudagraph_stage_sync: str,
    decode_cudagraph_suffix_stage: str,
    persistent_decode_cudagraph: bool,
    decode_stage_checksum: bool,
    hc_current_allreduce: bool,
    attention_projection_rank_local_input: bool,
    routed_ffn_rank_major_input: bool,
    model_router_rank_major_logits: bool,
    model_router_allreduce_logits: bool,
    gpu_route_plan: bool,
    actual_route_sync: bool,
    post_attention_slot_major_ffn_norm: bool,
    post_attention_skip_slot_major_ffn_norm: bool,
    masked_compact_copy: bool,
    cuda_visible_devices: str,
    nccl_no_sys_ring: bool,
    extra_profile_args: list[str],
    extra_server_args: list[str],
) -> list[str]:
    cmd = [
        sys.executable,
        "tools/ds4-v100-tp-ep-profile.py",
        "--run-mode",
        "http",
        "--tool",
        args.tool,
        "--artifact-dir",
        str(args.artifact_dir / case),
        "--ctx",
        str(args.ctx),
        "--slots",
        str(args.slots),
        *(
            ["--experimental-ctx-slot-cap", str(args.experimental_ctx_slot_cap)]
            if args.experimental_ctx_slot_cap is not None
            else []
        ),
        "--position",
        str(args.position),
        "--tokens",
        str(args.tokens),
        "--cuda-visible-devices",
        cuda_visible_devices,
        "--startup-warmup",
        args.startup_warmup,
        "--requests",
        str(args.requests),
        "--max-requests",
        str(args.max_requests),
        "--port",
        str(port),
        "--readiness-seconds",
        str(args.readiness_seconds),
        "--request-timeout-seconds",
        str(args.request_timeout_seconds),
        "--gpu-sample-interval-ms",
        str(args.gpu_sample_interval_ms),
        "--gpu-sampler",
        args.gpu_sampler,
        "--dcgmi-fields",
        args.dcgmi_fields,
        "--kill-stale-server",
        "--model-router-routes",
        "--compact-moe-decode",
        "--parallel-expert-load" if args.parallel_expert_load else "--disable-parallel-expert-load",
        "--lazy-output-head",
        "--vram-report",
        "--vram-min-free-mib",
        str(args.vram_min_free_mib),
        "--nccl-min-free-mib",
        str(args.nccl_min_free_mib),
        "--tp-runtime-scratch-mib",
        str(args.tp_runtime_scratch_mib),
    ]
    if args.prompt_file:
        cmd.extend(["--prompt-file", str(args.prompt_file)])
    if run_description:
        cmd.extend(["--run-description", run_description])
    if args.http_endpoint:
        cmd.extend(["--http-endpoint", args.http_endpoint])
    if nccl_no_sys_ring:
        cmd.append("--nccl-no-sys-ring")
    else:
        cmd.append("--disable-nccl-no-sys-ring")
    if args.nccl_algo:
        cmd.extend(["--nccl-algo", args.nccl_algo])
    if args.nccl_proto:
        cmd.extend(["--nccl-proto", args.nccl_proto])
    if args.nccl_rings:
        cmd.extend(["--nccl-rings", args.nccl_rings])
    if args.nccl_p2p_level:
        cmd.extend(["--nccl-p2p-level", args.nccl_p2p_level])
    if args.nccl_debug:
        cmd.extend(["--nccl-debug", args.nccl_debug])
    if args.nccl_debug_subsys:
        cmd.extend(["--nccl-debug-subsys", args.nccl_debug_subsys])
    if args.nccl_shm_disable:
        cmd.extend(["--nccl-shm-disable", args.nccl_shm_disable])
    cmd = [part for part in cmd if part]
    if args.disable_skip_tp_runtime_comp_state:
        cmd.append("--disable-skip-tp-runtime-comp-state")
    if args.defer_nccl_init:
        cmd.append("--defer-nccl-init")
    if args.post_attention_ffn_input:
        cmd.append("--post-attention-ffn-input")
    if args.post_attention_fixed_capacity_route_plan:
        cmd.append("--post-attention-fixed-capacity-route-plan")
    if decode_cudagraph:
        cmd.append("--decode-cudagraph")
    if decode_cudagraph_output_sync:
        cmd.append("--decode-cudagraph-output-sync")
    if decode_cudagraph_hc_current_sync:
        cmd.append("--decode-cudagraph-hc-current-sync")
    if decode_cudagraph_stage_sync:
        cmd.extend(["--decode-cudagraph-stage-sync", decode_cudagraph_stage_sync])
    if decode_cudagraph_suffix_stage:
        cmd.extend(["--decode-cudagraph-suffix-stage", decode_cudagraph_suffix_stage])
    if persistent_decode_cudagraph:
        cmd.append("--persistent-decode-cudagraph")
    if decode_stage_checksum:
        cmd.append("--decode-stage-checksum")
    if attention_projection_rank_local_input:
        cmd.append("--attention-projection-rank-local-input")
    if routed_ffn_rank_major_input:
        cmd.append("--routed-ffn-rank-major-input")
    if model_router_rank_major_logits:
        cmd.append("--model-router-rank-major-logits")
    if model_router_allreduce_logits:
        cmd.append("--model-router-allreduce-logits")
    if gpu_route_plan:
        cmd.append("--gpu-route-plan")
    if actual_route_sync:
        cmd.append("--post-attention-device-actual-route-sync")
    if post_attention_slot_major_ffn_norm:
        cmd.append("--post-attention-slot-major-ffn-norm")
    if post_attention_skip_slot_major_ffn_norm:
        cmd.append("--post-attention-skip-slot-major-ffn-norm")
    if nccl:
        cmd.extend(["--hc-current-stream-sync", "--hc-current-nccl-allgather"])
    if hc_current_allreduce:
        cmd.append("--hc-current-allreduce")
    if masked_compact_copy:
        cmd.append("--post-attention-masked-compact-copy")
    for server_arg in extra_server_args:
        cmd.append(f"--server-arg={server_arg}")
    cmd.extend(extra_profile_args)
    return cmd


def readiness_cmd(args: argparse.Namespace, case_dir: pathlib.Path, out: pathlib.Path) -> list[str]:
    cmd = [
        sys.executable,
        "tools/ds4-v100-http-readiness-check.py",
        "--case-dir",
        str(case_dir),
        "--out",
        str(out),
        "--expect-requests",
        str(args.requests),
        "--expect-tokens",
        str(args.tokens),
        "--expect-slots",
        str(args.slots),
        "--expect-ctx",
        str(args.ctx),
        "--min-server-decode-tok-s",
        str(args.min_server_decode_tok_s),
        "--min-client-generated-tok-s",
        str(args.min_client_generated_tok_s),
        "--min-gpu-util-avg",
        str(args.min_gpu_util_avg),
        "--min-gpu-samples",
        str(args.min_gpu_samples),
        "--min-free-mib",
        str(args.min_free_mib),
        "--max-vram-failures",
        str(args.max_vram_failures),
        "--require-summary",
        "--require-status",
        "--require-vram",
        "--require-gpu-samples",
        "--require-resident-kv",
        "--require-typed-kv",
        "--require-compact-moe",
        "--require-token-match",
        "--require-checksum",
    ]
    return cmd


def parity_cmd(args: argparse.Namespace, control_dir: pathlib.Path, candidate_dir: pathlib.Path) -> list[str]:
    cmd = [
        sys.executable,
        "tools/ds4-v100-http-response-parity.py",
        "--control-dir",
        str(control_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--out",
        str(args.artifact_dir / "response-parity.json"),
    ]
    if args.allow_missing_checksum:
        cmd.append("--allow-missing-checksum")
    if args.ignore_text:
        cmd.append("--ignore-text")
    return cmd


def tolerance_cmd(args: argparse.Namespace, control_dir: pathlib.Path, candidate_dir: pathlib.Path) -> list[str]:
    return [
        sys.executable,
        "tools/ds4-v100-http-response-tolerance.py",
        "--control-dir",
        str(control_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--out",
        str(args.artifact_dir / "response-tolerance.json"),
        "--min-pairs",
        str(args.requests),
        "--min-top1-agreement",
        str(args.tolerance_min_top1_agreement),
        "--max-selected-logit-relative-error",
        str(args.tolerance_max_selected_logit_relative_error),
    ]


def summarize_case(summary: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "http_200",
        "requests",
        "tokens",
        "client_generated_tok_s",
        "server_generated_tok_s_decode",
        "server_continuation_tok_s_decode",
        "gpu_sample_count",
        "gpu_sample_source",
        "gpu_util_avg",
        "gpu_util_max",
        "gpu_power_w_avg",
        "gpu_power_w_max",
        "gpu_pcie_rx_avg",
        "gpu_pcie_tx_avg",
        "gpu_startup_sample_count",
        "gpu_startup_util_avg",
        "gpu_steady_sample_count",
        "gpu_steady_util_avg",
        "gpu_steady_util_max",
        "gpu_timeline_peak_sm_util_avg_ma",
        "gpu_timeline_peak_elapsed_s",
        "gpu_timeline_peak_phase",
        "gpu_timeline_request_steady_sm_util_avg",
        "lifecycle_server_ready_elapsed_s",
        "lifecycle_requests_start_elapsed_s",
        "lifecycle_responses_complete_elapsed_s",
        "vram_min_free_mib",
        "vram_failures",
        "vram_after_lazy_output_head_close_min_free_mib",
        "vram_nccl_after_lazy_output_head_close_min_free_mib",
        "output_head_first_token",
        "scaffold_tp_hc_current_input_nccl_allgather",
        "scaffold_tp_hc_current_input_stream_sync",
        "scaffold_decode_cudagraph_capture_attempted",
        "scaffold_decode_cudagraph_capture_succeeded",
        "scaffold_decode_cudagraph_replay_attempted",
        "scaffold_decode_cudagraph_replay_succeeded",
        "scaffold_decode_cudagraph_persistent_cache_hits",
        "scaffold_decode_cudagraph_persistent_cache_misses",
        "scaffold_decode_cudagraph_persistent_invalidations",
        "scaffold_decode_cudagraph_persistent_invalidate_layer",
        "scaffold_decode_cudagraph_persistent_invalidate_slots",
        "scaffold_decode_cudagraph_persistent_invalidate_position",
        "scaffold_decode_cudagraph_persistent_invalidate_root_device",
        "scaffold_decode_cudagraph_persistent_invalidate_root_stream",
        "scaffold_decode_cudagraph_replay_ms",
        "graph_audit_steps",
        "graph_audit_sync_all_calls",
        "graph_audit_event_barrier_calls",
        "graph_audit_stream_sync_count",
        "graph_audit_rank_stream_sync_count",
        "graph_audit_dense_stream_sync_count",
        "graph_audit_copy_stream_sync_count",
        "graph_audit_output_head_outside_step",
        "graph_audit_host_selected_token_dependency",
        "graph_audit_helper_host_sync_blocker_classes",
        "graph_audit_capture_attempted",
        "graph_audit_capture_succeeded",
        "graph_audit_capture_error_code",
        "graph_audit_capture_error_name",
        "graph_audit_capture_nodes",
        "graph_audit_replay_attempted",
        "graph_audit_replay_succeeded",
        "graph_audit_replay_error_code",
        "graph_audit_replay_error_name",
        "graph_audit_persistent_cache_hits",
        "graph_audit_persistent_cache_misses",
        "graph_audit_persistent_invalidations",
        "graph_audit_persistent_invalidate_layer",
        "graph_audit_persistent_invalidate_slots",
        "graph_audit_persistent_invalidate_position",
        "graph_audit_persistent_invalidate_root_device",
        "graph_audit_persistent_invalidate_root_stream",
        "graph_audit_sum_instantiate_ms",
        "graph_audit_sum_replay_ms",
        "graph_audit_capture_eligible",
        "graph_audit_blocker",
        "scaffold_attention_projection_rank_local_input_gate",
        "scaffold_attention_projection_rank_major_input_gate",
        "scaffold_routed_ffn_rank_major_input_gate",
        "scaffold_model_router_rank_major_logits_gate",
        "scaffold_model_router_allreduce_logits_gate",
        "scaffold_post_attention_fixed_capacity_route_plan_gate",
        "scaffold_post_attention_device_actual_route_sync_gate",
        "scaffold_post_attention_slot_major_ffn_norm_gate",
        "scaffold_post_attention_skip_slot_major_ffn_norm_gate",
        "scaffold_post_attention_masked_compact_copy_gate",
        "scaffold_sum_hc_current_gather_ms",
        "scaffold_sum_hc_current_input_ms",
        "nccl_no_sys_ring",
        "nccl_env",
        "nccl_log_sys_mentions",
        "nccl_log_net_mentions",
        "nccl_log_p2p_mentions",
        "nccl_log_ring_mentions",
        "nccl_log_channel_mentions",
        "nccl_topology_sys_mentions",
        "nccl_graph_sys_mentions",
        "nccl_graph_channel_count",
        "nccl_graph_edge_count",
        "nccl_graph_nv1_edge_count",
        "nccl_graph_nv2_edge_count",
        "nccl_graph_sys_edge_count",
        "nccl_graph_sys_edges",
        "serving_aggregate_generated_tok_s_decode",
        "serving_aggregate_continuation_tok_s_decode",
    ]
    out = {field: summary.get(field) for field in fields if field in summary}
    out["ready"] = readiness.get("ready")
    out["failure_count"] = readiness.get("failure_count")
    return out


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    control = result["control"]
    candidate = result["candidate"]
    speedups = result["speedups"]
    lines = [
        f"# DS4 V100 TP/EP {result['candidate_label']} HTTP A/B",
        "",
        f"- Shape: `{result['shape']['requests']}` requests, `{result['shape']['slots']}` slots, `{result['shape']['ctx']}` ctx, `{result['shape']['tokens']}` generated tokens/request",
        f"- Control CUDA visible devices: `{result.get('control_cuda_visible_devices')}`",
        f"- Candidate CUDA visible devices: `{result.get('candidate_cuda_visible_devices')}`",
        f"- Control run: {result.get('control_description') or ''}",
        f"- Candidate run: {result.get('candidate_description') or ''}",
        f"- NCCL no-SYS ring: `{result.get('nccl_no_sys_ring')}`",
        f"- NCCL ring/env: algo=`{result.get('nccl_algo') or 'default'}`, proto=`{result.get('nccl_proto') or 'default'}`, rings=`{result.get('nccl_rings') or 'default'}`, p2p=`{result.get('nccl_p2p_level') or 'default'}`",
        f"- Control ready: `{control.get('ready')}`",
        f"- Candidate ready: `{candidate.get('ready')}`",
        f"- Parity match: `{result['parity'].get('match')}` (`{result['parity'].get('matched_pairs')}/{result['parity'].get('paired_count')}` pairs)",
        f"- Tolerance pass: `{result['tolerance'].get('pass')}` (top-1 `{result['tolerance'].get('selected_token_agreement')}`, selected-logit max rel `{result['tolerance'].get('max_selected_logit_relative_error')}`)",
        f"- Decision: **{result['decision']}**",
        "",
        f"| Metric | Control | {result['candidate_label']} | Candidate/control |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("server_generated_tok_s_decode", "server generated decode tok/s"),
        ("server_continuation_tok_s_decode", "server continuation decode tok/s"),
        ("client_generated_tok_s", "client generated tok/s"),
        ("gpu_util_avg", "avg GPU util %"),
        ("gpu_util_max", "max GPU util %"),
        ("gpu_startup_util_avg", "startup avg GPU util %"),
        ("gpu_steady_util_avg", "steady/request avg GPU util %"),
        ("gpu_steady_util_max", "steady/request max GPU util %"),
        ("gpu_timeline_peak_sm_util_avg_ma", "moving-average peak SM util %"),
        ("gpu_timeline_peak_phase", "moving-average peak phase"),
        ("gpu_timeline_request_steady_sm_util_avg", "request steady SM util %"),
        ("lifecycle_server_ready_elapsed_s", "server ready elapsed s"),
        ("lifecycle_requests_start_elapsed_s", "requests start elapsed s"),
        ("lifecycle_responses_complete_elapsed_s", "responses complete elapsed s"),
        ("vram_min_free_mib", "min free VRAM MiB"),
        ("vram_nccl_after_lazy_output_head_close_min_free_mib", "post-close NCCL free MiB"),
        ("scaffold_post_attention_device_actual_route_sync_gate", "actual route-sync gate"),
        ("scaffold_post_attention_slot_major_ffn_norm_gate", "force slot-major FFN norm gate"),
        ("scaffold_post_attention_skip_slot_major_ffn_norm_gate", "skip slot-major FFN norm gate"),
        ("scaffold_post_attention_masked_compact_copy_gate", "masked compact-copy gate"),
        ("scaffold_decode_cudagraph_replay_succeeded", "graph replay succeeded"),
        ("scaffold_decode_cudagraph_replay_ms", "graph replay ms"),
        ("graph_audit_sync_all_calls", "graph sync_all calls"),
        ("graph_audit_stream_sync_count", "graph stream sync count"),
        ("graph_audit_capture_attempted", "graph audit capture attempted"),
        ("graph_audit_capture_succeeded", "graph audit capture succeeded"),
        ("graph_audit_replay_attempted", "graph audit replay attempted"),
        ("graph_audit_replay_succeeded", "graph audit replay succeeded"),
        ("graph_audit_persistent_cache_hits", "graph persistent cache hits"),
        ("graph_audit_persistent_cache_misses", "graph persistent cache misses"),
        ("graph_audit_persistent_invalidations", "graph persistent invalidations"),
        ("graph_audit_persistent_invalidate_position", "graph position invalidations"),
        ("graph_audit_sum_replay_ms", "graph audit replay ms"),
        ("graph_audit_capture_eligible", "graph capture eligible"),
        ("graph_audit_blocker", "graph blocker"),
        ("scaffold_attention_projection_rank_local_input_gate", "rank-local attn input gate"),
        ("scaffold_attention_projection_rank_major_input_gate", "rank-major attn input gate"),
        ("scaffold_routed_ffn_rank_major_input_gate", "rank-major FFN input gate"),
        ("scaffold_model_router_rank_major_logits_gate", "rank-major router logits gate"),
        ("scaffold_model_router_allreduce_logits_gate", "allreduce router logits gate"),
        ("scaffold_sum_hc_current_gather_ms", "HC-current gather ms"),
        ("scaffold_sum_hc_current_input_ms", "HC-current input ms"),
    ]:
        left = control.get(key)
        right = candidate.get(key)
        mult = speedups.get(key)
        lines.append(f"| {label} | `{left}` | `{right}` | `{mult}` |")
    lines.extend(["", "## Artifacts", ""])
    lines.append(f"- Control: `{result['control_dir']}`")
    lines.append(f"- Candidate: `{result['candidate_dir']}`")
    lines.append(f"- Parity: `{result['parity_path']}`")
    lines.append(f"- Tolerance: `{result['tolerance_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--experimental-ctx-slot-cap", type=int)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--startup-warmup", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--position", type=int, default=262080)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument(
        "--cuda-visible-devices",
        default="0,1,2,3,4,5,6,7",
        help="default physical CUDA device order for both A/B legs",
    )
    parser.add_argument(
        "--control-cuda-visible-devices",
        default="",
        help="physical CUDA device order for the control leg; defaults to --cuda-visible-devices",
    )
    parser.add_argument(
        "--candidate-cuda-visible-devices",
        default="",
        help="physical CUDA device order for the candidate leg; defaults to --cuda-visible-devices",
    )
    parser.add_argument(
        "--nccl-no-sys-ring",
        action="store_true",
        default=True,
        help="forward the profile harness V100 no-SYS NCCL ring policy to both legs; default on",
    )
    parser.add_argument(
        "--disable-nccl-no-sys-ring",
        dest="nccl_no_sys_ring",
        action="store_false",
        help="diagnostic opt-out; lets NCCL choose topology without the no-SYS ring guardrail",
    )
    parser.add_argument(
        "--control-nccl-no-sys-ring",
        action="store_true",
        help="forward the profile harness V100 no-SYS NCCL ring policy to the control leg",
    )
    parser.add_argument(
        "--candidate-nccl-no-sys-ring",
        action="store_true",
        help="forward the profile harness V100 no-SYS NCCL ring policy to the candidate leg",
    )
    parser.add_argument("--nccl-algo", default="")
    parser.add_argument("--nccl-proto", default="")
    parser.add_argument("--nccl-rings", default="")
    parser.add_argument("--nccl-p2p-level", default="")
    parser.add_argument("--nccl-debug", default="")
    parser.add_argument("--nccl-debug-subsys", default="")
    parser.add_argument("--nccl-shm-disable", default="")
    parser.add_argument("--port-base", type=int, default=18410)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=500)
    parser.add_argument("--gpu-sampler", choices=["dmon", "dcgmi", "query"], default="dmon")
    parser.add_argument(
        "--dcgmi-fields",
        default="203,252,155,150,1002,1003,1005,1009,1010,1001,1011,1012",
        help=(
            "comma-separated dcgmi dmon field IDs forwarded to the profile harness. "
            "Default is V100 zero-multiplex SM/DRAM/PCIe/NVLink plus NVML health; "
            "use 1004 in a separate pass for tensor_active."
        ),
    )
    parser.add_argument("--tool", default="none")
    parser.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--http-endpoint", choices=["chat", "selected-token"], default="chat")
    parser.add_argument("--vram-min-free-mib", type=int, default=64)
    parser.add_argument("--nccl-min-free-mib", type=int, default=1536)
    parser.add_argument("--min-free-mib", type=float, default=1536.0)
    parser.add_argument("--max-vram-failures", type=int, default=0)
    parser.add_argument("--min-server-decode-tok-s", type=float, default=1.0)
    parser.add_argument("--min-client-generated-tok-s", type=float, default=1.0)
    parser.add_argument("--min-gpu-util-avg", type=float, default=0.0)
    parser.add_argument("--min-gpu-samples", type=int, default=1)
    parser.add_argument("--promotion-min-speedup", type=float, default=1.02)
    parser.add_argument("--allow-missing-checksum", action="store_true")
    parser.add_argument("--ignore-text", action="store_true")
    parser.add_argument("--use-tolerance-gate", action="store_true")
    parser.add_argument("--tolerance-min-top1-agreement", type=float, default=0.99)
    parser.add_argument("--tolerance-max-selected-logit-relative-error", type=float, default=1e-3)
    parser.add_argument("--disable-skip-tp-runtime-comp-state", action="store_true")
    parser.add_argument(
        "--parallel-expert-load",
        dest="parallel_expert_load",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--disable-parallel-expert-load",
        dest="parallel_expert_load",
        action="store_false",
    )
    parser.add_argument("--tp-runtime-scratch-mib", type=int, default=1024)
    parser.add_argument("--defer-nccl-init", action="store_true")
    parser.add_argument("--post-attention-ffn-input", action="store_true")
    parser.add_argument("--post-attention-fixed-capacity-route-plan", action="store_true")
    parser.add_argument("--control-hc-current-nccl", action="store_true")
    parser.add_argument("--control-hc-current-allreduce", action="store_true")
    parser.add_argument("--control-decode-cudagraph", action="store_true")
    parser.add_argument("--control-decode-cudagraph-output-sync", action="store_true")
    parser.add_argument("--control-decode-cudagraph-hc-current-sync", action="store_true")
    parser.add_argument("--control-decode-cudagraph-stage-sync", default="")
    parser.add_argument("--control-decode-cudagraph-suffix-stage", default="")
    parser.add_argument("--control-persistent-decode-cudagraph", action="store_true")
    parser.add_argument("--control-decode-stage-checksum", action="store_true")
    parser.add_argument("--control-attention-projection-rank-local-input", action="store_true")
    parser.add_argument("--control-routed-ffn-rank-major-input", action="store_true")
    parser.add_argument("--control-model-router-rank-major-logits", action="store_true")
    parser.add_argument("--control-model-router-allreduce-logits", action="store_true")
    parser.add_argument("--control-gpu-route-plan", action="store_true")
    parser.add_argument("--control-post-attention-slot-major-ffn-norm", action="store_true")
    parser.add_argument("--control-post-attention-skip-slot-major-ffn-norm", action="store_true")
    parser.add_argument(
        "--no-candidate-hc-current-nccl",
        dest="candidate_hc_current_nccl",
        action="store_false",
        default=True,
    )
    parser.add_argument("--candidate-hc-current-allreduce", action="store_true")
    parser.add_argument("--candidate-decode-cudagraph", action="store_true")
    parser.add_argument("--candidate-decode-cudagraph-output-sync", action="store_true")
    parser.add_argument("--candidate-decode-cudagraph-hc-current-sync", action="store_true")
    parser.add_argument("--candidate-decode-cudagraph-stage-sync", default="")
    parser.add_argument("--candidate-decode-cudagraph-suffix-stage", default="")
    parser.add_argument("--candidate-persistent-decode-cudagraph", action="store_true")
    parser.add_argument("--candidate-decode-stage-checksum", action="store_true")
    parser.add_argument("--candidate-attention-projection-rank-local-input", action="store_true")
    parser.add_argument("--candidate-routed-ffn-rank-major-input", action="store_true")
    parser.add_argument("--candidate-model-router-rank-major-logits", action="store_true")
    parser.add_argument("--candidate-model-router-allreduce-logits", action="store_true")
    parser.add_argument("--candidate-gpu-route-plan", action="store_true")
    parser.add_argument("--candidate-post-attention-slot-major-ffn-norm", action="store_true")
    parser.add_argument("--candidate-post-attention-skip-slot-major-ffn-norm", action="store_true")
    parser.add_argument("--candidate-post-attention-masked-compact-copy", action="store_true")
    parser.add_argument("--candidate-post-attention-device-actual-route-sync", action="store_true")
    parser.add_argument("--candidate-label", default="HC-current NCCL")
    parser.add_argument("--control-description", default="control baseline")
    parser.add_argument("--candidate-description", default="")
    parser.add_argument(
        "--control-run-arg",
        dest="control_run_arg",
        action="append",
        default=[],
        help=(
            "append one raw argument token to the control profile command; "
            "use --control-run-arg=--flag for values beginning with '-'"
        ),
    )
    parser.add_argument(
        "--control-profile-arg",
        dest="control_run_arg",
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--control-server-arg",
        action="append",
        default=[],
        help=(
            "append one raw argument token to the control serving binary command; "
            "use --control-server-arg=--flag for values beginning with '-'"
        ),
    )
    parser.add_argument(
        "--candidate-run-arg",
        dest="candidate_run_arg",
        action="append",
        default=[],
        help=(
            "append one raw argument token to the candidate profile command; "
            "use --candidate-run-arg=--flag for values beginning with '-'"
        ),
    )
    parser.add_argument(
        "--candidate-profile-arg",
        dest="candidate_run_arg",
        action="append",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--candidate-server-arg",
        action="append",
        default=[],
        help=(
            "append one raw argument token to the candidate serving binary command; "
            "use --candidate-server-arg=--flag for values beginning with '-'"
        ),
    )
    parser.add_argument("--global-lock-file", type=pathlib.Path)
    parser.add_argument("--no-global-lock", action="store_true")
    parser.add_argument("--wait-global-lock", action="store_true")
    parser.add_argument(
        "--lock-check-only",
        action="store_true",
        help="acquire the global lock, print its path, and exit without launching profiles",
    )
    args = parser.parse_args()
    if not args.control_cuda_visible_devices:
        args.control_cuda_visible_devices = args.cuda_visible_devices
    if not args.candidate_cuda_visible_devices:
        args.candidate_cuda_visible_devices = args.cuda_visible_devices
    if args.nccl_no_sys_ring and (
        args.control_cuda_visible_devices != NCCL_DEFAULT_VISIBLE_DEVICES
        or args.candidate_cuda_visible_devices != NCCL_DEFAULT_VISIBLE_DEVICES
    ):
        parser.error(
            "--nccl-no-sys-ring requires natural CUDA_VISIBLE_DEVICES "
            f"{NCCL_DEFAULT_VISIBLE_DEVICES}; use --disable-nccl-no-sys-ring "
            "for visible-order diagnostics"
        )

    repo = pathlib.Path.cwd()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_fh = None
    if not args.no_global_lock:
        lock_path = args.global_lock_file or default_global_lock_file()
        try:
            lock_fh = acquire_global_lock(lock_path, args.wait_global_lock)
        except RuntimeError as exc:
            print(f"ds4-v100-tp-ep-nccl-http-ab: {exc}", file=sys.stderr)
            return 73
        if args.lock_check_only:
            print(json.dumps({
                "global_lock_acquired": True,
                "global_lock_file": str(lock_path),
                "pid": os.getpid(),
            }, sort_keys=True), flush=True)
            lock_fh.close()
            return 0
    elif args.lock_check_only:
        print(json.dumps({
            "global_lock_acquired": False,
            "global_lock_file": None,
            "pid": os.getpid(),
        }, sort_keys=True), flush=True)
        return 0

    print(f"control run: {args.control_description}", flush=True)
    control_proc = run(
        case_profile_cmd(
            args,
            "control",
            args.port_base,
            run_description=args.control_description,
            nccl=args.control_hc_current_nccl,
            decode_cudagraph=args.control_decode_cudagraph,
            decode_cudagraph_output_sync=args.control_decode_cudagraph_output_sync,
            decode_cudagraph_hc_current_sync=args.control_decode_cudagraph_hc_current_sync,
            decode_cudagraph_stage_sync=args.control_decode_cudagraph_stage_sync,
            decode_cudagraph_suffix_stage=args.control_decode_cudagraph_suffix_stage,
            persistent_decode_cudagraph=args.control_persistent_decode_cudagraph,
            decode_stage_checksum=args.control_decode_stage_checksum,
            hc_current_allreduce=args.control_hc_current_allreduce,
            attention_projection_rank_local_input=args.control_attention_projection_rank_local_input,
            routed_ffn_rank_major_input=args.control_routed_ffn_rank_major_input,
            model_router_rank_major_logits=args.control_model_router_rank_major_logits,
            model_router_allreduce_logits=args.control_model_router_allreduce_logits,
            gpu_route_plan=args.control_gpu_route_plan,
            actual_route_sync=False,
            post_attention_slot_major_ffn_norm=args.control_post_attention_slot_major_ffn_norm,
            post_attention_skip_slot_major_ffn_norm=args.control_post_attention_skip_slot_major_ffn_norm,
            masked_compact_copy=False,
            cuda_visible_devices=args.control_cuda_visible_devices,
            nccl_no_sys_ring=args.nccl_no_sys_ring or args.control_nccl_no_sys_ring,
            extra_profile_args=args.control_run_arg,
            extra_server_args=args.control_server_arg,
        ),
        repo,
        args.artifact_dir / "control-profile.log",
    )
    if control_proc.returncode != 0:
        raise RuntimeError(
            f"control profile failed rc={control_proc.returncode}; "
            f"see {args.artifact_dir / 'control-profile.log'}"
        )
    candidate_description = args.candidate_description or args.candidate_label
    print(f"candidate run: {candidate_description}", flush=True)
    candidate_proc = run(
        case_profile_cmd(
            args,
            "candidate",
            args.port_base + 1,
            run_description=candidate_description,
            nccl=args.candidate_hc_current_nccl,
            decode_cudagraph=args.candidate_decode_cudagraph,
            decode_cudagraph_output_sync=args.candidate_decode_cudagraph_output_sync,
            decode_cudagraph_hc_current_sync=args.candidate_decode_cudagraph_hc_current_sync,
            decode_cudagraph_stage_sync=args.candidate_decode_cudagraph_stage_sync,
            decode_cudagraph_suffix_stage=args.candidate_decode_cudagraph_suffix_stage,
            persistent_decode_cudagraph=args.candidate_persistent_decode_cudagraph,
            decode_stage_checksum=args.candidate_decode_stage_checksum,
            hc_current_allreduce=args.candidate_hc_current_allreduce,
            attention_projection_rank_local_input=args.candidate_attention_projection_rank_local_input,
            routed_ffn_rank_major_input=args.candidate_routed_ffn_rank_major_input,
            model_router_rank_major_logits=args.candidate_model_router_rank_major_logits,
            model_router_allreduce_logits=args.candidate_model_router_allreduce_logits,
            gpu_route_plan=args.candidate_gpu_route_plan,
            actual_route_sync=args.candidate_post_attention_device_actual_route_sync,
            post_attention_slot_major_ffn_norm=args.candidate_post_attention_slot_major_ffn_norm,
            post_attention_skip_slot_major_ffn_norm=args.candidate_post_attention_skip_slot_major_ffn_norm,
            masked_compact_copy=args.candidate_post_attention_masked_compact_copy,
            cuda_visible_devices=args.candidate_cuda_visible_devices,
            nccl_no_sys_ring=args.nccl_no_sys_ring or args.candidate_nccl_no_sys_ring,
            extra_profile_args=args.candidate_run_arg,
            extra_server_args=args.candidate_server_arg,
        ),
        repo,
        args.artifact_dir / "candidate-profile.log",
    )
    if candidate_proc.returncode != 0:
        raise RuntimeError(
            f"candidate profile failed rc={candidate_proc.returncode}; "
            f"see {args.artifact_dir / 'candidate-profile.log'}"
        )

    control_summary_path = find_one_summary(args.artifact_dir / "control")
    candidate_summary_path = find_one_summary(args.artifact_dir / "candidate")
    control_dir = control_summary_path.parent
    candidate_dir = candidate_summary_path.parent

    control_ready_proc = run(
        readiness_cmd(args, control_dir, args.artifact_dir / "control-readiness.json"),
        repo,
        args.artifact_dir / "control-readiness.log",
    )
    candidate_ready_proc = run(
        readiness_cmd(args, candidate_dir, args.artifact_dir / "candidate-readiness.json"),
        repo,
        args.artifact_dir / "candidate-readiness.log",
    )
    parity_proc = run(
        parity_cmd(args, control_dir, candidate_dir),
        repo,
        args.artifact_dir / "response-parity.log",
    )
    tolerance_proc = run(
        tolerance_cmd(args, control_dir, candidate_dir),
        repo,
        args.artifact_dir / "response-tolerance.log",
    )

    control_summary = load_json(control_summary_path)
    candidate_summary = load_json(candidate_summary_path)
    control_readiness = load_json(args.artifact_dir / "control-readiness.json")
    candidate_readiness = load_json(args.artifact_dir / "candidate-readiness.json")
    parity = load_json(args.artifact_dir / "response-parity.json")
    tolerance = load_json(args.artifact_dir / "response-tolerance.json")

    speedup_fields = [
        "server_generated_tok_s_decode",
        "server_continuation_tok_s_decode",
        "client_generated_tok_s",
        "gpu_util_avg",
        "gpu_util_max",
        "gpu_startup_util_avg",
        "gpu_steady_util_avg",
        "gpu_steady_util_max",
        "gpu_timeline_peak_sm_util_avg_ma",
        "gpu_timeline_request_steady_sm_util_avg",
        "vram_min_free_mib",
        "vram_after_lazy_output_head_close_min_free_mib",
        "vram_nccl_after_lazy_output_head_close_min_free_mib",
        "scaffold_post_attention_device_actual_route_sync_gate",
        "scaffold_post_attention_slot_major_ffn_norm_gate",
        "scaffold_post_attention_skip_slot_major_ffn_norm_gate",
        "scaffold_post_attention_masked_compact_copy_gate",
        "scaffold_attention_projection_rank_local_input_gate",
        "scaffold_routed_ffn_rank_major_input_gate",
        "scaffold_model_router_rank_major_logits_gate",
        "scaffold_model_router_allreduce_logits_gate",
        "scaffold_sum_hc_current_gather_ms",
        "scaffold_sum_hc_current_input_ms",
    ]
    speedups = {
        field: ratio(candidate_summary.get(field), control_summary.get(field))
        for field in speedup_fields
    }

    arithmetic_match = bool(parity.get("match")) or (
        args.use_tolerance_gate
        and tolerance_proc.returncode == 0
        and bool(tolerance.get("pass"))
    )
    validation_ok = (
        control_proc.returncode == 0
        and candidate_proc.returncode == 0
        and control_ready_proc.returncode == 0
        and candidate_ready_proc.returncode == 0
        and arithmetic_match
    )
    decode_speedup = speedups.get("server_generated_tok_s_decode")
    if not validation_ok:
        decision = "do-not-promote-validation-failed"
    elif decode_speedup is not None and decode_speedup >= args.promotion_min_speedup:
        decision = f"promote-{args.candidate_label.lower().replace(' ', '-').replace('/', '-')}"
    else:
        decision = "keep-diagnostic-throughput-flat-or-slower"

    result = {
        "schema": "ds4_v100_tp_ep_nccl_http_ab.v1",
        "shape": {
            "ctx": args.ctx,
            "slots": args.slots,
            "experimental_ctx_slot_cap": args.experimental_ctx_slot_cap,
        "tokens": args.tokens,
        "startup_warmup": args.startup_warmup,
            "position": args.position,
            "requests": args.requests,
        },
        "control_hc_current_nccl": args.control_hc_current_nccl,
        "control_hc_current_allreduce": args.control_hc_current_allreduce,
        "control_cuda_visible_devices": args.control_cuda_visible_devices,
        "control_nccl_no_sys_ring": args.nccl_no_sys_ring or args.control_nccl_no_sys_ring,
        "control_decode_cudagraph": args.control_decode_cudagraph,
        "control_decode_cudagraph_hc_current_sync": args.control_decode_cudagraph_hc_current_sync,
        "control_decode_cudagraph_stage_sync": args.control_decode_cudagraph_stage_sync,
        "control_decode_cudagraph_suffix_stage": args.control_decode_cudagraph_suffix_stage,
        "control_persistent_decode_cudagraph": args.control_persistent_decode_cudagraph,
        "control_attention_projection_rank_local_input": args.control_attention_projection_rank_local_input,
        "control_routed_ffn_rank_major_input": args.control_routed_ffn_rank_major_input,
        "control_model_router_rank_major_logits": args.control_model_router_rank_major_logits,
        "control_model_router_allreduce_logits": args.control_model_router_allreduce_logits,
        "control_gpu_route_plan": args.control_gpu_route_plan,
        "control_post_attention_slot_major_ffn_norm": args.control_post_attention_slot_major_ffn_norm,
        "control_post_attention_skip_slot_major_ffn_norm": args.control_post_attention_skip_slot_major_ffn_norm,
        "control_description": args.control_description,
        "control_run_arg": args.control_run_arg,
        "control_server_arg": args.control_server_arg,
        "candidate_hc_current_nccl": args.candidate_hc_current_nccl,
        "candidate_hc_current_allreduce": args.candidate_hc_current_allreduce,
        "candidate_cuda_visible_devices": args.candidate_cuda_visible_devices,
        "candidate_nccl_no_sys_ring": args.nccl_no_sys_ring or args.candidate_nccl_no_sys_ring,
        "candidate_decode_cudagraph": args.candidate_decode_cudagraph,
        "candidate_decode_cudagraph_hc_current_sync": args.candidate_decode_cudagraph_hc_current_sync,
        "candidate_decode_cudagraph_stage_sync": args.candidate_decode_cudagraph_stage_sync,
        "candidate_decode_cudagraph_suffix_stage": args.candidate_decode_cudagraph_suffix_stage,
        "candidate_persistent_decode_cudagraph": args.candidate_persistent_decode_cudagraph,
        "candidate_attention_projection_rank_local_input": args.candidate_attention_projection_rank_local_input,
        "candidate_routed_ffn_rank_major_input": args.candidate_routed_ffn_rank_major_input,
        "candidate_model_router_rank_major_logits": args.candidate_model_router_rank_major_logits,
        "candidate_model_router_allreduce_logits": args.candidate_model_router_allreduce_logits,
        "candidate_gpu_route_plan": args.candidate_gpu_route_plan,
        "candidate_post_attention_device_actual_route_sync": args.candidate_post_attention_device_actual_route_sync,
        "candidate_post_attention_slot_major_ffn_norm": args.candidate_post_attention_slot_major_ffn_norm,
        "candidate_post_attention_skip_slot_major_ffn_norm": args.candidate_post_attention_skip_slot_major_ffn_norm,
        "candidate_post_attention_masked_compact_copy": args.candidate_post_attention_masked_compact_copy,
        "candidate_description": candidate_description,
        "candidate_run_arg": args.candidate_run_arg,
        "candidate_server_arg": args.candidate_server_arg,
        "candidate_label": args.candidate_label,
        "nccl_no_sys_ring": args.nccl_no_sys_ring,
        "nccl_algo": args.nccl_algo,
        "nccl_proto": args.nccl_proto,
        "nccl_rings": args.nccl_rings,
        "nccl_p2p_level": args.nccl_p2p_level,
        "nccl_debug": args.nccl_debug,
        "nccl_debug_subsys": args.nccl_debug_subsys,
        "nccl_shm_disable": args.nccl_shm_disable,
        "tp_runtime_scratch_mib": args.tp_runtime_scratch_mib,
        "defer_nccl_init": args.defer_nccl_init,
        "control_dir": str(control_dir),
        "candidate_dir": str(candidate_dir),
        "control_profile_returncode": control_proc.returncode,
        "candidate_profile_returncode": candidate_proc.returncode,
        "control_readiness_returncode": control_ready_proc.returncode,
        "candidate_readiness_returncode": candidate_ready_proc.returncode,
        "parity_returncode": parity_proc.returncode,
        "tolerance_returncode": tolerance_proc.returncode,
        "use_tolerance_gate": args.use_tolerance_gate,
        "control": summarize_case(control_summary, control_readiness),
        "candidate": summarize_case(candidate_summary, candidate_readiness),
        "speedups": speedups,
        "parity": {
            "match": parity.get("match"),
            "paired_count": parity.get("paired_count"),
            "matched_pairs": parity.get("matched_pairs"),
            "failed_pairs": parity.get("failed_pairs"),
            "missing_in_control": parity.get("missing_in_control"),
            "missing_in_candidate": parity.get("missing_in_candidate"),
        },
        "parity_path": str(args.artifact_dir / "response-parity.json"),
        "tolerance": {
            "pass": tolerance.get("pass"),
            "paired_count": tolerance.get("paired_count"),
            "selected_token_agreement": tolerance.get("selected_token_agreement"),
            "generated_sequence_agreement": tolerance.get("generated_sequence_agreement"),
            "max_selected_logit_relative_error": tolerance.get("max_selected_logit_relative_error"),
            "selected_logit_pairs": tolerance.get("selected_logit_pairs"),
            "min_top1_agreement": tolerance.get("min_top1_agreement"),
            "max_selected_logit_relative_error_threshold": tolerance.get("max_selected_logit_relative_error_threshold"),
        },
        "tolerance_path": str(args.artifact_dir / "response-tolerance.json"),
        "promotion_min_speedup": args.promotion_min_speedup,
        "decision": decision,
    }
    (args.artifact_dir / "ab-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.artifact_dir / "ab-summary.md", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if validation_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
