#!/usr/bin/env python3
"""Run a semantic HTTP A/B for the true-attention output serving path.

Unlike performance-only A/Bs, this harness does not require generated-token
parity between control and candidate. Enabling post-attention FFN input changes
the layer semantics relative to the current fast serving baseline. The promotion
gate here is operational: target-shape readiness, VRAM admission, and evidence
that the true-attention/post-attention timers are active.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


def run(cmd: list[str], cwd: pathlib.Path, log_path: pathlib.Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
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


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def ratio(candidate: Any, control: Any) -> float | None:
    cand = number(candidate)
    ctrl = number(control)
    if cand is None or ctrl is None or ctrl == 0.0:
        return None
    return cand / ctrl


def profile_cmd(args: argparse.Namespace, case: str, port: int, candidate: bool) -> list[str]:
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
        "--position",
        str(args.position),
        "--tokens",
        str(args.tokens),
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
        "--model-router-routes",
        "--compact-moe-decode",
        "--lazy-output-head",
        "--vram-report",
        "--vram-min-free-mib",
        str(args.vram_min_free_mib),
        "--nccl-min-free-mib",
        str(args.nccl_min_free_mib),
        "--hc-current-stream-sync",
        "--hc-current-nccl-allgather",
    ]
    if args.prompt_file:
        cmd.extend(["--prompt-file", str(args.prompt_file)])
    if args.http_endpoint:
        cmd.extend(["--http-endpoint", args.http_endpoint])
    if candidate:
        cmd.append("--post-attention-ffn-input")
        cmd.append("--disable-route-plan-async-upload")
        if args.candidate_attention_output_nccl:
            cmd.append("--attention-output-nccl-allgather")
    return cmd


def readiness_cmd(args: argparse.Namespace, case_dir: pathlib.Path, out: pathlib.Path) -> list[str]:
    return [
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


def response_first_sequence(case_dir: pathlib.Path) -> list[int] | None:
    path = case_dir / "response-00.txt"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\nHTTP_STATUS:" in text:
            text = text.rsplit("\nHTTP_STATUS:", 1)[0]
        data = json.loads(text)
    except Exception:
        return None
    meta = data.get("ds4_v100", {}) if isinstance(data, dict) else {}
    seq = meta.get("generated_token_sequence")
    return seq if isinstance(seq, list) else None


def summarize(summary: dict[str, Any], readiness: dict[str, Any], case_dir: pathlib.Path) -> dict[str, Any]:
    fields = [
        "http_200",
        "requests",
        "tokens",
        "client_generated_tok_s",
        "server_generated_tok_s_decode",
        "server_continuation_tok_s_decode",
        "gpu_sample_count",
        "gpu_util_avg",
        "gpu_util_max",
        "vram_min_free_mib",
        "vram_failures",
        "vram_after_lazy_output_head_close_min_free_mib",
        "vram_nccl_after_lazy_output_head_close_min_free_mib",
        "output_head_first_token",
        "scaffold_sum_pre_ep_attention_output_ms",
        "scaffold_sum_pre_ep_post_attention_ffn_input_ms",
        "scaffold_sum_pre_ep_attention_projection_ms",
        "scaffold_sum_pre_ep_attention_state_ms",
        "scaffold_sum_pre_ep_compressed_kv_ms",
        "scaffold_sum_hc_current_input_ms",
        "scaffold_sum_ep_ms",
    ]
    out = {field: summary.get(field) for field in fields if field in summary}
    out["ready"] = readiness.get("ready")
    out["failure_count"] = readiness.get("failure_count")
    out["response_00_sequence"] = response_first_sequence(case_dir)
    return out


def markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    control = result["control"]
    candidate = result["candidate"]
    ratios = result["ratios"]
    lines = [
        "# DS4 V100 TP/EP True-Attention HTTP A/B",
        "",
        f"- Shape: `{result['shape']['requests']}` requests, `{result['shape']['slots']}` slots, `{result['shape']['ctx']}` ctx, `{result['shape']['tokens']}` generated tokens/request",
        f"- Control ready: `{control.get('ready')}`",
        f"- Candidate ready: `{candidate.get('ready')}`",
        f"- Candidate active: `{result['candidate_active']}`",
        f"- Decision: **{result['decision']}**",
        "",
        "| Metric | Control | True-attn candidate | Candidate/control |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("server_generated_tok_s_decode", "server generated decode tok/s"),
        ("server_continuation_tok_s_decode", "server continuation decode tok/s"),
        ("client_generated_tok_s", "client generated tok/s"),
        ("gpu_util_avg", "avg GPU util %"),
        ("vram_min_free_mib", "min free VRAM MiB"),
        ("scaffold_sum_pre_ep_attention_output_ms", "attention output ms"),
        ("scaffold_sum_pre_ep_post_attention_ffn_input_ms", "post-attn FFN input ms"),
        ("scaffold_sum_pre_ep_attention_projection_ms", "attention projection ms"),
        ("scaffold_sum_pre_ep_attention_state_ms", "attention state ms"),
        ("scaffold_sum_pre_ep_compressed_kv_ms", "compressed KV ms"),
    ]:
        lines.append(f"| {label} | `{control.get(key)}` | `{candidate.get(key)}` | `{ratios.get(key)}` |")
    lines.extend([
        "",
        "## Response 0",
        "",
        f"- Control sequence: `{control.get('response_00_sequence')}`",
        f"- Candidate sequence: `{candidate.get('response_00_sequence')}`",
        "",
        "## Artifacts",
        "",
        f"- Control: `{result['control_dir']}`",
        f"- Candidate: `{result['candidate_dir']}`",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--position", type=int, default=262080)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--port-base", type=int, default=18430)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=500)
    parser.add_argument("--tool", default="none")
    parser.add_argument("--prompt-file", type=pathlib.Path)
    parser.add_argument("--http-endpoint", choices=["chat", "selected-token"], default="chat")
    parser.add_argument("--candidate-attention-output-nccl", action="store_true")
    parser.add_argument("--vram-min-free-mib", type=int, default=64)
    parser.add_argument("--nccl-min-free-mib", type=int, default=1536)
    parser.add_argument("--min-free-mib", type=float, default=1536.0)
    parser.add_argument("--max-vram-failures", type=int, default=0)
    parser.add_argument("--min-server-decode-tok-s", type=float, default=1.0)
    parser.add_argument("--min-client-generated-tok-s", type=float, default=1.0)
    parser.add_argument("--min-gpu-util-avg", type=float, default=0.0)
    parser.add_argument("--min-gpu-samples", type=int, default=1)
    args = parser.parse_args()

    repo = pathlib.Path.cwd()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    control_proc = run(profile_cmd(args, "control", args.port_base, False), repo, args.artifact_dir / "control-profile.log")
    candidate_proc = run(profile_cmd(args, "candidate", args.port_base + 1, True), repo, args.artifact_dir / "candidate-profile.log")

    control_summary_path = find_one_summary(args.artifact_dir / "control")
    candidate_summary_path = find_one_summary(args.artifact_dir / "candidate")
    control_dir = control_summary_path.parent
    candidate_dir = candidate_summary_path.parent
    control_ready_proc = run(readiness_cmd(args, control_dir, args.artifact_dir / "control-readiness.json"), repo, args.artifact_dir / "control-readiness.log")
    candidate_ready_proc = run(readiness_cmd(args, candidate_dir, args.artifact_dir / "candidate-readiness.json"), repo, args.artifact_dir / "candidate-readiness.log")

    control_summary = load_json(control_summary_path)
    candidate_summary = load_json(candidate_summary_path)
    control_ready = load_json(args.artifact_dir / "control-readiness.json")
    candidate_ready = load_json(args.artifact_dir / "candidate-readiness.json")
    control = summarize(control_summary, control_ready, control_dir)
    candidate = summarize(candidate_summary, candidate_ready, candidate_dir)
    ratio_fields = [
        "server_generated_tok_s_decode",
        "server_continuation_tok_s_decode",
        "client_generated_tok_s",
        "gpu_util_avg",
        "vram_min_free_mib",
        "scaffold_sum_pre_ep_attention_output_ms",
        "scaffold_sum_pre_ep_post_attention_ffn_input_ms",
        "scaffold_sum_pre_ep_attention_projection_ms",
        "scaffold_sum_pre_ep_attention_state_ms",
        "scaffold_sum_pre_ep_compressed_kv_ms",
    ]
    ratios = {field: ratio(candidate.get(field), control.get(field)) for field in ratio_fields}
    candidate_active = (
        number(candidate.get("scaffold_sum_pre_ep_attention_output_ms")) not in (None, 0.0)
        and number(candidate.get("scaffold_sum_pre_ep_post_attention_ffn_input_ms")) not in (None, 0.0)
    )
    validation_ok = (
        control_proc.returncode == 0
        and candidate_proc.returncode == 0
        and control_ready_proc.returncode == 0
        and candidate_ready_proc.returncode == 0
        and bool(control_ready.get("ready"))
        and bool(candidate_ready.get("ready"))
        and candidate_active
    )
    candidate_served = (
        candidate_proc.returncode == 0
        and candidate.get("http_200") == args.requests
        and candidate_active
    )
    candidate_reserve_blocked = (
        candidate_served
        and not bool(candidate_ready.get("ready"))
        and number(candidate.get("vram_failures")) not in (None, 0.0)
    )
    if validation_ok:
        decision = "true-attention-post-attention-serving-operational"
    elif candidate_reserve_blocked:
        decision = "true-attention-post-attention-serving-served-reserve-blocked"
    else:
        decision = "true-attention-post-attention-serving-blocked"

    result = {
        "schema": "ds4_v100_tp_ep_true_attn_http_ab.v1",
        "shape": {
            "ctx": args.ctx,
            "slots": args.slots,
            "tokens": args.tokens,
            "position": args.position,
            "requests": args.requests,
        },
        "control_profile_returncode": control_proc.returncode,
        "candidate_profile_returncode": candidate_proc.returncode,
        "control_readiness_returncode": control_ready_proc.returncode,
        "candidate_readiness_returncode": candidate_ready_proc.returncode,
        "control_dir": str(control_dir),
        "candidate_dir": str(candidate_dir),
        "control": control,
        "candidate": candidate,
        "ratios": ratios,
        "candidate_active": candidate_active,
        "candidate_served": candidate_served,
        "candidate_reserve_blocked": candidate_reserve_blocked,
        "decision": decision,
    }
    (args.artifact_dir / "ab-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown(args.artifact_dir / "ab-summary.md", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if validation_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
