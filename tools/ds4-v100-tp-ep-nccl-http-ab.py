#!/usr/bin/env python3
"""Run a DS4 V100 TP/EP HTTP A/B for HC-current NCCL allgather.

The harness intentionally composes the existing serving profile, readiness, and
response-parity tools. It gives the NCCL path a repeatable promotion gate at
the real serving shape instead of relying on ad hoc paired shell commands.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any


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


def case_profile_cmd(args: argparse.Namespace, case: str, port: int, nccl: bool) -> list[str]:
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
    ]
    if args.prompt_file:
        cmd.extend(["--prompt-file", str(args.prompt_file)])
    if args.http_endpoint:
        cmd.extend(["--http-endpoint", args.http_endpoint])
    if args.disable_skip_tp_runtime_comp_state:
        cmd.append("--disable-skip-tp-runtime-comp-state")
    if nccl:
        cmd.extend(["--hc-current-stream-sync", "--hc-current-nccl-allgather"])
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


def summarize_case(summary: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
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
        "scaffold_tp_hc_current_input_nccl_allgather",
        "scaffold_tp_hc_current_input_stream_sync",
        "scaffold_sum_hc_current_gather_ms",
        "scaffold_sum_hc_current_input_ms",
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
        "# DS4 V100 TP/EP HC-Current NCCL HTTP A/B",
        "",
        f"- Shape: `{result['shape']['requests']}` requests, `{result['shape']['slots']}` slots, `{result['shape']['ctx']}` ctx, `{result['shape']['tokens']}` generated tokens/request",
        f"- Control ready: `{control.get('ready')}`",
        f"- Candidate ready: `{candidate.get('ready')}`",
        f"- Parity match: `{result['parity'].get('match')}` (`{result['parity'].get('matched_pairs')}/{result['parity'].get('paired_count')}` pairs)",
        f"- Decision: **{result['decision']}**",
        "",
        "| Metric | Control | HC-current NCCL | Candidate/control |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [
        ("server_generated_tok_s_decode", "server generated decode tok/s"),
        ("server_continuation_tok_s_decode", "server continuation decode tok/s"),
        ("client_generated_tok_s", "client generated tok/s"),
        ("gpu_util_avg", "avg GPU util %"),
        ("gpu_util_max", "max GPU util %"),
        ("vram_min_free_mib", "min free VRAM MiB"),
        ("vram_nccl_after_lazy_output_head_close_min_free_mib", "post-close NCCL free MiB"),
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
    parser.add_argument("--port-base", type=int, default=18410)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=500)
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
    parser.add_argument("--disable-skip-tp-runtime-comp-state", action="store_true")
    args = parser.parse_args()

    repo = pathlib.Path.cwd()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    control_proc = run(
        case_profile_cmd(args, "control", args.port_base, nccl=False),
        repo,
        args.artifact_dir / "control-profile.log",
    )
    candidate_proc = run(
        case_profile_cmd(args, "candidate", args.port_base + 1, nccl=True),
        repo,
        args.artifact_dir / "candidate-profile.log",
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

    control_summary = load_json(control_summary_path)
    candidate_summary = load_json(candidate_summary_path)
    control_readiness = load_json(args.artifact_dir / "control-readiness.json")
    candidate_readiness = load_json(args.artifact_dir / "candidate-readiness.json")
    parity = load_json(args.artifact_dir / "response-parity.json")

    speedup_fields = [
        "server_generated_tok_s_decode",
        "server_continuation_tok_s_decode",
        "client_generated_tok_s",
        "gpu_util_avg",
        "gpu_util_max",
        "vram_min_free_mib",
        "vram_after_lazy_output_head_close_min_free_mib",
        "vram_nccl_after_lazy_output_head_close_min_free_mib",
        "scaffold_sum_hc_current_gather_ms",
        "scaffold_sum_hc_current_input_ms",
    ]
    speedups = {
        field: ratio(candidate_summary.get(field), control_summary.get(field))
        for field in speedup_fields
    }

    validation_ok = (
        control_proc.returncode == 0
        and candidate_proc.returncode == 0
        and control_ready_proc.returncode == 0
        and candidate_ready_proc.returncode == 0
        and parity_proc.returncode == 0
        and bool(parity.get("match"))
    )
    decode_speedup = speedups.get("server_generated_tok_s_decode")
    if not validation_ok:
        decision = "do-not-promote-validation-failed"
    elif decode_speedup is not None and decode_speedup >= args.promotion_min_speedup:
        decision = "promote-hc-current-nccl"
    else:
        decision = "keep-diagnostic-throughput-flat-or-slower"

    result = {
        "schema": "ds4_v100_tp_ep_nccl_http_ab.v1",
        "shape": {
            "ctx": args.ctx,
            "slots": args.slots,
            "tokens": args.tokens,
            "position": args.position,
            "requests": args.requests,
        },
        "control_dir": str(control_dir),
        "candidate_dir": str(candidate_dir),
        "control_profile_returncode": control_proc.returncode,
        "candidate_profile_returncode": candidate_proc.returncode,
        "control_readiness_returncode": control_ready_proc.returncode,
        "candidate_readiness_returncode": candidate_ready_proc.returncode,
        "parity_returncode": parity_proc.returncode,
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
