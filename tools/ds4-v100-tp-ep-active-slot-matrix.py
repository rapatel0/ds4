#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time


def parse_csv_ints(text, name):
    out = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a CSV of integers") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive")
        out.append(value)
    if not out:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")
    return out


def profile_case_dir(tool, hc_stream_sync, extra_profile_args=None):
    suffix = ""
    if hc_stream_sync:
        suffix += "-hc-stream-sync"
    for arg in extra_profile_args or []:
        if arg == "--async-output":
            suffix += "-async-output"
        if arg == "--decode-cudagraph":
            suffix += "-decode-cudagraph"
        if arg == "--batched-paged-attn":
            suffix += "-batched-paged-attn"
        if arg == "--model-router-routes":
            suffix += "-model-router"
        if arg == "--router-cublas":
            suffix += "-router-cublas"
        if arg == "--router-hash-fast":
            suffix += "-router-hash-fast"
        if arg == "--gpu-route-plan":
            suffix += "-gpu-route-plan"
        if arg == "--route-plan-async-upload":
            suffix += "-route-plan-async-upload"
        if arg == "--disable-route-plan-async-upload":
            suffix += "-no-route-plan-async-upload"
        if arg == "--compact-moe-decode":
            suffix += "-compact-moe"
        if arg == "--disable-compact-route-compose":
            suffix += "-no-compact-route"
    return f"{tool}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=pathlib.Path, default=pathlib.Path("/workspace/ds4-sprint181"))
    parser.add_argument("--profile-script", default="./tools/ds4-v100-tp-ep-profile.py")
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--requests-cases", default="1,4,8,16,32")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--position", type=int, default=262080)
    parser.add_argument("--port-base", type=int, default=18400)
    parser.add_argument("--max-requests", type=int, default=120)
    parser.add_argument("--tool", default="none")
    parser.add_argument("--http-endpoint", choices=["chat", "selected-token"], default="chat")
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=0)
    parser.add_argument("--request-timeout-seconds", type=int, default=1200)
    parser.add_argument("--readiness-seconds", type=int, default=600)
    parser.add_argument("--case-cooldown-seconds", type=int, default=0)
    parser.add_argument("--hc-current-stream-sync", action="store_true")
    parser.add_argument("--extra-profile-arg", action="append", default=[])
    args = parser.parse_args()

    request_cases = parse_csv_ints(args.requests_cases, "--requests-cases")
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, requests in enumerate(request_cases):
        case_root = args.artifact_dir / f"requests-{requests:03d}"
        cmd = [
            args.profile_script,
            "--run-mode",
            "http",
            "--http-endpoint",
            args.http_endpoint,
            "--tool",
            args.tool,
            "--artifact-dir",
            str(case_root),
            "--tokens",
            str(args.tokens),
            "--ctx",
            str(args.ctx),
            "--slots",
            str(args.slots),
            "--position",
            str(args.position),
            "--requests",
            str(requests),
            "--max-requests",
            str(max(args.max_requests, requests)),
            "--port",
            str(args.port_base + i),
            "--gpu-sample-interval-ms",
            str(args.gpu_sample_interval_ms),
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--readiness-seconds",
            str(args.readiness_seconds),
        ]
        if args.hc_current_stream_sync:
            cmd.append("--hc-current-stream-sync")
        cmd.extend(args.extra_profile_arg)
        (case_root / "matrix-command.txt").parent.mkdir(parents=True, exist_ok=True)
        (case_root / "matrix-command.txt").write_text(" ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=args.repo_dir, text=True, check=False)
        case_name = profile_case_dir(
            args.tool, args.hc_current_stream_sync, args.extra_profile_arg
        )
        summary_path = case_root / case_name / "summary.json"
        if proc.returncode != 0:
            raise SystemExit(f"profile case requests={requests} failed rc={proc.returncode}")
        if not summary_path.exists():
            matches = sorted(case_root.glob("*/summary.json"))
            if len(matches) == 1:
                summary_path = matches[0]
            else:
                raise SystemExit(f"profile case requests={requests} missing {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as src:
            summary = json.load(src)
        row = {
            "requests": requests,
            "artifact_dir": str(case_root / case_name),
            "http_200": summary.get("http_200", 0),
            "tokens": summary.get("tokens", args.tokens),
            "generated_tokens_meta": summary.get("generated_tokens_meta", 0),
            "coalesced_batch_size": summary.get("coalesced_batch_size", 0),
            "client_generated_tok_s": summary.get("client_generated_tok_s", 0.0),
            "server_generated_tok_s": summary.get("server_generated_tok_s", 0.0),
            "server_generated_tok_s_decode": summary.get("server_generated_tok_s_decode", 0.0),
            "server_continuation_tok_s_decode": summary.get("server_continuation_tok_s_decode", 0.0),
            "scaffold_projected_slot_step_tok_s": summary.get("scaffold_projected_slot_step_tok_s", 0.0),
            "compressed_kv_sum_ms": summary.get("compressed_kv_sum_ms", 0.0),
            "gpu_sample_count": summary.get("gpu_sample_count", 0),
            "gpu_util_avg": summary.get("gpu_util_avg", 0.0),
            "gpu_util_max": summary.get("gpu_util_max", 0.0),
            "gpu_mem_used_max_mib": summary.get("gpu_mem_used_max_mib", 0.0),
            "vram_min_free_mib": summary.get("vram_min_free_mib", 0.0),
            "vram_max_used_mib": summary.get("vram_max_used_mib", 0.0),
            "vram_threshold_mib": summary.get("vram_threshold_mib", 0.0),
            "vram_failures": summary.get("vram_failures", 0),
        }
        rows.append(row)
        if args.case_cooldown_seconds > 0 and i + 1 < len(request_cases):
            time.sleep(args.case_cooldown_seconds)

    json_path = args.artifact_dir / "active_slot_matrix.json"
    tsv_path = args.artifact_dir / "active_slot_matrix.tsv"
    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(
            {
                "schema": "ds4_v100_tp_ep_active_slot_matrix.v1",
                "ctx": args.ctx,
                "slots": args.slots,
                "tokens": args.tokens,
                "position": args.position,
                "http_endpoint": args.http_endpoint,
                "tool": args.tool,
                "gpu_sample_interval_ms": args.gpu_sample_interval_ms,
                "case_cooldown_seconds": args.case_cooldown_seconds,
                "cases": rows,
            },
            out,
            indent=2,
            sort_keys=True,
        )
        out.write("\n")
    with open(tsv_path, "w", encoding="utf-8", newline="") as out:
        fieldnames = [
            "requests",
            "http_200",
            "tokens",
            "generated_tokens_meta",
            "coalesced_batch_size",
            "client_generated_tok_s",
            "server_generated_tok_s",
            "server_generated_tok_s_decode",
            "server_continuation_tok_s_decode",
            "scaffold_projected_slot_step_tok_s",
            "compressed_kv_sum_ms",
            "gpu_sample_count",
            "gpu_util_avg",
            "gpu_util_max",
            "gpu_mem_used_max_mib",
            "vram_min_free_mib",
            "vram_max_used_mib",
            "vram_threshold_mib",
            "vram_failures",
            "artifact_dir",
        ]
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps({"cases": len(rows), "summary": str(json_path)}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
