#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys
import time


CASES = {
    "control": {
        "label": "Control",
        "flags": [],
    },
    "fp8-kv": {
        "label": "FP8 E5M2 KV",
        "flags": ["--fp8-e5m2-kv"],
    },
    "hc-nccl": {
        "label": "HC-current NCCL",
        "flags": ["--hc-current-nccl-allgather"],
    },
    "fp8-kv-hc-nccl": {
        "label": "FP8 E5M2 KV + HC-current NCCL",
        "flags": ["--fp8-e5m2-kv", "--hc-current-nccl-allgather"],
    },
}


def parse_json_line(text):
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def find_summary(case_dir):
    summaries = sorted(pathlib.Path(case_dir).glob("*/*/summary.json"))
    if not summaries:
        summaries = sorted(pathlib.Path(case_dir).glob("*/summary.json"))
    if not summaries:
        return {}
    with summaries[-1].open("r", encoding="utf-8") as src:
        return json.load(src)


def case_command(args, case_name):
    case_dir = args.artifact_dir / case_name
    cmd = [
        sys.executable,
        args.profile_script,
        "--run-mode",
        "direct-token-major",
        "--tool",
        args.tool,
        "--artifact-dir",
        str(case_dir),
        "--tokens",
        str(args.tokens),
        "--ctx",
        str(args.ctx),
        "--slots",
        str(args.slots),
        "--position",
        str(args.position),
        "--requests",
        str(args.requests),
        "--max-requests",
        str(args.max_requests),
        "--model-router-routes",
        "--compact-moe-decode",
        "--hc-current-stream-sync",
        "--nccl-min-free-mib",
        str(args.nccl_min_free_mib),
        "--vram-report",
        "--vram-min-free-mib",
        str(args.vram_min_free_mib),
    ]
    cmd.extend(CASES[case_name]["flags"])
    return cmd, case_dir


def compact_result(case_name, proc, summary, elapsed_s):
    return {
        "case": case_name,
        "label": CASES[case_name]["label"],
        "process_returncode": proc.returncode,
        "summary_returncode": summary.get("returncode"),
        "elapsed_s": elapsed_s,
        "first_token": summary.get("output_head_first_token"),
        "generated_decode_tok_s": summary.get("serving_aggregate_generated_tok_s_decode"),
        "continuation_decode_tok_s": summary.get("serving_aggregate_continuation_tok_s_decode"),
        "generated_wall_tok_s": summary.get("serving_aggregate_generated_tok_s_wall"),
        "generated_tokens": summary.get("serving_generated_tokens"),
        "continuation_tokens": summary.get("serving_continuation_tokens"),
        "vram_min_free_mib": summary.get("vram_min_free_mib"),
        "vram_max_used_mib": summary.get("vram_max_used_mib"),
        "vram_failures": summary.get("vram_failures"),
        "nccl_after_output_head_min_free_mib": summary.get(
            "vram_nccl_after_output_head_min_free_mib"
        ),
        "nccl_after_output_head_threshold_mib": summary.get(
            "vram_nccl_after_output_head_threshold_mib"
        ),
        "nccl_after_output_head_failures": summary.get(
            "vram_nccl_after_output_head_failures"
        ),
        "hc_current_gather_ms": summary.get("scaffold_sum_hc_current_gather_ms"),
        "hc_current_input_ms": summary.get("scaffold_sum_hc_current_input_ms"),
        "tp_hc_current_input_nccl_allgather": summary.get(
            "scaffold_tp_hc_current_input_nccl_allgather"
        ),
    }


def write_markdown(path, results):
    headers = [
        "Case",
        "Return",
        "First token",
        "Generated decode tok/s",
        "Continuation decode tok/s",
        "Min free VRAM",
        "NCCL threshold",
        "NCCL failures",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["label"],
                str(result.get("summary_returncode", result["process_returncode"])),
                str(result.get("first_token") or "n/a"),
                format_number(result.get("generated_decode_tok_s")),
                format_number(result.get("continuation_decode_tok_s")),
                format_mib(result.get("vram_min_free_mib")),
                format_mib(result.get("nccl_after_output_head_threshold_mib")),
                str(result.get("nccl_after_output_head_failures") or 0),
            ]
        )
    with path.open("w", encoding="utf-8") as out:
        out.write("# DS4 V100 TP/EP NCCL + KV Matrix\n\n")
        out.write("| " + " | ".join(headers) + " |\n")
        out.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            out.write("| " + " | ".join(row) + " |\n")


def format_number(value):
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return "n/a"


def format_mib(value):
    if isinstance(value, (int, float)):
        return f"{value:.0f} MiB"
    return "n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Run the TP/EP NCCL plus KV admission/performance matrix."
    )
    parser.add_argument(
        "--profile-script",
        default="tools/ds4-v100-tp-ep-profile.py",
    )
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--cases",
        default="control,fp8-kv,hc-nccl,fp8-kv-hc-nccl",
        help="Comma-separated cases. Known: " + ",".join(CASES),
    )
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--position", type=int, default=262080)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--nccl-min-free-mib", type=int, default=1536)
    parser.add_argument("--vram-min-free-mib", type=int, default=64)
    parser.add_argument("--tool", default="none")
    args = parser.parse_args()

    selected = [case.strip() for case in args.cases.split(",") if case.strip()]
    unknown = [case for case in selected if case not in CASES]
    if unknown:
        parser.error("unknown cases: " + ",".join(unknown))

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case_name in selected:
        cmd, case_dir = case_command(args, case_name)
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / "matrix-command.txt").open("w", encoding="utf-8") as out:
            out.write(" ".join(cmd) + "\n")
        start = time.time()
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        elapsed_s = time.time() - start
        (case_dir / "matrix-stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (case_dir / "matrix-stderr.txt").write_text(proc.stderr, encoding="utf-8")
        summary = find_summary(case_dir)
        if not summary:
            summary = parse_json_line(proc.stdout)
        result = compact_result(case_name, proc, summary, elapsed_s)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary_path = args.artifact_dir / "matrix-summary.json"
    with summary_path.open("w", encoding="utf-8") as out:
        json.dump({"cases": results}, out, indent=2, sort_keys=True)
        out.write("\n")
    write_markdown(args.artifact_dir / "matrix-summary.md", results)


if __name__ == "__main__":
    main()
