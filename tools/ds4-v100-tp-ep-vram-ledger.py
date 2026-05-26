#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import sys


def fields_from_line(line):
    parts = line.rstrip("\n").split("\t")
    if not parts:
        return None, {}
    tag = parts[0]
    fields = {}
    i = 1
    while i + 1 < len(parts):
        fields[parts[i]] = parts[i + 1]
        i += 2
    return tag, fields


def maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def find_case_files(case_dir):
    case_dir = pathlib.Path(case_dir)
    stdout_files = sorted(case_dir.glob("*/stdout.txt"))
    summary_files = sorted(case_dir.glob("*/summary.json"))
    if not stdout_files and (case_dir / "stdout.txt").exists():
        stdout_files = [case_dir / "stdout.txt"]
    if not summary_files and (case_dir / "summary.json").exists():
        summary_files = [case_dir / "summary.json"]
    return stdout_files[-1] if stdout_files else None, summary_files[-1] if summary_files else None


def parse_stdout(path):
    checkpoints = {}
    summaries = {}
    metadata = {}
    if path is None:
        return checkpoints, summaries, metadata
    with pathlib.Path(path).open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            tag, fields = fields_from_line(line)
            if tag == "tp_ep_vram":
                label = fields.get("label")
                gpu = maybe_int(fields.get("gpu"))
                if label is None or gpu is None:
                    continue
                checkpoints.setdefault(label, {})[str(gpu)] = {
                    "free_mib": maybe_int(fields.get("free_mib")),
                    "used_mib": maybe_int(fields.get("used_mib")),
                    "total_mib": maybe_int(fields.get("total_mib")),
                    "threshold_mib": maybe_int(fields.get("min_free_mib")),
                    "pass": fields.get("PASS") == "PASS" or line.rstrip().endswith("\tPASS"),
                }
            elif tag == "tp_ep_vram_summary":
                label = fields.get("label")
                if label is None:
                    continue
                summaries[label] = {
                    "min_free_mib": maybe_int(fields.get("min_free_mib")),
                    "max_used_mib": maybe_int(fields.get("max_used_mib")),
                    "threshold_mib": maybe_int(fields.get("threshold_mib")),
                    "failures": maybe_int(fields.get("failures")),
                    "pass": line.rstrip().endswith("\tPASS"),
                }
            elif tag == "tp_ep_hc_final_expand_shared":
                metadata["hc_control_bytes"] = maybe_int(fields.get("control_bytes"))
                metadata["hc_slots"] = maybe_int(fields.get("slots"))
            elif tag == "tp_ep_diagnostic_output_head_shared":
                metadata["output_weight_bytes"] = maybe_int(fields.get("output_weight_bytes"))
                metadata["output_logits_bytes"] = maybe_int(fields.get("logits_bytes"))
                metadata["output_vocab"] = maybe_int(fields.get("vocab"))
                metadata["output_rows_per_gpu"] = maybe_int(fields.get("rows_per_gpu"))
    return checkpoints, summaries, metadata


def load_summary(path):
    if path is None:
        return {}
    with pathlib.Path(path).open("r", encoding="utf-8") as src:
        return json.load(src)


def delta_table(checkpoints):
    labels = [
        "startup",
        "after_dense_f16_cache",
        "after_rank_buffers",
        "nccl_after_rank_buffers",
        "after_tp_runtime",
        "after_dense_ops",
        "after_hc_controls",
        "after_output_head",
        "nccl_after_output_head",
    ]
    rows = []
    previous = None
    for label in labels:
        gpu_rows = checkpoints.get(label)
        if not gpu_rows:
            continue
        row = {"label": label, "gpus": gpu_rows}
        if previous is not None:
            deltas = {}
            prev_rows = checkpoints.get(previous, {})
            for gpu, values in gpu_rows.items():
                prev = prev_rows.get(gpu)
                if not prev:
                    continue
                free = values.get("free_mib")
                prev_free = prev.get("free_mib")
                if isinstance(free, int) and isinstance(prev_free, int):
                    deltas[gpu] = prev_free - free
            row["delta_from_previous_label"] = previous
            row["delta_used_mib"] = deltas
        rows.append(row)
        previous = label
    return rows


def threshold_deficits(checkpoints, threshold_mib):
    label = "nccl_after_output_head"
    if label not in checkpoints:
        label = "after_output_head"
    deficits = {}
    for gpu, values in checkpoints.get(label, {}).items():
        free = values.get("free_mib")
        if isinstance(free, int):
            deficits[gpu] = max(0, threshold_mib - free)
    return label, deficits


def mib(value):
    if value is None:
        return None
    return value / (1024 * 1024)


def parse_case_arg(raw):
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--case must be NAME=DIR")
    name, path = raw.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--case must be NAME=DIR")
    return name, pathlib.Path(path)


def summarize_case(name, path, threshold_mib):
    stdout_path, summary_path = find_case_files(path)
    checkpoints, summaries, metadata = parse_stdout(stdout_path)
    profile_summary = load_summary(summary_path)
    deficit_label, deficits = threshold_deficits(checkpoints, threshold_mib)
    max_deficit = max(deficits.values()) if deficits else None
    failing_gpus = [gpu for gpu, deficit in sorted(deficits.items(), key=lambda item: int(item[0])) if deficit > 0]
    return {
        "name": name,
        "path": str(path),
        "stdout": str(stdout_path) if stdout_path else None,
        "summary": str(summary_path) if summary_path else None,
        "profile": {
            "returncode": profile_summary.get("returncode"),
            "first_token": profile_summary.get("output_head_first_token"),
            "generated_decode_tok_s": profile_summary.get("serving_aggregate_generated_tok_s_decode"),
            "continuation_decode_tok_s": profile_summary.get("serving_aggregate_continuation_tok_s_decode"),
            "vram_min_free_mib": profile_summary.get("vram_min_free_mib"),
            "vram_max_used_mib": profile_summary.get("vram_max_used_mib"),
        },
        "metadata": {
            **metadata,
            "hc_control_mib": mib(metadata.get("hc_control_bytes")),
            "output_weight_mib": mib(metadata.get("output_weight_bytes")),
            "output_logits_mib": mib(metadata.get("output_logits_bytes")),
        },
        "summaries": summaries,
        "checkpoints": checkpoints,
        "deltas": delta_table(checkpoints),
        "threshold_mib": threshold_mib,
        "deficit_label": deficit_label,
        "deficits_mib": deficits,
        "max_deficit_mib": max_deficit,
        "failing_gpus": failing_gpus,
    }


def format_mib(value):
    if isinstance(value, float):
        return f"{value:.1f}"
    if isinstance(value, int):
        return str(value)
    return "n/a"


def write_markdown(path, ledger):
    with pathlib.Path(path).open("w", encoding="utf-8") as out:
        out.write("# DS4 V100 TP/EP VRAM Ledger\n\n")
        out.write(f"Threshold: `{ledger['threshold_mib']} MiB`\n\n")
        out.write("## Case Summary\n\n")
        out.write("| Case | Return | First token | Min free | Max deficit | Failing GPUs | Decode tok/s |\n")
        out.write("|---|---:|---:|---:|---:|---|---:|\n")
        for case in ledger["cases"]:
            profile = case["profile"]
            out.write(
                "| {name} | {returncode} | {first_token} | {min_free} | {max_deficit} | {failing_gpus} | {decode} |\n".format(
                    name=case["name"],
                    returncode=profile.get("returncode", "n/a"),
                    first_token=profile.get("first_token", "n/a"),
                    min_free=format_mib(profile.get("vram_min_free_mib")),
                    max_deficit=format_mib(case.get("max_deficit_mib")),
                    failing_gpus=",".join(case.get("failing_gpus") or []) or "none",
                    decode=format_mib(profile.get("generated_decode_tok_s")),
                )
            )
        out.write("\n## Allocation Metadata\n\n")
        out.write("| Case | HC control MiB | Output weight MiB | Output logits MiB |\n")
        out.write("|---|---:|---:|---:|\n")
        for case in ledger["cases"]:
            meta = case["metadata"]
            out.write(
                "| {name} | {hc} | {ow} | {ol} |\n".format(
                    name=case["name"],
                    hc=format_mib(meta.get("hc_control_mib")),
                    ow=format_mib(meta.get("output_weight_mib")),
                    ol=format_mib(meta.get("output_logits_mib")),
                )
            )
        out.write("\n## Checkpoint Deltas\n\n")
        for case_idx, case in enumerate(ledger["cases"]):
            out.write(f"### {case['name']}\n\n")
            out.write("| Checkpoint | Min free | Failures | Delta used by GPU |\n")
            out.write("|---|---:|---:|---|\n")
            for row in case["deltas"]:
                label = row["label"]
                summary = case["summaries"].get(label, {})
                deltas = row.get("delta_used_mib") or {}
                delta_text = ", ".join(
                    f"gpu{gpu}:{delta}" for gpu, delta in sorted(deltas.items(), key=lambda item: int(item[0]))
                )
                out.write(
                    f"| `{label}` | {format_mib(summary.get('min_free_mib'))} | "
                    f"{format_mib(summary.get('failures'))} | {delta_text or 'n/a'} |\n"
                )
            if case_idx + 1 < len(ledger["cases"]):
                out.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize TP/EP VRAM checkpoints and NCCL admission deficits."
    )
    parser.add_argument("--case", action="append", type=parse_case_arg, required=True)
    parser.add_argument("--threshold-mib", type=int, default=1536)
    parser.add_argument("--out-json", type=pathlib.Path, required=True)
    parser.add_argument("--out-md", type=pathlib.Path, required=True)
    args = parser.parse_args()

    ledger = {
        "threshold_mib": args.threshold_mib,
        "cases": [
            summarize_case(name, path, args.threshold_mib)
            for name, path in args.case
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as out:
        json.dump(ledger, out, indent=2, sort_keys=True)
        out.write("\n")
    write_markdown(args.out_md, ledger)


if __name__ == "__main__":
    main()
