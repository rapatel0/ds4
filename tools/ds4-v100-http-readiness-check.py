#!/usr/bin/env python3
"""Validate one DS4 V100 TP/EP HTTP serving profile artifact."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any


RESPONSE_RE = re.compile(r"response-(\d+)\.txt$")


@dataclass
class ResponseArtifact:
    index: int
    path: pathlib.Path
    status: int | None
    body: dict[str, Any] | None
    error: str | None


def load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root is not an object")
    return value


def response_files(root: pathlib.Path) -> list[pathlib.Path]:
    paths: list[tuple[int, pathlib.Path]] = []
    for raw in glob.glob(str(root / "response-*.txt")):
        path = pathlib.Path(raw)
        match = RESPONSE_RE.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(paths)]


def parse_response(path: pathlib.Path) -> ResponseArtifact:
    match = RESPONSE_RE.search(path.name)
    index = int(match.group(1)) if match else -1
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ResponseArtifact(index, path, None, None, f"read_error: {exc}")
    status = None
    body_text = text
    if "\nHTTP_STATUS:" in text:
        body_text, raw_status = text.rsplit("\nHTTP_STATUS:", 1)
        try:
            status = int(raw_status.strip())
        except ValueError:
            return ResponseArtifact(index, path, None, None, f"bad_http_status: {raw_status!r}")
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        return ResponseArtifact(index, path, status, None, f"json_error: {exc}")
    if not isinstance(body, dict):
        return ResponseArtifact(index, path, status, None, "json_root_not_object")
    return ResponseArtifact(index, path, status, body, None)


def ds4_meta(response: ResponseArtifact) -> dict[str, Any]:
    if not response.body:
        return {}
    meta = response.body.get("ds4_v100")
    return meta if isinstance(meta, dict) else {}


def generated_sequence(meta: dict[str, Any]) -> list[int] | None:
    value = meta.get("generated_token_sequence")
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return None
        out.append(item)
    return out


def truthy_int(value: Any) -> bool:
    return value in (1, True)


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def count_gpu_samples(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as src:
            reader = csv.reader(src)
            rows = list(reader)
    except OSError:
        return 0
    if not rows:
        return 0
    header = rows[0]
    return max(0, len(rows) - 1) if any("gpu" in col.lower() for col in header) else len(rows)


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str,
              actual: Any = None, expected: Any = None) -> None:
    item: dict[str, Any] = {"name": name, "ok": bool(ok), "detail": detail}
    if actual is not None:
        item["actual"] = actual
    if expected is not None:
        item["expected"] = expected
    checks.append(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path)
    parser.add_argument("--status", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--expect-requests", type=int)
    parser.add_argument("--expect-tokens", type=int)
    parser.add_argument("--expect-slots", type=int)
    parser.add_argument("--expect-ctx", type=int)
    parser.add_argument("--expect-prompt-count", type=int)
    parser.add_argument("--expect-prompt-digest")
    parser.add_argument("--min-server-decode-tok-s", type=float, default=1.0)
    parser.add_argument("--min-client-generated-tok-s", type=float, default=1.0)
    parser.add_argument("--min-gpu-util-avg", type=float, default=0.0)
    parser.add_argument("--min-gpu-samples", type=int, default=0)
    parser.add_argument("--min-free-mib", type=float, default=64.0)
    parser.add_argument("--max-vram-failures", type=int, default=0)
    parser.add_argument("--require-summary", action="store_true")
    parser.add_argument("--require-status", action="store_true")
    parser.add_argument("--require-vram", action="store_true")
    parser.add_argument("--require-gpu-samples", action="store_true")
    parser.add_argument("--require-resident-kv", action="store_true")
    parser.add_argument("--require-typed-kv", action="store_true")
    parser.add_argument("--require-compact-moe", action="store_true")
    parser.add_argument("--require-token-match", action="store_true")
    parser.add_argument("--require-checksum", action="store_true")
    args = parser.parse_args()

    case_dir = args.case_dir
    summary_path = args.summary or case_dir / "summary.json"
    status_path = args.status or case_dir / "status.json"
    checks: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    status: dict[str, Any] = {}

    if summary_path.exists():
        try:
            summary = load_json(summary_path)
            add_check(checks, "summary_json", True, "summary loaded", str(summary_path))
        except ValueError as exc:
            add_check(checks, "summary_json", False, str(exc))
    else:
        add_check(
            checks,
            "summary_json",
            not args.require_summary,
            "summary missing" if args.require_summary else "summary missing but optional",
            str(summary_path),
        )

    if status_path.exists():
        try:
            status = load_json(status_path)
            add_check(checks, "status_json", True, "status loaded", str(status_path))
        except ValueError as exc:
            add_check(checks, "status_json", False, str(exc))
    else:
        add_check(
            checks,
            "status_json",
            not args.require_status,
            "status missing" if args.require_status else "status missing but optional",
            str(status_path),
        )

    responses = [parse_response(path) for path in response_files(case_dir)]
    metas = [ds4_meta(response) for response in responses]
    response_errors = [r.error for r in responses if r.error]
    http_ok = [r for r in responses if r.status == 200]
    add_check(checks, "responses_parse", not response_errors,
              "all responses parsed" if not response_errors else "; ".join(response_errors[:4]))
    add_check(
        checks,
        "http_200",
        len(http_ok) == len(responses) and bool(responses),
        "all responses have HTTP 200",
        {"http_200": len(http_ok), "responses": len(responses)},
    )

    expected_requests = args.expect_requests or summary.get("requests")
    if expected_requests is not None:
        add_check(
            checks,
            "request_count",
            len(responses) == int(expected_requests),
            "response count matches expected requests",
            len(responses),
            int(expected_requests),
        )

    summary_http_200 = summary.get("http_200")
    if summary_http_200 is not None:
        add_check(
            checks,
            "summary_http_200",
            int(summary_http_200) == len(http_ok),
            "summary HTTP count matches response files",
            int(summary_http_200),
            len(http_ok),
        )

    expected_tokens = args.expect_tokens or summary.get("tokens")
    if expected_tokens is not None:
        token_lengths = [len(seq) for meta in metas if (seq := generated_sequence(meta)) is not None]
        missing = len(metas) - len(token_lengths)
        add_check(
            checks,
            "generated_token_sequences",
            missing == 0 and all(length == int(expected_tokens) for length in token_lengths),
            "generated token sequences are present and have expected length",
            {"lengths": sorted(set(token_lengths)), "missing": missing},
            int(expected_tokens),
        )

    for name, expected, actuals in [
        ("slots", args.expect_slots, [meta.get("slots") for meta in metas]),
        ("ctx", args.expect_ctx, [meta.get("ctx") for meta in metas]),
    ]:
        if expected is not None:
            add_check(
                checks,
                name,
                all(value == expected for value in actuals),
                f"all response metadata {name} values match",
                sorted(set(actuals)),
                expected,
            )

    if args.expect_prompt_count is not None:
        add_check(
            checks,
            "prompt_count",
            summary.get("prompt_count") == args.expect_prompt_count,
            "summary prompt count matches",
            summary.get("prompt_count"),
            args.expect_prompt_count,
        )
    if args.expect_prompt_digest is not None:
        add_check(
            checks,
            "prompt_digest",
            summary.get("prompt_digest") == args.expect_prompt_digest,
            "summary prompt digest matches",
            summary.get("prompt_digest"),
            args.expect_prompt_digest,
        )

    if args.require_token_match:
        mismatches = [meta.get("token_mismatch") for meta in metas if meta.get("token_mismatch") not in (0, None)]
        nonmatches = [meta.get("token_match") for meta in metas if meta.get("token_match") not in (1, True, None)]
        add_check(
            checks,
            "token_match",
            not mismatches and not nonmatches,
            "token_match metadata is clean",
            {"token_mismatch_values": mismatches, "token_match_values": nonmatches},
        )

    if args.require_checksum:
        missing = [i for i, meta in enumerate(metas) if meta.get("checksum") is None]
        add_check(
            checks,
            "checksum",
            not missing,
            "all responses include DS4 checksum",
            missing,
        )

    if args.require_resident_kv:
        resident_bad = [
            i for i, meta in enumerate(metas)
            if not truthy_int(meta.get("kv_runtime_resident")) or
            not truthy_int(meta.get("hc_persist_state_gate"))
        ]
        add_check(
            checks,
            "resident_kv",
            not resident_bad,
            "KV runtime and HC persistence are resident in response metadata",
            resident_bad,
        )

    if args.require_typed_kv:
        required = [
            "true_ds4_attention_typed_kv_raw_gate",
            "true_ds4_attention_typed_kv_compressed_gate",
            "true_ds4_attention_typed_kv_indexer_gate",
            "true_ds4_attention_typed_kv_history_gate",
            "true_ds4_attention_typed_kv_skip_current_load_gate",
            "true_ds4_attention_typed_kv_quiet_gate",
            "true_ds4_attention_typed_kv_batch_rows_gate",
            "true_ds4_attention_typed_kv_stream_sync_gate",
        ]
        typed_bad = [
            {"index": i, "field": field, "value": meta.get(field)}
            for i, meta in enumerate(metas)
            for field in required
            if not truthy_int(meta.get(field))
        ]
        add_check(
            checks,
            "typed_kv",
            not typed_bad,
            "typed DS4 KV metadata gates are enabled",
            typed_bad[:16],
        )

    if args.require_compact_moe:
        compact_ok = truthy_int(summary.get("scaffold_compact_moe_decode_gate"))
        add_check(
            checks,
            "compact_moe",
            compact_ok,
            "summary confirms compact MoE decode gate",
            summary.get("scaffold_compact_moe_decode_gate"),
            1,
        )

    generated_meta = summary.get("generated_tokens_meta")
    if expected_requests is not None and expected_tokens is not None and generated_meta is not None:
        expected_generated = int(expected_requests) * int(expected_tokens)
        add_check(
            checks,
            "generated_tokens_meta",
            int(generated_meta) == expected_generated,
            "summary generated token total matches shape",
            int(generated_meta),
            expected_generated,
        )

    for field, threshold in [
        ("server_generated_tok_s_decode", args.min_server_decode_tok_s),
        ("client_generated_tok_s", args.min_client_generated_tok_s),
        ("gpu_util_avg", args.min_gpu_util_avg),
    ]:
        value = numeric(summary.get(field))
        add_check(
            checks,
            field,
            value is not None and value >= threshold,
            f"{field} is present and above threshold",
            value,
            threshold,
        )

    gpu_samples = int(summary.get("gpu_sample_count") or count_gpu_samples(case_dir / "gpu_util.csv"))
    if args.require_gpu_samples or args.min_gpu_samples:
        add_check(
            checks,
            "gpu_samples",
            gpu_samples >= args.min_gpu_samples and gpu_samples > 0,
            "GPU utilization samples are present",
            gpu_samples,
            args.min_gpu_samples,
        )

    if args.require_vram:
        failures = summary.get("vram_failures")
        min_free = numeric(summary.get("vram_min_free_mib"))
        add_check(
            checks,
            "vram_failures",
            failures is not None and int(failures) <= args.max_vram_failures,
            "VRAM admission failure count is acceptable",
            failures,
            args.max_vram_failures,
        )
        add_check(
            checks,
            "vram_min_free_mib",
            min_free is not None and min_free >= args.min_free_mib,
            "minimum free VRAM is above threshold",
            min_free,
            args.min_free_mib,
        )

    fail = [check for check in checks if not check["ok"]]
    output = {
        "schema": "ds4_v100_http_readiness.v1",
        "case_dir": str(case_dir),
        "summary": str(summary_path),
        "status": str(status_path),
        "ready": not fail,
        "failure_count": len(fail),
        "response_count": len(responses),
        "topline": {
            "http_200": len(http_ok),
            "requests": summary.get("requests"),
            "tokens": summary.get("tokens"),
            "client_generated_tok_s": summary.get("client_generated_tok_s"),
            "server_generated_tok_s_decode": summary.get("server_generated_tok_s_decode"),
            "gpu_util_avg": summary.get("gpu_util_avg"),
            "gpu_util_max": summary.get("gpu_util_max"),
            "vram_min_free_mib": summary.get("vram_min_free_mib"),
            "vram_failures": summary.get("vram_failures"),
            "output_head_first_token": summary.get("output_head_first_token"),
            "compressed_kv_sum_ms": summary.get("compressed_kv_sum_ms"),
        },
        "checks": checks,
    }

    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if output["ready"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
