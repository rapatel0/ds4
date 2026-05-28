#!/usr/bin/env python3
"""Compare DS4 V100 HTTP responses with an arithmetic-change tolerance gate."""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any


RESPONSE_RE = re.compile(r"response-(\d+)\.txt$")


@dataclass
class ParsedResponse:
    index: int
    path: pathlib.Path
    status: int | None
    body: dict[str, Any] | None
    parse_error: str | None


def parse_response(path: pathlib.Path, index: int) -> ParsedResponse:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ParsedResponse(index, path, None, None, f"read_error: {exc}")

    status = None
    body_text = text
    if "\nHTTP_STATUS:" in text:
        body_text, status_text = text.rsplit("\nHTTP_STATUS:", 1)
        try:
            status = int(status_text.strip())
        except ValueError:
            return ParsedResponse(index, path, None, None, f"bad_http_status: {status_text!r}")
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        return ParsedResponse(index, path, status, None, f"json_error: {exc}")
    if not isinstance(body, dict):
        return ParsedResponse(index, path, status, None, "json_root_not_object")
    return ParsedResponse(index, path, status, body, None)


def response_files(root: pathlib.Path) -> dict[int, pathlib.Path]:
    out: dict[int, pathlib.Path] = {}
    for raw in glob.glob(str(root / "response-*.txt")):
        path = pathlib.Path(raw)
        match = RESPONSE_RE.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def ds4_meta(body: dict[str, Any]) -> dict[str, Any]:
    meta = body.get("ds4_v100")
    return meta if isinstance(meta, dict) else body


def selected_token(body: dict[str, Any]) -> int | None:
    value = ds4_meta(body).get("selected_token")
    return value if isinstance(value, int) else None


def selected_logit(body: dict[str, Any]) -> float | None:
    value = ds4_meta(body).get("selected_logit")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def generated_sequence(body: dict[str, Any]) -> list[int]:
    value = ds4_meta(body).get("generated_token_sequence")
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if not isinstance(item, int):
            return []
        out.append(item)
    return out


def rel_error(control: float, candidate: float) -> float:
    denom = max(abs(control), 1.0)
    return abs(candidate - control) / denom


def compare(args: argparse.Namespace) -> dict[str, Any]:
    control_files = response_files(args.control_dir)
    candidate_files = response_files(args.candidate_dir)
    common = sorted(set(control_files) & set(candidate_files))
    pairs: list[dict[str, Any]] = []
    token_matches = 0
    token_total = 0
    sequence_matches = 0
    sequence_total = 0
    max_selected_logit_relative_error = 0.0
    selected_logit_pairs = 0
    parse_failures = 0
    http_failures = 0

    for index in common:
        control = parse_response(control_files[index], index)
        candidate = parse_response(candidate_files[index], index)
        pair: dict[str, Any] = {
            "index": index,
            "control_path": str(control.path),
            "candidate_path": str(candidate.path),
        }
        if control.parse_error or candidate.parse_error:
            pair["parse_error"] = {
                "control": control.parse_error,
                "candidate": candidate.parse_error,
            }
            parse_failures += 1
            pairs.append(pair)
            continue
        if control.status != candidate.status or control.status != 200:
            pair["http_status"] = {
                "control": control.status,
                "candidate": candidate.status,
            }
            http_failures += 1
        assert control.body is not None
        assert candidate.body is not None
        left_token = selected_token(control.body)
        right_token = selected_token(candidate.body)
        if left_token is not None and right_token is not None:
            token_total += 1
            if left_token == right_token:
                token_matches += 1
        left_seq = generated_sequence(control.body)
        right_seq = generated_sequence(candidate.body)
        for left, right in zip(left_seq, right_seq):
            sequence_total += 1
            if left == right:
                sequence_matches += 1
        left_logit = selected_logit(control.body)
        right_logit = selected_logit(candidate.body)
        pair["control_selected_token"] = left_token
        pair["candidate_selected_token"] = right_token
        pair["control_selected_logit"] = left_logit
        pair["candidate_selected_logit"] = right_logit
        if left_logit is not None and right_logit is not None:
            selected_logit_pairs += 1
            pair_rel = rel_error(left_logit, right_logit)
            pair["selected_logit_relative_error"] = pair_rel
            max_selected_logit_relative_error = max(max_selected_logit_relative_error, pair_rel)
        pairs.append(pair)

    token_agreement = (token_matches / token_total) if token_total else 0.0
    sequence_agreement = (sequence_matches / sequence_total) if sequence_total else 0.0
    enough_pairs = len(common) >= args.min_pairs
    logit_ok_advisory = (
        selected_logit_pairs > 0
        and max_selected_logit_relative_error <= args.max_selected_logit_relative_error
    )
    top1_ok = token_agreement >= args.min_top1_agreement
    sequence_ok = sequence_agreement >= args.min_top1_agreement
    passed = (
        not parse_failures
        and not http_failures
        and not set(control_files).symmetric_difference(candidate_files)
        and enough_pairs
        and top1_ok
        and sequence_ok
    )
    return {
        "schema": "ds4_http_response_tolerance.v1",
        "control_dir": str(args.control_dir),
        "candidate_dir": str(args.candidate_dir),
        "control_count": len(control_files),
        "candidate_count": len(candidate_files),
        "paired_count": len(common),
        "min_pairs": args.min_pairs,
        "missing_in_control": sorted(set(candidate_files) - set(control_files)),
        "missing_in_candidate": sorted(set(control_files) - set(candidate_files)),
        "parse_failures": parse_failures,
        "http_failures": http_failures,
        "selected_token_matches": token_matches,
        "selected_token_total": token_total,
        "selected_token_agreement": token_agreement,
        "generated_sequence_matches": sequence_matches,
        "generated_sequence_total": sequence_total,
        "generated_sequence_agreement": sequence_agreement,
        "min_top1_agreement": args.min_top1_agreement,
        "selected_logit_pairs": selected_logit_pairs,
        "max_selected_logit_relative_error": max_selected_logit_relative_error,
        "max_selected_logit_relative_error_threshold": args.max_selected_logit_relative_error,
        "max_selected_logit_relative_error_advisory_ok": logit_ok_advisory,
        "max_selected_logit_relative_error_policy": "advisory_only",
        "pass": passed,
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument("--min-top1-agreement", type=float, default=0.99)
    parser.add_argument("--max-selected-logit-relative-error", type=float, default=1e-3)
    args = parser.parse_args()

    summary = compare(args)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
