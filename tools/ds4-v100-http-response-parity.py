#!/usr/bin/env python3
"""Compare DS4 V100 HTTP response artifacts for generated-token parity."""

from __future__ import annotations

import argparse
import glob
import json
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
        status_text = status_text.strip()
        try:
            status = int(status_text)
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
        if not match:
            continue
        index = int(match.group(1))
        if index in out:
            raise ValueError(f"duplicate response index {index} under {root}")
        out[index] = path
    return out


def choice(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, dict) else {}


def message_content(body: dict[str, Any]) -> str | None:
    msg = choice(body).get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        return content if isinstance(content, str) else None
    text = choice(body).get("text")
    return text if isinstance(text, str) else None


def choice_token_ids(body: dict[str, Any]) -> list[int] | None:
    tokens = choice(body).get("token_ids")
    if not isinstance(tokens, list):
        return None
    out: list[int] = []
    for value in tokens:
        if not isinstance(value, int):
            return None
        out.append(value)
    return out


def ds4_meta(body: dict[str, Any]) -> dict[str, Any]:
    meta = body.get("ds4_v100")
    return meta if isinstance(meta, dict) else body


def generated_sequence(body: dict[str, Any]) -> list[int] | None:
    meta = ds4_meta(body)
    seq = meta.get("generated_token_sequence")
    if isinstance(seq, list):
        out: list[int] = []
        for value in seq:
            if not isinstance(value, int):
                return None
            out.append(value)
        return out
    return choice_token_ids(body)


def checksum(body: dict[str, Any]) -> int | str | None:
    value = ds4_meta(body).get("checksum")
    if isinstance(value, (int, str)):
        return value
    return None


def selected_token(body: dict[str, Any]) -> int | None:
    value = ds4_meta(body).get("selected_token")
    return value if isinstance(value, int) else None


def generated_text(body: dict[str, Any]) -> str | None:
    value = ds4_meta(body).get("generated_text")
    if isinstance(value, str):
        return value
    return message_content(body)


def compare_pair(control: ParsedResponse,
                 candidate: ParsedResponse,
                 require_checksum: bool,
                 require_text: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": control.index,
        "control_path": str(control.path),
        "candidate_path": str(candidate.path),
        "match": True,
        "mismatches": [],
    }
    if control.parse_error or candidate.parse_error:
        result["match"] = False
        result["mismatches"].append(
            {
                "field": "parse",
                "control": control.parse_error,
                "candidate": candidate.parse_error,
            }
        )
        return result
    assert control.body is not None
    assert candidate.body is not None

    checks: list[tuple[str, Any, Any, bool]] = [
        ("http_status", control.status, candidate.status, True),
        ("generated_token_sequence", generated_sequence(control.body), generated_sequence(candidate.body), True),
        ("choice_token_ids", choice_token_ids(control.body), choice_token_ids(candidate.body), False),
        ("selected_token", selected_token(control.body), selected_token(candidate.body), False),
        ("checksum", checksum(control.body), checksum(candidate.body), require_checksum),
        ("generated_text", generated_text(control.body), generated_text(candidate.body), require_text),
    ]
    for field, left, right, required in checks:
        if left != right:
            result["mismatches"].append({"field": field, "control": left, "candidate": right})
            if required:
                result["match"] = False

    result["control_first_token"] = (generated_sequence(control.body) or [None])[0]
    result["candidate_first_token"] = (generated_sequence(candidate.body) or [None])[0]
    result["control_checksum"] = checksum(control.body)
    result["candidate_checksum"] = checksum(candidate.body)
    result["control_generated_tokens"] = len(generated_sequence(control.body) or [])
    result["candidate_generated_tokens"] = len(generated_sequence(candidate.body) or [])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--allow-missing-checksum", action="store_true")
    parser.add_argument("--ignore-text", action="store_true")
    args = parser.parse_args()

    control_files = response_files(args.control_dir)
    candidate_files = response_files(args.candidate_dir)
    control_indices = set(control_files)
    candidate_indices = set(candidate_files)
    common = sorted(control_indices & candidate_indices)

    summary: dict[str, Any] = {
        "schema": "ds4_v100_http_response_parity.v1",
        "control_dir": str(args.control_dir),
        "candidate_dir": str(args.candidate_dir),
        "control_count": len(control_files),
        "candidate_count": len(candidate_files),
        "missing_in_control": sorted(candidate_indices - control_indices),
        "missing_in_candidate": sorted(control_indices - candidate_indices),
        "pairs": [],
    }

    all_match = not summary["missing_in_control"] and not summary["missing_in_candidate"]
    for index in common:
        control = parse_response(control_files[index], index)
        candidate = parse_response(candidate_files[index], index)
        pair = compare_pair(
            control,
            candidate,
            require_checksum=not args.allow_missing_checksum,
            require_text=not args.ignore_text,
        )
        summary["pairs"].append(pair)
        all_match = all_match and bool(pair["match"])

    matched = sum(1 for pair in summary["pairs"] if pair["match"])
    summary["paired_count"] = len(summary["pairs"])
    summary["matched_pairs"] = matched
    summary["failed_pairs"] = len(summary["pairs"]) - matched
    summary["match"] = all_match

    text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
