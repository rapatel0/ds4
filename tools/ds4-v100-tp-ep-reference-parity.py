#!/usr/bin/env python3
"""Reference-vector parity harness for the TP/EP diagnostic server.

This intentionally talks to the HTTP API instead of internal CUDA helpers. The
production gate we need is end-to-end: tokenizer input, prefill, decode,
output-head selection, detokenization, and session cursor updates.
"""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import sys
import time
from dataclasses import dataclass


@dataclass
class VectorCase:
    case_id: str
    ctx: int
    steps: int
    prompt_path: str
    selected_hex: list[str]


def parse_vectors(path: pathlib.Path) -> list[VectorCase]:
    cases: list[VectorCase] = []
    current: VectorCase | None = None
    with path.open("r", encoding="utf-8") as fp:
        for lineno, raw in enumerate(fp, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "case":
                if current is not None:
                    raise ValueError(f"{path}:{lineno}: nested case")
                if len(fields) != 5:
                    raise ValueError(f"{path}:{lineno}: bad case line")
                current = VectorCase(
                    case_id=fields[1],
                    ctx=int(fields[2]),
                    steps=int(fields[3]),
                    prompt_path=fields[4],
                    selected_hex=[],
                )
            elif fields[0] == "step":
                if current is None:
                    raise ValueError(f"{path}:{lineno}: step outside case")
                if len(fields) < 4:
                    raise ValueError(f"{path}:{lineno}: bad step line")
                step = int(fields[1])
                if step != len(current.selected_hex):
                    raise ValueError(f"{path}:{lineno}: non-contiguous step")
                current.selected_hex.append(fields[2])
            elif fields[0] == "top":
                continue
            elif fields[0] == "end":
                if current is None:
                    raise ValueError(f"{path}:{lineno}: end outside case")
                if len(current.selected_hex) != current.steps:
                    raise ValueError(f"{path}:{lineno}: incomplete case")
                cases.append(current)
                current = None
            else:
                raise ValueError(f"{path}:{lineno}: unknown line {fields[0]!r}")
    if current is not None:
        raise ValueError(f"{path}: unterminated case {current.case_id}")
    return cases


def post_json(host: str, port: int, path: str, body: dict, timeout: float) -> dict:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = json.dumps(body)
    conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    text = resp.read().decode("utf-8", errors="replace")
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
    return json.loads(text)


def wait_health(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            if resp.status == 200:
                return
        except OSError as exc:
            last = exc
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy: {last}")


def select_cases(cases: list[VectorCase], only: str | None, limit: int) -> list[VectorCase]:
    if only:
        cases = [c for c in cases if c.case_id == only]
    if limit > 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no vector cases selected")
    return cases


def run_case(root: pathlib.Path,
             host: str,
             port: int,
             case: VectorCase,
             timeout: float) -> dict:
    prompt = (root / case.prompt_path).read_text(encoding="utf-8")
    data = post_json(
        host,
        port,
        "/v1/chat/completions",
        {
            "model": "ds4-v100-tp-ep-diagnostic",
            "session_id": f"parity-{case.case_id}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": case.steps,
        },
        timeout,
    )
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    got_text = message.get("content", "")
    got_hex = got_text.encode("utf-8").hex()
    want_hex = "".join(case.selected_hex)
    meta = data.get("ds4_v100", {})
    return {
        "case": case.case_id,
        "ctx": case.ctx,
        "steps": case.steps,
        "expected_hex": want_hex,
        "actual_hex": got_hex,
        "expected_text": bytes.fromhex(want_hex).decode("utf-8", errors="replace"),
        "actual_text": got_text,
        "match": got_hex == want_hex,
        "tokenizer_ready": meta.get("tokenizer_ready"),
        "prompt_tokens": meta.get("request_prompt_token_ids"),
        "prompt_prefill_tokens": meta.get("prompt_prefill_tokens"),
        "generated_token_ids": meta.get("generated_token_ids"),
        "generated_token_sequence": meta.get("generated_token_sequence"),
        "slot_position": meta.get("slot_position"),
        "wall_tok_s": meta.get("timing_ms", {}).get("generated_tokens_per_second"),
        "decode_tok_s": meta.get("timing_ms", {}).get("generated_tokens_per_second_decode"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--vectors", default="tests/test-vectors/official.vec")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--only")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--wait-health", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = pathlib.Path(args.repo_root)
    cases = select_cases(parse_vectors(root / args.vectors), args.only, args.limit)
    if args.wait_health > 0:
        wait_health(args.host, args.port, args.wait_health)

    results = [run_case(root, args.host, args.port, case, args.timeout) for case in cases]
    passed = sum(1 for result in results if result["match"])
    summary = {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
