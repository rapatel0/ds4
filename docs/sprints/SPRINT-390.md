# Sprint 390: Permanent HTTP Response Parity Comparator

## Overview

Add a reusable comparator for TP/EP HTTP serving artifacts so promotion sprints
can prove generated-token/checksum parity without ad hoc JSON scripts.

Sprint 389 promoted compressed dense stats skip after manually comparing
`32` control and candidate chat responses. That manual step should become a
permanent tool because every future performance gate needs the same evidence:
same HTTP success count, same generated token sequence, same response checksum,
and a clear summary of metadata-only differences.

## Scope

- Add a standalone tool:
  `tools/ds4-v100-http-response-parity.py`.
- Compare response files written by the existing profile/matrix harnesses:
  `response-NN.txt` with trailing `HTTP_STATUS:NNN`.
- Treat generated token sequences and checksums as the primary parity contract.
- Report text/token/checksum/status mismatches in a machine-readable JSON
  summary and return non-zero on parity failure.
- Validate against Sprint 389 control/candidate artifacts.

## Out Of Scope

- No CUDA/kernel changes.
- No PP/layer-split work.
- No changes to the served model output.
- No MTP changes.

## Definition Of Done

- Comparator tool exists and has CLI help.
- Local Python syntax checks pass.
- Comparator passes on the Sprint 389 HTTP A/B artifacts.
- A negative fixture proves token mismatches fail non-zero.
- `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and
  `TEMP_STATUS_REPORT_390.md` are updated.
- Kept artifacts are committed.

## Risks

- Comparing too much metadata will create false failures from request/cache
  counters and timing fields.
- Comparing too little will miss correctness regressions. The tool should make
  generated token IDs and checksums first-class, and include generated text as
  a secondary check.

## Execution Plan

1. Implement the comparator with explicit response parsing and pair matching.
2. Run it on Sprint 389 HTTP control/candidate artifacts.
3. Create a small negative local fixture by mutating one copied response token
   and verify the comparator exits non-zero.
4. Document the tool and commit the sprint.

## Outcome

Complete.

Added `tools/ds4-v100-http-response-parity.py`, a standalone comparator for
HTTP response artifact directories. It parses the existing `response-NN.txt`
format, including the trailing `HTTP_STATUS:NNN` line, and compares paired
responses by:

- HTTP status
- generated token sequence
- choice token IDs
- selected token
- DS4 response checksum
- generated text

Primary parity failures return non-zero and are recorded in a JSON summary.
The tool intentionally ignores request/cache/timing metadata so serving A/B
runs do not fail because of expected metrology differences.

## Validation

Syntax/help:

```text
python3 -m py_compile tools/ds4-v100-http-response-parity.py
tools/ds4-v100-http-response-parity.py --help
```

Sprint 389 parity validation:

```text
match=true
control_count=32
candidate_count=32
paired_count=32
matched_pairs=32
failed_pairs=0
```

Negative fixture:

```text
match=false
failed_pairs=1
first_failed_field=generated_token_sequence
```

## Artifacts

- `logs/from-cluster/sprint389-skip-dense-stats/http-parity-summary.json`

## Decision

Promote the comparator as the standard response-parity check for future TP/EP
HTTP A/B sprints.
