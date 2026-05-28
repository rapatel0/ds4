# TEMP_STATUS_REPORT_390

Date: 2026-05-25

## Focus

Sprint 390 made the HTTP response parity check permanent. Sprint 389 promoted
a serving default using a manual comparison of generated token sequences and
checksums; that needed to become a reusable tool before more TP/EP performance
gates are promoted.

## Result

Added:

```text
tools/ds4-v100-http-response-parity.py
```

The tool compares paired `response-NN.txt` artifacts from control and candidate
HTTP runs. It parses the trailing `HTTP_STATUS:NNN` line and checks:

- HTTP status
- generated token sequence
- choice token IDs
- selected token
- DS4 response checksum
- generated text

It emits a JSON summary and exits non-zero on parity failure.

## Validation

Sprint 389 artifacts:

```text
control_count=32
candidate_count=32
paired_count=32
matched_pairs=32
failed_pairs=0
match=true
```

Negative fixture:

```text
match=false
failed_pairs=1
first_failed_field=generated_token_sequence
```

Local checks:

```text
python3 -m py_compile tools/ds4-v100-http-response-parity.py
tools/ds4-v100-http-response-parity.py --help
```

## Artifacts

- `logs/from-cluster/sprint389-skip-dense-stats/http-parity-summary.json`

## Next

Use this comparator in every future HTTP A/B promotion decision. The next
performance sprint can now focus on the remaining TP/EP serving bottlenecks
with a stable parity gate instead of manual response inspection.
