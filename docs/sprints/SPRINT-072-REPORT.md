# Sprint 072 Report: MTP Commit Throughput Decision Gate

## Summary

Sprint 072 shipped benchmark support for MTP `off`, `verify`, and `commit`
modes and ran the V100 decision gate. Exact commit remains safe and observable,
but it is not a throughput win on the measured one-slot exact-verify fixture.

## Implementation

- Added MTP controls to `tools/ds4-v100-sustained-decode-bench.sh`:
  `--mtp-model`, `--mtp-serving off|verify|commit`, `--mtp-top-k`,
  `--mtp-gpu`, and `--mtp-reserve-mib`.
- Kept non-MTP as the default benchmark mode.
- Guarded MTP sustained benchmark cases to slot tier `1`.
- Forwarded MTP serving flags to `tools/ds4-v100-replay` only when enabled.
- Recorded MTP attempted, accepted, rejected, committed, skipped, average draft
  ms, and total draft ms in case JSON and TSV output.
- Preserved server status snapshots in each case result for MTP server counters
  and mode evidence.

## V100 Evidence

Cluster build:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc \
  'cd /workspace/ds4-sprint072 && \
   bash -n tools/ds4-v100-sustained-decode-bench.sh && \
   CUDA_ARCH=sm_70 make tools/ds4-v100-replay'
```

Comparison fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- MTP model: `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`
- context: `1048576`
- slots: `1`
- queue policy: `sequential`
- tokens/request: `2`
- measured requests/mode: `4`
- warmup requests/mode: `1`
- expected first token hex: `3136`

| Mode | Status | Generated tok/s | Continuation tok/s | Avg latency ms | Avg GPU util | MTP attempted | MTP accepted | MTP committed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| off | `4/4` matched | `0.788607` | `0.394304` | `2535.818` | `10.301%` | `0` | `0` | `0` |
| verify | `4/4` matched | `0.774126` | `0.387063` | `2583.175` | `10.201%` | `4` | `4` | `0` |
| commit | `4/4` matched | `0.777308` | `0.388654` | `2572.592` | `10.093%` | `4` | `4` | `4` |

The default non-MTP path also ran without passing `--mtp-serving`, proving the
new flags preserve the default off behavior.

## Decision

Exact commit is correct but not throughput-positive:

- verify was `1.836%` slower than off on generated tok/s;
- commit was `1.433%` slower than off on generated tok/s;
- commit accepted and committed every measured draft;
- server status reported `mode="mtp_commit_one_slot"` and
  `mtp.serving_mode="commit"`.

The next optimization sprint should pivot to stage/kernel throughput. Recursive
MTP or skip-verify MTP should wait for a separate acceptance and safety study
because exact verification still pays for the target token and therefore cannot
deliver speculative speedup by itself.

## Artifacts

- `logs/from-cluster/sprint072-mtp-off`
- `logs/from-cluster/sprint072-mtp-verify`
- `logs/from-cluster/sprint072-mtp-commit`
- `logs/from-cluster/sprint072-mtp-default-off`
- `logs/from-cluster/sprint072-mtp-comparison`

## Validation

- `bash -n tools/ds4-v100-sustained-decode-bench.sh`
- `CUDA_ARCH=sm_70 make tools/ds4-v100-replay` on the V100 pod
- `make ds4_v100_replay.o tools/ds4-v100-replay.o`
- `python3 -m json.tool` on copied Sprint 072 result and comparison JSON files
- `git diff --check`
