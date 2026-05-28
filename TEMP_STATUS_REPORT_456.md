# TEMP Status Report 456

## Current Focus

Testing whether the now-promoted rank-major router+FFN serving baseline can
skip redundant slot-major FFN norm staging.

## Planned Validation

```text
artifact: /localpool/ds4/workspace/logs/s456-skip-slot-major-ffn-norm-s32-t32
shape:    32 requests / 32 slots / 256K context / 32 generated tokens
control:  Sprint 455 baseline
candidate: control + post-attention skip slot-major FFN norm
```

## Promotion Gate

Promote only if readiness passes, response parity is `32/32`, VRAM failures are
zero, and server decode improves by at least `1.02x`.

## Result

Artifact:

```text
/localpool/ds4/workspace/logs/s456-skip-slot-major-ffn-norm-s32-t32
```

Outcome: **do not promote**.

- Readiness: control `true`, candidate `true`
- HTTP: `32/32` in both legs
- Full response parity: failed, `0/32` matched
- First token: matched at output-head summary (`109865`) and response-0
  (`104565`)
- Server generated decode: `34.999820 -> 35.421446` tok/s (`1.0120x`)
- Server continuation decode: `35.039950 -> 35.392239` tok/s (`1.0101x`)
- Client generated tok/s: `14.767353 -> 14.791231` (`1.0016x`)
- Average GPU util: `11.845652% -> 11.674145%`
- Min free VRAM: `1734 MiB -> 1734 MiB`, `vram_failures=0`
- HC-current input: `393.599693 -> 391.250157 ms`
- HC-current gather: `5.893971 -> 5.763197 ms`

The candidate moved timing slightly in the right direction but did not pass
semantic parity or the throughput gate. Keep
`DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM` diagnostic-only.

## Observation

During the run, startup showed a memory/utilization wave through GPUs, then the
steady phase became more parallel across all GPUs but stayed low-utilization
around the mid-teens. That supports adding always-on lightweight domain
telemetry and reserving Nsight/ncu for short steady-state windows.

## Result

The exclusive rerun completed after killing unrelated stale 4-token benchmark
processes that were occupying the node.

```text
readiness:       pass/pass
response parity: 0/32
first token:     109865 -> 109865
checksum:        17913667570271397799 -> 17913667564178658333
server decode:   34.999820 -> 35.421446 tok/s
continuation:    35.039950 -> 35.392239 tok/s
client tok/s:    14.767353 -> 14.791231
avg GPU util:    11.85% -> 11.67%
min free VRAM:   1734 -> 1734 MiB
VRAM failures:   0 -> 0
```

## Decision

Rejected. Keep `DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM=0`.
The candidate preserves first token but changes the generated response
checksum for all `32` response pairs and is below the `1.02x` throughput gate.
