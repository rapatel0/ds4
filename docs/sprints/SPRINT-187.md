# Sprint 187 - Synthetic Prompt Replay Profile

Date: 2026-05-22

## Objective

Profile synthetic filled-context prompt replay so the next optimization target
is chosen from stage evidence rather than guessing.

## Scope

- Run direct replay with synthetic prompt mode and `--profile-decode`.
- Use a bounded synthetic tier that completes within the current work window.
- Prefer `--synthetic-prompt-len 1024`; fall back to `256` if synchronized
  profiling makes 1024 too slow.
- Record profile buckets:
  - HC attention prep
  - attention
  - HC FFN prep
  - FFN
  - HC final
- Copy evidence and update the vision.

## Non-Goals

- No kernel rewrite in this sprint.
- No full 256K prefill.
- No online-attention promotion.

## Definition of Done

- [x] A synthetic prompt replay profile completes or a concrete blocker is
      recorded.
- [x] Evidence is archived under `logs/from-cluster/`.
- [x] Sprint result records stage-profile sums.
- [x] Vision is updated with the profile signal.
- [x] Changes are committed.

## Execution

### Instrumentation Repair

The first preferred len-1024 profile completed but reported zero
`stage_profile` buckets even though `stage_decode` was populated. The cause was
in the single-slot layer executor: `ds4_v100_layer_execute_hc_decode_batch()`
recorded profile bucket timings, but the single-slot
`ds4_v100_layer_execute_hc_decode()` path did not.

The sprint therefore first patched the single-slot HC decode path to emit the
same buckets used by the batch path:

- HC attention prep
- attention
- HC FFN prep
- FFN
- HC final expansion

V100 pod build:

```bash
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

### Bounded Check

Patched len-256 synthetic profile:

```bash
tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --synthetic-prompt-token 926 \
  --synthetic-prompt-len 256 \
  --ctx 262144 \
  --tokens 2 \
  --profile-decode \
  --json
```

Result:

- Prompt replay: `21135.064 ms`
- Prompt throughput: `12.112573 tok/s`
- Continuation throughput: `13.187379 tok/s`
- Output IDs: `3955, 361`
- Stage-profile buckets populated successfully.

Evidence:

`logs/from-cluster/sprint187-synthetic-prompt-profile/len256-profile-patched/synthetic-len256-profile.json`

### Preferred Profile

Patched len-1024 synthetic profile:

```bash
tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --synthetic-prompt-token 926 \
  --synthetic-prompt-len 1024 \
  --ctx 262144 \
  --tokens 2 \
  --profile-decode \
  --json
```

Result:

- Prompt replay: `71203.549 ms`
- Prompt throughput: `14.381306 tok/s`
- Continuation throughput: `14.282227 tok/s`
- Output IDs: `926, 926`
- Stage decode ms: `[10497.333, 9781.688, 9821.666, 9802.618, 9787.269, 8202.683, 8137.685, 5022.917]`
- Stage-profile total ms: `[9157.970, 9288.436, 9316.642, 9306.576, 9282.759, 7773.933, 7669.785, 4725.167]`
- Handoff sum: `219.707 ms`

Stage-profile bucket sums:

| Bucket | Sum ms | Share |
|---|---:|---:|
| HC attention prep | 3069.656 | 4.6% |
| Attention | 37779.266 | 56.8% |
| HC FFN prep | 3754.986 | 5.6% |
| FFN | 21473.437 | 32.3% |
| HC final | 443.923 | 0.7% |
| Total | 66521.268 | 100.0% |

Evidence:

`logs/from-cluster/sprint187-synthetic-prompt-profile/len1024-profile-patched/synthetic-len1024-profile.json`

## Outcome

The direct synthetic profile path is now usable for single-slot filled-context
benchmarks. At len-1024 / ctx-262144, attention dominates the measured stage
profile at roughly `56.8%`, followed by FFN at roughly `32.3%`. Handoff is only
`219.707 ms` summed across boundaries for the run, so the next filled-context
optimization should target attention/KV execution or a larger fused execution
boundary, not inter-stage transfer.
