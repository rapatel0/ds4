# Sprint 413: Reduced-Slot Semantic Serving Admission

## Goal

Find the highest practical slot count where the TP/EP true-attention plus
post-attention FFN-input serving path is operationally admitted at `256K`
context.

Sprint 411 and Sprint 412 proved that the semantic path can serve HTTP at the
target `32` requests / `32` slots / `256K` shape, but it is not
production-admitted:

- `32/32` HTTP responses served
- semantic timers active
- minimum free VRAM `1328 MiB`
- `62` NCCL reserve-threshold failures against the `1536 MiB` guard
- server generated decode about `21 tok/s`

The long-term target remains `32` slots / `256K`. The user explicitly allowed
reducing slots to improve performance, so this sprint measures reduced-slot
serving tiers instead of blocking all semantic progress on the full 32-slot
shape.

## Experiment

Run the semantic HTTP A/B harness with HC-current NCCL, post-attention FFN
input, route-plan async upload disabled, lazy output head, compact MoE decode,
and attention-output NCCL enabled on the candidate.

Primary matrix:

```text
ctx      = 262144
position = 262080
tokens   = 32

slot tiers:
  24 requests / 24 slots
  16 requests / 16 slots
  optional bracket tier if 24 passes or barely fails
```

Control:

- promoted HC-current NCCL fast path

Candidate:

- HC-current NCCL
- true-attention output
- post-attention FFN input
- route-plan async upload disabled
- attention-output NCCL allgather

## Definition of Done

- [x] V100 reduced-slot A/B completes for at least one tier.
- [x] Identify the highest tested tier that is readiness-clean, or the next
      concrete blocker if none are clean.
- [x] Record for each tier:
      - HTTP success count
      - readiness
      - server decode tok/s
      - client generated tok/s
      - minimum free VRAM
      - reserve failures
      - semantic timers
- [x] Update status/vision and commit kept artifacts.

## Decision Rule

If a reduced slot tier is readiness-clean and semantic timers are active,
declare it the current practical semantic-serving tier for further correctness
and quality testing.

If no reduced tier is readiness-clean, use the memory telemetry to decide
whether the next implementation should reduce slot-coupled KV/scratch state,
reduce output-head or HC-control residency, or change the attention-output TP
kernel/collective structure.

Do not promote 32-slot semantic serving until it is readiness-clean at `256K`.

## Implementation

Relaxed the TP/EP launcher admission guard from exactly `32` slots to
`DS4_V100_SLOTS<=32`. The executable and TP runtime already consume the
requested slot count; the shell guard was the only blocker to reduced-slot
HTTP serving tests.

The launcher still requires:

- `DS4_V100_CTX=262144`
- `DS4_V100_ACTIVE_MICROBATCH == DS4_V100_SLOTS`
- `DS4_V100_SLOTS<=32`
- no MTP in TP/EP serving mode

## Validation

Local checks:

```text
bash -n tools/ds4-v100-run-appliance.sh
DS4_V100_SERVE_MODE=tp-ep DS4_V100_CTX=262144 \
  DS4_V100_SLOTS=24 DS4_V100_ACTIVE_MICROBATCH=24 \
  DS4_V100_APPLIANCE_DIR=/tmp/missing \
  DS4_V100_TP_EP_CONTRACT=/tmp/missing.tsv \
  DS4_V100_TP_EP_TM_INDEX=/tmp/missing.tsv \
  DS4_V100_TURBOMIND_LIB=/tmp/missing.so \
  DS4_V100_TP_EP_TOKENIZER_MODEL=/tmp/missing.gguf \
  DS4_V100_REQUIRE_GPUS=0 \
  tools/ds4-v100-run-appliance.sh --check --allow-missing
```

V100 HTTP A/B artifacts:

```text
logs/from-cluster/sprint413-post-attn-slot24-http-ab-rerun/
logs/from-cluster/sprint413-post-attn-slot28-http-ab/
logs/from-cluster/sprint413-post-attn-slot30-http-ab/
```

Shape for every run:

```text
ctx      = 262144
position = 262080
tokens   = 32/request
control  = promoted HC-current NCCL fast path
candidate = post-attention FFN input + attention-output NCCL
```

| Slots | Case | HTTP | Ready | Server decode tok/s | Client generated tok/s | Min free VRAM | Post-close free | VRAM failures | Attention output ms | Post-attn ms |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 24 | control | `24/24` | `true` | `90.974366` | `14.944926659484418` | `3198 MiB` | `3332 MiB` | `0` | `0.0` | `0.0` |
| 24 | candidate | `24/24` | `true` | `19.716583` | `7.367808881719989` | `2428 MiB` | `2562 MiB` | `0` | `365.433075` | `115.793207` |
| 28 | control | `28/28` | `true` | `100.245595` | `15.250440667793258` | `2566 MiB` | `2700 MiB` | `0` | `0.0` | `0.0` |
| 28 | candidate | `28/28` | `true` | `20.624419` | `7.922564755228921` | `1790 MiB` | `1924 MiB` | `0` | `422.604024` | `128.822428` |
| 30 | control | `30/30` | `true` | `105.035348` | `16.197411761359966` | `2332 MiB` | `2466 MiB` | `0` | `0.0` | `0.0` |
| 30 | candidate | `30/30` | `true` | `21.089170` | `8.21212433262282` | `1556 MiB` | `1692 MiB` | `0` | `437.980246` | `130.118340` |

For comparison, the prior target `32` slot semantic candidate from Sprint 412
served `32/32` HTTP responses but failed readiness with `1328 MiB` minimum free
VRAM and `62` reserve failures.

## Outcome

Decision:
`reduced-slot-semantic-serving-operational`.

The semantic TP/EP serving path is operational at reduced slots and `256K`
context.

- Highest clean tier tested: `30` slots / `30` concurrent requests.
- Practical tier for follow-on quality/performance work: `28` slots, because
  it leaves `1790 MiB` minimum free VRAM against the `1536 MiB` NCCL reserve.
- `30` slots is technically clean but too close to the guard: `1556 MiB`
  minimum free VRAM, only `20 MiB` above reserve.
- `32` slots remains blocked by reserve failure until the semantic path
  reduces attention-output/post-attention memory or changes the TP collective
  structure.

This does not solve performance. The semantic path remains about `5x` slower
than the fast control at comparable slot counts, with attention-output and
post-attention work now measured as the dominant semantic overhead. The next
implementation should make the `28`-slot tier the default semantic test shape,
then optimize the true attention-output projection/allgather and post-attention
FFN input path before moving back up to `32` slots.
