# Sprint 237 - Layer-2 Dense Coverage Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 236 proved descriptor-backed packed-F8 dense compute for one
representative tensor, `blk.2.attn_q_a.weight`, inside the TP/EP-only path.
Sprint 237 broadens that into a layer-2 dense coverage gate: the runtime should
discover every layer-2 F8 dense TP tensor, run the same packed-byte GPU compute
pattern for each compatible shape, and report per-tensor timings and oracle
checks.

This is still not serving and not a full DS4 layer. It is the step from "one
dense tensor can compute" to "the layer's F8 dense families are executable in
the TP/EP layout."

## Goals

- Keep the hard cut: no PP scheduler edits and no generic PP/TP abstraction.
- Extend the TP/EP full-layer smoke with an opt-in F8 dense coverage mode.
- Discover layer-2 `dense_tp` rows where `source_dtype=f8_e4m3_b128`.
- Group rows by tensor id and validate each group has eight TP shards.
- Run packed-F8 dense compute for compatible tensors using the Sprint 236
  kernel pattern.
- Report per-tensor:
  - tensor id;
  - rows/GPU;
  - columns;
  - packed bytes loaded;
  - compute ms;
  - repeat max_abs;
  - CPU oracle max_abs;
  - pass/fail.
- Preserve Sprint 235 full-layer scaffold checks.
- Preserve Sprint 236 single-tensor mode for focused debugging.
- Run on the V100 pod at `32` slots / `256K`, MTP off.

## Non-Goals

- No PP/layer-split work.
- No changes to `ds4_v100_scheduler.*`.
- No BF16 dense compute yet.
- No routed expert changes.
- No full attention graph.
- No logits equivalence claim.
- No serving integration.
- No MTP.
- No HMMA/CUTLASS optimization yet.

## Architecture

Extend the separate TP/EP full-layer smoke:

```text
tools/ds4-v100-tp-ep-full-layer-smoke.cu
  --dense-compute-tensor NAME      # Sprint 236 focused gate
  --dense-compute-all-f8           # Sprint 237 coverage gate
```

The coverage mode should reuse the same packed-F8 loader and kernel. It should
skip incompatible rows explicitly and report why, rather than silently claiming
coverage.

## Implementation

1. Add `--dense-compute-all-f8`.
2. Factor the Sprint 236 dense compute helper so it can run one tensor or a
   list of tensors.
3. Discover unique layer-2 F8 dense tensor ids from the parsed contract.
4. Run the dense compute gate for every compatible tensor.
5. Emit one `dense_compute_tensor` line per tensor.
6. Emit aggregate dense coverage summary:
   - discovered tensors;
   - executed tensors;
   - skipped tensors;
   - total packed bytes loaded;
   - worst compute ms;
   - worst oracle max_abs;
   - total failures.
7. Keep the final `tp_ep_full_layer_scaffold` PASS dependent on the aggregate
   dense coverage result when coverage mode is enabled.
8. Build and run on the V100 pod.
9. Copy evidence to
   `logs/from-cluster/sprint237-tp-dense-coverage/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | dense coverage mode |
| `docs/sprints/SPRINT-237.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint237-tp-dense-coverage/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Dense coverage mode is implemented only in the separate TP/EP tool.
- [x] No PP scheduler files are modified.
- [x] Tool discovers all layer-2 F8 dense TP tensor groups from the real
      contract.
- [x] Tool executes every compatible F8 dense tensor group.
- [x] Tool reports skipped tensors explicitly, if any.
- [x] Each executed tensor passes finite repeat checks.
- [x] Each executed tensor passes bounded CPU oracle comparison.
- [x] Existing full-layer scaffold, KV, and EP checks still pass.
- [x] Evidence is copied to
      `logs/from-cluster/sprint237-tp-dense-coverage/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Pack: `/workspace/packs/ds4-appliance-full-tm-gated-s181`

Contract:
`/workspace/logs/sprint228-tp-ep-pack-contract/contract/tp-ep-pack-contract.tsv`

Command shape:

- `32` slots;
- `256K` context;
- `top_k=6`;
- layer `2`;
- `--dense-compute-all-f8`;
- MTP off.

Executed F8 dense tensor groups:

| Tensor | Rows/GPU | Cols | Packed bytes | Compute ms | Oracle max_abs | Result |
|---|---:|---:|---:|---:|---:|---|
| `blk.2.attn_kv_latent.weight` | `64` | `4096` | `2113536` | `0.050893` | `0.000000007` | `PASS` |
| `blk.2.attn_output_a.weight` | `1024` | `4096` | `33816576` | `0.566170` | `0.000000007` | `PASS` |
| `blk.2.attn_output_b.weight` | `512` | `8192` | `33816576` | `0.556032` | `0.000000015` | `PASS` |
| `blk.2.attn_q_a.weight` | `128` | `4096` | `4227072` | `0.082227` | `0.000000007` | `PASS` |
| `blk.2.attn_q_b.weight` | `4096` | `1024` | `33816576` | `0.654029` | `0.000000007` | `PASS` |
| `blk.2.ffn_down_shexp.weight` | `512` | `2048` | `8454144` | `0.155238` | `0.000000002` | `PASS` |
| `blk.2.ffn_gate_shexp.weight` | `256` | `4096` | `8454144` | `0.149197` | `0.000000007` | `PASS` |
| `blk.2.ffn_up_shexp.weight` | `256` | `4096` | `8454144` | `0.149299` | `0.000000005` | `PASS` |
| `blk.2.indexer.attn_q_b.weight` | `1024` | `1024` | `8454144` | `0.169267` | `0.000000015` | `PASS` |

Aggregate dense coverage:

| Metric | Value |
|---|---:|
| compatible F8 tensor groups | `9` |
| executed F8 tensor groups | `9` |
| skipped F8 tensor groups | `0` |
| total packed bytes loaded | `141606912` |
| worst dense compute ms | `0.654029` |
| repeat max_abs | `0.000000000` |
| repeat bad/nan | `0 / 0` |
| worst CPU oracle max_abs | `0.000000015` |
| CPU oracle bad | `0` |

Full scaffold result remained passing:

| Metric | Value |
|---|---:|
| total layer rows | `288` |
| dense rows | `112` |
| control rows | `136` |
| expert rows | `16` |
| KV rows | `16` |
| comp rows | `8` |
| KV max_abs | `0.000000000` |
| worst EP ms | `0.241766` |
| result | `PASS` |

## Risks

- Some layer-2 F8 tensors may have very large local row counts. If runtime is
  too long for the coverage smoke, the tool should still report tensor-level
  progress and make the cap explicit rather than hiding the gap.
- The current dense kernel is a correctness gate, not final throughput. If
  coverage passes, the next optimization step should switch selected high-cost
  dense families to HMMA/CUTLASS-style kernels.
- BF16 compressor/indexer tensors remain outside this sprint.

## Decision

Complete. The TP/EP-only full-layer smoke now supports `--dense-compute-all-f8`
and executes every compatible layer-2 F8 dense TP tensor group from the real
contract. All nine groups pass finite repeat and bounded CPU oracle checks, and
the full descriptor scaffold, KV gate, and descriptor-backed EP path still pass.

This still excludes BF16 dense/control math and does not connect the tensor
outputs into a real next hidden state. The next sprint should either add BF16
compute coverage for compressor/indexer tensors or start composing the dense
outputs into the representative layer dataflow.
