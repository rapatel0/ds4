# SPRINT-025 Followups

## Critical

1. **Localize selected-token divergence**

   Add CPU-vs-V100 checkpoints after selected layer boundaries for the
   `short_reasoning_plain` prompt replay. Start with stage outputs after layers
   5, 11, 17, 23, 29, 34, 39, and 42.

2. **Top-k diagnostic output**

   Extend `cuda_v100_selected_token_smoke` to print top-5 token ids, token
   bytes, and logits. This will show whether the official token is close or
   absent.

3. **Output-head CPU parity microtest**

   Feed a known HC vector into both CPU `output_logits_one_decode_scratch` and
   `ds4_v100_stage_scheduler_select_token` to prove the output-head adapter in
   isolation.

## High

4. **Prompt replay counters**

   Report token count, per-token decode time, and per-stage decode time so the
   correctness fixture doubles as an early latency profile.

5. **Failure-preserving logs**

   When `--expected-token-hex` fails, keep the default exit failure but write a
   compact JSON diagnostic artifact with selected token, expected token, top-k,
   prompt token ids, and layer/stage metadata.

## Deferred

6. **Serving, MTP, throughput**

   Keep deferred until selected-token correctness is localized and fixed.
