# SPRINT-024 Followups

## Critical

1. **Output-head selected-token gate**

   Add a gpu7 output-head adapter that consumes the final scheduler HC,
   collapses it through `hc_head_*`, applies `output_norm.weight`, runs BF16
   `output.weight`, selects top-1, and compares against the source oracle.

2. **Failure-local full-chain reporting**

   The full-chain smoke reports aggregate success well. If a later stage fails,
   report the stage id, layer range, gpu, uploaded bytes, and last successful
   layer before returning.

## High

3. **Upload timing and resident memory report**

   Capture per-stage upload time and per-GPU resident bytes in the full-chain
   smoke output so regressions are visible without `nvidia-smi` polling.

4. **Relay optimization**

   Replace synchronous peer copies with the relay stream/buffer contract after
   selected-token correctness lands.

## Deferred

5. **MTP**

   Keep deferred until the selected-token path passes.

6. **Throughput**

   Measure after selected-token correctness and after we have enough timing
   counters to distinguish upload, stage execution, and relay costs.
