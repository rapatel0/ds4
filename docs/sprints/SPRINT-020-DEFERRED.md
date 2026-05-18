# SPRINT-020 Deferred Items

## Full 43-Layer Selected-Token Decode

Sprint 020 targets one representative layer with compressor/indexer and HC
scheduling. Full selected-token correctness remains deferred until the scheduler
can walk all layer classes and reach output-head logits.

## Public Serving

Serving remains deferred until selected-token correctness exists from the V100
layer-scheduled path.

## MTP

MTP remains deferred until base decode is correct and measurable.

## Throughput Optimization

Slot batching, wavefront scheduling, timing-driven kernel selection, and MTP
performance work remain deferred until the base HC layer scheduler is correct.

## Tensor Parallelism

Tensor-parallel output head or FFN splits remain evaluation candidates only.

