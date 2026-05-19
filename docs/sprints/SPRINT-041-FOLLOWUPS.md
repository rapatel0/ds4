# Sprint 041 Follow-Ups

## P0: Native Prompt-Token MTP Verify

The current `mtp_rollback` gate proves target rollback and synthetic MTP raw
visibility, but it does not produce a real draft token from the prompt-token
MTP path.

Next sprint should:

- read the committed target token embedding from the base model;
- read the target scheduler HC state after committing that token;
- feed both into the resident MTP forward path;
- compare real MTP draft top-1 against target top-1 using exact token equality;
- cover both accept and reject state transitions where possible.

## P1: Production Snapshot Boundary

The snapshot API is intentionally correctness-oriented and host-backed. Before
serving with speculative decoding, extract a lighter production transaction
boundary that only copies the mutable state needed for the active draft depth.

## P1: Compressor Rollback Stress

The scheduler snapshot smoke now covers ratio-4 compressed emissions and
indexer top-k visibility with eight decode positions. Add a longer targeted
variant for ratio-128 compressed emissions without putting it on every full
gate run.

## P2: Naming Cleanup

The binary remains `tools/ds4-v100-mtp-verify-smoke` while the gate label is
`mtp_rollback`. Rename or split once native prompt-token verify lands so
tool names match readiness semantics.
