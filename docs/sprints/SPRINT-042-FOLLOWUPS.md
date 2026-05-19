# Sprint 042 Follow-Ups

## P0: Production Deployment Package

The full gate now passes with `missing=production_deployment`. Next sprint
should make the appliance runnable as an operator-owned cluster service:

- supervised process or Kubernetes manifest;
- explicit config for model, MTP sidecar, pack index, context, slots, ports,
  GPU visibility, and reserve checks;
- health/status/metrics expectations;
- startup and restart behavior;
- log/artifact locations;
- rollback path to the base non-MTP service.

## P1: MTP Production State Object

The native verify path uses a tool-local MTP forward runner and a host hop for
`embed[4096]` plus post-commit HC. This is correct for the gate, but production
speculative serving should extract a resident MTP draft-session object with:

- persistent scratch tensors;
- persistent MTP raw cache visibility state;
- device-local committed embedding gather where practical;
- an explicit accept/reject transaction boundary.

## P1: Positive-Accept Fixture Set

The short fixture produced a positive MTP accept (`target_top1=1`,
`mtp_top1=1`). Add a small fixture set that covers:

- first-draft accept;
- first-draft reject;
- non-EOS accepted token if available;
- longer prompt context.

## P2: MTP Output Weight Duplication

The verify smoke uploads a separate gpu7 `output.weight` arena for the MTP
logits projection. It keeps more than 17 GB free after that upload on the
current cluster run, but production should avoid duplicate output-head residency
by sharing the scheduler-owned gpu7 output arena or adding a narrow runtime
binding.
