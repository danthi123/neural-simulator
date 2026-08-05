---
type: research-finding
status: engineering-substrate
date: 2026-08-04
mechanism: v14-snr-stageB-raw-scorer-readiness-transport
claim_check: synthesis
---

# V14 Stage B raw scorer and readiness transport

## Boundary

This is a seed-free engineering result, not an SNr physiology result. Synthetic
traces exercised source-bound scoring and candidate transport without running a
simulated neuron, constructing an executable parameter packet, or opening any
calibration, replication, or held-out seed. Stage B remains
`READINESS_UNDEFINED`.

## Implemented

- The scorer loads the fixture and target packets only through exact SHA-256
  declarations and recomputes observations from raw spike or conductance traces.
- Uncropped recording and burn-in declarations are mandatory. Claimed summary
  values are ignored.
- Source-derived bounded fixtures are scored independently. The HCN
  non-significance observation remains an unscored boundary rather than an
  invented equivalence interval.
- A valid out-of-band result is `process_status: completed` with
  `scientific_verdict: NO_GO`. Malformed traces, wrong digests, missing fixtures,
  or candidate-identity failures produce an infrastructure error and no score
  artifact.
- The readiness adapter binds two distinct candidate documents to isolated
  artifact directories under the same `numpy` / `readiness` cell. It selects no
  seed and does not use the reserved scientific harness.

## Receipt

The committed top-level receipt is
`research/findings/raw/v14_stageB_readiness_transport_20260804/readiness-dry-run.json`.
Its SHA-256 is
`7e304fd89f33ec19a62775f5bd25cdc38c4b31df4e5a725d8afea3a683a56f1b`.
Candidate digests are
`6c3a0dbe4f458bd37fcef5c7872b67e136bd8cc27da30cf7e1cb4b0091146a46`
and
`266f249e55c8e3beea5f916f6f0f57a4eccff8bc5d2fbb1865452b318ae8dd5f`.
The first synthetic profile exercised an in-band result. The second exercised
an out-of-band result that completed successfully as synthetic `NO_GO`.

<!--derived-->
The focused readiness, scorer, fixture, and metric suite passed `54` tests.

## Open gates

- Extend the scorer with source-authorized causal lesion, depolarization-block,
  rate-only compensation, release, and checkpoint-continuation controls.
- Resolve the remaining passive-property, HCN-equivalence, calcium/SK-prior,
  and transferred fast-spike authority decisions.
- Build candidate-specific authenticated packet compilation and adjudication.
- Run a real packet-backed physiology runner through the same identity and
  failure boundary.
- Source-seal the executable Stage B specification before opening calibration
  seed `590297`.
