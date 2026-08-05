---
status: live
type: finding
lane: gateb-v14-source-constrained-identification
date: 2026-08-05
---

# Kinetic identification readiness: target waveforms still required

The source-constrained search engine is not yet scientifically executable.
The repository has sealed Ding voltage commands and 18 scalar endpoints, but it
does not yet contain digitized population command-response curves. The papers
do not publish population mean current-versus-time waveforms or raw recordings.
Generating target traces from retired gate equations, or expanding fitted
Boltzmann and exponential summaries into synthetic traces, would train the new
models to reproduce old assumptions rather than measured currents.

The prospective command split is now sealed in
`research/specs/v14_snr_stageB_kinetic_identification_partition_v1.json`.
Calibration, validation, and one-shot held-out commands are disjoint. The most
diagnostic commands remain held out: sodium behavior at 0 mV, sodium
deactivation at -40 mV, Kv3 rise at +40 mV, and Kv3 deactivation at -50 mV.
Opening those values before a candidate and its analysis are sealed is
prohibited.

## Engine progress

- `tools/adaptive_design_update.py` now turns authenticated completed
  observations into an immutable next design version and emits a lineage
  receipt. This removes manual copying from the propose-run-ingest cycle.
- `sim/kv3_source_models.py` now accepts strict parameter documents for the
  published Labro and Desai constants. Exact defaults preserve the unmodified
  source comparators, and the two graphs remain separate.
- Existing proposal, experiment sealing, durable CPU/GPU/mini-PC execution,
  receipt verification, and observation compilation remain reusable.

## Required target-data gate

Each command admitted to fitting must bind numeric population samples or an
explicit source summary, units, normalization, primary-source locator,
acquisition or digitization method, uncertainty, and the immutable source asset
digest. Representative single-cell traces cannot silently become
population-mean targets. They may serve only as qualitative or secondary shape
evidence. Where the paper reports only a fitted curve or summary statistic,
that limitation must remain explicit and the objective must be scored as such.

The earlier requirement to fit complete empirical population waveforms is not
achievable from the published record and is therefore corrected. The campaign
will fit available population activation, inactivation, recovery, and
deactivation command-response curves plus independent scalar kinetics. It will
use identifiability diagnostics to preserve unresolved microscopic parameters
rather than claiming that sparse outputs uniquely determine them.

## Exact next action

The official figure assets and their hashes are now recorded in
`research/specs/v14_snr_stageB_primary_figure_asset_manifest_v1.json`. Next,
digitize the population command-response panels with an explicit pixel error
model and determine which partitions have defensible numeric targets. Do not
treat representative single-cell traces as population means. In parallel,
finish the resumable campaign supervisor; neither task authorizes execution
before the target-data gate passes.

Stage 2 integration remains unauthorized.
