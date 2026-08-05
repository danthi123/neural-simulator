---
type: finding
status: negative
date: 2026-08-04
mechanism: neural-vocal-action-credit-v14-stageB-event-count
---

# V14 Stage B source-bound endpoint execution

## Result

The Stage B engine now runs an authority-pinned production protocol rather than a short transport trace. Intact, Cav2.2-lesion, SK-lesion, and baseline-HCN-lesion arms record through 101 spontaneous spikes or a separately labelled 20-second operational timeout. The Nap-lesion arm records exactly one second. Raw traces are digest-bound and scored without caller-supplied aggregates.

The low endpoint is `UNAVAILABLE`: its four event-count arms produced zero spikes and reached the operational timeout. Its Nap lesion also produced zero spikes and passed the filed silence subgate. A timeout is not a physiology failure.

The high endpoint is `FAIL`: Nap removal left 73 spikes in one second, and Cav2.2 removal reduced ISI CV from `0.00895` to `0.00395` instead of increasing it. SK removal increased ISI CV from `0.00895` to `0.01031`; HCN removal preserved firing and changed baseline rate by about `0.84%`. Those positive subgates do not rescue the failed candidate (`research/findings/raw/v14_stageB_intrinsic_production_high_20260804/35850470eea7034c17119dbabefc78f8fd2ad395df6264e5cf68522c89464dec/intrinsic-lesion-score.json`).

No global scientific verdict is issued. Medium-AHP depth, HCN hyperpolarized input resistance, the 12-cell SK depolarization-block cohort, and Nap mean-voltage change remain unavailable because their source-faithful analysis protocols are incomplete.

## Evidence

- Source revision: `aff9fa7bb793a200759270f232372359f2f4c892`
- Protocol: `research/specs/v14_snr_stageB_intrinsic_protocol.json`, SHA-256 `9d808e9025f2bf731880e2bcf0975548f945a29ae5e820646e9297bb34e9588b`
- Low receipt: `research/findings/raw/v14_stageB_intrinsic_production_low_20260804/readiness-receipt.json`
- High receipt: `research/findings/raw/v14_stageB_intrinsic_production_high_20260804/readiness-receipt.json`
- Low execution: one 20,000-step fixed-duration arm plus four 400,000-step timeout traces; about 204 seconds wall time.
- High execution: all four event-count arms reached 101 spikes in 20,594-24,825 steps; about 17 seconds wall time.
- Backend: NumPy CPU reference path; no scientific seed and no stochastic runtime feature.

Atherton and Bevan report analysis over trains of 101 spontaneous action potentials. The exact rate formula `100 / (t101 - t1)` and population-CV convention are explicitly project analysis conventions, not source-reported formulas. The 20-second timeout is a resource bound only.

## Engineering consequence

The engine can now reject or retain candidate regions using resolved causal subgates without manual trace measurement. The next high-value engine slice is batched candidate screening with compact raw-trace storage and preregistered stop rules. A single neuron underuses the RTX 3090; GPU value comes from running many authenticated candidate/lesion trajectories concurrently, not from moving one Python-stepped neuron to CUDA.

Candidate search may use only resolved readiness subgates. It cannot promote a candidate until the missing AHP, HCN current-step, SK cohort, Nap voltage, held-out sensitivity, and later multi-seed controls are implemented.
