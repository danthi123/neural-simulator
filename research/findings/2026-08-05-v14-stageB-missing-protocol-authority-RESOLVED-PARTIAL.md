---
status: qualified
type: finding
date: 2026-08-05
---

# V14 Stage B missing protocol authority: RESOLVED-PARTIAL

## Scope

Four unresolved measurements kept the two authoritative NumPy survivors from a
complete intrinsic-lesion verdict. This audit re-read the primary sources,
separated source facts from simulator analysis choices, and implemented only the
measurement that existing traces can support without inventing a biological
protocol.

## Primary-source decisions

- **NaP voltage:** Ding, Wei, and Zhou report that 10 uM riluzole, applied after a
  stable baseline near -50 mV, hyperpolarized SNr GABA neurons by 11.9 +/- 1.1 mV
  and stopped spontaneous firing. They do not report the baseline duration,
  stability criterion, analysis window, or per-cell voltage estimator. A
  source-exact voltage statistic remains unavailable. A future same-cell phased
  directional assay may use project-defined windows, but must not claim
  equivalence between `gNaP=0` and riluzole. Primary source: [Ding et al.
  2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3234097/#sec19).
- **HCN input resistance:** Atherton and Bevan report that HCN blockade increases
  steady-state input resistance below approximately -60 mV. The current-clamp
  pulse duration, current increments, averaging windows, and resistance formula
  are not reported. Their one-second protocol belongs to a separate voltage-clamp
  experiment. A source-exact current-clamp contract remains unavailable. A
  project-operational paired current family can test the direction if TTX, HCN,
  current conversion, and fitting choices are all filed explicitly. Primary
  source: [Atherton and Bevan 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC6725542/#sec8).
- **SK depolarization block:** the source genuinely reports 4 of 12 cells entering
  depolarization block within ten minutes of 100 nM apamin. It does not define a
  detector voltage, silent duration, or exact onset rule. Twelve copies of one
  deterministic cell are not a biological cohort. This gate remains unavailable
  until twelve independently justified held-out cell parameterizations and a
  project-operational detector are preregistered.
- **Post-spike AHP:** the source supports loss of the post-spike AHP under SK and
  Cav2.2 blockade, but does not define a medium-AHP alignment window, reference
  voltage, or numerical amplitude. It does report a whole-AHP minimum of -68.49
  +/- 4.93 mV. The V2 contract therefore replaces the misleading
  `medium_ahp_depth_mV` name with a transparent directional total-AHP assay: the
  median discrete voltage nadir across all 100 complete interspike intervals in
  each 101-spike trace. This is a project analysis convention, not a
  source-measured medium-AHP amplitude.

## Implementation

- Preserved the original V1 causal and intrinsic protocol files byte-for-byte so
  completed campaign artifacts remain replayable.
- Added versioned V2 causal and intrinsic protocol files with the total-AHP nadir
  assay and explicit non-equivalence boundaries.
- Added a pure metric that measures every complete half-open interspike interval.
- The independent scorer recomputes the metric from authenticated raw or compact
  traces and never accepts runner-supplied aggregates.
- Runner, readiness controller, and scorer accept both explicitly authorized V1
  and V2 contracts. V1 behavior remains unchanged.

## Verification

Focused regression: **106 passed, 1 skipped** across physiology metrics, causal <!--derived-->
contracts, scorer, runner, readiness, batched execution, confirmation, and screen
aggregation tests. The skip is the existing optional-environment test.

Filed artifacts:

- `research/specs/v14_snr_stageB_causal_gates_v2.json`
- `research/specs/v14_snr_stageB_intrinsic_protocol_v2.json`

The two preregistered survivors were then rerun independently on authoritative
NumPy workers at revision `99cc46c72f32a2b3dcfceed8b11843370eff4efd` and
collected under digest-bound V2 job plan
`research/experiment-runtime/v14-stageB-v2-total-ahp/job-plan.json`. Both
collections passed artifact-manifest verification and independent local score
recomputation. The verifier permits only machine-scale float drift across NumPy
versions; structure, text, booleans, integer values, and material numerical
changes remain exact or fail closed.

V2 total-AHP observations:

| Candidate | Intact nadir (mV) | Cav2.2 lesion (mV) | SK lesion (mV) | Direction |
|---|---:|---:|---:|---|
| Sobol 284 | -78.317493 | -74.989601 | -78.297482 | both pass |
| Sobol 404 | -80.310486 | -77.033806 | -79.314903 | both pass |

The authenticated scores remain `UNAVAILABLE`, not failed: NaP voltage, HCN
hyperpolarized input resistance, and the heterogeneous 12-cell SK cohort were
not executable under V2. Evidence is in
`research/experiment-runtime/v14-stageB-v2-total-ahp/results/`.

## Result and next action

Status is **RESOLVED-PARTIAL**. The total-AHP contract is implemented and passed
by both survivors. NaP and HCN now have filed V3 project-operational contracts and
are being connected to separately authenticated companion traces. The SK cohort
must remain unavailable. Next, execute and independently score the V3 phased NaP
and paired HCN assays for both survivors.
