---
type: finding
status: partial
date: 2026-09-02
mechanism: confidence-forthcomingness (board #94) — the metacog confidence read that gates how much a rich
  KB-relation turn volunteers; made SCALE-INVARIANT so the #108 100k default flip is unblocked
lane: introspection
seeds: [42]
seed-waiver: numpy CPU SMOKE / de-risk on seed 42 at BOTH vocab scales (the shipped 15k core AND the board-#108
  100k bundle); the gating 6-seed cupy re-verify (42/43/44/100/101/102, both scales) is QUEUED on gpu_queue, not
  yet harvested — this is a PARTIAL-pending-soak, not a full GO
artifacts:
  - research/findings/raw/_confidence_100k_recalib/diagnose_15k_seed42.json
  - research/findings/raw/_confidence_100k_recalib/diagnose_100k_seed42.json
  - research/findings/raw/_confidence_100k_recalib/arms_15k_seed42.json
  - research/findings/raw/_confidence_100k_recalib/smoke_100k_seed42.json
  - research/findings/raw/_confidence_100k_recalib/smoke_15k_noregress_seed42.json
external: the cause is an internal extreme-value-statistics artifact of the composer's own runner-up margin,
  diagnosed by direct measurement of the candidate-score distribution at both vocab scales; corroborated by the
  retrieval/classification margin-calibration literature (cited in the External corroboration section below), and
  the runner-up's order-statistic growth is standard Gumbel extreme-value theory. NO NEW biological mechanism is
  missing — the fix is a within-substrate normalization choice
---

# Confidence-forthcomingness made SCALE-INVARIANT: `margin_norm` keyed on an order statistic that inflates with
# codebook size; the winner-vs-bulk z-score is the scale-invariant decisiveness read (numpy smoke fixes 100k,
# preserves 15k; 6-seed cupy queued)

**Verdict: PARTIAL-pending-soak.** numpy CPU smoke (seed 42) at BOTH scales passes all 8 checks; the gating
6-seed cupy re-verify is queued, not harvested. The #108 100k `confidence-forthcomingness` NO-GO is a de-risked
lead, not yet a landed GO.

**Artifacts** (all under `research/findings/raw/_confidence_100k_recalib/`): the fix is measured end-to-end by
`research/findings/raw/_confidence_100k_recalib/smoke_100k_seed42.json` (100k, now `vary_lesion_all_GO: true`) and
`research/findings/raw/_confidence_100k_recalib/smoke_15k_noregress_seed42.json` (15k, still `vary_lesion_all_GO:
true` — no regression). The diagnosis is `research/findings/raw/_confidence_100k_recalib/diagnose_15k_seed42.json`
+ `research/findings/raw/_confidence_100k_recalib/diagnose_100k_seed42.json` (per-role candidate-score
distributions) and `research/findings/raw/_confidence_100k_recalib/arms_15k_seed42.json` (the clean/degraded/lesion
winner-z reference used to anchor the map).

## The residual (board #108 R3, `research/FAILURE_LOG.md` newest row)

`confidence-forthcomingness` (#94) was validated only against the shipped `wikidata_core_15k` LTM. Against the
board-#108 `wikidata_100k` bundle the R3 re-verify read `measurement_all_GO: true` but `vary_lesion_all_GO: false`:
a CLEAN correct 100k recall's `mean_role_conf` never crossed the HIGH-confidence floor, so `confidence_forthcoming.
granted` stayed False on BOTH the clean and the degraded arm, the bonus never fired, and the metacog lesion had
nothing to collapse (a vacuous vary/lesion). The confidence read still DISCRIMINATED (clean mrc > degraded mrc),
but it sat below the operating point calibrated against the 15k core.

## Diagnosis — VERIFY-FIRST: is `margin_norm` scale-invariant? NO. It keys on an order statistic.

<!--derived-->
(The values in this section are ROUNDED per-role aggregates / ranges and hand-computed extreme-value theory,
derived from the full-precision per-role candidate-score distributions in the cited
`diagnose_15k_seed42.json` + `diagnose_100k_seed42.json` + `arms_15k_seed42.json`; the raw values live there.)

The read chain is `composer trace -> mean_role_conf -> spiking metacog WTA margin -> confident`. The scale problem
is entirely in the FIRST step, the composer's decisiveness scalar. `mean_role_confidence` prefers each role chip's
`margin_norm` = `(top - runner_up) / top` (`RFPhasorComposer._cleanup_all_score_stats`). It was INTENDED as a
peak-relative, scale-invariant ratio. It is not. Measuring the SAME clean correct recall
(`asimov_isaac employer university_of_boston`, present verbatim in both bundles, seed 42, through the real
`/api/brain-chat` handler; both bundles share D=128, vocab 7,032 -> 23,914):

| scale | vocab V | winner cosine (top) | runner-up | non-winner mean / std | p99 noise | **margin_norm** | **winner_z** |
|-------|---------|---------------------|-----------|-----------------------|-----------|-----------------|--------------|
| 15k   | 7,032   | 0.42–0.47           | 0.21–0.23 | ~0.000 / 0.062        | 0.144     | **0.497**       | **7.24**     |
| 100k  | 23,914  | 0.40–0.50           | 0.24–0.29 | ~0.000 / 0.062        | 0.144     | **0.395**       | **7.03**     |

The winner's cosine and the entire NON-WINNER BULK (mean ~0, std ~0.062 = ~1/sqrt(D), p99 0.144) are IDENTICAL
across the two scales — the recall is exactly as decisive. Only the single runner-up inflates: it is the MAX over
the V-1 non-winner candidates, an order statistic whose expectation grows as ~sqrt(2 ln V) (0.062·sqrt(2·ln 7032)
= 0.261, 0.062·sqrt(2·ln 23914) = 0.278). That inflation alone drags `margin_norm = 1 - runner/top` down from
0.497 to 0.395. The winner-vs-BULK z-score `(top - mean_nonwin)/std_nonwin` — dominance over the STABLE noise
floor — is scale-INVARIANT: **7.24 (15k) == 7.03 (100k)**. So this is the task's "should be scale-invariant but
isn't" case, NOT a legitimate operating-point shift: at 100k the winner still sits ~7 SD above the field.

## The fix (additive, guarded, scale-aware) — one operating point for all vocab scales

1. `RFPhasorComposer._cleanup_all_score_stats` (+`margin_snr`, ADDITIVE): each role chip now also carries the
   winner-vs-bulk z-score `(top_raw - mean_nonwin)/(std_nonwin + eps)`. `margin`/`margin_norm`/`confidence`/
   `conflict` are byte-identical (self_schema_honesty.py and the tests read `margin` unchanged).
2. `metacog_production_organ.mean_role_confidence` prefers `margin_snr` over `margin_norm` when a chip carries it,
   mapping the z-score LINEARLY onto the SAME `ROLE_CONF_LO`/`ROLE_CONF_HI` band via two universal anchors
   `SNR_LO=5.158` / `SNR_HI=7.273` (derived so the map reproduces the shipped 15k operating point <!--derived-->).
   The anchors are the 15k reference DEGRADED and CLEAN winner-z (degraded z=5.329
   <-> the shipped margin_norm 0.3161; clean z=7.237 <-> 0.4966; `arms_15k_seed42.json`), so the map reproduces the
   shipped 15k operating point at BOTH reference arms while being invariant to codebook size. These are a single
   universal read (same at 15k / 100k / 500k / 1M), NOT a per-bundle constant, and the metacog band itself is
   UNCHANGED. `OneBrainComposer` buffer chips never populate `margin_snr`, so the tiny-demo path falls through to
   `margin_norm`/`margin` byte-identically — only an LTM-sourced (`RFPhasorComposer`/`ShardedPhasorStore`) trace is
   remapped.

The lesion stays load-bearing untouched: the metacog lesion removes the evidence differential in the spiking WTA
(downstream of the read), so a would-be-confident answer still collapses regardless of the evidence scalar.

## numpy smoke (seed 42, SIM_BACKEND=numpy, both scales) — the de-risk

| bundle | arm | confident | mean_role_conf | n_sentences | verdict |
|--------|-----|-----------|----------------|-------------|---------|
| **100k** | clean | **True** | 0.4769 | 5 | was margin_norm 0.395 <!--derived--> / not-confident -> now confident |
| 100k | degraded | False | 0.343 | 4 | real match (not abstain), correctly not-confident |
| 100k | lesion | False | 0.4769 | 4 | spiking lesion collapses the confident read |
| **15k** | clean | True | 0.4966 | 5 | vs shipped 0.49655 — no regression (delta < 1e-4) |
| 15k | degraded | False | 0.3161 | 4 | matches shipped 0.31610 |
| 15k | lesion | False | — | 4 | collapses |

100k: `vary_lesion_all_GO: true` (all 8 checks) — the fix. 15k: `vary_lesion_all_GO: true` — the shipped #94
behavior is preserved (mean role-conf reproduced to < 1e-4). A fix that broke 15k would be worse than the residual;
it does not.

## Honest residual / what is NOT yet closed

* This is a numpy CPU SMOKE on seed 42 only. The GATE is the 6-seed cupy re-verify (both scales), QUEUED on
  gpu_queue (`research/findings/raw/_confidence_100k_recalib/cupy_6seed_verify.sh`, self-guarded on branch
  presence). Until harvested with `vary_lesion_all_GO: true` at 100k AND 15k, #94 stays PARTIAL and the #108
  100k-default flip stays blocked.
* The degraded arm's margin narrowed slightly (100k degraded z ~5.6 -> mrc 0.343 -> evidence 0.215 <!--derived-->;
  still clearly not-confident, flip needs evidence ~0.5). The winner-z read is a genuine competition read but is less
  sensitive to degradation than margin_norm was (noise damage shows more in the runner-up than in the winner-vs-
  bulk separation). The 6-seed cupy soak, with 6 independent noise draws, is what tests that this separation holds.
* The anchors `SNR_LO`/`SNR_HI` were fit to the single asimov reference recall (the 15k CLEAN recall is
  deterministic — mrc 0.49655 on every shipped seed — so its z anchor is robust; the DEGRADED anchor is the one
  seed-42 draw). Broader-traffic robustness is the soak's job.
* Not re-tested at 500k / 1M (bundles exist). The read is scale-invariant BY CONSTRUCTION (z over a stable noise
  floor), so those are expected to hold, but they are unmeasured.

## External corroboration

<!--derived-->
The diagnosis was reached by direct measurement, not from the literature, but it is corroborated by the
retrieval/classification margin-calibration literature (logged in `research/queue/.external_searches.jsonl`,
lane `introspection`):

* arXiv:2503.09218 (N2C2, 2025) — nearest-neighbor confidence calibration for retrieval classifiers; on-topic
  for a decode margin that shifts with datastore/vocabulary size.
* arXiv:1903.09215 (Gomez et al., 2019) — the top-1/top-2 score margin as a standard confidence measure (the
  exact family `margin_norm` belongs to).

The specific mechanism — the runner-up being the max over V-1 candidates, whose expectation grows as
sigma*sqrt(2 ln V) — is standard order-statistics / Gumbel extreme-value theory; the winner-vs-bulk z-score is
the distribution-relative decisiveness read that removes that V-dependence.
