---
type: finding
status: go
date: 2026-09-22
mechanism: metacog graded confidence marker — SPECULATE/HEDGE/ASSERT 3-band ladder self-calibrated on the
  workspace's own nmda_norm divisive-normalized NMDA-conductance balance margin (`metacog_production_organ.
  nmda_norm_margin`), measurement de-risk BEFORE any production wire-in
lane: introspection-self-model
backlog: §8 honesty=STATE-fidelity, Sub-arc B FIRST BUILD (workflow wg2byo6dg)
backend: numpy
runner: research/runners/_metacog_graded_marker_derisk.py
seeds: 42, 43, 44, 100, 101, 102
artifacts:
  - research/findings/raw/_metacog_graded_marker_derisk_6seed.json
  - research/findings/raw/_metacog_graded_marker_derisk_6seed.json.prov.json
  - research/findings/raw/_metacog_graded_marker_derisk/s42_intact_a.json
  - research/findings/raw/_metacog_graded_marker_derisk/s42_intact_b.json
  - research/findings/raw/_metacog_graded_marker_derisk/s42_lesion.json
---

# Softening the binary confidence hedge to a 3-band ladder (SPECULATE/HEDGE/ASSERT) is a GO on the workspace's own spiking margin, 6/6 seeds

**Verdict: GO** <!--derived--> (`research/findings/raw/_metacog_graded_marker_derisk_6seed.json`'s top-level
`verdict`/`preconditions` block, reproduced in full below). On all 6 pre-registered seeds (42/43/44/100/101/102),
the metacog workspace's own divisive-normalized NMDA-conductance balance margin (`nmda_norm_margin`, unchanged,
reuse-by-import) resolves **three** ordinal confidence bands — not just the two the production organ ships
today — self-calibrated from a synthetic LOW/MED/HIGH evidence battery with **no hand-set thresholds**. This is a
**measurement de-risk only**: nothing here touches `webapp/server.py` or `judge()`'s call sites; the ladder lives
entirely in the new runner, and `metacog_production_organ.py` is verified byte-identical to its pre-existing
baseline (git hash unchanged). Per [`docs/TERMS.md`](../docs/TERMS.md) this is a **de-risk**, not an
**integrated** faculty — the production wire-in is the explicitly-deferred next rung.

## Why this was a genuinely open question, not a re-derivation of a banked result

`bash tools/before_you_build.sh "metacog graded confidence marker binary to 3-band ladder"` was run before
writing a line of this runner (log: `research/queue/.corpus_checks.jsonl`). It surfaced three prior metacog
findings, none of which measure the same thing:

* [`2026-08-13-metacog-robust-confidence-GO.md`](2026-08-13-metacog-robust-confidence-GO.md) and
  [`2026-08-27-metacog-hedge-confidence-band-recalibration-GO.md`](2026-08-27-metacog-hedge-confidence-band-recalibration-GO.md)
  calibrate the SAME `nmda_norm` margin's existing **2-band** (confident/uncertain) split. A clean 2-way gap says
  nothing about whether that gap has a discriminable **interior** — the question this de-risk asks.
<!--derived-->
* [`2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md`](2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md)
  and [`2026-09-05-metacog-accumulation-to-bound-middle-band-PARTIAL.md`](2026-09-05-metacog-accumulation-to-bound-middle-band-PARTIAL.md)
  are the task brief's named "twice-measured hard case" — but they target a **different, upstream** mechanism
  (`RFPhasorComposer._spiking_margin[_accum]`, a discrete winner-vs-runner-up **spike-count** margin over the
  recall-cleanup competition that *derives the evidence scalar*), where the middle band sits near chance
  (AUC ~0.48–0.585, quoted from those two findings' own text, not this de-risk's artifact). This de-risk drives
  `nmda_norm_margin` directly — the workspace's own **downstream**,
  continuous, divisively-normalized **conductance** balance, already 8-way averaged (`READ_REPS`) by the
  production organ's own design — a different substrate one level further downstream, reusing neither runner's
  composer/capture machinery.

## What was built (additive only, no `sim/` edit, no edit to `metacog_production_organ.py` at all)

`research/runners/_metacog_graded_marker_derisk.py` reuse-by-imports `MetacogProductionOrgan`, `nmda_norm_margin`,
`SIG_LO`/`SIG_HI`/`BASE_PA`/`READ_REPS` from `metacog_production_organ.py`, and drives `organ.judge(evidence,
lesion=...)` — the actual production call path — for the sweep. It never edits that file: `ORGAN_FILE_BASELINE_HASH`
(`803c5a28b926ec55094942a3dd7999986f48977e`, captured via `git hash-object` before this runner was written) is
re-verified by `git hash-object` at aggregate time — an exact data compare, not an inference from "nothing was
touched" — and matched on every run (`organ_file_unchanged: true` in the artifact).

**Self-calibration.** At build time, on the SAME organ instance under test, a synthetic LOW (evidence 0.0–0.25) /
MED (0.375–0.625) / HIGH (0.75–1.0) battery of 6 points each — a fixed 0.125 buffer between bands so no
calibration point can straddle a future boundary — is read through `nmda_norm_margin`, and the two boundaries are
placed by `_boundary_between`, a literal generalization of `MetacogProductionOrgan.ensure_built`'s own existing
2-point placement rule (midpoint of the clean inter-band gap, or the class-mean midpoint if the gap is not clean)
applied twice. No threshold in this file was hand-set against the observed numbers. On the brief's "ride the
scale-invariant `margin_snr`/SNR anchors" note: those anchors exist to compare a *different*, codebook-size-variant
quantity (`mean_role_confidence`'s host cosine score) across vocab scales; `nmda_norm_margin` is already a ratio
of two conductances (Carandini & Heeger divisive normalization) with no codebook-size dependence, so no analogous
remap applies here — the two boundaries are calibrated directly from the margin's own battery.

**The ladder and its honest phrasing** (`graded_marker_prefix`, mirroring `hedge_prefix`'s existing convention):

| band | phrase (functional read-out only) |
|---|---|
| ASSERT | *(bare — no prefix)* |
| HEDGE | "I think — my decision-margin reads this as moderately confident, so take it as likely-but-not-certain: " |
| SPECULATE | "I'd only guess — my decision-margin reads this as low-confidence, so take it as a speculative guess, not an assertion: " |

Every phrase is a read of the spiking margin, never a felt/phenomenal/conscious claim — mechanically checked by
`_honesty_scan` against a banned-word list (feel/conscious/aware/experienc.../sentien.../subjectiv.../qualia/
phenomenal); all three pass (`honesty_scan_clean: true`).

## Pre-registered gate (fixed before any 3-band number was read)

GO iff, on ≥5/6 seeds: (1) the emitted band is monotone-non-decreasing across a 9-point evidence sweep (0..1,
i.e. drive 40..260 pA) AND Spearman(margin, evidence) ≥ 0.80; (2) both adjacent battery gaps are ordinally
separable — pairwise AUC(low,med) and AUC(med,high) ≥ 0.75 **and** each adjacent median-margin gap exceeds the
wider of its two flanking within-band standard deviations. Bars, battery ranges and sweep levels are fixed
constants in the runner file, written into its own header docstring before the first seed was measured.

## 6-seed results

<!--derived--> (`research/findings/raw/_metacog_graded_marker_derisk_6seed.json`)

| seed | monotone | Spearman rho | AUC(low,med) | AUC(med,high) | gap>spread (lo/med, med/hi) | both conditions |
|---:|:---:|---:|---:|---:|:---:|:---:|
| 42  | yes | 1.000 | 1.000 | 1.000 | yes / yes | PASS |
| 43  | yes | 1.000 | 1.000 | 1.000 | yes / yes | PASS |
| 44  | yes | 1.000 | 1.000 | 1.000 | yes / yes | PASS |
| 100 | yes | 1.000 | 1.000 | 1.000 | yes / yes | PASS |
| 101 | yes | 1.000 | 1.000 | 1.000 | yes / yes | PASS |
| 102 | yes | 1.000 | 1.000 | 1.000 | yes / yes | PASS |

**6/6 seeds pass both conditions** (gate needed ≥5/6).

<!--derived-->

Seed 42's raw calibration battery (`s42_intact_a.json`, `nmda_norm_margin` units, rounded to 4 decimals here)
shows the separation directly and non-overlapping: LOW 0.0212–0.0441, MED 0.0552–0.0709, HIGH 0.0792–0.0995.
The sweep bands land exactly where the battery predicts: evidence 0.0/0.125/0.25 → SPECULATE,
0.375/0.5/0.625 → HEDGE, 0.75/0.875/1.0 → ASSERT, on every seed.

## Anti-cheats (all six implemented; all pass, 6/6 seeds)

<!--derived-->

1. **SHUFFLE/YOKE control** — the intact sweep's own margins, permuted across the evidence axis with a fixed
   per-seed RNG, classified under the SAME boundaries: non-monotone on 6/6 seeds (seed 42's shuffled bands are
   `HEDGE, ASSERT, ASSERT, HEDGE, SPECULATE, SPECULATE, SPECULATE, HEDGE, ASSERT` against evidence
   `0.0..1.0` — visibly scrambled, not tracking). `n_shuffle_fails_to_track: 6`.
2. **LESION-load-bearing** — a fresh-subprocess rebuild at the SAME seed with `nmda_norm_margin(...,
   lesion=True)` at every sweep level, classified under the matching INTACT run's OWN boundaries (same seed →
   same underlying network). Collapses to all-SPECULATE and rho≈0 on 6/6 seeds; seed 42's lesioned margin is a
   flat 0.00290 at every evidence level (vs. an intact range of 0.0212–0.0995) — the evidence differential is
   removed, not merely attenuated. `n_lesion_collapses: 6`; per-seed lesion margins range 0.0027–0.0109 (small,
   seed-dependent numerical noise floor, never tracking evidence: rho=0.0 on every seed since a constant series
   has undefined/zero rank correlation by `_spearman`'s own zero-variance convention).
3. **NOT-a-2-level-relabel** — condition (2) requires BOTH the low/med AND med/high boundary separable above
   the within-band spread, scored across the full 6-point battery distribution at each level, never a single
   templated evidence value. Both adjacent gaps hold on 6/6 seeds (table above).
4. **DETERMINISM/null** — the intact arm ran twice per seed in independent fresh subprocesses (`intact_a`,
   `intact_b`); every calibration number and every sweep margin compares exactly equal
   (`determinism.byte_identical: true`, empty `diffs` list) on 6/6 seeds.
5. **BYTE-IDENTICAL OFF** — `metacog_production_organ.py`'s git blob hash is unchanged
   (`803c5a28b926ec55094942a3dd7999986f48977e`, re-verified by `git hash-object` at aggregate time, not inferred
   from the diff being absent). `organ_file_unchanged: true`.
6. **HONESTY BOUNDARY** — see the phrase table above; `_honesty_scan` clean on all three bands.

## Verdict block (as emitted by `tools.verdict.Verdict`)

<!--derived--> (`research/findings/raw/_metacog_graded_marker_derisk_6seed.json`'s `preconditions` list, all 14
checks `ok: true`; reproduced in condensed form — full detail including all 6 per-seed lesion `control` checks is
in the artifact)

```
require  n_seeds cond1 (monotone AND rho>=0.80), need >=5/6        measured=6  -> ok
require  n_seeds cond2 (3-band ordinally separable), need >=5/6    measured=6  -> ok
require  n_seeds BOTH conditions, need >=5/6                       measured=6  -> ok
require  shuffle/yoke control fails to track on all 6 seeds        measured=6  -> ok
require  lesion collapses to SPECULATE + rho~0 on all 6 seeds      measured=6  -> ok
require  intact-repeat byte-identical determinism on all 6 seeds   measured=6  -> ok
require  metacog_production_organ.py byte-identical to baseline    measured=True -> ok
require  graded-marker phrasing carries no phenomenal word         measured=True -> ok
control  seed {42,43,44,100,101,102}: lesion collapses |rho| gap   treatment=1 control=0, sep=1 > 0.4  (x6) -> ok
=> GO
```

## Why this reads cleaner than the sibling PARTIAL residuals — an honest structural explanation, not a discrepancy

The two "twice-measured hard case" findings this brief names both struggle in their middle band because their
signal is a **discrete winner-vs-runner-up spike COUNT** over a short competition window — small-integer counts
are inherently noisy near a decision boundary. `nmda_norm_margin` reads a **continuous NMDA conductance ratio**,
already averaged over `READ_REPS=8` fixed-jitter reads by the organ's own existing design (documented in
`metacog_production_organ.py`'s own module docstring as the mechanism that "denoises the tiny single-trial
margin"), and the underlying dynamic (sustained conductance rising smoothly with a stronger sustained drive
current) is monotonic by construction of the NMDA accumulator physics, not merely by post-hoc curve-fitting. This
is why AUC=1.000 / rho=1.000 appear rather than the ~0.48–0.83 range the sibling residuals report — a genuine,
structural difference in which substrate is being read (continuous conductance vs. discrete spike count), not an
artifact of a weaker test. The READ_SEED/READ_REPS design also makes a given evidence value map to a
**deterministic** confidence read (by the production organ's own stated intent — "so a given evidence -> a
deterministic confidence decision (reproducible per turn)"), which the determinism anti-cheat above confirms
independently rather than assumes.

## Honest scope / residuals (declared, not banked)

<!--derived-->

* **Not a production wire-in.** No call site in `webapp/server.py` / `judge()` was touched; the 3-band ladder
  lives only in this de-risk runner. Wiring it into production judge/reply construction is the explicitly-named
  next rung, out of scope here by the task brief.
* **Synthetic evidence, not real-traffic evidence.** The sweep drives `organ.judge()` with a hand-specified
  evidence scalar in [0,1]; it does not exercise the real upstream evidence-derivation pipeline
  (`mean_role_confidence` / `margin_snr` / real production traces) that feeds `judge()` in practice. This
  measures the workspace margin's own resolving *capacity*, not the end-to-end real-conversation band
  distribution — a real-traffic 3-band recalibration (mirroring `2026-08-27`'s real-traffic 2-band recalibration)
  is the natural follow-up before any wire-in.
* **6 points per calibration band.** A modest sample; the separation is not marginal (fully non-overlapping
  ranges on every seed, AUC=1.000), so this is not read as a fragile result, but a larger battery would be a
  cheap strengthening pass before production trust.
* **numpy/CPU only**, no GPU touched (the GPU stayed reserved for the concurrent episodic harvest per the task's
  compute constraint). Runtime: 6 seeds × 3 arms × ~13–17s/arm ≈ 2m7s wall-clock total
  (`time` output on this run), each arm wrapped in `tools/memcap.sh 4` (systemd --user scope confirmed available
  in this environment; no fallback was needed — `memcap_wrapped: true` on all 18 spawn records).

## Next rung (not attempted here, per the task's explicit scope)

Wire the 3-band ladder into `judge()`'s real call path behind a default-off flag, then re-run this exact
methodology (self-calibration + sweep + all six anti-cheats) against **real production evidence** derived from
`mean_role_confidence` on live traffic, mirroring how `2026-08-27-metacog-hedge-confidence-band-recalibration-GO.md`
recalibrated the 2-band split against real turns rather than only synthetic evidence.
