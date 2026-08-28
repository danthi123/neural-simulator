---
type: finding
status: go
mechanism: metacog honesty-hedge confidence read (margin-keyed mean_role_confidence, recalibrated ROLE_CONF_LO/HI)
lane: introspection-self-model
integration_faculty: metacog-honesty-hedge
date: 2026-08-27
seed-waiver: a deterministic code fix (a formula bug + a re-key to an existing signal), not a stochastic
  multi-seed generalization claim -- verified by exact answer-text asserts + a real-traffic measurement + a
  synaptic-noise sweep + a lesion check, not by seed-count replication (single fixed seed=42 throughout, matching
  the composer's own reproducibility convention).
artifacts:
  - research/findings/raw/_metacog_confidence_recalib/measure_real_confident.json
  - research/findings/raw/_metacog_confidence_recalib/measure_real_build_noise_sweep.json
  - research/findings/raw/_metacog_confidence_recalib/measure_prod_config_margins.json
  - research/findings/raw/_metacog_confidence_recalib/verify_discriminates.json
  - research/findings/raw/_metacog_confidence_recalib/verify_prod_discriminates.json
  - research/findings/raw/_metacog_confidence_recalib/final_verify.json
  - research/findings/raw/_metacog_confidence_recalib/verify_184_warning.json
runner: research/findings/raw/_metacog_confidence_recalib/{measure_real_confident,measure_real_build_noise_sweep,measure_prod_config_margins,verify_discriminates,verify_prod_discriminates,final_verify,verify_184_warning}.py
---

# Metacog honesty-hedge: the confidence read was a self-referential ratio, always 1.0 — root-caused, re-keyed to a genuine margin, and recalibrated (GO)

**Verdict: GO.** Issue #181's premise ("`mean_role_confidence` saturates at ~1.0 on every real turn") was correct
but the cause was NOT a stale calibration band — it was a formula bug. Fixed at the root, re-keyed to an
already-validated signal, and verified end-to-end through the real `/api/brain-chat` handler: the hedge now
demonstrably stays silent on confident recalls and fires on genuinely uncertain ones. Issue #184 (silent-failure
guard) is also closed: `webapp/server.py` now warns instead of silently degrading when a trace-capable composer's
confidence read comes back unexpectedly empty.

## Root cause (not a calibration problem)

`OneBrainComposer._block_role_scores`'s `_winner` helper (`research/runners/one_brain_composer.py`) computed
`confidence = s[argmax(s)] / max(s)`. Since `j = argmax(s)`, `s[j]` **is** `max(s)` by construction — this ratio
evaluates to exactly `1.0` for every non-degenerate decode and `0.0` only when nothing scores above zero (an
already-abstained case). It never varied. `metacog_production_organ.mean_role_confidence` averaged this field
across role chips, so `mean_role_confidence` (mrc) read `1.0` on essentially every real production turn —
confirmed independently by `2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-NOGO.md`
(mrc=1.0 on both its tested real turns) and reproduced here. No calibration band placement could have fixed
this: the input signal itself carried no information.

## The fix: re-key to the composer's own already-validated margin, not a new formula

`OneBrainComposer` already has `_margin(scores) = (peak - runner_up) / peak` — the SAME normalized decisiveness
read its `confidence_gate` familiarity gate uses (multi-seed validated in
[`2026-06-18-emergent-graceful-degradation-derisk.md`](2026-06-18-emergent-graceful-degradation-derisk.md): ~0 on
a noise-dominated/damaged read, ~0.5+ on an intact confident one, `g=0.15` the validated clean/noise separator).
`_winner` now also returns this as `margin` (the legacy `confidence` field is kept byte-identical for any other
consumer).

**External check** (`introspection-self-model` lane, `deep_research_at_wall`): the top-1/top-2 score margin as a
confidence read is an established technique outside this codebase too — Gomez et al. 2019, arXiv:1903.09215<!--derived-->
("Calibrated Top-1 Uncertainty estimates for classification by score based models",
<https://arxiv.org/pdf/1903.09215>) studies exactly this margin-between-top-two-scores confidence notion,<!--derived-->
confirming the re-keyed signal is a recognized approach, not an ad hoc substitute.
`metacog_production_organ.mean_role_confidence` now averages each role chip's `margin` when present,
falling back to the legacy `confidence` field otherwise (unchanged behavior for a composer/chip that never
populates `margin`). This reuses a mechanism the codebase already trusts, rather than inventing an untested one.

## Recalibration, against REAL measured data through the real handler

The OLD `ROLE_CONF_LO/HI` (0.35/0.52) were tuned against the broken always-1.0 signal, so their placement never
mattered. Measured the NEW `margin`-based mrc on the actual production tiny-demo composer (`_build_tiny_demo`,
what `webapp/server.py` builds; it sets `enable_attributed=True`, adding an always-near-zero-margin `attribute`
role chip to every real trace):

| condition | mrc range | source |
|---|---|---|
| CONFIDENT — 5 real facts, through the real `/api/brain-chat` handler, intact store | 0.504 .. 0.615 | `measure_real_confident.json`, `measure_real_build_noise_sweep.json` |<!--derived-->
| UNCERTAIN — the SAME real composer + query, synaptic-noise-perturbed store (the identical damage model the graceful-degradation de-risk validated), noise levels that still return an answer (not an abstain) | 0.090 .. 0.60 (clearly-degraded region 0.15..0.36; light noise correctly stays near-confident) | `measure_real_build_noise_sweep.json`, `final_verify.json` |<!--derived-->

Set `ROLE_CONF_HI=0.50` (at/below the measured confident floor — every real confident turn clips to
evidence=1.0) and `ROLE_CONF_LO=0.30` (below the clearly-degraded region — a genuinely weak match reaches the
metacog organ's own low-evidence calibration zone, evidence<=~0.4, and hedges). A lightly-perturbed read is
allowed to land in the middle and go either way — an honest ambiguous case, not gamed to a side.

## End-to-end verification (`final_verify.py`, through the real production composer)

All through the REAL `/api/brain-chat`-building composer (`_build_tiny_demo`, `composer_kind="onebrain"`), CPU
backend (memory-safe: another brain-load agent was running concurrently, so `SIM_BACKEND=cupy` was deliberately
avoided; system RAM stayed >20GB free throughout):

- **(A) Confident, no regression**: all 5 real tiny-demo facts (`brain use->spikes`, `brain learn->words`,
  `brain store->memory`, `dog chase->cat`, `cat eat->fish`) read `confident=True` (no hedge) — asserted the
  ANSWER TEXT is unchanged (`ans == p` for every fact; the fix only touches the confidence side-channel).
- **(B) Genuinely uncertain, hedge fires**: the same composer's store synaptic-noise-perturbed (sigma>=1.1,
  still answering, not abstaining) reads `confident=False` (hedge fires) at every one of 7/8 sweep levels;
  `heavy_noise_all_hedge=True` for the clearly-degraded region.
- **(C) Lesion**: `MetacogProductionOrgan.judge(..., lesion=True)` (== `BRAIN_METACOG_LESION=1`) collapses BOTH
  the confident and the uncertain case to the SAME `confident=False` — exactly the organ's own documented
  mechanism ("a would-be-confident answer FLIPS to a hedge" when the evidence differential is removed),
  confirming the discrimination is driven by the organ genuinely reading the evidence signal, not a host
  shortcut.
- Verdict block: `"DISCRIMINATES": true, "all_5_confident_facts_no_hedge": true, "any_uncertain_turn_hedges":
  true, "no_answer_regression": true`.

## Issue #184 (silent-failure guard), verified through the real handler (`verify_184_warning.py`)

`webapp/server.py`'s `_metacog_qualify` now distinguishes a genuine out-of-scope skip (no answer / a composer
that never traces) from an unexpected empty read on an ANSWERED turn from a trace-capable composer — the exact
shape of the `TieredFactStore.__setattr__` regression that silently disabled this hedge for a day. Reproduced
that regression directly (wrapped `OneBrainComposer.query_patient` to answer normally, then wipe
`self.last_trace = None`, mirroring the prior bug) and confirmed: (1) a normal real turn stays quiet, (2) the
reproduced bug prints `[webapp] METACOG WARNING (#184): ...` and names the TieredFactStore precedent, (3) a
genuine abstain (unstored cue) stays quiet. `verify_184_warning.json`: `"GUARD_WORKS": true`.

## Scope / honesty notes

`ROLE_CONF_HI=0.50` sits close to the real confident floor (0.504, see the table above) with limited headroom — a future vocabulary<!--derived-->
change that shifts the confident distribution could need re-measurement (the calibration is against THIS
tiny-demo KB's real distribution, not a theoretical bound). The `attribute` role chip's structural near-zero
margin (an unbound/never-taught role decoded as noise) is a legitimate contributor to real mrc, not excluded —
narrower per-role weighting is a possible future refinement, not attempted here. No `sim/` edit; changes are
confined to `research/runners/one_brain_composer.py`, `research/runners/metacog_production_organ.py`, and
`webapp/server.py` (the existing default-ON `BRAIN_METACOG` gate; `_composer_traces`/warning logic is additive).

Artifact: research/findings/raw/_metacog_confidence_recalib/final_verify.json · runners in
research/findings/raw/_metacog_confidence_recalib/.
