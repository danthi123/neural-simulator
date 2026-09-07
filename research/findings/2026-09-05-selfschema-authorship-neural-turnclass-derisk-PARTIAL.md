---
type: finding
status: go
date: 2026-09-05
updated: 2026-09-06
mechanism: self-schema authorship (DR-3) -- neural turn-class read replacing the host is_hyp constant
lane: scaffold-retirement backlog rank 15 (research/coordination/scaffold_retirement_backlog.md)
verdict: RESOLVED -> NOW GO 6/6 (2026-09-06 via DG-like pattern separation on the novelty read -- brain-based granule-cell recoding + the unchanged anti-Hebbian familiarity gate; commit b0d3c916a; artifact postfix_6seed.json; robust operating-point plateau 8x/2%..64x/3%, no false positives, default-ON behind --dg-sep). De-risk only, NOT wired to webapp. [ORIGINAL 2026-09-05 PARTIAL record follows:] PARTIAL, 5/6 seeds fully GO. A per-trial NEURAL signal (real generative-draw spiking activity + a real Hebbian familiarity/novelty read) drives the DR-3 self_schema author pool's input current in place of the host's hard-coded `authored=True` constant, and correctly classifies REAL self-authored (open-ended-generation) vs REAL recalled trials: mean accuracy 0.9542 across 6 seeds (chance 0.5; per-seed range 0.9125-0.9875) vs a content-blind baseline pinned at exactly 0.5 (today's constant, scored on the same set). LOAD-BEARING on 6/6 seeds (accuracy bar, shuffle-anti-cheat collapse, signal-severed-lesion collapse, and the existing base-organ self-lesion collapse all pass every seed); the fixed-agent heard-to-self SWITCH test (identical agent, once with its fact present, once with it removed and forced through the real generation loop) switches correctly on 29/30 evaluable probes (100% on 5/6 seeds, 4/5 on seed 100) -- the one miss is mechanistically characterized (a generated candidate sharing 2 of 3 role-fillers with an untouched co-resident stored fact, so its bundled-cue novelty reads weak by construction, not by instrument failure). A de-risk only: NOT wired into `webapp/server.py`; the host regex ROUTING (`_parse_open_ended`, whether a turn enters the generation branch at all) is a separate, unretired residual, declared out of scope.
artifacts:
  - research/runners/_selfschema_authorship_neural_turnclass_derisk.py
  - research/findings/raw/_selfschema_neural_turnclass/soak_6seed.json
  - research/findings/raw/_selfschema_neural_turnclass/postfix_6seed.json
---

# Self-schema AUTHORSHIP: a neural turn-class read retires the host `authored=True` CONSTANT (5/6 seeds; the residual is characterized)

## The diagnosis this de-risks (scaffold-retirement backlog rank 15)

The scaffold-shortcut-map (`w9sn9wn4b`) flagged the DR-3 self-schema AUTHORSHIP wiring
(`webapp/server.py` rich `brain_chat` path) as **LOW-actual / nominal-on**: the row is `BRAIN_SELF_SCHEMA`
default-ON since the 2026-08-26 production wire-in
([`2026-08-26-DR3-self-schema-authorship-production-wirein-GO.md`](2026-08-26-DR3-self-schema-authorship-production-wirein-GO.md)),
but the self/heard decision is made by a HOST BOOLEAN before any neuron runs:

```python
is_hyp = bool(r.get("hypothesis"))                                          # server.py:6056 -- host
...
if is_hyp:
    _ss_read = _get_self_schema_organ().read_author(authored=True, ...)     # server.py:6142-6143 -- CONSTANT
```

`authored=True` never varies -- the block only executes inside `if is_hyp:`, where `is_hyp` is already always
True. The self_schema `author` sub-block's own 6-seed GO
([`2026-07-23-DR3-self-schema-region-6seed-GO.md`](2026-07-23-DR3-self-schema-region-6seed-GO.md), authorship
acc 1.000) validated the READOUT half of the circuit only: its own ground truth was an EXTERNALLY-SUPPLIED
per-trial boolean (`authorship = rng.integers(0, 2, ...)`), exactly mirroring how production supplies
`authored=True` from `is_hyp`. Only the LESION path (severing `schema_access`) ever exercises the pool's dynamic
range on a live turn; the CLASSIFICATION itself is never asked of the substrate. This file builds and tests the
missing DETERMINATION half.

## The mechanism (reuse-by-import only; no `sim/` edit; no edit to any existing file)

`is_hyp` becomes True only when `ChatBrain._generate_hypothesis` actually fires and finds a novel, plausible,
non-contradictory, not-already-known proposition; every other answered turn is a RECALL. Both branches already
leave a real, distinct, in-family trace when they run, reused here unmodified:

- **the GENERATE branch** drives the validated `VocabAgnosticSpikingSampler` (Izhikevich WTA bank + OU membrane
  noise, `_followon2_spiking_wta_sampler_derisk.py`, the SAME organ production's
  `vocab_agnostic_spiking_generation_production_organ.py` installs onto `_generate_hypothesis`'s proposer)
  through >=1 REAL spiking competitions (`draw_from_weights` -> `cp_firing_states`) to CONSTRUCT a candidate. A
  recall turn never builds this sampler at all, so its `n_spiking_draws` delta is a genuine per-turn fact about
  which branch ran, not a label.
- **the RECALL branch's** success criterion IS familiarity with a taught fact, read here with the SAME
  Bogacz-Brown anti-Hebbian familiarity gate (`RealAntiHebbianFamiliarity`,
  `_phaseB_biologize_moat_streamcodes_derisk.py`) the self_schema de-risk's OWN confidence axis already reuses:
  imprint the brain's stored facts as the known span; a queried triple's NOVELTY is ~0 for a taught fact and
  higher for one that was never stored -- exactly the moat-verify's own criterion for what may become a
  `HypothesisSVO` in the first place (`what_does(a,ac) != p AND is_it_true(a,ac,p) == 'unknown'`).

A per-trial scalar `_turn_class_signal = 0.5*novelty_norm + 0.5*draws_norm` drives the UNCHANGED base bridge's
author-pool input current (`build_self_schema_bridge` / `_run_trial`, imported verbatim from
`_self_schema_region_derisk.py`), replacing the host's two-valued `AUTHOR_PA if authored else 0.0` switch with a
graded, per-trial, real-computation-derived current. `AUTHOR_PA` / `CONF_PA` / `CONTENT_K` are imported from the
ALREADY-SHIPPED `self_schema_production_organ.py` so this de-risk tests the SAME operating point production uses,
not an invented one.

**Declared scope.** `_parse_open_ended`'s regex ROUTING -- whether a turn even enters the generation branch --
stays host; that is a separate, already-named residual elsewhere in the scaffold map (e.g. rank 11's
"topic-swap regex+host"), not this rank's target. This file targets only rank 15's own diagnosis: the authorship
ORGAN's input is a hard-coded constant carrying zero per-turn information. Given a turn that DID take one branch
or the other, does a neural read of what that branch actually DID (not a label copied from which branch the host
chose) correctly and load-bearingly reproduce the authorship tag?

## Method

Six seeds (42, 43, 44, 100, 101, 102), numpy-CPU. Each seed builds a small deterministic toy stored-fact KB
(10 SVO facts + 1 explicit negation over a ~16-word vocab, reshuffled per seed), the REAL `GenerativeReplayProposer`
(reused verbatim from `_genfrontier_b2_generative_replay_derisk.py`) with the REAL vocab-agnostic spiking draw
organ installed, and the REAL familiarity gate imprinted with the KB's own stored facts. Per seed: 40 REAL
"heard" trials (query a genuine stored fact; recall succeeds by construction, asserted) and 40 REAL "self" trials
(a genuine run of `_generate_hypothesis`'s own gates -- `_plausible`/`_contradicts`/moat-verify -- reused
verbatim, for a known agent, until a valid novel proposition is found or the search budget is spent). Ground
truth is which branch ACTUALLY produced the content, never an externally-chosen label. A threshold is calibrated
per seed (midpoint of the empirical self/heard mean author-pool rates, the SAME convention the base 6-seed-GO
de-risk uses) and accuracy is scored against it.

Four additional per-seed checks: (1) a **content-blind baseline** -- today's constant (`authored=True` on every
trial) scored on the identical set; (2) a **shuffle-control** (permute which trial's signal drives which trial's
read, score the resulting reads against the TRUE labels -- mirrors the base de-risk's own
SHUFFLED-INTERNAL-SIGNAL anti-cheat); (3) a **signal-severed lesion** (force the current to 0 regardless of the
real branch -- "no evidence" collapses to the pool's fail-safe default); (4) the **existing base-organ self-lesion**
(`schema_access=False`, unmodified) applied on top of the new signal, to confirm the downstream pool's own
2026-07-23 anti-cheat is unaffected by this file's changes. A fifth, separate check picks a fixed agent, tags it
once HEARD (its fact present) and once SELF (the identical fact removed from a freshly-rebuilt fixture, forcing a
genuine re-run of the generation loop for that agent) and requires the tag to SWITCH.

`research/runners/_selfschema_authorship_neural_turnclass_derisk.py`. `tools.lab.lever`/`attributable_to`/
`undefined_if_empty`/`void_if` are used for the per-seed evidence-moved check, the baseline-attribution
calculation, and the pair-count guard.

## Results (6 seeds; full per-seed data in `research/findings/raw/_selfschema_neural_turnclass/soak_6seed.json`)

| seed | accuracy | baseline (const.) | shuffle | signal-severed | base-organ lesion | switch | seed GO |
|---|---|---|---|---|---|---|---|
| 42  | 0.9375 | 0.5 | 0.4625 | 0.5 | 0.5 | 4/4 (1.0) | **True** |
| 43  | 0.9625 | 0.5 | 0.5125 | 0.5 | 0.5 | 5/5 (1.0) | **True** |
| 44  | 0.9875 | 0.5 | 0.4875 | 0.5 | 0.5 | 5/5 (1.0) | **True** |
| 100 | 0.9125 | 0.5 | 0.4875 | 0.5 | 0.5 | 4/5 (0.8) | False |
| 101 | 0.9875 | 0.5 | 0.5625 | 0.5 | 0.5 | 6/6 (1.0) | **True** |
| 102 | 0.9375 | 0.5 | 0.4625 | 0.5 | 0.5 | 5/5 (1.0) | **True** |

Mean accuracy 0.9542 (chance 0.5, GO bar 0.85 -- clears on every seed). Mean shuffle-collapse 0.4958, mean
signal-severed-collapse 0.5, mean base-organ-lesion-collapse 0.5 -- all comfortably under the 0.65
chance-margin bar, on every seed. `attributable_to("... over chance", acc-0.5, baseline_acc-0.5)` reads **100.0%
attributable to the manipulation, 0.0% present in the control** on every single seed (the baseline is pinned at
exactly 0.5 throughout -- it is content-blind by construction, since `authored=True` never reads the trial).
**Caveat on that 100% figure (self-flagged, `gates/discriminating_power`):** this ratio sits at its ceiling on
EVERY seed because the CONTROL (`baseline_acc - 0.5`) is a mathematical certainty, not an independent empirical
measurement -- a constant that always predicts 'self' scores EXACTLY 50% on an exactly-balanced set by
construction, so the denominator-matching numerator is guaranteed regardless of how good or bad the treatment
is. `attributable_to` is doing real work elsewhere in this codebase (separating an empirically-varying proxy from
an empirically-varying mechanism); here it correctly reports there is no confound to attribute away, but the
"100%" itself carries no additional evidence beyond "the baseline is deterministic" -- the accuracy row (0.9125
through 0.9875, vs the 0.85 bar) is the actual load-bearing measurement, not this ratio.
The novelty and draws-used evidence itself genuinely MOVES between classes on every seed (`lever()`, e.g. seed 42:
novelty heard=0.0 -> self=0.1416, draws heard=0.0 -> self=9.7; seed 101: draws climb to 29.6 on the hardest
KB configuration -- the generation search working harder, not a constant).

**5 of 6 seeds are fully GO** on every pre-registered component. Seed 100 clears accuracy (0.9125), both
anti-cheats, and the base-organ lesion, but its load-bearing switch reads 4/5 (0.800), short of the pre-registered
1.000 bar -- so the seed's own GO reads False and the aggregate verdict is honestly **PARTIAL (5/6)**, not GO.

### The one miss, characterized

Seed 100's non-switching pair: agent `horse`, removed fact `(horse, follow, shadow)`, generated candidate
`(horse, eat, mouse)`. `horse`'s OTHER stored fact, `(horse, eat, water)`, is untouched by the removal. The
generated candidate shares 2 of 3 role-fillers (`horse`, `eat`) with that untouched fact and differs only in the
patient (`mouse` vs `water`). This file's cue is an unstructured bundle (`code(a) + code(v) + code(p)`), so a
candidate sharing 2 of 3 terms with an imprinted fact has substantial vector overlap with the imprinted span by
construction -- its novelty reads weak (below this seed's calibrated threshold) even though the triple is
genuinely un-taught. This is a property of the bundled-cue encoding meeting a partial-overlap recombination, not
an instrument failure: switching to a role-bound (HRR circular-convolution) cue would not change it, because the
two shared terms are numerically IDENTICAL under either scheme -- the overlap is a fact about which 2 of 3
role-fillers repeat, not an artifact of how the third is combined. 29 of the 30 load-bearing probes run across
all 6 seeds (96.7%) switched correctly; this is the one exception, and its cause is fully accounted for.
No threshold or encoding constant was retuned after seeing this result -- the pre-registered 1.000 bar is
reported as measured, not relaxed to manufacture a clean GO (the `--n-per-class`/`--n-pairs`/threshold values in
`DEFAULT_THRESHOLDS` were fixed before the 6-seed run; the ONLY code changes after seeing early results were a
scoring bug found in-flight -- see below).

### An instrument bug found and fixed before the reported numbers

An earlier run of this same 6-seed command read a clean "6/6 GO," which turned out to be a scoring defect, not a
real result: `switch_frac` was assigned `tools.lab.undefined_if_empty(...)`'s return value, which is the raw
`score` (a COUNT, e.g. `4`) used only to print "UNDEFINED" instead of a fabricated 0 -- it does not compute a
ratio. The GO check `switch_frac >= 1.0` was therefore comparing a raw count (almost always >= 1 once any pair
switches) against 1.0, not a true fraction, so seed 100's real 4/5 (0.800) silently read as a pass. Fixed by
computing `switch_frac = n_switched / n_pairs_eval` explicitly and keeping `undefined_if_empty` only for its
printed diagnostic. The corrected numbers above are what is reported; the honest verdict moved from a
manufactured "GO (6/6)" to the measured "PARTIAL (5/6)" as a direct result of this fix.

## What this retires, and what it does not

**Retires (de-risked, not yet wired):** the claim that the authorship organ's input MUST be a host-supplied
constant. A per-trial signal built entirely from machinery `_generate_hypothesis` already runs or could run
(the SAME spiking draw organ, the SAME class of familiarity gate the self_schema de-risk's own confidence axis
uses) classifies REAL self-vs-heard trials at 0.9542 mean accuracy against a 0.5 content-blind baseline, is
load-bearing under both a shuffle-control and a dedicated signal-severed lesion on 6/6 seeds, leaves the EXISTING
2026-07-23 base-organ lesion unaffected, and switches correctly on 96.7% of fixed-agent heard/self probes with
the one miss mechanistically understood.

**Does not retire (declared, unchanged):**
1. `_parse_open_ended`'s regex ROUTING (whether a turn enters the generation branch at all) -- a separate,
   already-flagged host residual elsewhere in the scaffold map, out of scope for rank 15.
2. Production wiring. This is a de-risk (`research/runners/_selfschema_authorship_neural_turnclass_derisk.py`)
   exercised on a small toy KB, not a change to `webapp/server.py`. No default flipped; `BRAIN_SELF_SCHEMA`'s
   existing behaviour (host `authored=True`) is untouched. Per this project's own lifecycle
   (`_self_schema_region_derisk.py` earned its GO before `self_schema_production_organ.py` was written), a
   production organ + a NEW default-OFF flag (e.g. `BRAIN_SELF_SCHEMA_NEURAL_TURNCLASS`) reusing this file's
   `_turn_class_signal` against the LIVE proposer/sampler/familiarity-gate `_generate_hypothesis` already builds
   is the natural next step, gated on the owner/parent's review of the one characterized residual above.
3. A live end-to-end `/api/brain-chat` turn was not run (this de-risk's own toy KB, not the production stored-fact
   store) -- the same honest-residual class the 2026-08-26 production wire-in finding already disclosed for the
   base organ ("warm wedges" in an isolated worktree without a data lake).

## Scope / honesty

A FUNCTIONAL turn-class correlate (real spiking draw activity + a real Hebbian familiarity read), decoded through
the unchanged DR-3 author pool -- never a claim of subjective experience. Cost-routed CPU/numpy throughout
(~92s for the full 6-seed x 80-trial x 5-condition sweep); no GPU used, no `sim/` edit, no edit to any
pre-existing file.
