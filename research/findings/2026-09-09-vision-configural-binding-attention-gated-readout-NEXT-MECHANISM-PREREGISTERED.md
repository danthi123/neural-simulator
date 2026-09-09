---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-09
mechanism: PER-CLASS, PER-TRIAL BIASED-COMPETITION READOUT GATE (--readout attention-gated in _vision_lindiscrim_readout_derisk.py) -- a top-down attentional template A_c = normalized |w_c|, read straight off the already-fitted signed linear discriminant (zero new learning), combines multiplicatively with each trial's bottom-up C2 drive; a k-WTA competition (--attn-kwta-frac) zeroes the losing conjunction units per (trial, class); the survivors' raw drive, gain-renormalized by the realized surviving fraction, goes through the unchanged excitatory/inhibitory (Dale's-law) sign-split read and LIF class-population spiking WTA. Grounded in research/biology/attention-gated-readout.md.
lane: vision (identity readout, D-perception configural binding)
seeds: [42, 43, 44, 100, 101, 102]
verdict: DECISIVE 6-seed run LANDED -- LINDISCRIM-READOUT-PARTIAL-beat2/6-lb3/6, a REGRESSION versus the competitive-selection baseline it is stacked on (beat4/6-lb6/6, RATE_lin_ceiling_held 0.4288 identical here since the readout change does not touch the front end/bank/ceiling). Not a task GO (needs beat>=5/6, landed 2/6) and, per the pre-registered honest-read discipline, a disappointment rather than progress even though technically a PARTIAL by the letter of the bands (beat0/lb0 would be the formal NO-GO threshold). The diagnosed cause, quantified per-seed below: the hard hard-zeroing hard-competitive gate discards genuine DISTRIBUTED signal in 4 of 6 seeds when compared like-for-like against the ungated linear score on the identical trained discriminant and identical spike code -- the same "fine distributed cosine modulation" this file's own module docstring already diagnoses as the lane's core representational character, which a hard top-k elimination is structurally the wrong tool for. Mechanism BANKED at this operating point (--attn-kwta-frac 0.5, hard k-WTA); a graded/soft gain field (no hard zeroing) is the named next rung, not another frac sweep of the same hard-competition form.
artifacts:
  - research/findings/raw/lanes/perception/conjbind_attngated_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/vlin_attngated_smoke.json
  - research/findings/raw/lanes/perception/vlin_attngated_frac1_smoke.json
  - research/findings/raw/lanes/perception/vlin_competitive_smoke.json
  - research/findings/raw/lanes/perception/conjbind_competitive_n1152_heldoutpos_scramblenull_6seed.json
---

# Reading, not just having, an existing conjunction bank -- attention-gating the readout landed a regression

**Status: mechanism BUILT (`--readout attention-gated`, `_attention_gated_class_read`), byte-identical-off
PROVEN at the disabling value (`--attn-kwta-frac 1.0`), GO gate PRE-REGISTERED before the run, and the decisive
6-seed run has LANDED, stacked on the lane-best `--conj-select competitive` at its own proven sweet-spot
operating point (`--conj-select-overcomplete 4 --conj-select-kwta-frac 0.1`, `--conj-n 1152`).**
**Result: `LINDISCRIM-READOUT-PARTIAL-beat2/6-lb3/6` -- WORSE than the competitive-selection baseline
(`beat4/6-lb6/6`) it is stacked on top of. Not a task GO, and per the pre-registered honest-read rule, a
regression rather than progress.**

## Why this lever (the pre-registered next mechanism, not a re-derivation)

<!--derived-->
The competitive-selection mechanism's own operating-point sweep (research/findings/2026-09-09-vision-
configural-binding-competitive-selection-NEXT-MECHANISM-PREREGISTERED.md) landed EXHAUSTED: every swept
`--conj-select-overcomplete`/`--conj-select-kwta-frac` variant was worse than the untuned default (4x/0.1,
`beat4/6-lb6/6`), so tuning the SELECTION operating point does not close the remaining gap to task-GO (>=5/6).
That finding's own closing line named the untried orthogonal axis: "the NEXT MECHANISM (no-defer, not yet
attempted) is now the attention-gated readout (reweight which conjunction units each class population listens
to per-trial), NOT another selection-tuning sweep." `--conj-select` decides which conjunction units EXIST in
the bank (a one-time, training-time, population-wide decision, frozen thereafter and shared identically by
every class); this lever is deliberately orthogonal -- for a GIVEN fixed bank, it decides which of the bank's
EXISTING units a given CLASS's decision listens to on a given TRIAL.

## The mechanism (built, `_attention_gated_class_read`, `research/biology/attention-gated-readout.md`)

<!--derived-->
Additive, gated by a new `--readout {linear,attention-gated}` flag, default `linear` (byte-identical to every
prior run of this file -- proven below), stacked on top of whichever `--conj-select`/`--conj-order` bank is
already in use:

1. **Top-down attentional template**, per class, read straight off the ALREADY-FITTED signed discriminant
   (zero new learning): `A_c = |w_c| / mean(|w_c|)`, the class's own learned weight MAGNITUDE, normalized to
   mean 1. Grounded in Kandel PNS-6e's feature-based attention (attending to a feature dimension -- here,
   "which conjunction units this class finds informative" -- rather than a spatial location: "a second kind of
   attention, feature attention: In your search, you ignore [task-irrelevant items] and attend only to
   [task-relevant ones]") and its top-down/task-driven framing ("Another top-down influence is perceptual
   task").
2. **Biased drive**, per trial: `bd[n,c,j] = r[n,j] * A_c[j]` -- bottom-up stimulus drive combined
   multiplicatively with the top-down template, the same "combine drive with a bias/gain field" primitive this
   file already uses (`_apply_s2_norm`'s alpha/satdiv forms, `_select_conjunctions_competitive`'s candidate
   drive).
3. **Compete**, per (trial, class): a k-WTA keeps only the top `--attn-kwta-frac` fraction of units by `bd` and
   ZEROES the rest -- the IDENTICAL top-k-by-current-drive competitive primitive already established TWICE in
   this file (`_bcm_learn_s2_templates`'s `competitive_frac`; `_select_conjunctions_competitive`'s
   per-presentation k-WTA), reused a THIRD time, now at READ time instead of weight-update or bank-selection
   time (Foldiak 1991 / Kohonen 1982).
4. **Read**: `gated_r[n,c,j] = r[n,j] * win_mask[n,c,j] / k_eff_frac` -- the RAW drive of the SURVIVING units
   only (losers contribute exactly zero, not merely a smaller weight), gain-renormalized by the realized
   surviving fraction so overall drive magnitude stays comparable across `--attn-kwta-frac` settings. The SAME
   Dale's-law excitatory-minus-inhibitory sign-split (`w = w+ - w-`) and LIF class-population port +
   spiking WTA downstream are UNCHANGED from `_spiking_class_read`.

All 5 class-read call sites in `run_seed` (LEARNED held/train, scramble-null, RANDOM control, label-shuffle
null) now route through a `_class_read` dispatcher, so every anti-cheat exercises the SAME readout the
capability path uses -- the RANDOM control is therefore genuinely like-for-like: an attention-gated read with
`V` untrained, not a plain linear read with `V` untrained.

## Byte-identical-off, proven

<!--derived-->
Re-ran the competitive-selection tiny smoke's own recipe (`vlin_competitive_smoke.json`) with `--readout`
omitted (defaults to `linear`): every decode/reframe/dissociation/verdict number matched the committed reference
exactly except `elapsed_seconds` (the only expected diff). Then re-ran the IDENTICAL smoke with
`--readout attention-gated --attn-kwta-frac 1.0` (the gate's disabling value: `win_mask` is all-ones,
`k_eff_frac == 1.0`, so `gated_r == r` for every class): every field again matched the `linear` run exactly
except `elapsed_seconds` -- the mathematical identity holds in practice, not just by construction. A separate
non-degenerate smoke at the real default (`--attn-kwta-frac 0.5`) ran end-to-end with no NaNs/crashes/collapse
(`vlin_attngated_smoke.json`) before the decisive run was launched.

## Pre-registered GO gate (fixed BEFORE the decisive run)

<!--derived-->
Identical criteria and anti-cheats to every prior lever in this file -- only the READOUT changes, stacked on
the lane-best `--conj-select competitive` at its proven sweet-spot operating point, so any result is
attributable to the readout gate, not to a different bank:

- **task GO**: `beats_config_c_nogo` (per-seed `learn_spkwta_held >= 0.34 + 0.10`) **AND**
  `learning_load_bearing` (`learned - random >= 0.10`), each at **>=5/6 seeds**, under
  `--heldout-position --scramble-null`.
- **Verdict bands, fixed in advance:** `beat>=5/6 & lb>=5/6` = GO. Some (>0) beats/lb short of 5/6 = PARTIAL.
  `beat0 & lb0` = NO-GO for this lever.
- **Read honestly, not by the letter of the band:** the number to beat is the competitive-selection default's
  own `beat4/6-lb6/6` (`RATE_lin_ceiling_held` 0.4288) -- a PARTIAL at or below that is a disappointment, not
  progress, even though both are technically "PARTIAL."

## The decisive result -- a regression, not progress

<!--derived-->
Same scale/op-point as the competitive-selection decisive run (`conj_n=1152`, `n_s2=96`,
`--heldout-position --scramble-null`, `--ridge 0.5`, `--conj-select competitive --conj-select-overcomplete 4
--conj-select-kwta-frac 0.1`), `--readout attention-gated --attn-kwta-frac 0.5` the only mechanism change:

| quantity | competitive-selection (baseline) | **attention-gated (this run)** |
|---|---|---|
| `overall_verdict` | `PARTIAL-beat4/6-lb6/6` | **`PARTIAL-beat2/6-lb3/6`** |
| `LEARNED_spkwta_held` (mean) | 0.4549 | **0.3507** |
| `RANDOM_spkwta_held` (mean) | (not tabulated; uniformly well below learned) | 0.2170 |
| `RATE_lin_ceiling_held` (mean) | 0.4288 | **0.4288 (identical -- confirms the bank/front-end/ceiling are untouched)** |
| `learning_load_bearing` (>=5/6) | 6/6 (perfect) | **3/6** |
| `beats_config_c_nogo` (>=5/6) | 4/6 | **2/6** |
| `capability_go` | (not the headline metric there) | 1/6 |
| `scramble_null_pass` | 6/6 | 6/6 |

<!--derived-->
Per-seed (`LEARNED_spkwta_held` attention-gated / competitive-selection baseline for the SAME seed / beat /
lb): seed 42: 0.4479 / 0.4479 (identical) / beat / lb; seed 43: 0.25 / 0.4688 (regression) / miss / miss;
seed 44: 0.4583 / 0.4375 (slight gain) / beat / lb; seed 100: 0.2292 / 0.3229 (regression) / miss / miss;
seed 101: 0.3958 / 0.5729 (regression) / miss / lb; seed 102: 0.3229 / 0.4792 (regression) / miss / miss.
4 of 6 seeds got WORSE than the identical bank's plain-linear read; only seed 44 improved and seed 42 was a
wash.

## The diagnosis, quantified per-seed (why this is a wall on the METHOD, not the capability)

<!--derived-->
`LEARNED_linscore_held` -- the plain, UNGATED signed-linear SCORE on the exact same trained discriminant and
the exact same spike C2 code (unaffected by `--readout`, since it always calls the original `_lin_score_pred`)
-- is the correct like-for-like comparison for isolating the gate's OWN causal contribution, holding the
front end, bank, and trained weights fixed: seed 42: gated 0.4479 vs ungated 0.4479 (no effect); seed 43:
0.25 vs 0.4688 (gate cost -0.2188); seed 44: 0.4583 vs 0.4167 (gate gained +0.0416); seed 100: 0.2292 vs 0.3229
(gate cost -0.0937); seed 101: 0.3958 vs 0.5625 (gate cost -0.1667); seed 102: 0.3229 vs 0.4583 (gate cost
-0.1354). In 4 of 6 seeds the hard k-WTA gate actively DESTROYS accuracy relative to reading the exact same
population without gating -- this is not a spiking-port artifact confound (the linscore comparator is spike-
code-based, not rate-based) and not a bank/front-end confound (`RATE_lin_ceiling_held` is bit-for-bit
identical to the baseline run). The gate itself, at this operating point, is the thing hurting the read.

<!--derived-->
**Why a hard k-WTA is structurally the wrong tool here.** This runner's own module docstring already diagnoses
the lane's representational character: the C2 code carries "a fine DISTRIBUTED cosine modulation... across-
template std ~0.042 on a common-mode ~0.80" -- the reason a SIGNED, POPULATION-WIDE linear readout (reading
every unit, weighted, never discarding any) was the mechanism that first cleared the config-C NO-GO floor at
all. Zeroing HALF the population per trial (`--attn-kwta-frac 0.5`) throws away exactly the kind of small,
broadly-distributed contribution this code's own signal is made of -- the top-down template `A_c` upweights the
class's LARGEST-magnitude units, but a distributed code's discriminating information does not live only in its
largest-magnitude units; many small, correlated contributions across the discarded half evidently mattered.
This is a diagnosis about the GATE'S FUNCTIONAL FORM (hard elimination), not about attention-gating as a
concept.

## The next rung (no-defer -- named, not run here)

<!--derived-->
Per the wall-reframe question (CLAUDE.md: "what does the real system run alongside this that we replaced with
a constant?"), the missing companion process is not competition itself (already reused correctly at bank-
selection time) but GRADED gain, not hard elimination, at read time: a real attentional gain field multiplies
responses by a continuous factor (never fully zeroing a unit that still carries some signal) rather than
imposing a binary keep/discard cutoff -- the Reynolds & Heeger (2009)-style normalization-model-of-attention
form (`response = drive * attention_gain`, no hard threshold) that this project's existing `_apply_s2_norm`
satdiv machinery already implements ONE level up in this same file for a different population. The named next
rung is therefore a SOFT/graded attention gain (multiplicative reweighting proportional to `A_c`, no k-WTA
elimination) as the next mechanism variant, not another sweep of `--attn-kwta-frac` on the same hard-
competition functional form -- a sweep of the SAME wrong tool is unlikely to reach >=5/6 when 0.5 already
destroys signal in 4/6 seeds and the direction (more elimination = worse) is not expected to reverse at a
different cutoff. Mechanism BANKED at this operating point/functional form; the biological grounding
(`research/biology/attention-gated-readout.md`) and the capability (a genuinely class-and-trial-varying read)
are NOT abandoned.

## External verification (deep-research-at-wall gate; this is the 3rd finding in this lane within 3 days)

<!--derived-->
This lane (triple-order NO-GO, competitive-selection sweep-exhausted, now this) crossed the `gates/deep_
research_at_wall` threshold (>=3 findings in one lane within 3 days), so a real external-literature check was
required and run before this finding was committed. Reynolds & Heeger (2009), "The Normalization Model of
Attention," *Neuron* 61:168 (https://pubmed.ncbi.nlm.nih.gov/19186161/) directly confirms the diagnosis above
from the primate physiology side, independent of this arc: a single operation -- multiplying stimulus drive by
a GRADED attentional gain field, THEN divisive normalization -- reproduces the full spectrum of measured
attentional effects, INCLUDING WTA-like suppression of unattended distractors, without any explicit hard
elimination step. WTA-like behavior EMERGES from graded gain + normalization; it is not implemented as a binary
keep/discard competition. This is direct external support for the named next rung (a soft/graded gain field, not
another hard-k-WTA-frac sweep) and against re-trying the same hard-elimination functional form at a different
cutoff.

## Reproduce

```bash
# byte-identical-off proof (both required; vlin_attngated_frac1_smoke.json matches
# vlin_competitive_smoke.json exactly except elapsed_seconds):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --seeds 42 --n-s2 24 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 96 --conj-offset-max 2 --readout attention-gated \
    --attn-kwta-frac 1.0 --n-pos-total 4 --n-ex 2 --n-glimpses 1 --heldout-position --scramble-null \
    --out research/findings/raw/lanes/perception/vlin_attngated_frac1_smoke.json

# tiny non-degenerate smoke (seconds, sanity only):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --seeds 42 --n-s2 24 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 96 --conj-offset-max 2 --readout attention-gated \
    --attn-kwta-frac 0.5 --n-pos-total 4 --n-ex 2 --n-glimpses 1 --heldout-position --scramble-null \
    --out research/findings/raw/lanes/perception/vlin_attngated_smoke.json

# the decisive 6-seed run reported above:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 1152 --conj-offset-max 4 --readout attention-gated \
    --attn-kwta-frac 0.5 --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_attngated_n1152_heldoutpos_scramblenull_6seed.json
```
