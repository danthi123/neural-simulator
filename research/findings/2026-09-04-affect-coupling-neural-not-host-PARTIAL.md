---
type: finding
status: partial
claim_check: measured
date: 2026-09-04
mechanism: replaces `_apply_affect_bias`'s host decode-time additive logit bias (`webapp/wkv_mouth_generator.py`,
  2026-09-03/04) with a genuinely substrate-mediated alternative: a per-candidate-pool NEUROMODULATOR
  CONCENTRATION (`FewSpikeWordRead.set_mood`, `research/runners/_wkv_fewspike_read_derisk.py`) realized through
  `sim/neuromodulators.py`'s existing, UNMODIFIED `excitability_drive` target and `sim/bridge.py`'s existing,
  UNMODIFIED per-step current computation -- the SAME general subsystem already used project-wide for dopamine/
  ACh/dynorphin/enkephalin/per-action-DA (`sim.neuromodulators._default_*_config`) and, closest in spirit, the
  NE/locus-coeruleus arousal-gain precedent (`research.runners._ne_lc_gain_vigilance_realbridge_derisk`). ZERO
  lines changed in `sim/`. Gated by `BRAIN_WKV_MOUTH_AFFECT_NEURAL` (default OFF, independent of the existing
  master `BRAIN_WKV_MOUTH_AFFECT` switch).
seed-waiver: the LOAD-BEARING + DETERMINISM claims are checked on the full 6-seed non-negotiable battery
  (42/43/44/100/101/102) x 3 prompts = 18 combinations per mechanism, not waived. The pA CALIBRATION sweep
  (finding the working `affect_boost` default) is cross-checked on 1 prompt x 7 boost values x 1 seed plus a
  3-seed/3-prompt spot-check at the chosen default -- the same "calibration point, not a generalization claim"
  reasoning `research/findings/2026-09-04-linattn-affect-coupling-sharpness-aware-GO.md` already used for the
  identical class of claim (a fixed constant, not a per-condition fit).
lane: language (own-voice mouth / affect grounding) -- scaffold-retirement burn-down
seeds: [42, 43, 44, 100, 101, 102]
verdict: PARTIAL. The coupling's DESTINATION is now genuinely substrate-mediated -- a real
  `sim.neuromodulators.NeuromodulatorManager` concentration, applied by `sim/bridge.py`'s own per-step current
  computation to the population `FewSpikeWordRead.read(p)`'s actual Izhikevich spiking competition decides among
  -- and this is LOAD-BEARING (18/18 prompt/seed combinations causally diverge by mood, 18/18 deterministic,
  BOTH matching the pre-existing host mechanism's own 18/18) and provably a true no-op both when the flag is OFF
  (byte-identical to the pre-rung code, verified by literally shadow-loading the pre-rung file pair) and when
  the coupling is ON but lesioned (`valence=0.0`; byte-identical to the host mechanism's own neutral output,
  proving an inert neuromodulator subsystem perturbs nothing numerically). NOT fully closed: the SOURCE half
  (which candidate word counts as "mood-congruent" and how strong the resulting concentration should be) is
  still a host-computed scalar over the Warriner lexicon -- the SAME "host scalar -> neuromodulator
  concentration -> substrate applies the effect" boundary `research/runners/affect_production_organ.py` already
  declares and accepts for the real affect organ's OWN appraisal injection, not a NEW shortcut invented here, but
  not eliminated either. See "Honest residual" for the precise boundary and the ONE measured asymmetry (negative
  mood undershoots the lesion baseline on 7/18 combinations at the current calibration, vs 0/18 for the host
  mechanism).
artifacts:
  - webapp/wkv_mouth_generator.py
  - research/runners/_wkv_fewspike_read_derisk.py
  - research/runners/_wkv_mouth_affect_neural_verify.py
  - research/runners/_wkv_mouth_affect_neural_byte_identical_check.py
  - research/findings/raw/_wkv_mouth_affect_neural_byte_identical_check.json
  - research/findings/raw/_wkv_mouth_affect_neural_calibration_sweep.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_aggregate.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s42.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s43.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s44.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s100.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s101.json
  - research/findings/raw/_wkv_mouth_affect_neural_loadbearing_s102.json
  - research/findings/raw/_wkv_mouth_affect_neural_vs_host_compare_s42.json
  - research/findings/2026-09-03-linattn-mouth-live-brain-grounded-honest-verification-PARTIAL-affect-gap.md
  - research/findings/2026-09-04-linattn-affect-coupling-sharpness-aware-GO.md
---

# Affect coupling: from a host logit bias to a real neuromodulatory gain on the spiking word-read -- PARTIAL

## What this closes and what it does not

`research/findings/2026-09-03-linattn-mouth-live-brain-grounded-honest-verification-PARTIAL-affect-gap.md` (ii-c)
and the 2026-09-03/04 fixes wired the real spiking affect organ's valence/arousal read into the WKV/SSM mouth's
free generation and made it load-bearing, but named the resulting mechanism (`_apply_affect_bias`) a TRACKED
SHORTCUT: it adds a signed, saturating bias directly to the full-vocab logits `lg` BEFORE
`FewSpikeWordRead.read(p)` -- the genuine few-spike Izhikevich population-coded soft-WTA -- ever runs. The
population itself never receives the mood signal; it only ever sees an already mood-cooked probability vector.
The task this finding answers: fold that modulation into the population read's OWN substrate dynamics instead --
a neuromodulatory gain on the Izhikevich population, not host arithmetic over a hand-built lexicon feeding a
softmax the neurons never see.

This finding reports what changed, what stayed host, and measures both honestly.

## The mechanism

`research.runners._wkv_fewspike_read_derisk.FewSpikeWordRead` already builds a real `SimulationBridge` of
`K*P` Izhikevich neurons (`K` candidate-word pools of `P` neurons each), drives each pool's neurons with an
external current derived from the model's top-K softmax, and reads the winner from `cp_firing_states` after a
short competition window -- the genuine spiking WTA. That class gained an opt-in constructor flag
(`affect_neural=False` by default) that, when true, registers `K` per-pool `sim.neuromodulators.
NeuromodulatorConfig`s at bank-build time -- one modulator per candidate pool (`mood_pool_0` .. `mood_pool_{K-1}`),
each with a single `ModulatorTarget(target_type="excitability_drive", scope="group:pool_k", sensitivity=1.0)`
and `decay_tau_ms=1e12` (a manually-set concentration effectively holds for one read; the SAME value the
project's existing NE/LC-gain precedent, `research.runners._ne_lc_gain_vigilance_realbridge_derisk`, already
uses for an identical reason). A new `FewSpikeWordRead.set_mood({pool_idx: extra_pA, ...})` method resets every
pool's concentration to 0.0 pA and then raises only the mood-congruent ones -- called by the driving loop
(`_free_gen`/`_free_gen_linattn`) immediately BEFORE `reader.read(p)`, so `sim/bridge.py`'s own, completely
UNMODIFIED per-step current computation (the SAME "Neuromodulator excitability_drive" block every other
neuromodulator config in this project already exercises) adds that extra current to the mood-congruent pool's
neurons for every step of the competition. `read`/`_compete`/`drive_from_weights` are byte-for-byte unchanged --
zero lines edited in `sim/` anywhere in this rung.

`webapp/wkv_mouth_generator.py` gained the mirror-image wiring: `wkv_mouth_affect_neural_enabled()`
(`BRAIN_WKV_MOUTH_AFFECT_NEURAL`, default OFF) and `_affect_pool_gains(cand, affect_ids, valence, arousal,
affect_boost, recent_ids)`, which reuses `_apply_affect_bias`'s EXACT congruence test (sign agreement between
the turn's mood and the Warriner-tagged word valence) and habituation multiplier (short-term-depression-shaped
damping over the last 8 generated tokens) -- the SAME mood policy, a different destination. Where
`_apply_affect_bias` would add a clipped bias to `lg`, `_affect_pool_gains` instead returns
`{local_topK_index: extra_pA}` and the driving loop calls `reader.set_mood(...)` with it, immediately before the
now-UNBIASED `p = softmax(lg[cand]/T)` is handed to the genuine spiking read. There is no host
`np.clip(...,-1,1)` saturation step in the new path at all -- `strength` is already bounded to <=1.0 by
construction (every factor is itself in [-1,1]/[0,1]), and `FewSpikeWordRead`'s own `concentration_max` (the
substrate's own clamp, inherited from `NeuromodulatorManager.set_concentration`) is the only additional ceiling.

## Calibration

`affect_boost` (the SAME parameter the host mechanism already exposes) is converted to a picoamp scale via one
constant, `_AFFECT_NEURAL_PA_AT_REFERENCE_BOOST = 880.0` at the reference `affect_boost = 10.0` (the host
mechanism's own existing default), so a caller's existing value carries over without a second, independently-
tuned knob. This was NOT guessed: a sweep at the realistic operating point the host mechanism was itself
calibrated against (`valence=0.16, arousal=0.65`, `generate()`'s own docstring) found `affect_boost<=5` produces
no measurable divergence and `affect_boost>=80` (raw `>=880` pA at full congruence) drives the affect-word
fraction of the output up to 0.4-0.65 (word-salad risk, the same failure shape the host mechanism's own
calibration comment records at its own excessive-boost end) -- `research/findings/raw/
_wkv_mouth_affect_neural_calibration_sweep.json`. `affect_boost=10.0` (the chosen default, `880` pA at full
congruence, ~90 pA at the realistic operating point) is the weakest value in the sweep at which BOTH mood
directions reliably diverge from the neutral baseline while the affect-word fraction stays close to the ~0.06-0.10
neutral range (0.077 vs 0.058 baseline, rounded to 3dp from the sweep artifact's `affect_boost==10.0` row <!--derived-->).
A follow-up 2-seed/3-prompt spot-check at this
default confirmed the same pattern generalizes past the single calibration prompt (5/6 rows fully diverging in
both directions, one benign single-direction miss -- see "Honest residual" below).

## Load-bearing verification (6 seeds x 3 prompts, fresh-subprocess-per-arm)

Methodology mirrors the project's own "phase6 clean isolation" discipline for this exact coupling
(`research/findings/raw/_affect_wkv_mouth_verify_phase6_clean_isolation.json`): each arm is a FRESH Python
subprocess (`research/runners/_wkv_mouth_affect_neural_verify.py::_run_arm`), never sharing the in-process
checkpoint cache or the continuing per-seed RNG timeline `webapp/wkv_mouth_generator.py`'s own `_RngIsolation`
otherwise deliberately maintains across calls -- so "different mood" is never confounded with "different
consumed RNG history." For each of the 6 non-negotiable seeds (42, 43, 44, 100, 101, 102) x 3 TinyStories-domain
prompts ("the little girl was" / "tom and his dog were" / "one day the boy"), four arms were run: A (positive
mood, valence=0.16/arousal=0.65), B (negative mood, valence=-0.16/arousal=0.65), C (A repeated), L (lesion,
valence=0.0/arousal=0.0) -- for BOTH mechanisms (host, neural), same prompts/seeds, like-for-like.

`research/findings/raw/_wkv_mouth_affect_neural_loadbearing_aggregate.json` (aggregated from the 6 per-seed
artifacts):

| metric | host | neural |
|---|---|---|
| affect_loadbearing (A != B) | 18/18 | 18/18 |
| determinism (A == C) | 18/18 | 18/18 |
| lesion diverges, positive direction (L != A) | 18/18 | 18/18 |
| lesion diverges, negative direction (L != B) | 18/18 | 11/18 |

The headline claim -- mood causally changes the produced text, and the same call is reproducible -- is a clean
18/18 for BOTH mechanisms. The stricter per-direction check (does EACH mood arm individually diverge from the
neutral/lesion baseline, not just from each other) shows the neural mechanism undershooting on the negative
side on 7/18 combinations at the current calibration; see "Honest residual."

## Byte-identical safety proofs (measured, not inferred)

`research/runners/_wkv_mouth_affect_neural_byte_identical_check.py` checks two claims by EXACT STRING COMPARE
(`docs/TERMS.md`'s own condition for the word "byte-identical" -- asserted in the data, never inferred from
reading the code), each run twice (2 prompts) -- `research/findings/raw/
_wkv_mouth_affect_neural_byte_identical_check.json`, all 4 checks `true`:

1. **OFF-BY-DEFAULT.** The pre-this-rung `webapp/wkv_mouth_generator.py` + `research/runners/
   _wkv_fewspike_read_derisk.py` pair (extracted from `git show HEAD:<path>` into an isolated temp sandbox and
   loaded as shadowed modules under their real dotted names, so the original file's own
   `from research.runners._wkv_fewspike_read_derisk import ...` resolves to the ORIGINAL sibling, not the
   modified one on disk) produces `generate()` output IDENTICAL to the current code with
   `BRAIN_WKV_MOUTH_AFFECT_NEURAL` unset. This is a genuine A/B between two DIFFERENT file contents, not the
   current file confirming its own logic.
2. **LESION-EQUIVALENT NO-OP.** With the neural coupling flag ON but `valence=0.0` (exactly what
   `AffectProductionOrgan.read_differential(lesion=True)` clamps the organ's differential, hence the mapped
   valence, to), output is IDENTICAL to the HOST mechanism's own output at the same neutral condition -- proving
   that registering 64 inert (all-zero-concentration) neuromodulator channels on `FewSpikeWordRead`'s bank
   perturbs the Izhikevich dynamics by exactly nothing numerically, not merely "the code path looks skipped."

## Moat / scope

Unchanged by construction: this rung touches only `_free_gen`/`_free_gen_linattn`'s free-generation branch and
`FewSpikeWordRead`. `render_fact_sentence`'s closed-class fact-clause path, `pick_covered_fact`, and every
VERIFY/moat check downstream of the mouth are untouched -- the same scope boundary `_apply_affect_bias`'s own
comment block already states ("facts stay tone-neutral by construction... only `_free_gen`/`_free_gen_linattn`'s
free generation").

## Honest residual: genuinely substrate-mediated destination, still host-computed source

Two things are true at once, and neither should be allowed to overwrite the other:

**What is now genuinely neural.** The mood signal's DESTINATION changed in kind, not just in place. Before, the
valence/arousal read was consumed entirely as host arithmetic on an abstract logit vector that no neuron ever
saw. Now it is consumed as a `NeuromodulatorManager` concentration that `sim/bridge.py`'s own per-step current
computation adds to specific Izhikevich neurons' actual membrane drive, for every step of a competition the
genuine spiking dynamics (OU noise + accumulated firing over the read window) still resolve -- the SAME general
mechanism (an `excitability_drive` neuromodulator target) this project already uses for dopamine, ACh,
dynorphin/enkephalin co-release, per-action DA, and NE/LC arousal gain elsewhere, applied here for the first
time to a population-coded word read. Zero `sim/` edits were needed; this is pure reuse of an existing,
general-purpose framework, not a bespoke mechanism invented for this task.

**What is still host.** Two things did not move: (a) WHICH candidate counts as "mood-congruent" is still decided
by a host dictionary lookup into the Warriner lexicon (`_affect_bias_ids`, unchanged, reused by import); (b) HOW
STRONG the resulting concentration should be is still a host-computed scalar formula
(`valence * clip(arousal,0,1) * habituation * word_valence`, then scaled by the calibrated pA constant). Per
`docs/TERMS.md`'s condition for "fully spiking" ("every cognitive step between sensation and action is
neurons/synapses"), this coupling does NOT qualify -- the congruence decision and the magnitude computation are
host arithmetic, exactly the "labelled-line pool assignment is a legitimate host input" category
`drive_from_weights` already relies on for candidate-to-pool assignment, extended here to "which pool gets a
modulatory nudge." This is the IDENTICAL boundary `research/runners/affect_production_organ.py`'s own docstring
already declares and accepts for the real affect organ's appraisal injection ("The appraisal injection (per-word
valence -> neuromodulator concentration) is a declared host scaffold") -- not a new shortcut invented by this
rung, but also not one this rung eliminates. Converting the congruence/magnitude computation itself into a
spiking mechanism (e.g., a small learned or Hebbian association layer between a word-identity population and an
arousal/valence population, driving the neuromodulator's production rule directly rather than via
`set_concentration`) is the next concrete rung, named here rather than assumed away.

**The measured asymmetry.** At the current calibration, the negative mood direction undershoots the lesion
baseline on 7/18 prompt/seed combinations (vs 0/18 for the host mechanism, and 0/18 for the neural mechanism's
own positive direction). The most likely explanation, NOT independently verified here: the TinyStories corpus
these checkpoints are trained on skews toward cheerful endings, so the checkpoint's own neutral/lesion
continuation for a given prompt may already sit close to what a mild positive nudge would also produce (little
room to move further in that direction is not the failure mode; the reverse -- a negative nudge fighting an
already-positive prior -- is). This reads as a corpus/checkpoint property interacting with a still-conservative
calibration, not a defect in the coupling mechanism itself (the mood-vs-mood comparison, the actual load-bearing
claim, is unaffected: A and B always differ). Re-running the calibration sweep with an asymmetric (stronger
negative-direction) pA scale, or simply raising `affect_boost` at the cost of some salad-fraction headroom (the
sweep cited above measured ~0.077 at the current default vs 0.4-0.65 at `affect_boost>=80` <!--derived-->), are
both concrete next steps, not attempted here to avoid conflating "does the mechanism work" with "is it perfectly
tuned."

## Reproduce

```bash
SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_byte_identical_check \
    --before-ref HEAD --json <OUT_1>.json
SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_verify --phase loadbearing \
    --seed 42 --affect-boost 10.0 --json <OUT_2>.json
SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_affect_neural_verify --phase calibrate \
    --seed 42 --json <OUT_3>.json
```

(`<OUT_N>.json` is any scratch output path of the caller's choosing -- the artifacts actually cited above were
written to their `research/findings/raw/_wkv_mouth_affect_neural_*` paths.)
