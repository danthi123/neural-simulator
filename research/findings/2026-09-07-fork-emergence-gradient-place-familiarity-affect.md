---
type: finding
status: live
date: 2026-09-07
tags: [agi-fork, emergence, affect, self-awareness, familiarity, place-cells, predictive-substrate]
verdict: A bare rate-recurrent self-supervised predictive substrate grows faculties in proportion to how inherent each is in the predictive task — place emerges strongly, familiarity weakly, affect not at all.
---

# The predictive-substrate emergence gradient — place emerges, familiarity weakly, affect not at all (AGI-fork)

## Question

The AGI fork (branch `agi-fork`, `sim/pcs_substrate.py`) is ONE rate-recurrent self-supervised
predictive substrate (JEPA next-latent prediction + reward/value/SR heads + a REINFORCE policy). The
owner steered it (2026-09-07) to advance the MAIN project goals — an affective, self-aware world-model —
with looser biology. This finding asks the sharp version of that: **which cognitive faculties EMERGE
from the bare predictive substrate on their own, and which do not?** Measured on three: spatial (place),
affect (valence), and self-awareness (familiarity).

## Instrument (the same sound one throughout, earned by the place-cell arc)

Each faculty is read out post-hoc and scored by a pre-registered GO that compares a TRAINED substrate
against an UNTRAINED reservoir replayed on identical inputs (`replay_untrained`, seed+777) — the
base-control that makes any separation attributable to LEARNED structure, not to the signal being
trivially present in the recurrence. Soundness uses split-half stability and a CYCLIC-SHIFT null (not the
statistically-invalid i.i.d. shuffle — see `research/FAILURE_LOG.md` 2026-09-07), plus `attributable_to`
(trained effect must exceed 2x untrained). Probes: `research/runners/_fork_pcs_emergence_derisk.py`
(place), `_fork_pcs_valence_probe.py` (affect), `_fork_pcs_familiarity_probe.py` (familiarity).

## Results

<!--derived: counts + AUROC/stability values read directly from the cited raw artifacts below-->

**PLACE — STRONG emergence (6/6).** Split-half rate-map stability ~0.8 (trained) vs ~0.3
(untrained-reservoir), 6/6 seeds, in BOTH the aux-loc and the PURE-prediction base control (no position
supervision). Artifacts: `research/findings/raw/_fork_pcs_base_poolseed42.json (seeds 42-102 in the set)`
(base, no aux-loc: stability 0.76-0.88 trained vs 0.25-0.41 untrained). Place emerges from prediction
ALONE.

**AFFECT / valence — NO emergence (0/6, four configurations).** Three methods, all pre-registered NO-GO:
- Instantaneous reward-PE readout: 0/6 (`research/findings/raw/_fork_pcs_valence_presence_seed42.json`) — the untrained
  reservoir decodes it as well (rpe = reward − reward-head-prediction is a trivial LINEAR read of h_t).
- Integrated "mood" (leaky-EMA of reward-PE): 0/6 (`research/findings/raw/_fork_pcs_valence_mood_seed42.json`) — the untrained
  reservoir's intrinsic memory carries the integrated signal too.
- A dedicated valence-FORECAST objective (predict discounted future reward; additive default-off
  `valence_weight`, gradcheck-clean): 0/6 at weight=1.0 (`research/findings/raw/_fork_pcs_valence_forecast_seed42.json`) and
  0/6 at weight=2.0 (`research/findings/raw/_fork_pcs_valence_forecast_w2_6seed.json`). The objective's own target is
  TAUTOLOGICALLY installed (h_t is directly optimized to linearly encode it), so it is not evidence of
  emergence either.

**FAMILIARITY / self-awareness — WEAK, inconsistent emergence (2-3/6).** AUROC of the substrate's own
one-step prediction error discriminating familiar vs marginal-matched scrambled-novel V1 inputs, trained
vs untrained reservoir. `research/findings/raw/_fork_pcs_familiarity_seed42.json` (h=128, seeds 42-102): 2/6. `research/findings/raw/_fork_pcs_familiarity_h512_6seed.json`
(h=512): 3/6. Trained beats untrained on 4/6 seeds (trained AUROC ~0.65-0.75 vs untrained ~0.34-0.79),
but only 2-3 clear the 0.65-bar + 2x-attributable; aggregate GO is FALSE at both scales. A 1-seed smoke <!--derived-->(0.999 vs 0.54) badly over-suggested — barely-trained tiny-scale gives an artificially clean split, which
is why the probe gated it "smoke doesn't decide emergence".

## The mechanism (why, not just that)

The three outcomes form a GRADIENT that tracks a single variable: **how INHERENT the faculty is in the
predictive task itself.**
- **Spatial structure is REQUIRED to predict observations** — you cannot predict your next egocentric
  view without an estimate of where you are (path integration), so a place code is forced into being by
  the prediction objective. Strong emergence.
- **Prediction-error IS novelty** — a trained substrate predicts familiar (in-distribution) inputs well
  and novel ones poorly, while an untrained reservoir predicts everything badly. So a familiarity signal
  is PARTIALLY inherent (it exists wherever prediction is learned) but weak, because the untrained
  reservoir's random dynamics also respond differently to some inputs, and the substrate partially
  generalizes to some scrambles. Weak emergence.
- **Valence is an EXTERNAL scalar** with no role in predicting observations. So every valence signal is
  either trivially present in the substrate's own dynamics (a linear read, or reservoir memory) or
  tautologically installed by an added objective — never grown by the predictive task. No emergence.

## Implication

This VALIDATES the main project's architecture choice: affect is built there as a DEDICATED spiking organ
(a valence forward-model), not expected to emerge — and this finding shows WHY that is the correct call.
It also matches the neuroscience: affect lives in dedicated subcortical/limbic structure, not in cortical
prediction. The design corollary for the fork: **faculties NOT inherent in prediction need dedicated
structure; do not expect them to emerge from the predictive substrate.**

## Honest caveats / what this does NOT show

- Familiarity's weakness may be partly a PROBE-design artifact: the novel set is an input-level V1
  scramble, which the trained substrate can partially predict; a STRUCTURAL / held-out novelty (a novel
  object type, a violated temporal contingency) might reveal a cleaner signal. Untested — banked as the
  next lever.
- LOAD-BEARING is not tested for any faculty here (this is presence only). A signal being present +
  attributable is necessary but not sufficient for a functional faculty; whether even the weak
  familiarity signal DRIVES behavior (confidence-gated action) is the deferred follow-on.
- "Emergence" here means "a trained-vs-untrained-reservoir gap on the sound instrument", never a
  phenomenal/felt claim. All read-outs are functional (honesty boundary).
