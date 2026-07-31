---
type: plan
status: live
date: 2026-05-20
---

# Per-regime metacognitive monitors: a constructive design grounded in the doubly-dissociable parallel metamemory streams that biology actually uses to gate compositional readouts

**Status:** Design (autonomous; no hand-back). Supersedes the
Pirazzini-reference stage, which closed as an honest negative AND
established the **triple convergent ceiling** across three biology-
distinct compositional architectures (Stage-1 static; SPEAR
rhythm-multiplexed synaptic_gain; Pirazzini disinhibition + correct-
Hasselmo-ACh-via-excitability_drive). This is a constructive design
not for another compositional mechanism but for the **threshold-vs-
mechanism reframe the convergent ceiling itself points at**.

**Date:** 2026-05-20

**Plain-language commitment:** ordinary scientific terms, each
defined once; no internal codenames or letter-number labels are
load-bearing; catalog identifiers in parentheses for traceability.

---

## 1. Why this direction, and what the triple convergent ceiling taught

Three biology-distinct compositional architectures decisively FAILed
at biological scale (Stage-1 static composition; SPEAR rhythm-
multiplexed synaptic_gain modulation; Pirazzini disinhibition with
correct-Hasselmo-ACh-via-excitability_drive). In all three: composed
read-out at `lang_output` never exceeded the calibrated no-confab
threshold (650; encoded direct-retrieval ~796 vs control max ~584)
for compositional queries; the trustworthy abstention moat held
robustly in every architecture (`abstain_correct = 1.00` across all
seeds, loads, and ablation arms); each named mechanism was
independently mechanistically active (SPEAR 14.15 mV bridge-state
divergence between encode/retrieve phases; Pirazzini 13.93 mV
divergence with theta on/off at the same ACh neutral setpoint).

The convergence rules out a whole class of candidate fixes (more
rhythm; different binding mechanism; different encoding scheme) AND
points sharply at the next biology-faithful direction at its **root**:
**the direct-retrieval-calibrated trustworthy-abstention threshold
itself is the rate-limiting factor for compositional readouts** --
not the choice of compositional mechanism. The brain demonstrably
does not abstain on every compositional question; it answers
compositionally with confidence below the direct-recall threshold,
because it uses *different* thresholds per memory regime.

## 2. Genuine biological investigation (done first, with citations)

Biology has **separate, doubly-dissociable parallel metacognitive
monitors** for different memory regimes:

- Miyamoto et al. 2017 *Science* [1] (the same paper cited in
  Stage-1's design §2 for parallel metamemory streams): a whole-brain
  fMRI search identified neural correlates of metamemory for
  temporally **remote** events in prefrontal area 9 (or 9/46d), and
  for **recent** events within area 6. **Reversible inactivation of
  each of these loci induced doubly-dissociated selective
  impairments** in metacognitive judgment performance on remote or
  recent memory, without impairing recognition itself. The brain has
  parallel metamemory streams supervising recognition networks for
  different memory regimes with *regime-appropriate* thresholds.
- Confidence integrates additional evidence *after* perceptual
  commit [2][3] -- the abstain-vs-answer decision uses a different
  signal stream than the decision itself, supporting the design
  separation between the readout's primary decision and a
  per-regime confidence monitor.
- Delay-period prefrontal persistent activity reflects confidence
  [4] -- the project's `dlpfc_verb` region with NMDA bistability
  (already validated) is a biology-faithful substrate for a
  prefrontal confidence-monitor head.
- Tuned inhibition / tuned normalisation in perceptual decision
  circuits explains how the same neural population can produce
  apparently-suboptimal confidence behaviour relative to its raw
  accuracy [5][6] -- supports the design idea that
  regime-appropriate thresholding (rather than mechanism redesign)
  is the right knob to turn.
- Spiking implementations of meta-learning + prefrontal working
  memory exist as precedent [7][8][9] -- not the specific
  doubly-dissociable monitor architecture, but useful evidence the
  bio-grounded SNN substrate can host meta-level monitoring without
  protected-module modifications.

The unifying biological picture: **a primary readout (validated v14/
v16 lang_output cosine_to_word path) produces confidence per word; a
per-regime metacognitive monitor head (rooted in dlpfc_verb persistent
activity) evaluates whether that confidence is *adequate for the
regime of the query*, with regime-appropriate calibrated thresholds;
the abstain-vs-answer decision is routed through the appropriate
monitor per query type.**

## 3. The reframe (explicit)

Unit of analysis is no longer "which compositional mechanism produces
high-enough confidence to exceed a single calibrated threshold"
(triple-convergent ceiling closes that line). Instead: **the brain
has separate calibrated thresholds per memory regime, and the
abstain-vs-answer decision uses the regime-appropriate threshold**.
The compositional readout from the project's validated subsystems
already produces meaningful confidence -- it just sits below the
direct-retrieval-calibrated 650 threshold by construction.
Per-regime monitors give it a second-look threshold appropriate
to compositional retrieval, while leaving direct retrieval gated
by the original 650.

There is **no necessity partition** anywhere; the original frozen
verdict modules + the no-confabulation moat + every prior verdict
module stay byte-unchanged. The single net-new test instrument is a
new pre-registered fixed-bar capability-verdict module for the
per-regime monitor, with a decisive built-in control:
single-threshold-applied-uniformly must score FAIL.

## 4. Inventory of validated subsystems to reuse byte-unchanged

Reuse-by-import only; no edits to any protected/frozen/validated
module or the no-confabulation moat:

- **The validated v14/v16 substrate + concept pools + hippocampus +
  dlpfc_verb** -- the Stage-1+SPEAR+Pirazzini-cleared substrate
  build via `build_biological_brain_regions(...,
  enable_hippocampus_consolidation=True, enable_dlpfc_verb=True)` +
  global `enable_nmda=True`. The dlpfc_verb region's NMDA bistability
  IS the biology-faithful substrate for the per-regime monitor head.
- **The validated readout path** -- `lang_output_pattern_during_*`
  + `cosine_to_word` + raw-firing-rate confidence (the Stage-1+SPEAR-
  cleared `_ranked_from_pattern` formula) produces the primary
  per-word confidence. The monitor head reads this AND the dlpfc_verb
  persistent activity.
- **The no-confabulation abstention moat: BYTE-UNCHANGED.**
  `abstention_gate.py` (`gate(ranked, 650.0)`; `DEFAULT_THRESHOLD =
  650.0`; 7/7 tests) stays exactly as-is. It remains the
  direct-retrieval regime monitor.
- **The compositional-regime monitor: a NEW calibrated module
  ALONGSIDE the existing one.** New file `research/runners/abstention_gate_compositional.py`
  with its own threshold constant (e.g. `COMPOSITIONAL_THRESHOLD =
  TO_BE_CALIBRATED_pre-registered`), its own 7-case adversarial
  test matrix mirroring `tests/test_abstention_gate.py`, and a
  thin routing layer in the new runner that selects the
  appropriate monitor per query type (direct -> existing 650 gate;
  compositional -> new gate). The existing `abstention_gate.py` is
  NOT touched.
- **All prior validated subsystems** (engram API, replay-
  consolidation, neuromodulator subsystem with excitability_drive
  scope=group:NAME, every frozen `*_core.py`, sim/bridge.py, etc.)
  stay byte-unchanged.

## 5. What is genuinely net-new (bounded precisely)

Two pieces of net-new code, both in the project's `research/runners/`
namespace, neither touching any protected file:

1. **A new compositional-regime abstention gate module**
   `abstention_gate_compositional.py`: standard library + typing
   only; its own pre-registered fixed `COMPOSITIONAL_THRESHOLD`
   calibrated separately from the direct-retrieval 650; mirrors the
   existing `abstention_gate.py` discipline exactly (instrument-
   validity first; malformed -> safe "cannot conclude"; threshold
   immovable; 7-case adversarial matrix in
   `tests/test_abstention_gate_compositional.py`). Calibration of
   the threshold is itself a pre-registered separate step --
   measured on a representative compositional ground-truth set, NOT
   tuned to PASS the verdict.
2. **A per-regime metacognitive-monitor runner** that builds the
   validated substrate, runs queries through the validated readout
   path, and routes each answer through the appropriate gate per
   query type. The compositional path uses `abstention_gate_compositional.gate`;
   the direct path uses `abstention_gate.gate` unchanged. The runner
   exposes BOTH regimes simultaneously so a single query can be
   evaluated under both, and the dlpfc_verb persistent-activity
   integration is the biology-faithful confidence-signal-augmentation
   layer feeding the compositional gate.

## 6. Three concrete architectures, honest ceilings, falsify-cheaply-first

- **A -- Minimal per-regime threshold separation (RECOMMENDED first;
  cheaply de-riskable).** A new compositional gate with a fixed
  pre-registered threshold calibrated on a small representative
  compositional ground-truth set; primary readout via the validated
  path; query-type routing in the runner. The decisive built-in
  control: a single-threshold-applied-uniformly variant must score
  FAIL (the per-regime separation must be the differentiator).
  Smallest net-new (one gate module + one routing controller).
- **B -- A + dlpfc-resident confidence-signal augmentation.**
  The compositional gate's confidence input includes a contribution
  from dlpfc_verb persistent-activity strength (the
  delay-period-activity-reflects-confidence finding [4]); the
  biology-faithful prefrontal monitor signal augments the
  cosine-to-word rate. Staged only if A passes.
- **C -- B + doubly-dissociable causal-inactivation control.**
  A faithful test that selectively inactivating the dlpfc_verb
  contribution to the compositional gate (mimicking Miyamoto 2017's
  reversible-inactivation experiment) impairs compositional
  metamemory specifically while leaving direct-retrieval metamemory
  intact -- the doubly-dissociable signature. Staged only if B passes.

**Recommendation:** build A first under a pre-registered fixed-bar
gate whose decisive control is the single-threshold-uniform variant.
B and C are pre-registered staged follow-ons.

## 7. Pre-registered gate, falsify-cheaply-first, anti-cheat

- **New frozen capability-verdict module** (its own file; standard
  library + typing only; does NOT import or change any existing
  verdict module or the moats). Fixed constants set now, never
  tuned: `_PR_FULL_MIN` (per-regime monitor compositional accuracy
  bar); `_PR_UNIFORM_CTRL_MAX` (single-threshold-uniform-variant
  control; must collapse, the capability must be attributable to
  the per-regime separation); `_PR_DIRECT_RETAIN_MIN` (direct
  retrieval continues to clear at 650; the per-regime monitor must
  NOT degrade direct retrieval); `_PR_ABSTAIN_CORRECT_MIN`
  (trustworthy property must hold under composition); fixed load
  ladder; min seeds 3; scale tolerance.
- **The decisive built-in control:** a single-threshold-applied-
  uniformly arm (the compositional gate's threshold set to 650 like
  the direct gate) must reduce the system to the triple-convergent-
  ceiling architecture and collapse. The capability must be
  attributable to the per-regime threshold separation -- a
  same-threshold solver structurally cannot score PASS.
- **The COMPOSITIONAL_THRESHOLD calibration is a separate pre-
  registered step.** Calibrated on a held-out compositional
  ground-truth signal (encoded compositional ~X vs control ~Y);
  the calibration value is set BEFORE the decisive verdict run and
  becomes a frozen constant for that run. Calibration changes
  retroactively would itself be goalpost-moving; the calibration
  process is recorded, the resulting threshold is frozen.
- **Falsify-cheaply-first** NumPy precursor: a small synthetic
  benchmark of the gate's routing logic before any GPU spend.
- **Anti-cheat (non-negotiable):** mandatory smell-test scrutinising
  a nominal PASS HARDER than a FAIL (the triple-convergent ceiling
  means a fourth-architecture PASS must clear an especially-
  skeptical review); a dedicated adversarial review of the
  net-new gate module + the net-new runner + the verdict module
  BEFORE the no-harm phase (primary mandate: is the capability
  genuinely emergent from the per-regime threshold separation vs
  a calibration artifact; is the single-threshold-uniform control
  faithful; can a degenerate / over-permissive gate score PASS;
  are the bars movable; any autograd; subsystems byte-unchanged);
  controller trust-but-verify with the full protected set
  byte-empty; both abstention gates (the existing 650 + the new
  compositional) + their tests byte-identical and 7/7; GPU/CuPy
  for every decisive run (NumPy only for the smoke); honest
  propagation of every outcome both remotes.

## 8. Honest ceiling (stated up front, never spun)

A clean scrutinised success = a biology-grounded per-regime
metacognitive-monitor architecture shows that the project's
existing compositional readout (which produces meaningful confidence
below the 650 direct-retrieval threshold) is correctly routed to a
regime-appropriate threshold above which compositional queries are
answered, while the direct-retrieval gate stays at 650 unchanged
AND the trustworthy property holds under composition (zero
confabulation). This is biology-faithful (Miyamoto 2017's
doubly-dissociable parallel metamemory streams; tuned-inhibition
confidence behaviour [5][6]) AND directly addresses the triple-
convergent ceiling at its root.

Explicitly NOT fluent open-ended language, NOT an LLM, NOT a
threshold-relaxation that defeats the trustworthy property. The
genuine durable contribution of this stage, regardless of outcome,
is a faithful, adversarially-hardened, fixed-bar test of whether
per-regime metacognitive monitoring -- biology's actual answer to
the threshold-vs-mechanism trade-off the triple-convergent ceiling
empirically localised -- yields the capability the single-threshold
abstention could not.

## 9. References

[1] [Causal neural network of metamemory for retrospection in primates](https://www.science.org/doi/10.1126/science.aal0162) (Miyamoto et al., 2017, Science)
[2] [Confidence integrates evidence after the perceptual decision -- separable neural signatures](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8423440/) (separable neural signatures of confidence during perceptual decisions)
[3] [Using artificial neural networks to reveal the human confidence computation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013827) (PLOS Comp Bio, 2025)
[4] [Working Memory in the Prefrontal Cortex](https://pmc.ncbi.nlm.nih.gov/articles/PMC5447931/) (working memory + delay-period activity reflecting confidence)
[5] [Tuned inhibition in perceptual decision-making circuits can explain seemingly suboptimal confidence behavior](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8032199/) (tuned inhibition explaining suboptimal confidence behaviour)
[6] [Tuned normalization in perceptual decision-making circuits can explain seemingly suboptimal confidence behavior (bioRxiv)](https://www.biorxiv.org/content/10.1101/558858.full.pdf)
[7] [Meta-Learning in Spiking Neural Networks with Reward-Modulated STDP](https://arxiv.org/pdf/2306.04410) (bio-plausible meta-learning in SNNs)
[8] [Synapses mediate the effects of different types of stress on working memory: a brain-inspired spiking neural network study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11961926/) (brain-inspired SNN working-memory study)
[9] [Unsupervised Spiking Neural Network Model of Prefrontal Cortex to study Task Switching with Synaptic deficiency](https://arxiv.org/pdf/2305.14394)

---

**Next:** writing-plans for this design (Task 0 pin; Task 1 the new
frozen capability-verdict module fully specified + adversarial
matrix; Task 2 the new compositional gate module + its test matrix;
Task 3 the net-new per-regime-monitor runner + tests; Task 4
dedicated adversarial review; Task 5 no-harm; Task 6 controller-
only decisive run + smell-test + honest propagation), then
subagent-driven-development under the pre-registered fixed-bar
gate, honest propagation of every outcome to both remotes,
iterating following the biology -- autonomous, no hand-back.
