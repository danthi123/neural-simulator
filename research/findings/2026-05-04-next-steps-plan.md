# Next steps after B3 v2 verdict

**Date:** 2026-05-04 22:55 EDT
**Context:** B3 v2 (gradient + cortical canon) showed 5/8 aligned with
3-factor `bio_grad_with_topo_fs` at 3/3 perfect. Last seed in flight
(ETA ~23:25 EDT). Final verdict will be 5-6/9 aligned — clearly
"gradient_works" territory.

---

## Decision tree once B3 v2 final verdict lands

### Path A — B3 final = "gradient_works" (most likely)

Three queued experiments, in order:

**1. Multi-seed validation** (~7 hours, low-risk)
- `experiments/bio_b3_validation.yaml` — 6 seeds × 3 conditions
- Doubles n from 3 → 6 per condition
- Confirms whether `topo_fs 3/3` was robust or lucky
- Run when GPU free; same parallel=2 setup

**2. Three-factor learning rule** (~7 hours, the SCIENTIFIC TEST)
- `experiments/bio_three_factor.yaml` — same architecture as B3
- Replaces gradient with biology-grounded rule:
  `Δw = lr × eligibility[pre,post] × dopamine_sign[motor]`
- Frémaux & Gerstner 2016 framework
- **The headline question:** can a biology-plausible rule do what
  gradient does? If yes, the project's continued bio-fidelity
  research is validated. If no, need fundamentally different
  framework.

**3. (Conditional on #2 success)** Apply findings to v2 architecture
- Modify g11_bg_runner.py motor pools to enable cortical canon
  (`internal_density=0.10`, `exc_fraction=0.8`, NMDA on)
- Re-run text_eval_embodied with the existing v2 cascade
- Tests: was the 28.5% W→A claim from May 2 actually real, or
  was it being suppressed by canon-less motor pools?

### Path B — B3 final = "partial" (3 aligned)

Same first two steps, but interpret partial as "training-dose ceiling
or sparse-code overlap":

**1. Multi-seed validation** — confirm 3 isn't lucky
**2. (Skip 3-factor; head straight to)** Sparse codes test:
   - `--token-sparsity 0.05` (~2-3 word overlap vs ~6-9 at 0.10)
   - 6 seeds × 2 conditions (vanilla vs topo_fs) × 0.05 = 12 runs
**3. Three-factor only after sparse-code result clarifies whether
   it's a learning-rule or representation issue.

### Path C — B3 final = "gradient_fails" (≤2 aligned)

Unlikely given current 5/8, but possible. Then:

**1. Investigate why gradient failed** at bio scale
   - Maybe push_to_gpu_every=64 is too coarse → try 16
   - Maybe expected_max_firing_per_neuron miscalibrated → measure
     actual firing rates and recalibrate
**2. (Skip 3-factor for now)** — if gradient can't do it, biology-
   plausible rules definitely can't.
**3. Architecture investigation** — token-sparsity, larger motor
   pools, alternative drive currents.

---

## Pre-staged artifacts (committed 88f121d)

| File | Purpose |
|---|---|
| `experiments/bio_b3_validation.yaml` | 6-seed B3 validation, 18 runs / parallel=2 / ~7 hr |
| `research/runners/bio_three_factor.py` | Three-factor learning rule runner |
| `experiments/bio_three_factor.yaml` | 3-factor sweep, 18 runs / parallel=2 / ~7 hr |
| `research/result_aggregator.py` | Built-in configs: bio_b3_gradient, bio_b3_validation, bio_three_factor |

---

## Why three-factor specifically?

Of the three biology-grounded frameworks in modern neuroscience:

**(a) Three-factor with eligibility** (Frémaux & Gerstner 2016):
- Local computation: each synapse only uses pre × post × scalar
- Eligibility = NMDA-mediated calcium decay (~50-200ms biological tau)
- Dopamine = global RPE signal from VTA/SNc
- Already implemented in our codebase as STDP+R-STDP — but we use
  it with paired-stim training (every event +1 reward), which is
  weak supervision. Three-factor with TARGETED supervision
  (sign-of-error per motor pool) is what we test here.

**(b) Apical-basal dendritic feedback** (Bono & Clopath 2017):
- Pyramidal neurons have apical dendrites that receive top-down
  prediction error
- More biological but requires multi-compartment neurons
- Not implemented in our point-neuron model — would be major
  architectural change

**(c) Predictive coding** (Rao & Ballard 1999):
- Each region predicts its inputs; error drives learning
- Different network organization than ours (paired generative +
  recognition pathways)
- Not implemented — different fundamental structure

**Why three-factor first:** smallest implementation effort,
clearest comparison vs gradient, leverages existing simulator
infrastructure.

If three-factor fails AND gradient succeeded, the sequence (b)
or (c) becomes the next research direction.

---

## Resource budget

If both validation + three-factor run sequentially: ~14 hours total
GPU time. Well within overnight budget.

If both confirm + extend to v2: ~20 hours total — 1.5 days at
parallel=2. Acceptable for a major scientific arc.

---

## After all three queued steps

If B3 + 3-factor + v2 all align ≥4/6: the W→A learning question is
answered. Project pivots to:
1. Multi-task learning (NESW + 8 directions)
2. Compositional language ("go north then east")
3. Real-image visual cortex pathway
4. Multi-modal integration (vision + language joint)

If 3-factor fails but B3 confirms: project pivot to:
1. Implement apical-basal dendritic learning (architectural change)
2. Or predictive coding pathway (alternative architecture)

If B3 itself doesn't validate (5/8 was lucky): more architectural
work needed first.
