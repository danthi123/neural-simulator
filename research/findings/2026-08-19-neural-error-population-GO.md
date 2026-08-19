---
type: finding
status: live
date: 2026-08-19
mechanism: urbanczik-senn-dendritic-prediction
lane: brain-based-purity burndown (board #39 — the "was I wrong?" teaching error)
verdict: de-risk-GO-6seed
artifacts:
  - research/findings/raw/_neural_error_population.json
---

# The "was I wrong?" teaching error is a neuron's OWN somato-dendritic mismatch (Urbanczik-Senn), not a host formula — 6-seed GO

**Board #39 (BRAIN-BASED-ONLY burndown).** The read-out that learns the brain's word choices was trained by a delta
rule whose per-output error `err_j = est_j - target_j` is a HOST subtraction (a Python formula). Under the
brain-based-only standard a prediction error computed by host code is a documented SHORTCUT even when numerically
correct — the *brain* was not computing the error, the host bookkeeping was. This de-risk replaces the host
subtraction with a POPULATION OF TWO-COMPARTMENT NEURONS that works the error out itself via the Urbanczik-Senn
dendritic-prediction rule: each read-out neuron's basal dendrite predicts its own teacher-nudged soma, and the
neuron's intrinsic `(soma_rate - phi(dendritic_voltage))` mismatch IS the local teaching error that drives
plasticity. No host error formula anywhere in the loop.

**Verdict: GO, 6 seeds (42, 43, 44, 100, 101, 102).** The neural error drives the read-out delta rule as well as
the exact host error (NEURAL == HOST held-out generalization within noise), and every anti-cheat collapses.

**Runner:** `research/runners/_neural_error_population_derisk.py` · **Raw:**
`research/findings/raw/_neural_error_population.json` · CPU/numpy, NO `sim/` edit, additive only.

## What is distinct from the 2026-06-17 neural-error GO (this is NOT a re-derivation)

The 2026-06-17 `2026-06-17-neural-error-localrule-derisk-GO.md` neuralised the same host subtraction with a
**Rao-Ballard TWO-neuron ON/OFF predictive-coding population** — a *separate* error unit fed exc=target,
inh=prediction, emitting `relu(target-est) - relu(est-target)`. This de-risk uses a **different, arguably stronger
biology**: the error is NOT a separate population, it is the **self-prediction mismatch internal to each read-out
neuron's own two compartments**. The soma is nudged by the teacher toward the target; the basal dendrite predicts
the soma from the forward synaptic drive through the plastic weights; the neuron's own soma-minus-dendrite voltage
comparison is the error — made physically local because the somatic spike back-propagates into the dendrite
(Kandel ch.13 active dendrites). The subtraction is the neuron's biophysics, not host arithmetic on two host-held
numbers. Biology binding: `research/biology/urbanczik-senn-dendritic-prediction.md`.

## Method (reuse-by-import; the shipped U-S rule computes the error)

- **Learning substrate imported from `sim/`:** `sim.dendritic_plasticity.urbanczik_senn_update` — THE shipped,
  literature-faithful rule `dw = outer(pre, gate*(soma_rate - sigma(v_basal)))`. This function computes the error;
  it is not reimplemented and `sim/` is not edited.
- **Task:** the role-filler word/sequence-acquisition systematicity harness the prior read-out de-risks use
  (`cortex_learned_binder_systematicity_probe`: `make_role_codes` / `make_systematicity_splits` / `native_argmax`),
  320 codes x 128-dim, R=4 roles, F=16 fillers, 3 leakage-free splits, 24000 steps. Metric = held-out (bundled,
  unseen role-filler combination) generalization.
- **Read-out neuron j (NEURAL arm):** dendrite `v_basal_j = act @ W_O[:,j]` (= the estimate, forward drive through
  the plastic weights); soma membrane `u_j = (1-beta)*est_j + beta*target_j` (finite teacher nudging, beta=0.5);
  spiking soma rate `s_j = Poisson(sigma(g*u_j)*W)/W` (W=20 spike window); the shipped rule returns
  `dw = outer(act, s_j - sigma(g*est_j))` = the soma-vs-dendrite mismatch. Decoded to an error estimate by dividing
  out the fixed small-signal transfer slope (rate decoding, exactly as the spike-rate runner divides counts by the
  window), then applied with the SAME lr/lam as the host arm. No host `est - target` anywhere.
- **HOST arm (reference / the default being replaced):** the identical binder trained by the exact host subtraction
  `err = est - target`. Same lr, lam, task, seeds — a like-for-like head-to-head.

## Result — 6 seeds, per-seed + pooled

<!--derived-->

All figures below are rounded reads/means of the cited raw artifact
`research/findings/raw/_neural_error_population.json` (pooled means, per-seed split-means, and 0.5x-NEURAL
thresholds computed from it); the raw per-run values live in that JSON and its `preconditions`/`attribution` blocks.

| arm | held-out generalization (6-seed mean) | per-seed [42, 43, 44, 100, 101, 102] |
|---|---|---|
| HOST-err (reference / default) | 1.000 | [1.000, 1.000, 1.000, 1.000, 1.000, 1.000] |
| **NEURAL-err (U-S soma-vs-dendrite)** | **0.964** | [1.000, 1.000, 0.926, 0.857, 1.000, 1.000] |
| LESION-nodend (silence dendritic self-prediction) | 0.032 | [0.000, 0.067, 0.000, 0.095, 0.000, 0.030] |
| LESION-noteach (silence somatic teaching) | 0.077 | [0.095, 0.000, 0.167, 0.143, 0.056, 0.000] |
| SCRAMBLE (mis-address error across outputs) | 0.087 | [0.190, 0.067, 0.167, 0.095, 0.000, 0.000] |

- **NEURAL == HOST within noise** (NEURAL = 96.4% of HOST; single-binding systematicity 1.000), NEURAL >= 0.85x
  HOST in **6/6** seeds. The read-out learns its word choices as well from the neuron's own error as from the host
  formula. The neural-vs-host gap (3.6 points, driven by seeds 44/100 at 0.926/0.857) is the honest residual from
  finite-nudging + spike-count noise; it is small and every seed clears the 0.85x bar.
- **The error is genuinely neural — three dissociations, each collapses learning (anti-cheat #1):**
  1. **Silence the dendritic self-prediction** (pin the dendrite so it no longer predicts the soma): mismatch
     stops tracking the estimate -> learning collapses to 0.032. The dendritic prediction — the neuron's error
     computer — is load-bearing; a residual host formula would be immune to this.
  2. **Silence the somatic teaching** (beta=0 -> soma == dendrite -> mismatch identically ~0): the error population
     emits nothing -> learning collapses to 0.077.
  3. **Scramble** (permute the error across outputs so err_j no longer addresses output j): collapses to 0.087 ~
     chance. The per-output error is load-bearing, not a noise artifact.
  All three floors sit far below 0.5x NEURAL (0.482), in every seed.
- **The default (HOST) arm genuinely differs from the lesioned neural arms** (HOST 1.000 vs lesions 0.03-0.09 at
  floor) — the dissociation is real, not a both-arms-fail or both-arms-pass artifact.
- **Attribution** (`tools.lab.attributable_to`, in the artifact's `attribution` block; fraction of the learning
  NOT present in each lesion control): dendritic self-prediction ~0.97, somatic teaching ~0.92, per-output
  addressing ~0.91. Almost all of the read-out's learning is owned by the neuron's own error machinery, not a
  residual host term — the gap#5 "measure whose the difference is" discipline applied here.

## Reading it (brain-based-only)

- The corrective "was I wrong?" error is **neuralisable as a neuron's own somato-dendritic mismatch**. What the
  host used to compute (`est - target`) is now the difference between two compartments of the same read-out neuron:
  the teacher-nudged soma (the "target" side) and the forward-driven dendritic prediction (the "estimate" side).
- **The target itself remains a legitimate env/teacher scaffold** — the supervised nudge current into the soma,
  the innate-teacher-teaches-a-learned-circuit pattern. What this de-risk removes is the host *subtraction* that
  turned (target, estimate) into the per-output error; that is now the neuron's biophysics.
- **Scope (respected):** this closes the ERROR-SOURCE only (HOW the teaching signal is computed). It does NOT touch
  the mouth/word-readout READ-REGIME (board #37, a separate in-flight frontier), and does NOT claim deep/hidden-layer
  credit assignment — the read-out here is a single plastic layer.

## Honest limits

- **Finite-nudging + spiking read is where any gap lives.** With beta<1 and Poisson spike counts the neural mismatch
  is a scaled, noisy estimate of `(target - est)`; the head-to-head measures exactly that degradation. It is small
  here (NEURAL = <PCT>% of HOST) at the 24000-step budget; at a shorter budget the neural arm lags then catches up.
- **The transfer nonlinearity is real but mild** at the code magnitude (gain=1.5x code-std); a much larger gain
  saturates the sigmoid and would bias the error. Not swept exhaustively — the reported point is a principled,
  not tuned-to-win, operating point (same lr/lam as host; only beta, gain, spike window are U-S-specific).
- **Numpy/CPU, task-scale.** The on-bridge spiking realization (deliver the soma-vs-dendrite mismatch through the
  read-out learning's existing per-synapse reward channel) is the natural next rung, not done here.

## Reproduce
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 SIM_BACKEND=numpy \
  python -u -m research.runners._neural_error_population_derisk --seeds 42,43,44,100,101,102 --steps 24000
```

## Biology
- Urbanczik & Senn, "Learning by the Dendritic Prediction of Somatic Spiking," Neuron 81:521-528, 2014
  (PubMed 24507189) — the local dendritic-voltage third-factor rule, shipped as `sim/dendritic_plasticity.py`.
- Mikulasch, Rudelt, Wibral & Priesemann, "Where is the error? Hierarchical predictive coding through dendritic
  error computation," Trends Neurosci 46:45-59, 2023 (PubMed 36577388) — prediction errors are computed locally in
  dendritic compartments, not in separate units.
- Kandel, Principles of Neural Science 6e, ch.13 (active dendritic properties: NMDA/voltage-gated Ca2+ spikes;
  back-propagating action potentials) — the biophysical substrate for the local soma-vs-dendrite comparison.
