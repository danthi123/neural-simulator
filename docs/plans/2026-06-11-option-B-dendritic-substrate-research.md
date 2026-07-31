---
type: plan
status: live
date: 2026-06-11
---

# Option B — the dendritic-substrate research (the opening move for the semantically-structured cortex)

**Status:** READ-ONLY deep-research + catalog/literature/existing-sim review (the project's standing "deep research
FIRST at a new direction" opening move, CLAUDE.md). No `sim/` code, no build, no GPU. The single deliverable is this
doc + one commit. **Date:** 2026-06-11. **Author role:** read-only deep-research subagent. Every load-bearing project
fact below was re-verified against the project's own record (file + line citations); the surprising ones were read in
full, not trusted from a prior summary.

**Decision being served:** the owner chose **OPTION B** — replace the conversational cortex's idealized exact-inverse
binding algebra with a **semantically-structured cortex that GENERALIZES**: keep the brain's overlapping/correlated
semantic codes (so "a cat is like a dog" inference is possible because similar concepts have similar codes), and add
the missing machinery to make binding work over them. Tonight's de-risk arc established the blocker precisely (four
mechanistically-distinct NEGATIVES, all committed): the brain's concept codes are **correlated** (between-code cosine
≈ 0.81), and **no point-neuron spiking mechanism can decorrelate them**, because decorrelation/whitening (removing the
shared "common mode" so codes become independent enough to bind/invert) is an **analog, pre-spike (DENDRITIC)**
computation that the project's point neurons structurally lack. This is the documented **Mikulasch-Priesemann
point-neuron limit** (CLAUDE.md:14). The fix is to give the substrate **dendritic machinery that performs analog /
pre-spike whitening.** This doc is the opening-move research for that substrate extension.

---

## 0. Terms (defined once — owner standing requirement; no undefined acronyms)

- **whitening = decorrelation = removing the shared "common mode."** Whitening is the stronger form: decorrelate AND
  equalize variance so the covariance matrix becomes the identity (all pairs of code-dimensions uncorrelated and unit
  variance). The "common mode" is the large shared component that all correlated concept codes carry; subtracting it
  exposes the small concept-distinguishing residual.
- **point neuron** — a neuron with ONE membrane state `(v, u)` (Izhikevich) or `(v, w)` (AdEx): all synaptic input is
  summed at a single soma compartment before the spike decision. The project's substrate. Cannot do per-input-source
  sub-threshold computation.
- **dendritic compartment** — a separate electrical sub-region of a neuron with its OWN local membrane potential, which
  integrates a SUBSET of the synaptic input and computes a local sub-threshold quantity BEFORE it reaches the soma. The
  capability point neurons lack.
- **Mikulasch-Priesemann limit** — the peer-reviewed theorem (Mikulasch, Rudelt, Priesemann, PNAS 2021) that a point
  neuron with a single global inhibitory pool CANNOT learn to whiten correlated inputs; local **dendritic** balance is
  the necessary ingredient. The project's documented decorrelation boundary.
- **reproducibility (the load-bearing gate)** — same input fed twice (under realistic spiking noise) must produce the
  same code: same-input cosine ≥ 0.9. This is the gate that **killed** tonight's spiking-dentate-gyrus attempt
  (achieved ≈ 0.05) and is the single hardest bar the dendritic front end must clear.
- **FHRR (Fourier Holographic Reduced Representation)** — the production binding algebra: each concept is a vector of
  phases; bind = element-wise complex product, unbind = multiply by the complex conjugate. Realised on the bridge's
  resonate-and-fire neurons + complex synapses. "Exact-inverse" because unbind is the exact algebraic inverse of bind.
- **NEF (Neural Engineering Framework)** — Eliasmith's method for realising a vector function as a spiking population.
  The project's validated localist cleanup is an NEF circuit.
- **VSA (Vector Symbolic Architecture)** — symbols as high-dimensional vectors bound by an invertible algebra; FHRR is
  one VSA scheme.
- **denoise64 codes** — the composer's REAL concept codes: captured + denoised firing of 16 concept pools on a trained
  bridge, cached at `research/findings/raw/activity_level_integration_cache/denoise64_seed{42,43,44}.npz`. Correlated
  (between-code cosine ≈ 0.81), lossy, grounded in the brain's own activity. The codes the whitening front end must
  consume.
- **Urbanczik-Senn** — the canonical learnable two-compartment (soma + dendrite) neuron where the dendrite predicts the
  somatic spiking and a local rule minimises the mismatch. Committed in the project (`sim/dendritic_plasticity.py`).
- **GLR-2017 / feedback alignment** — Guerguiev-Lillicrap-Richards 2017: segregated apical/basal dendrites with a
  FIXED-RANDOM top-down feedback projection (no "weight transport" — the feedback weights are never copied from the
  forward weights). Committed in the project (`sim/dendritic_neuron.py`, `sim/dendritic_mlp.py`).

---

## 1. THE PRECISE MECHANISM (diagnosis): why a dendrite whitens and a point neuron cannot

### 1.1 What a point neuron does each step (grounded in the real per-step update)

I traced the real membrane update. In `sim/bridge.py::_run_one_simulation_step` (line 5456), the engine assembles ONE
scalar drive per neuron, `total_input_current_pA`, by summing every current source — AMPA/GABA synaptic current, NMDA
(5749), NMDA-recurrent (5786), the dendritic-COINCIDENCE plateau (5859), GABA_B (5907), neuromodulator drive (5709),
OU noise (5950) — and then at line 5957 calls `fused_izhikevich2007_dynamics_update(cp_membrane_potential_v,
cp_recovery_variable_u, ...)` (or `fused_adex_dynamics_update`, 6123). **Everything collapses to a single somatic
`(v, u)` BEFORE the spike decision.** Excitation and inhibition from all sources are averaged together at one point.

This is exactly the structure Mikulasch-Priesemann prove cannot whiten. The reason (read directly from the paper, PMC
8685685): to whiten, the lateral inhibition between two units must match THEIR specific shared correlation — an
**input-specific** quantity. In a point neuron all inhibition arrives at the one somatic compartment and is averaged
together with excitation, so **the input-specific structure is destroyed before any plasticity rule can read it.** A
single global inhibitory pool can only apply uniform suppression; it cannot carve a different inhibitory weight per
correlated pair. Hence the point neuron + global inhibition reaches **mean** decorrelation at best, never the
**all-pairs / worst-pair** whitening that binding needs.

### 1.2 What a dendritic compartment computes in the sub-threshold / analog domain

Mikulasch-Priesemann's mechanism (extracted from the source): each neuron has **N dendritic compartments, one per
input dimension**. Each compartment `j,i` holds a local potential

```
u_ji(t) = F_ji · x_i(t)  +  Σ_k W_jki · z_k(t)
```

where `F_ji` is the feedforward (excitatory) weight, `x_i` the input, and the `Σ_k W_jki z_k` term is **recurrent
inhibition delivered locally to that compartment** (the lateral prediction from the rest of the population). The paper's
key identity: the local dendritic potential is **proportional to the coding error**, `u_ji = F_ji (x_i − x̂_i)` — the
part of the input NOT already explained (predicted) by the rest of the population's activity. The recurrent inhibition
**locally cancels the predictable (common-mode) part of the excitation in each compartment**, leaving only the residual.

Crucially: **this cancellation happens at the analog membrane-voltage level, BEFORE the soma spikes.** The paper states
decorrelation "occurs primarily at the analog membrane voltage level, not spike output." The soma then sums the
already-balanced compartments (`u_j = Σ_i u_ji`) and fires from the residual. So the whitening is a **sub-threshold,
graded (analog) computation distributed across the dendrites**, and only the whitened residual is ever converted to
spikes.

### 1.3 Why this is exactly what the four point-neuron NEGATIVES could not do

The four tonight-committed NEGATIVES are each a way of trying to whiten WITHOUT per-compartment analog cancellation, and
each fails for the reason §1.1 predicts:

| NEGATIVE (committed today) | what it tried | why it failed (per its own doc) |
|---|---|---|
| **vanilla Hopfield** (`-storkey-...NEGATIVE` §root-cause) | post-hoc clean correlated codes with a Hebbian attractor | common-mode eigenvalue ~18× the signal → all cues collapse to one attractor |
| **Storkey** (`-storkey-ca3-cleanup-NEGATIVE`) | a LOCAL covariance-corrected weight rule | locality wall: a local pairwise rule cannot reach `C⁻¹`; pseudo-inverse (global host op) recovers 1.000 |
| **spiking dentate-gyrus + rate-k-WTA** (`-dg-ratekwta-cleanup-NEGATIVE`) | decorrelate by sparse spiking expansion + competitive read | the DG input-driven signal is **below the spiking noise floor** (~15 spikes/600 neurons), so the read is noise-determined → **reproducibility ≈ separation at every k** (no decorrelation gain at usable repro) |
| **fixed random expansion + threshold** (`-fixed-expansion-decorrelation-NEGATIVE`) | a deterministic Marr-Albus granule expansion + top-k | the threshold sits in a smooth Gaussian activation region with a margin **~3,800× below the realistic input noise** → every boundary unit flips on every noise draw → repro 0.03–0.06 at σ=0.1, AND the common mode survives the linear expansion (between-cos stays ~0.55) |

The common thread: **all four operate on the SPIKE side (or a single-shot threshold on a summed-at-soma activation),
where the small concept-distinguishing residual sits under a large correlated common mode at low signal-to-noise.** This
is the same SNR wall the project already proved for opponency (`2026-06-05-B-opponency-rate-coded-SNR-wall-CONFIRMED`).
The dendritic mechanism removes the common mode in the **analog sub-threshold domain** where the membrane voltage
*averages over time and over compartments*, raising the SNR before the spike — which is precisely why Mikulasch-
Priesemann report it "prevented the sudden collapse" that point neurons suffer at transmission delays > 0.3 ms, and is
robust across 50 network realisations. **The dendrite is the analog averaging stage the four spike-side attempts lacked.**

### 1.4 The tie to Mikulasch-Priesemann, stated exactly

The project's decorrelation boundary IS the Mikulasch-Priesemann theorem promoted from diagnosis to citation. Their
central claim (quoted via the project's own deep-research doc `2026-06-06-decorrelation-blocker-deep-research.md` §2 and
re-verified against PMC 8685685): *"A point neuron with global somatic integration cannot decorrelate inputs; lateral
inhibition must be computed LOCALLY at dendrites to whiten the input representation."* The substrate is point neurons.
So the boundary is, precisely and citably, **a point-neuron limit, and the named escape (dendritic compartments) is the
one capability the bridge does not have.** Option B = build that capability.

---

## 2. RANKED, BIOLOGICALLY-GROUNDED OPTIONS for adding dendritic whitening to the bridge

**Critical context that reshapes the ranking (read before the options): the project is FAR less far from this than the
build-plan's "months-scale dendritic rewrite" framing implies — because two things already exist.**

1. **A committed, on-`main`, GPU-capable dendritic two-compartment + local-plasticity stack** (`sim/dendritic_neuron.py`,
   `sim/dendritic_plasticity.py`, `sim/dendritic_mlp.py`; git history `575a7cf1`→`18600441`; all on `main` per
   `git ls-tree main`). It implements: a spiking two-compartment pyramidal (basal forward + apical fixed-random feedback
   + Larkum BAC threshold lowering), the LOCAL **Urbanczik-Senn** somato-dendritic mismatch rule (apical-gated, no
   weight transport), and GLR-2017 feedback alignment. **BUT** it is currently a **supervised feedback-alignment
   classifier** (it learns MNIST with a teaching signal), it is **numpy/CuPy array math NOT wired into the bridge
   step-loop**, and its credit-assignment arc terminated in an honest **VOID** (`2026-05-18-dendritic-fairscale-SOUND-
   instrument-VOID...`: the discriminating instrument could not be built at MNIST scale — a *different* question from
   whitening; the machinery is sound, the supervised-credit-assignment science was unanswerable at feasible scale).
2. **A 6-seed validated result that a LOCAL whitening rule learns a code that COMPOSES** (`2026-06-06-option1-local-
   learning-whitening-VALIDATED-6seed.md`): a regularized similarity-matching lateral rule `ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij −
   λM_ij` (Pehlevan-Chklovskii form with a weight-decay fixed point) learns a *partial* whitening (effectively `C^{−1/3}`)
   that drives the full conversational composition to **100%, 6/6 seeds**, judged on COMPOSITION (the agent benchmark),
   with a BOUNDED weight matrix. AND the same arc REFUTED the "rate-coded spiking can't whiten" wall: spikes HOLD a
   whitened code AND COMPUTE one with analytic lateral inhibition + stable leaky dynamics (the membrane averages the
   rate-noise) — `2026-06-06-whitening-computation-spikes-CAN-compute-it.md`.

**The honest reconciliation of the apparent contradiction (this is the single most important thing in this doc — §6.1
expands it):** the 2026-06-06 arc validated whitening-that-COMPOSES at the *rate/algorithm level on grounded CIFAR
codes*. Tonight's 2026-06-11 arc failed the *REPRODUCIBILITY-under-realistic-noise gate on the denoise64 codes* — a
**different, stricter gate** that the 2026-06-06 arc never ran. Mikulasch-Priesemann is the bridge between them: their
paper's whole thesis is that dendritic balance specifically solves the **noise / transmission-delay regime** that breaks
point neurons — i.e. it is the mechanism that targets exactly the reproducibility gate still open. So the prior
6-seed-composing-whitening is the **strong prior that Option B will work**, and the dendritic substrate is the
**principled answer to the specific gate (reproducibility) that the rate-level result never had to face.**

### Option B1 (RECOMMENDED) — a SEPARATE analog sub-threshold "dendritic whitening" population, wired pre-spike, learning the Pehlevan-Chklovskii lateral rule

- **Mechanism.** Add a dedicated "IT-whitening" population whose job is to take the correlated concept code `x` and emit
  a whitened (decorrelated, variance-equalized, similarity-preserving) version `y` BEFORE it enters the spiking binder.
  Realise the whitening as **graded sub-threshold lateral inhibition** among the whitening units: settle
  `dr/dt = W_ff·x − r − M·r̂` to the fixed point `r = (I+M)⁻¹ W_ff x`, where `M` is the plastic lateral-inhibition
  matrix learned by the **regularized similarity-matching rule** `ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λM_ij` (the rule already
  validated at 6/6 composition, 2026-06-06). The lateral term is carried by the bridge's existing rate-coded FS
  inhibition; the membrane integration averages the rate-noise (the validated "spikes COMPUTE whitening" result). The
  whitened `y` then drives the FHRR binder (Option A's already-validated bind/unbind + localist NEF cleanup + the
  familiarity gate).
- **What existing code it extends.** This is the LIGHTEST substrate touch: it reuses the bridge's region framework, FS
  lateral inhibition, and homeostasis; the `−λM` is synaptic weight-decay (the bridge has homeostatic decay). The
  "dendritic" character is that the whitening is computed in the **graded sub-threshold settle** (analog), not in the
  spike code — which is the Mikulasch-Priesemann essence WITHOUT a full N-compartment neuron rewrite. It is
  "dendrite-FUNCTION without dendrite-MORPHOLOGY": one analog population standing in for the per-compartment local
  balance.
- **Fidelity vs biology.** MODERATE-HIGH for the *function* (it computes the right analog pre-spike whitening with a
  local biological rule), LOWER for the *morphology* (the analog cancellation lives in a population's recurrent settle,
  not in literal per-input dendritic compartments). This is the same "graded pre-spike whitening stage = the retina/LGN
  efficient-coding stage" the project's own deep research called **biologically FAITHFUL, not a cheat** (`2026-06-06-
  decorrelation-blocker-deep-research.md` §5, §11.3 — the retina whitens graded and pre-spike for the identical SNR
  reason; every spiking cortical model imports graded-whitened input).
- **Build cost: LOW-MODERATE** (weeks, not months — it is a new population + a plastic lateral rule + a settle loop;
  most of it is config + a small guarded current term, NOT a neuron-model rewrite). **Risk: MODERATE** — the open
  question is whether the *learned* (not handed-in) lateral matrix reaches the **reproducibility ≥0.9** bar on the
  denoise64 codes (the rate-level result was judged on composition, not reproducibility; and the 2026-06-06 learning
  de-risk itself caught two near-false-positives where the matrix blew up on rank-deficient toy data — §6 anti-cheats
  inherit this).
- **Composition with the validated pipeline.** Drops in as the **code front end** before the existing decorrelate → bind
  → clean → abstain pipeline: `correlated x → [B1 analog whitening] → similarity-preserving decorrelated y → FHRR bind →
  localist NEF cleanup → familiarity-gate abstention`. Everything downstream of `y` is already validated.

### Option B2 — a true two-compartment `NeuronModel` (soma + dendrite), learning Urbanczik-Senn whitening, computed in the bridge step-loop

- **Mechanism.** Add a `NeuronModel.TWO_COMPARTMENT` (or extend Izhikevich with a second `v_dend` state): each neuron
  gets a dendritic compartment that integrates the feedforward drive + LOCAL lateral inhibition and computes the
  coding-error residual `u = F(x − x̂)` in the analog domain; the soma fires from the residual. The lateral inhibitory
  weights learn the Mikulasch-Priesemann voltage-dependent rule `ΔW_jk ∝ −z_k · u_j` (inhibition learns to locally
  cancel the predictable input) and the feedforward weights are gated by the same local potential. This is the
  literature-faithful realisation.
- **What existing code it extends.** This is where the prior dendritic design doc (`docs/plans/2026-05-05-dendritic-
  learning-design.md`) and the committed `sim/dendritic_neuron.py` (the two-compartment BAC neuron) are directly
  reusable — but they must be (a) **wired into the bridge step-loop** (currently they are standalone numpy/CuPy), and
  (b) **repurposed from supervised feedback-alignment to unsupervised whitening** (the committed rule is apical-gated by
  a teaching signal; whitening needs the unsupervised local-balance rule instead). The slot in the step-loop is exactly
  where the **dendritic-COINCIDENCE plateau** already sits (`bridge.py:5788-5859`): a per-postsynaptic-neuron NONLINEAR
  restricted-matvec sub-unit, fully guarded (byte-identical when off), computed in the current-accumulation phase BEFORE
  the soma integrates. A dendritic-whitening compartment is the same shape — a guarded, per-neuron, restricted-matvec
  sub-threshold term added before line 5957.
- **Fidelity vs biology. HIGHEST** — this is the actual Mikulasch-Priesemann / Larkum / Urbanczik-Senn mechanism, the
  one that the catalog repeatedly flags as the missing capability (G.13 "active dendrites missing"; D.* "single
  compartment can only sum inputs"; B.17 dendritic linearization; the cerebellar Golgi max-pool needing dendrites).
- **Build cost: HIGH** (the prior design doc estimates **1.5–2 months** for the multi-compartment kernel + plasticity
  gate + routing + validation; the faithful Mikulasch-Priesemann is N compartments per neuron, one per input dim, which
  at D=512 is heavier still — though a *reduced* 2-compartment soma+dendrite per the committed `DendriticLayer` is the
  tractable form). **Risk: HIGH-MODERATE** — a new `NeuronModel` is a protected `sim/` edit touching the hottest code
  path; the byte-identity-when-off discipline (the dendritic-coincidence precedent) is the mitigation, but the
  reproducibility-≥0.9 question is still the live one.
- **Composition.** Same pipeline slot as B1; the difference is the whitening lives in literal per-neuron compartments
  rather than a separate analog population.

### Option B3 — the literature-faithful FIXED-Ω balanced-network wiring (the decisive cheap separator, NOT a production answer)

- **Mechanism.** Install the analytic whitening solution `Ω = ΓᵀΓ` (Deneve-Machens balanced spike-coding network) as
  **fixed** lateral inhibition among the whitening units, where `Γ` = the concept codebook. This is "ZCA-as-wiring": the
  whitening matrix handed in, not learned.
- **What it extends.** Only fixed lateral inhibition in the region framework — no plasticity, no new neuron model.
- **Fidelity / cost / risk.** Fidelity LOW as a *production* mechanism (it is a host-computed matrix installed as
  wiring — a documented shortcut by the project's brain-based bar, because the brain is not computing `ΓᵀΓ`). But cost
  is TRIVIAL and it is the **single most informative cheap experiment** (the project's own deep research, §12, names it
  as the test that "routes every subsequent decision"): it cleanly separates *"can spikes HOLD a whitening solution at
  the reproducibility bar at all"* from *"can a local rule LEARN it."* It is a **diagnostic, not an option B** — it tells
  you whether B1/B2 are worth building.
- **Composition.** Same slot; used to set the ceiling before committing to the learned versions.

### Ranking summary

| | Whitening computed by | Substrate touch | Fidelity (function / morphology) | Build cost | Reproducibility-≥0.9 risk |
|---|---|---|---|---|---|
| **B1 (recommend FIRST)** | separate analog population, learned PC lateral rule | new region + guarded current term | HIGH / MODERATE | **low-moderate (weeks)** | **moderate (the open question)** |
| **B2 (the faithful target)** | per-neuron soma+dendrite compartments, learned U-S rule | new `NeuronModel` (protected, hot path) | **HIGHEST / HIGHEST** | high (1.5–2 mo) | moderate |
| **B3 (decisive diagnostic)** | fixed `Ω=ΓᵀΓ` handed-in wiring | fixed lateral inhibition only | low (host shortcut) | **trivial** | it MEASURES the risk |

**Recommendation: B1 first**, because it removes the load-bearing analog-whitening gap on the lightest substrate touch,
reuses the 6/6-validated Pehlevan-Chklovskii rule + the bridge's existing FS/homeostasis machinery, and is "dendritic in
function" (analog pre-spike cancellation) without a full neuron-model rewrite — and per §6 it is cheaply falsifiable on
the exact reproducibility gate that is still open. **Run B3 as the half-day diagnostic FIRST** (does the handed-in
whitening even HOLD reproducibility ≥0.9 in spikes?) to set the ceiling and route the decision; **B2 is the
biologically-faithful follow-on** once B1 proves the function, escalated deliberately (it is the months-scale protected
edit), and is the path to the *proper brain analogue* the project's actual-goal framing wants.

---

## 3. REUSABLE PROJECT MACHINERY (concrete, file-cited, all verified present)

- **The committed dendritic stack (the biggest reuse; do NOT reinvent).** `sim/dendritic_neuron.py` (spiking
  two-compartment BAC pyramidal: basal forward + apical fixed-random feedback + Larkum threshold lowering),
  `sim/dendritic_plasticity.py` (LOCAL Urbanczik-Senn somato-dendritic mismatch rule, apical-gated, no weight
  transport), `sim/dendritic_mlp.py` (GLR-2017 feedback alignment, GPU-capable via `sim.backend`). All on `main`. **The
  repurposing for B2:** swap the supervised teaching-signal gating for the unsupervised local-balance rule; wire into the
  step-loop. The MORPHOLOGY (compartments, BAC, no-weight-transport, kill-safe, deterministic seeded init) is built and
  tested (`tests/test_*dendritic*`).
- **The 6-seed-validated whitening rule + harness.** `research/findings/raw/_A_whitening_compose_gate.py` (the
  regularized local rule that COMPOSES at 100% 6/6) + `_A_whitening_computation_derisk.py` (spikes COMPUTE whitening with
  leaky dynamics + the noiseless control) + `_A_whitening_learn_lateral_derisk.py` (the M-ratio guard that caught the
  false positives). **These ARE the B1 algorithm + its anti-cheats**, already written and validated at the rate level.
- **The dendritic-COINCIDENCE plateau (the step-loop integration precedent).** `bridge.py:5788-5859` +
  `sim/kernels.py:255` (`fused_coincidence_plateau`): the exact pattern for a per-neuron NONLINEAR restricted-matvec
  sub-threshold term, guarded to byte-identity when off, computed before the soma integration. A dendritic-whitening
  compartment slots in the same way (the byte-identity-when-off discipline is the protected-edit template).
- **The decorrelate → bind → clean → abstain downstream (all validated; B1/B2 only replace the front end).** FHRR
  bind/unbind on `NeuronModel.RESONATE_AND_FIRE` + complex synapses; the localist NEF/TPAM cleanup
  (`rf_phasor_composer._spiking_cleanup`, `core_sim_composition.NEF_CLEANUP_OP`, == numpy at D=2048 27/27); the learned
  anti-Hebbian familiarity gate (de-risked +0.982 margin); the sparse-distributed positive-control attractor
  (`research/runners/_D_sparse_heteroassoc.py`).
- **The reproducibility-gate harness + the convention caveat.** The code loader `cortex_storkey_ca3_cleanup_probe.
  load_real_codes` (line 62) is the exact denoise64 path the four NEGATIVES used; the fixed-expansion probe
  (`cortex_fixed_expansion_decorrelation_probe.py`) already computes between-cos + repro@σ + margin/noise — **reuse its
  reproducibility measurement verbatim.** ⚠️ **The surprising convention caveat** (from `2026-06-11-cortex-core-learned-
  binder-research.md` §1.2): sparse codes must be read in their NATIVE binary {0,1} / mean-removed form, NEVER
  median-bipolarized (which manufactures a common mode and produces a false NEGATIVE). The de-risk must assert this and
  unit-check the between-cos.
- **The vocab-ceiling probe + the merged one-bridge substrate.** `research/runners/vocab_ceiling_probe.py` (the full
  capability matrix at V=320, abstention floor + shuffled-fact control per cell); `research/runners/nav_conv_merged_
  bridge.py` (the substrate the whitened-code composer must run on). The acceptance bar, unchanged.
- **`sim/backend.py`** — any new kernel (the B1 settle term, the B2 compartment update) must route through
  `get_backend()`/`fuse()` so it works on both CuPy and NumPy (the existing dendritic stack already does this).
- **The prior dendritic design doc.** `docs/plans/2026-05-05-dendritic-learning-design.md` — the apical/basal scope,
  `target_compartment` routing on `RegionPathway`, and the 1.5–2 month estimate for B2.

---

## 4. THE CHEAP-FIRST DE-RISK (the load-bearing FALSIFICATION — specified precisely)

**The single open scientific question for Option B is one number: can a dendritic-whitening mechanism, computing in the
analog sub-threshold domain, take the brain's REAL correlated `denoise64` codes and produce codes that are simultaneously
(a) DECORRELATED, (b) REPRODUCIBLE under realistic noise at ≥0.9, and (c) genuinely analog-neural (not a disguised host
ZCA matrix)?** (a) and (c) are already established (the 2026-06-06 arc); (b) — the reproducibility gate that killed the
spiking-DG — is the ONLY untested one. The de-risk isolates exactly (b). CPU, numpy, NO substrate rewrite.

### 4.1 The probe (smallest thing that falsifies Option B)

**Codes / harness to reuse (verbatim):**
- The correlated codes: `cortex_storkey_ca3_cleanup_probe.load_real_codes(seed, proj_dim, rng)` (line 62) — the exact
  denoise64 path the four NEGATIVES used (mean over obs samples → Gaussian project → mean-center → unit-normalize;
  between-cos ≈ 0.80). Run seeds 42/43/44.
- The reproducibility + decorrelation measurement: reuse `cortex_fixed_expansion_decorrelation_probe.py`'s
  `between_cos`, `repro@σ`, and `margin/noise` instrumentation verbatim (so the result is directly comparable to the
  fixed-expansion NEGATIVE's 0.03–0.06).
- The whitening algorithm: the validated B1 rule from `research/findings/raw/_A_whitening_compose_gate.py` —
  `ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λM_ij` with the stable leaky settle `dr/dt = W_ff·x − r − M·r̂` from
  `_A_whitening_computation_derisk.py` (which already carries the noiseless-solver control). This is "the dendritic
  whitening, simulated as the analog sub-threshold settle, with the lateral matrix learned by the local rule" — NOT a
  `numpy.linalg` ZCA.

**The three gates (Option B GO requires ALL three):**
1. **DECORRELATION** — between-code cosine of the whitened codes ≤ **0.1** (the binding bar; the 2026-06-06 arc already
   hit ~0.03 at the rate level, so this should pass — it is the easy gate).
2. **REPRODUCIBILITY (the load-bearing one)** — feed each code twice with realistic spiking noise (σ = 0.1, the level
   the fixed-expansion probe used, where it scored 0.04) injected at the input, run the analog settle each time, and
   require same-input cosine ≥ **0.9**. **This is the EXACT bar the spiking dentate-gyrus read FAILED at ~0.05 and the
   fixed expansion at 0.03–0.06.** If the analog-settle whitening clears it, that is the new result; if it does not, that
   is the decisive NEGATIVE that says even dendritic whitening cannot reach reproducibility on codes this correlated
   (§6 risk).
3. **COMPOSITION (carry-over from the 2026-06-06 lesson — do NOT gate on coherence alone)** — bind + unbind over the
   whitened codes and confirm the full who/what-Q&A recovers (≥ the localist-cleanup parity), because a *noise-collapsed*
   output also passes (1) and (2) trivially but will NOT compose. The 2026-06-06 arc was nearly fooled THREE times by
   the coherence proxy; composition is the gate that caught it.

**If gates 1–3 pass on numpy: the minimal on-substrate (real bridge) confirmation** — wire the B1 analog-whitening
population into a small `SimulationBridge` (`SIM_BACKEND=numpy` for the first pass, then `cupy`), drive the denoise64
codes through it as graded sub-threshold lateral inhibition (the existing FS pool + the `−λM` homeostatic decay), read
the whitened rate code, and re-run gates 1–3 on the bridge output. Only on that GO does the GPU build of B1 + the
vocab-ceiling acceptance run proceed.

### 4.2 HOW TO PROVE it is analog-neural and NOT a disguised host ZCA (critical — the cheat to rule out)

A host `numpy.linalg.inv`/ZCA would pass gates 1–3 and be a shortcut (the brain is not computing a matrix inverse). The
proof it is genuinely analog-neural:
1. **It must emerge from the simulated sub-threshold dynamics.** The whitened code is the FIXED POINT of the leaky settle
   `dr/dt = W_ff·x − r − M·r̂` integrated over time steps (the membrane averaging the rate-noise), NOT the output of a
   one-shot `C^{−1/2} x` host op. The probe must run the iterative settle and show it CONVERGES to the whitened code
   over steps (a host ZCA has no settle).
2. **The lateral matrix `M` must be LEARNED by the local rule, not handed in.** Show `M` grows from zero under
   `ΔM_ij ∝ ⟨y_i y_j⟩ − δ_ij − λM_ij` and converges to a BOUNDED matrix (the M-ratio guard from
   `_A_whitening_learn_lateral_derisk.py`: `‖M_learned‖/‖M_analytic‖` near 1, NOT 9000× — the guard that caught the
   blow-up false-positive). B3 (the handed-in `Ω=ΓᵀΓ`) is run SEPARATELY as the explicit ceiling/shortcut reference so
   the learned-vs-handed-in distinction is visible.
3. **LESION the mechanism → whitening collapses.** Zero `M` (or freeze the settle) → between-cos returns to ~0.8 and
   reproducibility collapses → proves the decorrelation RIDES the simulated lateral dynamics, not a host transform or a
   leftover code property. (The fixed-expansion NEGATIVE already used exactly this lesion, §5; reuse it.)

### 4.3 Anti-cheats (mandatory, beyond the lesion)

- **The reproducibility ≥0.9 bar is the headline anti-cheat** (it IS gate 2; it is the spiking-DG killer; it must be
  front-and-center, measured at σ=0.1, reported alongside between-cos so a "decorrelated but irreproducible" result
  cannot masquerade as a win — the precise trap the fixed-expansion probe documented).
- **Native-convention unit check** — assert the codes are read binary/mean-removed and between-cos of the codes-as-read
  is ≈ the expected value (NOT ≈ 1.0 from accidental median-bipolarize). The exact caveat that would otherwise produce a
  false NEGATIVE.
- **M-ratio bound** (§4.2.2) — catches the noise-collapse false-positive (a blown-up `M` that decorrelates noise).
- **Composition-not-coherence gate** (gate 3) — catches the noise-collapsed output that passes decorrelation +
  reproducibility but cannot bind.
- **The shuffled-fact permuted control + the abstention floor** (carried from the vocab-ceiling acceptance) — at the
  on-substrate confirmation, who/what must collapse to chance under permuted fact pairings, and unstored cues must
  abstain (the no-confab moat), so the whitened-code composer is not echoing a fixed structure.

---

## 5. ANTI-CHEAT CONTROLS — including the NEW generalization capability the structured cortex must demonstrate

The de-risk anti-cheats (§4.3) gate the whitening front end. Two MORE controls gate the *whole* Option-B cortex —
because the entire POINT of Option B (vs the achievable-now flat cortex A) is **generalization**, and that must be
measured, not assumed.

### 5.1 The reproducibility ≥0.9 bar (front and center — the spiking-DG killer)

Stated again as the load-bearing control because it is the one that has killed two prior attempts. Any whitening result
that reports decorrelation WITHOUT reporting same-input reproducibility at σ=0.1 is incomplete and must be rejected. The
fixed-expansion NEGATIVE is the cautionary precedent: between-cos 0.55 looked like partial progress, but repro 0.04 made
it useless. Decorrelation and reproducibility must BOTH clear, simultaneously, on the same operating point.

### 5.2 The "disguised host ZCA?" proof

§4.2 in full: settle-convergence (not one-shot), learned-and-bounded `M` (not handed-in), and lesionability. If any of
the three is absent, the whitening is a host shortcut wearing neural clothes and must be labelled as such (B3 is the
honest handed-in reference; B1/B2 must beat it on the "learned + lesionable" axis to count as brain-based).

### 5.3 The GENERALIZATION control — the new capability the structured cortex must show that the flat one CANNOT

This is the reason Option B exists. The achievable-now flat cortex (Option A) is **semantically flat**: every concept is
equidistant, so it binds reliably but cannot infer "a cat is like a dog." Option B keeps the correlated/similar codes,
so it MUST demonstrate the generalization that flat codes structurally cannot. **Define a held-out generalization test
as a first-class acceptance gate:**
- **Setup.** Use the brain's REAL similarity-bearing codes (whitened by B1/B2 but **similarity-preserving** — see the
  §6 tension). Train relations / facts on SOME similar-concept pairs (e.g. learn properties for "dog", "cat", "wolf"),
  then test inference on a HELD-OUT similar pair never trained in that relation (e.g. query a property of a held-out
  near-neighbour concept).
- **The decisive contrast.** The test must PASS on the similarity-preserving Option-B codes and FAIL (or be at chance) on
  generated orthogonal Option-A codes — *because the inference only works if similar concepts have similar codes.* If
  Option B does not beat Option A on this exact test, the structured cortex has bought nothing over the flat one, and
  the whole Option-B premise is unsupported (an honest NEGATIVE that reshapes the arc — §6).
- **Anti-cheat on the generalization test itself.** A permuted-similarity control: shuffle which concepts are "similar"
  (break the code-similarity ↔ semantic-similarity correspondence) → the held-out inference must collapse to chance.
  Otherwise the "generalization" is an artifact of code overlap unrelated to meaning.

### 5.4 The unchanged vocab-ceiling acceptance matrix

In ADDITION to §5.3, the Option-B cortex must still match the full validated capability matrix the flat cortex passes:
who/what-Q&A + **abstention / no-confab moat 100% (20/20 every cell)** + negation/yes-no + embedded clause (needs code
dimension D≥256) + two-attribute binding, at **V=320, multi-seed (42–47)**, with the shuffled-fact permuted control at
zero false hits, on the merged one-bridge substrate. Generalization (§5.3) is the NEW gate; the matrix is the SAME bar —
Option B must add generalization WITHOUT regressing anything the flat cortex already does.

---

## 6. HONEST RISK REGISTER — every load-bearing assumption, flagged

### 6.1 ⚠️ THE LOAD-BEARING TENSION the controller must see: the project's own record is split, and tonight's framing may be too pessimistic

The build-plan (`docs/plans/2026-06-11-cortex-build-plan...`) frames Option B as a **months-scale dendritic-substrate
rewrite mandated by an impassable point-neuron wall.** But the project's OWN earlier record (2026-06-06, three findings +
a 6-seed result + a deep-research doc) reached a **materially more optimistic** conclusion that tonight's arc did not
cite: a regularized LOCAL whitening rule learns a code that COMPOSES at 100%, 6/6 seeds, and spikes were shown to HOLD
*and* COMPUTE whitening. **These are not in contradiction once you see the gate difference** — 2026-06-06 gated on
COMPOSITION of grounded CIFAR codes at the rate level; 2026-06-11 gated on REPRODUCIBILITY-under-noise of denoise64
codes — but the controller MUST be told both, because it changes the arc's shape:
- If the §4 de-risk shows the **analog-settle whitening (B1) clears reproducibility ≥0.9** on denoise64, then **Option B
  is achievable in WEEKS, not months** (B1 is a population + a validated rule + a settle loop), and the "months-scale
  dendritic rewrite" (B2) is the *fidelity* follow-on, not the prerequisite. This is the **likely-good outcome** given
  the 2026-06-06 priors.
- If the de-risk shows even the analog-settle whitening **cannot** clear reproducibility ≥0.9 on codes this correlated,
  then the wall is real at the strict gate, B2 (literal compartments) is genuinely required, and the months-scale
  estimate stands. **This is the live risk and the whole reason §4 is the cheapest-first falsification.**

**Recommendation to the controller: do NOT commit to the months-scale B2 rewrite until the §4 de-risk + the B3
diagnostic have run.** The 2026-06-06 arc strongly suggests B1 (cheap) may suffice; tonight's arc only proved the
SPIKE-side mechanisms fail, which B1's analog-settle is specifically designed to sidestep.

### 6.2 Dendritic whitening might ALSO not reach reproducibility ≥0.9 on codes this correlated

The genuine open risk. The four NEGATIVES all failed reproducibility because the concept-distinguishing residual sits
under a huge common mode at low SNR. The dendritic/analog-settle mechanism raises SNR by averaging over compartments +
time — but on codes at cos 0.81, the residual may STILL be below the noise floor even after analog averaging. The
2026-06-06 reproducibility number was never measured at σ=0.1 on denoise64 (it was composition on CIFAR), so this is
genuinely untested. If it fails, the honest deliverable is "even analog/dendritic whitening cannot reach VSA-grade
reproducibility on codes this correlated → the binding cortex must use generated decorrelated codes (Option A), and
semantic generalization must come from a SEPARATE linked representation (the dual cortico-hippocampal architecture), not
from binding the similar codes directly." That negative would itself be a major, citable result.

### 6.3 The B2 substrate rewrite may exceed the 1.5–2 month estimate

The prior design doc's estimate is for a *reduced* 2-compartment neuron. The literature-faithful Mikulasch-Priesemann is
**N compartments per neuron (one per input dim)** — at D=512 that is a far heavier state + kernel than the design doc
scoped, and the dendritic credit-assignment arc already showed the dendritic machinery, while sound, did not deliver a
*discriminating* science result at feasible scale (the VOID). The estimate is a floor, not a ceiling; B2 should be
escalated only after B1 proves the function.

### 6.4 ⚠️ "Decorrelated-but-similarity-preserving" is a GENUINE tension worth naming (the deepest risk)

This is the crux of Option B and may be a real impossibility, not a tuning problem. **Binding wants the codes maximally
decorrelated (orthogonal → invertible). Generalization wants them correlated (similar concepts → similar codes).** These
pull in opposite directions. The 2026-06-06 result is suggestive here — a *partial* (regularized, `C^{−1/3}`) whitening
COMPOSED where *full* whitening (`C^{−1/2}`) OVER-whitened and did NOT compose — which hints there is a "Goldilocks"
amount of whitening that decorrelates enough to bind while preserving enough structure to generalize. **But that result
was measured on composition, NOT on the §5.3 held-out generalization test, and NOT at the reproducibility gate.** So the
load-bearing assumption is: *there exists an operating point that is simultaneously (a) decorrelated enough to bind
reliably, (b) reproducible enough at ≥0.9, AND (c) similarity-preserving enough to generalize.* If those three regions
of operating-point space do not overlap, Option B's premise fails and the project should adopt the biology-faithful DUAL
architecture (similar codes in a "cortex" representation + decorrelated codes in a linked "hippocampal/cerebellar"
expansion, coupled by encode/decode — the complementary-learning-systems answer the build-plan §"deep tension" already
names). **The §5.3 generalization test + the §4 reproducibility gate, run TOGETHER on the SAME operating point, are the
experiment that resolves this — and it has never been run.**

### 6.5 The existing dendritic stack is supervised + off-bridge — repurposing is non-trivial

`sim/dendritic_neuron.py` / `dendritic_plasticity.py` implement *supervised, teaching-signal-gated* learning (the
apical signal IS a top-down teacher) and run as standalone array math, NOT in the bridge step-loop. B1/B2 need the
*unsupervised* local-balance rule and step-loop integration. The machinery (compartments, BAC, no-weight-transport,
kill-safe, deterministic init, GPU backend) is reusable; the *learning objective* and the *wiring* are new work. Calling
the dendritic stack "already done" would over-claim — it is the morphology scaffold, not the whitening front end.

### 6.6 No existing sim ships the mechanism at the project's reproducibility gate (so there is no drop-in port)

The literature check (the "check existing sims FIRST" mandate): the Mikulasch-Priesemann **Priesemann-Group/dendritic_
balance** code exists (Julia + Python) and is the most directly-relevant reference — BUT it is benchmarked on
representation efficiency (MNIST / natural images / speech), and the source paper reports robustness to transmission
delays + across 50 realisations, NOT the project's specific *same-input reproducibility ≥0.9 under σ=0.1* gate. The
Pehlevan-lineage SPIKING similarity-matching net (arXiv 1902.01429) is the closest spiking realisation of the B1 rule.
The gain-modulation route (Duong, the project's prior #1-ranked candidate) has **no demonstrated spiking realisation**
(confirmed by both the prior project doc and this search). **Net: the mechanism is well-supported in the literature and
the Mikulasch code is portable in principle (Julia → the bridge's CuPy/NumPy), but NO existing sim has cleared the exact
gate the project cares about — so the §4 de-risk is genuinely the project's own to run; we adopt the RULE
(Pehlevan-Chklovskii / Urbanczik-Senn / Mikulasch dendritic balance), not a turnkey implementation.**

---

## VERDICT

The owner's Option B — a semantically-structured cortex that generalizes — routes to **adding analog / pre-spike
(dendritic) whitening to the substrate**, because the four committed NEGATIVES converge precisely on the Mikulasch-
Priesemann point-neuron limit: a point neuron sums everything at one soma at low SNR, while a dendrite cancels the common
mode in the analog sub-threshold domain (averaging over compartments + time) where the SNR is high enough to be
reproducible. **The #1 recommendation is Option B1: a separate analog "dendritic-whitening" population that learns the
Pehlevan-Chklovskii lateral rule and computes the whitening in a graded sub-threshold settle BEFORE the spiking binder**
— it is dendritic in FUNCTION without a full neuron-model rewrite, reuses the bridge's FS/homeostasis machinery + the
6-seed-validated whitening rule, and is cheaply falsifiable on the one open gate. **Run the B3 fixed-Ω diagnostic
(half-day) FIRST** to set the ceiling, then the **§4 cheap-first numpy de-risk** (the load-bearing falsification: does
the analog-settle whitening clear *reproducibility ≥0.9 at σ=0.1* on the REAL denoise64 codes, with decorrelation +
composition + the analog-not-host-ZCA proof + the lesion + the M-ratio + the convention-caveat anti-cheats), and only on
GO escalate to the on-bridge B1 build, the §5.3 held-out generalization gate, and the V=320 vocab-ceiling matrix. **B2
(literal per-neuron compartments, Mikulasch-Priesemann / Urbanczik-Senn faithful, the months-scale protected edit) is the
biologically-faithful follow-on** — the path to a proper brain analogue — escalated deliberately after B1 proves the
function.

**The single most important thing for the controller (§6.1):** the project's own record is SPLIT, and tonight's framing
may be too pessimistic. The 2026-06-06 arc (6-seed, committed) already showed a LOCAL whitening rule that COMPOSES at
100% and that spikes HOLD + COMPUTE whitening — which tonight's arc did not cite. The reconciliation is the gate: 06-06
gated on composition, 06-11 on reproducibility. Mikulasch-Priesemann's dendritic balance is specifically the mechanism
for the noise regime tonight's gate tests. **So Option B may be achievable in WEEKS (B1) rather than months (B2) — and
the §4 de-risk is the cheap experiment that decides which.** Do not commit the months-scale rewrite before that
afternoon-scale falsification runs.

**The deepest honest risk (§6.4):** "decorrelated-enough-to-bind, reproducible-enough-at-0.9, AND
similarity-preserving-enough-to-generalize" may be three operating-point regions that do not overlap — in which case
Option B's premise fails and the answer is the biology-faithful DUAL architecture (similar cortical codes + linked
decorrelated hippocampal/cerebellar expansion), not whitening the similar codes directly. The §4 reproducibility gate
and the §5.3 generalization gate, run on the SAME operating point, are the never-yet-run experiment that resolves it.

**No banking.** Reported exactly as found, including the parts that reshape the arc.

## Sources (literature consulted beyond the in-repo catalog/code)

- Mikulasch, Rudelt, Priesemann, *Local dendritic balance enables learning of efficient representations in networks of
  spiking neurons* (PNAS 2021; PMC8685685; arXiv 2010.12395) + code **Priesemann-Group/dendritic_balance** (Julia +
  Python; MNIST / natural images / speech; robustness to transmission delays across 50 realisations — but NOT the
  project's same-input-reproducibility-≥0.9 gate). The mechanism: N dendritic compartments per neuron, each computing a
  local coding-error `u_ji = F_ji(x_i − x̂_i)` in the analog sub-threshold domain; voltage-dependent local rules
  `ΔW_jki ∝ −z_k·u_ji` (inhibition) and `ΔF_ji ∝ (1/F_ji)·z_j·u_ji` (excitation); decorrelation at the membrane-voltage
  level, not the spike output.
- Mikulasch, Rudelt, Wibral, Priesemann, *Dendritic predictive coding* (arXiv 2205.05303, 2022/2023) + *Where is the
  error?* (Trends in Neurosciences 2022) — the dendritic-error / predictive-coding framing; decorrelation from tight
  E/I balance; robustness to noise + neuron loss.
- Urbanczik & Senn 2014, *Learning by the dendritic prediction of somatic spiking* — the canonical learnable
  two-compartment neuron (committed in `sim/dendritic_plasticity.py`).
- Sacramento, Costa, Bengio, Senn 2018, *Dendritic cortical microcircuits approximate the backpropagation algorithm*
  (NeurIPS) + reimpl github.com/miyosuda/dendritic_bp — apical/basal dendritic error microcircuit; RATE-based, not
  spiking.
- Guerguiev, Lillicrap, Richards 2017, *Towards deep learning with segregated dendrites* — segregated apical/basal +
  fixed-random feedback (committed in `sim/dendritic_neuron.py`, `sim/dendritic_mlp.py`).
- Pehlevan & Chklovskii 2015, *Optimization theory of Hebbian/anti-Hebbian networks for PCA and whitening* (arXiv
  1511.09468) + *A Spiking Neural Network with Local Learning Rules Derived From Nonnegative Similarity Matching* (arXiv
  1902.01429) — the B1 lateral rule + its spiking realisation; the saddle-point reason the project's naive anti-Hebbian
  was unstable (the `−M` fixed-point term).
- Duong, Lipshutz, Heeger, Chklovskii, Simoncelli 2023, *Adaptive whitening in neural populations with gain-modulating
  interneurons* (ICML) + *...with fast gain modulation and slow synaptic plasticity* (NeurIPS) — the gain-modulation
  whitening route; rate/graded, **no demonstrated spiking realisation** (so not portable as-is).
- Deneve & Machens 2016, *Efficient codes and balanced networks* (Nat Neurosci) — the balanced-network `Ω=ΓᵀΓ` analytic
  whitening (the B3 handed-in diagnostic).

## Project cross-references (internal, all re-verified)

- The fork + build plan: `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`.
- Tonight's four NEGATIVES: `2026-06-11-cortex-storkey-ca3-cleanup-NEGATIVE.md`, `-dg-ratekwta-cleanup-NEGATIVE.md`,
  `-fixed-expansion-decorrelation-NEGATIVE.md`, and the prior `2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md` /
  `-learned-cleanup-derisk-PARTIAL.md`.
- The positive control: `2026-06-11-cortex-sparse-attractor-poscontrol-GO.md`; the core-binder research:
  `2026-06-11-cortex-core-learned-binder-research.md` (the convention caveat, §1.2).
- **The split-record (the §6.1 reconciliation):** `2026-06-06-decorrelation-blocker-deep-research.md` (the full literature
  ranking — Mikulasch dendritic balance #5/the named escape, gain-modulation #1, the stable-fixed-point rule #2),
  `2026-06-06-whitening-computation-spikes-CAN-compute-it.md` (spikes COMPUTE whitening — the wall REFUTED),
  `2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md` (the LOCAL rule COMPOSES at 100% 6/6),
  `2026-06-06-realobject-grounding-and-whitening-synthesis.md`.
- The committed dendritic stack + its honest terminus: `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`,
  `sim/dendritic_mlp.py`; `2026-05-18-dendritic-fairscale-SOUND-instrument-VOID-strongest-triangulation.md`,
  `2026-05-17-dendritic-faithful-instrument-TERMINUS.md`; the prior design `docs/plans/2026-05-05-dendritic-learning-
  design.md`.
- The step-loop integration precedent: `sim/bridge.py:5788-5859` (dendritic-COINCIDENCE plateau), `sim/kernels.py:255`.
