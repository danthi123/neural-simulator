# Learned generalizing cortex via the dendrite (Phase 3A/3B) — deep-research SCOPING + reconciliation (2026-06-24)

> **Type:** READ-ONLY deep-research + reference-catalog + literature gate (the standing "research-FIRST before
> committing build/GPU/`sim/` effort to overcome a deep frontier" + the SURPASS round before ACCEPTING a boundary).
> NO code written, NO experiments run, NO GPU, NO `sim/` edit. Single deliverable = this doc. Stayed on `main`.
> Every load-bearing claim trust-but-verified against the actual source/code/catalog/finding text (file:line + catalog
> IDs cited); toy-scale / single-seed / regime-bounded flagged where that is the truth. **This is a scoping/decision
> doc, NOT a brain-based result and NOT a commitment to build.**
>
> **Scope:** the burndown roadmap's DEEPEST item — Phase 3A (the dendritic substrate, the enabler) + Phase 3B
> (FHRR → learned generalizing cortex), inventory items **C-1 / H-2 / H-3** (the exact-inverse FHRR algebra + the
> host-designed structure) + **B-1 / B-3** (the dendritic-frontier nav read-outs). `docs/plans/2026-06-23-inventory-burndown-roadmap.md`.
>
> **Build-on, do NOT re-research:** this doc EXTENDS three prior load-bearing docs (read in full this pass) rather than
> re-deriving them — `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` (the fork),
> `2026-06-20-dendrite-substrate-unlock-deep-research.md` (the credit-assignment-vs-graded-read-out SPLIT), and
> `2026-06-22-conversational-scaling-vs-dendritic-scoping.md` (the "dendrite is NOT the conversational unlock" verdict).
> The CORE job the dispatch assigned — **reconcile "B needs the dendrite" (the 2026-06-11 fork + the 2026-06-14 D2 plan)
> vs "generalization achievable without it" (the 2026-06-15/16 PPMI + convergence findings)** — is done in §1, and it
> changes the burndown roadmap's Phase-3B framing.

---

## 0. TOP-LINE (read this first — it RE-FRAMES burndown Phase 3B)

**The burndown roadmap's 3B framing — "FHRR → learned generalizing cortex (the dendrite-gated path)" — is HALF
SUPERSEDED and must be split into two genuinely different residuals that the umbrella "the dendrite unblocks the
generalizing cortex" conflates:**

1. **GENERALIZATION across similar concepts is ALREADY achieved on point neurons — the dendrite is NOT gating it.**
   The 2026-06-11 fork put generalization behind option (B)/the dendrite because it assumed generalization needs
   *decorrelation/whitening* (the Mikulasch-Priesemann analog/pre-spike limit). **That premise was overturned by the
   project's own CYCLE-88→96 work** (`2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`):
   generalization needs **feedforward LOCAL normalization (PPMI)**, which is the *opposite* of decorrelation
   (whitening would *destroy* the semantic similarity that generalization rides on), and PPMI is achievable on point
   neurons with already-shipped local primitives. It is REALIZED on the real spiking substrate (rate-Hebbian
   co-occurrence + population code, `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`), and the
   capstone (perceive a NOVEL object → its concept neurons generalize + FIRE → recall a fact about the matched
   category, moat intact) is GO 3-seed via the hybrid (`2026-06-16-generalization-capstone-verbalize.md`, 0.92).
   **⇒ the D2-Phase-3 GATE ("answer about a held-out concept via a similar known one, moat intact") already PASSES on
   point neurons. Building the dendrite to deliver it would RE-deliver a shipped capability.**

2. **The genuine residual that the point neuron PROVABLY cannot do splits into TWO, and only ONE of them is the
   dendrite's still-open job:**
   - **(R1) Learnable MULTI-ATTRIBUTE composition (a fact = a superposition of bindings).** A from-scratch learned
     bind cannot bundle on point neurons (`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`:
     additive 0.193, learned-linear-inverse 0.056; the fixed ±1 self-inverse bundles 0.989). **The localization is
     exact: unbinding role *t* from a superposition needs a role-SPECIFIC MULTIPLICATIVE inverse `1/u_t`, and a learned
     LINEAR unbind cannot implement a role-dependent scaling — multiplication is a DENDRITIC operation.** This is the
     one place the inventory's C-1/H-3 ("the exact-inverse FHRR algebra is the host-designed idealization") is a
     genuine dendrite candidate. **BUT the cheap-first dendrite test of exactly this already came back NEGATIVE**
     (`2026-06-19-dendritic-binding-toy-derisk.md`: a learned dendritic sigma-pi/plateau conjunction MEMORIZES
     two-attribute bindings (train 0.422) but does NOT generalize (held-out 0.168) — and is WORSE than the fixed FHRR
     primitive it would replace (0.261)). So the dendrite's *learned multiplicative bind* is, on the toy evidence,
     **not the unlock for generalizable multi-attribute composition** either.
   - **(R2) Apical-basal CREDIT ASSIGNMENT** (needed for a *deep/learned* cortex that learns its own binding rule end
     to end). Also tested cheap-first NEGATIVE (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`) — but for a
     STRUCTURAL reason: the apical-basal dendrite does credit assignment **only in HIDDEN-LAYER / DEEP architectures**
     (Sacramento-Senn 2018; Payeur-Naud-Richards 2021, both verified), and the tasks were posed as SINGLE trainable
     layers ("nothing to align"). This residual is **not closed — it is UNTESTED in the regime where the literature
     says it works** (a multi-layer cortex). It is the deepest, highest-variance, and most genuinely-open item.

3. **The ONE dendrite job that is BUILT, validated multi-seed, and SHIPPED is the GRADED READ-OUT of a distributed
   code** — `enable_graded_dendritic_plateau` (`sim/kernels.py:280-330`, `sim/config.py:219-224`, the guarded
   `bridge.py:6441-6479` block, byte-identical when off). It overturned the old "graded value not realizable on point
   neurons" boundary (the nav critic δ = 1.33 via the dendritic plateau, `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`).
   This is the dendrite's REAL earned win, and it directly addresses inventory **B-1** (the place-value δ read-out).
   It does NOT, by itself, do learnable multi-attribute binding or credit assignment.

**Honest SURPASS verdict (the §5 detail):** the FHRR-idealization replacement (a learned cortex that learns its OWN
generalizable multi-attribute binding) is **dendrite-PLAUSIBLE but NOT cheaply dendrite-gated** — both cheap-first
dendrite tests of its two jobs are NEGATIVE, and the one job that survives the literature (credit assignment in a deep
cortex) requires *first re-posing the binder as a multi-layer learned network* (a design change, not a substrate swap),
which is the genuinely-months-scale, highest-variance bet. The generalization half of "3B" is ALREADY done without the
dendrite. **So the burndown roadmap should split 3B: the generalization deliverable is BANKED (point-neuron PPMI); the
"learned generalizable multi-attribute composition" is the genuine open frontier, and its cheapest decisive de-risk is
a DEEP (multi-layer) dendritic binder with the apical-basal credit-assignment machinery that the prior single-layer
NEGATIVEs did not test — run cheap-first, CPU, before any months-scale `sim/` commitment.** An honest "this is
genuinely months and may still be NEGATIVE" is a valid verdict, and it is the most likely one.

---

## 1. THE RECONCILIATION (the core dispatch job): "B needs the dendrite" vs "generalization achievable without it"

The two internal narratives are NOT contradictory once you separate the THREE distinct computations the umbrella
phrase "the generalizing learned cortex" silently bundles. The fork conflated them; the later findings disentangled them.

### 1.1 The three computations the umbrella bundles

| # | Computation | "needs the dendrite"? | Status (verified) |
|---|---|---|---|
| **(i)** | **Semantically-STRUCTURED codes** (similar concepts → similar codes, so the system can generalize cat~dog) | Fork said YES (whitening = dendritic). **Later: NO.** | **Delivered on point neurons** via PPMI feedforward LOCAL normalization (CYCLE 88-96), realized on the spiking substrate (rate-Hebbian + population code), 320-concept, moat clean. |
| **(ii)** | **Single-attribute role-filler BINDING over those codes** | No (a fixed coincidence primitive, or a learned additive bind) | **GO on real spikes** (`2026-06-16-...-single-attr-GO`: on-bridge held-out 0.833 = 100% of numpy). Learnable AND generalizes single-attribute. |
| **(iii)** | **MULTI-attribute composition** (a fact = a bundle/superposition of bindings, recoverable by a role-specific inverse) | **YES — this is the genuine multiplicative/dendritic residual** | **The fixed ±1/FHRR self-inverse bundles 0.989; a LEARNED bind CANNOT (additive 0.193, learned-linear-inverse 0.056).** The localization: unbind needs a role-dependent MULTIPLICATIVE scaling `1/u_t` — a learned LINEAR unbind is structurally incapable. **Multiplication is dendritic.** |

### 1.2 Why the fork said "(i) needs the dendrite" and why that was overturned

- **The fork's logic (2026-06-11):** binding wants DECORRELATED codes (clean, orthogonal → exactly invertible); semantic
  generalization wants CORRELATED codes (similar → similar). Four point-neuron mechanisms (vanilla Hopfield, Storkey,
  spiking DG, fixed random expansion) FAILED to *derive decorrelated codes FROM the brain's correlated codes* → the
  Mikulasch-Priesemann analog/pre-spike whitening limit → "(B) structured-generalizing cortex needs the dendritic
  rewrite." (`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`, the four-NEGATIVE convergence.)
- **The overturn (CYCLE 88, `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`,
  read in full):** *the off-diagonal decorrelation was a RED HERRING.* Generalization does NOT need cross-neuron
  decorrelation — it needs **feedforward LOCAL normalization** (PPMI = `log(count)` Weber-Fechner + `−log(per-hub)` =
  the shipped `input_mean_adapt` + `−log(per-concept)` = Carandini-Heeger divisive normalization + ReLU rheobase, ALL
  local point-neuron ops). Quote (verbatim): *"the months-scale dendritic-plus-lateral off-diagonal build is **(largely)
  unnecessary for the generalizing cortex** — its goal ('a learned, semantically-structured cortex that generalizes')
  is reached by PPMI local normalization."* And decisively: *"whitening over-processes (SM whitens → +0.35; PPMI just
  normalizes → +0.52)"* — i.e. the dendrite's job (whitening) would *hurt* generalization. **PPMI reaches host (+0.518),
  beats ZCA whitening (+0.49), and generalizes (held-out 0.86).**
- **The realization on the substrate (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`):**
  rate-Hebbian co-occurrence learning (6-seed `corr(M,C) +0.686`; STDP is the WRONG rule — symmetric co-occurrence has
  no pre→post order) + the **population code** (lifts single-neuron read-out 47% → 100-108% of host). The full
  conversation runs on the stream-learned codes (3-seed who/what recall 1.00; moat 0.96). **This IS the "learned
  cortex" — on point neurons.** The cross-modal generalization is also point-neuron (cat-acc 0.92 on real spikes via
  NMDA-integration + population read, `2026-06-16-generalization-graded-propagation.md`; the capstone perceive-novel→
  generalize→answer is GO 3-seed via the hybrid, `2026-06-16-generalization-capstone-verbalize.md`).

**⇒ Reconciliation of (i): the dendrite is NOT required for the structured-generalizing cortex. The fork's "(B) needs
the dendrite" was a wrong-hypothesis artifact (it tested "decorrelate the correlated codes," which IS dendritic and
fails on point neurons — but generalization never needed decorrelation; it needed local normalization, which point
neurons do).** This directly contradicts the burndown roadmap's 3B phrasing "generalizes across similar concepts (the
dendrite-gated path)" — generalization is NOT dendrite-gated.

### 1.3 What the dendrite IS still the candidate for (the genuine residual)

Strip away (i) [done on point neurons] and (ii) [done on real spikes], and the residual is **(iii) learnable
MULTI-ATTRIBUTE composition** — and this is where the inventory's C-1/H-2/H-3 actually lives:

- **C-1/H-3 (the exact-inverse FHRR algebra):** the production composer binds the LEARNED (PPMI) codes with a FIXED
  ±1/FHRR self-inverse primitive. That fixed primitive is what bundles multi-attribute facts (0.989). The inventory
  flags it as "host-designed idealization, the step-3 frontier." **The reconciliation: the BIND PRIMITIVE is the
  multiplicative/coincidence-detection operation, and the open question is whether a LEARNED multiplicative bind (a
  dendritic sigma-pi) can replace the fixed ±1 AND generalize multi-attribute.**
- **The honest blocker (2026-06-16 + 2026-06-19):** a learned MULTIPLICATIVE bind needs a role-specific reciprocal
  `1/u_t` to unbind from a superposition. A learned LINEAR inverse cannot be a reciprocal (it breaks even
  single-attribute → 0.056). The *only* operation that gives a role-dependent multiplicative scaling is a
  **dendritic multiplication / NMDA-plateau coincidence** — so this residual is genuinely dendrite-flavored, exactly
  as the inventory's C-1/H-3 → "dendritic frontier" pointer says.
- **BUT the cheap-first dendrite test of this exact job is NEGATIVE** (`2026-06-19-dendritic-binding-toy-derisk.md`,
  read in full): a learned dendritic sigma-pi/plateau conjunction binder MEMORIZES two-attribute bindings (train 0.422)
  but does NOT generalize (held-out 0.168, train→held gap +0.254 = the memorization signature), and is **below the
  fixed FHRR primitive (0.261)**. Lesion (plateau→identity) collapses the train-fit, so the supralinearity IS
  load-bearing for memorization — but generalization does not come. **Verdict from that doc (verbatim): "the binding
  wall is NOT (only) the missing dendritic multiplication … generalizable two-attribute composition is a deeper problem
  (more codes/capacity, a structurally different representation like the F=3 resonator, or a different learning target
  — not a dendritic nonlinearity)."**

**⇒ Reconciliation of (iii): the dendrite is the RIGHT KIND of operation (multiplicative) for multi-attribute binding,
but the cheap-first LEARNED-dendritic-bind test failed to generalize. The residual is real and dendrite-flavored, but a
naive learned dendritic multiplication is NOT (on the toy evidence) its solution.**

### 1.4 The precise residual statement (what STILL needs the dendrite — the dispatch's core ask)

> **PRECISELY what still needs the dendrite, after the reconciliation:**
>
> - **NOT (i) generalization** — done on point neurons (PPMI), and decorrelation (the dendrite's whitening job) would
>   actively HURT it. The D2-Phase-3 conversational-generalization GATE already passes (hybrid, 0.92 3-seed).
> - **NOT (ii) single-attribute binding** — done on real spikes (0.833 = 100% of numpy).
> - **(iii) learnable, GENERALIZABLE MULTI-attribute composition** is the genuine open residual — and it is
>   *dendrite-flavored* (it needs a role-specific multiplicative inverse, a dendritic op) but the *naive* learned
>   dendritic bind tested NEGATIVE (memorizes, doesn't generalize). The two ways it could still be dendrite-gated, BOTH
>   untested in the right regime:
>   1. **A DEEP (multi-layer) learned binder with apical-basal CREDIT ASSIGNMENT** — the single-layer NEGATIVE
>      (`2026-06-19-credit-assignment`) does NOT cover the regime where the literature says the apical-basal dendrite
>      works (Sacramento-Senn / Payeur: HIDDEN layers). A multi-layer learned binder (dendritic credit assignment doing
>      the deep credit it's designed for) is the genuine untested hypothesis for learnable generalizable composition.
>   2. **A structurally-different multi-attribute representation** (the F=3 resonator — already partially built, see
>      §3) that the dendrite's graded read-out could make robust on LEARNED (correlated) codes (the resonator degrades
>      to ~29% on correlated learned codes, `2026-06-19-resonator-on-learned-codes-derisk.md`).
>
> Plus the two NAV dendritic-frontier read-outs (inventory **B-1 place-value δ**, **B-3 TD temporal-credit**) — B-1's
> graded read is ALREADY the shipped `enable_graded_dendritic_plateau` (δ=1.33); B-3's TD-across-delay is the un-closed
> piece (a temporal-credit / eligibility computation the apical compartment could carry).

---

## 2. Catalog-grounded biology (the dendrite the build would add)

Verified against `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` + the biology-buildout-roadmap +
Kandel 6e.

- **G.02 — Active dendrites: local computation, dendritic spikes** (Kandel 6e Ch 13 p 293-298). *Verbatim Sim status:*
  *"missing. Single-compartment everywhere. **This is one of the largest abstractions in the simulator.** … we cannot
  reproduce experiments where the apical-basal coincidence detection is the substrate (e.g. perceptual inference via L5
  apical tuft activity) … **Compartmental neurons would be a major addition (~10× compute per neuron at minimum).**"*
  Biological role (verbatim): *"dendrites contain voltage-gated Na⁺, Ca²⁺, and HCN channels … NMDA spikes (NMDAR-driven
  plateau potentials), and Ca²⁺ spikes at the apical tuft. These produce nonlinear summation rules (e.g. cluster of
  inputs on one branch ≫ scattered inputs on many branches), gain modulation by apical-basal coincidence (Larkum's
  two-layer model)."* ⇒ **G.02 is the canonical entry; it is the multiplicative/coincidence nonlinearity the (iii)
  residual needs, and the multi-attribute-binding hypothesis IS "cluster of inputs on one branch ≫ scattered" applied
  to role-filler conjunctions.**
- **J.08 — NMDA receptor, voltage-dependent coincidence detector** (Kandel 6e Ch 13 p 281-286). *Sim status:
  IMPLEMENTED* (`fused_nmda_update_and_current`; the Jahr-Stevens Mg²⁺ block; slow τ ~50-150 ms). ⇒ the NMDA-spike
  kinetics the dendritic plateau reuses are already in-engine; the `fused_coincidence_plateau` /
  `fused_graded_dendritic_plateau` kernels are NMDA-spike-faithful (Jahr-Stevens block, dual-exponential).
- **B.17 — Sp-Sp dendritic inhibition / voltage-dependent dendritic linearization** (Wilson, PBR-160 ch 6) — confirms
  the striatum's MSN dendrite is electrotonically nonlinear (KIR2/Kv-2), the substrate the graded-plateau read-out
  exploits for the value critic.
- **F.12 (codon/granule sparse expansion) + D.12 (DG pattern separation)** — the structural-expansion decorrelation the
  2026-06-11 fork leaned on. Now de-emphasized (decorrelation is the red herring for generalization), but the codes
  remain validated (`generate_sparse_patterns`, sparse-distributed composes 1.000 to 320 — the FLAT cortex).
- **T3.A — Compartmental neurons (Cluster G + I), biology-buildout-roadmap.** *Verbatim:* tier =
  **"Modelable with major architecture change … Months to years; warrants a separate research arc decision."** *What:*
  *"at minimum a 2-compartment 'soma + apical dendrite' version of the L5 pyramidal cell, supporting active-dendrite
  computation, NMDA spikes, and Larkum's BAC firing."* *Cost (verbatim):* *"~10× compute per neuron … New GPU kernel
  architecture (compartment-coupled membrane equations). Does not compose cleanly with existing kernels — requires
  substantial rewrite of `sim/kernels.py`."* **Decision criterion (verbatim): "pursue when a target experiment requires
  it AND we've exhausted single-compartment alternatives."** ⇒ the catalog's own gate for the full two-compartment
  neuron is "a target experiment requires it" — and the present finding is that the named conversational experiments
  do NOT require it (generalization done on point neurons), while the one that MIGHT (learnable generalizable
  multi-attribute composition) has NOT exhausted single-compartment alternatives (the F=3 resonator + more
  codes/capacity, §3).

**Literature (verified, from the prior gates):** Larkum 2013 (BAC firing); Major-Larkum-Schiller 2013 (NMDA spike);
Poirazi-Brannon-Mel 2003 (two-layer subunit — the model behind `fused_coincidence_plateau`); Mikulasch-Rudelt-Wibral-
Priesemann *Trends Neurosci* 2023 (dendritic error computation = the point-neuron limit, PubMed 36577388);
Urbanczik-Senn *Neuron* 2014 (the local dendritic third-factor rule = `sim/dendritic_plasticity.py`, PubMed 24507189);
**Sacramento-Costa-Bengio-Senn NeurIPS 2018 + Payeur-Naud-Richards Nat Neurosci 2021 — apical-basal credit assignment
requires MULTILAYER/HIERARCHICAL architectures** (the load-bearing reason the single-layer NEGATIVE doesn't close R2).

---

## 3. What is ALREADY BUILT (do NOT re-build) — the D2 dendritic stack + the resonator

Verified file:line. **Substantially more is built than "D2 Phase 0-2 + Phase 3 pending."**

| Asset | What it is | File:line | Status |
|---|---|---|---|
| `DendriticLayer` | spiking two-compartment BAC neuron: basal forward `x@W_basal`, apical via FIXED-RANDOM `B_apical` (feedback alignment, no weight transport), soma BAC (apical depol LOWERS effective threshold) | `sim/dendritic_neuron.py:20-58` | Built, numpy, biologically-local |
| `urbanczik_senn_update` | the LOCAL somato-dendritic third-factor plasticity rule | `sim/dendritic_plasticity.py:17-41` | Built |
| `DendriticMLP` | a **DEEP** feedback-alignment MLP (per-hidden-layer fixed-random `B` → hidden learning via Urbanczik-Senn) — **the hidden-layer credit-assignment machine R2 would need** | `sim/dendritic_mlp.py` | Built (the deep credit-assignment scaffold ALREADY EXISTS) |
| `fused_graded_dendritic_plateau` | the GRADED, non-saturating dendritic-plateau READ-OUT (a graded analog value from a distributed code; Mikulasch-Priesemann); NMDA-spike kinetics + no-drive floor subtraction | `sim/kernels.py:280-330` | **SHIPPED, byte-inert when off** |
| `enable_graded_dendritic_plateau` (+ center/slope/strength/tau) | the config + guarded bridge block (alloc `cp_conductance_g_graded_plateau` None when off; per-step block) | `sim/config.py:219-224`; `sim/bridge.py:1428-1430,6441-6479` | **SHIPPED** (test `test_graded_dendritic_plateau.py` 5/5; byte-identity-when-off proven) |
| `enable_dendritic_divisive_gain` | the per-source divisive gain (D2 Phase-1 narrower form) | `sim/config.py:260` | SHIPPED, byte-inert when off, found NOT load-bearing for the cortex (Phase-2 inversion) |
| `--dendrite-critic` (graded plateau as the nav value V) | the dendritic-plateau value read-out deployed into the production nav-RL critic (δ=1.33, multi-seed value clean) | `g11_bg_runner.py:476-479,4697-4720` | SHIPPED default-off; **B-1 graded read = DONE** |
| F=3 resonator (two-attribute) | the FHRR resonator for two-attribute binding (lifted K=5 on random codes; **degrades ~29% on LEARNED correlated codes**) | `2026-06-19-resonator-on-learned-codes-derisk.md` | the single-compartment alternative for (iii), partially built |

**D2 phase status (verified `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`):**
- **Phase 0** (numpy spiking two-compartment, `dendritic_d1p7_spiking_twocompartment_derisk.py`): **SURVIVES** (the
  per-compartment advantage holds through a genuine spiking soma).
- **Phase 1** (protected `sim/` edit): the SHIPPED realization is the **graded-plateau read-out** (`enable_graded_dendritic_plateau`)
  + the divisive gain — **NOT** a full `NeuronModel.TWO_COMPARTMENT` second-state (`v_dend`) neuron. That broader form
  (a second per-neuron state + a compartment-coupled membrane equation, the T3.A architecture) is the larger un-built
  edit.
- **Phase 2** (learned graded cortex embedding): **HONEST NEGATIVE for the gain's necessity** (the clean-readout control
  inverts: with enough temporal integration the point neuron recovers the structure on its own; the gain HURTS).
- **Phase 3** (#23, plug learned graded codes → bind/unbind/cleanup → conversational matrix; GATE = generalization-in-
  conversation, moat intact): **PENDING — and its target capability already ships on point neurons (§1.2), so it is
  redundant for generalization.**

**⇒ The reuse is high: the dendrite NEURON, the deep credit-assignment MLP, the graded read-out, AND the F=3 resonator
all exist. The genuinely-NEW build for the (iii) residual is a DEEP (multi-layer) LEARNED binder wiring the existing
`DendriticMLP` credit-assignment machine to a multi-attribute composition task — which has never been tested.**

---

## 4. The CHEAPEST-FIRST de-risk (the smallest probe that proves a dendritic binder learns generalizable
## MULTI-attribute composition — the known point-neuron NEGATIVE)

The dispatch asks for the smallest probe that proves a dendritic two-compartment binder learns MULTI-attribute
composition + generalizes held-out (the known point-neuron NEGATIVE). The prior cheap-first dendrite-bind test
(`2026-06-19-dendritic-binding-toy-derisk.md`) was NEGATIVE — but it tested a **SINGLE-layer** learned dendritic
sigma-pi. **The untested hypothesis (the one the literature says could work) is a DEEP (multi-layer) learned binder
with apical-basal credit assignment.** That is the cheapest decisive next de-risk, and it is CPU/numpy with NO
`sim/` edit (it reuses `DendriticMLP` + `DendriticLayer` + `urbanczik_senn_update` + the existing rigorous binding
harness).

**Name it:** *Does a DEEP (≥2 hidden-layer) learned dendritic binder — credit-assigned by the apical-basal feedback-
alignment machinery the `DendriticMLP` already implements — learn invertible MULTI-attribute (3-way bundled) composition
that GENERALIZES to held-out role-filler combinations, where the SINGLE-layer learned dendrite (0.168) and the learned-
linear inverse (0.056) provably cannot, and ideally reaching the fixed-FHRR ceiling (0.989 bundle / 0.261 the prior
two-attr held-out bar)?*

**Stage 0 (CPU/numpy, NO `sim/` edit, ~minutes-hours — the decisive gate before any bridge work):** extend the existing
`_phaseB_dendritic_bind_derisk.py` / `_phaseB_multiplicative_bind_bundled_derisk.py` harness (leakage-free
systematicity splits, memorization-floor, the fixed-FHRR positive control, R=4/F=16, 3 leakage-free splits, 3 seeds —
all already implemented). Add ONE arm: a **DEEP dendritic binder** = `DendriticMLP` (≥2 hidden layers, the deep
credit-assignment regime) producing the bound vector + a learned dendritic unbind, trained with the Urbanczik-Senn
local rule (NO backprop weight transport — feedback alignment, the brain-faithful path). Arms on the IDENTICAL
codes/splits:

| arm | binder | role |
|---|---|---|
| `single_layer_dendrite` (prior NEGATIVE, the control) | the 2026-06-19 single-layer sigma-pi | MUST fail held-out (0.168 — already shown) |
| `learned_linear_inverse` (control) | additive bind + learned linear unbind | MUST fail (0.056 — already shown) |
| `deep_dendrite` (TEST) | `DendriticMLP` ≥2 hidden layers + Urbanczik-Senn credit assignment | the question: does held-out generalize > 0.40 (the GO bar), ideally → the fixed-FHRR primitive |
| `fixed_FHRR` (positive control / ceiling) | the ±1 self-inverse algebra | bundles 0.989 / two-attr held-out 0.261 |
| `memorization_floor` | lookup table | MUST ≈ 0.000 on leakage-0 held-out |

**The decisive metric (pre-register, fixed):** held-out (leakage-asserted) MULTI-attribute (3-way bundled) recovery,
3-seed. **GO** = `deep_dendrite` held-out ≥ 0.40 AND > both point-neuron controls AND > the single-layer dendrite's
0.168, with the train→held gap SMALL (generalizes, not memorizes). **BOUNDARY** = it beats the single-layer dendrite
but stays below the fixed FHRR (a characterized partial). **NEGATIVE** = the deep dendrite ALSO memorizes-but-doesn't-
generalize (then the dendrite is comprehensively ruled out for learnable generalizable composition — itself a valuable,
months-saving deliverable, and the strong prior given the two existing NEGATIVEs).

**The complementary cheaper alternative-path de-risk (run in parallel, also CPU, NO `sim/` edit):** the F=3 resonator
on LEARNED codes + the SHIPPED graded-plateau read-out — does the dendrite's graded read-out lift the resonator's
~29%-on-correlated-codes (`2026-06-19-resonator-on-learned-codes-derisk.md`) back toward its random-code ceiling? This
tests the "structurally-different representation" path (§1.4 option 2) that the 2026-06-19 binding doc named as the
likelier route than a learned dendritic nonlinearity. This one CAN reuse `enable_graded_dendritic_plateau` directly.

**Stage 1 (only if Stage 0 GO — the protected `sim/` edit):** the full `NeuronModel.TWO_COMPARTMENT` second-state
(`v_dend`) neuron (the T3.A architecture the D2 Phase-1 plan named but did NOT ship — the shipped form is the read-out
term only), guarded `cfg.enable_two_compartment`, byte-identical when off, byte-level diff review. This is the
months-scale `sim/kernels.py` rewrite the catalog T3.A flags; it is ONLY warranted if Stage 0 GO.

**Why this de-risk and not re-running the single-layer one:** the single-layer learned-dendritic-bind is NEGATIVE and
the literature explains why a single layer fails (nothing to credit-assign); the genuinely-untested hypothesis is the
DEEP regime the apical-basal dendrite is *designed* for — and the `DendriticMLP` machine to test it ALREADY EXISTS, so
this is a reuse-by-import CPU probe, not a build.

---

## 5. Anti-cheats the de-risk needs

1. **Fodor-Pylyshyn held-out SYSTEMATICITY split, leakage-asserted.** Hold out R novel (role, filler) combinations such
   that every role AND every filler still appears in some training combo — only the specific PAIRINGS are held out. A
   `leakage_count == 0` assert (train ∩ held-out = ∅, every atom covered). This is the EXACT protocol the prior probes
   use (`2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED.md`); reuse it.
2. **Memorization-floor control.** A pure lookup table must score ≈ 0.000 / chance on the leakage-0 held-out combos. (If
   the test arm only matches the lookup table, it memorized.)
3. **Fixed-FHRR positive control / ceiling (the harness-soundness gate).** The fixed ±1 self-inverse MUST bundle ~0.989
   / two-attr held-out 0.261 on the IDENTICAL harness — this is what makes the NEGATIVEs real (the harness DETECTS
   working bundling). Re-assert it in the SAME run, not cited.
4. **Apical/plateau LESION (the decisive dendrite anti-cheat).** Replace the regenerative dendritic nonlinearity with a
   passive/identity compartment — the dendritic arm MUST collapse to the additive/point-neuron floor. (If generalization
   survives the lesion, it isn't coming from the dendritic nonlinearity — the exact confound the 2026-06-19 lesion
   caught.)
5. **Multi-seed (42/43/44 minimum; 6-seed for any GO claim).** The single-layer credit-assignment toy was
   seed-unstable (0.917 / 0.083 / 0.583) — a deep-binder GO MUST be multi-seed-robust, not a lucky-alignment seed.
6. **Brain-faithfulness: feedback alignment, NO weight transport.** The credit assignment uses the fixed-random
   `B_apical` (the `DendriticMLP` design), NOT backprop'd `Wᵀ`. A backprop arm is allowed ONLY as a fenced
   positive-control ceiling (the `DendriticMLP` oracle), never as the deliverable.
7. **Correlated-code regime (the real test).** Run on the LEARNED/correlated PPMI stream codes (the production regime),
   not just decorrelated codes — generalizable composition on CORRELATED codes is the actual target (the fixed FHRR
   degrades there; the question is whether a learned deep dendrite does better).

---

## 6. The SURPASS verdict — is the FHRR-idealization replacement dendrite-gated, and how cheaply?

Per the standing SURPASS discipline (a boundary is accepted ONLY after it survives: isolate+quantify the genuine
residual → reframe via real biology → rank cheap surpass mechanisms → verdict surpassable-or-precisely-why-not):

**(1) ISOLATE + QUANTIFY the genuine residual.** Of the "FHRR-idealization replacement" umbrella:
- generalization (i) — **NOT residual** (done on point neurons, PPMI; the dendrite would HURT it).
- single-attribute binding (ii) — **NOT residual** (done on real spikes, 0.833).
- **multi-attribute composition (iii) — the genuine residual.** Quantified: the fixed ±1 bundles 0.989; a learned
  bind cannot (additive 0.193 / learned-linear-inverse 0.056); a learned SINGLE-LAYER dendritic sigma-pi memorizes but
  doesn't generalize (held-out 0.168, below the fixed primitive 0.261). **The genuine residual is exactly: "a LEARNED
  generalizable multi-attribute bind," and it is ~a single multiplicative-inverse operation the point neuron and the
  single-layer dendrite both fail.** Plus the two nav read-outs (B-1 done via graded plateau; B-3 TD-across-delay open).

**(2) REFRAME via real biology.** The residual is multiplicative (role-specific reciprocal `1/u_t`), which IS dendritic
(NMDA-plateau coincidence, G.02). The reframe the 2026-06-19 binding doc already made: *is the binding wall the missing
dendritic multiplication, or a deeper problem (capacity / a different representation / a different learning target)?*
The cheap-first single-layer dendrite test answered "deeper" — memorizes, doesn't generalize. **The remaining
biology-reframe: the apical-basal dendrite does credit assignment in DEEP/HIERARCHICAL circuits (Sacramento-Senn,
Payeur) — so the untested hypothesis is a DEEP learned binder, not a single-layer one.** Real cortex binds with a
hierarchy (V1→V2→IT), not one layer.

**(3) RANK cheap surpass mechanisms (cheapest PAST it first):**
- **(a) [CHEAPEST, NO `sim/` edit] The DEEP dendritic binder de-risk (§4 Stage 0).** Reuses `DendriticMLP` (the deep
  credit-assignment machine already exists). Tests the one untested regime. CPU/hours. The decisive cheap experiment.
- **(b) [CHEAP, NO `sim/` edit] The F=3 resonator + graded-plateau-read-out on LEARNED codes.** Tests the
  "structurally-different representation" path (the 2026-06-19 doc's named likelier route); reuses the SHIPPED
  `enable_graded_dendritic_plateau`. The non-dendrite-bind alternative for (iii).
- **(c) [CHEAP, ALREADY DONE for B-1] The graded read-out.** The B-1 place-value δ is ALREADY surpassed by
  `enable_graded_dendritic_plateau` (δ=1.33). B-3 (TD temporal-credit-across-delay) is the open nav piece — the apical
  compartment carrying an eligibility/temporal-credit trace is a bounded next probe.
- **(d) [DEEP, MONTHS, only if (a) GO] The full two-compartment `NeuronModel.TWO_COMPARTMENT` rewrite (T3.A).** The
  ~10×-compute, `sim/kernels.py`-rewrite the catalog gates on "a target experiment requires it." Warranted only if (a)
  proves a deep dendritic binder generalizes.

**(4) VERDICT — surpassable-and-how-cheaply, vs genuinely-far:**

> **The generalization deliverable (the burndown 3B headline) is ALREADY surpassed WITHOUT the dendrite (point-neuron
> PPMI) — BANK it; the roadmap's "dendrite-gated generalization" phrasing is superseded.**
>
> **The FHRR-idealization replacement for LEARNABLE GENERALIZABLE MULTI-ATTRIBUTE composition (the genuine residual,
> C-1/H-3) is dendrite-PLAUSIBLE but NOT cheaply dendrite-gated, and is genuinely FAR (months, high-variance, likely
> still NEGATIVE):**
> - both cheap-first dendrite tests of its two jobs (learned multiplicative bind; single-layer credit assignment) are
>   NEGATIVE;
> - the one surviving hypothesis (a DEEP learned binder with apical-basal credit assignment, the regime the literature
>   supports) is **untested but cheaply testable** (§4 (a), reuse `DendriticMLP`, CPU/hours, NO `sim/` edit) — run THIS
>   before any commitment;
> - even a Stage-0 GO leads only to a months-scale `sim/kernels.py` rewrite (T3.A) for the full two-compartment neuron,
>   gated on the cheap de-risk;
> - the strong prior (two existing NEGATIVEs + the 2026-06-22 "dendrite is not the conversational unlock" verdict + the
>   2026-06-19 "binding wall is a deeper problem than a dendritic nonlinearity") is that the deep-binder de-risk is
>   ALSO likely NEGATIVE — in which case the dendrite is comprehensively ruled out for learnable generalizable
>   composition, the fixed ±1/FHRR primitive STAYS (load-bearing, biology-grounded as binding-by-coincidence — NOT a
>   host shortcut), and the residual is honestly a CHARACTERIZED point-neuron boundary, not a fixable one.
>
> **This is the deepest, highest-variance burndown item, and an honest "genuinely months, and the cheap de-risk will
> probably confirm the dendrite is not the unlock either" is the most likely verdict.** It is accepted as a boundary
> ONLY after the §4(a) deep-binder de-risk survives the SURPASS round — i.e. run the cheap CPU probe first; if NEGATIVE,
> the FHRR idealization is the honest brain-grounded primitive and the residual is a documented boundary, not a fix.

---

## 7. Recommended sequence + `sim/`-edit flags

| Step | Action | `sim/` edit? | Cost | Gate |
|---|---|---|---|---|
| **A** | **BANK the reconciliation:** correct burndown 3B — generalization is DONE on point neurons (PPMI), NOT dendrite-gated. Update the roadmap/inventory so 3B = "learnable generalizable MULTI-attribute composition" (the genuine residual), not "generalization." | NO | minutes (doc) | — |
| **B (cheapest-first, RECOMMENDED NEXT)** | **The DEEP dendritic binder de-risk (§4 Stage 0).** Reuse `DendriticMLP` + `DendriticLayer` + `urbanczik_senn_update` + the existing binding harness. 3-seed, all §5 anti-cheats. | **NO** (reuse-by-import) | hours, CPU | GO = held-out multi-attr ≥ 0.40, multi-seed, lesion collapses, > both controls. NEGATIVE = dendrite ruled out for learnable composition. |
| **C (parallel, cheap)** | **The F=3 resonator + graded-plateau read-out on LEARNED codes.** Tests the "different representation" path; reuses the SHIPPED `enable_graded_dendritic_plateau`. | NO | hours, CPU/GPU | GO = the graded read-out lifts the resonator's ~29%-on-correlated back toward its random-code ceiling. |
| **D (bounded nav)** | **B-3 TD temporal-credit-across-delay** via the apical compartment (eligibility trace). B-1 graded read-out is ALREADY done (δ=1.33). | maybe (additive, byte-review) | days | the merged-anti-cheat that the B-3 standalone GO couldn't certify value-driven. |
| **E (ONLY if B GO — months)** | **The full `NeuronModel.TWO_COMPARTMENT` second-state neuron (T3.A).** A second per-neuron `v_dend` state + compartment-coupled membrane eqn; guarded `cfg.enable_two_compartment`, byte-identical when off, byte-level diff review. The `sim/kernels.py` rewrite the catalog T3.A gates. | **YES (deep, protected hot-path)** | months | the D2 Phase-3 conversational gate, now scoped to learnable generalizable composition (NOT generalization, which is done). |

**`sim/`-edit summary:** Steps A-D need NO new `sim/` edit (B/C reuse-by-import; D is a small additive byte-reviewed
term if pursued). The ONLY months-scale protected `sim/` edit is Step E (the full two-compartment neuron), and it is
gated behind the cheap-first Step-B GO that, on the strong prior, is unlikely. **Per the owner's standing "close all
genuine shortcuts" + "dendrite is fair game when it's the obvious unlocker" + "honest negatives ARE the deliverable":
run B (and C) cheap-first; if NEGATIVE, the FHRR primitive is the honest brain-grounded binding-by-coincidence and the
multi-attribute residual is a CHARACTERIZED boundary — not a months-scale build to chase.**

---

## 8. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (read in full / file+line cited):**
- The fork (flat-A vs structured-B-needs-dendrite): `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`,
  read in full (the four-NEGATIVE decorrelation convergence + the Mikulasch-Priesemann limit).
- The PPMI overturn of (i): `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (verbatim
  "dendritic off-diagonal build is unnecessary for the generalizing cortex"; whitening over-processes) + the on-bridge
  realization `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`; the generalization-frontier survey
  (10 docs, 2026-06-15/16) confirming point-neuron PPMI + visual-similarity + Hebbian-convergence + NMDA-graded
  propagation, all explicitly "dendrite NOT required / off the critical path."
- The (iii) residual localization: `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`, read in
  full (additive 0.193 / learned-linear-inverse 0.056 / fixed-FHRR 0.989; "multiplication is a DENDRITIC operation").
- The single-layer dendritic-bind NEGATIVE: `2026-06-19-dendritic-binding-toy-derisk.md`, read in full (held-out 0.168,
  below FHRR 0.261; "the binding wall is NOT (only) the missing dendritic multiplication … a deeper problem").
- The credit-assignment SPLIT + the graded-read-out recommendation: `2026-06-20-dendrite-substrate-unlock-deep-research.md`,
  read in full (the credit-assignment-vs-graded-read-out distinction; single-layer "nothing to align"; the
  Sacramento-Senn/Payeur "needs hidden layers" verification).
- The prior conversational-vs-dendritic verdict: `2026-06-22-conversational-scaling-vs-dendritic-scoping.md`, read in
  full (the three premise-overturning findings; dendrite NOT the conversational unlock).
- The D2 build plan + phase status: `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`, read in full (Phase 3
  gate verbatim).
- The SHIPPED dendrite code: `sim/dendritic_neuron.py:20-58`, `sim/kernels.py:253-330` (both plateau kernels),
  `sim/config.py:219-224`, read directly.
- The graded-plateau B-1 win: `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md` (δ multi-seed clean; the via the
  D2-state survey).
- Catalog: G.02 (active dendrites MISSING, ~10× compute), J.08 (NMDA coincidence detector IMPLEMENTED), B.17, T3.A
  (compartmental neurons, "months to years, separate research arc decision") — read directly from
  `feature-catalog.md` + `biology-buildout-roadmap.md`.
- The inventory framing of C-1/H-2/H-3/B-1/B-3: `2026-06-23-cheats-shortcuts-integration-inventory.md`, read in full.

**Flagged (could NOT fully verify / honest uncertainty):**
1. **Whether a DEEP learned dendritic binder (§4 Stage 0, the recommended de-risk) would generalize multi-attribute** —
   GENUINELY OPEN and the whole point of the de-risk. The strong prior (two single-layer NEGATIVEs + the "deeper
   problem" finding) is that it is ALSO likely NEGATIVE, but the DEEP regime (the one the literature supports) is
   untested. I do NOT predict it; flagged as a research bet.
2. **The F=3 resonator's exact ~29%-on-learned-codes number** — read from `2026-06-19-resonator-on-learned-codes-derisk.md`
   (not re-run, read-only); internally consistent, not load-bearing for the direction.
3. **The exact PPMI decimals (+0.518 / host +0.442 / ZCA +0.49)** — read from CYCLE 88-96 findings, not re-run; not
   load-bearing for the *direction* (generalizing cortex shipped on point neurons).
4. **Some D2-survey file:line specifics** (e.g. the precise `bridge.py` guarded-block line range 6441-6479) were
   reported by a read-only sub-survey of the code; the kernel + config lines I verified directly, the bridge-block
   range I did not re-open line-by-line (consistent with the surveyor's report + the byte-identity test claim).

---

## Sources

**Project record (re-verified this pass):**
- `docs/plans/2026-06-23-inventory-burndown-roadmap.md` (Phase 3A/3B); `research/findings/2026-06-23-cheats-shortcuts-integration-inventory.md` (C-1/H-2/H-3/B-1/B-3).
- `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md` (the fork); `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` + `docs/plans/2026-06-14-step3-cortex-fork-resolved-dendritic-D2-decision.md` (D2).
- `research/findings/2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`, `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`, `2026-06-15-biology-faithful-online-stream-cortex-reaches-target.md`.
- `research/findings/2026-06-16-generalization-{frontier-scoping,crossmodal-unify-cheap-first,optionB-visual-similarity,onsubstrate-convergence,graded-propagation,capstone-vision-to-concept,capstone-verbalize}.md`.
- `research/findings/2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`; `2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED.md`.
- `research/findings/2026-06-19-dendritic-binding-toy-derisk.md`; `2026-06-19-dendrite-credit-assignment-toy-stage1.md`; `2026-06-19-resonator-on-learned-codes-derisk.md`.
- `research/findings/2026-06-20-dendrite-substrate-unlock-deep-research.md`; `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`; `2026-06-20-burndown-9-critic-graded-readout.md`.
- `research/findings/2026-06-22-conversational-scaling-vs-dendritic-scoping.md` (the prior reconciliation this extends).
- Code: `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`, `sim/dendritic_mlp.py`, `sim/kernels.py:253-330`, `sim/config.py:219-224,260`, `sim/bridge.py:6441-6479`, `research/runners/g11_bg_runner.py` (`--dendrite-critic`).

**Catalog (`E:\Documents\Projects\sim-catalog\references\`):** `feature-catalog.md` G.02 (active dendrites), J.08 (NMDA coincidence detector), B.17 (dendritic linearization), F.12/D.12 (sparse expansion/pattern separation); `biology-buildout-roadmap.md` T3.A (compartmental neurons — "months to years, separate research arc decision").

**Literature (verified via prior gates):** Larkum 2013 (BAC firing); Major-Larkum-Schiller 2013 (NMDA spike); Poirazi-Brannon-Mel 2003 (two-layer subunit); Mikulasch-Rudelt-Wibral-Priesemann *Trends Neurosci* 2023 (PubMed 36577388, dendritic error = point-neuron limit); Urbanczik-Senn *Neuron* 2014 (PubMed 24507189, the local rule); Sacramento-Costa-Bengio-Senn NeurIPS 2018 + Payeur-Naud-Richards *Nat Neurosci* 2021 (apical-basal credit assignment needs MULTILAYER/HIERARCHICAL — PubMed 34728832).

_Read-only deep-research deliverable. NO code, NO experiments, NO `sim/` edit. The dispatch's core ask — reconcile
"B needs the dendrite" vs "generalization achievable without it" — resolves to: generalization is DONE on point neurons
(PPMI; the dendrite would hurt it), so burndown 3B's "dendrite-gated generalization" is superseded; the genuine
dendrite-flavored residual is LEARNABLE GENERALIZABLE MULTI-ATTRIBUTE composition (the FHRR-idealization replacement),
which both cheap-first dendrite tests came back NEGATIVE on, with ONE untested surviving hypothesis (a DEEP learned
binder with apical-basal credit assignment) that is cheaply testable (reuse `DendriticMLP`, CPU/hours, NO `sim/` edit)
and, on the strong prior, likely also NEGATIVE — an honest "genuinely months and probably not the unlock" verdict,
accepted as a boundary only after the cheap deep-binder de-risk survives._
