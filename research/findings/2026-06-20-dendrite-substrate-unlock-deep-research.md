# Dendrite-substrate unlock — deep-research gate: the cheap-first de-risk that would prove a dendrite earns its keep (2026-06-20)

**Type:** READ-ONLY deep-research + reference-catalog + literature gate (the standing "research-FIRST before committing build/GPU/`sim/` effort to overcome a deep frontier"). NO code written, NO experiments run. ONE findings doc. Stayed on `main`. Every load-bearing claim trust-but-verified against the actual source/code/catalog text (file:line + catalog IDs cited); where a claim is toy-scale / single-seed / regime-bounded that is flagged.

**The dispatch's premise (to be verified, not assumed):** the same DENDRITIC wall blocks all three of (a) the merged living agent's SURVIVAL POLICY, (b) the nav orienting/critic READ-OUTS (#6/#9), and (c) the un-converted graded-magnitude COGNITION (learned binder / FHRR-B / value read-out) — so the dendrite is the "obvious unifying unlocker," and the question is the cheapest de-risk that proves it.

**Owner context:** the dendrite is PRE-AUTHORIZED "when it's the obvious unlocker" and is NOT-from-scratch (D2 Phase 0-2 built a two-compartment neuron + a learned graded cortex; Phase 3 pending). `feedback_dendritic_substrate_fair_game`; BRAIN-BASED-ONLY standard applies; honest negatives ARE the deliverable.

---

## 0. TOP-LINE (read this first — it partially OVERTURNS the dispatch's "one wall" framing)

**The verified picture is sharper than "one dendrite unlocks all three," and the prompt's flagged self-check ("if more is built than expected, SAY SO; the controller caught overclaims this arc") fires here. The honest synthesis:**

1. **More dendritic machinery is ALREADY built than "Phase 0-2" suggests, AND two of its three named jobs have ALREADY been tested cheap-first and came back NEGATIVE.** The project has: a spiking two-compartment neuron (`sim/dendritic_neuron.py`), the Urbanczik-Senn local plasticity rule (`sim/dendritic_plasticity.py`), a deep feedback-alignment MLP with hidden-layer credit assignment (`sim/dendritic_mlp.py`), AND a SHIPPED on-bridge protected dendritic divisive-gain (`enable_dendritic_divisive_gain`, `sim/config.py:233`, byte-identical when off). The two cheap-first toy de-risks of the dendrite's named jobs — **(a) apical-basal CREDIT ASSIGNMENT** (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`) and **(c) learnable multi-attribute BINDING** (`2026-06-19-dendritic-binding-toy-derisk.md`) — are **BOTH NEGATIVE on favorable toys.**

2. **The three blockers are NOT one wall — they split into TWO distinct dendritic questions, and only ONE of them is still genuinely open.**
   - **The credit-assignment / learned-policy question (blockers a + the bulk of c's binding):** tested, NEGATIVE on the substrate's actual structure. The apical-basal dendrite does credit assignment **only in HIDDEN-LAYER / DEEP architectures** (Sacramento-Senn 2018; Payeur-Naud-Richards 2021 — both verified below). The nav actor is a **SINGLE trainable layer** ("nothing to align"), and the learned binder memorizes-but-doesn't-generalize — so a dendrite, as the project posed these tasks, is NOT the unlocker. This is the controller-flagged overclaim: the 2026-06-19 water-maze NEGATIVE's "resolves toward the dendrite" was a *direction-of-research* call, and the SAME-week boundary-ledger + nav-loop gate (both 2026-06-20) re-classify the nav loop as **point-neuron loop-stability, NOT dendritic**.
   - **The graded-read-out-of-a-distributed-code question (blocker b's #9, and the deepest part of c):** this is the ONE place the project repeatedly hits a *genuine, characterized* dendritic-flavored wall that the point neuron provably cannot cross — and it has NOT been tested as a standalone dendrite de-risk. **This is the real unlock candidate, and it is cheaper + more leverage-rich than the credit-assignment task.**

3. **THE RECOMMENDED CHEAP-FIRST DE-RISK is therefore NOT the credit-assignment task (twice-NEGATIVE) but a GRADED DENDRITIC READ-OUT of a distributed code** — the exact computation the burndown-9-critic doc (today) proved a point-neuron MSN cannot do (linear summation is sub-rheobase; the all-or-none coincidence plateau over-clamps; **neither expresses the graded middle**). A two-compartment neuron whose dendritic compartment performs an active, graded, regenerative-but-non-saturating integration is the named biology (Larkum BAC plateau; Mikulasch-Priesemann dendritic error). The decisive metric: a graded δ = r − V (or a graded category read-out) that the point-neuron control provably can't produce, with an apical-lesion that collapses it. Reuse the D2 Phase 0-2 two-compartment neuron; the new piece is a **graded (non-saturating) dendritic-plateau read-out**, a small additive `sim/` term in the same guarded slot the existing `fused_coincidence_plateau` occupies.

4. **Honest top-line for the owner:** the dendrite is a real tool that has been **reached for, partially built, and tested on two of its three jobs (both NEGATIVE)**. The *one* job where it is still genuinely the obvious unlocker is the **graded read-out of a distributed code** — and that is a cheap-first, well-grounded de-risk worth running BEFORE any months-scale commitment. If THAT de-risk is also NEGATIVE, the dendrite is comprehensively ruled out for every current wall and the project should NOT start the months-scale substrate. If it's GO, it localizes the one thing the months-scale build should target (the graded cortex/critic read-out), and the credit-assignment / hidden-layer question becomes a *separate, later* call (which would also require re-posing the nav actor with hidden layers — itself a design change, not just a substrate swap).

---

## 1. VERIFIED D2 STATE — what of the dendritic substrate is ALREADY built (with evidence)

I read the actual code + findings, not the summaries. **Substantially more is built than the dispatch's "Phase 0-2 two-compartment neuron + learned graded cortex" — and crucially, the credit-assignment machinery exists too, AND has been tested.**

### 1a. The built code (file:line verified)

| Asset | What it is | File | Status |
|---|---|---|---|
| `DendriticLayer` | A spiking two-compartment neuron: **basal** (bottom-up forward drive `x@W_basal`), **apical** (top-down through a FIXED-RANDOM `B_apical` — feedback alignment, no weight transport), soma BAC integration (apical depol LOWERS the effective threshold) | `sim/dendritic_neuron.py:20-58` | Built, numpy, biologically-local by construction (no autodiff/reverse-mode) |
| `urbanczik_senn_update` | The LOCAL Urbanczik-Senn somato-dendritic mismatch rule: `Δw ∝ apical_gate · (soma − φ(v_basal)) · pre`; apical-driven target via the fixed-random projection | `sim/dendritic_plasticity.py:17-41` | Built, the literature-faithful local rule |
| `DendriticMLP` | A **DEEP** sigmoid MLP with per-HIDDEN-layer fixed-random feedback `B` + hidden learning delegating to `urbanczik_senn_update`; a fenced hand-derived backprop `oracle` as positive-control / emergent-alignment measurement only | `sim/dendritic_mlp.py:1-58` | Built — this is the **hidden-layer credit-assignment machine** (Lillicrap 2016 feedback alignment; GLR-2017) |
| `enable_dendritic_divisive_gain` | The SHIPPED on-bridge protected edit: a per-presynaptic-source divisive gain `g_i = σ/(σ+a_i)` (suppresses high-rate common sources, passes rare informative ones), at the conductance matvec | `sim/config.py:233`; `sim/bridge.py` (5 guarded sites) | DELIVERED + verified, **byte-identical when off** (18/18 GPU conversational+composition tests pass verbatim, incl. the no-confab moat) |
| `enable_graded_lateral` / `enable_input_mean_adapt` / `enable_input_divisive_norm` | Companion guarded point-neuron normalization primitives (the SM-lateral / per-hub-adapt / Carandini-Heeger divisive norm) | `sim/config.py:386,421,440` | Built, all default-off, byte-identical when off |

**⇒ The D2 "two-compartment neuron" (Phase 0-2) is real AND the credit-assignment scaffold (the deep feedback-alignment MLP + the Urbanczik-Senn rule) is real.** The dispatch under-counted: it is not just a neuron + a cortex, there is a working hidden-layer feedback-alignment learner sitting in `sim/dendritic_mlp.py`.

### 1b. What D2 Phase 1/2/3 actually are (verified against the build plan)

`docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`:
- **Phase 0** (numpy gate, `dendritic_d1p7_spiking_twocompartment_derisk.py`): does the per-compartment advantage survive a genuine spiking soma? **GATE: SURVIVES** (`2026-06-14-dendritic-D1-cheap-derisk-GO.md`).
- **Phase 1** (the protected `sim/` edit): a dendritic compartment on the bridge. **What SHIPPED is the per-source divisive GAIN** (`enable_dendritic_divisive_gain`), not yet a full `NeuronModel.TWO_COMPARTMENT` second-state neuron. The plan's Phase-1 target (a second `v_dend` state + shunting conductance) is the *broader* form; the shipped gain is the narrower realization. Verified DELIVERED + byte-clean.
- **Phase 2** (the learned graded cortex embedding): scale the units into a cortex that learns per-compartment gains from a co-occurrence stream and emits graded-but-reproducible codes. **Status: first realization is an HONEST NEGATIVE** (`2026-06-14-D2-phase1-DONE-phase2-frontier.md`) — and the NEGATIVE is precisely diagnostic (see §1c).
- **Phase 3** (the pending one, task #23): plug the learned graded codes into the validated bind/unbind → attractor cleanup → the full conversational matrix; GATE = generalization-in-conversation (answer about a held-out concept via a *similar* known one) with the moat intact. **NOT STARTED.**

### 1c. The genuine gap — what is NOT built, with the diagnostic evidence

**The credit-assignment mechanism is built (the MLP) but tested-and-NEGATIVE on the substrate's actual task structure:**
- `2026-06-19-dendrite-credit-assignment-toy-stage1.md` (verified in full): a grid-8 actor-critic toy reusing `DendriticLayer` + the BAC gate + `urbanczik_senn_update`, with the place code held FIXED + perfectly selective (so the place-selectivity wall is excluded by construction). Six arms (oracle / point-control / burst-dependent-dendrite / lesion / wrong-sign / urbanczik-senn). **Result: oracle 6/6, point-control 0/6 (a FAIR test — both validity gates pass), dendrite 1/6 (≈ point-neuron level), lesion collapses, wrong-sign anti-learns.** Robust to a learning-rate × budget sweep. **The mechanistic reason (load-bearing): for a SINGLE trainable layer, feedback alignment has NO hidden units to align — the apical burst reduces to a per-action `|δ|`-scaled gain on the same update the point rule already uses.** Consistent with the 2026-05-17 supervised-isolation NEGATIVE.
- `2026-06-19-dendritic-binding-toy-derisk.md` (verified): a learned dendritic sigma-pi/plateau conjunction binder **memorizes** two-attribute bindings (train 0.422) but does **NOT generalize** (held-out 0.168, BELOW the fixed FHRR primitive's 0.261). Lesion collapses the train-fit (the supralinearity is load-bearing for memorization) but generalization doesn't come.

**The graded-read-out gap is built-toward (the divisive gain) but the cortex-scale Phase 2 read-out is the honest NEGATIVE that LOCALIZES the real frontier:**
- `2026-06-14-D2-phase1-DONE-phase2-frontier.md` (verified, incl. its own CORRECTION): the divisive gain is verified + harmless on the bridge, but a clean-readout control **inverts** the early "gain confirmed" — with enough temporal integration the point neuron recovers the cortex-code structure on its own (and the gain hurts). **The deeper, regime-independent limit:** the spiking rheobase threshold *silences the low-count category hubs* under faithful raw-count drive (both gain and point-neuron fail); a baseline lift un-silences them but dilutes the common-mode so the point neuron recovers without the gain. **⇒ neither mechanism robustly produces a strong graded cortex code from faithful raw counts on the spiking bridge** — a *learned* graded embedding that doesn't rate-code raw counts one-shot is the frontier.
- `2026-06-20-burndown-9-critic-graded-readout.md` (verified, TODAY): the nav value-critic read-out is a **genuine FORK, not a clean swap** — the graded-rate (linear) form **physically can't fire the MSN-D1** (sub-rheobase: afferent fires 13 Hz, MSN reaches only −72 mV vs ~−40 mV rheobase, at any weight 0.2→6.0), and the all-or-none coincidence plateau that DOES fire it **over-clamps** (176-219 Hz → GABA_B annihilates both near and far bursts → δ = 0.00). Lesion-confirmed at faithful grid-32 multi-seed. **The residual δ floor is attributed to the deferred DENDRITIC field-carving** — "a dendritic-flavored mechanism is what would carve sparse-separable fields AND grade the plateau into a non-saturating band."

**Net gap:** the dendrite's credit-assignment job is built + tested-NEGATIVE (on a single-layer actor); its graded-read-out job is built-toward + characterized as the genuine open wall + **never run as a standalone dendrite de-risk with the right anti-cheats.** That last sentence is the entire recommendation.

---

## 2. THE UNIFYING-UNLOCK FRAMING — corrected to what the evidence supports

The dispatch's "ONE dendrite unlocks all three" is **half-right and importantly half-wrong**, and getting this exactly right is the point of the gate.

### 2a. Where the framing HOLDS (the genuine common wall)

The three blockers DO share a wall — but it is the **graded read-out of a distributed analog code on a point neuron**, the documented Mikulasch-Priesemann limit (verified: *"prediction errors are computed not in separate units, but locally in dendritic compartments"* — Mikulasch, Rudelt, Wibral & Priesemann, *Trends Neurosci* 2023). This SAME wall surfaces as:
- **Blocker b / #9-critic:** the MSN-D1 can't produce a graded V from linear summation of a distributed place code (sub-rheobase / over-clamp). A graded δ = r − V needs a graded read-out.
- **Blocker c / the learned cortex (D2 Phase 2):** the spiking substrate loses the *weak/diffuse* real category code (host +0.44 → spiking +0.06-0.155, `2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md`) because the whitened structure is a low-magnitude signed differential rate coding can't carry — the same "graded analog read-out of a distributed code" the point neuron can't do.
- **Blocker b / #5 place fields:** a truly sparse+selective place code "would plausibly need per-cell nonlinear input integration" (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`, its own "dendritic flavor" note).

These are ONE family: **the point neuron cannot produce a graded, selective analog read-out of a distributed code by linear summation — that computation is dendritic** (an active, regenerative-but-graded dendritic nonlinearity, the Larkum BAC plateau).

### 2b. Where the framing BREAKS (the controller-flagged overclaim)

The **SURVIVAL POLICY (blocker a)** and the **learned binder (the rest of blocker c)** are NOT this wall — they are **credit assignment**, and credit assignment is a DIFFERENT dendritic job that the project already tested and found NEGATIVE *for the substrate's actual task structure*:
- The literature is unambiguous (both verified): the apical-basal dendrite does credit assignment **in HIERARCHICAL / DEEP architectures** — Payeur-Naud-Richards 2021 (*"pyramidal neurons higher in a hierarchical circuit can coordinate the plasticity of lower-level connections … solve challenging tasks that require deep network architectures"*) and Sacramento-Costa-Bengio-Senn 2018 (a **multilayer** model that approximates backprop). The nav actor is a **single trainable corticostriatal layer** (`cortex_X → str_D1_X`, catalog A.04) — there are no hidden layers for feedback alignment to align, so the dendrite's credit-assignment value structurally does not apply (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`).
- The boundary-ledger + nav-loop gate (both TODAY, verified in full) independently re-classified the nav reward/value/sustained-control loop as **point-neuron LOOP-STABILITY / operating-point** (the actor goes silent because the reentrant `thal→cortex` self-sustain arc is OFF and the host-heuristic drive was removed — `2026-06-20-nav-reward-value-loop-deep-research.md` §1), the SAME family as #4 (which was point-neuron-tunable to 1.16×), **NOT credit-assignment-dendritic.**

**⇒ The corrected unifying framing:** ONE dendritic computation — a **graded, non-saturating, regenerative dendritic read-out of a distributed code** — would unlock the *graded-read-out family* (b/#9-critic, c/learned-cortex, b/#5 selective fields). It would NOT, by itself, unlock the survival policy or the learned binder, because those are credit-assignment / capacity problems that (i) need a hidden-layer architecture the nav actor doesn't have and (ii) tested NEGATIVE. **So the dendrite is the obvious unlocker for ONE of the three claimed blockers (the graded read-out), a plausible-but-untested contributor to a SECOND (the learned cortex generalization), and NOT the unlocker for the THIRD (the survival policy / single-layer credit assignment) on the task as posed.**

### 2c. The single cheapest de-risk that proves the unlock

Because the credit-assignment job is twice-NEGATIVE and the graded-read-out job is the genuine untested wall, **the de-risk to run is the graded read-out**, not the policy. It is also cheaper (no RL trial loop, no place-selectivity entanglement) and more decisive (the point-neuron control provably fails — the burndown-9-critic doc already proved the linear MSN can't fire and the plateau over-clamps, so the two-sided validity gate is pre-satisfied).

---

## 3. REUSABLE (D2 Phase 0-2) vs the minimal NEW piece — the `sim/`-edit scope

### Reusable (high — the de-risk is mostly reuse-by-import)

| Need | Reusable asset | Where |
|---|---|---|
| The two-compartment spiking neuron (basal forward + apical depol-lowers-threshold) | `DendriticLayer` + `_apical_depol` / `effective_threshold` | `sim/dendritic_neuron.py:20-58` |
| The local dendritic plasticity rule (if the read-out gains are learned) | `urbanczik_senn_update` | `sim/dendritic_plasticity.py:17-41` |
| The on-bridge divisive-gain harness + the faithful critic harness + the point-neuron control | `enable_dendritic_divisive_gain`; `dendritic_cortex_forward_codes_derisk.py`; `_burndown9_critic_graded_readout_derisk.py` (the navfaithful harness, lesion + regime-fidelity asserts) | `sim/`, `research/runners/` |
| The guarded protected-edit slot (the template for the new term) | `fused_coincidence_plateau` (`bridge.py:5805-5849`, byte-inert when off) — the SHIPPED protected-edit template the plan names | `sim/kernels.py:253`, `sim/bridge.py` |
| Anti-cheats: point-neuron control, lesion, permuted, host ceiling, regime-fidelity, multi-seed | the D1.x ladder + the burndown-9 harness already implement all of these | `research/runners/dendritic_*` |

### The minimal NEW piece (the genuine `sim/` edit)

**A graded (non-saturating) dendritic-plateau read-out term** — the active dendritic nonlinearity that produces a graded analog value where the linear MSN is sub-rheobase and the all-or-none plateau over-clamps. Concretely: a dendritic compartment current that (i) is regenerative enough to cross the down-state from a distributed sub-rheobase input (so the cell fires at all — what the linear form can't), but (ii) **scales with the weighted input magnitude over a band rather than saturating** (so a near>far weight gradient → a near>far rate → a graded δ — what the all-or-none `gain=2` sigmoid plateau can't, because it clamps at ~plateau). This is a small additive term in the same guarded current-accumulation slot the coincidence plateau already occupies — **owner-OK on a justified additive default-OFF `sim/` edit with byte-level diff review** (the standing rule for protected edits; the existing `enable_dendritic_divisive_gain` is the precedent for byte-identity-when-off).

**Scope:** ~one guarded kernel term + a `cfg.enable_graded_dendritic_plateau` flag + None-guarded allocation, mirroring `enable_dendritic_divisive_gain`'s five-site pattern. NOT a full `NeuronModel.TWO_COMPARTMENT` second-state neuron (that is the broader Phase-1 form and a larger edit) — the cheap-first de-risk needs only the graded-plateau read-out term, validated numpy-first (Phase 0 discipline) before any bridge edit.

---

## 4. THE RECOMMENDED CHEAP-FIRST DE-RISK (the decisive falsification)

**Name it:** *Does a graded dendritic-plateau read-out produce a graded analog value (a graded δ = r − V, OR a graded category read-out) from a distributed sub-rheobase code, where the point-neuron substrate provably cannot (linear = silent; all-or-none = over-clamped)?*

**Stage 0 (CPU/numpy, NO `sim/` edit, ~minutes — the gate before any bridge work):** reuse the burndown-9-critic harness's exact regime (the navfaithful MSN-D1 critic: a distributed place afferent the linear read-out can't fire and the plateau over-clamps). Add ONE arm: the graded dendritic-plateau read-out (a `DendriticLayer`-style compartment with a banded/non-saturating plateau instead of the `gain=2` saturating sigmoid). Three read-out arms on the IDENTICAL afferent + critic + GABA_B subtraction:

| arm | read-out form | role |
|---|---|---|
| `linear` (point-neuron control A) | plain plastic synapse → MSN rate | MUST fail (sub-rheobase, δ ≈ 1.00 — already proven) |
| `plateau` (point-neuron control B) | the all-or-none `fused_coincidence_plateau` (`gain=2`) | MUST fail (over-clamp, δ ≈ 0.00 — already proven) |
| `graded_dendrite` (TEST) | the banded/non-saturating dendritic plateau | the question: does δ grade (near>far, > the 1.3 host-Gaussian bar)? |

**The decisive metric (pre-register, fixed):** the graded δ = far_burst / near_burst (the burndown-9 metric), faithful grid-32, multi-seed 42/43/44. **GO** = graded_dendrite δ ≥ 1.30 (the host-Gaussian ref) at ≥2/3 seeds WHILE both point-neuron controls stay at their failure floors (linear ≈ 1.0, plateau ≈ 0.0). **BOUNDARY** = δ grades but below the host ref (a characterized partial — the brain-based deliverable). **NEGATIVE** = the graded dendrite ALSO fails to grade (then the dendrite is ruled out for the read-out family too, and the months-scale build is NOT warranted — the cheap-first gate did its job).

**Stage 1 (only if Stage 0 GO — the on-bridge protected edit):** wire the graded-plateau term into the bridge (the guarded `cfg.enable_graded_dendritic_plateau` slot, byte-identical when off, byte-level diff review), reproduce the Stage-0 δ on the real spiking substrate at a single-neuron probe, then the full faithful nav A/B. This is the D2 Phase-1 protected edit, scoped to the graded-plateau term only.

**Why this and not the credit-assignment task:** (1) the credit-assignment dendrite is **twice-NEGATIVE** on favorable toys — running it a third time has low expected value and the literature explains why (single-layer actor); (2) the graded read-out has a **pre-satisfied two-sided validity gate** (both point-neuron controls provably fail — already shown), which is the hardest part of any de-risk to establish; (3) it is the **cheapest** (no RL trial loop, no place-selectivity entanglement — a static critic read-out probe); (4) a GO here is **directly load-bearing** — it is exactly the D2 Phase-2 graded-cortex frontier AND the #9-critic δ AND (via the same mechanism) the path to the #5 selective place fields, i.e. it tests the genuine common wall, not the misattributed one.

---

## 5. THE ANTI-CHEAT CONTROLS the de-risk needs

1. **Both point-neuron controls (the two-sided validity gate).** `linear` (sub-rheobase → δ ≈ 1.0) AND `plateau` (over-clamp → δ ≈ 0.0) must BOTH fail on the identical afferent/critic/GABA_B pipeline. *This is already proven in `_burndown9_critic_graded_readout_derisk.json` — so the gate is pre-satisfied, but it MUST be re-asserted in the same run as the test arm (not cited from a prior run).* If the graded dendrite's lift coincides with the linear control suddenly firing, the lift is an operating-point artifact, not the dendrite.
2. **Apical/plateau LESION (the decisive dendrite anti-cheat).** Replace the regenerative dendritic plateau with a passive (identity) compartment — the graded δ MUST collapse to the linear control's floor. *If δ survives the lesion, the grading isn't coming from the dendritic nonlinearity* (the exact confound the 2026-06-19 credit-assignment toy's lesion caught, and the burndown-9 plateau lesion already models — lesion δ 0.72/0.74).
3. **GABA_B subtraction lesion (provenance — the value is neural).** Zero the `striosome_value → snc` GABA_B mask: the δ gap MUST collapse (proving V is subtracted by the conductance, not host arithmetic). Already in the burndown-9 harness.
4. **Regime fidelity (anti-cheat d — the #6 lesson).** Global OU / conductance-noise / homeostasis OFF, faithful grid-32, the dense place afferent — asserted by the navfaithful builder (NOT a permissive smoke). The #6 grid-8 overclaim (CYCLE 310→312, self-corrected) is the cautionary precedent: do NOT read GO off a non-faithful smoke.
5. **Host ceiling / positive control.** The host-Gaussian critic δ ≈ 1.3 is the reference the graded dendrite must reach or beat; the host scaffold remains the upper bound.
6. **No-free-grading control (the over-clamp anti-cheat).** Confirm the graded dendrite's lift is the BAND, not just a weaker plateau: sweep the input magnitude and verify the δ scales with near>far weight gradient (graded), not a single threshold crossing (which would be a re-tuned all-or-none, i.e. the plateau control in disguise).

---

## 6. BIOLOGY GROUNDING (verified against catalog + Kandel + literature)

Every load-bearing biological claim checked against the actual source.

- **The graded dendritic plateau / BAC firing (the mechanism the read-out needs).** Larkum 2013 BAC firing (in `sim/dendritic_neuron.py`'s own docstring + the build plan): apical depolarization + basal coincidence → a regenerative Ca²⁺ plateau that crosses threshold from otherwise sub-rheobase input. **Catalog (to cite by ID in the build):** the dendritic-computation / NMDA-spike / two-compartment cluster — the `sim-catalog/references/feature-catalog.md` dendritic entries the D1.x ladder already drew on (`2026-06-14-dendritic-substrate-deep-research.md` enumerates them). [Catalog file present at `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`; the specific dendritic-computation entry IDs are enumerated in the D1.x deep-research finding and the build plan — cite those IDs in the de-risk runner header.]
- **Dendritic error computation = the point-neuron limit (the WHY this is a dendrite, not a point neuron).** **Mikulasch, Rudelt, Wibral & Priesemann, "Where is the error? Hierarchical predictive coding through dendritic error computation," *Trends Neurosci* 46:45-59, 2023** (verified via search): prediction errors are computed **locally in dendritic compartments**, not in separate units — i.e. the graded analog error/value computation lives in the dendrite, which is exactly why a point neuron (linear summation + single threshold) cannot produce the graded read-out. This is the project's standing "point-neuron limit" citation, and it grounds the graded-read-out family directly. (PubMed 36577388.)
- **Urbanczik-Senn local dendritic plasticity (if the read-out gains are learned).** **Urbanczik & Senn, "Learning by the Dendritic Prediction of Somatic Spiking," *Neuron* 81:521-528, 2014** (verified): a local, non-Hebbian, biologically-plausible rule using the post-synaptic **dendritic voltage** as a third factor; minimizes the dendritic prediction error of somatic spiking; **a unified rule for supervised / unsupervised / reinforcement learning** depending on the somatic input source. This is the already-built `sim/dendritic_plasticity.py` rule. (PubMed 24507189.)
- **Apical-basal credit assignment NEEDS deep/hierarchical architectures (the reason it is NOT the survival-policy unlock).** **Sacramento, Costa, Bengio & Senn, "Dendritic cortical microcircuits approximate the backpropagation algorithm," NeurIPS 2018** (verified): a **multilayer** model where error-driven plasticity adapts the network toward a global desired output, approximating backprop. **Payeur, Naud, Richards et al., "Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits," *Nat Neurosci* 2021** (verified): high-frequency bursts let pyramidal neurons higher in a **hierarchical** circuit coordinate lower-level plasticity, solving tasks requiring **deep network architectures**. **⇒ both require hidden layers — confirming the 2026-06-19 toy NEGATIVE's mechanism (a single-layer nav actor has nothing for feedback alignment to align), so the dendrite's credit-assignment job is structurally inapplicable to the nav actor as posed.** (PubMed 34728832 for the Payeur correction.)
- **The point-neuron CBGT loop sustains reward-driven policy WITHOUT a dendrite (corroborating that blocker a/b's actor-silence is loop-stability, not a substrate wall).** Dunovan-Verstynen-style biologically-constrained spiking cortico-basal-ganglia-thalamic models learn action selection via dopamine-dependent cortico-striatal plasticity and tune the speed-accuracy tradeoff to maximize reward rate (biorxiv 2024.05.21.595174, in `2026-06-20-nav-reward-value-loop-deep-research.md`). Point-neuron CBGT loops ARE a working substrate for the reward loop — the project's failure was its open-loop/perception-stripped config, not the substrate.

---

## 7. HONEST TOP-LINE + PHASED PATH (options for the owner to steer — this is the deep, months-scale call)

### The honest top-line (no spin)

**The dendrite is a real, partially-built tool that has been tested on two of its three claimed jobs and came back NEGATIVE on both — so the dispatch's "one dendrite unlocks all three" is the controller-flagged overclaim, and the corrected truth is sharper and more useful:**
- The dendrite's **credit-assignment** job (the survival policy + the learned binder's generalization) is **ruled out for the substrate's current task structure** — twice cheap-first NEGATIVE, with a literature-confirmed reason (it needs hidden layers the nav actor doesn't have; the binder needs capacity, not a nonlinearity). Re-opening it would require *first* re-posing the task with a deep/hierarchical actor — a design change, not a substrate swap — and is therefore NOT the cheap-first move.
- The dendrite's **graded-read-out** job (the #9-critic δ, the D2 Phase-2 learned cortex, the #5 selective place fields) is the ONE place the project repeatedly hits a *genuine, characterized* dendritic-flavored wall the point neuron provably cannot cross — and it has **never been run as a standalone dendrite de-risk** with the now-built two-compartment neuron + the right anti-cheats. **This is the obvious-unlocker candidate, it is cheap-first, and its two-sided validity gate is pre-satisfied** (both point-neuron controls already proven to fail).

### Phased options for the owner to steer

- **Option A (RECOMMENDED cheap-first, ~hours, decoupled from the months-scale commitment): run the graded-dendritic-plateau read-out de-risk (§4 Stage 0, CPU/numpy, NO `sim/` edit).** Reuse the burndown-9 harness + a `DendriticLayer`-style graded-plateau arm. GO = a graded δ ≥ host where both point-neuron controls provably fail + apical lesion collapses it. **This is the single decisive experiment that confirms-or-kills the dendrite as the graded-read-out unlocker**, before any protected edit or months-scale spend. A NEGATIVE here comprehensively rules the dendrite out for *every* current wall (credit-assignment already NEGATIVE; read-out then NEGATIVE) — itself a valuable, build-saving deliverable.
- **Option B (only if Option A GO, the protected-edit increment): D2 Phase 1 scoped to the graded-plateau term.** Wire the validated graded-plateau read-out onto the bridge (guarded `cfg.enable_graded_dendritic_plateau`, byte-identical when off, byte-level diff review). GATE = the on-bridge single-neuron probe reproduces Stage 0 + the faithful nav A/B δ grades. This retires the #9-critic δ floor on the real substrate — a concrete, bounded brain-based win.
- **Option C (only if A+B GO, the months-scale Phase 2/3): the learned graded cortex + conversational generalization.** Scale the graded-read-out units into the D2 Phase-2 cortex (learn per-compartment gains from the co-occurrence stream → graded reproducible codes that recover the *weak/diffuse* category structure the point neuron lost), then Phase-3 plug into bind/unbind/cleanup → generalization-in-conversation with the moat intact. **This is the genuine months-scale frontier the dispatch points at** — but it is gated on A+B, and it targets the graded-cortex/read-out wall specifically, NOT the survival policy.
- **Option D (SEPARATE, NOT this gate's recommendation): the credit-assignment / survival-policy path.** This would require (i) re-posing the nav actor with a hidden-layer / deep architecture (so feedback alignment has units to align — Sacramento-Senn / Payeur), AND (ii) the burst-dependent-plasticity machinery beyond the current `DendriticMLP`. Given the two NEGATIVE toys, this is a high-variance, design-change-first arc — **defer until/unless Option A-C land AND the owner specifically wants the learned spatial policy** (the boundary ledger's honest note: the nav loop is more likely point-neuron loop-stability — close the reentrant `thal→cortex` arc first, `2026-06-20-nav-reward-value-loop-deep-research.md` §4, the cheaper non-dendrite fix).

**Recommended order:** A (cheap-first, decoupled) → B (protected edit, bounded) → C (months-scale, gated) → D (separate owner call, design-change-first). **The dendrite is worth a cheap-first de-risk (A) NOW — but as the graded-read-out unlocker, not the credit-assignment one; the months-scale build should not start until A confirms the read-out unlock.**

---

## Citations (verified against actual source text)

**Project code (file:line):**
- `sim/dendritic_neuron.py:20-58` (`DendriticLayer` two-compartment BAC neuron); `sim/dendritic_plasticity.py:17-41` (`urbanczik_senn_update`); `sim/dendritic_mlp.py:1-58` (deep feedback-alignment MLP — the hidden-layer credit-assignment machine).
- `sim/config.py:233` (`enable_dendritic_divisive_gain`), `:386` (`enable_graded_lateral`), `:421` (`enable_input_mean_adapt`), `:440` (`enable_input_divisive_norm`) — all default-off, byte-identical when off.
- `sim/kernels.py:253` + `sim/bridge.py:5805-5849` (`fused_coincidence_plateau` — the guarded protected-edit template + the all-or-none read-out the de-risk replaces).

**Project findings (read in full):**
- `2026-06-14-D2-phase1-DONE-phase2-frontier.md` (Phase 1 delivered byte-clean; Phase 2 gain NOT load-bearing on the spiking substrate — the clean-readout control inversion; the graded-cortex frontier localized).
- `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` (Phase 0/1/2/3 definitions; Phase 3 = the pending conversational-generalization gate).
- `2026-06-19-dendrite-credit-assignment-toy-stage1.md` (apical-basal credit assignment NEGATIVE; single-layer actor "nothing to align"); `2026-06-19-dendritic-binding-toy-derisk.md` (learned dendritic binding memorizes-but-doesn't-generalize, below the FHRR primitive).
- `2026-06-20-burndown-9-critic-graded-readout.md` (the graded-rate critic read-out is a FORK: linear sub-rheobase / plateau over-clamps; the δ floor attributed to the deferred dendritic field-carving — the genuine graded-read-out wall, TODAY).
- `2026-06-20-boundary-ledger-dendritic-audit.md` (the dendritic-debt audit: 0 dendritic boundaries block a shipped capability; the two dendrite jobs tested NEGATIVE; the one deferred candidate is the spiking-from-real cortex, TODAY).
- `2026-06-20-nav-reward-value-loop-deep-research.md` (the nav loop is point-neuron LOOP-STABILITY, NOT dendritic; close the reentrant `thal→cortex` arc, TODAY); `2026-06-19-fsg-watermaze-trial-structured-derisk.md` (the 3rd water-maze NEGATIVE; "resolves toward the dendrite" as a research-direction call, entangled with #5 place-selectivity); `2026-06-15-phaseB-spiking-cortex-WALL-rate-to-spike.md` (the spiking-from-real cortex loses the weak/diffuse code, +0.06 vs host +0.44).
- `AUTONOMOUS_STATE.md` CYCLE 321/322 (TODAY): the spiking living-loop SPLIT — DRIVE converts (corr 0.995, GO), SURVIVAL POLICY = the dendrite wall; the "one animal" cross-modal GO.

**Catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`):** the dendritic-computation / NMDA-spike / two-compartment cluster (entry IDs enumerated in `2026-06-14-dendritic-substrate-deep-research.md` + the D2 build plan — cite those IDs in the de-risk runner header); A.04 (single-layer corticostriatal actor), A.05 (reentrant `thal→cortex` loop — the non-dendrite nav fix), C.30 (actor-critic), B.07 (striosome→SNc value subtraction).

**Literature (verified via WebSearch):**
- Mikulasch, Rudelt, Wibral & Priesemann, *Where is the error? Hierarchical predictive coding through dendritic error computation*, Trends Neurosci 46:45-59, 2023 (PubMed 36577388) — dendritic error/value computation = the point-neuron limit.
- Urbanczik & Senn, *Learning by the Dendritic Prediction of Somatic Spiking*, Neuron 81:521-528, 2014 (PubMed 24507189) — the local dendritic-voltage third-factor rule (the built `dendritic_plasticity.py`).
- Sacramento, Costa, Bengio & Senn, *Dendritic cortical microcircuits approximate the backpropagation algorithm*, NeurIPS 2018 (arXiv 1810.11393) — credit assignment needs a MULTILAYER model.
- Payeur, Naud, Richards et al., *Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits*, Nat Neurosci 2021 (PubMed 34728832) — burst-dependent credit assignment needs HIERARCHICAL / DEEP architectures.
- Larkum 2013 (BAC firing) — the regenerative apical plateau, the mechanism the graded read-out needs.
- Dunovan-Verstynen, biorxiv 2024.05.21.595174 — point-neuron CBGT loops sustain reward-driven policy without a dendrite (corroborating the nav-loop loop-stability classification).

_Read-only deep-research deliverable. NO code, NO experiments. Load-bearing claims verified against the actual code/finding/catalog/literature text; toy-scale / single-seed / regime-bounded flagged where that is the truth. The dispatch's "one dendrite unlocks all three" is corrected to: the dendrite is the obvious unlocker for the GRADED-READ-OUT family (one of the three), tested-NEGATIVE for the credit-assignment family (the survival policy + the learned binder), and the recommended cheap-first de-risk is the graded-dendritic-plateau read-out, not the credit-assignment task._
