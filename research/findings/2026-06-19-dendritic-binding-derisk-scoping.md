# Dendritic binding de-risk — scoping (the genuine dendrite shot: learnable two-attribute COMPOSITION)

**Date:** 2026-06-19. **Type:** READ-ONLY deep-research + scoping (no build, no runs). **Decision this scopes
(owner-approved 2026-06-19):** test the dendrite's *other* candidate job — **learnable multi-attribute binding via
dendritic coincidence/multiplication** (candidate (a), "the binding wall") — which is UNTESTED and is a DIFFERENT
mechanism from the apical-basal credit-assignment that just landed NEGATIVE (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`).

**One-line answer (the recommendation):** the existing two-compartment dendrite is the WRONG primitive for this (it is
an additive threshold-shift / spatial-decorrelation machine — verified by reading the code), so the de-risk needs a
**small new dendritic-multiplication primitive** (a sigma-pi / NMDA-plateau supralinear conjunction). The cheapest-first
test is a numpy/CPU extension of the **already-built** `_phaseB_multiplicative_bind_bundled_derisk.py` harness that
swaps the broken *learned-linear* unbind for a **dendritic supralinear/plateau conjunctive** bind+unbind and asks the
one decisive question: does a *learned dendritic-multiplication* binder recover **two-attribute bundling that
GENERALIZES to held-out attribute pairs**, where the point-neuron / learned-linear baseline hits the documented K=5
wall? No `sim/` edit for the cheap gate (the dendritic op lives in the de-risk runner, numpy); a protected `sim/` edit
is deferred to the *build*, gated behind this numpy GO.

---

## 1. The mechanism map — how dendritic multiplication makes a LEARNABLE two-attribute bind

### 1.1 The wall, verified (not re-derived)
- **Production state (read `CLAUDE.md`, `rf_phasor_composer.py:317–333`).** The conversational composer binds
  role-filler and attribute structure by a **fixed multiplicative primitive** (the ±1 self-inverse / FHRR phasor
  conjugate). **SINGLE-attribute binding WORKS** (production). **TWO-attribute binding is the documented K=5-load
  BOUNDARY** — `rf_phasor_composer.py:327` literally tags `attribute2` as "the +-1 scheme's K=5 boundary — does FHRR
  lift it?". The F=3 two-attribute resonator is a **numpy REFERENCE** retained only as the FHRR ceiling
  (`spiking_phasor_fhrr.py:7`).
- **The prior boundary this attacks (read `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`).**
  The decisive contrast already on file:

  | bind | single-attribute | 3-way bundle (a fact) |
  |---|---|---|
  | **fixed ±1 / FHRR algebra** (self-inverse role) | 1.000 | **0.989** |
  | **learned additive** (point-neuron) | 0.806 → real-LIF 0.833 | **0.193** |
  | **learned multiplicative + learned LINEAR inverse** | 0.083 (broken) | 0.056 (broken) |
  | chance (1/F) | 0.062 | 0.062 |

  The localization is exact and load-bearing: **unbinding role *t* from a superposition requires applying the
  role-specific inverse to the bundle (`bundle / u_t`) — a MULTIPLICATION by a role-dependent factor `1/u_t`.** A
  shared *linear* unbind is structurally incapable of a role-dependent scaling (it breaks even single-attribute,
  0.083); an *additive* bind has no inverse at all (0.193). So the production keeps the **fixed** self-inverse
  primitive, and the open frontier is a **LEARNED** multiplicative bind that bundles AND generalizes. **Multiplication
  is a dendritic operation** — the same point-neuron limit (Mikulasch-Priesemann) the project repeatedly meets.

### 1.2 The dendritic mechanism (deep-research, controller should verify the load-bearing claims)
The linchpin paper is **Singh & Schultheiss / Bicknell-style "Local, calcium- and reward-based synaptic learning rule
that enhances dendritic nonlinearities can solve the nonlinear feature binding problem"** (eLife reviewed-preprint
97274, 2024). It gives both halves — the *nonlinearity* and the *learning rule* — and is the closest published analogue
to exactly this de-risk:

**(i) The multiplicative computation = NMDA-plateau supralinear branch conjunction (sigma-pi).**
- Catalog **G.02 Active dendrites** (`feature-catalog.md:2644–2652`): dendrites have voltage-gated Na⁺/Ca²⁺/HCN →
  **NMDA spikes (plateau potentials), supralinear branch summation** — "cluster of inputs on one branch ≫ scattered
  inputs on many branches." Catalog **J.08 NMDA receptor** (`:3593–3601`): ligand+voltage gated, Mg²⁺ unblock above
  ~−40 mV → a **coincidence detector**; this is the existing `fused_nmda_update_and_current` kernel (so the substrate
  already has the *element*, just not as a per-branch product unit). Catalog **G.03 feature binding** (`:2658–2668`):
  "missing — no binding mechanism" → this de-risk is the first attempt at a substrate binding op.
- The abstraction is **Mel's Clusteron / sigma-pi neuron**: dendrites supply **low-order PRODUCT terms between groups
  of synapses** that are summed at the soma; the branch responds supralinearly to *co-activated clusters*. **Two
  factors → two clustered input groups on one branch; the plateau nonlinearity computes their conjunction
  ≈ a product A·B** (a soft AND), where a linear sum cannot. This is the formal bridge VSA work names directly:
  binding requires multiplication/coincidence, "computations which can be implemented by active dendritic mechanisms …
  such as sigma-pi neurons" (Kleyko/Frady "Variable Binding for Sparse Distributed Representations", arXiv 2009.06734).
- **Concretely for the composer's A⊗B:** put attribute-A's code as one synaptic cluster and attribute-B's code as a
  second cluster on the *same* dendritic branch; the branch's supralinear (NMDA-plateau / sigmoidal) transfer emits a
  strong output only when *both* clusters co-activate → the per-branch output approximates the **element-wise product
  A⊙B** (the bind). Unbinding is the same op against the conjugate/inverse factor (the dendritic conjunction with a
  role-specific cluster recovers the matching filler; the other facts fall below plateau threshold → noise).

**(ii) The LEARNING rule (what shapes the bind — the part the fixed ±1 primitive does NOT have).**
The eLife rule is a **LOCAL, three-factor, calcium-and-dopamine** rule (NOT backprop, NOT weight transport):
- **LTP** when NMDA-calcium sits in a bell-shaped window AND dopamine peaks: `Δw ∝ η₁·σ'([Ca]_NMDA − θ_LTP)`.
- **LTD** above an L-type-calcium threshold AND dopamine pause: `Δw ∝ −η₂·([Ca]_Ltype − θ_LTD)`.
- **Metaplasticity** (the kernel midpoint shifts with dopamine pauses/peaks) stabilizes learned synapses.
- **Inhibitory plasticity** (BCM-like) does the dendritic **compartmentalization** — it sculpts *which* features land
  on *which* branch, so the relevant conjunctions cluster together. This is the load-bearing learning step: the rule
  *learns to route* co-relevant attributes onto a shared branch so the plateau binds them. **That routing/clustering
  is exactly the "learned bind" the point-neuron substrate cannot do** — a point neuron has one summation node and no
  branch to compartmentalize onto.
- **Result they report:** the rule solves the **nonlinear feature binding problem** (respond to red-strawberry &
  yellow-banana, stay silent for the cross pairs — a linearly *non-separable* discrimination where all four
  combinations carry identical total synaptic input) at ~90–100% with clustered synapses, while a **linear/point
  summation provably cannot** (the four pairs are somatically identical).

**The synthesis (1–2 sentences):** put each attribute's code as a separate synaptic cluster on a shared dendritic
branch; the branch's **NMDA-plateau supralinear nonlinearity computes their conjunction A⊙B (a sigma-pi product) — the
multiplicative bind a point neuron's single linear summation cannot** — and a **local calcium/dopamine three-factor
rule (with inhibitory compartmentalization) LEARNS which co-relevant attributes to route onto a shared branch**, which
is the learnable part the fixed ±1 primitive lacks.

---

## 2. Why the navigation-NEGATIVE reason does NOT apply here (stated explicitly)

The credit-assignment NEGATIVE (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`) failed for a **structural,
single-layer** reason: *"For a single trainable layer (the actor) there are no hidden units to assign credit to, so
feedback alignment has nothing to align — the apical compartment reduces to a per-action |δ|-scaled non-negative GAIN
on the same update direction the point rule already uses."* That is a property of **apical-basal credit assignment**
(candidate (b)) needing a hidden layer to route error to.

**Binding is a DIFFERENT mechanism and the single-layer reason is irrelevant.** The bind is a **single-OP feedforward
multiplicative computation** — `A⊙B` evaluated in one dendritic step. There is no hidden-layer credit to route and no
feedback-alignment ask; the supralinear/plateau conjunction is the **dendrite's native analog forward op** (catalog
G.02/J.08). The learning (eLife rule) shapes *which inputs cluster on which branch*, locally, with no error-transport.
So "single layer, nothing to align" cannot fire here — the failure mode of candidate (b) does not exist for candidate
(a). (This is precisely the distinction the owner drew and the `feedback_dendritic_substrate_fair_game` memory records:
"learnable multi-attribute COMPOSITION via dendritic coincidence/multiplication (the binding wall)" vs "apical-basal
credit assignment (the learning-rule gap)" — two different jobs.)

---

## 3. Right-dendrite check — existing primitive is WRONG; a SMALL new one is needed

**Verdict: the existing `sim/dendritic_neuron.py` is the WRONG dendrite — it has NO multiplicative branch nonlinearity.
A small new sigma-pi / plateau conjunction primitive is required.** Verified by reading both modules:

- **`sim/dendritic_neuron.py` `DendriticLayer.step` (line 49–58):**
  ```
  v_basal   = leak*v_basal + x_basal @ W_basal        # additive basal sum
  v_apical  = teacher @ B_apical                       # additive apical sum (fixed-random feedback)
  theta_eff = theta_high - apical_gain*|v_apical|      # apical only SHIFTS the threshold
  soma_rate = sigmoid(v_basal - theta_eff)             # == sigmoid(basal_sum + gain*|apical_sum|)
  ```
  This is **additive throughout** — the apical compartment is a **threshold-shift / gain modulation** of the basal
  sum (Larkum BAC / Guerguiev-Lillicrap-Richards). There is **no `A·B` product term**; two input vectors are never
  multiplied. It is, as `feedback_dendritic_substrate_fair_game` already notes, a **SPATIAL decorrelation** machine —
  the right dendrite for the *Phase-2 graded-cortex normalization* (the separate D2 arc), the WRONG dendrite for
  binding.
- **`sim/dendritic_plasticity.py` `urbanczik_senn_update` (line 17–41):** a **somato-dendritic mismatch** rule
  (`Δw ∝ gate·(soma − φ(v_basal))·pre`). Additive-mismatch; it does NOT learn a multiplicative conjunction. (This is
  also the rule that landed NEGATIVE on credit assignment.)

**What the new primitive must add (keep it SMALL + additive + default-OFF):** a per-branch **supralinear conjunction
node** — given two input clusters `a, b`, emit `g_plateau(a, b)` where `g` is an all-or-none / sigmoidal-on-the-product
transfer (e.g. `relu(a·b − θ)` or `σ(κ·(a·b − θ))`, the sigma-pi product), approximating the NMDA-plateau. **For the
cheapest gate this lives entirely in the de-risk runner (numpy) — NO `sim/` edit at all.** If the gate is GO, the
*build* adds a guarded `NeuronModel.TWO_COMPARTMENT_SIGMAPI` (or a guarded `fused_dendritic_plateau` sub-threshold
term, the `fused_coincidence_plateau` template at `bridge.py:5805–5849` is the exact precedent), **default OFF =
bridge byte-identical**, byte-level diff review per the owner's standing rule. **The cheap gate must not touch `sim/`.**

**Relationship to D2 task #23 (`docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`).** D2 Phase 0–2 built a
two-compartment neuron + learned graded cortex for **per-input divisive normalization** (generalizing *codes*); task
#23 (Phase 3, pending) plugs those codes into the conversational pipeline. **That dendrite (decorrelation) and this
dendrite (multiplication) are complementary, not the same.** This de-risk attacks the **binding OPERATION** (the
exact-inverse algebra idealization); D2 attacks the **representation** (the codes). A full step-3 cortex plausibly
wants both: D2's normalized codes flowing through a sigma-pi *learned* bind. This scoping is about the bind only.

---

## 4. The cheapest-first de-risk — exact config, GO bar, controls

**The smallest test that decides the question.** Reuse the **already-built** harness
`research/runners/_phaseB_multiplicative_bind_bundled_derisk.py` (it already has the systematicity protocol, the
memorization-floor, the chance line, the bundled-held-out metric, and the GO/PARTIAL/NEGATIVE bars). That de-risk
tested a learned-multiplicative bind with a **learned LINEAR unbind/cleanup** (`est = act @ W_O`) and is expected
NEGATIVE (its own header predicts it). **The new de-risk changes exactly ONE thing: replace the learned-linear unbind
+ cleanup with a learned DENDRITIC SUPRALINEAR (sigma-pi / plateau) conjunction**, and re-ask whether bundling now
generalizes.

### 4.1 Mechanism under test (numpy, in the runner — NO `sim/`)
- **Bind:** `g_r = plateau(role_r ⊙ filler_r)` where `plateau(z) = σ(κ·(z − θ))` (or `relu(z − θ)`) applied
  element-wise to the product of the two clustered factors — the sigma-pi/NMDA-plateau conjunction.
  `bundle = Σ_r g_r` (superposition, as now).
- **Unbind role t:** the dendritic conjugate conjunction — `act = plateau(bundle ⊙ inv(role_t))` — then read the
  nearest filler (`native_argmax` against the codebook). **The cleanup stays the matched-filter argmax already in the
  harness** (do NOT introduce a learned-linear cleanup — that is the broken element this de-risk is removing).
- **Learning:** train the per-branch routing/weights with a **local, three-factor, calcium/reward-shaped** update
  faithful to the eLife rule (a bell-shaped-window LTP + thresholded LTD on the plateau-Ca surrogate, with the
  inhibitory/BCM compartmentalization that routes co-relevant attributes onto a shared branch). Keep it **local — NO
  backprop through the products** (the prior de-risk used Adam-backprop; the *brain-based* claim wants the local rule,
  and a backprop control may be run alongside as a non-biological ceiling only).

### 4.2 Config (CPU/numpy first, multi-seed)
- Reuse: `make_role_codes`, `make_systematicity_splits`, `native_argmax` from
  `cortex_learned_binder_systematicity_probe.py`; the cached **320 stream codes**
  (`research/findings/raw/_phaseB_stream_codes_320_seed42.npy`, **confirmed present**) normalized as now.
- `R=4` roles, `F=16` fillers, `N_SPLITS=3` leakage-free splits, `D_h=64` (match the harness; sweep `D_h` only if
  PARTIAL). **The two-attribute frontier:** evaluate **bundled SVO + ATTRIBUTE (K≥2 bound attributes)** — the exact
  load `rf_phasor_composer.py` flags at K=5 — split by train-combo vs **held-out-combo**.
- `SIM_BACKEND=numpy`, 3 seeds (42/43/44) for the cheap gate; **escalate to 6 seeds (42/43/44/100/101/102) before any
  GO claim** (the project's standing 6-seed rule). GPU only if a scaled confirmation is wanted after the numpy GO.

### 4.3 GO bar (pre-registered, not tuned to result)
A **GO** requires ALL of:
1. **Two-attribute bundled held-out-combo recall ≫ the point/linear wall** — concretely the harness's own bar:
   `bundle_held ≥ 0.40` AND `bundle_held ≥ 0.6·bundle_train`, **and well above** the additive NEGATIVE (0.193) and
   chance (1/F = 0.062). (For the *attribute* load specifically, "above the K=5 boundary" = the fixed-±1 / point
   baseline measurably failing at the same K while the dendritic binder holds.)
2. **It GENERALIZES, not memorizes** — `bundle_held ≈ bundle_train` (small systematicity GAP); a memorization-floor
   lookup scores chance (the control below). This is the make-or-break clause.
3. **Single-attribute still works** — `single_held ≥ 0.40` (the dendritic bind must not break what the additive bind
   already had at 0.806; if single-attribute regresses, the op is wrong).
4. **The baselines fail on the identical pipeline** — the point-neuron / learned-linear unbind (the existing de-risk's
   own arms: additive 0.193, learned-linear 0.056) stay at/near chance on the same data + same splits.
5. **Multi-seed** (≥6 seeds) — the GO holds, not a lucky seed.

### 4.4 Controls (load-bearing anti-cheats — all must behave)
- **Memorization-floor** (already in the harness as `MemorizationBinder`): a lookup table that can only return
  training-seen combos → **MUST score chance** on held-out. If the dendritic binder beats this AND generalizes, the
  lift is real recombination, not table lookup.
- **Permuted control:** shuffle the role↔filler (or attribute-pair) assignment → the learned routing should collapse
  (no spurious structure). Mirrors the project's permuted-label discipline.
- **Lesion:** ablate the plateau nonlinearity (set `plateau → identity`, i.e. fall back to the linear product/sum) →
  bundling collapses to the additive/linear NEGATIVE. This proves the **supralinearity** (not just calcium + a
  learning rule) is what binds — the eLife paper's own decisive control (distributed-synapses-without-clustering
  solved only 16%).
- **Chance line** (1/F = 0.062) printed alongside, as now.
- **The FHRR F=3 resonator as the reference CEILING** (`spiking_phasor_fhrr.py`): the fixed-algebra two-attribute
  resonator is the upper bound the *learned* dendritic binder is measured against (the learned bind earns its keep if
  it approaches the fixed ceiling while being learned + generalizing).
- **The no-confab moat as anti-cheat at the agent level** (deferred to the build, not the cheap gate): if/when wired
  into `rf_phasor_composer` / `one_brain_composer`, the abstention (`is None` on unstored cues) must stay intact (zero
  false-accepts) — never weakened to manufacture a GO.

---

## 5. Honest risk + stop criterion (be my own skeptic)

### 5.1 The biggest ways this misleads
1. **Memorization masquerading as binding (the primary risk).** A learned binder with enough capacity can *memorize*
   the training attribute pairs and score high on a test set that secretly overlaps. **Mitigation is non-negotiable:**
   the leakage-free systematicity splits (assert empty train∩test), the memorization-floor control scoring chance, and
   the held-out≈train GAP clause in the GO bar. **The eLife paper itself did NOT test held-out generalization** (it
   trains continuously on a fixed 4-feature discrimination, no test set) — so the published result does NOT by itself
   answer the composer's question, and *our* de-risk's whole value is adding the held-out test the paper lacks. Treat
   any "it binds" number without the GAP clause as uninformative.
2. **Solving the wrong problem (the NFBP ≠ the VSA bind).** The eLife "nonlinear feature binding problem" is an
   XOR-like **discrimination** (fire for relevant pairs, stay silent for cross pairs). The composer needs an
   **invertible bind-and-unbind that recovers the filler from a superposition** — a stronger requirement. A dendritic
   op could pass NFBP-discrimination yet fail invertible unbind-from-bundle. **Mitigation:** the de-risk metric is
   *recall of the unbound attribute from the K≥2 bundle* (not a fire/silent discrimination), so it tests the actual
   composer requirement.
3. **The single-attribute distractor.** The fixed ±1 primitive already does single-attribute (production); a sloppy
   test could "succeed" by riding single-attribute and never isolate the two-attribute frontier. **Mitigation:** the
   GO bar gates on **two-attribute / K≥2 bundled held-out** specifically, with single-attribute reported separately as
   a must-not-regress floor.
4. **Plateau threshold tuning = a hidden cheat.** If the plateau `θ/κ` are hand-tuned per seed to the answer, the
   "learning" is smuggled into hyperparameters. **Mitigation:** fix `θ/κ` a-priori (one value, all seeds) OR let the
   local rule adapt them (metaplasticity), and report; never per-seed-tune to a target.
5. **Local-rule vs backprop confound.** If only a backprop-trained version works, the "brain-based" claim is unmet.
   **Mitigation:** the headline arm is the **local calcium/reward rule**; a backprop arm may run only as a
   non-biological ceiling, clearly labelled.

### 5.2 The clear decision
- **GO** (learned dendritic two-attribute bind clears the bar in §4.3, generalizes to held-out pairs, baselines fail,
  multi-seed): **the dendrite earns its keep on the conversational binding wall.** Recommend the fuller build — the
  small protected `sim/` sigma-pi/plateau primitive (default-OFF, byte-reviewed) + wiring into the composer's
  attribute path, gated phase-by-phase like the D2 plan. This would be the first *learned* lift of the K=5 two-attribute
  boundary and a genuine biology-translatable composition mechanism.
- **PARTIAL** (multiplication helps — `0.25 ≤ bundle_held < 0.40` — but isn't decisive): localize per the harness's
  own branch (add an iterative **resonator** cleanup in the unbind loop, or more branch capacity) before committing to
  the `sim/` build.
- **NEGATIVE** (even a learned dendritic multiplication does not lift two-attribute bundling above the wall, or it
  binds but does NOT generalize): **honest finding — the binding wall is NOT (only) the missing dendritic
  multiplication.** Stop; do not escalate to the protected `sim/` edit or the months-scale build (the discipline from
  the credit-assignment NEGATIVE: a cheap NEGATIVE on the favorable numpy toy is the terminus, it SAVES the spend).
  The production keeps the fixed ±1 / FHRR primitive (which already bundles 0.989) and the open question moves
  elsewhere (e.g. a fixed self-inverse role + a learned filler cortex — which, per the prior finding, **is already the
  production composer binding learned codes**).

**Stop criterion (pre-registered):** run the numpy 3-seed gate first; if it does not clear §4.3 clause-1+clause-2 at
3 seeds, it is NEGATIVE/PARTIAL — do **not** spend on GPU, the `sim/` primitive, or 6-seed escalation. Only a 3-seed
indicator that clears earns the 6-seed confirmation, and only a 6-seed GO earns the build.

---

## Appendix — files read / evidence

- **The wall + composer:** `CLAUDE.md` (K=5 two-attribute boundary; FHRR F=3 reference);
  `research/runners/rf_phasor_composer.py` (`store`/`_encode`/`_bind`/`_unbind_phases`; `attribute2` K=5 tag at :327);
  `research/runners/one_brain_composer.py` (sibling API); `research/runners/spiking_phasor_fhrr.py:7` (F=3 reference
  header).
- **The prior boundary:** `research/findings/2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`
  (the capability map; additive 0.193 / learned-linear 0.056 / fixed-±1 0.989 / chance 0.062).
- **The credit-assignment NEGATIVE (why (b)≠(a)):** `research/findings/2026-06-19-dendrite-credit-assignment-toy-stage1.md`.
- **The existing harness to reuse:** `research/runners/_phaseB_multiplicative_bind_bundled_derisk.py` (systematicity +
  memorization-floor + chance + GO/PARTIAL/NEGATIVE bars; its own header predicts the linear NEGATIVE and points to
  "the two-compartment dendritic substrate"); `research/runners/cortex_learned_binder_systematicity_probe.py`
  (`make_role_codes`/`make_systematicity_splits`/`native_argmax`/`MemorizationBinder`, leakage-free protocol);
  codes present at `research/findings/raw/_phaseB_stream_codes_320_seed42.npy`.
- **The dendrite infra (WRONG primitive, verified):** `sim/dendritic_neuron.py` (`DendriticLayer.step` :49–58 —
  additive threshold-shift, no product term); `sim/dendritic_plasticity.py` (`urbanczik_senn_update` :17–41 —
  additive mismatch). Protected-edit template: `bridge.py:5805–5849` `fused_coincidence_plateau`.
- **Catalog:** `sim-catalog/references/feature-catalog.md` G.02 active dendrites (:2644–2652, NMDA plateau +
  supralinear branch summation), J.08 NMDA coincidence detector (:3593–3601), G.03 feature binding (:2658–2668,
  "missing — no binding mechanism").
- **The D2 relationship:** `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` (Phase 0–2 done = the
  decorrelation dendrite for codes; task #23 Phase 3 pending); `feedback_dendritic_substrate_fair_game` memory ((a)
  multiplication-binding = "the genuine dendrite unlocker"; D2 neuron = WRONG dendrite, a spatial decorrelation
  machine).
- **Literature (deep-research; controller verify load-bearing claims):**
  - eLife reviewed-preprint **97274** — "Local, calcium- and reward-based synaptic learning rule that enhances
    dendritic nonlinearities can solve the nonlinear feature binding problem" (the linchpin: NMDA-plateau conjunction
    + local Ca/dopamine three-factor rule + inhibitory compartmentalization; ~90–100% on NFBP; **no held-out test —
    the gap our de-risk fills**).
  - **Mel, "The Clusteron"** + sigma-pi neuron (dendritic low-order product terms between synapse groups).
  - **Legenstein & Maass 2011** (branch-specific dendritic-spike learning binds features in different branches).
  - **Kleyko/Frady, "Variable Binding for Sparse Distributed Representations"** (arXiv 2009.06734) — binding =
    multiplication/coincidence, implementable by sigma-pi/active dendrites (the VSA↔dendrite bridge).
  - Larkum 2013 (BAC firing); Mikulasch-Priesemann (the point-neuron analog/dendritic limit, the project's recurring
    wall).
