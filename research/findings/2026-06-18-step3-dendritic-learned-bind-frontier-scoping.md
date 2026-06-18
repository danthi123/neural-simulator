# Step-3 / dendritic learned-bind frontier — synthesis + the single cheapest-first next build

**Status:** READ-ONLY deep-research SYNTHESIS (the standing "deep research + catalog review FIRST at a new
direction", CLAUDE.md). NO `sim/` edits, NO build, NO GPU, NO git commit. Single deliverable = this doc.
**Date:** 2026-06-18. **Author role:** read-only synthesis subagent. The frontier is heavily pre-researched;
this doc SYNTHESIZES the established record and pins the one cheapest-first concrete next build — it does NOT
re-derive known results. Every load-bearing claim below is file/finding/line-cited and re-verified against the
repo this pass; the surprising ones were read in full.

---

## 0. The one-paragraph answer (the rest is the evidence)

"Step 3" = replace the conversational composer's exact-inverse FHRR (Fourier Holographic Reduced Representation)
binding algebra with a LEARNED spiking-cortical binder. The project has now de-risked this to ground, and the
record SPLITS the original single "step-3 cortex" goal into **two questions with opposite resolutions**: **(1) the
LEARNED-CODE / generalizing-cortex question is CLOSED-POSITIVE on point neurons** — the PPMI local-normalization
stream cortex (learned from the conversation stream, on the real spiking substrate, 320 concepts, multi-seed, moat
intact) generalizes and ships; and the dendritic *decorrelation* rationale that motivated D2 was **falsified as
unnecessary** (the off-diagonal residual a dendrite was hypothesized to buy does not exist as a reachable gap —
local PPMI-centering already reaches the whitening ceiling; `2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md`,
3 seeds). **(2) The LEARNED-BIND question is SETTLED-WITH-ONE-OPEN-CORNER:** a learned role-filler bind generalizes
SINGLE-attribute bindings and is validated on real LIF spikes (on-bridge held-out 0.833 = 100% of the numpy
reference), but **multi-attribute BUNDLING (a fact = a superposition of bindings) is NOT learnable from scratch on
the point-neuron substrate** (additive has no inverse 0.193; a learned LINEAR inverse cannot be a reciprocal 0.056),
while a **FIXED ±1 self-inverse bind bundles 0.989** on the same harness — so the current production answer is
"learned representations through a FIXED biology-grounded coincidence primitive." **The single deepest unresolved
question, and the only place the existing D2 dendritic substrate might unlock something point neurons provably
cannot, is OP-B — dendritic MULTIPLICATION as the superposition-inverse for bundling** (the bundling-NEGATIVE
itself names "multiplication is a dendritic operation"). Crucially, the bundling NEGATIVE's "multiplicative" arm was
CONFOUNDED — it used a *learned linear* inverse, not a genuine product with a *fixed self-inverse* role — and the
dendritic-multiplication primitive (`fused_coincidence_plateau`) is **already on the bridge, guarded, but never
wired as a binding primitive**. **⇒ The cheapest-first decisive next build is an afternoon CPU/numpy A/B that
re-runs the EXACT 2026-06-16 bundling NEGATIVE harness with ONLY the binding op swapped to a fixed-self-inverse
role + LEARNED filler codes (the untested middle), against the established additive/learned-linear NEGATIVE and the
fixed-±1 positive control.** This A/B was fully specified 2026-06-17 (`2026-06-17-dendritic-multiplicative-binding-scoping.md`)
and is **confirmed still un-run** (no `2026-06-18` result doc, no new self-inverse-bind runner — verified). It is
the project's own un-executed recommendation, and either outcome is a deliverable.

---

## 1. DIAGNOSIS — the established state, synthesized faithfully (do NOT contradict these)

### 1.1 The fork (2026-06-11) and what it became
The step-3 cortex was framed (`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`,
`docs/plans/2026-06-14-step3-cortex-fork-resolved-dendritic-D2-decision.md`) as:
- **(A) a semantically-FLAT cortex** — generated decorrelated codes + the validated binder + cleanup + the no-confab
  gate. Achievable on point neurons; cannot generalize across similar concepts. **DELIVERED** (the 2,048-concept
  curated cortex, `2026-06-14-phase1-production-32bridge-2048-concept-cortex-DELIVERED.md`).
- **(B) a semantically-STRUCTURED / generalizing cortex** — learns the similarity from experience so similar concepts
  get similar codes. *Suspected* (2026-06-11/14) to require the months-scale dendritic rewrite, because **five
  mechanistically-distinct point-neuron NEGATIVEs** (vanilla Hopfield common-mode collapse; Storkey locality wall;
  spiking dentate-gyrus sub-reproducible read; fixed-random/Marr-Albus expansion; Option-C learn-from-text
  `Pearson(S_learned,S_true)=−0.008` vs host PPMI+SVD `+0.532`) all converged on the **Mikulasch-Priesemann
  point-neuron limit** (decorrelation/whitening is an analog, pre-spike, DENDRITIC computation a single-soma point
  neuron structurally cannot do; PNAS 2021; catalog G.02 "Active dendrites — MISSING"). On that basis the owner
  approved the D2 dendritic build.

### 1.2 The REFRAME that overtook (B) — the generalizing cortex ships ON POINT NEURONS (CYCLE 88–96, 2026-06-15/16)
Two findings (`2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`,
`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`, `2026-06-16-harden-320-stream-cortex-learned-moat-GO.md`,
commit `4bf19c02`) substantially overtook the dendritic motivation. **The settled, verbatim-faithful conclusions:**
- The "off-diagonal decorrelation" was a **RED HERRING.** A generalizing cortex needs **feedforward LOCAL
  normalization** (PPMI = log [Weber-Fechner] + per-hub frequency normalization + per-concept Carandini-Heeger
  divisive + ReLU threshold — all point-neuron-local diagonal ops), **NOT** cross-neuron decorrelation (which would
  *destroy* generalization by whitening away the semantic similarity).
- PPMI codes reach host (`+0.518` vs host `+0.442`), generalize (held-out `0.859`, 3 seeds, real TinyStories), and
  land in the binding sweet spot (between-cos ≈ 0.014; a wide 0.05–0.13 window where codes BOTH generalize AND bind).
- It is **realized on the real spiking substrate**: population coding lifts the read-out from 47% (1 neuron) to
  **100–108% of host** (8–32 neurons/concept), retiring the single-neuron rate-code wall. The biology-faithful
  **online STREAM cortex** (hears the corpus window-by-window; online Hebbian co-occurrence + running-frequency; NO
  preprocessing, NO global matrix) reaches the target (`corr(M,C_stream) +0.885`), and the **full conversation on the
  stream-learned codes is GO** (3-seed who/what recall 1.00; moat 0.96–1.00, the single tail false-accept restored to
  0 with more stream — code-fidelity cost, not a moat-mechanism weakening). **Scaled to 320 concepts, multi-seed:
  recall 1.00 every seed, moat clean** (the learned Bogacz-Brown gate is *cleaner* than the host threshold).

⇒ **The generalizing conversational cortex is shipped on point neurons, learned from experience.** This is the single
most important correction to the original "(B) needs the dendrite" framing.

### 1.3 D2 — what Phases 0–2 actually built + validated (pinned down)
The D2 build (`docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`, owner-approved after the D1.x ladder) added a
**dendritic per-presynaptic-source DIVISIVE gain** to the bridge:
- **Phase 0 (D1.7) = SURVIVES.** A full spiking two-compartment numpy probe (dendritic shunting → LIF soma,
  spike-count code): dendritic structure `+0.976` (≈ host) while the point-neuron control stays `−0.012` at every
  threshold; the somatic threshold *enhances* it. Owner-approved Phase 1 on this basis.
  (`dendritic_d1p7_spiking_twocompartment_derisk.py`.)
- **Phase 1 = DELIVERED + fully verified protected `sim/` edit.** `enable_dendritic_divisive_gain` — each presynaptic
  source's spike scaled by `g_i = σ/(σ + a_i)` (`a_i` = that source's firing-rate EMA), suppressing common
  high-frequency sources toward 0, passing rare informative ones near 1; the per-input divisive normalization a single
  soma cannot do, at the synaptic input. **Five guarded sites, default-OFF, byte-identical when off** — proven by
  **18/18 GPU conversational + composition tests verbatim** (incl. the no-confab moat) + 4 dedicated tests
  (`tests/test_dendritic_divisive_gain.py`; suppresses a high-activity source to 44.6%). Verified present this pass:
  `sim/config.py:233`, `sim/bridge.py:357,1347,5878`.
- **Phase 2 = HONEST NEGATIVE for the gain's NECESSITY (the load-bearing correction).** A clean-readout control
  **inverted** the early "gain confirmed" claim: point-neuron `+0.167` (gen 0.422) vs dendritic `+0.042` — **with
  enough read-out temporal integration the point neuron recovers the cortex-code structure on its own, and the gain
  HURTS** (`2026-06-14-D2-phase1-DONE-phase2-frontier.md` §CORRECTION). The doc's own conclusion, verbatim: *"on a
  spiking substrate, the firing-rate THRESHOLD (presence-binarization) + readout TEMPORAL INTEGRATION provide the
  common-mode robustness the dendritic gain was designed for — so the gain is not load-bearing for the cortex code."*
  And **why the rate-level D1 ladder over-stated it:** D1's point-neuron control was a single *global divisive gain at
  the rate level*, which lacked the spiking substrate's threshold + temporal integration. ⇒ the gain mechanism is
  *verified + harmless* but *not load-bearing* for the cortex code.
- **Phase 3 = PENDING, and now largely REDUNDANT for the conversational product.** Its verbatim gate
  (generalization-in-conversation about a held-out concept via a similar known one, moat intact, multi-seed) **has
  already been passed on point neurons** by the PPMI/stream cortex + the generalization capstone
  (`2026-06-16-generalization-capstone-verbalize.md`, `2026-06-16-harden-320-stream-cortex-learned-moat-GO.md`). So
  D2 Phase 3 would re-deliver, via a months-scale dendritic build, a capability the point-neuron substrate already
  delivers — unless a *measured* off-diagonal residual justifies it.

### 1.4 The decorrelation dendritic question is CLOSED-NEGATIVE (2026-06-17, 3 seeds — decisive)
The one residual the dendrite might uniquely buy for the cortex was the **off-diagonal (cross-neuron, low-rank)
decorrelation** PPMI's diagonal normalization leaves (diagonal ~+0.31 → host +0.44 → offline-ZCA +0.49/+0.52). The
fully-specified afternoon de-risk was run (`_phaseB_offdiagonal_dendritic_pc_derisk.py`, CPU, 3 seeds;
`2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md`). **NEGATIVE, decisively:** the Duong
fixed-frame-+-gains mechanism reaches +0.519, but its **lesion (g=0, centered-PPMI input, no learned off-diagonal
gains) gives the SAME +0.519**, at eff-rank 53 (not ~8). The learned off-diagonal gains are **inert** — **local,
feedforward PPMI-centering already reaches the whitening ceiling** (+0.519 ≈ ZCA +0.524), and there is nothing left
for a cross-neuron dendritic decorrelator to add. **⇒ the dendritic OP-A (divisive/whitening) question is CLOSED for
the conversational cortex; ship the flat cortex; reserve the dendrite for the artificial-life goal only.**

### 1.5 The LEARNED-BIND capability map (2026-06-16) — the settled conclusion, verbatim-faithful
`2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` (CYCLE 102–103) places each piece of the
conversational bind precisely. **The decisive contrast table (do not contradict):**

| Bind | Single-attribute | 3-way bundle (a fact) |
|---|---|---|
| **Fixed ±1 / FHRR algebra** (self-inverse role) | 1.000 | **0.989** |
| **Learned additive** (point-neuron) | **0.806** → real-LIF on-bridge **0.833** | 0.193 |
| **Learned multiplicative + learned LINEAR inverse** | 0.083 (broken) | 0.056 (broken) |
| chance | 0.062 | 0.062 |

The settled decomposition: **(1)** concept codes = LEARNED on the spiking substrate (the stream cortex) ✅; **(2)**
single-attribute binding = LEARNABLE and validated end-to-end in real spikes (numpy 0.806 → real-LIF on-bridge 0.833
= 100% of numpy) ✅; **(3)** multi-attribute BUNDLING (a fact = a superposition of bindings) requires a **FIXED,
self-inverse algebraic structure** — additive lacks any inverse (0.193), and a learned *linear* inverse cannot be a
reciprocal `1/u` (0.056, breaks even single-attribute), while the fixed structure bundles at 0.989. **The honest
conclusion, verbatim:** the conversational bind is *"learned representations flowing through a fixed,
biologically-grounded coincidence/multiplicative binding primitive"* — and that primitive is NOT a host shortcut; it
is binding-by-coincidence / dendritic-multiplication, a STRUCTURAL neural primitive, not learnable from scratch on
point neurons. The production composer already realizes exactly this (fixed ±1 coincidence binding the learned stream
codes; biologization sweep, 0.92 who-Q&A).

**The localization (why additive + a linear unbind cannot bundle):** unbinding role *t* from a superposition
`bundle = Σ_i (uᵢ ⊗ wᵢ)` requires applying the **role-specific inverse** `bundle ⊗ u_t⁻¹` — an element-wise PRODUCT
of two content streams (the role code × the bound vector) and a role-dependent reciprocal scaling. A shared LINEAR
unbind provably cannot implement a role-dependent multiplicative scaling — structurally incapable, independent of
capacity or training. **This is the same Mikulasch-Priesemann point-neuron limit, now for binding-superposition: the
operation is MULTIPLICATIVE, and multiplication is a DENDRITIC operation** (the finding explicitly notes "the project
already has a two-compartment dendritic neuron on the bridge (D2 arc)").

---

## 2. THE PRECISE OPEN QUESTION (the single deepest unresolved capability)

> **Is learnable multi-attribute composition — specifically, a multiplicative (fixed-self-inverse-role) bind with
> LEARNED filler codes that BUNDLES + generalizes — achievable on the substrate, recovering the superposition
> capacity that the learned ADDITIVE bind (0.193) and the learned-LINEAR-inverse bind (0.056) provably cannot, where
> a FIXED ±1 self-inverse bind reaches 0.989?**

This is the ONE corner of step-3 that is genuinely open AND is the only place the existing dendritic substrate might
unlock something point neurons provably cannot do. It is **NOT** the decorrelation/whitening question (OP-A — CLOSED
NEGATIVE, §1.4; the generalizing cortex ships on point neurons, §1.2). The crux distinction (the
`2026-06-17-dendritic-multiplicative-binding-scoping.md` correction, load-bearing): **the four "walls" do NOT all
share one op — they split into two dendritic ops mapping to two wall-families:**
- **OP-A — dendritic DIVISIVE normalization** (per-compartment inhibitory gain; Carandini-Heeger; Mikulasch-Priesemann)
  → the *decorrelation/whitening* family. **BUILT (D2 Phase 1) and found NOT load-bearing** on the spiking substrate.
- **OP-B — dendritic MULTIPLICATION / superposition-inverse** (a supralinear, all-or-none NMDA-plateau product on
  co-located coincident clustered inputs; Poirazi-Brannon-Mel two-layer subunit; Major-Larkum-Schiller NMDA spike) →
  the *binding / bundling / nested-composition / non-adjacent agreement* family. **This is the open question's op.**

**The biological mechanism (why a dendrite CAN where a summing soma cannot):** a dendritic branch computes a
supralinear PRODUCT of co-located synaptic inputs — a cluster on ONE branch produces a regenerative NMDA plateau ≫ the
same inputs scattered across branches. That branch-local product IS the `role ⊗ bound` multiplication a single summing
soma cannot form (it sums linearly + commutatively). **Catalog entries (verified this pass):** **G.02 "Active
dendrites — MISSING"** (NMDA spikes, Larkum two-layer "cluster ≫ scattered" nonlinear summation, ~10× compute/neuron,
"one of the largest abstractions in the simulator", `feature-catalog.md:2644–2652`); **B.17** (dendritic
linearization); **J.08** (NMDA receptor — voltage-dependent coincidence detector). Literature: Poirazi-Brannon-Mel
*Neuron* 2003 (two-layer subunit); Major-Larkum-Schiller *Annu Rev Neurosci* 2013 (the NMDA spike); Larkum BAC firing.

**The non-obvious un-exploited opening (load-bearing):** the bundling NEGATIVE "localized here" but its "multiplicative"
arm (`_phaseB_multiplicative_bind_bundled_derisk.py`) was **CONFOUNDED** — it used a *learned LINEAR* `W_Rinv`, which
provably cannot be a reciprocal, so it broke even single-attribute (0.056). It did NOT test a genuine PRODUCT with a
**fixed self-inverse** role. AND the dendritic-multiplication PRIMITIVE — `fused_coincidence_plateau` (`sim/kernels.py:253`,
guarded `enable_coincidence_detection`, `cp_coincidence_synapse_mask`, byte-inert when off; verified present this pass)
— **is already on the bridge and has NEVER been wired as a binding primitive.** So the project built the wrong-half
dendritic op (OP-A divisive, which the substrate didn't need) and has not de-risked the right-half (OP-B product) as a
binding mechanism. That is the precise, un-exploited gap.

---

## 3. RANKED OPTIONS for the next build (leverage × cheapness)

Ranked for the project's north star (artificial life / biology-translatable; capabilities instrumental; honest
negatives are the deliverable; the no-confab moat is a PLUS not a hard gate per owner 2026-06-17).

### Option 1 (RECOMMENDED FIRST — afternoon CPU/numpy, NO `sim/` edit) — the multiplicative (self-inverse-role) BUNDLING A/B
- **Demonstrable claim:** a multiplicative bind with a **FIXED self-inverse role + LEARNED filler codes** recovers
  multi-attribute bundling (held-out ≥ 0.40) that the learned-additive (0.193) and learned-linear-inverse (0.056)
  binds cannot, on the IDENTICAL corpus/splits/seeds, while the fixed-±1 positive control (0.989) confirms the harness
  detects working bundling. *This is the untested MIDDLE of the capability map* (fixed-on-both-sides bundles 0.989;
  learned-on-both-sides collapses; fixed-role-+-learned-filler is the open cell).
- **Existing machinery it reuses (all verified present this pass; new code ≈ 60–100 lines):**
  `research/runners/_phaseB_multiplicative_bind_bundled_derisk.py` (the confounded NEGATIVE to fix — swap the learned
  linear `W_Rinv` for a fixed self-inverse role), `_phaseB_fixed_fhrr_bundled_control.py` (the 0.989 positive control),
  `_phaseB_learned_bind_bundled_facts_derisk.py` (the 0.193 additive NEGATIVE),
  `cortex_learned_binder_systematicity_probe.py` (`make_role_codes` / `make_systematicity_splits` / `native_argmax` —
  leakage-free splits + the full anti-cheat battery), cached `raw/_phaseB_stream_codes_320_seed42.npy`.
- **Effort:** an afternoon, CPU/numpy, ≥6 seeds (42/43/44/100/101/102), NO `sim/` edit, NO GPU.
- **Dud-risk:** LOW as a decision tool (either outcome is a deliverable). Genuine empirical unknown: whether
  fixed-role-+-LEARNED-filler bundles, or only fixed-on-both-sides does. **Honest nuance (do not over-claim a GO):** a
  fixed-self-inverse-role bind is precisely what the production FHRR composer already does at 0.989 — so a GO here
  primarily proves the **LEARNED-FILLER + spike-read** version holds (lifting the composer's "learned codes" boundary),
  NOT that "multiplication from scratch" is a new capability. A NEGATIVE is a clean, citable boundary (the fixed algebra
  is load-bearing on BOTH sides; the learned-bind frontier is closed for bundling).

### Option 2 (weeks, gated on Option 1 GO — small protected `sim/` edit, NOT a new NeuronModel) — wire the coincidence-plateau as a binding primitive
- **Demonstrable claim:** routing two content streams (a fixed self-inverse role + the bound vector) into a co-located
  coincidence cluster so `fused_coincidence_plateau` computes the `role ⊗ bound` product on the real spiking bridge
  recovers bundling that the linear unbind cannot — the *on-substrate* realization of Option 1's numpy result.
- **Existing machinery:** the coincidence-plateau primitive is **already built + guarded** (`fused_coincidence_plateau`,
  `cp_coincidence_synapse_mask`); the new work is *additive wiring* (a binding-specific synapse-mask + a fixed
  self-inverse role projection on the RF complex-synapse path the composer already uses), mirroring the existing mask
  machinery — **the scale of the D2 Phase-1 protected edit (5 guarded sites, byte-identical-when-off, delivered in days),
  NOT the months-scale `NeuronModel.TWO_COMPARTMENT`.** Protected-edit template: `sim/bridge.py:5805–5849` +
  `sim/kernels.py:253` (the guarded coincidence block).
- **Effort:** WEEKS (additive guarded wiring + a binding harness), gated on Option 1 GO + owner byte-level diff review.
- **Dud-risk:** MODERATE. Open: whether the plateau's *thresholded supralinear switch on a sum* (the Poirazi subunit)
  is expressive enough for the *full element-wise product* a binding unbind needs (vs needing a per-pair product) —
  exactly what Option 1's spike-count arm and an on-bridge single-neuron probe must confirm first.

### Option 3 (months, HIGH-variance, DEFER) — D2 Phase 3 / a true two-compartment `NeuronModel`, as a learning-rule (credit-assignment) unlock
- **What "D2 Phase 3 as currently scoped" actually targets — assessed:** Phase 3's verbatim gate is
  generalization-in-conversation with the moat intact — **which is already shipped on point neurons (§1.3)**, so D2
  Phase 3 *as scoped* re-delivers an existing capability and is **largely redundant** for the product. Its only unique
  payoff is the biology-translatable off-diagonal cortex science — **which the 2026-06-17 NEGATIVE (§1.4) closed.** A
  genuinely-new dendritic target is **credit assignment** (apical-basal; Sacramento-Costa-Bengio-Senn NeurIPS 2018;
  Guerguiev-Lillicrap-Richards 2017) — a *learning-rule* unlock (better credit → fewer samples → deeper learnable
  networks), the piece the committed off-bridge stack (`sim/dendritic_neuron.py` Larkum/GLR2017,
  `sim/dendritic_plasticity.py` Urbanczik-Senn, `sim/dendritic_mlp.py`) most directly supports. **But this is a
  DIFFERENT problem from the binding walls** (representational, not credit-assignment), it has a **prior sound-instrument
  VOID** (`2026-05-18-dendritic-fairscale-SOUND-instrument-VOID...`, on a CIFAR-scale credit-assignment question), and
  the #2c feedback-alignment de-risk for the binder backward pass was seed-unstable (0.528 mean; needs e-prop). **Wrong
  lever for the open binding question; defer until binding is unlocked and the bottleneck demonstrably shifts to
  sample-efficiency.**
- **Effort:** MONTHS (1.5–2 floor; hot-path protected edit). **Dud-risk:** HIGH (prior VOID; redundant target).

### Ranking summary
| | What it tests/builds | Substrate touch | Cost | Decides |
|---|---|---|---|---|
| **1 (do FIRST)** | fixed-self-inverse-role + learned-filler bundling (numpy A/B) | NONE | **afternoon** | does dendritic MULTIPLICATION unlock learnable bundling |
| **2 (weeks, gated on 1)** | coincidence-plateau as a binding primitive on the bridge | small guarded edit (Phase-1-scale) | weeks | the on-substrate realization |
| **3 (months, DEFER)** | D2 Phase 3 / two-compartment NeuronModel (credit-assignment) | new NeuronModel (guarded) | months | a scaling/learning-rule unlock (redundant for binding) |

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK (the single highest-leverage cheapest-first probe)

**= Option 1, verbatim from `2026-06-17-dendritic-multiplicative-binding-scoping.md` §4** (afternoon, CPU/numpy, NO
`sim/` edit, NO GPU). It is the cheapest possible thing that decides the only open step-3 corner, and either outcome
is a deliverable. It is the project's own un-executed recommendation — **confirmed STILL UN-RUN** (no `2026-06-18`
result doc; no new self-inverse-bind runner; the latest AUTONOMOUS_STATE cycle, CYCLE 194, names *this exact arc* as
the next frontier).

**What to test (the single op change — the lever the prior NEGATIVE was missing):** re-run the EXACT 2026-06-16
bundling NEGATIVE harness with ONLY the binding op changed:
- The prior (confounded) arm: `unbind = bundle ⊙ (role @ W_Rinv)` with a **learned LINEAR** `W_Rinv` (can't be a
  reciprocal → broke even single-attribute, 0.056).
- **The de-risk arm:** `bind = role_fixed ⊙ (filler @ W_F)` with a **FIXED self-inverse** `role_fixed` (a ±1 /
  unit-phasor hypervector, its own inverse under ⊙ — the dendritic-coincidence analogue: a *structural* product, not a
  learned linear map), `unbind = bundle ⊙ role_fixed`, **LEARNED filler codes `W_F`**. Isolates the one question: does a
  genuine multiplicative (self-inverse-role) bind let the LEARNED fillers bundle + generalize. **Optional second arm:**
  read it as a finite Poisson **spike count** through the coincidence-plateau's supralinear switch, to face the
  spike-noise floor that killed prior attempts cheaply, before any bridge work.

**Existing knobs/runners to reuse:** the four harness files in §3-Option-1 + the cached 320 stream codes (all verified
present). New code ≈ 60–100 lines (the fixed-self-inverse-role arm + the optional spike-count read).

**Quantitative GO bar (pre-registered ≥6 seeds, fractional ≥5/6 per `feedback_6seed_validation`):**
- **GO:** bundled held-out-combo `≥ 0.40` AND `≥ 0.6 × train-combo` AND single-binding generalizes `≥ 0.40`, on ≥5/6
  seeds, WHILE the additive (0.193) + learned-linear (0.056) arms stay at their NEGATIVE on the identical harness. ⇒ the
  multiplicative (self-inverse-role) bind with learned fillers recovers superposition the point-neuron linear bind
  cannot → Option 2 (the small protected wiring edit) is warranted.
- **BOUNDARY:** the multiplicative bind bundles (≥0.40) but the **spike-count** arm falls short of the float arm → the
  lever is real but the spiking realization needs the on-bridge plateau (one more cheap on-bridge single-neuron arm
  before the wiring edit).
- **NEGATIVE:** even the fixed-self-inverse-role multiplicative bind with learned fillers stays ≤ 0.25 bundled held-out
  → multiplication *with learned representations* is insufficient; superposition needs either an iterative resonator
  cleanup in the loop, OR the fixed-FHRR algebra stays the production binding primitive (the honest status quo, V=320).

**What an honest NEGATIVE means (and that it IS the deliverable):** a NEGATIVE pins exactly which dendritic ingredient
(product vs iterative cleanup) the superposition-inverse needs, and confirms the production conclusion — "learned codes
+ a FIXED biology-grounded coincidence bind primitive" — is the right resting point, closing the learned-bind frontier
for bundling on a measured signal. That re-confirms a real point-neuron/representation boundary and maps the dendritic
substrate's reach for binding — a citable artificial-life result under "honest negatives are the deliverable." It does
NOT reopen the (already-CLOSED) decorrelation question.

---

## 5. ANTI-CHEAT CONTROLS (mandatory — what makes the result real + brain-based)

All already in the harness; this is the standard battery, foregrounding the established contrasts.
1. **POINT-NEURON / ADDITIVE + LEARNED-LINEAR MUST FALL SHORT (the headline A/B):** the additive (0.193) + learned-linear
   (0.056) arms must stay at their NEGATIVE on the IDENTICAL corpus/splits/seeds while the multiplicative arm exceeds
   0.40 — **mirroring the established 0.989-fixed-vs-0.193-additive contrast.** If the controls don't fall short, the
   harness isn't reproducing the wall → re-tune before trusting any GO. (The dendritic GO only counts against a
   point-neuron NEGATIVE on the same data.)
2. **FIXED-±1 POSITIVE CONTROL CARRIES (0.989):** the harness must DETECT working bundling (proves a NEGATIVE is real,
   not a broken harness).
3. **HELD-OUT systematicity with a leakage assertion vs a memorization floor:** `make_systematicity_splits` (train-combo
   vs never-seen held-out-combo); the GO bar is on **held-out** (generalization, not memorization). Memorization-floor
   (lookup table → 0.0) + chance line (1/F = 0.062) reported.
4. **PERMUTED-ROLE control:** shuffle role→filler assignments → recall collapses to chance (the bind reads role
   structure, not a code-overlap artifact; mirror of `permuted_label_check.py`).
5. **LESION (the bind/compose must be NEURAL not host):** freeze the fixed role to identity / replace the product with a
   sum → the bundling lift collapses to the additive value (proves the lift RIDES the multiplicative op, not a host
   transform or leftover code property).
6. **COMPOSITION-NOT-COHERENCE (the 2026-06-06 lesson):** the metric is *unbind-recovers-the-right-filler*, not a
   decorrelation/reproducibility proxy — a noise-collapsed output fails it.
7. **NEVER WEAKEN THE NO-CONFAB MOAT:** if this reaches an on-substrate confirmation, the shuffled-fact
   zero-false-accept + the abstention floor carry from V=320; the gate threshold is *not* tuned on the test. (A PLUS not
   a hard gate per owner 2026-06-17, but free here.)
8. **THE DENDRITE-UNLOCKS-MULTI-ATTRIBUTE CLAIM REQUIRES A POINT-NEURON NEGATIVE ON THE SAME HARNESS:** if Option 1 (or
   the on-bridge Option 2) claims the dendritic product unlocks bundling, the additive/linear point-neuron arms MUST
   fail on the identical corpus/splits/seeds (control #1) — the established 0.989-fixed-vs-0.193-additive contrast is the
   template; a "dendritic GO" without the point-neuron NEGATIVE on the same data is rejected.
9. **≥6 seeds (42/43/44/100/101/102), fractional ≥5/6; CPU/numpy for the cheap-first, CuPy for any decisive on-bridge
   promotion** (`feedback_gpu_not_numpy`).

---

## 6. Trust-but-verify flags (load-bearing claims; verified vs. could-not)

**Verified directly this pass (file/finding/line cited):**
- The generalizing cortex ships on point neurons via PPMI (off-diagonal a red herring; 320-concept stream cortex
  recall 1.00 + moat clean, commit `4bf19c02`) — read `2026-06-15-off-diagonal-red-herring...`, `2026-06-15-on-bridge-hebbian-co-occurrence...`,
  `2026-06-16-harden-320-stream-cortex-learned-moat-GO.md`.
- The off-diagonal dendritic de-risk is **NEGATIVE, 3 seeds** (mechanism == lesion +0.519, eff-rank 53; PPMI-centering
  reaches the ceiling) — read `2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md` verbatim.
- The learned-bind capability map (single-attr GO 0.806→real-LIF 0.833; bundling NEGATIVE additive 0.193 / learned-linear
  0.056; fixed-±1 0.989) — read `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` in full.
- D2 Phase 1 DELIVERED + Phase 2 NEGATIVE-for-necessity (clean-readout inversion point-neuron +0.167 vs dendritic +0.042)
  — read `2026-06-14-D2-phase1-DONE-phase2-frontier.md` §CORRECTION; flags `enable_dendritic_divisive_gain` /
  `cp_dendritic_source_activity` confirmed at `sim/config.py:233`, `sim/bridge.py:357,1347,5878`.
- The multiplicative-binding A/B is fully specified and **STILL UN-RUN** — read `2026-06-17-dendritic-multiplicative-binding-scoping.md`
  §4; confirmed no `2026-06-18` result doc + no new self-inverse-bind runner (`ls`); AUTONOMOUS_STATE CYCLE 194 names
  this exact arc as the next frontier.
- The coincidence-plateau primitive exists, guarded, byte-inert when off — `sim/kernels.py:253` (`fused_coincidence_plateau`),
  `sim/config.py:173` (`enable_coincidence_detection`), `sim/bridge.py:270` (`cp_coincidence_synapse_mask`).
- The bundling harness files + cached 320 stream codes all present — `ls` verified.
- Catalog G.02 "Active dendrites — MISSING" (NMDA spikes, Larkum two-layer cluster≫scattered, ~10× compute/neuron),
  B.17, J.08 — read `sim-catalog/.../feature-catalog.md:2644–2652, 375, 3593`.

**Could NOT fully verify (flagged honestly):**
1. **Whether the §4 A/B returns GO/BOUNDARY/NEGATIVE** — the genuine open empirical question; deliberately not predicted.
   The fixed-±1 control bundles (0.989), the learned-linear collapses (0.056); the fixed-role-+-LEARNED-filler MIDDLE is
   untested. The de-risk pre-registers all three outcomes.
2. **Whether the coincidence-plateau's thresholded-supralinear-switch-on-a-sum form is expressive enough for the full
   element-wise `role ⊗ bound` product** (vs needing a per-pair product) — exactly what the §4 spike-count arm + an
   on-bridge probe (Option 2) must confirm before the wiring edit.
3. **The exact byte-size of the Option-2 on-bridge binding-routing edit** — asserted "weeks (Phase-1-scale), not months
   (NeuronModel-scale)" from the existing `cp_coincidence_synapse_mask` machinery; a builder should scope the
   binding-mask routing concretely (not drafted here).

**No banking.** Reported exactly as the record stands — including that the *original* dendritic motivation (the
generalizing cortex) is gone (shipped on point neurons), the decorrelation dendritic question is CLOSED-NEGATIVE, and
the ONE open dendritic question (multiplication-as-binding for bundling) has a confounded prior NEGATIVE and an
afternoon-scale A/B that decides it.

## Sources (the load-bearing project record; literature in the cited findings)
- The fork + D2 decision/plan: `docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`,
  `docs/plans/2026-06-14-step3-cortex-fork-resolved-dendritic-D2-decision.md`,
  `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`.
- The reframe (generalizing cortex on point neurons): `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`,
  `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`, `2026-06-16-harden-320-stream-cortex-learned-moat-GO.md`.
- D2 build state: `2026-06-14-D2-phase1-DONE-phase2-frontier.md`, `2026-06-14-dendritic-D1-cheap-derisk-GO.md`,
  `2026-06-14-dendritic-substrate-deep-research.md`.
- The decorrelation-dendrite NEGATIVE: `2026-06-17-offdiagonal-dendritic-derisk-NEGATIVE-ship-flat-cortex.md`,
  `2026-06-17-dendritic-substrate-frontier-scoping.md`.
- The learned-bind map + the OP-A/OP-B split + the A/B spec: `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`,
  `2026-06-17-dendritic-multiplicative-binding-scoping.md`.
- On-bridge primitives: `sim/kernels.py:253` (`fused_coincidence_plateau`), `sim/bridge.py:6053–6107` (coincidence
  block), `sim/bridge.py:357,1347,5878` + `sim/config.py:233` (divisive gain); the dendritic stack
  `sim/dendritic_neuron.py`, `sim/dendritic_plasticity.py`, `sim/dendritic_mlp.py`.
- Harness + codes: `research/runners/_phaseB_multiplicative_bind_bundled_derisk.py`,
  `_phaseB_fixed_fhrr_bundled_control.py`, `_phaseB_learned_bind_bundled_facts_derisk.py`,
  `cortex_learned_binder_systematicity_probe.py`; `research/findings/raw/_phaseB_stream_codes_320_seed42.npy`.
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` G.02 (line 2644), B.17 (375), J.08 (3593).
