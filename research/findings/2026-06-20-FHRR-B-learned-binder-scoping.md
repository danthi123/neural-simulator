# FHRR-B (the host-designed exact-inverse bind algebra) — learned-binder scoping under the "boundary is not an exit" rule

**Type:** READ-ONLY deep-research + catalog/literature/code scoping (the standing "research-FIRST before committing
build/GPU/`sim/` effort to push past a frontier"). NO code written, NO experiments run, NO GPU. ONE findings doc.
Stayed on `main`. Every load-bearing project claim is file/finding-cited and was read in full (not grepped) where it
is load-bearing; literature is paper-cited.

**The shortcut (FHRR-B), defined precisely.** The production composer binds `role ⊗ filler` and **bundles** a fact
(= superposes the three role-filler bindings of an SVO sentence into one vector) using a FIXED, exactly-invertible
Fourier-Holographic-Reduced-Representation (FHRR) algebra: bind = element-wise complex product of phasors, unbind =
multiply by the conjugate. The bind/unbind *operations* are already on-substrate spiking (`NeuronModel.RESONATE_AND_FIRE`
+ complex synapses — not a shortcut). **The shortcut is the FORM:** the exact-inverse algebraic structure is
host-DESIGNED (an Eliasmith Spaun / Semantic-Pointer idealization), where a real cortex would have a LEARNED, lossy,
redundant read-out. "Bundle" throughout = superpose multiple role-filler bindings into one fact vector; "unbind from a
bundle" = recover one role's filler from that superposition.

**The owner's rule that governs this scope (CYCLE 329, commit `286f8368`):** a BOUNDARY is **not** an exit — it is a
prompt to research more + try NEW mechanisms until past it; the arc cannot end while a cognitive computation is still a
host-designed shortcut. So this scope does **not** recommend "close as honest-negative." It synthesizes WHY the prior
learned-bind attempts failed, then ranks the NEXT mechanisms to try — honestly likelihood-ordered, because the rule is
"try the most-promising first," not "all are equal."

---

## 0. TOP-LINE (the honest state, then the path)

**A great deal of FHRR-B has ALREADY been converted from host-designed to learned/spiking — the residual that is still
genuinely "designed, not learned-by-the-brain" is narrow and precisely localized.** Verified against the findings:

| piece of the bind | status | evidence |
|---|---|---|
| the bind/bundle/unbind OPERATIONS (resonate-and-fire + complex synapses) | **on-substrate SPIKING — not a shortcut** | `RFPhasorComposer`/`OneBrainComposer` (CLAUDE.md "OPPONENCY ESCAPED") |
| the concept CODES (the fillers) | **LEARNED + ships** (PPMI stream cortex, 320, real spikes) | `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` |
| **SINGLE-attribute role-filler binding** | **LEARNED + validated on real LIF spikes** (numpy 0.806 → on-bridge 0.833 = 100% of numpy) | `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md` |
| **the bundle INVERSE — its exact-inverse FORM** | **host-DESIGNED — THE load-bearing residual of FHRR-B** | the same finding; the FORM is the idealization |

**So FHRR-B has shrunk to ONE op: the exact-inverse FORM of the *multi-attribute bundle inverse*** (recover one role's
filler from a superposition of three bindings). Everything else in the composer's bind is already learned or already
spiking. This is the precise, narrow target the next mechanisms must hit — and the honest synthesis below shows it is
the single hardest op in the whole conversational stack, because **a superposition (a sum) has no exact inverse by a
theorem of the algebra**, so recovering a clean bundled filler requires a *separate, structured* multiplicative inverse
— which is exactly what a from-scratch learner has repeatedly failed to discover.

---

## 1. WHY IT FAILED — the synthesis (BINDING wall vs CODE wall, kept strictly separate)

Four from-scratch learned-bind attempts are on the record, all on the IDENTICAL leakage-free systematicity harness
(`cortex_learned_binder_systematicity_probe.py`: F=16 fillers × R=4 roles; the held-out split holds out specific
(role,filler) *pairings* while every role and every filler still appears in training — the Fodor-Pylyshyn 1988
never-seen-combination test; with a memorization-floor lookup-table control, a permuted-role control, and a leakage
assertion). **The decisive metric is held-out generalization, never raw recall.**

### 1.1 The capability map (verbatim-faithful; the headline contrast)

| Bind | single-attribute held-out | 3-way bundle (a fact) held-out | source |
|---|---|---|---|
| **Fixed ±1 / FHRR self-inverse** (the current shortcut) | 1.000 | **0.989** | `2026-06-16-...bundling-NEGATIVE.md` |
| **Learned ADDITIVE** (point-neuron) | 0.806 → real-LIF **0.833** | **0.193** (fails even on TRAIN combos → crosstalk) | same |
| **Learned MULTIPLICATIVE + learned-LINEAR inverse** | 0.083 (broken) | **0.056** (broken — even single-attr) | same |
| **Learned dendritic σ-π / NMDA-plateau** conjunction | 0.500 | **0.168** (train 0.422 → +0.254 mem gap; **below** fixed FHRR 0.261) | `2026-06-19-dendritic-binding-toy-derisk.md` |
| chance (1/F) | 0.062 | 0.062 | — |

### 1.2 The localization — the BINDING wall (the load-bearing one), in one paragraph

Unbinding role *t* from a bundle `Σᵢ (uᵢ ⊗ wᵢ)` requires applying the **role-specific reciprocal** `bundle ⊗ uₜ⁻¹` — an
**element-wise PRODUCT of two content streams** (the role code × the bound vector) followed by a **role-dependent
reciprocal scaling**. Three independent ways to learn this from scratch each fail for a *structural*, not a tuning,
reason: **(a) additive** bind has no inverse at all (0.193, fails even on train combos — pure superposition crosstalk);
**(b) a learned LINEAR inverse** provably cannot be a reciprocal `1/u` (0.056, breaks even single-attribute — a linear
map is the wrong function class); **(c) a learned dendritic multiplication** *can* form the product so it MEMORIZES the
train bundles (0.422) but does not *generalize* (held-out 0.168) — the supralinear product is load-bearing for the fit
(lesion → 0.032) but a learned plateau does not discover the *systematic* reciprocal. The one form that works — **a
fixed self-inverse role** (`a ⊗ a = 1`, so `(a⊗b)⊗a = b`; the MAP/±1 algebra) — bundles at 0.989 *because* the inverse
is given by the algebra, not learned. **⇒ the BINDING wall is: the bundle inverse is a structured multiplicative
reciprocal that a from-scratch learner does not discover; only a fixed (self-)inverse structure supplies it.** This is
a theorem of vector-symbolic algebra (bundling is similarity-preserving additive superposition, which by construction
has no exact inverse — ACM Computing Surveys VSA survey; the self-inverse property is what makes a bundled element
recoverable), corroborated by the project's harness, not an artifact of it.

### 1.3 The CODE wall — SEPARATE, and largely already solved (do not conflate)

A *second* axis — whether the *codes* are decorrelated enough to bind/unbind cleanly — is the
`2026-06-11-cortex-learned-binder-systematicity-NEGATIVE-ON-CORRELATED.md` result: a learned binder is systematic on
**decorrelated** codes (held-out 1.000) and fails on the brain's **correlated** codes (≈ chance). **This is a CODE
property, not a binding-op property,** and it has been resolved on a *different* axis: the PPMI stream cortex learns
codes that land in the binding sweet-spot (between-cos ≈ 0.014) AND generalize across similar concepts (held-out 0.86),
on real spikes, 320 concepts (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`). **The strategic
prize a learned binder was originally wanted for — generalization across similar concepts — is therefore ALREADY
delivered on the codes axis.** Closing FHRR-B (the bind FORM) does NOT add that prize; it is a brain-based-purity goal
(retire the host-designed exact-inverse FORM), not a capability goal. This must be stated plainly so the effort is
scoped to what it actually buys.

### 1.4 The crucial partial that bounds the problem (the FRLF result)

The one untested CELL of the map — **fixed self-inverse role + LEARNED filler** — was run and is **GO**
(`2026-06-18-learned-filler-fixed-bind-bundles-GO.md`, CYCLE 196): bundled held-out **0.639** (3-seed) / 0.603 (6-seed),
single-binding held-out 0.833, beats additive (0.193) and learned-linear (0.056) at every seed; lesion (drop the
multiplicative self-inverse → sum) collapses to chance; on-bridge who/what 1.000 6-seed with the moat intact
(`2026-06-17-onbridge-learned-composer-step2-GO.md`). **Reading:** *learned fillers are fully compatible with a fixed
self-inverse bind* — but the bundle INVERSE is still the fixed structure (FRLF retires the *curated-codes* half of the
idealization, NOT the *exact-inverse-FORM* half). It lands ~0.35-0.39 below the fixed-algebra ceiling (0.989/0.993).
**So the genuinely-open question this scope targets is narrower and harder than "a learned binder":** can the
multi-attribute bundle INVERSE itself be *learned* (or learned-and-iterative) and stay systematic, recovering the
~0.35 gap to the fixed ceiling — where additive (0.193), learned-linear (0.056), and learned-dendritic (0.168) all
failed?

### 1.5 The dendrite, specifically — why "dendrite-first" is NOT the answer this time (the key update vs prior scopings)

The two prior scopings (`2026-06-17-dendritic-multiplicative-binding-scoping.md`,
`2026-06-18-step3-dendritic-learned-bind-frontier-scoping.md`) recommended the dendritic-multiplication A/B as the
cheapest-first decisive move. **That A/B has since been RUN (`2026-06-19-dendritic-binding-toy-derisk.md`) and is
NEGATIVE** (memorizes 0.422, generalizes 0.168, *below* the fixed FHRR 0.261 it would replace). The dendrite's *other*
named job — apical-basal credit assignment — was ALSO run and is NEGATIVE (`2026-06-19-dendrite-credit-assignment-toy-
stage1.md`: a single-layer actor has nothing for feedback-alignment to align). **So both named dendrite jobs for
binding are now cheap-first NEGATIVE.** The dendritic substrate (`sim/dendritic_neuron.py` Larkum-BAC two-compartment;
`sim/dendritic_plasticity.py` Urbanczik-Senn local rule; `sim/dendritic_mlp.py` deep feedback-alignment MLP;
`enable_dendritic_divisive_gain` shipped on-bridge) is real and reusable — but a *bare dendritic product as the bind
op* is ruled out for generalization. **The honest correction to the task's "the dendrite is a prime candidate": it is a
candidate only in a SPECIFIC reframed form (a HIDDEN-LAYER architecture where its credit-assignment value actually
applies — §2 Option 2), not as the binding nonlinearity itself (already NEGATIVE).** The dendrite literature is
unambiguous that apical-basal credit assignment needs DEEP/hierarchical nets (Sacramento-Senn 2018; Payeur-Naud-Richards
2021) — so the dendrite can only help if the binder is posed as a deep net with hidden units, which none of the prior
attempts were.

---

## 2. RANKED NEXT-MECHANISMS (likelihood-ordered; what to try to actually CLOSE the bundle-inverse FORM)

Ranked by P(closes the bundle-inverse FORM with held-out systematicity) × cheapness. Each is a mechanism the project
has **not** yet tested for the bundle inverse. The bar to "close FHRR-B" is: a *learned* (non-host-designed) bundle
inverse that recovers held-out bundled fillers ≥ ~0.9 (parity with the fixed 0.989), systematically, on real spikes.

### Option 1 (DO FIRST — afternoon CPU/numpy, NO `sim/` edit) — the FRLF capacity + LEARNED-ITERATIVE-cleanup sweep

**This is the highest-leverage cheapest probe and it is the one un-run thread with positive signal.** The FRLF GO
(0.639) used a single-pass nearest-cosine cleanup at one bind-space dimension. Two levers, never swept, plausibly lift
it toward the fixed ceiling — and crucially, **the cleanup is where the from-scratch learned-inverse can legitimately
re-enter**, because a bundle has no one-shot inverse but DOES have an *iterative* decomposition (the resonator-network
principle: alternate "unbind by the fixed structure → clean up each factor → re-bundle the residual"):
- **(a) capacity:** sweep bind-space `D_h` 64 → 128 → 256 → 512 (VSA bundling capacity is dimension-bounded; the F=3
  resonator needs D ≥ 2048 to even decode two attributes on clean codes, `2026-06-19-resonator-on-learned-codes-derisk.md`
  — so the FRLF 0.639 at low D_h is plausibly capacity-starved, not mechanism-limited).
- **(b) a LEARNED iterative cleanup in the loop** (the genuinely-new lever): replace the single nearest-cosine cleanup
  with a **learned multiplicative / resonator-style iterative decomposition** (Frady-Sommer resonator networks; the AID
  attention-based iterative decomposition for tensor-product representations, arXiv 2406.01012) — learn the *cleanup /
  decomposition* (which IS allowed to be learned, and is the part a from-scratch additive bind lacked), while the bind
  stays the fixed self-inverse. This directly attacks the ~0.35 gap with a learned component on the one sub-op that can
  carry it.
- **Why first:** reuses the entire FRLF harness + the 320 stream codes (all present); CPU/numpy; either outcome is a
  deliverable. **GO** (held-out → ~0.9 on ≥5/6 seeds, additive/linear arms still NEGATIVE on the identical harness) ⇒
  the learned-iterative-cleanup + fixed-bind is the on-bridge realization to wire (Option 4). **NEGATIVE** ⇒ the gap is
  fundamental at this representation and the next-mechanism is a structurally-different code (Option 3).
- **Honest nuance:** even a GO here keeps the *bind* fixed and learns the *cleanup* — so it closes FHRR-B's "the
  read-out is host-designed" half (the cleanup becomes learned + lossy + redundant, exactly the realistic-cortex
  property the idealization lacked) and shrinks the residual to "the bind op is a fixed self-inverse structure" — which
  is a *structural neural primitive* (coincidence/dendritic-product), not a host computation. That is a genuine,
  defensible reduction of the shortcut under the BRAIN-BASED-ONLY standard, even though it is not "a bind learned
  end-to-end from scratch."

### Option 2 (weeks, gated on Option-1 signal — a DEEP / hidden-layer learned binder, where the dendrite CAN re-enter)

**The reframe that makes the dendrite relevant again.** Every from-scratch attempt (additive, learned-linear, dendritic)
was a *shallow* one-projection bind + one-readout unbind. The dendrite's credit-assignment value is real but **only in
HIDDEN-LAYER nets** (Sacramento-Senn 2018; Payeur 2021). So the untested mechanism is a **multi-layer learned binder**
(a 2-3-hidden-layer bind+unbind net) trained either by **(a) surrogate-gradient BPTT** (the host-shortcut *ceiling*
characterization, using `sim/bptt_snn*` + `sim/surrogate_grad.py`) or **(b) e-prop / a local three-factor eligibility
rule** (the brain-faithful target; Bellec 2020; the bridge's `cp_eligibility_trace` substrate), with the dendritic
apical-basal machinery (`sim/dendritic_mlp.py`, already a deep feedback-alignment MLP) routing hidden-layer credit. The
biology grounding is direct: the **bilinear-gating** circuit (arXiv 2606.10891, 2026) makes a two-compartment neuron's
burst the *product* of a soma (filler) and dendrite (role) signal, trained by a LOCAL three-factor rule, and claims
zero-shot generalization *by the multiplicative inductive bias*. **Why it might beat the shallow NEGATIVEs:** a deep net
has the capacity + the credit-routing the shallow learned-linear inverse structurally lacked. **Why it is ranked second,
not first:** it is weeks not an afternoon; surrogate-BPTT is a host-shortcut characterization (a PASS there is "a spiking
binder of this form *can* be systematic," not "the brain binds"); and the prior credit-assignment NEGATIVE means the
e-prop/dendritic version is genuinely at-risk. **Run the BPTT ceiling FIRST as the cheap gate** (CPU, < 30 min/seed): if
even a deep BPTT learned binder cannot generalize the bundle inverse, no local rule will, and that NEGATIVE closes the
"learned bundle inverse" question on a measured signal — itself the deliverable.

### Option 3 (weeks, parallel-able — a structurally-different REPRESENTATION: distinct per-attribute role tags + tensor-product)

The bundle inverse is hard *because* multiple attributes share one commutative codebook (the resonator's permutation
symmetry, which correlation defeats — `2026-06-19-resonator-on-learned-codes-derisk.md`). **Remove the symmetry:** bind
each attribute under a DISTINCT named role tag (a tensor-product representation, TPR: `Σ_role role ⊗ filler` with
*orthogonal* roles), so unbind is `bundle · role_t` with no permutation ambiguity. The 2024-25 literature's own
resolution is exactly this — **learn the fillers + the decomposition, keep a structured (tensor-product) binding
primitive** (AID, arXiv 2406.01012; differentiable tree operations, arXiv 2306.00751; "RNNs implicitly implement TPR"
arXiv 1812.08718). This is more *cortex-faithful* than a commutative product (cortex binds *named* slots, not a
symmetric bag) and removes the exact failure mode the F=3 resonator hit. **Why ranked third:** it changes the
representation (fixed attribute *slots*, a real cost) and the *binding* primitive stays fixed (orthogonal-role TPR is
still a designed structure) — so like Option 1 it retires the codes/cleanup half, not the bind-FORM half. But it is the
literature-endorsed path to *systematic* multi-attribute binding and is worth a cheap numpy A/B (orthogonal-role TPR vs
the commutative resonator, same harness) that may simply *dissolve* the two-attribute boundary (#20 in the boundary
ledger) — a concrete capability win regardless of the FHRR-B FORM question.

### Option 4 (weeks, gated on Option-1 or Option-2 GO — the on-bridge protected edit)

If Option 1 (learned iterative cleanup) or Option 2 (deep learned binder) is GO at numpy, the on-bridge realization is a
**small, guarded, byte-reviewable `sim/` edit, NOT a new `NeuronModel`**: route the learned-cleanup / hidden-layer
through the existing complex-synapse RF path + the already-built, guarded `fused_coincidence_plateau` (the supralinear
product primitive, `sim/kernels.py:253`, byte-inert when off) — mirroring the D2-Phase-1 protected-edit pattern (5
guarded sites, byte-identical-when-off, `enable_dendritic_divisive_gain`). Validate against `vocab_ceiling_probe` at
V=320, 6 seeds, with the no-confab moat carried verbatim. **This is the brain-based-purity payoff: the bind read-out
becomes learned/spiking on the real bridge.**

### Ranking summary

| | mechanism | what it learns | substrate touch | cost | P(closes the bundle-inverse FORM) |
|---|---|---|---|---|---|
| **1 (do first)** | FRLF capacity + **learned iterative cleanup** sweep | the *cleanup/decomposition* (the part a bundle CAN have) | none | afternoon | **moderate** — best cheap shot; closes the read-out half for sure, may lift to parity |
| **2 (weeks)** | DEEP/hidden-layer learned binder (BPTT ceiling → e-prop/dendrite) | the *bind+unbind* end-to-end | none (numpy) → small guarded | weeks | low-moderate — the only path to an end-to-end *learned* bind; the dendrite's real re-entry; prior shallow NEGATIVEs are the headwind |
| **3 (weeks, parallel)** | distinct per-attribute role tags (orthogonal TPR) | fillers + decomposition; bind = orthogonal-role TPR | none → small guarded | weeks | moderate for the CAPABILITY (dissolves the 2-attr boundary); low for the FORM (TPR is still designed) |
| **4 (weeks, gated)** | on-bridge wiring of 1/2 | — | small guarded `sim/` edit (Phase-1-scale) | weeks | n/a (realization of a GO) |

---

## 3. REUSABLE MACHINERY (all verified present this pass)

**The learned-binder + bundling harness (extend these; do NOT re-build the splits/anti-cheats):**
- `research/runners/cortex_learned_binder_systematicity_probe.py` — `BilinearBinder`, `make_systematicity_splits`
  (leakage-free), and ALL four anti-cheats (memorization-floor, shuffled-label, abstention, leakage-assert).
- `research/runners/_phaseB_fixed_role_learned_filler_bundled_derisk.py` — the **FRLF GO harness (0.639)** = the exact
  base for Option 1's capacity + learned-cleanup sweep.
- `_phaseB_learned_bind_bundled_facts_derisk.py` (additive NEGATIVE 0.193), `_phaseB_multiplicative_bind_bundled_derisk.py`
  (learned-linear NEGATIVE 0.056), `_phaseB_dendritic_bind_derisk.py` (dendritic NEGATIVE 0.168),
  `_phaseB_fixed_fhrr_bundled_control.py` (the 0.989 positive control) — the established A/B arms to run alongside.
- Cached production codes: `research/findings/raw/_phaseB_stream_codes_320_seed42.npy` +
  `_phaseB_stream_codes_320_neural_seed{42,43,44}.npy` (the real learned/grounded codes).

**The dendritic substrate (Option 2's deep/credit-assignment re-entry) — reuse-by-import, NO bridge edit for the toy:**
- `sim/dendritic_neuron.py` (`DendriticLayer`: basal forward + FIXED-random apical feedback-alignment + BAC threshold
  lowering — verified, 58 lines, no autodiff/weight-transport).
- `sim/dendritic_plasticity.py` (`urbanczik_senn_update`: local apical-gated mismatch, no weight transport — verified).
- `sim/dendritic_mlp.py` (`DendriticMLP`: **a DEEP feedback-alignment MLP with per-hidden-layer fixed-random feedback**
  — this is the hidden-layer credit-assignment machine Option 2 needs; verified, GPU-backed via `sim.backend`).
- The shallow-actor credit-assignment toy `research/runners/_dendrite_ca_toy_derisk.py` (the NEGATIVE harness — its
  anti-cheat scaffold and the two-sided validity gate transfer directly).

**The surrogate-gradient / e-prop substrate (Option 2's two training routes):**
- `sim/bptt_snn.py` / `sim/bptt_snn_gpu.py` / `sim/surrogate_grad.py` (the BPTT ceiling, confirmed on `main`).
- On-bridge eligibility three-factor: `cp_eligibility_trace`, `fused_eligibility_trace_decay`, the reward→weight block
  (the e-prop substrate); `research/runners/compose_temporal_bind.py` already trains a compositional A→B bind through
  the eligibility kernel (the closest precedent).

**The on-bridge binding primitives (Option 4):**
- `NeuronModel.RESONATE_AND_FIRE` + `rf_kick`/`rf_set_complex_weights`/`rf_read_phases` (the complex-synapse path the
  composer already uses); the masked-RF-ops edit (owner-approved, default-off byte-identical) for co-residence.
- `fused_coincidence_plateau` (`sim/kernels.py:253`, guarded `enable_coincidence_detection`, byte-inert when off) — the
  supralinear product, the protected-edit template; `enable_dendritic_divisive_gain` (`sim/config.py:260`) — the
  byte-identical-when-off pattern to mirror.

**The acceptance harness + drop-in target:**
- `research/runners/vocab_ceiling_probe.py` (full who/what + abstention-floor + shuffled-fact, V=320, 6 seeds) — the
  ship gate; `BrainConversationalAgent`/`OneBrainComposer` interface (the drop-in: write a new composer, don't rewire
  the agent).
- The F=3 resonator path `research/runners/nested_composition_agent.py` + `_resonator_on_learned_codes_probe.py` (the
  two-attribute boundary Option 3 attacks).

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK + ANTI-CHEATS

**Run Option 1 first: the FRLF capacity + learned-iterative-cleanup sweep** (CPU/numpy, NO `sim/` edit, NO GPU,
afternoon). Extend `_phaseB_fixed_role_learned_filler_bundled_derisk.py`:
1. sweep `D_h ∈ {64,128,256,512}`;
2. add a cleanup arm = a **learned multiplicative / resonator-style iterative decomposition** in the unbind loop (vs the
   current single nearest-cosine), keeping the bind a fixed self-inverse;
3. report bundled held-out vs train vs memorization-floor vs chance, ≥6 seeds (42/43/44/100/101/102), against the
   established additive (0.193) / learned-linear (0.056) / fixed-±1 (0.989) arms on the identical harness.

**Pre-registered verdict (fixed bars, never tuned to the result):**
- **GO:** bundled held-out **≥ 0.90** AND ≥ 0.6× train, on ≥5/6 seeds, WHILE additive + learned-linear stay NEGATIVE on
  the same data ⇒ the learned iterative cleanup + fixed bind reaches fixed-algebra parity → Option 4 (the small guarded
  on-bridge wiring) is warranted; FHRR-B's read-out half is closed (learned + lossy + redundant).
- **BOUNDARY:** held-out lifts (e.g. 0.639 → 0.75-0.85) but short of parity ⇒ the cleanup learns but the bind-FORM gap
  is partly fundamental → proceed to Option 2/3 (deep binder / orthogonal-TPR) as the next mechanisms; record the
  characterized partial.
- **NEGATIVE:** held-out stays ≈ 0.639 across the sweep ⇒ the gap is fundamental at this representation → the next
  mechanism is the structurally-different representation (Option 3 orthogonal-role TPR) and/or the deep binder (Option 2);
  the single-pass FRLF + fixed bind is the current resting point. **A NEGATIVE here does NOT close the arc** (per the
  owner's rule) — it directs to Options 2/3.

**Anti-cheats (mandatory — all already in the harness; this is the standard battery):**
1. **POINT-NEURON/ADDITIVE + LEARNED-LINEAR MUST FALL SHORT on the identical corpus/splits/seeds** (the headline A/B —
   a learned-cleanup GO only counts against the established 0.193/0.056 NEGATIVE on the same data).
2. **FIXED-±1 POSITIVE CONTROL CARRIES (0.989)** — proves the harness detects working bundling (a NEGATIVE is real, not
   a broken harness).
3. **HELD-OUT systematicity, leakage-asserted, vs the memorization-floor (lookup → 0.0) + chance (0.062)** — the bar is
   on held-out generalization, never raw recall (the exact confound that retracted the 2026-05-14 transitive-inference
   and the 2026-05-03 permuted-label results).
4. **PERMUTED-ROLE control** — shuffle role→filler → collapse to chance (the bind reads role structure, not a
   code-overlap artifact).
5. **LESION** — drop the multiplicative self-inverse → sum; the bundling lift collapses to the additive floor (proves
   the lift rides the bind op, not the cleanup or a leftover code property). FRLF already showed this collapses to 0.082.
6. **COMPOSITION-NOT-COHERENCE (the 2026-06-06 lesson)** — the metric is *unbind-recovers-the-right-filler*, not a
   decorrelation/reproducibility proxy; a noise-collapsed output fails it.
7. **DECORRELATED-vs-CORRELATED codes both reported** — run on the clean stream codes AND the grounded/correlated
   production codes (the §1.3 CODE-wall axis), so a GO is not a clean-code artifact (the F=3 resonator's exact failure
   mode, `2026-06-19-resonator-on-learned-codes-derisk.md`).
8. **PROVENANCE / BRAIN-BASED audit** — for any on-bridge promotion (Option 4), assert bind/unbind/cleanup are neuron
   firing + synaptic current; host code legitimate ONLY for presenting tokens + reading the final argmax. For Option
   2's BPTT arm, EXPLICITLY label it a host-side *characterization* (ceiling), not "the brain binding."
9. **NEVER WEAKEN THE NO-CONFAB MOAT** — the shuffled-fact zero-false-accept + abstention floor carry from V=320; the
   gate threshold is not tuned on the test. (A PLUS not a hard gate per owner 2026-06-17, but free here.)
10. **≥6 seeds (42/43/44/100/101/102), fractional ≥5/6; CPU/numpy for the cheap-first, CuPy for any decisive on-bridge
    promotion.**

---

## 5. LIKELIHOOD VERDICT (honest, per the "try the most-promising first" rule)

**Is FHRR-B plausibly closable?** Partially yes, and in a precisely-bounded sense:
- **The read-out / cleanup half is closable now** (Option 1, moderate confidence): a learned iterative cleanup + fixed
  self-inverse bind is the realistic-cortex form (lossy, redundant, learned read-out) and is the highest-probability
  cheap win. This genuinely reduces FHRR-B — it retires the host-designed *exact-inverse READ-OUT* and shrinks the
  residual to "the bind op is a fixed self-inverse *structure*."
- **The bind-FORM half (an end-to-end *learned* bundle inverse) is the deep, lower-probability frontier** (Options 2/3):
  the theorem (a sum has no exact inverse; only a self-inverse structure recovers a bundled element) + three shallow
  NEGATIVEs + the dendritic-multiplication NEGATIVE all say a from-scratch learned bundle inverse is fighting the
  algebra. The deep/hidden-layer learned binder (Option 2) is the one *untried* path with a real mechanism (capacity +
  credit-routing the shallow attempts lacked) and is where the approved dendrite legitimately re-enters — but it is
  weeks-scale and at-risk (the prior credit-assignment NEGATIVE). The orthogonal-role TPR (Option 3) most likely
  *dissolves the two-attribute capability boundary* but keeps a *designed* (orthogonal) binding structure, so it closes
  the capability gap more than the FORM shortcut.

**The disciplined honest statement under the owner's rule:** FHRR-B is NOT yet at a true terminus — there ARE untried
mechanisms (learned iterative cleanup; deep/hidden-layer learned binder with the dendrite re-entering; orthogonal-role
TPR), and the rule says to try them most-promising-first. **But it is equally honest that the bind op's fixed
self-inverse structure may be the *correct biology-grounded resting point* (coincidence-detection / dendritic-product
is a STRUCTURAL neural primitive, not a host computation), in which case "closing FHRR-B" means making the CODES and the
CLEANUP learned/spiking (mostly done + Option 1) while the *structural* bind primitive stays fixed — which is a
defensible BRAIN-BASED-ONLY close, distinct from the indefensible host-designed-exact-inverse-READ-OUT.** The next move
that respects the rule and is cheapest-first is **Option 1**, and its outcome routes to Option 2/3 if it does not reach
parity. We do not stop; we run Option 1, then escalate to the deep binder / TPR per the result.

**Likelihood-ordered try-list:** (1) FRLF capacity + learned-iterative-cleanup sweep [afternoon, moderate P] → (2a)
deep BPTT learned-binder ceiling [< 30 min/seed CPU, the gate] → (2b) e-prop / dendritic-MLP deep learned binder
[weeks, the brain-faithful target, dendrite re-entry] AND/OR (3) orthogonal-role TPR A/B [weeks, parallel, dissolves
the 2-attr boundary] → (4) on-bridge guarded wiring of whichever is GO [weeks, gated].

---

## 6. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file/finding/line):**
- The capability map (single-attr GO 0.806→0.833; bundling additive 0.193 / learned-linear 0.056; fixed-±1 0.989) —
  `2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`, read in full.
- The dendritic-multiplication A/B is RUN + NEGATIVE (0.168 < fixed 0.261) — `2026-06-19-dendritic-binding-toy-derisk.md`,
  read in full.
- The apical-basal credit-assignment toy is RUN + NEGATIVE (single-layer actor, nothing to align) —
  `2026-06-19-dendrite-credit-assignment-toy-stage1.md`, read in full.
- The FRLF cell is GO (0.639 numpy; on-bridge who/what 1.000) — `2026-06-18-learned-filler-fixed-bind-bundles-GO.md` +
  `2026-06-17-onbridge-learned-composer-step2-GO.md`.
- The CODE-wall (systematic on decorrelated, fails on correlated) — `2026-06-11-cortex-learned-binder-systematicity-
  NEGATIVE-ON-CORRELATED.md`; the prize delivered on the codes axis (PPMI 320, held-out 0.86) —
  `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`.
- The F=3 resonator degrades on correlated learned codes (100% clean → 29% learned) —
  `2026-06-19-resonator-on-learned-codes-derisk.md`.
- The dendrite substrate code is present + reusable — `sim/dendritic_neuron.py` (58 lines, read in full),
  `sim/dendritic_plasticity.py` (41 lines, read in full), `sim/dendritic_mlp.py` (196 lines, read in full),
  `sim/config.py:219` (`enable_graded_dendritic_plateau`) + `:260` (`enable_dendritic_divisive_gain`).
- The owner CYCLE-329 rule ("boundary is not an exit") — commit `286f8368`, the live HEAD.

**Could NOT fully verify (flagged honestly):**
1. **Whether Option 1's learned-iterative-cleanup sweep reaches parity** — the genuine open empirical question;
   deliberately not predicted. The FRLF base is 0.639 at low D_h with single-pass cleanup; capacity + learned iterative
   decomposition is the untested lever.
2. **Whether a DEEP learned binder (Option 2) generalizes the bundle inverse where the shallow ones failed** — untested;
   the capacity/credit-routing argument is a hypothesis, against the headwind of the shallow + dendritic NEGATIVEs.
3. **Whether the bilinear-gating paper's "zero-shot by multiplicative inductive bias" claim survives the bundle
   (superposition) case** — the paper is on motor-primitive interpolation, not a 3-way symbolic superposition unbind; its
   transfer to bundling is the unknown Option 2 would measure.

**No banking.** Reported exactly as the record stands — including that FHRR-B has already shrunk to one op (the bundle
inverse FORM), that the dendrite-as-binding-op is already NEGATIVE (correcting the task's "prime candidate" framing to a
specific deep/hidden-layer re-entry), that the strategic prize (generalization) is delivered elsewhere, and that the
honest reduction available now (learned codes + learned iterative cleanup + a fixed *structural* bind primitive) is a
defensible brain-based close of FHRR-B even if an end-to-end-learned bundle inverse remains the deep, lower-probability
frontier the owner's rule says to keep pushing.

## Sources

**Project record (all read in full this pass; file-cited above):** the learned-bind capability map, the dendritic
binding + credit-assignment NEGATIVEs, the FRLF GO, the CODE-wall, the PPMI prize, the F=3 resonator boundary, the
boundary ledger (`2026-06-20-boundary-ledger-dendritic-audit.md`), the FHRR frontier decision-prep
(`2026-06-20-fhrr-frontier-decision-scoping.md`), the dendrite-substrate unlock gate
(`2026-06-20-dendrite-substrate-unlock-deep-research.md`), the two prior dendritic-binding scopings (06-17, 06-18); the
dendrite substrate code (`sim/dendritic_{neuron,plasticity,mlp}.py`), the on-bridge primitives (`sim/kernels.py:253`,
`sim/config.py:219,260`), the harness + acceptance runners.

**Literature (verified this pass + cited in the prior findings):**
- VSA theory: bundling = similarity-preserving additive superposition with NO exact inverse; the self-inverse property
  recovers a bundled element (ACM Computing Surveys VSA survey Part I; MAP `a⊗a=1`).
- Frady & Sommer, *Robust computation with rhythmic spike patterns* (PNAS 2019) + Frady-Kleyko-Sommer resonator networks
  (arXiv 2208.12880) — iterative resonator decomposition (Option 1's learned-cleanup grounding; the on-substrate RF
  reference).
- Tensor-product / learnable-decomposition: AID *Attention-based Iterative Decomposition for TPR* (arXiv 2406.01012,
  2024); *Differentiable Tree Operations Promote Compositional Generalization* (arXiv 2306.00751); *RNNs implicitly
  implement TPR* (arXiv 1812.08718) — the literature's resolution = learn fillers + decomposition, keep a structured
  binding primitive (Options 1/3).
- Dendritic credit assignment needs DEEP/hierarchical nets: Sacramento, Costa, Bengio, Senn, *Dendritic cortical
  microcircuits approximate the backpropagation algorithm* (NeurIPS 2018, arXiv 1810.11393); Payeur, Naud, Richards et
  al., *Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits* (Nat Neurosci 2021) —
  Option 2's grounding (and why the shallow dendrite NEGATIVE does not transfer to a deep binder).
- Bilinear gating: *Bilinear gating of motor primitives* (arXiv 2606.10891, 2026) — two-compartment burst = soma×dendrite
  product, local three-factor rule, zero-shot by multiplicative inductive bias (Option 2's direct circuit).
- e-prop: Bellec et al., *A solution to the learning dilemma for recurrent networks of spiking neurons* (Nat Commun 11:3625,
  2020) — the brain-faithful local three-factor approximation to BPTT (Option 2b).
- Urbanczik & Senn, *Learning by the Dendritic Prediction of Somatic Spiking* (Neuron 2014); Guerguiev-Lillicrap-Richards,
  *Towards deep learning with segregated dendrites* (eLife 2017) — the built `sim/dendritic_plasticity.py` /
  `sim/dendritic_neuron.py` rules.
- Lake & Baroni, *Human-like systematic generalization through a meta-learning neural network* (Nature 2023) — learned
  binders are systematic only with deliberate structure (the multiplicative/TPR inductive bias IS that structure).

_Read-only scoping deliverable. NO code, NO experiments, NO GPU. Stayed on `main`. Per the owner's CYCLE-329 rule, this
does NOT recommend closing FHRR-B as an honest-negative; it ranks the untried next-mechanisms (learned iterative
cleanup first, then the deep/hidden-layer learned binder with the dendrite re-entering, then orthogonal-role TPR) and
the cheapest-first de-risk that decides the first one._
