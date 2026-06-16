# On-substrate LEARNED role-filler binder — deep-research scoping (2026-06-16)

**Status:** READ-ONLY deep-research + catalog/literature review (the project's standing "deep research FIRST at a new
direction / roadblock" opening move). No `sim/` code, no GPU, no build, no experiment run. The single deliverable is
this doc.
**Author role:** read-only computational-neuroscience research subagent. Every load-bearing project fact below is cited
to file + line; the literature claims are cited to paper.
**Question (verbatim from the tasking):** how do we realize a **LEARNED, systematically-generalizing role-filler
binder on the spiking substrate** (neurons + synapses on the project's `SimulationBridge`), to replace the fixed
exact-inverse vector-symbolic-algebra (VSA) bind the conversational pipeline currently uses?

---

## 0. Terms (defined once — no undefined acronyms)

- **bridge** — one `sim.bridge.SimulationBridge`: a network of simulated spiking neurons stepped by one
  `_run_one_simulation_step` loop. "The brain."
- **role / filler / bind / unbind** — a *role* is a slot (agent / action / patient / attribute / polarity); a *filler*
  is a concept word. **bind** combines (role, filler) → one composite vector; **unbind** recovers the filler given the
  composite + the role ("who is the agent?"). A fact "dog go north" is the vector **sum** of three bound pairs.
- **VSA (vector symbolic architecture)** — symbols as high-dim vectors, bound by an algebraic operation with a defined
  inverse so unbind is exact up to noise.
- **FHRR (Fourier Holographic Reduced Representation)** — the production VSA scheme; bind = element-wise complex
  product of phasors, unbind = multiply by the conjugate. On-substrate via `NeuronModel.RESONATE_AND_FIRE` + complex
  synapses (`rf_kick`, `rf_set_complex_weights`, `rf_read_phases`, `sim/bridge.py:5447–5556`).
- **the fixed bind we want to replace** — that exact-inverse FHRR (and the legacy ±1 Hadamard) algebra: spiking-
  realized but **hand-designed, not learned** (the documented "principled idealization" / "step 3"). The OPERATION is
  identical for every operand, which is *why* it is systematic for free — and exactly the property a learned binder
  must reproduce without that crutch.
- **`BilinearBinder`** — the de-risked numpy learned binder (`research/runners/cortex_learned_binder_systematicity_
  probe.py:277`). Bind `bound = tanh(role @ W_R + filler @ W_F + b)` (line 330); unbind = a 2-layer net keyed on the
  role (line 335). Adam-trained on a SUBSET of role-filler combos. **Crucial nuance, §1:** its bind is *additive* in
  role and filler, not a literal role×filler product.
- **systematicity (Fodor-Pylyshyn 1988)** — if the system binds "dog as agent" and "cat as agent" and "dog as
  patient," it should bind "cat as patient" without that specific combination ever being trained. A symbolic algebra
  has this for free; learned nets notoriously do not. The central risk.
- **stream-cortex codes** — concept codes the bridge LEARNS from the raw conversation stream by rate-Hebbian
  co-occurrence + population code, validated multi-seed at 64 and 320 concepts (`2026-06-15-on-bridge-hebbian-co-
  occurrence-learning-mechanism-GO.md`). Graded, real-valued, **moderately decorrelated** (between-code cos ≈ 0.05);
  semantic structure lives in the magnitudes.
- **BPTT** — backpropagation-through-time. **surrogate gradient** — replaces the non-differentiable spike with a
  smooth surrogate in the backward pass (`sim/surrogate_grad.py`, `sim/bptt_snn*.py`).
- **e-prop (Bellec 2020)** — the biologically-plausible, online, LOCAL approximation to BPTT for recurrent spiking
  nets: each weight update is a **three-factor** product of an **eligibility trace** (a local pre×post running
  product) and a per-neuron learning signal. The bridge for "BPTT learned binder" ↔ "three-factor/eligibility binder"
  (they are the same thing under e-prop).
- **two-compartment / Larkum-BAC neuron** — a pyramidal model with a basal (bottom-up) compartment and an apical
  (top-down) compartment; coincident basal+apical drive triggers a burst (a multiplicative AND). The project HAS this
  in numpy (`sim/dendritic_neuron.py`), NOT on the bridge.
- **three-factor / eligibility substrate (on the bridge)** — reward-modulated plasticity: STDP/Hebbian deposits an
  eligibility trace (`cp_eligibility_trace`, `sim/bridge.py:469,690`), decayed by `fused_eligibility_trace_decay`
  (`bridge.py:6673`), and converted to weight change by a global reward signal × `reward_learning_rate`
  (`bridge.py:6745–6784`). Per-pathway plasticity gates freeze/thaw it (`set_plasticity_gate`, `bridge.py:3033`).

---

## 1. DIAGNOSIS — what is actually hard about a *spiking* learned binder

### 1.1 The single most important clarification: the de-risked binder is NOT a literal multiplication

The tasking asks "is a multiplicative bind (role×filler) realizable locally, and how?" — but the binder that already
generalizes does not multiply role by filler. Read directly from `cortex_learned_binder_systematicity_probe.py:330`:

```
bind:   bound = tanh( role @ W_R + filler @ W_F + b )        # ADDITIVE in role & filler, then a pointwise tanh
unbind: filler_hat = [bound ; role @ W_RP] @ W_U + b        # role used as a KEY (concatenated), linear readout
```

The systematicity the §2026-06-16 finding measured (held-out 0.889 vs memorization-floor 0.000) comes from **learned
linear projections of role and filler into a shared hidden space, a pointwise saturating nonlinearity, and a learned
unbind that conditions on the role** — NOT from a role⊗filler outer product. This matters enormously for the spiking
realization, because it means **the hard, "is multiplication local?" question is OPTIONAL, not mandatory.** Two
distinct realizations follow, and the diagnosis must keep them separate:

- **(Form A — additive/projection bind, the literal de-risked form).** Everything in the `BilinearBinder` is a
  matrix-vector product (a synaptic projection: pre-rate vector @ weight matrix) plus a pointwise tanh/saturation (a
  neuron's f-I curve) plus a concatenation (two populations feeding one). **Every primitive here already exists on a
  point-neuron bridge.** There is no role×filler product anywhere. So the *literal* de-risked binder is realizable on
  point neurons; the open question is only whether it stays systematic when the smooth tanh is replaced by spikes and
  the exact gradient by a spiking-compatible learning rule.

- **(Form B — true multiplicative/conjunctive bind).** A genuinely *bilinear* bind (the role GATES the filler:
  output = Σ role_k · filler_k, or a population tuned to role-AND-filler conjunctions) is the more cortex-like and
  more obviously-systematic form (it inherits the algebra's operand-independence). This DOES need a multiplication,
  and on a point neuron a product of two firing rates is not a primitive (a point neuron sums currents; it cannot
  multiply two inputs without a nonlinearity that approximates a product). This is where the substrate limit bites —
  and where the dendritic option earns its place (§2, Option i).

**The honest framing:** the project does NOT have to solve multiplication to ship a learned binder, because the
de-risked binder is additive. Multiplication is the route to the *stronger* (more systematic, more cortex-faithful)
binder, not a prerequisite for the *first* one.

### 1.2 Why "spiking" is harder than the numpy `BilinearBinder` — three concrete gaps

The numpy binder works because it has three things the spiking substrate makes expensive or impossible:

1. **An exact, global gradient.** Adam backprops MSE through the unbind, through the concat, through the tanh, into
   `W_R`/`W_F`. On the spiking substrate, the spike is non-differentiable (Heaviside), so the gradient must be either
   (a) a *surrogate* gradient through an unrolled BPTT graph (`sim/bptt_snn*`), which is a host-side training shortcut
   — legitimate as a *characterization* but not "the brain doing it"; or (b) a **local three-factor approximation**
   (e-prop) that the bridge's eligibility substrate can compute, which is brain-faithful but is an *approximation* to
   the gradient and may lose systematicity. **This is the central open question, not multiplication.**

2. **Smooth real-valued activations.** `tanh` is a graded, signed, exactly-differentiable transfer. A spiking neuron's
   output is a rate (a count of binary events) over a window: positive-only, noisy, quantized. The project's own
   **single-neuron rate-code SNR wall** (`2026-06-15-...GO.md`, "single-neuron read-out plateaus at ~50% of host")
   says one neuron's rate is too noisy to carry a graded code — the **population code** (≈8–32 neurons/dimension
   recovers 94–108% of host, same finding) is the established fix. So the spiking binder must be a *population*
   binder: each "dimension" of the bound vector is a pool, not a neuron. That multiplies the neuron count but is a
   solved pattern.

3. **A clean, ZCA-conditioned code.** The numpy binder was de-risked on the stream-cortex codes at cos ≈ 0.05 (the
   sweet-spot regime). The **Mikulasch-Priesemann point-neuron limit** (PubMed 34876505; already in CLAUDE.md as the
   "decorrelation is analog/pre-spike" reframe) says a point-neuron pairwise-Hebbian rule provably cannot *discover*
   decorrelation on correlated codes. **This is NOT a blocker for the binder, because the codes are ALREADY
   decorrelated by the rate-Hebbian + population stream cortex (cos 0.05, structural sparse-expansion-like).** The
   binder consumes decorrelated codes; it does not have to produce them. The decorrelation problem is upstream and
   already solved (§ the stream-cortex GO).

### 1.3 Is the dendritic-arc NEGATIVE a blocker? — NO, and this is load-bearing

The project's dendritic arc reached an honest NEGATIVE (`2026-05-17-dendritic-credit-assignment-NEGATIVE.md`): the
local Urbanczik-Senn rule with fixed-random feedback did **not** do **hidden credit assignment** in a **W2-frozen
isolation** test at feasible local scale. It is tempting to read this as "local dendritic learning can't train a
binder." **That reading is wrong, for a precise structural reason:**

- The dendritic NEGATIVE was about **deep credit assignment** — training a HIDDEN layer (W1) through a FROZEN output
  layer (W2), so the only way loss can drop is if the local rule itself solves the multi-layer credit-assignment
  problem. That is the hard case (and GLR-2017/Sacramento-Senn show it needs scale + machinery the cheap slice can't
  afford).
- **A role-filler binder is not that problem.** The `BilinearBinder`'s bind is **one projection layer** (`W_R`, `W_F`)
  and the unbind is **one readout layer** (`W_U`). There is no deep stack of hidden layers needing credit routed
  through frozen downstream weights. The bind's hidden layer is trained by a teaching signal at the *unbind output*
  that is **one synaptic hop away** — exactly the regime where a local three-factor rule with feedback works (it is
  the shallow case the dendritic NEGATIVE explicitly did NOT cover; the both-layers regime "trained both layers" and
  the isolation test froze the output). So the dendritic NEGATIVE **does not transfer** to the binder; the binder is
  the shallow, teaching-signal-adjacent case.

**Conclusion of diagnosis.** The hard part of a spiking learned binder is **the learning rule, not the architecture
and not multiplication**: can a spiking-compatible rule (surrogate-BPTT, or — the brain-faithful target — e-prop /
three-factor eligibility) train the bind+unbind so that it **stays systematic** (held-out ≈ train, ≫ memorization
floor)? The architecture is reachable (additive form = pure projections + population codes; multiplicative form =
two-compartment burst-AND, which the project has in numpy). The codes are already decorrelated upstream. The
dendritic-credit NEGATIVE is about a deeper problem and does not apply. A multiplicative bind IS realizable locally —
on a two-compartment neuron via the basal×apical burst coincidence (§2 Option i, grounded in the bilinear-gating
paper below) — but is not required for the first, additive binder.

---

## 2. RANKED, BIOLOGICALLY-GROUNDED OPTIONS for the on-substrate learned binder

Organized by **how much systematicity each puts at risk** and **how brain-faithful the learning rule is** (the
BRAIN-BASED-ONLY standard: the binding must be neurons + synapses + their communication, not host bookkeeping). Each
option below is for the *learned binder itself*; all of them sit downstream of the already-decorrelated stream codes
and upstream of the already-solved localist NEF cleanup (`2026-06-11-cortex-core-learned-binder-research.md` §1.3) and
the already-de-risked learned familiarity gate / no-confab moat (same doc §1.4). Build those shared pieces once.

### Option (i) — RECOMMEND: a TWO-COMPARTMENT BILINEAR-GATE binder trained by a LOCAL THREE-FACTOR rule

- **Mechanism.** Realize the bind as a **bilinear gate on two-compartment (Larkum-BAC) neurons**: the **basal**
  compartment carries the filler projection, the **apical** compartment carries the role projection, and the neuron's
  **burst probability is the product** of the two compartments' activations — a native, hardware-level multiplication
  by dendritic coincidence. This is **Form B** (§1.1): a genuinely multiplicative, conjunctive bind. The bound
  representation is the population burst pattern; unbind drives the same population with the role on the apical
  compartment and reads the basal-recoverable filler.
- **Biology citation — this is the strongest grounding I found.** *"Bilinear gating of motor primitives: a principle
  linking dendritic computation to rapid goal-directed adaptation"* (arXiv 2606.10891, 2026) gives this exact circuit:
  a population of two-compartment L5 pyramidal neurons computes `p_burst = ⟨ p_s(V_soma) · p_d(V_dend) ⟩` — the product
  of a state/filler signal and a goal/role signal — instantiating the bilinear decomposition `μ(s,g) = Σ_k G_k(g)
  Y_k(s)`. Catalog grounding: **G.02 active dendrites / NMDA plateau / Larkum two-layer apical-basal coincidence**
  (catalog line 2644–2652: "gain modulation by apical-basal coincidence (Larkum's two-layer model)"); **J.08 NMDA
  voltage-dependent coincidence detector** (line 3593) is the molecular substrate of the AND.
- **How it learns — LOCAL three-factor, NO backprop, NO weight transport.** The bilinear-gating paper trains the
  apical (goal/role) weights by a **three-factor Hebbian rule** `Δw_j ∝ r(t) · x_j(t) · ∂μ/∂y(t)`, "entirely local:
  each synapse updates using only presynaptic input, the neuron's own burst probability, and a global reward — no
  backpropagation across the network." This maps **directly onto the bridge's existing eligibility substrate**:
  `x_j(t)` = the eligibility trace (pre×post), `r(t)` = the global reward signal, `∂μ/∂y` = a function of the neuron's
  own membrane state. The project already has all three (`cp_eligibility_trace`, the reward modulation block,
  `reward_learning_rate`; `bridge.py:6745–6784`).
- **Does it generalize systematically?** The paper's headline is **yes, by construction**: "the multiplicative
  structure itself provides the inductive bias for robust out-of-distribution generalization … zero-shot
  generalization to unseen directions through linear interpolation within the learned goal-vector space." This is the
  same reason the fixed algebra is systematic — multiplication is operand-independent — but here the gate weights are
  *learned*, not hand-set. So it keeps the algebra's systematicity GIFT while making the binding learned. **This is the
  best of both worlds and the reason it is ranked first.**
- **Risk.** (a) The two-compartment neuron is **NOT on the bridge** — it is numpy-only (`sim/dendritic_neuron.py`).
  Putting it on the bridge is a *protected* `NeuronModel` edit (catalog G.02 estimates ~10× compute/neuron;
  CLAUDE.md/the dendritic findings already scoped a protected two-compartment `NeuronModel` as the "Phase 1" after the
  numpy gates). That is the real cost, and the owner has flagged it as a months-scale arc. (b) The dendritic-credit
  NEGATIVE shows the project's local rule didn't do *hidden* credit assignment — but here the gate is trained by a
  *reward/teaching signal one hop away* (the unbind output), the shallow regime that NEGATIVE did not cover (§1.3), so
  this is a genuinely different and easier ask. (c) Mitigation: the entire de-risk can be done in **numpy first**
  (the project's `DendriticLayer` + `urbanczik_senn_update` + a three-factor reward term) — no protected edit until a
  numpy systematicity PASS.

### Option (ii) — a small SURROGATE-GRADIENT / BPTT SNN binder (the spiking analogue of the `BilinearBinder`)

- **Mechanism.** Build the additive **Form A** binder (§1.1) as a 2-layer spiking net: a bind layer (LIF population
  fed `role` and `filler` population codes through `W_R`, `W_F`) and an unbind layer (LIF population fed the bound
  population + role). Train end-to-end by surrogate-gradient BPTT — the literal spiking translation of the numpy net.
- **Biology citation.** Surrogate-gradient SNN training (Neftci-Mostafa-Zenke 2019, arXiv 1901.09948; "remarkable
  robustness" Zenke-Vogels 2021, Neural Computation 33(4)). Catalog: this is the project's own `sim/bptt_snn*` +
  `sim/surrogate_grad.py` stack (confirmed on `main`: `git ls-tree main sim/` → `bptt_snn.py`, `bptt_snn_gpu.py`,
  `surrogate_grad.py`; their "path-f-hybrid only" headers are stale).
- **How it learns.** Surrogate-gradient BPTT — a **host-side training shortcut** (the backward unroll is not something
  the brain does). Per the BRAIN-BASED-ONLY standard this is **characterization, not the deliverable**: it answers
  "does ANY spiking binder of this form stay systematic?" If even the BPTT spiking binder fails systematicity, no
  local rule will; if it passes, it bounds the achievable and provides a teacher for the local rule. It is the
  spiking analogue of the numpy de-risk — the natural NEXT cheap step.
- **Does it generalize systematically?** This is exactly what the de-risk measures and is unknown for the spiking
  case. The numpy net got 0.889 held-out; whether spikes + surrogate gradient preserve that is the open question.
  Form A (additive) has *less* built-in systematicity than Form B (multiplicative) — it relies on the learned
  projections, not on operand-independent multiplication — so this is the more at-risk form, which is precisely why
  measuring it is informative.
- **Risk.** (a) BPTT is a host shortcut → it is a *characterization*, and a PASS here is not "the brain binds," it is
  "a spiking binder of this form can be systematic." (b) Surrogate-gradient training of resonate-and-fire / LIF over
  the bind window can be unstable — but **Balanced Resonate-and-Fire neurons** (BRF, arXiv 2402.14603, 2024) "provide
  much faster and more stable training convergence … bridging hundreds of timesteps during BPTT," a known fix if
  stability bites. (c) Cost: small (toy R×F grid, CPU/numpy-backend BPTT), so it is cheap to run before committing.

### Option (iii) — a THREE-FACTOR / e-prop binder on the bridge's existing eligibility substrate (the brain-faithful target)

- **Mechanism.** Same additive **Form A** architecture as (ii), but trained **online and locally** by **e-prop**
  (Bellec 2020): each synapse keeps an eligibility trace (pre×post local product), and a per-population learning
  signal (the unbind error projected back through a FIXED random feedback matrix — feedback alignment, no weight
  transport) gates the trace into a weight change. This is the project's existing reward-modulated three-factor block,
  generalized from a scalar reward to a per-population teaching signal.
- **Biology citation.** Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, *"A solution to the learning
  dilemma for recurrent networks of spiking neurons,"* Nature Communications 11, 3625 (2020) — e-prop is the
  data-inspired biologically-plausible local approximation to BPTT, explicitly "fits the framework of three-factor
  learning rules" for spiking neurons. Catalog: **J cluster** (synapses & plasticity, eligibility traces); the
  project's `compose_temporal_bind.py` already routes credit through the REUSED `fused_eligibility_trace_decay`,
  proving the eligibility substrate trains compositional A→B bindings across a temporal gap (its validated mechanism).
- **How it learns.** Local three-factor with eligibility traces — **brain-faithful** (this is the genuine
  "neurons + synapses + their communication" deliverable, not a host shortcut). It is also the SAME rule family as
  Option (i)'s three-factor Hebbian gate — Options (i) and (iii) differ only in whether the bind is multiplicative
  (two-compartment, Form B) or additive (point-neuron, Form A).
- **Does it generalize systematically?** Unknown — e-prop is an *approximation* to the gradient, so it may lose some
  of the numpy binder's systematicity. Measuring the gap (e-prop held-out vs BPTT held-out vs numpy 0.889) is itself a
  clean scientific result (maps how much systematicity the local approximation costs).
- **Risk.** (a) e-prop is the most engineering to stand up (per-population learning signals + feedback-alignment
  matrices on the bridge), though much of the substrate exists. (b) The systematicity-loss risk is real and is the
  honest negative this option might surface. (c) Recommend it AFTER (ii) bounds the achievable — train (iii) toward
  the (ii) BPTT ceiling.

### Option (iv) — FAST-WEIGHTS / Hebbian temporary binding

- **Mechanism.** Bind by a transient, rapidly-written associative weight (a "fast weight" overlaid on slow weights):
  a fact (role, filler) is stored by a one-shot Hebbian outer-product into a short-lived weight matrix; unbind reads
  it back by driving the role. This is the classic Hebbian-temporary-binding / fast-weights idea (Hinton-Plaut 1987;
  Schmidhuber 1992; Ba et al. 2016 "Using Fast Weights to Attend to the Recent Past").
- **Biology citation.** Short-term synaptic plasticity (the project's STP: `stp_U`, `stp_tau_d`, `stp_tau_f`) +
  gamma-binding-by-synchrony (**catalog N.19**, line 1028: "neurons firing within the same gamma cycle are
  co-grouped"; STDP-window matched to the gamma cycle so co-bound assemblies become eligible for storage). The engram
  API (**catalog D.14**, the Tonegawa tag, SHIPPED on the bridge) is the project's codebook-free "store the pattern
  that fired" mechanism.
- **How it learns.** The *binding* is one-shot Hebbian (not gradient-learned), so this is more a learned *memory* than
  a learned *bind transform*. It is genuinely brain-based and cheap.
- **Does it generalize systematically?** **This is the weak point.** Fast-weights binding MEMORIZES the specific
  (role, filler) pairs it writes — it does not learn a *binding relation* that transfers to novel combinations. It is
  exactly the **memorization floor** the systematicity probe is built to catch. So it is NOT a systematic binder; it is
  the project's existing engram/heteroassociative store under another name. Useful as a fallback store and as the
  negative-control floor, NOT as the systematic-binder answer.
- **Risk.** Fails systematicity by design (it is the memorization baseline). Keep it as the no-confab store / negative
  control, not the binder.

### Option (v) — a LEARNED CODEBOOK + the existing FHRR ops (the hybrid; this is the documented Option A)

- **Mechanism.** Do NOT learn the bind; keep the exact-inverse FHRR phasor product, but feed it the **learned
  stream-cortex codes** (the decorrelated codes the bridge already learns) instead of curated codes. Cleanup =
  localist NEF; abstention = learned familiarity gate.
- **Biology / status.** This is the already-recommended Option A from `2026-06-11-cortex-core-learned-binder-
  research.md` §4, validated to V=320 (the flat-distinct composition). It removes the clean-code demand (I-2) and the
  host cleanup/abstention (I-3) but **keeps the hand-designed bind operation (I-1)** — so it is NOT a learned binder.
- **Why it is listed here.** It is the honest *lowest-risk* shippable, and it is the correct fallback if (i)–(iii)
  surface a systematicity NEGATIVE: a learned-codes + fixed-op + computed-cleanup/abstention composer is brain-based
  in everything EXCEPT the bind operation, and it works. But it does not answer the tasking's question (a *learned*
  bind); it is the safety net, ranked last among the binder options precisely because the bind itself is not learned.

### Ranking summary

| | Form | Bind learned? | Learning rule | Brain-faithful rule? | Systematicity by construction | On bridge today? | Cost | Rank |
|---|---|---|---|---|---|---|---|---|
| **(i) two-compartment bilinear gate** | multiplicative (B) | **yes (gate weights)** | local 3-factor Hebbian | **yes** | **yes (multiplication is operand-independent)** | no (numpy; protected edit later) | high (protected `NeuronModel`) — but numpy-first | **1 (target)** |
| **(ii) surrogate-BPTT SNN** | additive (A) | yes | surrogate-grad BPTT | no (host shortcut → characterization) | partial (learned projections) | yes (`bptt_snn*`) | **low** | **2 (cheapest next probe)** |
| **(iii) e-prop / 3-factor** | additive (A) | yes | local eligibility 3-factor | **yes** | partial | mostly (eligibility substrate) | moderate | 3 |
| (iv) fast-weights / Hebbian | one-shot | no (memorizes) | one-shot Hebbian | yes | **NO (this is the memorization floor)** | yes (STP/engram) | low | 4 (control/fallback store) |
| (v) learned codes + fixed FHRR | n/a (op fixed) | **no** | n/a | yes except the op | yes (algebra's) | yes | low | 5 (safety net, not a learned bind) |

**Recommendation.** The TARGET is **Option (i)** — a two-compartment bilinear-gate binder trained by a local
three-factor rule — because it is the only option that is simultaneously (a) genuinely learned, (b) brain-faithful in
its learning rule, and (c) systematic by construction (the multiplicative inductive bias), with a direct 2026 biology
paper (bilinear-gating) handing the project the exact circuit + rule. But the CHEAPEST-FIRST de-risk is **Option (ii)
in numpy/tiny-bridge first**, because it directly extends the already-passing numpy `BilinearBinder` probe to spikes
with the smallest possible change, and a PASS/FAIL there gates the entire arc before any protected `NeuronModel` edit.
Options (i) and (iii) share the same local three-factor rule, so the numpy three-factor de-risk (below) simultaneously
de-risks both.

---

## 3. REUSABLE PROJECT MACHINERY (concrete file/function pointers — all verified present)

- **The de-risked numpy binder + its full systematicity harness** — `research/runners/cortex_learned_binder_
  systematicity_probe.py`: `BilinearBinder` (line 277), the leakage-free `make_systematicity_splits` (line 465), and
  ALL FOUR anti-cheats coded (`score_shuffled_label` 562, `score_memorization_floor` 583, `score_abstention` 629,
  the leakage assert 725). **The spiking de-risk should subclass/extend this so the splits + anti-cheats are byte-
  identical** — only the binder's internals change (numpy tanh → spiking population), so the comparison numpy-0.889
  vs spiking-held-out is apples-to-apples.
- **The two-compartment neuron + local plasticity (numpy, for Option i / iii first pass)** — `sim/dendritic_neuron.py`
  (`DendriticLayer`: basal forward, FIXED-random apical feedback, BAC threshold-lowering) + `sim/dendritic_plasticity.py`
  (`urbanczik_senn_update`: local apical-gated mismatch, NO weight transport, NO autodiff) + `sim/dendritic_mlp.py`
  (the assembled feedback-alignment MLP with a GPU-backed option). These give the bilinear-gate circuit + a local rule
  in numpy WITHOUT any bridge edit — the cheapest realization of Options (i)/(iii) for the systematicity gate.
- **The surrogate-gradient BPTT stack (Option ii)** — `sim/bptt_snn.py` (numpy reference: `LIFLayer`, `forward_unroll`,
  `backward_unroll` with ATan surrogate + hard-reset recurrent chain rule), `sim/bptt_snn_gpu.py` (CuPy, validated ==
  numpy to fp32), `sim/surrogate_grad.py` (ATan + fast-sigmoid surrogates). **All confirmed on `main`.** The 2-layer
  bind/unbind net drops straight onto these.
- **The on-bridge three-factor / eligibility substrate (Option iii)** — `cp_eligibility_trace` alloc
  (`sim/bridge.py:469,690`), STDP deposit into the trace (`:6627`), `fused_eligibility_trace_decay` (`:6673`), reward
  → weight update (`:6745–6784`), per-pathway `set_plasticity_gate` (`:3033`) / `set_transmission_gate` (`:3059`).
  `research/runners/compose_temporal_bind.py` already trains a compositional A→B binding through this exact eligibility
  kernel across a temporal gap — the closest existing precedent for an eligibility-trained bind.
- **The population-code lift (mandatory for the spiking binder per §1.2)** — the rate-Hebbian + population-code
  machinery in `research/runners/_phaseB_stdp_cooccurrence_derisk.py` (`--n-per`); 8–32 neurons/dimension recovers
  94–108% of host (`2026-06-15-...GO.md`). Each bound-vector "dimension" must be a pool, not a neuron.
- **The decorrelated learned codes (upstream input, already solved)** — `2026-06-15-on-bridge-hebbian-co-occurrence-
  learning-mechanism-GO.md`; stream codes cached `research/findings/raw/..._phaseB_stream_codes_320_seed42.npy`. The
  binder consumes these; it does not have to produce or decorrelate them.
- **The RF/FHRR phasor substrate (for Option v / the fixed-op fallback + the reference ceiling)** —
  `NeuronModel.RESONATE_AND_FIRE` + `rf_kick`/`rf_set_complex_weights`/`rf_read_phases` (`sim/bridge.py:5447–5556`);
  the masked-RF-ops edit (`rf_kick(neuron_mask=)`, owner-approved, default-off byte-identical) lets a binder co-reside.
- **The shared downstream pieces (build once, all options need them)** — the localist NEF/TPAM cleanup
  (`core_sim_composition.NEF_CLEANUP_OP`, `rf_phasor_composer._spiking_cleanup` line 202 — SOLVED, == numpy at 320) and
  the learned Bogacz-Brown familiarity gate / no-confab moat (de-risked +0.982 margin, `cortex_learned_cleanup_
  derisk.py` TEST 3; cleaner than the host threshold per the 320 multi-seed GO).
- **The capability spec + acceptance harness** — `research/runners/vocab_ceiling_probe.py` (full matrix, abstention
  floor + shuffled-fact control per cell, V=320, 6 seeds). The learned binder must pass THIS verbatim to ship.
- **The composer interface (drop-in target)** — `BrainConversationalAgent` / `BridgeParser`
  (`research/runners/brain_conversational_agent.py`) delegate to a composer object (`store`, `query_agent`,
  `query_patient`, `ask_yes_no`, `render_fact`, `elaborate`). A learned binder implementing this interface is a
  drop-in — step 3 is "write a new composer class," not "rewire the agent."

---

## 4. RECOMMENDED CHEAP-FIRST DE-RISK (run this BEFORE any build) + expected wall-clock

**The single cheapest experiment that falsifies/validates the most-promising option before any protected edit:**

> **Extend the EXISTING numpy systematicity probe (`cortex_learned_binder_systematicity_probe.py`) with a SPIKING
> binder variant and re-run the IDENTICAL splits + anti-cheats on the stream-cortex codes — measuring whether the
> spiking binder stays systematic (held-out vs train vs memorization floor) relative to the numpy 0.889.**

Run TWO spiking binders in the same harness, cheapest first:

1. **(ii) surrogate-BPTT spiking binder (Form A).** Replace the numpy `BilinearBinder._bind`/`_unbind` with a 2-layer
   spiking population net (LIF, population code per dimension via `--n-per`), trained by the project's
   `sim/bptt_snn.forward_unroll`/`backward_unroll` (numpy backend, CPU). This is the *minimal* spiking translation of
   the de-risked binder. **This first** because it bounds the achievable and is the smallest change.

2. **(i)/(iii) local three-factor binder (Form A point-neuron AND Form B two-compartment).** In the same harness, train
   the bind+unbind by a **local three-factor rule** — for Form B, instantiate the bilinear gate with the project's
   `sim/dendritic_neuron.DendriticLayer` (basal=filler, apical=role) + `urbanczik_senn_update` plus a reward/teaching
   term; for Form A, a point-neuron eligibility-trace rule. Compare held-out systematicity to the BPTT ceiling and to
   numpy-0.889. **This is the brain-faithful de-risk; do it second, targeting the BPTT ceiling.**

**The exact metric (verbatim from the existing protocol — do not invent a new one):** on the leakage-free R×F splits
(every role + every filler appears in train; the specific held-out (role, filler) pairings never appear in train),
report **`held_out_acc` (the systematicity number) vs `train_acc` vs the lookup-table `memorization_floor` held-out vs
chance (1/F)**, multi-seed (42/43/44), with the FHRR exact-inverse as the systematic-by-construction reference ceiling
(1.000). **GO gate:** spiking `held_out_acc` ≫ memorization floor (≈ 0) and ≫ chance, with the shuffled-label control
collapsing — i.e. the spiking binder reproduces the numpy binder's systematicity (numpy got 0.889; a spiking GO is
"clearly above floor and chance, ideally within a stated margin of 0.889," not necessarily equal). **NO-GO / honest
negative:** spiking held-out ≈ chance/floor while train is high → the spiking realization memorizes (maps exactly
where a spiking learned binder stops being systematic — itself the scientific deliverable).

**Why this is the right cheapest-first move.** It (a) reuses the entire validated harness (splits + 4 anti-cheats) so
the result is directly comparable to the numpy 0.889; (b) runs on CPU / `SIM_BACKEND=numpy` in **minutes per seed**
(the BPTT toy is a 2-layer net over a small R×F grid, T≈30; the numpy probe already runs in seconds–minutes — the
spiking version with population codes is a small constant factor more, **est. ≈ 5–20 min total for both binders ×
3 seeds**); (c) makes ZERO `sim/` edits and NO protected `NeuronModel` change (the two-compartment is used in its
existing numpy `DendriticLayer` form); and (d) gates the entire arc: if even the surrogate-BPTT spiking binder
(Option ii, the host-shortcut characterization) fails systematicity, no local rule (i/iii) will, and the honest
negative is reported without spending GPU. If (ii) passes and (i)/(iii) approach it, the protected two-compartment
on-bridge build (Option i, Phase 1) is warranted, and the next step is `vocab_ceiling_probe` verbatim at V=320.

**Expected wall-clock:** the numpy `BilinearBinder` probe (3 seeds × 2 F-values × 3 splits × 800 epochs) already
completes in well under an hour on CPU. The spiking extensions add a forward-unroll + surrogate-backward per training
step (a small constant factor) and population codes (×8 neurons/dim); **estimate < 30 min total on CPU for both
spiking binders across 3 seeds at F=8/16.** No GPU required for the de-risk; GPU is only for the eventual V=320
`vocab_ceiling_probe` integration after a GO.

---

## 5. ANTI-CHEAT CONTROLS the de-risk needs

The existing systematicity protocol already carries FOUR controls; **all four apply unchanged**, plus two additions
specific to the spiking realization:

1. **Leakage assert (existing, mandatory).** Programmatically verify NO held-out (role, filler) combo is in the train
   set; confirm every role and every filler is covered in train (`cortex_learned_binder_systematicity_probe.py:725`).
   Guards against the binder seeing the test combination during training.
2. **Shuffled-held-out-label control (existing).** Score held-out predictions against SHUFFLED true fillers → must
   collapse to chance (`score_shuffled_label`, line 562). Confirms held-out accuracy is real signal, not a readout
   artifact.
3. **Memorization floor (existing — the load-bearing one).** A pure lookup table scores chance on held-out
   (`score_memorization_floor`, line 583). The spiking binder beating this floor is THE systematicity signal; a
   spiking binder that only matches the floor is memorizing (and that is the honest negative to report).
4. **No-confab / abstention floor (existing).** Confidence gap between seen fillers and a never-bound novel filler
   (`score_abstention`, line 629). On the spiking binder, this becomes the familiarity-gate readout (the moat
   replacement); require max-known confidence < min-novel separated cleanly.

**Additional controls the SPIKING realization needs (new):**

5. **Lesion control (new, brain-based-audit).** Zero the learned bind/unbind weights (or, for Option i, the apical
   gate weights) → held-out accuracy must collapse to chance. Confirms the systematicity rides the LEARNED synaptic
   weights, not leftover structure, the population-code geometry, or a host path. (This is the lesion anti-cheat the
   familiarity-gate de-risk already used: zero the weights → margin → 0.)
6. **Provenance / brain-based audit (new, per the BRAIN-BASED-ONLY standard).** Assert that bind, unbind, and the
   plasticity update are neuron firing + synaptic current + the eligibility/three-factor rule on the substrate — host
   code legitimate ONLY for presenting the (role, filler) tokens and reading the final argmax. For Option (ii)
   (surrogate-BPTT) the audit must EXPLICITLY label the BPTT training as a host-side *characterization*, not "the
   brain binding" — the brain-faithful claim attaches only to Option (i)/(iii)'s local three-factor rule. Guards
   against a covert host-side gradient or a numpy fast-path doing the binding the substrate is credited with.

**Two specific failure modes these controls guard against, called out explicitly:**
- **The binder secretly memorizing** — caught by controls 1 (leakage) + 3 (memorization floor) + 5 (lesion). A spiking
  binder that quietly indexes training pairs scores at the memorization floor and collapses under lesion.
- **The codes leaking the answer** — caught by the unit check (between-code cos ≈ 0.05, not ≈ 1.0; the §1.2 caveat:
  read sparse codes in native binary/mean-removed form, NEVER median-bipolarized — that manufactures a common mode and
  produces a FALSE result) + control 2 (shuffled label collapses to chance). Guards against the population code or the
  role codes carrying the filler identity directly.

---

## 6. HONEST FRAMING / where each mechanism is speculative vs grounded

- **GROUNDED:** the additive binder is realizable on point neurons (it is pure projections + tanh + concat + population
  codes — all existing primitives); the two-compartment bilinear gate + its local three-factor rule is **directly
  grounded** in a 2026 paper (bilinear-gating, arXiv 2606.10891) AND the project's existing numpy `DendriticLayer` +
  `urbanczik_senn_update`; e-prop as the local-BPTT bridge is grounded (Bellec 2020, Nature Comms); the codes are
  already decorrelated upstream (stream-cortex GO); the cleanup + moat are solved/de-risked; the numpy binder's
  systematicity (0.889) is measured.
- **SPECULATIVE (the live risks):** (a) whether a SPIKING binder — surrogate-BPTT or local three-factor — *preserves*
  the numpy binder's 0.889 systematicity is **unmeasured**; the rate-code SNR wall + the local-rule's gradient
  approximation could erode it. (b) The two-compartment neuron is NOT on the bridge; Option (i) eventually needs a
  protected `NeuronModel` edit (~10× compute/neuron, catalog G.02), a months-scale arc the owner has flagged. (c) The
  dendritic-credit NEGATIVE is about a *deeper* (hidden-credit-assignment) problem and I argue (§1.3) it does NOT
  transfer to the shallow teaching-adjacent binder — but that argument is itself a hypothesis the de-risk tests; if
  the local three-factor binder ALSO fails systematicity in isolation, that would be a meaningful (and honest)
  extension of the dendritic boundary to the binding case. (d) Per the systematicity literature (Lake-Baroni Nature
  2023 vs the contested arXiv 2506.01820, 2025), learned binders are systematic only with deliberate structure — the
  multiplicative inductive bias (Option i) IS that structure, which is why it is the target, but a vanilla additive
  spiking readout (Option ii) should NOT be assumed systematic; it must be measured.

---

## SUMMARY (the 6–8 lines requested)

**Top recommended option:** a **two-compartment bilinear-gate binder trained by a LOCAL three-factor rule** (Option i)
— the basal compartment carries the filler, the apical the role, and the neuron's burst probability is their PRODUCT
(a native dendritic multiplication), with the gate weights learned by a local reward-gated Hebbian rule. It is the
only option that is simultaneously genuinely-learned, brain-faithful in its rule, AND systematic by construction (the
multiplicative inductive bias), and it is handed to the project almost turnkey by a 2026 paper ("Bilinear gating of
motor primitives," arXiv 2606.10891) plus the project's EXISTING numpy two-compartment neuron (`sim/dendritic_neuron.py`)
+ local plasticity (`sim/dendritic_plasticity.py`) + the bridge's eligibility/three-factor substrate.

**Crucial clarification:** the already-de-risked numpy `BilinearBinder` does NOT multiply role×filler — its bind is
ADDITIVE (`tanh(role@W_R + filler@W_F)`), so the hard "is multiplication local?" question is OPTIONAL; the literal
de-risked binder is realizable on point neurons. Multiplication is the route to the STRONGER (more systematic) binder
(Option i), not a prerequisite for the first one.

**Cheap-first de-risk:** extend the EXISTING systematicity probe (`cortex_learned_binder_systematicity_probe.py`) with
a SPIKING binder — first the surrogate-BPTT version (Option ii, the cheapest, using `sim/bptt_snn*`), then the local
three-factor version (Options i/iii) — and re-run the IDENTICAL leakage-free splits + 4 anti-cheats on the stream
codes, measuring spiking **held-out vs train vs memorization-floor** against the numpy 0.889. CPU/numpy, NO `sim/`
edits, NO protected `NeuronModel` change, **est. < 30 min for both binders × 3 seeds.** It gates the whole arc before
any build.

**Single biggest risk:** that the SPIKING realization (surrogate-BPTT or local three-factor) **loses the numpy binder's
systematicity** — the rate-code SNR noise and the local rule's gradient approximation could push held-out accuracy
down toward the memorization floor, in which case the honest negative (mapping exactly where a spiking learned binder
stops being systematic) is the deliverable, and the documented fallback is Option v (learned codes + the fixed FHRR op,
already validated to V=320 — brain-based in everything except the bind operation itself).

## Sources (literature consulted beyond the in-repo catalog/code)

- "Bilinear gating of motor primitives: a principle linking dendritic computation to rapid goal-directed adaptation"
  (arXiv 2606.10891, 2026) — two-compartment burst = product of soma×dendrite; trained by a LOCAL three-factor Hebbian
  rule; systematic/zero-shot by the multiplicative inductive bias. **The direct grounding for Option (i).**
- Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, "A solution to the learning dilemma for recurrent
  networks of spiking neurons," Nature Communications 11:3625 (2020) — e-prop: biologically-plausible LOCAL three-factor
  approximation to BPTT (Option iii / the bridge between BPTT and three-factor).
- Neftci, Mostafa, Zenke, "Surrogate Gradient Learning in Spiking Neural Networks" (arXiv 1901.09948, 2019) +
  Zenke & Vogels, "The Remarkable Robustness of Surrogate Gradient Learning…" (Neural Computation 33(4), 2021) — the
  surrogate-BPTT basis for Option (ii).
- Higuchi et al., "Balanced Resonate-and-Fire Neurons" (arXiv 2402.14603, 2024) — stable BPTT training of R&F neurons
  over hundreds of timesteps (the stability fix if Option (ii) training is unstable).
- Frady & Sommer, "Robust computation with rhythmic spike patterns" (PNAS, 2019) + Frady, Kleyko, Sommer, "Variable
  binding for sparse distributed representations" (IEEE TNNLS, 2021) — VSA binding on resonate-and-fire spikes and for
  SPARSE codes (the project's decorrelated-code regime); the on-substrate reference for the fixed-op fallback.
- Mikulasch, Leugering, Priesemann, "Local dendritic balance enables learning of efficient representations in networks
  of spiking neurons" (PNAS/PubMed 34876505, 2021) — decorrelation needs dendritic/voltage plasticity, not pairwise
  Hebbian; explains why the codes must be decorrelated UPSTREAM (already done), not in the binder.
- Lake & Baroni, "Human-like systematic generalization through a meta-learning neural network" (Nature, 2023) +
  "Fodor and Pylyshyn's Legacy — Still No Human-like Systematic Compositionality in Neural Networks" (arXiv 2506.01820,
  2025) — learned binders are systematic only with deliberate structure (the multiplicative bias IS that structure).

**No banking — reported exactly as found.**
