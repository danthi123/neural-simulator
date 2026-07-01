# Dendritic cortex for EMERGENCE (not hand-design) — the research-gate scoping (2026-07-01)

> **Type:** READ-ONLY deep-research + reference-catalog + literature gate (the standing "research-FIRST before
> committing months/GPU/`sim/` effort to a deep frontier" + the SURPASS round before ACCEPTING a boundary). NO code
> written, NO experiments run, NO GPU, NO `sim/` edit. Single deliverable = this doc. Stayed on `main`. Every
> load-bearing claim trust-but-verified against the actual source/code/catalog/finding text (file:line + catalog IDs +
> arXiv/PubMed cited); toy-scale / single-seed / regime-bounded flagged where that is the truth. **This is a
> scoping/decision doc, NOT a brain-based result and NOT a commitment to build.**
>
> **The owner's question (verbatim intent):** the conversational/knowledge stack was built FEATURE-BY-FEATURE — each
> feature routes *some* computation through spiking neurons, but the "conversation layer" is largely HAND-DESIGNED host
> orchestration (a binding algebra, discourse-plan templates, an intent dispatcher), not biology reproducing itself.
> The realization: **the DENDRITIC CORTEX may be the highest-priority substrate investment if we want capabilities to
> EMERGE inherently from experience — the way real brains acquire them — rather than being hand-designed one at a
> time.** Is that intuition correct? What is the minimal dendritic mechanism? Honest tractability verdict.
>
> **What this doc ADDS to the prior dendritic record (it does NOT re-derive it):** the project has already run the
> dendrite against SPECIFIC point-neuron walls and mostly found NEGATIVE — credit-assignment on a single-layer nav
> actor (`2026-06-19-dendrite-credit-assignment-toy-stage1.md`), multiplicative binding
> (`2026-06-19-dendritic-binding-toy-derisk.md`), the graded read-out (SHIPPED, the one WIN), and the whole
> "dendrite-is/isn't-the-conversational-unlock" reconciliation (`2026-06-20-dendrite-substrate-unlock-deep-research.md`,
> `2026-06-24-learned-cortex-dendrite-phase3-scoping.md`, `2026-06-22-conversational-scaling-vs-dendritic-scoping.md`,
> `2026-06-20-boundary-ledger-dendritic-audit.md` — all read in full this pass). **The owner's question is a genuinely
> DIFFERENT one from all of those.** They asked "does a dendrite fix *this specific wall*"; the owner is asking "is the
> dendrite the substrate that makes competence EMERGE from experience instead of being installed." The crucial gap: the
> prior NEGATIVEs all used **single-trainable-layer toy tasks** — and the literature is unanimous that dendritic credit
> assignment is a **DEEP/hierarchical, representation-learning** phenomenon. **The emergence question has never been
> posed in the regime where the mechanism is claimed to work.** That is the whole content of this gate.

---

## 0. TOP-LINE (read this first)

**The owner's intuition is DIRECTIONALLY RIGHT about the DIAGNOSIS and the MECHANISM, but the tractability verdict is
sobering, and one word in the framing must be split.**

1. **RIGHT (the diagnosis):** the reason the project hand-designs the conversation layer one feature at a time is, at
   root, that **the point-neuron + Hebbian/STDP substrate cannot learn DEEP compositional structure from experience**,
   and deep credit assignment on a biological substrate *is* a dendritic phenomenon (apical/basal compartments,
   burst-dependent plasticity). This is the settled literature — Sacramento-Costa-Bengio-Senn 2018, Payeur-Guerguiev-
   Zenke-Richards-Naud 2021, Whittington-Bogacz 2017/2019, Guerguiev-Lillicrap-Richards 2017. The project's own record
   corroborates it in the negative: **every emergent-structure capability that shipped was either (a) shallow /
   single-hop and learnable by local rules, or (b) hand-installed as an algebra/template.** The dendrite is genuinely
   the missing substrate for "structure develops instead of being installed."

2. **RIGHT (the two proven point-neuron NEGATIVEs are dendritic by nature):** decorrelation/whitening (Mikulasch-
   Priesemann: analog/pre-spike, point neurons provably can't) and multiplicative binding/bundling (the role-specific
   reciprocal `1/u_t` is a dendritic multiplication) are both, correctly, dendrite-flavored. The owner has these right.

3. **MUST-SPLIT (the "red herring" caveat is more consequential than it looks):** the owner already flags that
   decorrelation was a "red herring" for the PPMI cortex. That is not a footnote — it is the KEY to isolating what
   actually needs a dendrite. **Local normalization, co-occurrence learning, and single-hop generalization DID emerge
   on point neurons** (PPMI stream cortex, `corr(M,C) +0.686`, held-out 0.86, `2026-06-15-*`). So "emergence" is not
   monolithic: SHALLOW representational emergence already works on point neurons; **only DEEP/hierarchical/compositional
   emergence (a cortex that learns its own multi-level binding + grammar from a stream) is the genuine dendritic
   frontier.** The owner's "hand-designed one at a time" pain is real, but a large fraction of it is *not* a substrate
   limit — it is **missing engineering + missing a generative sequence learner**, not a missing dendrite (see §1.3, the
   "intent dispatcher / discourse template" analysis — control-flow is not a representation a dendrite learns).

4. **THE SOBERING PART (tractability — the silver-bullet risk):** even granting a working dendritic credit-assignment
   substrate, the field's own 2024–2026 results say **biologically-plausible deep learning (feedback alignment,
   predictive coding, dendritic microcircuits) WORKS at small scale (MNIST, ~5–7 layers) but the test-set gap widens
   with depth and it does NOT scale to hard/large problems (ImageNet-class)** — error-delay decays exponentially with
   depth, DFA scales poorly in deep/conv nets, and PC degrades substantially past ~7 layers absent special
   parameterization. **This is exactly the silver-bullet risk the owner asked me to flag:** a dendrite is *necessary
   infrastructure* for emergence but is **NOT sufficient** — scale, data volume, experience-stream richness, and
   training time are all still-binding walls, and the honest expectation at our scale (thousands–millions of neurons,
   one 3090, local) is a **characterized partial**, not a from-scratch emergent conversationalist.

5. **THE HONEST VERDICT + RECOMMENDATION:** the dendrite IS the highest-leverage *substrate* lever for the
   *artificial-life / biology-translatable* north-star (emergent competence from experience) — that judgment survives.
   But two things must precede any months-scale build: **(A)** the ONE cheap-first de-risk that has never been run — the
   prior credit-assignment NEGATIVE used a single-layer actor; **run the DEEP (≥2 hidden-layer) representation-emergence
   probe** where the literature says the dendrite works, on a toy where point-neuron/single-layer provably can't, and
   measure whether structure EMERGES that Hebbian can't produce. The `DendriticMLP` deep-feedback-alignment machine
   **already exists** (`sim/dendritic_mlp.py`) — this is reuse-by-import, CPU, hours. **(B)** honest acknowledgement
   that the *conversational product* does not depend on this (it ships on point neurons + the algebra + a small
   generator) — this is an artificial-life *substrate* bet, not a conversational-fluency bet. If (A) is GO, it localizes
   the months-scale build to the deep-cortex-learns-its-own-structure target and the next wall becomes scale/data. If
   (A) is NEGATIVE (the strong prior, given the shallow-emergence-works + doesn't-scale evidence), the dendrite is a
   *characterized* frontier, not a fix — and that is itself a build-saving deliverable.

---

## 1. MOVE 1 — ISOLATE + QUANTIFY: what provably needs dendrites vs what point neurons already do

The owner's core reconciliation ask: (a) the Mikulasch-Priesemann point-neuron limit + the multi-attribute-bundling
NEGATIVE say "point neurons can't"; (b) the PPMI stream-cortex says "a generalizing cortex was achieved on point
neurons, decorrelation was a red herring." **These are not contradictory once you separate the specific computations.**

### 1.1 The capability boundary (verified, exact)

| Computation | Emergent on POINT neurons? | Evidence | Verdict |
|---|---|---|---|
| **Local normalization** (log/Weber-Fechner + per-hub + per-concept mean-subtraction + rheobase) | **YES** | `2026-06-15-off-diagonal-red-herring-ppmi-*.md` — PPMI reaches host (+0.518), *beats* ZCA whitening (+0.49); whitening would *hurt* generalization | point-neuron-achievable |
| **Co-occurrence / associative learning from a stream** | **YES** | `2026-06-15-on-bridge-hebbian-co-occurrence-*.md` — rate-Hebbian `corr(M,C) +0.686` 6-seed (STDP is the WRONG rule; symmetric co-occurrence has no pre→post order) | point-neuron-achievable |
| **Graded read-out of a distributed analog value** | **YES — but only via the SHIPPED dendritic plateau** | `enable_graded_dendritic_plateau` (`sim/kernels.py:280-330`, δ=1.33); the linear point read is sub-rheobase, the all-or-none plateau over-clamps — a *graded* dendritic term was needed | dendrite-DELIVERED (the one WIN) |
| **Single-hop / single-attribute generalization** (cat~dog via similar codes) | **YES** | `2026-06-16-onsubstrate-learned-binder-single-attr-GO-*.md` — on-bridge held-out 0.833 = 100% of numpy; the capstone (perceive-novel→generalize→answer) GO 3-seed | point-neuron-achievable |
| **Decorrelation / whitening** (derive orthogonal codes from correlated ones) | **NO** | 4 point-neuron mechanisms NEGATIVE (`2026-06-11-cortex-*`); Mikulasch-Priesemann *Trends Neurosci* 2023 (PubMed 36577388): the common-mode removal is an **analog/pre-spike dendritic** computation | genuinely dendritic — **but a RED HERRING for generalization** (whitening hurts it) |
| **Multiplicative / role-specific binding** (a fact = superposition; unbind needs `1/u_t`) | **NO (learned)** | `2026-06-16-*-bundling-NEGATIVE.md` — additive 0.193, learned-linear-inverse 0.056; fixed ±1/FHRR bundles 0.989. Multiplication is dendritic (NMDA-plateau coincidence, G.02) | genuinely dendritic — **but the naive LEARNED dendritic bind tested NEGATIVE** (memorizes, doesn't generalize) |
| **DEEP credit assignment** (a multi-LEVEL cortex learns its own compositional structure end-to-end from a stream) | **UNTESTED in the right regime** | the single-layer NEGATIVE (`2026-06-19-credit-assignment`) is NOT the regime the literature (Sacramento-Senn, Payeur) says the dendrite works — HIDDEN layers | **THE GENUINE OPEN DENDRITE QUESTION** |

### 1.2 The reconciliation (the exact statement)

- **The apparent contradiction dissolves:** "point neurons can't (Mikulasch-Priesemann)" is TRUE *for decorrelation*,
  and "point neurons can (PPMI)" is TRUE *for generalization* — because **generalization never needed decorrelation.**
  Whitening removes the very semantic similarity that generalization rides on; PPMI *preserves* it. The fork's
  "structured cortex needs the dendrite" (`2026-06-11`) tested the wrong hypothesis (decorrelate-the-correlated-codes,
  which is dendritic and fails on point neurons) and was overturned by CYCLE 88-96.
- **What is LEFT after removing the red herring:** the *only* computations that provably need a dendrite are (i)
  multiplicative binding, and (ii) DEEP credit assignment for a cortex that learns its OWN multi-level structure. And
  (i)'s *naive* form already tested NEGATIVE (a single-layer learned dendritic sigma-pi memorizes, doesn't generalize).
- **So the genuine dendrite frontier collapses to ONE thing:** **can a DEEP (hierarchical) dendritic cortex, credit-
  assigned by the apical-basal / burst-dependent machinery, LEARN generalizable compositional structure from a stream
  where a point-neuron / single-layer network provably cannot?** That is (ii), and if it works it *subsumes* (i) (the
  binding rule would be learned, not installed). This is untested — because every prior probe was single-layer.

### 1.3 The part of the owner's pain that is NOT a dendrite (be honest about this)

The owner names three hand-designed pieces. They are NOT the same category, and conflating them inflates the dendrite's
apparent leverage:

- **The binding ALGEBRA (FHRR exact-inverse):** this IS a genuine dendrite candidate — it stands in for a learned
  compositional bind, and a deep dendritic cortex could in principle *develop* it. (But see the NEGATIVE prior.)
  Dendrite-relevant.
- **Discourse-plan TEMPLATES / the SVO-frame grammar:** partly dendrite-relevant (a learned generative sequence model
  would develop word-order/grammar from a stream), but **the mechanism that produces free fluent generation is a
  learned autoregressive SEQUENCE model, not a dendrite per se** (`2026-06-22-conversational-scaling-vs-dendritic-*.md`
  §0: free generation + open grammar are a *different* missing mechanism — a generative learned-sequence substrate,
  e.g. the benched surrogate-grad SNN / the ~21M generator, not the dendritic cortex). A dendrite helps *learn* it;
  it is not *itself* it. Partial.
- **The intent DISPATCHER (which query-type / which handler):** this is **control-flow / routing**, not a
  representation a dendrite learns from experience. Real brains route via learned gating (thalamocortical, BG
  disinhibition — the project's `transmission_gate`) — a dendrite is not the natural substrate for it. **Least
  dendrite-relevant; largely an engineering/gating question.** (The project already has neural gating primitives.)

**⇒ Honest scoping of the owner's pain:** of the three, ~1 is squarely a dendrite job (learned binding), ~1 is a
*generative-sequence*-model job the dendrite would *assist* but not *be*, and ~1 is a *gating/control-flow* job that is
not dendritic at all. The dendrite is the highest-leverage single substrate lever, but it does **not** dissolve the
whole hand-design surface — a meaningful slice is generative-sequence-model + neural-gating engineering.

---

## 2. MOVE 2 — REFRAME via real biology: how does the brain make these EMERGENT, and the minimal mechanism

**The reframe the owner asked for:** the brain does not *install* a binding algebra or a discourse template — it
*develops* cortical hierarchy from experience via **local, dendritically-computed error signals** that solve the
credit-assignment problem *without weight transport*. The canonical mechanisms (all verified):

- **Larkum BAC / active dendrites (G.02, Kandel 6e Ch 13 p293-298; catalog verbatim: "one of the largest abstractions
  in the simulator … ~10× compute per neuron"):** the apical tuft integrates top-down feedback; basal integrates
  bottom-up; their *coincidence* produces a regenerative Ca²⁺ plateau → a burst. This burst is the physical carrier of
  a top-down teaching signal reaching the synapse locally. The nonlinear summation rule ("cluster on one branch ≫
  scattered") is the multiplicative/binding primitive.
- **Urbanczik-Senn 2014 (*Neuron*, PubMed 24507189 — ALREADY BUILT as `sim/dendritic_plasticity.py`):** a local,
  non-Hebbian, three-factor rule that minimizes the dendritic prediction error of somatic spiking. **Crucially, it
  UNIFIES supervised / unsupervised / reinforcement learning depending only on what drives the soma** — i.e. the SAME
  local rule gives *self-supervised representation emergence* (unsupervised mode) that the owner wants, not just
  supervised classification. This is the single most important biology fact for the "emergent from experience" framing:
  **the dendritic prediction-error rule is intrinsically a representation-LEARNING rule, not a label-fitting one.**
- **Sacramento-Costa-Bengio-Senn 2018 (NeurIPS, arXiv 1810.11393) + Payeur-Guerguiev-Zenke-Richards-Naud 2021 (*Nat
  Neurosci*, PubMed 34728832):** in a **multilayer** dendritic microcircuit, error-driven plasticity gated by
  high-frequency bursts lets higher-level neurons coordinate lower-level plasticity — approximating backprop, solving
  tasks that **require deep architectures**, WITHOUT weight transport (feedback alignment). **This is the mechanism that
  makes deep structure emerge instead of being installed.** The load-bearing constraint: it needs HIDDEN LAYERS (the
  single-layer NEGATIVE is off-regime by construction).
- **Whittington-Bogacz 2017/2019 + the predictive-coding lineage (Rao-Ballard 1999; `sim/predictive_coding.py`
  already exists):** predictive-coding networks with local updates approximate backprop; error is computed in apical
  compartments; PC has "inspired innovations in unsupervised and self-supervised learning" — again the *emergence*
  (not label-fitting) framing. But (the counter-fact, §4): "PC has tended to focus on small-scale experiments;
  scalability … an open problem."

**The minimal dendritic mechanism (the owner's "is there a single primitive that buys all three?"):**

> **A two-compartment (soma + apical) spiking pyramidal with (i) basal bottom-up drive, (ii) apical top-down feedback
> through a FIXED-RANDOM projection (feedback alignment — no weight transport), (iii) a burst/plateau that gates
> local plasticity (Urbanczik-Senn / burst-dependent), stacked into ≥2 hidden layers.** This single primitive buys:
> - **(i) credit assignment for emergent hierarchy** — YES, this is exactly Sacramento-Senn / Payeur (the point of the
>   whole mechanism).
> - **(ii) the decorrelation the point neuron can't** — the apical/inhibitory-gated compartment does analog pre-spike
>   normalization (Mikulasch-Priesemann) — YES, *if* needed (but it is a red herring for generalization, §1.2).
> - **(iii) multiplicative/binding operations** — YES, the apical-basal coincidence IS the multiplicative nonlinearity
>   ("cluster on one branch ≫ scattered", G.02) — the sigma-pi / NMDA-plateau bind.

So the biology answer is genuinely elegant: **ONE compartmental primitive (the Larkum two-compartment pyramidal with
burst-gated local plasticity) is the substrate for all three.** The catch is not the primitive — it is the SCALE at
which the emergence it enables actually materializes (§4).

---

## 3. MOVE 3 — What D2 already has, the Phase-3 gap, and the ranked cheap-first de-risks

### 3.1 The AUDIT — substantially more is built than "D2 Phase 0-2" (verified file:line)

| Asset | What it is | File:line | Status |
|---|---|---|---|
| `DendriticLayer` | spiking two-compartment BAC neuron: basal `x@W_basal`, apical via FIXED-RANDOM `B_apical` (feedback alignment, no weight transport), soma BAC (apical depol LOWERS effective threshold) | `sim/dendritic_neuron.py:20-58` | **BUILT** (numpy, biologically-local) |
| `urbanczik_senn_update` | the LOCAL somato-dendritic third-factor rule (unifies sup/unsup/RL) | `sim/dendritic_plasticity.py:17-41` | **BUILT** |
| **`DendriticMLP`** | **a DEEP (multi-hidden-layer) feedback-alignment MLP** — per-hidden-layer fixed-random `B`, hidden learning via `urbanczik_senn_update`, a fenced backprop `oracle` as positive-control ONLY | `sim/dendritic_mlp.py` | **BUILT — this is the hidden-layer credit-assignment machine the emergence de-risk needs, and it ALREADY EXISTS** |
| `predictive_coding.PredictiveCoder` | Rao-Ballard top-down predictor + prediction-error learning signal (order-sensitive) | `sim/predictive_coding.py` | **BUILT** (numpy, off-substrate) |
| `enable_graded_dendritic_plateau` (+ kernels) | the SHIPPED graded, non-saturating dendritic-plateau READ-OUT (the nav critic δ=1.33) | `sim/kernels.py:280-330`; `sim/config.py:219-224`; `sim/bridge.py:~6441-6479` | **SHIPPED, byte-inert when off** (`test_graded_dendritic_plateau.py` 5/5) |
| `enable_dendritic_divisive_gain` | per-source divisive gain (the D2 Phase-1 narrower form) | `sim/config.py:~260` | SHIPPED, byte-inert when off; **found NOT load-bearing for the cortex code** (Phase-2 inversion) |
| `--dendrite-critic` | the graded plateau deployed as the nav value V | `g11_bg_runner.py` | SHIPPED default-off; B-1 graded read = DONE |

**D2 phase status (verified `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md`):**
- **Phase 0** (numpy spiking two-compartment): **SURVIVES** — the per-compartment advantage holds through a genuine
  spiking soma.
- **Phase 1** (protected `sim/` edit): what SHIPPED is the **graded-plateau read-out + divisive gain**, NOT a full
  `NeuronModel.TWO_COMPARTMENT` second-state (`v_dend`) neuron (that broader form = the larger un-built edit).
- **Phase 2** (learned graded cortex from a co-occurrence stream): **HONEST NEGATIVE for the gain's necessity** — the
  clean-readout control inverted ("with enough temporal integration the point neuron recovers the structure; the gain
  HURTS"). This is the Phase-2 wall.
- **Phase 3 (PENDING — task #23):** plug learned graded codes → bind/unbind/cleanup → the conversational matrix; GATE =
  generalization-in-conversation with the moat intact. **Two independent scoping docs found Phase-3's GATE ALREADY
  PASSES on point neurons (PPMI + hybrid, 0.92 3-seed)** — so *as originally framed (generalization)* Phase 3 would
  re-deliver a shipped capability. **The Phase-3 that is NOT redundant is the EMERGENCE reframe the owner is raising:
  not "generalize across similar concepts" (done) but "DEVELOP the compositional/binding/grammar structure from
  experience instead of installing it" — and THAT has never been the tested Phase-3 gate.**

**⇒ The Phase-3 gap, precisely:** the built D2 machinery (`DendriticLayer` + `DendriticMLP` + `urbanczik_senn_update` +
the graded plateau) covers the *neuron* and the *read-out* and even the *deep credit-assignment learner* — but it has
**never been run on the emergence question in the deep regime**: does a DEEP dendritic cortex, learning locally from a
stream, DEVELOP compositional structure (a binding rule / a hierarchy) that point-neuron Hebbian cannot? The one prior
deep-ish artifact (`DendriticMLP`) was only ever used for supervised classification and grad-alignment measurement, not
for unsupervised structure-emergence.

### 3.2 Ranked cheap-first de-risks (the emergence hypothesis, before any months-scale build)

Each is CPU/numpy, reuse-by-import, NO `sim/` edit, with the mandatory anti-cheats. The honest metric throughout:
**"did the structure EMERGE (develop from experience), or was it still SUPPLIED (installed / memorized)?"**

**De-risk EMERGE-1 (RECOMMENDED FIRST — the one never run): DEEP dendritic representation emergence.**
- **Question:** does a DEEP (≥2 hidden-layer) dendritic network, trained with the local Urbanczik-Senn / burst-gated
  rule via feedback alignment (NO weight transport), DEVELOP hierarchical representations that a SINGLE-layer dendrite
  and a point-neuron Hebbian net provably cannot — on a task with genuine compositional/hierarchical structure (e.g.
  hierarchical-feature classification, or self-supervised next-element prediction on a structured sequence)?
- **Reuse:** `DendriticMLP` (the deep FA machine, already built) + `DendriticLayer` + `urbanczik_senn_update`.
- **Arms (identical data/splits/seeds):** `point_hebbian` (control, must fail on the deep-structure task) ·
  `single_layer_dendrite` (the prior-NEGATIVE regime, must fail) · `deep_dendrite_FA` (TEST) · `oracle_backprop`
  (fenced ceiling ONLY) · `memorization_floor` (lookup, ≈chance on held-out) · `apical_lesion` (passive compartment,
  must collapse the TEST to the point floor) · `wrong_sign` (must anti-learn).
- **Honest metric:** held-out (leakage-asserted) generalization AND an emergence measure — hidden-layer
  representational structure (e.g. RSA / linear-probe on frozen hiddens, or emergent grad-alignment climbing during
  training, which `DendriticMLP.hidden_grad_alignment` already measures) that is ABSENT in the point/single-layer arms.
- **GO** = deep_dendrite_FA generalizes AND its hidden structure emerges, > both controls, multi-seed (42/43/44),
  lesion collapses. **NEGATIVE** = it too memorizes/fails to develop structure → the dendrite is characterized-frontier,
  not a fix (build-saving).

**De-risk EMERGE-2 (parallel, cheap): learned generalizable MULTI-attribute composition via the DEEP binder.**
- **Question:** the naive single-layer dendritic bind was NEGATIVE (memorizes, doesn't generalize, `2026-06-19`); does
  a DEEP dendritic binder (`DendriticMLP` credit-assigning a multi-attribute bind) learn invertible bundling that
  GENERALIZES to held-out role-filler combos, where single-layer (0.168) and learned-linear-inverse (0.056) can't?
- **Reuse:** the existing binding harness (leakage-free systematicity splits, fixed-FHRR positive control at 0.989,
  memorization floor) + `DendriticMLP`. **GO** = held-out ≥ 0.40, > all controls, > the single-layer 0.168, small
  train→held gap. (This is the §4-de-risk `2026-06-24-learned-cortex-dendrite-phase3-scoping.md` already scoped — it is
  the SAME cheap probe, framed as "did the bind EMERGE" rather than "did the bind fit.")

**De-risk EMERGE-3 (parallel, cheapest, on-substrate-adjacent): self-supervised stream cortex on the SHIPPED graded
plateau.** Does the SHIPPED `enable_graded_dendritic_plateau` + a local prediction-error rule let a cortex DEVELOP
graded reproducible codes from a raw co-occurrence stream that the point neuron lost at Phase 2 (+0.06 vs host +0.44)?
This directly retests the D2 Phase-2 NEGATIVE with the graded read-out + a self-supervised (predict-next) target rather
than a supervised one — the Urbanczik-Senn "unsupervised mode." Reuses shipped code; the one that most directly tests
"emergence from experience on the real substrate."

### 3.3 Which hand-designed scaffold each dendritic capability would let the brain DEVELOP

| Dendritic capability (if a de-risk GOes) | Hand-designed scaffold it would let the brain DEVELOP instead |
|---|---|
| Deep credit assignment (EMERGE-1) | the whole feature-by-feature install pattern — a cortex that learns structure from a stream |
| Learned generalizable binding (EMERGE-2) | the FHRR exact-inverse binding **algebra** (the composer idealization) |
| Self-supervised graded stream cortex (EMERGE-3) | the curated concept-code + PPMI-scaffold; graded codes that emerge, not normalized-by-hand |
| (NOT dendrite) generative sequence model | the discourse-plan **templates** + free-generation — a *separate* learned-sequence bet |
| (NOT dendrite) neural gating | the intent **dispatcher** — a `transmission_gate`/BG-disinhibition routing question |

---

## 4. MOVE 4 — VERDICT: is dendritic cortex the top lever, how tractable, where's the next wall, silver-bullet risk

### 4.1 Is the owner's intuition right? Necessary vs sufficient.

- **NECESSARY (for the artificial-life north-star):** YES. If the goal is competence that DEVELOPS from experience the
  way brains do it, deep dendritic credit assignment is the settled biological mechanism, and the point-neuron substrate
  demonstrably cannot develop deep compositional structure (that is *why* the project hand-installs it). The dendrite is
  the single highest-leverage *substrate* investment for emergence. The intuition survives.
- **NOT SUFFICIENT (the crux):** a dendrite alone does not conjure competence. Three independent facts bound it:
  1. **Scale/data/time still bind.** The field's 2024-2026 evidence: bio-plausible deep learning (FA, PC, dendritic
     microcircuits) matches backprop only at SMALL scale / shallow depth (≤~5-7 layers, MNIST-class); the test-set gap
     WIDENS with depth; DFA scales poorly in deep/conv nets; PC "degrades substantially" past ~7 layers absent
     special (Depth-µP) parameterization; and it does NOT generalize to ImageNet-class problems. **Emergent
     conversational/knowledge competence is an ImageNet-class-or-harder problem.** At thousands-millions of neurons on
     one 3090, the honest expectation is a *characterized partial* (toy-scale emergence demonstrated), not a
     from-experience conversationalist.
  2. **A generative sequence model is a SEPARATE lever.** Free fluent generation + open grammar are categorically not a
     dendrite (`2026-06-22-*`) — they need a learned autoregressive sequence substrate. The dendrite would help *learn*
     it, but the two are distinct bets.
  3. **The project's own strong prior is NEGATIVE-leaning.** Two cheap-first dendrite jobs already NEGATIVE; six
     "needs-dendrites" verdicts historically overturned on point neurons; the boundary ledger finds **ZERO shipped
     capabilities gated by a dendrite.** The base rate says the deep-emergence de-risk is more likely BOUNDARY than GO.

### 4.2 Where the NEXT wall is (after a hypothetical dendrite GO)

In order of when they bite: **(1) DEPTH-scaling of the local rule** (the FA/PC error-delay-decay wall — the very first
thing to hit past a few layers; the field's open problem); **(2) DATA VOLUME + experience-stream RICHNESS** (emergence
needs a rich, long, structured stream — the project's stream is thin; "thin-but-true" grounded knowledge, per
`project_deep_knowledge_brain_fluency_build`, is a data wall a dendrite doesn't fix); **(3) TRAINING TIME / compute**
(~10× compute per compartmental neuron per G.02/T3.A → months-scale even at toy scale, and the `sim/kernels.py` rewrite
does not compose cleanly with existing kernels); **(4) the generative-sequence gap** (still there). The dendrite moves
the wall from "can't develop deep structure at all" to "develops it at toy scale but doesn't scale" — a real advance for
the *science* question, not obviously a product unlock.

### 4.3 The silver-bullet risk (flagged explicitly, as asked)

**The risk is treating "dendrites" as the one substrate change that makes everything emerge.** The evidence says the
opposite: the dendrite is *necessary infrastructure* whose payoff is *gated behind scale/data/time*, and a large slice
of the owner's hand-design pain (generation, gating) is not even a dendrite job. The project has a documented history of
crying "dendrite" and then finding a point-neuron reframe (6 times) OR finding the dendrite NEGATIVE (twice) — the
standing scepticism is well-calibrated and should hold here. **Concretely: do NOT start the months-scale
`NeuronModel.TWO_COMPARTMENT` `sim/kernels.py` rewrite (T3.A, "months to years, separate research arc") on the
emergence premise until the cheap deep-regime de-risk (EMERGE-1) is GO.** The catalog's own gate for the full
two-compartment neuron is "a target experiment requires it AND we've exhausted single-compartment alternatives" — and
the single-compartment alternatives for the *conversational* goal are NOT exhausted (PPMI + generator + gating).

### 4.4 Recommended cheap-first sequence + honest tractability

| Step | Action | `sim/` edit? | Cost | Gate |
|---|---|---|---|---|
| **A** | **EMERGE-1: the DEEP dendritic representation-emergence probe** (reuse `DendriticMLP`). The one never run — the prior NEGATIVE was single-layer. | **NO** | hours, CPU | GO = deep structure emerges + generalizes, > point/single-layer, lesion collapses, multi-seed. |
| **B** | **EMERGE-2 + EMERGE-3 in parallel** (deep learned binder; self-supervised stream cortex on the shipped graded plateau). | **NO** | hours, CPU/GPU | GO = generalizable bundling emerges / graded stream codes emerge past the point floor. |
| **C** (only if A GO) | **Scope** the deep dendritic cortex on the substrate (the emergence-framed D2 Phase 2/3), with the DATA/stream-richness + depth-scaling walls named as the gates, NOT assumed away. | maybe (protected, byte-review) | weeks (scoping) | a target experiment that requires the full two-compartment neuron + a plan for the scale wall. |
| **D** (only if C warrants — MONTHS) | The full `NeuronModel.TWO_COMPARTMENT` second-state neuron (T3.A `sim/kernels.py` rewrite), guarded, byte-identical when off. | **YES (deep, protected hot-path)** | months | emergence-in-conversation at the scale the de-risks proved reachable. |

**Honest tractability-at-our-scale verdict:** emergent SHALLOW representation (co-occurrence, normalization, single-hop
generalization) is *already reachable and demonstrated* on point neurons. Emergent DEEP compositional competence
(structure that develops instead of being installed) is **reachable to a characterized PARTIAL at toy scale via a
dendrite** — that is a genuine, novel, biology-translatable science deliverable and worth the cheap de-risk. But a
from-experience conversational/knowledge agent at product quality is **NOT reachable at our scale** on the field's own
evidence (bio-plausible deep learning doesn't scale to hard problems, the stream is thin, ~10× compute per neuron, and
generation/gating are separate levers). **The honest deliverable is: dendrite = the right substrate lever for the
emergence *science*, gated behind a cheap de-risk, with scale/data as the wall behind it — NOT a silver bullet for the
hand-design surface.**

---

## 5. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (read in full / file+line / arXiv+PubMed):**
- The built D2 stack: `sim/dendritic_neuron.py:20-58`, `sim/dendritic_mlp.py` (the DEEP FA machine — read in full),
  `sim/dendritic_plasticity.py:17-41`, `sim/predictive_coding.py`, `sim/kernels.py` graded plateau, `sim/config.py`
  flags — read directly.
- The prior dendrite NEGATIVEs + reconciliations: `2026-06-19-dendrite-credit-assignment-toy-stage1.md` (single-layer,
  NEGATIVE, "nothing to align"), `2026-06-20-dendrite-substrate-unlock-deep-research.md` (the split),
  `2026-06-24-learned-cortex-dendrite-phase3-scoping.md` (the three-computation reconciliation),
  `2026-06-20-boundary-ledger-dendritic-audit.md` (ZERO shipped capabilities dendrite-gated; 2 NEGATIVE; 6 overturned),
  `2026-06-22-conversational-scaling-vs-dendritic-scoping.md` (generation/gating are separate levers) — all read in
  full.
- The D2 build plan (Phase 0/1/2/3 + Phase-2 NEGATIVE + the "months-scale floor"): `docs/plans/2026-06-14-D2-dendritic-cortex-build-plan.md` — read in full.
- Catalog G.02 (active dendrites MISSING, ~10× compute, "one of the largest abstractions") — read directly from
  `feature-catalog.md:2644-2652`.
- Literature: Sacramento-Costa-Bengio-Senn NeurIPS 2018 (arXiv 1810.11393); Payeur-Guerguiev-Zenke-Richards-Naud *Nat
  Neurosci* 2021 (PubMed 34728832, bioRxiv 2020.03.30.015511); Urbanczik-Senn *Neuron* 2014 (PubMed 24507189,
  unifies sup/unsup/RL); Mikulasch-Rudelt-Wibral-Priesemann *Trends Neurosci* 2023 (PubMed 36577388); Whittington-Bogacz
  2017/2019 + PC reviews (arXiv 2202.09467); Guerguiev-Lillicrap-Richards 2017 (segregated dendrites). All verified via
  WebSearch this pass.
- **The tractability/scaling counter-evidence (the load-bearing sobering fact):** FA matches BP on MNIST/CIFAR train but
  test-gap widens + fails on ImageNet (arXiv 1811.03567, 1812.06488, 1609.01596); PC degrades past ~5-7 layers, error
  decays exponentially with depth, needs Depth-µP for 100+ layers (arXiv 2506.23800, 2505.13124, 2407.01163); "PC has
  tended to focus on small-scale; scalability an open problem." Verified via WebSearch this pass.

**Flagged (honest uncertainty):**
1. **Whether the DEEP dendritic emergence de-risk (EMERGE-1) would GO** — GENUINELY OPEN, the whole point of the gate.
   The strong prior (two single-layer NEGATIVEs + the doesn't-scale evidence + shallow-emergence-already-works) is
   BOUNDARY, not GO. I do NOT predict it; flagged as a research bet.
2. **Exact PPMI decimals (+0.518 / host +0.442) and the F=3 resonator numbers** — read from prior findings, not re-run;
   not load-bearing for the direction.
3. **The precise `bridge.py` graded-plateau line range** — reported by a prior code survey; the kernel + config lines I
   verified directly.

## Sources
- Project code/findings/plans as cited inline (file:line).
- Catalog: `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` G.02 (active dendrites), J.08 (NMDA
  coincidence detector), T3.A (compartmental neurons — "months to years, separate research arc").
- Literature (verified via WebSearch, 2026-07-01):
  Sacramento, Costa, Bengio & Senn, *Dendritic cortical microcircuits approximate the backpropagation algorithm*,
  NeurIPS 2018 — https://arxiv.org/pdf/1810.11393
  Payeur, Guerguiev, Zenke, Richards & Naud, *Burst-dependent synaptic plasticity can coordinate learning in
  hierarchical circuits*, Nat Neurosci 2021 — https://www.nature.com/articles/s41593-021-00857-x (PubMed 34728832)
  Urbanczik & Senn, *Learning by the Dendritic Prediction of Somatic Spiking*, Neuron 2014 —
  https://pubmed.ncbi.nlm.nih.gov/24507189/
  Mikulasch, Rudelt, Wibral & Priesemann, *Where is the error? Hierarchical predictive coding through dendritic error
  computation*, Trends Neurosci 2023 — PubMed 36577388
  Whittington & Bogacz / PC review, *Predictive Coding: towards a future of deep learning beyond backpropagation?* —
  https://arxiv.org/pdf/2202.09467
  Bio-plausible learning scaling: https://arxiv.org/pdf/1811.03567 ; https://arxiv.org/pdf/1812.06488 ;
  https://arxiv.org/pdf/1609.01596
  PC depth-scaling: https://arxiv.org/pdf/2506.23800 ; https://arxiv.org/abs/2505.13124 ; https://arxiv.org/pdf/2407.01163

_Read-only deep-research deliverable. NO code, NO experiments, NO `sim/` edit. The owner's intuition is
directionally right (dendrite = the substrate for emergent deep structure; the two proven point-neuron NEGATIVEs are
dendritic) but must be split (shallow emergence already works on point neurons; only DEEP compositional emergence is
the frontier) and is bounded by tractability (bio-plausible deep learning does not scale to hard problems on the
field's own evidence). The recommended move is the ONE cheap-first de-risk never run — deep-regime dendritic
representation emergence (reuse `DendriticMLP`, CPU, hours) — before any months-scale `sim/` build; the honest
expectation is a characterized partial, and the silver-bullet risk (dendrite-as-panacea) is flagged._
