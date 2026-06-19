# Phase 2 — case-marking cue → true cross-language comprehension (scoping)

**Date:** 2026-06-19
**Type:** Deep-research + catalog-review scoping (standing opening move for Phase 2 of the multi-cue competition arc). READ-ONLY; no code built/edited/run.
**Owner directive (2026-06-19):** make conversational comprehension LANGUAGE-AGNOSTIC and ROBUST to imperfect input. Phase 1 (robust ENGLISH) is COMPLETE; Phase 2 = case-marking → true cross-language.
**Status of this doc:** confirm/sharpen the Phase-1 scoping's Phase-2 plan → mechanism map (how a case PARTICLE enters the SAME competition) → reuse-vs-new → cheapest-first de-risk + GO bar + controls → the explicit Phase-2 (isolating particle) vs Phase-3 (fused morphology) boundary + honest risks/stop. Controller reviews the load-bearing claims before any build.

---

## 0. TL;DR (the recommendation)

- **The hypothesis is CORRECT, well-grounded, and the canonical Competition-Model cross-linguistic result.** Adding a **CASE cue** to the SAME multi-cue competition (the Phase-1 spiking role-WTA + plastic cue→role projections + the three-factor cue-validity learner) handles case-marking, free-word-order languages. The cross-linguistic literature is decisive: **Japanese assigns agent/patient by case (ga→agent, o/wo→patient), with case the dominant cue and word-order weak**, exactly inverting English (order-dominant). It is the **same architecture, different learned cue weights** (Bates-MacWhinney; MacWhinney-Bates-Kliegl 1984 English/German/Italian; the Japanese case-vs-order studies below). "Easily adaptable" = the same parser, the three-factor learner re-run on the new language's corpus, re-learns `w_case` high / `w_position` low — **no per-language hand-coding**. This is the literal "language-agnostic" payoff.
- **The case cue is "just another cue" — the Phase-1 machinery already accepts it.** The validated spiking de-risk (`_phaseB_multicue_competition_spiking_derisk.py`) is built around a `CUES` tuple (`position, animacy, verbfit, lexbias`) where each cue is a population that emits a signed vote into the role-WTA through a **plastic cue→role projection whose weight IS its learned validity**. Adding `case` is **a 5th tuple entry + a `_case_vote` function + a tiny case-marked lexicon** — structurally identical to how `animacy`/`verbfit` already plug in. **No architecture change; no `sim/` edit anticipated.**
- **The morphology boundary is the crux, and it is crisp.** A case **PARTICLE that is its own token** (Japanese ga/を, Korean 이/가·을/를) is read as a cue **with NO new representational layer** — the particle word lights a strong role-bias population, exactly like an animacy feature lights the animacy cue. A **FUSED/portmanteau case suffix** (Russian *-a*/*-u*, Latin *-us*/*-um*) is phonologically integrated into the stem and bundles case+number+gender in one morpheme → it needs **sub-word morphological segmentation** = the genuinely-new **Phase-3** layer. **Phase 2 targets the isolating-particle case first (Japanese-style ga/を) — the cheapest, no-morphology-layer route to true cross-language.**
- **Point-neuron risk is LOW and the same as Phase 1** (rate-coded reliability-weighted accumulation, NOT the analog/dendritic decorrelation that walled before). The honest residuals are inherited from Phase 1, not new: (1) the cue VALIDITIES must genuinely LEARN (the no-learning + lesion controls decide this); (2) on-substrate three-factor learning is **seed-variable** in how hard it down-weights the now-unreliable cue (Phase 2 can demonstrate via the **install-path** validities like Phase 1's robust 5/6 arm, with on-substrate learning the shared follow-on).
- **The immediate next build = the Phase-2 cheapest-first de-risk** (§3): does the SAME competition read roles by CASE on a FREE-word-order Japanese-style toy where position-only collapses, with the case cue's validity LEARNED high (not hand-set), AND the SAME parser on the ENGLISH corpus still weights position high (= the cross-linguistic dissociation demo)? GO bar + controls in §3.

---

## 1. Confirming/sharpening the Phase-1 scoping's Phase-2 plan

The Phase-1 scoping (`2026-06-19-multicue-competition-parser-scoping.md` §4 Phase 2) named the plan: *"add a case/agreement cue family (a morphological marker → role vote) as another competing cue; demonstrate a free-word-order toy language where roles are assigned by case even when order varies, by re-learning the cue weights (`w_case` high, `w_pos` low)."* **This doc confirms that plan and sharpens it on three points:**

1. **The learning rule is THREE-FACTOR, not plain Hebbian** (the load-bearing Phase-1 finding, `2026-06-19-multicue-competition-spiking-derisk.md` §5). Plain Hebbian co-firing computes co-occurrence, not error-corrected validity, so it cannot down-weight a high-co-occurrence-but-unreliable cue. In Japanese, **word order is the cue that must be down-weighted** (free order → order is *available* but *unreliable*), so Phase 2 inherits the same requirement: the **error/reward term** is what learns `w_case` high and `w_position` low. This is the exact mechanism the literature names — discriminative-learning models of Japanese find case-marking "struggles to compete with the more frequently available cue of word-order" *unless* the learner is error-driven and gets enough free-order input for order's empirical validity to drop. **Phase 2's case-toy training distribution must contain enough free-/non-canonical order** (the `noncanon_train_frac` knob, set HIGH for the case language) so the three-factor rule discovers order is unreliable — the direct Japanese analogue of the Phase-1 `noncanon_train_frac≈0.55`.

2. **Target the isolating particle FIRST; defer fused morphology to Phase 3 explicitly** (the morphology boundary, §4). The Phase-1 scoping lumped "case/agreement" together; this doc draws the precise line: isolating particle = Phase 2 (a cue, no new layer); fused suffix = Phase 3 (a segmentation layer).

3. **The cross-linguistic DISSOCIATION DEMO is the headline deliverable, not just "a case toy works."** The owner's "easily adaptable" goal is proven by showing the SAME parser, trained on two corpora, lands on TWO different cue-weight profiles (English → `w_position` high; Japanese-toy → `w_case` high) with NO code difference between the two runs. That dissociation IS the Competition Model's central cross-linguistic result and the literal demonstration of language-agnosticism. The de-risk (§3) makes it a gated control.

---

## 2. Mechanism map — the case particle inside the SAME competition

### 2.1 How a case PARTICLE enters the competition (the central mechanism)

The Phase-1 spiking competition (`SpikingRoleCompetition`) is: per-role Wong-Wang accumulator pools `sel_agent`/`sel_patient` in mutual inhibition (via `sel_FS_*`, the Rutishauser selective-inhibition motif), fed by cue populations through **plastic cue→role projections** whose synaptic weights are the learned cue validities. Each cue emits a **signed vote** in {−1,0,+1} toward (agent:+ / patient:−); a `cue_c_pos` sub-pop projects to `sel_agent`, a `cue_c_neg` sub-pop to `sel_patient`; both share the cue's learned validity weight.

**The case cue is identical in shape — it is the strongest, most reliable vote-emitter in a case language:**

```
   surface tokens          cue populations                role assemblies
   (any word order)        (one per cue family)           (agent / patient)

   "ringo-o   inu-ga       [ POSITION cue ]  --w_pos(LOW in JP)-->  ┌──────────┐
    apple-ACC dog-NOM"     [ ANIMACY  cue ]  --w_anim----------->   │ sel_agent│◄─┐
                           [ VERBFIT  cue ]  --w_vfit----------->   │ sel_patient│◄┘ mutual
   per noun:               [ CASE     cue ]  --w_case(HIGH in JP)-> └──────────┘   inhibition
     dog-NOM  -> ga  -> CASE vote +1 (agent)        ▲                              (Rutishauser
     apple-ACC-> o   -> CASE vote -1 (patient)       │                               sel-FS WTA)
                                          reliability-weighted ACCUMULATOR per role
                                          (Wong-2002 NMDA integrator sums weighted cue drive;
                                           role crossing bound = the assignment)
```

- **The case particle → a strong role-bias.** The lexical front-end (the legitimate boundary, §2.3) detects the case particle attached to / following each noun: a NOMINATIVE marker (ga) → the noun's CASE cue emits a +1 (agent) vote, lighting `cue_case_pos`; an ACCUSATIVE marker (o/wo) → a −1 (patient) vote, lighting `cue_case_neg`. The case cue's `_case_vote(noun, marker)` returns ±1 from the marker, exactly as `_animacy_vote` returns ±1 from the animacy lexicon today.
- **In a case language the competition LEARNS `w_case` HIGH and `w_position` LOW.** Because Japanese order is free, the position cue makes **prediction errors** on the free-order training majority (a fronted object gets the agent-position vote but is the patient). The **three-factor rule** (spike-eligibility × reward/error × vote) penalizes position for those errors → `w_position` drops; case is *always* right (it never errs) → its error term net-reinforces → `w_case` climbs to the top. The synaptic cue→role weights re-settle to **Japanese's cue validities**: `w_case` ≫ `w_position`. This is the spiking analogue of the Phase-1 numpy `w_position 0.34 ≪ w_animacy 0.76`, just with `case` as the winning cue instead of `animacy`.
- **The accumulator settles roles by case even when order is misleading.** A free-/object-fronted Japanese sentence drives a *misleading* position vote but a *correct, high-weight* case vote. The summed reliability-weighted drive into `sel_agent`/`sel_patient` is dominated by the case cue → the WTA settles correctly → **free-word-order role assignment**. Same dynamic that made Phase-1's competition carry scrambled English on the surviving (semantic) cues, now carried on the case cue.

### 2.2 The cross-linguistic DISSOCIATION (the language-agnosticism payoff, one mechanism two profiles)

The competition architecture is **identical** across languages; only the **learned cue weights** differ:

| Language (corpus) | Learned profile (after the same three-factor training) | Reads free order? |
|---|---|---|
| **English** (Phase 1) | `w_position` HIGH, `w_animacy`/`w_verbfit` moderate, `w_case` ≈ 0 (no case markers in input → case cue never fires → its validity stays at floor) | carried by semantic cues when order degrades |
| **Japanese-toy** (Phase 2) | `w_case` HIGH, `w_animacy`/`w_verbfit` moderate, `w_position` driven LOW (free order → position unreliable) | carried by the case cue at any word order |

**This is the entire deliverable:** run the SAME parser/learner on two corpora; it lands on two cue-weight profiles; English weights order, Japanese weights case — *with no code path difference between the two runs.* The literature confirms exactly this dissociation (English-L1 learners of Japanese "did not use case information and relied predominantly on word-order, choosing the first noun as the agent"; native Japanese speakers show "the predominant strength of case marking"). The model reproduces both the cross-linguistic profile AND the L1-transfer error signature — a strong artificial-life validation hook.

### 2.3 What is on-substrate vs the legitimate front-end (honest, per BRAIN-BASED-ONLY)

| Piece | Status | How |
|---|---|---|
| Role COMPETITION (sel_agent/sel_patient WTA) | **SPIKING** (reused) | The Phase-1 re-pointed biased-competition WTA, unchanged. |
| Reliability-weighted ACCUMULATION + WINNER | **SPIKING** (reused) | The case cue population drives evidence through the new plastic `cue_case → role` projection; the accumulator sums it; the WTA settle is the decision. |
| Case cue VALIDITY = `cue_case → role` weight | **SPIKING-LEARNED** (install-path fallback) | The three-factor rule learns `w_case` high on the case-toy corpus (the brain-based claim); the **install-path** places the validated value for the robust multi-seed arm (the Phase-1 GO-bar's explicit fallback — an installed learned parameter, like a pre-trained weight). |
| Detecting WHICH case marker a noun carries (ga vs o) | **HOST front-end** (flagged, legitimate) | The same lexical/morphology boundary `FrameParser` + the buffer's `content_bias_target` + Phase-1's animacy/verb lexicons already occupy: the front-end supplies each cue's VALUE (which case population to light), NOT the role decision (that is the learned-weight spiking competition). For an **isolating particle** this is trivial — the particle is its own token, so detection is "is this token in the case-marker set?" (no segmentation). Conversion target = a learned lexical-feature/morpheme map (the documented boundary; the same one Phase 1 flags for its feature lexicons). |
| The reward signal in the three-factor learner | **HOST teaching signal** (flagged) | The legitimate environment/body boundary, identical to Phase 1 + the nav reward-RPE scaffolds: the *eligibility* is spike-measured, the *weight update* is on real synapses, the reward *computation* (did the settled winner match gold) is host (the documented follow-on neuralizes it). |

### 2.4 Biology anchors (verified)

- **Competition Model cross-linguistic dissociation** (the empirical target): Bates & MacWhinney 1982/1989; MacWhinney-Bates-Kliegl 1984 (English=order, German=agreement+animacy, Italian=agreement); the Japanese case-vs-order studies (Sasaki & MacWhinney; Kilborn & Ito) — case is the dominant, most-reliable cue in Japanese; ga biases agent, o biases patient; English-L1 transfer relies on order. (Cue validity = availability × reliability; the comprehender's weights track input cue validity — exactly the learned cue→role weights.)
- **Catalog G.12 (Broca)** — the textbook statement that **semantic/constraint cues carry comprehension when the order/grammar cue is hard** ("The apple the girl ate was green" succeeds, semantically constrained). The case cue is the same kind of non-position cue carrying non-canonical comprehension.
- **Catalog G.18 (LIP accumulator)** — "integrates *any* evidence weighted by reliability... each source contributes its known logLR additively." The case cue is *another reliability-weighted source* added to the same accumulator. G.16/G.17 (drift-diffusion / first-bound-crossing WTA) settle the decision; the project's BG cascade is functionally this accumulator.
- **Case-particle neural distinctness (minor anchor, NOT load-bearing):** an fMRI study (Yano et al., *Neuropsychologia*/PMC3967534) shows Japanese case particles ga/o have **distinct neural representations** in left middle/inferior frontal gyrus (Broca-adjacent) and "the information contained in a case particle affects prediction/anticipation" in incremental SOV comprehension. (Caveat recorded: the study used isolated particles without sentential nouns/verbs, so it does **not** test cue competition for role assignment — it anchors only that case particles are a dissociable, frontally-processed signal; the role-assignment claims rest on the behavioral Competition-Model literature.)
- **Point-neuron limit (why it does NOT apply):** Mikulasch & Priesemann (PNAS 2021925118) — decorrelation/whitening is the dendritic/pre-spike computation; cue-integration-to-a-winner is rate-coded reliability-weighted accumulation, which attractor/accumulator networks do near-optimally with rate codes (PNAS 2214441119; eLife 20047). The Phase-1 spiking WTA already settled fine; the case cue adds one more summand, not a new computational class.

---

## 3. Reuse-vs-new + the cheapest-first de-risk

### 3.1 What transfers WHOLESALE (reuse-by-import / additive)

| Need | Reuse from | What transfers |
|---|---|---|
| The role-COMPETITION WTA (sel_agent/sel_patient + sel_FS_*) | `_phaseB_multicue_competition_spiking_derisk.py` `SpikingRoleCompetition` | The **entire spiking competition substrate** — Wong-Wang accumulators + Rutishauser selective inhibition, the cue-population → plastic cue→role projection pipeline, `set_cue_weight`/`cue_weights`/`freeze_all_cue_plasticity`, `assign_roles` (the WTA read + moat gate). **Unchanged.** |
| The three-factor cue-validity LEARNER | same, `learn_error_gated` (+ `_settle_with_eligibility`, `_precompute_cue_edge_slots`, `_fast_set_cue_weight`) | The validated **spike-eligibility × reward × vote** rule that learns the validity SPREAD plain Hebbian cannot. Re-run on the case-toy corpus → it learns `w_case` high / `w_position` low. **Unchanged rule; new training data.** |
| The install-path fallback (robust multi-seed arm) | same, `INSTALLED_CUE_WEIGHTS` + `learn_mode="install"` | The GO-bar's explicit fallback: install the validated case-language validities into the WTA + run the free-order battery + controls. (Reported honestly as installed-not-spiking-learned, like Phase 1's 5/6 arm.) |
| The degradation/battery + dataset + moat machinery | same, `build_dataset`/`_battery_accuracy`/`_moat_breaches`/`_calibrate_abstain_margin`/`run_seed` | Dataset construction (canonical-majority + non-canonical-minority training; the eval battery; the ambiguous moat set), the battery accuracy harness, the moat calibration + breach counter, the per-seed GO gates + ≥5/6 aggregator. **Re-parameterized** (the case toy makes the *case-absent* battery the position-degrading set; raise `noncanon_train_frac`). |
| The production drop-in shape | `multicue_role_parser.py` `MultiCueRoleParser` | `parse(words, voice) -> {agent, action, patient}` + `parse_decisive` + the lexical verb front-end + the moat content gate. A Phase-2 case-aware parser is the same class with a case-marker front-end + the case cue in evidence. |
| The agent wire-in (default-OFF flag) | `brain_conversational_agent.py` `enable_multicue_competition` + `hear_multicue` | The exact opt-in pattern (lazy/cached build, byte-identical when off, moat preserved end-to-end). A Phase-2 flag (e.g. `multicue_lang="ja"` selecting the case-aware parser) reuses it verbatim. |
| The no-confab MOAT | same, the `_semantic_contrast` content gate / `parse_decisive` | Abstain when no decisive content cue (two animate nouns + a symmetric verb → tie). **Unchanged** — case adds a cue, it does not weaken the gate. |

### 3.2 The MINIMAL NEW pieces (small; additive; flag any `sim/` touch — none anticipated)

1. **A `case` entry in `CUES`** + a **`_case_vote(noun, marker)`** function (returns +1 for a nominative marker, −1 for accusative, 0 if unmarked) — structurally identical to `_animacy_vote`/`_verbfit_vote`. The `SpikingRoleCompetition.__init__` already builds a `cue_{c}_pos`/`cue_{c}_neg` population + a plastic `cue_{c} → role` projection **for every `c in CUES`**, so adding `"case"` to the tuple auto-creates its populations + projection with **zero constructor edits**.
2. **A tiny case-marked toy lexicon/corpus** — a Japanese-style toy where each noun carries a particle token (`inu ga`, `ringo o`) and word order is FREE (SOV canonical + OSV + scrambled all present in training, at a HIGH `noncanon_train_frac` so order's empirical validity is low). The cue VALUE front-end maps the particle → the case vote. (The corpus is a legitimate environment artifact — the language's data; the same boundary as Phase 1's training sentences.)
3. **The case-marker front-end in the parser** — extend `MultiCueRoleParser._split_verb_nouns` / `_evidence_for_nouns` to detect the particle token attached to each noun and pass its case marker into `cue_evidence`. For the **isolating particle this is a set-membership check on the token** (no segmentation). 

**Net:** Phase 2 = **+1 `CUES` entry + 1 vote function + a case-marked toy corpus + a case-marker front-end**, all reuse-by-import / additive. **NO `sim/` edit anticipated** (the populations + projection are auto-built from the `CUES` tuple). If a per-cue gain or a sim-level change turns out necessary, it gets a byte-level diff review per the standing rule.

### 3.3 The cheapest-first DE-RISK (the immediate next build)

**Question it decides:** *Does the SAME multi-cue competition read thematic roles by CASE on a FREE-word-order, case-marked toy where the position-only parser collapses — with the case cue's validity genuinely LEARNED HIGH (not hand-set) — AND does the SAME parser, trained on the ENGLISH corpus, still weight position high (the cross-linguistic dissociation)?*

**Setup (smallest CPU/numpy, hours — the Phase-1 de-risk found CPU ~9× faster than GPU for this tiny 200-neuron bridge):**
- Reuse `SpikingRoleCompetition` with `CUES = (position, animacy, verbfit, case, lexbias)`. Build a **Japanese-style case toy**: animacy-balanced nouns each carrying a particle (nominative→agent, accusative→patient); asymmetric verbs (so verb-fit is informative on some items) + symmetric verbs (feed the moat). **Training is free-order naturalistic** (canonical SOV + OSV + scrambled, `noncanon_train_frac` HIGH so position's empirical validity is low). Held-out test FILLERS + VERBS disjoint from training (role correctness vocab-agnostic → not memorizing examples).
- Two learn paths, exactly as Phase 1: **install** (validated case-language validities — the robust multi-seed arm) AND **error_gated** (the three-factor on-substrate learner — the brain-based claim).

**The test battery (the case toy's position-degrading set):**
1. **FREE-WORD-ORDER (OSV / scrambled with case markers present):** the case markers are correct but position is misleading → the case cue must carry the role assignment. This is the Phase-2 analogue of Phase-1's scramble/object-front.
2. **CASE-PRESENT, POSITION-MISLEADING canonical-inverse:** an object-fronted sentence where a position-only parser maps the fronted noun to agent (and fails); case overrides.
3. (separately reported, like Phase-1's drop-verb) a **case-ABSENT** condition (particles dropped) where the case cue is silent and only animacy/verb-fit survive — characterizes graceful degradation, NOT gated (position is also degraded here so it is not the load-bearing position-only-collapse metric).

**GO bar (pre-register; FROZEN; ≥6 seeds; fractional ≥5/6 — matching Phase 1's standard):**
- **The case-path recovers roles on the free-word-order battery at ≥ 0.80** mean role accuracy (vs chance 0.50 for 2-role agent/patient), on ≥5/6 seeds, AND
- **the LOAD-BEARING control — POSITION-ONLY baseline — COLLAPSES on the SAME free-order battery** (≤ chance+0.12 ≈ 0.45 / ideally near chance; a genuine position-only parser maps the fronted noun to agent and FAILS free order). *If position-only does NOT collapse, the battery is not actually free-order/position-degrading → INVALID, fix the toy first.* This proves the win is the **added case cue carrying free order**, not a generically better parser. AND
- **the native canonical (SOV) still comprehends** (no regression; case-path ≥ position-only on clean SOV — with the honest Competition-Model trade-off noted in Phase 1: a learner that down-weights position to win free order may pay a small canonical cost, so the gate requires clean comprehension stays STRONG, not that it beats a pure-position parser on its home turf), AND
- **the no-confab MOAT holds** (two animate nouns + a symmetric verb, case markers ambiguous/absent → no decisive content cue → abstain; 0 breaches on every seed).

**Controls (each must pass or it is NOT a GO):**
1. **CASE-LESION (THE decisive control):** zero the learned `cue_case → role` weights, keep the others. Free-order robustness must **collapse back to the position-only/semantic-only level**. A lesion that does NOT break free-order robustness = the case cue was not load-bearing = NOT a GO. (Mirrors the Phase-1 cue-lesion + the buffer's bias-lesion control.)
2. **NO-LEARNING control ("are weights learned, not hand-set?"):** run the case cue→role weights FROZEN at uniform init (skip the three-factor learner). If free-order robustness is the same as trained, the weights were not doing the work → hand-tuned → NEGATIVE. (The honest-risk guard.)
3. **PERMUTED-CASE control:** train the case cue→role map against a **scrambled** case-marker→role assignment (nominative→patient, accusative→agent). Free-order accuracy must collapse to chance → proves the case cue carries *real* role information, not a relabelled position signal.
4. **THE CROSS-LINGUISTIC DISSOCIATION (the language-agnosticism control — the headline):** run the SAME parser/learner UNCHANGED on the **English** Phase-1 corpus (no case markers → case cue never fires) and confirm it lands on `w_position` HIGH / `w_case` ≈ 0, AND on the **Japanese-toy** corpus it lands on `w_case` HIGH / `w_position` LOW — *with no code-path difference between the two runs.* This is the literal demonstration that "adapt to a new language = re-learn the weights, not re-code." If both corpora produce the same profile, the learner is not actually tracking language-specific cue validity → the adaptation claim fails.
5. **Held-out fillers + verbs:** the test uses fillers/verbs unseen in training (role correctness is vocab-agnostic) → not memorizing the examples.
6. **Moat intact:** assert 0 false role-commitments on genuinely ambiguous input. The moat is not weakened.

**Outcomes:**
- **GO** → the same competition reads roles by case on free word order, case cue load-bearing + learned, AND the cross-linguistic dissociation holds → promote a case-aware `MultiCueRoleParser` variant into `BrainConversationalAgent` behind a default-OFF language flag (byte-identical when off); multi-seed GPU validate the full who/what + moat pipeline on the case toy; THEN the Phase-3 morphology decision (§4).
- **BOUNDARY** → free order recovers but (e.g.) the on-substrate three-factor learner is seed-fragile in down-weighting position at the spiking scale (the documented Phase-1 residual) → ship the **install-path** arm as the robust result + report the learning seed-variance honestly, exactly as Phase 1 did.
- **NEGATIVE** → the case cue doesn't carry free order (competition fails to integrate it), OR the no-learning/lesion/permuted controls don't collapse (hand-tuned), OR the dissociation control gives the same profile on both corpora (not tracking language-specific validity) → report the honest negative + the fallback (§5).

---

## 4. The Phase-2 (isolating particle) vs Phase-3 (fused morphology) boundary — the crux

This is the explicit scope line the prompt asks for. It is grounded in standard morphological typology (isolating ↔ agglutinative ↔ fusional continuum), verified against the case literature.

### 4.1 Phase 2 = ISOLATING-PARTICLE case (a cue, NO new layer)

A case **particle that is its own token** is read as a cue **with no representational layer below the word**:
- **Japanese ga / を(wo/o) / に(ni)** are **distinct, isolating bound morphemes** — each is essentially its own token following the noun (`inu ga`, `ringo o`). Detecting which case a noun carries = **a set-membership check on the particle token** (`is this token in {ga, o, ni, ...}?`). No segmentation, no sub-word composition.
- **Korean 이/가 (NOM), 을/를 (ACC)** are likewise **agglutinative, near-isolating case particles** (with a phonologically-conditioned allomorph pair, but each is a discrete, segmentable suffix-particle, not a fused portmanteau) — the same "particle → cue" treatment, with at most a 2-way allomorph lookup in the front-end (still no morphological *decomposition*).
- **Why this is cheap:** the particle token lights the case cue population exactly as an animacy feature lights the animacy cue. The case cue's VALUE comes from a token-level lookup — the **same legitimate lexical front-end boundary** Phase 1's animacy/verb lexicons + the buffer's `content_bias_target` already occupy. **No new layer; the Phase-1 competition machinery accepts it as a 5th cue.**

**Phase 2 target: Japanese-style ga/を first** (the cleanest isolating particle; the canonical Competition-Model case language; richest literature). Korean is a near-identical follow-on (the same mechanism, a 2-way allomorph front-end).

### 4.2 Phase 3 = FUSED / portmanteau morphology (a genuinely new segmentation layer)

A **fused case suffix** is phonologically integrated into the stem and **bundles multiple grammatical meanings (case + number + gender) in one morpheme**:
- **Russian** is **fusional**: `-a`/`-u`/`-om`/`-e` etc. fuse case+number+gender and integrate with the root (`sobak-a` NOM vs `sobak-u` ACC). The role-relevant case is NOT a separable token; it is **entangled in a word-final morpheme** that also encodes number and gender.
- **Latin** is the same (`lup-us` NOM vs `lup-um` ACC — case+number+gender fused in the ending).
- **Why this needs a new layer:** to read the case cue, the system must **segment the word into stem + suffix and decode WHICH grammatical features the (fused) suffix encodes** — a **sub-word morphological-segmentation + composition** tier *below* the word level. That is a genuinely new representational layer (and plausibly interacts with the deferred dendritic/compositional-binding substrate, since decomposing+recomposing a fused morpheme is a compositional-binding problem). It is **NOT** "just another cue" — the cue's VALUE can no longer come from a token-level lookup.

**Phase 3 is correctly DEFERRED** — the only months-scale NEW layer in this arc, with the narrowest reward (only fusional/agglutinative-fused languages need it; isolating-particle languages, English, and lightly-inflected languages never do). It is named here so the boundary is honest, not pursued now.

### 4.3 The boundary in one line
**Phase 2 ends where the case marker stops being a separable token and starts being a fused morpheme.** Isolating particle (Japanese ga/を, Korean 이/가·을/를) = Phase 2 (a cue, token-level front-end, NO new layer). Fused/portmanteau suffix (Russian -a/-u, Latin -us/-um) = Phase 3 (sub-word segmentation + composition, a new representational tier).

---

## 5. Honest risks + the stop rule

### 5.1 The biggest way it could MISLEAD — hand-tuned case cue masquerading as a learned model (inherited from Phase 1)
The seductive failure is identical to Phase 1: the **case-marker front-end is a host scaffold** (it supplies the cue VALUE). If the `cue_case → role` weight is effectively hand-set (or the front-end does the discrimination and the "competition" just reads it out), then "reads Japanese by case" is **engineering, not a learned cue-validity model** — and would NOT transfer to a new language by re-learning, defeating the language-agnostic goal. **Guard:** the **NO-LEARNING** + **CASE-LESION** + **PERMUTED-CASE** controls (§3) directly test this; the win must require *trained* case→role weights and vanish when they are removed/scrambled. AND the **cross-linguistic dissociation control** is the additional Phase-2-specific guard: the same code must produce DIFFERENT profiles on English vs Japanese — if it produces the same profile, it is not tracking language-specific validity. (This is the same load-bearing-controls-first standard that retracted the 2026-05-14 compose-concept and transitive-inference claims.)

### 5.2 The Phase-1 seed-variance residual is INHERITED, not new
The Phase-1 spiking de-risk found on-substrate three-factor learning is **seed-variable** in how hard it down-weights the unreliable cue at the spiking operating scale (the **install path was the robust 5/6 GO**; the error-gated path GOes on the seeds where it learns the spread well). Phase 2 inherits this exactly — in fact Phase 2's "unreliable cue to down-weight" is **word order** (which in a free-order language is even more clearly unreliable than English position was), so the three-factor learner has a *cleaner* error signal, but the same operating-point friction applies. **Mitigation = the same as Phase 1:** demonstrate via the **install-path** validities for the robust multi-seed arm; treat full multi-seed on-substrate validity-learning robustness as the **shared follow-on** (the same residual for both phases), and neuralizing the learner's reward as the last host scaffold. This is NOT a Phase-2-specific blocker.

### 5.3 The point-neuron risk is LOW and the same as Phase 1
Adding a case cue is adding one more **reliability-weighted summand** to a rate-coded accumulator — categorically NOT the analog/dendritic decorrelation that walled the conversational whitening blocker and the structured-cortex generalization (Mikulasch-Priesemann). The Phase-1 WTA already settled fine with four cues; a fifth is the same computational class. (Where a graded/dendritic computation re-enters is the separate generalization-across-similar-concepts arc — NOT Phase 2's free-word-order goal.) The narrow residual, as in Phase 1, is whether a finely-graded cue confidence needs a population-coded representation; the case cue is near-binary (nominative/accusative/unmarked), so this is mild — and the fallback (population-coded cue confidence) is the documented rate-code-wall lift, not a dendritic rewrite.

### 5.4 Clear cheap-first GO vs NEGATIVE
- **GO** = free-order battery role accuracy ≥0.80 (≥5/6) **AND** position-only collapses on the same battery **AND** case-lesion collapses **AND** no-learning/permuted controls collapse **AND** the cross-linguistic dissociation holds (English→position, Japanese→case, same code) **AND** clean SOV unregressed **AND** moat 0-breach.
- **NEGATIVE** = the case cue doesn't carry free order, OR the no-learning/lesion/permuted controls don't collapse (hand-tuned), OR the dissociation control gives the same profile on both corpora (not tracking language-specific validity). Report it; the fallback is (a) the install-path arm + honest learning-seed-variance report (if the issue is on-substrate learning robustness, the documented Phase-1 residual), or (b) population-coded cue confidence (if cue resolution is the blocker). The dendritic/morphology layer is NOT required for Phase 2 (it is Phase 3, and only for fused morphology).

---

## 6. Provenance (verified sources)
- **Competition Model + cross-linguistic dissociation:** Bates & MacWhinney 1982/1989; MacWhinney-Bates-Kliegl 1984 "Cue validity and sentence interpretation in English, German, and Italian" (*JVLVB*). Cue validity = availability × reliability; learned weights track input cue validity.
- **Japanese case-vs-order (the Phase-2 empirical target):** the Competition-Model Japanese studies (Sasaki & MacWhinney; Kilborn & Ito) + recent L2-Japanese case/order work — case is the **dominant, most-reliable** cue in Japanese (ga→agent, o/wo→patient); word order is available-but-unreliable (free order); English-L1 learners transfer the word-order strategy ("first noun = agent") and "did not use case information"; native speakers show "the predominant strength of case marking"; discriminative-learning models find case "struggles to compete with the more frequently available cue of word-order" unless error-driven with enough free-order input. ["Sentence Processing within the Competition Model" (Columbia SALT); "Decoding case markers: L1 Chinese L2 Japanese learners' comprehension of Japanese OSV sentences" (*Linguistics* 2023); "The use of case marking for predictive processing in second language Japanese" (*Bilingualism: Language and Cognition*).]
- **Morphology typology boundary (the crux):** standard isolating↔agglutinative↔fusional continuum — Japanese ga/o/ni are **distinct isolating bound morphemes** (particle = token-level cue); Korean 이/가·을/를 are agglutinative near-isolating case particles (a 2-way allomorph); Russian/Latin are **fusional** (case+number+gender fused in one stem-integrated suffix → needs segmentation). [Morphological-typology references; "What Is An Agglutinative Language?" overview.]
- **Case-particle neural distinctness (minor, NOT load-bearing):** Yano et al., "Neural differences in processing of case particles in Japanese: an fMRI study" (PMC3967534) — ga/o have distinct left-IFG/MFG representations; case-particle information drives incremental prediction. (Caveat: isolated particles, no role-assignment competition test — anchors only that case particles are a dissociable frontally-processed signal.)
- **Neural cue-integration / accumulation:** catalog G.18 (LIP integrates any reliability-weighted evidence additively), G.16/G.17 (drift-diffusion / first-bound-crossing WTA), G.12 (Broca; semantic/constraint cues carry non-canonical comprehension); PNAS 2214441119; eLife 20047 (rate-coded near-optimal reliability-weighted cue integration).
- **Biased competition:** Desimone & Duncan 1995; Wong & Wang 2006 (the project's WTA core).
- **Point-neuron limit (why it does NOT apply to cue competition):** Mikulasch & Priesemann (PNAS 2021925118).
- **Phase-1 (the reused, validated mechanism):** `research/findings/2026-06-19-multicue-competition-parser-scoping.md`; `research/findings/2026-06-19-multicue-competition-derisk.md` (numpy GO 6/6); `research/findings/2026-06-19-multicue-competition-spiking-derisk.md` (spiking GO, install 5/6 + the three-factor learning finding).
- **Project machinery (reuse-by-import):** `research/runners/_phaseB_multicue_competition_spiking_derisk.py` (`SpikingRoleCompetition`, `CUES`, `learn_error_gated`, `INSTALLED_CUE_WEIGHTS`, `build_dataset`, `run_seed`), `research/runners/multicue_role_parser.py` (`MultiCueRoleParser`), `research/runners/brain_conversational_agent.py` (`enable_multicue_competition`/`hear_multicue`), `research/runners/biased_competition_buffer.py` (`ANIMACY`/`VERB_SELECTS`, the `sel_X`/`sel_FS_X` WTA). Catalog: `sim-catalog/references/feature-catalog.md` Cluster G (G.12, G.16-G.18).
