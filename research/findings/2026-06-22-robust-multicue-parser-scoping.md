# Robust multi-cue COMPETITION parser — deep-research scoping (the conversational PRIMARY, Tier-1)

**Date:** 2026-06-22
**Type:** Deep-research + catalog-review scoping (standing opening move; the Tier-1 conversational PRIMARY per
`project_conversational_primary_robust_multicue_parser` + `project_post_conversational_roadmap_tiers`). READ-ONLY;
no code built / edited / run.
**Owner directive (2026-06-19):** the conversational PRIMARY = robust, language-agnostic COMPREHENSION via a
MULTI-CUE COMPETITION parser (robust-to-imperfect-English AND language-agnostic from ONE mechanism, Bates-MacWhinney
Competition Model), prioritized ABOVE further syntax expansion. Phase 1 = robust-English, REUSING the existing
"#2 biased-competition" mechanism.

---

## 0. TL;DR — the load-bearing finding the controller must register first

**The Phase-1 (robust-English) and Phase-2 (case) multi-cue competition parsers are ALREADY BUILT, spiking-validated,
and wired into the agent behind flags (2026-06-19).** This scoping was dispatched as if the parser were the next
*build*; on reading the code it is the next *integration + hardening* job. Specifically, since the owner's directive:
- `MultiCueRoleParser` (`research/runners/multicue_role_parser.py`) — the production drop-in over the spiking
  `SpikingRoleCompetition` (the re-pointed `biased_competition_buffer.py` Wong-Wang/Rutishauser WTA over thematic
  ROLES + plastic cue→role projections). **GO, install path 5/6 seeds** (`2026-06-19-multicue-competition-spiking-derisk.md`).
- `CaseAwareRoleParser` (`research/runners/case_aware_role_parser.py`) — Phase-2, the 5th `case` cue + the
  cross-linguistic dissociation. **GO 5/6** (`2026-06-19-case-cue-crosslanguage-derisk.md`).
- Both wired into `BrainConversationalAgent` behind default-OFF flags (`enable_multicue_competition` /
  `enable_case_competition`); CI guards green; flag-OFF byte-identical.
- The on-substrate three-factor cue-validity LEARNING is firmed (correct validity signature 6/6 seeds) AND the
  reward is neuralized (spiking SNc RPE, signature 6/6) — `2026-06-19-multicue-learning-firm-and-neural-reward.md`.

**So the genuine frontier is NOT "build the multi-cue parser." It is the four concrete gaps below, in cheapest-first
order. The recommended Phase-1 cheap-first de-risk is GAP-1 (the comprehension-level no-confab moat) — it is the only
gap that touches the moat, it is hours of numpy work, and it converts a latent confabulation risk into a validated
abstention.**

| # | Gap (since the wire-in) | Why it matters | Effort |
|---|---|---|---|
| **GAP-1** | **The comprehension content-gate is NOT in `hear()`** — `hear_multicue`/`hear_case` call `parse()` (always commits a role), not `parse_decisive()`. A genuinely ambiguous degraded sentence stores a CONFABULATED fact. | Directly a moat hole at comprehension. The composer abstains on *unstored* facts but cannot un-confabulate a *wrong stored* one. | **LOW (the recommended de-risk)** |
| GAP-2 | Multi-cue is a 2-NOUN-only front-end, MUTUALLY EXCLUSIVE with `hear_attributed` (adj+noun) and `hear_multiframe` (frame). Degraded input with an adjective ("apple the big dog ate") has no robust path. | The owner's "imperfect English" includes attributed entities; today they don't compose. | MEDIUM |
| GAP-3 | NO production-default flip (every flag OFF); the default `hear()` is still position-only / onebrain-SVO. | The robustness exists but is not the shipped behavior. A deliberate flip gated on the full who/what + moat suite at production scale. | LOW–MEDIUM |
| GAP-4 | The residual seed-variance is a tiny-scale Wong-Wang WTA **object_front operating-point friction** (re-calibrate selective-inhibition gain vs pool size); the install path masks it but the *learned* path exposes it. | Bounds the end-to-end ceiling on the hardest items; a genuine operating-point study (flagged, not a quick lever). | MEDIUM (deferred) |

---

## 1. DIAGNOSIS of the current parser

### 1.1 The position-only default (`BridgeParser`) — verified
`BridgeParser` (`research/runners/brain_conversational_agent.py:28`) learns a Hebbian map from **6 conjunction units**
(`k = position*2 + voice`, `_GT` at `:25`) onto **3 role ensembles** (agent/action/patient) via the v16 embodied-Hebbian
co-firing rule (`enable_hebbian_learning=True`, `:73`; `_train` at `:110`). `role_of(position, voice)`
(`:123`) drives the conjunction ALONE and reads the max-firing role; `parse(words, voice)` (`:139`) **asserts exactly
3 words** and assigns each word the role its `(position, voice)` conjunction reads out. The on-bridge `OneBrainComposer`
parser (`one_brain_composer.py:6`, a `BridgeParser` on slice `[0:P]`) is the same mechanism co-resident.

**The single structural fact:** every default path uses **ONE cue family — serial position** (voice/frame are just
*which position→role table*). No semantic cue, no agreement/case cue, no competition. **Where it breaks (verified by
the wire-in CI test `test_default_position_only_parser_inverts_object_fronted`):**
- **Object-fronted / scrambled order** ("apple eat dog"): the position table assigns roles **backwards** (agent=apple,
  patient=dog) and **stores the inverted fact** — not a None, a *wrong commit*.
- **Dropped function words / missing arguments**: position indices shift; the `assert len(words)==3` in
  `BridgeParser.parse` (`:142`) means a 2- or 4-token input is out of scope entirely.
- **A different language's cue structure**: a free-word-order case-marked sentence (Japanese ga/wo) is unparseable by
  position — position-only comprehension IS a typological commitment to a fixed-word-order language.

### 1.2 The two failure modes are ONE failure (the Competition-Model frame is correct)
Cross-language failure and degraded-English failure are the same structural fact: the parser stakes everything on one
cue, so it dies whenever that cue is unavailable — whether because the *language* doesn't use it or the *utterance*
degraded it. (This diagnosis was confirmed in `2026-06-19-multicue-competition-parser-scoping.md` §1 and is
unchanged.)

### 1.3 The mechanism that ALREADY fixes it — verified present
The spiking `SpikingRoleCompetition` (`research/runners/_phaseB_multicue_competition_spiking_derisk.py:192`) is real:
- **2 role accumulators** `sel_agent`/`sel_patient` (`:222`, NMDA-slow Wong-Wang, `enable_nmda=True`, soft-WTA
  `sel_recurrent_weight=0.30`) in mutual inhibition via selective inhibitory pools `sel_FS_{r}` (`:228`,
  `exc_fraction=0.0` → inhibitory traits; `sel_r→sel_FS_r→sel_(s≠r)`, the Rutishauser motif, `:268–278`).
- **4 cue populations** (`CUES=("position","animacy","verbfit","lexbias")`, `:84`), each a signed pair `cue_c_pos`/`cue_c_neg`
  (`:236`), projecting through **plastic cue→role synapses** (`:280`, `plasticity_gate=f"cue_{c}"`) — the synaptic
  weights ARE the cue validities.
- Install-path validities `INSTALLED_CUE_WEIGHTS = {position 6, animacy 20, verbfit 20, lexbias 2}` (`:857`).

`MultiCueRoleParser` wraps it, installs those validities, freezes plasticity, and exposes
`parse(words,voice)→{agent,action,patient}` + `parse_decisive(...)→(roles, decisive)` (`multicue_role_parser.py:95,122`).
The verb is identified lexically from a caller-supplied `known_verbs` set (the legitimate lexical boundary,
`_split_verb_nouns` `:72`); the two nouns' roles are the spiking WTA decision.

### 1.4 The precise remaining gaps (verified in code — the real frontier)
1. **GAP-1 — the comprehension moat is not wired.** In `hear()` (`brain_conversational_agent.py:348`), when
   `enable_multicue_competition` is ON it routes to `hear_multicue` (`:332`), which calls `parse()` (`:342`) — **always
   commits a role assignment**, then `composer.store(...)`. `parse_decisive` (which reports `decisive=False` on a
   content-ambiguous sentence — two animate nouns + a symmetric verb + scrambled order, `multicue_role_parser.py:122`)
   **is never called on the production path**. So an ambiguous degraded sentence **confabulates and stores a fact**.
   The downstream composer Q&A abstains only on *unstored* facts (`one_brain_composer.py`); it cannot un-store a
   confabulated one. This is the one gap that touches the no-confab moat directly — and the cheapest to close.
2. **GAP-2 — front-ends are mutually exclusive, 2-noun only.** `hear()` dispatches exactly one of: multicue / case /
   onebrain-parser / position (`:362–369`); `hear_attributed` (adj+noun, `:380`) and `hear_multiframe` (frame, `:394`)
   are SEPARATE entry points. `MultiCueRoleParser.parse` falls back to a surface-order read for any non-2-noun input
   (`:104–114`). So degraded input *containing an adjective* has no robust path.
3. **GAP-3 — no production-default flip** (every flag default-OFF; `:362–365` skipped; the wire-in findings §5 confirm
   this is deliberate, awaiting a gated flip).
4. **GAP-4 — the WTA object_front operating-point friction** (`2026-06-19-multicue-learning-firm-and-neural-reward.md`
   §Part-1: naive levers — more epochs, more read_steps, bigger n_sel — do NOT fix it; a genuine selective-inhibition-gain
   vs pool-size re-calibration).

---

## 2. The existing "#2 biased-competition" mechanism to REUSE

**It is `BiasedCompetitionContextBuffer` (`research/runners/biased_competition_buffer.py:96`)** — promoted to
production from `_phaseB_biased_competition_derisk.py` (de-risk GO `2026-06-19-multireferent-biased-competition-derisk.md`;
the "#2" in the agent constructor's "richer-syntax #1/#2 … biased competition" comment lineage, and the wire-in
template the multicue wire-in explicitly mirrors — `2026-06-19-multicue-competition-agent-wirein.md` §6). It is the
**Desimone-Duncan 1995 biased competition + Wong-Wang 2006 attractor WTA + the Rutishauser selective-inhibition
motif** the navigation `sel_X`/`sel_FS_X` read-out already uses.

**What it is + where it lives:**
- Per held referent: a Wong-Wang accumulator pool `sel_X` (NMDA-slow, soft-WTA α<1, `:140`) + a selective inhibitory
  pool `sel_FS_X` (`exc_fraction=0.0`, `:146`); wiring `cortex_assembly[X]→sel_X` (ff evidence, read-only tap),
  `sel_X→sel_FS_X` (exc), `sel_FS_X→sel_Y≠X` (inh) (`:163–176`).
- A small CONTENT bias injector (`content_bias_target`, `:79`; the `ANIMACY`/`VERB_SELECTS` lexicons `:64,71`) — the
  crux is that the bias is a CONTENT signal (not recency, not magnitude), and the recurrence amplifies the small
  content asymmetry into a SUPPRESSIVE winner.

**How it carries multi-cue competition (the realization — already done):** the spiking de-risk
**re-points `sel_X`/`sel_FS_X` from REFERENT to ROLE** (agent/patient) and turns the single host content-bias into
**N plastic cue→role projections weighted by cue validity** (`SpikingRoleCompetition`, §1.3). The "small content bias"
becomes the *summed reliability-weighted cue drive*; the same mutual-inhibition recurrence that picks a referent picks
a role under partial evidence. **One biased-competition / normalization circuit, weighting cues by validity** — exactly
the owner's framing. The buffer's HOST-SCAFFOLD flag (`content_bias_target` → "a LEARNED SYNAPTIC FEATURE-COMPATIBILITY
MAP", `:33–39`) is the *same conversion target* as the parser's learned cue→role weights.

---

## 3. REFRAME via the biology (Competition Model → spiking biased-competition / divisive-normalization)

### 3.1 The Competition Model (Bates & MacWhinney 1982/1989) maps cleanly onto the substrate
A finite inventory of **cues** (word order, agreement, case, animacy, selectional plausibility) each carries
role information; each has a **cue validity** = availability × reliability in a language; the comprehender's cue
**weights** are *learned to track cue validity*; at comprehension cues **compete** and the role assignment is the
**settled winner**. Degrade a high-weight cue → lower-weight cues carry it (graceful degradation). Switch languages →
re-learn the weights (English=order high; Japanese=case high) = the "adapt by re-learning, not re-coding" property.

The neural realization (verified against the catalog + current literature):
- **Cue validity = the cue→role synaptic weight** (a learned reliability scaling). Catalog **G.18** (Probabilistic
  reasoning, LIP logLR; `feature-catalog.md:2850`): *"LIP's accumulator is not specific to perceptual evidence; it
  integrates ANY evidence weighted by reliability. Each [source] contributes its known logLR additively."* This is
  cue-validity-weighted integration as a firing-rate computation. (Sim status: missing as a primitive — the parser is
  the first realization.)
- **The competition/settle = biased competition + an accumulator-to-bound.** Catalog **G.16/G.17** (drift-diffusion /
  LIP accumulator, `:2826,2838`): *"accumulator integrates the difference over time; decision terminates when it hits
  ±bound"*, *"two anti-correlated accumulators terminate at first-bound-crossing"* — the project's BG cascade is
  *"functionally equivalent to a bounded accumulator"*. Catalog **A.04** (GPi/SNr selective disinhibition, `:128`):
  *"Selection is an emergent property of the entire reentrant network"* (focused-vs-broad inhibition = the biased-competition
  motif, though "Desimone-Duncan" terminology is not in the catalog).
- **Why semantic cues carry non-canonical comprehension = the textbook statement of the mechanism.** Catalog **G.12**
  (Broca, `:2774`): behavioral validation *"'The girl that the boy is chasing is tall' comprehension fails
  (grammar-dependent); 'The apple the girl ate was green' succeeds (semantically constrained)"* — i.e. when the syntactic/order
  cue is hard, semantic cues carry comprehension. That is multi-cue competition, stated by Kandel.

### 3.2 Divisive normalization as the cue-integration circuit — the owner's question, answered
**Current literature (2024–2025) directly grounds "cue competition as a divisive-normalization circuit," AND draws the
honest point-neuron line:**
- **Point-neuron / rate-code version is feasible and stable.** A 2025 *unified cortical circuit* combines **divisive
  normalization + self-excitation → a continuous attractor** for robust encoding + stable retention (arXiv 2508.12702);
  and **divisive-normalization recurrent circuits are UNCONDITIONALLY stable** for arbitrary dimensionality
  (ORGaNICs, PMC11469413, 2025). This is the WTA-stability theory the project's
  Wong-Wang/Rutishauser WTA approximates — and it is exactly the lever for **GAP-4** (the object_front operating-point
  friction is a *calibration* of the normalization/selective-inhibition gain, NOT a substrate wall).
- **Bayes-OPTIMAL graded cue weighting is DENDRITIC.** "Conductance-based dendrites perform Bayes-optimal cue
  integration" (PMC11168673, *PLoS Comput Biol* 2024; arXiv 2006.15099) shows the *crucial ingredient* is **divisive
  normalization of compartmental membrane potentials via conductance-based synapses** — a *multi-compartment* property.
  This is the honest **point-neuron caveat** (the Mikulasch-Priesemann family): *optimal, finely-graded* cue weighting
  wants dendrites. **But Phase-1 does NOT need it** — the project's cues are **near-binary** (animate/inanimate; fits/doesn't;
  particle present/absent), so the rate-coded point-neuron WTA suffices (which is precisely what the spiking de-risk
  found: GO 5/6, the residual being WTA calibration, not graded-cue resolution). The dendritic version is the
  Tier-4 / D2-Phase-3 upgrade *if a finely-graded plausibility cue is ever the blocker* — not a Phase-1 requirement.
- **Reliability-weighting is LEARNED from stimulus statistics by crossmodal plasticity** (PMC9393257) — the project's
  three-factor cue-validity learner is the analogue (already firmed 6/6 signature).

### 3.3 The field precedent + the gap the project fills (verified, current)
- **NEMO / Assembly Calculus parser** (Mitropolsky et al., arXiv 2108.02189; Papadimitriou et al. "Language Organ",
  arXiv 2306.15364; "Simulated Language Acquisition", biorxiv 2025.07.15.664996) — Role areas under MUTUAL INHIBITION,
  order learned by Hebbian plasticity, **struggles with object-initial orders**, **single-cue (order only; no animacy/case)**.
  This is *(a)* a positive precedent that thematic-role-by-biased-competition + Hebbian order works on a spiking
  substrate, and *(b)* exactly the single-cue brittleness the project's multi-cue competition fixes.
- **A neurobiologically-inspired sentence-comprehension model (2025)** (Crocker et al., *Lang. Cogn. Neurosci.*,
  tandfonline 10.1080/23273798.2025.2473537) — lexicon + syntax + semantics modules; confirms the field is converging
  on exactly the multi-module, cue-integrating comprehension architecture the project builds in spikes.

---

## 4. RANKED biologically-grounded options for Phase 1 (robust-English), cheapest-first

All reuse the shipped `MultiCueRoleParser` + `SpikingRoleCompetition` + `biased_competition_buffer`; none anticipates
a `sim/` edit (additive runner/agent wiring, the established pattern).

| Rank | Option | Expected payoff | Reuses |
|---|---|---|---|
| **1 ★** | **GAP-1: wire the comprehension content-gate into `hear()`** — route the production `hear_multicue`/`hear_case` through `parse_decisive`; on `decisive=False` (content-ambiguous degraded sentence), **ABSTAIN at comprehension** (don't store; return a no-commit marker) instead of confabulating a role. | Closes a concrete moat hole at the comprehension layer (today a wrong fact is *stored*; the composer can't un-store it). Strengthens the no-confab moat where it is currently silent. **Hours, numpy.** | `MultiCueRoleParser.parse_decisive` (`:122`, already built), the `hear()` dispatch (`:362`). |
| 2 | **GAP-2: unify the front-ends** — one `hear()` that runs the multi-cue *role* competition AND admits an adjective (route the non-noun-role tokens to `hear_attributed`'s attribute role after the role WTA settles), so degraded *attributed* input ("apple the big dog ate") parses. | Extends robustness to the attributed-entity case the owner's "imperfect English" includes; removes the mutually-exclusive-front-end limitation. **Days.** | `AttributedBridgeParser` (`attributed_parser.py`), `MultiCueRoleParser`, the composer's attribute role (`hear_attributed` `:380`). |
| 3 | **GAP-3: gated production-default flip** — flip `enable_multicue_competition` ON in the production demos (not the library constructor default, to preserve CPU/numpy portability), gated on the FULL who/what + moat + clean-canonical suite at production V=320. | Ships the robustness as the default behavior. **Low–medium** (mostly validation + the deliberate flip + a regression gate). | The wire-in flag + CI guards; the 320-scale demo harness. |
| 4 | **GAP-4: re-calibrate the WTA** (deferred) — selective-inhibition gain (`fs_to_sel_weight`) vs accumulator pool size (`n_sel`) study against the divisive-normalization stability theory (ORGaNICs) to lift the object_front ceiling on the *learned* path. | Raises the end-to-end ceiling on the hardest items (object_front), unifying install-path and learned-path robustness. **Medium; a genuine operating-point study, deep-research-gated.** | The `SpikingRoleCompetition` layout knobs; ORGaNICs stability result (PMC11469413). |

**Recommended immediate build = Option 1.** It is the cheapest, it is the only one that touches the moat (and the
project's hardest rule is "never weaken the moat"), and it converts a latent confabulation into a validated abstention
— the highest reward-per-effort and the most defensible first step.

---

## 5. REUSABLE project machinery (what transfers)

| Need | Reuse from | Status |
|---|---|---|
| The spiking multi-cue role competition | `multicue_role_parser.py` (`MultiCueRoleParser`) over `_phaseB_multicue_competition_spiking_derisk.py` (`SpikingRoleCompetition`, `cue_evidence`, `INSTALLED_CUE_WEIGHTS`) | **Built, GO 5/6** |
| The "#2 biased-competition" WTA substrate | `biased_competition_buffer.py:96` (`sel_X`/`sel_FS_X` Wong-Wang + Rutishauser; `ANIMACY`/`VERB_SELECTS`) | **Built, GO** |
| The comprehension content-gate (the moat at parse time) | `MultiCueRoleParser.parse_decisive` (`:122`) + `_default_margin` (`:140`) | **Built, NOT wired into `hear` (GAP-1)** |
| The case (Phase-2) cue + cross-linguistic dissociation | `case_aware_role_parser.py` (`CaseAwareRoleParser`) | **Built + wired, GO 5/6** |
| On-substrate cue-validity LEARNING + spiking RPE reward | `_phaseB_multicue_competition_spiking_derisk.py` (`learn_error_gated`; the `snc` pool) | **Built; signature 6/6 (host + spiking RPE); end-to-end readout-bounded (GAP-4)** |
| The agent's flag + dispatch + CI guards | `brain_conversational_agent.py` (`enable_multicue_competition` / `enable_case_competition`, `hear` `:348`); `tests/test_multicue_competition_agent.py`, `tests/test_case_cue_crosslanguage_agent.py` | **Built** |
| The regions framework / transmission-gate / divisive-norm primitives | `sim/regions.py` (`BrainRegion`/`RegionPathway`, `transmission_gate`), `cp_plasticity_rate_gain` (plasticity gates), `_phaseC_S5_divnorm_derisk.py` (divisive-norm score bridge) | Shipped |
| The attributed + multiframe front-ends (for GAP-2) | `attributed_parser.py` (`AttributedBridgeParser`), `frame_parser.py` (`FrameParser`/`MultiFrameParser`) | Built (separate entry points) |
| **The no-confab MOAT (MUST NOT be weakened)** | the composer's `query`-time abstention (`one_brain_composer.py`) + `parse_decisive` content gate | Active downstream; **GAP-1 adds it at comprehension** |

---

## 6. The CHEAP-FIRST de-risk (Option 1 / GAP-1) — numpy/CPU, with anti-cheats + the GO bar

**Question it decides:** *When the production `hear()` runs the multi-cue competition on a genuinely ambiguous degraded
sentence (two animate nouns + a symmetric verb, scrambled — no decisive content cue), does routing through
`parse_decisive` make the agent ABSTAIN at comprehension (store nothing / return a no-commit marker) instead of
confabulating and storing a role assignment — WITHOUT regressing the decisive degraded cases or clean canonical, and
without ever weakening the existing query-time moat?*

### 6.1 Setup (smallest CPU/numpy, hours)
- `BrainConversationalAgent(composer_kind="rf", enable_multicue_competition=True, multicue_verbs={...}, concepts=<explicit
  vocab so denoise64 cache is not needed>)` (the CPU-runnable pattern the existing CI guards use).
- Add `decisive`-aware routing in `hear_multicue` (and `hear_case`): call `parse_decisive`; if `decisive=False`, **do
  NOT** `composer.store(...)` — return a sentinel (e.g. `None` / `{"_abstain": True}`); else store as today.
- Test set: (a) DECISIVE degraded ("apple eat dog" — inanimate patient → resolvable by content); (b) AMBIGUOUS degraded
  ("wolf chase dog" scrambled — two animate + symmetric verb → genuinely undecidable); (c) clean canonical
  ("dog eat apple"). Held-out fillers vs the install-path lexicon where possible (role correctness is vocab-agnostic).

### 6.2 GO / NEGATIVE bar (pre-register; FROZEN; ≥6 seeds; fractional ≥5/6)
- **GO** requires ALL of:
  1. **AMBIGUOUS degraded → ABSTAIN at comprehension** (no fact stored; a follow-up who/what returns None) on **≥5/6
     seeds**, with **0 confabulated stores** across the ambiguous set (the moat at comprehension).
  2. **DECISIVE degraded → still resolved + stored correctly** (the object-fronted win is NOT lost): who/what correct on
     ≥5/6 seeds (no regression vs the current `parse()` path on the decisive items).
  3. **Clean canonical → unregressed** (who/what correct, fact stored).
  4. **The query-time moat is preserved** (an unstored fact still abstains; 0 query-time breaches) — i.e. GAP-1 ADDS a
     comprehension-layer abstention without removing the downstream one.
  5. **Flag-OFF byte-identical** (the existing `test_brain_conversational_agent` + `test_multicue_competition_agent`
     pass verbatim).
- **NEGATIVE** = either the content gate fails to fire on the ambiguous set (it confabulates anyway → the margin
  calibration `_default_margin` is wrong for the production path), OR closing the gate also kills the *decisive*
  degraded cases (the gate is too aggressive → over-abstention), OR it perturbs clean canonical. Report honestly; the
  fallback is a margin re-calibration (a 1-knob study on `_default_margin`/`abstain_margin`), NOT a mechanism change.

### 6.3 Anti-cheat controls (each must pass or it is not a GO)
1. **Margin-LESION (the decisive control):** set `abstain_margin=0` (the gate can never fire). The ambiguous set must
   then **confabulate + store** (reproducing today's GAP-1 hole) — proving the abstention is *caused by the gate*, not
   by the parser silently failing on ambiguous input. (If it abstains anyway with margin=0, the "win" is an artifact.)
2. **PERMUTED-CUE:** drive the content gate with a scrambled animacy/verb-fit assignment (animate→patient). The
   *decisive* set must collapse toward the ambiguous (abstain) outcome — proving the gate reads *real* content
   contrast, not a relabelled position signal.
3. **Decisive-vs-ambiguous CONTRAST (no over-abstention):** assert the *same surface form* is decisive WITH a real
   content asymmetry and non-decisive WITHOUT it (the `test_multicue_moat_ambiguous_sentence_not_decisive` pattern,
   extended to the *production hear path* rather than the parser in isolation) — the gate must DISCRIMINATE, not blanket-abstain.
4. **Held-out fillers / 0 confabulation:** the ambiguous abstention holds on fillers unseen at install, and the count of
   confabulated stores on the ambiguous set is **exactly 0** (a hard moat assertion, the project's load-bearing
   standard).
5. **Moat-never-weakened invariant:** the query-time false-accept count on unstored facts stays **0** (GAP-1 must be
   strictly additive to the moat).

### 6.4 If GO → next steps (in order)
Promote the `decisive`-aware `hear()` behind the existing flag (still default-OFF), multi-seed GPU-validate the full
who/what + moat at production V=320, then proceed to GAP-2 (unify front-ends) → GAP-3 (gated default flip) →
GAP-4 (WTA re-calibration, deep-research-gated). Each banks behind the existing flag pattern; no mechanism is new.

---

## 7. Honest risks + the point-neuron line
- **Hand-tuned-cues masquerading as learned** — already guarded by the de-risk's PERMUTED-CUE + NO-LEARNING +
  cue-LESION controls (all passed; `2026-06-19-multicue-competition-spiking-derisk.md` §4). GAP-1's de-risk re-uses
  the permuted-cue control (§6.3.2).
- **The point-neuron risk for cue competition is LOW** (§3.2): reliability-weighted accumulation to a winner is a
  rate-code/attractor operation (stable per ORGaNICs); it is categorically NOT the analog/pre-spike *decorrelation/whitening*
  that walled before. The genuinely dendritic version (Bayes-optimal graded weighting, PMC11168673) is a Tier-4 upgrade
  *only if a finely-graded plausibility cue becomes the blocker* — Phase-1's near-binary cues do not need it.
- **The honest residual is GAP-4** (the tiny-scale WTA object_front operating-point friction) — flagged, not escalated;
  it bounds the *learned*-path end-to-end ceiling, not the install-path deployment, and is a calibration study
  (ORGaNICs stability) not a substrate wall.

## 8. Provenance (verified)
- **Project code (file:line verified this scoping):** `research/runners/brain_conversational_agent.py` (`BridgeParser`
  :28, `hear` :348, `hear_multicue` :332, `hear_attributed` :380, `hear_multiframe` :394, dispatch :362–369);
  `research/runners/multicue_role_parser.py` (`parse` :95, `parse_decisive` :122, `_default_margin` :140);
  `research/runners/_phaseB_multicue_competition_spiking_derisk.py` (`SpikingRoleCompetition` :192, `CUES` :84,
  `cue_evidence` :134, `INSTALLED_CUE_WEIGHTS` :857); `research/runners/biased_competition_buffer.py`
  (`BiasedCompetitionContextBuffer` :96, `content_bias_target` :79, `resolve_referent` :309);
  `research/runners/case_aware_role_parser.py`; `research/runners/one_brain_composer.py`.
- **Prior findings:** `2026-06-19-multicue-competition-parser-scoping.md`, `-multicue-competition-derisk.md`,
  `-multicue-competition-spiking-derisk.md`, `-multicue-competition-agent-wirein.md`,
  `-multicue-learning-firm-and-neural-reward.md`, `-case-cue-crosslanguage-derisk.md`,
  `-case-cue-crosslanguage-agent-wirein.md`, `-multireferent-biased-competition-derisk.md`.
- **Catalog (`sim-catalog/references/feature-catalog.md`):** G.18 (logLR reliability-weighted accumulation, :2850),
  G.16/G.17 (drift-diffusion / LIP accumulator, :2826/:2838), G.12 (Broca; semantic cues carry non-canonical
  comprehension, :2774), A.04 (selective disinhibition WTA, :128), E.03 (population coding / Bayesian decode, :1367),
  E.05 (lateral inhibition / decorrelation, :1391). **Catalog gaps noted:** no explicit "biased competition"
  (Desimone-Duncan), "divisive normalization" (Carandini-Heeger), "multisensory Bayesian cue combination", or
  "Rutishauser α>1 WTA stability" entry — candidate new catalog entries surfaced by this work. No `glossary.md` in
  that references/ dir.
- **Competition Model:** Bates & MacWhinney 1982/1989; MacWhinney-Bates-Kliegl 1984 (English=order, German=agreement,
  Italian=agreement). Good-enough/NVN: Ferreira 2003; Ferreira & Patson 2007.
- **Current literature (2024–2025, WebSearch):** divisive-normalization + self-excitation → continuous attractor
  (arXiv 2508.12702, 2025); ORGaNICs unconditional stability (PMC11469413, 2025); conductance-based DENDRITES do
  Bayes-optimal cue integration via compartmental divisive normalization (PMC11168673 / arXiv 2006.15099, 2024 — the
  point-neuron caveat); crossmodal-plasticity reliability weighting (PMC9393257); neurobiologically-inspired sentence
  comprehension (tandfonline 10.1080/23273798.2025.2473537, 2025); NEMO / Assembly Calculus parser (arXiv 2108.02189,
  2306.15364; biorxiv 2025.07.15.664996).
