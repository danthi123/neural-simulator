# Multi-cue Competition Parser — language-agnostic + degraded-input-robust comprehension (scoping)

**Date:** 2026-06-19
**Type:** Deep-research + catalog-review scoping (standing opening move for a NEW DIRECTION). READ-ONLY; no code built/edited/run.
**Owner directive (2026-06-19):** make conversational comprehension LANGUAGE-AGNOSTIC and ROBUST to imperfect input.
**Status of this doc:** diagnosis → verified neural mechanism → reuse-vs-new map → phase order (reward-vs-effort) → Phase-1 cheapest-first de-risk (the immediate next build) + GO bar + load-bearing control → honest point-neuron risk + stop rule. Controller reviews load-bearing claims before any build.

---

## 0. TL;DR (the recommendation)

- **The hypothesis is CORRECT and well-grounded.** The current parser assigns thematic roles by WORD POSITION (`(position × voice) → role`). Its two brittleness modes — (1) can't handle case/morphology-marked languages, (2) mis-parses dropped-word / scrambled / degraded English — are **the same problem**, and the **Competition Model** (Bates & MacWhinney) is the right frame: word-order + animacy + verb-selectional-fit + (later) case/agreement **cues compete with learned weights** to assign roles; degrade one cue, the others carry it. Robustness AND language-agnosticism fall out of one mechanism. Adapting to a new language/register = **re-learning the cue weights from its data** (no per-language hand-coding) = the "easily adaptable" goal.
- **The verified neural realization is the project's own existing mechanism.** Weighted multi-cue role assignment = **biased competition** (Desimone-Duncan 1995) over role assemblies + an **evidence-accumulator** that sums **reliability-weighted cue evidence** (catalog **G.18**: the LIP accumulator "integrates *any* evidence weighted by reliability", each source contributing additively — this is exactly cue-validity-weighted integration, realized as a firing-rate computation). The project already has BOTH pieces on the substrate: the **navigation Wang-2002 accumulator + Rutishauser selective-inhibition WTA**, and — closest of all — **`research/runners/biased_competition_buffer.py`** (`BiasedCompetitionContextBuffer`: mutual inhibition between assemblies + a small CONTENT bias from an animacy × verb-selectional-restriction lexicon). That buffer does this competition **for referents today**; the proposal re-points the *same wiring* at **cues**.
- **Point-neuron risk is LOW and clean.** Weighted-cue-→-winner is a **firing-rate competitive-dynamics** computation (attractor networks do near-Bayesian reliability-weighted cue integration with rate codes — PNAS 2214441119; eLife 20047). That is categorically NOT the analog/pre-spike **decorrelation/whitening** that walled before (Mikulasch-Priesemann point-neuron limit). The honest risk is elsewhere: **do the cue WEIGHTS actually LEARN** (else it's hand-tuned), which the Phase-1 de-risk decides directly with a lesion + a no-learning control.
- **Phase order (confirmed, my call held):** **Phase 1 = robust ENGLISH multi-cue competition** (handles dropped words / scrambled order / non-canonical order — the owner's use case; ~90% reuse) FIRST; **Phase 2 = case/agreement marking + a non-English toy** (true cross-language); **Phase 3 = sub-word morphology** (agglutinative langs) DEFERRED (the only months-scale NEW layer; English does not need it).
- **The immediate next build = the Phase-1 cheapest-first de-risk** (CPU/numpy, hours): does adding competing cues (animacy + verb-fit) to the position parser make role-assignment **robust to degraded word order, where the position-only parser collapses**? GO bar + the load-bearing position-only control are specified in §6.

---

## 1. Diagnosis — why the position parser is brittle, and why it's ONE problem

### 1.1 What the parser does today
`BridgeParser` (`research/runners/brain_conversational_agent.py:28`) learns a Hebbian map from **6 conjunction units** (`k = position*2 + voice`, voice ∈ {active, passive}) onto **3 role ensembles** (agent/action/patient). `role_of(position, voice)` drives the conjunction and reads the max-firing role; `parse(words, voice)` assigns each word the role its `(position, voice)` conjunction reads out. `FrameParser` (`research/runners/frame_parser.py`) generalizes the structural cue to **verb-position → frame** (SVO/VSO/OSV) and `MultiFrameParser` learns `(position × frame) → role`. **Every one of these uses exactly ONE structural cue family: serial position** (voice and frame are just *which position-→-role table* to apply). There is no semantic cue, no agreement cue, no competition.

### 1.2 The two failure modes are the same failure
- **Cross-language failure.** Position-only comprehension is a *typological commitment to a fixed-word-order language*. Case-marking languages (Japanese -ga/-o, Russian/Turkish case, agreement-heavy Italian/German) signal roles by **morphology, not order** — the very cue the project's parser ignores. The canonical Competition Model result (Bates, MacWhinney, et al.; MacWhinney-Bates-Kliegl "Cue validity and sentence interpretation in English, German, and Italian", *JVLVB* 1984): **English speakers rely overwhelmingly on word order; Germans on agreement + animacy; Italians extremely on agreement.** A position-only parser IS an English-only parser.
- **Degraded-English failure.** Real users drop words ("dog north"), scramble order ("north the dog goes"), front objects ("the bone, the dog ate"), typo, and use non-canonical voice. Each of these **corrupts the position cue**. With position as the *sole* cue, a corrupted position → a corrupted (or absent) role assignment. The position parser has no fallback.

**Both are the same structural fact:** the parser stakes everything on one cue, so it dies whenever that one cue is unavailable — whether because the *language* doesn't use it (cross-language) or the *utterance* degraded it (imperfect input). The Competition Model's answer is identical in both cases: **carry the role assignment on whatever OTHER cues survive.**

### 1.3 The Competition Model in one paragraph (verified frame)
(Bates & MacWhinney 1982/1989; MacWhinney 2005 "Extending the Competition Model".) A finite inventory of **cues** (word order, noun-verb agreement, case marking, animacy, stress/prosody, semantic/selectional plausibility) each carries information about thematic roles. Each cue has a **cue validity** = availability × reliability in a given language. The comprehender's cue **weights** are *learned to track cue validity in the input language*. At comprehension time the cues **compete**; the role assignment is the **settled winner of the weighted competition**. When a high-weight cue is degraded or absent, lower-weight cues determine the outcome (graceful degradation). This is why English-learners weight order high and Italian-learners weight agreement high — **same architecture, different learned weights** = exactly the "adapt to a new language by re-learning weights, not re-coding" property the owner wants.

### 1.4 The neural realization is already a named, validated computation — the catalog confirms it
The crux is *how a brain integrates competing cues to settle on a role*. The catalog (`sim-catalog/references/feature-catalog.md`, Cluster G — Working memory / PFC / cortical integration) supplies the load-bearing entries:

- **G.12 Broca's area — "supports comprehension of grammatically complex (non-canonical) sentences."** Its behavioral validation IS the Competition Model dissociation: *"The girl that the boy is chasing is tall" comprehension fails (grammar-dependent); "The apple the girl ate was green" succeeds (**semantically constrained**)."* That is the textbook stating directly: **when the syntactic/order cue is hard, semantic (selectional/plausibility) cues carry comprehension** — i.e. multi-cue competition is the brain's documented mechanism for robust role assignment.
- **G.18 Probabilistic reasoning from symbols — logLR accumulation in LIP.** "LIP's accumulator is **not specific to perceptual evidence; it integrates *any* evidence weighted by reliability.** Each [source] contributes its known logLR additively." This is **cue-validity-weighted integration**, realized as a neural firing-rate accumulator. The cue weight = the cue's reliability (logLR scaling); the role decision = the accumulator's bounded winner.
- **G.16 / G.17 Drift-diffusion / LIP accumulator.** "Two anti-correlated accumulators... terminate at first-bound-crossing" — the competitive WTA that settles the decision. Catalog notes the project's **BG cascade is functionally a bounded accumulator** already.
- **G.06 / G.08 PFC working memory — recurrent attractor dynamics**, the substrate that *holds* the competing role hypotheses while they settle.

**Conclusion:** the multi-cue parser is not a new mechanism that needs inventing — it is **biased competition over role assemblies, fed by a reliability-weighted evidence accumulator**, both of which the catalog documents and the project already runs on the substrate (navigation + the biased-competition buffer). The cue *weights* are the cue *validities*, learned by Hebbian co-firing (the same v16 rule the parser already uses).

---

## 2. Mechanism map — the multi-cue competition realized in spikes

### 2.1 The architecture (what's new is small; the parts exist)
```
                cue populations                  role assemblies
                (one per cue family)             (agent / action / patient)
   word i ->  [ POSITION cue unit_i ] --w_pos-->  ┌───────────────┐
              [ ANIMACY cue unit_i  ] --w_anim-->  │  agent (R)    │◄─┐
              [ VERBFIT cue unit_i  ] --w_vfit-->  │  action (R)   │  │ mutual
              [ (Phase2) CASE unit_i] --w_case-->  │  patient (R)  │◄─┘ inhibition
                                                   └───────────────┘   (biased competition,
                     learned cue->role weights          ▲                Rutishauser sel-FS)
                     (= cue VALIDITY, Hebbian)           │
                                              evidence ACCUMULATOR per role
                                              (Wang-2002 NMDA integrator; sums the
                                               reliability-weighted cue drive; the
                                               role that crosses bound = the assignment)
```
- **Cue populations.** Each cue family is a small population whose firing encodes *that cue's vote* for word i. Position cue = the existing conjunction unit (it already votes a role from serial position). Animacy cue = "this word is animate/inanimate" (a lexical-feature population). Verb-selectional-fit cue = "given the verb, does this filler fit the agent vs patient slot" (a relational population). These are the **same content signals** the biased-competition buffer already computes (animacy + verb-selectional-restriction); here they vote for a *role*, not a *referent*.
- **Learned cue→role weights = cue validity.** Each cue population projects to the role assemblies through **plastic synapses trained by Hebbian co-firing** (the v16 rule, `enable_hebbian_learning=True`, the parser's own training loop). A cue that reliably co-fires with the correct role in the training language grows a strong weight (high cue validity); an unreliable cue stays weak. **Switching languages = re-running this Hebbian training on that language's data → the weights re-settle to that language's cue validities.** No per-language hand-coding — this is the "easily adaptable" mechanism, and it's the load-bearing claim the de-risk must prove (the weights LEARN, not hand-set).
- **Biased competition + accumulation = the settle.** The role assemblies are in **mutual inhibition** (each role's assembly recruits a selective FS pool that suppresses the other roles — the Rutishauser `sel_FS_X` motif the navigation read-out and the biased-competition buffer both use). A per-role **NMDA-slow accumulator** (Wang-2002) integrates the summed weighted cue drive; the recurrence amplifies a small cue-asymmetry into a clean SUPPRESSIVE winner. **This is precisely where robustness comes from:** if the position cue is degraded (its vote is weak/absent), the surviving cues' votes still drive their role's accumulator past the others → the right role still wins. The same dynamic that makes biased competition pick a referent makes it pick a role under partial evidence.

### 2.2 Why this yields BOTH robustness and language-agnosticism (one mechanism, two payoffs)
- **Robustness to degraded input.** Degrading a cue = removing one summand from a *sum of reliability-weighted votes*. As long as the surviving cues' summed weighted vote still separates the correct role, the accumulator/WTA still settles correctly. (PNAS 2214441119 / eLife 20047: attractor accumulators perform **near-optimal reliability-weighted cue integration with rate codes**, and degrade gracefully as cues drop.) The Ferreira "good-enough" / **NVN heuristic** literature is the human signature of exactly this: when structure is hard, comprehenders fall back on the semantic (animacy/plausibility) cues — sometimes producing the *human error* "The mouse was eaten by the cheese" → mouse-as-agent. **A correct multi-cue model reproduces both human robustness AND the human error pattern** — a strong validation hook (§6).
- **Language-agnosticism.** A new language has different cue validities. Re-learning the weights re-settles which cue dominates the competition. English data → high `w_pos`; Japanese data → high `w_case` (and `w_pos` drops out because Japanese order is freer). The *competition architecture is identical*; only the learned weights differ. This is the Competition Model's central cross-linguistic claim (English/German/Italian dissociation), realized as Hebbian weight learning.

### 2.3 The biology anchors (verified)
- **Desimone-Duncan 1995 biased competition** — the WTA-by-mutual-inhibition core (already in the buffer + the navigation read-out).
- **Catalog G.18 / G.16 / G.17** — reliability-weighted additive evidence accumulation in the LIP-style accumulator (the project's BG cascade is functionally this).
- **Catalog G.12 (Broca)** — the textbook statement that semantic cues carry comprehension when syntax is hard; the Competition Model dissociation as the validation.
- **Hagoort MUC (Memory-Unification-Control; PubMed 23874313; Baggio-Hagoort N400 dynamic account)** — the *integration-by-current-summation* picture: frontal (unification) injects currents that combine with circulating temporal (memory/lexical) activity at each word; **N400/P600 are the cue-conflict signals** (the ERP signature when cues disagree). This is the macro-circuit framing for a future Broca/Wernicke split; not needed for Phase 1.
- **NEMO / Assembly Calculus parser (Papadimitriou-Vempala-Dabagia-Mitropolsky; biorxiv 2025.07.15.664996 "Simulated Language Acquisition in a Biologically Realistic Model of the Brain"; arxiv 2507.11788).** The field's leading biologically-plausible parser uses **Role areas under MUTUAL INHIBITION** for thematic roles ("who is doing what to whom"), learns constituent order (SVO/SOV/VSO) by **Hebbian plasticity**, and **struggles with object-initial orders (OSV/OVS)**. **Two takeaways:** (a) it is a *positive precedent* that thematic-role assignment by biased competition + Hebbian order-learning works on a biologically-plausible spiking substrate (de-risks the substrate); (b) it is *exactly the gap the project fills* — NEMO relies on **word order + pre-grounded scene analysis and does NOT integrate animacy/case cues**, so it has the *same single-cue brittleness*. The project's multi-cue competition is a principled extension of the strongest existing model.

---

## 3. Reuse-vs-new — what transfers, the minimal new wiring

### 3.1 Transfers almost entirely (reuse-by-import / additive)
| Need | Reuse from | What transfers |
|---|---|---|
| Mutual-inhibition WTA over assemblies | `biased_competition_buffer.py:96` `BiasedCompetitionContextBuffer` | The **entire competition substrate**: `sel_X` Wong-Wang accumulator pools + `sel_FS_X` selective inhibitory pools (`exc_fraction=0.0` → inhibitory traits → out-synapses route to `g_i`) + the `sel_X→sel_FS_X→sel_Y!=X` wiring + the `bias(concept,pA)` injector + the substrate facts already handled (E/I sign = pre-neuron trait; re-present competitors during read). **Re-point `sel_X` from "referent X" to "role X"** (agent/action/patient). This is the single biggest reuse — the competition is already built and de-risked. |
| The CONTENT cue lexicons | `biased_competition_buffer.py:64` `ANIMACY`, `:71` `VERB_SELECTS`, `:79` `content_bias_target` | The **animacy** feature lexicon + **verb-selectional-restriction** lexicon + the helper that returns which candidate the content selects. Today it picks a *referent* for a pronoun; the parser uses the same lexicons to score *which role* a filler fits given the verb. **Already flagged HOST-SCAFFOLD for conversion to a learned synaptic feature-compatibility map** — the parser's learned cue→role weights ARE that conversion (the cue's vote becomes a synaptic projection, not a host lookup). |
| The position cue + Hebbian cue→role training | `brain_conversational_agent.py:28` `BridgeParser` (+ `_phaseB_multiframe_comprehension_derisk.py` `MultiFrameParser`, `frame_parser.py` `FrameParser`) | The **position cue population** (the conjunction units), the **role ensembles**, and the **v16 Hebbian co-firing training loop** (`_train`: teacher-drive cue + correct role; the validated rule). The new cue families are *additional projections into the same role ensembles*, trained by the same loop. Multi-frame already proved `(position × frame) → role` is learnable and productive (GO 6/6) — adding non-position cues is the same shape. |
| Holding role hypotheses while they settle | `content_selection_spiking.py:66` `build_loop_wm_bridge` / `SpikingLoopContextBuffer` | The cortex_ctx↔dlpfc_wm recurrent loop + per-pattern attractors + the `update()`/`read()` window machinery the buffer subclasses. |
| The no-confab moat | the existing familiarity gate / `resolve_referent` abstain logic (`biased_competition_buffer.py:309`) | Abstain when the competition has **no decisive winner** (tie → margin below threshold) or **nothing is held** — reuse verbatim: a parser that can't settle a role abstains rather than confabulating one. |
| The structural cue (which order/frame) | `_phaseB_frame_selection_derisk.py` `FrameSelector` (verb-position → frame, Hebbian) | The verb-position cue → frame selector; in the multi-cue view this is *one cue's vote* (the order cue) rather than a hard gate — but the existing GO selector is a drop-in starting point for the order cue. |

### 3.2 The minimal NEW wiring (small; additive; flag any `sim/` touch for byte-review)
1. **Two new cue→role projection families** into the existing role ensembles: `animacy_cue_unit_i → {agent,action,patient}` and `verbfit_cue_unit_i → {agent,action,patient}`, **plastic, Hebbian**, trained by the existing `_train` loop. (Pure `set_pathway_weights(add_missing=True)` + the framework — the pattern `BridgeParser`/`MultiFrameParser` already use. **No `sim/` edit.**)
2. **A small driver** that, given a sentence, lights each cue population for each word: position cue (already), animacy cue (lexicon lookup → drive the animate/inanimate unit), verb-fit cue (verb + candidate filler → selectional lexicon → drive the fit unit). The cue *firing* is on-substrate; *which lexical feature a word has* is a legitimate lexical/morphology front-end (the same boundary the existing verb-lexicon lookup and `content_bias_target` already occupy — and the same boundary NEMO uses). **Flag for conversion:** the feature lexicons are host scaffolds (teaching the cue values); the validated win is the spiking competition. The follow-on neuralizes the feature lookup into a learned lexical-feature map (already the buffer's documented conversion target).
3. **Re-point** the buffer's `sel_X`/`sel_FS_X` from referent-indexed to role-indexed (3 roles). Mechanical rename of the construction loop.

**Net:** Phase 1 is **assembly of validated parts + 2 plastic cue projections + a cue driver**, all reuse-by-import / additive, **NO `sim/` edit anticipated**. (If a `sim/` edit turns out necessary — e.g. a per-cue gain — it gets a byte-level diff review per the standing rule.)

---

## 4. Phase order — reward vs effort (CONFIRMED; my call held)

The project's cadence is **deep-research-scope → cheapest-first de-risk → multi-seed validate → integrate (default-OFF) → production flip**. Each phase below follows it.

### Phase 1 — Robust ENGLISH multi-cue competition  ★ DO FIRST
- **What:** position + animacy + verb-fit cues compete (learned weights) to assign roles; comprehension survives **dropped words, scrambled order, non-canonical/object-fronted order, degraded grammar** — the owner's stated use case (real users producing imperfect English).
- **Reward:** HIGH + IMMEDIATE. Directly fixes the headline brittleness; every downstream conversational capability (who/what Q&A, negation, dialogue, multi-hop) inherits the robustness because they all consume the parser's role output. Reproduces the human "good-enough"/NVN robustness (and error) signature — a strong artificial-life deliverable.
- **Effort:** LOW (~90% reuse, §3). The de-risk is CPU/numpy, hours. Validate → integrate behind `BrainConversationalAgent(enable_multicue=True)` default-OFF (byte-identical when off, like `enable_multiframe`/`enable_attributed`).
- **Dependency:** none new. Builds on `BridgeParser` + the biased-competition buffer, both shipped.

### Phase 2 — Case/agreement marking + a non-English toy (true cross-language)
- **What:** add a **case/agreement cue family** (a morphological marker → role vote, e.g. -ga/-o or a case suffix population) as another competing cue; demonstrate a **free-word-order toy language** (case-marked) where roles are assigned by case even when order varies, by **re-learning the cue weights** on that language's data (`w_case` high, `w_pos` low).
- **Reward:** HIGH (the literal "language-agnostic" goal), but a step removed from the owner's immediate English-robustness use case.
- **Effort:** MEDIUM. The cue family is *another plastic cue projection* (same shape as Phase 1's animacy/verb-fit), and re-learning weights is *re-running the Hebbian loop on new data* — both reuse Phase 1's machinery. The genuinely-new bit is a **small case-marked toy corpus** + showing the weight re-settling. Still no architecture change.
- **Dependency:** Phase 1 (the competition + learned-weight machinery). The case cue is "just another cue" once the multi-cue substrate exists.

### Phase 3 — Sub-word morphology (agglutinative languages) — DEFERRED
- **What:** parse roles from *sub-word morphemes* (Turkish/Finnish agglutination: a word = stem + stacked case/agreement/number suffixes). This needs a **morpheme-segmentation + composition layer** below the word level — a genuinely new representational tier.
- **Reward:** MODERATE, NARROW. Only matters for agglutinative langs; English (and isolating/lightly-inflected langs) never needs it.
- **Effort:** HIGH (months-scale) — the only phase that is a *new layer*, not a new cue. Likely interacts with the deferred dendritic/compositional substrate (sub-word composition is a compositional-binding problem).
- **Dependency:** Phases 1–2 + plausibly the dendritic-substrate arc. **Explicitly out of scope for the current direction;** named here so the phase boundary is honest.

**Confirmation:** the proposed order is correct. Phase 1 is the highest reward-per-effort and matches the owner's immediate use case; Phase 2 is the cross-language payoff at modest incremental effort once Phase 1 exists; Phase 3 is correctly deferred as the only months-scale new layer with the narrowest reward.

---

## 5. (kept brief — Phases 2/3 de-risks are downstream; the immediate build is Phase 1, §6)

---

## 6. Phase-1 cheapest-first DE-RISK — the immediate next build (the crux)

**Question it decides:** *Does adding competing cues (animacy + verb-fit) to the position parser make role-assignment ROBUST to degraded word order, where the position-only parser collapses — with the cues genuinely LOAD-BEARING and the weights genuinely LEARNED (not hand-set)?*

### 6.1 Setup (smallest CPU/numpy, hours)
- **Substrate:** reuse `BridgeParser`'s role ensembles + the biased-competition buffer's WTA, re-pointed to 3 roles. Cue families: **position** (existing conjunction unit), **animacy** (`ANIMACY` lexicon), **verb-fit** (`VERB_SELECTS` selectional lexicon). Two new plastic cue→role projections, trained by the existing `_train` Hebbian loop on a small clean-English training set (canonical SVO sentences with the cue values present). `SIM_BACKEND=numpy` is fine for the smoke (tiny bridge); a GPU multi-seed confirm follows if the smoke is GO.
- **Test vocabulary:** small, animacy-balanced (e.g. animate agents {dog,cat,fox}, inanimate patients {ball,apple,rock}) + verbs with clear selectional restriction (eat/chase → animate eater; roll/float → inanimate roller). Crucially the held-out test FILLERS differ from training (role correctness is vocab-agnostic, like the multi-frame de-risk's held-out tuples) — **assert it's not memorizing examples.**

### 6.2 The degradation battery (the test conditions)
For each clean canonical sentence, produce degraded variants and measure role-assignment accuracy:
1. **DROP-A-WORD:** omit a non-verb word ("dog north" from "dog go north") → position indices shift; the surviving cues must still place roles.
2. **SCRAMBLE-ORDER:** permute word order ("north dog go", "go north dog") with the **same words** → position cue is now misleading; animacy + verb-fit must override.
3. **NON-CANONICAL/OBJECT-FRONTED:** an OSV-style order that the *position-only* table mis-maps (the NEMO object-initial weakness) → semantic cues must carry it.

### 6.3 GO bar (pre-register; FROZEN; ≥6 seeds; fractional ≥5/6)
- **MULTI-CUE recovers roles on the degraded battery** at **≥ 0.80** mean role accuracy (vs chance 1/3 ≈ 0.33), on ≥5/6 seeds, AND
- **the LOAD-BEARING control — POSITION-ONLY baseline — COLLAPSES on the SAME degraded battery** (≤ chance+0.12 ≈ 0.45; ideally near chance). *If position-only does NOT collapse on the degraded set, the test is not actually degrading the position cue — INVALID, fix the battery before any claim.* This control is the whole point: it proves the win is the **added cues carrying degraded input**, not a generically better parser.
- **the NATIVE clean SVO still comprehends** (no regression; multi-cue ≥ position-only on clean input), AND
- **the no-confab MOAT holds** (an unparseable/ambiguous input with no decisive role winner → abstain; 0 breaches).

### 6.4 Anti-cheats (each must pass or it's not a GO)
1. **Cue-LESION (THE decisive control):** zero the learned animacy + verb-fit cue→role weights, keep position. Robustness on the degraded battery must **collapse back to the position-only level**. A lesion that does NOT break degraded-input robustness = the cues weren't load-bearing = NOT a GO. (Mirrors the buffer's bias-lesion control.)
2. **NO-LEARNING control (the "are weights learned, not hand-set?" control):** run with the cue→role weights frozen at their **initial design values** (skip the Hebbian `_train` for the cue projections). If degraded-input robustness is the same as with training, the weights weren't doing the work — it's hand-tuned, NOT a learned cue-validity result → NEGATIVE. (This is the honest-risk guard from §7.)
3. **PERMUTED-CUE control:** train the animacy/verb-fit cue→role map against a **scrambled** cue-value assignment (animate→patient, etc.). Comprehension on the degraded battery must collapse to chance — proves the cues carry *real* role information, not a relabelled position signal.
4. **Held-out fillers:** the degraded test uses fillers unseen in training (role correctness is vocab-agnostic) → **not memorizing the degraded examples.**
5. **Moat intact:** assert 0 false role-commitments on genuinely ambiguous input (e.g. two animate nouns + an order-scrambled symmetric verb → tie → abstain). The moat is not weakened.

### 6.5 Outcomes
- **GO** → the multi-cue competition makes English comprehension robust to degraded word order, cues load-bearing + learned → promote into `BrainConversationalAgent(enable_multicue=True)` (default-OFF), multi-seed GPU validate the full who/what + moat pipeline on degraded input, then add the case cue (Phase 2).
- **BOUNDARY** → some degradations recover but others (e.g. object-fronting) stay seed-fragile → localize (cue weight balance / more training / per-cue sub-pools), report honestly.
- **NEGATIVE** → either the cues don't carry degraded input (competition doesn't integrate them) OR the no-learning control matches the trained one (hand-tuned, not learned) → report the honest negative + the fallback (§7).

---

## 7. Honest risks + the stop rule

### 7.1 The biggest way it could MISLEAD: hand-tuned cues masquerading as a learned model
The seductive failure is that the **feature lexicons (animacy, verb-selectional-fit) are host scaffolds** (the buffer already flags this). If the cue→role weights are effectively hand-set (or the host lexicon does the discrimination and the "competition" just reads it out), then "robustness" is **engineering, not a learned cue-validity model** — and it would NOT transfer to a new language by re-learning, defeating the whole "easily adaptable" goal. **Guard:** the **NO-LEARNING control** (§6.4.2) + the **cue-LESION** (§6.4.1) directly test this — the win must require *trained* cue→role weights, and must vanish when they're removed. If degraded-input robustness survives freezing the weights at init, the result is hand-tuned → honest NEGATIVE, do not ship as "learned." (This is the same standard that retracted the 2026-05-14 compose-concept claims and the transitive-inference claim — load-bearing controls first.)

### 7.2 The point-neuron risk (named, and why it's LOW here)
Weighted multi-cue integration *could* be feared to need an **analog/graded, pre-spike computation** that point neurons can't do — the documented **Mikulasch-Priesemann limit** that walled the conversational decorrelation/whitening blocker and the structured-cortex generalization (where *decorrelation* genuinely requires dendritic compartments). **But cue-integration is a categorically different computation:** it is **reliability-weighted evidence accumulation to a winner**, which attractor/accumulator networks do **near-optimally with firing-rate codes** (PNAS 2214441119 "flexible integration of continuous sensory evidence"; eLife 20047 competitive attractor accumulation; the project's own Wang-2002 navigation accumulator + the biased-competition buffer). Summing reliability-weighted votes and letting recurrence pick the winner is a *rate-code* operation; it does **not** require removing a common mode or decorrelating channels in the dendrite. **So the substrate wall does not apply to the cue competition itself.** Where a graded/dendritic computation *would* re-enter is the **structured-similarity generalization** frontier (treating "dog"/"cat" as related cue-bearers) — but that is the separate, already-mapped generalization arc, NOT Phase 1's robustness goal. **The honest residual point-neuron risk for Phase 1** is narrower: whether a *single* graded cue-confidence (a continuous selectional-plausibility score) can be represented finely enough by a small rate-coded population to tip the competition — the de-risk's animacy/verb-fit cues are near-binary (animate/inanimate; fits/doesn't), so this is mild; if a *finely-graded* plausibility cue is later needed and rate-coding is too coarse, the fallback is a **population-coded cue confidence** (the same population-code lift that took the stream cortex read-out 47% → 100%) — not a dendritic rewrite.

### 7.3 Clear cheap-first GO vs NEGATIVE
- **GO** = degraded-battery role accuracy ≥0.80 (≥5/6) **AND** position-only collapses on the same battery **AND** cue-lesion collapses **AND** no-learning control collapses **AND** clean SVO unregressed **AND** moat 0-breach.
- **NEGATIVE** = the cues don't carry degraded input (competition fails to integrate them → degraded accuracy ~ position-only), OR the no-learning/lesion controls do NOT collapse (hand-tuned, not learned). Report it; the fallback is (a) population-coded cue confidence (if the issue is cue resolution) or (b) the documented dendritic/PPMI arc only if a *structured-similarity* cue is the blocker (which Phase 1 does not require).

---

## 8. Provenance (verified sources)
- **Competition Model:** Bates & MacWhinney 1982/1989; MacWhinney 2005 "Extending the Competition Model" (*Int. J. Bilingualism*); MacWhinney-Bates-Kliegl 1984 "Cue validity and sentence interpretation in English, German, and Italian" (*JVLVB*). English=order, German=agreement+animacy, Italian=agreement.
- **Constraint-satisfaction neural realization:** St. John & McClelland 1990 "Learning and applying contextual constraints in sentence comprehension" (*Artificial Intelligence*) — real-valued constraint strengths compete/cooperate, learned from data, graceful with vague/incomplete input. McClelland et al. 2020 PNAS "Placing language in an integrated understanding system" (mutual constraint satisfaction as the domain-general principle).
- **Good-enough / NVN heuristic:** Ferreira 2003 "The misinterpretation of noncanonical sentences" (*Cognitive Psychology*); Ferreira & Patson 2007 "The 'good enough' approach to language comprehension." The semantic-cue fallback + the human error signature ("The mouse was eaten by the cheese").
- **Neural cue-integration / accumulation:** catalog G.18 (LIP integrates any reliability-weighted evidence additively), G.16/G.17 (drift-diffusion / LIP accumulator), G.12 (Broca; semantic cues carry non-canonical comprehension); PNAS 2214441119; eLife 20047 (rate-coded near-optimal reliability-weighted cue integration in attractor accumulators).
- **Biased competition:** Desimone & Duncan 1995; Wong & Wang 2006 (the project's WTA core).
- **MUC:** Hagoort 2013 "MUC and beyond" (PubMed 23874313); Baggio & Hagoort N400 dynamic account — integration-by-current-summation; N400/P600 as cue-conflict markers.
- **Biologically-plausible parser precedent + the gap:** NEMO / Assembly Calculus — Papadimitriou-Vempala-Dabagia-Mitropolsky, biorxiv 2025.07.15.664996 "Simulated Language Acquisition in a Biologically Realistic Model of the Brain" (arxiv 2507.11788); Mitropolsky et al. "A Biologically Plausible Parser" (arxiv 2108.02189). Role areas under mutual inhibition + Hebbian order learning; object-initial weakness; single-cue (no animacy/case integration) = the gap the project fills.
- **Point-neuron limit (why it does NOT apply to cue competition):** Mikulasch & Priesemann (PNAS 2021925118 "Local dendritic balance...") — decorrelation/efficient-coding is the dendritic/pre-spike computation; cue-integration-to-a-winner is rate-coded and feasible.
- **Project machinery:** `research/runners/biased_competition_buffer.py`, `research/runners/brain_conversational_agent.py:28` (`BridgeParser`), `research/runners/frame_parser.py`, `research/runners/_phaseB_multiframe_comprehension_derisk.py`, `research/runners/_phaseB_frame_selection_derisk.py`, `research/runners/content_selection_spiking.py`, `research/runners/multi_turn_agent.py`. Catalog: `sim-catalog/references/feature-catalog.md` Cluster G.
