# BURN-DOWN LIST — toward the all-spiking, one-substrate, no-host-shortcut, full-functionality production one-brain

**North-star (owner, 2026-08-12):** full functionality (ALL capabilities) implemented on the DEFAULT production one-brain,
ALL spiking on a SINGLE substrate, NO host shortcuts. Anything that can't be there yet (lacking mechanism, lacking
biologization, un-wired) goes ON THIS LIST and is worked down in parallel ASAP (research + testing immediately).

**How to read a row:** each item = a host shortcut / external scaffold / un-wired validated faculty / missing-or-boundary
mechanism currently standing between the production default and the north-star. STATUS ∈ {IN-PROGRESS, QUEUED, RESEARCH-NEEDED}.
A row is CLOSED only when the capability runs on the default production turn, spiking, one-substrate, lesion-load-bearing,
with the no-confab moat intact and the host/external piece deleted-or-demoted-to-test-oracle.

---

## A. ARTICULATION — the "mouth" (the #1 shortcut)
- **A1. Fluent surface = EXTERNAL pretrained Qwen-0.5B transformer** (`model.generate()` off-bridge; only 3 nonlinearities
  rate-code-approximated — NO spikes/membrane/LIF; external weights, not the brain's own circuitry). The single biggest host
  shortcut. **Burn-down → a brain-native SPIKING generation circuit.** STATUS: RESEARCH-NEEDED / IN-PROGRESS. The residual now
  NARROWS to open ARBITRARY prose (the banked deep-context wall) — the structured GENERATE surface is spiking (A1a).
  - **A1a. GENERATE-channel SURFACE — WIRED brain-native SPIKING (2026-08-12).** The #3E GENERATE channel's hypothesis
    SURFACE is now spoken by FIRING NEURONS, transformer-free: `ChatBrain.render_hypothesis_verified` renders a structured
    (transitive SVO) hypothesis via the composed spiking BROCA — `"perhaps the <S> <V-3sg> the <O>"`, word order = the
    per-pool spiking-RATE ranking on a real Izhikevich `SimulationBridge` (EMERGE-59/61 × the #3E draw, reuse-by-import from
    `research/runners/_spiking_fluent_surface_derisk.py`, 6-seed GO) — replacing the agrammatic host f-string
    `"perhaps bear walk foot"`. Re-parse VERIFIED (the SAME moat the recall path uses → recovers the drawn SVO) or it falls
    back to the raw flagged template (0 leaks); the guess stays clearly FLAGGED. Default-ON (`BRAIN_SPIKING_MOUTH=0` escape
    → the Qwen/stub mouth, byte-identical), lesion-load-bearing (OFF → reverts). Verified SYNCHRONOUSLY numpy-CPU:
    end-to-end `"what might a dog chase"` → `"perhaps the dog chases the hare  [a guess …]"` (moat-recovered), 4/4 direct
    verbs grammatical+faithful, independent held-out parser recovers the SVO, discriminative verify (wrong content rejected),
    no torch on the path, recall/abstain/anaphora/discourse unregressed (byte-identical smoke verdict pristine==modified).
    **RESIDUAL (still Qwen):** open ARBITRARY prose the spiking Broca can't frame = the banked deep-context wall (A1 above).
- **A2. Template-stub renderer** (the GPU-free fallback) — a deterministic SVO template, not fluent. Acceptable only as the
  CPU test-oracle; the production default should be the fluent path (A1 target). STATUS: QUEUED.

## B. GENERATION mechanisms (the brain's own content generation)
- **B1. The generative DRAW (#3E). WIRED — genuinely SPIKING (F1, 2026-08-13).** The #3E open-ended DRAW (picking WHICH
  verb/object filler from the brain's likelihood) was the host b2 oracle (`np.random.choice`) because the b2 spiking WTA
  sampler hardcodes an 8×8 taxonomy + KeyErrors on runtime vocab. `/api/brain-chat` (via `ChatBrain._generate_hypothesis`)
  now routes that DRAW through a VOCAB-AGNOSTIC spiking soft-WTA (`VocabAgnosticSpikingDrawOrgan` →
  `research/runners/vocab_agnostic_spiking_generation_production_organ.py`, reuse-by-import): role pools are INDUCED from
  the brain's OWN stored-fact concepts (no taxonomy), intersected with the plausibility graph row, and a taxonomy-free
  `VocabAgnosticSpikingSampler` is pre-injected onto the proposer — the unchanged generate loop then draws the winner off
  `cp_firing_states` of a co-resident Izhikevich bank (OU noise IS the stochasticity). Default-ON (`BRAIN_SPIKING_DRAW=0`
  → the host oracle draw, byte-identical), lesion-load-bearing (`BRAIN_SPIKING_DRAW_LESION=1` → likelihood ablated →
  plausibility collapses). Every downstream gate (`_plausible`/`_contradicts`) + the #3E moat verify are UNCHANGED (0 leaks
  by construction). Verified numpy-CPU: organ verify GO (draw 0 host-rng/>0 spiking; LESION plausible-frac 0.828→0.035,
  95.7% attributable; noise-ablation deterministic argmax; flag-off 16==16 byte-identical) + through the REAL handler (5/6
  open-ended prompts return flagged spiking-drawn hyps @ 400 spiking draws/0 host-rng, render "perhaps the dog chases the
  mouse [a guess…]", LESION collapses the handler proposer's plausible-frac 0.862→0.009, flag-off a no-op) + smoke
  byte-identical. STATUS: **WIRED**. Commit `6670bda25`. **RESIDUALS** (ride existing rows): the plausibility LIKELIHOOD
  matrix, the SVO template, and the RF-composer moat remain host scaffolds; the WTA bank is co-resident (rides the
  one-brain merge, #1). (Was B1/B2 open-ended-generation host-DRAW.)
- **B2. The plausibility gate (#3E) = host** (selectional-preference over the brain's clean fact co-occurrence graph).
  **Burn-down → spiking plausibility.** STATUS: QUEUED.
- **B3. Non-contradiction ASSERTION-gate. WIRED (Gate-B, 2026-08-12).** The generation-path non-contradiction check was
  historically inert on onebrain; the B3 de-risk (`_burndown_B3_onebrain_negation_moat_derisk`, 6-seed GO D=128) showed the
  onebrain composer DOES recall a stored NEGATE polarity on the substrate (`ask_yes_no` → `_spiking_select` over
  `cp_firing_states`, `enable_spiking_cleanup=True`), so the USER-ASSERTION non-contradiction gate now fires there.
  `/api/brain-chat`: when the user ASSERTS a transitive fact whose POLARITY contradicts the brain's stored polarity for the
  EXACT same SVO ("the dog eats grass" vs a stored "a dog does NOT eat grass"), the brain REJECTS it instead of silently
  overwriting a held belief (reuse-by-import from `research/runners/b3_noncontradiction_production_organ.py`; the gate runs
  BEFORE the store so a reject returns before `_maybe_acquire` overwrites). STORE-SIDE wired too: `brain_chat_tui._maybe_acquire`
  now acquires a heard assertion with its DETECTED polarity (a heard negation stores as NEGATE) via the organ's extractor, so
  the gate has negations to fire against (guarded, legacy AFFIRM path when B3 disabled). Moat-INVERTING (an unknown SVO →
  accept, never a fabricated rejection), MUTUALLY EXCLUSIVE with D2 surprise (patient-mismatch → "unknown" → accept). Default-ON
  (`BRAIN_NONCONTRADICTION_GATE=0` → byte-identical), lesion-load-bearing (`BRAIN_NONCONTRADICTION_LESION=1` bypasses the spiking
  polarity recall → every recall "unknown" → the gate goes INERT, contradictions slip through). Verified numpy-CPU: organ verify
  6/6 ALL_OK (intact 12 rejections, recall-lesion 0 → 100% attributable) + through the REAL handler (reject on the contradiction,
  lesion inert, flag-off no-op). **RESIDUALS** (declared, host upstream): negation DETECTION (`detect_polarity`) + verb
  MORPHOLOGY (surface-first/lemma-fallback) are host input-tagging (the composer already RECALLS polarity on the substrate); a
  learned spiking polarity classifier + the shared D4 lemmatizer are the next rungs. No co-resident bridge added (B3 reads the
  ONE production recall composer directly). STATUS: **WIRED**. Finding `2026-08-12-B3-noncontradiction-production-organ-built-and-verified.md`.
- **B4. Reconsolidation / belief revision (PE-gated in-place fact UPDATE). WIRED (Gate-B, 2026-08-12).** The production
  memory was APPEND-ONLY (tell the brain "the dog went north" then "actually south" → two contradictory facts coexist, the
  STALE one answered first). `/api/brain-chat` now RECONSOLIDATES: when the asserted patient CONTRADICTS the stored one (the
  D2 spiking surprise window is OPEN), the stored fact is UPDATED IN PLACE — no contradictory duplicate. The window-open
  decision REUSES the SAME `cp_firing_states[surprise]` read the D2 block just computed (Nader-Schafe-LeDoux reconsolidation;
  PE-NECESSITY); the in-place rewrite reuses the composer's OWN `update_on_mismatch` (rf + onebrain,
  `_write_block`+`_compose_phases`). Reuse-by-import from `research/runners/reconsolidation_production_organ.py`. Moat-safe:
  ABSTAINS on a missing trace (never fabricates), NEVER writes on a re-statement (window closed), only rewrites a fact the
  brain ALREADY HOLDS; it only PREPENDS an honest notice ("Updated — I'd stored that dog go north; I've revised it in place
  to dog go south"). Default-ON (`BRAIN_RECONSOLIDATION=0` → append-only byte-identical), lesion-load-bearing
  (`BRAIN_RECONSOLIDATION_LESION=1` fires the window but BLOCKS the update → append-only → recall returns the STALE fact).
  **Integration fix (2026-08-12):** the organ's in-place rewrite now RECRUITS a runtime-novel corrected patient into a
  vocab_headroom cleanup slot BEFORE the rewrite (the composer's own `_recruit_word`, the exact pattern `_store_fact` uses) —
  without it `update_on_mismatch`→`_patient_prediction_error` KeyErrored on a word never seen this session ("actually south"
  when only "north" was taught) and silently fell back to append-only. Verified numpy-CPU: organ verify GO (rf 6/6, onebrain
  3/3, window 5.61 Hz open on contradict vs 0 Hz on re-statement, 100% attributable, flag-off byte-identical) + through the
  REAL handler (INTACT novel-patient rewrite dog-go north→south, ONE fact, no duplicate; lesion → stale persists; flag-off
  append-only). **RESIDUALS**: reactivation SELECTS the fact by the same host kb cue-match recall performs (rides the one-brain
  merge); the synaptic-literal engram tag-and-capture tier + a cupy production-scale 6-seed onebrain sweep are the next rungs.
  STATUS: **WIRED**. Finding `2026-08-12-reconsolidation-production-organ-belief-revision-wireable.md`.

## C. MOAT / VERIFY
- **C1. `_verify` decomposition/coverage/entailment logic = host** (a legitimate verification HARNESS, like the existing
  `_verify` — the per-clause role-parse IS spiking via BridgeParser). The claim-level entailment generalization is de-risked
  GO (0 confab leaks, 6-seed). **GENERALIZATION WIRED (2026-08-12):** the RichAnswerComposer multi-fact path now routes
  each rendered sentence through the de-risked `ClaimEntailmentVerifier` over the SET of facts the turn gathered
  (`ChatBrain._verify_claim_set` / `rich_answer_composer._verify_rendered`, imported from `_moat_claim_entailment_derisk`,
  NOT reimplemented) — genuinely free-form MULTI-CLAUSE grounded prose now survives the moat, while any response carrying
  one ungrounded/contradictory clause is rejected (verified numpy-CPU through the production wiring: 0 leaks over the full
  de-risk suite, load-bearing under the `BRAIN_CLAIM_MOAT=0` lesion). Escape flag + single-fact turn keep the single-triple
  `_verify`. STATUS: **WIRED**; RESIDUAL: the clause decomposition + coverage + synonym/negation/hedge bookkeeping remain a
  HOST verification harness (biologizing it is RESEARCH-NEEDED, low priority — a verification harness is defensible).

## D. UN-WIRED VALIDATED FACULTIES (clean GOs, currently default-OFF — the owner's named list)
_(the faculty-integration audit is designing the concrete wiring for each; wire ONLY the clean GOs, load-bearing + default-on)_
- **D1. Affect / emotion** (6-seed GO: affective concept-tagging, affect-state region, graded bistable ladder, spiking
  active-clear quench-gate). **CONTENT-coloring WIRED (Gate-B, 2026-08-12):** `/api/brain-chat` reads the live mood
  NEURALLY off the co-resident graded-affect ladder (`research/runners/affect_production_organ.py`) and colors the
  DEFAULT turn — mood-congruent **forthcomingness** (how many gate-matched facts to volunteer) AND prose **MANNER**
  (the Qwen mouth phrases the same fact warmer/curter). Default-ON (`BRAIN_AFFECT=0` escape), moat-safe (colors only an
  already-matched answer), lesion-load-bearing (`affect_out=0` collapses the coloring; matched fact byte-identical).
  Also the honest inner-state read-out ("how do you feel" → the live valence differential). STATUS: **WIRED**; residuals:
  (a) the MANNER-coloring conditions the EXTERNAL Qwen mouth (host-mediated) — **rides on A1** (brain-native spiking
  mouth); (b) the appraisal VALUE is now the **DR-2 LEARNED distributional valence** (2026-08-12): each strong word's
  valence is inferred by leave-one-out label-propagation over the brain's learned co-occurrence graph (6-seed held-out
  r≈+0.81; 97.4% sign-agreement with the norm on the gated words), sourced from a cached map, NOT a hardcoded lookup —
  this closed the audit's #1 over-credit (the injection was a raw Warriner lookup mislabeled "DR-2 learned"). REMAINING
  host residual: the affect-word SALIENCE GATE + the SEED norms are still Warriner (DR-2 is SEEDED from them — the
  lexicon is propagated, NOT retired; a full drop-in would color plain content/action words like cat/sit/run, breaking
  neutral-default — measured), the LEARNING is numpy not spiking, and the injection is host. **Next rung = a
  fully-spiking on-bridge opponent V+/V- appraisal population** (moves the learning onto the substrate). Default-ON
  (`BRAIN_AFFECT_DR2=0` → raw-norm value, byte-identical oracle). See finding
  `2026-08-12-D1-affect-appraisal-value-learned-DR2-not-hardcoded-lexicon`;
  (c) the affect organ is a co-resident affect/honesty/arbiter bridge run ALONGSIDE the recall composer, not merged
  onto the ONE recall bridge — **rides on the one-brain merge (§ below)**; (d) the held value is a quantized bistable
  LADDER (sign + level), a smooth-magnitude **continuum is still a BOUNDARY**. Curiosity/self-model/reward (rest of the
  old D1 bundle) remain QUEUED.
- **D2. Expectation / consequences / RPE** (limbic-core RPE battery GO, neural-reward GO, value-critic RANK1 GO). RPE =
  prediction-error = SURPRISE on expectation-violating input. **WIRED (Gate-B, 2026-08-12).** `/api/brain-chat` now runs a
  genuinely-SPIKING predictive-coding MISMATCH unit when the user asserts a fact `(agent,action,patient)` for which the
  brain HOLDS a stored `(agent,action)→patient` (reuse-by-import from `research/runners/surprise_production_organ.py` →
  the 6/6-GO D2 de-risk): cue→`patient_expected` (FS/PV-like, GABA_A subtractive inhibition = the recalled prediction);
  `patient_asserted`→`surprise` (excitation). Confirm cancels (~0 Hz), contradict/novel fires. The EXPECTED patient is
  RECALLED by the brain's own spiking `what_does` (not a host lookup); the mismatch is a `cp_firing_states[surprise]` read
  (NO host `recalled==asserted` compare, `current_reward_signal==0`). On a firing surprise the brain PREPENDS an honest
  functional NOTICE ("that surprises me — I'd learned <stored>"). Default-ON (`BRAIN_SURPRISE=0` escape → byte-identical
  oracle), lesion-load-bearing (`BRAIN_SURPRISE_LESION=1` removes the prediction → the SAME confirm input flips 0.00→4.92 Hz
  surprised). Verified SYNCHRONOUSLY numpy-CPU through the real handler (8/8: teach→null; confirm 0.00 Hz not-surprised;
  contradict 5.61 Hz surprised → honest notice; contradict≥3× confirm; lesion makes the same confirm fire; flag-off null).
  STATUS: **WIRED**. **RESIDUALS** (ride existing rows): (a) CO-RESIDENT on its own mismatch-circuit bridge, not merged onto
  the one recall bridge — **rides on the one-brain merge (§ below)**; (b) PRECISION BOUNDARY — at low prediction gain the GO
  drops to 3/6 (the divisive-normalization/gain-match companion process, proxied by a fixed weight); wired at the ROBUST
  `cue_to_expected_weight=0.8` (6/6 GO); fully-learned CA3 all-to-all recall + homeostatic gain precision = the named next
  rungs; (c) the `(agent,action)` recall + patient-block map key on surface tokens (light inflection tolerance). Finding
  `2026-08-12-GateB-surprise-expectation-violation-production-chat.md`. (Surprise-gated plasticity on the live turn = a
  QUEUED optional next step; the honest NOTICE is the wired deliverable.)
- **D3. Curiosity** (curiosity-inversion GO, learning-progress selection GO, curiosity-veto). **WIRED (Gate-B, 2026-08-12).**
  `/api/brain-chat` now reads a genuinely-SPIKING CURIOSITY (crave) drive on an ABSTAIN: the brain's own epistemic gap
  (it holds no answer → a maximal novelty scalar, the SAME signal the no-confab moat uses) feeds the `from_novelty`
  neuromodulator (ALREADY committed additive/default-off in `sim/`, byte-identical when unused → NO new `sim/` edit) →
  an `excitability_drive` on a spiking ASK pool → the ASK pool SPIKES; the wanting is read DIRECTLY off
  `cp_firing_states[ask]` (reuse-by-import from `research/runners/curiosity_production_organ.py` → the DR-1 crave-drive,
  on-bridge 6-seed CPU GO + 6/6-SAFE in Stage-A step-3; **corr(gap,SPIKING-want)=+0.996 reproduced numpy-CPU**, lesion→0
  asks). When the ASK pool CRAVES (want ≥ a build-calibrated threshold) the brain APPENDS an honest FOLLOW-UP QUESTION
  ("My curiosity is piqued — I haven't learned about `<topic>` yet: what can you tell me about `<topic>`?") — crave,
  don't refuse. The moat is INVERTED, not broken: the answer stays an abstain (never a confabulated fact); the added
  text is unambiguously a QUESTION. Default-ON (`BRAIN_CURIOSITY=0` escape → byte-identical oracle), lesion-load-bearing
  (`BRAIN_CURIOSITY_LESION=1` removes the drive pathway → the SAME novel abstain's want collapses 129.2→5.4 Hz below
  threshold → NO follow-up). Verified SYNCHRONOUSLY numpy-CPU through the real handler (24/24: novel abstain → curious
  follow-up with topic; familiar recall → null/no follow-up; lesion → no follow-up; flag-off byte-identical +
  additive:default==off+suffix; single-fact path too; recall/abstain/affect/D2/D4/E1/E2 unregressed). STATUS: **WIRED**.
  **RESIDUALS** (ride existing rows): (a) ONLY the crave-DRIVE is wired (the 6-seed / 6/6-SAFE spiking part); the
  learning-progress SELECTOR is a CPU-proxy host formula (on-bridge memory seed-fragile 1/6) + the noisy-TV VETO is a
  host ELP TD tracker (survives the critic lesion) — NEITHER wired (a single-topic chat follow-up needs no multi-armed
  LP selection nor a noise veto); (b) NOVELTY = the ABSTAIN (a BINARY epistemic gap), a declared host boundary — a
  graded Bogacz-Brown familiarity novelty + curiosity on a low-confidence RECALL (already metacog-hedged) are named next
  rungs; (c) the wh-FRAME is a host language scaffold (only the topic CONTENT is brain-surfaced); (d) CO-RESIDENT on its
  own ASK bridge (rides on the one-brain merge, §). Finding `2026-08-12-GateB-curiosity-followup-production-chat.md`.
- **D4. Comprehension MEASUREMENT** ("measurement of understanding of spoken language"). **WIRED (Gate-B, 2026-08-12).**
  `/api/brain-chat` now reads a genuinely-SPIKING comprehension signal BEFORE acting on an incoming transitive assertion:
  the co-resident `SpikingRoleCompetition`'s two Wong-Wang pools (`sel_agent`/`sel_patient`, mutual inhibition), driven by
  the SEMANTIC (animacy+verbfit) cues only, settle to a firing margin `|agentEv_0−agentEv_1|` read off `cp_firing_states`
  (reuse-by-import from `research/runners/comprehension_production_organ.py` → the 6/6-GO D4 de-risk, AUC=1.000, lesion→0.500;
  the host `_semantic_contrast` dot-product never called). On a LOW margin (OOV / content-ambiguous input the substrate could
  not resolve) the brain honestly ABSTAINS ("my role-binding didn't resolve — I didn't follow that") instead of ingesting it
  — this STRENGTHENS the moat. SCOPE (non-regressive): fires ONLY on a competent 3-content-token transitive (fully cue-covered
  OR fully OOV); questions / self-queries / anaphora / open-ended / real-but-untabled vocab are OUT OF SCOPE → byte-identical.
  GUARD: never abstains on an (agent,action) the brain KNOWS (`what_does` truthy). Default-ON (`BRAIN_COMPREHENSION_GATE=0`
  escape → byte-identical oracle), lesion-load-bearing (`BRAIN_COMPREHENSION_LESION=1` zeroes cue→role synapses → margin 0.338→0.000
  on a well-formed input). Verified SYNCHRONOUSLY numpy-CPU through the real handler (7/7: comprehensible m=0.338 PASSES;
  OOV m=0.026 + ambiguous m=0.142 → honest didn't-follow abstain; known-but-ambiguous m=0.088 HONORED; recall/abstain/anaphora
  unregressed; flag-off comprehension:null). STATUS: **WIRED**. **RESIDUALS** (ride existing rows): (a) CO-RESIDENT on its own
  `SpikingRoleCompetition` bridge, not merged onto the one recall bridge — **rides on the one-brain merge (§ below)**;
  (b) VOCAB CEILING — the cue lexicon is the toy 2-noun transitive scope; a graded/near-threshold battery + a LEARNED cue
  lexicon are the next rung; (c) structural malformedness (no verb/wrong arity) still a host arity check. Finding
  `2026-08-12-GateB-comprehension-monitor-production-chat.md`. Distinct from the E1 lane-C metacog BOUNDARY (a different faculty).
- **T1-6. Conversational OTHER-REPAIR** (a TARGETED clarification, not a dead-end abstain). **WIRED (Gate-B, 2026-08-13).**
  On a low-comprehension turn the D4 gate would ABSTAIN on, `/api/brain-chat` now asks a TARGETED clarification that NAMES
  what did not resolve, instead of the bare "I didn't follow that". It COMPOSES the D4 monitor: the SAME co-resident
  `SpikingRoleCompetition` per-noun agent-evidence (`a0,a1 = sel_agent−sel_patient` off `cp_firing_states`,
  `ComprehensionProductionOrgan.repair_target`, reuse-by-import) localises the failure — `sign(a0+a1)` names the
  OVER-subscribed role (a two-inanimate transitive → both nouns claim PATIENT → the **AGENT** slot is unresolved:
  "my role-binding didn't resolve the AGENT — which of them is doing the 'carry', the book or the cup?"; a symmetric
  two-animate → near-zero net lean → an honest GENERIC role-swap question), `max(|a0|,|a1|)` confirms the roles are ACTIVE.
  An OOV transitive names the unknown token (a declared HOST-LEXICAL scaffold, like curiosity's topic extractor — NOT
  load-bearing). Default-ON (`BRAIN_REPAIR=0` escape → the bare abstain, byte-identical), lesion-load-bearing (the D4
  spiking signal zeroed → pair-max 0.000 → no target → the bare abstain). MOAT-SAFE: a clarification is unambiguously a
  QUESTION (never asserts/confabulates a fact; the turn stays an abstain). Verified SYNCHRONOUSLY numpy-CPU through the real
  handler (6/6: 2-inanim → role=AGENT clarification; 2-animate → generic role-swap; fully-OOV → token naming;
  comprehensible → NO false repair; lesion → bare abstain; flag-off → bare abstain no-key) + a 9-turn pristine-vs-modified
  byte-identical regression (flag-off IDENTICAL across all 9; default-ON changes ONLY the turns D4 already abstained on —
  recall/moat-abstain/anaphora/D2/comprehensible-D4 byte-identical). NO `sim/` edit. STATUS: **WIRED**. **RESIDUALS** (ride
  existing rows): (a) the OOV token branch is a HOST-LEXICAL scaffold (not load-bearing on spikes) — a spiking unknown-word
  read is the next rung; (b) two-animate direction is UNDETERMINED by the substrate → the generic question is the honest
  read; (c) the clarification WORDING is a host language template (only the DECISION + ROLE TARGET are brain-surfaced);
  (d) D4-SCOPE INHERITANCE — the repair fires on exactly the D4-abstain set (inherits D4's OOV-transitive edge cases);
  (e) CO-RESIDENT — reuses D4's own bridge (rides the one-brain merge, §). Finding `2026-08-13-T1-6-other-repair-production-chat.md`.
- **D5. Episodic memory — recall of PAST TURNS. WIRED (Gate-B, 2026-08-12).** `/api/brain-chat` now runs a genuinely-SPIKING
  hippocampal RECALL GATE on a referential turn ("you mentioned X", "earlier you told me about X"): a spoken TOPIC BTSP-forms
  a CA3 assembly (Hook B, the WRITE); a later referential cue COMPLETES it cue-specifically via the two-compartment apical
  dAP UP-state read (Hook A, the READ) — reuse-by-import from `research/runners/d5_episodic_production_organ.py` → the kt=8
  `EpisodicDapMemory` (6/6-GO gap#5 dAP readout, n_ca3=2000). A completed assembly → honest disclosure ("my hippocampal
  readout completes its assembly for it, dendritic dAP completion 0.91"); a non-completing cue → honest "I don't recall
  discussing X" (a genuine spiking completion failure, NEVER a confabulation — the honesty floor). Runs FIRST (referential-
  first, right after AFFECT) so the comprehension/surprise/B3 gates cannot pre-empt it. CONVERSATION-SCOPED (one memory per
  session, cleared on reset — Hook C). Default-ON (`BRAIN_EPISODIC=0` → the referential turn falls through, byte-identical),
  lesion-load-bearing (`BRAIN_EPISODIC_LESION=1` reads the UNFORMED baseline recurrent weights → completion collapses
  0.909→0.000 → "not in memory"). Verified numpy-CPU: organ verify ALL_OK (intact cue=0.909 perm=nocue=lesion=0, attribution
  1.0, wall 790s; 6-seed committed GO both backends) + through the REAL handler (referential-first + honest not-in-memory
  disclosure + IN-MEMORY fire after a store + lesion collapse). **RESIDUALS** (declared, ride existing rows): (a) the fact
  CONTENT surfaced on a completed topic is the host-oracle chat recall the moat already governs (the GATE is spiking; the
  retrieved sentence is the next conversion); (b) temporal/recency ORDER is a host store-index (no spiking WHEN pool yet);
  (c) the gap#5 converse→sleep→replay→converse CAPSTONE (a separate 6-seed GO) is OFFLINE consolidation, deliberately NOT on
  this per-turn path; (d) LATENCY — a BTSP store is ~seconds on cupy but ~430–510s/topic on numpy@2000, so Hook B is GATED
  behind cupy (`_episodic_store_ok`; `BRAIN_EPISODIC_STORE=1` forces it); on a numpy deployment the WRITE is DEFERRED (the
  recall GATE stays spiking) — a declared latency residual; (e) CO-RESIDENT on its OWN dAP readout bridge (n_ca3=2000), rides
  the one-brain merge (burn-down #1). STATUS: **WIRED**. Finding `2026-08-12-D5-episodic-production-organ-spiking-recall-gate-wireable.md`.
- **D6. Advanced WM binding — HOLD >=2 discourse referents across a span. WIRED (Gate-B, 2026-08-12).** `/api/brain-chat` now
  holds ≥2 discourse referents on a genuinely-SPIKING multi-register buffer (R disjoint slow-NMDA bistable banks on ONE
  bridge sharing ONE FS pool; reuse-by-import from `research/runners/d6_multiref_wm_production_organ.py` → the 6-seed-GO
  `MultiSlotHold` + RUNG6c HebbianBinder). MAINTAIN: a coordinated-NP turn ("the dog and the cat …") LOADS each referent into
  its own register (role-by-position) and HOLDS across the intervening span (write-only, changes no reply). READ-OUT: an
  explicit "who/what are we talking about / keeping in mind" query READS BACK every held referent off `cp_firing_states`
  (what a single-attractor store CANNOT do — it ties to one) → honest functional read-out ("I'm holding 2 referents in
  working memory at once: dog and cat"). PER-SESSION buffer (the organ singleton's referent codebook is process-global, so a
  shared buffer would leak other sessions' referents; cleared on reset). Default-ON (`BRAIN_MULTIREF=0` → byte-identical),
  lesion-load-bearing (`BRAIN_MULTIREF_LESION=1` builds recur=0 → the slow-NMDA hold dies → the ≥2 read-back collapses,
  all_recovered 1.000→0.000). Verified numpy-CPU: organ verify PASS (k=2/3/4 all_recovered=1.000, lesion=0.000, 100%
  attributable, superposed-single collides ~1/k) + through the REAL handler (MAINTAIN n=2 all_recovered; hold-query readout
  holds dog+cat; lesion collapses; flag-off byte-identical). **RESIDUALS** (declared): (a) the learned SPIKING WRITE-GATE is
  the open rung — register assignment is a role-by-position host MARKER (gap#4 credit-assignment); (b) referent EXTRACTION is
  a host regex + small lexicon (vocab-ceiling, same class as the comprehension organ); (c) the register READ is a host argmax
  over each bank's firing rates (read-out instrument), binder-capped at 6 distinct referents (ceiling k=5); (d) cross-turn
  persistence of WHICH referents is a host codebook (the load-bearing spiking part is the within-span HOLD); (e) CO-RESIDENT
  on its own `MultiSlotHold` bridge (rides the one-brain merge, burn-down #1). STATUS: **WIRED**. Finding
  `2026-08-12-D6-multiref-WM-production-organ-holds-two-plus-referents-lesion-load-bearing.md`.

## E. MISSING MECHANISMS / BOUNDARIES (need RESEARCH before they can be wired — do NOT fake integration)
- **E1. Self-model / metacognition — WIRED (Gate-B, 2026-08-12)** (was BOUNDARY -> DE-RISKED GO 6/6 -> now WIRED onto the
  DEFAULT turn). `/api/brain-chat` now reads a genuinely-SPIKING confidence of the answer the brain is about to give off the
  co-resident balance-of-evidence monitor (`|rate(asm1)-rate(asm0)|` from `cp_firing_states`, reuse-by-import from
  `research/runners/metacog_production_organ.py` -> the E1 balance de-risk); the evidence is the brain's OWN mean role-decode
  confidence. On a LOW-confidence answer the brain honestly QUALIFIES it (an honest FUNCTIONAL hedge — "my decision-margin
  reads this as low-confidence"), never a phenomenal claim, never a content change. Default-ON (`BRAIN_METACOG=0` escape ->
  byte-identical oracle), moat-safe (only qualifies an already-produced answer; an abstain is skipped), lesion-load-bearing
  (`BRAIN_METACOG_LESION=1` removes the evidence differential -> a confident answer FLIPS to hedged). Verified SYNCHRONOUSLY
  numpy-CPU through the real handler (11/11: high-conf recall no hedge; low-conf recall hedged; abstain skipped; lesion flips
  high->hedge; flag-off null+byte-identical; recall/anaphora/abstain/D2/D4 unregressed). Finding
  `2026-08-12-GateB-metacog-confidence-readout-production-chat.md`. **RESIDUALS** (ride existing rows): (a) EVIDENCE = the parse
  confidence (a COMPONENT of answer confidence), not a full recall-vs-alternatives balance (the rf magnitude/frac signals are
  saturated on the tiny-demo) — a richer recall-margin evidence is the next rung; (b) NOT type-1/type-2 DISSOCIABLE (the
  balance read is an ENCODING read, so the load-bearing lesion is on the encoding; the dissociable `margin_abs` comparator is
  seed-fragile = the named next rung); (c) CO-RESIDENT on its own metacog-workspace bridge — rides on the one-brain merge (§).
  **The original de-risk record (for provenance):** (1) INSTRUMENT — the "mis-calibrated GO gate" bug did NOT exist; the gate
  already required `type2_auc≥0.65 AND meta_d>0 AND m_ratio≥0.60 AND controls`. The old boundary finding conflated a chance
  `meta_rate` run's type-2 numbers with a genuine `learned_acc` run's GO verdict (correction banner added to it). Hardened
  anyway (CLAUDE.md rule 9): extracted a pure `_seed_go_decision`, added `selftest()`/`--selftest` that FAILS in its failing
  direction (chance type-2 → NO-GO even with type-1 fine), fixed a latent always-False `domain_control`. (2) MECHANISM —
  following the D4 lead, a `balance` confidence read = `|rate(asm₁)−rate(asm₀)|`, the workspace WTA margin read directly from
  `cp_firing_states` (Vickers balance-of-evidence / Kepecs distance-to-bound), clears the corrected gate 6/6: type2_auc
  0.668–0.815, meta_d 1.04–2.25, m_ratio 0.66–1.84, permuted collapses to chance, within-class holds, read ENTIRELY from
  spikes. **First PURE-SPIKING meta-d′>0 to clear 6/6** (the prior `learned_acc` 6/6 uses a HOST logistic regression = a
  shortcut). Unblocks the honesty-boundary self-report read-out ("my familiarity monitor reads this as novel → I'm
  uncertain"). **RESIDUAL/next rung (mapped):** loop-ablation does NOT collapse the balance read — it is "confidence =
  balance of evidence" (a genuine decision-variable read) but NOT architecturally type-1/type-2 DISSOCIABLE; the dissociable
  comparator (`margin_abs`) is seed-fragile 0/6 = the named next rung (make it robust via the D4 content-sensitive read).
- **E2. Internal worldview / affective world-model** — **WIRED (Gate-B, 2026-08-12)** (was RESEARCH-NEEDED -> DE-RISKED
  GO 6/6 -> now WIRED onto the DEFAULT turn). `/api/brain-chat` now runs the co-resident 2-channel spiking predictive-coding
  VALENCE forward model (reuse-by-import from `research/runners/worldmodel_production_organ.py`): the next-turn-affect
  prediction is QUERYABLE ("what do you expect / how is this going" -> the two-pool spike-rate read, an early-return honest
  read-out), and an affect-trajectory VIOLATION fires a genuinely-SPIKING surprise (`cp_firing_states[surprise]`) that
  PREPENDS an honest "that shifts the mood unexpectedly" notice. Default-ON (`BRAIN_WORLDMODEL=0` escape -> byte-identical
  oracle), moat-safe (only READS/NOTICES — never manufactures a fact or flips an abstain), lesion-load-bearing
  (`BRAIN_WORLDMODEL_LESION=1` zeroes the learned transition -> the queryable prediction margin collapses 411->0 AND an
  EXPECTED observation flips 0->52 Hz surprised). Verified SYNCHRONOUSLY numpy-CPU through the real handler (15/15: queryable
  +411 Hz margin; violation 24.3 Hz > thr 12.2 fires the notice; persistence no-surprise; lesion collapses both; flag-off
  null + not-intercepted; recall/anaphora/abstain/D2/D4/E1 unregressed). Finding
  `2026-08-12-GateB-worldmodel-affective-forward-model-production-chat.md`. **RESIDUAL (the mission's named NEXT RUNG, not
  faked):** GENERIC pos/neg pools — binding the state+observation to the ACTUAL interlocutor affect (the P0.3 valence latch +
  the W5 ToM channel) is the next rung (un-wired); the persistence state-SELECTION is a declared host mapping; Markov-1
  first-order transition (HTM-TM high-order is a rung); teacher-driven (not self-organized); CO-RESIDENT on its own bridge
  (rides on the one-brain merge, §). **The original de-risk record (for provenance):**
  The corpus audit confirmed the building blocks existed (D2 mismatch unit, HTM-TM sequence predictor,
  W5 other-tagged affect, P0.3 valence latch) but NO validated *affective forward model*. Built one, brain-based, NO
  `sim/` edit (`research/runners/_affective_world_model_derisk.py`): a 2-channel spiking predictive-coding VALENCE forward
  model on the Izhikevich bridge — `state→pred_{pos,neg}` is an all-to-all plastic transition LEARNED FROM ZERO by Hebbian
  co-fire (each state learns which valence follows — a 2-way discrimination that sidesteps the n-way CA3 pattern-separation
  wall); the prediction is delivered as subtractive GABA_A inhibition to `surprise_{pos,neg}` error units that `obs_{pos,neg}`
  excites. Expected turn→prediction cancels observation→~0 Hz; violated→un-inhibited channel fires; prediction read = a
  two-pool spike-rate difference; surprise = a `cp_firing_states` read (NO host argmax/reward/compare). 6/6 GO: predicted-
  valence acc 1.00, expected-turn surprise 0 Hz vs violated 37–46 Hz; lesion 3/3 (zero the learned transition → ratio→1.0×,
  100% attributable); dual-scored shuffle 3/3+3/3 (structure not template); update-on-error shifts the prediction toward the
  new observation. **Instrument verified in BOTH directions** (caught a false-null from explicit-Euler instability at high
  cue AND a false-GO from a weak-firing cancellation artifact — fixed by the pred→surprise gain match, the precision
  companion the wall-reframe predicts). **RESIDUAL/next rungs:** Markov-1→HTM-TM high-order (context-dependent); 2-way
  valence→CA3 sparse pattern-sep for a full state rollout; generic pos/neg→bind to P0.3 latch + W5 ToM so it predicts the
  INTERLOCUTOR's affect; teacher-driven (learned but not `self-organized`, a declared boundary). Wire → a queryable "what do
  you expect / how is this going" + surprise on interlocutor-affect violation. Reuses the D2 [[surprise]] mechanism family.
- **E3. Deeper LEARN — a BTSP/plateau per-turn LASTING trace** (today LEARN writes the RF store, a synaptic write, but a
  BTSP plateau lasting trace is the fuller "the turn writes synapses"). **DE-RISKED GO 6/6 (2026-08-12), with a HOST caveat.**
  The corpus check found prior BTSP work measured the WRITE (2026-07-18 on-bridge BTSP GO: held plateau potentiates
  one-shot, held_dw ~110 vs transient ~13) + recall IMMEDIATELY after — but never whether the write LASTS. This de-risk
  (`research/runners/_gap4_btsp_lasting_trace_recall_after_delay_derisk.py`) closes that: a real on-bridge BTSP write
  (`enable_btsp` + bistable BDSP apical) + spiking recall, with a synaptic TAG-AND-CAPTURE persistence model (Frey-Morris
  1997 / Lisman CaMKII / Bittner-Magee 2017 — supra-barrier synapses stabilized, sub-barrier passively decay). 6/6 GO: the
  plateau still recalls after a 200-step decay window (54–92 Hz) while transient/moat/static writes decay below recall;
  lesion-load-bearing (the IDENTICAL plateau write stays ~1100 with capture, decays to 0.3 without → recall fails, 95%
  attributable); the crux control shows the failure is DECAY not a weak write; instrument shown CAPABLE OF FAILING (β=0 and
  barrier=100 both correctly return BOUNDARY). **HOST CAVEAT (honest):** the LASTING/capture side is a runner host model,
  NOT a spiking kernel yet; NOT `consolidation` in the TERMS sense (no replay executes); low absolute firing → "decayed
  below recall" is a within-seed ~17× contrast. **NEXT RUNG (named):** port tag-and-capture to a guarded default-OFF
  byte-identical-when-off `sim/` kernel (alongside `hebbian_weight_decay`), then WIRE under production LEARN so a taught
  fact's per-turn write is a genuine on-substrate BTSP plateau + capture. Related to [[surprise]]/gap#4 credit family.
- **E4. gap#4 deep credit on the Izhikevich production substrate** (the read-regime residual; fixed-FA doesn't converge on
  Izhikevich where LIF does). RESEARCH-NEEDED (KP/microcircuit levers being swept).
- **E5. gap5-R4 emergent-assembly BTSP completion = BOUNDARY** (writes but cue-completion ≈0 on emergent assemblies).
  RESEARCH-NEEDED.
- **E6. Perception / motor** = PARTIAL, not wired. RESEARCH-NEEDED for the conversational-perception path.

## F. OTHER host shortcuts / residuals in the production path
- **F1. Host `QuestionRouter` fallback** — retired for factual-SVO questions (the on-brain parser owns them) but still the
  self/identity + noisy-anaphora fallback. Burn-down → neural self/identity + a robust anaphora WM. STATUS: QUEUED.
- **F2. `_learned_assoc` graph polluted by the `__free` reserve-slot codes** (a latent interaction from the recruit-an-assembly
  vocab_headroom fix — noise edges like "dog use worm"). Bug to fix. STATUS: QUEUED.
- **F3. discourse-register — WIRED / default-ON (D3, F2, 2026-08-13).** `/api/brain-chat` now answers "who was doing it
  BEFORE?" across a discourse connective off the held spiking prev-slot (four FS-WTA attractor slots read from
  `cp_firing_states`), while still tracking "who is doing it NOW?" — a single-event register structurally cannot do this.
  PART A: both register-construction sites in `brain_chat_tui.py` (`_build_tiny_demo` + `_load_self_knowledge`) build
  `make_discourse_register` (the validated spiking twin) instead of `PairEventRegister(spiking=False)`. PART B: the
  `brain_chat` handler runs an additive fold of a discourse SVO clause (a connective marks the boundary → SHIFT; pure
  side-effect, reply byte-identical) + a disjoint before/now query short-circuit answered off the register + the moat
  abstain (a before-answer only after a connective boundary actually opened this conversation). Reuse-by-import from
  `research/runners/d3_discourse_event_register_production_organ.py`. Default-ON (`BRAIN_DISCOURSE_REGISTER=0` → the
  register is built spiking=False AND the endpoint block is skipped → byte-identical), lesion-load-bearing
  (`BRAIN_DISCOURSE_REGISTER_LESION=1` silences the spiking prev-hold → who-was-before collapses, NOW preserved). Verified
  numpy-CPU: organ verify seed42 ALL_OK (BEFORE 0.900/NOW 0.917, LESION→0.150 83% attributable, moat 1.000; source branch
  42/43/44 all ALL_OK) + real handler (before/now on spikes; LESION collapse + register-type flip; flag-off block skipped)
  + smoke flag-off byte-identical, default-ON discourse resolves. STATUS: **WIRED**. Commit `d817effd0`. **RESIDUALS**: the
  transition-δ RNN (`multislot_rnn`, rate-learned), the boundary/connective detection + referent/verb parse are host; the
  register is co-resident (rides the one-brain merge, #1). **neural-render PARTIAL** — still not default-on (QUEUED).
- **T1-4. Causal WHY / WHAT-IF forward model — WIRED / default-ON (2026-08-13).** `/api/brain-chat` now answers a real
  "what happens if <agent> <action>?" (forward-SIMULATION of an unseen consequence — the substrate rolls
  A=(dog,go,east)->B=(dog,reach,river)->D=(dog,drink,water) though A->D was NEVER taught) and "why did <agent>
  <action>?" (the directed cause that survives a Pearl DO-probe — Y=(dog,wake,morning) reads C=(sun,rise,sky), never
  the correlate X=(bird,sing)) — the reasoning rung a host triple-JOIN cannot serve. The `brain_chat` handler runs a
  DISJOINT why-did/what-happens-if turn class (after the affect/episodic/worldmodel/multiref/discourse read-outs,
  before comprehension/surprise/rich), routed through the co-resident spiking forward model (reuse-by-import from
  `research/runners/causal_whatif_production_organ.py` — the grounded de-risk, 6/6 GO — + its toy primitives; NO `sim/`
  edit). MOAT-SAFE: the consequence/cause is emitted ONLY when `composer.query_patient` CONFIRMS it (the no-confab moat
  the live recall uses); an unconfirmed/unmapped causal query ABSTAINS to the honest `_honest_causal_answer` disclaimer
  (INTEGRATION #5 fallback) — 0 confabulation. The grounding is READ-ONLY (the organ's event set + causal curriculum are
  gated by the LIVE composer's moat recall; it never writes a fact). Default-ON (`BRAIN_CAUSAL=0` -> the block is fully
  skipped, byte-identical), lesion-load-bearing (`BRAIN_CAUSAL_LESION=1` zeroes the learned forward edges -> BOTH
  why/what-if collapse to the honest abstain). Verified numpy-CPU through the REAL handler: what-if moat-confirmed
  consequence; why DO-surviving moat-confirmed cause; unmapped + grounding-unconfirmed abstains (0 confab); LESION
  collapse; byte-identical-when-off (recall/abstain panel flag on==off + causal query flag-off no key) + canonical
  `brain_chat_tui --smoke` byte-identical (server.py stashed vs present). STATUS: **WIRED**. Finding
  `2026-08-13-causal-whatif-production-organ-wired-into-brain-chat.md`. **RESIDUALS** (declared next rungs):
  (a) GROUNDING-BY-DERIVATION not shared-substrate-merge — the events are DERIVED from + gated by the composer's moat
  recall, but the composer's unbind SPIKES do not yet directly DRIVE the forward-model event blocks in ONE merged
  bridge (co-resident forward-model bridge, rides the one-brain merge #1); (b) the DA sign + causal episode ORDER are
  teacher-delivered (a spiking-mismatch DA is the next rung); (c) FIRST-ORDER + the canonical CHAIN/CONFOUND causal
  STRUCTURE is teacher-rendered (wired scope = the validated chain-source what-if + confound why; anything outside
  abstains honestly). NOT `scaffold_retired` (the host causal disclaimer remains the abstain fallback).
- **F4. Any host-computed reward / value / neuromodulator** on the live turn (audit D2 to confirm none are host-formula).
- **F5. The co-resident organs are COMBINED by HOST PYTHON, not by the substrate** — `ChatBrain.gate()` (reached from
  `webapp/server.py::brain_chat`) snapshots a spiking recall + a reverse-binding VERIFY + a router fallback and fuses
  them with an `if recalled == p`. The organs are neural; their COMBINATION is host `if/else` — the audit's #1
  structural lever. **Burn-down → the spiking GNW N-organ ignition bus IS the organ-combination.** STATUS: **SHADOW
  WIRED / default-OFF (first step, 2026-08-13).** The de-risked N-organ bus (`_gnw_norgan_bus_derisk`, 6/6 GO) is now
  wired into the `brain_chat` single-fact path as an ADDITIVE, DEFAULT-OFF verification/shadow path
  (`webapp/gnw_bus_shadow.py::shadow_report`, reuse-by-import — NO `sim/` edit). With `BRAIN_GNW_BUS=1` the SUBSTRATE
  re-derives the host `gate()` combination by routing the SAME real organ reads (recall + VERIFY re-check + reverse-
  binding VERIFY) through ONE warm spiking workspace: three corroborating subthreshold votes ACCUMULATE + IGNITE the
  patient slot (a decoy is WTA-suppressed; a recall-miss = no ignition = abstain). Verified numpy-CPU through the REAL
  production ChatBrain + handler: AGREEMENT 9/9 (5 stored ignite the host's exact patient, 4 abstain withhold — the
  moat as a substrate property); LESION-load-bearing (assembly self-recurrence zeroed → the bus collapses to abstain
  while the forward-recall reflex survives); BYTE-IDENTICAL-when-off (real handler carries no `gnw_bus` key + host
  fields unchanged). This PROVES the substrate can combine the live organs; the host orchestration is NOT removed yet
  (flip-to-default is gated on the wiring design's §4 criterion). Files: `webapp/gnw_bus_shadow.py`,
  `webapp/server.py brain_chat`, verify `research/runners/_gnw_bus_shadow_production_verify.py`; design
  `docs/plans/2026-08-13-gnw-norgan-bus-production-wiring.md`; finding
  `2026-08-13-gnw-norgan-bus-shadow-wired-into-production-brain-chat.md`. **RESIDUALS**: host `gate()` still authors
  the answer (shadow only); organ B == organ A's forward recall (distinct consensus = recall + reverse-VERIFY,
  faithful to gate()); single-fact path only (rich path follow-on); workspace bridge co-resident (rides #1).

---

## Parallelization note
The BURN-DOWN work runs in parallel with the WIRING work: research de-risks (A1/B1 spiking generation, E1-E5 mechanisms) run
in isolated worktrees / on the compute lanes ASAP; the WIRING of already-clean GOs (D1-D6, C1, A2) is sequential on the
production files (brain_chat_tui.py / server.py / rich_answer_composer.py) — one agent at a time to keep the working chat
coherent. This list is the master worklist; every closed row is a capability now genuinely on the default spiking one-brain.
