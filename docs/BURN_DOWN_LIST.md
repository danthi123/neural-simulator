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
- **B1. The generative DRAW (#3E) = host b2 oracle.** The spiking WTA sampler hardcodes an 8×8 taxonomy + KeyErrors on
  arbitrary vocab, so the #3E open-ended draw runs on the host. **Burn-down → a VOCAB-AGNOSTIC spiking generative draw.**
  STATUS: RESEARCH-NEEDED (part of the spiking-generation de-risk).
- **B2. The plausibility gate (#3E) = host** (selectional-preference over the brain's clean fact co-occurrence graph).
  **Burn-down → spiking plausibility.** STATUS: QUEUED.
- **B3. Non-contradiction gate INERT on onebrain** (the onebrain composer doesn't store negations retrievably, so the
  non-contradiction check only fires on rf). **Burn-down → onebrain negation storage.** STATUS: QUEUED.

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
  mouth); (b) the appraisal INJECTION (host DR-2 valence lexicon → neuromodulator) is a declared host scaffold;
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
- **D3. Curiosity** (curiosity-inversion GO, learning-progress selection GO, curiosity-veto). Wire → drives a follow-up
  question on a novel/uncertain topic. STATUS: QUEUED (audit in flight).
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
- **D5. Episodic memory** (gap5 one-brain CAPSTONE 6/6 GO: converse→sleep-replay→converse). Wire → recall of PAST TURNS on the
  live turn. STATUS: QUEUED.
- **D6. Advanced WM binding** (wm-binding-advanced, de_risked YES). Wire → ≥2-referent anaphora / multi-slot. STATUS: QUEUED.

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
- **E2. Internal worldview / affective world-model** — **DE-RISKED GO 6/6 (2026-08-12), QUEUED-FOR-WIRING** (was
  RESEARCH-NEEDED). The corpus audit confirmed the building blocks existed (D2 mismatch unit, HTM-TM sequence predictor,
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
- **F3. discourse-register PARTIAL**, neural-render PARTIAL — not default-on. STATUS: QUEUED.
- **F4. Any host-computed reward / value / neuromodulator** on the live turn (audit D2 to confirm none are host-formula).

---

## Parallelization note
The BURN-DOWN work runs in parallel with the WIRING work: research de-risks (A1/B1 spiking generation, E1-E5 mechanisms) run
in isolated worktrees / on the compute lanes ASAP; the WIRING of already-clean GOs (D1-D6, C1, A2) is sequential on the
production files (brain_chat_tui.py / server.py / rich_answer_composer.py) — one agent at a time to keep the working chat
coherent. This list is the master worklist; every closed row is a capability now genuinely on the default spiking one-brain.
