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
  prediction-error = SURPRISE on expectation-violating input. **DE-RISKED GO 6/6 (2026-08-12), QUEUED-FOR-WIRING:** the
  existing spiking RPEs are all over a SCALAR reward; the conversational need is a CONTENT-space contradiction ("dog eats
  grass" vs stored "(dog,eats)→meat"), which today would be a host `recalled==asserted` string compare (a shortcut). The
  de-risk (`research/runners/_spiking_expectation_rpe_derisk.py`) built the genuinely-spiking replacement: a predictive-
  coding MISMATCH unit — a `surprise` pool gets EXCITATION from the asserted-patient code + topographic SUBTRACTIVE
  INHIBITION (GABA_A/PV-like) from the recalled expectation; confirm→cancel→~0 Hz, contradict/novel→un-inhibited→fires.
  6/6 GO (confirm 0.3–2.5 Hz vs violate 7.5–9.9 Hz, 3.5–30.9×), brain-based (a `cp_firing_states[surprise]` read, no host
  subtraction, `current_reward_signal==0`), lesion-decisive (zeroing the prediction collapses to ~1.0× → 100% attributable
  to the prediction). **RESIDUAL/boundary:** precision — at low prediction gain GO drops to 3/6 (the divisive-normalization/
  gain-match companion process, proxied by a fixed weight); the which-patient mapping is a topographic prior with Hebbian
  strength (fully-learned CA3 all-to-all recall + homeostatic gain precision = the named next rungs). Wire → an honest
  functional NOTICE ("my mismatch monitor reads this as surprising") + surprise-gated plasticity on the live turn.
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
- **E1. Self-model / metacognition = BOUNDARY** (type-2 at chance despite a mis-calibrated GO gate, 2026-08-12). Genuine
  meta-d′ > 0 is the missing mechanism. RESEARCH-NEEDED (+ audit the mis-calibrated gate). Do NOT wire until it's a real GO.
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
  BTSP plateau lasting trace is the fuller "the turn writes synapses"). RESEARCH-NEEDED.
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
