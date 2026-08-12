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
  shortcut. **Burn-down → a brain-native SPIKING generation circuit.** STATUS: RESEARCH-NEEDED / IN-PROGRESS (spiking-generation
  de-risk agent has a 6-seed run in flight; the emerge stream-cortex hit a deep-context BOUNDARY on spikes, banked).
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
  GO (0 confab leaks, 6-seed). Lower burn-down priority (a verification harness is defensible), but the decomposition could be
  biologized later. STATUS: the GENERALIZATION is QUEUED to wire (into `_verify`); biologizing the host harness is RESEARCH-NEEDED (low priority).

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
  prediction-error = SURPRISE on expectation-violating input. Wire → a surprise signal on the live turn. STATUS: QUEUED
  (audit in flight) — VERIFY it's a genuine spiking RPE, not a host formula (a host-formula RPE is itself a shortcut).
- **D3. Curiosity** (curiosity-inversion GO, learning-progress selection GO, curiosity-veto). Wire → drives a follow-up
  question on a novel/uncertain topic. STATUS: QUEUED (audit in flight).
- **D4. Comprehension MEASUREMENT** (multiframe / passive / object-relative comprehension GOs — "measurement of understanding
  of spoken language"). Wire → the brain knows when it understood vs didn't → an honest "I didn't follow that". STATUS: QUEUED.
- **D5. Episodic memory** (gap5 one-brain CAPSTONE 6/6 GO: converse→sleep-replay→converse). Wire → recall of PAST TURNS on the
  live turn. STATUS: QUEUED.
- **D6. Advanced WM binding** (wm-binding-advanced, de_risked YES). Wire → ≥2-referent anaphora / multi-slot. STATUS: QUEUED.

## E. MISSING MECHANISMS / BOUNDARIES (need RESEARCH before they can be wired — do NOT fake integration)
- **E1. Self-model / metacognition = BOUNDARY** (type-2 at chance despite a mis-calibrated GO gate, 2026-08-12). Genuine
  meta-d′ > 0 is the missing mechanism. RESEARCH-NEEDED (+ audit the mis-calibrated gate). Do NOT wire until it's a real GO.
- **E2. Internal worldview / affective world-model** — likely ABSENT/unvalidated as a production faculty (audit confirming).
  RESEARCH-NEEDED.
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
