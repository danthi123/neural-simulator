# Communicable-brain console integration — scoping (READ-ONLY)

**Date:** 2026-06-25
**Scope:** Backlog #2 (owner's TOP frontier — the communicable brain). The 3 core mechanisms are de-risked GO; this is the **integration plan** to wire them into the production conversational agent + the interact console, so the owner genuinely CONVERSES with the brain (not a retrieve→render→abstain RAG-feel pipeline). NO edits/runs/webapp performed; design only.

**The north-star (verbatim memory `project_communicable_brain_not_rag`):** "I want a brain I can communicate with, not just a fancy RAG." Open-ended, associative, value-driven conversation — grounded-by but NOT restricted-to stored knowledge (respond to "hi", form a view on a newly-described thing, speculate while flagging uncertainty). The BRAIN does the cognition (REJECT the LLM-free-generate cheat). The moat RELAXES to graded-confidence / speak-while-flagging, NOT removed. Talkativeness is LEARNED from interaction (the owner's 2026-06-24 refinement), not a hardcoded knob.

---

## 0. What is already GO (the inputs — verified from the JSON verdicts)

| Mechanism | Runner | Verdict | Key numbers |
|---|---|---|---|
| **GENERATE** (spiking draw) | `_followon2_spiking_wta_sampler_derisk.py` → wired DEFAULT-ON into `GenerativeReplayProposer` | **GO 6-seed** | spiking/host quality 0.948 (min 0.72), 15.5× random advantage, calibration KL 0.215, 0 moat leaks, lesion+shuffle collapse, provenance: draw read from `cp_firing_states`, 0 host-rng draws |
| **DECIDE-TO-SPEAK** (value/salience appraisal) | `_value_salience_appraisal_derisk.py` (`SpikingSpeakAccumulator` + `AppraisalTurn`) | **GO 3-seed** | value arm emits 4.0 vs plausibility-only 1.0 (+3.0 value-driven/seed), grounded adv 4.0×, calib spearman 1.0, 0 leaks, lesion collapses to plausibility-only, value⊥plausibility corr −0.039 |
| **LEARN-TALKATIVENESS** (learned speak-value Q) | `_learned_talkativeness_derisk.py` (`LearnedSpeakValue`) | **GO 3-seed** | Q taught 1.62 > simU 0.65 > disU 0.35, monotone learning curve, context-specific (decorrelated gap −0.13 vs value gap +0.71), non-circular (corr −0.047), 0 leaks, lesion (pin DA) abolishes |
| **TURN composition** (Probe-1) | `_communicable_brain_probe1_whatdoyouthink.py` (`WhatDoYouThinkTurn`) | **GO 3-seed** | emitted 7.7/seed (min 6), novel 1.0, render+VERIFY 1.0, grounded adv 18.2×, calib spearman 1.0, 0 leaks, LESION (LLM free-gen) caught-by-VERIFY 23/23 |

**The dependency chain is already wired in the de-risks:** the talkativeness turn (`LearnedTalkativenessTurn`) imports `SpikingSpeakAccumulator` + `triple_value` from the appraisal de-risk; both import `plausibility_score`/`hedge_for` from Probe-1; all three import the `GenerativeReplayProposer` (with the default-on spiking sampler) + the PPMI cortex + `RFPhasorComposer` + `BrainConversationalAgent`. **They already run on the SAME brain object.** What does NOT exist yet: a single class that fuses them into one turn, the wire-in to the production agent/console, the GPU-Qwen fluency drop-in for the generative path, and the console feedback UI + certain-vs-hypothesis rendering.

**Honest scope statement:** this is the owner's deepest, highest-variance frontier and a **real multi-stage integration**, not a one-shot. The mechanisms are individually validated at 3–6 seeds on the **CPU testbed brain** (the b2 PPMI cortex over the 8×8 taxonomy, `RFPhasorComposer` at D=64, the CPU `TemplateStubFaculty`). The integration must (a) fuse them into ONE turn, (b) prove the fused turn keeps every safety invariant, (c) wire it onto the **production** `BrainConversationalAgent`/`MultiTurnAgent`/`ChatBrain` over the **developed-brain bundle** codes, and (d) surface it in the console with the GPU spiking-Qwen and the teaching/feedback UI. Each is a gated stage below.

---

## 1. TURN ARCHITECTURE — how a user message flows

The current production turn (`ChatBrain.answer` → `gate → render → verify`, server.py `brain_chat`) has ONE channel: match a stored fact, render it, verify, or abstain. The communicable turn keeps that as the **known-fact channel** and ADDS a parallel **novel/generative channel** + a **social/phatic channel**, with an appraisal deciding which/whether to speak.

```
                          ┌─────────────────────────────────────────────────────┐
USER MESSAGE  ──►  classify the turn intent (the router, extended):
                          │  • a who/what/yes-no QUESTION about a known cue       │
                          │  • "what do you think about X" / a newly-described    │
                          │    thing  (an OPINION / view request)                 │
                          │  • a phatic / social opener ("hi", "how are you")     │
                          │  • a teaching signal ("tell me more", "elaborate",    │
                          │    "stop", "too much")  → routes to LEARN, §3         │
                          └───────────────┬─────────────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼──────────────────────────────────────┐
        ▼                                  ▼                                       ▼
 KNOWN-FACT CHANNEL              NOVEL / GENERATIVE CHANNEL                 PHATIC / SOCIAL
 (hard-gated, certain)          (flagged hypothesis, graded)               (templated, no claim)
 ── existing ChatBrain.gate     ── ASSIMILATE(X): PPMI neighborhood        ── a small fixed
    + composer.what_does/        ── PROPOSE candidate set: the b2            social repertoire
    who_does/is_it_true             GenerativeReplayProposer draws           ("Hi! Ask me about
 ── the no-confab moat:             NOVEL, graph-plausible, non-             what I know, or what
    abstains when nothing           contradictory triples ABOUT X           I think about X.")
    matches                         — each filler is a SPIKING draw         ── makes NO factual
 ── recall is CERTAIN                (cp_firing_states, GENERATE)            claim → no moat risk
                                  ── APPRAISE+RANK by WORTH =
                                     talkativeness·(w_v·VALUE +
                                     w_p·plausibility + w_f·familiarity)
                                     (VALUE = the LEARNED speak-value Q
                                      for this context; §3)
        │                         └───────────────┬───────────────────────┘
        │                                          │
        ▼                                          ▼
   (gate hit a fact)            SPIKING SPEAK/SILENCE ACCUMULATOR (DECIDE-TO-SPEAK)
        │                       ── the top candidate's WORTH drives the speak pool;
        │                          a fixed reticence drives the silence pool;
        │                          the spiking WTA winner decides emit-vs-silent
        │                       ── talkativeness scales the speak drive (the LEARNED
        │                          threshold; §3)  → speaks-more-where-supported
        │                                          │  (speak wins)
        └──────────────┬───────────────────────────┘
                       ▼
              RENDER via the spiking-Qwen fluency faculty (FLUENCY ONLY)
                       │   known-fact:  CONSTRAIN render of the recalled SVO
                       │   novel:       CONSTRAIN render of the proposed SVO
                       ▼
              VERIFY: re-parse the prose → SVO; require == the brain's SVO
                       │   (the gate→constrain→verify contract, both channels)
                       ▼
              EMIT
                  • known-fact  → asserted as CERTAIN ("Dog chases the cat.")
                  • novel       → FLAGGED HYPOTHESIS with the worth/plausibility
                                  hedge ("I'm not sure, but I'd guess dog likes bones.")
                  • silence     → either nothing, or a brief "I don't have a strong
                                  view on that." (the honest no-grounded-candidate case)
```

### The three turn cases the owner named, concretely:

1. **A who/what/yes-no question about a known cue** → the **known-fact channel ONLY** (the existing path). The generative channel does not fire for a direct recall (the answer is certain, not a hypothesis). `ChatBrain.gate` already does this; unchanged.

2. **"What do you think about X" / form a view on a newly-described thing** → the **novel/generative channel**. This is Probe-1's `WhatDoYouThinkTurn`, now with the appraisal deciding whether the top candidate is worth saying (vs Probe-1's "always emit the single best-plausibility"). For a **newly-described** thing: the description's content words are assimilated into the PPMI graph at runtime (the topic must be in the brain's learned vocab; an entirely-unknown word → "I don't know that word yet" — the honest abstain). The generated proposition is a **flagged hypothesis**, never a stored fact.

3. **Phatic / social ("hi")** → the **phatic channel**: a small fixed repertoire, makes no factual claim, so it carries no moat risk. This is NEW (the current `ChatBrain` abstains on "hi" because no fact matches). It needs the router to recognize a social opener and a tiny canned-response table — deliberately NOT generative (a greeting is not a grounded proposition).

### The decision logic (which channel, in one place):

A new `CommunicableTurn` orchestrator (the integration glue — see §3) runs:
- **router.classify(msg)** → {`question`, `opinion`, `phatic`, `teaching`}.
- `question` → known-fact channel (certain).
- `opinion` → generative channel: propose candidate set → appraise → spiking speak decision → render+verify → emit-flagged-or-silent.
- `phatic` → phatic table.
- `teaching` ("tell me more"/"stop") → the LEARN update (§3), AND optionally re-run the prior turn's topic with the updated talkativeness (so "tell me more" immediately elaborates).
- **A question that the known-fact channel ABSTAINS on** MAY fall through to the generative channel as a flagged hypothesis ("I wasn't told, but I'd guess …") — this is the owner's "grounded-by but not restricted-to stored knowledge". This fall-through is an **owner-steer fork** (§5): some owners want a clean "I don't know" for a direct question; others want the speculative guess. Default proposal: **fall through ON for "what about / do you think" framings, OFF for a bare factual "what does X do" question** (a bare question expects a fact; an opinion framing invites a guess).

---

## 2. SAFETY INVARIANTS preserved (and HOW each is enforced)

These are the load-bearing guarantees. Every one is already enforced in the de-risks; the integration must preserve them by **construction**, not re-implement them.

1. **No-confab moat HARD on the known-fact channel.** The known-fact channel is the unchanged `ChatBrain.gate` + `composer.what_does/who_does/is_it_true`, which abstain when nothing matches. The generative channel is an **additive emission channel**; it NEVER writes to the store (`comp.store` is not called on a proposed triple) and a who/what query on a proposed (un-stored) triple still abstains. Enforced + measured in all three de-risks as `moat_leaks == 0` (a who/what/yes-no on every emitted proposition still abstains). **Integration gate: assert 0 known-fact-channel leaks on the fused turn.**

2. **The novel channel speaks-while-flagging — never as a certain fact.** Every generative emission carries a hedge (`hedge_for`), graded by worth/plausibility ("I'd say…" → "I'm not sure, but I'd guess…"). The render layer must distinguish CERTAIN (known-fact) from HYPOTHESIS (novel) in the surface form AND in the console UI (§4). The de-risks assert `all_flagged` (every emission flagged). **Integration gate: every novel emission flagged; the console renders the two visually distinct.**

3. **LLM fluency-only + lesion-provable.** The spiking-Qwen faculty supplies SURFACE FORM only; the BRAIN supplies the content (the proposed/recalled SVO) and the choice (which/whether). VERIFY re-parses the faculty's prose back to an SVO and requires it to MATCH the brain's SVO; on a mismatch it regenerates once, then falls back to the brain's raw triple (`ChatBrain.render`). The **LESION anti-cheat** (sever the brain's proposal → let the faculty free-generate the content → VERIFY MUST reject) is GO 23/23 in Probe-1 (`InjectingStubFaculty` / the adversarial render). The grounded-language arc already proved the REAL spiking Qwen's drift is caught (it flipped a role-inversion; VERIFY rejected it — `2026-06-23-grounded-lang-INTEGRATION-GO`). **Integration gate: the lesion (free-generate) arm collapses on the fused turn with the REAL faculty (§4 stage 3).**

4. **Provenance — the cognition is the brain's, end to end.**
   - The generative DRAW is a spiking event (read from `cp_firing_states`; the proposer's `_followon2` provenance check asserts 0 host-rng draws on the draw path).
   - The speak DECISION is a spiking pool's firing (`SpikingSpeakAccumulator.decide` — a real Izhikevich WTA on a `SimulationBridge` slice), not a host `if score > thr`.
   - The talkativeness VALUE is learned by the three-factor rule (`LearnedSpeakValue`), lesion-provable (pin DA → no learning).
   - The PPMI likelihood + the RF composer store are the brain's, unchanged.
   - The host's only roles: routing which assembly fired, the SWR replay seed (which memory reactivates), and the surface bookkeeping — all explicitly within the BRAIN-BASED-ONLY boundary (`feedback_brain_based_only_standard`: the body/environment + bookkeeping are legitimate host code).

5. **Moat relaxed-not-removed, the OWNER-SANCTIONED relaxation.** Per `feedback_moat_not_hard_lossy_memory_ok`, the moat is a plus, not a hard gate — but the relaxation is SPECIFIC: the known-fact channel stays hard-gated (0 leaks), and the novel channel is added as a FLAGGED-hypothesis-only path. The de-risks call this "MOAT RELAXED-NOT-REMOVED (HARD)" and gate on it. The integration does not weaken the moat; it adds a clearly-flagged speculative channel beside it.

---

## 3. INTEGRATION POINTS — where each mechanism wires in

### 3.1 The new glue: a `CommunicableTurn` / `CommunicableComposer` (NEW, ~the size of `WhatDoYouThinkTurn` + the appraisal/learned wrappers fused)

The cleanest seam is a **new wrapper class** that holds a `BrainConversationalAgent` (or `MultiTurnAgent`) + the proposer + the value system + the spiking accumulator + the learned-talkativeness Q, and exposes the unified turn. It is the production-grade fusion of `WhatDoYouThinkTurn` (Probe-1) + `AppraisalTurn` (value) + `LearnedTalkativenessTurn` (learned) — which ALREADY share the same brain object and the same render+VERIFY contract, so this is **mostly reuse-by-import + a routing shell, NOT new mechanism.**

Reuse-by-import (verbatim, no `sim/` edit):
- `GenerativeReplayProposer` (default spiking sampler) + `build_plausibility`/`_category_pools` — PROPOSE (the draw is the spiking GENERATE).
- `SpikingSpeakAccumulator` (from the appraisal de-risk) — the spiking speak DECISION.
- `LearnedSpeakValue` + `context_code`/`code_overlap` (from the learned-talkativeness de-risk) — the LEARNED talkativeness Q + the three-factor update.
- `plausibility_score`, `hedge_for` (from Probe-1) — graded confidence.
- `_extract_svo_from_prose`, `_build_inflection_map` + the faculty's `render_svo`/`render_svo_regen` — RENDER+VERIFY (the existing `ChatBrain._verify` contract).

New glue (the only genuinely new code):
- `classify(msg)` — extend the existing `QuestionRouter` (in `brain_chat_tui.py`) with intent labels {question, opinion, phatic, teaching}. The question/who-what/yes-no matching already exists; add opinion-framing detection ("what do you think about X", "tell me about X") + a phatic table + a teaching-signal detector ("tell me more"/"elaborate"/"stop"/"too much").
- `turn(msg)` — the §1 decision logic, returning a structured record (channel, surface, certain-vs-hypothesis, worth, confidence, supporting/proposed SVO, the spiking-decision margin, abstained).
- `feedback(signal, topic)` — the teaching path (§3.3).

### 3.2 The agent — `BrainConversationalAgent` / `MultiTurnAgent`

The existing agent stays the **comprehension + known-fact + recall** core (unchanged; the moat lives here). The `CommunicableTurn` wraps it. Two integration touch-points:
- **Reuse the agent's `composer` + `parse`** for VERIFY (the brain's own comprehension) and the known-fact channel — exactly as Probe-1 does (`BrainConversationalAgent(...).parse` is the VERIFY re-parse).
- **Reuse the agent's vocabulary/codes** so the generative channel proposes over the SAME concepts the brain knows (the de-risks build the proposer over the same PPMI graph the composer's codes come from). On the production path the agent is built on the developed-brain bundle's `grounded_codes`; the PPMI graph for the proposer must be the **same learned co-occurrence the bundle was developed from** (the develop loop already learns it — see §3.4).

No change to `hear()`/`what_does()`/`describe()` is required; the communicable turn is a NEW entry point alongside `answer()`. (An optional later step: route `BrainConversationalAgent.answer` through `CommunicableTurn` behind a default-OFF flag, mirroring the `enable_*` pattern the agent already uses for every additive feature.)

### 3.3 The owner's reward-policy (the teaching / perceived-feedback set)

Per `project_communicable_brain_not_rag` (the 2026-06-24 refinement) the talkativeness is LEARNED from a **small, transparent perceived-feedback set**:
- **`elaborate` / "tell me more" / "go on"** → a perceived conversational reward → a phasic DA burst → `LearnedSpeakValue.update(topic, taught=True)` raises Q for that context (and PPMI-similar contexts via the eligibility kernel). **= positive (+).**
- **`stop` / "too much" / "that's enough"** → a perceived negative → a DA dip. The de-risk only modeled the positive (taught) reward; the integration adds the **negative** as `da < da_baseline` → a negative RPE → Q DECREASES for that context. **= negative (−).** (This is a small extension to `LearnedSpeakValue.update`: allow `da_reward` to be a signed delta, or add a `da_punish`. Cheap; the three-factor rule already handles `rpe < 0`.)
- **everything else** → neutral (baseline DA, no update).

The reward is **BRAIN-COMPUTED** (a DA-reward-US fires the SNc/striosome-value system). The de-risk used a CPU DA stand-in; the GPU follow-on (Tier-2 one-brain limbic core) reads the REAL shared `dopamine` concentration off the merged bridge. The **lesion anti-cheat** (pin DA → no learning) must hold on the production path — it distinguishes "the brain learned its talkativeness via reward-modulated plasticity" from "a Python counter".

**Persistence:** the learned Q is per-context state that must **survive across sessions** (the brain develops its conversational style over time — ties to the artificial-life develop loop). The natural home is the **developed-brain bundle** (`developed_brain_io.save_developed_brain` → `brain.json` manifest). Add an optional `speak_value_Q` blob to the manifest (a `{context_key: Q}` map, or the context codes + Q vector), loaded by `load_developed_brain` and re-installed into the `LearnedSpeakValue`. This is the "the brain develops its conversational style via interaction" loop made persistent. (Owner-steer: whether to persist Q per-bundle or globally — §5.)

### 3.4 The console — `ChatBrain` + `/api/brain-chat` + `_build_chat_brain` (server.py) + the TUI

The console (the INTERACT centerpiece) routes a turn through `ChatBrain.gate → render`. Integration:
- **`ChatBrain` gets a communicable mode.** Add `CommunicableTurn` as the turn engine behind a flag (default OFF preserves the current gate→render behavior byte-for-byte). When ON, `ChatBrain.answer(msg)` calls `CommunicableTurn.turn(msg)` and returns the structured record. The known-fact path inside it IS the current `ChatBrain.gate` (reused), so the moat is unchanged.
- **The renderer is already the spiking-Qwen faculty** on the console (`_get_warm_qwen_renderer` → `QwenRenderer`, the off-bridge spiking Qwen; the bridge-co-resident faculty is the demonstrated alternative). The generative channel renders through the SAME faculty + the SAME `_verify`. **No new renderer is needed** — the fluency-only + VERIFY contract is identical for a proposed SVO and a recalled SVO.
- **The `/api/brain-chat` response schema extends** (additive): add `channel` ("known"|"novel"|"phatic"|"silence"), `certain` (bool — known-fact vs flagged hypothesis), `confidence`/`hedge` (for novel), `proposed_svo` (for novel), `speak_decision` margin (for the B3-style activity panel). The existing `answer`/`abstained`/`recalled_svo`/`verified`/`activity` fields stay. The B3 brain-activity panel already shows decoded role chips + the matched engram + the RF firing gauge; ADD the speak-decision (speak vs silence spikes + margin) + which channel fired — so the appraisal is VISIBLE (the owner sees "the brain weighed this at worth 0.7 and chose to speak").
- **The teaching/feedback UI** (the genuinely new console surface): the owner needs a way to deliver the perceived-feedback set. Two options (owner-steer §5): (a) explicit buttons ("tell me more" / "that's enough") beside each turn, or (b) natural-language ("tell me more" typed as the next message, recognized by the router). Either fires `CommunicableTurn.feedback`, which updates Q and persists it to the bundle. The UI must SHOW that the feedback was registered (e.g. "noted — I'll say more about robots") so the teaching loop is transparent.

### 3.5 The one summary of reuse-vs-new

| Piece | Reuse-by-import | New glue |
|---|---|---|
| GENERATE (spiking draw) | `GenerativeReplayProposer` (default-on sampler) | — |
| DECIDE (spiking speak WTA) | `SpikingSpeakAccumulator` | — |
| LEARN (talkativeness Q) | `LearnedSpeakValue`, `context_code` | the signed-negative-feedback extension (`stop` → −) |
| graded confidence | `plausibility_score`, `hedge_for` | — |
| RENDER+VERIFY | `_extract_svo_from_prose`, `ChatBrain._verify`, `QwenRenderer` | — |
| known-fact channel + moat | `ChatBrain.gate`, `composer.what_does/...` | — |
| **turn orchestration** | — | `CommunicableTurn` (router intents + the §1 decision logic + feedback) |
| **phatic channel** | — | a tiny canned-response table |
| **console schema + feedback UI** | the `/api/brain-chat` + B3 activity scaffolding | the channel/certain/hedge fields + the teaching buttons |
| **Q persistence** | `developed_brain_io` bundle | a `speak_value_Q` manifest blob |

---

## 4. CHEAP-FIRST STAGED build order (with GO gates + anti-cheats)

Each stage is a real deliverable with a falsification gate. Stages are ordered cheapest-first; a NEGATIVE at any stage localizes the next bounded build rather than failing the whole arc.

### Stage A — CPU unified turn on ONE harness (the fused mechanism)
**Build:** a CPU de-risk runner (`_communicable_unified_turn_derisk.py`) that fuses all three mechanisms + Probe-1's render+VERIFY into ONE `CommunicableTurn` on the b2 testbed brain (PPMI cortex + RFPhasorComposer D=64 + CPU `TemplateStubFaculty`). Run the four turn cases (question / opinion / phatic / teaching) end-to-end.
**GO gate (≥3 seeds, promote to 6 if GO):**
- the unified turn REPRODUCES each component's GO in composition: novel emissions are generated+flagged (Probe-1), the value/learned-talkativeness arm speaks-more-where-taught (appraisal+learned), the spiking speak decision is load-bearing, the learning curve rises monotonically.
- **MOAT (HARD): 0 known-fact-channel leaks** across all turn cases; every novel emission flagged; stored facts still answer.
- **a "tell me more" feedback round raises the next-turn talkativeness on that topic** (the teaching loop closes end-to-end), and the **lesion (pin DA) abolishes it**.
- the **negative-feedback ("stop") extension** lowers Q for that context (the signed-RPE check).
**Anti-cheats:** shuffled-PPMI-graph collapses the novel-channel groundedness ≥3×; the LESION (free-generate the content) is caught-by-VERIFY; value⊥plausibility corr ≈0 (non-circular); the decorrelated-credit control flattens the taught/untaught gap (context-specific, not global vigor).

### Stage B — agent wire-in (the production `BrainConversationalAgent`)
**Build:** add `CommunicableTurn` as a default-OFF entry point on `BrainConversationalAgent`/`MultiTurnAgent` (mirroring the existing `enable_*` additive-feature pattern). Wire the proposer over the agent's own PPMI graph + codes. Persist/restore the learned Q via the bundle manifest.
**GO gate:** the full existing conversational test suite (`test_brain_conversational_agent`, `test_multi_turn_agent`, `test_one_brain_composer_agent`) passes VERBATIM with the flag OFF (byte-identical regression). With the flag ON: the unified turn works on the production agent + composer at the validated vocab; the moat (0 leaks) holds; a loaded bundle restores its Q (talkativeness survives a save/load round-trip).
**Anti-cheats:** flag-OFF byte-identity (the regression guard); the moat assertions carry over; provenance (spiking draw + spiking decision + learned Q) asserted on the production composer.

### Stage C — GPU spiking-Qwen fluency drop-in
**Build:** swap the CPU `TemplateStubFaculty` for the REAL spiking-Qwen faculty (`SpikingQwenFaculty` / `QwenRenderer`, the off-bridge spiking Qwen the console already warms) on the generative channel's render+VERIFY. This is the SAME contract the grounded-language INTEGRATION arc already passed (the faculty's drift was caught), now exercised on GENERATED (not just recalled) propositions.
**GO gate (GPU, ≥3 seeds):** the real faculty renders the brain's PROPOSED novel hypotheses fluently AND VERIFY accepts the faithful ones; the **LESION (steer the faculty to free-generate a self-chosen patient) is caught-by-VERIFY** on the generative channel (the decisive provenance proof with the real LLM in the loop); the moat holds; render+VERIFY rate is acceptable (the grounded-lang arc's 0.5B faculty occasionally object-fronts under the loose constrain prompt → the regen path recovers, as in `grounded_reply`).
**Anti-cheat:** the lesion arm with the REAL faculty MUST collapse (this is the "the LLM is not doing the cognition" gate — the whole point of the fluency-only design).
**Note (latency):** the console deliberately uses the host discourse planner for snappiness (`_get_rich_composer neural_planner=False`); the generative channel adds a propose+appraise+spiking-decision cost per opinion turn. Measure the per-turn wall-clock; the spiking accumulator is ~10ms/decision (CPU) and the propose loop is bounded by `n_attempts`. Per `feedback_long_local_runs_ok_confirm_cloud_cause`: this is wall-clock, not a VRAM wall — keep local; only flag cloud if VRAM exceeds 24 GB (it won't — the faculty is the same ~14 GB already demonstrated).

### Stage D — console surface (the INTERACT centerpiece)
**Build:** extend `ChatBrain` (communicable mode behind the flag) + `/api/brain-chat` (the additive schema fields) + the B3 activity panel (add the speak-decision + channel) + the teaching/feedback UI (the buttons or NL recognition) + the certain-vs-hypothesis rendering (visually distinct: a known fact stated plainly; a hypothesis prefixed with the hedge + a "hypothesis" badge + the confidence). Persist Q to the bundle on feedback.
**GO gate:** a live console session (the owner's actual UX): ask a known fact → certain answer; "what do you think about X" → a flagged hypothesis; "hi" → a social reply; "tell me more" → the brain elaborates AND its talkativeness on that topic rises (visible in the activity panel); reload the brain → the learned talkativeness persists. The moat is VISIBLE (a who/what on an unstored cue → "I don't know about that", the B3 panel shows scanned-N-none-matched).
**Anti-cheat:** the console must NEVER present a hypothesis as a certain fact (the certain-vs-hypothesis flag is load-bearing in the UI); a flagged hypothesis is never written to the store (a subsequent direct query on it still abstains — testable live).

---

## 5. OWNER-STEER FORKS (what needs the owner's call)

1. **Known-fact-abstain → generative fall-through.** When a direct question hits the moat (no stored fact), should the brain offer a flagged guess ("I wasn't told, but I'd guess…") or stay clean ("I don't know about that")? **Proposal:** fall-through ON for opinion framings ("what about/do you think"), OFF for bare factual questions. The owner may want it always-OFF (strict) or always-ON (chatty). This sets the line between "communicable" and "RAG-clean".

2. **The teaching/feedback UI form.** (a) Explicit buttons ("tell me more" / "that's enough" beside each reply) — unambiguous, transparent, easy to log. (b) Natural-language ("tell me more" typed, router-recognized) — feels more like real conversation, the owner's stated mental model ("like a real person"), but the router must reliably distinguish a teaching signal from a content request. **Proposal:** start with (a) for the de-risk-clean signal, add (b) as the natural-feel layer once the reward-policy is validated. Owner decides the priority.

3. **How `certain` vs `hypothesis` is surfaced.** A badge + hedge ("🟡 hypothesis: I'd guess…") vs only the linguistic hedge vs a separate UI lane. Load-bearing for the safety invariant (the owner must never mistake a guess for a fact). Owner's call on the exact visual; the REQUIREMENT (they are distinct) is non-negotiable.

4. **Talkativeness Q persistence scope.** Per-bundle (each developed brain has its own conversational style — fits the artificial-life "watch a brain develop" north-star) vs a global Q (one talkativeness profile across brains). **Proposal:** per-bundle (ties to the develop loop). Owner decides.

5. **Multi-turn / discourse.** Should the generative channel participate in the multi-turn WM loop (a flagged hypothesis becomes a discourse referent a later "tell me more about that" can elaborate)? The `MultiTurnAgent` WM loop already exists; the question is whether a HYPOTHESIS (not a fact) should be held as a referent. **Proposal:** yes, but tagged as a hypothesis-referent so a follow-up about it stays in the speculative channel (never promotes to certain). Owner's call.

6. **The default talkativeness prior + the value weights.** `talkativeness`, `w_value`/`w_plaus`/`w_fam`, the speak/silence drive bands — the appraisal de-risk used conservative defaults (speaks somewhat more than Probe-1, only on grounded topics). The owner may want the starting point more or less chatty before the learning kicks in. **Proposal:** ship the de-risk's conservative defaults; the learning adapts from there. Owner can set the prior.

7. **GPU vs CPU faculty on the console for the generative path.** The console warms the off-bridge spiking Qwen (~14 GB, demonstrated local). The bridge-co-resident faculty is the "one brain" purist option but slower. **Proposal:** off-bridge warm Qwen for the console (snappy, the current default); the co-resident faculty is the later one-brain-consolidation step. Owner's call on whether the console must be co-resident.

---

## Appendix — file map (where each thing lives / will be touched)

**De-risk inputs (read-only, GO):**
- `research/runners/_followon2_spiking_wta_sampler_derisk.py` — `SpikingWTASampler` (GENERATE; wired default-on into the proposer)
- `research/runners/_value_salience_appraisal_derisk.py` — `SpikingSpeakAccumulator`, `AppraisalTurn`, `triple_value`, `build_concept_value` (DECIDE)
- `research/runners/_learned_talkativeness_derisk.py` — `LearnedSpeakValue`, `context_code`, `code_overlap`, `LearnedTalkativenessTurn` (LEARN)
- `research/runners/_communicable_brain_probe1_whatdoyouthink.py` — `WhatDoYouThinkTurn`, `plausibility_score`, `hedge_for` (the TURN composition)

**Production surfaces (reuse, mostly unchanged):**
- `research/runners/brain_conversational_agent.py` — `BrainConversationalAgent`, `BridgeParser` (comprehension + known-fact + recall + the moat; `.parse` is the VERIFY re-parse)
- `research/runners/multi_turn_agent.py` — `MultiTurnAgent` (discourse WM)
- `research/runners/_genfrontier_b2_generative_replay_derisk.py` — `GenerativeReplayProposer` (PROPOSE; the wire-in seam is `_sample_weighted`, default spiking)
- `research/runners/_grounded_lang_integration_derisk.py` — `SpikingQwenFaculty`, `_extract_svo_from_prose`, `grounded_reply` (the GPU fluency + the gate→constrain→verify)
- `research/runners/brain_chat_tui.py` — `ChatBrain` (`gate`/`render`/`_verify`), `QwenRenderer`, `StubRenderer`, `QuestionRouter`, `_build_tiny_demo`/`_load_self_knowledge` (the console turn engine + the router to extend)
- `research/runners/developed_brain_io.py` — `save_developed_brain`/`load_developed_brain`/`is_developed_brain_bundle` (bundle IO; the Q-persistence home)
- `webapp/server.py` — `/api/brain-chat` (`brain_chat`), `_build_chat_brain`, `_get_warm_qwen_renderer`, `_get_rich_composer`, `/api/brains` (the console endpoints + schema to extend)

**New code (the integration glue):**
- `research/runners/_communicable_unified_turn_derisk.py` (Stage A — the CPU fused-turn de-risk)
- a `CommunicableTurn` / communicable mode on `ChatBrain` + `BrainConversationalAgent` (Stages B/D)
- the `speak_value_Q` manifest blob + the console feedback UI + the certain-vs-hypothesis rendering (Stages C/D)
