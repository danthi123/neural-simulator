# Realigned plan: sim as standalone conversational agent

**Date:** 2026-05-11 (post-user-checkin)
**Status:** ACTIVE — primary path, supersedes earlier Path 3 framing
**Author:** autonomous arc, after user clarified goal

## Goal

Biology-grounded spiking neural simulator as a **standalone**
conversational agent. **No external LLM** — not as orchestrator, not
as feature extractor, not as fallback. Fully local execution (CPU or
GPU). Eventually capable of multi-word, multi-turn, semantically
meaningful conversation.

## Why the previous Path 3 framing was wrong

The 2026-05-11 strategic re-eval framed Path 3 as **"LLM + sim-as-memory-
subsystem"** — an external 3B-param LLM (Phi-3 / Llama / Qwen) doing
language, sim doing persistent memory. That's tractable but it abandons
the harder goal. The user's actual goal: the sim does language too.

Concrete consequence: the work shipped on 2026-05-11 that I'm carrying
forward vs. deprecating:

| Component | Status |
|---|---|
| BridgeMemory API (store/recall/speak/forget/consolidate) | KEEP — these are the sim's cognitive operations, not LLM-facing tools |
| Bridge Lineage Manager + continuous learning | KEEP — this is the differentiator |
| Synapse tiering + auto-growth + NumPy backend | KEEP — scaling infrastructure |
| Dashboard chat panel + chips | KEEP — direct sim interaction (the "user types, sim responds") UI |
| LLMMemoryOrchestrator + MockLLM | KEEP-BUT-REFRAME — MockLLM is just a pattern-matched intent router (sim-native), not an "LLM" |
| OllamaLLM adapter + SIM_LLM_BACKEND env var | KEEP-AS-SECONDARY — useful if user ever wants the vector-DB-for-LLMs path; **not active development** |
| Phase 3.3 design (real LLM integration) | DEPRECATED for primary path |
| Phase 3.4 design (multi-session assuming LLM driver) | NEEDS REFIT — the multi-session test is still relevant, but reframed around sim-native interaction |

## Capabilities table (where we are, where we need to go)

| Capability | Status | Distance |
|---|---|---|
| Bind/recall direction words | ✅ Working | done |
| A→W generation | ✅ Working | done |
| Sleep-replay consolidation | ✅ Real-ops shipped | done |
| Multi-session persistence | ✅ Lineage system working | done |
| Catastrophic-forgetting resistance | ✅ Phase 1.4 BRANCH A 5/6 retention | done |
| 8-word vocab (synonym) | ✅ Tier 2.1, 5/6 W→A | done |
| **In-vivo new-vocab binding** | ❌ **0/4 → 1/4 at 200/400 events** | **BLOCKER — must fix** |
| 12-16 word vocab | partial (synonym12/16); needs multi-seed | small (gated on in-vivo fix) |
| 20-30 word vocab | infra (auto-grow Phase A); not validated | medium |
| 64+ word vocab | NEGATIVE @ XL encoding 2026-05-11 | large — arch insufficient |
| Compositional syntax (2-word phrases) | infra (Tier 2.3 PFC verb pool); never positive | large |
| Sentence-level understanding | not started | huge |
| Sentence-level generation | not started | huge |
| Reasoning | not started | huge |
| Hundreds-of-words conversation | not started | enormous |

The "in-vivo new-vocab binding" row is the immediate blocker. Today's
sweep confirmed novel keys (apple/river/mountain/forest) don't bind
reliably even at 800 events. Every subsequent scaling step depends on
this working.

## Step-by-step plan

### Step 1 (active, ~1-2 weeks): Fix in-vivo new-vocab binding

Run `research/runners/investigate_invivo_binding_fix.py` testing four
variants:
- V0 — vanilla control
- V1 — pre-bind anchoring (zero edges, build from zero)
- V2 — curriculum interleave with anchor words
- V3 — recall-only tail fine-tune

Validation: ≥ 4/6 seeds correct on 4 made-up keys at default n_events.

If V1/V2/V3 works → bake into `BridgeMemory.store()` as the new default.

If NONE works → escalate: either Step 4 immediately (dendritic learning
rewrite) or design a 5th variant (e.g. cortex_X anchored learning).

### Step 2 (~2-3 weeks): Multi-seed validate synonym12 / synonym16

**Step 2a — 12-word vocab: ✅ ALREADY VALIDATED (2026-05-09).**
3/3 GO unanimous at scaled arch (n_motor=2000): primary retention
95.5%, synonym retention 115.0%. Counts toward this step; no new
work needed.
Source: `research/findings/2026-05-09-Phase1.3-Tier2.1-12word-scaled-3seed-CONFIRMED.md`

**Step 2b — 16-word vocab: PARTIAL (1/1 smoke seed PASSES).**
16-word smoke at seed 42 (`consolidation_synonym_16word_scaled_smoke`
preset, n_motor=2000) gives **primary retention 90%, synonym
retention 108.7%, verdict GO**. Single-seed positive — capacity
hypothesis appears to extend to 16 words.

Real Step 2b work: run `consolidation_synonym_16word_scaled_medium`
preset at 3-6 seeds (already wired up in webapp; ~3.5 hr/seed at
medium config). No new code needed. ~10-20 hours total wall clock.

Validation: ≥ 4/6 seeds at ≥ 80% primary retention, ≥ 60% synonym
retention (matches 12-word criteria).

If Step 1's fix enables in-vivo binding, Step 2b can also test
adding NEW words on top of the trained 16-word vocab (capacity
test with novel keys).

If passes: 16-word vocab unlocked. Update capability_status.json.

### Step 3 (~2-4 weeks): Compositional syntax — Tier 2.3 PFC verb pool

Re-activate the existing Tier 2.3 infrastructure (`phrase_trainer.py`,
`phrase_eval.py`), but with an **action_gate redesign** addressing the
2026-05-07 finding ("Tier 2.3 6seed PARTIAL"):

> **Phrase accuracy < direction-only accuracy across ALL 6 seeds.**
> Phrase composition consistently HURTS, not helps. Hypothesis:
> `action_gate` boosts ALL motor pools indiscriminately when PFC is
> active, making non-target pools fire too easily. Indiscriminate
> excitatory boost ≠ verb-context-gated selection.

Two redesigns to try (sketched per `text_minimal_isolation.py:622`):

**v2a — Per-direction PFC subpools (recommended).** Split `dlpfc_verb`
into 4 subpools, one per direction. Each subpool fires only for
verbs that map to its motor pool. Train each subpool to gate ONLY its
own motor pool. 4 neuromodulators (`action_gate_N/E/S/W`) instead of
1. Biologically plausible: real PFC has direction-tuned subpopulations.

**v2b — Inhibitory action_gate.** action_gate inhibits non-target motor
pools rather than exciting the target. Mechanism: when dlpfc_verb is
active, all motor pools EXCEPT the target get an inhibitory drive.
The motor_TARGET pool fires by default; verb context just suppresses
competitors. Biologically plausible: basal ganglia disinhibition
model (Mink 1996).

With Step 1's binding fix + v2a or v2b, the two-word phrase task
should have a real shot.

Validation: ≥ 4/6 seeds where two-word phrases produce coherent
motor sequences (not just one motor pool) AND phrase ≥ direction-
only accuracy (not the inversion we saw in 2026-05-07).

If passes: compositional syntax unlocked. This is a real research
contribution.

If fails: clear signal that the current arch can't compose even with
verb-direction gating fixed. Escalate to Step 4 (dendritic learning).

### Step 4 (months, real research bet): Dendritic learning rewrite

The 2026-05-05 W→A verdict said "global scalar feedback fails at
biological scale; need apical-basal dendritic learning OR predictive
coding." Step 3's outcome will tell us if Step 4 is necessary or not.

Design doc: `docs/plans/2026-05-05-dendritic-learning-design.md`
(1.5-2 month scope).

If Step 3 succeeds compositionally without dendritic learning, Step 4
can be deferred. If Step 3 fails, Step 4 becomes the only path to
sentence-level capability.

### Step 5 (after Step 4 lands): Scale to 64-word vocab + sentence-level

The earlier NEGATIVE 64-word @ XL encoding result was on the current
architecture. With dendritic learning, retry the same experiment.

Validation: 64-word vocab with reliable binding + recall + at least
short-phrase composition.

### Step 6+ (year+ horizon): Sentence-level understanding + generation,
reasoning, conversation.

These are the huge-distance items. Each will require its own design
arc once we have the foundation from Steps 1-5.

## What gets validated at each step

The user's risk tolerance is high but each step has a clear PASS / FAIL
criterion. Negative results are publishable findings just as positive
ones are.

| Step | PASS criterion | FAIL implication |
|---|---|---|
| 1 | ≥ 4/6 seeds correct recall on novel keys | Escalate to Step 4 immediately |
| 2 | ≥ 4/6 seeds at ≥ 60% W→A on synonym12 + synonym16 | Architecture won't scale linearly; need redesign |
| 3 | ≥ 4/6 seeds with coherent 2-word phrase motor sequences | Need dendritic learning before composition possible |
| 4 | Dendritic learning produces measurable credit-assignment improvement vs scalar DA at toy scale | Pivot to predictive coding alternative |
| 5 | 64-word vocab works with dendritic learning | Need scale beyond what local hardware allows |
| 6+ | Sentence-level emergence | Hard limit of biology-grounded approach at our scale |

## What stays warm but inactive

The Ollama / SIM_LLM_BACKEND scaffolding remains in the codebase as
the **secondary path** (sim as continuous-learning memory layer for
external LLM agents). This isn't garbage code — it's a valid
production application that several real groups would pay for.

Active development on the secondary path resumes ONLY when:
- The primary path hits a hard ceiling we can't overcome, OR
- The user explicitly redirects

## Wall clock estimates

- Step 1: 1-2 weeks (mostly compute; experiments are quick to set up)
- Step 2: 2-3 weeks
- Step 3: 2-4 weeks
- Step 4: 1.5-2 months (real implementation + validation)
- Step 5: 1 month
- Step 6+: open-ended

Total to "demonstrable conversational sim" with compositional syntax
and reasoning: **6-12 months** at autonomous pace, possibly longer
with negative-result detours.

## Local-only commitment

Every step above runs on local hardware. CuPy on the RTX 3090 (24 GB)
or NumPy on CPU. No cloud dependencies. No external LLM. The deploy_
to_cloud.sh script from earlier is for sweep parallelization only; it
is NEVER required for the sim itself to function.

If a future step hits the 24 GB VRAM ceiling, the sim still runs on
CPU (slower). Synapse tiering already allows pathway-grained eviction
to SSD, so even >24 GB synapse counts are reachable on the same box.

## What this plan does NOT promise

- A 6-month "ChatGPT replacement". Sentence-level conversation is the
  endpoint, not the next milestone. The capability table is honest
  about distance.
- Working 64+ word vocab at the current scale. The 2026-05-11 NEGATIVE
  result stands; Step 5 retries it AFTER Step 4 lands.
- That every step will produce a positive result. Some will produce
  NEGATIVE findings, and those are valuable.

This plan is a **research arc**, not a feature roadmap. Each step
informs the next; some steps may pivot the entire plan.

## Resume directive

I'm resuming autonomous work on Step 1 now. The n_events sweep is
finishing in the background (will give us the final "more events
doesn't help" datapoint). Once that lands, I'll launch the four-
variant binding-fix experiment immediately.
