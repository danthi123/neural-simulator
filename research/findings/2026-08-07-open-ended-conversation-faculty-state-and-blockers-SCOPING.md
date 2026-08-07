---
type: finding
status: contributing
date: 2026-08-07
mechanism: open-ended-grounded-conversation-scoping
lane: E-language
---

# Open-ended fluent GROUNDED conversation — where we stand + the blockers (owner-directed deep-research scoping, 2026-08-07)

Owner steer (verbatim intent): "itching to have a live sim brain I can talk to reasonably fluently and in an
open-ended manner, not just question→answer" — and "it wasn't just about training speech generation but also
grounding it in world-view, perception, emotion, and understanding of the effects of saying certain things."
This doc maps the CURRENT state (post-2026-07-23 pivot, post-August work) of every faculty open-ended grounded
conversation needs, and the ranked blockers. Method: 6 parallel deep-research faculty maps (RAG + read-in-depth)
+ the coordinator's own reads of the framing plans (`docs/plans/2026-07-22-genuine-conversation-...md`,
`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`) and the generation wall (DE-RISK-4). Read-only; no `sim/` edit.

## THE UNIFYING FINDING — the dominant blocker is INTEGRATION, not missing capability

Across all six faculties the same pattern holds: **the pieces are overwhelmingly BUILT and 6-seed-validated as
ISOLATED REGIONS / SELECTORS on toy bridges — they are NOT WIRED TOGETHER into one sustained conversational loop on
the real substrate.** The deployed console feels "Q→A" not because the faculties are missing but because it is,
by design, a **retrieve-a-stored-triple → render-in-a-fixed-carrier-frame → verify-against-the-store** pipeline,
with the no-confab moat structurally excluding free generation from asserting facts. Underneath that integration
gap sit a small number of genuinely-UNBUILT organs (grounded free generation, a consequence-of-speaking layer, a
predictive forward model). Perception grounding is real but narrow. This is good news: the path to "talk to it
open-ended" is mostly **grounding + wiring faculties we already have**, plus ~3 targeted new builds — not inventing
the stack.

## Faculty-by-faculty state (honest scope)

| Faculty | State | Honest scope / the gap |
|---|---|---|
| **Grammar + bounded production** | HAVE (emergent, on spikes) | grammar self-organized (function words/order/inventory), 7 constructions incl. ditransitive; renders on the spiking WKV. BUT bounded to corpus-attested inventory. |
| **The brain's OWN emergent production** | HAVE (real faculty, on spikes, learned) — but BOUNDED | competitive-queuing serial order (EMERGE-59), self-organized grammar (EMERGE-62..77), and today's brain-native percept/knowledge→word selection — the CONTENT is sourced from the brain's own knowledge/state and structured by emergent grammar. THIS is the target faculty. Bounded to short/framed messages + a small inventory. |
| **Open-domain fluency via the 267M WKV LM** | EXISTS but it is a SCAFFOLD (a spiking LLM), UNGROUNDED + UNINTEGRATED | the 267M `run4_d2048` WKV autoregressively free-generates coherent multi-sentence prose + scales + is spiking-forward-faithful 6/6 — but it is a **BPTT-trained, next-token, transformer-adjacent LM**: it tells TinyStories (not what the brain knows), is not wired into the turn, and single-pass multi-fact synthesis confabulates (DE-RISK-4). ⚠️ **OWNER STEER (2026-08-07): a spiking LLM is NOT the goal — it is a temporary fluency scaffold/teacher to be minimized+retired in favor of the brain's OWN emergent production faculty (the row above).** Do not enshrine autoregressive-LM generation as the end state. |
| **World-model + reasoning (F1/F2)** | GO (real cognition) but RETRIEVES/INHERITS/COMPLETES, cannot SIMULATE | associative co-occurrence graph on spikes + emergent taxonomy (EMERGE-30) + learned valence (r=0.81) + genuine inference (inheritance, cancellation, transitive B>D, graded completion, re-entrant multi-hop deliberation P1.2). NO learned forward model `s,a→s'` → no causal/counterfactual/hypothesize→test→revise; premises acquired only for the co-occurrence case. |
| **Perception grounding** | Real but NARROW + half-joined | validated spiking retina→Gabor-V1→complex + a category-pooler feeding spiking inference→conversation (6-seed) + today's brain-native percept→word naming (on-bridge 6-seed GO). BUT the invariance-BINDING stage (V1→IT/pooler) is the join gap (raw-V1 naming = boundary), and it's a handful of oriented-bar toy objects — the conversational vocabulary is NOT perceptually grounded at breadth. |
| **Emotion / affect (F3)** | Standing state GO (6-seed) but a bistable LATCH + UNWIRED to speech | valence/arousal sources GO on spikes; 3 neuromodulator axes dissociate 6-seed; a standing V×A affect-STATE region persists across a turn + causally biases recall/speak (6-seed); wired neurally into a workspace ONCE (P1.2). BUT it's a good/bad bistable latch (not a graded circumplex → can't grade enthusiasm/hesitance), appraisal input is host-fed, and it does NOT run on the production composer → doesn't color real speech yet. |
| **Curiosity (F5)** | Drive + reward + selection GO but UNWIRED, no question-emission | DR-1 curiosity inversion on-bridge spiking GO; reward-omission veto GO; learning-progress-MAXIMIZING selection 6-seed CPU-proxy GO (noisy-TV-immune). BUT no natural-language wh-QUESTION generation (A→W spell is a scope comment, not code), no interactive teacher answering it, and on-bridge LP memory is fragile (1/6). |
| **Self-model / metacognition (F4)** | Strong as COMPONENTS, not fused | five 6-seed-GO spiking regions (attention/confidence self-schema, learned agency/BORN, ACC/aPFC uncertainty monitor, familiarity gate) + a GNW workspace that speaks its reasoned answer. BUT isolated — NOT fused into one persistent self the conversation reads; the honesty wire-in is PARTIAL (routes a recall-score, not the calibrated correctness monitor); the neural abstain/hedge behavior is 6-seed PARTIAL (3/6). |
| **Pragmatics / effects-of-speaking** | FRAGMENTED probes, NONE wired — the owner's named gap | recursive ToM WITH an RSA listener model (6-seed GO), affective ToM (GO), a first-person speech→consequence loop (GO, but the "listener" is a world-rule), learned "worth-saying" value (GO). BUT none is wired into the speaking loop; the brain speaks as a consequence-free read-out; NO communicative-success reward (reward-from-being-understood); NO discourse-referent (token-vs-type) layer (the named prerequisite). |
| **Conversational loop + memory (F6)** | Within-session context REAL (6-seed); episodic memory MISSING | spiking multi-turn context buffer (referent + running SVO carry across turns, pronoun resolution, moat-safe, 6-seed); semantic memory persists across sessions/reboots (BridgeLineage); genuine endogenous elaboration (volunteers unsaid facts). BUT the develop-loop's "converse" is one-way static fact-injection (the interactive teacher-loop P2.1 does NOT exist; the contingent-teacher atom is near-GO 5/6, blocked on an instrument fix); the brain's own affect/curiosity/self-state does NOT drive the next turn; NO episodic/autobiographical memory (can't say "yesterday you told me X"). |

## The ranked blockers to open-ended grounded conversation

1. **OPEN-ENDED GENERATION — and the fork the owner named (scaffold-LM vs the brain's OWN faculty).** Two paths, and
   they must not be conflated:
   - **(near-term, functional) GROUNDED FREE GENERATION via the WKV LM scaffold:** condition the 267M WKV on the
     brain's retrieved knowledge-neighbourhood, generate one free multi-sentence reply, enforce the moat POST-HOC
     (re-parse VERIFY). Gets "discusses freely, grounded" QUICKLY — but it is a **spiking-LLM scaffold**, and per the
     owner's 2026-08-07 steer it is a TEACHER/stand-in to be retired, NOT the end state. Legitimate only as a
     temporary bridge + a distillation teacher for the faculty below.
   - **(the real target) THE BRAIN'S OWN EMERGENT PRODUCTION scaled to open-domain:** the content forms from the
     world-model/affect/self, gets encoded by the emergent grammar + competitive-queuing circuitry (EMERGE-59..77)
     and articulated — no next-token LM. This is BOUNDED today (short/framed messages). The master roadmap's R4
     surpass is exactly this: **scale the spiking HTM Temporal-Memory generator (`fused_htm_permanence_update`) +
     teacher-distilled soft targets + developmental domain growth, and RETIRE the transformer** (gap#4 for the fully
     learned version). This is the honest end-state and it is a real frontier (~4-orders data/scale + the mechanism),
     grown via the teacher-loop — not a quick win. **The scaffold-LM path buys a talkable demo now; the emergent-
     production path is the mission.** Track the scaffold as a burn-down item, never a destination.
2. **THE INTEGRATION LOOP (co-resident single-bridge, self-driven).** Wire the standing affect state (colors
   speak-margin/word-choice/forthcomingness), the curiosity ask-loop, the self-schema/metacog honesty read, the
   multi-turn context, and memory into ONE conversational bridge (`OneBrainComposer`). Makes conversation
   self-driven + affect-colored + curiosity-asking, not reactive Q→A. Reuses validated components; never run together.
3. **PRAGMATICS / consequence-of-speaking (+ communicative-success reward) — the owner's named gap.** NOT near-absent:
   the ToM ladder (false belief, recursive+RSA implicature, affective ToM — all 6-seed GO) exists, and the CLOSEST
   existing piece is **common-ground audience design** (`2026-06-27-tier2.4-common-ground-GO`: VOLUNTEER informative
   vs SUPPRESS redundant — a real GO model of what saying-a-thing does to a listener). What is genuinely MISSING is
   the **perlocutionary** model ("say X → the listener now believes/does Y") and a **communicative-success reward
   (reward-from-being-understood)** — because **speaking is not a BG-selected action that lands in the world with a
   consequence** (the one closed speech→consequence loop returns a physical/energy change, not a listener-belief
   update; `structural-mechanism-map.md` §8 names this OPEN). BUILD: make SPEAK a BG-selected act whose contingent
   consequence returns as reward, let the (existing) ToM/common-ground model GATE what is said at production time, on
   a discourse-referent (token-vs-type) foundation. The *learned* version routes through gap#4; a first *wired*
   version (fixed-operating-point ToM gating production + a contingent success signal) does not.
4. **THE FORWARD MODEL `s,a→s'` — the one missing cognitive organ.** Gates causal/counterfactual/hypothesize→
   test→revise reasoning + a simulation-grounded (not just retrieval) world-model + value bound from lived reward.
   BUILD on the UNBLOCKED substrate (frozen reservoir + local readout / numpy-rate deep-credit) — do NOT wait on
   deep-credit-on-spikes (gap#4-on-spikes is mapped/deprioritized; the emergence engine routes around it).
5. **THE INTERACTIVE TEACHER-LOOP (P2.1) — the developmental engine.** Turn the develop-loop's one-way "converse"
   into a real exchange: teacher selects ZPD material, corrects the brain's own spiking outputs (the near-GO arc-A
   e-prop contingent-teacher atom, after its instrument fix), and ANSWERS the brain's curiosity questions (couples
   F5+F6). This is how all the above GROW toward open-endedness rather than being hand-tuned.
6. **EPISODIC / AUTOBIOGRAPHICAL MEMORY wiring.** Semantic facts persist; episodes don't — needed to refer to shared
   conversational history. Hippocampal/engram (Tier-3) territory.
7. **DEEPER (the long frontier, not first-loop prerequisites):** fully-spiking locally-learned free generation
   (gap#4-gated; currently at parity), open-domain generation data-scale (~4 orders; interim = the minimized
   transformer scaffold), graded affect (SFA/dendritic line-attractor vs the bistable latch), perception breadth
   (invariance stage + natural images), scaffold retirement.

## The honest bottom line (how far is "talk to it open-ended, not Q→A")

- **A FIRST integrated open-ended loop is much closer than it looks — IF we accept the scaffold-LM for fluency.**
  Blockers 1(near-term)+2 are mostly grounding+wiring components that are already GO, and would yield a brain that
  discusses a topic freely, colored by a standing affect state, that sometimes asks its own questions, over a real
  multi-turn context — a genuine "talk to it more openly" demo, not Q→A. ⚠️ But the fluency in that demo rides the
  WKV **spiking-LLM scaffold**, which the owner has explicitly said is NOT the end state. So this demo is a
  functional milestone + a distillation teacher, not the mission being met.
- **GENUINE open-endedness — reason to novel conclusions, model the effects of what it says, feel-and-grade affect,
  refer to its own history — needs the unbuilt organs** (forward model #4, pragmatics #3, episodic memory #6) and
  the developmental teacher-loop (#5) to GROW them. That is months, not weeks, and it is the honest Stage-4
  "Conversant" bar in the master roadmap.
- **The mission's non-negotiables are intact:** the honesty boundary is a real, reachable functional read-out
  (metacog monitor separates correct-from-error); the whole thing stays one spiking substrate (the deep-credit and
  fully-spiking-generation purity items are the deferred-not-abandoned frontier, routed around by the reservoir +
  local-readout emergence engine for now).

## Staged build plan (the path)

- **STAGE A — the first integrated open-ended loop (the nearest big win):** (A1) grounded free-generation turn
  (blocker 1); (A2) co-resident single-bridge integration of affect + curiosity + self/metacog + multi-turn + memory
  on the composer (blocker 2); (A3) close the curiosity ask→A→W question-emission + the interactive teacher-loop P2.1
  (blockers 5, partial). Verify they COMPOSE (adversarial: does grounded-gen + moat + affect-coloring hold together?).
- **STAGE B — the cognitive organs:** (B1) the forward model on the unblocked substrate (blocker 4); (B2) the
  pragmatics/consequence-of-speaking layer + communicative-success reward + discourse-referent token/type (blocker 3);
  (B3) episodic/autobiographical memory (blocker 6).
- **STAGE C — deepen + biologize:** graded affect line-attractor; fully-spiking locally-learned free generation
  (gap#4 via the teacher bridge); open-domain scale; perception invariance + breadth; retire scaffolds.

Stage A is a parallel decompose-build-integrate-VERIFY effort across ~3 workstreams that must then compose — the
kind of multi-agent orchestration (ultracode/workflows) pays off on.

## Sources (the decisive findings the maps rest on — read these to verify)
- Generation: `2026-07-20-grounded-fluent-conversation-DE-RISK-4-multifact-prose-wall-and-workaround.md`;
  `2026-07-19-gap1-WKV-learned-KV-recurrence-RUNG1a-6seed-GO-...md`; `2026-07-01-open-ended-grounded-discussion-scoping.md`.
- World-model/reasoning: `2026-08-02-gap4-depth-rescue-untestable-on-spikes-...md`;
  `2026-07-11-SYNTHESIS-emergent-generation-ladder-and-emergence-bar.md`; `2026-07-02-emerge30-emergent-superordinate-GO.md`.
- Perception: `2026-07-23-perception-closure-scoping.md`; `2026-08-07-grounded-retina-percept-naming-raw-V1-boundary.md`.
- Affect/curiosity: `2026-07-24-P0.3-affect-state-region-6seed-GO.md`; `_p1_2_affect_directed_deliberation_derisk.py` (commit b30981b58);
  `2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md`; `2026-07-23-DR1-curiosity-inversion-ONBRIDGE-spiking.md`.
- Self/pragmatics: `2026-06-27-conv-thinking-research-discourse-pragmatics.md` (the pragmatics blueprint);
  `2026-08-01-W4-recursive-theory-of-mind-...6seed-GO.md` (RSA); `2026-08-07-laneC-neural-abstain-hedge-gate-...` (honesty behavior PARTIAL).
- Loop/memory: `2026-08-01-arcA-integrated-brain-learns-from-contingent-teacher-via-eprop-on-spikes-5of6.md`;
  `2026-07-10-D3-event-multiturn-coherence-GO.md`; `2026-07-01-fluid-conversation-phase17-persistence-GO.md`.
- Framing: `docs/plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md` (F1-F6);
  `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` (faculty map, Stage-4 Conversant, walls ledger R4).
