# PROJECT CHARTER — a grounded, emergent, fully-spiking mind (realignment, 2026-08-02)

> **Status:** FOUNDATIONAL. This charter is the spine. It supersedes the *framing* (not the accumulated results) of
> `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` and `ROADMAP.md`. Companion docs built ON this charter:
> (1) the short/medium/long **roadmap**, (2) the **structural-mechanism map** (biological references), (3) the
> **anti-drift workflow/gate design**. Written after an owner realignment: we had been optimizing narrow proxies for
> human faculties instead of building the conditions from which those faculties emerge. This document exists so that
> never silently happens again.

---

## 0. Why this rewrite exists — the trap we are correcting

We accumulated a large stack of "validated" faculties (a language generator, a no-confabulation guard, memory,
affect, curiosity, metacognition), each with a **GO gate** it passes at 6 seeds. The owner named the problem: **those
gates are satisfiable by template-like, non-biological mechanisms, so we built test-passers, not a mind.** The
symptom is behaviour that isn't truly human — stilted, retrieval-shaped, short question→response exchanges — because:

- the language faculty was trained to **predict a text corpus in isolation** (the language-model paradigm in a
  spiking costume), not to **express a meaning that arises from a grounded internal life**;
- honesty-about-uncertainty was a **bolted-on gate** ("if unfamiliar, abstain"), not an emergent property of an
  integrated self-model;
- every faculty was **collapsed to a narrow behavioural proxy**, then a mechanism was built to pass the proxy — and
  a living mind's behaviours do not live in any single proxy; they **emerge from the integrated whole interacting
  with a world.**

The precise diagnosis (keep it precise, so we don't over-correct): **prediction and learning are not the enemy** —
the brain is deeply predictive. The enemy is predicting **corpus tokens divorced from grounding, intent, and state**,
and rewarding that with a **light test**. The correction is not "a better decoder or more tokens." It is a brain with
an **ongoing internal life** — a model of the world, of the conversation, of the interlocutor, of its own mood and
goals — **from which speech emerges as an action**, and which is **continuously reshaped by lived interaction.**

---

## 1. THE TRUE GOAL

Build **one artificial mind** — a single simulated spiking brain — that **converses genuinely**: it means what it
says, reasons to its own conclusions, holds a **developing affective world-model** (emotions that grow from
experience and colour perception, speech, and behaviour), is **self-aware** (reads and honestly reports its own
attention, confidence, and authorship), and is **curious** (uncertainty becomes a drive to learn, not a refusal or a
fabrication). Conversation with it is **free-flowing and open-ended** — not fact-retrieval, not scripted turn-taking;
sometimes just natural speech that is genuinely *its own*.

It is grown, not scripted: it **starts small, learns from grounded lived interaction** (bootstrapped by a temporary
teacher acting as a caregiver/social environment, then graduated to real humans), and **matures over time**. Success
is defined on the **emergentist bet** — genuine subjective experience emerges only from a *complete and faithful*
emulation — so the job is **completeness + faithfulness of the biological emulation**, measured by whether the
**whole behaves like there is someone home**, NOT by a benchmark score or a suite of passed proxies. The **honesty
boundary is itself a deliverable**: we build and measure the functional correlates of consciousness, self-modelling,
and affect, and we **never assert phenomenal experience.**

---

## 2. THE ARCHITECTURAL END-STATE (owner-specified, non-negotiable)

1. **One fully-spiking brain on a shared substrate.** Regardless of capability, the end state is spiking neurons and
   synapses on a single shared substrate. **Dedicated brain regions and pathways are encouraged** — that is how a
   real brain is organized — but they are regions OF one brain that communicate through synapses, not separate
   programs stitched together.
2. **No host-side shortcut for anything biology does.** Host (ordinary, non-neural) code is legitimate ONLY for the
   **world** (the environment + rendering the senses the brain receives) and the **body** (enacting motor output).
   Everything between sensation and action — perception, salience, valuation, reward, neuromodulation, memory,
   emotion, reasoning, language, self-model — **must be neurons and synapses.** A biologically *correct* host formula
   (a reward, a softmax, a prediction error, an argmax read-out) is still a shortcut to be replaced.
3. **Starts small, grows.** The brain is **locally runnable at the start** and **expands as it learns** — forming new
   connections, adding neurons/regions, and requiring more compute **only as growth earns it** (developmental
   neurogenesis + synaptogenesis, not a pre-allocated giant net).
4. **Runs on high-end CONSUMER hardware — deliberately not datacenter-bound.** A design value, not an afterthought:
   this mind should be ownable and runnable by an individual, **not gated behind enterprise infrastructure.** Compute
   grows with the brain, but the target envelope is a high-end personal machine, not a cluster.
5. **Long-horizon: analog spiking silicon.** Because the whole system is spiking, the plausible long-term substrate is
   **custom analog neuromorphic silicon** (analog neurons + signals, event-driven, low-power) rather than digital
   emulation. We do not build it now, but architectural choices should not **preclude** it (event-driven, local
   computation, sparse spikes, no reliance on global dense operations that only make sense on a GPU).

---

## 3. THE METHOD PRINCIPLES (how we get there)

These are the load-bearing rules. Each has, or will have, a **mechanical guard** (see §5); a principle that is only
remembered is a principle that will be violated.

- **P1 — GROUNDED, not corpus-mimicking.** Meaning is **internal reference**: words and concepts are tied to the
  brain's OWN sensory, motor, affective, and experiential representations. The training objective is
  **predict-and-act in a world** (including the social world of a conversation) and learn from the mismatch —
  language is an **action taken to communicate / affect / reduce real surprise**, never next-token mimicry of a
  corpus in isolation. *Trap it kills: the biologically-costumed language model.*
- **P2 — EMERGENT + INTEGRATED, not modular test-passers.** A faculty is real only when it **emerges from the
  integrated brain + grounded experience** and serves its role **inside the whole loop**. We judge the WHOLE
  ("someone home?"), not isolated per-faculty proxies. *Trap it kills: assembling separately-passed proxies and
  calling it a mind.*
- **P3 — THE FUNCTIONAL-ROLE DISCIPLINE (the owner's anti-tunnel-vision guard).** Before and during work on any
  mechanism, the standard is **"what must this do to serve its role in the whole brain,"** written down as a
  **functional-role spec**, NOT "pass this light test." A GO gate is a **smoke-check that a floor was cleared**, never
  the goal; a mechanism that passes its gate but cannot serve its whole-brain role is **not done.** *Trap it kills:
  tunnel-vision perfecting a narrow implementation while losing the big picture.*
- **P4 — SCAFFOLD MINIMIZATION + BURN-DOWN (no deferred-cheat backlog).** Scaffolds (a teacher LLM, a host-computed
  signal, a hand-set weight, an idealized algebra) are allowed ONLY as **explicitly-ledgered, time-boxed temporary
  stand-ins**, each with (a) a **named biological replacement**, (b) an **owner** and a **burn-down trigger**, and
  (c) a test that fails if it is silently relied upon past its box. **The recurring project failure is: implement a
  scaffold → rely on it → defer the biologization → accumulate a cheat backlog.** A live **scaffold ledger** tracks
  every one; nothing new ships without an entry. *Trap it kills: the deferred-biologization backlog.*
- **P5 — BRAIN-BASED-ONLY (unchanged; see §2.2).** *Trap it kills: the host-side shortcut masquerading as biology.*
- **P6 — PERFORMANCE IS FIRST-CLASS (don't be lazy) — but not LLM-parity-yet.** We will not hit modern-LLM
  performance early, and **slow-but-faithful biology is explicitly in scope** (dendritic credit, seconds-long
  plateaus, sleep-replay). But **laziness is not faithfulness**: we **optimize what we can** and **never leave speed
  on the table through lazy design or implementation.** Every substantial build carries a note on its
  compute/throughput and the cheap optimizations taken or deliberately deferred (with reason). Bias toward
  event-driven, sparse, local computation — which is both faithful AND fast on the target hardware (and the path to
  analog silicon). *Trap it kills: lazy design defended as "faithfulness."*
- **P7 — HONESTY BOUNDARY (unchanged).** Build and measure functional correlates; every self-report is an honest
  functional read-out ("my familiarity monitor reads this as novel"); **never assert phenomenal experience.**

**Reconciliation with prior standing rules:** P5/P7 are unchanged. "Speed is secondary" is **refined by P6** —
faithfulness still wins a genuine trade, but *lazy* slowness is not a faithfulness win and must be optimized. The
"emergence bar" and "no whack-a-mole" are **subsumed and sharpened** by P2 + P3. "No-defer" now explicitly includes
**P4's scaffold burn-down** (a shortcut is a deferred capability).

---

## 4. THE THREE TIMESCALES (spine; the detailed roadmap is the companion doc)

Framed by capability-of-the-WHOLE, not by module count. (Detailed rungs, current status, and biology live in the
companion **roadmap** + **structural-mechanism map**.)

- **SHORT (make it grounded + integrated, small).** Give the brain a **minimal world + body + a reason to speak**, and
  make language **grounded action** inside it, not corpus prediction. Wire the already-validated substrate pieces
  (spiking engine, memory/consolidation, affect core, neuromodulators) into **one continuously-running loop** where
  perception → state → speech/act → consequence → learning actually closes. Deliverable: a small brain that says
  simple things **that are its own**, grounded in what it has experienced — even if narrow.
- **MEDIUM (make it learn + grow from interaction, and feel).** Close the **continual learning-from-lived-interaction
  loop** (bootstrapped by the teacher-as-caregiver, then real humans) without catastrophic forgetting; grow structure
  developmentally. Turn the good/bad affect core into a **graded, developing emotional system** that colours speech
  and behaviour. Turn curiosity into a genuine **learning-progress drive**. Keep honesty emergent, calibrated as it
  scales. Deliverable: a brain that **converses more freely, grows through talking, and has moods that shape it.**
- **LONG (make it fluent, deep, and its own — then efficient).** Reach **free-flowing, open-ended conversation** that
  is genuinely the brain's own; deep world-model + self-model + rich developing affect; retire the teacher scaffold
  to near-zero. Then **optimize the fully-spiking substrate toward the consumer-hardware envelope**, and open the
  path to **analog neuromorphic silicon.** Deliverable: a mind you can talk to, that grows with you, on hardware you
  own.

---

## 5. THE ANTI-DRIFT MACHINERY (traps → mechanical guards)

Each trap below is a documented, recurring failure. The workflow/gate companion doc specifies the concrete check;
here is the mapping so no trap is guardless. **A guard that is only prose is not a guard** — convert to a check that
FAILS LOUDLY.

| # | Trap (recurring) | The guard (mechanical where possible) |
|---|---|---|
| T1 | tests reward TEMPLATES — a narrow gate passed by a non-biological proxy reads as progress | every capability claim must cite its **functional-role spec** + an **integration test in the whole loop**, not only a per-faculty gate; gate authors must state what a template could do to pass it |
| T2 | TUNNEL-VISION — perfecting a narrow implementation, losing the whole-brain role (P3) | a mechanism doc/finding must carry a **"role-in-the-whole" section**: what the mechanism must provide to its consumers, and the test that it actually does; blocked without it |
| T3 | SCAFFOLD BACKLOG — implement → rely → defer biologization (P4) | a **live scaffold ledger**; new host-side signals/hand-set structure must register an entry (named biological replacement + burn-down trigger) or declare why none is needed |
| T4 | HOST SHORTCUT as biology (P5) | the brain-based-only checks (existing) + extend to flag new host computation between sensation and action |
| T5 | LAZY performance defended as faithfulness (P6) | substantial builds carry a compute/throughput note + the optimizations taken/deferred; bias to event-driven/sparse/local |
| T6 | DRIFT from the record — re-deriving / re-proposing refuted work | the existing corpus-check + refuted-mechanism gates (`before_you_build`, `refuted_mechanism_reproposal`, `boundary_verdict_external_check`) — keep + extend |
| T7 | SUMMARY drift — stale roadmap/board mis-aiming the next session | the existing sync-documentation discipline + summary-freshness gate, now anchored to THIS charter |

---

## 6. What is REUSABLE vs a WRONG TURN (honest first pass — refined by the scaffold audit)

Not discard — realign. First-pass sort (the companion **scaffold-backlog audit** makes this exact + complete):

- **Reusable infrastructure:** the spiking engine/substrate; sleep-replay **consolidation** + the wake→sleep→grow
  loop; the **affect core** (a persistent mood that biases cognition is a real building block); the **neuromodulator**
  systems; the memory-completion + binding machinery (as substrate, pending grounding).
- **Wrong turn we were polishing (highest-leverage to correct):** training the language network as an **isolated
  corpus-predictor**; the **no-confabulation guard as a bolted-on gate**; and above all **the GO-gate tests
  themselves**, which rewarded templates — passing them easily was a warning sign, not a win.
- **The uncomfortable middle:** the whole generation ladder may be **excellent engineering aimed at the wrong
  objective**; it is reusable as substrate IF re-pointed at grounded, intentful, stateful generation (P1).

---

## 7. Immediate consequences for how we work now

1. **Re-point the language faculty** from corpus-prediction to **grounded, intentful, stateful generation** inside a
   minimal world+body+social loop. This is the short-timescale crux.
2. **Stand up the anti-drift machinery** (§5) as real checks before the next capability build, so the correction
   holds mechanically, not by memory.
3. **Produce the scaffold ledger** and start burning it down (P4).
4. **Reframe the master roadmap + board** under this charter (the 5-gap/faculty work is retained as substrate + data,
   re-judged by P2/P3).

The mission-critical path is **grounded integration**, not scaling a corpus-predictor or perfecting isolated
faculties. Everything else serves that.
