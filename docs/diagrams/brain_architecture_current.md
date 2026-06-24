# Brain architecture — current state (2026-06-23)

Maintainable, **as-implemented** Mermaid flowcharts of the whole simulated
brain, kept in step with `research/findings/AUTONOMOUS_STATE.md` and the
2026-06-23 findings. These render directly on GitHub.

The whole-brain pipeline, top to bottom:

> **spiking substrate** (`SimulationBridge`: Izhikevich + resonate-and-fire
> complex-synapse neurons) → **conversational pipeline** (parser → composer
> role-filler binding → the no-confab **moat** → hippocampal **sleep
> consolidation**) → the **learned cortex** (PPMI stream cortex: learn word
> meanings from listening) → the **grounded-language faculty** (a
> spiking-Qwen LLM gated by the brain's knowledge: **gate → constrain →
> verify** — brain = knowledge, LLM = phrasing only, hallucination-proof by
> construction) → **bridge co-residence** (the faculty consolidated onto the
> `SimulationBridge`) → the artificial-life **develop loop** (day → converse
> → consolidate(sleep) → grow(auto-growth tiers) → persist(lineage) → next
> day).

These three Mermaid diagrams are the **current-state, whole-stack** view.
The two exhaustive region/pathway detail graphs (every region, every distinct
synapse, the honesty legend) remain the hand-authored SVGs in this folder —
they are scoped to a single builder each and are the source of truth for the
region inventory:

| Detail graph | Scope | Files |
|---|---|---|
| **Navigation brain** | every region + pathway of `build_bg_brain_regions()` — BG action-selection cascade, spiking superior-colliculus orienting, spiking actor-critic, thalamus/TRN, the accumulate→commit spiking decision (now the default read-out), cerebellum, hippocampus, dlPFC | [`brain_navigation.svg`](brain_navigation.svg) · [`.png`](brain_navigation.png) |
| **Conversational brain** | the production `OneBrainComposer` pipeline + the `build_biological_brain_regions()` region inventory (language I/O, Wernicke/semantic/Broca, concept pools, multimodal hub, hippocampal consolidation, dlPFC verb WM) | [`brain_conversational.svg`](brain_conversational.svg) · [`.png`](brain_conversational.png) |
| **Master map (SVG)** | the earlier hero overview: one brain (nav + conversation as disjoint slices) + the 3 validated cross-brain routes + the co-resident generalization stack + the honesty legend | [`brain_master.svg`](brain_master.svg) · [`.png`](brain_master.png) |

> The SVG master map predates the grounded-language faculty, bridge
> co-residence, and the develop loop (the 2026-06-23 arc). **Diagram 1 below
> is the current master map** — it adds those three layers on top of the
> consolidated one brain. See "Honesty & scope" at the bottom for the
> brain-based-only standard the markers encode.

---

## 1. Master architecture map — the whole brain

The full stack: the shared **spiking substrate** carries the **conversational
pipeline** and the **navigation** brain as disjoint neuron slices on one
update loop; the **learned cortex** supplies grounded concept codes; the
**grounded-language faculty** turns retrieved facts into fluent prose under a
firewall; and the **develop loop** runs the whole thing forward over simulated
days. Solid arrows = the live signal path; the faculty's firewall edges are
highlighted.

```mermaid
flowchart TB
    %% ---- world / body I/O (host-legitimate boundary) ----
    World([🌍 World · retina render · sensory input]):::io
    Body([🦾 Body · act on motor output]):::io
    UserQ([💬 User question / sentence]):::io
    Reply([🗣️ Grounded fluent reply · or 'I don't know']):::io

    subgraph BRIDGE["🧠 ONE SimulationBridge — one spiking update loop (Izhikevich + resonate-and-fire complex synapses)"]
      direction TB

      subgraph SUB["Spiking substrate"]
        direction LR
        IZH["Izhikevich point neurons<br/>(cascades · pools · parser)"]:::sub
        RF["Resonate-and-fire neurons<br/>+ complex synapses<br/>(phasor binding · the composer)"]:::sub
      end

      subgraph CORTEX["Learned cortex — learn word meanings by listening"]
        direction LR
        STREAM["PPMI stream cortex<br/>online rate-Hebbian co-occurrence<br/>(hears the corpus word-by-word)"]:::mem
        CODES["grounded concept codes<br/>(generalize across similar concepts)"]:::mem
        STREAM -->|corr M,C ≈ 0.89| CODES
      end

      subgraph CONV["Conversational pipeline"]
        direction TB
        PARSER["PARSER<br/>word-order × voice → who-did-what<br/>(voice-invariant, vocab-agnostic)"]:::conv
        COMPOSER["COMPOSER (role-filler binding)<br/>bind words → facts · persistent fact store<br/>query · negate · clauses · multi-hop"]:::conv
        MOAT{{"no-confab MOAT<br/>exact-match recall →<br/>abstain when unknown"}}:::moat
        SLEEP["hippocampal SLEEP CONSOLIDATION<br/>SWR replay → cortex (no forgetting)"]:::mem
        PARSER -->|roles| COMPOSER
        COMPOSER --> MOAT
        COMPOSER <-.->|encode / retain| SLEEP
      end

      subgraph NAV["Navigation brain (disjoint slices, co-resident)"]
        direction LR
        V1["V1 / IT perception"]:::nav --> SCOL["superior colliculus<br/>orienting (spiking)"]:::nav --> BG["basal-ganglia cascade<br/>→ accumulate→commit decision"]:::dec --> MOT["motor read-out<br/>(spiking WTA, default)"]:::mot
        SNC["dopamine SNc<br/>reward / value (spiking)"]:::da -.->|RPE broadcast| BG
      end

      subgraph FAC["GROUNDED-LANGUAGE FACULTY — spiking Qwen2.5-0.5B (co-resident, ◇ see Diagram 2)"]
        direction LR
        GATE["① GATE<br/>brain has content?"]:::fac
        CONSTR["② CONSTRAIN<br/>condition on the fact's<br/>words + roles"]:::fac
        RENDER["spiking LLM render<br/>(phrasing only)"]:::fac
        VERIFY["③ VERIFY<br/>re-parse output ==<br/>stored fact?"]:::fac
        GATE --> CONSTR --> RENDER --> VERIFY
      end
    end

    %% ---- I/O wiring ----
    World -->|pixels| V1
    MOT -->|which way to move| Body
    UserQ -->|a sentence| PARSER

    %% ---- the conversational answer path through the faculty firewall ----
    CODES -.->|grounded codes| COMPOSER
    MOAT ==>|content| GATE
    MOAT ==>|abstain · 'I do not know'| Reply
    VERIFY ==>|verified| Reply
    VERIFY -.->|drift → reject / regenerate| RENDER

    %% ---- cross-brain routes (each 6/6-seed GO; ◇ see master SVG) ----
    PARSER -.->|A · spoken instruction → command_route gate| BG
    BG -.->|B/C · perceived object → engram-tag / grounded phasor| COMPOSER

    %% ---- the develop loop wraps the whole bridge (◇ see Diagram 3) ----
    LOOP{{"♻️ DEVELOP LOOP — repeat per simulated day:<br/>WAKE·converse → SLEEP·consolidate → GROW·auto-growth tiers → PERSIST·lineage"}}:::loop
    BRIDGE -.->|one simulated day| LOOP
    LOOP -.->|next day, resume| BRIDGE

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef sub fill:#eae6f3,stroke:#6c4fb0,color:#1d1d1f;
    classDef mem fill:#dcefd3,stroke:#2f8f4e,color:#1d1d1f;
    classDef conv fill:#d6eaf8,stroke:#2e6da4,color:#1d1d1f;
    classDef moat fill:#fdebd0,stroke:#c8791a,color:#1d1d1f;
    classDef nav fill:#fdebd0,stroke:#c87f2e,color:#1d1d1f;
    classDef dec fill:#e9dcf5,stroke:#7d3c98,color:#1d1d1f;
    classDef mot fill:#f8c9c4,stroke:#b03a2e,color:#1d1d1f;
    classDef da fill:#fcf3cf,stroke:#d68910,color:#1d1d1f;
    classDef fac fill:#d1f2eb,stroke:#138d75,color:#1d1d1f;
    classDef loop fill:#fce8f0,stroke:#c0397b,color:#1d1d1f;
```

**How to read it.** The brain holds the **knowledge** (learned cortex +
composer + moat); the faculty supplies **phrasing only**. The conversational
answer leaves the moat one of two ways: **content** (a stored fact exists → it
flows into the faculty's gate→constrain→verify firewall and emerges as a
verified fluent reply) or **abstain** (no fact matches → the reply is "I don't
know", the moat never breached). Navigation is a co-resident, disjoint-slice
brain on the same bridge, joined to the conversation by three validated
cross-brain routes (drawn in full on the master SVG). The whole bridge is run
forward over simulated days by the develop loop.

---

## 2. The grounded-language faculty — gate → constrain → verify

The owner's decoupling, realized: **the brain supplies and verifies the
CONTENT; the spiking LLM supplies only the fluent SURFACE FORM.** Three
enforcement layers (cheapest first) make the faculty hallucination-proof *by
construction* — a fluent-but-false render is re-parsed and rejected before it
can reach the user. Validated end-to-end with the **real spiking Qwen2.5-0.5B**
in the loop: grounded → fluent-correct, untaught → abstain, adversarial drift →
caught.

```mermaid
flowchart TB
    Q([💬 User query]):::io
    P["PARSER<br/>query → cue {agent, action} (or topic)"]:::conv

    subgraph BRAINSIDE["🧠 BRAIN = KNOWLEDGE (holds + verifies the content)"]
      direction TB
      STORE["COMPOSER / fact store<br/>exact-match recall over bound facts"]:::conv
      G{"① GATE<br/>does a stored fact match the cue?"}:::gate
      STORE --> G
    end

    ABSTAIN([🚫 'I don't know'<br/>moat held — no fabrication]):::abstain

    subgraph FIREWALL["🛡️ THE FIREWALL — content cannot originate in the LLM"]
      direction TB
      C["② CONSTRAIN<br/>condition the faculty on the retrieved<br/>fact's WORDS + ROLES (slot-fill, not free generation)<br/>→ degrees of freedom = grammar + phrasing only"]:::fac
      subgraph LLMSIDE["🤖 LLM = PHRASING ONLY (spiking Qwen2.5-0.5B, T=16)"]
        R["spiking render<br/>fact → fluent surface form"]:::llm
      end
      V{"③ VERIFY<br/>re-parse the output with the SAME parser;<br/>asserted SVO == the stored fact?"}:::gate
      C --> R --> V
    end

    OUT([🗣️ Grounded fluent reply]):::io

    Q --> P --> STORE
    G ==>|match → content| C
    G ==>|no match → abstain| ABSTAIN
    V ==>|matches stored fact| OUT
    V -->|DRIFT e.g. role inversion 'Rabbit chased fox' → reject → regenerate| R

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef conv fill:#d6eaf8,stroke:#2e6da4,color:#1d1d1f;
    classDef gate fill:#fdebd0,stroke:#c8791a,color:#1d1d1f;
    classDef fac fill:#d1f2eb,stroke:#138d75,color:#1d1d1f;
    classDef llm fill:#e8dff5,stroke:#7d3c98,color:#1d1d1f;
    classDef abstain fill:#f8d7da,stroke:#b03a2e,color:#1d1d1f;
```

**The three layers.**
- **① Gate** (the moat) — the composer / familiarity gate decides *whether
  there is content to speak*. No stored fact matches the cue → the faculty is
  given **nothing to render** → the reply is "I don't know". Recall 1.0 +
  **0 false-accepts** at ~67 facts / 138-word vocab across 3 seeds.
- **② Constrain** (the hallucination-reducer) — the faculty is conditioned on
  the retrieved fact's **words and roles** (slot-filling / constrained
  decoding), so its only freedom is grammar and phrasing — *not* facts.
- **③ Verify** (the moat-preserver) — the faculty's output is **re-parsed by
  the same parser** and its asserted subject-verb-object checked against the
  stored fact. Any drift is rejected and regenerated. The proof: the real 0.5B
  LLM *did* try to hallucinate (it inverted a role, "Rabbit chased fox"), and
  **verify caught it** — the conservative failure direction is over-abstention,
  never confabulation.

> Mapping to grounded-generation SOTA: the brain's structured store **is** the
> knowledge graph; the composer's abstention **is** the calibrated-confidence
> gate — a *stronger* guarantee than soft retrieval (exact binding-match vs
> cosine similarity).

**Bridge co-residence.** The faculty is not a bolted-on service: the full
24-layer spiking Qwen2.5-0.5B (494M) runs **on the live `SimulationBridge` RF
substrate** — 14 GB resident (fits < 24 GB → local), bit-exact to the
off-bridge spiking forward, coherent generation. The "one brain" north star for
language: the fluent faculty is co-resident on the brain's own substrate.
(Honest scope: a feasibility demonstration — it runs but is slow; the perf
lever is the usability follow-on, and full functional integration on one bridge
is the deeper step.)

---

## 3. The artificial-life develop loop — the brain DEVELOPS over simulated days

The whole brain is run forward over simulated time. Each "day" the brain
**hears** a developmentally-graded curriculum and **learns** concept codes
(real stream-cortex Hebbian learning), **converses** on the codes it learned,
**consolidates** them in sleep so they stick (no catastrophic forgetting),
**grows** as it masters a tier, and **persists** so the next day resumes the
same brain. Validated at GPU scale, 1 seed: over 4 days vocab 6→24, facts
2→11, recall 1.0, retention 1.0, moat 0-false-accept; persists + resumes across
days; the frozen-brain anti-cheat holds (plasticity-off learns nothing).

```mermaid
flowchart LR
    PREV([📅 Day N · the developed brain<br/>resumed from the lineage]):::day

    subgraph DAY["One simulated day — every stage maps to a validated subsystem"]
      direction LR
      WAKE["☀️ WAKE / LEARN<br/>stream cortex HEARS the day's curriculum<br/>word-by-word → learns new concept codes<br/>(rate-Hebbian co-occurrence, corr M,C ≈ 0.89)"]:::wake
      CONV["💬 CONVERSE<br/>MultiTurnAgent on the learned codes<br/>parse · store · recall · abstain · yes-no · chain<br/>(the no-confab moat: 0 false-accepts)"]:::conv
      SLEEP["🌙 SLEEP / CONSOLIDATE<br/>SWR self-replay → the day's learning STICKS<br/>+ retain OLD facts (no catastrophic forgetting)"]:::sleep
      GROW["📈 GROW (if a tier mastered)<br/>auto-growth TierPromoter scales the brain<br/>+ logs a growth event"]:::grow
      PERSIST["💾 PERSIST<br/>BridgeLineage atomic save<br/>(learned codes · facts · vocab · tier · dev-log)"]:::persist
      WAKE --> CONV --> SLEEP --> GROW --> PERSIST
    end

    NEXT([📅 Day N+1 · resume + keep developing<br/>not a blank slate]):::day

    PREV --> WAKE
    PERSIST --> NEXT
    NEXT -.->|next simulated day| PREV

    FROZEN([🧪 anti-cheat: FROZEN brain<br/>plasticity OFF → hears but learns nothing<br/>→ competence must NOT rise]):::cheat
    FROZEN -.->|controls| WAKE

    classDef day fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef wake fill:#fcf3cf,stroke:#d4ac0d,color:#1d1d1f;
    classDef conv fill:#d6eaf8,stroke:#2e6da4,color:#1d1d1f;
    classDef sleep fill:#dcdcf5,stroke:#5b5bd6,color:#1d1d1f;
    classDef grow fill:#dcefd3,stroke:#2f8f4e,color:#1d1d1f;
    classDef persist fill:#fce8f0,stroke:#c0397b,color:#1d1d1f;
    classDef cheat fill:#f8d7da,stroke:#b03a2e,color:#1d1d1f;
```

**Feasibility (the north-star number).** ~2.2 min/day at GPU smoke scale → a
compressed "week" ≈ 16 min, a "month" ≈ 1 hr, a "year" ≈ ~13.5 hr (an
overnight **local** run, no VRAM wall). Simulating weeks/months/years of
development is a tractable, local problem.

**Honest scope.** 1-seed; small smoke vocab (24); consolidation = the validated
self-replay stand-in (full SWR on the conversational bridge deferred); growth =
the TierPromoter *decision* + lineage growth-event (the heavy arch rebuild +
weight-transfer is the GPU follow-on). "Development" here is vocab/facts
accumulation + retention, not yet open-ended conversational sophistication.

---

## Honesty & scope (the brain-based-only standard)

These diagrams are accurate **to the degree the biology is implemented in this
simulator**, not to the degree real brains are organized — an honest map of the
code, including its reductions.

**The brain-based-only standard** (owner directive, `CLAUDE.md`): even where a
host-side computation is biologically correct, it is a **shortcut** if the
*brain* isn't doing it. Host code is legitimate only for the **environment**
(world state + sensory rendering, the 🌍 I/O nodes) and the **body** (acting on
motor output, the 🦾 node). Everything between sensation and action is meant to
be neurons and synapses.

Two current load-bearing honest residuals, drawn here:

- The **grounded-language faculty's** spiking LLM is **co-resident** on the
  bridge and bit-exact, but it is a *feasibility* demonstration — slow (the
  perf lever is pending) and not yet *functionally* interacting with the
  conversational brain on one bridge (the deeper integration step). Its
  knowledge does NOT originate in the LLM — the firewall (Diagram 2) enforces
  that the brain holds and verifies all content.
- The **develop loop's** consolidation and growth stages run validated
  *stand-ins* (self-replay retention re-test; TierPromoter decision + lineage
  event) at 1-seed smoke scale; the full SWR-on-the-conversational-bridge and
  the arch-rebuild weight-transfer are flagged GPU follow-ons.

For the exhaustive per-region / per-synapse honesty markers (collapsed `×N`
pools, host I/O boundary `⌂`, documented shortcut `⚠`, negative/boundary
pathways `✗`, the substrate-wide SH-1…SH-14 list), see the hand-authored detail
SVGs and the extraction spec
[`research/findings/2026-06-09-brain-architecture-flowchart-spec.md`](../../research/findings/2026-06-09-brain-architecture-flowchart-spec.md).

## Provenance

These Mermaid diagrams are composed from the 2026-06-23 findings (read directly):

- `research/findings/2026-06-23-grounded-lang-INTEGRATION-GO.md` — the
  gate→constrain→verify capstone (real spiking Qwen renders gated facts; drift
  caught).
- `research/findings/2026-06-23-grounded-lang-SCALED-GO.md` — robust at ~67
  facts / 138-word vocab; moat 0-false-accept × 3 seeds.
- `research/findings/2026-06-22-grounded-language-faculty-scoping.md` — the
  P1 (fluency) / P2 (knowledge) / P3 (grounding) architecture + the firewall
  sketch.
- `research/findings/2026-06-23-bridge-coresidence-DEMONSTRATED.md` — the full
  24-layer spiking Qwen on the SimulationBridge, 14 GB local, bit-exact.
- `research/findings/2026-06-23-longitudinal-develop-loop-GPU-GO.md` +
  `research/runners/_longitudinal_develop_loop_gpu.py` /
  `_longitudinal_develop_loop.py` — the WAKE→SLEEP→GROW→PERSIST day cycle.
- `research/findings/AUTONOMOUS_STATE.md` — the current top-of-stack ordering.
