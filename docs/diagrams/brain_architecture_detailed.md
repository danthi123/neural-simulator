# Brain architecture — detailed diagrams (current)

Exhaustive, **as-implemented** diagrams of the whole simulated brain: every region, every distinct pathway, grouped by function, in plain language backed by the biology it reproduces. These render directly on GitHub (Mermaid) and are the maintainable, **current** replacement for the earlier hand-drawn SVGs (a 2026-06 snapshot). For the plain-language overview see [`brain_architecture_current.md`](brain_architecture_current.md); for the full stage-by-stage development path and status see [`ROADMAP.md`](../../ROADMAP.md).

**Last synced:** 2026-07-10.

### How to read every diagram

Each box is a group of neurons doing one job, labelled with its **plain function** and, in smaller text, the **brain structure it reproduces**. A small marker in the label shows what kind of thing it is:

| Marker | Meaning |
|---|---|
| *(no marker)* | a spiking neural circuit — the default |
| ✅ | **learned from experience** (the structure was discovered, not hand-designed) |
| ⚙ | a **fixed, hand-designed** mechanism — biologically defensible, but not itself learned (a stand-in to replace) |
| 🧩 | a **temporary external model** (the one crutch, to be replaced by circuitry) |
| 🌍 | the legitimate **world / body interface** (allowed non-neural code: the environment and the muscles) |
| ◐ | present in a **reduced / partial** form |

Arrow styles: a **solid** arrow is an excitatory (driving) signal; a **dotted** arrow is inhibitory or a gating/modulatory signal; a **thick** arrow is the main signal path.

---

## 1. Master map — the whole brain, one engine, two configurations

The simulator is *one* network of spiking neurons on a single update loop. From that one substrate, two configurations are assembled — a **navigating** brain and a **conversing** brain — that share the same engine, the same memory, and one dopamine/limbic core, and are joined by validated synaptic links.

```mermaid
flowchart TB
    World(["🌍 Simulated world — renders what the agent sees, enacts its movements"]):::io

    subgraph ENGINE["🧠 One brain — spiking neurons + synapses on a single update loop"]
      direction TB

      subgraph SENSE["Sensing"]
        direction LR
        VIS["Vision — retina to primary visual cortex<br/><small>oriented-edge (Gabor) detectors · what and where streams</small>"]:::sense
      end

      subgraph SM["The navigating brain — reach goals by moving"]
        direction LR
        NAV["Action selection and navigation<br/><small>basal-ganglia go/no-go loops · superior-colliculus orienting · place cells · goal working-memory</small>"]:::nav
      end

      subgraph LANG["The conversing brain — understand, think, speak"]
        direction LR
        COMP["Understanding<br/><small>parser + reservoir: word order to who-did-what</small>"]:::conv
        CONCEPT["Concepts and meaning ✅<br/><small>categories learned from experience · reasoning</small>"]:::conv
        SPEAK["Speaking ✅<br/><small>self-organized grammar · every word on spikes</small>"]:::gen
        DISC["Conversation<br/><small>tracks who/what across turns · the 'I don't know' guard</small>"]:::plan
      end

      subgraph SHARED["Shared core (used by both brains)"]
        direction LR
        MEM["Memory — hippocampus<br/><small>separate and complete patterns · tag · replay in 'sleep'</small>"]:::mem
        REW["Reward and drive — dopamine<br/><small>one shared limbic core for both brains</small>"]:::reward
        LRN["Learning rules<br/><small>spike-timing · Hebbian · three-factor · dendritic</small>"]:::learn
      end
    end

    Body(["🌍 Body — carries out the chosen movement"]):::io
    Reply(["🗣️ Spoken reply — grounded, checked, or 'I don't know'"]):::io

    World ==>|pixels| VIS
    VIS ==> NAV
    NAV ==>|movement| Body
    World -->|a spoken command steers movement| NAV

    VIS -.->|what it saw while moving| MEM
    COMP ==> CONCEPT ==> DISC
    CONCEPT --> MEM
    DISC ==> SPEAK ==> Reply
    MEM -.-> DISC
    REW -.->|modulates learning and confidence| SM
    REW -.->|modulates learning and confidence| LANG
    LRN -.-> SHARED

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef sense fill:#d6eaf8,stroke:#2e6da4,color:#10303f;
    classDef nav fill:#fdebd0,stroke:#c87f2e,color:#5b3a10;
    classDef conv fill:#d6eaf8,stroke:#2471a3,color:#10303f;
    classDef gen fill:#d1f2eb,stroke:#138d75,color:#0c3d33;
    classDef plan fill:#e9dcf5,stroke:#7d3c98,color:#3b1d4e;
    classDef mem fill:#d4efdf,stroke:#1d8049,color:#0f3d23;
    classDef reward fill:#fcf3cf,stroke:#b9770e,color:#5b3a10;
    classDef learn fill:#eae3f3,stroke:#5b4a8a,color:#2c2247;
```

---

## 2. The navigating brain — reach goals by moving, using only what it sees

The project's oldest and most-tested behaviour: an agent explores a world and reaches (or re-reaches, when it moves) a goal, choosing each step by a **spiking evidence-race** through the basal ganglia — the hand-coded winner-take-all shortcut is retired. Per-direction pools (north/east/south/west) are drawn once with a ×4 badge.

```mermaid
flowchart TB
    World(["🌍 World — renders the scene the eyes receive"]):::io

    subgraph PERC["Seeing — visual cortex (what and where)"]
      direction TB
      RET["Retina 🌍<br/><small>on/off contrast image</small>"]:::sense
      V1["Primary visual cortex ✅<br/><small>oriented-edge (Gabor) detectors — Hubel-Wiesel</small>"]:::sense
      HI["Higher visual areas ◐<br/><small>object 'what' stream toward temporal cortex</small>"]:::sense
      PLACE["Place and goal signals ⚙<br/><small>where-I-am / where-the-goal-is (hippocampal place cells · some still supplied as code)</small>"]:::sense
      RET ==> V1 ==> HI --> PLACE
    end

    subgraph ORIENT["Where to look — superior colliculus (orienting reflex)"]
      direction TB
      SCM["'Look here' map<br/><small>retinotopic sheet · a single bump at the goal's location</small>"]:::orient
      SCFS["Contrast sharpening<br/><small>Mexican-hat surround — inhibits neighbours to sharpen the bump</small>"]:::orient
      SCM -.-> SCFS
    end

    subgraph GOALWM["Holding the goal — prefrontal working memory"]
      PFC["Goal working-memory<br/><small>keeps the current goal active across steps (persistent activity)</small>"]:::plan
    end

    subgraph BG["Choosing the move — basal-ganglia action-selection loop (×4 directions)"]
      direction TB
      CTX["Motor cortex ×4<br/><small>one channel per direction (regular-spiking pyramidal)</small>"]:::nav
      D1["'Go' pathway ×4 · learns<br/><small>direct striatal cells (D1) — release the brake on this action</small>"]:::str
      D2["'No-go' pathway ×4<br/><small>indirect striatal cells (D2) — suppress competing actions</small>"]:::str
      FSI["Feedforward competition ×4<br/><small>fast-spiking interneurons — sharpen the winner</small>"]:::stri
      GPE["Pallidum GPe ×4<br/><small>pacemaker relay of the no-go pathway</small>"]:::pall
      STN["Subthalamic nucleus<br/><small>diffuse 'hold everything' excitation (shared)</small>"]:::pall
      GPI["Output gate GPi/SNr ×4<br/><small>tonically inhibits the thalamus · releasing it selects the action</small>"]:::pall
      CTX ==> D1
      CTX ==> D2
      CTX -.-> FSI -.-> D1
      D2 -.-> GPE -.-> GPI
      CTX --> STN -.-> GPI
      D1 -.->|removes the brake| GPI
    end

    subgraph DECIDE["Committing — the spiking decision (default)"]
      direction TB
      THAL["Thalamus ×4<br/><small>disinhibited when its action's gate opens</small>"]:::thal
      ACC["Evidence race<br/><small>each action accumulates evidence (Wang 2002 attractor)</small>"]:::decide
      COMMIT["Commitment burst<br/><small>the first to cross threshold fires an all-or-none commit (Lo-Wang) — retires the old code shortcut</small>"]:::decide
      THAL ==> ACC ==> COMMIT
    end

    subgraph REWARD["Learning the route — dopamine reward signal"]
      direction TB
      SNC["Dopamine neurons<br/><small>fire the reward-prediction-error (actual minus expected) · this same core also drives conversation</small>"]:::reward
      VCRIT["Value critic ◐<br/><small>how good is this situation (the expected part) — partly code, being built as a circuit</small>"]:::reward
      SNC -.->|three-factor learning| D1
      VCRIT --> SNC
    end

    subgraph MEMNAV["Spatial memory — hippocampus"]
      HIPPO["Place-cell map and replay<br/><small>entorhinal to dentate to CA3 to CA1 · consolidates during 'sleep'</small>"]:::mem
    end

    Body(["🌍 Body — moves in the chosen direction"]):::io
    Spoken(["💬 A spoken command 'go north' — from the conversing brain"]):::io

    World ==> RET
    PLACE ==> CTX
    PLACE --> PFC --> CTX
    SCM ==>|biases the winning direction| CTX
    World -->|the goal's location on the eye| SCM
    HIPPO --> PLACE
    COMMIT ==> Body
    Spoken -.->|routes the command to a direction| CTX
    COMMIT -.-> SNC

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef sense fill:#d6eaf8,stroke:#2e6da4,color:#10303f;
    classDef orient fill:#f4ecf7,stroke:#6c3483,color:#3b1d4e;
    classDef plan fill:#e9dcf5,stroke:#7d3c98,color:#3b1d4e;
    classDef nav fill:#e8e1f2,stroke:#5b4a8a,color:#2c2247;
    classDef str fill:#f5cba7,stroke:#7e3300,color:#4a2100;
    classDef stri fill:#fad7a0,stroke:#6e2c00,color:#4a2100;
    classDef pall fill:#f6ddcc,stroke:#7e3300,color:#4a2100;
    classDef thal fill:#d5f5e3,stroke:#1e8449,color:#0c3d22;
    classDef decide fill:#d1f2eb,stroke:#138d75,color:#0c3d33;
    classDef reward fill:#fcf3cf,stroke:#b9770e,color:#5b3a10;
    classDef mem fill:#d4efdf,stroke:#1d8049,color:#0f3d23;
```

---

## 3. The conversing brain — understanding and memory

How a sentence becomes a stored, recallable fact. The brain holds and verifies the *content*; the "I don't know" guard makes fabrication impossible by construction.

```mermaid
flowchart TB
    Words(["💬 A sentence — flexible word order"]):::io

    subgraph LEARNCX["Word meaning — learned by listening ✅"]
      STREAM["Word-meaning cortex ✅<br/><small>anterior-temporal concept hub · meaning learned from a text stream — similar words end up close together (~320 concepts)</small>"]:::conv
    end

    subgraph UNDERSTAND["Understanding — who did what to whom"]
      direction TB
      PARSE["Sentence parser ✅<br/><small>Wernicke-style · maps word position to role, regardless of active/passive voice</small>"]:::conv
      RES["Reading reservoir ✅<br/><small>fronto-striatal recurrent network (Hinaut-Dominey) · its lingering activity carries structure, resolving long-distance dependencies</small>"]:::conv
      ROLES{{"Actor · action · thing-acted-on"}}:::conv
      PARSE ==> RES ==> ROLES
    end

    subgraph FACT["Fact memory"]
      direction TB
      BIND["Bind into a fact ⚙<br/><small>phase-based resonate-and-fire neurons + complex synapses · the binding is a fixed algebra standing in for a learned cortical binder</small>"]:::conv
      STORE["Fact store<br/><small>each fact written as a pattern into synapses · holds many, persistently</small>"]:::mem
      BIND ==> STORE
    end

    subgraph REASON["Reasoning beyond what it was told ✅"]
      INHERIT["Inference ✅<br/><small>inheritance, exceptions, transitivity emerge from overlapping concept codes — no separate 'inference engine' (a robin inherits that birds fly)</small>"]:::conv
    end

    subgraph GUARD["The 'I don't know' guard"]
      direction TB
      MATCH{{"Is there a stored fact<br/>matching the question?"}}:::gate
      RECALL["Recall and unbind<br/><small>reconstruct the fact, separate the roles, clean up, read the answer</small>"]:::conv
      IDK(["🚫 'I don't know' — nothing invented"]):::abstain
      MATCH ==>|match| RECALL
      MATCH ==>|no match| IDK
    end

    subgraph HIPPO["Episodic memory — hippocampus (shared)"]
      HP["Separate · complete · replay<br/><small>dentate separates, CA3 completes from a fragment (Marr), replay consolidates in 'sleep' with no forgetting</small>"]:::mem
    end

    Speak(["to the speaking brain, section 4"]):::io

    Words ==> PARSE
    STREAM -.->|concept codes| PARSE
    STREAM -.->|concept codes| BIND
    STREAM --> INHERIT
    ROLES ==> BIND
    STORE --> MATCH
    INHERIT -.-> MATCH
    STORE <-.->|consolidate / recall| HP
    RECALL ==>|answer content| Speak

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef conv fill:#d6eaf8,stroke:#2471a3,color:#10303f;
    classDef mem fill:#d4efdf,stroke:#1d8049,color:#0f3d23;
    classDef gate fill:#fdebd0,stroke:#c8791a,color:#5b3a10;
    classDef abstain fill:#f8d7da,stroke:#b03a2e,color:#5b1512;
```

---

## 4. The conversing brain — speaking and conversation

Turning a verified fact into a spoken reply, tracking who is being talked about across turns, and the two phrasing paths (both behind the guard). Every word — content *and* function words — is produced as spiking activity.

```mermaid
flowchart TB
    Content(["✔️ Verified answer content — from the fact memory, section 3"]):::io
    UserTurn(["💬 The ongoing conversation"]):::io

    subgraph DISC["Tracking the conversation — discourse working memory"]
      direction TB
      REG["Who-is-acting register ✅<br/><small>prefrontal working-memory slots (Grosz-Sidner focus stack) · a topic-shift pushes the current actor aside, a return pops it back</small>"]:::plan
      ANAPH["Pronoun resolution<br/><small>'it / they' to the referent held in working memory across turns</small>"]:::plan
      REG --> ANAPH
    end

    subgraph PLAN["Deciding what to say"]
      DLPFC["Dialogue planning — prefrontal<br/><small>picks the next on-topic thing to say (self-sustaining working-memory loop)</small>"]:::plan
    end

    subgraph GRAMMAR["Grammar — self-organized from experience ✅"]
      direction TB
      DISCOVER["Discovered grammar ✅<br/><small>which words are function words, the word order, the sentence templates — all mined from example sentences (Broca's grammatical role)</small>"]:::gen
      ORDER["Word-order generator ✅<br/><small>a spiking 'say-this-first' ranking (competitive queuing) sets the order — no fixed template</small>"]:::gen
      DISCOVER ==> ORDER
    end

    subgraph ARTIC["Saying it on spikes — articulation"]
      direction TB
      AW["Sound-out read-out ✅<br/><small>every word — content and function alike — decoded from the neurons' spike output (Broca articulation)</small>"]:::gen
      SPEECHM["Speech motor 🌍<br/><small>the produced word sequence</small>"]:::io
      AW ==> SPEECHM
    end

    subgraph OPEN["Open-ended prose — the two phrasing paths"]
      direction TB
      EMERG["Home-grown generator ✅ (first rung)<br/><small>a fixed reservoir + a locally-trained next-word read-out (no backpropagation) — beats the standard baselines, so far over a small vocabulary</small>"]:::gen
      CRUTCH["Small conventional model 🧩<br/><small>a temporary crutch for fluent open prose — the one remaining external model, being replaced by the home-grown path</small>"]:::crutch
    end

    Reply(["🗣️ Grounded, re-checked spoken reply"]):::io

    Content ==> DISCOVER
    UserTurn ==> REG
    ANAPH -.-> DLPFC
    DLPFC -.-> DISCOVER
    ORDER ==> AW
    SPEECHM ==>|re-read and verified vs the stored fact| Reply
    Content -.->|for open prose only, behind the guard| EMERG
    Content -.->|for open prose only, behind the guard| CRUTCH
    EMERG -.-> Reply
    CRUTCH -.-> Reply

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef plan fill:#e9dcf5,stroke:#7d3c98,color:#3b1d4e;
    classDef gen fill:#d1f2eb,stroke:#138d75,color:#0c3d33;
    classDef crutch fill:#fdebd0,stroke:#c8791a,color:#5b3a10;
```

---

## 5. Shared systems — memory, reward, learning, development

Systems both brains use, and how the whole brain lives over time.

```mermaid
flowchart LR
    subgraph MEMORY["Memory — hippocampus and consolidation"]
      direction TB
      EC["Entorhinal cortex<br/><small>gateway in and out</small>"]:::mem
      DG["Dentate gyrus<br/><small>keeps memories separate</small>"]:::mem
      CA3["CA3 autoassociator<br/><small>completes a memory from a fragment (Marr)</small>"]:::mem
      CA1["CA1 read-out"]:::mem
      REPLAY["Sleep replay<br/><small>replays the day's learning so it sticks — and does the job backpropagation-through-time does (no forgetting)</small>"]:::sleep
      TAG["Memory tags<br/><small>tag a trace, reactivate it later (engram cells)</small>"]:::mem
      EC ==> DG ==> CA3 ==> CA1
      CA3 <-.-> REPLAY
      CA3 -.-> TAG
    end

    subgraph MOTIV["Reward, value and drive — one shared limbic core"]
      direction TB
      DA["Dopamine — reward-prediction-error<br/><small>actual minus expected reward (Schultz)</small>"]:::reward
      DRIVE["Hunger / drive<br/><small>a hungry brain gets more careful about what it claims to know</small>"]:::reward
      DA -.-> DRIVE
    end

    subgraph LEARNRULES["Learning rules (on every plastic synapse)"]
      direction TB
      STDP["Spike-timing plasticity"]:::learn
      HEBB["Hebbian co-occurrence ✅<br/><small>the rule the word-cortex learns by</small>"]:::learn
      THREE["Three-factor (dopamine-gated)"]:::learn
      DEND["Dendritic deep learning ◐<br/><small>the top open lever — a two-compartment burst rule (Payeur-Naud) · works idealized, being brought fully onto spikes</small>"]:::learn
    end

    subgraph LIFE["Living over simulated days"]
      direction LR
      WAKE["Wake and learn<br/><small>hears the day's words</small>"]:::wake
      CONV["Converse"]:::conv
      SLEEP2["Sleep and consolidate"]:::sleep
      GROW["Grow if mastered"]:::grow
      SAVE["Save and resume<br/><small>not a blank slate next day</small>"]:::persist
      WAKE ==> CONV ==> SLEEP2 ==> GROW ==> SAVE ==> WAKE
    end

    DA -.->|gates| THREE
    HEBB -.-> MEMORY
    REPLAY -.-> SLEEP2

    classDef mem fill:#d4efdf,stroke:#1d8049,color:#0f3d23;
    classDef sleep fill:#dcdcf5,stroke:#5b5bd6,color:#2a2a55;
    classDef reward fill:#fcf3cf,stroke:#b9770e,color:#5b3a10;
    classDef learn fill:#eae3f3,stroke:#5b4a8a,color:#2c2247;
    classDef wake fill:#fcf3cf,stroke:#d4ac0d,color:#5b4a0e;
    classDef conv fill:#d6eaf8,stroke:#2471a3,color:#10303f;
    classDef grow fill:#dcefd3,stroke:#2f8f4e,color:#173d25;
    classDef persist fill:#fce8f0,stroke:#c0397b,color:#5b1a3a;
```

---

## What the diagrams deliberately simplify

- **Per-direction and per-concept pools are collapsed** (a "×4" or "×N" badge) — the real network has one pool per direction (north/east/south/west) and per concept.
- **A few regions are drawn once but are shared** by both configurations (the hippocampus, the dopamine core).
- The brain is **assembled per configuration** from a declarative region/pathway grammar; a given run builds a subset of what's shown (these diagrams show the fuller picture).
- Anything marked ⚙ (a fixed stand-in), 🧩 (the external crutch), or ◐ (partial) is a tracked item on the way to being replaced or completed — see [`ROADMAP.md`](../../ROADMAP.md) sections 8 and 9 for each one's replacement plan.
