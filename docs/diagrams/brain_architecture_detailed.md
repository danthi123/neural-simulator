# Brain architecture — detailed diagrams (current)

Exhaustive, **as-implemented** diagrams of the whole simulated brain: every region, every distinct pathway, grouped by function, in plain language backed by the biology it reproduces. Each diagram keeps its boxes short (so they render cleanly) and is followed by a **detail table** giving each box's function, the brain structure it reproduces, and its status. These render directly on GitHub (Mermaid) and are the maintainable, **current** replacement for the earlier hand-drawn SVGs (a 2026-06 snapshot). See also the plain-language overview [`brain_architecture_current.md`](brain_architecture_current.md) and the full development [`ROADMAP.md`](../../ROADMAP.md).

**Last synced:** 2026-07-11.

### Status markers (used in every box and table)

| Marker | Meaning |
|---|---|
| *(none)* | a spiking neural circuit — the default |
| ✅ | **learned from experience** (the structure was discovered, not hand-designed) |
| ⚙ | a **fixed, hand-designed** mechanism — biologically defensible, but not itself learned (a stand-in to replace) |
| 🧩 | a **temporary external model** (the one crutch, to be replaced by circuitry) |
| 🌍 | the legitimate **world / body interface** (allowed non-neural code: the environment and the muscles) |
| ◐ | present in a **reduced / partial** form |

Arrow styles: **solid** = an excitatory (driving) signal; **dotted** = inhibitory or a gating/modulatory signal; **thick** = the main signal path. *Working memory is one shared faculty that appears in several diagrams — holding the goal (§2), the reading network's lingering activity (§3), and the discourse register (§4).*

---

## 1. Master map — the whole brain, one engine, two configurations

```mermaid
flowchart TB
    World(["🌍 Simulated world"]):::io
    Turn(["💬 A sentence"]):::io

    subgraph ENGINE["🧠 One brain — spiking neurons on a single update loop"]
      direction TB
      VIS["Vision — visual cortex"]:::sense
      subgraph SM["Navigating brain"]
        NAV["Action selection + navigation"]:::nav
      end
      subgraph LANG["Conversing brain"]
        direction LR
        COMP["Understanding"]:::conv
        CONCEPT["Concepts + meaning ✅"]:::conv
        SPEAK["Speaking ✅"]:::gen
        DISC["Conversation"]:::plan
      end
      subgraph SHARED["Shared core (both brains)"]
        direction LR
        MEM["Memory — hippocampus"]:::mem
        REW["Reward + drive — dopamine"]:::reward
        LRN["Learning rules"]:::learn
      end
    end

    Body(["🌍 Body"]):::io
    Reply(["🗣️ Spoken reply"]):::io

    World ==>|pixels| VIS ==> NAV ==>|movement| Body
    Turn ==>|words| COMP ==> CONCEPT ==> DISC ==> SPEAK ==> Reply
    World -->|spoken command steers movement| NAV
    VIS -.->|seen while moving| MEM
    CONCEPT --> MEM -.-> DISC
    REW -.->|modulates learning + confidence| SM
    REW -.->|modulates learning + confidence| LANG
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

| Box | What it is | Status |
|---|---|---|
| Vision — visual cortex | Retina → primary visual cortex; orientation-selective edge detectors; "what" and "where" streams | ⚙/◐ |
| Action selection + navigation | The navigating brain: basal-ganglia go/no-go loops, superior-colliculus orienting, place cells, goal working-memory (detailed in §2) | 🟩 |
| Understanding | Reads a sentence into who-did-what — parser + reading network (detailed in §3) | ✅ |
| Concepts + meaning | Categories and word meaning learned from experience; reasoning beyond what it was told | ✅ |
| Speaking | Grammar self-organized from experience; every word produced on spikes (detailed in §4) | ✅ |
| Conversation | Tracks who/what across turns; the "I don't know" guard | ✅ |
| Memory — hippocampus | Separate + complete patterns, memory tags, replay in "sleep" (detailed in §5) | 🟩 |
| Reward + drive — dopamine | One shared limbic core driving both brains | 🟩 |
| Learning rules | Spike-timing, Hebbian, three-factor, dendritic (detailed in §5) | 🟩 |

---

## 2. The navigating brain — reach goals by moving, using only what it sees

The project's oldest and most-tested behaviour: an agent explores a world and reaches (or re-reaches, when it moves) a goal, choosing each step by a **spiking evidence-race** through the basal ganglia — the earlier hand-coded pick-one-winner shortcut is retired. Per-direction pools (north/east/south/west) are shown once with a ×4 badge.

```mermaid
flowchart TB
    World(["🌍 World"]):::io
    Spoken(["💬 Spoken command 'go north'"]):::io

    subgraph PERC["Seeing"]
      direction TB
      RET["Retina 🌍"]:::sense
      V1["Primary visual cortex ⚙"]:::sense
      HI["Higher visual areas ◐"]:::sense
      PLACE["Place + goal signals ⚙"]:::sense
      RET ==> V1 ==> HI --> PLACE
    end

    subgraph ORIENT["Where to look — superior colliculus"]
      SCM["'Look here' map"]:::orient
      SCFS["Contrast sharpening"]:::orient
      SCM --> SCFS
      SCFS -.->|inhibits| SCM
    end

    PFC["Goal working-memory"]:::plan

    subgraph BG["Choosing the move — basal ganglia (×4 directions)"]
      direction TB
      CTX["Motor cortex ×4"]:::nav
      D1["'Go' pathway ×4 · learns"]:::str
      D2["'No-go' pathway ×4"]:::str
      FSI["Feedforward competition ×4"]:::stri
      GPE["Pallidum GPe ×4"]:::pall
      STN["Subthalamic nucleus"]:::pall
      GPI["Output gate GPi/SNr ×4"]:::pall
      CTX ==> D1
      CTX ==> D2
      CTX --> FSI
      FSI -.->|winner-take-all| D1
      D2 -.->|inhibits| GPE -.->|inhibits| STN
      CTX -->|hyperdirect| STN
      STN ==>|raises the brake| GPI
      D1 -.->|removes the brake| GPI
    end

    subgraph DECIDE["Committing — the spiking decision"]
      direction TB
      THAL["Thalamus ×4"]:::thal
      ACC["Evidence race"]:::decide
      COMMIT["Commitment burst"]:::decide
      THAL ==> ACC ==> COMMIT
    end

    subgraph REWARD["Learning the route — dopamine"]
      SNC["Dopamine neurons"]:::reward
      VCRIT["Value critic ◐"]:::reward
      VCRIT -.->|subtracts expected value| SNC
      SNC -.->|three-factor learning| D1
    end

    HIPPO["Place-cell map + replay"]:::mem
    Body(["🌍 Body"]):::io

    World ==> RET
    PLACE ==> CTX
    PLACE --> PFC --> CTX
    SCM ==>|biases the winner| CTX
    World -->|goal on the eye| SCM
    HIPPO --> PLACE
    GPI -.->|gate opens| THAL
    COMMIT ==> Body
    Spoken -.->|routes to a direction| CTX

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

| Box | What it is (brain structure · function) | Status |
|---|---|---|
| Retina | On/off contrast image of the scene — the world/body interface | 🌍 |
| Primary visual cortex | Orientation-selective edge detectors (Hubel-Wiesel simple cells, modelled as fixed Gabor filters) | ⚙ |
| Higher visual areas | Object "what" stream toward temporal cortex | ◐ |
| Place + goal signals | Where-I-am / where-the-goal-is (hippocampal place cells; still partly hand-supplied) | ⚙ |
| "Look here" map | Superior-colliculus retinotopic sheet — a single bump at the goal's location | ● |
| Contrast sharpening | Mexican-hat surround — excited by the map, inhibits it back to sharpen the bump | ● |
| Goal working-memory | Prefrontal cortex — keeps the current goal active across steps (persistent activity) | ● |
| Motor cortex ×4 | One channel per direction (regular-spiking pyramidal) | ● |
| "Go" pathway ×4 | Direct striatal cells (D1) — release the brake on this action; **the learning site** | ● learns |
| "No-go" pathway ×4 | Indirect striatal cells (D2) — suppress competing actions | ● |
| Feedforward competition ×4 | Fast-spiking interneurons — sharpen the winner | ● |
| Pallidum GPe ×4 | Relay of the no-go pathway (inhibits the subthalamic nucleus) | ● |
| Subthalamic nucleus | Glutamatergic — diffuse "hold everything" excitation onto the output gate (the shared brake) | ● |
| Output gate GPi/SNr ×4 | Tonically inhibits the thalamus; releasing it selects the action | ● |
| Thalamus ×4 | Disinhibited when its action's gate opens | ● |
| Evidence race → Commitment burst | Each action accumulates evidence (Wang 2002); the first past threshold fires an all-or-none commit (Lo-Wang) — replaces the earlier hand-designed pick-the-winner step | ● (default) |
| Dopamine neurons | Fire the reward-prediction-error (actual − expected); this same core also drives conversation | ● |
| Value critic | The "expected" part — subtracts expected value to form the error (partly code, being built as a circuit) | ◐ |
| Place-cell map + replay | Hippocampus (entorhinal → dentate → CA3 → CA1); consolidates during "sleep" | 🟩 |

---

## 3. The conversing brain — understanding and memory

How a sentence becomes a stored, recallable fact. The brain holds and verifies the *content*; the "I don't know" guard makes fabrication impossible by construction.

```mermaid
flowchart TB
    Words(["💬 A sentence"]):::io

    STREAM["Word-meaning cortex ✅"]:::conv

    subgraph UNDERSTAND["Understanding — who did what to whom"]
      direction TB
      PARSE["Sentence parser ✅"]:::conv
      RES["Reading network ✅"]:::conv
      ROLES{{"Actor · action · object"}}:::conv
      PARSE ==> RES ==> ROLES
    end

    subgraph FACT["Fact memory"]
      direction TB
      BIND["Tie the roles into a fact ⚙"]:::conv
      STORE["Fact store"]:::mem
      BIND ==> STORE
    end

    INHERIT["Inference ✅"]:::conv

    subgraph GUARD["The 'I don't know' guard"]
      direction TB
      MATCH{{"Stored fact matches<br/>the question?"}}:::gate
      RECALL["Recall + unbind"]:::conv
      IDK(["🚫 'I don't know'"]):::abstain
      MATCH ==>|match| RECALL
      MATCH ==>|no match| IDK
    end

    HP["Hippocampus"]:::mem
    Speak(["→ speaking brain (§4)"]):::io

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

| Box | What it is (brain structure · function) | Status |
|---|---|---|
| Word-meaning cortex | Anterior-temporal concept hub; meaning learned from a text stream so similar words end up close together (~320 concepts; one normalization step still done in code) | ✅ |
| Sentence parser | Language-comprehension cortex (Wernicke's area) — maps word position to role, regardless of active/passive voice | ✅ |
| Reading network | A fixed recurrent network (Hinaut-Dominey) whose lingering activity carries structure; a **trained read-out** pulls out the roles, resolving long-distance dependencies | ✅ (read-out) |
| Tie the roles into a fact | Timing/phase-signalling neurons lock actor + action + object into one reusable pattern; a fixed mathematical rule stands in for a binding step the cortex would normally learn | ⚙ |
| Fact store | Each fact written as a pattern into synapses; holds many, persistently | ● |
| Inference | Inheritance, exceptions, transitivity emerge from overlapping concept codes — no separate "inference engine" (a robin inherits that birds fly) | ✅ |
| The "I don't know" guard | Checks for a matching stored fact first; recalls + unbinds it if found, else abstains — nothing is invented | ● |
| Hippocampus | Episodic store: dentate separates, CA3 completes from a fragment (Marr), replay consolidates in "sleep" with no forgetting | 🟩 |

---

## 4. The conversing brain — speaking and conversation

Turning a verified fact into a spoken reply, tracking who is being talked about across turns, and the two phrasing paths (both behind the guard). Every word — content *and* function words — is produced as spiking activity.

```mermaid
flowchart TB
    Content(["✔️ Verified content (§3)"]):::io
    UserTurn(["💬 The conversation"]):::io

    subgraph DISCG["Tracking the conversation"]
      direction TB
      REG["Who-is-acting register ✅"]:::plan
      ANAPH["Pronoun resolution"]:::plan
      WM["Nesting / bounded stack ◐"]:::plan
      REG --> ANAPH
    end

    DLPFC["Dialogue planning"]:::plan

    subgraph GRAMMAR["Grammar — self-organized ✅"]
      direction TB
      DISCOVER["Discovered grammar ✅"]:::gen
      ORDER["Word-order generator ✅"]:::gen
      DISCOVER ==> ORDER
    end

    subgraph ARTIC["Saying it on spikes"]
      direction TB
      AW["Sound-out read-out ✅"]:::gen
      SPEECHM["Speech motor 🌍"]:::io
      AW ==> SPEECHM
    end

    subgraph OPEN["Open prose — two phrasing paths"]
      direction TB
      EMERG["Home-grown generator ✅"]:::gen
      CRUTCH["Small conventional model 🧩"]:::crutch
    end

    Reply(["🗣️ Grounded, re-checked reply"]):::io

    Content ==> DISCOVER
    UserTurn ==> REG
    ANAPH -.-> DLPFC -.-> DISCOVER
    ORDER ==> AW
    SPEECHM ==>|re-read + verified vs the fact| Reply
    Content -.->|open prose only, behind the guard| EMERG
    Content -.->|open prose only, behind the guard| CRUTCH
    EMERG -.-> Reply
    CRUTCH -.-> Reply

    classDef io fill:#eef1f4,stroke:#7a8794,color:#1d1d1f;
    classDef plan fill:#e9dcf5,stroke:#7d3c98,color:#3b1d4e;
    classDef gen fill:#d1f2eb,stroke:#138d75,color:#0c3d33;
    classDef crutch fill:#fdebd0,stroke:#c8791a,color:#5b3a10;
```

| Box | What it is (brain structure · function) | Status |
|---|---|---|
| Who-is-acting register | Prefrontal working-memory slots (Grosz-Sidner focus stack) — a topic shift pushes the current actor aside, a return pops it back | ✅ |
| Pronoun resolution | "It / they" → the referent held in working memory across turns | ● |
| Nesting / bounded stack | A slotted buffer (Lisman-Idiart theta-gamma) matches nested structure up to depth ~3 — the human-faithful limit | ◐ |
| Dialogue planning | Prefrontal — picks the next on-topic thing to say (self-sustaining working-memory loop) | ● |
| Discovered grammar | Which words are function words, the word order, the sentence templates — all mined from example sentences (Broca's grammatical role) | ✅ |
| Word-order generator | A spiking "say-this-first" ranking (competitive queuing) sets the order — no fixed template | ✅ |
| Sound-out read-out | Every word (content and function alike) decoded from the neurons' spike output (Broca articulation) | ✅ |
| Home-grown generator | For open prose: a fixed recurrent network + a small next-word read-out trained locally (not by the usual global method) — beats simple predictors, so far over a small vocabulary (first rung) | ✅ |
| Small conventional model | A temporary crutch for fluent open prose — the one remaining external model, being replaced by the home-grown path | 🧩 |

---

## 5. Shared systems — memory, reward, learning, development

```mermaid
flowchart LR
    subgraph MEMORY["Memory — hippocampus"]
      direction TB
      EC["Entorhinal cortex"]:::mem
      DG["Dentate gyrus"]:::mem
      CA3["CA3 autoassociator"]:::mem
      CA1["CA1 read-out"]:::mem
      REPLAY["Sleep replay"]:::sleep
      TAG["Memory tags"]:::mem
      EC ==> DG ==> CA3 ==> CA1
      CA3 <-.-> REPLAY
      CA3 -.-> TAG
    end

    subgraph MOTIV["Reward + drive"]
      direction TB
      DA["Dopamine — reward error"]:::reward
      DRIVE["Hunger / drive"]:::reward
      NE["Arousal / surprise"]:::reward
      DRIVE -.->|raises dopamine| DA
    end

    subgraph LEARNRULES["Learning rules"]
      direction TB
      STDP["Spike-timing"]:::learn
      HEBB["Hebbian ✅"]:::learn
      THREE["Three-factor"]:::learn
      DEND["Dendritic credit ◐"]:::learn
    end

    subgraph LIFE["Living over days"]
      direction LR
      WAKE["Wake + learn"]:::wake
      CONV["Converse"]:::conv
      SLEEP2["Sleep"]:::sleep
      GROW["Grow"]:::grow
      SAVE["Save + resume"]:::persist
      WAKE ==> CONV ==> SLEEP2 ==> GROW ==> SAVE ==> WAKE
    end

    DA -.->|gates| THREE
    NE -.->|speeds| STDP
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

| Box | What it is | Status |
|---|---|---|
| Hippocampal loop (entorhinal → dentate → CA3 → CA1) | Dentate separates memories, CA3 completes one from a fragment (Marr), CA1 reads out | 🟩 |
| Sleep replay | Replays the day's experience during "sleep" so it sticks — learning sequences over time without overwriting older memories | 🟩 |
| Memory tags | Tag a trace, reactivate it later (engram cells) | 🟩 |
| Dopamine — reward error | Actual − expected reward (Schultz) — gates three-factor learning | ● |
| Hunger / drive | A hungry brain raises dopamine, making it more careful about what it claims to know | ● |
| Arousal / surprise (noradrenaline) | An unexpected outcome speeds up learning | ● |
| Learning rules | Spike-timing plasticity; Hebbian co-occurrence (✅ how the word-cortex learns); three-factor (dopamine-gated); **dendritic credit assignment** (◐ the open lever for the *deep-composition* ceilings — a two-compartment burst rule, Payeur-Naud, idealized now, being brought fully onto spikes); **local input-representation learning on a fixed reservoir** (the *long-range-language* lever — a fixed random recurrent scaffold that learns only its input, beating full backprop; ~78% biology-legal, going onto spikes now with no engine edit — see [`ROADMAP.md`](../../ROADMAP.md) §9.1) | mixed |
| Living over days | Wake + learn → converse → sleep + consolidate → grow → save + resume (not a blank slate next day) | 🟩 |

---

## What the diagrams deliberately simplify

- **Per-direction and per-concept pools are collapsed** (a "×4" or "×N" badge).
- **Some regions are shared** and drawn once (the hippocampus, the dopamine core).
- The brain is **assembled per configuration** from a declarative region/pathway grammar; a given run builds a subset of what's shown.
- **Noradrenaline / serotonin / acetylcholine** are framework-supported but only partly used (◐); dopamine is fully deployed.
- **The human interfaces** — the real-time 3-D viewer, the talkable chat console, the develop-over-days launcher, the experiment system, and the biological validation suite — are the windows into the brain, not brain computations, and are described in [`ROADMAP.md`](../../ROADMAP.md) §7 rather than drawn here. The **cerebellum** is intentionally absent (cell presets only, no circuit — an open item, ROADMAP §6).
- Anything marked ⚙ (a fixed stand-in), 🧩 (the external crutch), or ◐ (partial) is a tracked item — see [`ROADMAP.md`](../../ROADMAP.md) §8–9 for each one's replacement plan.
