# ROADMAP — a brain you can talk to

> **This is the source of truth** for *what the project has accomplished, what it is working on now, and what is left* on the path to the goal. It is kept current as a standing part of the workflow. If anything here disagrees with a detailed write-up in `research/findings/`, the finding is right and this file gets corrected.
>
> **Last synced:** 2026-07-10.

**A note on names.** This document is written to be read without knowing the codebase. A few conventions:
- **Status badges:** ✅ *emergent* (done, and the structure was **learned from experience** on the spiking brain) · 🟩 *done* (validated, but with one hand-designed part that is biologically defensible) · 🟨 *partial* (works in a reduced form) · 🟧 *boundary* (a limit we've mapped precisely; the next mechanism is named) · 🧩 *scaffold* (a temporary stand-in to be replaced by real circuitry) · ⬜ *open* (not built yet).
- **"On the spiking brain" / "on the substrate"** means the computation is done by simulated neurons firing and synapses, not by ordinary code.
- **"The no-confabulation guard"** (elsewhere in the codebase called "the moat") is the mechanism that makes the brain say *"I don't know"* instead of making something up.
- **"Validated across 6 seeds"** means the result held up over six independent random initializations (three used during development, three held back as a blind check) with control experiments that rule out cheating.
- Internal milestone codes like *EMERGE-30*, *D3*, or *RUNG-1* are just our own progress tags; you can ignore them — they're kept only so a claim can be traced to the exact experiment.

---

## 1. The goal

Build **artificial life** — a real brain-analogue that lives, learns, and grows — whose **conversational ability approaches that of a large language model** (open-ended, fluid, grounded, able to carry context), built the honest way: as **one simulated spiking brain** whose language is genuinely its own, learned from experience, with no permanent external AI model doing the thinking for it.

## 2. The rules we hold ourselves to

These decide what counts as real progress. Anything that breaks them is a temporary scaffold, not a milestone.

1. **The brain does the thinking, not the code.** Ordinary (non-neural) software is allowed only for the **world** (the environment, and drawing the picture the eyes receive) and the **body** (moving when the motor neurons fire). Everything in between — perceiving, deciding, valuing, remembering, reasoning, speaking — must be done by simulated neurons and synapses. A biologically *correct* Python formula (a reward, a winner-take-all, a prediction error) is still a shortcut to be replaced.
2. **One brain.** All the faculties are populations of neurons on a single shared brain that talk to each other through synapses — not separate programs stitched together.
3. **Learned, not hand-built.** Structure should be *discovered from experience*, not designed by hand one feature at a time. Building capabilities by hand, one at a time, is a treadmill, not biology.
4. **Fully spiking in the end.** We sometimes use a temporary stand-in to move fast, but the finished version runs entirely on the spiking brain, and every stand-in is tracked and removed.
5. **No permanent external model.** A conventional AI model may be a temporary crutch for fluency, but the end state simulates the actual brain circuitry.
6. **An honest "no" is a result.** When the brain genuinely can't do something, we measure exactly where it breaks and report it — then go find the missing mechanism. A limit is never a stopping point.

## 3. The one-screen picture

**What is already done and genuinely brain-learned:** perceiving categories from experience, understanding sentences (including who-did-what-to-whom in tricky word orders), reasoning that goes beyond what it was told (if a robin is a bird and birds fly, a robin flies), a grammar it worked out for itself from listening, speaking every word out of neural activity, tracking who's being talked about across a conversation, and knowing when to say "I don't know." **Navigation** — moving through a world to reach goals using only what it sees — is its oldest and most thoroughly-tested behavior.

**The two honest gaps between this and a large language model:**
1. **Open-ended fluent speech.** Right now, truly free-flowing prose is produced by a small conventional AI model (a temporary crutch) while the brain supplies all the meaning and grounding. The home-grown replacement has taken its first real step this cycle, but only over a small controlled vocabulary so far.
2. **A deeper learning rule.** Some ceilings (how much structure it can compose, how deep it can nest ideas) come down to one missing capability: a dendritic learning rule that lets a *deep* network teach itself on spikes. We have this working in an idealized form; getting it to train a real spiking network end-to-end is the live frontier.

**Bottom line:** the foundations — the brain engine, movement, memory, understanding, concepts, and self-organized speech — are done and mostly emergent. The distance to LLM-level conversation is real, bounded, and measured in months of focused work, not a single demo — and nothing is fundamentally blocked.

---

## 4. The brain engine (the platform everything runs on) — 🟩 Done

*This is the "hardware": the neurons, synapses, and wiring framework that every faculty runs on. It's mature and production-grade.*

| Part | What it reproduces (biology) | Where |
|---|---|---|
| The GPU simulation engine | The whole brain as one continuously-updated system of realistic conductance-based neurons; scales to 100,000+ neurons; saves/restores state; fully reproducible | `sim/bridge.py` |
| Neuron types — Izhikevich, Hodgkin-Huxley, adaptive exponential, and a resonate-and-fire "phasor" neuron | ~50 cell-type presets across cortex, hippocampus, basal ganglia, thalamus, cerebellum, spinal cord (catalog cluster I; the phasor neuron follows Frady-Sommer 2019) | `sim/enums.py`, `sim/kernels.py` |
| The region-and-pathway framework | Multiple interacting brain regions declared as populations with wiring between them; runtime "gates" that open/close a pathway (a model of how the thalamus routes cortical traffic, Logiaco-Abbott-Escola 2021) | `sim/regions.py` |
| The learning-rule family | Spike-timing-dependent plasticity (Bi & Poo), short-term facilitation/depression, Hebbian learning, dopamine-gated three-factor learning (Schultz), homeostasis (Turrigiano), and a dendritic burst-dependent rule (catalog cluster J) | `sim/kernels.py` |
| The neuromodulator system | Dopamine, noradrenaline, serotonin, acetylcholine concentration dynamics and their effects; **one shared dopamine system drives both movement and conversation** (catalog cluster C) | `sim/neuromodulators.py` |
| Lifelong-learning persistence | Learning that accumulates across sessions without erasing old knowledge | `sim/lineage.py`, `sim/synapse_storage.py` |
| A two-compartment "dendritic" neuron (early) — 🟨 | Active dendrites and burst-based learning — the substrate for the deeper learning rule (see §9) | `enable_bdsp` in the engine |

---

## 5. The developmental path

*Roughly the order a developing brain builds these. Each stage names the brain regions/functions it reproduces (with citations), a status, what's done, what's open, and the next step. Grouped into the sensorimotor brain (5.1–5.6), the thinking-and-language brain (5.7–5.12), and the living whole (5.13).*

### — The sensorimotor brain —

### 5.1 Perception — *seeing the world* · 🟨 Partial
- **Goal:** turn the picture the eyes receive into neural codes that capture real similarity (two apples look alike; an apple and a river don't), so the rest of the brain has something meaningful to learn from.
- **Biology:** edge/contrast detection (center-surround cells), then oriented-edge detectors in primary visual cortex (Hubel & Wiesel; catalog E; Kandel Ch 22), splitting into a "what" stream (object identity, ventral) and a "where/how" stream (location and movement, dorsal) (Kandel Ch 24–25).
- **Done:** a real oriented-edge (Gabor) visual front end plus a retina renderer, giving genuine visual similarity structure (objects of the same category look alike to it, verified against the raw pixels); population coding lifts a single neuron's noisy read to full accuracy.
- **Open:** only the first cortical visual stage exists (no deeper object-recognition hierarchy, no separate location stream as its own module, no hearing/smell/touch); the edge detectors are fixed rather than learned.
- **Next:** build a deeper, self-organizing visual stage only where a later capability needs it.

### 5.2 Attention & orienting — *where to look* · 🟩 Done (orienting) / 🟨 Partial (attention)
- **Goal:** decide where to point the eyes/attention and commit the movement; emphasize what matters.
- **Biology:** the superior colliculus (a brainstem "look here" map, released to act by removing an inhibitory brake; Kandel Ch 35), and noradrenaline-driven arousal that turns up the gain on surprising events.
- **Done:** a spiking superior-colliculus orienting reflex, on by default (a small, honest ~16% cost over an idealized version); an arousal/surprise signal that speeds learning after unexpected outcomes.
- **Open:** no top-down "pay attention here" controller yet; no dedicated eye-movement burst generator; feature-binding by neural synchrony is available but not deployed.
- **Next:** add a spiking attention controller when scenes with several objects demand it.

### 5.3 Action selection — *choosing what to do* · 🟩 Done
- **Goal:** pick one action from competing options, gather evidence until confident, and commit — entirely in spikes.
- **Biology:** the basal ganglia "go/no-go/stop" loops that select an action by releasing its brake while suppressing rivals (Kandel Ch 38), and cortical evidence-accumulation to a decision threshold (Wang 2002; Lo & Wang 2006).
- **Done:** a full basal-ganglia selection circuit (per-action go/no-go channels, lateral competition) that solved a long-standing "stuck-motor" failure; and a **fully-spiking decision** — neurons accumulate evidence and fire a commitment burst — is now the default, having **retired the old ordinary-code shortcut** (a winner-take-all `argmax`), at a modest cost over the idealized version.
- **Open:** the global "cancel that action" pathway isn't wired; only the movement loop is modeled (not the parallel loops for choosing *what to say* or *what to think about* — those reuse the same machinery when needed); most striatal interneuron types are unmodeled.
- **Next:** add the parallel selection loops when non-movement choices (which fact to say, which topic) need the same circuit.

### 5.4 Reward & value — *what was worth doing* · 🟩 Done (reward signal + learning + drive) / 🟨 Partial (value critic)
- **Goal:** compute a reward-prediction-error teaching signal, learn how good situations are, and let motivation shape learning and choice — in spikes, not by formula.
- **Biology:** midbrain dopamine neurons signaling *actual minus expected* reward (Schultz 1998), three-factor learning where dopamine gates synaptic change, a value "critic," and hunger/incentive drives (Kandel Ch 41–43).
- **Done:** a spiking dopamine reward-prediction-error signal; three-factor learning in the engine; a value signal proven to actually drive choices (removing it collapses the high-value pick); and **one shared dopamine system that drives both the moving brain and the conversing brain** — a hungry brain literally becomes more careful about what it claims to know.
- **Open:** the explicit "how good is this situation" critic population isn't built yet (the highest-value next step); some reward is still computed by formula rather than by a circuit; no fear/aversion system.
- **Next:** build the explicit spiking value critic; convert the remaining formula-computed reward into a real circuit.

### 5.5 Navigation & spatial cognition — *moving through a world to reach goals* · 🟩 Done (the flagship embodied behavior) / 🟧 learned policy deferred
- **Goal:** an embodied agent that explores a world, figures out where it is and where the goal is *from what it perceives*, and navigates there — the project's first and most thoroughly-tested end-to-end behavior. This stage is where perception (5.1), orienting (5.2), action selection (5.3), reward (5.4), and spatial memory come together into a living behavior.
- **Biology:** hippocampal **place cells** (neurons that fire at specific locations; O'Keefe & Nadel) and entorhinal **grid cells** for a spatial map; the basal-ganglia action loop for movement; dopamine reward-prediction-error for learning the route; and prefrontal working memory holding the current goal (Kandel Ch 38, 54).
- **Done:** a gridworld agent that reaches (and re-reaches, when the goal moves) targets using **only perception** — the "cheat" inputs were progressively removed (no being handed its own coordinates, the goal's coordinates, a hand-coded heuristic, or a distance-based reward), so it navigates from a rendered visual/landmark scene through a learned perception→cortex pathway. Place cells emerge from landmark perception; prefrontal working memory holds the goal; adaptive dopamine and a maturation-style curriculum improve learning; the whole thing scales to larger grids with *tighter* variance, and the movement decision is the fully-spiking commit-burst from 5.3. The flagship result closes four of five original shortcuts and spends a large fraction of its time sitting on the goal.
- **Open:** the *learned* spatial policy (deciding the best move by reinforcement learning) currently uses a validated rate-based stand-in for the value function — replacing it with a truly self-taught deep policy is gated on the deeper learning rule (§9); a proper entorhinal **grid-cell** map and path-integration are not built (location is via place cells + landmarks); the world is a corridor/grid, not an open environment.
- **Next:** replace the rate-based policy stand-in with the dendritic-learned policy once that rule lands; enrich the world and add a grid-cell spatial map.

### 5.6 Memory — *holding on to experience* · 🟩 Done (mechanisms) / 🟧 one deep-consolidation boundary
- **Goal:** store episodes so similar ones don't blur together yet a partial cue can bring one back, tag memories so they can be reactivated, and move them into long-term cortical storage during "sleep" — all without erasing old memories.
- **Biology:** the hippocampal three-step loop (entorhinal → dentate → CA3 → CA1; Kandel Ch 54), where the dentate gyrus keeps memories separate and CA3 completes a memory from a fragment (Marr 1971); activity-tagged memory traces ("engrams"; Tonegawa, Liu 2012); and sharp-wave ripple **replay** during rest that both consolidates memories and — importantly — does the job that backpropagation-through-time does in artificial networks (Foster & Wilson 2006).
- **Done:** the full three-step loop with measured pattern-separation and pattern-completion; a production engram-tagging interface (tag a memory, reactivate it later) with ~90% cued retrieval; sleep-replay consolidation that moves memories to cortex with **no catastrophic forgetting** (verified with strict controls); and a striking result — **replay genuinely replaces backpropagation-through-time** for learning the conversation's discourse memory.
- **Open (🟧):** consolidating *composed* memories (a whole structured fact, not a single item) currently strands in the hippocampus and doesn't reach cortex — a characterized boundary; CA3 completes a single snapshot but not a *sequence*; no "theta rhythm" pacemaker or the finer sleep-stage generators yet.
- **Next:** attack the deep-consolidation boundary via a sequence-completing CA3 plus theta-rhythm compression, and build a theta pacemaker to improve replay quality.

### — The thinking-and-language brain —

### 5.7 Concept formation — *carving the world into categories* · ✅ Emergent
- **Goal:** discover categories and abstract concepts from experience, and learn the structure of sequences — unsupervised, and in spikes.
- **Biology:** a sparse-expansion coding trick from the cerebellum (Marr 1969, Albus 1971), the anterior temporal lobe as a concept hub with distributed word representations (Patterson & Lambon Ralph 2007; Pulvermüller), and a cortical sequence-memory mechanism using two-compartment neurons (Bouhadjar & Diesmann 2022).
- **Done:** **categories are discovered from experience** with increasing self-sufficiency — from co-occurrence, then from varied contexts, then from a self-organizing competitive layer, then grounded in real vision — fully in spikes; a **word-meaning cortex learned just from listening** to a stream of sentences (it picks up which words mean similar things); and a cortical sequence-memory that learns to predict the next symbol in context, with a built-in refusal to make things up.
- **Open:** one normalization step in the word-cortex read-out is still done by ordinary code rather than a circuit (a designed replacement exists); scaling the learned vocabulary to ~320 concepts needs a bigger source corpus; getting the memory attractors to *form themselves* (rather than be installed) is a separate open problem.
- **Next:** build the normalization circuit; scale the word cortex; pursue self-forming attractors (feeds the deeper learning rule).

### 5.8 Language comprehension — *understanding what is said* · ✅ Emergent / 🟧 recursion boundary
- **Goal:** map word-forms to who-did-what-to-whom and to meaning, including awkward word orders and long-distance dependencies — learned, in spikes, with no special-case code per sentence shape.
- **Biology:** the dual-stream model of language (a meaning stream and a sound-to-articulation stream; Hickok & Poeppel; Kandel Ch 55), Wernicke's area for word selection, and a frontal-striatal "reservoir" that maps word order onto roles (Hinaut & Dominey 2013).
- **Done:** a parser that assigns roles regardless of active/passive voice ("the dog chased the cat" and "the cat was chased by the dog" get the same agent); a **reservoir that learns the word-order→role mapping itself** (retiring an earlier hand-written labeler; it was hardened after adversarial review caught a too-easy first version); resolving a long-distance dependency across ~33 words, in spikes; and a neural router that figures out what kind of question is being asked (replacing a keyword lookup).
- **Open (🟧):** deeply nested sentences ("the cat the dog the man saw chased fled") degrade past a depth of ~3 — which is, honestly, roughly the human limit too; cross-language grammar (case markers, rich morphology) is a deliberate opt-in, not on by default; no hearing front end (it reads structured input, it doesn't process sound).
- **Next:** build the spiking working-memory buffer that pushes nesting depth on the brain itself.

### 5.9 Semantic reasoning — *inference beyond what it was told* · ✅ Emergent / ⬜ probabilistic accumulation open
- **Goal:** infer things it was never told — inheritance, exceptions, transitivity — and weigh evidence over symbols.
- **Biology:** the hippocampus as a relational/inference network (Eichenbaum; O'Keefe & Nadel), hierarchical semantic memory (Collins & Quillian), and the same evidence-accumulation math used for perceptual and value decisions applied to symbols (Kandel Ch 56).
- **Done:** **inheritance, multi-level inheritance, exceptions, and transitivity all emerge on their own** from overlapping neural codes plus the next-state predictor — no separate "inference engine" — over both given and *self-discovered* categories; a large adversarial audit found and fixed a class of measurement flaws, and every result survived its corrected test.
- **Open:** no "weigh evidence with learned reliability" reasoning primitive yet; the fully-spiking version of recalling a fact about a *newly seen* object is an honest boundary (a hybrid where code routes which concept fired works well); genuinely open-world inference beyond learned facts is a field-wide unsolved problem (managed by staying on-topic and abstaining).
- **Next:** wire a spiking evidence-accumulator for probabilistic reasoning (reusing the decision machinery from 5.3).

### 5.10 Language production — *speaking* · ✅ Emergent (within a bounded vocabulary) / 🟧 open prose deferred
- **Goal:** turn meaning into correctly-ordered speech — choosing function words, word endings, word order — with **every word produced by neural activity**, and the grammar worked out from experience rather than programmed.
- **Biology:** Broca's area for grammatical encoding and articulation (Kandel Ch 55), sequence-generation circuits for serial order (Grossberg; Kandel Ch 34), and the discovery of grammatical structure from statistics (construction grammar; Tomasello, Goldberg).
- **Done:** **the entire grammatical structure self-organizes from a corpus** — which words are function words, the slot order, the inventory of sentence templates — with the hand-written grammar removed as an input; **every word (content and function alike) is spelled out from the neurons' spike output**; the whole answer is produced by one brain in one process; and it renders seven sentence constructions in spikes, including transitive and ditransitive ("the dog gives the cat a bone").
- **Open (🟧):** it produces a **bounded, corpus-attested set of sentence patterns**, not open prose (the honest, deferred "roughly four orders of magnitude too small" wall); the spoken vocabulary per neural pathway is limited and scaling it is straightforward but linear; the emergent *open-ended* generator (see §9) is currently only at the token level over a small vocabulary.
- **Next:** scale the spoken vocabulary, then drive the open-generation ladder (§9) to replace the conventional-AI crutch.

### 5.11 Discourse & conversation — *tracking who-and-what across turns* · ✅ Emergent (the spiking core) / 🧩 fluent chat still leans on the crutch
- **Goal:** track who and what is being talked about across a multi-turn conversation, resolve pronouns, abstain when it doesn't know, and hold a growing, grounded chat.
- **Biology:** prefrontal working memory holding discourse referents (Kandel Ch 52), an attentional stack that pushes a topic when the focus shifts and pops it back on return (Grosz & Sidner; O'Reilly & Frank gating), and a familiarity signal that distinguishes known from unknown.
- **Done:** a spiking **discourse memory that tracks who's-acting-now versus who-was-acting-before** across a connective, learned by *replay* rather than backpropagation-through-time; multi-turn pronoun resolution on a persistent working-memory loop; a learned **no-confabulation guard** that matches a reference standard with zero breaches and is consulted *first* (so the fluency crutch is never even invoked when the brain should abstain); and a working "talkable brain" console where you can teach it facts, ask questions, and watch it reason, speak on spikes, and say "I don't know."
- **Open:** picking which of several remembered people a bare "it/they" refers to is an unsolved case (the fix is specified — a winner-take-all competition between the candidates); truly fluent single-pass synthesis over many facts still confabulates on the small conventional model; open-domain non-fact chatter is a field wall.
- **Next:** build the multi-referent disambiguator; drive the emergent generator (§9) to replace the conventional-AI fluency.

### 5.12 Working memory, sequence & recursion — *holding structure in mind* · 🟩 Done (working memory + graded memory) / 🟧 recursion depth
- **Goal:** hold an ordered set of items across a delay, process sequences with graded fading memory, and match nested structure with a bounded stack.
- **Biology:** a theta-gamma "7±2 slots" buffer (Lisman & Idiart 1995), NMDA-based persistent-activity working memory (Wang 2002), and a fading-memory reservoir (Hinaut & Dominey).
- **Done:** persistent working-memory attractors that hold items across a conversation; a reservoir that holds a distant cue across ≥16 intervening words, in spikes; and a slot-based buffer that pushes nesting depth to ~3 (scrambling the slots breaks it — proving the ordered structure is doing the work).
- **Open (🟧):** the *spiking* version of the slot buffer is designed but not yet built on the brain; depth beyond ~3 is the human-faithful bounded limit; no theta-rhythm pacemaker yet.
- **Next:** build the spiking slot buffer and a theta pacemaker (shared with the comprehension and memory stages).

### — The living whole —

### 5.13 Artificial life — *living, developing, remembering* · 🟩 Done (in pieces) / 🟨 Partial (as one unified life)
- **Goal:** a persistent brain that lives, perceives and remembers its own experience, develops over time, is driven by one shared motivational core, and can be talked to about its own life — all on one brain.
- **Biology:** complementary learning systems (hippocampus + cortex; McClelland 1995) with self-replay, a shared dopamine/hunger drive, and memory persistence across sessions.
- **Done:** a **develop-over-time loop** where a brain's vocabulary and knowledge grow day by day from real learning, with *zero* forgetting and the no-confab guard holding every day; the **moving brain and the conversing brain merged into one** with real synaptic interaction between them (a spoken command steers movement; what it perceives while navigating becomes something it can later be asked about and *compose into a new fact*); one shared drive affecting both halves; and persistence so a brain lives across a reset (resume its body and its lived memories, or start empty).
- **Open:** the learned movement policy is still the rate-based stand-in (the deferred deep-learning wall, §9); persistence re-instates saved facts rather than the raw synaptic tensor; the world is a corridor, not open-ended; the fact-composition path still uses an idealized binding scheme (§8).
- **Next:** merge the conversation machinery into a single co-resident brain; replace the stand-in policy with a self-taught one; move toward raw-synapse persistence and richer worlds.

---

## 6. The body & supporting biological systems

*The parts that surround the thinking brain, and biological systems present as building-blocks or planned.*

- **The body / motor output** — 🟨 the legitimate "body" interface. Per-action motor-neuron pools drive movement; there is no muscle model, spinal central-pattern-generators, or neuromuscular junction yet (catalog cluster H, M). Built out only if embodied tasks require it.
- **Cerebellum** — ⬜ cell-type presets exist (Purkinje, granule, climbing fibers) but the error-correcting microcircuit itself is not built; it's the natural home for predictive timing and fine motor/temporal learning (catalog cluster F; Marr-Albus-Ito).
- **Sleep architecture** — 🟨 sharp-wave-ripple replay is built and used; the finer NREM/REM stage generators (slow oscillations, spindles) are not (catalog cluster N).
- **Neuromodulation breadth** — 🟨 dopamine is fully deployed; noradrenaline / serotonin / acetylcholine are supported by the framework but only partially used.
- **Future / validation directions** — ⬜ disease models (Parkinson's, schizophrenia, epilepsy) are natural stress-tests of the basal-ganglia and cortical circuits; glia, neurovascular coupling, and molecular/transcriptional detail are deliberately out of scope at the current level of abstraction (catalog clusters P, Q).

## 7. How you watch it, talk to it, and grow it (interfaces & tooling)

*Not brain computations — these are the windows into the brain and the ways to interact with it.*

- **Real-time 3D viewer** — a live OpenGL view of every neuron firing in space, with camera, picking, and overlays (`neural-simulator.py`, the `viz/` package). Watching the brain think is a first-class feature.
- **The talkable-brain console** — an interactive chat where you teach the brain facts, ask questions, and watch it comprehend → reason → speak on spikes → abstain, all under the no-confab guard; a web front end with a brain picker and per-turn "brain activity" view.
- **The develop-run launcher** — a hands-off, resumable, pause-able way to let a brain live and develop over many simulated days, saving a loadable snapshot per day so you can load any day and talk to the brain at that stage of its life.
- **The experiment & stimulus system** — programmable stimulus injection, input/output neuron groups, readouts, and multi-phase training protocols for running controlled experiments.
- **The biological validation suite** — automated checks that the neurons reproduce textbook results (spike-timing plasticity curves, excitation/inhibition balance, paired-pulse facilitation, gamma oscillations, homeostasis).

---

## 8. Temporary stand-ins still in place (and how each gets replaced) · 🧩

1. **A small conventional AI model (~21M parameters) for open-ended fluency.** It supplies *only* the fluent wording, and only after the brain has decided what to say and checked it — it's never used when the brain should abstain. It's the one forbidden "permanent external model," so it must go. **Replacement:** the home-grown generation ladder in §9, whose first rung just landed.
2. **An idealized word-binding scheme** (a clean, exactly-reversible algebra for relating words into facts). Its *operations* already run in spikes, but the clean exact-reversibility isn't how a real cortex works, and one part (combining several attributes at once) provably can't be learned from scratch by simple neurons — so a fixed structural trick stands in. **Replacement:** a *learned* cortical binder — the same dendritic-learning frontier as §9.
3. **A few movement read-outs and reward values still computed by code** rather than by a circuit. **Replacement:** the fully-spiking decision is already the default; the remaining reward is a de-risked circuit away, plus the explicit value critic (5.4).
4. **A backpropagation-based training stand-in** used only as a development yardstick, at toy scale. **Replacement:** the biological dendritic learning rule — exactly as replay already replaced backpropagation for the discourse memory.
5. **One word-cortex normalization step done in code.** **Replacement:** a designed on-brain inhibition/adaptation circuit.
6. **Fact-based (rather than raw-synapse) persistence, and a rate-based movement policy.** **Replacement:** raw-synapse continuity and a self-taught policy, once the deeper learning rule lands.

## 9. The honest frontier (what's left, and the real walls) · 🟧 / ⬜

1. **A deep learning rule that works on spikes — the top lever.** Several ceilings (composing structure, deep nesting, the learned binder, the self-taught movement policy) all trace back to one missing capability: letting a *deep, multi-layer* spiking network teach itself. We have the feed-forward half working in an idealized (off-brain) form, and a two-compartment dendritic neuron that completes patterns a simple neuron can't. Getting a full spiking network to train end-to-end to real accuracy is being actively worked (this cycle we found and fixed a bug that had silenced the network, and it now learns above chance). *This is one candidate route — the dendritic one — and it is unproven on spikes; it is **not** on the critical path for the open-generation ladder below, which needs no deep learning rule.*
2. **Open-ended fluent generation without the conventional-AI crutch.** The home-grown replacement's first rung works — an emergent, on-brain, no-backpropagation next-word model that genuinely beats the standard baselines — but so far only over a small, controlled vocabulary. The remaining rungs (conditioning on working memory, generalizing to new combinations, spelling out an open vocabulary, multi-sentence discourse) are mapped but unbuilt and depend on scale. Fully model-free open-domain fluency is a field-wide wall even for LLMs (they lean on enormous scale and retrieval); our honest interim is to shrink the crutch and keep it fluency-only, behind the guard.
3. **Nesting deeper than ~3** — this is the human-faithful bounded limit, so it's a feature, not a bug; the only open build is the spiking version of the slot buffer.
4. **Consolidating composed/relational memories to cortex** — currently stuck in the hippocampus; the sequence-completing-CA3 path is the next lever.
5. **Choosing among several remembered referents for a bare pronoun** — the mechanism (a winner-take-all competition) is specified; a bounded build when multi-person dialogue is prioritized.
6. **The fully-spiking version of recalling a fact about a newly-seen object** — the hybrid works; the all-spiking version needs the learned binder.
7. **Genuinely open-world reasoning and open-domain chit-chat** — unsolved by anyone unconstrained; managed here (as LLMs do) by staying grounded, on-topic, and willing to abstain. This is the honest scope, not a near-term wall.

**Breadth items** (each a scoped build when a downstream need calls for it, not a fundamental limit): a deeper visual hierarchy and a separate location stream; hearing/smell/touch; the explicit value critic; a fear/aversion system; a theta-rhythm pacemaker; the finer sleep-stage generators; the global action-cancel pathway; the non-movement selection loops; grid cells and path integration.

---

## Appendix — how this roadmap is maintained

Kept current as a standing part of the workflow: whenever an experiment lands a result, surpasses a boundary, or removes a stand-in, the relevant stage above is updated in the same work cycle (status badge, the done/open notes, the next step, the citation). Periodically a deeper re-sync reads the source material in depth — the biological-mechanism catalog, the *Principles of Neural Science* textbook, and the experiment write-ups in `research/findings/` — to re-verify the biology map and the honest frontier. This file, not any single experiment write-up, is the intended at-a-glance source of truth for tracking progress toward the goal.
