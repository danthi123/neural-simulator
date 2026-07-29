# Science Roadmap

**The full project plan — past, present, and the path to the end.**

This is the project's living plan, written for three readers at once: a
curious non-expert who wants to know where this is going, a neuroscientist
who wants the mechanism and the evidence, and a contributor who wants to
know what to build next. Each item is stated plainly first; the technical
detail and the linked write-up follow.

**Last updated:** 2026-07-23

> **▶ Current primary plan.** The ordered, forward-looking plan of record is the
> **master development roadmap,
> [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md)**
> (foundation:
> [`plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md`](plans/2026-07-22-genuine-conversation-affective-self-aware-brain-plan.md)).
> As of the **2026-07-23 direction pivot** the north-star has sharpened from
> "a conversation approaching a large language model's" to a sim-brain that
> **converses *genuinely*** — reasons to its own conclusions, and has an
> **affective world-model, emotion, self-awareness, and curiosity** — pursued on
> the honest **emergentist bet** that genuine experience emerges only from a
> *complete and faithful* biological emulation (functional consciousness
> correlates are built and measured; phenomenal experience is never asserted).
> This Science Roadmap remains valid as the deep scientific narrative and the
> launch-pad record; the master plan is the primary forward-looking surface, and
> the "five research frontiers" below are subsumed into its faculty-map +
> walls-ledger.

> If you only read one paragraph: this project is building **artificial life
> with a real, biologically translatable brain** — a network of simulated
> neurons that learns the way a brain does (from local rules: spikes,
> synapses, dopamine), to find out **how much of intelligence emerges from
> those rules alone.** Today, as **one network**, it navigates from simulated
> vision; holds a continual, trustworthy memory of a few hundred concepts (it
> does not forget old facts as it learns new ones, and it refuses to make
> things up); binds concepts into structured facts in spikes and answers
> questions about them; learns word meanings by "listening" to text;
> **discovers categories and simple taxonomies on its own** and reasons by
> inheritance ("a robin can fly because a bird can"); and increasingly
> **produces its spoken answers through its own spiking speech circuitry**
> (modelled on Broca's area, the brain's speech-production region). It can
> also live a simulated life — foraging, remembering what it meets, and
> growing day over day without forgetting. What remains are the open research
> frontiers the project is now working through: **open-ended fluent
> generation** by the brain's own circuitry, a **learned** concept-binding
> circuit, **resolving ambiguous references**, **dendrite-based credit
> assignment** (a local learning rule to replace backpropagation), and
> **memory replay and imagination** — plus scaling the learned model cortex
> and settling the deepest open question: whether a richer neuron with
> branching dendrites is ultimately required, or whether the simpler neuron
> used here can go all the way.

---

## How to read this roadmap

The work is organized by **horizon**, from what is already validated to what
is genuinely far off:

- **[The North Star](#the-north-star)** — the actual end goal.
- **[A note on method](#a-note-on-method-why-the-negatives-matter)** — how
  claims here are checked; this is part of the contribution.
- **[Where we are now](#where-we-are-now-the-launch-pad)** — what is already
  built and validated (condensed; this is the launch pad, not the focus).
- **[Near-term](#near-term-the-active-frontiers)** — the frontiers being
  actively worked, with honest "de-risked vs. open" labels.
- **[Mid-term](#mid-term)** — scaling, fluency, longer timescales.
- **[Long-term and the end-state](#long-term-and-the-end-state)** — the
  deepest open question, what "done" looks like, and the honest limits.
- **[Appendix: the historical foundation](#appendix-the-historical-foundation)**
  — the early credibility-floor work (analysis tools, biological-benchmark
  validation, performance, and the navigation-learning arc) kept for the
  record.

**Status labels** used throughout:

| Label | Meaning |
|---|---|
| **Done** | Built and validated, usually across several random seeds with controls. |
| **In progress** | Actively being built or validated right now. |
| **De-risked** | A cheap test says the approach should work; the full build is not finished. |
| **Planned** | Designed or scoped, not yet started. |
| **Open question** | A genuine unknown the project has deliberately not resolved yet. |

A note on vocabulary: a few recurring terms are defined the first time they
appear. The shortest glossary — *spike*, *plasticity*, *catastrophic
forgetting*, *composition/binding*, *the (model) cortex* — lives in the
[README](../README.md#glossary).

---

## The North Star

**Build artificial life with a proper, biologically translatable brain
analogue** — and use it to measure how much of intelligence emerges from
local biological rules *alone*.

Concretely, that means a self-contained agent whose every cognitive step —
perceiving, deciding, valuing, remembering, reasoning, conversing — is
carried out by **simulated neurons firing and synapses changing strength**,
using only mechanisms that have a counterpart in real brains (and are
therefore translatable *back* to neuroscience). No backpropagation through a
frozen graph, no symbolic shortcuts standing in for cognition, no external
language model doing the thinking.

**The 2026-07-23 sharpening.** The end target of "conversing" is now stated
precisely: not fact-recall/RAG and not a large model's plausible-sounding text,
but a brain that **converses *genuinely*** — reasons to *its own* conclusions,
carries an **affective world-model** (emotion, mood, valence), is
**self-aware** (reads and reports its own attention, confidence, and
authorship), and is **curious** (turns uncertainty into a drive to ask and
learn rather than a refusal). Stated at its most ambitious and most honest, the
end goal is **genuine subjective experience / true consciousness**. It is
pursued on the **emergentist bet**: consciousness emerges when a brain's full
capabilities and behaviour are emulated *completely and faithfully enough*. The
deliverable is therefore **completeness and faithfulness of the biological
emulation**, not a benchmark score. Early growth is accelerated by a **temporary AI teacher**
(the social environment, not the brain's cognition), then graduated into real
human interaction, with every scaffold biologized away toward the one spiking
brain. Two hard rules tighten under the pivot: **don't defer any needed
functionality** (every wall gets a real-biology surpass, never a permanent
shortcut), and **speed is secondary to faithfulness** (slow-but-faithful
mechanisms — deep dendritic credit, seconds-long plateaus, sleep-replay
consolidation — are explicitly in scope). Crucially, the **honesty boundary is
a deliverable, not a caveat:** every faculty delivers the standard *functional
correlates* of access-consciousness, self-modelling, and functional affect, and
every self-report is written as an honest functional read-out — never an
unlicensed claim of felt experience. The emergentist bet is the *reason to
pursue completeness*; it is not a license to *assert* the experience has
arrived.

Two consequences shape everything below:

1. **Capabilities are milestones, not the goal.** "It can navigate" or "it
   can converse" matter because they are *evidence* about what the substrate
   can do on its own — not as products to ship.
2. **An honest negative is a real result.** When the biological version of a
   capability *underperforms* a shortcut, that maps a genuine limit of the
   substrate, and that map is the scientific deliverable. The project's
   working standard ("brain-based only — anything not done by neurons,
   synapses, and their communication is a shortcut, even if the off-brain
   calculation is biologically correct") is what makes those negatives
   meaningful.

The boundary the standard draws: ordinary code is legitimate only for **the
world** (the environment's state and rendering what the agent's senses
receive) and **the body** (acting on the motor output). *Everything between
sensation and action is the brain's job.*

---

## A note on method (why the negatives matter)

Three practices are themselves part of the contribution, because they are
what let you trust the rest of the document:

- **Multi-seed validation.** A result re-run from several different random
  starting points ("seeds") so it is not a one-off fluke. Single-seed
  numbers are treated as indicators only; the standing bar for a generalized
  claim is **six seeds**.
- **Anti-cheat controls.** Before a capability is believed, a deliberately
  broken version is run to confirm the capability actually comes from the
  mechanism claimed — e.g. a *lesion* (sever the pathway → the behavior must
  collapse), a *permuted/derangement* control (scramble the mapping → the
  signal must die), and a *floor* baseline (unstructured inputs → chance).
- **Forthright retractions.** Several promising numbers in this project's
  history were **withdrawn** when a control later failed (a notable example:
  a whole batch of "concept-concept conversation" results in May 2026 turned
  out to be a measurement artifact and were retracted in full). Those
  corrections are kept in the record, not hidden. Honest negatives under
  strict biology are the point.

Every dated write-up referenced below lives under `research/findings/`
(including the negative ones — they are first-class findings).

---

## Where we are now (the launch pad)

This is the validated base the rest of the plan builds on. It is condensed
on purpose — each line is a capability that already works, with its evidence.

### It navigates from vision — *Done*

The agent finds a goal on a grid (including *moving* goals) using **simulated
retinal input**, reaching the goal well above chance. The decision of *where
to step* is made by a simulated basal-ganglia circuit (the brain's "action
selector," where competing options race and the winner is released), a neural
reward signal, and a spiking orienting reflex — i.e. *in neurons*, with no
off-brain shortcut between seeing and acting. As of June 2026 the
move-decision is **made in spikes by default**: an accumulator integrates the
evidence and the race ends on an all-or-none committing burst, retiring the
last off-brain "pick the best option" step (kept only as an optional
baseline). This is a *genuine* neural decision at an honest, reported cost —
roughly a sixth more steps than that shortcut, the irreducible price the
simple-neuron substrate pays to make the decision itself a spike (validated
across six seeds).

> **Accuracy note (mid-2026 internal audit).** Several older navigation
> headline claims were re-audited and corrected, and are *not* repeated here:
> a widely-copied description of one configuration as "navigates with no
> heuristic / all shortcuts closed" was found to still have the
> goal-direction heuristic switched on by default, and a "32×32 grid beats
> 16×16 by 13.3%" comparison mixed two *different* distance metrics.
> Navigation performance is real and characterized across grid sizes and
> multiple random seeds, but different configurations close different
> shortcuts; treat any single specific benchmark percentage with care and
> check the dated finding it comes from. The honest, verifiable statements are
> qualitative (it navigates from vision, the decision is a spike) plus the
> six-seed spiking-decision cost above.
→ spiking decision: `research/findings/2026-06-19-spiking-decision-default-on-GO.md`;
the audit corrections:
`research/findings/2026-07-16-anchor-claim-audit-10-defects-in-the-record-incl-my-own-correction.md`,
`research/findings/2026-07-16-clusterKv2-NO-heuristic-claim-is-FALSE-the-flag-that-closes-it-is-absent.md`.
The navigation arc is summarized in the
[Appendix](#appendix-the-historical-foundation).

### It holds a continual, trustworthy memory — *Done*

You can teach it word–concept facts ("apple is big"); it recalls them on cue,
and — the genuinely hard part — it **keeps old memories intact while learning
new ones** (avoiding *catastrophic forgetting*, the usual failure mode of
networks that keep learning). It holds roughly **320 distinct concepts**
across a five-part model cortex, validated across many seeds. The biology:
words stored as distributed cell assemblies spread across the cortex
(Pulvermüller 2001), recalled as sparse scattered patterns (Kanerva 1988),
each memory a re-triggerable tagged ensemble of neurons (Liu/Tonegawa 2012),
and protected by a hippocampus→cortex transfer with replay during simulated
sleep (complementary learning systems — McClelland, McNaughton & O'Reilly
1995).
→ `research/findings/2026-05-16-G20-sparse-ensemble-320concept-SHIPPED.md`;
forgetting validated in `research/findings/2026-05-08-Phase1.3-Tier2.1-strict-anti-cheat-3seed-CONFIRMED.md`.

### It refuses to make things up — *Done*

Asked about something it was never taught, it answers "I don't know" rather
than confidently fabricating — a trust property today's large language models
notably lack. This is not a wish; it is a measured **confidence gap** between
what it knows and what it does not, and it is guarded as a hard line (no
change is allowed to "fix" a capability by widening what the system will
confabulate). This is the project's **no-fabrication safeguard**: it declines
to answer rather than making something up.
→ `research/findings/2026-06-11-familiarity-gate-v320-GO.md` (a neural,
learned version of the abstention decision, validated at 320 concepts with
zero breaches).

### It binds concepts into structured facts — in spikes — *Done*

Beyond storing single words, it combines them into structured facts ("who did
what to whom," attributes, yes/no including **negation**, and even nested
clauses like "the dog sees the cat chase the ball") and answers questions
about them. The binding and unbinding are computed by **spiking neurons**,
not a lookup table. The full conversational stack is now complete and consolidated into one
production agent. It does four further things:

- **reasons across several facts** — multi-step chaining: "dog eats cat, cat
  eats mouse" → "what does the thing the dog eats, eat?" → "mouse";
- **tracks referents across turns** — a later "it" resolves to the right thing;
- understands a **described object** — "the dog ate the big apple" → "big apple";
- handles **flexible word orders**, beyond plain subject-verb-object.

All four are validated at the 320-concept scale, with zero fabrications. (The engine that does this was also
sped up 10–20×.) One honest boundary stays open: a *two*-attribute object
("big red ball") is not yet reliable on the learned codes.
→ consolidation onto the core network:
`research/findings/2026-06-04-conversational-pipeline-consolidated-onto-core-sim.md`;
recursive nesting: `research/findings/2026-06-03-recursive-clause-nesting-RESOLVES-depth3-capacity.md`;
multi-hop reasoning + multi-turn memory:
`research/findings/2026-06-17-multihop-query-chain-GO.md`,
`research/findings/2026-06-17-multiturn-anaphora-derisk-GO.md`;
described-object + flexible word order, and the 10–20× speedup:
`research/findings/2026-06-19-consolidation-attr-multiframe.md`,
`research/findings/2026-06-19-latency-csr-cache-GO.md`.

### One network does all of it — *Done*

Navigation, perception, the conversational comprehension network, the
working-memory/dialogue planner, and the fact-binding circuitry all run as
**separate groups of neurons on a single simulated brain** with one update
loop — not separate programs glued together. And the parts genuinely
**interact**: a *spoken command* the system parses can steer the body
(language → action), and the agent can **navigate to see an object, then
recall or compose a fact about what it saw** (perception → memory → reasoning).
→ merge: `research/findings/2026-06-10-step2b-rf-composer-coresident-COMPLETE.md`;
spoken-command steering: `research/findings/2026-06-10-spoken-instruction-nav-GO.md`;
navigate-to-see-then-recall: `research/findings/2026-06-16-navigate-to-see-then-answer.md`;
navigate-to-compose: `research/findings/2026-06-16-navigate-to-compose-then-answer.md`.

### It can learn concept *meanings* from a conversation stream — *Done at small scale*

A model cortex that **hears a stream of sentences word by word** and learns,
from co-occurrence alone (no preprocessing, no global bookkeeping), internal
codes that carry **meaning-similarity** — so "dog" and "cat" end up close
together and the system can answer about a concept by analogy to a similar
one. This is realized on the real spiking substrate (Hebbian "fire together,
wire together" co-occurrence learning), and the full conversational pipeline
runs on the stream-learned codes. Validated at **64 concepts** so far.
→ `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`;
the "what we were missing" reframe (it needs *local* normalization, not
cross-neuron decorrelation):
`research/findings/2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`.

### The conversational cognition is biologized — *Done*

The handful of cognitive steps in the conversational pipeline that used to be
done by ordinary code now have validated neural mechanisms: the
"don't-make-things-up" gate (a learned familiarity circuit), the cleanup that
maps a noisy pattern back to the nearest known concept (a spiking version of
a standard neural-engineering cleanup), the binding operation itself, and the
normalization on the read-out (neural firing-rate adaptation plus feedforward
inhibition). Even sentence **word-order generation** — the last output that
was a literal code template — now has an opt-in spiking serial-order
generator.
→ `research/findings/2026-06-16-biologization-sweep-conversational-pipeline.md`;
word-order generator:
`research/findings/2026-06-16-sentence-generation-serial-order-cheap-first-GO.md`.

### Generalization across similar concepts — *Done (mechanism), demonstrated end-to-end*

The hallmark of a real cortex: answer about a never-seen thing by reasoning
from a similar known one. The full mechanism is de-risked on the simple
neuron substrate (no richer neuron needed): meaning-similar visual inputs
(from the existing edge-detector visual front end) converge, via Hebbian
learning, onto concept neurons that then **fire for the right category** for a
brand-new object. End-to-end: a novel object seen through the real visual
front end makes its concept neurons spike in the correct category, and the
system recalls a fact about that category.
→ mechanism (four pieces, all multi-seed):
`research/findings/2026-06-16-generalization-graded-propagation.md` and the
sibling `2026-06-16-generalization-*` docs; capstone:
`research/findings/2026-06-16-generalization-capstone-vision-to-concept.md`.

### It discovers categories and reasons by inheritance — on its own — *Done at small scale (research-stage)*

Given only *experience* — a stream of co-occurring words, or objects *seen*
through the visual front end — the brain **discovers categories and simple
taxonomies by itself**, with no category labels supplied. From a handful of
told class facts it then **inherits**: a never-taught robin "can fly" because
a bird can; a penguin is an **exception** that walks; and it can inherit
through several levels at once (an owl "breathes" because an animal does,
two levels up). It can also do **transitive inference** — told only that A
beats B, B beats C, … it infers the never-shown A-vs-C ordering. All of this
runs on the spiking network, unsupervised, and you can then **converse with
it about what it discovered** ("can a robin breathe?" → "Yes"), while the
no-fabrication safeguard still holds for things it was never told ("I don't
know what a *zzz* is"). These are validated across multiple seeds but at
small scale — a research-stage capability, not a production feature.
→ the emergent-language experiments: inheritance and exceptions
`research/findings/2026-07-02-emerge26-emergent-inheritance-GO.md`;
multi-level taxonomy `…-emerge27-multilevel-taxonomy-GO.md`;
transitive inference `…-emerge28-transitive-inference-GO.md`; categories
discovered from what it *sees* `…-emerge34-perception-grounded-emergence-GO.md`.

### It speaks its grounded answers — increasingly in its own spiking circuitry — *Research-stage, validated across seeds*

Fluent phrasing arrives in two layers. **(a)** A **small, locally-trained
language generator** supplies fluent English *phrasing only*. It holds tens of
millions of parameters — far fewer than a typical large language model. The
brain decides *what* is true, and whether to answer at all. The generator is
**never invoked when the brain chooses to abstain**, so the no-fabrication
safeguard holds by construction. This generator is a deliberate, temporary
scaffold. **(b)** For a **bounded set of sentence forms**, the brain's **own
spiking circuitry** now produces the words *and their order* — modelled on
**Broca's area**, the human speech-production region — and, importantly, it
**learns that sentence structure from a text stream** (which words are the
"grammar glue," the slot order, and each construction's slot inventory are all
discovered from a corpus rather than hand-written) and spells every word out
of firing neurons. This is validated across several seeds; its honest scope is
a bounded inventory of sentence forms, *not* open-ended prose (that is a
near-term frontier below).
→ the two-layer fluent-conversation system:
`research/findings/2026-07-01-fluid-conversation-console-capstone.md`;
the brain's own speech-production circuitry (grammar self-organized from a
corpus, every word produced on spikes):
`research/findings/2026-07-03-emerge65-self-organized-producer-GO.md`,
`research/findings/2026-07-03-emerge69-console-fully-spiking-GO.md`.

---

## Near-term (the active frontiers)

These are the things in flight or next in line. Each is labelled honestly for
how far along it is.

### The five current research frontiers

> **Now a sub-view of the master roadmap.** The five frontiers below were the
> near-term spine before the 2026-07-23 pivot; they remain valid and actively
> worked, but they are now **subsumed into the master development roadmap's
> faculty-map + walls-ledger** ([`plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md)),
> which broadens the frontier set to the full pivot: an **affect-state system**
> (emotion, mood, valence-tagging, appraisal), **self / metacognition** (a
> self-schema and second-order confidence monitor), **curiosity** (uncertainty
> as a learning drive), **memory consolidation** of composed/relational
> memories, and **social cognition / theory-of-mind** — each with a named
> biological surpass. The three Phase-0 foundations of that broader push landed
> this cycle, each 6-seed GO: *curiosity inversion*
> (`research/findings/2026-07-23-DR1-curiosity-inversion-6seed-GO.md`),
> *affective concept tagging*
> (`research/findings/2026-07-23-DR2-affective-concept-tagging-6seed-GO.md`), and
> a *self-schema region*
> (`research/findings/2026-07-23-DR3-self-schema-region-6seed-GO.md`).

The project's near-term work is organized around **closing the remaining
capability gaps so that every step is done by spiking neurons on one brain.**
Five open research questions define that push (each is expanded below and, for
the deeper ones, in later sections). They are frontiers being *actively worked*
— not solved features:

1. **Open-ended fluent generation** — moving beyond a bounded set of sentence
   forms toward free conversation, produced by the brain's *own* circuitry
   rather than a bolted-on language model. *(Open — the hardest of the five;
   see "Open-ended language toward fluency" under [Mid-term](#mid-term).)*
2. **A learned concept-binding circuit** — replacing today's fixed, exact
   scheme for combining concepts into facts with one the brain *learns*.
   *(Open — detailed status in "A genuinely learned binding circuit" below.)*
3. **Resolving ambiguous references** — deciding which of several remembered
   things a bare pronoun ("it") means. *(Recently de-risked — a winner-take-all
   "biased competition" between the candidate memories, validated across six
   seeds; detailed status in "Resolving ambiguous references" below.)*
4. **Dendrite-based credit assignment** — a *local* learning rule (how a
   neuron works out which of its inputs to strengthen) that does not rely on
   backpropagation. This is the likely enabler for open-ended generation, and
   is the near-term face of the deepest open question about the neuron model
   (see "is a richer neuron ultimately required?" under
   [Long-term and the end-state](#long-term-and-the-end-state)). *(Open.)*
5. **Memory replay and imagination** — the brain internally *replaying* and
   *recombining* stored sequences (as the hippocampus does during rest), to
   support planning and imagination. *(Open — pattern-completion/replay in the
   CA3 hippocampal circuit is the current build target; several earlier
   sequence-replay attempts were honest negatives.)*

The remaining subsections below give the detailed status of each build in
flight, including three of these five frontiers (learned binding, reference
disambiguation, credit assignment) and the artificial-life development loop
they all feed.

### 0. The artificial-life **development loop** — *De-risked at small scale (the current north-star arc)*

The project's actual end goal (see [The North Star](#the-north-star)) is
**artificial life that develops**: a brain that lives through simulated days
of conversation, grows, and remembers — and that a human can then talk to. As
of June 2026 this loop runs end-to-end on the spiking network at small scale.
In a single GPU smoke run the brain **develops over four simulated days**:
hearing a daily curriculum, it learns the word co-occurrence with a plain local
rule (learning quality ~0.89), grows its vocabulary (6→24 words) and its store
of facts (2→11), recalls them perfectly, **keeps the old memories** (no
catastrophic forgetting), and **never fabricates** (zero false-accepts every
day). It **persists and resumes** — a later run picks up where the last left
off and lives more days — and a frozen-brain control (no learning) develops
nothing, confirming the development is real. The timing is tractable: about two
minutes per simulated day, so a compressed "week" is ~16 minutes and a "year"
is an overnight run, all on local hardware — no cloud needed. What remains is
to scale the horizon (a "month"/"year" brain), confirm across more seeds, and
wire up the human-talks-to-the-developed-brain step.
→ `research/findings/2026-06-23-longitudinal-develop-loop-GPU-GO.md`; scoping:
`research/findings/2026-06-23-artificial-life-longitudinal-test-scoping.md`.

### 0b. A **grounded-language faculty** — a language model that speaks the brain's knowledge, hallucination-proof — *De-risked end-to-end*

A separation the owner asked for is now realized: the language model supplies
**fluency only**, while the **brain** supplies the **knowledge, the grounding,
and the verification**. This was first demonstrated with a real small language
model (Qwen2.5-0.5B), converted to run **fully in spikes** by the project's own
graded-read mechanism, generating coherent English (within ~8% of the
original's quality). It is then **gated and verified by the brain**: it may
only phrase facts the brain actually holds, and every sentence is parsed back
and checked before it is allowed out. The decisive proof: in testing the real
model *did* try to hallucinate — it inverted a fact into its false opposite
("rabbit chased fox") — and the architecture **caught and rejected it**, so the
no-fabrication guarantee holds *with a real generative model in the loop*.
Validated end-to-end and at small scale (~67 facts, several seeds).

**Since then (July 2026) the everyday conversational scaffold has shrunk
dramatically.** The design now leans on a **~21-million-parameter** locally
trained generator — 15–25× smaller than that half-billion-parameter model —
which is enough to carry fluent phrasing inside the same *decide-then-phrase,
verify, abstain* loop (multi-turn pronoun tracking, learning new facts live,
persistence across sessions, and grounded discussion built from real
encyclopedic facts). This keeps the transformer **minimized** rather than
eliminated: the small generator is a deliberate temporary scaffold, and the
brain's *own* spiking speech circuitry (see the launch-pad entry, "It speaks
its grounded answers") is progressively taking over the phrasing for the
sentence forms it can already produce.
→ the Qwen demonstration: `research/findings/2026-06-23-grounded-lang-INTEGRATION-GO.md`,
`research/findings/2026-06-23-grounded-lang-P1b-GO.md`,
`research/findings/2026-06-23-grounded-lang-SCALED-GO.md`;
the current small-generator conversational system:
`research/findings/2026-07-01-fluid-conversation-console-capstone.md`.

### 0c. **Bridge co-residence** — the language faculty *on the brain's own engine* — *Demonstrated (feasibility); perf is the open item*

The "one brain" goal for language: run the whole spiking language faculty **on
the simulation engine itself**, alongside the conversational brain, rather than
as a separate process. This was shown feasible and **local**: the full
494-million-parameter, 24-layer spiking model runs on the simulation engine's
resonate-and-fire substrate on a single 24 GB GPU (~14 GB used), producing
**identical output** to the off-engine version. The honest catch is **speed,
not memory** — it runs but slowly (prefill is usable at ~187 tokens/sec after
a first optimization; token-by-token generation is still launch-bound and is
the next engineering lever, a key-value cache). No simulation-engine code was
changed. The faculty *running on* the brain is demonstrated; making it fast,
and making it *interact* with the conversational brain, are the follow-ons.
→ `research/findings/2026-06-23-bridge-coresidence-DEMONSTRATED.md`; perf:
`research/findings/2026-06-23-bridge-coresidence-perf-dense-matvec-GO-WITH-CAVEAT.md`.

### 1. Scale the **learned** cortex to production size (~2,048 concepts) — *In progress*

There are two model cortices in this project, and the distinction matters:

- The **memory cortex** above (~320 concepts) stores each concept as its own
  *separate* pattern. It is trustworthy and continual, but it cannot reason
  from similarity — to it, "dog" and "cat" are simply different.
- The **learned cortex** gives concepts internal codes that **carry
  meaning-similarity**, so similar concepts sit close together and
  generalization becomes possible.

A learned cortex spanning **~2,048 concepts** (across many small spiking
sub-networks) has been **built and its core capabilities directly confirmed
at full scale**: every sub-network passes the who/what/negation/clause
conversational matrix, and within-network generalization (inferring about one
concept from a similar one) scores ~0.99 — about four times chance — across
maximally different categories, with the no-fabrication safeguard clean.
→ `research/findings/2026-06-14-phase1-production-32bridge-2048-concept-cortex-DELIVERED.md`.

**Honest status of the two remaining pieces:**

- *Where the similarity comes from.* The 2,048-concept build used a
  **curated** similarity scheme (a hand-organized taxonomy plus a brain-based
  learning step) — a principled stepping-stone. Learning that similarity from
  **raw experience** instead is validated at **64 concepts** (the stream
  cortex above) and is the harder, more faithful target at production scale;
  scaling the *stream-learned* path to 2,048 concepts is open work (it needs
  a larger corpus-grounded vocabulary).
- *A clean full run.* The 2,048-concept run was deliberately **stopped
  partway** for efficiency — a memory-accumulation slowdown over dozens of
  sequential builds made the remainder a re-confirmation of an already-saturated
  result. A clean, uninterrupted full run awaits a build-system fix (release
  memory between sub-networks); this is an engineering item, not a science gap.

### 2. Finish the **all-spiking** conversational pipeline — *De-risked; a few pieces open*

Most of the conversational cognition is now neural (see the launch pad). The
remaining off-brain steps are small handoffs and read-outs, each with a
validated or de-risked neural replacement in hand:

- The fully-spiking **fact recall from a generalized percept** (perceive a
  novel object → fire its concept → recall a fact about the matched category)
  currently works via a **hybrid**: the generalization is genuinely spiking,
  but the recall is keyed by the validated conversational binding system
  (ordinary code only routes *which* concept fired). A *fully* spiking version
  of that last read-out
  hits an honest boundary today (the winner-take-all over fact tags plus a
  spiking confidence gate is too noisy) and is a bounded refinement, not a
  wall.
  → `research/findings/2026-06-16-generalization-capstone-verbalize.md`.
- Other small read-outs (embedded-clause word order, adjective–noun order,
  multi-frame syntax) remain literal templates and are scoped follow-ons to
  the spiking word-order generator already shipped.

### 3. A genuinely **learned** binding circuit — *Open (frontier 2 above)*

Today the conversational binding system combines concepts into facts using a
**fixed, exact algebra** (a mathematical scheme for combining and separating
patterns, run in spikes) rather than a circuit that *learned* to bind. That
fixed algebra is the project's most useful idealization: it buys the
no-fabrication safeguard and reliable composition almost for free — but it is
not how a real cortex does it (a real cortex learns lossy, redundant
read-outs). Replacing it with a *learned* binder is the honest endpoint of
"make the whole binding system neural."

Where this stands, precisely:

- A learned role–filler binding **generalizes single-attribute facts** and is
  validated **on real spikes** — held-out performance matching the reference.
- But **bundling** (a fact as a superposition of several bindings) is **not
  learnable from scratch** on the simple neuron substrate: additive
  superposition has no clean inverse, and a learned *linear* inverse cannot be
  a true reciprocal. A *fixed* coincidence-style binding primitive bundles
  cleanly on the same test — so the current production answer is **learned
  representations flowing through a fixed, biology-grounded binding
  primitive** (binding-by-coincidence is itself a structural neural primitive,
  not an off-brain shortcut).
→ `research/findings/2026-06-16-onsubstrate-learned-binder-single-attr-GO-bundling-NEGATIVE.md`.

A fully learned, fully spiking binder (abandoning the fixed algebra) is a
deliberately **lower-priority, months-scale, uncertain** trade — revisited
only after the cheaper wins land, and brought to an explicit decision when it
is.

### 4. Close the unified agent's last generalization wobble — *In progress (cheap fix in flight)*

The one network that navigates + perceives + generalizes + composes +
converses is **robust across all six seeds on everything except one
sub-skill**: the vision→concept generalization keys the *wrong* category at
two of six seeds. This is **not** a fabrication-safeguard failure and **not** a
firing failure — the concept neurons fire strongly, they just vote for the
wrong category — and it is **not** fixable by loosening any gate.

The cause has been pinned down honestly. The first idea (more example images
per category, an "object-prototype" fix) *did* raise accuracy — but it **broke
the no-fabrication safeguard**, so it was **rejected outright** (the safeguard
is never weakened to manufacture a pass). A cheaper analysis of existing data
then
re-localized the real cause to **co-residence**: the very same generalization
that works standalone at those seeds degrades once it shares the network with
everything else, because of neuron-to-neuron variability at its location. The
safeguard-preserving fix — average the category read over more neurons (a
population code), which sharpens the read **without** broadening categories —
is the test in flight.
→ `research/findings/2026-06-16-vision-to-concept-spiking-npercat-sweep.md`
(the rejected exemplar lever + the co-residence diagnosis);
scoping: `research/findings/2026-06-16-vision-to-concept-fidelity-scoping.md`.

### 5. Resolving ambiguous references — *De-risked (frontier 3 above)*

When a conversation holds *several* remembered things and the next sentence
says "it," which one does "it" mean? Recency alone does not decide it, and a
simple salience boost was an honest negative. The mechanism that does work is a
**winner-take-all "biased competition"**: the candidate memories inhibit one
another until a single one wins, biased by how well each fits the current
sentence — exactly how cortex is thought to resolve competing options. This has
now been validated **across six seeds** with the usual controls, and wired into
the multi-turn dialogue system. It is a good illustration of a frontier moving
from *open* to *de-risked*: the earlier failed ideas are kept in the record as
the negatives that pointed to the right mechanism.
→ the mechanism, six-seed:
`research/findings/2026-07-21-gap3-biased-competition-multireferent-6seed-GO.md`;
the earlier honest negatives that located it:
`research/findings/2026-06-17-multireferent-disambiguation-NEGATIVE.md`.

### 6. Dendrite-based credit assignment — *Open (frontier 4 above)*

Every neuron in the system today is a **point neuron** — a single electrical
compartment — and learning is driven by local spike-timing and reward rules.
To grow beyond hand-shaped learning toward *open-ended* capability (frontier
1), the project is working toward a **local, dendrite-based credit-assignment
rule**: a biologically plausible way for a neuron to figure out *which of its
inputs* to strengthen, without the global backpropagation that ordinary neural
networks rely on. This is the near-term face of the project's deepest open
question — whether a richer, branching-dendrite neuron is ultimately required —
and it is the likely enabler for open-ended fluent generation. It is genuinely
open; see "is a richer neuron ultimately required?" under
[Long-term and the end-state](#long-term-and-the-end-state) for the full
framing and the evidence on both sides.

### 7. Memory replay and imagination — *Open (frontier 5 above)*

Real brains **replay** stored experience during rest — the hippocampus
re-runs and recombines sequences, which is thought to underpin planning,
consolidation, and imagination. Bringing this to the spiking brain means the
**pattern-completion** step of the hippocampal CA3 circuit: cue it with a
fragment and have it complete or *replay* a stored sequence, and — the harder
part — recombine fragments into something *new* (imagination). Several earlier
sequence-replay attempts on this substrate were honest negatives (direct
memory binding consolidates well, but faithful sequence *completion* did not
fall out of it), which is why CA3 completion is now an explicit build target
rather than an assumed capability.
→ the honest negatives that scoped it:
`research/findings/2026-05-24-c-generative-replay-decisive-NEGATIVE-loop-at-n-iterations-1-doesnt-produce-above-chance-completion-pivot-direction-identified.md`;
learned CA3 completion de-risk:
`research/findings/2026-06-09-learned-graded-ca3-derisk-RESULT.md`.

---

## Mid-term

The horizon beyond the active frontiers — bigger, harder, and honestly not
yet started in earnest.

### Scaling — *Planned*

- **More concepts.** Beyond 2,048: the architecture scales roughly linearly
  in the number of sub-networks, so the limits are training time and corpus
  size more than a wall — but each scale-up needs its own validation, not an
  assumption.
- **More neurons per region.** Today each region is thousands of neurons
  versus 10⁴–10⁶ in biology. Larger regions may change what reliably forms;
  this needs measuring, not extrapolating.
- **Harder tasks.** Larger grids, more goal types, multi-step plans for
  navigation; richer fact structures and multi-turn dialogue for conversation.

### Open-ended language toward fluency — *Open (foundation proven, the full learn-loop now demonstrated, fluency still far)*

The system's *own* spiking network is being trained to generate language from
a local text corpus, using a spike-compatible form of gradient learning. The
**foundation is validated** — it provably learns *real* text structure, not
noise — but it is **not yet fluent**, and honestly far from a large language
model: a measured capability gap of roughly four orders of magnitude, which
matches what the field sees at this scale. A separate model may be used *only*
as a **training-time teacher** (knowledge distillation); after training the
system runs entirely on its own, fully local, with no external model and no
hand-written reply templates. Closing the fluency gap on biologically faithful
hardware is a genuinely hard, open frontier — possibly the hardest in the
project.

**New (June 2026): the full generative *loop* is now demonstrated** — train on
a distribution → generate → **grow** the network → confirm it **did not
forget** the old distribution — end-to-end on the spiking generator, across
three seeds. A "scale wall" that had stopped this turned out to be a tuning bug
(a fine-tuning learning rate 30× too high); with it fixed, self-replay during
the grow step **causally prevents forgetting** (retention 0.88 with replay vs
0.39 without). The generator's last math operations (its normalization and
output steps) now also run **in spikes**. One honest boundary is mapped: the
tiny demonstration network cannot hold *two* similar distributions at once — a
genuine capacity wall that would need a larger (~50–200M-parameter, still
local) generator, and which affects only the optional *free-generation* upgrade,
not the development loop's memory (which is carried by the stream cortex and
consolidation, both validated to retain).
→ `research/findings/2026-06-23-generative-loop-DEMONSTRATED.md`,
`research/findings/2026-06-23-C2-moderate-shift-NEGATIVE-scale-wall.md`;
the original ceiling + field-context:
`research/findings/2026-06-02-generative-ceiling-spiking-LM-NEGATIVE-overfit-not-size.md`,
`research/findings/2026-06-03-pre-compute-review-the-tiny-LLM-gap-is-ALREADY-MEASURED.md`.

### Richer working memory and longer timescales — *Planned*

Prefrontal working memory currently holds a goal for *seconds*. Extending to
longer, more structured, multi-item working memory — and adding the slower
forms of memory consolidation that real brains use — is needed before
genuinely extended reasoning or long conversations are realistic.

---

## Long-term and the end-state

### The deepest open question: is a richer neuron ultimately required? — *Open question (deliberately deferred)*

Every neuron in the system today is a **point neuron** — a single electrical
compartment. Real neurons have **dendrites**: elaborate input branches that
do their own local computation before the cell decides to fire. A recurring
result in this project is that certain computations (a particular kind of
input normalization called *whitening*, which decorrelates signals before
they are spiked) appear to be things a point neuron **fundamentally cannot
do** — biology does them in the analog, pre-spike stage inside a dendrite or
the retina. This is a known theoretical limit (the Mikulasch–Priesemann
point-neuron bound), and the project has hit it repeatedly.

The honest and important finding so far: **the present generalizing cortex
does *not* require the dendritic rewrite.** Generalization across similar
concepts needs three things: *local, feedforward* normalization, similarity
from shared features, and Hebbian convergence. A point-neuron, feedforward
network can do all three. The "decorrelate the codes" framing *would* have
demanded dendrites, but it turned out to be a red herring for this
capability. So the months-scale dendritic-substrate rewrite is
**deliberately deferred**, and shown not to be on the critical path for what
is being built now.

But it remains the project's deepest open question. There is a real fork:

- **Path A — a semantically *flat* cortex** (the present direction):
  achievable now, passes the full conversational matrix, but cannot
  generalize across *similar* concepts unless similarity is supplied.
- **Path B — a semantically *structured* cortex** that learns and preserves
  similarity from raw experience and generalizes broadly. The current
  evidence is that the point-neuron substrate, with the right *local*
  mechanisms, can go a long way here (the stream cortex at 64 concepts is the
  proof of concept) — but whether it reaches full generality, or whether some
  capability genuinely demands dendrites, is **not yet known**. The dendritic
  rewrite is the highest-variance, highest-cost item on the whole map, and a
  deliberate owner call to defer until the cheaper paths are exhausted.
→ the fork and its evidence:
`docs/plans/2026-06-11-cortex-build-plan-decorrelate-then-bind.md`;
the reframe that deferred the rewrite:
`research/findings/2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md`;
the substrate limit itself:
`research/findings/2026-06-06-decorrelation-blocker-deep-research.md`.

### Toward a self-contained, brain-like agent — *Planned (the synthesis)*

The end-state is the synthesis of everything above into **one always-on
agent**, fully local, that:

- perceives its world through simulated senses;
- decides and acts through simulated motor circuits;
- remembers continually and trustworthily, without forgetting and without
  fabricating;
- reasons by analogy across what it knows;
- and converses about all of it — every step carried out *in neurons*, with
  ordinary code confined to the world and the body.

Much of the scaffolding exists (one network, the cross-region interactions,
the learned cortex, the continual memory). The end-state is reached when the
last off-brain shortcuts are converted or honestly bounded, the learned
cortex is at production scale and learning its similarity from experience, and
the whole thing runs as a single persistent instance.

### What "done" would look like — and the honest limits

"Done" is **not** "matches GPT." It is: a single, persistent, fully-local
agent in which *no cognitive step is a shortcut* — perception, decision,
value, memory, reasoning, and language all realized by spiking neurons and
plastic synapses — that learns continually from interaction, generalizes from
similarity, holds its trust property (no fabrication), and whose every
mechanism is **translatable back to neuroscience**. Reaching that, even at a
modest scale, would be a concrete answer to the project's question: *this
much* of intelligence emerges from local biological rules alone.

The limits that will remain even then, stated plainly:

- **Scale vs. real biology.** Thousands of neurons per region versus millions;
  far fewer learning experiences than a developing brain gets. The claims are
  about *mechanism faithfulness*, not biological-scale equivalence.
- **A single timescale.** Learning here is millisecond spike-timing
  plasticity only; biology also has slow, protein-synthesis-dependent
  consolidation over hours and days, which is not modeled.
- **Static structure.** Developmental wiring changes — synaptic pruning,
  cortical-layer formation — are not modeled; the architecture is declared,
  not grown.
- **Not LLM-fluent.** Local hardware caps open-ended generation well below
  cloud models. The contribution is *integrity* (no cheating, no fabrication,
  self-contained, biology-faithful), not parity.
- **Research software, not validated for any clinical or diagnostic use.**

These limits are not failures to hide; they are the honest boundary of "how
much of intelligence emerges from biology alone," which is the whole point.

---

## Appendix: the historical foundation

The early work below established the credibility floor — the analysis tooling,
the biological-benchmark validation, the performance engineering, and the
reward-learning/navigation arc that the rest of the project stands on. It is
condensed here; the blow-by-blow lives in `CLAUDE.md`, `CHANGELOG.md`, and the
dated findings under `research/findings/`.

### Analysis tooling — *Done (2026-04)*

A parameter-sweep framework (`run_parameter_sweep.py`), spectral analysis
(per-band power at phase transitions), population-synchrony metrics (Fano
factor), and an automatic statistics layer (Welch's t-test, Cohen's d). These
turned "runs experiments" into "produces analyzable results."

### Biological-benchmark validation — *Done (2026-04), all passed*

The simulator reproduces a battery of published neuroscience results, which is
what licenses any novel claim:

| Benchmark | Result |
|---|---|
| STDP timing curve (Bi & Poo 1998) | Kernel matches theory to ~3×10⁻⁸; full simulation verified |
| Excitation/inhibition balance | 80/20 split, biologically correct firing rates and irregularity |
| Short-term plasticity (Tsodyks–Markram) | Depressing and facilitating paired-pulse ratios as published |
| Gamma oscillations (PING) | Peak 27–45 Hz; a connectivity bug affecting *all* spatial networks was found and fixed here |
| Homeostatic firing-rate regulation | 10× perturbation recovers to baseline within ~1 s |

→ run via `python run_benchmarks.py --benchmark <name>`.

### Performance engineering — *Done (2026-04)*

Batched sparse matrix multiplication, GPU memory-pool hygiene (removing
mid-simulation stalls), render frame-skipping, and LZ4-compressed recording.
Net effect: roughly **7–8× faster** than the project's original single-file
implementation, and the modular package layout (`sim/`, `viz/`, `ui/`,
`experiment/`) that everything since has been built on.

### The reward-learning and navigation arc — *Done (2026-04 → 2026-05)*

The arc that produced vision-only navigation, told as its key turning points:

- **The silent-motor trap, and the basal-ganglia fix.** A shared "reservoir"
  network with a pick-the-maximum action selector had a structural bias that no
  runner-side trick could fix (seven negative attempts). The resolution was a
  **per-action basal-ganglia cascade** where each action has its own
  populations and selection emerges from independent disinhibition gates —
  the silent-motor trap then cannot occur structurally.
- **Curriculum and per-pathway plasticity control.** Letting sensory layers
  mature before association layers (a real critical-period idea), via
  per-pathway "freeze/thaw" gates, made plastic input layers work where
  cold-start learning had failed.
- **Working memory.** A recurrent prefrontal region holding a goal in mind
  (NMDA-based persistent activity) gave a clean multi-seed improvement.
- **Closing the perception and reward shortcuts.** Step by step, some direct
  coordinate access and the hand-coded distance reward were replaced with
  perceived beacon/landmark sensing and a sensed-reward gradient, and in the
  headline configuration the biology-grounded version was at least competitive
  with the shortcut-allowed one. *(A mid-2026 audit is a caution here. Several of the original "biology beats
  the shortcut" and "all shortcuts closed" headlines were later found to be
  favourable-seed selections. Others had mixed up which shortcuts a given
  configuration actually closed. Treat the qualitative story as real, and the
  specific old numbers with care.)*
- **Re-classification under the strict standard (2026-06).** Several of those
  navigation wins were biologically *shaped* but still computed by ordinary
  code; under the project's "brain-based only" standard they were honestly
  reclassified as shortcuts, and their fully-spiking versions (a spiking
  orienting reflex, a neural reward/value system, a spiking dopamine signal,
  a spiking action read-out) became the real targets — now largely built, as
  the launch-pad navigation entry reflects.

This is also where the project's method crystallized: multi-seed validation
(after a 3-seed result failed to replicate at 6 seeds), anti-cheat controls,
and treating negative results as findings — the practices that the rest of
this roadmap depends on.
