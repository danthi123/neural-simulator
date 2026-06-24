# Science Roadmap

**The full project plan — past, present, and the path to the end.**

This is the project's living plan, written for three readers at once: a
curious non-expert who wants to know where this is going, a neuroscientist
who wants the mechanism and the evidence, and a contributor who wants to
know what to build next. Each item is stated plainly first; the technical
detail and the linked write-up follow.

**Last updated:** 2026-06-23

> If you only read one paragraph: this project is trying to build
> **artificial life with a real, biologically translatable brain** — a
> network of simulated neurons that learns the way a brain does (from local
> rules: spikes, synapses, dopamine), and to find out **how much of
> intelligence emerges from those rules alone.** Today it navigates from
> vision, holds a continual and trustworthy memory of ~320 concepts (it does
> not forget, and it refuses to make things up), binds concepts into
> structured facts in spikes, and runs all of this as **one network** that
> can navigate, perceive, generalize, compose, and converse. What remains is
> to make every last step run *in neurons* (a few small bridges still lean
> on ordinary code), to scale the **learned** model cortex to production
> size, and — the deepest open question — to find out whether a richer
> neuron (one with branching dendrites) is ultimately required, or whether
> the simpler neuron we use can go all the way.

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

The agent finds a goal on a grid using only **simulated retinal input** — no
direct coordinates and no hand-coded distance signal — reaching the goal far
above chance (about 38% of steps at the goal on a 16×16 grid in the validated
configuration). The decision of *where to step* is made by a simulated
basal-ganglia circuit (the brain's "action selector," where competing options
race and the winner is released), a neural reward signal, and a spiking
orienting reflex — i.e. *in neurons*, with no off-brain shortcut between
seeing and acting. As of June 2026 the move-decision is **made in spikes by
default**: an accumulator integrates the evidence and the race ends on an
all-or-none committing burst, retiring the last off-brain "pick the best
option" step (kept only as an optional baseline). This is a *genuine* neural
decision at an honest, reported cost — about 16% more steps than the shortcut
(the irreducible price the simple-neuron substrate pays to make the decision
itself a spike, validated across six seeds).
→ `research/findings/2026-06-19-spiking-decision-default-on-GO.md`; the
navigation arc is summarized in the
[Appendix](#appendix-the-historical-foundation), and the flagship recipe and
full cheat-closure history live in `CLAUDE.md`.

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
confabulate). The project calls this the **no-confab moat**.
→ `research/findings/2026-06-11-familiarity-gate-v320-GO.md` (a neural,
learned version of the abstention decision, validated at 320 concepts with
zero breaches).

### It binds concepts into structured facts — in spikes — *Done*

Beyond storing single words, it combines them into structured facts ("who did
what to whom," attributes, yes/no including **negation**, and even nested
clauses like "the dog sees the cat chase the ball") and answers questions
about them. The binding and unbinding are computed by **spiking neurons**,
not a lookup table. The full conversational stack is now comprehensively
complete and consolidated into one production agent: it also **reasons across
several facts** (multi-step chaining — "dog eats cat, cat eats mouse" → "what
does the thing the dog eats, eat?" → "mouse"), **tracks referents across
turns** (a later "it" resolves to the right thing), understands a **described
object** ("the dog ate the big apple" → "big apple"), and handles **flexible
word orders** beyond plain subject-verb-object — all validated at the
320-concept scale with zero fabrications. (The engine that does this was also
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

---

## Near-term (the active frontiers)

These are the things in flight or next in line. Each is labelled honestly for
how far along it is.

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

### 0b. A **grounded-language faculty** — a spiking LLM that speaks the brain's knowledge, hallucination-proof — *De-risked end-to-end*

A separation the owner asked for is now realized: an external language model
supplies **fluency only**, while the **brain** supplies the **knowledge,
the grounding, and the verification**. A real small language model
(Qwen2.5-0.5B), converted to run **fully in spikes** by the project's own
graded-read mechanism, generates coherent English (within ~8% of the original's
quality). It is then **gated and verified by the brain**: it may only phrase
facts the brain actually holds, and every sentence is parsed back and checked
before it is allowed out. The decisive proof: in testing the real model *did*
try to hallucinate — it inverted a fact into its false opposite ("rabbit chased
fox") — and the architecture **caught and rejected it**, so the
no-fabrication guarantee holds *with a real generative model in the loop*.
Validated end-to-end and at small scale (~67 facts, several seeds).
→ `research/findings/2026-06-23-grounded-lang-INTEGRATION-GO.md`;
the spiking faculty: `research/findings/2026-06-23-grounded-lang-P1b-GO.md`;
the scaled run: `research/findings/2026-06-23-grounded-lang-SCALED-GO.md`.

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
maximally different categories, with the no-confab moat clean.
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
  but the recall is keyed by the validated composer (the host only routes
  *which* concept fired). A *fully* spiking version of that last read-out
  hits an honest boundary today (the winner-take-all over fact tags plus a
  spiking confidence gate is too noisy) and is a bounded refinement, not a
  wall.
  → `research/findings/2026-06-16-generalization-capstone-verbalize.md`.
- Other small read-outs (embedded-clause word order, adjective–noun order,
  multi-frame syntax) remain literal templates and are scoped follow-ons to
  the spiking word-order generator already shipped.

### 3. A genuinely **learned** binding circuit — *Open (deepest of the near-term)*

Today the system combines concepts into facts using a **fixed, exact
algebra** (a vector-symbolic scheme, run in spikes) rather than a circuit
that *learned* to bind. That fixed algebra is the project's most useful
idealization: it buys the no-confab moat and reliable composition almost for
free — but it is not how a real cortex does it (a real cortex learns lossy,
redundant read-outs). Replacing it with a *learned* binder is the honest
endpoint of "fully biologize the composer."

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
two of six seeds. This is **not** a moat failure and **not** a firing
failure — the concept neurons fire strongly, they just vote for the wrong
category — and it is **not** fixable by loosening any gate.

The cause has been pinned down honestly. The first idea (more example images
per category, an "IT-prototype" fix) *did* raise accuracy — but it **broke
the no-confab moat**, so it was **rejected outright** (the moat is never
weakened to manufacture a pass). A cheaper analysis of existing data then
re-localized the real cause to **co-residence**: the very same generalization
that works standalone at those seeds degrades once it shares the bridge with
everything else, because of neuron-to-neuron variability at its location. The
moat-safe fix — average the category read over more neurons (a population
code), which sharpens the read **without** broadening categories — is the
test in flight.
→ `research/findings/2026-06-16-vision-to-concept-spiking-npercat-sweep.md`
(the rejected exemplar lever + the co-residence diagnosis);
scoping: `research/findings/2026-06-16-vision-to-concept-fidelity-scoping.md`.

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
concepts was shown to need *local, feedforward* normalization plus
similarity-from-shared-features plus Hebbian convergence — all of which a
point-neuron, feedforward network can do — and the "decorrelate the codes"
framing that *would* have demanded dendrites turned out to be a red herring
for that capability. So the months-scale dendritic-substrate rewrite is
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

- **The silent-motor trap, and the basal-ganglia fix.** A shared
  reservoir-plus-argmax action selector had a structural bias that no
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
- **Closing the perception and reward shortcuts.** Step by step, the direct
  coordinate access and the hand-coded distance reward were replaced with
  perceived beacon/landmark sensing and a sensed-reward gradient — and the
  biology-grounded version *beat* the shortcut-allowed version, a recurring
  and instructive theme.
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
