# Chat demo guide — talking to the biology-grounded conversational brain

A guide to running and understanding the conversational demos, from the
**grounded conversational agent** (the whole pipeline running on one spiking
brain) through the newer research-stage capabilities (discovering categories,
fluent replies, developing over days) down to the small, portable word-binding
demos that the whole thing was built on.

**Last updated:** 2026-07-23.

> **Where the project is now.** The conversational pipeline — parse a sentence
> into who-did-what, store facts, recall them, decline to answer what it was
> never told, handle yes/no and negation, generate a word-ordered reply, bring
> up an on-topic associate, reason across several stored facts, and track a
> referent across turns (a later "it" resolves to the earlier subject) — runs
> as **one shared spiking network on a single update loop**, and converses on
> the word-meaning representations it **learned by listening** to a sentence
> stream (a few-hundred-word vocabulary, validated across multiple random
> seeds, with zero fabricated answers). Its defining feature is a
> **no-fabrication safeguard**: when nothing it has stored matches the
> question, it says it doesn't know instead of guessing.
>
> This is an active research project. The grounded who/what conversation is the
> robust core; the "discover categories," "fluent free-form reply," and "develop
> over days" capabilities below are newer and clearly marked research-stage. The
> small Tier-1 / Tier-2.1 word-binding demos near the bottom are the earlier
> foundation — kept because they are small, fast, and run on a laptop CPU.

---

## The easiest way to chat (start here)

The friendliest way to hold a conversation is the chat console
(`brain_chat_tui`). It loads a trained/developed brain, resolves pronouns
across turns, recalls what it knows, and declines to answer what it doesn't. It
can phrase replies fluently through a small language generator, or fall back to
a **GPU-free template renderer** so it runs anywhere:

```bash
# GPU-free smoke: a tiny CPU brain + the template renderer (no GPU, no model download)
SIM_BACKEND=numpy python -m research.runners.brain_chat_tui --stub-renderer --tiny-demo

# Talk to a saved "developed" brain, with fluent phrasing (needs a CUDA GPU + the local generator):
SIM_BACKEND=cupy python -m research.runners.brain_chat_tui --load <developed-brain-dir-or-codes.json>
```

In the chat loop, `/facts` lists what the brain knows, `/raw` toggles the
brain's own unvarnished neural rendering (no language model in the loop), and
`/quit` exits. Words like "you"/"your"/"I" map to the brain itself, so
"what are you?" resolves against the brain's own self-facts.

The **web console's Interact tab** wraps this same chat with a brain picker and
a renderer selector — see `docs/webapp-frontend-guide.md`. The fluent-phrasing
language generator is a deliberate, temporary scaffold: the brain decides *what*
is true and whether to answer, and the generator is never invoked when the brain
chooses to abstain, so the no-fabrication safeguard holds regardless of which
renderer is used.

---

## The grounded conversational agent

These demos run the real conversational agent (`BrainConversationalAgent` plus
the conversational binding system that combines concepts into facts and reads
them back). They want a CUDA GPU (`SIM_BACKEND=cupy`) for the full
fully-spiking path; the smaller pieces run on CPU under `SIM_BACKEND=numpy`.

### A. The whole conversation on one spiking brain (flagship)

The agent converses on the word representations the "cortex" learned by
listening to a sentence stream (a few-hundred-word vocabulary). It recalls every
stored fact, refuses to invent answers it was never told, handles yes/no,
generates a word-ordered description, and brings up an on-topic associate. It
defaults to the unified one-brain path (the whole who/what pipeline on one
persistent spiking network, fully spiking by default):

```bash
# Fully-spiking flagship (needs a CUDA GPU):
SIM_BACKEND=cupy python -m research.runners.consolidated_320_conversation_demo \
    --seeds 42 43 44 --readout neural

# CPU-portable / reference path (the numpy reference binding system):
SIM_BACKEND=numpy python -m research.runners.consolidated_320_conversation_demo \
    --seeds 42 --readout neural --composer rf
```

Per seed it checks: every stored fact is recalled; the brain declines on the
unstored set with **zero false accepts** (a single false accept is a
no-fabrication-safeguard breach and a hard stop); yes/no is correct; a known
agent gets a correctly-ordered description and an unknown one gets "I don't
know"; and the on-topic associate is relevant. The learned word-representation
caches (`research/findings/raw/_phaseB_stream_codes_320_*.npy`) ship in the
repo; a seed with no cache is skipped with a message.

`--composer onebrain` (the default) runs the unified one-brain binding system,
fully spiking. `--composer rf` is the numpy reference / CPU path.
`--no-spiking-cleanup` and `--no-integrated-loop` switch individual steps back
to their ordinary-code equivalents for comparison.

### B. Multi-turn conversation — pronouns and reasoning across turns

A persistent neural working-memory loop holds what was just talked about, so a
later "it" resolves to the right thing, and a multi-step reasoning chain carries
across turns:

```bash
python -m research.runners.multi_turn_conversation_demo --composer rf
# --composer onebrain runs it on the one spiking network (wants SIM_BACKEND=cupy)
```

It defaults to `--composer rf` (the reference / numpy-CPU path) so it runs
anywhere; pass `--composer onebrain` for the fully-spiking network.

### C. What a trained conversation looks like

```
> remember the dog chased the cat
  OK, I'll remember that.
> who chased the cat?
  The dog did.
> is the dog big?
  I don't know. I haven't been told that.
> the dog is big
  OK, I'll remember that.
> is the dog big?
  Yes, the dog is big.
```

The binding and unbinding that store and read back each fact are computed by
spiking neurons (not a lookup table), and the "I don't know" is the measured
**no-fabrication safeguard**: a clean confidence gap between what it knows and
what it does not. One honest boundary: an object with *two* attributes
("big red ball") is not yet reliable on the learned representations — a
documented limit, not a hidden one.

---

## Research-stage capabilities

These build on the grounded agent above and are newer. Treat them as active
research, not finished features.

### Discovering categories and inheriting properties

By observing co-occurrences — or by *seeing* objects through the visual front
end — the brain can **discover simple categories and taxonomies on its own**,
then **inherit** properties down the "is-a" chain, with exceptions. You can
teach it an is-a taxonomy and class properties in plain sentences, then ask
questions whose answers were never stated:

```bash
python -m research.runners._emerge29_inference_console --demo
```

A scripted teach-and-ask session looks like:

```
teach:  a robin is a bird ·  a bird is an animal
teach:  a bird can fly    ·  an animal breathes
ask:    can a robin fly?        -> Yes, a robin can fly.        (inherited, never told)
ask:    does a robin breathe?   -> Yes, a robin breathes.       (inherited two levels up)
ask:    can a robin swim?       -> I don't know whether a robin can swim.   (honest — not inherited)
ask:    can a zzz fly?          -> I don't know what a zzz is.   (unknown concept)
```

There is no separate inference engine — the inheritance emerges from the shared
representations plus the brain's next-state prediction, and the no-fabrication
safeguard still applies to anything it can't reach.

### Fluent, LLM-like conversation

A separate console lets you talk to the brain more freely — ask grounded
questions, use pronouns across turns, teach it new facts on the fly, and have it
decline on what it hasn't learned — with a **small, locally-trained language
generator** (far smaller than a typical large language model) supplying the
fluent phrasing:

```bash
SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl --showcase
SIM_BACKEND=numpy python -m research.runners._fluidconv_chat_repl \
    --script "what does the dog eat?|the wolf eats rabbit|what does the wolf eat?"
```

> **Note:** the fluent path uses a small language generator whose weights are
> **built locally and not checked into the repo** (they are regenerable). Without
> that generator present, use the GPU-free chat console at the top of this guide
> (`brain_chat_tui --stub-renderer`), which renders replies from a template
> instead. The brain does all the comprehension, knowledge, grounding, and the
> no-fabrication safeguard either way; the generator only shapes the surface
> wording, and is a temporary scaffold on the path to the brain producing its own
> words entirely in spikes.

### Development over days

The brain can live a simulated life: forage under a hunger drive, perceive and
remember objects it encounters, grow its vocabulary and factual knowledge day
over day **without catastrophically forgetting** older knowledge, and persist
across restarts. You can then load the brain at a given "day" and talk to it
about what it lived through. The multi-day loop is driven by `develop_run.py`
(and loaded for chat via `brain_chat_tui --load <bundle>`); this is a
longer-running research workflow, not a quick demo.

---

## The foundational word-binding demos (small, fast, CPU-portable)

Everything below is the earlier **tier ladder** that the grounded agent was
built on: bind direction words to motor pools, scale up to synonyms, and show
continual learning without forgetting. They are small enough to train on a
laptop CPU and are useful for understanding the substrate. They are *not* the
full conversational agent above.

### 1. Tier 1 chat demo (single seed, ~6 min on RTX 3090)

```bash
python -m research.runners.chat_demo \
    --seed 43 \
    --train-events 200 \
    --transcript-out research/findings/chat-demo-result.md
```

Trains the Tier 1 architecture on a 4-word vocabulary (north/east/south/west)
via embodied Hebbian co-firing (no backpropagation). Then runs 12 turns
(3 rounds × 4 words) and writes a transcript showing:
- the user's word,
- the brain's predicted direction,
- motor-pool spike counts (per cardinal direction),
- a confidence ratio.

**Expected accuracy:** ~33% mean ± 12% across 6 seeds (range 17–50%, validated
2026-05-07). Chance is 25%; 5/6 seeds beat or tied chance. This is a
substrate-understanding demo, not a benchmark of the conversational agent.
See `research/findings/2026-05-07-chat_demo-multi-seed.md`.

### 1c. Interactive Tier-1 chat REPL (single seed, ~6 min training + interactive)

```bash
# First time: train + save the bridge
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --train-events 200 \
    --save-bridge simulation_checkpoints_h5/chat_tier1_seed43.simstate.h5

# Future sessions: load the saved bridge (skips ~6 min training, starts in ~30 sec)
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --load-bridge simulation_checkpoints_h5/chat_tier1_seed43.simstate.h5

# Or with 8-word synonym vocab:
python -m research.runners.chat_repl --mode synonym --seed 42 \
    --train-events 400 \
    --save-bridge simulation_checkpoints_h5/chat_synonym_seed42.simstate.h5
```

Then type direction words at the `>` prompt:
```
> north
[OK] [TIER1 seed=43] sim hears 'north', activates motor_N (delta N+205, x2.1)
> up                  # in synonym mode only
[OK] [SYNONYM] sim hears 'up', activates motor_N (delta N+87, x1.7)
> what
[?] 'what' is not in vocab; tracking deltas anyway
> quit
[DONE] 3 turns total. In-vocab accuracy: 2/2 = 100%
```

This is a genuinely interactive REPL: you type and the brain responds. It uses
the same baseline-vs-driven delta methodology as the scripted chat demos.

**Checkpoint save/load:** use `--save-bridge` on the first run to persist the
trained network, then `--load-bridge` on later runs to skip the ~6 min
training. Note that saved checkpoints don't preserve firing thresholds /
short-term-plasticity / eligibility state, but for inference (REPL chat) the
weights are sufficient — the dynamic state self-recovers in a few timesteps.

### 1b. Tier 2.1 synonym chat demo (single seed, ~15–20 min on RTX 3090)

```bash
python -m research.runners.chat_synonym_demo \
    --seed 42 \
    --train-events 400 \
    --transcript-out research/findings/chat-synonym-demo-result.md
```

Trains the scaled-up Tier 2.1 architecture (larger language-input and motor
pools) on an 8-word synonym vocabulary: {north,up}, {east,right}, {south,down},
{west,left}. Type "north" OR "up" → the north motor pool activates.

Then runs 16 turns (2 rounds × 8 words), tracking primary words, synonym words,
and per-motor accuracy separately.

**Validated across 6 seeds (2026-05-06):** word→action aligned on 5/6 seeds,
action→word aligned on 6/6. Demonstrates capacity-driven binding: bigger motor
pools give the spike-timing rule enough room for functional sub-populations
(different synonyms activate different sub-pools).
See `research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md`.

### 2. Continual-learning (catastrophic forgetting) test (~25 min single seed)

```bash
python -m research.runners.continual_forgetting_eval \
    --seed 42 --phase-a-events 200 --phase-b-events 200 \
    --out-stats research/findings/raw/g11_bg/forgetting_smoke.json
```

Checks that learning new vocabulary (synonyms up/right/down/left) does NOT erase
the old vocabulary (north/east/south/west). 6-seed result: 5/6 retain ≥ 80%,
mean 103%.

### 3. Memory-consolidation test (~25 min single seed)

```bash
python -m research.runners.consolidation_trainer \
    --seed 42 --n-awake-events-per-word 100 --n-sleep-swr-events 100 \
    --consolidation-interval 4 \
    --out-stats research/findings/raw/g11_bg/consolidation.json
```

Checks that memory transfers from a hippocampus-style region into cortex via
sharp-wave-ripple sleep replay. 3-seed result: 3/3 retain ≥ 50% after the
hippocampus is silenced (mean 96%) — the complementary-learning-systems idea,
demonstrated empirically.

### 4. Sequence-cortex pretraining (research branch)

```bash
git checkout path-f-hybrid
python -m research.runners.cortex_pretraining \
    --task shakespeare --T 32 --hidden-layers 256,256 \
    --epochs 200 --batch-size 32 --lr 0.003 \
    --n-train-samples 2000 --seed 42 \
    --backend auto \
    --out-checkpoint research/findings/raw/path_f/shakespeare_pretrained.npz
```

Trains a 4-layer spiking network on Tiny Shakespeare via surrogate-gradient
learning (~11 min on RTX 3090; loss 12.18 → 1.016). This validates the
training infrastructure at toy scale. A follow-up test found that these
char-level features do **not** transfer to word-action binding at toy scale — an
informative negative result: much larger scale would be needed for transfer.

## What works vs what's limited (foundational demos)

### Validated across multiple seeds

| Capability | Result | Source |
|---|---|---|
| Tier 1 4-word binding | word→action 5/6, action→word 6/6 aligned | Tier 1 |
| Tier 2.1 8-word synonyms | word→action 5/6, action→word 6/6 aligned | Tier 2.1 |
| Continual learning | 5/6 retain, mean 103% | continual-forgetting test |
| Memory consolidation | 3/3 retain, mean 96% | consolidation test |
| 8-word + consolidation | 3/3, primary 91% / synonym 128% | 2026-05-08 |

### Documented limits

| Capability | Limit | Source |
|---|---|---|
| Visual + language binding | did not bind (parked) | early Tier 2.2 |
| Compositional two-word phrases | ~41% ceiling | Tier 2.3 sweep |
| Toy-scale pretrained transfer | below random | sequence-cortex test |

## How to read a chat-demo transcript

Example turn:
```
[OK] You: north  -> Sim: north  (N204 E189 S190 W131, confidence x1.1)
```

- `[OK]` / `[X]`: the prediction matches / doesn't match the user's word.
- `motor counts`: spikes during the stimulus window per cardinal direction.
- `confidence x1.1`: winner-to-runner-up ratio (1.0 = a tie; larger = more
  confident).

Confidence ratios of 1.0–1.2 are typical for the Tier-1 standard architecture —
it barely differentiates words, so predictions are weak but above chance. The
scaled Tier 2.1 architecture differentiates more strongly.

## Where the conversational arc stands now

The Tier-1 word-binding demo was the *first* conversational artifact (May 2026).
The arc since then built, on top of that substrate, the full grounded
conversational agent at the top of this guide. The conversational stack now:

- **parses** a sentence (word order × active/passive voice → who-did-what),
  with flexible word orders beyond plain subject-verb-object;
- **stores** who-did-what facts, attributes, and nested clauses, bound in spikes;
- **recalls** them on cue, and **declines** ("I don't know") on what it was never
  told — the measured no-fabrication safeguard;
- **handles yes/no and negation** ("is the dog big?");
- **generates** a word-ordered reply (order produced by spiking neurons);
- **plans a little dialogue** (brings up an on-topic associate);
- **reasons** across several facts (chaining stored who-did-what links);
- **tracks referents** across turns (a later "it" resolves to the right thing);
- **learns word meanings by listening** — a "cortex" that learns what a
  few-hundred everyday words mean purely from a sentence stream, then converses
  on those learned representations.

The whole loop runs as **one shared spiking network on a single update loop**,
fully spiking by default at the few-hundred-concept scale.

### Open research frontiers

- **Open-ended fluent generation** — moving beyond a bounded set of sentence
  forms toward free conversation produced by the brain's own circuitry.
- **Learned concept binding** — replacing today's hand-designed scheme for
  combining concepts into facts (a clean, exactly-invertible vector algebra,
  whose *operations* already run in spikes) with one the brain *learns*.
- **Two-attribute objects** ("big red ball") are still unreliable on the learned
  representations — a documented boundary with the specific fixes it would need
  written down.
- **Resolving ambiguous references** — deciding which of several things a bare
  pronoun means.

## Where to find more

- Conversational agent: `research/runners/brain_conversational_agent.py`,
  `research/runners/one_brain_composer.py`,
  `research/runners/rf_phasor_composer.py`
- Chat consoles: `research/runners/brain_chat_tui.py`,
  `research/runners/_fluidconv_chat_repl.py`,
  `research/runners/_emerge29_inference_console.py`
- Flagship demos: `research/runners/consolidated_320_conversation_demo.py`,
  `research/runners/multi_turn_conversation_demo.py`
- Web console: `docs/webapp-frontend-guide.md`
- Current state: [`docs/CURRENT-STATE.md`](CURRENT-STATE.md)
- History & milestones: [`CHANGELOG.md`](../CHANGELOG.md)
- Architecture details: `CLAUDE.md`
