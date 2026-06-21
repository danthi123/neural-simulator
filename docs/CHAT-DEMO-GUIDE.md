# Chat demo guide -- biology-grounded conversational sim

A guide to running and understanding the conversational sim demos, from the
**production conversational agent** (the whole pipeline on one spiking brain)
down to the small portable word-binding demos and the foundational tier ladder.

**Last updated:** 2026-06-22.

> Where the project is now: the conversational pipeline — parse a sentence,
> store who-did-what facts, recall them, abstain on what it was never told,
> handle yes/no and negation, generate a word-ordered reply, plan dialogue,
> reason across several facts, and track referents across turns — runs as
> **one persistent interacting spiking loop on a single `SimulationBridge`**,
> and converses on the codes it **learned from conversation** (320 concepts,
> multi-seed, zero fabrications). The flagship demo runs fully spiking by
> default. The Tier-1 / Tier-2.1 word-binding demos below are the earlier
> foundation; they are kept because they are small, fast, and CPU-portable.

---

## The production conversational agent (start here)

These run the real production agent (`BrainConversationalAgent` +
`RFPhasorComposer` / `OneBrainComposer`). They want a CUDA GPU
(`SIM_BACKEND=cupy`) for the full fully-spiking path; the smaller pieces run on
CPU under `SIM_BACKEND=numpy`.

### A. The whole conversation on one spiking brain, at 320 concepts (flagship)

The agent converses on the **320 word-codes the cortex learned by listening**
to a sentence stream — recalls every fact, refuses to invent answers it was
never told, handles yes/no, generates a word-ordered description, and brings up
an on-topic associate. Defaults to `--composer onebrain` (the whole who/what
pipeline on **one persistent spiking bridge**, fully spiking by default):

```bash
# Fully-spiking flagship (needs a CUDA GPU):
SIM_BACKEND=cupy python -m research.runners.consolidated_320_conversation_demo \
    --seeds 42 43 44 --readout neural

# CPU-portable / test-oracle path (the numpy reference composer):
SIM_BACKEND=numpy python -m research.runners.consolidated_320_conversation_demo \
    --seeds 42 --readout neural --composer rf
```

Per seed it gates on: recall == 1.00 on every stored fact, abstain == 1.00 on
the unstored set (**zero false-accepts** — a single one is a moat breach and a
hard stop), yes/no correct, a correctly-ordered `describe()` for a known agent
and `None` for an unknown one, and an on-topic `elaborate()`. The stream-learned
code caches (`research/findings/raw/_phaseB_stream_codes_320_*.npy`) ship in the
repo; a seed with no cache is skipped with a message.

`--composer onebrain` runs the integrated one-brain composer (production
default, fully spiking); `--composer rf` is the numpy reference / test oracle
(also the CPU path). `--no-spiking-cleanup` and `--no-integrated-loop` switch
individual steps back to their host-oracle equivalents for comparison.

### B. Multi-turn conversation — pronouns and reasoning across turns

A persistent neural working-memory loop holds what was just talked about, so a
later "it" resolves to the right thing, and a multi-step reasoning chain is
carried across turns:

```bash
python -m research.runners.multi_turn_conversation_demo --composer rf
# --composer onebrain runs it on the one spiking bridge (wants SIM_BACKEND=cupy)
```

Defaults to `--composer rf` (the test-oracle / numpy-CPU path) so it runs
anywhere; pass `--composer onebrain` for the fully-spiking bridge.

### C. What a trained conversation looks like

```
> remember the dog is big
  OK, I'll remember dog is big.
> is the dog big?
  Yes, dog is big.
> is the apple small?
  I don't know. I haven't been told.
> who ate the apple?
  Dog did.
```

The binding and unbinding are computed by spiking neurons (not a lookup
table), and the "I don't know" is the measured **no-confabulation moat**: a
clean confidence gap between what it knows and what it does not. One honest
boundary: an object with *two* attributes ("big red ball") is not yet reliable
on the learned codes — a documented limit, not a hidden one.

---

## The foundational word-binding demos (small, fast, CPU-portable)

Everything below is the earlier **tier ladder** that the production agent was
built on: bind direction words to motor pools, scale to synonyms, and show
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

Trains Tier 1 architecture on 4-word vocab (north/east/south/west)
via embodied Hebbian co-firing (no backprop). Then runs 12 turns
(3 rounds x 4 words) and outputs a transcript showing:
- User's word
- Sim's predicted direction
- Motor pool spike counts (per cardinal)
- Confidence ratio

**Expected accuracy:** ~33% mean ± 12% std across 6 seeds (range
17-50%, validated 2026-05-07). Chance baseline is 25%. 5/6 seeds
beat or tied chance. Best seed (101) reaches 50%; outlier seed
(42) below chance at 17%. See
`research/findings/2026-05-07-chat_demo-multi-seed.md`.

### 1c. Interactive chat REPL (single seed, ~6 min training + interactive)

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

This is the master plan's "build conversational demo on Phase 1.4
architecture" milestone — a true interactive REPL where you type and
the sim responds. The implementation uses the same baseline-vs-driven
delta methodology as the scripted chat demos.

**Checkpoint save/load** (added 2026-05-08): use `--save-bridge`
on first run to persist the trained network state, then `--load-bridge`
on subsequent runs to skip the ~6 min training phase. This makes the
demo near-instant after the first invocation. Per CLAUDE.md gotcha:
`save_checkpoint` doesn't preserve firing thresholds / STP / eligibility,
but for inference (REPL chat), weights are sufficient — dynamic state
self-recovers in a few timesteps.

### 1b. Tier 2.1 synonym chat demo (single seed, ~15-20 min on RTX 3090)

```bash
python -m research.runners.chat_synonym_demo \
    --seed 42 \
    --train-events 400 \
    --transcript-out research/findings/chat-synonym-demo-result.md
```

Trains Tier 2.1 v4 scale-up arch (n_lang=4096, n_motor=1000,
n_motor_fs=120) on 8-word synonym vocab: {north,up}, {east,right},
{south,down}, {west,left}. Type "north" OR "up" -> motor_N activates.

Then runs 16 turns (2 rounds x 8 words), separately tracking:
- PRI (primary words: north, east, south, west)
- SYN (synonym words: up, right, down, left)
- Per-motor accuracy

**Validated 6-seed (2026-05-06):** W->A 5/6 aligned, A->W 6/6 aligned,
A->W mean 63.7%. Demonstrates capacity-driven binding: bigger motor
pools (1000 vs Tier 1's 500) give STDP enough room for functional
sub-populations within each motor_X (different synonyms activate
different sub-pops, no winner-take-all).

See `research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md`.

### 2. Phase 1.4 catastrophic forgetting test (~25 min single seed)

```bash
python -m research.runners.continual_forgetting_eval \
    --seed 42 --phase-a-events 200 --phase-b-events 200 \
    --out-stats research/findings/raw/g11_bg/forgetting_smoke.json
```

Validates that learning new vocab (synonyms up/right/down/left)
does NOT erase old vocab (north/east/south/west). 6-seed
result: 5/6 PASS at >= 80% retention, mean 103%. The
foundational Path F validation.

### 3. Phase 1.3 consolidation test (~25 min single seed)

```bash
python -m research.runners.consolidation_trainer \
    --seed 42 --n-awake-events-per-word 100 --n-sleep-swr-events 100 \
    --consolidation-interval 4 \
    --out-stats research/findings/raw/g11_bg/consolidation.json
```

Validates that memory transfers from hippocampus to cortex via
SWR sleep replay. 3-seed result: 3/3 PASS at >= 50% hippo-OFF
retention, mean 96%. McClelland 1995 / Buzsaki 2013
complementary learning systems theory empirically validated.

### 4. Phase 2.2 SNN cortex pretraining (path-f-hybrid branch)

```bash
git checkout path-f-hybrid
python -m research.runners.cortex_pretraining \
    --task shakespeare --T 32 --hidden-layers 256,256 \
    --epochs 200 --batch-size 32 --lr 0.003 \
    --n-train-samples 2000 --seed 42 \
    --backend auto \
    --out-checkpoint research/findings/raw/path_f/shakespeare_pretrained.npz
```

Trains a 4-layer SNN on Tiny Shakespeare via surrogate-grad BPTT.
~11 min on RTX 3090. Validated: loss 12.18 -> 1.016 (92%
reduction). Infrastructure works at toy scale.

**Note:** Phase 2.3a tested whether pretrained features transfer
to word-action binding. Result: NEGATIVE at toy scale (pretrained
22% < random 28% < baseline 33%). Char-level next-char features
don't carry word-action information. Project Nord scale (~1B
params) would be needed.

## What works vs what's limited

### Validated multi-seed (biology-grounded)

| Capability | Result | Source |
|---|---|---|
| Tier 1 4-word binding | 5/6 + 6/6 aligned | Tier 1 BREAKTHROUGH |
| Tier 2.1 8-word synonyms | 5/6 + 6/6 aligned | Tier 2.1 BREAKTHROUGH |
| Tier 2.1 12-word vocab | 5/6 + 6/6 aligned | Tier 2.1 BREAKTHROUGH |
| Continual learning | 5/6 PASS, 103% retention | Phase 1.4 BRANCH A |
| Memory consolidation | 3/3 PASS, 96% retention | Phase 1.3 |
| 8-word + consolidation | **3/3 GO**, pri 91% / syn 128% | 2026-05-08 |
| 8-word strict anti-cheat | **3/3** identical to non-strict | 2026-05-08 |
| 12-word + consolidation (default arch) | 2/3 GO PARTIAL | 2026-05-08 (capacity edge) |
| 12-word + consolidation (scaled arch) | seed 43 GO (pri 100%, syn 138%) | 2026-05-08 (multi-seed in flight) |

### Capacity scaling table (motor pool size vs vocab size)

| Vocab | Sub-pops/motor_X | Default arch (n_motor=1000) | Scaled arch (n_motor=2000) |
|---|---|---|---|
| 4-word | 1 | ✅ 5/6 + 6/6 | not needed |
| 8-word (synonyms) | 2 | ✅ 5/6 + 6/6 | also passes |
| 12-word (3 synonyms) | 3 | ⚠️ 2/3 PARTIAL (capacity edge) | ✅ seed 43 GO |
| 16-word (4 synonyms) | 4 | not tested (predicted FAIL) | available (not yet validated) |

**Key insight (Tier 2.1 BREAKTHROUGH 2026-05-06, capacity hypothesis
empirically confirmed 2026-05-08):** as vocab size grows, each motor_X
needs to differentiate more sub-populations. The motor pool capacity
must scale with vocab size for clean binding. n_motor=1000 supports up
to ~3 sub-pops (8-word fine, 12-word edge); n_motor=2000 supports
4+ sub-pops (12-word clean, 16-word in test).

### Documented limits

| Capability | Limit | Source |
|---|---|---|
| Visual+language binding | 0/6 | Phase 1.1 Tier 2.2 (parked) |
| Compositional 2-word | 41% mean ceiling | Tier 2.3 (sweep confirmed) |
| Pretrained transfer (toy) | < random | Phase 2.3a (informative negative) |

## How to interpret a chat demo transcript

Example turn:
```
[OK] You: north  -> Sim: north  (N204 E189 S190 W131, confidence x1.1)
```

- `[OK]` / `[X]`: prediction matches / doesn't match user's word
- `motor counts`: spikes during stim window per cardinal direction
- `confidence x1.1`: winner-to-runner-up ratio (1.0 = tie, larger = more confident)

Confidence ratios at 1.0-1.2 are TYPICAL for Tier 1 standard arch.
The architecture barely differentiates words; predictions are
weak but above-chance. This is consistent with Phase 1.4 BRANCH A
baseline of ~33% W->A.

For stronger differentiation, would need:
- Tier 2.1 v4 scale-up arch (n_lang=4096, n_motor=1000)
  -> validated to ~35-41% W->A
- More training events
- Better word encoding (current uses hash-based sparse codes)

## Where the conversational arc stands now

The Tier-1 word-binding demo above was the *first* conversational artifact (May
2026). The arc since then built, on top of that substrate, the full production
conversational agent at the top of this guide. As of mid-2026 the conversational
stack is comprehensively complete:

- **parse** a sentence (word order × voice → who-did-what), with flexible word
  orders beyond plain subject-verb-object;
- **store** who-did-what facts, attributes, and nested clauses, bound in spikes;
- **recall** them on cue, and **abstain** ("I don't know") on what it was never
  told — the measured no-confabulation moat;
- **negate / yes-no** ("is the dog big?");
- **generate** a word-ordered reply (order produced by spiking neurons);
- **plan dialogue** (bring up an on-topic associate);
- **reason** across several facts (multi-hop chaining);
- **track referents** across turns (a later "it" resolves to the right thing);
- **learn word meanings from conversation** — a "cortex" that learns what ~320
  everyday words mean purely by listening to a sentence stream, then converses
  on those learned codes.

The whole loop runs as **one persistent interacting spiking loop on a single
`SimulationBridge`**, fully spiking by default at the 320-concept scale.

### Open frontiers

- **Two-attribute objects** ("big red ball") are still unreliable on the
  learned codes — a documented boundary, with the specific fixes it would need
  written down.
- **The composer's binding is a principled idealization** (a clean
  exactly-invertible vector algebra); the binding *operations* run in spikes,
  but replacing the exact-inverse algebra with a fully *learned* cortical binder
  is the deferred final frontier (it needs the dendritic substrate).
- **Open-ended fluency.** The own-network text generator's foundation is proven
  (it provably learns real text structure), but it is not yet fluent and is far
  from a large language model.

## Where to find more

- Production agent: `research/runners/brain_conversational_agent.py`,
  `research/runners/one_brain_composer.py`,
  `research/runners/rf_phasor_composer.py`
- Flagship demos: `research/runners/consolidated_320_conversation_demo.py`,
  `research/runners/multi_turn_conversation_demo.py`
- Current state: [`docs/CURRENT-STATE.md`](CURRENT-STATE.md)
- History & milestones: [`CHANGELOG.md`](../CHANGELOG.md)
- Validation findings: `research/findings/` (chronological)
- Architecture details: `CLAUDE.md`
