# Chat demo guide -- biology-grounded conversational sim

Quick guide to running and understanding the conversational
sim demos.

## What you can run today

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

## Roadmap

The current demo is the FIRST conversational artifact. Future
extensions:

### Near-term (1-2 weeks)
- Multi-seed demo (run 6 seeds, show variance)
- ~~Tier 2.1 synonym demo (8-word vocab)~~ — **shipped 2026-05-07** (`chat_synonym_demo`)
- ~~Continual-learning demo (train primary, then synonym, show retention)~~
  — **shipped 2026-05-07** (`chat_continual_demo`)
- Consolidation demo (sleep replay between training rounds)

### Medium-term (1-3 months)
- Interactive REPL mode (Python prompt, type words live)
- Larger Tier 2.1 vocab (16-30 words)
- Path F scale-up (10x params, word-level pretraining)

### Long-term
- Tier 3 dendritic learning for compositional binding
- Project Nord scale conversational LM (1B+ params)

## Where to find more

- Master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
- Validation findings: `research/findings/2026-05-07-*.md`
- Architecture details: `CLAUDE.md` (W->A history entries 1-12)
