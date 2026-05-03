# Permuted-label control — NO REAL LEARNING in text I/O system

**Date:** 2026-05-03 ~08:10 EDT (autonomous overnight)
**Status:** **MAJOR NEGATIVE FINDING** — recontextualizes all prior text I/O results

---

## TL;DR

The 28.5% W->A baseline is NOT real word-action learning. We tested
this by computing accuracy under all 24 permutations of the (token ->
action) mapping for every existing eval. The TRUE mapping is the BEST
permutation in **0 of 16 seeds** across 3 conditions.

If the network had learned real word-action mapping, the TRUE mapping
should be the BEST. Instead, every seed has a DIFFERENT permutation
that scores 3-13pp higher than the true labels.

The architecture has ~8pp of "structure" above chance — but it's
randomly oriented per-seed, not aligned with task labels.

## The control test

For each existing W->A eval JSON (n=16 across baseline / v2+SWR / H4):

1. Take the confusion matrix `cm[word][action]`.
2. For all 24 permutations of `(N, E, S, W)`, compute the accuracy
   `acc(perm) = sum(cm[word_i][perm_i] for word_i, perm_i in zip(WORDS, perm)) / total_trials`.
3. Find the BEST permutation and its accuracy.
4. Compare to the TRUE permutation `(north->N, east->E, south->S, west->W)`.

If learning is real, true mapping should equal best mapping. If learning
is illusory, true mapping is just one of 24 random permutations.

## Results

| Condition | seeds | true mean | best mean | excess (best - true) | true=best (aligned) |
|---|---|---|---|---|---|
| v2 baseline | 6 | 28.5% | **32.8%** | +4.3pp | **0/6** |
| v2 + SWR | 6 | 24.3% | **31.8%** | +7.5pp | **0/6** |
| H4 isolation | 4 | 23.0% | **32.8%** | +9.8pp | **0/4** |

**Aligned 0 of 16 across all conditions.**

If the network had learned real word-action mapping, we'd see aligned ~6/6
(true = best for every seed). We see 0/16. This means the architecture
NEVER produces the labeled mapping as its strongest signal.

## Per-seed best permutations

For seed 42 v2 baseline (best perm = `('E', 'N', 'S', 'W')`, acc 35%):
- The network learned: north -> E, east -> N (these two swap), south -> S
  (correct), west -> W (correct).
- The architecture's actual word-pattern -> motor pool mapping has
  N and E swapped relative to the task labels.

For seed 100 v2 baseline (best perm = `('W', 'E', 'S', 'N')`, acc 33%):
- Network learned: north -> W, east -> E (correct), south -> S
  (correct), west -> N. North and west swap.

These permutations are seed-dependent and arbitrary. The architecture's
"learning" depends on initial random weights, not on training-data
labels.

## What this means

The 28.5% W->A baseline (and the 24.3% v2+SWR regression) reflect
architectural noise, not task learning:
- Per-seed cascade biases produce ~8pp of structure above chance
- That structure isn't aligned with task labels (random permutation
  of which-word-fires-which-motor)
- We've been celebrating a noisy aggregate around a true mapping that
  the network ignores

The architecture currently does NOT learn word-action mapping. The
v2 baseline isn't learning poorly — it's not learning at all.

## What this means for tonight

The v2 vs v2+SWR vs H1 comparison is essentially "how do different
training procedures affect the random architectural noise?" The
"regression" with SWR isn't degrading real learning — it's just
shifting the noise.

The arch sweep (auto-launches after H1) might find a variant where:
1. The architecture's per-seed structure is more pronounced (best perm
   gives 40-50% instead of 33%)
2. AND the structure is more often aligned with labels (more aligned/n
   ratio)

If a variant gives e.g. best=42% and aligned=4/6, that would be REAL
learning emerging. If all variants give best ~32% with aligned=0/6,
the architecture is fundamentally incapable of word-action mapping
under current training regimes.

## What this means for tomorrow

Top priority: figure out why the architecture has unaligned structure.
Hypotheses:
1. **Cascade biases dominate before training** — The cluster_a/e
   pathways create per-seed cortex_X firing biases. When language is
   driven, the cascade firing dominates over language-driven activity,
   so motor pools fire based on cascade state, not language input.
   Test: increase language drive 5-10x relative to cascade drive.
2. **STDP doesn't differentiate words enough in 100 ep** — More
   episodes might bring true alignment. Needs a longer-training run.
3. **Soft-bound STDP at stdp_w_max=5 is too restrictive** — Test
   with stdp_w_max=10 to allow more weight differentiation.
4. **Word embedding dimensionality too low** — 256 dims with sparsity
   0.1 = 26 active. Test with 1024-dim embeddings.

Priority 1 and 3 are quick CLI tests; 2 and 4 are longer.

## Files

- `research/findings/raw/g11_bg/text_eval_*_seed*.json` — confusion
  matrices for all conditions
- This analysis is fully reproducible from JSON files alone — no GPU
  rerun needed.

## Code (reproducer)

```python
import json, itertools, statistics
WORDS = ["north", "east", "south", "west"]
TRUE_MAP = {"north": "N", "east": "E", "south": "S", "west": "W"}

def acc_for_mapping(cm, mapping):
    correct = total = 0
    for word, row in cm.items():
        target = mapping[word]
        for action, count in row.items():
            total += count
            if action == target:
                correct += count
    return correct / max(total, 1)

# Per file:
cm = json.load(open("path/to/eval.json"))["word_to_action_eval"]["confusion_matrix"]
true_acc = acc_for_mapping(cm, TRUE_MAP)
best_acc = max(acc_for_mapping(cm, dict(zip(WORDS, perm)))
               for perm in itertools.permutations(["N", "E", "S", "W"]))
print(f"true={true_acc:.3f} best={best_acc:.3f} excess={best_acc-true_acc:.3f}")
```
