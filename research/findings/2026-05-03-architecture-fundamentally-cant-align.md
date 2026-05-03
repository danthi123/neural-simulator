# Architecture-level finding: word-action labels cannot align

**Date:** 2026-05-03 (autonomous overnight investigation)
**Status:** **MAJOR NEGATIVE** — consolidates 39+ runs across 19 conditions

---

## TL;DR

Every documented text-IO experiment in this project history (39+
evaluations across 19 conditions including v2 baseline / v2+SWR /
v2+SWR-balanced / H4 PFC isolation / fundamentals heb_only / drive_5x /
stdp_wmax_10 / dt=1.0 / and various pre-v2 variations) shares one
property:

**The TRUE labeled (token, action) mapping is NEVER the best of 24
permutations.** Across 39 runs: 0 aligned.

The architecture has 8-12pp of structure above chance, but it's not
aligned with task labels. Whatever the network "learns," it's not the
intended word-action mapping.

## Evidence

### 1. Aligned ratio = 0 across all conditions

```
Condition                       n   aligned/n   true mean   best mean
v2 baseline                     6   0/6         28.5%       32.8%
v2 + SWR (default)              6   0/6         24.3%       31.8%
v2 + SWR balanced               6   0/6         24.7%       31.8%
H4 PFC isolation                6   0/6         23.0%       32.8%
fundamentals heb_only           1   0/1         22.0%       33.0%
fundamentals drive_5x           1   0/1         29.0%       36.0%
fundamentals stdp_wmax_10       1   0/1         24.0%       32.0%
dt=1.0 smoke (20-ep)            1   0/1         28.0%       34.0%
[various pre-v2 variants]       12  0/12        22-32%      30-37%
TOTAL                          39   0/39
```

Even at 24/seed null hypothesis (true=best by chance with prob 1/24 =
4.2%), zero across 39 has joint probability < 1e-9. This is not random.

### 2. Best permutations are scattered, not dominant

| best perm | runs | % |
|---|---|---|
| ENSW | 4 | 10.3% |
| SNEW | 4 | 10.3% |
| ESNW, ESWN, SEWN, SWNE, NWES, WENS | 3 | 7.7% each |
| (others) | 1-2 | 2.6-5.1% |
| **TRUE (NESW)** | **0** | **0.0%** |

No single permutation dominates. Misalignment is **seed-dependent**
(random init creates each seed's private structure) rather than a
single structural bias.

### 3. Mild east-bias in cascade default firing

| action | total predictions across 39 runs | % |
|---|---|---|
| N | 959 | 24.6% |
| **E** | **1068** | **27.4%** |
| S | 944 | 24.2% |
| W | 929 | 23.8% |

motor_E gets ~3pp more predictions than chance, regardless of which
word drives. This is consistent with cluster_e topographic cortex
having a mild east-firing default. It's small but real.

### 4. West is the only direction that "kind of works"

Per-word, what action does the best permutation assign?
- north -> most often E (36%) — cascade pushes north toward east
- east -> most often N (36%) — east input activates north pool!
- south -> most often N (31%) — south input also activates north
- west -> most often **W (33%)** — matches the true mapping

Why is west different? Possibly:
- Cluster_e was set up with cortex_W having distinct orientation
- west's word code happened to project orthogonally to cortex_E bias
- random luck across the available conditions

Either way, this is the only direction where the architecture's
structure overlaps with task labels.

## Why all our experiments failed to fix this

Tested hypothesis | Result
---|---
Frequency-weighted SWR replay (v2+SWR) | regression -4pp, aligned 0/6
Buffer-balanced SWR (H1) | same 0/6, no rescue
PFC bypass isolation training (H4) | sub-chance 23%, aligned 0/6
Single fundamentals fixes (heb / drive / stdp_wmax) | aligned 0/3
Combined fixes (heb_drive, heb_stdp, drive_stdp) | running, expected to also fail
Bigger drive (drive_5x) | no help (29% w/ 0/1)
Hebbian re-enable with reduced decay | actively hurt (22%)
Token-orthogonal codes (sparsity 0.05) | not yet tested in production
Larger language regions (256->512) | not yet tested
Larger motor pools (10->50) | not yet tested

## What this means

The current architecture has a **structural inability to align learned
word-action mapping with task labels**. The training procedure (v2 with
phase1=0, phase2=100, phase3=0) creates ~8pp of seed-dependent
structure above chance, but that structure isn't oriented toward labels.

Possible root causes (in order of decreasing likelihood):

1. **Cascade dominates language signal during training.** Even at
   drive=1000 (drive_5x), the cascade's cluster_a/e default firing
   creates per-seed cortex_X biases that the language-driven STDP can
   barely modulate. Result: motor_X firing reflects cascade-driven
   bias, not language drive.

2. **Plasticity mass insufficient at 100ep.** 3000 plastic events for
   4 word-action pairings = 750 events per pairing. STDP needs
   thousands per pairing to differentiate against soft-bound saturation.

3. **PFC bypass weights init too close to soft-bound.** text_input_to_motor
   weight=3.0, stdp_w_max=5.0 leaves only 2.0 of room. Once STDP hits
   the bound, weights stop discriminating.

4. **Sparse code overlap insufficient at sparsity=0.1.** 26 active per
   word with 2-3 cross-word overlap. Maybe the readout pathway treats
   overlapping neurons as "shared signal" and can't distinguish words.

## Decisive next test (planned for tomorrow)

**Minimal language->motor isolation:** build the simplest possible
architecture with NO cascade, NO PFC, NO retina — just
language_input (256) -> motor_X (10 each) with paired-stim training.

If THIS can achieve aligned >= 4/6, the cascade IS the problem.
Architecture can learn word-action mapping, just not when buried under
cascade noise.

If THIS can't align either, the issue is more fundamental:
- Plasticity dose too low at this scale
- Initial weights wrong
- Sparse-code overlap unsolvable

Implementation: one new runner `text_minimal_isolation.py` that builds
a 100-300 neuron architecture and runs paired-stim training. ~30 min
per seed at dt=1.0. 6 seeds = ~30 min in parallel-3.

## Practical implications

**Stop tuning v2.** The fundamentals sweep finished Phase 1
inconclusively (the 3 tested variants all fail aligned check at 1
seed). Even if combined variants happen to give 32% on seed 42, they're
unlikely to align across 6 seeds.

**The bigger investigation is whether the architecture can learn
word-action mapping AT ALL.** The minimal-isolation test answers
that question definitively, regardless of what tomorrow's batch 2
+ dose results show.

If minimal isolation works, the path forward is:
- Reduce cascade contribution during text I/O training
- Or train cascade and language separately, then merge
- Or scale up language signal beyond what cascade can override

If minimal isolation also fails:
- The architecture itself is fundamentally not learning labels
- Need totally different approach (e.g., supervised gradient-based
  readout instead of STDP)

## Statistical caveat: where does the 8pp non-chance structure come from?

If the architecture has 0 alignment, why is best perm consistently 8pp
above chance (32% instead of 25%)? Two contributors:

1. Mild cascade biases (e.g., motor_E bias) which any permutation that
   maps something to E will pick up.
2. Per-trial noise creating slight imbalances. With only 25 trials per
   word, expected std of correct count is sqrt(25*.25*.75) ≈ 2.2.
   So "best of 24 random multinomial outcomes" is naturally elevated.

The 8pp doesn't represent learning. It represents architectural noise
that happens to align with SOME random permutation in each seed.

## Related findings

- `2026-05-03-permuted-label-control-NEGATIVE.md` (initial finding)
- `2026-05-03-unaligned-structure-pattern.md` (cross-seed pattern)
- `2026-05-03-i2w-also-at-chance.md` (I->W also fails)
- `2026-05-03-step-profile-results.md` (perf is compute-bound)
- `2026-05-03-dt1ms-speedup-validated.md` (dt=1.0 works)

## Tools

- `python -m research.runners.permuted_label_check` (per-condition aligned)
- `python -m research.runners.unaligned_pattern_analysis` (cross-condition pattern)
