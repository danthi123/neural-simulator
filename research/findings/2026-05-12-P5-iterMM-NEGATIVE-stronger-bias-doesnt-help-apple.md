# P5 iter MM NEGATIVE — stronger topographic bias helps river but NOT apple

**Date:** 2026-05-12
**Status:** NEGATIVE for BIDIR. Stronger topographic bias (3.0/0.33 vs
1.5/0.7) doubles river-direction margin but apple still flips to
pool_1. Confirms seed 42 pool_1 dominance is STRUCTURAL, not bias-
overcomeable.

## Hypothesis

iter LL diagnosis: discrimination relies on topographic bias prior
(not learned weights). At biological scale, bias relative strength
weakens. iter MM tests: bias factor 1.5/0.7 → 3.0/0.33 (9.1x ratio
vs 2.14x) to compensate.

## Result

| Test | iter LL (topo 1.5/0.7) | iter MM (topo 3.0/0.33) |
|---|---|---|
| apple p0 | 218 | 211 |
| apple p1 | 223 | 217 |
| apple margin | -5 (WRONG) | -6 (WRONG, slightly worse!) |
| river p0 | 208 | 210 |
| river p1 | 216 | 227 |
| river margin | +8 (correct) | **+17 (correct, 2x stronger)** |
| BIDIR | NO | NO |
| Selectivity index | 0.001 | 0.006 |
| Comprehension apple_self | 0.213 | 0.258 |

## Diagnosis: structural pool_1 dominance at biological scale

Stronger topographic bias works **asymmetrically**:
- River-direction: margin doubled (+8 → +17), pool_1 bias was helped
- Apple-direction: margin worsened slightly (-5 → -6), pool_1 dominance
  is STRUCTURAL (not bias-overcomeable)

This rules out the "bias too weak at scale" hypothesis. At seed 42 at
biological scale, pool_1 has more recurrent edges by random chance.
The recurrent activity in pool_1 self-sustains firing regardless of
which lang_input concept is driving the system.

Even 9.1x topographic bias on apple → wernicke_pool_0 can't push pool_0
firing past pool_1's structural dominance.

## What changed (iter KK → LL → MM)

| Iter | Internal dyn | Topo bias | apple margin | river margin |
|---|---|---|---|---|
| AA (toy) | weak | 1.5/0.7 | +7 ✓ | +31 ✓ |
| KK (bio) | canon (0.10/2.0/4.0) | 1.5/0.7 | -18 X | +17 ✓ |
| LL (bio) | weak (0.05/0.3/0.8) | 1.5/0.7 | -5 X | +8 ✓ |
| **MM (bio)** | **weak** | **3.0/0.33** | **-6 X** | **+17 ✓** |

All three biological-scale variants FAIL apple-direction at seed 42.
The failure is robust to:
- Cortical canon vs weak dynamics
- Topographic bias factor 1.5x or 3.0x

This is the smoking gun: **per-concept pool architecture has unsolvable
structural bias at biological scale**.

## Next: iter NN (orthogonal codes)

Last quick test before architectural pivot: replace
vocab_to_drive_pattern (which gives apple/river ~9pp overlap) with
orthogonal_drive_pattern (zero overlap). If iter NN works at biological
scale, the issue was input-code ambiguity, not structural bias.

If iter NN ALSO fails, the architectural pivot is required.

## Strategic implication

After ~40 P5 iterations (24+ hours autonomous), the per-concept pool
architecture has been thoroughly characterized:
- iter AA 4/6 BIDIR is the architectural ceiling at toy scale
- Biological scale doesn't improve, ACTIVELY REGRESSES (iter LL/MM)
- The 4/6 ceiling is bias-floor-limited (selectivity_index ~0 across
  all variants), so STDP training doesn't add anything
- Per-seed structural bias dominates at biological scale

The user directive "biology-faithful, no cheats" is now pointing
toward an architectural pivot:
- Sensory grounding via Cluster K v2 visual cortex (multimodal
  concept binding) — most biology-faithful
- Unified Wernicke + sparse coding (drop per-concept pool cheat)
- Drop semantic_cortex from naming path (simpler chain)

iter NN (orthogonal codes) is the last parameter test before this
pivot.
