# P5 iter KK NEGATIVE — Tier 1 cortical canon amplifies structural bias at biological scale

**Date:** 2026-05-12
**Status:** NEGATIVE. Single-seed smoke (seed 42) regressed below iter
AA's pool_readout PASS. Tier 1 cortical canon (internal_density=0.10,
exc=2.0, inh=4.0) applied to wernicke_pool + lang_output_pool causes
pools to self-sustain and amplifies seed-specific structural bias
instead of averaging it out.

## User directive (2026-05-12 07:30 EDT)

> "I don't know why we keep testing at toy scale if larger scale (that
> still fits locally) is clearly needed? And also you have my permission
> to autonomously do arch work to continue working towards conversational
> capabilities. Just keep in mind the reference catalog and the goal of
> staying biologically accurate, no cheats."

## Hypothesis tested

iter AA's 4/6 ceiling at toy scale (5K neurons, 100-neuron pools)
was diagnosed (in retrospect) as having wernicke_pool / lang_output_pool
internal dynamics ~6x WEAKER than Tier 1 motor pools (which achieved
6/6 on direction-word binding). Specifically:

| Variable | iter AA | Tier 1 motor | Ratio |
|---|---|---|---|
| internal_density | 0.05 | 0.10 | 2x |
| exc_weight_mean | 0.3 | 2.0 | 6.7x |
| inh_weight_mean | 0.8 | 4.0 | 5x |

**Hypothesis:** Apply Tier 1 cortical canon to wernicke_pool +
lang_output_pool at biological scale → 6/6 BIDIRECTIONAL recognition.

## Configuration

```bash
python -m research.runners.validate_ventral_semantic --seed 42 \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 \
    --apply-wernicke-topographic
# wernicke_pool + lang_output_pool internal: 0.10 / 2.0 / 4.0 (canon)
```

Architecture: 8636 neurons total, 2.35M synapses. Build 3.6s, total 410s.

## Result

| Test | iter AA s42 (toy weak) | iter KK s42 (bio + canon) |
|---|---|---|
| apple p0 spikes | 92 | 236 |
| apple p1 spikes | 85 | 254 |
| **apple winner** | **p0 OK** | **p1 WRONG** |
| river p0 spikes | 80 | 242 |
| river p1 spikes | 111 | 259 |
| **river winner** | **p1 OK** | **p1 OK** |
| **BIDIR** | **YES** | **NO** |
| Comprehension apple_self | 0.50? | 0.214 (FAIL) |
| Comprehension apple_river | 0.40? | 0.226 |
| Naming ratio | 1.30+ | 1.43x |

**Pool firing exploded 2.5-3x** (cortical canon doing its job — pools
fire vigorously). **BUT discrimination collapsed**: both pools fire
similarly, with pool_1 marginally dominant for BOTH stimuli.

## Diagnosis: "saturation + amplification of structural bias"

Strong recurrent excitation makes pools self-sustain once they start
firing. After the 3-hop chain
(lang_input → wernicke → semantic_cortex → lang_output_pool), the
input signal is weaker than internal recurrence. Pools then amplify
whatever random structural bias the seed has — for seed 42 at
biological scale, pool_1 has slightly more recurrent connections,
so it dominates BOTH apple and river responses.

iter AA seed 101 had identical phenomenology at toy scale: pool_0
dominance from random connectivity. We thought scaling would average
out the bias (sqrt(N) reduction in structural variance). Instead,
**stronger dynamics amplify the bias** even after scaling.

The biology: Tier 1 motor pool canon worked because:
1. Single hop (lang_input → motor_X direct)
2. Strong cross-pool FS that effectively WTAs at output
3. Per-trial transient firing (motor pools don't need to sustain)

P5 with canon has:
1. 3-hop chain — input signal diluted
2. Cross-pool FS only at wernicke layer, NOT at lang_output
3. Pools should sustain (Wernicke holds concept) but lang_output
   should be transient → mismatch

## Comparison

| Iter | Scale | Internal dyn | apple p0/p1 | BIDIR seed 42 |
|---|---|---|---|---|
| AA | 5K (100/200 pools) | weak (0.05/0.3/0.8) | 92 / 85 | YES |
| **KK** | **8.6K (500/500 pools)** | **canon (0.10/2.0/4.0)** | **236 / 254** | **NO** |

Iter KK has 1.7x more neurons but 2.7x worse apple discrimination
(margin +7 → -18 — REVERSED winner).

## Lesson learned

**Cortical canon is not universally portable across architectures.**
Tier 1 motor binding works with canon because:
- Short chain (lang_input → motor)
- Output-level FS WTA
- Brief per-trial firing

P5 ventral semantic has a longer chain + needs sustained attractor
at the wernicke level. Just applying Tier 1's recurrent strength
without re-engineering the WTA and the chain length breaks the
balance.

## Next iteration: iter LL (scale only)

Reverted the canon change (parameterized via CLI; defaults back to
iter AA weak). Running iter LL = biological scale WITHOUT canon to
isolate the user's "scale is what's needed" hypothesis. Same recipe
as iter AA but at 5x pool size + 2x lang_input.

If iter LL passes 6/6: scale alone is sufficient (user was right).
If iter LL fails: the architecture has a deeper issue than scale or
canon, and we need to redesign (probably: shorter chain via direct
wernicke→lang_output binding, dropping semantic_cortex from recognition).

## Code

Parameterized in commit `9dad1ef` — 6 new CLI flags:
- `--wernicke-pool-internal-density` (default 0.05)
- `--wernicke-pool-exc-weight` (default 0.3)
- `--wernicke-pool-inh-weight` (default 0.8)
- `--lang-output-pool-internal-density` (default 0.05)
- `--lang-output-pool-exc-weight` (default 0.3)
- `--lang-output-pool-inh-weight` (default 0.8)

Cortical canon = `--wernicke-pool-internal-density 0.10
--wernicke-pool-exc-weight 2.0 --wernicke-pool-inh-weight 4.0
--lang-output-pool-internal-density 0.10
--lang-output-pool-exc-weight 2.0 --lang-output-pool-inh-weight 4.0`
(no longer default; available for future experimentation).
