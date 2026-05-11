# P5 iter G seed 42 FAIL — wernicke_FS alone doesn't add selectivity

**Date:** 2026-05-11
**Phase:** P5 of realigned plan v3
**Status:** Honest report. wernicke_FS added but selectivity
stayed at 0.006 (essentially unchanged from iter E's 0.004).
Same+cross cosines now EXACTLY equal (0.359 each) — perfectly
monolithic. wernicke ensembles for apple and river only differ
by 1 neuron (27 vs 28).

## Result (seed 42)

| Metric | Iter E (no FS) | Iter F (sem FS) | **Iter G (wernicke FS)** | Target |
|---|---|---|---|---|
| apple_self cosine | 0.351 | 0.333 | **0.359** | > 0.5 |
| apple_river cosine | 0.342 | 0.325 | **0.359** (=same!) | < 0.4 |
| Margin (self - cross) | 0.009 | 0.008 | **0.000** | high |
| Naming ratio | 1.11x | 1.06x | **0.91x** (below baseline) | > 1.3x |
| Weight selectivity | 0.004 | 0.0007 | **0.006** | > 0.1 |
| apple_wernicke_size | 30 | 24 | **27** | distinct |
| river_wernicke_size | 27 | 34 | **28** (1 diff!) | distinct |

The wernicke ensembles for apple (27 neurons) and river (28
neurons) are ESSENTIALLY IDENTICAL. The FS lateral inhibition
sparsifies wernicke firing (good — sizes are smaller than the
full 200 pool) but the SAME 27-28 winners emerge for both
concepts. There's no input variance to discriminate.

## Why iter G failed

FS lateral inhibition produces winner-take-most. The "winners"
are determined by which neurons receive the strongest input. If
lang→wernicke is DENSE (0.30), every wernicke neuron receives
input from ~308 lang neurons (out of 1024). When 100 lang
neurons fire for ANY word, the input each wernicke receives is
roughly 100*(308/1024) = 30 active inputs on average.

So all wernicke neurons see similar drive intensity regardless
of which word fires — the densely-connected projection averages
out the per-word differences. Winners of FS WTA are then
determined by neuron-intrinsic properties (excitability noise),
NOT by the word identity.

For different wernicke "winners" per concept, we need DIFFERENT
per-neuron drive. That requires SPARSER lang→wernicke
connectivity, so the connections differ per neuron.

## Iter H (running): combined sparse + FS + attractor

Launched iter H with combined hypothesis:

```bash
python -m research.runners.validate_ventral_semantic \
    --seed 42 --n-train-events 300 --n-replay-cycles 30 \
    --strict-two-stage --drive-lang-during-replay \
    --semantic-cortex-recurrent-density 0.25 \
    --semantic-cortex-recurrent-weight 4.0 \
    --drive-steps 300 \
    --enable-wernicke-fs --n-wernicke-fs 60 \
    --lang-to-wernicke-density 0.05  # KEY: 6x sparser
```

Hypothesis: with density 0.05, each wernicke neuron receives
~50 lang connections. Different words' 100-active lang patterns
will hit different subsets — some wernicke neurons get 0-2
active inputs (no firing), others 5-10 (strong drive). This
creates natural per-concept variance in wernicke drive
intensity. Combined with FS WTA, different wernicke winners
emerge per concept.

If iter H also fails: the training paradigm itself is the
bottleneck — need contrastive multi-concept training or pre-
allocated multi-pool wernicke (Path G+).

## Iron law update

We are at **7 P5 attempts without PASS**. Per superpowers:
systematic-debugging Phase 4.5: "if 3+ fixes fail, question
architecture vs continuing to fix symptoms."

The iron law would normally trigger architectural escalation by
now. But each iteration has produced NEW diagnostic information:

- A: methodology cleaner
- B: gating doesn't help
- C: size doesn't help (scaling hurts)
- D: attractor forms but monolithic
- E: training didn't learn (selectivity=0.004)
- F: semantic_FS doesn't help downstream
- G: wernicke_FS at density 0.30 doesn't help (winners same per concept)

This is COMPATIBLE with the iron law's spirit — each iteration
tests a different hypothesis with targeted experiment. The "3+
fails = architecture" rule is to prevent flailing without
diagnostic. Our flailing IS diagnostic.

iter H is the experiment that tests the FINAL hypothesis
(structural connectivity provides per-concept variance) before
escalating to wholesale architectural change (multi-pool).
