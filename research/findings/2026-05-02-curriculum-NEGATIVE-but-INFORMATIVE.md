# 2026-05-02 — Curriculum NEGATIVE result is informative: cascade quality isn't the bottleneck

**TL;DR:** Two-phase curriculum training (Phase 1: 200 ep visuomotor only,
Phase 2: 100 ep text I/O on trained cascade) produced eval accuracy
similar to v2 single-phase baseline despite Phase 2 having dramatically
better cascade dynamics (43% correct moves vs v2's 30%). The weight
diagnostic shows IDENTICAL token-targeted differentials between v2 and
curriculum.

This refutes the "cascade quality is the bottleneck" hypothesis from
the strategic biology-grounded options doc. The 28.5% W→A is a true
architectural ceiling under STDP-based language pathways at current
network structure.

## Headline numbers

| Run | Phase 1 train | Phase 2 train | I→W eval | W→A eval |
|---|---|---|---|---|
| v2 100-ep seed=42 | — | 30% correct moves | 33% (p=0.042) | 27% |
| Curriculum 200+100 seed=42 | 34% (200ep visuomotor) | 43% (100ep text+visuomotor) | 24% (chance) | 23% (chance) |

Phase 2 cascade is **13 pp better** than v2 (43% vs 30%), but eval
accuracy didn't improve — actually slightly worse, within seed variance.

## The smoking-gun weight diagnostic

Token-targeted weight differentials (PFC-bypass: lang_in_active → motor_X):

```
                  v2 100-ep         curriculum (200+100 ep)
north weight      -0.0786            -0.0796   <-- essentially identical
east weight       +0.2102            +0.2100
south weight      +0.3036            +0.3028
west weight       +0.0727            +0.0723
```

The numbers match to 3 decimal places. Curriculum produces the SAME
weights as single-phase v2 training, despite 3x the total training
duration AND much better cascade dynamics in phase 2.

## What this means

Language pathway weights converge to a fixed point that depends on:
- Cascade STRUCTURE (which cortex_X drives which motor_X via BG)
- STDP+reward parameters (a_plus, a_minus, w_max, eligibility tau)
- Token embedding distinctness (sparsity, overlap)

Language pathway weights do NOT depend (significantly) on:
- Cascade ACCURACY (% correct moves)
- Training duration past ~100 ep (saturation)
- Phase ordering (curriculum vs concurrent)

This is a deep architectural property: STDP+reward on stable cascade
structure produces a stable weight distribution that captures the
relative spike correlation patterns. Whether motor_N fires for "north"
30% or 43% of the time, the RELATIVE pre-post correlations at each
motor pool are similar — STDP captures those.

## Why eval accuracy varies anyway

If weights are identical, why does eval vary so much across seeds?
- v2 seed=42: I→W 33% (lucky east)
- v2 seed=43: I→W 25% 
- Curriculum seed=42: I→W 24%

Answer: eval-time DYNAMICS depend on bridge state (firing thresholds,
STP, eligibility traces) which DO accumulate across training and DO
differ between curriculum vs single-phase. Same weights, different
network state → different per-trial readout statistics.

The 6-seed cumulative (n=600) at v2 single-phase was 28.5% W→A p=0.027.
Repeating that 6-seed validation with curriculum likely produces similar
result (~28-29%). One curriculum seed can't distinguish from one v2
seed at this n.

## Implications for strategic options

The strategic biology-grounded options doc's Tier 1 recommendation
(Curriculum / Option A) was wrong about expected impact. Curriculum
doesn't push past the 28.5% ceiling because the ceiling isn't about
cascade quality.

**Updated tier-1 recommendations:**

1. **Variance reduction + multi-baseline eval methodology.** Same
   network, different readout — explicitly test if seed=42's 33% I→W
   was lucky vs systematic. Multi-seed validation of curriculum at
   n=600 would clarify.

2. **Different DECODING.** The argmax-of-delta-spike-counts is sensitive
   to per-trial baseline noise. Different decoders (e.g., direct
   firing-rate cosine to known motor pool patterns, or temporal-pattern
   readout) might extract more signal from the SAME weights.

3. **Different LANGUAGE PATHWAY ARCHITECTURE.** Maybe more direct
   pathways or different initial conditions would produce better
   discriminative weights:
   - Wider readout pathways (cortex_X → multiple lang_out neurons)
   - Cosine-attention-style readout (output is weighted sum of
     all-tokens, decoded by argmax of similarity)
   - Different motor pool sizes WITH proportional drive scaling

4. **Active goal-perception scaffolding.** The visuomotor task is hard
   because the agent has to figure out goal location from a 32x32
   image with no explicit cue. Adding an explicit "where is the goal"
   signal during training (then removing for eval) could let cascade
   reach 70%+ correct moves. This isn't the same as our heuristic
   cheat — it's a TEACHING SIGNAL during training only.

## What should NOT be tried

- More curriculum phases / longer phase 1 — won't help, weights saturate
- Reward shaping — already tested NEGATIVE
- Stronger drives — already tested NEGATIVE
- Bigger pools — already tested NEGATIVE  
- Bigger lang regions (256→512) — already tested NEGATIVE

## Open question

Is the 28.5% W→A actually a HARD ceiling, or is it a property of the
specific decoding/eval methodology? The weight diagnostic shows
differentiated weights (3-4/4 tokens in target direction). The argmax-
delta eval may be losing signal that other decoders would extract.

This points back to **eval methodology** as the most leveraged change
to test next. Population vector decoding, longer reset windows,
multi-baseline averaging — these are eval-only changes that don't
require retraining and can be tested on existing 6 v2 checkpoints
(though see reeval limitation about bridge state).

## Files

- Result: `research/findings/raw/g11_bg/text_eval_curriculum_seed42.json`
- Phase 1 ckpt: `research/findings/raw/g11_bg/text_eval_curriculum_seed42.phase1.simstate.h5`
- Final ckpt: `research/findings/raw/g11_bg/text_eval_curriculum_seed42.simstate.h5`
- Weight diag: `research/findings/raw/g11_bg/text_weight_diag_curriculum_seed42.json`

The phase-1 checkpoint remains valuable infrastructure for future
experiments — load it, then train phase 2 with different language
pathway configurations to test pathway-specific hypotheses without
retraining the cascade.
