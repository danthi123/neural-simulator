# Eval methodology is BROKEN — perfect weights give chance accuracy

**Date:** 2026-05-04 ~10:00 EDT
**Source:** `text_eval_sanity_check_*.json` (24 runs, 4 modes × 6 seeds)
**Predecessor:** `2026-05-04-biology-sweep-VERDICT-B.md`

---

## TL;DR

The B1 sanity check (auto-fired by the post-biology decision waiter,
re-run after fixing two bugs) found that **hand-built PERFECT
language→motor weights give 25.3% TRUE accuracy across 6 seeds — not
distinguishable from random or zero weights.**

This is the largest negative finding of the autonomous arc. **Every
prior W→A eval in this project has been measuring noise, not learning.**

| Condition | Mean TRUE | Mean best perm | Aligned/n |
|---|---|---|---|
| **perfect weights** (density 0.30, weight=8.0 on correct edges) | **25.3%** | 32.8% | **1/6** |
| **perfect weights** (density 1.0, full connectivity) | **25.2%** | 32.3% | **0/6** |
| **wrong weights** (rotated mapping, weight=8.0 on wrong edges) | 25.5% | 32.3% | 0/6 |
| **random weights** (U[0, 8.0] on all edges) | 24.8% | 32.2% | 0/6 |

All four conditions are statistically identical. The eval cannot tell
that 625 strong synapses (in density 1.0) are driving motor_correct
with weight 8.0 vs nothing.

## What this means

The 0/N alignment streak across the entire W→A investigation arc is
explained by this single fact: **the eval is signal-blind to
language→motor weight patterns.**

This rules out the following lines of investigation as moot until
the eval is fixed:
- Biology-grounded fixes (Pulvermuller topographic, Vogels PV-FSI)
- Cascade reintroduction (A2)
- Sparse-code orthogonality (B2)
- Long training dose (B4)
- Supervised gradient learning (B3)

ALL of these were going to be evaluated against this same eval. If
even hand-built perfect weights can't be detected, no learning rule
or architecture change matters.

## What's likely going wrong

The hand_build_perfect_weights with density=1.0 + weight=8.0 produces:
- 25 active language neurons (from sparse drive at sparsity=0.10)
- 25 motor neurons per pool, 4 pools = 100 total motor neurons
- Density 1.0 → every lang_active connects to every motor neuron
- Per-pool: 25 × 25 = 625 synapses. Correct pool weight=8.0; others weight=0.0
- Drive=200 pA on each lang_active neuron during stim window

That should produce overwhelming preferential drive to motor_correct.
Yet TRUE accuracy is 25.2%. Possible failure modes:

1. **Motor pool dynamics saturate**: 625 synapses × weight 8.0 may push
   motor_correct into refractory rapidly, capping its firing rate at
   the same level motor_other reaches via noise alone.

2. **OU noise dominates**: with `ou_std_current_pA = 80.0`, the noise
   on all 100 motor neurons may swamp the differential signal.

3. **Synaptic time constants are too fast**: the conductance decay may
   close the synaptic window before motor pools differentiate.

4. **Eval measurement window mismatch**: `evaluate_word_to_action`
   measures population firing rates over the stim window, but the
   network may not have reached steady-state in that window.

5. **Reset window doesn't actually reset**: trial N may carry residual
   activation from trial N-1 that contaminates the measurement.

6. **The eval's argmax over deltas vs baselines may be biased**: the
   delta calculation might subtract baseline rates that are NOT
   homogeneous across motor pools.

## How we found this

The post-biology decision chain (`wait_biology_then_decide.ps1`)
correctly fired B1 (eval_sanity_check) when the biology sweep gave
verdict B. The first B1 run had two bugs:
1. `hand_build_perfect_weights` had a `'mode': mode` summary entry that
   crashed perfect/wrong-mode runs with TypeError.
2. `result_aggregator --out FILE` only printed "Wrote X" to stdout
   (not the report), so the waiter's verdict regex never matched.

After fixing both bugs (commit `cfc9487`) and re-running B1, all 24
runs succeeded. The result above is the actual perfect-weights eval.

## Next steps

Before continuing the W→A arc, the eval mechanics need investigation.
Recommended diagnostic experiments:

1. **Drive sweep:** sweep `--lang-input-drive-pA` over [50, 200, 500,
   1000] with hand-built perfect weights. Does any drive level produce
   >50% TRUE accuracy?

2. **Stim window sweep:** sweep `--stim-steps-per-step` over [50, 100,
   200, 500]. Does longer measurement window improve SNR?

3. **OU noise sweep:** sweep `--ou-std-current-pA` over [0, 20, 80,
   200]. Does removing OU noise reveal the signal?

4. **Direct firing rate inspection:** instrument
   `evaluate_word_to_action` to print actual motor pool firing rates
   per word, not just argmax outcomes. Quantify whether motor_correct
   fires MORE than motor_other under perfect weights.

5. **Motor pool drive direct test:** bypass language_input entirely;
   inject current directly into motor_N for word "north", motor_E for
   "east", etc. Does THIS give clean alignment? Tests whether the issue
   is the synaptic propagation or the eval scoring itself.

If experiment 5 ALSO gives chance accuracy, the issue is in the eval
scoring (argmax / delta / baseline subtraction). If experiment 5 works,
the issue is in synaptic propagation through language_input → motor_X.

## Implication for the cheat-5 ON HOLD reframe

The 2026-04-28 reframe described cheat-5 (cross-projections) as "ON HOLD
pending biology buildout." This finding suggests the W→A eval has been
silently broken since the original baseline measurements. The 28.5%
"validated result" debunked by the 2026-05-03 permuted-label control was
the surface symptom; this is the underlying mechanism.

The cheat-5 navigation arc is on a different eval (Manhattan distance to
goal, robust mechanical metric) and is unaffected. Only the W→A arc is
blocked by this finding.

## Why the chain correctly stopped

The decision waiter parsed `b1Verdict = "eval_broken"` and exited with
"Eval methodology BROKEN -> stopping chain. Manual review needed." This
is exactly what the pre-staged chain was supposed to do — the chain
saved us from auto-firing B2 (sparse codes) on top of a broken eval,
which would have wasted ~2 hours of GPU time.

The chain also caught its own bugs (B1 first run had two failures →
manual fix → re-run → real result). That's the kind of resilience the
autonomous-runs skill is designed for.
