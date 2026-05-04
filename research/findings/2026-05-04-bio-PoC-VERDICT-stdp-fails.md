# Bio-scale STDP proof-of-concept — VERDICT: stdp_fails_at_bio

**Date:** 2026-05-04 18:43 EDT
**Chain:** Stage 2 of 2026-05-04 biological-scale test plan
**Source:** `research/findings/raw/g11_bg/text_eval_bio_*_seed*.json` (12 runs)
**Aggregated to:** `research/findings/2026-05-04-bio-proof-of-concept-results.md`

---

## TL;DR

Cortical canon (recurrence + E/I balance + NMDA bistability + N=500
motor pool) does NOT unlock STDP+R-STDP learning of the W→A mapping
at biological scale. Adding the biology fix (Pulvermüller topographic
prior + Vogels PV-FSI lateral inhibition) provides a real but small
benefit (+7.3pp mean TRUE accuracy) — not enough to clear the
"learning achieved" threshold.

| Condition | Aligned | Mean TRUE | Mean best perm |
|---|---|---|---|
| bio_baseline (canon, no fix) | 0/6 | 23.5% | 32.8% |
| bio_topo_fs (canon + biology) | 1/6 | 30.8% | 34.2% |

The single aligned seed (44) had TRUE accuracy 37% with best_perm =
NESW = TRUE. Consistent with prior pattern of one-seed-per-condition
random alignment from lucky initialization, NOT systematic learning.

**Orchestrator auto-fired bio_b3_gradient (PID 23940)** to test the
plasticity-rule-bottleneck hypothesis: if supervised gradient learning
can find the right weights at biological scale, STDP is the bottleneck;
if even gradient fails, the architecture itself is limited.

## What this rules in / rules out

**Rules out:**
- "Eval is fundamentally broken" — bio_sanity_check at perfect weights
  showed eval works at biological scale (4/6+ aligned both densities).
- "Architecture too small" — bio scale has 4288 neurons, ~1.5M synapses,
  cortical canon enabled. Same architecture supports perfect-weight
  alignment at sanity check.
- "Biology fix is irrelevant" — topo_fs gives consistent +7pp TRUE
  improvement vs baseline. Real signal, just small.

**Rules in:**
- Plasticity rule (STDP+R-STDP) is the dominant bottleneck. Cannot
  reliably differentiate sparse codes under paired-stim training,
  even with cortical canon + biology priors.
- Possible secondary issue: training dose (1000 events/dir × 4 dirs =
  4000 total events). Real cortical critical periods see 10⁵+ paired
  presentations. We may simply need more.
- Possible secondary issue: sparse code overlap at sparsity=0.1.
  Real human Wernicke-area concept neurons may be MORE orthogonal
  than our hash-derived patterns.

## Comparison with prior findings

This narrows the 2026-05-04 morning "eval broken" → "STDP fails" pivot:
- 2026-05-03 evening: cascade-as-cause hypothesis FALSIFIED (minimal-iso
  INVERSION finding, mean 16.7% TRUE).
- 2026-05-04 ~10:00: minimal-arch sanity check showed perfect weights
  give 25% TRUE. Initially read as "eval broken."
- 2026-05-04 13:16: bio sanity check showed perfect weights give 67%+
  TRUE at biological scale. "Eval broken" was actually "arch too minimal."
- 2026-05-04 18:43: PoC at bio scale shows STDP fails to LEARN. Eval
  works; architecture supports the answer; STDP can't find it.

The investigation arc has narrowed from "something is wrong with W→A"
through three layers (cascade interference → eval methodology → minimal
arch) to land on "STDP+R-STDP at biological scale, with biology
priors, is insufficient to learn this 4-class language→motor mapping
with this training procedure."

## What's running next (auto)

`bio_b3_gradient` at biological scale (lang=2048, motor=500/action):
- 3 conditions × 3 seeds = 9 runs at parallel=2
- ETA: ~7.5 hours (~02:15 EDT)
- Conditions:
  - `bio_grad_vanilla`: gradient + no biology fix
  - `bio_grad_with_topo_fs`: gradient + topo + FS
  - `bio_grad_with_topo`: gradient + topo only

If gradient succeeds where STDP fails: STDP is the dominant bottleneck.
Future work: biology-grounded learning rules with better credit
assignment (apical-basal dendrites per Bono & Clopath 2017,
three-factor with eligibility per Frémaux & Gerstner 2016).

If gradient also fails: architecture or training-dose ceiling.
Investigate sparser codes (0.05/0.02), longer training (10x events),
or different action representation (population vector).

## Lessons for future experimentation

1. **Always test learning rule on canonical architectures first.** Our
   18-day W→A 0/N alignment streak across v2 + biology-sweep variants
   was on architectures missing cortical canon. We can't trust those
   negative results because the architecture couldn't even pass
   sanity-check eval.

2. **The biology fix has a real but modest effect.** Topographic prior
   + lateral inhibition gives +7pp on canon-enabled architectures.
   Worth keeping in future experiments. Not sufficient alone.

3. **Compute budget for STDP needs to be higher.** 4000 events/seed
   may be 10-100x below the dose required for STDP to differentiate
   4-class sparse codes. Future bio-scale experiments should plan
   for 40K-400K events if attempting STDP-only training.

4. **B3 supervised gradient is the right next experiment.** It
   isolates the plasticity-rule question from the architecture
   question. Specifically asks: "given correct biology + correct eval,
   does ANY learning rule succeed?"
