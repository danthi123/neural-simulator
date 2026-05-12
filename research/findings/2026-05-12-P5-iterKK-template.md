# P5 iter KK — Tier 1 cortical canon at biological scale (TEMPLATE)

**Date:** 2026-05-12
**Status:** PENDING — single-seed smoke running; multi-seed launch ready
**Hypothesis:** iter AA's 4/6 ceiling at toy scale was caused by sub-Tier-1
internal dynamics in wernicke_pool / lang_output_pool. Applying Tier 1
cortical canon (internal_density=0.10, exc=2.0, inh=4.0) at biological
scale (500-neuron pools, 2048 lang_input) gives 6/6 BIDIRECTIONAL.

## User directive (2026-05-12 07:30 EDT)

> "I don't know why we keep testing at toy scale if larger scale (that
> still fits locally) is clearly needed? And also you have my permission
> to autonomously do arch work to continue working towards conversational
> capabilities. Just keep in mind the reference catalog and the goal of
> staying biologically accurate, no cheats."

## Architectural diagnosis (the biological mismatch)

iter AA wernicke_pool / lang_output_pool internal dynamics:
- internal_density=0.05 (vs cortex 10-20% per Lefort 2009)
- exc_weight_mean=0.3 (vs Tier 1 motor pool 2.0)
- inh_weight_mean=0.8 (vs Tier 1 motor pool 4.0)

These pools were ~6x weaker than Tier 1 motor pools that achieved 6/6
on direction-word binding. The biology: real cortex has 10-20% recurrent
density (Lefort 2009 sensory cortex) and strong recurrent excitation
creating NMDA-bistable attractor dynamics (Wang 2002). iter AA was
sub-biological in both density and weight magnitude.

## Iter KK changes (single-variable + scale)

Single-variable change (Tier 1 cortical canon applied identically to
wernicke_pool + lang_output_pool):
- internal_density: 0.05 → 0.10 (Lefort 2009 sensory cortex)
- exc_weight_mean: 0.3 → 2.0 (Tier 1 motor canon)
- inh_weight_mean: 0.8 → 4.0 (Tier 1 motor canon)

Combined biological-scale up:
- n_per_wernicke_pool: 100 → 500 (Tier 1 motor pool size; Schieber 2001)
- n_per_wernicke_pool_fs: 12 → 60 (Tier 1 FS size)
- n_per_lang_out_pool: 200 → 500
- n_lang_input: 1024 → 2048 (Tier 1 size)

Other variables kept identical to iter AA:
- multi-pool wernicke (Path A) — ON
- per-concept lang_output_pools — ON
- interleaved training — ON
- wernicke FS=4.0 (iter AA default; iter BB regression at 8.0)
- no lang_output FS (iter CC regression)
- Topographic bias on lang_input → wernicke_pool (Pulvermüller
  2001-2003 cortical somatotopy; biology-faithful additive prior)
- Multi-trial averaging at recognition (n=5; addresses iter AA seed
  44 borderline failure)

Total neurons: ~7K (vs iter AA 5K). Total synapses: ~2-3M (vs iter AA 1M).

## Recipe (iter KK production)

```bash
python -m research.runners.validate_ventral_semantic --seed N \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 500 --n-per-wernicke-pool-fs 60 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 500 \
    --apply-wernicke-topographic \
    --n-recognition-trials 5 --inter-trial-rest-steps 100
```

## Results (PENDING)

| Seed | apple→p0 | apple→p1 | Apple OK | river→p0 | river→p1 | River OK | BIDIR |
|---|---|---|---|---|---|---|---|
| 42  | ? | ? | ? | ? | ? | ? | ? |
| 43  | ? | ? | ? | ? | ? | ? | ? |
| 44  | ? | ? | ? | ? | ? | ? | ? |
| 100 | ? | ? | ? | ? | ? | ? | ? |
| 101 | ? | ? | ? | ? | ? | ? | ? |
| 102 | ? | ? | ? | ? | ? | ? | ? |

**TOTAL: apple ?/6, river ?/6, BIDIR ?/6**

## Comparison to baselines

| Iter | Scale | Internal dyn | Result | BIDIR |
|---|---|---|---|---|
| AA | 5K (100-neuron pools) | weak (0.05/0.3/0.8) | 4/6 ceiling | 4/6 |
| BB | 5K | wernicke FS 8.0 | catastrophic | 0/6 |
| CC | 5K | lang_output FS | trades errors | 2/6 |
| **KK** | **7K (500-neuron pools)** | **Tier 1 (0.10/2.0/4.0)** | **PENDING** | **?/6** |

## Interpretation (to fill after results)

(Will fill in after 6-seed multi-seed completes)

## Next steps (conditional)

**If iter KK passes 6/6 BIDIR:**
- Scale concept count from 2 → 4 (validate_ventral_semantic_multi)
- If 4-concept passes: scale to 8 concepts (vocabulary growth)
- Begin P6 Broca's grammar scaffolding (catalog G.12)
- Update CLAUDE.md / dashboard capability_status.json

**If iter KK partial (≥ iter AA's 4/6 but < 6/6):**
- Add bias-correction training (more river events than apple to
  compensate for residual apple-bias)
- Try further pool size scale (1000-neuron pools per Tier 2.1 recipe)

**If iter KK regresses below iter AA:**
- Architectural interference — Tier 1 canon is too strong for the
  long chain (lang_input → wernicke → semantic_cortex → lang_output)
- Try: canon on wernicke_pool only (not lang_output_pool)
- Try: smaller exc_weight_mean (1.0 instead of 2.0)
- Consider: anchored concepts via Cluster K v2 visual cortex
  (semantic grounding from sensory features)

## Biology citations

- **Lefort 2009** (Neuron 61:301-316): mouse barrel cortex layer 2/3
  recurrent density ~10-15%; the cortical canon value
- **Wang 2002** (Neuron 36:955-968): NMDA-bistable attractor dynamics
  in PFC working memory; recurrent excitation must be strong enough
  to sustain persistent activity without input
- **Schieber 2001** (J Neurophys 86:2125-2143): primate M1 distributed
  representations; 500-neuron pool size matches functional motor
  representation size in monkey cortex
- **Pulvermüller 2001-2003**: cortical somatotopy of language; words
  cluster in topographically-organized cortical regions per their
  motor/sensory referents
- **Hickok & Poeppel 2007** (Nat Rev Neurosci 8:393-402): dual-stream
  model (G.11); ventral semantic stream maps sound → meaning
- **Wernicke / Kandel 6e pp 1380-1387**: semantic comprehension at
  posterior superior temporal + middle temporal gyrus

## Catalog alignment

- G.11 Hickok & Poeppel dual-stream ✓
- G.13 Wernicke's area (posterior superior temporal) ✓
- No motor-decoder cheat ✓
- No external LLM cheat ✓
- Biology-grounded: cortical canon from Lefort 2009 + Tier 1 motor
  binding canon (the same one that gave Tier 1 6/6) ✓
