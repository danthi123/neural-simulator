---
type: finding
status: negative
date: 2026-08-18
mechanism: perception-v1-pooler-learned-decorrelation
runner: research/runners/_laneD_v1_pooler_trace_invariance_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/v1_pooler_trace_decorr_baseline_off_6seed.json
  - research/findings/raw/lanes/perception/v1_pooler_trace_decorr_lr05_tp01_6seed.json
---

# lane D perception: a LEARNED anti-Hebbian lateral-inhibition DECORRELATION stage on the V1-complex features does NOT open invariant identity (NO-GO) — it regresses the one baseline GO into over-sparsification and lifts none of the position-blind seeds

**One-line verdict.** The board's named representation-side lever for the seed-43/44 upstream residual on the
V1 -> OnSubstratePooler trace route was "learned V1-complex normalization / decorrelation upstream of the pooler"
(2026-08-11 next-mechanism (b)). Built it as an additive, default-off learned anti-Hebbian lateral-inhibition
decorrelation stage (Foldiak 1990; SAILnet, Zylberberg-Murphy-DeWeese 2011 PLoS Comput Biol 7(10):e1002250) on
the V1-complex features BEFORE the pooler top-k. At the banked honest op-point (`--n-ex 8`, 24 held images,
`--inhib-frac 0.67`), 6 seeds: **decorrelation is NO-GO (0/6)**. It does not lift the blind seeds; it regresses
the single baseline GO (seed 42) into an over-sparsified NO-GO, moving the route TRACE-ROUTED-PARTIAL-1/6 ->
TRACE-ROUTED-NOGO. This CONFIRMS and EXTENDS the 2026-05-31 anti-Hebbian-decorrelation boundary
(`2026-05-31-foldiak-learned-decorrelation-BOUNDARY-over-sparsifies-near-ortho-bar-is-GENERAL-not-method-specific-3-methods-converge`)
to the position-invariance route, for a quantified mechanistic reason: lateral inhibition DECORRELATES but does
not POOL across retinotopic positions, and the V1-complex code has essentially no position-invariance margin to
amplify at this level.

## The lever (runner-side, additive, NO sim/ edit)

An `AntiHebbianDecorr` stage sits between `_normalize_complex` and the pooler top-k selection, gated behind a
default-off `--decorr` flag with a `--decorr-lr` learning rate (and `--decorr-target-p`, `--decorr-epochs`,
`--decorr-settle`). Feedforward is the identity (one decorrelation unit per V1-complex feature); the LEARNED
object is the symmetric, non-negative, per-pair lateral inhibitory weight `W_ij`. The output settles under
recurrent inhibition `y = relu(x - W y)` (bounded: `y_i <= x_i`, `W >= 0`, so stable by construction), and `W`
grows anti-Hebbianly toward a co-activity target `p^2` (SAILnet's inhibitory rule): `dW_ij = lr (y_i y_j - p^2)`,
`i != j`, then `W = max(W, 0)`, symmetric, zero-diagonal. It is PLASTIC per-pair decorrelation learned once per
seed on the train ensemble and applied identically to train / held / scramble (and therefore to the V1-direct
control) — NOT fixed inhibition, divisive normalization, or homeostatic synaptic scaling (all already refuted on
this route). This is the biologically-standard local anti-Hebbian sparse-coding decorrelator; the external
literature is convergent that every plausible off-diagonal decorrelator is exactly this recurrent cross-neuron
interaction (`2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research`).

The primary run engages the mechanism deliberately: `--decorr-lr 0.05 --decorr-target-p 0.1` drives the output
co-activity below its natural mean so `W` genuinely grows (the auto target `p^2 = natural mean co-activity` is
inert — see the operating-point note). Params were fixed by the representation criterion (the setting where the
stage genuinely engages and prunes columns), NOT by the decode gate.

## Anti-cheats (all pass)

- **lr=0 byte-identical to the `--inhib-frac 0.67` control** — verified at the op-point (seed 42) by exact JSON
  compare: every decision-relevant block (`v1_complex`, `v1_pooler_trace`, `shuffled_temporal`, `no_learning`,
  `decision`, `over_sparsification_guard`) is identical between `--decorr`-off and `--decorr --decorr-lr 0`; only
  an informational `decorr` provenance block (recording lr=0, `corr_reduction`=0.0) is added. `W` inits to zero,
  so lr=0 keeps the transform an exact identity for the non-negative V1-complex features.
- **De-quantized readout** — `--n-ex 8` = 24 held images (decode step 1/24), the honest readout the 2026-08-11
  finding mandated after the 6-image readout was shown to INFLATE the baseline. The banned 6-image readout is not
  used.
- **shuffled-temporal + pixel-scramble + V1-direct (no-pooler) controls** — computed per seed and part of the
  fixed 5-gate; the V1-direct control reads the SAME decorrelated features (like-for-like).
- **Over-sparsification guard** (2026-05-31 boundary) — reported per seed: pooler-column alive fraction (all-alive
  check) + within-identity held-code reliability (reliability floor). Under decorrelation 3/6 seeds trip it.
- **Determinism** — numpy backend, `W` init 0, deterministic train-image order; pooler seeded via its `seed`
  arg exactly as the baseline.

## Result — 6-seed per-seed 5-gate (banked honest op-point)

Artifacts: `research/findings/raw/lanes/perception/v1_pooler_trace_decorr_baseline_off_6seed.json` (decorr OFF
reference) and `research/findings/raw/lanes/perception/v1_pooler_trace_decorr_lr05_tp01_6seed.json` (decorr ON).

Op-point: `--position-axis y --complex-norm local_orient_div --n-col 240 --k-win 8 --pool-lr-pot 0.08
--pool-lr-depress 0.01 --trace-decay 0.75 --n-ex 8 --inhib-frac 0.67`. Gates per seed: decode >= chance+0.10;
trace margin beats shuffled-temporal (>=0.05); beats V1-direct (>=0.02); beats no-learning (>=0.05); pixel-scramble
does not decode. GO = all five.

BASELINE (decorr OFF) — reproduces the banked baseline: **TRACE-ROUTED-PARTIAL-1/6** (only seed 42 GO).

| seed | decode | decode_ok | >shuffled | >V1 | >no-learn | scramble-collapse | GO | over-sparsified |
|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 42  | 0.58 | Y | Y | Y | Y | Y | **Y** | . |
| 43  | 0.25 | . | . | . | . | . | . | . |
| 44  | 0.33 | . | . | . | . | Y | . | Y |
| 100 | 0.38 | . | . | . | . | Y | . | . |
| 101 | 0.42 | . | . | . | . | Y | . | . |
| 102 | 0.38 | . | . | Y | . | Y | . | . |

DECORR ON (`--decorr-lr 0.05 --decorr-target-p 0.1`): **TRACE-ROUTED-NOGO (0/6)**. corr_reduction mean +0.0021,
within-identity reliability mean 0.5424, pooler-column alive fraction mean 0.1479; 3/6 seeds over-sparsified.

| seed | decode | decode_ok | >shuffled | >V1 | >no-learn | scramble-collapse | GO | over-sparsified |
|---:|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 42  | 0.50 | Y | . | Y | Y | Y | . | Y |
| 43  | 0.25 | . | . | . | . | Y | . | . |
| 44  | 0.25 | . | . | . | . | . | . | Y |
| 100 | 0.17 | . | . | . | . | Y | . | . |
| 101 | 0.42 | . | . | . | . | Y | . | . |
| 102 | 0.46 | Y | . | Y | . | Y | . | Y |

The single baseline GO (seed 42) LOSES the trace-beats-shuffled gate under decorrelation (margin +0.112 ->
+0.0783; shuffled +0.0409 -> +0.0292) and becomes over-sparsified. No blind seed (43/44/100/101/102) gains a
trace-specific margin. Decorrelation makes the route strictly worse (PARTIAL-1/6 -> NOGO-0/6).

## Root cause — quantified (why decorrelation cannot help here)

Numbers in this section are from a throwaway diagnostic (not a committed artifact); the committed 6-seed
artifacts back the verdict tables above.

<!--derived-->
A throwaway diagnostic measured the crux directly at the V1-complex level (post `local_orient_div`), before any
pooler: the same-category / cross-position cosine (held vs train positions) is **0.989**, and the cross-category
cosine is **0.989** — an invariance margin of **~0.000**. The V1-complex code is nearly isotropic in cosine:
same-orientation-different-position images are no more similar than different-orientation images. There is no
position-invariance margin at this level for a decorrelator to amplify.

<!--derived-->
Anti-Hebbian lateral inhibition DECORRELATES (removes redundancy / makes units compete) but does NOT POOL across
retinotopic positions — so it cannot manufacture the absent invariance. Swept across strength (lr 0.05 -> 1.0,
target_p 0.1 -> 0.005), the stage produces one of two outcomes and never opens a same-vs-cross-position margin:
(1) at higher lr, `W` overshoots on a sample, the settled output collapses to near-zero, the anti-Hebbian term
`(y_i y_j - p^2)` goes NEGATIVE, and `W` is driven back to zero — the equilibrium transform is the identity (the
output second-moment and Pearson correlation return to the input values exactly); (2) in the narrow band where
`W` stays finite (~lr 0.05), it over-sparsifies (within-identity reliability collapses, columns die) without
lifting the margin. This is the SAME separation-vs-reliability frontier the 2026-05-31 finding traced across
three independent coding methods; here it manifests on the position axis.

## Relation to the record

CONFIRMS + EXTENDS `2026-05-31-foldiak-learned-decorrelation-BOUNDARY-...` (anti-Hebbian decorrelation traces the
separation-vs-reliability frontier rather than threading it) to the position-invariance route. Consistent with
`2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research` (recurrent cross-neuron interaction is the
only local decorrelator) and `2026-06-14-L1-learned-cortex-fair-test-GO` (subtractive feedforward inhibition is
more robust than recurrent lateral — but neither pools). The harder-kWTA contributor stays as banked in
`2026-08-11-laneD-v1-pooler-harder-kWTA-...`: this finding tests and REFUTES its next-mechanism sub-lever (b)
"learned decorrelation"; sub-lever (a) "wider transformation task" remains untested.

## NOT-A-WALL — next lever (quantified residual + named mechanism)

The residual is now sharply localized: the trace rule needs a representation in which same-orientation features
at DIFFERENT positions are GROUPED (pooled), and the two operations that could produce that grouping —
competition (harder k-WTA, 2026-08-11) and decorrelation (this finding) — are both now tested and insufficient,
because neither POOLS. The genuinely-untested mechanism is therefore representation-side and additive, a learned
CROSS-POSITION POOLING upstream of the trace pooler, NOT another competition/decorrelation stage (both refuted):
a complex-cell-style OR-pooling / slow-feature grouping that learns to map the retinotopic positions of one
orientation onto a shared unit (trace/slow-feature learning applied to the V1 FEATURES themselves, so identity is
made position-tolerant BEFORE the pooler binds it). The measured target it must move is the ~0.000 V1-level
cross-position invariance margin. The complementary sub-lever the 2026-08-11 finding named — a wider
transformation task (more categories/positions) so the trace has more invariance evidence — is also still open.
An honest verdict settles the op-point: at the banked op-point, learned decorrelation is refuted, so decorr stays
default-off (lr=0), and the route's residual is a missing pooling operation, not a missing decorrelator.
