# Learned graded-embedding: a BIOLOGICAL HOMEOSTATIC recurrent gives CYCLE-INDEPENDENT faithfulness — GO

**Date:** 2026-06-11
**Runner:** `research/runners/learned_graded_embedding_homeostasis_probe.py`
**Raw:** `research/findings/raw/_lge_homeostasis_multiseed.json` (+ `_lge_homeostasis_smoke3.json`)
**Backend:** `SIM_BACKEND=cupy` (GPU, RTX 3090), foreground, multi-seed 42/43/44.

## TL;DR — GO (cycle-independent)

The last open mechanism question for the dual/CLS learned-embedding is **RETIRED**. A proper
biological homeostatic mechanism — **Turrigiano synaptic scaling** (and **Oja's rule**, ~equally
well) on the spiking-Hebbian recurrent — gives **CYCLE-INDEPENDENT faithfulness**: the learned `W`
tracks the co-occurrence counts AND recovers the graded structure AND generalizes, **flat across
cycles {2,5,10,20,40} AND under store-volume stress**, on all 3 seeds — where the un-normalized
recurrent collapses by cycle 20. The architecture gates (G1 graded + 2nd-order margin, G2
generalization + controls) **re-confirm at the high-cycle end (cycles=40), not just at the
hand-picked cycles=2.** ⇒ the build no longer depends on hand-picking cycles; it can start with
**zero open mechanism risks**, using the homeostatic recurrent as the default learner.

Adding synaptic scaling / Oja makes the learner **MORE realistic** (real neurons have these
bounded-Hebbian mechanisms to prevent runaway potentiation), not less — so this is a strict
improvement over the cycles=2 hand-pick, biologically and operationally.

**Honest caveat on the consensus label:** the runner's auto-consensus printed
`MIXED_with_GO:42=GO,43=BOUNDARY,44=GO`. The seed-43 "BOUNDARY" is **a brittle anti-cheat
sub-condition artifact, NOT a homeostasis or anti-cheat failure** — see §4. The load-bearing
cycle-independence + the substantive anti-cheats hold 3/3.

## The question (the last open mechanism unknown)

The brain-based learned-embedding is fully de-risked + fully brain-based end-to-end
(`2026-06-11-learned-graded-embedding-confirm-GO_full.md` + `-divnorm-readout-GO.md`, commits
`e6e277e3` + `9fa90d74`) — but at a **HAND-PICKED operating point**: the un-normalized excitatory
Hebbian recurrent (`LearnedAssocGraph`) **SATURATES** with cycles (`Pearson(W, raw_counts)` +0.69 @2cyc
→ +0.06 @20cyc), so the recipe uses cycles=2. A multiplicative gamma-decay arm was REAL but INFERIOR.
Both are hand-picked, not self-regulating.

**The de-risk:** does a proper BIOLOGICAL HOMEOSTATIC mechanism — **Oja's rule** (Oja 1982: Hebbian +
the `−y²w` normalization; the canonical bounded-Hebbian whose fixed point is the unit-norm weight
vector) and/or **synaptic scaling** (Turrigiano: normalize each neuron's total synaptic input to a
target) — give **cycle-independent** faithfulness, so the build is robust without hand-picking cycles?

## What was run

`HomeostaticAssocGraph` (subclass of the desaturate probe's `DesaturatingAssocGraph`) applies a
biological homeostatic rule **per co-fire cycle**, restricted to the **pool↔pool recurrent**, **per
POSTSYNAPTIC neuron** (`cp_connections` is `(pre→post)`: neuron *j*'s incoming weights = the *j*-th
**column** — so the per-post group-reduce is a segmented `cp.add.at` over the post-column index of
each pool↔pool data entry). **Runner-side only — NO `sim/` edits.** Two rules, both with a **FIXED
homeostatic set-point** (NOT fit to `S_true`):

- **Oja** — renormalize each postsynaptic neuron's **incoming L2 norm** to a set-point (the Oja-1982
  fixed-point form; clip-only-above, so the set-point is a ceiling on total potentiation).
- **Synaptic scaling** — renormalize each postsynaptic neuron's **incoming sum (L1)** to a set-point
  (Turrigiano multiplicative scaling).

Set-points were **calibrated** to the natural per-post-neuron incoming norm at the *faithful*
early-learning regime (cycles=2: median incoming L2 ≈ 13, median incoming sum ≈ 120 in this config):
**Oja target = 15, scaling target = 150.** (Initial uncalibrated targets of 2 / 10 crushed the
recurrent to mean ≈ 0.012 — a mis-scaled set-point, fixed by calibration.)

Read-out is **FIXED** to the validated FULLY BRAIN-BASED divnorm recipe (the `-divnorm-readout-GO`
winner: Carandini-Heeger divisive normalization, `interleave` order, steps=2, σ=0.001, exp=2.0) — so
the measured variation is the **LEARN's**, not the read-out's.

Sweep (per seed): **variants** {un-normalized, gamma=0.95 (inferior ref), oja-t15, scaling-t150} ×
**cycles** {2,5,10,20,40}; **store-volume stress** {reps 1,2,3} at cycles=10 (scaling vs
un-normalized); **gate re-confirm** {cycles 2,10,40} (full G1 + G2 + controls). Config: 30 concepts
(6 hubs + 24 members), 120 facts, n_pool=1000, pattern_size=60 (reduced from the de-risk's
2000/100 for tractable wall-clock — the cycle-independence question is about learning *dynamics*, not
scale; host ceiling reproduces at +0.929/gen 1.000). Total ≈ 3.3 h on the 3090.

## 1. The CORE result — cycle-independence (cycles 2→40, all 3 seeds)

`Pearson(W, raw_counts)` (does the learned `W` track the co-occurrence COUNTS?), with its
least-squares slope per cycle:

| variant | seed 42 (2→40) | seed 43 | seed 44 | slope/cyc (mean) |
|---|---|---|---|---|
| **un-normalized** | +0.747 → **+0.243** | +0.739 → **+0.274** | +0.720 → **+0.371** | **−0.0122** |
| **Oja (t=15)** | +0.838 → +0.758 | +0.840 → +0.786 | +0.826 → +0.756 | **−0.0019** |
| **synaptic scaling (t=150)** | +0.854 → +0.778 | +0.856 → +0.793 | +0.844 → +0.770 | **−0.0020** |

The homeostatic slope is **~6× flatter** than the un-normalized. The brain-based read-out's recovery
holds in lock-step — worst-case (min over the cycle sweep) `Pearson(sim, S_true)` and generalization:

| variant | min Pearson(sim,S_true) (42/43/44) | min generalization (42/43/44) |
|---|---|---|
| un-normalized | +0.094 / +0.154 / +0.142 | 0.375 / 0.433 / 0.583 |
| gamma=0.95 (inferior ref) | +0.429 / +0.426 / +0.254 | 0.675 / 0.783 / 0.683 |
| **Oja (t=15)** | +0.632 / +0.655 / +0.519 | 0.808 / 0.958 / 0.842 |
| **synaptic scaling (t=150)** | **+0.655 / +0.665 / +0.563** | **0.833 / 0.958 / 0.892** |

⇒ **both Oja and synaptic scaling are cycle-independent on all 3 seeds**; the un-normalized collapses;
gamma-decay is genuinely inferior (confirms the desaturate finding). Scaling is auto-selected best on
all 3 seeds (marginally above Oja). Host ceiling = +0.929 / gen 1.000.

## 2. Store-volume / scale stress (toward production) — HOLDS 3/3

At the best homeostatic variant (scaling t=150, cycles=10), inflate the store (each fact stored
{1,2,3}× — a larger corpus; counts scale, structure is invariant). `Pearson(sim,S_true)` / gen as the
store grows, scaling vs un-normalized:

| seed | scaling (rep1 / rep2 / rep3) | un-normalized (rep1 / rep2 / rep3) | holds |
|---|---|---|---|
| 42 | 0.804/0.94 → 0.812/0.94 → 0.821/0.94 | 0.712/1.00 → 0.429/0.96 → **0.276/0.88** | **True** |
| 43 | 0.840/1.00 → 0.855/0.98 → 0.863/0.98 | 0.747/1.00 → 0.407/0.88 → **0.269/0.68** | **True** |
| 44 | 0.811/1.00 → 0.850/1.00 → 0.861/1.00 | 0.300/0.59 → 0.074/0.47 → **0.050/0.43** | **True** |

The homeostatic recurrent's faithfulness **IMPROVES slightly** as the store grows (more co-fire,
better-estimated structure, bounded magnitude); the un-normalized **saturates faster** with more
volume. The production concern ("many facts shouldn't saturate the recurrent") is **answered**: the
homeostatic recurrent does not saturate under store-volume stress. (3/3 holds.)

## 3. Gate re-confirm at the cycle-independent operating point (cycles 2 / 10 / 40)

Full G1 (Pearson ≥ 0.5 + graded + 2nd-order cat~dog margin ≥ +0.10) + G2 (A1 generalization ≥ 0.7 +
A2 orthogonal-collapse + A3 permuted-property-collapse), on the FULLY BRAIN-BASED divnorm read-out,
at the best homeostatic variant:

| seed | cycles=2 | cycles=10 | cycles=40 |
|---|---|---|---|
| 42 | G1✓(P+0.81,2nd+0.70) G2✓(A1 1.00/A2/A3) | G1✓(P+0.81,2nd+0.71) G2✓(0.94) | **G1✓(P+0.66,2nd+0.56) G2✓(0.83)** |
| 43 | G1✓(P+0.85,2nd+0.73) G2✓(1.00) | G1✓(P+0.84,2nd+0.71) G2✓(1.00) | **G1✓(P+0.67,2nd+0.55) G2✓(0.96)** |
| 44 | G1✓(P+0.88,2nd+0.74) G2✓(1.00) | G1✓(P+0.81,2nd+0.69) G2✓(1.00) | **G1✓(P+0.66,2nd+0.55) G2✓(0.89)** |

**G1 and G2 pass at EVERY cycle on EVERY seed** — including the high-cycle end (cycles=40) the
un-normalized cannot reach. This is the cycle-independence bar: the gate is no longer tied to cycles=2.

## 4. Anti-cheats + the seed-43 consensus-label caveat (honest)

Load-bearing anti-cheats — **all hold 3/3, every cycle:**
- `Pearson(W, raw_counts) < 0.99` (genuine learning, not pass-through): W stays ~+0.77–0.86 (well
  below 0.99 AND well above the +0.06 collapse floor) — `W-distinct=True` everywhere.
- the homeostatic rule is a **FIXED set-point** (calibrated to the natural early-learning incoming
  norm; **not fit to `S_true`**).
- G2 controls collapse: A2 orthogonal acc ≈ 0.14–0.19, A3 permuted-property acc ≈ 0.11–0.33 (≤ 1.5×
  chance 0.375) on all seeds/cycles.
- the **permuted-co-occurrence** control (re-learn on a scrambled corpus) collapses the **Pearson**:
  +0.004 / +0.072 / +0.105 (vs the real +0.85), and permuted generalization → chance.

**The seed-43 `BOUNDARY` label** comes from one brittle sub-condition of the G5 permuted-co-occurrence
criterion: `g5 = (|perm_pearson| < 0.30) AND (NOT perm_is_graded) AND (perm_gen ≤ 1.5·chance)`. On
seed 43, `perm_pearson` (+0.004/+0.072/+0.105) and `perm_gen` BOTH collapsed correctly, but
`perm_is_graded` came back **True** — and that flag is just `within_cluster_cos > between_cluster_cos`,
which on a **structureless scrambled matrix is a coin-flip** (the standalone re-derivation gives
permuted within=+0.160, between=+0.120, **margin +0.040 ≈ 0**, Pearson +0.072 — vs the REAL corpus
margin +0.722, Pearson +0.840). So the scramble IS structureless; the boolean flipped on a ~0 margin
by chance on 1/3 seeds. **This is a control-criterion robustness note, not evidence the homeostatic
learn cheats.** (The same brittle sub-condition exists in the confirm/divnorm probes; it just didn't
hit there.) Recommended fix for the build's anti-cheat: gate G5 on a **margin/Pearson threshold**
(e.g. permuted 2nd-order margin < +0.10) rather than the bare `is_graded` boolean.

⇒ Substantively, **3/3 seeds pass cycle-independence + the load-bearing anti-cheats**; the consensus
string is `MIXED` only because of the brittle boolean.

## Recommended homeostatic recipe (for the build)

> Default learner = the spiking-Hebbian recurrent (`LearnedAssocGraph`) **+ Turrigiano synaptic
> scaling** applied **per cycle, per postsynaptic neuron, on the pool↔pool recurrent**: rescale each
> neuron's incoming pool↔pool **sum** to a fixed set-point (clip-only-above-target), the set-point
> **calibrated to the natural early-learning incoming sum** (~the per-post-neuron incoming sum at
> cycles=2 ≈ 120–150 in this config; scale with `pattern_size`/`density`). **Oja** (incoming-L2 renorm
> to ~15) is an equally-valid alternative (cycle-independent on all 3 seeds, marginally below scaling).
> Read-out = the validated brain-based divnorm (Carandini-Heeger, interleave, steps 2, σ=0.001, exp 2).

With this, **cycles is no longer a tuned knob** — any cycle count in {2..40} (and growing store
volume) yields a faithful, graded, generalizing embedding that passes the gates. The set-point IS the
one parameter, and it is a **principled biological quantity** (the homeostatic norm target), not a
fit to the labels.

## Decision

**GO (cycle-independent).** A biological homeostatic mechanism (synaptic scaling / Oja) retires the
last open mechanism unknown: the learned-embedding is robust **without hand-picking cycles** —
faithfulness holds across cycles {2..40} and under store-volume stress, gates re-confirm at the
high-cycle end, anti-cheats hold, multi-seed. **The build can start with zero open mechanism risks.**

## Explicit next step

Build the dual/CLS learned-embedding with the homeostatic recurrent as the default learner (synaptic
scaling, calibrated set-point) — cycles is no longer a tuned constraint. When wiring the build's
anti-cheat harness, replace the G5 permuted-co-occurrence `is_graded` sub-condition with a
margin/Pearson threshold (the only loose end here — a control-criterion robustness fix, not a
mechanism risk). Optionally re-confirm the recipe once at the de-risk's full scale (n_pool=2000,
pattern_size=100) with the set-point re-calibrated to that scale's natural incoming norm, before the
build's first integration.
