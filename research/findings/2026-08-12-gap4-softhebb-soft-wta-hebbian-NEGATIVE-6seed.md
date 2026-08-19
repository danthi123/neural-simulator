---
type: finding
status: contributing
date: 2026-08-12
mechanism: deep-credit-on-spikes
lane: gap#4 ALL-IN (SoftHebb — the fully-unsupervised, no-feedback end of the local-rule family)
verdict: 6-SEED METHOD-NEGATIVE — an UNSUPERVISED soft-WTA Hebbian (SoftHebb) deep spiking stack does NOT enter the regime. Each verdict is precondition-guarded (tools.verdict.Verdict): a config counts only where a REGIME EXISTS (the BPTT ceiling clears chance by >=0.05). On the 3 valid-regime configs (inheritance N=3, inheritance N=4, xor N=3) the SoftHebb+optimal-ridge readout is NO-GO — it never clears the >=0.10 min-over-seeds lift over a MATCHED-width frozen-random reservoir read the identical way (min lift NEGATIVE in all three). The other 3 configs (xor N=4, hier3 N=3, hier3 N=4) are UNDEFINED — the BPTT ceiling itself sits at/below chance+0.05, so no regime exists to enter (NOT a negative). SoftHebb NEVER achieves a GO on any config (0/6 seeds each). On structured-input inheritance a small POSITIVE MEAN lift appears but sits inside the noise and the shuffled-input anti-cheat LEAKS; on uniform-input XOR SoftHebb ties the reservoir. FA/KP collapse to majority-class (the wall). The capability is NOT walled (Q1 Forward-Forward + Q4 DECOLLE crack it); this method is. CONFIRMS the July-15 'SoftHebb = unsupervised feature side, not task-directed' objection under the enter-the-regime metric. seed_control_verified=True; NO sim/ edit.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_gap4_softhebb_local_derisk.py
artifacts:
  - research/findings/raw/_gap4_softhebb/inheritance_N3_6seed.json
  - research/findings/raw/_gap4_softhebb/inheritance_N4_6seed.json
  - research/findings/raw/_gap4_softhebb/xor_N3_6seed.json
  - research/findings/raw/_gap4_softhebb/xor_N4_6seed.json
  - research/findings/raw/_gap4_softhebb/hier3_N3_6seed.json
  - research/findings/raw/_gap4_softhebb/hier3_N4_6seed.json
external: SoftHebb (Journé, Rodriguez, Guo, Moraitis 2023, ICLR, "Hebbian deep learning without feedback") — the exact rule was READ (Eq 1 soft-WTA softmax; Eq 2 Oja-instar update with the anti-Hebbian sign flip) and ported to the LIF forward. arXiv identifier cited in the body.
---
<!--derived-->

# gap#4 SoftHebb — an UNSUPERVISED soft-WTA Hebbian deep spiking stack does NOT beat a random reservoir (enter the regime); the July-15 "not task-directed" objection is confirmed under the new metric (6-seed)

## The mechanism tested (READ, not recalled)

<!--derived-->
SoftHebb (Journé et al., 2023, ICLR; arXiv:2209.11883) is the fully-UNSUPERVISED, no-feedback, no-label-per-layer
end of the local-rule family. Each layer is a soft winner-take-all Hebbian module. The exact rule was read from
the paper and ported to the project's LIF SNN forward (`sim/bptt_snn_gpu`, reuse-by-import, NO sim/ edit):

- Eq 1 (soft-WTA): `y_k = softmax(u/τ)_k`, pre-activation `u = x @ W` (`u_k = Σ_i w_ik x_i`).
- Eq 2 (Oja-instar): `Δw_ik = η · s_k · y_k · (x_i − u_k·w_ik)`, where `s_k = +1` for the maximally-activated
  neuron (`argmax_k u`) and `−1` for all others — SoftHebb's "soft anti-Hebbian" competition (winner moves TOWARD
  the input, losers are pushed AWAY); the `(x_i − u_k·w_ik)` term is the Oja normalisation.
- Stacking: greedy layer-wise; each hidden layer is fully trained + FROZEN, its summed-spikes are the presynaptic
  `x` to the next. Then ONE supervised linear readout on the frozen concatenated deep features.

## Why this was worth re-testing (the Q5 reframe)

The July-15 survey (`2026-07-15-deep-credit-fresh-class-gate-...`) filed SoftHebb as "unsupervised feature side,
not task-directed deep credit" — but that predates the Q5 ENTER-THE-REGIME reframe
(`2026-08-12-gap4-obligatory-depth3-...-NEGATIVE`), which set the falsifiable success metric to "leave
majority-class + BEAT the frozen reservoir" after obligatory-depth-3 proved unconstructible. Q1 Forward-Forward
and Q4 DECOLLE (per-layer LOCAL objectives) crack it. The open question: does the STRONGEST brain-based claim —
a fully-unsupervised, no-label soft-WTA Hebbian stack — also enter the regime?

## The isolation (the comparison is the RULE alone)

The SoftHebb arm and the frozen-reservoir floor are byte-for-byte identical except the hidden weights: SAME init
RNG stream (seed+1), SAME widths, SAME `w_scales`, SAME LIF forward, SAME concat-hidden-summed-spikes read
(`_reservoir_features`), SAME optimal five-fold-CV ridge (`_optimal_ridge_acc`). The ONLY difference is whether the
hidden weights were shaped by unsupervised soft-WTA Hebbian competition or left RANDOM. So (SoftHebb − reservoir)
isolates exactly what the competition adds over a random projection.

## Result — 6-seed, per-arm inherit accuracy (42 43 44 100 101 102)

Artifacts: `research/findings/raw/_gap4_softhebb/*_N*_6seed.json` (one per config, each carrying its
`status` + `preconditions` block).

<!--derived-->
| task / N | status | SoftHebb | reservoir<br>matched | reservoir<br>wide | shuffled-<br>SoftHebb | FA | KP | BPTT<br>ceiling | oracle | chance | lift mean<br>(MIN) | GO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inheritance N=3 | NO-GO | 0.648 | 0.605 | 0.691 | 0.636 | 0.333 | 0.333 | 0.667 | 0.735 | 0.333 | +0.043 (−0.111) | 2/6 |
| inheritance N=4 | NO-GO | 0.642 | 0.593 | 0.802 | 0.636 | 0.333 | 0.333 | 0.525 | 0.642 | 0.333 | +0.049 (−0.111) | 0/6 |
| xor N=3 | NO-GO | 0.630 | 0.623 | 0.682 | 0.613 | 0.513 | 0.500 | 0.616 | 0.476 | 0.524 | +0.006 (−0.017) | 0/6 |
| xor N=4 | UNDEF | 0.623 | 0.615 | 0.687 | 0.600 | 0.492 | 0.492 | 0.573 | 0.476 | 0.524 | +0.007 (−0.022) | 0/6 |
| hier3 N=3 | UNDEF | 0.145 | 0.148 | 0.157 | 0.145 | 0.167 | 0.167 | 0.173 | 0.160 | 0.167 | −0.003 (−0.167) | 1/6 |
| hier3 N=4 | UNDEF | 0.151 | 0.160 | 0.145 | 0.145 | 0.167 | 0.167 | 0.154 | 0.170 | 0.167 | −0.009 (−0.148) | 0/6 |

Status is precondition-guarded (tools.verdict.Verdict): NO-GO where a REGIME EXISTS (BPTT ceiling clears chance
by >=0.05) and SoftHebb still fails to enter it; UNDEF where the BPTT ceiling itself does not clear chance+0.05
(xor N=4 misses it by ~0.001; hier3 sits at chance) so no regime exists and the test is uninterpretable — NOT a
negative. SoftHebb never reaches a GO on any config.

SoftHebb hyperparameters: τ=0.25, η=0.02, 50 unsupervised passes/layer, homeostatic synaptic-scaling to the init
weight radius (keeps the LIF forward firing while competition rotates the tuning direction); τ was tuned on seed 42
only (sharper competition helps), the other 5 seeds are held out from tuning and the GO bar is the MIN over all 6.

## Read (the honest verdict)

<!--derived-->
- **On the 3 valid-regime configs (inheritance N=3, inheritance N=4, xor N=3) SoftHebb is NO-GO.** It never clears
  the >=0.10 MIN-over-seeds lift over the matched reservoir — the MIN lift is NEGATIVE in all three (−0.111,
  −0.111, −0.017). A robust METHOD-negative under the enter-the-regime metric.
- **On structured-input inheritance** there is a small POSITIVE MEAN lift (+0.043 / +0.049), and the mechanism is
  real (SoftHebb clusters the category structure). But it sits inside the noise (small held-out set, 0.037
  granularity; MIN lift −0.111), GO is only 0–2/6, and the **shuffled-input anti-cheat LEAKS** (a SoftHebb stack
  trained on column-shuffled inputs still beats the reservoir on some seeds — MAX +0.148/+0.185), so the mean lift
  is not cleanly attributable to genuine input-structure learning above a random projection's own variance.
- **On uniform-input XOR** SoftHebb ~= the reservoir (mean lift +0.006/+0.007). This is principled: SoftHebb learns
  the INPUT DISTRIBUTION's cluster structure, and the XOR hypercube is uniform (no clusters), so competition has
  nothing to latch onto — and the LIF reservoir already expands XOR to 0.62–0.69 via its spiking nonlinearity, so
  there is little headroom either way.
- **Attribution (`tools.lab.attributable_to`, treatment = SoftHebb lift over reservoir, control = shuffled-input
  lift):** on inheritance only 29% (N=3) / 12% (N=4) of the small mean lift is attributable to genuine
  structure-learning — the remaining 71–88% is reproduced by a SoftHebb stack trained on structure-destroyed
  input, so even the positive mean is mostly artifact. On XOR the fraction is >1 (a null — the control moved
  opposite the near-zero treatment). Nothing survives the control cleanly.
- **3 configs are UNDEFINED, not negative** (the precondition-guard doing its job): xor N=4 (BPTT ceiling 0.573
  vs chance+0.05 = 0.574 — a hair under, deep XOR is barely learnable at N=4) and hier3 N=3/N=4 (the BPTT ceiling
  itself sits at chance, 0.17/0.15 vs 0.167 — the depth-3 compositional task is unlearnable by ANY arm at this
  scale, consistent with the Q5 obligatory-depth-3-is-unconstructible finding). With no regime, "SoftHebb vs
  reservoir" is uninterpretable — reporting a negative there would fabricate a result from an instrument failure.
- **FA/KP collapse to majority-class in every config** (the wall, re-confirmed): inheritance/hier3 at chance,
  XOR below chance.
- **Per-layer selectivity**: SoftHebb DOES build rising per-layer class-selectivity on XOR (5/6 seeds the deepest
  hidden layer is more class-informative than the first) — but NOT more than the frozen reservoir does
  (over-reservoir 1/6). A random projection produces layers just as selective.
- **Anti-cheat source-guard**: the SoftHebb update reads only presynaptic `x` and its own pre-activation `u` —
  never a label, a target, or a top-down feedback matrix (`no_label_no_feedback_all=True`, all configs).
- **Determinism**: re-running xor N=3 seed 42 reproduces SoftHebb 0.638 / reservoir 0.649 / BPTT 0.632
  byte-identical (numpy RNG seeded via `default_rng(seed+offset)`; no sim bridge, so the `cfg.seed` trap does not
  apply here).

## What this settles for the gap#4 assault

- **CONFIRMS the July-15 objection under the new metric.** "SoftHebb = unsupervised feature side, not
  task-directed" was written for the old (obligatory-depth) framing; it now holds under enter-the-regime too. A
  fully-UNSUPERVISED soft-WTA objective does not build deep spiking features that beat a random reservoir. This is
  a METHOD-negative — the CAPABILITY (deep spiking credit) is NOT walled: Q1 Forward-Forward and Q4 DECOLLE
  already enter the regime.
- **Sharpens the emerging insight.** The winning ingredient is not "a per-layer LOCAL objective" alone — it is a
  per-layer local objective that carries TASK signal (FF's contrastive goodness on positive/negative data;
  DECOLLE's per-layer supervised readout). SoftHebb removes the task signal entirely and the lift disappears. So
  the two independent cracks (FF, DECOLLE) share TASK-DIRECTEDNESS, not unsupervised competition; SoftHebb is a
  clean control that isolates that.
- **Honest scope.** This tests SoftHebb as a feature-builder read by an optimal linear head, on this LIF
  point-neuron substrate, at matched width 32 (wide-256 reservoirs already solve inheritance — the "denied width"
  caveat applies). It does not claim unsupervised Hebbian competition is useless in general (it separates
  OVERLAPPING categories elsewhere, EMERGE-38); it claims it does not, on its own, ENTER THE REGIME where the
  top-down transport-free rule collapses. Runner-side; NO sim/ edit.
