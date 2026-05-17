# Generator G1.5 — order-sensitive trajectory readout: honest NEGATIVE (and it sharpens the route to P)

## TL;DR

G1.5 — the cheapest pre-staged FAIL-branch probe after the G1
negative: a *readout-only* change (decode the ordered trajectory of
the pool's response in the un-driven gap after each slot, instead of
one argmax of the final residual; no architecture rewrite) —
**FAILED its pre-registered held-out gate, and the signal is *worse*
than G1's, not better.** This is a real, honest negative. More
importantly, its pre-registered Step-0 calibration data **falsifies
the next pre-staged branch (G1.6) too**, legitimately pruning the
decision tree by evidence straight to **P (predictive-coding
top-down)**.

## The pre-registered gate result (FIXED bars, never touched)

`song_g1_gate.py --readout trajectory` on the trained
trajectory-regime checkpoint (epoch 59), held-out props only,
sidecar-frozen trajectory-regime floor `g1_abstain=46.0` (NOT 650,
NOT G1's final-regime 72.0, NOT recomputed), readout MATCH asserted,
`meta_smoke=False`, bars `_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5`
untouched:

| held-out prop | true | best_perm | gate_cleared | top_rate | verdict |
|---|---|---|---|---|---|
| int=4 `old hard` | 0.000 | 0.000 | N (11.0 < 46.0) | 11.0 | FAIL |
| int=5 `ride smell` | 0.000 | 0.000 | N (18.0 < 46.0) | 18.0 | FAIL |

Aggregate (`g1_verdict` on the means): mean_true 0.000,
mean_best_perm 0.000, 0/2 cleared → **GATE: FAIL**. Full 60-epoch
protocol: `mean_reward=0.0000` every epoch (n_gate_cleared ≈ 0).

## The decisive, decision-relevant evidence (why this routes PAST G1.6 to P)

The hypothesis G1.5 tested — "the order signal exists in the
substrate's dynamics but was discarded by G1's argmax-of-final
readout" — is **falsified, and inverted**:

- G1 final-readout Step-0: encoded (intended order) 73.2 vs
  control-max 72.0, **AUC 0.775** (thin, but encoded ≥ control).
- G1.5 trajectory-readout Step-0: encoded **14.0** vs control mean
  18.8 / max 46.0, **AUC 0.40**.

**AUC 0.40 is anti-signal** (< 0.5): under a richer, ordered
trajectory readout, intended-order productions decode *worse* than
scrambled/random order. A better readout did not recover an order
signal; there is none to recover.

Crucially, the Step-0 "encoded" measurement uses **direct ignition
of the intended order — a perfect controller, zero cold-start, zero
learning**. It still cannot be distinguished from scrambled order by
the self-comprehension judge, under *either* readout (final: thin
0.775; trajectory: anti 0.40). Therefore the bottleneck is provably:

- **NOT the controller's learning** (a perfect, directly-ignited
  controller already fails) → so it is **not a cold-start /
  sparse-reward problem**, which is exactly what the next pre-staged
  branch **G1.6 (developmental scaffolding) targets**. G1.6's premise
  is falsified by G1.5's own pre-registered calibration data.
- **NOT the readout** (both the final-residual and the
  ordered-trajectory readout fail; trajectory is worse).
- **The recognition-only G.20 substrate's response simply does not
  encode recoverable sequence ORDER.** Igniting concepts A→B vs B→A
  leaves no reliably decodable difference. There is no bottom-up
  order trace for any controller or readout to exploit.

This is a legitimate, evidence-based pruning of the pre-staged
decision tree (not skipping rigor): the cheap probe's *own*
pre-registered data falsifies both the readout hypothesis (G1.5) and
the cold-start hypothesis (G1.6). The falsify-cheaply discipline
worked exactly as designed — two hypotheses ruled out for the cost of
one probe.

## Route: directly to P (predictive-coding top-down)

The only remaining branch addresses the *actual* diagnosed cause: the
substrate has **no generative model that represents or predicts
order**. Neither a better controller (G1, G1.6) nor a better readout
(G1.5) can substitute for machinery the architecture does not have.
P — Rao-Ballard top-down generative + prediction-error pathways on
the concept cortex (Friston active inference; Bastos 2012 canonical
microcircuit) — *adds* that machinery: a top-down generative model
that predicts the next concept and whose prediction error is the
order-sensitive learning signal the bottom-up substrate cannot
provide. This is the design's explicitly pre-registered deep "FAIL"
branch; G1+G1.5 have now justified it by evidence rather than
assumption, and pruned the cheaper intermediate (G1.6) for free.

## Anti-cheat discipline (maxed-integrity negative, again)

`g1_verdict`/`score_order`/`permuted_order_controls` bars
(`_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5`) NEVER touched. 650 never
used. The trajectory-regime floor was pre-registered control-max,
frozen to an isolated sidecar, **never recomputed at gate time**;
cross-readout/cross-smoke sidecar reuse is hard-refused.
`--readout final` is byte-identical to G1 (G1's recorded negative
stays reproducible). The no-harm re-proof PASSed (13/13) — the
additive trajectory code does not regress the validated path. The
controller was never config-cranked; the full pre-registered
60-epoch protocol ran to completion. Maxed-integrity honest negative.

## The robust validated asset is unchanged

Generation remains unproven. The trustworthy grounded continual
memory + no-confabulation abstention (G.20 sparse ensemble,
160@100%/320@98.4% multi-seed; CLS no-forgetting) is untouched; the
no-harm probe re-proved the songbird/readout code does not regress
it.

## Files

- `research/runners/song_g1_ignite.py`
  (`ignite_and_trajectory_decode`, write-only),
  `song_g1_train.py`/`song_g1_gate.py` (`--readout` mode, isolated
  `song_g1.traj.*` namespace, regime-recalibrated frozen floor)
- Evidence: `research/findings/raw/g11_bg/song_g1_traj_train.log`,
  `song_g1.traj.ckpt.npz` (+ `.meta.json`: smoke=False,
  readout=trajectory, g1_abstain=46.0, traj_rate_rule=min),
  `song_g1_traj_gate.json`
- Plan/design: `docs/plans/2026-05-16-generator-G1.5-trajectory-readout-implementation.md`,
  `docs/plans/2026-05-16-generative-G1-followup-branches-design.md`
- Prior: `2026-05-16-generator-G1-songbird-NEGATIVE.md`
