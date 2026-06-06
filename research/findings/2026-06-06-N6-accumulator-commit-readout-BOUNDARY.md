# N6 action-selection readout — spiking ACCUMULATE-THEN-COMMIT (Wang-2002 NMDA accumulator → Lo-Wang/SC commit burst): the spiking decision is REAL and decisive, but the cheat-5 nav score is a BOUNDARY (grid-8, seed 42) — 2026-06-06

**Verdict: BOUNDARY.** The host argmax (cheat N6) is replaced by a genuine, biologically-faithful, fully-spiking
**accumulate-then-commit** decision — the winner's accumulator ramps and its all-or-none burst fires (the guard
proves it; 80% of committed actions match the clean thalamic winner). But on the cheat-5 multi-goal nav metric the
spiking readout scores **4.71** vs the host-argmax-over-thalamus target **2.34** (GATE was ≤ ~3). The mechanism
works; matching the host argmax's *score* on this metric does not, for a characterized reason (below).

Opt-in (`--readout-source spiking_wta`), runner-side, **NO `sim/` edits**.

## What N6 is

The nav agent's BG cascade runs genuine GPi→thalamus disinhibition (N8, multi-seed GO;
`2026-06-06-N8N6-combined-readout-GO.md`). Under disinhibition the released `thal_X` is the cleanly-selective
signal (only the chosen action's thal fires; others 0.000). **N6** is the residual: committing that
released-thalamus winner to a motor action *as a spiking decision*, not a host `argmax` over pool rates. The
thal-source readout biologized the SIGNAL but still used a host argmax (target **2.336**, the cheat-5 multi-goal
SUM of per-phase final-quarter mean distance, LOWER better).

## The research diagnosis (`2026-06-06-action-selection-readout-deep-research.md`)

A prior `--readout-source spiking_wta` built `sel_X` selection pools (a read-only `thal_X → sel_X → sel_FS_X →
sel_Y` soft-WTA) but FAILED (28.0 / 7.0 / 5.87). The deep-research finding diagnosed why and prescribed the fix:

- The `sel_X` pools had `internal_density=0.0` / `exc_weight_mean=0.0` — **NO recurrent self-excitation** → a
  PASSIVE INSTANTANEOUS COMPARATOR, which provably cannot manufacture a winner from a weak signal.
- The brain commits decisions in **TWO STAGES**: (1) ACCUMULATE — a recurrent attractor (Wang 2002 *Neuron*;
  network τ = τ_syn/|1−w_rec|, NMDA-slow) amplifies + integrates the weak released drive to a bound; (2) COMMIT —
  a downstream tonically-inhibited burst structure (Lo-Wang 2006 *Nat Neurosci* SC threshold; Stine-Shadlen 2023
  LIP-accumulate/SC-commit) fires ALL-OR-NONE on threshold crossing. The host-argmax stands in for BOTH stages at
  once, instantaneously — which is why an instantaneous WTA on a weak signal cannot replace it.
- The recurrence must be a STABLE soft-WTA (gain α < 1, Rutishauser-Douglas-Slotine 2011) with STRUCTURED
  (per-pool), not symmetric-blanket, inhibition.

## The design implemented (additive, read-only, NO `sim/` edit)

Extended the existing `sel_X`/`sel_FS_X` layer in `build_bg_brain_regions` (gated by `readout_source="spiking_wta"`):

1. **ACCUMULATE** — each `sel_X` pool now has NMDA-SLOW recurrent self-excitation (`sel_recurrent_density=0.5`,
   `sel_recurrent_weight=1.0`) and `enable_nmda=True`. The sel slice (and only it, since `--enable-pfc-nmda` is off
   here) is put in the bridge's per-region `cp_nmda_neuron_mask` (the SAME mechanism `enable_pfc_nmda` uses), so the
   recurrence integrates with NMDA τ_decay=100ms — the biological integration constant (Wang 2002). `cfg.enable_nmda`
   is turned on globally + `nmda_ratio=0.5` for the accumulator. The feedforward `thal_to_sel_weight` was lowered
   60→30 (modest EVIDENCE, not instant saturation) and the cross-inhibition `sel_fs_to_sel_weight` lowered 28→5
   (GENTLE — see "diagnosis trail" below).
2. **COMMIT** — a downstream `commit_X` burst pool (SC / saccade-generator EBN analogue, H.24/H.25) driven by
   `sel_X → commit_X` (`sel_to_commit_weight=22`), with its own recurrence (`commit_recurrent_weight=0.6`) for the
   all-or-none regeneration. **The decision = which `commit_X` bursts (threshold crossing), NOT a host argmax over
   graded rates.**
3. **Fallback chain:** commit burst (primary — the threshold crossing) → sel accumulator's leading pool (sub-threshold
   lean; Shadlen affordance / Stine 2023 — the accumulator keeps a candidate even when the SC burst hasn't fired) →
   random (only if both silent). This replaced the old random-only fallback and was the single biggest score lever
   (8.97 → 4.71).

**The shared `commit_OPN` omnipause pool (the textbook SC/OPN gate, H.24) is built and correctly inhibitory, but
DEFAULT OFF (`commit_opn_tonic_pa=0`):** a CONSTANT OPN drive induces SYNCHRONIZED REBOUND BURSTING across all commit
pools on this rate-coded substrate (the symmetric-inhibition instability — verified: 500pA → all commit fire; 200pA →
none fire; no stable constant middle). The commit threshold is instead enforced by the sel→commit drive + the commit
pool's intrinsic IZH threshold (the deep-research finding's documented "minimal variant"). Stated honestly, not hidden.

## Diagnosis trail (why the parameters are what they are)

A per-pool conductance trace (`N6_DEBUG`) made the calibration empirical, not guesswork:
- The gain-0 layer AND the first recurrent version had **all four sel pools saturate to 700** even though only ONE
  `thal_X` fires (the thalamus IS cleanly selective: winner ~30-50 spikes, losers 0). Cause: the original
  `thal_to_sel_weight=60` + strong symmetric cross-inhibition (`sel_fs_to_sel_weight=28`) drove a SYNCHRONIZED
  POPULATION OSCILLATION — every pool fired together, inhibited everyone (gi hit 450+, v → -170), rebounded, repeat →
  no winner. This is exactly the Rutishauser "symmetric mutual inhibition is unstable" warning.
- Fix: lower feedforward (30) + GENTLE structured inhibition (5) + modest NMDA recurrence (1.0). Then the sel layer
  became selective (winner ramps, losers ~0) and the commit burst fired decisively for the winner only.

## Smoke arc (grid-8, seed 42, multi-goal; `--genuine-thal-disinhibition --genuine-gpi-tonic-pa 1300 --genuine-thal-tonic-pa 750 --readout-source spiking_wta`)

cheat-5 SUM (sum of per-phase final-quarter mean distance, LOWER better; GATE ≤ ~3):

| readout / config | SUM | per-phase | commit-silent | note |
|---|---|---|---|---|
| host argmax over thalamus (the target/reference) | **2.336** | [0.51, 0.52, 0.54, 0.76] | — | what we are trying to match |
| gain-0 passive comparator (prior FAIL) | 28.0 / 7.0 / 5.87 | — | — | the research-diagnosis subject |
| accumulate-then-commit, random fallback (smoke #1) | 8.97 | [1.06, 1.33, 1.70, 4.89] | 47% | 47% silent commit → random |
| **accumulate-then-commit, sel fallback (smoke #2) [DEFAULT]** | **4.71** | [0.64, 0.84, 1.49, 1.74] | 34% | best; the production default |
| stronger ramp (rec 1.8, commit-thr 30) (smoke #3) | 5.68 | [0.50, 0.97, 2.13, 2.09] | 36% | stronger recurrence WORSENS full-run (hysteresis) |
| + per-trial accumulator reset (smoke #4) | 6.93 | [0.66, 1.14, 1.58, 3.55] | 55% | reset removes carried drive → MORE silent commit, WORSE |

Best (and the shipped default) = **smoke #2, 4.71.** The early stable-goal phases (1-2) are excellent
(0.64 / 0.84, essentially at the reference); the cost is concentrated in the post-goal-change phases (3-4),
where the thalamic drive is weaker/messier and the accumulator either rams the old winner (hysteresis) or
goes sub-threshold (sel fallback).

## The accumulation/commit GUARD — the decision is REAL, not an argmax

Per-substep trace, smoke #2 trial 1 (a clean E-winner trial). The winner's `sel_E` accumulator RAMPS while the
three losers stay at exactly 0, and the `commit_E` burst fires as the ramp crosses:

```
substep | winner sel cum | mean-loser sel cum | commit burst (winner)
   10    |      20        |       0.0          |   0     (ramping)
   30    |      42        |       0.0          |   0
   50    |      70        |       0.0          |   3     (crossing → burst)
   70    |     103        |       0.0          |   1
   99    |     144        |       0.0          |   0
```

Windowed over the whole run: **commit burst winner mean 15.3 vs runner-up mean 0.0 (separation ~500×); sel
accumulator winner 15.2 vs runner-up 0.1.** When the burst fires it is DECISIVE and SELECTIVE — a thresholded,
accumulated, committed winner, not an instantaneous comparator. **80% of committed actions match the clean
thalamic winner** (1181/1486). At short horizons / stable goals the commit fires on ~100% of trials and alignment
is 90-96%; the residual 20% are the goal-change phases (silent commit → sel fallback, or one-trial hysteresis).

## Why it is a BOUNDARY (the honest reason the score doesn't reach ≤3)

Two intrinsic, characterized properties of a *spiking* accumulate-then-commit readout — both ABSENT from a host
argmax — cost exactly the goal-change phases:

1. **Cross-trial NMDA hysteresis.** The NMDA-slow accumulator (τ=100ms) persists ~one full inter-trial, so when the
   goal/thal switches the previous winner lingers for a trial (a working-memory latch — biologically real, but it
   mis-commits at the boundary). Resetting it each trial (smoke #4) removes the carried drive that helps the burst
   fire → commit goes silent on 55% of trials → WORSE. So the persistence is kept (net better) and its boundary cost
   is accepted.
2. **Sub-threshold silent-commit in weak-drive phases.** When (late-phase, plasticity-leaked) thalamic drive is weak,
   the accumulator doesn't ramp to the burst bound within the 100ms window → no commit → the sel-fallback fires
   (the accumulator's lean, ~80% correct, but not the decisive burst). A host argmax has no threshold to miss.

The host argmax is *instantaneous and threshold-free*, so it never pays either cost — which is precisely the
deep-research finding's point that the argmax compresses both stages into one zero-latency operation. The spiking
mechanism pays a small, biology-faithful price for being a genuine accumulate-then-commit circuit. **The decision is
biologized; the cheat-5 SUM lands at 4.71 (boundary), not ≤ the 2.34 host-argmax target.**

## Honest scope

- Seed 42, grid-8 smoke (the CONTROLLER runs multi-seed). Single-seed.
- The commit-OPN omnipause gate is structurally faithful (built, inhibitory) but OFF by default (rate-coded rebound
  instability); the commit threshold is the sel→commit + intrinsic-threshold "minimal variant".
- Stronger recurrence and per-trial reset were both tried and are NET NEGATIVE on the full multi-goal run (kept as
  opt-in flags `--sel-recurrent-weight`, `--reset-accumulator`).
- NO `sim/` edits anywhere. Reuses `BrainRegion` / `RegionPathway` / the per-region NMDA mask
  (`cp_nmda_neuron_mask`) exactly as `enable_pfc_nmda` does.

## What would move BOUNDARY → GO (next directions, not done here)

- An **urgency / collapsing-bound** signal (Cisek 2009; Thura-Cisek 2014) so weak-drive late phases still commit
  within the window (fixes silent-commit without the reset's lost-drive penalty).
- A **gentle reset of only the LOSING pools' NMDA** at trial start (clear hysteresis without zeroing the eventual
  winner's carried drive).
- Tuning the readout window / `commit` intrinsic threshold per phase, or a DA-modulated speed-accuracy bound
  (Lo-Wang: the threshold is the cortico-striatal weight) — the biologically-principled knob, left for the controller.

## Files

- `research/runners/g11_bg_runner.py` — `--readout-source spiking_wta` now builds the accumulate-then-commit layer.
  New flags: `--sel-recurrent-density`, `--sel-recurrent-weight`, `--no-commit-burst`, `--n-commit-per-action`,
  `--n-commit-opn`, `--sel-to-commit-weight`, `--commit-recurrent-density`, `--commit-recurrent-weight`,
  `--opn-to-commit-weight`, `--commit-opn-tonic-pa`, `--reset-accumulator`. Result dict adds `commit_counts`,
  `accum_trace`, `use_commit_readout`, and a thal-guard (thal_counts logged under spiking_wta).
- `research/findings/raw/_n6_accum_commit_smoke{1,2,3,4}_*.json` — the smoke arc (#2 = default config).
- `research/findings/raw/_n6_analyze.py` — analysis (score + thal-alignment + accumulation/commit guard).
