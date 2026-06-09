> # ⚠️ RETRACTED (2026-06-09, same day) — the "substrate limit" conclusion below is WRONG.
> A CPU forensic (controller-reproduced) shows the deployed nav bridge is **current-identical** to
> the isolation probe and **DOES bootstrap the critic** (afferent→critic weight 0.2005 → 3.31 over
> 40 trials; ≥0.71 by trial 20). There is **no −79.6 mV plateau** (both bridges integrate to ~−69 mV)
> and the Gabor growth does **not** corrupt the afferent→critic synapses. The 1800-step CuPy freeze
> is therefore a **CuPy-path divergence or a measurement artifact**, NOT a substrate boundary — N9 is
> recoverable. See **`2026-06-09-N9-forensic-substrate-is-NOT-the-wall-plus-plastic-mask-bug.md`**.
> This doc is kept for the trail; treat its "robustly mapped substrate limit" verdict as superseded.
> (The factual run measurements below — critic 0/1800, weight frozen 0.20078, nav 2.136 — still hold;
> only the *interpretation* was wrong.)

# N9 nav value-subtraction — warm-up does NOT deploy: the MSN critic can't fire in the full nav bridge (honest negative, robustly mapped) [RETRACTED — see banner]

**Date:** 2026-06-09
**Type:** deployed-nav warm-up smoke (GPU/CuPy, seed 42) + root-cause reconciliation.
**Verdict:** **CRITIC_STILL_SILENT** — the `--critic-warmup-trials 20` deadlock-breaker, validated in isolation, does **not** fire the `striosome_value` critic in the deployed 47-region nav bridge. The neural value subtraction (r − V via GABA_B) does **not** engage. Nav is unaffected (excellent via the raw-reward RPE). This is the **6th probe-vs-deployment gap** of the arc and a robustly-confirmed BRAIN-BASED-ONLY deliverable: it maps a real limit of the substrate-as-assembled.

## The run

Exact command = the validated 6-seed-A/B *neural* condition + the warm-up flag:
```
g11_bg_runner --moving-goal --goal-schedule multi --deterministic
  --enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi
  --enable-cluster-a-closed-loop --enable-cluster-e-topography --enable-dlpfc-wm --enable-pfc-nmda
  --enable-visual-cortex --visual-cortex-action-warmup-steps 600 --grid-size 32
  --spiking-snc --n-steps 1800
  --enable-neural-critic --enable-place-goal-readout --enable-critic-homeostasis
  --critic-warmup-trials 20 --seed 42
```

## Result (research/findings/raw/g11_bg/_n9_warmup_smoke_s42.json, elapsed ~18 min)

| Measure | Value | Read |
|---|---|---|
| `striov_rate_log` (the critic) | max **0.00**, mean 0.0000, **0/1800 nonzero** | critic NEVER fired |
| critic weight init → warmup_post → final | **0.20078 → 0.20078 → 0.20078** | byte-frozen, zero LTP |
| `snc_rate_log` (dopamine) | mean 6.93 Hz, 98.6% active, max 14 | reward system fine |
| `mean_distance_overall` / quarters | **2.136** / [4.11, 1.37, 1.56, 1.50] | excellent nav |
| `n_steps_at_goal` | 828 / 1800 | excellent |
| `global_homeostasis_off` / per-region mask | True / True | deterministic regime correct |

The 20 warm-up trials × 4 goals produced **zero** critic spikes → eligibility stayed 0 → the
`vs_place_context → striosome_value` weight never grew → the GABA_B subtraction onto the SNc was
never carried by a learned V. Nav ran on the raw-reward RPE (= effectively Stage-A; the neural
critic is a no-op).

## Root cause (already pinned in the warm-up code; now confirmed a 3rd time)

`g11_bg_runner.py:4352-4356` (the `_run_critic_warmup` body) documents the prior forensic verbatim:

> *"even 10× drive (g_exc 0.5 > the de-risk's firing 0.35) did NOT fire the MSN-D1 critic in the
> deployed nav bridge (the critic's membrane plateaus at ~−79.6 mV where the byte-identical de-risk
> critic integrates to −71 mV on the same g_exc — the unresolved nav-bridge blocker)."*

So the warm-up's drive reaches the critic, but in the **full 47-region bridge** the critic's
membrane plateaus ~8 mV below where the **isolation** critic reaches on identical input — below
spike threshold. No spike → no STDP post-event → the LTP-bootstrap deadlock is NOT broken. This
is independent of warm-up trial count: the critic cannot fire from the afferent at warm-up time in
deployment, at 1× or 10× drive.

Three independent confirmations now: (1) the 1800-step no-warmup smoke (critic silent), (2) the
10× forensic (crit_spk=0, −79.6 mV plateau), (3) this 20-trial warm-up smoke (0/1800, weight frozen).

## Reconciliation with the integration diagnostic (important honesty note)

The read-only integration diagnostic (`a9342a0bd`) concluded the −71 vs −79.6 mV gap was **"two
points on the same homeostasis-adaptation curve, NOT a bug; probe and nav behave identically; the
warm-up infra is the correct lever."** This deployed-nav test **contradicts the load-bearing half
of that conclusion**: the warm-up is NOT a working lever in deployment — it cannot fire the critic.
Whether the −79.6 mV plateau is "early on the same homeostasis curve" or a genuine deployment-only
suppression, the **operational outcome is the same**: the neural value critic does not engage in the
full nav, and the warm-up does not change that. The diagnostic's "not a bug" framing may still be
correct at the single-neuron level, but its "warm-up is the lever" optimism was falsified by the
deployment run — a reminder that read-only single-neuron diagnostics don't substitute for the
deployed test (the arc's recurring lesson).

## What is banked vs open

**Banked (unaffected):** the GABA_B/GIRK conductance `sim/` edit (Pavlovian de-risk GO,
byte-identical-when-off); the per-region homeostasis `sim/` edit (byte-verified both global states);
the place-critic *isolation* de-risk PASS (the mechanism works in a small bridge); excellent nav
(2.136); the N9 reward-prediction-error **loop is already neural in deployment** via
`--spiking-snc` (the SNc FIRES δ) — it is the neural **value-subtraction** (a learned V critic
subtracting r − V) that does not deploy.

**The honest negative (robustly mapped):** a spiking MSN-D1 value critic reading a place afferent,
with GABA_B subtraction at the SNc, **validates in a small isolated bridge but does not fire in the
deployed 47-region nav bridge** — the critic's membrane plateaus ~8 mV below firing threshold there,
so no learning ever seeds. This maps exactly what the substrate-as-assembled can and cannot do.

## Decision fork (owner steer)

1. **BANK** the negative + move to **N5** (the neural reward *signal* — research done `943a0a6e`,
   `R5-APPROACH-CELL`; the SNc-fires-δ leg is already neural, the residual is the reward formula).
   The N9 RPE loop is neural via spiking-SNc; the value-subtraction limit is a documented, honest
   substrate boundary. **Recommended** for momentum toward "fully biologize nav."
2. **INVESTIGATE** the −79.6 mV deployment plateau (why does the same MSN integrate ~8 mV lower in
   the big bridge?). Per the standing practice this warrants a deep-research / catalog pass first,
   then likely a protected `sim/` investigation (hidden inhibition? a per-synapse-array / driving-
   force interaction unique to the assembled bridge?). Higher cost; if it's a real fixable
   suppression it would unblock N9 *and* be a general fidelity win.

Tools: `research/findings/raw/g11_bg/_n9_warmup_smoke_s42.{json,log}`;
`_navcritic_ab_6seed_warmup.ps1` (the A/B, NOT run — the firing gate failed);
`WARMUP_DEBUG=1` env hook (`g11_bg_runner.py:4390`) for per-trial crit_spk if deeper forensics wanted.
