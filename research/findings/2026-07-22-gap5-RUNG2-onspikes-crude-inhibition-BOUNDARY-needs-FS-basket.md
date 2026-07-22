# gap#5 RUNG-2 (on-spikes theta/gamma replay) — crude external inhibition is NOT robustly achievable (mapped); the next method is FS-basket FEEDBACK inhibition + adaptation

**2026-07-22, numpy CPU, coexisting with the fluency training.** The numpy timing MECHANISM is validated (3/3 GO,
`2026-07-22-gap5-gamma-WTA-timing-fixes-replay-order-cheap-GO.md`): a gamma-WTA + post-fire silence turns RANK 2's
marginal weight-only replay order into a reliable forward order on the learned weights. This is the on-spikes realization
attempt (`_gap5_spiking_gamma_replay_derisk.py`): during the spontaneous-replay REST phase over RANK 2's real BTSP chain +
bistable within-attractors, apply a theta/gamma self-avoidance (detect a reactivated assembly, silence it so the forward
chain drives the next).

## The mapped result — a crude injected inhibition cannot robustly hit the release-vs-suppress window
| inhibition | seed 42 | seed 43 | seed 44 | note |
|-----------|---------|---------|---------|------|
| fixed -4000 | act=[0,0,0] | — | — | over-suppresses (kills reactivation) |
| **fixed -1500** | **GAMMA fwd 0.667, act=[3,3,3]** | act=[0,0,0] | act=[0,0,0] | **works seed 42 ONLY — seed-dependent** |
| proportional (~firing) | act=[0,0,0] | act=[0,0,0] | act=[0,0,0] | scaling over-suppresses all |

- The NO-GAMMA baseline reactivates every seed (act=[2,2,2]/[3,3,3]) at chance/reverse order — the sequencing, not the
  reactivation, is what is missing.
- There is a genuine window (seed 42, -1500: gamma fwd 0.667 > chance 0.500, reactivation ENHANCED to act=[3,3,3], more
  events, NO-NOISE=0, NO-ENCODE=0) — so the mechanism CAN organize the order on spikes. But the window is **too narrow
  and seed-dependent** for a crude external current: too weak and it does not release the bistable within-attractor; too
  strong and it kills the reactivation below the detection peak. Neither a fixed nor a firing-proportional injected
  current hits it across seeds.

## Root cause + the next method (per THE LAW — a verdict on the crude METHOD, not the capability)
This is the on-spikes form of RANK 2's within-vs-chain tension: the **bistable within-attractor** that makes reactivation
robust also RESISTS release, and a crude external inhibition either fights it (killing detection) or under-shoots. The
proper release must SELF-SCALE through the real neural loop, which a hand-injected current cannot. The ranked next method
is the biological one: a **gamma-driven FS-basket FEEDBACK inhibition** (the `ca3_pv_basket` pool already wired with
`ca3_fb_inhib` feedback) — feedback inhibition scales with the assembly's OWN firing through the actual synaptic loop, so
it releases proportionally without the seed-dependent over/under-shoot — combined with intrinsic spike-frequency
adaptation for the per-assembly self-avoidance (de Almeida-Idiart-Lisman E%-max WTA). This is RUNG 3 (a `sim/`-level
gamma-FS-pool build), a deeper multi-parameter mechanism than the RUNG-2 scaffold. Building it next.

## Honest status
- Numpy timing mechanism: **validated** (the principle works on the weights).
- On-spikes RUNG-2 via crude injected inhibition: **NOT robust** (seed-42-only window; mapped across fixed + proportional).
- Next: the FS-basket feedback + adaptation (RUNG 3). The capability (on-spikes ordered/imaginative replay) stays OPEN and
  pursued; this maps the crude method's boundary and names the proper mechanism. Driver + all variants:
  `_gap5_spiking_gamma_replay_derisk.py`; raw `research/findings/raw/gap5_r4/spiking_gamma_{gentle,2seed,prop}.log`.
