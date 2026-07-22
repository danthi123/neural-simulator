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

## FS-basket FEEDBACK inhibition SOLVES the reactivation-preservation (the crude approach's fatal flaw); ordering is the residual
Built `--fs-gamma` (gamma-rhythm drive of the `ca3_pv_basket` -> feedback inhibition self-scales via the real synaptic loop):
- **FS-basket ALONE:** reactivation PRESERVED + ENHANCED (act=[5,5,5], 13 events vs baseline's 5 -- the self-scaling
  feedback does NOT over-suppress, unlike the crude injected current that gave act=[0,0,0]) but fwd=0.400 (no ordering --
  the windows gate timing but the strongest within-attractor re-wins each window; no self-avoidance).
- **FS-basket + gentle self-avoidance (-800), 3-seed:** 1/3 -- reactivation preserved every seed (act=[5,5,5]/[5,5,5]/[1,1,1]),
  and seed 43 orders FORWARD (fwd 0.600 > chance 0.500, GO), but seeds 42 (reverse-leaning) + 44 (weak) fail. ⇒ the
  FS-basket self-scaling FIXED the over-suppression (the crude approach's fatal flaw), but robust FORWARD ORDERING across
  seeds is the residual -- it is seed-dependent (the on-spikes order does not track the numpy asym; the WTA-timing + gentle
  silence orders on some seeds, not all).
- **⇒ characterized state:** windows ✓ (FS-basket), reactivation-preservation ✓ (self-scaling feedback), ordering PARTIAL
  (1/3, seed-dependent). The next lever for ROBUST ordering is **theta PHASE-PRECESSION** (Skaggs-McNaughton: encode the
  sequence order in theta PHASE, read by phase rather than by a fragile WTA-timing race) -- the mechanism that makes
  ordered replay robust in biology, and the same lever RANK 2's 4/6 wants. The on-spikes ordered-replay capability stays
  OPEN with the mechanism space now thoroughly mapped (numpy-principle ✓, crude-inhib ✗, FS-basket-reactivation ✓,
  FS-basket-ordering partial, theta-precession = next).

## FS-basket parameter sweep EXHAUSTED (gamma_period 25 x inhib {-500,-800,-1200}, 3-seed each) — ordering NOT robust; theta-precession confirmed as the required lever
The FS-basket + self-avoidance forward ordering is 0/3, 1/3, 0/3 across the ~40Hz gamma sweep (and 1/3 at gp12) — a stable
~1/3 ceiling at every operating point. ⇒ the FS-basket self-scaling reliably fixes reactivation-preservation, but robust
forward ORDERING via a WTA-timing race (which assembly wins each gamma window) is NOT achievable across the parameter
space on the spiking substrate. **Decisive conclusion: robust on-spikes ordered replay needs THETA PHASE-PRECESSION** —
encode the sequence order in theta PHASE during the BTSP chain (Skaggs-McNaughton), then read the order by phase rather
than racing a WTA each window. This is the biologically-robust ordered-replay mechanism (and the same lever RANK 2's 4/6
wants). The FS-basket WTA-timing approach is now exhaustively mapped as insufficient; theta-precession is the next build.
Mechanism space fully mapped: numpy-principle ✓ | crude-inhib ✗ (over-suppress) | FS-basket-reactivation ✓ (self-scaling) |
FS-basket-ordering ✗ (0-1/3, exhausted) | theta-precession = the confirmed next method.
