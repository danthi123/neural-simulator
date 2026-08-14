---
status: live
type: finding
lane: laneC
date: 2026-08-14
---

# GNW Rung-2b — spike-frequency-adaptation (SFA) eviction on an async competitive workspace: BOUNDARY (6/6 no clean salience-eviction; the intrinsic-adaptation fatigue that would evict the dense recurrent attractor instead self-extinguishes it, so a more-salient challenger co-ignites rather than displacing it — the named next mechanism is a transient salience-gated dis-inhibition pulse)

**Date:** 2026-08-14
**Runner:** `research/runners/_gnw_rung2b_sfa_workspace_eviction_derisk.py` (reuse-by-import of the Rung-2 `build_competitive_bridge` + Rung-1 assembly-loop/snapshot helpers; SFA applied by writing `cp_izh_d_increment`/`cp_izh_a` on the workspace region after build — config-frozen, **NO `sim/` edit**).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** a decisive, well-anti-cheated **method-negative** — *intrinsic* Izhikevich spike-frequency adaptation is NOT the eviction effector on the async competitive workspace. This is a verdict on the METHOD, not on the CAPABILITY: salience-eviction is NOT abandoned; the next method (a transient salience-gated dis-inhibition pulse) is named below.
**Builds on / cites:** the Rung-2 PENDING finding `research/findings/2026-07-07-GNW-rung2-competitive-access-mutual-exclusion-GO-salience-eviction-PENDING.md` (mutual-exclusion 6-seed GO; salience-eviction + causal-swap PENDING, naming SFA as next) and **(Dehaene & Changeux, 2011)**, *Neuron* 70(2):200-227 (metastability: an ignited workspace state must be able to be "destabilized" and "spontaneously replaced by another"). No external-literature search was run this session — `NO-EXTERNAL-NEEDED:` the mechanism, its Izhikevich basis, and the named next method are all fixed by the corpus (the Rung-2 finding + the banked affect-lane eviction negatives) and the cited source; this is a method-negative that launches the next build, not a capability wall.

## What was tested (the one limb Rung-2 never tried)
The workspace neuron model is Izhikevich-2007, so SFA is the intrinsic spike-triggered recovery increment: after each spike `u += d`, with `du/dt = a·(b·(v−vr) − u)` — larger `d` = stronger fatigue, smaller `a` = slower decay (more accumulation). The SAME `d`/`a` were written onto BOTH assemblies A and B (one init-invariant fatigue rule), the FS pool kept its fast-inhibition params, `enable_homeostasis`/`enable_short_term_plasticity` stayed OFF. The de-risk searched `(izh_d, izh_a, fs_to_ws, ou_noise, incumbent_settle)` for the window where the incumbent ignites, HOLDS a weak challenger, and is EVICTED by a strong one (leaving exactly one content ignited).

## Result — no such window exists (the dichotomy is complete and robust)

<!--derived-->
Heterogeneity raises the ignition knee (drive 2500 is now sub-threshold; ~4000-5000 ignites), and the ignited assembly is a **rigid period-3 rate attractor (exactly 1/3) or nothing** (the cited seed JSONs' a_rates/b_rates read 0.3333333333333333) — het+OU up to 120 pA did NOT desync it into the graded async rate the mechanism assumed. Against that all-or-none attractor, intrinsic SFA has only two outcomes, and the swept SEARCH RANGES (method params, not measured results) — `d` 60→600, `a` {0.005,0.01,0.02,0.03}, `fs` 16→48, `ou` {0,20,40,120}, `settle` 60→200, `drive_inc` 2500→5000 — found no third:

<!--derived-->
- **`d` low enough to hold** → the incumbent is un-evictable: its rate holds at the ~1/3 plateau through the whole window (console 20-step bins 0.31–0.35) regardless of challenger strength or persistence; a strong challenger **CO-IGNITES** (both → 1/3, mutual exclusion breaks) rather than displacing it.
- **`d` high enough to threaten it** → the incumbent **SELF-EXTINGUISHES alone** (n_ignited=0, no challenger present); any subsequent "B win" is B igniting into an *empty* workspace and is itself phase-erratic — not an eviction.

The fatigue that would evict equals the fatigue that kills. At the frozen hold-point (d=400, a=0.03, fs=28, ou=40, settle=150, drive_inc=5000, chal in [0,8000]) the incumbent A wins the ENTIRE sweep on 5/6 seeds and co-ignites at the top on seed 43 — never a single clean "B":

| seed | winners[chal 0→8000] | mutual excl | takes_strong | causal_swap | continuous takeover | n_ignited post |
|---|---|---|---|---|---|---|
| 42 | AAAAAAAAA | ✓ | ✗ | 0.00 | ✗ | 1 |
| 43 | AAAAAAAAX | ✗ (co-ignite) | ✗ | 0.00 | ✗ | 2 |
| 44 | AAAAAAAAA | ✓ | ✗ | 0.00 | ✗ | 1 |
| 100 | AAAAAAAAA | ✓ | ✗ | 0.00 | ✗ | 1 |
| 101 | AAAAAAAAA | ✓ | ✗ | 0.00 | ✗ | 1 |
| 102 | AAAAAAAAA | ✓ | ✗ | 0.00 | ✗ | 1 |

Smoke grid (seed 42, `_gnw_rung2b_smoke42.json`) makes the dichotomy explicit: d=200 → holds but co-ignites (mutual_excl False); d=400/a=0.01 and all d=600 → n_ignited=0 (self-extinction). `any_op_go=False`.

## Anti-cheats (each held — this is a real negative, not an artifact)
- **SFA is load-bearing / live:** high-`d` self-extinguishes the assembly (n_ignited=0) — the injected adaptation demonstrably drives the dynamics; it simply cannot thread the needle.
- **WTA lesion (`fs_to_ws=0`) co-ignites** (both A and B ignited at the 1/3 plateau; lesion A/B in each cited seed JSON under control_wta_lesion) on all 6 seeds → mutual exclusion is caused by the shared inhibition, not by SFA silencing one assembly.
- **SFA-OFF (RS d/a) reproduces the phase-erratic negative** on all 6 (monotone-but-no-crossover / causal_swap fail) — consistent with Rung-2; and SFA-ON does not rescue it.
- **Scaffold removed / continuous:** the headline takeover attempt ran with ZERO `_restore_state` calls (`no_restore_calls=True` all 6) — the eviction failure is genuine fatigue-dynamics, not a missing per-hop wash-out. This confirms the continuous no-reset protocol works; it is the SFA effector that fails.
- **Anti-annihilation gate is exactly what fires:** the "clean-eviction" outcome collapses into either n_ignited=2 (co-ignition) or n_ignited=0 (self-extinction), never 1-that-is-the-challenger.
- **Determinism:** build-twice at one seed → identical hash of the seed-derived Izhikevich params (heterogeneity seeded from `cfg.seed`, NOT `actual_seed_used`) on all 6.

## Why the banked affect-lane negatives now TRANSFER here (they didn't before)
The affect-lane eviction negatives (`2026-07-31-affect-eviction-slow-GABAB-KILLED...`, `2026-08-01-affect-ratchet-STP-annihilates...`) were on a single self-exciting pool with no competitor. The GNW workspace is competitive, so the a-priori argument was that incumbent silencing hands the workspace to a challenger. It does not: the dense recurrent NMDA attractor is too robust for an *intrinsic* per-neuron adaptation current to grade down — between "un-evictable hold" and "self-extinction" there is no graceful-yield regime, and the async desync that would have made the rate gradable never materialized at these knobs. So the bistable-latch negative transfers to the competitive async rate regime too.

## The named next mechanism (this launches the next build; it is NOT a wall)
Per the Rung-2 diagnosis and this result, the eviction effector should be a **transient salience-gated dis-inhibition pulse** — a phasic attention-shift *release* of the shared inhibition, biologically thalamic-reticular / pulvinar gating of the workspace (Dehaene-Changeux's destabilization driven from *outside* the fatiguing assembly rather than from its own recovery current). This is the competitive, brain-based analogue of the banked active-clear FS quench, but **salience-DRIVEN** rather than a host reset: the more-salient challenger's drive gates a brief dip in `fs→ws` inhibition onto the incumbent's slot, letting the challenger capture the freed workspace while the incumbent's own recurrence (now unsupported) collapses. That is Rung-2c.

## Do-NOT-retread (banked)
GABA_B eviction KILLED on a valid sweep; STP annihilates across all τ_d/U; active-clear FS quench works but is a host shortcut; ALIF adaptation negative as a memory feature on open text; **and now: intrinsic Izhikevich SFA cannot evict the dense recurrent workspace attractor — it self-extinguishes before it yields.**

## Remaining scaffolds (named, not claimed closed — would still stand on a future PASS)
1. **Salience is host-supplied external drive** — an emergent salience (a value/surprise organ writing the drive) is a later rung.
2. **The assemblies are hand-wired** (dense E→E loop at fixed weight), not self-organized.

## Files
`research/runners/_gnw_rung2b_sfa_workspace_eviction_derisk.py`; `research/findings/raw/_gnw_rung2b_seed{42,43,44,100,101,102}.json` (6-seed frozen-point); `research/findings/raw/_gnw_rung2b_smoke42.json` (dichotomy grid).
