# Nav Stage-B neural-critic + GABA_B — 2-seed smoke NEGATIVE (SNc silenced; honest negative + diagnosis)

**Date:** 2026-06-08
**Type:** Honest NEGATIVE de-risk (the cheap-first smoke gate failed → the 6-seed A/B was NOT run, per the pre-registered gate).
**Predecessors:** `2026-06-08-gabab-girk-stageB-derisk-GO.md` (the GABA_B edit + CPU de-risk that PASSED); the nav integration build (commit pending, runner-side only, `sim/` byte-empty).

## What was tested

The BRAIN-BASED-ONLY nav completion: `--enable-neural-critic` adds a dedicated GABAergic `striosome_value`
critic (driven by the perceived ventral object code `cortex_it`, plastic afferent trained by the SNc δ),
routes `striosome_value → snc` `receptor="gaba_b"` + `cfg.enable_gabab=True`, and DROPS the host
`−snc_value_gain·_V_scaffold` term in `_I_snc` so the value subtraction `r − V` happens at the SNc membrane
via the neural GABA_B/GIRK inhibition. Smoke = flagship A+E+G v2.5 + `--spiking-snc --enable-neural-critic`,
seed 42 (a complete 1800-step run; seed 43 not needed once seed 42 failed decisively).

## Result — NEGATIVE (the critic config catastrophically fails)

| Metric | Stage A (`--spiking-snc`, host scaffold) | Neural-critic (`--enable-neural-critic`) |
|---|---|---|
| SNc firing (`snc_rate_log` mean) | **7.16 Hz** (max 13) | **0.0 Hz** (all zero) |
| Critic `striosome_value` firing (reward windows) | n/a | **0.0** (all zero in the logged windows) |
| Critic weight (`cortex_it→striosome_value`) | n/a | **2.990 → 2.990 (no learning)** |
| Nav: summed final-quarter distance | **2.0** | **133.0** |
| Nav: steps at goal | (good) | **0** (never reached the goal) |

The neural-critic path **silenced the SNc** (0 Hz vs Stage A's 7 Hz). With the SNc silent, the dopamine signal
(`from_region_firing_signed` on the SNc pool) is dead → no three-factor learning anywhere → the critic weight
stays frozen at init AND the actor cannot learn → navigation collapses (distance ~32 on a 32×32 grid, never
reaching the goal). This is an honest negative: it maps a real integration failure, **not** a property of the
GABA_B conductance itself (which passed its CPU de-risk 3/3 and is byte-identical-when-off).

## Diagnosis so far (the paradox + the prime suspect)

The puzzle: the neural-critic drive `_I_snc = snc_tonic(220) + snc_reward_gain·max(0,reward)` is **larger**
than Stage A's `220 + reward − snc_value_gain·_V_scaffold` (Stage A *subtracts* the host value; the neural
path drops that subtraction). A *larger* depolarizing drive producing *zero* firing means **something is
adding strong hyperpolarization** in the neural-critic path. Candidates, in order of suspicion:

1. **GABA_B over-inhibition from a critic that fires outside the logged windows (PRIME SUSPECT).**
   `striov_rate_log` only counts critic spikes during the brief reward-hold windows (runner ~line 5268), so
   `striov_rate_log == 0` does **not** prove the critic never fires — it may fire during the regular nav
   steps (driven by `cortex_it`), pumping the **slow** GABA_B conductance (τ = 150 ms) on the SNc. The slow
   decay would sustain a hyperpolarizing K⁺ current (E_K = −90 mV) across many steps → SNc silenced. The CPU
   de-risk did NOT expose this because it was a clean train-then-test Pavlovian schedule, not a continuous
   free-running nav loop with persistent cortical drive to the critic.
2. **SNc readout index** (`_snc_idx_host`) reading the wrong neurons after the `striosome_value` region was
   added — but this is unlikely the whole story, because nav itself collapses (the RPE really is dead, not
   just mis-measured).
3. **A per-action DA / synapse-mask misalignment** from the added region/synapses shifting the synapse count.

## Next — instrumented root-cause (NOT brute-force)

A focused diagnostic must localize the SNc silence (the load-bearing question, since the fix depends on the
cause): a short (~100–200 step) neural-critic build that logs `cortex_it` / `striosome_value` / `snc` firing
**every step** (not just reward windows), the GABA_B conductance magnitude on the SNc, and a NaN check —
compared against the same with `cfg.enable_gabab` forced off (critic on) and against Stage A. Likely fixes if
the prime suspect holds: a much faster GABA_B τ for the nav loop, a weaker `gabab_propagation_strength`, gating
the critic→SNc GABA_B to the reward window only, or rebalancing the SNc tonic. An honest "the slow GABA_B
subtraction is incompatible with the continuous nav loop without further mechanism" would itself be a valid
deliverable (it maps a limit of porting the Pavlovian-validated mechanism into free-running navigation).

## ROOT CAUSE — RECOVERED (the instrumented diagnostic; supersedes the "prime suspect" above)

A diagnostic subagent instrumented this before crashing on an API-500; its artifacts
(`research/findings/raw/_neural_critic_snc_diag.py` + `_neural_critic_snc_diag_result.json`,
`_gabab_mask_probe.py`) were recovered + the mask probe re-run by the controller. The recovered evidence
**refutes the "slow GABA_B accumulates" prime suspect** and finds TWO distinct blockers:

**Decisive 3-condition nav trace (150 steps each, every-step probe):**

| Condition | `enable_gabab` | SNc rate | `g_gabab` on SNc | nav dist | NaN | cortex_it rate |
|---|---|---|---|---|---|---|
| A (failing) | ON | **0.0 Hz** | **0.0** | 50.9 (broken) | none | **0.0** |
| B (gabab forced off) | OFF | **8.16 Hz** | 0.0 | **0.5 (works!)** | none | **0.0** |

**SNc-tonic-only isolation (critic forced silent, drive SNc 220 pA, 200 steps):** `enable_gabab` ON → SNc
50 Hz; OFF → 45 Hz; `g_gabab` on SNc = 0.0 both. So **GABA_B does not directly silence the SNc** — with the
critic silent the conductance is zero and the SNc fires either way. The mask is **correct** (407 synapses, all
PRE in `striosome_value`, all POST in `snc`; E_gabab=−90 on the SNc).

**Blocker 1 — `enable_gabab=True` destabilizes the FULL nav with `g_gabab`=0.** Condition A vs B differ only
in the flag, the GABA_B current is provably zero in both (`gabab_all_max`=0, no NaN, V_snc normal at −52.6
mV), yet A silences the whole network and B navigates perfectly (dist 0.5). So the breakage is a subtle
*system-level* interaction of enabling GABA_B in the full step pipeline (NOT the conductance hyperpolarizing
the SNc — the isolated SNc fires fine with it on). Mechanism not yet pinned (a deeper dig was cut off by the
subagent crash); it is reproducible and flag-gated.

**Blocker 2 — the critic's afferent `cortex_it` never fires in nav (`it_mean`=0 over 16k steps), so the
critic has no input and cannot learn V** (weight frozen 2.990→2.990; `striov`=0 in both A and B). `cortex_it`
is the **position-invariant ventral "what" stream** — the exact pathway CLAUDE.md records the nav was
root-caused to AVOID (spatial value needs the **dorsal "where"** stream / place cells). So the afferent choice
is doubly wrong: inactive in nav AND position-invariant. Even with Blocker 1 fixed, the critic as wired cannot
learn a spatial value.

## Honest verdict

The GABA_B/GIRK membrane-subtraction validated in the **Pavlovian** CPU de-risk (clean cue→reward schedule,
3/3 PASS) does **not** port to the **continuous free-running nav loop as designed**. Two independent blockers:
a flag-level instability (enable_gabab breaks full nav with zero GABA_B current) and a wrong/inactive critic
afferent (ventral position-invariant `cortex_it`). Fixing it is a **redesign** (a position-sensitive, active
critic afferent — dorsal/place-cell-driven — plus root-causing the enable_gabab system interaction), not a
one-line calibration. The GABA_B `sim/` edit itself is UNAFFECTED and remains a validated GO (its Pavlovian
de-risk passed; it is byte-identical-when-off; condition B confirms the nav runs fine when it is simply not
engaged).

## Status

GABA_B conductance (protected `sim/` edit, commit a7370d49): VALIDATED + shipped, byte-identical-when-off,
unaffected. The nav-runner neural-critic integration (commit e194bd4c, opt-in `--enable-neural-critic`,
default off → existing nav byte-unchanged): an honest NEGATIVE — it does not work as designed (two blockers
above). The 6-seed A/B gate was NOT run (the smoke gate failed). Decision point for the owner: pursue the nav
redesign (right afferent + the enable_gabab dig) vs bank the Pavlovian GABA_B win + this honest negative and
move to a different remaining nav item (the N2/N7 characterizations toward the "fully biologized nav" finding).
