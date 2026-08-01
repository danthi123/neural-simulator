---
type: finding
status: live
date: 2026-08-01
mechanism: affect-state-region
artifacts:
  - research/findings/raw/affect/stp_negative/
---

# The affect ratchet is a BISTABLE saturated attractor: STP depression annihilates it across every τ_d and strength — an outward brake cannot grade-evict it

**One-line verdict:** short-term synaptic depression, swept comprehensively, does not grade-evict the mood
ratchet — at every strength and timescale that engages at all, it collapses the held state to zero. This is
the signature of a bistable attractor, and it retires the whole *brake* class of evictor (GABA_B, STP, and by
the same argument SFA), not just one method.

## What was measured (this session)

`--stp` / `--stp-tau-d` / `--stp-U` are now reachable, recorded knobs on `_affect_eviction_derisk.py`
(committed `75b4e1ce`, `3d80d446`; the lever was verified to engage — STP-ON held collapses to
zero while the untouched baseline instrument brain stays byte-identical, so the knob reaches only the
eviction brain). With GABA_B off (`--sweep-weights 0.0`) so STP is the sole evictor, the held-state readout across the
full sweep:

| swept | values | result (held[0], 3 seeds each) |
|---|---|---|
| τ_d (fast) | 50, 100, 150, 200 ms | **0.000** every cell |
| stp_U (strength) | 0.01, 0.02, 0.05, 0.15 | **0.000** every cell |
| τ_d (slow) | 500, 1000, 2000, 4000 ms | **0.000** every cell |

Every cell: `held(LOW) = [0,0,0,0,0]`, verdict `UNDEFINED (held[0]~0: no state to evict)`. The baseline
ratchet reproduces, so the instrument is valid; the pool still ignites *during* drive but holds nothing
afterward. Aggregate (the numbers above):
`research/findings/raw/affect/stp_negative/_stp_negative_aggregate.json` — `n_cells` 27, `n_seeds_per_cell` 3,
`held0_max_across_all_cells` 0.0, `held0_all_zero` true, `baseline_ratchet_ratio_min`/`max` 1.045/1.064; the
27 per-cell smokes are in the same directory.

## Why — and why this is not just "one more method failed"

The persistence being braked is the **slow-NMDA reverberatory attractor**, which the record already says
*"IGNITES at a low threshold and SATURATES"* (`2026-07-24-P0.3-affect-state-region-6seed-GO.md`: mood
retains 0.62 of peak with NMDA on vs 0.00 with NMDA off) <!--derived: quoted from 2026-07-24 P0.3 GO-->. A saturated recurrent loop is **bistable** — it sits
in a self-sustaining ON basin. An outward synaptic brake has exactly two outcomes against such a loop:

- too weak to cross the basin boundary → the loop stays ON → no eviction (**GABA_B**, killed 0/80 with G1
  eviction failing on a valid sweep);
- strong enough to cross it → the loop falls out of the ON basin entirely → **annihilation** (**STP**, this
  finding, at every τ_d 50–4000 ms and U 0.01–0.15).

There is no graded middle because a bistable attractor has no graded middle. This directly **refutes the
2026-07-31 prediction** that a seconds-scale τ_d would grade-evict ("200 ms is too fast to accumulate against
the seconds-long reverberation"): slow τ_d {500–4000} annihilates identically to fast τ_d. The timescale was
never the issue; the bistability is.

## The mission consequence — the next mechanism is a NON-brake, and it needs a research gate not another sweep

Per THE LAW this kills the *brake method class*, not the *evict-the-ratchet capability*. Running more brake
sweeps (a slower SFA, a different GABA_B cap) is now predictably redundant — they are all outward brakes on a
bistable loop. The capability needs a mechanism of a different kind. The candidates, for a deep-research gate:

1. **Make the attractor graded, not saturated** — reduce the recurrent NMDA gain so mood is an adjustable bump
   rather than a latch, then a brake *can* grade it down. Risk: the 07-24 GO shows the saturation IS the
   persistence, so this trades persistence for evictability — the real question is whether a graded attractor
   can hold long enough to be the mood substrate.
2. **An active clear / quench gate** — a transient strong inhibitory pulse (a biological "clear" signal) that
   knocks the loop out of the ON basin on command; eviction is all-or-none by design, and persistence is
   re-established by the next input. This matches how attractor working-memory is cleared in vivo.
3. **Neuromodulatory gain control** — dynamically lower the loop's effective gain (release the attractor)
   rather than subtract current against it.

None is a pool-sweep; each is a wiring + measurement arc gated on reading the attractor-clearing literature.
This is the affect evictor's real frontier and it is now precisely stated.

## Honest scope
Single-seed-per-cell smokes (3 seeds per row), not a 6-seed gate — but a held state of exactly 0.000 in
**27/27** STP cells is not a marginal read, and the bistable argument is structural, not statistical. Also
still owed (independent of this): STP/SFA arms lack a valid instrument control — G6's `evict_out` lesion gates
GABA_B synapses, not the per-synapse STP dynamic or the intrinsic `cp_izh` adaptation — so a *positive* brake
result would have needed an STP-lesion control before any GO. No positive result arose, so this negative
stands on the annihilation itself.
