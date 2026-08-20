---
type: finding
status: live
date: 2026-08-20
mechanism: continuous-state-engine
lane: continuous-substrate
seeds: [42]
seed-waiver: A deterministic MECHANISM demonstration — does a curiosity-gain FLOOR let the previously-dead tail concept surface under inhibition-of-return. The evidence is the concept SEQUENCE of two arms (IOR-only vs IOR+floor) on the same organ/seed, a within-subject presence/absence of full coverage; the single seed is the substrate build seed.
instrument: research/runners/_continuous_wander_gainfloor_coverage_derisk.py — two cupy arms x 10 successive wanders (IOR-only vs IOR + gain floor)
runner: research/runners/_continuous_wander_gainfloor_coverage_derisk.py
artifacts:
  - research/findings/raw/_continuous_live_cupy/wander_gainfloor.json
external: NO-EXTERNAL-NEEDED — a steering-gain-floor on an in-repo mechanism, following the in-repo SFA-locus diagnosis; no capability wall or paradigm claim.
---
# GO: a curiosity-gain FLOOR closes the wander store-coverage residual — full 4/4 concepts now surface

Artifact: research/findings/raw/_continuous_live_cupy/wander_gainfloor.json

**One line.** The between-turn wander with inhibition-of-return reached only 3 of the 4 stored concepts under any
recovery/strength — the 4th ("fish") never won. The SFA-locus diagnosis (2026-08-20-per-neuron-SFA-wrong-locus)
showed the winner is set by the tonic STEERING gain, so the tail concept never surfaced because its steering gain was
too LOW to win even against the IOR-fatigued top 3. This lands the named secondary lever: a curiosity-gain FLOOR
(clamp each basin's base steering gain up to a floor before IOR fatigue). Result (cupy, seed 42, 10 wanders):
IOR-only reaches `cat, dog, bird` (3/4); IOR + gain-floor 1.6 reaches `cat, dog, bird, fish` (4/4, full coverage) —
the previously-dead tail concept now wins when the top basins are fatigued. VERDICT GO.

## Why it works (and confirms the diagnosis)
The wander is a winner-take-all competition on the steering drive. The 4th basin's base curiosity gain was below the
level needed to beat even an IOR-suppressed leader, so it stayed dead. Raising the FLOOR (a change to the STEERING
drive — the mechanistically-correct locus, per the per-neuron-SFA diagnosis, not intrinsic excitability) gives the
tail enough residual drive to win once the top basins fatigue. This is the population-level counterpart of the same
lesson: fatigue AND floor the DRIVE, not the intrinsic excitability.

## Wired live
Added `IOR_GAIN_FLOOR = 1.6` to `webapp/continuous_engine.py`: the per-session IOR now clamps the captured base gains
up to the floor, so a live idle session's wander reaches full 4/4 coverage (was 3/4). Default-on with the continuous
engine's IOR; byte-compatible off (BRAIN_WANDER_IOR=0). With this, the between-turn "trains of thought" faculty is
now COMPLETE at the store's capacity: it varies across every stored concept, driven by the spiking self-init wander,
fatigued + floored on the correct (steering) locus. The remaining continuous-arc work is integration (port the GO
mechanisms to sim/ kernels + wire idle consolidation under the tick), not this faculty. (Substrate-run, GO.)
