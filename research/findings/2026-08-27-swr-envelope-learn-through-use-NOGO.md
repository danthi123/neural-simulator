---
type: finding
status: no-go
lane: memory-learn-through-use
date: 2026-08-27
mechanism: SWR-state E/I-transient envelope replay op-point on the DECOUPLED bistable-completion CA3 store
artifacts: research/findings/raw/gap5_r4/swr_envelope_v2_6seed_char.json, research/findings/raw/gap5_r4/swr_envelope_v2_seed42_designed.json
runner: research/runners/_gap5_swr_envelope_replay_derisk.py
---
# SWR-envelope replay CANNOT reach the [1,1,1] forward-ordered op-point on the current CA3 store — 6-seed NO-GO; the wall is the bistable-completion ARCHITECTURE, so D5 learn-through-use stays blocked pending the Ecker-2022 AdEx CA3 build

Artifact: `research/findings/raw/gap5_r4/swr_envelope_v2_6seed_char.json` (6-seed op-point characterization) +
`research/findings/raw/gap5_r4/swr_envelope_v2_seed42_designed.json` (canonical runner, full anti-cheats, seed 42).
Runner: `research/runners/_gap5_swr_envelope_replay_derisk.py` (+ `_scratch` GO-only sweeps, reuse-by-import; NO `sim/` edit).

## Question and result
Can the SWR E/I-transient envelope move the CA3 replay op-point OFF co-fire TOWARD discrete `[1,1,1]` forward-ordered
replay (the convergent unblock for #71/#107 learn-through-use)? **NO — 0/6 seeds.** `[1,1,1]` does not close.

## The op-point does not exist (6-seed, n_ca3=2000, cupy)
The store builds correctly every seed (within≈200, adj_fwd≈38.6, adj_rev≈5.0). Reading it under the envelope:

| arm | per_asm_active | forward | silence_frac | co_active_frac | discrete events |
|-----|---------------|---------|--------------|----------------|-----------------|
| DEFAULT envelope | [0,0,0] 6/6 | 0.00 | 0.080 | **0.966** | 0/6 |
| DESIGNED (latch-release + fwd-gain 4x + random cue) | [0,0,0] 6/6 | 0.00 | **0.003** | **0.983** | 0/6 |

co_active_frac = fraction of active steps with >=2 assemblies simultaneously on: **0.97–0.98 = CO-FIRE, not a sequence.**
The net never rests (silence <=0.09), so it never SEGMENTS into discrete SWR events. Adding the forward-link gain makes
it MORE continuous (silence 0.080->0.003), not more discrete. The canonical seed-42 runner agrees: GO=False, every
store-lesion anti-cheat (NO-SWR, SHUFFLED, REVERSE-ASYM, NO-ENCODE) at forward 0.000.

## Why (root cause, quantified) — and what it is NOT
A drive/inhibition sweep (seed 42; env_exc 180–3000, noise 800–6000, basket-boost 200–5000 = 25x, envelope 60–120,
sel_inhib_spare 0 and 20) NEVER segments: more inter-event inhibition LOWERS silence (over-suppression -> low mush),
it does not buy discreteness. The store's strong bistable within-attractors (≈200) reverberate semi-continuously and
co-ignite all assemblies. So the blocker is the STORE ARCHITECTURE (bistable *completion* is antithetical to a
moving-bump *hand-off*), not link strength, not op-point tuning — confirming + quantifying the 2026-07-24 UPDATE-2 NO-GO.

## Consequence for learn-through-use (D5 / #71 / #107)
There is NO store-dependent discrete forward-replay op-point to feed the D5 organ, so SWR-envelope replay does NOT
unblock learn-through-use here; the D5 recall-change test has no valid replay input to run. Capability NOT abandoned.

## Next mechanism (a wall defers a METHOD, not the capability)
The **Ecker et al. 2022 eLife 11:e71850** AdEx CA3 model: weaker/self-terminating recurrent attractors + a
synaptic-depression hand-off + a PV-basket ripple oscillator that generates DISCRETE replay events (Buzsáki 2015
Hippocampus 25:1073, SWR as a transient self-terminating E>I burst). The `--fwd-gain` + `--onset-cue-asm` machinery are
validated reusable ingredients for that build.
