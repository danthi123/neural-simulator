---
type: finding
status: live
date: 2026-08-20
mechanism: continuous-state-engine
lane: continuous-substrate
seeds: [42]
seed-waiver: A deterministic content-variety probe — do successive idle wanders on ONE session's self-init organ surface DIFFERENT concepts, or the same one. The evidence is the concept SEQUENCE across 6 successive wanders (a within-run variety count), not a stochastic effect size across a seed population; the single seed is the substrate build seed (and re-seeding per organ is precisely the confound being characterised).
instrument: research/runners/_continuous_wander_variety_cupy.py — one SelfInitiationOrgan(seed=42), 6 successive speak() calls on cupy
runner: research/runners/_continuous_wander_variety_cupy.py
artifacts:
  - research/findings/raw/_continuous_live_cupy/wander_variety.json
---
# HONEST NEGATIVE: the between-turn wander is content-DEGENERATE (6/6 'cat' on cupy) — the coupling drives, the source doesn't wander

Artifact: research/findings/raw/_continuous_live_cupy/wander_variety.json

**One line.** The continuous engine's "a THOUGHT wanders between turns" property was shown load-bearing (the wandered
concept drives the next reply's lead, and it vanishes under lesion). This probe asks the next, sharper question the
drive-not-observe bar demands: does the wander actually WANDER? On cupy, six successive idle wanders on one session's
self-initiation organ surfaced **the same concept every time — 'cat', 6/6**. So the coupling is real but its SOURCE
is degenerate: while a user is away, the brain always "mulls over cat", never anything else. A load-bearing coupling
to a constant is still a constant — genuine trains-of-thought requires the wander to vary, and here it does not.

## What this corrects
The continuous-engine v1 finding (2026-08-20-continuous-state-engine-v1) stated the numpy light path surfaces the
stable curiosity-top concept while "the stochastic multibasin CA3 wander (varied concepts) is the cupy path." This
measurement FALSIFIES the parenthetical for the production default store: the cupy wander is NOT varied here — it is
the same single concept across all six draws. The earlier two observations ('cat' in the in-process load-bearing test
and 'cat' in the live long-gap server test) were both FRESH seed-42 organs (same starting noise); this probe advances
one organ's noise state across six wanders and still gets 'cat' every time, so re-seeding is not the (only) cause.

## Why (candidate causes, to disambiguate next)
The self-init wander is meant to ignite "whichever balanced basin the coincidental noise overlap favours, curiosity-
biased" (`self_initiated_production_organ` / `_self_initiation_multibasin_derisk`). Degeneracy means one of: (a) the
curiosity gain over-determines the winner so the SAME basin always ignites regardless of noise; (b) the default
concept store has too few well-encoded basins (a coverage problem, not a wander problem); (c) the OU noise amplitude
is too low to move the winner off the dominant basin. The artifact records only the sequence; the next probe should
print the store's basin count + the per-basin ignition margin to separate (a)/(b)/(c).

## The biological next lever (INHIBITION-OF-RETURN / adaptation)
Real spontaneous cognition does not fixate — an active representation ADAPTS (spike-frequency adaptation) or its
recurrent loop briefly DEPRESSES (short-term synaptic depression), so the next wander is biased AWAY from the
just-visited basin (inhibition of return). This is a spiking, on-substrate mechanism (both SFA and STD already exist
in the engine): after a wander ignites basin X, fatigue X for the next tick so a different basin wins. That both
breaks the 'cat'-every-time degeneracy AND is how a mind actually moves between thoughts. Secondary lever: verify /
widen the concept-store basin coverage so there is more than one strongly-encoded target to move to.

## Scope
The wander-DRIVE coupling (#86) remains correct and load-bearing — this does not retract it; it characterises the
UPSTREAM source feeding it as degenerate, and names the fix. Until fixed, the honest read of the continuous engine is:
FEELING genuinely evolves between turns (mood relaxes, a real spiking read), but the wandered THOUGHT is currently a
constant, not a train — so the "trains-of-thought" continuous property is NOT yet met, only scaffolded.

## Sources
- (Posner & Cohen, 1984) "Components of visual orienting" — inhibition of return: attention/representation is biased
  AWAY from a just-visited item, the cognitive-level anti-fixation phenomenon the wander needs.
- Local substrate precedent: 2026-08-14-gnw-rung2b-sfa-workspace-eviction-BOUNDARY — spike-frequency adaptation
  already prototyped to EVICT the current workspace winner; the same SFA/STD primitive fatigues the just-ignited
  wander basin. So the next lever is a documented mechanism with an in-repo precedent, not a novel paradigm.
