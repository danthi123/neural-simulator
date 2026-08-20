---
type: finding
status: live
date: 2026-08-20
mechanism: generative-attractor-wander
lane: continuity
seeds: [42, 43, 44, 45, 46, 47]
instrument: sparse-Hopfield attractor de-risk, blended multi-pattern cue, 6 seeds x 2 scales + 3 anti-cheats
runner: research/runners/_generative_attractor_wander_derisk.py
artifacts:
  - research/findings/raw/_generative_attractor_wander/evidence.json
  - research/runners/_generative_attractor_wander_derisk.py
---
# Generative attractor-wandering de-risk: a blended cue settles to a NOVEL (never-stored) stable state — genuine novelty from the dynamics

Artifact: research/findings/raw/_generative_attractor_wander/evidence.json (runner: research/runners/_generative_attractor_wander_derisk.py)

**One line (continuous rung 4 — genuine NOVELTY).** A sparse associative-attractor completion (same family as the
project's GO CA3 harnesses) CAN settle into a state that was never stored: cueing it with a blend of THREE stored
patterns yields a STABLE fixed point balanced 0.611 with all three cued sources (not any single item, well
above the 0.188 overlap with non-cued patterns) — novelty from the DYNAMICS, not the nodes. numpy de-risk,
6-seed x 2-scale, anti-cheats clean; on-substrate port is the next step.

## The mechanism that mattered
A sharp rank-based top-k WTA (the direct FS-lateral-inhibition analogue) is TOO SHARP — it collapses every blended
cue onto ONE stored source in 1-3 iterations. Replacing it with a mean+std dynamic THRESHOLD (the ca3_ff_inhib /
ca1_ff_inhib feedforward-inhibition mechanism ALREADY validated on-substrate: threshold rises with drive, a ~constant
fraction fires) gives the balanced novel fixed point. So the novelty capability rides an already-GO spiking mechanism.

## Anti-cheats (6 seeds 42-47, scales n=400/k=40/n_mem=6 and n=1200/k=60/n_mem=20)
- **Positive control:** a partial cue of ONE stored pattern recovers it at overlap 1.000, ~0.10-0.15 on all others (the completion works + is specific).
- **Untrained (W=0):** the blended cue settles to overlap 0.000 with every pattern (the threshold rule alone does NOT fake completion).
- **Fixed point:** the blended state is stable under another full sweep at every seed.
- Honest caveat (reported, not gated): a pure-NOISE cue can occasionally settle balanced-between-two by chance at these small n/k/n_mem — a capacity-limit, printed every run. The Amit-1989 odd/even mixture-state prediction was tested and does NOT hold under this threshold dynamics (a reported negative on the theory).

## Residual / next step
Algorithmic (numpy) de-risk, not yet on the spiking SimulationBridge CA3 harness. Next: swap the on-substrate CA3
partial-cue readout for a mean+std dynamic threshold, drive it with a multi-concept blended cue, re-measure
novelty+balance+anti-cheats at larger scale (closes the noise caveat), THEN wire the idle-tick wander
(webapp/continuous_engine.py) to occasionally drive a blended multi-concept cue instead of the curiosity-top single concept.
