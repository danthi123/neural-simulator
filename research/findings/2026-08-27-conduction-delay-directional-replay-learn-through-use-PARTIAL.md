---
type: finding
status: partial
lane: memory-learn-through-use
date: 2026-08-27
mechanism: forward-edge axonal CONDUCTION DELAY on the Ecker AdEx CA3 offline-replay recurrent band (host-side delay-line on the g_e channel; forward drive reaches the next assembly delay_steps*dt ms later so the leading assembly self-terminates before the next ignites), on top of the BTSP-eligibility directional write (sim.kernels.fused_btsp_update)
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py (--write-rule btsp --fwd-delay-steps 90)
runner: research/runners/_gap5_ecker_replay_learn_through_use_derisk.py
external:
  - Izhikevich 2006 "Polychronization Computation with Spikes" Neural Computation vol18 no2 (doi neco.2006.18.2.245), https://consensus.app/papers/details/b70432d790495819ade03d39979d4835/ — axonal conduction delays + STDP spontaneously self-organize neurons into time-locked DIRECTIONAL (polychronous) groups; the delay is the ingredient that makes a coincidence-read rule direction-selective.
  - Yu et al. 2025 Neuromorphic Computing and Engineering — trainable axonal delays let SNNs exploit temporal/DIRECTIONAL structure (networks with delays fail under spike time-reversal; without delays they do not).
  - Baron et al. 2026 Hippocampus "The Role of Plasticity in Replay" — same Ecker CA3 model: an asymmetric-Hebbian STDP kernel biases replay directionality (corroborates directionality is reachable on this substrate).
artifacts:
  - research/findings/raw/gap5_ecker_adex/conduction_delay_directional_ltu_6seed.json
  - research/findings/raw/gap5_ecker_adex/conduction_delay_control_delay0_6seed.json
---
# A forward-edge axonal CONDUCTION DELAY (9 ms) SEPARATES the replay volleys (overlap 0.58 -> 0.28) and flips the BTSP write NET-DIRECTIONAL on 6/6 seeds (dw_fwd > dw_rev every seed; was 0/6, dw_rev ~1.9x dw_fwd) — surpassing the exact volley-overlap blocker the prior PARTIAL isolated. But learn-through-use does NOT clear the GO bar (1/6): under the long-period detection needed to separate the volleys, forward recall is at ceiling (weak/full ~1.0) so the directional deepening has no recall HEADROOM, and a residual reverse band still grows

Artifact: `research/findings/raw/gap5_ecker_adex/conduction_delay_directional_ltu_6seed.json` (6-seed decisive, delay=90) +
`research/findings/raw/gap5_ecker_adex/conduction_delay_control_delay0_6seed.json` (matched delay=0 control).
Runner: `research/runners/_gap5_ecker_replay_learn_through_use_derisk.py --write-rule btsp --fwd-delay-steps 90` (additive, default `-1`=OFF; the delay-line + overlap metric are new, guarded off, determinism-verified: exact re-run match on dw_fwd/dw_rev/overlap; the STDP/BTSP paths are untouched; reuse-by-import of `fused_btsp_update`; NO `sim/` edit).

## Question
[[2026-08-27-btsp-directional-write-learn-through-use-PARTIAL]] proved (a 14-point op-point sweep) the residual blocker is NOT the write rule — it is VOLLEY OVERLAP: the leading assembly keeps firing AFTER the next ignites, so pre-and-post overlap and NO coincidence-read write can be forward-selective. Named next mechanism: a forward-edge CONDUCTION DELAY separating the volleys. The engine propagates every spike with a UNIFORM 1-step delay (no per-synapse delay exists), so the delay is a host-side delay-line on the SAME g_e conductance channel: forward edges zeroed in the matrix, their exact increment re-added `delay_steps` later; reverse keeps its 1-step delay. Does separating the volleys make the write forward-selective and unblock learn-through-use?

## Result — the delay SEPARATES the volleys and makes the write NET-DIRECTIONAL 6/6 (but recall stays at ceiling)
<!--derived-->
At delay=90 steps (9.0 ms ≈ the volley duration; elig_tau=80 ms > delay, plat_tau=1.0 ms, eta=0.001 sub-saturation), volley overlap drops 0.58 (matched delay=0 control) -> **0.280** and the write flips forward-selective: **dw_fwd 203.6 vs dw_rev 170.9** (forward 1.19x reverse) — **6/6 seeds directional** (dw_fwd > dw_rev on every seed; per-seed dw_fwd/dw_rev: 246/219, 146/114, 239/205, 233/220, 164/117, 194/150). The forward band deepens more than reverse (adj_fwd 319.4 -> 522.9, adj_rev_after 182.4). This is the prior PARTIAL's exact residual SURPASSED: it was 0/6, dw_rev ~1.9x dw_fwd; a shorter plateau/longer eligibility never flipped it.
The matched delay=0 control at the identical op-point is 0/6 directional (dw_fwd 215.9 < dw_rev 245.6, overlap 0.582) -> the delay is CAUSAL for both the overlap collapse (0.582 -> 0.280) and the 0/6 -> 6/6 flip. Lesion-the-replay (NO-SEED): dw = 0.000 on ALL 6 seeds -> 100% attributable.

## Residual (not forced): learn-through-use does NOT clear the GO bar (1/6)
<!--derived-->
The GO gate is 1/6 — the RECALL-CHANGE criterion fails 5/6. Cause: the long SWR period + short detection window REQUIRED to separate the delayed volleys make the recall read saturate — full-cue forward 1.0 and weak-cue forward 1.0 BEFORE (both well above chance), so the directional deepening has no headroom to manifest as a recall GAIN; after consolidation weak-cue forward even slips (1.0 -> 0.807) because a residual reverse band still grows (dw_rev +170.9 > 0). Fewer encode laps did not open headroom (the read still saturates). So the WRITE-DIRECTIONALITY wall is surpassed, but the CAPABILITY (a replayed episode durably STRENGTHENS forward recall) is not demonstrated here — the recall instrument is compromised by the very op-point that separates the volleys.

## Verdict + next mechanism (a wall defers a METHOD, not the capability)
PARTIAL — a real, banked advance: a biological forward-edge conduction delay eliminates most volley overlap (0.58 -> 0.28) and makes the replay-time write NET-DIRECTIONAL on 6/6 seeds, decisively surpassing the volley-overlap blocker that held the prior BTSP PARTIAL at 0/6 (delay-controlled, lesion-controlled, Izhikevich-2006 polychronization-grounded).
Banked positives carry forward: the Ecker store segments; replay drives durable lesion-controlled plasticity; the delay separates the volleys.
Residual, precisely isolated: learn-through-use does not clear the bar because recall is at ceiling under the separation op-point + reverse still grows.
Named next: (1) Braun 2022 inhibitory GAP-CODING — sparse sequences with NO overlapping synfire excitation, so volleys separate WITHOUT a long-period read that saturates recall; (2) a recall instrument with headroom decoupled from the separation op-point (degraded-cue read on a shorter period).
Production D5 wire-in stays separately blocked at the AdEx soma-recurrence seam ([[2026-08-20-ecker-real-d5-store-does-NOT-reactivate-via-soma-recurrence-dendritic-latch-is-the-read]]).
