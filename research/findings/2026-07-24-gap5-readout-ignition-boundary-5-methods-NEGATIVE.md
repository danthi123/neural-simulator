# gap#5 imaginative-replay READOUT — the residual is discrete single-assembly IGNITION, and it is now a 5-method boundary (the store COMPLETES statically but will NOT ignite from the readout cue) (2026-07-24)

## The result that sharpens it: re-latching the dendrite did NOT produce ignition
The research gate (2026-07-24) reframed gap#5's residual as **reactivation/ignition, not ordering** (ordering is numpy-GO
via the gamma-WTA), and its TOP pick was to **re-latch the dendritic bistability for the read** (`self_regen_read` 0.0 →
0.15, the completion-GO band) so the cued theta-disinhibition sweep would ignite a discrete single assembly. Tested
6-seed (`_gap5_theta_sweep_replay_derisk --self-regen-read 0.15`, pool): **HONEST NEGATIVE 0/6, per_asm_active [0,0,0]
every seed** — the basket theta reference is live and the cue fires, but NO discrete assembly ignites. ⇒ **the re-latch
was NOT the missing piece; the residual is the IGNITION DRIVE.**

## The 5-method boundary (all this arc, all multi-seed, all controls sound)
| method | result | signature |
|---|---|---|
| (A) spontaneous bistable ignition | NEG | ev≈0 (weak-between store won't self-ignite) |
| (B) targeted DG-detonator ignition | NEG | max_ev=0 across 32 configs to 32× drive |
| (C) Kamondi ramp on POINT neurons | NEG 0/6 | R≈0.05 (no phase locking) |
| (②) Kamondi ramp on TWO-COMPARTMENT dendrite | NEG 0/6 | R≈0.10 == point-neuron; dendrite doesn't rescue |
| (①) network gamma-slot re-latch (self_regen_read=0.15) | NEG 0/6 | **[0,0,0] no ignition** |

Every readout hits the SAME wall the roadblock finding named: strong-between store → diffuse co-fire `[3,3,3]`; weak /
latch-driven → no ignition `[0,0,0]`. Neither ignition-drive strength (to 32×), nor the intrinsic dendritic pacemaker,
nor re-latching the dendritic bistability produces a discrete self-limiting single-assembly burst in the READOUT context.

## The crux (the precise, load-bearing dichotomy for the next research gate)
**The SAME decoupled/dendritic-bistable store COMPLETES statically** (the 2026-07-18 CA3-completion GO: a strong
recall cue `recall_drive=700, recall_steps=150, recall_k_thresh=110` ignites a specific assembly, 6/6) **but does NOT
ignite from the READOUT cue** (a per-theta DG-detonator inside the theta-disinhibition sweep). So the question is NOT
"can the store ignite" (it can, statically) — it is **"why does the readout-context cue fail to ignite what the
completion-context cue ignites, and what mechanism supplies a discrete, self-terminating, sequentially-advancing
ignition for offline replay?"** Candidate directions the gate must weigh (read the sources, don't assume): (i) the
readout cue must MATCH the completion cue's igniting operating point (drive/steps/k_thresh) — a cue-parameter mismatch,
not a mechanism gap; (ii) SWR/sharp-wave physiology — replay ignition is driven by a CA2/CA3 population SPW burst +
a specific E/I transient, not a theta-disinhibition sweep (theta and SWR are different brain states — theta = encoding,
SWR = offline replay; the arc may be applying a theta-state readout to an SWR-state phenomenon); (iii) a slow
depolarizing envelope (not a fast cue) that walks the assembly chain. Direction (ii) is the strongest reframe: **replay
happens in the SWR state, not the theta state** — the readout may be in the wrong brain-state regime entirely.

## Status (per THE LAW — closure cannot be deferred; capability OPEN)
All 5 readout METHODS banked NEGATIVE; the imaginative-replay CAPABILITY stays OPEN. A focused research gate is
dispatched on the ignition problem (esp. the theta-vs-SWR-state reframe + the cue-operating-point-match). NO `sim/`
edit anywhere in this readout arc. This is a memory-system wall (roadmap walls-ledger gap#5), NOT the pivot's core
faculties (affect/reasoning/self/curiosity), which are being built in parallel — gap#5 keeps moving via the gate, but
does not block the pivot.

## UPDATE (2026-07-24, same day) — REFINED by the SWR-envelope build: it's op-point/STATE, NOT a substrate wall; "[0,0,0]" was partly a detection artifact
The SWR research gate's Option-1 diagnostic (`_gap5_swr_envelope_replay_derisk --option1`, GPU seed 42) showed that at
the COMPLETION operating point (sustained recall_drive=700/150 + self_regen=0.15 + k_thresh=110), assembly-0 IGNITES
(`per_asm_frac` 24.4%). ⇒ ignition IS achievable; the 5-method boundary was operating-point / brain-STATE (theta vs
SWR), **NOT a substrate wall**. The `per_asm_active [0,0,0]` no-ignition signature reported above was PARTLY a windower
detection artifact (sustained cued firing isn't a discrete "event" for the windower — `per_asm_frac` shows it WAS
igniting). The genuine residual is attractor-SELECTIVE forward HAND-OFF, not ignition-from-scratch. The SWR-envelope
readout then confirmed 3/4 SWR ingredients (discrete ignition · self-termination · noise-seeded) with the 4th
(attractor-selective hand-off) precisely localized → full result + the next mechanism (latch-then-release) in
`2026-07-24-gap5-SWR-envelope-Option1-POSITIVE-Option2-3of4-selective-handoff-residual.md`. Capability OPEN + advancing.
