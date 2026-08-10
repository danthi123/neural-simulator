---
type: finding
status: contributing
date: 2026-08-10
mechanism: ca3-completion
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: a HAND-INSTALLED perfect within-assembly potentiation on the ca3->ca3 recurrent (idealized outcome of a perfect recurrent LTP; frozen plasticity) is swept over weight W, with the recurrent ELEMENT as the one manipulated variable (somatic slow-NMDA reverberatory vs the fast/AMPA control, identical FS basket + sparse density + long read). Decomposed quantities: held-out completion magnitude (held_cue), permuted-cue reactivation of the SAME held-out members (held_perm, specificity teeth), silent-rest firing (held_nocue, self-ignition teeth), and sustain-after-cue-release (held_sustain, bistable-persistence teeth). Load-bearing controls: NO-ENCODING (weak baseline, no install -> completion collapses) + RECURRENCE-ZERO (zero ca3->ca3 -> collapses) + OU-off/OU-on (deterministic vs noise-driven). GO gate (6-seed): held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10.
---

# Gap #5 lead (b) reaches GO — a SOMATIC slow-NMDA REVERBERATORY attractor (Wang 2002) holds a bistable + cue-specific CA3 completion on the POINT soma, 6/6 seeds (deterministic AND noise-robust) — the horn the AMPA/point-neuron recurrent upper bound could not hold

This BUILDS the cheapest indicated lead from the 2026-07-18 research gate
(`2026-07-18-gap5-bistable-completion-mechanism-research-gate.md`, mechanism #1) and today's redirect
(`2026-08-10-ca3-point-neuron-attractor-completion-trilemma-NEGATIVE-redirect-dendritic-plateau.md`, which named
lead (b) "the open frontier — whether it reaches GO is the open frontier"). It reaches GO. NO `sim/` edit — a new
runner reuses the committed `exc_receptor="nmda_slow"` / `enable_nmda_recurrent` machinery (byte-inert when off).

## Why this was genuinely untested (not a re-derivation)

The 2026-07-18 "Wang seed-42" result is RETRACTED (a plasticity+OU-noise confound; frozen it read w~49 DEAD) — but that
tested a Hebbian-grown WEAK attractor at the WRONG operating point, never a proper reverberatory one. The 2026-08-10
UPPER BOUND swept a hand-installed perfect potentiation on the FAST/AMPA recurrent (short read, no dedicated FS basket,
dense recurrence) and found the point-neuron trilemma (perm overtakes comp, net self-ignites). Neither ran the Wang
somatic slow-NMDA (tau_decay 100 ms, Mg2+ self-limiting, AMPA suppressed on those synapses) reverberatory element at a
completion-scale operating point + FS working point + a read window >= 2.5*tau_NMDA. That is the one untested variable,
and it is NOT covered by the AMPA upper bound (slow NMDA's temporal integration + Mg self-limiting is a distinct axis).

## The result (6-seed 42/43/44/100/101/102; frozen plasticity; hard-silence resets `g_nmda_recurrent` between conditions)

<!--derived-->

Headline config: `_gap5_ca3_nmda_slow_reverberatory_derisk.py` — n_ca3=400, density 0.12, 3 disjoint ~72-cell
assemblies, ca3->ca3 = `exc_receptor="nmda_slow"`, shared FS basket `ca3_fb_inhib=60`, drive 300 pA, read
200-warm + 200-hold steps (~4*tau_NMDA), hand-installed within-assembly weight W. Determinism verified (build-twice
`cp_neuron_firing_thresholds` hash identical -> SEEDED, via `cfg.seed`).

| element | OU | shared-GO window (all 6 seeds) | held_cue (at W5000) | held_perm | held_nocue | held_sustain |
|---|---|---|---|---|---|---|
| **slow-NMDA** | OFF | **W 2500 / 5000 / 9000 = 6/6 each** | 0.328-0.356 | **0.000** | **0.000** | 0.331-0.355 (≈cue) |
| **slow-NMDA** | ON (fb=60) | **6/6** (W5000-9000) | 0.31-0.44 | 0.000 (43: 0.034) | 0.000 | ≈cue |
| AMPA control | OFF (fb=60) | W5000 / 9000 = 6/6 (**W2500 = 0/6**) | 0.210-0.260 | 0.000 | 0.000 | ≈cue |
| AMPA control | OFF (fb=20) | **best 5/6** (self-ignites) | ~0.21-0.26 | 0.000 | **0.19-0.34 at high W** | ≈cue |

- **slow-NMDA = clean 6/6 GO, deterministic AND noise-robust.** Every seed passes at every swept W (2500-9000) OU-off;
  and at fb=60 every seed also passes OU-on. `held_sustain ≈ held_cue` (the held-out members KEEP firing after the cue
  is RELEASED) with `held_nocue = 0.000` — genuine bistability: a switchable persistent HIGH state coexisting with a
  silent LOW rest, NOT the retracted always-on self-sustaining artifact (which nocue=0 rules out).
- **Every teeth has teeth:** NO-ENCODING = 0.000, RECURRENCE-ZERO = 0.000 (the reverberation, not cue re-drive, does
  the completion), permuted-cue = 0.000 (a random cue does NOT ignite the assembly), non-member = 0.000.
- **Scale robustness:** n_ca3=1000, 120-cell assemblies (density 0.06) hold GO at W5000 (cue ~0.30, perm/nocue 0) on
  seeds 42/100 — the Kopsick 150-300 robust range direction.

## Attribution — what the slow-NMDA ELEMENT is load-bearing for (honest, matched AMPA control)

<!--derived-->

The runner emits `attributable_to` for completion-vs-no-encoding and completion-vs-permuted every cell. The matched
AMPA control (identical FS basket + install + long read; only the recurrent element differs) is the decisive foil:

- **Bistable-window WIDTH / efficiency:** slow-NMDA reaches the 0.20 magnitude bar at **W2500**; the AMPA control needs
  **W5000** (~2x the recurrent weight) — the slow, temporally-integrating NMDA folds more drive into the held-out
  members per synapse (Wang 2002's "wide bistable window from slow NMDA").
- **Silent-rest robustness to the inhibitory set-point:** at fb=20 the AMPA control SELF-IGNITES (held_nocue 0.19-0.34)
  at the high W that seed-fragile seeds need -> best 5/6, no single W GO on all six; slow-NMDA keeps held_nocue=0.000
  at every W already at fb=20 -> 6/6. The Mg2+-self-limiting slow conductance stabilizes the LOW state.
- **What slow-NMDA is NOT solely responsible for (do not overclaim):** the SPECIFICITY (perm=0) is SHARED — at fb=60
  the AMPA control ALSO reaches perm=0.000 (the FS basket + sparse density + long integration carry much of it). So
  slow-NMDA is the load-bearing element for the WINDOW (width + silent-rest stability at low inhibition), not the sole
  specificity mechanism. This also reconciles with the 2026-08-10 AMPA upper bound (which had NO dedicated FS basket,
  a short read, and DENSE recurrence -> perm overtook): the specificity there failed for lack of the basket+sparsity+
  integration, not because AMPA is intrinsically non-specific at THIS operating point.
- **Noise-robustness = the Amit-Brunel inhibitory set-point, not automatic:** at fb=20 OU-on is a NEGATIVE (noise drives
  held_nocue 0.24-0.47, cue≈perm — pure noise, not completion). Raising the feedback-inhibition set-point (fb 20->60,
  the Amit-Brunel spontaneous-state stabilizer) recovers 6/6 OU-on. Named and closed within this de-risk.

## Honest scope / the open residual (per THE LAW — a characterized boundary, not a wall)

<!--derived-->

- **HAND-INSTALLED attractor (idealization of a perfect recurrent LTP), point-soma only.** This de-risks the ATTRACTOR
  ELEMENT / dynamics + readout, the same idealization the standing 6/6 readout surpass
  (`2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md`, a two-compartment READOUT — NOT the
  refuted two-compartment deep-credit rule of `2026-05-17-dendritic-credit-assignment-NEGATIVE`) and the 2026-08-10
  upper bound both use. This somatic slow-NMDA route is a peer to that readout surpass: a SECOND, soma-only route to
  bistable+specific completion. The EMERGENT-FORMATION path (whether a biological rule GROWS this operating point) is
  NOT closed here — the substrate's rate-Hebbian collapses ca3->ca3 (2026-07-17) and the retracted Hebbian-grown Wang
  read w~49 DEAD; the install W (2500-9000, >~1600x baseline) is a completion-scale idealization, not a per-synapse
  physiological weight.
- **The open residual is EMERGENT FORMATION into this reverberatory operating point** — BTSP one-shot plateau-gated
  storing (`_gap4_btsp_stores_recurrent_assembly_derisk.py` machinery) that writes a completion-scale within-assembly
  slow-NMDA recurrent WITHOUT the rate-Hebbian collapse, then read by THIS gate; and folding in Kopsick homeostatic
  divisive downscaling + `w_max ∝ 1/size` so the emergent weight is moderate-absolute + high-SNR rather than hand-set.

## Verdict

**Lead (b) — somatic slow-NMDA reverberatory attractor — REACHES GO: 6/6 bistable + cue-specific CA3 completion on the
point soma, deterministic and noise-robust, all mandatory anti-cheats clean.** The point-neuron trilemma the 2026-08-10
AMPA upper bound hit is ESCAPED by the slow-NMDA element (wide bistable window + Mg-self-limited silent rest) plus the
E/I working point — a soma-only peer to the standing 2026-07-08 readout surpass. What remains OPEN is EMERGENT
FORMATION into this operating point; the readout/dynamics itself is now de-risked by two independent routes (the
somatic slow-NMDA reverberation here, and the previously-established readout surpass).

Artifacts (SIM_BACKEND=cupy, provenance sidecars record backend + argv + git SHA):
`research/findings/raw/_gap5_nmda_slow/nmda_slow_reverberatory_fb60_6seed.json` (headline, both OU),
`research/findings/raw/_gap5_nmda_slow/nmda_slow_reverberatory_6seed.json` (fb=20, both OU — OU-on negative),
`research/findings/raw/_gap5_nmda_slow/ampa_control_fb60_6seed.json` +
`research/findings/raw/_gap5_nmda_slow/ampa_control_6seed.json` (matched AMPA attribution controls).
Reproducer: `research/runners/_gap5_ca3_nmda_slow_reverberatory_derisk.py`. NO `sim/` edit.

### Sources
- Wang X-J. *Probabilistic decision making by slow reverberation in cortical circuits.* Neuron 36:955-968 (2002).
- Amit D.J., Brunel N. *Model of global spontaneous activity and local structured activity...* Cereb. Cortex 7:237-252 (1997).
- Kopsick J.D., Kilgore J.A., Adam G.C., Ascoli G.A. *Formation and Retrieval of Cell Assemblies in a Biologically
  Realistic Spiking Neural Network Model of Area CA3...* (2024) PMC10996657.
