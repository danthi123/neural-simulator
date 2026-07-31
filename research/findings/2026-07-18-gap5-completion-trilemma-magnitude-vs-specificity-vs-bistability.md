---
type: finding
status: contributing
date: 2026-07-18
mechanism: ca3-completion
---

# Gap #5 — the FROZEN cue-specific completion TRILEMMA (magnitude vs specificity vs bistability), and the mechanisms that address each horn

**2026-07-18.** Continuing the frozen-attractor completion work (after retracting the Wang plasticity+noise confound
and finding assembly-selective inhibition gives specificity). Pushing toward a GO working point (cue ≥ 0.20, cue ≥ 3×
perm, nocue ≤ 0.10) surfaced a precise three-way tension on the point-neuron rate-coded substrate.

## The trilemma (each corner tested, seed 42, FROZEN recall + OU off)

| regime | config | cue | nocue | perm | reading |
|---|---|---|---|---|---|
| **strong attractor** | strong w (encode kt 15), any inhibition | 0.499 | 0.499 | 0.499 | SELF-SUSTAINS — non-specific, not bistable |
| **specific but weak** | weak w (encode kt 100) + assembly-selective inhib | 0.045 | 0.000 | 0.014 | specific (3.2×) + bistable, but MAGNITUDE ~0 |
| **structural sep, strong w** | zero non-member→member + strong w | 0.499 | 0.499 | 0.499 | strong w STILL self-sustains (leak path not the issue) |
| **all-three combo** | structural_sep + strong w + rate_homeo (bias −2000 cap) | 0.499 | 0.499 | 0.499 | even a −2000 pA per-cell suppressive bias CANNOT quench the self-sustain (recurrent drive > any bias) |

The three constraints pull against each other:
- **MAGNITUDE** (cue ≥ 0.20) needs STRONG within-ensemble weights (a strong correct completion).
- **BISTABILITY** (nocue ≤ 0.10, silent rest) needs the attractor NOT to self-sustain — but a strong recurrent
  attractor on a single-compartment point soma HAS NO INTRINSIC BISTABILITY, so strong weights → self-sustain.
- **SPECIFICITY** (cue ≥ 3× perm) needs a permuted cue NOT to complete — but a strong attractor completes from any
  partial activation (large basin), and the leaked-activation path is only part of it (structural separation alone,
  zeroing non-member→member recurrents, does NOT fix it because strong weights self-sustain regardless).

## The mechanisms built, and which horn each addresses (all default-off / byte-identical, NO sim/ edit)

1. **Assembly-selective inhibition** (`selective_inhib`, Kim-Kim 2025 "spare your own engram") — addresses SPECIFICITY:
   spare the assembly's cells from the basket I→E, inhibit non-members → a permuted cue's non-members are quenched
   before they avalanche. Gives cue/perm 0.94 → 3.19 (the FIRST cue-specific frozen completion in the project) — but in
   the WEAK regime only (sparing a strong-weight assembly disinhibits it → self-sustain).
2. **`rate_homeo`** (Turrigiano per-neuron intrinsic-excitability homeostatic) — addresses BISTABILITY: auto-calibrate a
   per-cell suppressive bias so the self-sustaining rest state is quenched to a genuine low state.
3. **`structural_sep`** (DG pattern separation: zero non-member→member recurrents) — addresses SPECIFICITY structurally:
   a permuted cue's non-members cannot reach the assembly. But alone it does NOT resolve the trilemma (strong weights
   self-sustain even with no leak path).
4. **`recall_k_thresh`** — decouple encode (low, strong weights) vs recall (high, specific) dAP threshold.

## The root cause (the honest verdict on the METHOD)

The single-compartment point-neuron soma has NO INTRINSIC BISTABILITY: a recurrent attractor strong enough to complete
a partial cue is also strong enough to self-sustain (there is no stable silent-rest fixed point when the recurrent gain
is high). So magnitude and bistability are in direct opposition, and specificity requires shrinking the basin (which
weakens completion). This is the SAME family as the documented point-neuron limits (the Mikulasch-Priesemann
whitening/decorrelation wall, the graded-magnitude/divisive-normalization family): a computation that biology does with
DENDRITIC nonlinearity, which a point soma cannot.

## Next mechanism (per THE LAW — not a wall, an undiscovered mechanism)

The missing piece is **intrinsic cellular bistability** — a DENDRITIC PLATEAU / NMDA-dependent UP-state that HOLDS a
stable high fixed point per-cell (not just the dAP coincidence READOUT, which detects co-activity but cannot hold). A
cell with a bistable dendrite has TWO stable states (silent + plateau) at the SAME recurrent input, so the network can
have a silent rest AND a completed state without the recurrent gain being cranked to self-sustain — resolving the
magnitude-vs-bistability opposition. Then structural separation + assembly-selective inhibition give specificity on top.
This is the Wang-2002 intent (somatic slow-NMDA bistability) done at the DENDRITE (Major-Larkum-Schiller dendritic
plateau; the project's `enable_two_compartment_dap` + `apical_R`/`apical_g_couple` machinery is the substrate for it).
Research-gate the dendritic-bistability parameterization, then combine with structural_sep + selective_inhib, 6-seed.
