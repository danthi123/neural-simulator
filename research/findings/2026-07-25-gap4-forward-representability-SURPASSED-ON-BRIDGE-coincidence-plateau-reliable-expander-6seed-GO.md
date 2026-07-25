# gap#4 forward-representability boundary SURPASSED ON-BRIDGE (spiking, 6-seed GO): the coincidence dendritic-PLATEAU reliable expander breaks the input-drivenness↔reliability tradeoff — the forward is now held-out-linearly separable, so the CPU-rate-GO credit has features to shape (2026-07-25)

## The capstone of the gap#4 arc (this session)
1. **Characterized** the on-bridge blocker: the sparse-spiking FORWARD destroys the input's generalizable class structure
   (levers a–h, 6-seed held-out; input generalizes 0.99 → every hidden readout ≤0.34) — NOT credit, NOT readout, NOT drive,
   NOT population, NOT unpooling. `2026-07-24-gap4-sparse-spiking-forward-representability-degeneracy-characterized-levers-a-h-6seed.md`.
2. **Research-gated** it (5-agent workflow `wf_1f9812d7-0eb`) → the escape is a fixed nonlinear **EXPANSION** (every lever
   a–h attacked compression/credit/readout; none the missing expansion), already validated in our own record (EMERGE-35),
   never tried on gap#4.
3. **Numpy 6-seed GO**: a fixed random-feature expansion lifts held-out LINEAR 0.284 → 0.772 (clean anti-cheats after
   replacing an ill-designed control with label-shuffle). `2026-07-24-gap4-forward-representability-SURPASSED-nonlinear-expansion-numpy-GO...`.
4. **Pinned the on-bridge residual**: a precise INPUT-DRIVENNESS↔RELIABILITY tradeoff (input_cv↑ ⟹ reproducibility↓; all
   points degenerate held-out) — the noisy rate code can't be input-driven AND reliable at once.
   `2026-07-24-gap4-onbridge-expander-residual-is-input-drivenness-vs-reliability-tradeoff...`.
5. **THIS: broke the tradeoff on-bridge** with the named escape — the coincidence dendritic-PLATEAU read.

## The mechanism (reuse-by-import, NO `sim/` edit)
`research/runners/_gap4_plateau_expander_probe.py` (`PlateauExpander`): a fixed decorrelated coincidence EXPANSION on a
real `SimulationBridge` — n_in feature cells → N_COL=200 columns, each sampling SAMP=3 features, coincidence-driven
(`enable_coincidence_detection` + `coincidence_weighted_drive` + `coincidence_plateau_strength=160` + `enable_two_compartment_dap`).
Each input is presented via `_prime_from_winners` (EMERGE-12), which **RESETS the soma + apical** and holds the active
features SYNCHRONOUSLY for 6 steps, so the dendritic plateau `cp_v_apical` rises **deterministically** — no noisy rate
settle, no state carryover. The codon = {columns with `cp_v_apical > FLOOR`}. This is input-driven (coincidence) AND
reliable (plateau threshold-crossing + full reset) **at once** — the exact fix for the reproducibility-0.07 collapse.

## Result — 6-seed held-out GO (semantic-inheritance, k=5, n_ho=27, seeds 42/43/44/100/101/102)
| representation (held-out) | ho-LINEAR | reproducibility | note |
|---|---|---|---|
| INPUT (ceiling) | 0.284 | — | task is nonlinear; the linear structure to create |
| sparse-spiking H2 (the boundary) | ~0.284–0.34 | 0.07 (input-driven) / 0.5 (pinned) | the characterized wall |
| **CODON expand PLATEAU** | **0.611 ±0.047** | **1.000** | **6/6 seeds 0.556–0.704, all ≫ 0.34** |
| non-expanding control (N_COL=n_in) | 0.352 ±0.111 | 1.000 | expansion is load-bearing (+0.26) |
| label-shuffle control | 0.247 ±0.055 | — | ≈ chance → the lift is REAL class structure |
| pool-silence lesion (no active feats) | 0.333 | 0.000 | degenerate → genuinely from column coincidence |

**⇒ 6-seed GO.** The coincidence-plateau expander is perfectly reliable (reproducibility **1.000**, vs 0.07 for the rate
code) AND input-driven (codon varies by input, sparsity 0.627), lifting held-out LINEAR decodability off the boundary
(0.34) to **0.611** — the forward is now held-out-**linearly separable**, which the sparse-spiking forward never was. The
gap#4 forward-representability boundary is SURPASSED on the spiking substrate.

## Verify-go (why I trust this GO — having been burned twice this session)
- **Reproducibility 1.000 is not a constant-codon artifact:** ho-lin 0.611 ≫ chance proves the codon VARIES across inputs
  (input-driven); reproducibility 1.000 proves it's constant FOR THE SAME input (reliable). Both hold — that IS the escape.
- **Not overfitting:** held-out (27 unseen items) + label-shuffle → chance (0.247). **Expansion is load-bearing:**
  non-expand 0.352 ≪ 0.611. **Genuinely from coincidence:** pool-silence → degenerate.
- **6 seeds**, all above 0.55.

## Honest scope + the follow-ons (NOT a full close yet)
- **0.611 = the binary CODON ceiling for this 7-dim continuous input** (matches the numpy codon 0.617), NOT the full
  random-ReLU 0.772. A binary top-k coincidence codon on only 7 continuous features is limited; a **graded/continuous
  reliable expander** (a reliable spiking random-feature/graded-coincidence read) is the follow-on for the full 0.772.
- **This surpasses the FORWARD boundary; it does not by itself deliver gap#4 ACCURACY.** The unblock: the forward is now
  representable, so the **CPU-rate-GO learned interneuron microcircuit credit** (56c90d67) / a trained readout now has
  linearly-separable features to shape on-bridge — the next build (wire the microcircuit/readout onto the plateau-codon
  forward, measure held-out ACCURACY vs the reservoir, 6-seed).

## Verdict (per THE LAW)
- gap#4 on-bridge forward-representability boundary: **SURPASSED, 6-seed GO** — reliable + input-driven + expanded spiking
  code, held-out-linearly separable (0.611 vs boundary 0.34), NO `sim/` edit. The reliability tradeoff is broken by the
  reset-based coincidence-plateau read (biology: the bistable dendritic plateau as a reliable coincidence detector).
- **NEXT:** (1) wire the microcircuit credit / trained readout onto this forward → gap#4 on-bridge ACCURACY; (2) a
  graded/continuous reliable expander for the full 0.772.

## Provenance
`research/runners/_gap4_plateau_expander_probe.py` (+ `scratchpad/plateau_expander{,_6seed}.log`). Reuses EMERGE-35
(`_emerge35_spiking_pooler_derisk.py`), EMERGE-12 `_prime_from_winners`, EMERGE-14 `_host`, `_gap4_representability_probe`.
Builds on the numpy expansion GO (`c2eb6d0c`) + the research gate + the reliability-tradeoff finding (`27be121f`).
