# Promoted CoreSimComposer is vocab-robust to V=320 (multi-seed, hard correlated codes) — 2026-06-04

**One line:** The consolidation deliverable — `CoreSimComposer`, the conversational composer now running ON the
core `SimulationBridge` — inherits the validated V=320 vocab-robustness of the archived `_insubstrate` coincidence
bind/unbind (pillar n=111). Multi-seed (42/43/44), at the **hard** correlated-code regime (between-cos ~0.60,
matching the `denoise64` substrate codes' ~0.70): **what 12/12, who 12/12, no-confab abstain 12/12 every seed**.

## Why this run (cheap-first, in-scope)

Immediately after the consolidation arc completed (the conversational pipeline moved onto the core sim), the
natural cheap-first question is: *did the production module inherit the property pillar n=111 validated on the
probes?* Pillar n=111 established the spiking composition layer is **vocabulary-robust to V=320** (spiking cleanup
recovery 1.000 up to 320) and that "the real ~320 limit lives in the RECOGNITION FRONT-END, not in composition."
This run confirms the **promoted** `CoreSimComposer` did not lose that in promotion. This is validation of the
consolidation deliverable at production vocab — **not** a months-scale scaling commitment.

The composer exposes a `concepts=` constructor hook, so 320 distinct concept codes can be injected directly
(bypassing the V=16 `denoise64` cache) and the spiking bind/unbind exercised at production vocab.

## Result (`research/findings/raw/_core_composer_v320_capacity_probe.py`, GPU, ~6400-neuron bridge)

| seed | code regime | between-cos (mean/max) | what (query_patient) | who (query_agent) | abstain (no-confab moat) |
|---|---|---|---|---|---|
| 42 | near-orthogonal | -0.000 / 0.144 | 12/12 | 12/12 | 12/12 |
| 42 | correlated (ρ=0.6) | 0.601 / 0.671 | 12/12 | 12/12 | 12/12 |
| 43 | correlated (ρ=0.6) | ~0.60 | 12/12 | 12/12 | 12/12 |
| 44 | correlated (ρ=0.6) | ~0.60 | 12/12 | 12/12 | 12/12 |

The **hard** regime (ρ=0.6, between-cos ~0.60) is the decisive one — it matches the real `denoise64` substrate
codes' ~0.70 correlation, where naive cleanup would be expected to confuse 320 highly-overlapping codes. It holds
at 100% across all three seeds, and critically the **abstention threshold transfers**: the no-confab moat (return
`None` when no stored fact's agent matches the cue) is perfect at 320 concepts without recalibration.

## Relational-memory axis: a 20-fact KB at V=320 (multi-seed, hard codes)

The K=1 rows above vary the vocabulary at single-fact load. The complementary axis is **relational-memory depth**:
store a full knowledge base of distinct facts in ONE composer and query each. At V=320, hard codes (ρ=0.6), a
**20-fact KB** with distinct (agent, action) cues:

| seed | between-cos | what (each fact's patient) | who (each fact's agent) | abstain (20 unstored cues) |
|---|---|---|---|---|
| 42 | 0.601 | 20/20 | 20/20 | 20/20 |
| 43 | 0.600 | 20/20 | 20/20 | 20/20 |
| 44 | 0.598 | 20/20 | 20/20 | 20/20 |

All three seeds: every stored fact recalled correctly (who AND what), and every one of 20 never-stored cues
correctly abstained — no cross-fact confusion and no confabulation at a 20-fact KB on 320-word vocab. This matches
the separate n=113 result ("KB ≥30 facts perfect at 320 scale") and confirms the **promoted** module carries it.
(`research/findings/raw/_core_composer_v320_kb20_rho60_s{42,43,44}.json`.)

## Honest scope (what this is and isn't)

- **Is:** the COMPOSITION layer (spiking coincidence bind/unbind + cleanup + the abstention moat) of the promoted
  brain composer, vocab-robust to 320 concepts, multi-seed, even at high code correlation. Confirms the promotion
  preserved pillar n=111.
- **Isn't (yet):** (1) **K=1 load** — a single fact is stored then queried; this is the **vocabulary** axis, NOT the
  binding-load axis (the K~4–6 bindings-per-structure cap is separately validated in n=111). (2) Codes are
  **injected synthetic** (random near-orthogonal, or shared-component correlated), NOT captured from a real
  320-concept bridge's concept pools — so this validates the composition layer's vocab headroom, not the **grounded
  front-end** (getting 320 clean distinct codes FROM the substrate = the documented hard problem; n=111: the real
  ~320 limit is recognition, not composition). The grounded 320-concept brain agent (capture concept-pool activity
  for >16 concepts, or wire the sparse-distributed G.20 codes through the composer) is the heavier follow-on and a
  genuine owner-steerable scaling direction.

## Files

- `research/findings/raw/_core_composer_v320_capacity_probe.py` (the probe, with `--rho` correlation knob)
- `research/findings/raw/_core_composer_v320_capacity_s42.json` (near-orthogonal) +
  `_core_composer_v320_capacity_rho60_s{42,43,44}.json` (correlated, multi-seed)
- Consolidation finding: `2026-06-04-conversational-pipeline-consolidated-onto-core-sim.md`
