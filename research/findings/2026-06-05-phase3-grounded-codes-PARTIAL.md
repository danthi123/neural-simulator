# Conversion Phase 3 (cheat A, grounded codes) — grounding INTERFACE validated; full grounding is the BOUNDARY — 2026-06-05

Third phase of the cheat-conversion plan (`docs/plans/2026-06-05-conversational-cheat-conversion-plan.md`). The RF
composer's concept/role codes are `rng.uniform` random phases (cheat A) — not grounded in sensory input nor learned.
This is the documented "recognition front-end / code grounding" hard problem. Per the plan + the cheat-A research
(`2026-06-05-cheat-A-code-grounding-research.md`), cheat A is **TRACTABLE for the groundable subset; PARTIAL at scale;
abstract concepts are the embodied-cognition limit; deep grounding is a multi-month arc.** So this phase delivers the
TRACTABLE INCREMENT (the grounding INTERFACE) and DISCLOSES the boundary — it does not claim full grounding.

## De-risk: the RF composer works on REAL V1-grounded codes (GO, multi-seed)
`research/findings/raw/_phase3_grounded_codes_derisk.py`: each concept = a distinct visual stimulus (oriented bars +
spots) → the REAL biological V1 Gabor bank (`sim/visual_cortex.py` `build_v1_simple_weights`, Hubel-Wiesel simple
cells, 8192 cells) → an 8192-d grounded code → a complex random projection to D → phases `φ = angle(P·v1code)/2π`.
Inject these grounded codes into the composer; store facts; query; abstain.

| seed (D=256) | grounded who/what | random-code baseline | abstain | grounded-code cosine (mean / max) |
|---|---|---|---|---|
| 42 | 6/6 | 6/6 | 1/1 | 0.175 / 0.579 |
| 43 | 6/6 | 6/6 | 1/1 | 0.188 / 0.539 |
| 44 | 6/6 | 6/6 | 1/1 | 0.181 / 0.526 |

**Verdict: GO.** The RF composer's who/what Q&A + the no-confab moat work on SENSORY-GROUNDED codes at the
random-code baseline, multi-seed. The grounded codes carry genuine V1 structure (mean phasor-cosine ~0.18, vs random's
~0 — the V1 separability the cheat4 finding measured), and the composition pipeline (bind/bundle/unbind/cleanup) is
indifferent to it. The grounding INTERFACE works on the RF phasor substrate.

## Integration: the `grounded_codes` opt-in
`RFPhasorComposer(grounded_codes={word: phases[D]})` and `BrainConversationalAgent(grounded_codes=...)` override the
random codes for the provided words (random for the rest). Regression test `test_..._grounded_codes_interface`
(multi-seed): the composer USES the provided codes and still does Q&A + abstention. Default `None` = the random codes
(unchanged). NO `sim/` edits — reuses the existing V1 bank.

## HONEST BOUNDARY (disclosed, not forced — this is PARTIAL, not DONE)
- **The word→stimulus mapping is arbitrary.** There is no real object-image dataset (dog→a dog image, apple→an apple
  image); the de-risk maps words to bars/spots. The codes are V1-GROUNDED (from real sensory features) but not
  SEMANTICALLY grounded (dog ≠ a dog image). Real semantic grounding needs an image dataset — out of scope.
- **Abstract concepts have no canonical image** (motor directions, verbs, function words). Grounding them in raw
  sensation is the embodied-cognition limit (best case: grounded in the motor/lexical referent, not vision).
- **Full on-bridge decorrelation at 320 concepts is seed-fragile** (the project's concept-pool arc; the labelled-ZCA
  stand-in may remain a disclosed boundary).
- So cheat A is **PARTIALLY converted: the grounding interface is validated + a clean opt-in shipped; the codes CAN be
  sensory-grounded. Producing meaningful grounded codes at the full vocabulary (real images + abstract concepts) is
  the named boundary** — a multi-month arc, the recognition-front-end hard problem, not a shortcut to paper over.

## Artifacts
`research/findings/raw/_phase3_grounded_codes_derisk.py` (V1-grounded codes → RF composer, multi-seed). Reuses
`sim/visual_cortex.py` (V1 Gabor) + `research/runners/_visual_grounding_probe.py` (stimulus rendering, the 2026-06-04
cheat4 visual-grounding work). NO sim/ edits.
