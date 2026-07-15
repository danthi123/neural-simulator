# MECHANISM CORRECTION: the fluid console's discourse-marker routing is SEMANTIC-NEAREST-INTENT + a NOVELTY THRESHOLD (both already-GO), NOT a deep-credit compositional dispatch — the deep-credit dispatch classifier has no deployment home in this console's actual routing

**Date:** 2026-07-15 · **Status:** honest test→read-substance mechanism correction (6-seed). Follows the target-correction (`2026-07-15-wire-in-target-correction-...`). NEGATIVE for "deep credit is the discourse-router mechanism" + a POSITIVE redirect to the composed GO mechanism. NO `sim/` edit.

## The test
`_discourse_marker_dispatch_derisk.py`: learn the fluid console's discourse-marker→intent routing (`{share,common}`→SHARE, `{compare,different,difference}`→COMPARE, `{classify,trace,ancestry}`→TAXONOMY, wh/content→FALLTHROUGH) over SEMANTIC marker codes (each intent group shares a PPMI-like semantic block + per-word identity), and generalise to a HELD-OUT SYNONYM per group ("alike"/"versus"/"lineage"/"chase") — the open-vocabulary capability a fixed keyword set cannot do. Deep 2-hidden e-prop classifier; 1-NN memorisation floor; OOD marker → FALL (moat); permuted anti-cheat. 6-seed:
```
[marker s42] parity=0.400 HELDOUT-SYN=0.500 (memfloor 1.000) ood->fall=0.000 permuted=0.400
[marker s43] parity=0.800 HELDOUT-SYN=0.500 (memfloor 1.000) ood->fall=1.000 permuted=0.200
[marker s44] parity=1.000 HELDOUT-SYN=0.750 (memfloor 1.000) ood->fall=1.000 permuted=0.467
[marker s100] parity=1.000 HELDOUT-SYN=0.750 (memfloor 1.000) ood->fall=1.000
[marker s101] parity=1.000 HELDOUT-SYN=0.500 (memfloor 1.000) ood->fall=1.000
[marker s102] parity=1.000 HELDOUT-SYN=1.000 (memfloor 1.000) ood->fall=1.000
```

## The substance (why 0/6 GO is a mechanism finding, not a failure)
- **`memfloor=1.000` every seed** — a plain 1-NN over the raw codes ALREADY routes every held-out synonym perfectly, because the synonym shares its group's SEMANTIC block, so its L2-nearest training marker is a same-group marker. The routing is **semantic nearest-neighbour**, not a composition. The held-out synonym generalises by proximity, which is exactly what the PPMI stream cortex (already GO) gives for free.
- **The deep-credit net adds nothing here AND is less stable** — HELDOUT-SYN 0.5–1.0 (≤ the 1.0 memfloor) and parity collapses on 2 seeds (tiny 16-example, 4-class task). Deep credit's value is COMPOSITIONAL generalisation (subject-category × question-type → intent, validated GO in `_learned_dispatch_derisk`); discourse-marker routing has no such composition — the intent is a function of the marker's SEMANTIC IDENTITY alone.
- **The one piece 1-NN cannot do — OOD→fallthrough — is the NOVELTY THRESHOLD.** An OOD marker (novel semantic block) has a random nearest neighbour → 1-NN would force it into some marker intent; the correct behaviour (route to the neural-parse fallthrough) needs a DISTANCE/FAMILIARITY threshold = the Bogacz-Brown novelty gate (already GO). The deep net got this right on 5/6 but not by design.

## The corrected wire-in mechanism (composes only GO pieces; NO deep credit)
Replace `FluidChat.turn`'s keyword-set checks with: **PPMI semantic marker codes → nearest-intent readout, gated by a novelty threshold** (near a known intent cluster → that intent; far from all → the neural-parse fallthrough = the moat). This makes discourse routing OPEN-VOCABULARY (a novel synonym "versus"/"unlike"/"lineage" routes correctly by semantic proximity) — a real capability add over the closed keyword set — using the PPMI cortex + the familiarity gate, both already validated. The deep-credit dispatch classifier is NOT the tool for this subproblem.

## Where the deep-credit compositional dispatch (validated GO) actually belongs
Not in the current fluid console: its routing decomposes into (wh→type: already neural, Phase-7 `_neural_parse`) + (discourse-marker→intent: semantic-nearest + novelty, above) + (membership: familiarity gate). None is a subject-category × question-type COMPOSITION. The deep-credit dispatch would earn its keep only on a console whose response-FRAME is a genuine composition of (subject category) × (question type) — i.e. a richer intent grammar than this console currently has. So the GO deep-credit dispatch is a validated capability held in reserve for that richer console, not a drop-in for today's fluid router.

## POSITIVE CONTROL — the corrected mechanism is a CLEAN 6-seed GO
The composed mechanism (per-intent centroid over the SEMANTIC block = PPMI cortex + nearest-intent + a novelty radius threshold from the max train-marker-to-centroid distance) on the same task's codes:
```
[nearest+thr s42..102] heldout-synonym=1.000  ood->fall=1.000   (all 6 seeds)
[nearest+thr] mean heldout-synonym=1.000  ood->fall=1.000  GO=True
```
So the open-vocabulary discourse routing works decisively: a novel synonym routes to the correct intent by semantic proximity (1.000), and an OOD marker routes to the neural-parse fallthrough by the novelty threshold (1.000) — the moat. This is the de-risked wire-in mechanism; it uses ONLY already-GO pieces (PPMI semantic codes + the Bogacz-Brown novelty gate) and needs NO deep credit.

## REAL-PPMI prerequisite CONFIRMED (not just faithful-geometry) — 6-seed GO
The deployment needs the markers to carry SEMANTIC codes (the fluid console currently uses RANDOM codes, `_fluidconv_chat_repl.py:171`). Confirmed the canonical `ppmi()` machinery (EMERGE-30/62) over a distributional discourse corpus (each intent group's markers co-occur with that intent's CONTEXT words — contrastive `vs/unlike/whereas/contrast` · commonality `together/mutual/jointly` · taxonomy `kind/descends/genus` · interrogative `is/the/eat`, overlapping not a hard partition) produces marker codes that cluster by intent:
```
[realPPMI s42..102] heldout-synonym-nearest-intent=1.000 (all 6)   within-cos=0.81  between-cos=0.05  (16x separation)
[realPPMI] mean heldout-synonym=1.000  GO=True
```
So a novel synonym (versus/alike/lineage) routes to the correct intent by REAL distributional-semantic proximity. The wire-in is de-risked END-TO-END with real machinery: discourse corpus → `ppmi()` → marker codes → nearest-intent + novelty threshold → open-vocab routing + OOD-fallthrough moat. The remaining work is the FluidChat integration (source marker codes from PPMI instead of random; replace the keyword checks; additive default-off; 6-seed parity + open-vocab + moat).

## Honest arc status
- Deep-credit compositional dispatch: **GO** (interpolative), reserved for a future compositional intent grammar.
- Fluid discourse-marker routing: mechanism = **semantic-nearest-intent + novelty threshold** (composes PPMI + familiarity gate, both GO) → open-vocabulary routing. This is the deployable wire-in; it does NOT need deep credit.
- Runners `_learned_dispatch_console_wire_derisk.py` + `_discourse_marker_dispatch_derisk.py` retained as the artifacts that produced the target + mechanism corrections.
