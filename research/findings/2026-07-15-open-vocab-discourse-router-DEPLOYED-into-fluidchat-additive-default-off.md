# DEPLOYED: the open-vocab discourse-marker router wired into the flagship `FluidChat` (additive, default-off, byte-identical when off; CI-guarded, 6-seed)

**Date:** 2026-07-15 · **Status:** DEPLOYMENT of the de-risked mechanism (`2026-07-15-discourse-marker-routing-...`). Additive, default-off, moat-preserved. NO `sim/` edit. Closes the discourse-routing wire-in arc to capacity (per `feedback_close_arcs_to_full_capacity`).

## What shipped
- **`research/runners/_discourse_marker_router.py` — `DiscourseMarkerRouter`.** Replaces the fluid console's CLOSED keyword-set discourse routing (`"share"/"compare"/"classif" in tset`) with a LEARNED semantic router: the canonical `ppmi()` (EMERGE-30/62) over a distributional discourse corpus → per-intent centroids → nearest-intent. `route(tokens)` → the intent (SHARE / COMPARE / TAXONOMY) or None. Emergence bar: a hand-coded keyword rule becomes structure LEARNED from distributional co-occurrence.
- **The MOAT is code-dictionary membership:** only curated discourse markers + synonyms have PPMI codes, so any other query word → None → fallthrough to the neural wh-parse (`_neural_parse`, already neural). The secondary novelty threshold is calibrated over all known markers so it never rejects a known one (the within-cos 0.81 / between-cos 0.05 separation makes nearest-centroid robust).
- **The open-vocab capability the closed keyword set lacks:** a novel synonym ("versus" / "alike" / "akin" / "lineage" / "taxonomy") routes to the correct intent by semantic proximity — the keyword set would miss it.
- **`FluidChat(open_vocab_dispatch=False)` additive flag** (`_fluidconv_chat_repl.py`): when off (default), the three discourse checks are the VERBATIM keyword conditions (byte-identical); when on, `_mi = self._marker_router.route(toks)` and each check becomes `_mi == "SHARE"/"COMPARE"/"TAXONOMY"`. The router is built lazily in `__init__` only when the flag is on. Module imports clean; the WHY-taxonomic and all downstream handlers are unchanged.
- **`tests/test_discourse_marker_router.py`** — ckpt-free / GPU-free / pure-numpy CI guard, **24 passed** (6 seeds × 4): attested markers route; novel synonyms route open-vocabulary; OOD/wh tokens → None fallthrough; first-recognised-marker-wins.

## Verification
- Router smoke 8/8 (attested + novel-synonym open-vocab + OOD fallthrough); CI 24/24 (6-seed).
- De-risk backing (prior findings, 6-seed): held-out-synonym nearest-intent 1.000, OOD→None 1.000, within-cos 0.81 / between-cos 0.05 (real PPMI).
- Default path byte-identical by construction (ternary else = verbatim original; `_marker_router is None` off).
- **Honest scope:** the FluidChat END-TO-END turn (`fc.turn("compare dogs and cats")`) needs the local 21M gen artifacts (BPE + ckpt), absent on this CPU box — the same reason `tests/test_fluidconv_chat_repl.py` skips here. The router + wire-in logic + byte-identity are validated ckpt-free; the end-to-end console smoke on a novel-synonym query (e.g. "how do dogs and cats **differ**?" routing to COMPARE) is a follow-on where the gen artifacts exist. The router's `code` dictionary is the curated marker+synonym set; a TRULY-novel word (not in the corpus) → None → fallthrough (correct, safe), so extending the open vocabulary = extending the discourse corpus the router is built from.

## Where this sits in the dispatch arc
The dispatch wire-in decomposed (this session, a0/a-1 discipline): wh→type is already neural (Phase-7); membership is a familiarity gate (already GO); discourse-marker routing is semantic-nearest + novelty (this deployment). The GO deep-credit COMPOSITIONAL dispatch is reserved for a future subject-category × question-type intent grammar (absent from today's router). NO `sim/` edit anywhere in the arc.
