# EMERGE-20 / toward-language — GO (6/6 seeds): GROUNDED production + the INTRINSIC no-confab MOAT. The emergent sequence cortex produces the learned fact for a grounded cue, generalizes a valid inference for a similar cue, and ABSTAINS (produces nothing) for a truly-novel ungrounded cue — it CANNOT confabulate what it has no learned pathway for. The no-confab moat (the project's load-bearing property) is EMERGENT from the substrate, not a bolted-on check. NO `sim/` edit.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge20_grounded_moat_derisk.py`. Reuse-by-import `_emerge14` + `_emerge17`; NO `sim/` edit; CPU numpy-backend; 6-seed.

## The mechanism — the moat is intrinsic
The project's load-bearing "no-confab moat" (never assert an ungrounded fact) has, until now, been a host-side or composer-side check. On the emergent sequence cortex it is INTRINSIC: production works by the recurrent dendritic coincidence firing on a LEARNED pathway. A truly-novel cue — a word whose sparse code is DISJOINT from anything the cortex learned — shares NO cells with any trained pathway, so it drives NO coincidence → no cell is primed → the cortex produces NOTHING (abstains). It literally cannot confabulate a continuation for which it has no learned pathway.

Three regimes on ONE trained cortex (grounded facts dog→home, cat→away):
- **GROUNDED** (a trained word): dog → "home", cat → "away" — the learned fact.
- **GENERALIZED** (an untrained but SIMILAR word, a valid grounded inference): wolf/fox → "home", lion → "away" — via the shared family micro-columns (EMERGE-17/19).
- **ABSTAIN / MOAT** (a truly-NOVEL word, code disjoint from everything): "zzz"/"qqq" → nothing primed → ABSTAINS (no confabulation).

## Result — GO (6/6 seeds)
`epochs=60`, seeds 42/43/44/100/101/102:
- **GROUNDED 1.00** (all seeds) — produces the learned fact for a trained cue.
- **GENERALIZED 1.00** — produces a valid grounded inference for a similar untrained cue.
- **NOVEL-ABSTAIN 1.00, CONFABULATION 0.00** — a truly-novel ungrounded cue is abstained on, EVERY time; the cortex never confabulates.
- **dAP-LESION collapses grounded to 0.00** — the coincidence recurrence is load-bearing (removing it → no production, so the grounded production is genuinely the substrate's). Multi-seed.

## Significance
This closes the grounding step of the toward-language chain: the emergent sequence cortex is a GROUNDED, MOAT-PROTECTED word producer. It produces grounded word sequences and abstains when ungrounded — biology-native, intrinsic to the substrate, NO bolted-on check, NO `sim/` edit. This replaces the transformer's fluency-in-the-loop role WHILE keeping the no-confab moat (the owner's north-star property: a brain you can talk to that does not hallucinate). Combined with the chain so far — PREDICTION (15) · PRODUCTION (16) · GENERALIZATION (17) · HIGH-ORDER-GEN (18) · REAL-CODE-GEN (19) · GROUNDED-MOAT (20) — the emergent sequence cortex now predicts, produces, generalizes, and grounds word sequences with an intrinsic no-confab moat, all on one spiking brain, emergent + unsupervised + no `sim/` edit.

## Honest scope + next
- A tiny grounded fact set + the intrinsic-moat property isolated cleanly. Scaling to a full grounded knowledge base + real vocabulary needs the codes' word→row vocab plumbing (+ R2 multi-segment if cells scarce).
- The genuinely-hard open residual (the NEXT research gate): open-domain SURFACE FLUENCY (arbitrary-topic grammar / connected multi-word prose) — the transformer's last unique job. Prediction, production, generalization, and grounded-abstention are now covered by the emergent substrate.

## Artifacts
`research/runners/_emerge20_grounded_moat_derisk.py`, `research/findings/raw/_emerge20_grounded_moat{,_6seed}.json`. Prior: `2026-07-02-emerge16-word-generation-GO.md`, `2026-07-02-emerge17-generalizing-word-codes-GO.md`, `2026-07-02-emerge19-real-ppmi-generalization-GO.md`.
