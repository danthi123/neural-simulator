# OPEN-WORLD INFERENCE beyond stored structure — spreading-activation semantic completion (GO, 6-seed): an ADJACENT unknown (a concept with NO stored fact) gets a HEDGED best-guess property by spreading activation to its nearest neighbour in the brain's own learned co-occurrence code space (guess-acc 1.000, coverage 1.000), the DERANGED-code control collapses to chance (0.229), a genuinely DISJOINT unknown still HARD-ABSTAINS (1.000 — moat preserved), and confidence tracks neighbourhood tightness (gap 0.71). The no-confab moat is UPGRADED (graded best-guess for the adjacent unknown; hard-abstain for the disjoint), not weakened. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_spreading_activation_completion_derisk.py`. Reuse-by-import (`learn_stream_codes` emergent co-occurrence codes). numpy. NO `sim/` edit.
**Research gate:** `2026-07-08-open-domain-grounded-conversation-frontier-research-gate.md` (ranked this #1 cheapest highest-value).
**Verdict:** GO (6-seed) — open-world inference beyond stored structure, moat upgraded.

## Why this ran (the open-domain research gate's #1)
The open-domain research gate isolated the residual to "any topic": breadth = a data lever (in flight); fluent open generation = the minimized transformer's job (scaffold); the genuinely-NEW cheap SPIKING frontier = open-world INFERENCE beyond stored structure. Its #1 cheapest highest-value mechanism: spreading-activation semantic completion — turn "bounded inventory + hard abstain" into "grounded best-guess about the ADJACENT unknown, flagged as a guess" (the moat-as-plus upgrade). Biology: Collins-Loftus spreading activation; Rogers-McClelland semantic cognition; CA3 pattern completion (catalog D.13).

## The mechanism
A query about a concept with NO stored fact does not immediately hard-abstain. Activation spreads through the brain's own LEARNED co-occurrence codes to the nearest KNOWN concept (max code cosine). If that neighbour clears a tightness threshold theta (calibrated from the known set's own 20th-percentile neighbour cosine, frozen per seed), its property is offered as a HEDGED, graded-confidence best guess (confidence = the cosine). If no neighbour clears theta (a genuinely disjoint/novel code), the brain HARD-ABSTAINS — the moat holds for the truly-unknown; only the ADJACENT unknown gets a guess.

## The result — 6-seed (4 categories x 8 members, 3 held-out per category)
```
guess-acc (adjacent unknown -> correct category property)   = 1.000 every seed
coverage  (adjacent unknowns that get a guess)              = 1.000 every seed
DERANGED  (shuffle codes<->concepts, 10-shuffle avg)        = 0.229  (~= chance 0.25 -> the learned similarity is load-bearing)
DISJOINT-abstain (random novel code -> hard-abstain)        = 1.000 every seed  (moat preserved for the truly-unknown)
confidence gap (adjacent nearest-cos 1.00 vs disjoint 0.30) = 0.71   (confidence tracks neighbourhood tightness)
```

## Anti-cheats (all from the research gate, all pass)
- **(A) DERANGED neighbourhood** — shuffling which code sits at which known concept makes the nearest neighbour random, so the guess collapses to chance (0.229 ~= 0.25). The completion rides the learned code similarity, not coincidence. (Averaged over 10 shuffles; a single shuffle is high-variance at 4 categories — the same averaging the taxonomy derangements use.)
- **(B) DISJOINT code** — a genuinely novel random unit code has no neighbour above theta, so it hard-abstains (1.000). The moat is NOT weakened for the truly-unknown; the guess is offered ONLY for the adjacent unknown.
- **(C) CONFIDENCE tracks tightness** — the adjacent-unknown nearest-cosine (~1.00) far exceeds the disjoint nearest-cosine (~0.30), gap 0.71, so the graded confidence is meaningful (a tight neighbourhood -> a confident hedge; a loose one -> abstain).

## Honest scope
Validated on a controlled synthetic category-structured co-occurrence stream (clean anti-cheat controls). It demonstrates the mechanism cleanly; the next steps are (1) run it on REAL codes (TinyStories / the natural-text animal tree), and (2) wire the hedged completion into the console (a "probably, like a neighbour" phrasing) behind a default-off flag, keeping the hard-abstain for disjoint. This is a graded best-guess (a hedge), NOT a definite answer — deliberately, per the moat-as-plus directive.

## What this establishes
The first open-world inference mechanism beyond the bounded stored inventory: the brain guesses about the ADJACENT unknown from its own learned semantic neighbourhood, with a graded confidence, while still hard-abstaining on the genuinely-disjoint unknown — the no-confab moat upgraded from a hard gate to a graded hedge exactly where the research gate prescribed. Follow-on: real-code validation; console wiring; the research gate's #2 (schema/script default-filling via the HTM next-state predictor) and #3 (analogical transfer on corpus-mined clean relational codes).

## Files
`research/runners/_realcorpus_spreading_activation_completion_derisk.py`; reuses `_emergent_vocab_breadth_scale_derisk.learn_stream_codes`, `_realcorpus_inheritance_rung1_derisk._unit_rows`. Research gate: `2026-07-08-open-domain-grounded-conversation-frontier-research-gate.md`.
