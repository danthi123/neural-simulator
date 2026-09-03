---
type: finding
status: contributing
date: 2026-09-03
mechanism: configural-binding conjunction sweep (--conj-bind fixed --conj-mode prod) + adversarial anti-cheat verification of a 6/6 capability_go crossing
lane: vision (identity readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: capability_go crosses (6/6 at prod/n1152) but the ANTI-CHEAT shows it is MOSTLY CAPACITY (z-norm + high feature count), binding a marginal +1-seed boost — and this CORRECTS the earlier satdiv-based "readout axis exhausted" claim
artifacts:
  - research/findings/raw/lanes/perception/conjbind_prod_n1152_6seed.json
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_6seed.json
  - research/findings/raw/lanes/perception/conjbind_prod_n1152_shuffleoffsets_6seed.json
---

# The vision-readout bar DOES cross — but it's mostly capacity, not binding; the anti-cheat caught it

**Status:** the first vision-identity `capability_go` crossing in the arc (6/6), BUT the adversarial verification shows it is primarily a CAPACITY effect (feature count under z-norm), not the configural binding it was built to test. Configural binding is a real but MARGINAL contributor. The verification is the deliverable — without it this would have been an over-claimed "binding breakthrough."

## Result — the crossing, decomposed by anti-cheat

<!--derived-->
From the cited artifacts, deepest bar `capability_go` fraction (≥5/6 = strict GO), 6-seed (42/43/44/100/101/102):

<!--derived-->
| arm (prod mode, conj-offset-max 4, z-norm, ridge 0.5) | capability_go |
|---|---|
| configural binding, n1152 (right offsets) | 6/6 |
| **width-matched flat ELM, n_s2 1152, NO binding** | **5/6** |
| binding n1152, offsets SHUFFLED (lesion) | 2/6 |
| binding n1152, Δ=0 degenerate (same-location AND) | 2/6 |
| binding n1024 / n1280 (peak-breadth) | 5/6 / 4/6 |

<!--derived-->
Reading: the correct-offset binding (6/6) beats the flat control (5/6) beats the wrong-offset binding (2/6). So (a) the relative-offset binding IS load-bearing — shuffling the offsets collapses 6/6→2/6, and a wrong conjunction is worse than a plain feature; (b) BUT a flat ELM with the same feature count (no binding at all) already clears 5/6 — so the crossing is MOSTLY capacity, with binding adding a marginal +1-seed boost (6/6 vs 5/6). The n1152 peak is also slightly non-monotonic (n1024 5/6, n1152 6/6, n1280 4/6), consistent with binding contributing near the seed-variance margin.

## The correction it forces (drift #12 — a stale claim falsified)

<!--derived-->
The earlier finding `2026-09-03-...satdiv-width... readout-axis-exhausted` concluded the frozen-random-S2-bank readout axis was exhausted (satdiv, all configs 0/6 `capability_go`). That was drawn at the SATDIV operating point. **This sweep falsifies it for Z-NORM: z-norm + a high feature count (~1152) crosses the strict bar (5/6 flat, 6/6 with binding) — something no satdiv config reached.** The readout axis was not exhausted; the satdiv normalization simply does not scale with feature count the way z-norm does. The "next mechanism = configural binding" framing is therefore tempered: capacity (features under z-norm) is the primary driver of the crossing; configural binding is a genuine but marginal add.

## Honest verdict + next

Configural binding is a REAL contributor (offset-load-bearing, +1 seed over capacity) but NOT the primary driver of the crossing — capacity under z-norm is. The genuine open questions: (1) does the flat-z-norm-capacity crossing (5/6) hold under the arc's OTHER anti-cheats (held-out position, scramble nulls) or is it partly an ELM-overfit at high width; (2) can learned/selective conjunctions (vs fixed-random) lift binding's marginal boost into a robust, capacity-independent gain. The adversarial width-matched control is what kept this honest — it is now a required arm for any high-feature-count vision-readout GO.

## Reproduce

```bash
# prod-n1152 binding (6/6) vs its width-matched flat control (5/6) — the decisive pair:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-mode prod --conj-n 1152 --conj-offset-max 4 \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/lanes/perception/conjbind_prod_n1152_6seed.json
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind none --n-s2 1152 \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_6seed.json
```
