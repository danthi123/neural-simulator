# Roadmap phase 2, step 3b — the no-confab MOAT reads off the persistent bridge (peak score = familiarity): GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** (3 seeds × 2 D, 6/6, clean separation
every seed/D). The cleanup matched-filter's **peak score** (the max concept-neuron membrane on the persistent
bridge, step 3a) IS a neural familiarity signal: a stored, correctly-cued fact makes the recovered Q strongly match
one concept (HIGH peak); an unbound query (a role the fact lacks) makes Q cross-talk noise that matches nothing (LOW
peak). A threshold on that on-bridge peak = abstain-vs-answer — the no-confab moat, read off the substrate, **no host
equality check**.
**Runner:** `research/runners/_phaseB_onebrain_moat_onbridge_derisk.py` | **Raw:**
`research/findings/raw/_phaseB_onebrain_moat_onbridge.json`

## Result — 3 seeds × {D=64, D=128}

| metric | result |
|---|---|
| answer accuracy (bound roles) | **1.000** |
| bound peak min vs unbound peak max | e.g. 380M vs 126M, 745M vs 212M — **bound ≈ 2–3× unbound, every seed/D** |
| clean separation (every bound > every unbound) | **6/6** |
| bound > threshold / unbound < threshold (threshold = measured midpoint, not tuned) | 1.000 / 1.000 |

The bound/unbound peak **ratio** (~2–3×) is the robust, seed-stable signal (the absolute membrane values are large
because the RF matched-filter accumulates unnormalized; the ratio is what gates).

## Where the one-brain pipeline stands

**Four GO steps** on ONE persistent bridge, register→register, no host round-trips between ops:
- step 1 — synaptic phase handoff (bind→unbind),
- step 2 — full fact store + query (bind→bundle→store→unbind),
- step 3a — cleanup matched-filter (concept neurons read Q, membrane = score),
- step 3b — the moat (peak score gates abstain-vs-answer).

⇒ the full who/what LOOP — store a fact, query a role, clean up to the answer, **abstain when there is no fact** —
runs on one brain. The no-confab moat is preserved and is now a substrate-read familiarity signal.

## Honest scope + next

- The final **selection** is still a host argmax over the on-bridge scores (the "which neuron won" read-out, like
  reading which motor pool fired). Folding the validated spiking Izhikevich WTA (`_spiking_cleanup` stage 2) makes it
  spiking — the next biologization.
- The **parser front-end** (drive the operand registers from the parser's role firing) closes comprehend→store→
  query→answer as one spiking flow, host = text I/O only. That completes the integrated one-brain who/what turn.
- Then phase 3: make fully-spiking the default + retire the legacy numpy production paths (keeping numpy as the test
  oracle).

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_moat_onbridge_derisk --seeds 42,43,44 --dims 64,128
```
