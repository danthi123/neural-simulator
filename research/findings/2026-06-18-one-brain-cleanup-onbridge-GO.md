# Roadmap phase 2, step 3a — the CLEANUP folds onto the persistent bridge; the full chain runs on one brain: GO

**Date:** 2026-06-18 (the real "one brain" headline arc). **Status:** **GO** (3 seeds × 2 D, 6/6). The cleanup's
matched-filter — concept-score neurons that read the recovered **Q register** via `conj(codebook)` complex synapses,
their membrane (re) = the match score — folds onto the SAME persistent bridge as the store+query (steps 1+2). The
on-bridge cleanup's argmax == the numpy cleanup (the validated oracle) 100%, and a random-codebook control collapses
to chance. ⇒ **bind → bundle → store → unbind → cleanup ALL run register→register on ONE persistent bridge with no
host round-trips** — the answer is read straight off the concept neurons' membranes.
**Runner:** `research/runners/_phaseB_onebrain_cleanup_onbridge_derisk.py` | builds on
`2026-06-18-one-brain-{register-handoff,fact-store-query}-GO.md`.

## Result — 3 seeds × {D=64, D=128}

| metric | mean | reading |
|---|---|---|
| on-bridge cleanup self-recovery | **1.000** (6/6) | reads Q, picks the right concept |
| == numpy cleanup (the oracle) | **1.000** (6/6) | identical to the validated cleanup |
| random-codebook control (no real match) | 0.17 | collapses to ≈chance |

The cleanup is the same RF complex-synapse matvec as unbind (one matched-filter step Q→concept), so it composes
naturally onto the persistent bridge — concept neuron `j` accumulates `conj(code_j) · Q`, and `Re(c_j)` is exactly
the numpy cosine score (per `_spiking_cleanup` stage 1). The final argmax over the on-bridge membranes is the answer.

## Where the one-brain pipeline stands

bind ✓ · bundle ✓ · store-in-register ✓ · unbind ✓ · cleanup matched-filter ✓ — all register→register on one
persistent bridge, no host round-trips between ops. The conversational CORE (store a fact, query a role, clean up to
the answer) runs on one brain.

## Honest scope + next

- The final **selection** is still a host argmax over the on-bridge scores. The validated spiking WTA
  (`_spiking_cleanup` stage 2, Izhikevich) folds it to a spiking winner-take-all (co-resident Izhikevich neurons
  driven by the scores → argmax-over-firing) — the next biologization (analogous to "read which pool fired").
- Remaining to close the full who/what TURN on one bridge: the spiking WTA selection, the **familiarity-gate moat**
  (abstain when no concept scores high), and the **parser front-end** (drive the operand registers from the parser's
  role firing) so comprehend→store→query→answer is one spiking flow, host = text I/O only.
- Top risk stays phase coherence as the chain lengthens (multi-window settle mitigates; a phase-latch on the stored
  register is the fallback). So far the chain holds to 5 ops at 1.000.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_cleanup_onbridge_derisk --seeds 42,43,44 --dims 64,128
```
