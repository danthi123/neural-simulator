# Hardening the 320-concept stream cortex — piece 1 (the no-confab MOAT): GO, brain-based, seed-43 closed

**Date:** 2026-06-16
**Owner directive:** "Approved" → harden the 320-concept learned-from-conversation cortex.
**Status:** **Piece 1 (the abstention moat) = GO, 3-seed, integrated into the production runner.** Piece 2 (the
on-brain read-out normalization) is next.

## The starting point (already 3-seed validated, with one loose end)

The 320-concept on-bridge stream cortex (CYCLE 96-97, `_phaseB_onbridge_stream_conversation_derisk.py`, 40×8
corpus-grounded taxonomy, 150 000 stream windows/seed) was already validated across 3 seeds: who-Q&A recall
**1.00** every seed; the no-confab abstention moat abstained **1.00 on seeds 42 + 44** (0 false-accepts) but
**0.88 on seed 43** (1 false-accept). Critically, on seed 43 the present/absent confidences were cleanly
separable (+0.464 vs +0.064) — so the single false-accept was a **GATE-PLACEMENT artifact of the FIXED host
threshold** (`GATE=0.25`), not a binding/representation failure.

## The fix (brain-based, not a tuned threshold)

Replace the fixed host confidence threshold with the **learned Bogacz-Brown anti-Hebbian familiarity gate**
(catalog D.04, perirhinal repetition suppression): imprint each stored fact's verb+object composite into a
learned projector; at query time read the novelty `N(x) = ||x||² − xᵀWx` (familiar → ~0 → accept; novel → ~1 →
abstain), threshold at the a-priori unit-norm midpoint 0.5. This gate was already shown *cleaner* than the host
moat on seed-42 codes (`_phaseB_biologize_moat_streamcodes_derisk.py`) — but had **never been tested on seed-43's
own (lower-fidelity) codes**, exactly where the host gate failed. This is that test.

## Result — `_phaseB_harden_320_learned_moat_derisk.py` (CPU, the 3 cached per-seed code sets)

| seed | recall (learned gate) | false-accepts: LEARNED vs HOST | novelty margin | lesion |
|---|---|---|---|---|
| 42 | 1.00 | **0** vs 0 | +0.873 | +0.000 |
| 43 | 1.00 | **0** vs **1** | +0.889 | +0.000 |
| 44 | 1.00 | **0** vs 0 | +0.884 | +0.000 |

**GO.** The learned gate keeps recall **1.00** and drives false-accepts to **0 on every seed — closing seed-43's
one** (which the fixed host threshold left). The novelty separation is far wider than the host confidence gap
(**+0.882** mean vs the host's ~+0.4), so the a-priori 0.5 threshold sits in a large clean margin. **Anti-cheat:**
the 0.5 threshold is a-priori (NOT tuned on the test); the gate imprints only the *stored* facts (train); absent
cues are genuinely absent; lesioning the learned projector collapses the margin to ~0.000 (the decision rides the
learned synapses, not a fixed rule).

## Integrated into the production runner

`_phaseB_onbridge_stream_conversation_derisk.py` now takes `--moat {learned,host}`, **default `learned`** (the
hardened brain-based gate); `host` preserves the original fixed-threshold behaviour for comparison. Re-validated
end-to-end on the 3 cached seeds (loads the stream-learned codes, skips the ~100-min re-stream):

```
MEAN (3 seeds): who-Q&A recall 1.00 | no-confab abstain 1.00 (total false-accepts 0)
                | familiarity gap present +0.453 vs absent +0.051   ==> GO
```

⇒ the 320-concept stream cortex now has a **clean 3-seed GO** (recall 1.00 + abstain 1.00, 0 false-accepts) with
the abstention decision made by a **learned neural familiarity gate**, not a host threshold. The seed-43
loose end is closed without weakening — in fact strengthening — the no-fabrication guarantee.

## Next — piece 2: the on-brain read-out normalization

The per-concept code is currently read out with a host-side `double_center(log1p(·))` normalization. The
brain-based replacement (CYCLE 93b: per-hub spike-frequency adaptation + per-concept feedforward inhibition,
de-risked at ~96% of host) needs folding into the code derivation — which requires re-deriving the codes with
the on-brain circuit (the cached `.npy` are post-host-normalization), i.e. a re-stream. That is the remaining
"fully brain-based" piece. NO `sim/` edit in piece 1.

## Reproduce

```bash
# the cheap-first de-risk (CPU, cached codes):
SIM_BACKEND=numpy python -u -m research.runners._phaseB_harden_320_learned_moat_derisk --seeds 42,43,44
# the hardened production conversation (learned moat, cached codes):
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onbridge_stream_conversation_derisk \
    --seeds 42,43,44 --taxonomy 40x8 --n-per 16 --moat learned \
    --codes-npy research/findings/raw/_phaseB_stream_codes_320_seedSEED.npy
```
