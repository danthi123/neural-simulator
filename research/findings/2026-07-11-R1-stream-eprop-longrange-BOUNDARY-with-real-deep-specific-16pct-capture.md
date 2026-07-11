# R1 (cheap-first, pre-registered): contiguous-stream one-step-local biological credit (e-prop/RFLO) on the long-range LM frontier is an HONEST BOUNDARY on the pre-registered frac — BUT it produces a genuine, DEEP-SPECIFIC, credit-structure-dependent gain (~16% of the learnable deep margin), exactly the Bellec/Murray prediction; + a metric confound to isolate next (BPTT trains the input embedding, e-prop does not)

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_stream_eprop_lm_derisk.py` (torch/GPU; FD grad-check PASS 2.68e-9; built by subagent, controller-verified line-by-line; NO `sim/` edit). TinyStories 2M tokens, V=2000, n_pool=300, block=128, 4 epochs, seed 42, 5 arms. Pre-registered by the research-gate + adversarial-verify workflow (`raw/_biocredit_recurrent_gate_workflow.json`).
**Verdict:** the pre-registered gate returns **BOUNDARY** (`frac_deep < 0.25`), the literature-predicted honest negative — one-step-local recurrent credit does NOT reach full-backprop's within-block deep-context long-range. But reading the substance: e-prop produces a **real, deep-specific, credit-structure-dependent** partial capture (~16% of the learnable deep margin), and the pre-registered denominator conflates input-embedding learning with recurrent credit — isolating that is the immediate next rung.

## The result — margin vs the add-1 stream bigram, by within-block context depth (+ = arm beats the bigram)
| arm | ctx1 | ctx4-8 | ctx9-16 | **DEEP 17-127** | SHALLOW |
|---|---|---|---|---|---|
| fixed_reservoir (echo-state floor) | −3.301 | −3.266 | −3.169 | **−2.760** | −3.279 |
| **plastic_eprop** (the mechanism) | −3.303 | −3.268 | −3.167 | **−2.158** | −3.281 |
| shuffle_elig (credit broken) | −3.301 | −3.266 | −3.170 | −2.762 | −3.279 |
| zero_signal (L:=0) | −3.301 | −3.266 | −3.169 | −2.760 | −3.279 |
| **BPTT_same_net** (matched full-backprop ceiling) | +0.457 | +0.804 | +0.848 | **+0.902** | +0.737 |

- **Pre-registered `frac_deep` = plastic_deep / BPTT_deep = −2.158 / +0.902 = −2.39 → BOUNDARY** (< 0.25 while the BPTT ceiling deep margin is clearly positive). Honest negative on long-range, exactly as pre-registered.
- **Anti-cheats all PASS:** `zero_signal` W_rec == `fixed_reservoir` W_rec **byte-identical** (True); `shuffle_elig` deep −2.762 ≈ `fixed` deep −2.760 (permuting the eligibility **removes** the gain → the credit STRUCTURE is load-bearing, not extra capacity).

## The substantive nuance — a SMALL-but-REAL, DEEP-SPECIFIC, credit-dependent capture (NOT zero)
- **e-prop's W_rec learning improves the DEEP bucket by +0.60** over the fixed echo-state floor (−2.760 → −2.158) while leaving SHALLOW unchanged (−3.279 → −3.281). The gain is **deep-specific** (long-range, not short-range) and **collapses under `shuffle_elig`** → it is genuine recurrent credit.
- As a **fraction of the LEARNABLE deep margin** (plastic's gain over fixed ÷ BPTT's gain over fixed): (−2.158 − (−2.760)) / (+0.902 − (−2.760)) = **0.602 / 3.662 = ~16%**. ⇒ one-step-local biological credit captures a **small but real (~16%), deep-specific** fraction of the long-range structure full backprop captures — precisely the Bellec-2020 (e-prop needed synthetic-gradient DNI for long-range PTB) / Murray-2019 (RFLO matches BPTT only at short horizons) prediction, pre-registered as an honest first-class deliverable.
- **Why nonzero despite a ~1/alpha≈3-token eligibility horizon:** the forward-filtered own-unit eligibility is short, but e-prop still tunes W_rec so the RECURRENT STATE (which itself carries context) becomes more predictive at depth — a partial deep credit that stops well short of BPTT's full off-diagonal (cross-unit-across-time) gradient (e-prop = a diagonal RTRL truncation).

## The metric confound to isolate NEXT (rung R1b) — the denominator over-credits BPTT
- **BPTT_same_net trains ALL params (incl. `W_in`, the input embedding); the e-prop arms use a FIXED random `W_in`** (the standard reservoir/e-prop convention). So the fixed floor is catastrophically below the bigram even at ctx1 (−3.30) because a random 300-d projection of a 2000-way one-hot is lossy — the read-out can't recover bigram-level from it — whereas BPTT LEARNS good embeddings (ctx1 +0.46). ⇒ a large part of BPTT's advantage, especially at shallow, is **input-embedding learning, not recurrent long-range credit** — which the current `frac_deep` conflates.
- **R1b (immediate next, single-variable):** add a **`BPTT_fixed_win`** arm (full backprop over W_rec/W_out/b but with `W_in` FROZEN to the same random projection the e-prop arms use). Then `frac_deep = plastic_deep / BPTT_fixed_win_deep` isolates the pure recurrent-credit rule (e-prop diagonal vs BPTT full off-diagonal) on identical embeddings. Expected: a HIGHER (cleaner) fraction than the −2.39 all-params number, likely still partial — the honest recurrent-credit-only long-range capture.

## What this launches (boundary = the next mechanism, per the standing discipline)
The boundary is diagnosed to the **off-diagonal RTRL truncation** (e-prop drops the cross-unit-across-time gradient the deep margin needs). The pre-registered cheap-first lever (ALIF adaptation-as-state horizon extension) only lengthens the OWN-UNIT eligibility horizon — it does NOT restore the off-diagonal credit, so it is expected to help only incrementally. The biologically-motivated mechanism that DOES target the dropped indirect signal is a **future-error / synthetic-gradient predictor population** (the biological analogue of Bellec's DNI — a second cortical population learning to predict the downstream error, supplying the "indirect influence" e-prop drops) and/or the **multi-layer spatial-credit (DFA/learned-feedback) path**. Rung ladder: **R1b** (isolate recurrent credit via BPTT_fixed_win) → **R2** (ALIF horizon lever, cheap) → **R3** (biological future-error predictor, the diagnosed-cause mechanism) → multi-layer. Only after a rate GO: the spiking `enable_bdsp` realization.

## Honest scope
Single seed (42), 2M tokens, single layer, n_pool=300 — a cheap-first de-risk, as designed; the point was to convert the whole build's central risk into a measured deep-fraction cheaply, which it did. The +16% deep-specific capture is real and anti-cheated; the BOUNDARY on the pre-registered metric is honest; the W_in confound means the pre-registered fraction is a LOWER bound on the recurrent-credit capture (R1b isolates it). All rate-level torch, GPU, grad-checked, anti-cheated, NO `sim/` edit.

## Files
`_emerge_stream_eprop_lm_derisk.py`; raw `research/findings/raw/_stream_eprop_lm.json` + `_stream_eprop_lm.log` + `_stream_eprop_lm_smoke.json`. Pre-registration + verdict: `2026-07-11-CEILING-...md` + `raw/_biocredit_recurrent_gate_workflow.json`.
