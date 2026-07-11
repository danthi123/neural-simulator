# D1 on-bridge non-learning — a0 trace of one teach window reveals THREE interacting mechanism failures (transient LTD, Pbar self-extinction, silent non-target = no LTD)

**Date:** 2026-07-10
**Runner:** `research/runners/_d1_transient_trace.py` (numpy; NO `sim/` edit). Reads E/B/Pbar/dev at the target vs non-target output pool across one 60-step teach window (bursting regime soma_g=500, a class-1 example, top-error apical [-0.5, +0.9]).
**Method:** a0 — read my OWN substrate directly rather than trust the pending research summaries.

## What the trace shows (target pool = class 1, apical +0.9; non-target = class 0, apical -0.5)
```
step |  E_tgt  B_tgt  Pbar_t  dev_tgt |  E_oth  B_oth  dev_oth
  6  | 0.0667 0.0000 0.4588  -0.0306  | 0.0000 0.0000  0.0000
  9  | 0.0819 0.0167 0.5344  -0.0271  | 0.0000 0.0000  0.0000
 10  | 0.0737 0.0650 0.5575  +0.0239  | 0.0000 0.0000  0.0000
 24  | 0.0169 0.2556 0.7838  +0.2424  | 0.0000 0.0000  0.0000
 48  | 0.0013 0.1707 0.9368  +0.1694  | 0.0000 0.0000  0.0000
```

## THREE mechanism failures, in ascending importance
1. **Transient LTD (minor).** `dev_tgt` is NEGATIVE for steps 6–9 (B lags E — a burst needs a 2nd spike, established ~4 steps after the 1st) then flips POSITIVE at step 10. The `sim/` kernel applies `dw` EVERY step, so the early spurious LTD partially cancels the correct LTP.
2. **Pbar self-extinction within ONE sample (moderate).** `Pbar_t` races **0.31 → 0.95 in 50 steps**, so the teaching deviation `P − Pbar` collapses within a single teach window (dev peaks at step 24 then decays). Cause: on-bridge Pbar is EMA-updated EVERY sim step (50×/sample); the numpy reference updates Pbar ONCE PER BATCH (`_gnw_...:284`). So on-bridge the Pbar EMA runs ~50× faster and catches up to P within one sample — the sustained teaching signal extinguishes itself.
3. **The DOMINANT failure — the non-target output pool is SILENT (E_oth = B_oth = 0 for all 60 steps) → it receives ZERO credit → NO LTD on the wrong class.** The credit is `dev = B − Pbar·E`; with E_oth = 0, dev_oth ≡ 0, so the non-target's input weights never decrease. Only the TARGET pool gets LTP; over a balanced class set that is symmetric LTP on both pools with NO discriminative LTD → the two pools grow alike → no class separation → chance accuracy. The numpy reference's outputs are `sigmoid(basal)` — NEVER exactly 0 — so its negative-apical LTD always works. On real spikes a pool can be genuinely silent. Compounding it: the `bdsp_apical_couples_soma` edit makes the negative apical SUPPRESS the non-target's events (E), which also **violates the burst-multiplexing invariant** (E must be apical-INVARIANT, set by basal drive; only B/P should track apical). The coupling that was needed to raise the target's B also silences the non-target's E.

## The surpass this points to (to de-risk next; corroborate with the running research workflow + isolation probe)
The operating point must keep **E > 0 and apical-invariant on ALL output pools** (the multiplexing invariant), with apical modulating ONLY B — so the non-target (negative apical → B below Pbar·E) gets real LTD. Concretely, ranked:
1. **Keep all output pools tonically firing** (higher basal/output drive so E>0 everywhere) AND make the apical coupling gentle/one-signed enough NOT to silence a pool — ideally a proper two-compartment BAC mechanism where the apical plateau adds a 2nd spike (B) to an already-firing soma without suppressing a silent one (the honest dendritic `sim/` build).
2. **Slow Pbar** on-bridge (update once per sample, or scale ema_alpha by 1/steps-per-sample) so P−Pbar persists across the window.
3. **Settle past the transient**, then apply the credit from settled B/E (discard steps 0–9).

## Confirming follow-up (a0): tonic drive does NOT rescue the non-target — the coupling actively suppresses E
Re-ran the trace at `output_bias` 200 and 400 (`--output-bias`). The non-target pool E_oth stays **0.0000 for essentially the whole window** (one tiny flicker E_oth=0.011 at step 54, output_bias=400). So a stronger basal/tonic drive does NOT make the non-target fire — the **negative apical (via the soma coupling) actively suppresses its events**. This confirms failure #3's mechanism: the additive apical→soma coupling moves the TOTAL current → moves E, so it cannot modulate B without silencing E, and a negative-apical pool goes silent → no LTD. ⇒ the surpass is NOT a bias tweak; it requires a TRUE two-compartment burst where the apical plateau adds a 2nd spike (B) to an already-firing soma WITHOUT changing whether it fires (E) = BAC firing / the `enable_bdsp_microcircuit` dendritic path (the owner's flagged top emergence lever; built, its on-bridge learning-to-accuracy causality is the open test).

## Honest scope
(1), (2), and the coupling-suppresses-E mechanism of (3) are directly confirmed by the trace. (3) the silent-non-target = no-LTD is confirmed for this sample/pool and is the leading dominant cause; the isolation probe (`_d1_learn_isolation_probe.py`, single_layer + settled_credit) + the research workflow will confirm it is the general cause and rank the fix. This maps the on-bridge learning failure to a concrete, biology-grounded operating-point + credit-timing problem — the honest learning-substrate frontier, with a named dendritic surpass.

## Files
`_d1_transient_trace.py`; extends `2026-07-10-D1-onbridge-boundary-is-per-sample-credit-noise-not-bursting.md` (the "noise" framing is refined: it is these three timing/operating-point failures, not raw noise).
