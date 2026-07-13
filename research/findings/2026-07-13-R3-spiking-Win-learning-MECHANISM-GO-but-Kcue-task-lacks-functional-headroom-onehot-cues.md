# R3 spiking-W_in-learning: the on-bridge BDSP MECHANISM is validated GO, but the K-cue distal-decode DE-RISK TASK does NOT test the functional "learn W_in beats fixed W_in" claim — a fixed random W_in already solves it (one-hot cues are maximally separable; the RATE reference confirms learn≈fixed). The functional test needs STRUCTURED input (the language regime), not arbitrary orthogonal symbols

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_onbridge_learn_win_derisk.py` (committed 69b56c09; the R3 spiking realization — learn W_in on a FROZEN spiking Izhikevich reservoir via the committed `enable_bdsp` rule on a plastic input→reservoir pathway; NO `sim/` edit). Dist sweep `raw/_r3_distsweep.log`.
**Status:** MECHANISM GO + a METHODOLOGY NEGATIVE on the task — the cheap K-cue de-risk cannot evaluate the R3 functional claim; the diagnosis names the valid task.

## The mechanism is validated (all controls GO, every operating point)
On the K-cue distal-decode (`[CUE_k] filler×d [QUERY]`, K=12, n_pool=200), the committed BDSP rule learns the input→reservoir W_in correctly:
- **W_in moves, W_rec frozen:** `dw_win` 0.014→0.135 (grows with dist/eligibility) while `dw_rec = 0.000000` at every dist (the frozen recurrence is untouched — verified by the COO pathway masks against `sim/bridge.py:7246-7273`).
- **Directed credit:** `learn_win` moves W_in ~2–5× the `apical_lesion` (apical=0) arm; `B_rises` True (apical raises the measured burst rate → directed credit); `wrong_sign` mirrors `learn_win`'s dw (anti-symmetric credit).
- **Anti-cheats collapse:** `input_lesion` → 0.083 (chance); `scramble` → 0.000; `no_weight_transport` True (Y own RandomState, never modified).
⇒ the R3 spiking realization — "the committed dendritic burst rule learns the reservoir's INPUT projection on spikes, no `sim/` edit, no weight transport" — WORKS as a mechanism.

## The functional claim FAILS on this task — and the RATE reference proves it is the TASK, not spiking
Dist sweep (seed 42, K=12, n_pool=200), decode acc (chance 0.083):
| dist | fixed_win (spiking) | learn_win (spiking) | RATE-ref fixed | RATE-ref learn |
|---|---|---|---|---|
| 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | — | — |
| 16 | 1.000 | 1.000 | — | — |
| 24 | 1.000 | 1.000 | 0.917 | 0.917 |
| 32 | 1.000 | 1.000 | 0.833 | **0.667** |
- **A fixed random W_in decodes all 12 cues perfectly to dist=32** — the reservoir HOLDS the cue that far AND a random projection separates it. There is NO collision → NO headroom for learning W_in.
- **The RATE reference (the R3 mechanism at its full-gradient best) shows learn ≈ fixed** (0.917=0.917 at dist=24) and learn WORSE than fixed at dist=32 (0.667<0.833) — i.e. **the task has no "learn W_in beats fixed W_in" property AT ALL**, independent of the spiking substrate.

## Root cause (systematic-debugging): one-hot cues have NOTHING to learn
The cues are arbitrary ORTHOGONAL one-hot symbols → they are already maximally separable, so a LEARNED input projection cannot beat a RANDOM one (both separate orthogonal inputs equally; learning only adds instability, hence learn<fixed at dist=32). The R3 reframe's "learn W_in is worth ~3× learning W_rec / beats full BPTT" was measured on the **LANGUAGE next-token task** (TinyStories V=2000), where the input tokens have **distributional/semantic STRUCTURE** that a random projection scrambles but a learned embedding exploits. The K-cue task strips exactly that structure out. ⇒ the scoping doc's assumption ("K large enough that a fixed-random W_in COLLIDES the cues") does not hold with orthogonal one-hot cues + an over-provisioned reservoir; a random projection separates ≤~n_pool orthogonal cues regardless of distance.

## ⇒ The redirect (the valid functional test)
Testing the R3 W_in-learning claim on spikes needs a task with **input-representation headroom** — where a fixed random W_in genuinely underperforms a learned one:
1. **STRUCTURED cue codes** (the cheap faithful fix): cues = OVERLAPPING/correlated codes (shared input features), so a random W_in scrambles the exploitable structure but a learned W_in maps them to a separable subspace. Validate the regime with the RATE reference FIRST (does rate-learn > rate-fixed?), then run the spiking arm only where the regime is proven valid.
2. **The LANGUAGE task on spikes** (the real R3 regime): learn W_in on the spiking reservoir, by-depth next-token CE, vs fixed W_in — expensive (the fork base's on-bridge W_rec LM run boundaried at 2/6), but the faithful regime.
The RATE reference is the load-bearing GATE for task validity: only run the expensive spiking arm at an operating point where the rate reference SHOWS the R3 property (rate-learn > rate-fixed). NO `sim/` edit.

## Files
`_reslm_onbridge_learn_win_derisk.py`; `raw/_r3_distsweep.log`. Corrects the task assumption in `2026-07-12-spiking-realization-scoping-learn-Win-on-fixed-reservoir-via-committed-BDSP-no-sim-edit.md`. Ties to `2026-07-11-R3-REFRAME-*` (the language regime where W_in-learning DOES win).
