# On-substrate systematicity — SCALE HARDENING (6-seed): the correction loop CLOSED — the small-task weakness was small-sample-noise-limited; at a larger task (12×12=144 combos, n_held=43) the on-substrate systematicity RECOVERS the full-control margin (RUNG-1 5/6, RUNG-2 6/6 vs the parent's 1-NN memfloor + linear-raw)

**Date:** 2026-07-15 · **Runner:** `research/runners/_onsubstrate_systematicity_scale_hardening.py` (monkeypatches `_fixedbind_systematicity_derisk.N_CAT/N_QTYPE` = 12; re-runs RUNG-1 + RUNG-2 `run_one` verbatim with their post-correction FULL controls; the bind dim D=12 is unchanged, only the combo count grows; numpy-CPU; NO `sim/` edit). Closes the loop the RUNG-1/2 CORRECTION opened.

## The question
The 2026-07-15 adversarial verify showed the RUNG-1/2 "6/6 beats the learner" was overstated (weak baseline + dropped controls); held to the parent's FULL controls at the small 7×7 task (n_held=14), RUNG-1 cleared +0.15 on 1/6 seeds, RUNG-2 on 3/6 — spiking noise eroded the margin on the leaky/near-degenerate held-out cells. The named follow-up: is the win REAL (small task was noise-limited) or is the on-spikes bind genuinely POINT-NEURON-NOISE-LIMITED (a boundary)? A larger task (more held-out, fewer degenerate cells) decides it.

## Result (12×12 = 144 combos, n_held = 43; chance 0.25; vs the FULL parent controls memfloor 1-NN + linear-raw, +0.15)
| rung | 7×7 (n_held 14) GO vs full controls | **12×12 (n_held 43) GO vs full controls** | mean bind/read | mean memfloor |
|---|---|---|---|---|
| **RUNG 1** (fixed spiking bind) | 1/6 | **5/6** (only s42 misses, by +0.14) | 0.829 | 0.539 |
| **RUNG 2** (transport-free read-out) | 3/6 | **6/6** | 0.795 | 0.539 |
- Per-seed RUNG-1 bind: 0.744/0.861/0.837/0.674/0.930/0.930 — all well above chance (0.25) and above memfloor; RUNG-2: 0.767/0.884/0.814/0.628/0.861/0.814.
- ⇒ **HARDENS.** The small-task 1/6 was SMALL-SAMPLE-NOISE-LIMITED (14 held-out cells, several near-degenerate where a 1-NN memorizer coincidentally matched). At 43 held-out cells the on-substrate systematicity clears the FULL parent controls robustly (RUNG-1 5/6, RUNG-2 6/6) — the underlying mechanism is REAL, not a point-neuron boundary.

## ⇒ The honest, complete story of the systematicity-on-substrate arc (correction + hardening together)
- The spiking coincidence bind genuinely computes the multiplicative product (|corr(bound, cat·qt)| 0.52–0.70) and, at a task scale where small-sample leakage doesn't dominate, **extrapolates to held-out compositions above the parent's stronger non-compositional controls (1-NN memfloor + linear-raw), 5/6 (RUNG-1) / 6/6 (RUNG-2)** — the from-scratch e-prop learner memorizes + fails.
- The transport-free read-out over the fixed bind is learnable by biological credit (≈ gradient) AND extrapolates at scale (6/6 vs full controls).
- So systematicity is realized on the real spiking substrate as a **fixed spiking binding primitive + a biologically-learned read-out over it**, validated against the full control set — the emergence-bar target, honestly earned (the adversarial-verify correction ensured the claim is against the FULL controls, not a weak baseline).
- The one honest open rung remains RUNG-3 (a from-scratch on-bridge BDSP RATE read hits the rate-coded-SNR-wall; the fully-spiking PLACED read is validated via the composer; named surpass = phasor/population-coordinate learning).

Lesson (reinforced): a de-risk ported to a new substrate MUST keep the parent's full control set AND be run at a scale where the held-out set isn't small-sample-degenerate; the adversarial verify + the scale hardening together are what make the claim trustworthy.
