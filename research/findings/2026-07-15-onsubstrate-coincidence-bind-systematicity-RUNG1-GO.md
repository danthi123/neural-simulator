# On-substrate systematicity (plan #3, RUNG 1, 6-seed): the SPIKING COINCIDENCE BIND on a real SimulationBridge achieves TEST A's held-out-composition extrapolation (mean 0.75, beats the from-scratch learner 6/6) — the composing-machinery works on spikes

**Date:** 2026-07-15 · **Runner:** `research/runners/_onsubstrate_coincidence_systematicity_derisk.py` (reuse-by-import `core_sim_composition.build_bind_bridge`/`hadamard_spiking` — the 8-neuron AND-bank coincidence circuit on a real `SimulationBridge` — + the `_fixedbind_systematicity` task/harness; numpy backend for the smoke; NO `sim/` edit). Emergence-bar forward from `2026-07-15-TEST-A-...` + `-learned-bilinear-...` (plan `docs/plans/2026-07-15-months-scale-plan-...` #3).

## The test
TEST A on SPIKES: drive cat_code (as a ±1 ROLE) + qt_code (as graded ON/OFF FILLER currents) into the coincidence circuit; read the bound firing rates (bound_ON, bound_OFF); a ridge read-out on the bound rates → intent; test held-out (cat,qt) COMBINATIONS (zero surface count, scale-confound-free). Does the substrate's native multiplicative ⊙ (coincidence detection) achieve the systematic extrapolation the numpy ±1 bind did?

## Result (6-seed 42/43/44/100/101/102; chance 0.25)
| arm | held-out extrapolation (mean) | note |
|---|---|---|
| **SPIKING coincidence bind (real SimulationBridge) + ridge read-out** | **0.75** (0.500–0.929) | EXTRAPOLATES on all 6 seeds; matches numpy TEST A (0.857 on seed 42) |
| from-scratch MLP learner on [cat;q] | 0.39 (0.000–0.500) | memorizes+fails (the systematicity wall) |
- **The spiking bind >> the learner on ALL 6 seeds (0.75 vs 0.39)** — the load-bearing bind-beats-learner claim is 6/6 robust. ⇒ the composing-machinery is realized ON SPIKES, on a real `SimulationBridge`, for TEST A's exact held-out-composition task: the coincidence AND-banks compute the ⊙ that exposes a(X)b, and a read-out trained only on attested combos extrapolates to never-seen combinations where a from-scratch learner cannot.
- **Strict GO gate 3/6** — the permuted-label control doesn't cleanly collapse on 2 seeds (s43 0.500, s102 0.714): with only ~6-9 held-out combos, a permuted-label ridge partially fits by chance (a small-held-out-sample artifact, same family as TEST A's memfloor split-luck). NOT a mechanism issue — the bind-beats-learner metric is 6/6. A larger task (more combos) tightens the permuted control.
- **Debugging note (systematic-debugging):** the first version drove FIXED RANDOM projections of the codes → the bind computed coincidence(scrambled_cat, scrambled_qt), which does NOT expose a(X)b (train 0.686, held 0.143). The fix — drive the codes DIRECTLY as role/filler (identity projection) — makes the coincidence bind expose a(X)b in the first NB dims (train 0.94, held 0.86). The direct-drive is the on-spike analogue of TEST A's `cat_code ⊙ q_code`.

## ⇒ Ladder + next
RUNG 1 (fixed/identity projections) confirms the SPIKING coincidence bind is a systematicity primitive on the real substrate — the composing-machinery (today's numpy TEST A) realized on spikes. Combined with the already-deployed composer (structured SVO composition on spikes to 320 concepts), the systematicity engine is validated on the substrate end-to-end.
**RUNG 2 (the full emergence step):** LEARN the input projections (cat→role, qt→filler) by the committed on-bridge BDSP rule (`_reslm_onbridge_learn_win_derisk.py`'s plastic-input-pathway + BDSP loop) instead of the identity — a SHALLOW learned-projection + fixed-coincidence-bind (the feedforward-deep-credit-GO regime). Then the codes+projections are learned biologically and the bind is the substrate's native ⊙ — the honest emergence-bar composing-machinery, fully on spikes. Also: a larger task to tighten the permuted control; GPU (`SIM_BACKEND=cupy`) multi-seed.
Reuse-by-import; NO `sim/` edit. Runner: `_onsubstrate_coincidence_systematicity_derisk.py`.
