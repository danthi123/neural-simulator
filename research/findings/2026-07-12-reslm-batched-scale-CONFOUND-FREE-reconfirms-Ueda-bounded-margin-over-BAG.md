# Batched reslm scale sweep, CONFOUND-FREE: re-confirms the Ueda-bounded reservoir co-scale negative — now vs the memoryless BAG, a stronger control (2026-07-12)

**One-line:** the batched-forward enabler let me run the reservoir-LM data/co-scale sweep confound-free (headline = margin over a **memoryless bag-of-prefix** read-out, not the add-1 bigram a bag could fake beating). It re-confirms the ALREADY-CLOSED result — the fixed reservoir is n-gram-bounded at our tractable scale — and the (a-1) gate flagged that this sweep was **re-deriving a conclusion the record already held**. Corrects the next-action pivot.

## The result (confound-free, multi-seed, all reservoirs `active`)
Headline metric `margin_over_bag = bag_ce − res_ce` (>0 ⇒ the recurrent dynamics beat a memoryless read-out over the same prefix; must GROW with data if the generator scales). Fixed eval-set + fixed vocab across the whole n_train sweep (a prior sliding-window/per-point-vocab would measure the curve on drifting data).

| reservoir | nt=1400 | nt=2800 | nt=5600 |
|---|---|---|---|
| **np=300** (baseline) | **+0.189** (n=5) | **+0.126** (n=4) | **+0.100** (n=3) |
| **np=600** (BIGGER, co-scale) | **−0.081** (n=3) | **−0.061** (n=3) | **−0.086** (n=2) |

- **np=300:** the dynamics beat the bag, but the margin **SHRINKS with data** (+0.189→+0.100) — the memoryless bag catches up as data grows. Not a growing (scaling) margin.
- **np=600:** a bigger reservoir is **WORSE — margin NEGATIVE at every scale** (the bag beats it). The extra read-out features overfit the tractable-scale data (more parameters, same small data ⇒ worse generalization). Doubling reservoir size does not help; it hurts.

## Why this is NOT a new mechanism boundary — it's the closed Ueda scale law (the a-1 catch)
Our own record already holds this (`AUTONOMOUS_STATE.md`, the 2026-07-11 ceiling arc + the prior co-scale probe):
- **Ueda et al. 2025 (same fixed-Wrec/Win setup):** a fixed reservoir needs **16–65k units / 100M words** to reach its *own* bounded ceiling (~60% BLiMP, which still LOSES to a 512-unit LSTM's 67.8%). We run **np=300–600 / ~15k–110k tokens = 50–200× below** that scale → the reservoir is n-gram-level → the bigram/bag catches up + overtakes, exactly as Ueda predicts.
- **The prior co-scale probe already went +0.242 → −0.076** (margin over the bigram). This sweep re-confirms it **confound-free** — vs the memoryless BAG (a stronger baseline than the bigram, the reservoir's own dynamics-load-bearing bar) — a methodological upgrade, not a new discovery.
- **The ceiling is already run + resolved (2026-07-11):** at real scale (TinyStories 23.7M, WikiText-103 60M) a transformer AND a full-backprop LSTM both reach the +1.5 growing-with-depth long-range; **recurrence holds long-range**. The path past the reservoir ceiling is deep credit for a multi-layer RECURRENT net — not more reservoir scaling.

**⇒ the a-1 lesson, in action:** this sweep RE-DERIVED a closed conclusion. The batched-forward enabler (67× — genuinely validated + useful) unblocked a run whose *answer was already in the record*. Caught by checking our own findings; logged so we stop re-running it.

## The corrected pivot (NOT PATH B — the record is explicit)
A first instinct was "reservoir bounded ⇒ pivot to PATH B (deep credit / learned recurrence on spikes)." **The record corrects this:** `2026-07-12-deep-credit-on-spikes-FA-family-exhausted-BurstCCN-mechanism-gate.md` UPDATE 2 + `ROADMAP.md` §12 item 1 state deep-credit-on-spikes is a **thoroughly-mapped, PARKED, NON-critical-path boundary** (the open-generation ladder "needs no deep learning rule").

**The mission-critical path is the open-generation LADDER** (Rungs 1–4 GO + the emergence-bar close). The next **tractable, buildable-now, no-deep-credit** rung is **Rung 5 — open-vocabulary spiking spell-out:** wire the reslm next-token output (already beats bigram at tractable scale, Rung 1) through the validated EMERGE-67/68 A→W spiking read-out, so the emergent generator SPELLS its predicted token on spikes. The reservoir's scale ceiling does not block Rung 5 — Rung 5 is an EXPRESSIVENESS rung (spell what it already predicts on spikes), not a scale rung.

## Status / next
- Reservoir-scale question: **CLOSED** (Ueda-bounded at tractable scale; ceiling resolved at real scale). Do NOT re-run tractable-scale reservoir sweeps.
- Batched-forward enabler: **VALIDATED + retained** (67×) for whoever runs the real-scale (16–65k-unit) reservoir arc, if ever pursued (bounded payoff).
- **NEXT CONCRETE ACTION:** a0-read the reslm ladder (`_emerge_reservoir_lm_*`) + the EMERGE-67/68 A→W spell-out, then BUILD Rung 5 (open-vocab spiking spell-out). NO `sim/` edit anticipated (reuse-by-import).

Reuse-by-import; no `sim/` edit. Multi-seed as tabled (np=300: 5/4/3 seeds; np=600: 3/3/2 — the np=300 blind-seed ladder was cut short when the sweep was killed to free cores for gaming, but the dev-seed decline + the np=600 negative are decisive and reproduce the closed finding).
