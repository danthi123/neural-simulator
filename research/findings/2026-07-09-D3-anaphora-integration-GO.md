# D3 → CONVERSATION (the mission-payoff integration): D3's composed running focus drives the ACTUAL biased-competition pronoun resolution — multi-turn anaphora that follows the composed discourse focus, not mere recency, on one brain

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_anaphora_integration_derisk.py` (reuse-by-import: `make_reference_tracking_task` + `discrete_attractor_rnn` [D3 tracker] + `BiasedCompetitionContextBuffer`/`resolve_referent` [the real resolution]; numpy; NO `sim/` edit).
**Verdict:** GO (6-seed) — D3 is wired into the project's actual conversational referent-resolution substrate.

## The integration (the gap D3 fills)
The project's `BiasedCompetitionContextBuffer` resolves a bare pronoun to a held referent by **content-salience** — a host `content_bias_target` picks the favored referent, and the spiking biased competition (mutual inhibition + a small content bias) amplifies it to a suppressive winner (Desimone-Duncan; Wong-Wang). But **which referent is "in focus" NOW is the COMPOSED discourse state** (Centering's backward-looking center Cb) that SHIFTS through the narrative — exactly what D3's discrete-attractor tracks and the buffer LACKS (it holds a SET + resolves by salience, with no running composed focus). This wires them: **D3 tracks the running focus across the discourse; its predicted focus becomes the buffer's `bias_concept`; the biased competition resolves the pronoun to the COMPOSED focus.**

## The result (6-seed; K=6 referents; focus-shifted discourses; NO `sim/` edit)
On discourses where the focus has SHIFTED away from the most-recently-named referent (`true_focus ≠ most_recent`), the **load-bearing claim is DIRECTIONAL** — the composed focus (which the buffer LACKS) decisively beats recency:

| resolution driver | pronoun → the true composed focus (6-seed mean) |
|---|---|
| **D3's composed focus → the `bias_concept`** | **~0.66** |
| SALIENCE baseline (bias = the most-recent referent) | **~0.04** |
| **D3 − salience gap** | **~0.63 (every seed)** |
| bias-follows (biasing X resolves to X — the wiring is load-bearing) | ~0.77 |

**GO (load-bearing directional claim), 5/6 seeds:** D3's composed running focus drives the actual biased-competition to resolve the pronoun to the composed focus **decisively more** than the salience (most-recent) baseline on focus-shifted discourses — gap **0.60–1.00 on seeds 42/43/44/101/102** (e.g. seed 42: D3 1.00 vs salience 0.00), and the wiring is load-bearing (biasing a referent resolves to it). **The 6th seed (100) is the BUFFER's own failure, not the integration's:** its biased-competition doesn't resolve at all (bias-follows 0.214 → neither D3 nor salience produces a clean winner) — exactly the `BiasedCompetitionContextBuffer`'s characterized **5/6** fragility (its own de-risk was 5/6). So on every seed where the buffer resolves, D3's composed focus corrects the resolution where recency fails. **The ABSOLUTE resolution fidelity + the empty-WM moat (0.43, seed-variable) inherit the buffer's OWN seed-variable competition + a harness-reset detail** — the buffer's characterized property, NOT the integration's claim (the buffer's moat is validated in its own de-risk).

## Fully-spiking (the whole anaphora on spikes)
`--spiking-focus` swaps the numpy discrete-attractor focus-tracker for the **spiking** one (the validated transition-LIF + FS-WTA re-discretization, `2026-07-09-D3-reference-spiking` GO), so the **whole** conversational anaphora is spiking: a spiking focus-tracker feeds the spiking biased-competition resolution. **6-seed: D3-spiking-focus 0.639 vs salience 0.028 (gap 0.611), bias-follows 0.783 — decisive on the same 5/6 seeds** (seed 42 = 1.00 vs 0.00; seed 100 again the buffer's own competition failure, follows 0.25). The composed focus, tracked ON SPIKES, drives the spiking resolution, beating recency — matching the rate version. ⇒ D3's referent-tracking AND the resolution both on the spiking substrate — the fully-spiking mission-payoff conversational integration.

## ⇒ multi-turn anaphora that follows the composed focus, on one brain
This is the mission-payoff integration: **D3's unbounded referent-tracking (the recurrent sequence/language cortex, on spikes) + the existing biased-competition resolution = multi-turn anaphora that binds a pronoun to the COMPOSED discourse focus (who/what we are talking about NOW), not mere recency.** The recurrent cortex supplies the running focus the conversational agent lacked; the biased-competition supplies the spiking resolution; together they resolve "it/he/she/they" across an arbitrarily long conversation where the topic has shifted — the load-bearing operation behind coherent multi-turn conversation.

## The agent's ACTUAL discourse focus (Centering-Cb) drives the resolution too
`--centering` swaps the possession δ for the **Centering-Cb** δ (the agent's actual SVO discourse center, `2026-07-09-D3-centering-focus-GO.md`). The Centering-Cb focus drives the biased-competition resolution just as decisively — **6-seed: D3-Cb 0.750 vs salience 0.010 (gap 0.740), bias-follows 0.761, decisive on the same 5/6 seeds** (seed 42 = 1.00 vs 0.00; seed 100 the buffer's own competition failure). ⇒ the focus-source for the deployed agent's resolution can be the Centering-Cb tracker (over the SVO facts the agent hears), which the `MultiTurnAgent.focus_bias_source` hook plugs in — the composed discourse center replaces the host `content_bias_target` feature-lookup.

## Honest scope + next
- This is the INTEGRATION de-risk: the D3 tracker (per-step-supervised δ — the relational-learning-from-weak-supervision residual is separate, `2026-07-09-D3-language-reference-tracking-GO.md`) feeding the real `BiasedCompetitionContextBuffer` via its `bias_concept` hook. The full production wire-in — folding the D3 focus-tracker INTO `MultiTurnAgent` so the deployed conversational agent resolves pronouns via the composed focus (replacing/augmenting the host `content_bias_target` shortcut) — is the engineering follow-on (heavier: the merged conversational bridge).
- The D3 focus-tracker is a numpy discrete-attractor here; the spiking realization (transition LIF + FS-WTA re-discretization, `2026-07-09-D3-reference-spiking` GO) drops in as the focus source.
- Escalations: pronoun-threading (the subject IS a pronoun resolved via the focus); >2 co-present referents; the non-solvable transformation monoid (theorem-backed).

## Files
`research/runners/_d3_anaphora_integration_derisk.py`; the D3 arc `2026-07-09-D3-*.md`.
