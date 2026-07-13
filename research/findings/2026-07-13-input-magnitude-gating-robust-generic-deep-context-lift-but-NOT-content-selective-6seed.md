# Input-dependent SHUNTING (divisive gain) on the reservoir's recurrent loop gives a ROBUST +0.09 deep-context lift over the plain ESN (6-seed) — but it is GENERIC input-magnitude gating, NOT Mamba's CONTENT-SELECTIVE mechanism (the shuffled-input anti-cheat matches, 0/6); the content-selective delta-memory premise needs structured codes, not one-hot

**Date:** 2026-07-13
**Runner:** `research/runners/_ssm_context_depth_derisk.py` (`--beta`, arms `shunt`/`shunt_shuffled`; numpy-CPU; the by-depth reservoir-minus-bigram margin + bag control; NO `sim/` edit). The cheap-first probe the deep-research gate (`genuine-frontier-gate` Workflow) recommended for the input-dependent-gating axis (Mamba-Delta / ORGaNICs).
**Status:** POSITIVE-but-GENERIC (a real 6-seed reservoir improvement) + an honest deflation of the content-selective (delta-memory) premise. Done in the NON-scale-confounded regime (concat=5 long-context, where the plain ESN robustly beats the bigram +0.199 — so a null/positive here is interpretable, unlike the toy-scale next-token CE).

## The mechanism (input-dependent divisive shunt = Mamba's Delta / Carandini-Heeger)
On the mixing ESN, per token: `h_t = tanh( g_t · (W·h_{t-1}) + W_in·x_t )` with an INPUT-DEPENDENT per-unit gate `g_t = leak/(1 + β·|W_in·x_t|)` (β=0 → plain ESN). A strong input drive gates DOWN the recurrent memory (write new / forget old); a weak drive (filler) keeps `g_t ≈ leak` (RETAIN the held content across fillers) — the state holds the last informative token across intervening fillers. ANTI-CHEAT `shunt_shuffled`: the gate is driven by a PERMUTED token's drive (content↔gate correspondence broken) — if the lift survives, the selectivity is NOT content-driven.

## Result — 6-seed (42/43/44 dev + 100/101/102 blind), concat=5 long-context, deep = depth 16+, margin over the bigram
| arm | mean deep margin | beats plain ESN? | is it content-driven? |
|---|---|---|---|
| **shunt** (input-magnitude gated mixing ESN) | **+0.289** | **6/6** (+0.09 over ESN) | — |
| shunt_shuffled (gate from a PERMUTED token) | +0.287 | 6/6 | **shunt beats it 0/6** (Δ~0.003) |
| plain random ESN | +0.199 | — (reference) | — |
| multitimescale (input-INDEPENDENT fixed timescales) | −0.055 | no | — |
| bag-of-prefix | −0.61 | no | — |
- **POSITIVE (6/6):** input-magnitude divisive gating robustly lifts the deep-context margin +0.09 over the plain ESN AND beats the fixed multi-timescale control decisively — the **best fixed-reservoir variant for deep-context language** tested this session. Biology-grounded (shunting inhibition / divisive normalization, Carandini-Heeger; input-driven gain control), emergence-clean (fixed local circuit, zero credit).
- **NOT content-selective (0/6):** `shunt ≈ shunt_shuffled` on every seed — the lift survives shuffling which token gates. So the benefit is a GENERIC input-magnitude gain fluctuation (an adaptive RANGE of effective recurrent timescales), NOT the content-addressed selectivity Mamba/DeltaNet use. Root cause: the codes are one-hot × random-W_in, so every token's `|drive|` is drawn from a similar distribution → a permuted-token gate has the same statistical structure → the same lift. There is no content-vs-filler informativeness structure for a SELECTIVE gate to exploit (the same limitation as the R3 arc: one-hot codes have nothing for representation/selectivity to learn).

## ⇒ Verdict for the deep-research-gate's top-pick (state-as-weights / delta-memory)
The gate's premise — "if input-dependent gating lifts deep-context above the plain ESN + the input-independent control, the axis is LIVE → invest in the delta memory" — is HALF-confirmed: the axis gives a real generic lift, but the CONTENT-SELECTIVE part (the higher-ceiling mechanism the delta-memory actually needs — a content-addressed key/value/query memory) is NOT demonstrated here (the shuffled control matches). So the delta-memory / content-selective long-range investment is gated on the SAME regime the SSM + R3 arcs mapped: **structured codes with informativeness structure** (content vs filler distinguishable) AND/OR the **validated GPU scale** (TinyStories 23.7M/V=2000, where the deep-context signal is demonstrably rich) — NOT the toy one-hot regime. The generic shunt lift is a legitimate reservoir improvement to keep; the content-selective ceiling is the scale/structured-code frontier.

## Files
`_ssm_context_depth_derisk.py` (`--beta`, `shunt`/`shunt_shuffled` arms); `raw/_ssm_shunt_6seed.json`. Follows the `genuine-frontier-gate` deep-research gate. Ties to `2026-07-13-SSM-multitimescale-RECURRENCE-does-NOT-help-language-*` (the fixed-recurrence family) + `2026-07-13-R3-spiking-Win-learning-*` (one-hot codes lack learnable/selectable structure). NO `sim/` edit.
