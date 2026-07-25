# gap#4 forward-representability boundary SURPASSED (mechanism, numpy 6-seed GO): a fixed NONLINEAR EXPANSION linearizes the task the sparse-spiking forward destroyed — the never-tried lever the research gate found; on-bridge input-driven reliable spiking expander is the next build (2026-07-24)

## The arc that reached this
Same-day: the gap#4 on-bridge SPIKING blocker was characterized as a sparse-spiking FORWARD-representability degeneracy
(`2026-07-24-gap4-sparse-spiking-forward-representability-degeneracy-characterized-levers-a-h-6seed.md`): input generalizes
(held-out mlp 0.988) but every sparse-spiking hidden readout collapses to ≤0.34, ruled out by credit (a,b), readout (c),
drive (d), population (g), unpooling (h). A 5-agent research-gate workflow (`wf_1f9812d7-0eb`) then diagnosed the exact
gap and found the escape ALREADY VALIDATED in our own record but NEVER TRIED on gap#4.

## The research-gate diagnosis (decisive, verdict = surpassable-cheaply)
The load-bearing tell is exact: the sparse-spiking hidden code's held-out **LINEAR** decodability (0.284) EQUALS the raw
input's held-out linear decodability (0.284) to three decimals, while the input's held-out MLP is 0.988. **So the forward
is linearly-transparent but nonlinearly-DESTRUCTIVE — it adds ZERO linearly-usable structure.** The task needs a NONLINEAR
hidden transform (input mlp 0.988 vs lin 0.284); the sparse-spiking forward COMPRESSES (mean-pools) instead of EXPANDS.
Every lever a–h attacked compression/drive/credit/population/readout; **none attacked the missing NONLINEAR EXPANSION** —
the operation reservoir/kernel theory + the project's own Marr-Albus coincidence codon (EMERGE-35, held-out inheritance
1.00 6/6 on the SAME task family) supply. R3 reframe corroborates: FIX the forward as an expander + train only the
readout (deep-learning the forward is what FREEZES it at the sparse point).

## The cheapest-first de-risk — HYPOTHESIS CONFIRMED (numpy, 6 seeds, held-out)
`scratchpad/gap4_codon_expansion_numpy.py`: swap the gap#4 forward for a FIXED nonlinear expander, measure the exact
held-out linear/mlp representability metric (semantic-inheritance, k=5, n_ho=27, seeds 42/43/44/100/101/102). n_in=7.

| representation (held-out) | ho-LINEAR | ho-mlp | note |
|---|---|---|---|
| INPUT (ceiling) | 0.284 | 0.988 | task is nonlinear; the ceiling to reach |
| sparse-spiking H2 (the boundary) | ~0.284 | ≤0.34 | adds zero linear structure |
| **RANDFEAT-ReLU EXPANSION (200-dim)** | **0.772 ±0.069** | 0.944 | **6/6 seeds; task now LINEARLY separable** |
| non-expanding control (7-dim, same ReLU) | 0.377 | 0.623 | **expansion is load-bearing (+0.395)** |
| label-shuffle control | 0.210 | 0.191 | **≈ chance — the lift is real class structure** |
| EMERGE-35 coincidence codon (200-col) | 0.617 | 0.605 | underperforms: 7 continuous feats too few for a rich binary codon |

**⇒ GO.** A fixed random-feature nonlinear EXPANSION lifts held-out LINEAR decodability 0.284 → **0.772** (toward the
0.988 ceiling), with the non-expanding control at 0.377 (dimensionality-expansion is what lifts) and label-shuffle at
chance (real structure, not overfitting). The gap#4 forward-representability boundary is SURPASSED at the mechanism level.
**Verify-go note:** the research-gate's "permuted-features" anti-cheat was ILL-DESIGNED for a random expander (permuting
the 7 input columns is a no-op — it just relabels random directions; it read 0.809, NOT a collapse) — REPLACED with the
valid **label-shuffle** control (→ chance 0.21). Do not use permuted-features as an anti-cheat for a permutation-agnostic
random expansion.

## Honest scope + the ON-BRIDGE next build
- This is the numpy MECHANISM confirmation (the escape works), NOT the on-bridge spiking close. NO `sim/` edit.
- **The exact reason lever-g (on-bridge population sweep) failed while this GO'd:** lever-g's spiking columns were
  TONIC-PINNED (E~0.04, input-insensitive, mean-pooled to a near-constant 0.354), so they were NOT the input-driven
  random features the numpy ReLU is. Also lever-h's UNPOOLED spiking code overfit (train 1.0 / ho 0.247) because the raw
  spiking code is NOISY, while the numpy expander is DETERMINISTIC → generalizes.
- **⇒ the on-bridge realization = a fixed random EXPANSION FF projection to INPUT-DRIVEN, RELIABLE spiking columns**
  (random weights as synapses; operating point tuned so input drive dominates the threshold → columns fire
  differentially, NOT tonic-pinned; population-averaged per column → reliable, so held-out generalizes). This is the
  spiking version of random-ReLU / the EMERGE-35 input-driven-coincidence-column architecture generalized to continuous
  input. Read all N_COL columns (do NOT mean-pool to compress). Then the CPU-rate-GO learned microcircuit credit
  (56c90d67) / a linear readout closes it on real spikes.

## Verdict (per THE LAW)
- gap#4 forward-representability boundary: **SURPASSED at the mechanism level (numpy 6-seed GO)** — the escape is nonlinear
  EXPANSION, confirmed with clean anti-cheats. NOT a wall.
- **NEXT ACTION:** build the on-bridge input-driven reliable spiking random-feature/coincidence expander (reuse the
  EMERGE-35 coincidence-column machinery + operating-point discipline), re-run `_gap4_representability_probe` with the
  expander forward, GO iff held-out LINEAR on the spiking column code rises off 0.34 toward 0.988 on ≥5/6 seeds; then wire
  the learned-microcircuit credit / trained readout for accuracy.

## Provenance
`scratchpad/gap4_codon_expansion_numpy.py`; research gate `wf_1f9812d7-0eb` (5 agents, journal.jsonl). Reuses:
`_gap4_representability_probe.py`, `_emerge35_spiking_pooler_derisk.py`. Reconciled with the 2026-07-24 characterization +
the 2026-07-22 credit-signal finding (the credit is CPU-rate GO; this fixes the FORWARD it had no features to shape).
