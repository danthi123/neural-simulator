# Research gate — surpassing the spiking-WTA object-relative read boundary (SURPASSABLE, divisive-normalization)

**Date:** 2026-07-05
**Type:** read-only deep-research mechanism survey (catalog + Kandel 6e + literature + bio-research MCP). No code touched.
**Launches:** the objrel-surpass de-risk ladder (RANK-1 first-to-fire → RANK-2 recurrent divisive normalization).
**Supersedes the comfortable verdict of:** `2026-07-05-rungB1c-objrel-ff-inhibition-BOUNDARY.md` (+ the fixed-signed / learned-signed-from-scratch negatives). Per the SURPASS directive, the boundary is the START of the research, not the end.

## Bottom line — SURPASSABLE and cheaply; NOT the analog/whitening wall

The object-relative read fails through the SPIKING winner-take-all (WTA) but a linear argmax gets it ~100% — so the role
info is present + linearly separable. The genuine residual is a **specific computational mismatch, not a representational
wall**: a rate-driven WTA fires in proportion to **TOTAL drive**, but the discriminating role signal is a **per-draw-
variable additive common-mode-shifted DIFFERENTIAL** (a sub-1% margin riding a large, per-sentence-varying pedestal =
Dale-shift baseline + fixed floor). The pedestal, being the largest component of total, sets the WTA ignition order and the
sub-1% differential loses.

**This is NOT the Mikulasch-Priesemann limit** (that is decorrelating cross-neuron covariance for representation-whitening,
a matrix-inverse/dendritic op). Here it is: divide a per-draw scalar drive by a per-draw scalar pool-sum before selection —
a within-draw gain op implemented by recurrent summing inhibition (the same conductance motif already shipped for MSN
lateral inhibition, catalog B.04/E.05). Point-neuron-native.

## Why the three prior mechanisms failed (confirms the diagnosis)

- **Fixed subtractive FF-inhibition (BOUNDARY, see-saw):** a FIXED subtraction SHIFTS the operating point; it cannot cancel
  a PER-DRAW-VARIABLE pedestal (over-subtracts some draws → canonical flips to ~0.33; under-subtracts others). Carandini-
  Heeger: **"shunting inhibition ALONE is subtractive, not divisive."** Subtraction is categorically the wrong operation for
  a variable pedestal — you need DIVISION, with the pedestal in a denominator recomputed per draw.
- **Fixed signed-conductance ridge (NEGATIVE, traded 0.31/0.92):** same class — a single fixed linear map has no per-draw
  adaptation; one operating point can't satisfy two draws with different pedestals.
- **Learned-signed delta-rule from scratch (NEGATIVE, position basin 0.69/0.31):** the **Barak-Rigotti-Fusi mixed-
  selectivity generalization-discrimination trade-off** — with the pedestal dominating input variance, the lowest-loss
  linear-through-WTA solution is the HIGH-variance direction (position), not the low-variance role differential. Gradient
  descent through the WTA follows total-drive variance → the position basin. A PRE-PROCESSING failure (fix the input
  geometry = whiten before the learner), not a rule failure. (This predicts the in-flight INIT_RIDGE init-from-ridge may
  start in the objrel basin but be pulled BACK toward position unless the input is whitened.)

## The ranked cheap-first ladder (each: reuse-by-import, multi-seed-blind, 4 anti-cheats — canon-not-regressed / objrel-recovers / differential-load-bearing / scramble-collapses)

**RANK 1 (cheapest, try FIRST) — rank-order / first-to-fire WTA read.** Read the existing WTA as first-to-fire (latency)
instead of integrated-rate. Rank-order coding (Thorpe-Gautrais 1998; VanRullen-Thorpe) is intrinsically invariant to an
additive/multiplicative intensity offset ("less subject to changes in intensity of the stimulus"): a shared additive
pedestal advances all pools' latencies ~equally (latency is compressive near threshold) while the DIFFERENTIAL still sets
who-crosses-first. Likely NO new region, NO `sim/` edit — a read-mode change + near-threshold f-I gain tuning. **dt pre-
check first (10 min):** does the <1% margin map to a > dt latency difference? If yes it resolves; if dt-blocked, honest
sub-boundary → RANK 2.

**RANK 2 (highest-confidence build) — recurrent divisive-normalization pool → WTA (Louie-Glimcher two-stage).** Insert a
shared inhibitory "sum pool" G receiving from all selection pools, divisively/recurrently inhibiting each (`R_i ← V_i /
(B + Σ_j R_j)`), then the existing WTA selects on R_i. Per-draw adaptive by construction (denominator recomputed each
draw); order-preserving so canonical is protected (the two-stage decision: normalize THEN choose — Louie et al. 2014 J.
Neurosci. 34:16046). Reuse the shipped MSN lateral-inhibition motif. One free param `k` (G→R_i strength): `k=0` MUST
reproduce the boundary exactly (byte-identity check), sweep up. AC3 lesion = silence G → objrel collapses, canon
unchanged. Add the Rutishauser-Douglas-Slotine contraction-stability check so the recurrent pool doesn't oscillate.

**RANK 3 (learning-basin fix, compose if learning the read-out end-to-end) — adaptive-whitening interneurons before the
learner** (Duong-Lipshutz-Chklovskii 2023, gain-modulating interneurons; spiking-compatible, per-input adaptive) so the
delta-rule sees the differential not the pedestal-variance → escapes the position basin. + margin/relative loss + init-
from-ridge basin-seeding.

**RANK 4 (defer) — theta-gamma phase-coded WTA** (N.15 Lisman-Idiart; phase is intensity-invariant) — heaviest (needs the
oscillator; possibly reuse EMERGE-85's theta-gamma WM buffer). Below 1-3.

RANK 1 + RANK 2 compose (normalize the drive AND read first-to-fire). Try RANK 1 → RANK 2 → both, in that cost order.

## Verdict

SURPASSABLE, cheaply. The residual is one missing **per-draw relativization stage** between the reservoir state and the
WTA (representation already solved by the linear-argmax control; ridge weights already known). The one genuine empirical
risk is the sub-1% margin vs the WTA ignition/latency resolution — a TUNING problem (sharpen f-I gain / soft-then-hard
ramp), not a mechanism wall, since the differential is present + linearly separable. The brain does object-relative
comprehension via LIP-style divisive-normalization decision circuits → the mechanism exists and is point-neuron-native.

## Key sources
Carandini-Heeger 2012 (Nat Rev Neurosci, PMC3273486 — divisive normalization, shunting-alone-is-subtractive); Louie-
LoFaro-Webb-Glimcher 2014 (J Neurosci 34:16046, PMC4244470 — `R_i∝V_i/(B+ΣR_j)`, normalize-then-WTA); Rutishauser-Douglas-
Slotine 2011 (Neural Comp — soft/hard WTA gain, contraction stability); PNAS 2020 10.1073/pnas.2005417117 + PMC12139785
2024 (spiking divisive-normalization); Thorpe-Gautrais 1998 + arXiv 2212.00081 (rank-order intensity-invariance); Duong et
al. 2023 arXiv 2301.11955 (adaptive whitening interneurons); Barak-Rigotti-Fusi 2013 (J Neurosci 33:3844 — the position-
basin trade-off); Machens et al. 2022 (eLife 82426 — dynamic switch normalize↔WTA). Catalog B.04/B.52/E.05 (lateral-
inhibition WTA motif, shipped), G.16/G.17 (LIP accumulator, Kandel 6e Ch 56).
