# objrel CORRECTION — there is NO reservoir problem; the pure linear ridge reads objrel 1.00 on ALL 10 seeds. The ENTIRE residual unifies to the SPIKING read-out reproducing the linear-ridge discriminant. (Supersedes the "reservoir info-absence" framing + the reservoir-robustness gate.)

**Date:** 2026-07-06
**Supersedes:** the "(A) reservoir info-absence" half of `2026-07-06-objrel-basin-escape-restart-NEGATIVE-*.md` (CYCLE 931h) AND the reservoir-robustness research gate `2026-07-06-objrel-reservoir-robustness-research-gate-*.md` (CYCLE 931i) — both were built on a MISREAD.
**Trigger:** building the reservoir sweep (the diagnostic the mechanical-parallelism discipline mandates) caught that the premise was wrong — the "103/104 ridge = 0.00/0.17" I diagnosed as reservoir info-absence was NOT the pure linear ridge.

## The misread (the 4th self-caught error this session — the discipline working)
The value I read per seed as "the analytic RIDGE (is objrel LINEARLY present?)" was `analytic_dale_reference.objrel_slot0_THEME` — which is the **SPIKING Dale-legal read at the graded op-point** (`D._analytic_dale_readout` scored through `D._score`, IN_SCALE=0.5, sign-clipped, LIF spike-count argmax), NOT a pure linear ridge. I conflated the two and concluded "the reservoir feature lacks objrel on 103/104."

## The correction (fanned across 10 cores, ~17s — a pure linear ridge on the SAME on-bridge spiking reservoir feature, held-out)
| metric | seeds 42-102 (8) | seed 103 | seed 104 | all 10 |
|---|---|---|---|---|
| **PURE LINEAR RIDGE** objrel-slot0 (host `np.linalg.solve`, lam=0.1) | 1.00 | **1.00 (12/12)** | **1.00 (12/12)** | **1.00 on ALL 10** |
| SPIKING analytic-Dale read (graded op-point) | 1.00 | **0.00** | **0.33** | 8/10 |
| canonical (linear ridge) | 1.00 | 1.00 | 1.00 | 1.00 all |

**⇒ The objrel signal is LINEARLY PRESENT in the reservoir feature on ALL 10 seeds (ridge 1.00, and it does not even move with input scaling 320→80).** There is NO reservoir info-absence, NO reservoir seed-fragility, NO capacity issue. The reservoir is robust. The reservoir-robustness research gate (input-scaling / ρ→1 / orthogonal-W / ensemble) addressed a NON-PROBLEM.

## The UNIFIED residual — it is ALL the SPIKING read-out
The entire objrel residual is now ONE thing: **the SPIKING Dale-legal read-out cannot reproduce the linear-ridge discriminant on some seeds.** The linear ridge (a full-precision host linear read) reads objrel on all 10 seeds; the SPIKING read (Dale-shift to excitatory + graded op-point + LIF spike-count quantization) loses it on:
- **103/104** — even the ANALYTIC Dale read (the "ideal" spiking read, adversarially verified 1.00 on the original 6 seeds) fails (0.00/0.33). The graded op-point / Dale-shift loses the discriminant the ridge holds.
- **45/101** — the LEARNED read-out (delta rule) is fragile (45 always-THEME collapse; 101 non-monotone), also where the ridge succeeds.
This IS the core objrel arc's DALE-SHIFT diagnosis, now sharply bounded: the read-out must carry a discriminant that PROVABLY exists (ridge 1.00 on every seed) through Dale-legal spiking without losing it on thin-margin seeds. It is a READ-OUT precision/op-point problem, NOT a reservoir or representation problem.

## The NEXT mechanism (cheap, targeted — the read-out op-point, not the reservoir)
The ridge (1.00 on every seed) is the achievable CEILING. Make the spiking read reproduce it on all seeds:
1. **Read-out op-point sweep (cheapest):** READ_T (more spike-count resolution → finer approximation of the graded ridge value), IN_SCALE, threshold, leak — sweep so the SPIKING analytic-Dale read reads objrel ≥0.9 on all 10 seeds (fanned across cores; measure the spiking-analytic read, the `--read spiking` variant the sweep runner can add). The 103/104 failures are likely thin-margin seeds the current op-point quantizes away.
2. **A 2-stage / calibrated read** (the EMERGE-77-style per-pool bias calibration at a reference current — a Turrigiano homeostatic per-unit normalization) so the spike-count read tracks the graded margin on every seed.
3. **The learned-read-out fragility (45/101)** — separate, but likely also eased once the op-point reproduces the ridge (the learned read targets the same discriminant).
The read-out plasticity is SOLVED where the spiking read is faithful (5/10 clean); the residual is making the spiking read FAITHFUL to the provably-present linear discriminant on every seed.

## Files
- `research/runners/_rungB1c_objrel_reservoir_robustness_sweep_derisk.py` (the pure-linear-ridge-per-seed measurement that caught this), `research/findings/raw/_resv_ridge_s*.json` (10 per-seed: ridge objrel 1.00 all).
