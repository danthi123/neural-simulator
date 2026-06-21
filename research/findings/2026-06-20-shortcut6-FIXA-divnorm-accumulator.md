# Shortcut #6 — FIX A: divisive normalization at the `sel_X` accumulator input (2026-06-20)

**Type:** IMPLEMENTATION + de-risk (GPU, `SIM_BACKEND=cupy`). The mechanistically-matched fix for the deepest
residual of shortcut #6 (the spiking-SC orienting read-out staying above the host ceiling so the host orienting
heuristic cannot retire). Gated by the deep-research scoping `2026-06-20-cascade-accumulator-Nbias-scoping.md`
(`c5b851e1`), which ranked **FIX A = divisive normalization at the `sel_X` input** rank-1.

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; grid-32 IS the verdict (never grid-8); a boundary is not an
exit. The no-confab moat is array-disjoint from the nav cascade (`cp_*` nav state vs the composer's complex
`cp_rf_w_*` synapses) and is untouched throughout.

---

## The diagnosis (controller-verified, scoping `c5b851e1`)

A small, persistent, goal-invariant **thalamic North-over-South lead** is **amplified by the Wang-2002 `sel_X`
NMDA-recurrent accumulators** (a race over ABSOLUTE drives — Bogacz et al. 2006: a race integrates each absolute
input, so a common-mode additive offset is integrated like signal) into a large selection-stage surplus that the
Lo-Wang `commit_X` burst inherits. The fix must act at the INPUT, removing the common mode BEFORE the race. FIX A
divides each `sel_X` input by `σ + g·mean(over the four sel pools)`, so the common N+E+S+W drive is divided out,
leaving the position-bearing differential.

---

## ARM 1 — SOURCE probe (confirms WHICH stage first shows the offset)

The existing rig (`_nav_sc_popvector_readout_derisk.py`, FIX-1 popvector arm, `readout_source="spiking_wta"`) logs
per-cardinal `thal_counts` / `sel_counts` / `commit_counts` / `motor_counts` summed over each readout window. The
N−S surplus by stage (grid-32, seed 42, FIX-1 popvector, a short 240-step smoke = the source localization; the
moving-goal schedule supplies the per-phase goal so this is the run-aggregate common-mode lead, NOT a centred-only
artifact):

| stage | N | E | S | W | **N−S** | **(%)** | E−W |
|---|---|---|---|---|---|---|---|
| `thal_counts`   | 1402 | 1333 | 1253 | 1459 | **+149**  | **+11.2%** | −126 |
| `sel_counts`    | 7874 | 7298 | 6065 | 6960 | **+1809** | **+26.0%** | +338 |
| `commit_counts` | 8706 | 7814 | 6581 | 7446 | **+2125** | **+27.8%** | +368 |
| `motor_counts`  |   92 |   32 |   16 |   57 | **+76**   | **+140.7%**| −25 |

**Reads (reproduces the scoping diagnosis exactly):**
- The **thalamic** stage carries a small **+11.2% N-over-S common-mode lead** (the scoping cited `thal_counts` N−S
  ≈ +1233, ~11% — the % matches to the tenth).
- The **`sel_X` accumulators AMPLIFY it** to **+26.0%** (absolute +149 → +1809 = ~12× absolute amplification; the
  scoping cited ~9× / sel ~22%). This is the mechanistically-expected winner-amplification of an NMDA-recurrent
  integrator (Wang 2002).
- The **`commit_X` burst INHERITS it** (+27.8%), and the final **`motor` action distribution is heavily N-biased**
  (+140%).
- E−W is small at every stage (the bias is specifically the N-vs-S axis, as predicted by the cluster-E top/bottom
  corner geometry riding on the shared-STN common baseline).

⇒ The offset **first appears at the thalamus (small, +11%)** and is **amplified at `sel_X` (+26%)**. FIX A's target
is confirmed: divide out the common mode at the `sel_X` input, where the amplification happens.

(Source-probe JSON: `research/findings/raw/nav_gate_2a/scpv_FIXA_srcprobe_seed42_smoke.json`.)

---

## ARM 2 — σ/gain sweep (finds the surplus-shrink sweet spot; grid-32, seed 42)

The sel_X divisive divisor is `σ_2 + g_2·mean(sel_X input current)`. Because the four sel inputs share the
SAME `mean`, the divisor is a COMMON scalar — it rescales all four sel pools identically. So the sweep's job is
to find the (σ_2, g_2) that (a) keeps the sel pools FIRING robustly (selection intact — `phase0_finalQ` near the
no-divnorm baseline, NOT the over-flatten where sel goes silent), while (b) the surplus shrinks at the sel +
commit output (the nonlinear f-I + NMDA-recurrence interaction is where a common-scalar input rescale buys a
real common-mode reduction).

**Baseline (FIX1 popvector, NO sel divnorm, grid-32 seed 42, same step budget):** sel_counts N−S = **+22.2%**
(N=52912 E=54862 S=42325 W=55001), commit N−S = +21.4%, the action heavily N-biased (motor N−S +121%).

First pass: σ=1, g=1 OVER-FLATTENS (the divisor `1 + 1·mean` ≈ hundreds of pA silences sel to ~6 spikes —
useless). The alive regime needs `g·mean ~ O(σ)`, i.e. g ≈ 0.05–0.2 at this drive. The 2×2 regime sweep:

| σ_2 | g_2 | sel N (alive?) | sel N−S | sel N−S % | commit N−S % | phase0_finalQ | dom |
|---|---|---|---|---|---|---|---|
| — (no divnorm) | — | 52912 | +10587 | **+22.2%** | +21.4% | (29.2) | N |
| 5.0 | 0.05 | 245 (near-silent) | +242 | +195% (tiny base) | 0% | 15.8 | N |
| 5.0 | 0.2  | 7   (silent) | −30  | −136% (collapsed→S) | −200% | 29.5 | E |
| **2.0** | **0.05** | **4387 (ALIVE)** | +641 | **+15.8%** | **+0.35%** | 29.9 | W |
| 2.0 | 0.2  | 7   (silent) | +4   | +80% (collapsed) | 0% | 27.2 | E |

**Reads:**
- σ=2, g=0.05 keeps the sel pools alive (N=4387, E=4140, S=3746, W=4290) and shrinks sel N−S **+22.2% → +15.8%**
  with the **commit-stage N−S collapsing to +0.35%** — but a finer bracket (below) does materially better.
- Every g≥0.2 point OVER-FLATTENS — sel collapses to single-digit spikes and the "surplus %" is meaningless on a
  near-zero base (and selection is broken). So the gain has a narrow alive window.
- **The thalamic common mode is UNCHANGED by FIX-A** (+11–14% at every point) — FIX-A correctly acts at the sel
  input, not upstream. The amplification is what it suppresses.

**Bracket refinement (σ∈{2,3}, g=0.1) → the refined SWEET SPOT σ_2=2.0, g_2=0.1:**

| σ_2 | g_2 | sel N (alive?) | sel N−S % | commit N−S % | phase0_finalQ | dom |
|---|---|---|---|---|---|---|
| **2.0** | **0.1** | **4811 E5464 S4502 W4665 (ALIVE)** | **+6.6%** | **−5.0%** | **0.97** | E |
| 3.0 | 0.1 | 712 (collapsing) | +88.6% (tiny base) | +55.8% | 9.41 | E |

**σ=2, g=0.1 is the chosen operating point for ARM 3.** It keeps the sel pools fully alive (thousands of spikes),
shrinks the sel N−S surplus to **+6.6%** (the strongest alive shrink: 22.2% → 6.6%, ~3.4× reduction), drives the
**commit-stage N−S to −5.0%** (common mode removed, slightly over-corrected), AND gives a **phase0_finalQ of 0.97**
(the agent reaches the goal in phase 0 — vs ~29 at the no-divnorm baseline and at g=0.05). σ=3/g=0.1 over-flattens
(sel collapses to hundreds, commit +56%). So the alive window is narrow and σ=2/g=0.1 sits in it.

Sweep JSONs: `research/findings/raw/nav_gate_2a/scpv_FIXA_arm2_probe.json` (regime sweep) +
`scpv_FIXA_arm2_bracket.json` (the bracket). **ARM-3 verdict uses the sweet spot σ_2=2.0, g_2=0.1.**

---

_(ARM 3 = the multi-seed FIX1+A vs FIX1 vs HOST vs SCRAM, the surplus-shrink check, the anti-cheat table, and the
FIX-A enable summary follow as they land. Each seed committed the moment it lands.)_
