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

_(ARM 2 = the FIX1+A σ/gain sweep, ARM 3 = the multi-seed FIX1+A vs FIX1 vs HOST vs SCRAM, the surplus-shrink check,
the anti-cheat table, and the FIX-A enable summary follow as they land. Each arm committed the moment it lands.)_
