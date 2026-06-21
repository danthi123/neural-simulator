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

## ARM 3 — multi-seed verdict (FIX1+A vs FIX1 vs HOST vs SCRAM; grid-32, full 1800 steps)

Per seed, four arms at the sweet spot σ_2=2.0, g_2=0.1. Decisive checks: (1) surplus-shrink, (2)
re-orient/ceiling, (3) nav-not-regressed, (4) SCRAM-collapses, (5) moat-untouched.

| seed | metric | HOST | FIX1 | **FIX1+A** | SCRAM |
|---|---|---|---|---|---|
| 42 | post_change_finalQ_sum | 1.93 | 68.65 | **115.84** | 118.20 |
| 42 | gate_score (4 phases) | 2.53 | 94.10 | **116.90** | 121.15 |
| 42 | sel N−S % | — | **18.18%** | **28.35%** | 18.9% |
| 42 | sel N−S abs | — | **8535** | **820** | 241 |
| 42 | tracks_goal (dominants) | True (N,W,S,E) | **True (N,W,W,E)** | **FALSE (E,E,E,E)** | FALSE (E,E,E,E) |
| 42 | host/FIX1A post ratio | — | — | **0.017 (~60×)** | — |
| 42 | FIX1A/SCRAM post ratio | — | — | **0.98 (NOT collapsed)** | — |

**VERDICT seed 42 — HONEST NEGATIVE (decisively diagnosed). FIX-A is the WRONG lever, and is
COUNTER-PRODUCTIVE for re-orienting:**

1. **ABSOLUTE surplus-shrink WORKS** — sel N−S absolute 8535 → 820 (~10×), sel total ~206K → ~12.8K. The
   divnorm genuinely scales the sel pools down (as the probe showed).
2. **PERCENT surplus does NOT shrink — it slightly GROWS (18.2% → 28.4%).** A divisive divisor that is a
   COMMON SCALAR over the four sel pools (they share the `mean`) preserves the N/S RATIO by construction;
   the lower-drive f-I nonlinearity even mildly amplifies it. **A divisive common-scalar removes additive
   common-mode AMPLITUDE but cannot remove a RELATIVE (ratio) bias** — and the residual is a relative bias.
3. **NO re-orient — and FIX-A HURT tracking.** Per-phase (goals N,W,S,E): HOST → N,W,S,E (finalQ ~0.6 each,
   perfect). **FIX1 (no divnorm) → N,W,W,E (tracks 3/4; phase-1 W-goal finalQ 1.1 = goal reached).** FIX1+A →
   E,E,E,E (stuck-E, post_change 115.84 ≫ FIX1's 68.65). **FIX1 tracks the goal BETTER than FIX1+A** — the
   divisive flattening threw away the goal-direction SC signal along with the common mode.
4. **DECISIVE anti-cheat: SCRAM does NOT collapse** (118.20 vs FIX1+A 115.84, ratio 0.98 ≈ 1). The retinotopy
   scramble lesion is statistically identical to the intact read-out ⇒ the orienting is NOT carried by the
   retinotopic decode even with FIX-A (the SAME signature as the prior grid-32 NEGATIVE `c3fef1ff`). FIX-A
   flipped the bias axis (N→E) but did not remove the relative bias and did not make the read-out
   load-bearing.
5. **Moat untouched** by construction (no `--with-conv` ⇒ `cp_rf_w_*` never allocated; the nav cascade is
   `cp_connections`/`cp_membrane_potential_v`/`cp_firing_states`, array-disjoint).

**Seed 43 (3 primary arms HOST/FIX1/FIX1+A complete; SCRAM killed mid-run by the harness ~50 min, moot here
since FIX1+A doesn't track) — CONFIRMS the negative:**

| seed | metric | HOST | FIX1 | FIX1+A |
|---|---|---|---|---|
| 43 | sel N−S abs | — | 11601 | **151 (~77× smaller)** |
| 43 | sel N−S % | — | 23.68% | **10.26% (shrinks here)** |
| 43 | post_change_finalQ_sum | 1.59 | 78.27 | **111.42** |
| 43 | tracks_goal (dominants) | — | True (N,W,E,W) | **FALSE (N,N,N,N stuck)** |

**2-seed read (the negative is robust on the load-bearing metric):** the ABSOLUTE surplus-shrink is large in
both (8535→820 seed-42, 11601→151 seed-43, ~10–77×). The PERCENT effect is SEED-VARIABLE (grows 18.2→28.4%
at seed 42; shrinks 23.7→10.3% at seed 43) — consistent with a common-scalar divisor whose net effect on the
ratio rides on the lower-drive f-I nonlinearity, i.e. it does not reliably reduce the relative bias. But in
BOTH seeds **FIX-A fails to re-orient and is WORSE than FIX1** (FIX1 tracks 3–4/4 distinct dominants; FIX1+A
is stuck on a single cardinal — E at seed 42, N at seed 43; FIX1+A post_change ≫ FIX1 ≫ host). The divisive
flatten throws away the SC goal-direction signal. ⇒ a robust, mechanistically-clear NEGATIVE; the host
orienting heuristic does NOT retire on FIX-A.

(Note: the full 1800-step grid-32 runs are ~50 min/seed on the shared GPU and the harness terminated the
seed-43 SCRAM arm mid-run; the 2-seed agreement on the structural re-orient null + the SURPASS round below
is the deliverable. Per-seed JSON: `scpv_FIXA_arm3_seed{42,43}.json`.)

---

## SURPASS round (a boundary is not an exit) — ISOLATE → REFRAME → RANK → verdict

**Move 1 — ISOLATE + QUANTIFY the genuine residual.** FIX-A already did one real job: it removed the
ABSOLUTE common-mode amplification (sel total 206K → 12.8K; N−S abs 8535 → 820). That part is solved. The
genuine residual is TWO-fold: (a) the **relative (ratio) sel bias** (FIX1+A: E 3848 > N 3302 > W 3226 > S
2482, 28.4%) — a per-action tuning asymmetry, not an additive offset; and (b) the deeper fact, proven by
SCRAM≈FIX1+A, that the **goal-direction SC signal is too weak at the sel ring to overcome that relative
bias** (the SC pop-vector margin is present — FIX1 tracks 3/4 — but FIX-A's flattening REDUCED it).

**Move 2 — REFRAME via "how does real biology actually do this?"** Carandini-Heeger divisive normalization
removes a COMMON GAIN (contrast normalization); it is the wrong operator for a RELATIVE channel bias. Real
SC/BG selection wins not by common-mode removal but by **opponent (push-pull) competition** between
antagonistic direction channels, which integrate the DIFFERENCE of the two members of an axis (Bogacz et
al. 2006: a difference-integrator cancels a shared offset AND a relative bias structurally, where a
race-over-absolutes integrates both as signal). And critically — the data shows the position signal is
already there (FIX1 tracks 3/4); the failure is that a divisive flatten threw signal away. So the fix must
ENHANCE the differential, not shrink the absolutes.

**Move 3 — RANK cheap-first SURPASS mechanisms (the path PAST it):**
1. **FIX-B — opponent-pair the sel accumulators (RANK 1, pre-built).** Re-weight the sel ring into balanced
   axis-partner inhibition (N↔S, E↔W at `sel_opponent_weight`, cross-axis at `sel_crossaxis_weight`=0) so
   each axis integrates the DIFFERENCE → a common-mode N−S cancels structurally AND a relative ratio bias is
   suppressed (Bogacz 2006). This is the mechanistically-CORRECT operator for the residual that FIX-A
   provably cannot touch. Already wired (`--fixB`, `enable_sel_opponent_pair` in `g11_bg_runner`,
   `bridge.py:2247` re-weights the sel topology). Cheapest next run.
2. **Usher-McClelland LEAK on the sel accumulators (RANK 2).** Add accumulator leak so a constant bias does
   not integrate unboundedly; combined with the (present) SC signal the winner tracks. This was the
   load-bearing lever for the 2026-06-19 spiking-decision-default-on GO (the finite-size-noise + leak pair).
3. **Stronger SC drive into the cortex_X ring (RANK 3, low).** The prior calibration found "more SC drive is
   the wrong lever" (`5c42fa33`); deprioritized but available (`SC_CORTEX_W`).

**Move 4 — Verdict: SURPASSABLE, and the cheapest path is FIX-B.** FIX-A is a CHARACTERIZED NEGATIVE for
the #6 accumulator residual — *not* because the residual is irreducible, but because divisive normalization
is the wrong operator (it removes additive common-mode amplitude; the residual is a relative bias + a
signal-strength shortfall). The genuine residual is precisely pinned (the relative sel bias + the SC margin
swamped at the sel ring), and the mechanistically-matched cheap-first next move is **FIX-B opponent-pairing
of the sel accumulators** (integrate the difference, Bogacz 2006), with Usher-McClelland leak as the rank-2
follow-on. The host orienting heuristic does NOT retire on FIX-A; whether it can retire is now a question
for FIX-B.

_(Confirmation seeds for the FIX-A negative + the FIX-B run land below as they complete; each committed the
moment it lands.)_
