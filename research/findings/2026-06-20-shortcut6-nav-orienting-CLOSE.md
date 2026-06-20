# Shortcut #6 (nav SC orienting read-out) — the FAITHFUL grid-32 verdict (2026-06-20)

**Type:** GPU experiment, the decisive grid-32 verdict for shortcut #6 (the spiking-SC orienting read-out).
**Why this doc exists:** the scoping (`2026-06-20-shortcut6-nav-orienting-spiking-scoping.md`) found that the
"closed honest-negative" ledger row for #6 was NOT supported by a faithful-scale test of the GEOMETRY fix — the
population-vector build's grid-32 confirm was an empty `<!-- FILL -->` placeholder
(`2026-06-20-burndown-6-popvector-readout-build.md:69/75/90/99`), and its only data was a mis-calibrated grid-8 smoke
(`gain=1`, self-documented as over-attenuating). **Two subsequent subagents also failed to deliver the faithful
verdict — one ran the arms at grid-8 (the documented false-GO scale; the `scpv_*_seed42.json` files dated Jun 20
13:57–14:12 are all `grid_size=8`, `n_steps=480`, 2 phases only), and one's config was unrecoverable.** This doc runs
the genuine point-neuron mechanism (the pop-vector cosine read-out + correctly-calibrated bump-mass divisive
normalization + the already-deployed #4 WTA ring) at FAITHFUL grid-32 / 1800 / warmup-600 and settles the
convert-vs-residual question.

**Owner standard (load-bearing):** BRAIN-BASED-ONLY. The verdict is grid-32, NEVER grid-8. A boundary is not an exit —
if NEURAL does not re-orient OR the SCRAMBLE lesion does not collapse, the next mechanism (R1: the NEF-FS lateral-inhib
WTA; then Option-E inhibition-of-return) is attempted, never "closed boundary". The no-confab moat is array-disjoint
from the SC read-out (the spiking read-out is `cp_*` nav-cascade state; the conversational composer's complex
`cp_rf_w_*` synapses are untouched) and is NOT weakened by any of this.

---

## The mechanism under test (verified line-by-line against `g11_bg_runner.py`)

Shortcut #6 = the SC orienting read-out `sc_map → cortex_{N,E,S,W}`. The DEPLOYED host default is a signed half-plane
LINEAR RAMP (`:298-300`, `popvector=False`) — an un-normalized weighted SUM that mass-codes ("how much SC fires")
instead of position-coding ("where the bump is"), so it cannot track a moved goal (the
`2026-06-20-nav-sc-drive-reorient-derisk.md` stuck-N NEGATIVE). The fix is the SC's canonical population-VECTOR decode
(`:287-296`, `popvector=True`): each `sc_map` site's preferred-direction unit vector cosine-projected onto each
cardinal axis (`wv = max(0, û_a·u_site)`, bounded [0,1]), with the bump-mass divisive normalization supplied by
flagging the four `cortex_X` pools `input_divisive_norm=True` (`sim/bridge.py` Carandini-Heeger primitive), and the
inter-cardinal competition supplied by the already-deployed #4 `sel_X`/`commit_X` WTA ring (the `--spiking-sc` config
routes `readout_source="spiking_wta"`). PURE POINT-NEURON; NO `sim/` edit (a runner read-out-weight formula + an
existing-primitive flag).

---

## The 3 conditions (+ the original NEGATIVE as a 4th reference)

All on the SAME faithful base nav config: **grid-32, 1800 steps, warmup-600, the merged-het-off SC op-point**
(`SC_RET_SC=160, SC_REC=12, SC_RET_DRIVE=3500, SC_ROS_US=40` — the de-risk's merged-tuned values, identical to the
NEGATIVE), the goal-schedule `[phase0 NE, phase1 far-W, phase2 SW, phase3 SE]` (3 re-orients), `enable_visual_cortex`
on (perception NOT stripped), `enable_d1_d2_asymmetry / enable_striatal_fsis / enable_cluster_a_closed_loop /
enable_cluster_e_topography / enable_pfc_nmda` (the full cascade), `stdp_w_max_override=400`.

| Condition | What it is | How realized (the probe arm) |
|---|---|---|
| **HOST** | the host orienting scaffold (host Manhattan orienting + host reward, NO spiking SC). Re-orients for free (centroid+argmax = a position decode). The genuine host scaffold #6 must match. | `host` arm: `heuristic` on, no `enable_spiking_sc` |
| **NEURAL** | the position-coding pop-vector decode + CALIBRATED divnorm + the #4 WTA ring. | `sc_popvector` arm: `enable_spiking_sc=True, enable_spiking_sc_approach=True, spiking_reward_us=True, enable_neural_critic=True, spiking_snc=True, heuristic_strength=0, sc_popvector_readout=True, sc_popvector_divnorm_sigma=5, sc_popvector_divnorm_gain=0.02, SC_CORTEX_W=18` |
| **SCRAM** | NEURAL with the `sc_map→cortex` retinotopy SCRAMBLED (the lesion control). MUST collapse, else the decode is not load-bearing. | `sc_popvector_scr` arm: NEURAL + `SC_SCRAMBLE=1` (`install_spiking_sc_wiring(scramble=True)` permutes the sc-site target assignment, `:248-249`) |
| (ref) **RAMP** | the deployed half-plane ramp (= the original NEGATIVE), matched drive. | `sc_ramp` arm: NEURAL kwargs but `sc_popvector_readout=False` |

The SCRAM and RAMP arms hold the SAME `SC_CORTEX_W=18` as NEURAL (the matched-drive anti-cheat: any lift is
attributable to the read-out GEOMETRY, not a covert drive increase).

---

## Calibration (the step the build abandoned) — grid-8 micro-sweep, CALIBRATION ONLY

`research/runners/_nav_sc_popvector_calibrate.py` swept `(divnorm_sigma, divnorm_gain)` on the `sc_popvector` arm at
grid-8/480 (standalone, seed 42, `SC_CORTEX_W=18`). The scoping predicted the responsive band is `gain ≪ 1` (the
default `gain=1` over-attenuates the O(tens-pA) SC drive). Confirmed — **`any_tracks_goal=True`; the best cell is
`sigma=5, gain=0.02`** (dominant cardinal `[N→W]`, tracking the far-west phase-1 goal; phase0 finalQ 0.894,
post-change finalQ 0.75). `gain ∈ {0.0, 0.02, 0.05}` mostly tracks; `gain=0.1` at sigma 1/20 reverts to stuck-N.

**Grid-8 is the calibration screen ONLY — it completes just 2 of 4 goal phases and the cascade N-bias + OU dominate at
small scale; it is the documented false-GO cautionary tale. The verdict below is grid-32.** (`scpv_calibrate.json`.)

Chosen operating point for the grid-32 verdict: **`sigma=5, gain=0.02`**.

---

## THE FAITHFUL GRID-32 VERDICT (seed 42/43/44, n=1800, warmup-600)

Per-phase `final_quarter_mean_distance` (lower = better; the goal MOVES each phase so phases 1–3 are re-orients):

<!-- FILL: the grid-32 per-phase finalQ table for HOST / NEURAL / SCRAM / (RAMP), 3 seeds -->

**Re-orient discriminator:** NEURAL is GO iff (a) NEURAL RE-ORIENTS — sum ≈ HOST (within ~25%), post-change phases LOW
not stuck, the per-phase dominant cardinal TRACKS the goal (W-heavy for the far-west goal, E-heavy for the SE goal);
AND (b) SCRAM COLLAPSES at grid-32 (materially worse, post-change phases stuck, dominant cardinal goal-invariant). If
SCRAM ≈ NEURAL the decode is NOT load-bearing.

<!-- FILL: the verdict (CLOSED / next-mechanism-in-flight) -->

---

## Anti-cheat table

<!-- FILL: the anti-cheat results table -->

---

## EXACT commands (all 3 conditions + calibration)

```bash
# --- Calibration micro-sweep (grid-8, CALIBRATION ONLY — never a verdict) ---
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_calibrate \
    --seed 42 --n-steps 480 --grid-size 8 --warmup-steps 100 --sc-cortex-w 18 \
    --sigmas 1,5,20 --gains 0.0,0.02,0.05,0.1,0.2 \
    --out research/findings/raw/nav_gate_2a/scpv_calibrate.json

# --- The grid-32 verdict: HOST / NEURAL / SCRAM (one arm per invocation for incremental commit) ---
# HOST (host orienting scaffold)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms host --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --out research/findings/raw/nav_gate_2a/grid32_s6/scpv_summary_HOST_s42.json

# NEURAL (pop-vector decode + calibrated divnorm sigma=5 gain=0.02 + #4 WTA ring)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 \
    --out research/findings/raw/nav_gate_2a/grid32_s6/scpv_summary_NEURAL_s42.json

# SCRAM (NEURAL + retinotopy scramble lesion)
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector_scr --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 \
    --out research/findings/raw/nav_gate_2a/grid32_s6/scpv_summary_SCRAM_s42.json
```

(The probe `_nav_sc_popvector_readout_derisk.py` IS the faithful harness — it imports `run_moving_goal_episode` and
sets the merged-het-off SC op-point env defaults. Each arm writes a per-arm episode JSON
`grid32_s6/scpv_{arm}_seed{seed}.json` containing `phase_stats[i].{final_quarter_mean_distance, goal, action_counts}`,
committed as it lands.)

---

## Note on the prior false-GO + unrecoverable configs

The status ledger (`2026-06-20-shortcut-burndown-status.md:41`) marked #6 "CLOSED (characterized honest-negative)"
citing a grid-32 test — but the scoping found that row cites a DIFFERENT knob's (`SC_CORTEX_W` drive-strength) grid-32
sweep, NOT the geometry fix, whose grid-32 confirm was the empty FILL placeholder. The intervening subagent arm files
(`scpv_host/sc_ramp/sc_popvector/sc_popvector_scr_seed42.json`, Jun 20 13:57–14:12) are **all `grid_size=8`** — the
documented false-GO scale (2 phases only). This doc supersedes those with the genuine grid-32 geometry-fix verdict.

<!-- FILL: closing summary -->
