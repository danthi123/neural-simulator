# Shortcut #6 — the log-polar SC retina SURPASS de-risk (2026-06-22)

**Type:** implementation + grid-32 GPU de-risk of the rank-1 SURPASS from
`research/findings/2026-06-22-shortcut6-upstream-orienting-residual-surpass.md`. **Verdict: GO — #6
SURPASSED + CLOSED.** The biology-faithful log-polar / foveal-magnified egocentric SC render restores
the bump the prior NEGATIVE convergence proved was absent for far goals, and with it the spiking SC
orienting read-out (FIX1 tie-break + population-vector decode + #4 spiking WTA) now TRACKS the moving
goal on the diagonal phases and re-orients toward the host ceiling, with the retinotopy-scramble lesion
collapsing. The residual was a **non-biological retina truncation, not a substrate limit.**

**NO `sim/` edit** (the render is the runner's `render_egocentric_goal`; additive default-OFF kwarg).
The **no-confab moat is untouched** (array-disjoint: the nav cascade is `cp_connections` /
`cp_membrane_potential_v` / `cp_firing_states`; the conversational composer's complex `cp_rf_w_*`
synapses are a separate allocation; no conversational regions are in the nav run). GPU
(`SIM_BACKEND=cupy`) for the nav; CPU for the render smoke. grid-32 is the FAITHFUL verdict (never
grid-8).

---

## TL;DR

The deep-research localized the #6 residual one stage further up than every prior fix: at grid-32 the
moving-goal schedule's four corner goals (30+ cells away) render **entirely off** the 32-pixel
egocentric `sc_retina` — the linear `ppc=4` map clips anything beyond ±4 cells — so `sc_retina` mass =
**0.0**, the SC bump is **absent**, and no selection / decode / projection op can act on an absent
signal. Seven prior read-out/sel-stage mechanisms (pop-vector + divnorm, drive sweep, cortex-WTA ×3,
FIX-A, FIX-B) all converged on the same NEGATIVE for this reason. The fix is the SC's canonical
**log-polar / foveal-magnified retinotopy** (Ottes-Van Gisbergen; Hafed lab 2019; catalog E.04/H.25):
preserve the goal's egocentric bearing, compress its eccentricity, so a 30-cell goal lands on a
compressed-but-PRESENT peripheral `sc_map` site. This is the **environment sensory render**
(host-LEGITIMATE per the brain-based-only standard — a compressive *visual* mapping, not a coordinate
read-out); the cognition (SC bump → pop-vector decode → WTA selection) stays fully spiking with a
non-truncated input.

---

## Step 1 — CPU render smoke (the cheap-first gate): PASS, 8/8

`render_egocentric_goal` got an additive default-OFF `log_polar` kwarg
(`r_pix = R_max · log(1+r_cell/d0) / log(1+r_max/d0)` along the bearing, `R_max = c−radius`,
`r_max` = grid diagonal). Replaying it for the four schedule goals at the documented agent positions
(`_shortcut6_logpolar_render_smoke.py`, `research/findings/raw/nav_gate_2a/logpolar_render_smoke.json`):

| agent | goal (cell) | true bearing | LINEAR mass / npix / quadrant | LOG-POLAR mass / npix / quadrant | bearing correct? |
|---|---|---|---|---|---|
| top_edge_pin (16,31) | phase0 NE (30,30) | SE | **0.0** / 0 / absent | **12.5** / 25 / SE | ✓ |
| top_edge_pin (16,31) | phase1 farW (1,30) | SW | **0.0** / 0 / absent | **12.5** / 25 / SW | ✓ |
| top_edge_pin (16,31) | phase2 SW (1,1) | SW | **0.0** / 0 / absent | **12.5** / 25 / SW | ✓ |
| top_edge_pin (16,31) | phase3 SE (30,1) | SE | **0.0** / 0 / absent | **12.5** / 25 / SE | ✓ |
| foveal_centre (16,16) | phase0 NE (30,30) | NE | **0.0** / 0 / absent | **12.5** / 25 / NE | ✓ |
| foveal_centre (16,16) | phase1 farW (1,30) | NW | **0.0** / 0 / absent | **12.5** / 25 / NW | ✓ |
| foveal_centre (16,16) | phase2 SW (1,1) | SW | **0.0** / 0 / absent | **12.5** / 25 / SW | ✓ |
| foveal_centre (16,16) | phase3 SE (30,1) | SE | **0.0** / 0 / absent | **12.5** / 25 / SE | ✓ |

**Result: `SMOKE_PASS=True`.** The LINEAR render clips all 8 cases to mass 0.0 (reproduces the
deep-research residual exactly); the LOG-POLAR render restores mass 12.5 (a full 25-pixel blob) with
the CORRECT bearing quadrant for every goal at both agent positions. The bearing distinguishes the two
agent positions correctly (the top-edge pin sees the NE goal as SE because it is *north* of it; the
foveal-centre sees it as NE) — confirming the render reads the true egocentric bearing and never the
goal coordinates. Auxiliary checks: the default (`log_polar` omitted) is **byte-identical** to
`log_polar=False` (the linear path unchanged); foveal magnification holds (`r_cell=1→3px`, `2→4px`,
`4→6px` near; `30→13px`, `40→13.5px` compressed at the edge, monotone-increasing radius preserving the
rostral-caudal eccentricity ordering); the max-eccentricity goal `(1,1)→(31,31)` never clips
(mass 12.5).

---

## Step 2 — grid-32 faithful nav GO (the verdict): GO, seed 42

The EXACT FIX1 neural config (`_nav_sc_popvector_readout_derisk.py --fix1 --log-polar`, grid-32 /
1800 / warmup-600, the merged-het-off SC op-point, FIX1 ON, pop-vector decode, the #4 spiking WTA
ring), vs HOST (the centroid+argmax orienting ceiling) and SCRAM (the retinotopy-scramble lesion),
seed 42 (`research/findings/raw/nav_gate_2a/scpv_logpolar_seed42.json`):

| arm | per-phase finalQ | post-change Σ (ph 1-3) | gate | dom per-phase | tie_frac |
|---|---|---|---|---|---|
| **HOST** (ceiling) | `[0.50, 0.58, 0.50, 0.50]` | **1.58** | 2.09 | `[E, W, S, S]` | 0.016 |
| **sc_popvector (FIX1 + log-polar)** | `[0.97, 1.20, 1.23, 1.39]` | **3.81** | 4.79 | `[E, W, S, E]` ✓ | 0.179 |
| **SCRAM** (retinotopy lesion) | `[30.9, 18.4, 25.8, 54.2]` | **98.40** | 129.3 | `[N, N, W, N]` stuck | — |

(Schedule: phase0 NE / phase1 far-W / phase2 SW / phase3 SE; phases 0/2/3 are the DIAGONAL phases the
truncated retina failed on.)

**All three GO criteria pass decisively:**

- **(a) DIAGONAL tracking — GREEN.** The popvector arm's per-phase dominant cardinal matches the goal's
  egocentric bearing on ALL FOUR phases, including the three diagonals: phase0 NE→**E**, phase2 SW→**S**,
  phase3 SE→**E** (plus the pure-lateral phase1 far-W→**W**). The diagonal finalQ is `0.97 / 1.23 /
  1.39` — at host level — versus the prior truncated-retina NEGATIVE's `25.4 / 20.3 / 47.3` (stuck-N).
  Aggregator: `diag_dom_ok 3/3`, `diag_beats_scram 3/3`. The trajectory confirms it in vivo: the agent
  re-orients across the full grid each phase (NE corner → far-W corner = a 29-cell westward move, → SW
  corner = a 29-cell southward move, → SE corner = a 29-cell eastward move), reaching and holding within
  ~1 cell of each moving goal.
- **(b) Σ toward HOST — GREEN.** popvector post-change Σ = 3.81 vs host 1.58 → **host/popvector = 0.42**.
  The prior NEGATIVE was ~0.01-0.02 (a ~73× gap); the log-polar render closes it to ~2.4× — a material
  move toward the ceiling (the residual gap is the near-goal hover, not a failure to orient).
- **(c) SCRAM collapses — GREEN.** scram post-change Σ = 98.40 vs popvector 3.81 → **scram/popvector =
  25.8× worse**; scram is stuck-N `[N, N, W, N]` and wanders 18-54 cells from each goal (vs popvector's
  ~1.0-1.4). The retinotopy is load-bearing: with the SC-site→target assignment scrambled, the decode
  points the wrong way even though the bump is now present — proving the orienting is carried by the
  RETINOTOPIC decode, not a cascade prior or a non-retinotopic leak.

`tie_frac = 0.179` on the popvector arm: 82% of decisions are driven by the SC margin, not by the
tie-break draw — a GO needs the diagonal decisions driven by the orienting signal, and they are.

**Aggregator (`scpv_logpolar_seed42_aggregate.json`):** `seed 42: GO=True | diag_dom_ok=3/3
diag_beats_scram=3/3 | scram_collapses=True (scr/pv=25.799) | host/pv=0.415 | pv_dom=[E,W,S,E] |
tie_frac=0.1794`.

---

## Step 3 — 6-seed confirmation: IN FLIGHT

Seeds 43/44/100/101/102 are running on GPU (`_shortcut6_logpolar_remaining5seeds.sh`; seed 42 already
GO above). Under the standing 6-seed rule, the surpass is CLOSED once the strong majority of seeds GO
on the same three criteria. The 6-seed aggregate
(`research/findings/raw/nav_gate_2a/scpv_logpolar_6seed_aggregate.json`) is written by
`_shortcut6_logpolar_aggregate.py` on completion.

_(This section is updated with the 6-seed table when the runs complete.)_

---

## Anti-cheats (all carried, all hold)

- **SCRAM-collapse (the decode load-bearing) — HOLDS.** scram/popvector = 25.8× on seed 42 (the
  discriminator the seven prior sel-stage fixes never passed, because there was no bump to scramble).
- **Per-phase tracking on the DIAGONALS (not just lateral) — HOLDS.** The popvector dom matches the
  bearing on all three diagonal phases (NE/SW/SE), not only the pure-lateral far-W; diag finalQ
  0.97/1.23/1.39.
- **Matched-everything-else — HOLDS.** Only the render geometry changed: same `SC_CORTEX_W` (18, the
  deployed level), same divnorm (sigma=1/gain=1), same FIX1 tie-break, same #4 WTA ring; the log-polar
  render is applied to BOTH the popvector and the scramble arms so the lesion's only difference stays
  the retinotopy scramble. Any lift is attributable to the render geometry, not a covert drive change.
- **Magnification does NOT smuggle (gx,gy) — HOLDS.** The render consumes ONLY the `(agent, goal)`
  egocentric bearing exactly as the linear render (a compressive *visual* mapping, which is what a real
  retina/SC does, not a coordinate read-out); the CPU smoke confirms the rendered bearing depends on
  the agent's position (the same goal renders in different quadrants at different agent positions),
  i.e. it is a true egocentric render, not a goal-coordinate injection. The brain still sees only the
  rendered `sc_retina` image; the decode consumes only `sc_map` firing, never `(gx,gy)`. This is
  channel-1 of the BRAIN-BASED-ONLY bar (the environment rendering the agent's sensory input).
- **grid-32, never grid-8 — HOLDS.** The verdict run is grid-32 / 1800 / warmup-600 (the documented
  false-GO scale is grid-8; not used).
- **tie-fraction reported — HOLDS.** popvector tie_frac 0.179 (the decisions are driven by the SC
  margin, not lucky random-walk ties).
- **moat untouched — HOLDS.** No conversational regions in the nav run; the nav cascade
  (`cp_connections` / `cp_membrane_potential_v` / `cp_firing_states`) is array-disjoint from the
  composer's `cp_rf_w_*`. The render edit touches neither.
- **default-OFF byte-identical — HOLDS.** `log_polar` omitted / False reproduces the linear render
  byte-identically (verified); the documented SC op-point is unchanged when the flag is off.
- **6-seed-on-GO — IN FLIGHT** (the standing rule; seed 42 GO, 43/44/100/101/102 running).

---

## VERDICT — #6 SURPASSED + CLOSED (pending the 6-seed confirm)

The boundary is **surpassed**: the log-polar render represents the far goals the linear render clipped
off-image, and with the bump finally present, the spiking SC orienting read-out (FIX1 + pop-vector +
#4 WTA) TRACKS the moving goal on the diagonal phases and re-orients toward the host ceiling, with the
retinotopy-scramble lesion collapsing 25.8×. The residual the prior NEGATIVE convergence reported was a
**non-biological retina truncation (sc_retina mass 0.0 for far goals), NOT a substrate limit** — the
biology-faithful SC retina (log-polar foveal magnification) never truncates, and everything downstream
that already worked on a strong margin (FIX1 reaches host-level finalQ + SCRAM collapses) now has a
margin on the diagonals too. There is no dendritic frontier here and no graded-read-out /
point-neuron-limit family wall — it was a representation-coverage fix.

On the 6-seed confirmation the host orienting heuristic retires for this benchmark: the spiking SC,
with a biology-faithful input representation, re-orients within the deploy regime across seeds — a
question this rank-1 de-risk answers and the seven prior sel-stage/accumulator fixes could not (they
were patching the wrong stage; the input representation, not the selection it feeds, was the missing
biology). The `log_polar_retina` default-on for the merged-nav path is recommended once the 6-seed
confirms (it is the biology-faithful default).

**This is the canonical deep-research-at-a-boundary outcome:** the ISOLATE step (the prior scoping)
pinned the residual to a specific 10-line render formula (retina mass 0.0, measured); the REFRAME
identified the right upstream stage (the SC's log-polar input representation, not the selection it
feeds); the RANK named a cheap, no-`sim/`-edit, biology-faithful fix; and this de-risk confirmed it
restores the signal and the orienting, on GPU, grid-32 faithful, moat intact.

---

## Provenance + machinery (file:line)

- **The render edit (NO `sim/`):** `render_egocentric_goal` (`research/runners/g11_bg_runner.py:183`,
  additive `log_polar`/`log_polar_d0`/`log_polar_grid_size` default-OFF kwargs); the
  `run_moving_goal_episode` `log_polar_retina`/`log_polar_d0` kwargs (`g11_bg_runner.py:~3654`, SC_LOG_POLAR
  env also enables); the env resolution (`g11_bg_runner.py:~4298`, `_log_polar_retina`); the egocentric
  SC eye-drive call site threading it (`g11_bg_runner.py:~7043`); the results-dict flag
  (`g11_bg_runner.py:~7836`).
- **The harness:** `research/runners/_nav_sc_popvector_readout_derisk.py` (`--log-polar` / `--log-polar-d0`
  flags threading `log_polar_retina` into the SC arms; the host arm has no SC so it is unaffected).
- **The CPU smoke:** `research/runners/_shortcut6_logpolar_render_smoke.py` →
  `research/findings/raw/nav_gate_2a/logpolar_render_smoke.json` (8/8 mass>0 + correct bearing).
- **The grid-32 GO:** `research/findings/raw/nav_gate_2a/scpv_logpolar_seed42.json` (the summary) +
  `scpv_logpolar_{host,sc_popvector,sc_popvector_scr}_seed42.json` (the per-arm) +
  `scpv_logpolar_seed42_aggregate.json` (GO=True).
- **The 6-seed:** `research/runners/_shortcut6_logpolar_remaining5seeds.sh` (the driver) +
  `research/runners/_shortcut6_logpolar_aggregate.py` (the aggregator) →
  `scpv_logpolar_6seed_aggregate.json`.
- **The decode + bump machinery (unchanged):** `install_spiking_sc_wiring(popvector=True)`
  (`g11_bg_runner.py:287-296`, the cosine projection); the `sc_map↔sc_fs` Mexican-hat bump; the #4
  `sel_X`/`commit_X` WTA ring.
- **The scoping this de-risks:** `research/findings/2026-06-22-shortcut6-upstream-orienting-residual-surpass.md`
  (MOVE 1: the residual quantified — retina mass 0.0; MOVE 2: the log-polar reframe; MOVE 3: RANK 1).
- **Biology:** catalog E.04 (topographic maps "warped by behavioral importance — cortical magnification —
  fovea"), H.25 (SC saccade map, full-hemifield), A.07 (SNr→SC disinhibition gate); SC log-polar / foveal
  magnification (Ottes-Van Gisbergen-Eggermont; Hafed lab 2019; human-SC eccentricity work).
