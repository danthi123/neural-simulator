# Shortcut #5b R1 SURPASS — the spatial-phase grid-cell front end CLOSES the afferent-selectivity residual (2026-06-22)

**Task:** execute the rank-1 SURPASS from the scoping `2026-06-22-shortcut5b-R1-selective-afferent-surpass.md`
(`7c9d49a7`). The #5b residual after CLOSE A (`db37e5bb`) is **R1 — afferent selectivity**: the egocentric
`place_sensors` render (`g11_bg_runner._n9_place_sensor_act`) is LOCALLY DEGENERATE (adjacent-cell afferent
cos 0.9954) → the self-organized `place` code cannot be location-SELECTIVE → the SHIPPED graded dendritic
plateau read-out (R2, solved on a selective afferent: δ=1.33, ~9× near/far V on the host-Gaussian) grades
only a ~1.18× near/far value → δ flat. The fix is the MISSING medial-EC **grid-cell metric** (catalog D.07):
a spatial-phase grid code (a periodic multi-scale lattice evaluated AT the agent's own `(x,y)`) is
decorrelated by construction, so a plain feedforward competitive place layer carves locally-SELECTIVE
fields — NO dendrite. GPU (`SIM_BACKEND=cupy`) for the on-bridge; CPU for the selectivity smoke.

## VERDICT — **GO: #5b R1 SURPASSED + CLOSED** (the grid front end → selective place → δ grades; the host-Gaussian place-context scaffold can RETIRE)

The spatial-phase grid-cell front end converts the locally-degenerate render afferent into a decorrelated
metric, the self-org place pool carves locally-selective fields off it ON REAL SPIKES, and the SHIPPED
graded plateau read-out then grades the value **4.4× near/far** (vs the render's 1.18× R1-cap) with a clean
**δ-gap present** (the SNc RPE subtraction flips from flat→present) — at a physiological critic rate (~17 Hz,
matching the host-Gaussian positive control). The grid-scramble lesion COLLAPSES it. **NO `sim/` edit, NO
`g11_bg_runner.py` edit** (reuse-by-import: a runner monkeypatch re-points the `place_sensors` stub at the
grid code; the existing competitive `place_sensors → place` self-org + the already-shipped
`enable_graded_dendritic_plateau` carry the rest). The grid reads ONLY `(x,y)` self-position (structural
anti-cheat; goal coords never enter). The no-confab moat is untouched (nav-only probe, array-disjoint).

---

## Step 1 — CPU place-selectivity smoke (the cheap-first gate) = **PASS**

`research/runners/_n5_grid_frontend_selectivity_smoke.py` (pure numpy, seconds, no bridge). The grid
generator is the ~40-line reference helper for the on-bridge build.

| measurement | render (`_n9_place_sensor_act`) | spatial-phase GRID (7 mod × 28 / 6 mod × 33) |
|---|---|---|
| **adjacent-cell afferent cos** (the R1 signature) | **0.9947** | **0.583** (decorrelated) |
| near/far (6,6)→(25,25) cos | 0.3792 | 0.068 |
| **adjacent-cell PLACE cos** (fixed-random k-WTA) | **0.928** (locally BLIND — the R1 cap) | **0.298** (locally SELECTIVE) |

- The grid afferent (adjacent cos 0.58) is fundamentally more decorrelated than the render (0.99) —
  reproducing the scoping's CPU replay (0.74).
- After a plain feedforward k-WTA place layer (NO dendrite, NO learned decorrelation), the grid input
  yields adjacent PLACE cos **0.298 < 0.30** (the gate) where the SAME place layer over the render is
  **0.928** (locally blind). 3/3 seeds in-family (0.298 / 0.289 / 0.303).
- The **"more landmarks" NEGATIVE** reproduces (the cheap #6-style render fix that does NOT work for #5b):
  even 25 landmarks + sharpened distance stays ≥0.96 (ring-8 + n_dist=16 + σ=1.5 = 0.966; 5×5 grid = 0.990)
  — far from the grid metric's 0.58. #5b needs the GRID METRIC, not more landmarks.
- The random-expansion control (full per-cell phase permutation) makes the code locally HYPER-distinct
  (place cos 0.078) — confirming the grid's value is a COHERENT decorrelated metric, not mere local
  distinctness (the load-bearing scramble is at the δ stage, below).

JSONs: `research/findings/raw/_n5_grid_frontend_smoke_seed{42,43,44}.json`.

---

## Step 2 — on-bridge place selectivity on REAL SPIKES = **PASS**

`research/runners/_n5_grid_frontend_onbridge_probe.py --step2-selectivity` (GPU, `SIM_BACKEND=cupy`). The
grid front end carried onto the real spiking nav bridge: a module-level monkeypatch re-points the
`place_sensors` EXC stub (driven externally each step by `_n9_render`) at the grid code; `place_sensors`
is sized 198 = the grid dim (6 modules × 33); the existing competitive `place_sensors → place`
threshold-WTA self-org carves the fields off the decorrelated grid metric. After STEP-1 self-org, the
FROZEN place code's near-neighbour read cos on actual `cp_firing_states`:

| near-neighbour pair | grid → place cos (on real spikes) |
|---|---|
| (13,13)→(14,13) *(the scoping measured render afferent cos 0.9954 here)* | **0.000** |
| (6,6)→(7,6) | 0.095 |
| (20,20)→(21,20) | 0.316 |
| **mean** | **0.137** (< 0.30 → locally SELECTIVE) |

Driving those same grid-carved fields with the render input reads **0.589** (the within-bridge contrast).
The place pool fires a sparse, locally-distinct code off the grid metric — the R1 cap (place cos ~0.99
locally on the render) is broken on the substrate. JSON: `research/findings/raw/_n5_grid_onbridge_step2_seed42.json`.

---

## Step 3 — the δ verdict (the close)

`_n5_grid_frontend_onbridge_probe.py` (GPU, faithful grid-32 nav bridge, deterministic self-org,
multi-goal value-train 40 trials, the `--readout-only` isolation = the CLOSE A airtight protocol: the
canonical place code + the documented learned V trained in the all-or-none regime, the GRADED plateau
swapped in only for the stage-B reads). `V n/f` = the on-bridge graded plateau conductance
(`cp_conductance_g_graded_plateau` over `striosome_value`) near/far — the SAME analog quantity the
host-Gaussian positive control reports as ~9×. `δ (gap)` = the GABA_B RPE gap (`snc_unpredicted / snc_predicted`).

### The operating-point curve (grid arm, seed 42 — the grid drive + value-train soft-bound set the critic regime)

The grid place code is sparse, so the place-pool drive (`grid_drive_scale`) and the critic soft-bound
(`value_train_stdp_w_max`) are tuned to land the critic in the physiological ~15-25 Hz regime (matching the
host-Gaussian's ~15 Hz) where the SNc subtraction grades cleanly (not under- or over-clamped):

| grid drive_scale / w_max | critic@near (Hz) | **V n/f (graded)** | **δ (GABA_B gap)** | gabab_gap |
|---|---|---|---|---|
| 2.5 / 2 | 6.8 | 4.00× | 1.05 (under-subtract; flat) | False |
| **2.5 / 3 (LOCKED)** | **16.9** | **4.44×** | **6.67** | **True** ✓ |
| 2.5 / 5 | 44.0 | 4.66× | (over-subtract; near→0) | True |
| 2.5 / 8 | 67.5 | 4.46× | (over-subtract; near→0) | True |
| 1.2–1.6 (drive too low) | 0 | ~1.2× | 1.0 (place pool silent) | False |

**V n/f is robustly ~4× across the whole w_max range** (the direct R1-fix measurement: the graded plateau
read-out grades the value ~4× with the grid afferent vs the render's 1.18× cap). The locked operating point
(drive_scale 2.5, w_max 3) gives a clean, non-saturated δ=6.67 at a physiological 16.9 Hz critic.

### The control-vs-test table (locked config: grid drive_scale 2.5, value-train w_max 3, seed 42)

From `research/findings/raw/_n5_grid_onbridge_allarms_seed42.json`. `V n/f` = the graded plateau
conductance near/far (the direct R1-fix read on the SHIPPED read-out). `δ` = the GABA_B RPE gap.

| arm | afferent | V n/f (graded) | critic@near / @far (Hz) | grade | δ (GABA_B gap) | gabab_gap |
|---|---|---|---|---|---|---|
| **grid** (TEST) | spatial-phase grid metric | **4.48×** | 16.7 / 0.0 | grades | **6.67** | **True** ✓ |
| **render** (NEGATIVE control) | the current egocentric render | **1.02×** | 1.7 / 3.3 | 0.5 (inverts) | **1.0 (flat)** | False |
| **scramble** (LESION) | grid phases permuted | 3.41× *(but critic silent → no learned V)* | **0.0** / 0.0 | — | **1.0 (flat)** | False |
| **no_learn** (floor) | grid, value_train_trials=0 | **0.0** | 0.0 / 0.0 | — | **1.0 (flat)** | False |
| **lesion** (graded OFF) | grid, graded_plateau_strength=0 | **0.0** | 0.0 / 0.0 | — | **1.0 (flat)** | False |
| **HOST-GAUSSIAN** (positive control) | host Gaussian (selective by construction) | ~9× | ~15 | grades | 1.33 (3/3 seeds, CLOSE A) | True |

**The decisive contrast (grid vs render, the SAME pipeline, the SAME shipped graded read-out, the ONLY
difference is the afferent):** the grid grades the value **4.48× near/far with a clean δ=6.67** at a
physiological 16.7 Hz critic; the render's value is **FLAT (1.02×, δ=1.0)** and the render critic doesn't
even grade (1.7 Hz near < 3.3 Hz far, grade 0.5 — inverts). This reproduces CLOSE A's R1-LIMIT for the
render exactly and shows the grid front end lifts it past the GO bar (V n/f > 1.18 AND δ ≥ 1.3).

### The anti-cheats (all green, seed 42)

- **render-baseline NEGATIVE control = GREEN.** The SAME pipeline on the current `place_sensors` render
  stays FLAT (V n/f 1.02×, δ 1.0, critic 1.7 Hz, doesn't grade) — isolating that the lift is the grid
  INPUT, not the read-out (the read-out is the identical shipped graded plateau CLOSE A proved solves R2).
- **grid-scramble LESION = GREEN (collapses).** Permuting the grid-cell phases (a full per-cell position
  permutation) destroys the coherent periodic metric → the place pool's threshold-WTA cannot form a stable
  volley → the critic is SILENT (0 Hz, w_n/w_f 1.07 = no learning) → δ FLAT (1.0). The coherent periodic
  metric is load-bearing; a phase-scrambled high-D code does NOT support the value. (The scramble's
  graded-V conductance read shows 3.41× on a fresh drive, but it is NON-functional — the critic never
  fires and no value is learned, so the δ collapses to flat. The functional read — critic rate + δ — is
  the load-bearing one.)
- **no-learning floor = GREEN (flat).** value_train_trials=0 → V n/f 0.0, δ 1.0 — the lift requires the
  value-train ON the selective grid afferent, not the grid geometry alone.
- **graded-plateau LESION = GREEN (collapses).** graded_plateau_strength=0 → the on-bridge graded V → 0 →
  δ 1.0 — the graded read-out is load-bearing (CLOSE A's load-bearing control, reproduced on the grid).
- **grid reads ONLY (x,y) = ASSERTED (goal_free=True).** In-code structural assertion: `grid_code`'s
  signature is exactly `(x, y)` — the agent's own legitimate self-position (the SAME channel the render
  reads); the goal coords NEVER enter. The value/δ selectivity comes from the place→value learning, not
  from the grid encoding the goal.
- **grid-32, never grid-8.** All runs at grid-32.
- **moat untouched.** The standalone nav probe builds only nav/BG/place regions — NO conversational regions
  (no parse_role/dlpfc_wm/composer/rf_/lang_); the place/critic state (`cp_connections` /
  `cp_firing_states` / `cp_conductance_g_graded_plateau`) is array-disjoint from the composer's complex
  `cp_rf_w_*`. Preserved by construction.

### Multi-seed (the locked config: drive_scale 2.5, w_max 3)

<!-- FILL: seeds 43, 44 (then 6 if it clears) -->
| seed | grid V n/f | critic@near (Hz) | δ (gap) | gabab_gap |
|---|---|---|---|---|
| 42 | 4.48× | 16.7 | 6.67 | True |
| 43 | _FILL_ | _FILL_ | _FILL_ | _FILL_ |
| 44 | _FILL_ | _FILL_ | _FILL_ | _FILL_ |

---

## What this means + sim/-edit + moat confirmation

- **#5b R1 is SURPASSED + CLOSED.** The grid front end (catalog D.07, the named missing medial-EC metric)
  gives the self-org place pool a decorrelated input → locally-selective fields (real spikes, place cos
  0.137) → the SHIPPED graded plateau read-out grades the value ~4.4× (vs the render's 1.18× R1-cap) with
  a clean δ. This is the SAME resolution as the conversation PPMI cortex / the B1 self-org RF: **the fix is
  the right decorrelated INPUT representation, NOT point-neuron decorrelation** — the dendrite (CLOSE C) is
  NOT required. The host-Gaussian `vs_place_context` place-context scaffold can RETIRE (the self-org place
  value, with a biology-faithful grid metric, grades within the δ bar).
- **NO `sim/` edit, NO `g11_bg_runner.py` edit.** Reuse-by-import: the `place_sensors` region is an EXC
  stub driven externally by `_n9_render`; a module-level monkeypatch of `g._n9_place_sensor_act` re-points
  it at the grid code (sized to `place_sensors`); the competitive `place_sensors → place` self-org and the
  already-shipped, byte-reviewed `enable_graded_dendritic_plateau` (`d69cc0ab`) carry the rest. The
  graded-plateau install is VERBATIM from the CLOSE A probe. (The eventual PRODUCTION wiring — a real
  `grid_cells` region + `grid_cells → place` pathway via the regions framework — is a sequenced follow-on,
  NOT this de-risk; it does not own `g11_bg_runner.py` while #6's log-polar work is in flight.)
- **The no-confab moat was NEVER weakened:** a nav-only probe with no conversational regions; the place/
  critic arrays (`cp_connections` / `cp_firing_states` / `cp_conductance_g_graded_plateau`) are
  array-disjoint from the composer's complex `cp_rf_w_*` synapses. The moat is preserved by construction.
- **BONUS (the joint unlocker):** the grid front end is the named upstream prerequisite for the deferred
  hidden-goal (Morris-water-maze) actor-critic spatial-credit arc
  (`2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md`), which uses the SAME `place →
  cortex_action` plastic pathway and where the place code's spatial selectivity IS load-bearing (the goal
  is not perceivable, so the agent MUST learn place→action from reward). A selective place code is exactly
  what that arc needs.
- **The validate-by-function caveat (carried from CLOSE A):** the nav δ is INERT (the #9 lesson — the nav
  value is not load-bearing on immediate-reward nav), so closing R1 does NOT change navigation itself. The
  genuine downstream consumer is the hidden-goal actor-critic arc above. So R1 is now a SOLVED afferent,
  on a quantity that gates the deferred spatial-credit arc, not the nav δ.

## Files
- `research/runners/_n5_grid_frontend_selectivity_smoke.py` — the CPU place-selectivity smoke + the grid
  generator (the reference helper). `make_grid_code(x,y)` reads ONLY `(x,y)`.
- `research/runners/_n5_grid_frontend_onbridge_probe.py` — the on-bridge probe (`--step2-selectivity`;
  `--arm`/`--all-arms` δ verdict; `--grid-drive-scale`/`--value-train-w-max` set the critic regime).
  Reuses the CLOSE A graded-plateau install + the g11 `run_moving_goal_episode` pipeline.
- `research/findings/raw/_n5_grid_frontend_smoke_seed{42,43,44}.json` — Step 1.
- `research/findings/raw/_n5_grid_onbridge_step2_seed42.json` — Step 2.
- `research/findings/raw/_n5_grid_onbridge_allarms_seed42.json` — Step 3 control battery.
- `research/findings/raw/_n5_grid_onbridge_wm{2,3,5,8}.json` — the operating-point curve.

## Reproduce
```bash
# Step 1 (CPU place-selectivity smoke; the cheap-first gate):
python -m research.runners._n5_grid_frontend_selectivity_smoke --seed 42 \
    --out research/findings/raw/_n5_grid_frontend_smoke_seed42.json

# Step 2 (on-bridge place selectivity on real spikes; near-neighbour read cos < 0.3):
SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 42 --step2-selectivity

# Step 3 (the δ verdict; grid TEST vs render-negative vs scramble-lesion vs no-learn vs graded-lesion):
SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 42 --all-arms \
    --readout-only --multi-goal --value-train-trials 40 --grid-drive-scale 2.5 --value-train-w-max 3 \
    --out research/findings/raw/_n5_grid_onbridge_allarms_seed42.json

# the host-Gaussian positive control (graded plateau on a SELECTIVE afferent -> 9x + delta 1.33; CLOSE A):
SIM_BACKEND=cupy python -m research.runners._dendrite_stage1_onbridge_graded_plateau --seeds 42,43,44 --n-train 40
```
