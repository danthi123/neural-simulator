# Shortcut #5b — the R1 selective-afferent residual: SURPASS round (2026-06-22)

**Type:** READ-ONLY deep-research SURPASS scoping (no code edits — this doc is the only write; no GPU run beyond
reading existing JSONs + two CPU replays of the exact render function this session). The gated next step after the
CLOSE A de-risk (`db37e5bb`, `2026-06-22-shortcut5b-closeA-graded-on-selforg-derisk.md`) resolved the #5b fork to an
honest **R1-LIMIT**: the graded dendritic-plateau read-out (R2) is SOLVED on a SELECTIVE afferent (δ=1.33, ~9×
near/far V on the host-Gaussian, 3/3 seeds), but the genuine residual is **R1 — the self-organized `place` afferent
is not location-SELECTIVE**: the egocentric `place_sensors` render is locally degenerate, so the read fires the same
dominant cells at near AND far (V n/f ~1.18× from 1.91× near-selective weights) → δ stays flat ≤1.0. Per the
no-boundary rule, this scopes the SURPASS for R1. This is the canonical "deep-research at a multiply-confirmed
boundary" move.

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; the verdict is grid-32 (never grid-8); a boundary is not an
exit. The no-confab moat is array-disjoint from the nav cascade (the place/critic state is `cp_*` nav-cascade
arrays — `cp_connections` / `cp_membrane_potential_v` / `cp_firing_states`; the conversational composer's complex
`cp_rf_w_*` synapses are a separate allocation) and is untouched throughout; nothing here weakens it. Protected
`sim/` edits are APPROVED for the eventual fix — flagged for byte-review where applicable, not gated.

**Terms defined once.** *R1* = the afferent-selectivity residual (the `place` code fires the same dominant cells at
many locations). *R2* = the read-out residual (the all-or-none plateau is binary; SOLVED by the graded plateau).
*`place_sensors`* = the egocentric landmark-sensor render (`_n9_place_sensor_act`, `g11_bg_runner.py:96`): per fixed
landmark, 12 cosine-tuned bearing sensors + 8 Gaussian distance sensors — the sole input to the place pool. *`place`*
= the self-organized spiking place pool (competitive threshold-WTA on `place_sensors`, then frozen). *Object-vector
cell* = a cell tuned to distance+direction to a specific landmark (catalog D.09 — exactly what `place_sensors` are).
*Grid cell* = a medial-EC cell firing on a periodic triangular lattice tiling the whole arena, a context-invariant
metric (catalog D.07). *Adjacent-cell afferent cos* = the cosine between the sensor vectors at two grid cells one step
apart — the direct measure of whether the INPUT distinguishes nearby locations (1.0 = blind; lower = selective).
*δ (delta)* = the dopamine reward-prediction-error gap V(near)/V(far) the critic delivers.

---

## TL;DR — the verdict in four sentences

The CLOSE A de-risk correctly localized the residual to R1 (afferent selectivity), and this SURPASS round pins R1
one stage further up still — to the egocentric `place_sensors` **render itself**: a CPU replay of `_n9_place_sensor_act`
shows the **adjacent-cell afferent cos is ~0.99** (the input at two cells one step apart is near-identical), so the
place pool — a point-neuron that can only spike on the structure the INPUT carries (Mikulasch-Priesemann) — produces
a locally non-selective code no read-out can rescue. Crucially, the cheap "more landmarks" fix (the direct analog of
#6's render fix) **does NOT work**: bearing+distance to MORE fixed landmarks stays locally smooth (25 landmarks +
sharp distance tuning only reaches adjacent cos ~0.95), because the egocentric vector code is inherently
slowly-varying across nearby cells. The biology-faithful fix is the **medial-EC grid-cell metric** (catalog D.07,
the named missing piece): a CPU replay shows a 5-module grid code reaches adjacent-cell cos **0.74** (vs 0.99) and,
after a k-WTA place layer, adjacent-cell *place* cos **0.53** (vs the render's 0.98) — a fundamentally more
decorrelated, locally-selective metric. **The boundary is SURPASSABLE, but unlike #6 it is NOT a ~10-line render
tweak — it needs a new grid-cell front end (rank-1), which is a moderate build, not cheap-trivial.**

---

## MOVE 1 — ISOLATE + QUANTIFY: where the overlap arises, and how overlapping

The CLOSE A finding pinned the residual to R1 by elimination (R2 works on the host-Gaussian, fails on self-org). This
round answers the next question: **WHERE in the self-org path is the overlap — the RENDER (the sensory input) or the
self-org LEARNING?** A 5-line CPU replay of the EXACT render function `_n9_place_sensor_act` (`g11_bg_runner.py:96`,
defaults `n_bearing=12, n_dist=8, max_intensity=450, falloff=0.03, dist_sigma=4.0, dist_max=grid·1.42, bexp=4`,
landmarks `(0,0),(31,0),(15.5,31)`) localizes it decisively to **the RENDER, before any learning**.

### 1a. The afferent (render) is locally degenerate — quantified

| measurement | afferent (render) cos | reading |
|---|---|---|
| **adjacent cells, 1 step apart** (13,13)→(14,13) | **0.9954** | the INPUT barely distinguishes neighbouring locations |
| 2 cells apart (13,13)→(15,13) | 0.9833 | |
| 3 cells apart | 0.9652 | |
| 5 cells apart | 0.9111 | |
| 8 cells apart | 0.7800 | |
| **mean adjacent-cell cos over the interior** | **0.9921** | the render is near-blind to local position |
| near/far value-train pair (6,6)→(25,25), 19 cells | 0.3792 | far-apart locations DO separate — but the value-train must separate the trained location from its neighbours, not just from the antipode |
| self-org sweep (36 positions): mean pairwise | 0.4209 | the sweep positions are far apart (≥5 cells) so they look separable in aggregate — masking the local degeneracy |

**The decisive read:** the afferent is **locally degenerate** (adjacent-cell cos ~0.99) even though far-apart
locations separate (near/far cos 0.38). This is exactly the R1 signature CLOSE A measured downstream (V n/f ~1.18×
from 1.91× weights): a value-train can grow near-selective WEIGHTS (the few cells that happen to fire slightly more
at the goal), but because the active ENSEMBLE is near-identical at the goal and its neighbours, the read collapses
that weight gradient to a ~1.18× value. **~80% of the learned selectivity is lost in the read because the input the
read operates on is locally non-selective.** The overlap is in the RENDER, not the learning — the self-org rule is
faithfully carving fields from an input that has almost no local structure to carve.

### 1b. WHY the render is locally degenerate — the component structure

The render is an **egocentric object-vector code** (catalog D.09): per landmark, a bearing block (which direction the
landmark lies, `cos_align**4`) + a distance block (how far, a Gaussian over `dist_max`). The CPU replay shows why
this is locally smooth:

- **Bearing saturates and varies slowly.** With `bexp=4`, only ~6/12 bearing sensors per landmark are non-zero (the
  `cos**4` is sharp), and the bearing to a *fixed, distant* landmark changes by a tiny angle when the agent moves one
  cell — so the active bearing sensors and their magnitudes are nearly unchanged step-to-step.
- **Distance is broad.** `dist_sigma=4.0` on a 32-cell grid means the distance Gaussians span ~8 cells; moving one
  cell shifts `d` by ≤1, well inside one sigma, so the distance block barely moves.
- **Net:** the full sensor vector is a slowly-varying function of position. Two adjacent cells produce
  near-collinear vectors (cos 0.99). This is intrinsic to "bearing + distance to a few fixed points," not a tuning
  artifact — it is the geometry of an object-vector code with a handful of anchors.

⇒ **The genuine residual, pinned to the byte: `_n9_place_sensor_act` produces a LOCALLY non-selective egocentric
object-vector code (adjacent-cell cos ~0.99), so the point-neuron `place` pool cannot carve locally-selective fields
— it has almost no local structure in the input to separate.** R1 is a RENDER (sensory-input-representation) problem,
exactly as #6's residual was, and the CLOSE A V-n/f ~1.18× cap is its downstream fingerprint.

---

## MOVE 2 — REFRAME: how does the brain get a SELECTIVE, DECORRELATED place code?

The project has been treating the place code as "object-vector sensors → competitive WTA → place fields" (the
Hartley-Burgess landmark route). That route IS biological, but it is MISSING the decorrelated metric the real
medial-EC place system is built on. Biology says the local selectivity comes from upstream of the place layer — from
the **grid-cell metric**, not from sharpening the object-vector sensors.

### 2a. The catalog names the exact missing piece — grid cells (D.07)

Reading the catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`), the project's status on the
relevant entries is unambiguous:

| ID | Mechanism | Project status (catalog's own words) |
|---|---|---|
| **D.06** | Place cells (O'Keefe) — sparse, location-SELECTIVE allocentric fields | "partial — cells are **sensor-driven not allocentric**, ... the population is undifferentiated" — exactly R1 |
| **D.09** | Border / **object-vector cells** | "**partial — landmark sensors in `g11` provide distance + direction to landmarks, which is functionally object-vector encoding at the sensor stage; no grid alignment to borders since no grid cells**" |
| **D.07** | **Grid cells** (Moser) — periodic triangular lattice, context-invariant Cartesian metric, "tiling the entire environment" | "**missing** — no path-integration substrate, no periodic spatial firing" |
| **D.12** | Pattern separation (DG): strong feedforward inhibition → sparse, orthogonalized | the sparsity mechanism — but CLOSE A confirmed sparsity ≠ selectivity (W=10 already 6% sparse, still non-selective) |

**The catalog's D.09 note is the load-bearing sentence:** the `place_sensors` ARE object-vector cells, but with
**"no grid alignment ... since no grid cells."** The medial-EC place system gets its fine, locally-selective metric
from the **grid cells** (D.07), which the project is **missing entirely**. The object-vector sensors anchor the metric
to landmarks; the grid cells PROVIDE the metric. Without grid cells, the place layer is reading only the
locally-smooth object-vector code — exactly what MOVE 1 measured.

### 2b. The CPU replay confirms the grid-cell metric is locally selective where the render is not

A 30-line CPU replay of a canonical grid code (5 modules × 20 cells, each module a rectified sum of 3 plane waves at
60° with module-specific spatial scale `λ` geometrically spaced 3.5→26 cells and random per-cell phase — the
Moser/Hafting oscillatory-interference / attractor form):

| code | adjacent-cell cos (1 step) | near/far (6,6)→(25,25) cos | sweep mean pairwise | after k-WTA place layer: adjacent-cell *place* cos |
|---|---|---|---|---|
| **render (`place_sensors`)** | **0.9921** | 0.379 | 0.421 | **0.976** (locally blind — the R1 cap) |
| **grid-cell metric (5 modules × 20)** | **0.7379** | 0.266 | 0.267 | **0.528** (locally selective) |

**The reads (decisive):**
- The grid metric's **adjacent-cell cos is 0.74 vs the render's 0.99** — a fundamentally more decorrelated input at
  the local scale that R1 needs. The periodic multi-scale lattice changes substantially between adjacent cells (the
  short-`λ` modules cycle every few cells), which is precisely the property the slowly-varying object-vector code
  lacks.
- **After a fixed-random k-WTA place layer, the grid input yields adjacent-cell place cos 0.53 vs the render's 0.98.**
  The place layer over the grid metric is locally SELECTIVE; the same place layer over the render is locally blind.
  This is the on-substrate version of the fix: a competitive place pool reading a grid metric carves locally-distinct
  fields, which a value-train can then grade.
- This is the canonical biological route (catalog D.07): **grid (decorrelated metric) → place (competition selects
  single fields)** is how the real medial-EC → hippocampus path produces sparse, locally-selective place cells.

### 2c. The crucial NEGATIVE — "more landmarks" (the cheap #6-style render fix) does NOT fix R1

This is the most important reframe finding, because it rules out the cheapest path. #6's residual was fixed by a
~10-line render change (log-polar). The direct #5b analog — "richer / more landmarks for a less-overlapping
egocentric code" (CLOSE A's own ranked option 1) — was tested by CPU replay and **fails**:

| render variant | mean adjacent-cell cos |
|---|---|
| baseline (3 landmarks, bexp=4) | 0.9921 |
| 4 / 6 / 8 / 12 landmarks (ring) | 0.9915 / 0.9904 / 0.9874 / 0.9904 |
| 3×3 / 4×4 / 5×5 grid of landmarks (9/16/25) | 0.9908 / 0.9884 / 0.9871 |
| 3 landmarks + sharper distance (`dist_sigma` 4→1) | 0.9556 |
| 8-ring + `n_dist=16` + `dist_sigma=1.5` (combined) | 0.9557 |

**Even 25 landmarks + maximally-sharpened distance tuning only reaches adjacent-cell cos ~0.95** — still far from the
grid metric's 0.74. The reason is structural (1b): bearing to fixed anchors varies slowly, and adding anchors adds
more slowly-varying channels, not local structure. The render-sharpening levers help marginally (sharper distance is
the only real mover) but cannot reach the local selectivity R1 requires. **The cheap render tweak that worked for #6
does NOT work for #5b — this is where the two afferent problems diverge.** #6's render CLIPPED the signal (mass 0.0,
a coverage problem a remapping fixes); #5b's render is locally SMOOTH (a decorrelation problem only a richer metric —
grid cells — fixes).

### 2d. Why this is not a point-neuron wall (the substrate question)

R1 looks Mikulasch-Priesemann-flavoured ("the point neuron can't decorrelate"), but the CPU replay shows it is NOT a
substrate limit: the decorrelation is supplied by the **INPUT** (the grid metric is decorrelated by construction —
plane-wave geometry, not a learned whitening), and a plain feedforward k-WTA place layer (no dendrites, no learned
decorrelation) reads a locally-selective code off it (place cos 0.53). The point-neuron place pool spikes on the
structure the input carries (Mikulasch-Priesemann) — so the fix is to give it a decorrelated input, not to make the
point neuron decorrelate. This is the SAME resolution as the conversation PPMI cortex ("decorrelation is a red
herring; the fix is the right INPUT representation, not cross-neuron decorrelation", CYCLE 88) and the B1 self-org RF
("the structure comes from the input statistics + the rule, not the substrate alone"). The dendrite (CLOSE C) is the
LAST resort, justified only if a grid front end proves insufficient — and the CPU evidence says it should not.

---

## MOVE 3 — RANK the cheap-first SURPASS mechanisms (the path PAST R1)

All ranked by leverage × cheapness × reuse × directness-of-attacking-R1. Each: the mechanism, the reusable machinery,
the cheap-first de-risk, the anti-cheats, and the `sim/`-edit-or-not flag.

### RANK 1 (RECOMMENDED) — a medial-EC grid-cell metric front end → the place layer (D.07)

- **Mechanism.** Insert a `grid_cells` region between position and `place`: a periodic multi-scale metric (5 modules
  × ~20 cells, each a rectified sum of 3 plane waves at 60° with a module-specific scale and a fixed per-cell phase
  — the oscillatory-interference / attractor grid form), driven from the agent's own position, then projected to the
  `place` pool which carves locally-selective fields by the EXISTING competitive threshold-WTA. The grid code is the
  context-invariant Cartesian metric the place layer is currently missing.
- **Sourcing the grid code legitimately (the anti-cheat that matters).** The grid drive reads ONLY the agent's own
  position `(x,y)` — which is ALREADY the legitimate egocentric self-knowledge `place_sensors` reads ("(x,y) enters
  the brain ONLY through this legitimate sensory render", `g11_bg_runner.py:98`; and the hidden-goal diagnostic's
  "own position is legitimate egocentric self-knowledge under BRAIN-BASED-ONLY"). A spatial-phase grid code (the
  periodic lattice evaluated AT the agent's position) is a fixed sensory transform of the SAME legitimate
  self-position input — it never reads the GOAL coordinates. This sidesteps the catalog D.07 "no path-integration
  substrate" gap: the project has no velocity/heading sensors (confirmed — no HD/velocity/grid machinery in the
  runner), so a *path-integration* grid generator is out of cheap scope, but a *spatial-phase* grid code (the
  attractor's fixed-point firing pattern as a function of position) is the cheap, equivalent realization and is what
  the CPU replay validated.
- **Why it attacks R1 directly.** It converts the place layer's input from adjacent-cell cos 0.99 (locally blind) to
  0.74 (locally selective) → the place fields become locally distinct (place cos 0.53) → the value-train's
  near-selective weights now read a near-selective VALUE → δ can grade. It is the named missing biology (D.07), the
  canonical grid→place route, and the only tested lever that reaches the required local selectivity.
- **Reusable machinery.** The regions framework (`BrainRegion` + `RegionPathway`) to declare `grid_cells` +
  `grid_cells → place` (plastic, competitive — the SAME role the current `place_sensors → place` plays); the existing
  `_run_place_selforg` self-org loop (re-point it at the grid input, unchanged WTA); the `_n9_place_ensemble` /
  diff-cos provenance gates; the CLOSE A graded-plateau read-out (`enable_graded_dendritic_plateau`, already shipped
  + byte-reviewed, `d69cc0ab`) for the value read; the `_n5_closeA_graded_on_selforg_probe.py` /
  `_n5_place_sparsify_probe.py` harnesses; the B1 dev-random precedent for the fixed-phase draw
  (`2026-06-21-B1-v1-gabor-selforg-derisk.md` — a genome-style `rng(seed)` draw is the accepted self-organized bar).
  The grid code generator is a ~40-line runner helper (the CPU replay IS the reference).
- **Cheap-first de-risk.** (1) **CPU place-selectivity smoke (seconds, no GPU, mostly done this session):** grid code
  → fixed-random k-WTA place layer → confirm adjacent-cell place cos materially below the render's 0.98 (the replay:
  0.53) AND the near/far VALUE separation a value-train would see grades. (2) **on-bridge STEP-1 confirm (GPU,
  seed 42):** build the `grid_cells → place` path, run `_run_place_selforg`, read the diff-cos provenance AND the
  read-regime adjacent/near-far cos — confirm the FROZEN place code is locally selective (read cos < 0.3 on a
  near-neighbour pair, where the render gave ~0.99 locally). (3) **the δ verdict (GPU):** run the CLOSE A
  fixed-readout-only test (canonical place code over the grid metric + the graded plateau) — does V n/f now exceed
  the render's ~1.18× and δ cross 1.3? numpy-CPU smoke → GPU seed 42 → 6-seed if it clears 1.3.
- **Anti-cheats (all carried + the new ones).** (a) **HOST-GAUSSIAN positive control** (CLOSE A's δ=1.33 anchor) —
  the grid place code must approach it. (b) **the render-baseline negative control** (the SAME pipeline on the
  current `place_sensors` render must stay flat, V n/f ~1.18× — isolates that the lift is the grid INPUT, not the
  read-out, which CLOSE A already proved is solved). (c) **no-learning floor** (value_train_trials=0 → the grid place
  code's raw near/far geometry, NOT learned value — the lift must come from the value-train ON a selective afferent,
  reproducing CLOSE A's no-learning control). (d) **grid-scramble lesion** (permute the grid-cell phases/wiring → the
  metric becomes non-selective → selectivity collapses → proves the periodic metric is load-bearing, not a generic
  expansion — the analog of the SCRAM / OSI=0 controls). (e) **NEW — the grid code must not smuggle the goal:** assert
  the grid drive reads ONLY agent `(x,y)` (the legitimate self-position channel), NEVER `(gx,gy)` — the value/δ
  selectivity must come from the place→value learning, not from the grid code encoding the goal. (f) **grid-32, never
  grid-8.** (g) **plateau-lesion** (CLOSE A's load-bearing control — graded-plateau strength=0 collapses δ). (h)
  **6-seed on a GO.** (i) **moat untouched** (no conversational regions in the nav run; the place/critic state is
  array-disjoint from `cp_rf_w_*` — preserved by construction, as CLOSE A's build-region-list assertion confirms).
  (j) **default-off byte-identity** (the grid front end behind a default-off flag; `test_nav_conv_merged_agent` 8/8 +
  `test_nav_conv_step2b_coresident` 7/7).
- **`sim/` edit?** **Likely NONE for the core build** — the grid generator is a runner helper, and `grid_cells` +
  `grid_cells → place` are declared via the regions framework (no `sim/` edit, exactly as B1's self-org RF reused the
  existing plastic pathway). The value read-out (`enable_graded_dendritic_plateau`) already ships. The only plausible
  `sim/` touch is IF the grid drive needs a current-injection path the framework doesn't expose — but the existing
  `cp_external_input_current` per-region drive (used for `place_sensors`) covers it. **Flag for byte-review only if a
  new `sim/` injection path is needed; the expectation is runner-only.** This is the strongest combination: directly
  attacks R1, biology-faithful (the named missing D.07), reuses the self-org loop + the shipped graded read-out, and
  needs no protected edit.
- **HONEST cost (must be stated):** unlike #6's ~10-line render tweak, RANK 1 is a **moderate build** (a new sensory
  region + its self-org + the full δ re-validation), not cheap-trivial. The CPU evidence is strong that it works, but
  it is a day-scale build, not an afternoon. It is still far below the months-scale dendrite (CLOSE C).

### RANK 2 — sharper egocentric render (sharper distance tuning + more landmarks) as an ABLATION, not a fix

- **Mechanism.** The CLOSE A option-1 lever: more landmarks + sharper `dist_sigma` + more `n_dist` for a
  less-overlapping object-vector code. Runner-local (the `place_sensors` size + `_n9_place_sensor_act` params), NO
  `sim/` edit.
- **Why ranked below RANK 1 — it is a documented-by-replay NEGATIVE as a standalone fix.** MOVE 2c quantified it:
  even 25 landmarks + `dist_sigma=1.5` + `n_dist=16` reaches only adjacent-cell cos ~0.95, far short of the grid
  metric's 0.74. The egocentric vector code is intrinsically locally smooth; sharpening it cannot reach the required
  local selectivity. Its VALUE is as an **ablation alongside RANK 1**: if the sharpened render (RANK 2) still gives a
  flat δ but the grid front end (RANK 1) grades, that attributes the lift specifically to the decorrelated METRIC, not
  merely to "more sensors." Run it as the control, not the fix.
- **Reusable machinery / de-risk / anti-cheats.** Identical harness to RANK 1 (same `_n5_closeA` δ probe with a
  richer render). **NO `sim/` edit.**

### RANK 3 (the deep substrate route — NOT cheap; the honest deferred fork) — dendritic per-cell field carving (D.06/G.02)

- The Major-Larkum-Schiller / Poirazi-Mel two-compartment NMDA-plateau nonlinearity to carve selective fields per
  cell from overlapping input — the named months-scale deferred dendritic rewrite (the recurring
  Mikulasch-Priesemann wall, CLOSE A's option 3). **Out of cheap scope, and per MOVE 2d the CPU evidence says it is
  NOT necessary** (a plain feedforward k-WTA place layer over a grid metric already reaches local selectivity). Only
  justified if RANK 1 (a decorrelated grid input) proves insufficient AND the spatial-value δ is re-prioritized.
  Reserve.

---

## MOVE 4 — VERDICT: SURPASSABLE via a grid-cell front end (RANK 1), moderate-build not trivial

**The boundary is SURPASSABLE; it survives the SURPASS round as a precisely-located, non-irreducible residual with a
biology-faithful fix — but the fix is a moderate build (a new grid-cell front end), not the ~10-line render tweak that
closed #6.** CLOSE A correctly localized R1 to afferent selectivity; this round pins R1 to the `place_sensors` render
itself (adjacent-cell afferent cos ~0.99, measured) and identifies the right upstream stage: the **medial-EC
grid-cell metric (catalog D.07), which the project is missing entirely** ("no grid alignment ... since no grid cells").

**The genuinely-irreducible part is SMALL and is NOT a substrate limit:** the only thing "irreducible" about the
current setup is that an egocentric object-vector code with a handful of fixed anchors is locally smooth — a
representation-coverage gap (the missing grid metric), not a point-neuron limit. The CPU replays show (i) the grid
metric is locally selective where the render is not (adjacent cos 0.74 vs 0.99), (ii) a plain feedforward k-WTA place
layer over the grid input is locally selective (place cos 0.53 vs 0.98) — NO dendrite needed, and (iii) the cheap
"more landmarks" alternative provably cannot reach it (≤0.95 at 25 landmarks). Once the place layer reads a
decorrelated metric, the value-train's near-selective weights read a near-selective value, and the already-validated
graded read-out (CLOSE A, R2 solved) grades it — δ should cross 1.3.

**Recommended rank-1 de-risk (the precise next move):** build a spatial-phase grid-cell front end (a ~40-line runner
helper, the CPU replay as reference) driving a `grid_cells → place` competitive pathway via the regions framework
(no `sim/` edit expected); (1) CPU place-selectivity smoke (confirm adjacent-cell place cos ≪ the render's 0.98 — the
replay gives 0.53); then (2) on-bridge STEP-1 confirm (the FROZEN place code is locally selective, read cos < 0.3 on
a near-neighbour pair); then (3) the CLOSE A fixed-readout-only δ verdict (grid place code + graded plateau vs the
render-baseline negative control vs the host-Gaussian positive control). **GO bar = V n/f exceeds the render's ~1.18×
AND δ crosses 1.3, with the grid-scramble lesion collapsing it, the render-baseline staying flat, the no-learning
floor flat, and the moat intact** — then 6-seed. The host-Gaussian `vs_place_context` retires only if the self-org
place code, with a biology-faithful decorrelated metric, grades the value within the δ bar across seeds — a question
this rank-1 de-risk answers and the render-sharpening levers never could.

**The validate-by-function caveat (carried from CLOSE A §4 / the scoping §4, restated):** the nav δ is INERT (the #9
lesson — the nav value is not load-bearing on immediate-reward nav), so closing R1 would NOT change navigation
itself. BUT the right downstream consumer of a selective place code is the **hidden-goal (Morris-water-maze)
actor-critic arc** (`2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md`) — which uses the SAME
`place → cortex_action` plastic pathway and where the place code's spatial selectivity IS load-bearing (the goal is
not perceivable, so the agent MUST learn place→action from reward). A selective place code (R1 fix) is the named
upstream prerequisite for that arc — so RANK 1 is not just a δ-cosmetic; it is the afferent the deferred spatial-credit
arc needs. The grid front end (RANK 1) is therefore the joint unlocker for both the #5b δ-lift AND the hidden-goal
spatial-credit wall.

**This is the canonical deep-research-at-a-boundary outcome: the comfortable R1-LIMIT verdict was the START of the
research, not the end. The ISOLATE step pinned the residual to the render's locally-degenerate object-vector code
(adjacent cos 0.99, measured); the REFRAME identified the right upstream stage (the medial-EC grid metric, D.07, the
named missing piece) and ruled out the cheap "more landmarks" path by replay; and the RANK named a biology-faithful,
likely-no-`sim/`-edit grid-cell front end. The boundary is surpassable — at moderate build cost, not trivially.**

---

## Provenance + machinery (file:line, for the controller's trust-but-verify)

- **The residual, quantified (CPU replays this session):** `_n9_place_sensor_act` (`g11_bg_runner.py:96`, defaults
  `n_bearing=12, n_dist=8, max_intensity=450, falloff=0.03, dist_sigma=4.0, dist_max=grid·1.42, bexp=4`); landmarks
  `_n9_place_landmarks` (`:88`, `(0,0),(31,0),(15.5,31)`); the per-step nav drive (`:7000-7007`). Adjacent-cell
  afferent cos **0.9921** (interior mean); the 25-landmark + sharp-distance ceiling **~0.95**; the grid-cell metric
  **0.74** → k-WTA place **0.53** vs the render place **0.976**.
- **The R1 localization (CLOSE A):** `2026-06-22-shortcut5b-closeA-graded-on-selforg-derisk.md` /
  `db37e5bb` (R2 solved on host-Gaussian δ=1.33 9×; R1-limited on self-org V n/f 1.18× δ 0.94–1.00; the
  no-learning control = a direct R1 fingerprint); the scoping `2026-06-21-shortcut5b-sparse-place-fields-scoping.md`
  (the sparsify run-to-ground; R1/R2 split).
- **The self-org loop (re-pointable at the grid input):** `_run_place_selforg` (`g11_bg_runner.py:5499-5553`);
  `_n9_place_ensemble` (`:5482`); the `place_sensors → place` plastic competitive pathway (`:1895`, gate
  `landmark_to_place`); the `place_fs` FS-PING.
- **The value read-out (already shipped, reused unchanged):** `enable_graded_dendritic_plateau` (`d69cc0ab` +
  `f941a39b`; `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`); the CLOSE A probe
  `_n5_closeA_graded_on_selforg_probe.py`; the sparsify harness `_n5_place_sparsify_probe.py`.
- **The downstream consumer (validate-by-function):** the hidden-goal actor-critic arc
  `2026-06-19-limbic-core-load-bearing-hidden-goal-diagnostic.md` (the same `place → cortex_action` plastic pathway,
  `g11_bg_runner.py:1450`); `2026-06-19-spiking-actor-critic-advantage-routing-derisk.md`.
- **The B1 precedent (self-org sensory front end, no `sim/` edit, dev-random bar):**
  `2026-06-21-B1-v1-gabor-selforg-derisk.md` (`0594b3b2`); the dev-random precedent `sim/dendritic_neuron.py:25`.
- **The #6 parallel (afferent/render residual, the diverging shape):** `2026-06-22-shortcut6-upstream-orienting-residual-surpass.md`
  (`2a94226a` — #6's render CLIPS / coverage problem → log-polar; #5b's render is locally SMOOTH / decorrelation
  problem → grid cells; different fix).
- **Biology (catalog, verified):** `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` — **D.07** grid
  cells "missing — no periodic spatial firing" (Kandel 6e Ch 54 pp 1361–1364); **D.09** border/object-vector cells
  "no grid alignment ... since no grid cells"; **D.06** place cells "sensor-driven not allocentric"; **D.12** pattern
  separation. Mikulasch-Priesemann point-neuron decorrelation limit (CLAUDE.md) — resolved here the same way as the
  PPMI cortex (the fix is the decorrelated INPUT, not point-neuron decorrelation).

_READ-ONLY SURPASS scoping. This doc is the only write; no code edited, no protected `sim/` touched. grid-32 IS the
verdict (never grid-8). The no-confab moat is array-disjoint from the nav cascade and untouched. Load-bearing claims
cited to `g11_bg_runner.py` line numbers + the existing CLOSE A / B1 findings + the catalog (D.06/D.07/D.09/D.12); the
render-degeneracy residual and the grid-metric reframe were confirmed by CPU replays of the exact render function this
session._
