# CLOSE A #5b de-risk — the SHIPPED graded dendritic plateau wired onto the SPARSE self-org place afferent: the read-out (R2) is SOLVED but the afferent selectivity (R1) caps δ → honest **R1-LIMIT** (2026-06-22)

**Task:** execute CLOSE A from the scoping `2026-06-21-shortcut5b-sparse-place-fields-scoping.md` — the one
genuinely-unexploited cheap close for the #5b boundary (the self-org place code's value-train δ is flat at
nav scale). CLOSE A = combine the **sparse self-org place** afferent (`place_sensors_to_place_weight=10`, the
sweet spot that lifted LEARNS-V to 1.91×) with the already-SHIPPED, validated **graded dendritic plateau
read-out** (`enable_graded_dendritic_plateau`, commit `d69cc0ab` — gave a clean monotone ~9× near/far V on the
HOST-GAUSSIAN afferent, 3/3 seeds). The two have NEVER been combined; the runner gates the graded plateau OFF
under self-org (`g11_bg_runner.py:4496`). GPU (`SIM_BACKEND=cupy`), faithful nav bridge (NOT a numpy proxy).

## VERDICT — **R1-LIMIT** (honest; the read-out is solved, the afferent selectivity is the genuine residual)

**The fork resolves to R1-LIMIT, not a genuine close.** On the airtight clean test (canonical W=10 place code +
the documented 1.91× learned-graded value weights + the VALIDATED graded plateau read-out — the same read-out
that gives 9× V + δ=1.33 on the host-Gaussian afferent), the critic rate near/far is **FLAT (grade ratio ≈1.0)**
and **δ stays flat (0.94–1.00, never ≥1.3)**, because the graded read-out grades an **OVERLAPPING value**: the
graded-V conductance near/far is only **~1.18–1.23×** even though the learned weights are **1.91× near-selective**.
The FS-PING-open read regime fires the same dominant place cells at near AND far, so the near-selective weight
gradient cannot localize V to the near location.

- **R2 (the binary read-out) is SOLVED.** The host-Gaussian positive control (the validated
  `_dendrite_stage1_onbridge_graded_plateau`, run fresh this session, 3/3 seeds): graded plateau → **δ=1.33**,
  `V near 0.130 > mid 0.081 > far 0.014` (**~9× near/far**), `w_near 2.72 / w_far 0.20` (13× learned-selective),
  graded-3=True, loc-sel=True. The graded read-out grades a SELECTIVE afferent cleanly. So the binary all-or-none
  read-out (the prior BOUNDARY's R2) is no longer the residual — the graded read-out fixes it **on a selective
  afferent**.
- **R1 (the afferent selectivity) is the genuine substrate-limit residual.** On the self-org afferent the SAME
  graded read-out grades V n/f only ~1.18× → δ flat. The read regime collapses the 1.91× near-selective WEIGHTS
  down to a ~1.18× near-selective VALUE — i.e. **~80% of the learned selectivity is lost in the read** because a
  few dominant cells fire everywhere. This is the catalog's **sparsity ≠ selectivity** wall (D.06/D.07/D.12): the
  egocentric `place_sensors` are heavily overlapping across nearby locations, so the point-neuron place pool spikes
  on whatever (overlapping) structure the INPUT carries (Mikulasch-Priesemann).

**⇒ Do NOT flip the merged default critic afferent to the self-org place code.** The host-Gaussian
`vs_place_context` (position-specific BY CONSTRUCTION) remains the documented better-δ scaffold (δ ~1.3). Per the
BRAIN-BASED-ONLY standard, this neural-underperforms-host mapping IS the deliverable. Note the scoping's §4
reframe also stands: the nav δ is INERT anyway (the #9 lesson — the nav value is not load-bearing on
immediate-reward nav), so even closing R1 would not change navigation; the real spatial-credit test is the
deferred actor-critic hidden-goal arc, NOT the nav δ.

---

## The control-vs-test table (the decisive measurement)

All GPU (`SIM_BACKEND=cupy`), faithful grid-32 nav bridge, deterministic self-org, multi-goal value-train
(40 trials), the **readout-only** isolation (the COUNT plateau ON through STEP-1 self-org + STEP-2 value-train
= the CANONICAL place code + the documented learned V; the graded read-out swapped in only at the value-train
freeze, just before the stage-B reads — so the read-out is the SOLE difference vs the all-or-none baseline).

| arm | afferent | read-out | STEP-1 cos | LEARNS-V (w_n/w_f) | graded-V near/far (n/f) | critic@near/far (Hz) | **δ (r−V gap)** |
|---|---|---|---|---|---|---|---|
| **HOST-GAUSSIAN control** (positive) | host Gaussian (selective by construction) | graded | — | True (13.4×) | 0.130 / 0.014 (**9.0×**) | — | **1.33** (3/3 seeds) |
| **test** seed 42 | self-org W=10 (sparse) | **graded** | 0.219 | True (**1.91**) | 100.6 / 85.6 (**1.18×**) | 15.4 / 15.3 (grade 1.009) | **0.94** (flat) |
| test seed 43 | self-org W=10 | graded | 0.694 | False (1.05) | 95.0 / 77.2 (1.23×) | 11.0 / 10.8 (grade 1.013) | **1.00** (flat) |
| test seed 44 | self-org W=10 | graded | 0.667 | False (0.93) | 122.7 / 103.4 (1.19×) | 17.5 / 18.9 (grade 0.926) | **1.00** (flat) |
| **allnone** (baseline) seed 42 | self-org W=10 | all-or-none (graded OFF) | 0.219 | True (**1.91**) | — | 4.58 / 4.17 (grade 1.10) | **1.04** (flat; reproduces the documented BOUNDARY) |

**Reading:** the graded read-out reproduces ~9× V + δ=1.33 on the host-Gaussian afferent (R2 solved), but on the
self-org afferent it grades only ~1.18× V → δ flat (R1-limited). The allnone baseline reproduces the documented
W=10 multi-goal result exactly (LEARNS-V 1.91×, critic ~4.5 Hz physiological, δ 1.04) — confirming the harness is
faithful and the test arm reads the SAME canonical config.

## The controls (all green)

| control | result | reading |
|---|---|---|
| **HOST-GAUSSIAN positive control** | graded plateau → δ=1.33, V 9× near/far, 3/3 seeds | the read-out (R2) WORKS on a selective afferent → the failure on self-org is the AFFERENT (R1), not the read-out |
| **graded-plateau LESION** (strength=0) | graded-V → 0.0, δ → 1.00, GABA_B gap collapses (3/3) | the graded plateau is LOAD-BEARING (V→0 when off) |
| **no-learning floor** (value_train_trials=0) | graded-V n/f ≈ 1.76× (single-goal) — **identical to the trained test arm** | the V near/far separation is RAW afferent geometry, NOT learned value (the learned 1.91× weights add ~0 to the read selectivity) — a direct R1 fingerprint |
| **no-sparsification control** (dense W=28 + graded) | critic over-fires 290–347 Hz, graded-V n/f 0.88× (inverted), δ flat 0.98 | the documented over-clamp regime; graded read-out does not rescue the dense code either |
| **moat (no-confab)** | the standalone nav probe builds 43 regions, ALL nav/BG/place — **NO conversational regions** (no parse_role/dlpfc_wm/composer/rf_/lang_/cortex_it); the graded-plateau array is indexed by `striosome_value` (critic), array-disjoint by construction | **moat intact** — never weakened (asserted from the build region list) |

## ISOLATE R1 (the genuine residual) — and the ranked next SURPASS move

**The genuine irreducible residual is SMALL-and-precise: the FS-PING-open read regime collapses the place code's
near-selectivity from 1.91× (weights) to ~1.18× (value).** R1 is *not* field DENSITY (sparsity is fine: W=10 →
6–8% active) and *not* the read-out (R2 is solved by the graded plateau on a selective afferent). R1 is
**afferent location-selectivity**: a few dominant place cells fire at MANY locations because the egocentric
`place_sensors` (bearing/distance to 3 fixed landmarks) are heavily overlapping across nearby grid cells, so the
point-neuron place pool — which can only spike on the structure the INPUT carries (Mikulasch-Priesemann; catalog
**D.06/D.07/D.12**, "sparsity ≠ selectivity") — produces a sparse-but-overlapping read code. The catalog's own
D.06 note names exactly this: the project's place-cell activations are *sensor-driven*, not true allocentric
fields.

**Ranked next SURPASS moves (cheapest first; each addresses R1 — the INPUT, not the read-out):**

1. **(cheap, INPUT-side) richer / less-overlapping landmark sensors → a more-decorrelated place afferent.** The
   read overlap is set by 3 fixed landmarks (`_n9_place_landmarks`) giving correlated bearing/distance across
   nearby cells. More landmarks (or boundary-vector / object-vector sensors with sharper distance tuning) →
   less-overlapping egocentric input → the threshold-WTA carves more-selective fields. Runner-local (more
   `place_sensors` + the render params), NO `sim/` edit. RISK: the 2026-06-19 sweep already showed FFI on the
   OUTPUT can't fix selectivity (sparsity≠selectivity); this targets the INPUT, the correct lever per the catalog
   — but it is a sensory-front-end change of unknown sufficiency (the egocentric geometry may stay correlated).
2. **(cheap-moderate, INPUT-side) a grid-cell-like / conjunctive front end (D.07).** A periodic, context-invariant
   metric (grid modules) → place competition selects single fields — the canonical biological route to selective
   place fields from a decorrelated metric input. Larger build (a new sensory region) but point-neuron/feedforward;
   the catalog flags grid cells as the missing piece (`g11` has object-vector sensors but no grid alignment).
3. **(deep, NOT cheap — the honest deferred fork) dendritic per-cell field carving (D.06/G.02).** The
   Major-Larkum-Schiller / Poirazi-Mel two-compartment NMDA-plateau nonlinearity ("cluster-on-one-branch ≫
   scattered") to carve selective fields per cell from overlapping input — the named months-scale dendritic
   rewrite (the recurring Mikulasch-Priesemann wall). Only justified if (1) the spatial-value δ is RE-prioritized
   AND (2) the INPUT-side moves (1/2) prove insufficient.

**The validate-by-function caveat (the scoping's §4, restated):** the nav δ is INERT (the #9 lesson — the nav
value is not load-bearing on immediate-reward nav), so closing R1 would NOT change navigation. The genuine
spatial-credit test for "the place code grades SPATIAL value better than the host Gaussian" is the deferred
actor-critic hidden-goal arc (the 3×-NEGATIVE wall the dendrite is the named unlocker for) — NOT the trace task
(which uses an orthogonal cue, sidestepping the spatial afferent), and NOT the nav δ. So R1 is a real
substrate-limit residual, but on a quantity that does not gate navigation.

---

## Why CLOSE A is NOT a genuine close (the honest mechanism)

The prior BOUNDARY (`2026-06-19-place-code-sparsify-default-BOUNDARY.md`) found two reachable read-out regimes for
the **all-or-none** coincidence plateau: low-weight (under-discriminates → flat δ) and high-weight (over-clamps the
SNc → δ→0). It named "a graded rate read-out that scales smoothly with V" as the specified next move. CLOSE A
supplies exactly that (the validated graded plateau). It **does** restore a physiological, non-over-clamping
critic (15 Hz, not 0 or 238) AND a graded-V that monotonically separates near>far — so the binary-read-out
residual (R2) is genuinely fixed. But the graded read-out grades whatever V the afferent supplies, and the self-org
afferent supplies an OVERLAPPING V (n/f 1.18× from a 1.91× weight gradient), because the read-regime active
ensemble is non-selective. So the graded read-out **grades the wrong thing** — exactly the HONEST RISK the scoping
flagged ("if the FS-open read still fires the same dominant cells at near AND far, even a perfectly-graded
read-out grades an overlapping V"). CLOSE A was the right first experiment (cheap, and it cleanly ISOLATES R1 from
R2: R2 works on the host-Gaussian, fails on self-org → R1 is localized), and it resolves the fork to the
deeper-selectivity side.

## sim/-edit flag

**NO `sim/` edit, NO `g11_bg_runner.py` edit.** CLOSE A is realized entirely in a standalone probe via an
init-time monkeypatch flipping `enable_graded_dendritic_plateau=True` + the validated graded params on
`self.core_config` before `_initialize_simulation_data` (the graded-plateau `sim/` code already ships, byte-reviewed,
commit `d69cc0ab`; the per-step block routes on the EXISTING `coincidence_detector` mask — no new wiring). The
graded plateau on the self-org path is therefore a default-OFF capability the runner could expose with a one-line
gate relaxation IF a selective afferent (R1 fix) is ever found — but that is gated on R1, which this de-risk shows
is the wall.

## Files
- `research/runners/_n5_closeA_graded_on_selforg_probe.py` — the CLOSE-A probe (NEW; standalone, no sim/ or
  g11_bg_runner.py edit). `--readout-only` isolates the read-out (canonical training regime + graded read at
  stage-B); `--all-arms` runs test/lesion/no_learn/dense/allnone; `--multi-goal` for the 1.91×-LEARNS-V regime.
- `research/findings/raw/_n5_closeA_seed42.json` — seed-42 single-goal 5-arm sweep (graded-ON-during-train).
- `research/findings/raw/_n5_closeA_seed42_multigoal_test.json` — multi-goal test (graded-ON-during-train →
  LEARNS-V perturbed to 0.99, motivating readout-only).
- `research/findings/raw/_n5_closeA_seed42_multigoal_readoutonly.json` — multi-goal all-arms; the allnone arm
  reproduces the documented canonical W=10 (LEARNS-V 1.91, δ 1.04).
- `research/findings/raw/_n5_closeA_seed{42,43,44}_RO_fixed_test.json` — the AIRTIGHT 3-seed fixed-readout-only
  test (canonical code + graded read-out): δ 0.94/1.00/1.00, graded-V n/f 1.18/1.23/1.19 → R1-LIMIT.
- Host-Gaussian positive control: `research/runners/_dendrite_stage1_onbridge_graded_plateau.py` (run 3/3 seeds:
  δ=1.33, V 9×; reproduces `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`).

## Reproduce
```bash
# the airtight fork test (canonical W=10 code + 1.91x learned V + graded read-out):
SIM_BACKEND=cupy python -m research.runners._n5_closeA_graded_on_selforg_probe \
    --seed 42 --multi-goal --readout-only --arm test --value-train-trials 40
# -> STEP-1 cos 0.219, LEARNS-V 1.91, critic@near 15.4/@far 15.3 Hz (grade 1.009),
#    graded-V n/f 1.18, delta 0.94 (FLAT) => R1-LIMIT

# the full anti-cheat battery (test/lesion/no_learn/dense/allnone), single-goal:
SIM_BACKEND=cupy python -m research.runners._n5_closeA_graded_on_selforg_probe \
    --seed 42 --all-arms --value-train-trials 40 --out research/findings/raw/_n5_closeA_seed42.json

# the host-Gaussian positive control (graded plateau on a SELECTIVE afferent -> 9x + delta 1.33):
SIM_BACKEND=cupy python -m research.runners._dendrite_stage1_onbridge_graded_plateau \
    --seeds 42,43,44 --n-train 40 --lead-ms 150
```
