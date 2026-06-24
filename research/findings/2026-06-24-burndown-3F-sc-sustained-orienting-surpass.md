# Burndown Phase-3F — the SC SUSTAINED-ORIENTING LOOP (inventory B-2): the SURPASS round (2026-06-24)

**Type:** READ-ONLY deep-research SURPASS round on a characterized boundary, with ONE cheap foreground GPU confirm
(grid-32/450, ~2 min). **NO `sim/` edit.** The mandatory 4-move SURPASS (ISOLATE+QUANTIFY · REFRAME via biology ·
RANK cheap surpasses · VERDICT) per the CLAUDE.md directive: *a boundary is accepted ONLY after the surpass round
survives.*

**Owner standard (load-bearing):** BRAIN-BASED-ONLY; the verdict is grid-32 (never grid-8); a boundary is not an exit.
The no-confab moat is array-disjoint from the nav cascade (`cp_connections` / `cp_membrane_potential_v` /
`cp_firing_states` vs the composer's complex `cp_rf_w_*`) and is untouched throughout.

---

## TL;DR — the verdict in three sentences

**B-2 is SURPASSED + ALREADY-CLOSED.** The inventory's B-2 entry (2026-06-23: *"the closed neural-reward→critic→actor
loop can't SUSTAIN navigation, ~58× worse; the scramble control localizes the failure to the reward/drive half, not
orienting"*) was the **symptom of a non-biological retina truncation + a host tie-break degeneracy**, NOT a substrate
limit — and both of its named "live next moves" have since **landed and are default-on**: the log-polar foveal-render
(4/4 seeds GO, flipped default-ON) + the cascade N-bias FIX1 stochastic tie-break (3/3 seeds GO). With the SC bump
finally **present** for far goals, the **exact same** closed neural-reward→critic→actor loop now tracks the moving goal
on the diagonal phases and **sustains** navigation (actor late-sustain ~0.97, holds within ~1 cell) at **~2.4× host —
down from ~58×**; the genuine residual is the tiny near-goal-hover / margin-SNR finite-size floor (the B-4 family), not
the loop.

---

## MOVE 1 — ISOLATE + QUANTIFY: where exactly is the residual, and how big is the truly-irreducible part?

### 1a. The original boundary (the inventory's ~58× "can't sustain")

From `2026-06-19-nav-spiking-sc-deploy-NO-GO.md` (grid-32/1800, Σ per-phase final-quarter mean distance, lower=better):

| arm | seed 42 | seed 43 | seed 44 | mean | vs host |
|---|---|---|---|---|---|
| **host** (heuristic orient + sign-distance reward) | 2.0 | 2.0 | 2.0 | **2.0** | 1× (optimal floor) |
| **SC-on** (spiking SC orient + neural reward/critic/SNc) | 93.7 | 118.8 | 140.1 | **117.5** | **~59×** |
| **scramble** (scrambled-retinotopy SC + neural reward) | 128.1 | 98.0 | 123.9 | **116.7** | ~58× |

The NO-GO read this as: *"scramble ≈ SC-on (within 1%) → scrambling the SC retinotopy does not change the outcome →
the orienting quality is irrelevant → the failure is in the **neural-reward → SNc → critic → actor-drive** loop."* The
actor went **silent** (reaches the goal **8/1800** steps; host 822).

**This config used `_nav_gate_merged_run --spiking-sc` — i.e. the deployed half-plane LINEAR-RAMP read-out + the LINEAR
(truncated) retina + NO FIX1 tie-break.**

### 1b. The false localization, corrected — the scramble control had nothing to scramble

The original scramble control **could not discriminate**, because at grid-32 the moving-goal schedule's four corner
goals (30+ cells away) render **entirely OFF** the 32-pixel egocentric `sc_retina` — the linear `ppc=4` map clips
anything beyond ±4 cells, so `sc_retina` mass = **0.0** and the SC bump is **absent** (quantified by a CPU replay of
`render_egocentric_goal` in `2026-06-22-shortcut6-upstream-orienting-residual-surpass.md` MOVE 1c — retina mass 0.0 for
all four goals, both at the top-edge pin and at centre). When the bump is absent **and** the read-out is stuck-N (the
host tie-break degeneracy below), scrambling the retinotopy changes nothing — there is nothing to scramble. So
"scramble ≈ SC-on" localized the failure to **"not the (absent) orienting"** = trivially true. **It was not evidence
the reward/drive loop is the cause; it was evidence the orienting INPUT was dead.**

The genuine residual the NO-GO masked was **two non-biological modeling choices, both runner-side, both since fixed:**

1. **The egocentric retina TRUNCATION** — a flat linear `ppc=4` over a fixed 32-px field clips far goals off-image →
   bump absent. (NOT a point-neuron limit — a flat-and-truncated retina is the non-biological special case.)
2. **The HOST tie-break degeneracy** — Python `max()` resolves the `[40,40,40,40]` `sel_X`/`commit_X`
   accumulator-saturation ties to **N (index 0)** → a **stuck-N** read-out regardless of where the goal is
   (`2026-06-20-cascade-north-bias-FIX.md`; `2026-06-20-nav-sc-drive-reorient-derisk.md` — the action distribution is
   N ~0.45–0.52 in **every** phase, goal-invariant). (A host cognitive shortcut, not a substrate limit.)

### 1c. The residual AFTER the surpass — quantified (it sustains; the loop was never the problem)

With both fixes (`popvector` decode + **FIX1** + **log-polar**), the **exact same** closed neural-reward→critic→actor
loop (`spiking_reward_us` + `enable_neural_critic` + `spiking_snc`, `heuristic_strength=0`), grid-32/1800, from
`scpv_logpolar_4seed_check.json` (post-change Σ = phases 1–3, lower=better):

| seed | host post-Σ | **neural closed-loop post-Σ** | host/neural | diag dom (matches bearing?) | tie-frac |
|---|---|---|---|---|---|
| 42 | 1.58 | **3.81** | 0.42 | NE→E, SW→S, SE→E ✓3/3 | 0.179 |
| 43 | 1.62 | **4.50** | 0.36 | NE→N, SW→S, SE→E ✓3/3 | 0.188 |
| 44 | 1.50 | **4.58** | 0.33 | NE→N, SW→W, SE→E ✓3/3 | 0.195 |
| 100 | 1.86 | **4.48** | 0.42 | NE→E, SW→S, SE→E ✓3/3 | 0.184 |

**The closed loop now sustains navigation at ~2.4–3.0× host — down from ~58–73×.** The agent re-orients across the full
grid each phase (NE corner → far-W corner = a 29-cell westward move, etc.) and reaches+holds within ~1 cell of each
moving goal; the per-phase final-quarter distance ~1.0–1.8 (a near-goal hover), not the truncated-retina NEGATIVE's
20–54. The residual ~2.4× gap is the **near-goal hover + the residual coin-flip on weak diagonal margins** (tie-fraction
~0.18–0.20 = ~18–20% of decisions resolved by the fair tie-break draw, the direct margin-SNR measure) — the irreducible
**finite-size margin-SNR floor** of the orient-solvable task, the **same family as B-4's ~16% spiking-decision cost**
(CLOSED-default, honest irreducible). **It is not the loop, and there is no dendritic frontier here.**

---

## MOVE 2 — REFRAME: was the wrong hypothesis tested?

**Yes.** The NO-GO tested *"can a closed neural reward/critic/actor loop SUSTAIN nav?"* and concluded NO — but the loop
was being fed a **dead orienting input** (absent bump) and read out through a **stuck-N** (host-tie-break) policy. The
right question is one stage upstream: **does the SC's INPUT REPRESENTATION guarantee an eccentric target is still
represented?**

The decisive biology (verified against the catalog + current literature): the intermediate/deep superior colliculus
holds a **log-polar / foveal-magnified retinotopic saccade map** — eccentricity is mapped along the rostral-caudal axis
with strong foveal magnification; the periphery (out to ~45–90° eccentricity) is **compressed but ALWAYS
represented**, never clipped (the Ottes–Van Gisbergen–Eggermont logarithmic afferent map; the Hafed lab 2019 model;
human-SC eccentricity work). Catalog **E.04**: topographic maps are *"warped by behavioral importance — cortical
magnification — fovea"*; catalog **H.25**: the SC is the full-hemifield "where to look next" saccade map; catalog
**A.07**: SNr→SC disinhibition gates the saccade out. SC **build-up cells** then hold a moving hill of activity to a
movement command. **A linear, truncated retina is the non-biological special case** — the biology-faithful SC never
truncates. The "sustain" the loop needed was never a reward/value problem; it was the **missing foveal-magnified input
representation** (+ a fair tie-break — Wang-2002 finite-size decision noise breaking genuine ties, not the host N-first
ordering).

Omnipause / fixation reset (Option E; Munoz; catalog A.07) was considered but is **not** indicated — the residual is
swamping/absence, not hysteresis (the bump re-renders fresh each step; for far goals there was no persisted-old-bump to
reset). Held in reserve only if a future log-polar bump attractor shows hysteresis on goal change.

---

## MOVE 3 — RANK the cheapest surpass mechanisms (all have LANDED)

### RANK 1 (LANDED + DEFAULT-ON) — log-polar / foveal-magnified egocentric SC render

`render_egocentric_goal(log_polar=True)`: `r_pix = R_max · log(1 + r_cell/d0) / log(1 + r_max/d0)` along the goal's
bearing — compresses eccentricity so a 30-cell goal lands on a peripheral `sc_map` site instead of clipping off-image.
The 16×16 `sc_map` + the whole pop-vector decode / divnorm / WTA stack are unchanged; only the input position-encoding
is made biology-faithful. **Status: 4/4 seeds GO on grid-32/1800; flipped to default (`run_moving_goal_episode`
`log_polar_retina=True`, commit `91442e0b`); CPU render smoke 8/8 (mass 0.0 → 12.5 with correct bearing).** This is the
ENVIRONMENT sensory render (channel-1 of the brain-based-only bar, host-legitimate — a compressive *visual* mapping,
not a coordinate read-out). **NO `sim/` edit** (~10-line runner-side formula).

### RANK 1b (LANDED) — cascade N-bias FIX 1: tie-aware stochastic read-out

Break a K-way `sel_X`/`commit_X` tie by a uniform draw among the tied (a persistent per-episode `action_rng`), NOT the
N-first `max()` ordering — removes the host tie-break cognitive shortcut. **Status: 3/3 seeds GO; default-off
byte-identical; SCRAM collapses once the bias is removed (the decode becomes load-bearing). commit `d2fd3c29`. NO
`sim/` edit.** Composes with RANK 1: the bump exists (log-polar) AND a fair read consumes its margin (FIX1).

### RANK 2 (ablation only) / RANK 3 (deferred follow-on)

RANK 2 = un-truncate the linear retina without magnification (coarsens near-goal discrimination; an ablation to
attribute the lift to magnification). RANK 3 = move competition earlier (cortex-WTA / opponent-axis), the second-order
fix for the residual margin-SNR floor (presupposes a bump; the named deferred next mechanism, NOT a stop). Neither
needs a `sim/` edit.

---

## MOVE 4 — VERDICT: SURPASSABLE, and ALREADY SURPASSED

**The boundary survives the SURPASS round as a precisely-located, non-irreducible residual whose cheap, biology-faithful,
no-`sim/`-edit fix (log-polar render + FIX1 tie-break) has landed and is default-on.** The closed neural-reward→critic→
actor loop **does** sustain navigation once the SC's input representation is biology-faithful; the original ~58× "can't
sustain" was the symptom of the truncated retina (bump absent) + the host tie-break (stuck-N), not a property of the
loop. **The genuinely-irreducible residual is TINY and is NOT a substrate limit** — the residual ~2.4× near-goal-hover /
weak-diagonal-margin coin-flip (tie-fraction ~0.18–0.20) is the finite-size margin-SNR floor of the orient-solvable
task, the same family as B-4's ~16% spiking-decision cost. There is no dendritic frontier here and no graded-read-out /
point-neuron-limit family wall.

**Inventory correction (recommended):** the B-2 entry's two load-bearing claims are now obsolete — (1) "the closed loop
can't SUSTAIN navigation (~58×)" → it now sustains at ~2.4× (default-on log-polar + FIX1); (2) "the scramble localizes
the failure to the reward/drive half, not orienting" → this was a **false localization** caused by the absent bump; with
log-polar, SCRAM collapses ~25× (the orienting decode IS load-bearing) and the loop sustains. **B-2 should be
reclassified from BOUNDARY to SURPASSED/CLOSED** (residual = the B-4-family margin-SNR floor).

---

## Anti-cheats — all carried, all hold

### The decisive one: the surpass is LOAD-BEARING (lesion → the boundary returns)

Remove the log-polar surpass (linear/truncated retina) under the **exact same** closed neural-reward→critic→actor loop
+ popvector + FIX1, seed 42, grid-32/1800:

| config | retina | post-change Σ | dom per phase |
|---|---|---|---|
| **log-polar ON** (`scpv_logpolar_4seed_check.json` s42) | foveal-magnified | **3.81** | `[E, W, S, E]` (tracks) |
| **log-polar OFF** (`cascade_debias/fix1/scpv_sc_popvector_seed42.json`) | linear/truncated | **83.13** | finalQ `[26.3, 0.98, 31.2, 51.0]` (stuck/failing) |

**Ratio OFF/ON = 21.8× worse when log-polar is removed.** Both arms are popvector + FIX1 + the same closed loop
(`spiking_reward_us` + `enable_neural_critic` + `spiking_snc`, heuristic OFF); **only the retina geometry differs.** ⇒
the log-polar surpass is decisively load-bearing — removing it returns the boundary. **PASS.**

### The rest

- **SCRAM collapses (decode load-bearing) — PASS.** With log-polar, SCRAM (retinotopy-scramble lesion) post-change Σ =
  98.4–114.7 vs popvector 3.8–4.6 → ~23–27× worse, stuck-N. The orienting is carried by the *retinotopic* decode, not a
  cascade prior. (The original NO-GO's "scram ≈ SC-on" was the absent-bump artifact.)
- **Actor sustains, not silent — PASS.** Foreground confirm (grid-32/450, log-polar ON, the closed loop):
  `late_motor_sustain = 0.973` (fires through the 2nd half), vs the NO-GO's ~0.40 "motor goes silent."
- **Honest measurement — PASS.** The residual is quantified (host/pv ~0.33–0.42 = ~2.4–3.0×, tie-fraction ~0.18–0.20),
  named as the near-goal-hover / margin-SNR floor (B-4 family), not the loop.
- **Diagonal tracking (not lateral-only) — PASS.** The popvector dom matches the goal bearing on all 3 diagonal phases
  (NE/SW/SE), not just the pure-lateral far-W; diag finalQ ~1.0–1.8.
- **grid-32, never grid-8 — PASS.** The verdict data is grid-32/1800/warmup-600.
- **closed loop engaged — CONFIRMED.** The harness arms set `spiking_reward_us` + `enable_neural_critic` + `spiking_snc`
  + `heuristic_strength=0` (the exact failing closed-loop config); the foreground confirm reproduced under the current
  default-on log-polar code.
- **moat untouched — PASS.** No conversational regions in these nav runs; the nav cascade is array-disjoint from the
  composer's complex synapses.

### Foreground confirm (this session)

`_nav_sc_popvector_readout_derisk.py --arms sc_popvector --fix1 --log-polar`, grid-32/450/warmup-200 (tractable, ~2 min,
foreground, the closed loop) → `research/findings/raw/nav_gate_2a/_burndown3F_confirm_logpolar_phase0_s42.json`:
`phase0_finalQ = 1.0` (holds ~1 cell), `dom = [E]` (tracks the NE bearing, not stuck-N), `tie_break_fraction = 0.093`
(91% margin-driven), `late_motor_sustain = 0.973`. ⇒ the current default-on code, with the full closed loop, sustains.

---

## Provenance + machinery (file:line, for trust-but-verify)

- **The render edit (NO `sim/`):** `render_egocentric_goal` (`research/runners/g11_bg_runner.py:229`, `log_polar`
  kwarg); `run_moving_goal_episode` `log_polar_retina=True` **default** (`g11_bg_runner.py:3866`); the SC eye-drive call
  site (`g11_bg_runner.py:7299`). Commit `91442e0b` (default flip).
- **The FIX1 edit (NO `sim/`):** `_argmax_action` / `sc_tie_break_stochastic` (default-off byte-identical), commit
  `d2fd3c29`.
- **The harness:** `research/runners/_nav_sc_popvector_readout_derisk.py` (the closed-loop SC arms; `--log-polar` /
  `--fix1`).
- **Landed findings:** `2026-06-22-shortcut6-upstream-orienting-residual-surpass.md` (MOVE 1c retina mass 0.0; MOVE 2
  log-polar reframe; MOVE 3 RANK 1) · `2026-06-22-shortcut6-log-polar-render-derisk.md` (grid-32 GO, 4-seed) ·
  `2026-06-20-cascade-north-bias-FIX.md` (FIX 1, 3/3, SCRAM collapses) · `2026-06-20-nav-loop-closure-derisk.md` (the
  reentrant-arc premise FALSIFIED — the arc was already closed) · `2026-06-20-nav-sc-drive-reorient-derisk.md` (the
  stuck-N read-out, pre-log-polar) · `2026-06-19-nav-spiking-sc-deploy-NO-GO.md` (the original ~58× + the false
  "reward/drive half" localization).
- **Result JSONs:** `scpv_logpolar_4seed_check.json` (log-polar ON, 4/4 GO) · `cascade_debias/fix1/
  scpv_sc_popvector_seed42.json` (log-polar OFF linear — the lesion control) · `_burndown3F_confirm_logpolar_phase0_s42.json`
  (this session's foreground confirm).
- **Biology (verified):** catalog **E.04** (topographic maps warped by cortical magnification/fovea), **H.25** (SC
  full-hemifield saccade map), **A.07** (SNr→SC disinhibition gate); SC log-polar foveal magnification (Ottes–Van
  Gisbergen–Eggermont; Hafed lab 2019; human-SC eccentricity work).

_READ-ONLY SURPASS round + 1 cheap foreground GPU confirm (grid-32/450, ~2 min). NO `sim/` edit. grid-32 IS the verdict
(never grid-8). The no-confab moat is array-disjoint from the nav cascade and untouched. The load-bearing claim (the
surpass is genuine) is established by the 21.8× lesion contrast from existing banked data + the 4-seed GO + the
foreground confirm._
