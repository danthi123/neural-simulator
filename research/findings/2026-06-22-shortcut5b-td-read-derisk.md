# Shortcut #5b RANK-1 — the TD-difference dopamine read de-risk (2026-06-22)

**Type:** probe-only de-risk (NO `sim/` edit). GPU (`SIM_BACKEND=cupy`), 3 seeds (42/43/44), the full
control battery. Governing scoping: `research/findings/2026-06-22-shortcut5b-value-read-cleanup-scoping.md`
(RANK 1). Probe: `research/runners/_n5_grid_frontend_onbridge_probe.py` (`--td-read` opt-in, the raw read
stays the default → one-process A/B).

## VERDICT — HONEST FALLBACK (not a forced GO)

**The TD-difference δ read does NOT reliably separate learned value from structural place-code magnitude on
the point-neuron substrate.** It collapses the make-or-break magnitude-matched `shuffle_v` on only **1 of 3
seeds**. This is precisely the scoping's MOVE-4 fallback: the structural place-code magnitude survives the
weight-shuffle (it lives upstream in the grid code's per-location drive density, not in the shuffled
place→value weights) and dominates the graded-V read, so the TD difference — which reads the graded-V —
survives whenever the post-shuffle structural residual is large.

**Disposition (per the scoping):** the host-Gaussian `vs_place_context` scaffold **still retires on R1
grounds** (the grid front end's afferent selectivity + the genuinely-learned near/far value are closed 3/3 —
see below: the TD δ HOLDS 3/3 on the genuine learned `grid` arm, and the value-train DOES learn a real ratio
`w_n/f` 1.27→2.59). The residual is now **precisely characterized**: it is the **value READ operator's
structural/learned inseparability**, a dendritic-frontier item (separate apical/basal compartments for
structural drive vs learned value), NOT a place-code or value-train defect, and NOT a blocker for retiring
the host-Gaussian. Do NOT flip the grid front end's δ-read to production-default-on the TD form; keep the
grid front end as the production place code and document the value-read contamination as the dendritic
frontier.

## The fix tested (RANK 1, probe-only)

The raw read in `_read_graded_v_delta` is the `no_bootstrap` form `δ ≈ r − V(s)` evaluated separately at NEAR
and FAR (`gabab_gap = snc_unpred(FAR) > 1.30 × snc_pred(NEAR)`) — it reads TOTAL afferent magnitude
(structural + learned). The fix is the biologically-correct **temporal-difference** read across the FAR→NEAR
transition (FAR=`s`, NEAR=`s′`):

```
δ_TD = r + γ·V(near) − V(far)          (γ = td_value_critic.GAMMA = 0.95)
```

reusing the SAME per-location reads (`_read_graded_v_near_far`, the per-location SNc burst). Three faithful V
estimators (all reuse-by-import, NO `sim/` edit):
- **td1 graded-V** — V(loc) = the graded-plateau conductance read (the critic's learned value estimate). *The
  primary form.*
- **td2 snc-burst** — V(near) = burst(FAR) − burst(NEAR); V(far) = 0; r = burst(FAR). The bootstrapped
  difference of the burst-derived RPE.
- **td3 adjacent** — V at NEAR vs an ADJACENT near-neighbour MID location (the cleanest structural-baseline-
  cancellation test, since the structural baseline is ~common across adjacent states).

Run config: `--all-arms --with-shuffle-v --readout-only --multi-goal --deterministic-read --td-read` (the
validated determinism-close that pins the per-location reads seed-stable).

## The 3-seed δ table — genuine learned value (the `grid` TEST arm)

The TD δ HOLDS 3/3 on the genuine learned value (td1 graded-V, the primary form):

| seed | `grid` learns? `w_n/f` | V_n/f | **RAW** `delta_snc`/gap | **TD** td1 ratio/gap | td2 | td3 |
|---|---|---|---|---|---|---|
| 42 | 1.27 | 2.43 | 1.75e7 / True | **1.66 / True** | 1.95 / True | 1.38 / True |
| 43 | 1.58 | 7.63 | 2.75e7 / True | **4.12 / True** | 1.89 / True | 4.42 / True |
| 44 | 2.59 | 3.04 | 0.0 / False¹ | **1.94 / True** | 0.00 / False¹ | 0.97 / False¹ |

¹ seed 44 is the documented SNc over-clamp seed (critic@near 276 Hz → the somatic burst saturates,
`burst_far=0`), so the **burst-based** reads (RAW `delta_snc`, td2, td3) over-clamp to 0/False on seed 44 even
on grid. The graded-V form **td1 is the over-clamp-robust read** and HOLDS 3/3. (This over-clamp is a known,
separately-characterized seed-44 issue, not a TD-read defect.)

**grid td1 HOLD = 3/3** (1.66 / 4.12 / 1.94). The TD δ reads the genuine learned value on all three seeds.

## The control battery — the make-or-break `shuffle_v`, side-by-side vs the RAW read

`shuffle_v` = the magnitude-matched metric-lesion: grid + graded + value-train, then PERMUTE the learned
place→value V across place neurons at the freeze (destroys the learned near/far *difference*, `w_n/f` → ~1.0)
while leaving the structural magnitude matched. **The raw read PASSES this (the bug); the TD read must FAIL
it for a GO.**

| seed | post-shuffle `w_n/f` | post-shuffle V_n/f | **RAW** `gabab_gap` | **TD** td1 ratio / gap |
|---|---|---|---|---|
| 42 | 0.95 | **1.43** (small residual) | **True (SURVIVES)** | 1.18 / **False (COLLAPSES)** ✓ |
| 43 | 0.76 | **4.18** (large residual) | **True (SURVIVES)** | 2.48 / **True (SURVIVES)** ✗ |
| 44 | 1.08 | **3.26** (large residual) | False² | 2.05 / **True (SURVIVES)** ✗ |

² under `--deterministic-read` the RAW read's shuffle_v survival is itself seed-variable (2/3 True) — seed 44
over-clamps the somatic burst. (The scoping's "raw survives 3/3" cited the non-deterministic
`synscale_shufflev_magmatched` runs, which give True 3/3.)

**TD td1 collapses shuffle_v = 1/3** (only seed 42). The discriminating quantity is the **post-shuffle
V_n/f**: when it is small (seed 42, 1.43) the TD δ collapses; when it is large (seeds 43/44, 4.18/3.26) the TD
δ survives — because the **structural place-code magnitude asymmetry survives the weight-shuffle** and
dominates the graded-V read. The shuffle permutes the place→value *weights*, but the structural drive density
lives in the grid code's per-location activations, upstream of those weights, so V(near)≫V(far) persists. The
TD difference of two structurally-asymmetric V's stays positive. **The forms also disagree on the large-
residual seeds** (seed 44: td1=2.05 survives, td3=0.97 collapses because V_mid happened to exceed V_near) —
further evidence the separation is not robust.

**The clean parts of the battery (TD td1, all 3 seeds):**

| arm | what it tests | TD td1 across seeds | required | result |
|---|---|---|---|---|
| `grid` | genuine learned value HOLDS | 1.66 / 4.12 / 1.94 | HOLD 3/3 | ✓ HOLD 3/3 |
| `no_learn` | no value-train → no V | 0.00 / 0.00 / 0.00 | COLLAPSE 3/3 | ✓ COLLAPSE 3/3 |
| `lesion` | no graded read-out → no V | 0.00 / 0.00 / 0.00 | COLLAPSE 3/3 | ✓ COLLAPSE 3/3 |
| `render` | R1-limit egocentric (flat) | 0.99 / 0.96 / 0.84 | COLLAPSE 3/3 | ✓ COLLAPSE 3/3 |
| `shuffle_v` | **learned destroyed, mag-matched** | 1.18 / 2.48 / 2.05 | **COLLAPSE 3/3** | ✗ COLLAPSE **1/3** |

`no_learn` / `lesion` / `render` all collapse cleanly 3/3 (the TD δ correctly requires a learned, read-out-
able value). It is **specifically the `shuffle_v` discriminator** — the one that holds the structural
magnitude fixed while destroying the learned gradient — that the TD difference fails to collapse robustly.
(`scramble` legitimately HOLDS on the TD δ because it builds a *different* place code that the value-train
then genuinely learns on, `LEARNS-V=True w_n/f` 1.69/2.18 on seeds 42/43 — it is not a required-collapse
control.)

## Algorithmic positive control (free, pure-array math)

`sim/td_value_critic.run_pavlovian` — the TD form vs the `no_bootstrap` form (= the raw read's form), 3 seeds:

| mode | vrmse-vs-V* | scale-free transfer |
|---|---|---|
| `td` (`r + γV(s′) − V(s)`) | **0.003** | **0.997** |
| `no_bootstrap` (`r − V`, the raw read) | 182.3 | 0.203 |
| `permuted` | 0.194 | 0.072 |

This confirms — in pure array math, isolated from the substrate — that the TD form learns/reads the value
where the raw read does not. The TD principle is correct **in the abstract**; the on-bridge fallback is a
substrate property: the structural and learned magnitude are inseparable at the graded-V input, so the TD
difference inherits the contamination through V(near), V(far).

## Why the fallback (the precise residual)

The scoping (MOVE 4) anticipated this exactly: *on a point-neuron substrate where the value-train learns the
near/far V by amplifying the place code's intrinsic structural magnitude differences, a value read cannot
fully separate the learned increment from the structural baseline, because they are the same physical
quantity (afferent drive magnitude) differing only in how it was set.* The TD difference cancels a baseline
that is **consistent across the two states** — but near and far are the EXTREMES, and the structural baseline
differs strongly between them (near has more drive). The adjacent-state form (td3) was the attempt to make
the two states close enough for the baseline to cancel; it too is dominated by the structural residual on the
large-residual seeds (and noisily inverts on seed 44).

This is the **dendritic frontier**: a two-compartment neuron (apical = structural place drive, basal =
learned value) could route the structural magnitude away from the learned-value read-out, which a point
neuron cannot. It is a deferred deep-frontier item, NOT a blocker for retiring the host-Gaussian on R1
grounds.

## Disposition / next

- **#5b is closeable on R1 grounds** (grid afferent selectivity + genuinely-learned near/far value, both 3/3)
  → the host-Gaussian `vs_place_context` retires; the grid front end becomes the production place code.
- **The value-read residual is the dendritic frontier** (the value-read's structural contamination), precisely
  characterized here. Do NOT promote the TD δ read to a production-default discriminator (1/3 shuffle_v
  collapse is not a clean separation).
- The grid front end's δ-read remains opt-in; the `--td-read` flag is retained in the probe as the
  characterization tool.

### Flip-path note (for the controller, IF the host-Gaussian is retired on R1 grounds)

The grid front end is selected via the `nav_critic_grid_frontend` flag (the production wiring that re-points
`place_sensors` at the grid code); the value READ stays the existing graded-plateau / GABA_B path. The TD δ
read is NOT part of the production flip (it is a characterization read only) — the R1 close does not depend on
it. The controller should wire `nav_critic_grid_frontend` + run the nav gates with the existing read; the
value-read structural contamination is carried as the documented dendritic-frontier residual.

## Moat + sim/-edit confirmation

- **Moat:** nav-only probe (regions: `place_sensors` / `striosome_value` / `snc` / BG cascade — 43 regions,
  1280 neurons; NO conversational/composer regions, NO `cp_rf_w_*` complex synapses). The place/critic/SNc
  state is array-disjoint from the composer's complex synapses. The no-confab moat is preserved by
  construction and is untouched (no read-mechanism change can affect it; verified the probe has zero
  composer/rf references).
- **`sim/` edit: NONE.** The TD δ is computed entirely in the probe from two existing per-location reads,
  reusing `sim/td_value_critic.GAMMA` by import. No protected code touched. The raw read stays the probe
  default (`--td-read` is opt-in).

## Reproduce

```bash
SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe \
    --seed {42,43,44} --all-arms --with-shuffle-v --readout-only --multi-goal \
    --deterministic-read --td-read \
    --out research/findings/raw/_n5_td_read_allarms_seed{N}.json
# algorithmic positive control:
python -c "from sim.td_value_critic import run_pavlovian; print([run_pavlovian(m,42) for m in ('td','no_bootstrap')])"
```

Raw JSONs: `research/findings/raw/_n5_td_read_allarms_seed{42,43,44}.json`.
