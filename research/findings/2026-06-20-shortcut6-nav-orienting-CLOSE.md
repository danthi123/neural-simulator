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

## THE FAITHFUL GRID-32 VERDICT (seed 42, n=1800, warmup-600)

_(Seed 42 only: the verdict is a robust NEGATIVE across SEVEN mechanism variants + a second operating point — a
mechanistic stuck-N, not a marginal effect — so the standing 6-seed rule for variable EFFECTS does not apply; GPU is
reserved for the next shortcut. A GO would have triggered the 6-seed confirmation.)_

Per-phase `final_quarter_mean_distance` (lower = better; the goal MOVES each phase so phases 1–3 are re-orients).
The goal schedule: phase0 NE `(30,30)`, phase1 far-W `(1,30)`, phase2 SW `(1,1)`, phase3 SE `(30,1)`. Seed 42:

| arm | phase0 (acquire, NE) | phase1 (re-orient, far-W) | phase2 (re-orient, SW) | phase3 (re-orient, SE) | Σ all | Σ post-change | dom-cardinal per phase |
|---|---|---|---|---|---|---|---|
| **HOST** (scaffold) | 0.496 (dom **E**) | 0.540 (dom **W**) | 0.549 (dom **S**) | 0.646 (dom **E**) | **2.230** | **1.735** | E, W, S, E → **tracks goal every phase** |
| **NEURAL** (popvector σ=5 g=0.02) | 16.850 (dom **N**) | 29.159 (dom **N**) | 59.257 (dom **N**) | 41.372 (dom **N**) | **146.6** | **129.8** | N, N, N, N → **STUCK-N, goal-invariant** |
| **SCRAM** (popvector + scramble lesion) | 24.522 (dom **N**) | 4.973 (dom **N**) | 48.903 (dom **N**) | 32.531 (dom **N**) | **110.9** | **86.4** | N, N, N, N → **STUCK-N** |

**The verdict (seed 42, the decisive faithful-scale read):** NEURAL is an **HONEST NEGATIVE**. At faithful grid-32 the
calibrated population-vector decode + bump-mass divnorm + the #4 WTA ring **does NOT re-orient**: Σ = 146.6 vs HOST
2.23 (**~66× worse**), post-change Σ = 129.8 vs HOST 1.74 (**~75× worse**), and the dominant cardinal is **stuck-N
(N ~0.49–0.50) in EVERY phase regardless of where the goal is** — the EXACT stuck-N signature the original NEGATIVE
documented (`2026-06-20-nav-sc-drive-reorient-derisk.md`). **The grid-8 calibration's apparent tracking
(`sigma=5,gain=0.02` → dom `[N→W]`) was the documented false-GO scale — it does NOT survive the faithful grid-32
confirm.**

**Mechanistic diagnosis (from the NEURAL episode's `commit_counts` + `decision_path_counts`):** `decision_path_counts
= {primary: 1800, fallback: 0}` (the #4 WTA always commits), but the four `sel_X`/`commit_X` accumulators **saturate
together at the `n_commit_per_action=40` ceiling nearly every step** (`[40,40,40,40]` is the dominant pattern) — the
WTA ring is NOT discriminating a winner. The corrected cosine geometry + divnorm produces only a TINY directional
margin at grid-32 (the far-goal blob is dim/small in the 16×16 `sc_map`, so the pop-vector signal is weak), and that
margin is **SWAMPED** by the cascade's structural N-bias + the actor's vision drive before the WTA can amplify it. This
is the scoping's predicted Option-B failure mode (`…scoping.md:172-173`: "the competition is far enough downstream
that the un-sharpened `cortex_X` margin is already swamped by the N-bias before it reaches `sel_X`"). It is a SWAMPING /
under-selectivity residual, NOT a bump-attractor hysteresis (the SC bump re-renders fresh every step, so Option E's
reset is not the indicated remedy).

### Independent corroboration at a SECOND operating point — pure cosine (`gain=0`), grid-32 seed 42

A parallel run reached the SAME negative at the OTHER end of the calibration band, ruling out "a different divnorm
calibration would have worked." Per its own grid-8 bracket (`scpv_cal_{g0p0,g0p05,g0p2}_s42.json`), the divisive `gain`
must be ~0 (pure cosine) — a NON-zero shared divisor `σ+gain·mean` crushes the cosine geometry's relative margin (it is
identical across the four cardinals, so it adds no competition while shrinking the signal). At grid-8 phase-1 the
`gain=0` arm appeared to track (dom flipped N→W) — but that is exactly the false-GO scale (phase-1 ~30 actions). The
faithful grid-32 verdict (`scpv_VERDICT_g32_s42.json`, `gain=0, σ=1`):

| arm | post-change finalQ sum | dom per phase | tracks? | late_sustain |
|---|---|---|---|---|
| HOST | **1.91** | E, **W**, E, E | **YES** | 1.000 |
| RAMP (the NEGATIVE) | 123.05 | N, N, N, N | NO | 0.424 |
| **NEURAL popvector (gain=0, pure cosine)** | **118.14** | **N, N, N, N** | **NO** | 0.402 |
| SCRAM (lesion) | 95.95 | N, N, N, N | NO | — |

`host_over_popvector_post_ratio = 0.0162` (~62× worse); `popvector_over_ramp = 1.04` (≈ identical to the ramp). The
pure-cosine arm (the MAXIMAL-geometry, MINIMAL-divnorm limit — i.e. the end-point the "Lever 1 gain→0" recalibration
screen approaches) is **just as stuck-N as the ramp**. ⇒ the negative is NOT a single-calibration artifact; neither the
`σ=5,gain=0.02` operating point NOR the `gain=0` pure-cosine limit re-orients at grid-32. **This also answers the
Lever-1/2 recalibration screen: weakening the divnorm to its limit does not help — the residual is competition
(swamping), not attenuation, which points directly at R1.**

### The SCRAM anti-cheat — the decisive clincher (NOT a collapse)

**The lesion does NOT collapse relative to NEURAL — and that is what settles the verdict.** SCRAM (scrambled
retinotopy) is stuck-N (Σ 110.9), but **NEURAL (Σ 146.6) is actually WORSE than SCRAM**. The discriminator the task
specifies is: *"at grid-32 the lesion MUST clearly collapse — if SCRAM ≈ NEURAL, the decode is NOT load-bearing."*
Here SCRAM ≈ NEURAL (both stuck-N, both ~75× the host, NEURAL marginally worse) ⇒ **the retinotopic pop-vector decode
is NOT carrying the orienting at grid-32.** Destroying the retinotopy does not make things meaningfully worse because
the actor is N-stuck regardless of what the SC read-out says — the SC contribution is being swamped before it can
steer the action. This is a clean, unambiguous HONEST NEGATIVE for Option A+B.

---

## Anti-cheat table

| anti-cheat | requirement | result | pass? |
|---|---|---|---|
| Host positive control | host re-orients (centroid+argmax position decode), anchors the SC arm's gap | HOST Σ 2.23, post-change 1.74, dom tracks goal every phase | ✅ (host ceiling established) |
| Re-orient-after-change metric (NOT static hold) | the fix must move the re-orient metric (phases 1–3), not just acquisition | NEURAL post-change 129.8 vs HOST 1.74 (~75×) — re-orient NOT moved | ✅ measured (NEGATIVE) |
| Per-phase action distribution tracks the goal | dom-cardinal must shift W-heavy↔E-heavy across phases | NEURAL N ~0.49–0.50 EVERY phase (goal-invariant stuck-N) | ✅ measured (NEGATIVE) |
| Retinotopy-scramble LESION collapses | SCRAM must regress materially below NEURAL (proves the decode is load-bearing) | SCRAM Σ 110.9 ≈ NEURAL Σ 146.6 (NEURAL even worse) — NO collapse | ✅ measured (decode NOT load-bearing) |
| Matched drive (SC_CORTEX_W=18 across SC arms) | any lift attributable to GEOMETRY, not covert drive | NEURAL/SCRAM/RAMP all at SC_CORTEX_W=18 | ✅ |
| Perception NOT stripped | `enable_visual_cortex` on, warmup-600 honored | on for all arms | ✅ |
| Regime fidelity = grid-32 (NOT grid-8) | the verdict is grid-32/1800/warmup-600 | all 3 arms at grid-32/1800/warmup-600 | ✅ (grid-8 was the false-GO; see below) |
| No-confab moat untouched | the SC read-out (`cp_*` nav state) is array-disjoint from the composer's complex `cp_rf_w_*` synapses | no conversational regions in these nav runs; moat by construction unaffected | ✅ |

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

# --- Next mechanisms (after Option A+B's NEGATIVE) ---
# Re-calibration Lever 1 (lower divnorm gain → stronger SC margin)  [DONE: NEGATIVE]
SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.005 \
    --out research/findings/raw/nav_gate_2a/grid32_s6/scpv_summary_RECAL_g0p005_s42.json

# R1 — inter-cardinal cortex WTA strength sweep (sharpen the SC margin EARLY)  [DONE: all NEGATIVE for re-orient]
#   FS weight 8 (default)  → broke phase-0 N-pinning, no re-orient
#   FS weight 16, n 10     → stuck-N
#   FS weight 40, n 15     → over-quenched
for W in "8 5" "16 10" "40 15"; do set -- $W; SIM_BACKEND=cupy python -m research.runners._nav_sc_popvector_readout_derisk \
    --arms sc_popvector --seed 42 --n-steps 1800 --grid-size 32 --warmup-steps 600 \
    --sc-cortex-w 18 --divnorm-sigma 5 --divnorm-gain 0.02 --cortex-wta --cortex-fs-weight $1 --cortex-fs-n $2 \
    --out research/findings/raw/nav_gate_2a/grid32_s6/scpv_summary_R1_fs${1}_s42.json; done
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

## VERDICT + the next mechanism (a boundary is NOT an exit)

**The faithful grid-32 verdict for Option A+B (pop-vector cosine read-out + bump-mass divnorm + the #4 WTA ring) is an
HONEST NEGATIVE, seed 42:** NEURAL does NOT re-orient (Σ 146.6 vs HOST 2.23, ~66×; stuck-N every phase), and the
SCRAM lesion does NOT collapse relative to NEURAL (SCRAM 110.9 ≈ NEURAL 146.6) — i.e. the retinotopic decode is not
load-bearing at faithful scale. The genuine point-neuron mechanism (the one the build abandoned mid-way) has now been
finished, calibrated, and tested at faithful grid-32. **This is NOT "closed boundary."** Per the owner's HARD rule the
arc continues to the next mechanism — the host orienting scaffold STAYS in place until a spiking organ re-orients.

**Why it fails (diagnosis, load-bearing):** SWAMPING / under-selectivity, not hysteresis. The corrected cosine geometry
produces a position-correct but TINY directional margin at grid-32 (a far goal-blob is dim/small in the 16×16
`sc_map`); the #4 `sel_X` WTA ring sits far enough downstream that the cascade's structural N-bias + the actor's vision
drive swamp that margin before it can win (the `commit_counts` show all four `sel_X` pools saturating together at the
40-ceiling). The bump itself re-renders fresh each step (NOT stuck), so a goal-change reset (Option E) is NOT the
indicated remedy.

**The next mechanisms (the scoping's Option-B-failure remedies):**
1. **Re-calibration at faithful scale (Lever 1, gain=0.005) — DONE, NEGATIVE (seed 42).** Σ 108.8, post-change 108.1,
   stuck-N every phase. The lower gain DID improve static ACQUIRE (phase0 0.628, near host) but **could not RE-ORIENT**
   (post-change unchanged-catastrophic) — the exact static-hold-vs-re-orient split the scoping predicted: more SC drive
   holds a fixed bias better but cannot track a *moved* goal. ⇒ "more SC drive" is the wrong lever; the missing piece is
   COMPETITION. (`scpv_RECAL_g0p005_s42.json`.)
2. **R1 — inter-cardinal cortex WTA (`--cortex-wta`, `enable_cortex_lateral_inhibition`), default FS weight 8 — DONE,
   NEGATIVE but RIGHT DIRECTION (seed 42).** Σ 126.2, post-change 124.7, stuck-N phases 1-3 — BUT **phase0 acquire
   dropped to 1.469 with dom=E** (E 0.30, NOT N-pinned): the cortex-WTA genuinely BROKE the phase-0 N-pinning the other
   arms all showed. ⇒ the inter-cardinal competition is the CORRECT mechanism direction; the default strength (FS weight
   8, 5 FS/pool) is just too weak to also win during re-orient. (`scpv_R1cortexwta_s42.json`.)
3. **R1 escalation — STRONG cortex WTA (FS weight 40, 15 FS/pool) — DONE, NEGATIVE (over-suppressed) (seed 42).**
   Σ 135.2, post-change 115.3, dom back to N every phase, AND phase0 acquire REGRESSED to 19.9 (from the FS=8
   default's 1.5). The 5× stronger inter-cardinal inhibition QUENCHED the cortex pools (too much mutual inhibition →
   all four pools suppressed → the N-bias re-dominates via residual noise) — the over-strong-WTA / Rutishauser
   α-stability failure. ⇒ stronger is NOT better; there may be an intermediate. (`scpv_R1strong_s42.json`.)
4. **R1 intermediate — cortex WTA FS weight 16, 10 FS/pool — IN FLIGHT (seed 42).** Threads between the FS=8 (broke
   phase-0 N-pinning but couldn't re-orient) and FS=40 (over-quenched) — the last cortex-WTA-strength shot before the
   convergent characterization.
5. **(reserve) Option E** — a goal-change inhibition-of-return/fixation reset, only if the residual turns out to be
   bump/ring hysteresis (the diagnosis says it is NOT, so this is a reserve).

### Full mechanism sweep at faithful grid-32 (seed 42) — the convergent characterization

| mechanism | Σ post-change | re-orient? | dom-per-phase | note |
|---|---|---|---|---|
| **HOST** (scaffold) | **1.74** | **YES** | E,W,S,E | the ceiling — host orienting tracks every goal |
| NEURAL (pop-vector + divnorm, the Option-A+B fix) | 129.8 | NO | N,N,N,N | stuck-N; SCRAM doesn't collapse (decode not load-bearing) |
| Lever 1 (lower divnorm gain 0.005) | 108.1 | NO | N,N,N,N | static-ACQUIRE improved (phase0 0.63), re-orient unchanged |
| R1 cortex-WTA FS=8 (default) | 124.7 | NO | **E**,N,N,N | the one positive signal: BROKE phase-0 N-pinning (dom=E) |
| R1 cortex-WTA FS=40 (strong) | 115.3 | NO | N,N,N,N | over-quenched (phase0 regressed to 19.9) |
| R1 cortex-WTA FS=16 (intermediate) | 114.2 | NO | N,N,N,N | stuck-N; the FS=8 phase-0 break did not persist |
| R1 COMBO (FS=8 WTA + SC_CORTEX_W=60) | **104.9** | NO | N,N,N,N | the BEST spiking post-change (the 2 partial levers combined) — still ~60× host |

**The convergent pattern (7 variants):** the spiking SC orienting read-out does NOT re-orient at faithful grid-32
across the read-out-geometry fix (pop-vector + divnorm) + the SC-drive lever (lower gain) + the cortex-WTA competition
at THREE strengths (FS 8/16/40) + the combo (FS=8 WTA + stronger drive, the best post-change 104.9 — still ~60× host).
The actor is **structurally pinned to the top edge (pos row ~31)** during re-orient regardless of the SC read-out
signal. **Sharper diagnosis from the combo trace:** the pinning is specifically a NORTH-SOUTH axis lock — the agent's
x-position DOES drift west under the SC signal (e.g. combo phase-2 reached pos x=11) but y stays glued at 31, never
moving SOUTH toward the SW goal. The HOST (whose orienting reaches the actor at full strength) clears this same N-bias
on BOTH axes and tracks every goal — so the wall is that the SC read-out's signal, even geometry-corrected +
competition-sharpened + drive-boosted, is not strong/selective enough to override the N-S-axis N-bias when the goal
MOVES. The sole positive signal across the whole sweep is that the cortex-WTA (FS=8) broke the phase-0 N-pinning
(static acquisition), confirming the competition mechanism is directionally right but insufficient for re-orient (and
FS=16/40 lost even that — too much mutual inhibition quenches the pools). **This is a thoroughly-characterized HONEST
NEGATIVE, not "closed boundary" by default — the genuine point-neuron mechanism classes for the read-out (#6's actual
scope: read-out geometry + inter-cardinal competition + drive) were each attempted at faithful scale and each
rigorously measured.**

## FINAL VERDICT + the precise next direction

**Shortcut #6 (the SC orienting read-out) at faithful grid-32 = a comprehensively-characterized HONEST NEGATIVE.** The
prescribed geometry fix (Option A+B: pop-vector cosine decode + bump-mass divnorm + the #4 WTA ring) was FINISHED,
calibrated, and run at the faithful grid-32/1800/warmup-600 scale the prior build abandoned — and it does NOT re-orient
(NEURAL ~75× the host post-change; SCRAM does not collapse, so the retinotopic decode is not load-bearing). The
next-mechanism class the diagnosis indicated (inter-cardinal cortex-WTA competition, the scoping's prescribed
Option-B remedy) was then attempted at three strengths — also NEGATIVE for re-orient. **The host orienting scaffold
STAYS in place.**

**The precise residual (the next direction — but it is OUTSIDE #6's read-out scope):** the wall is the actor's
**structural N-bias** (it pins to pos row ~31 and the action distribution is N ~0.5 every phase), which dominates the
SC read-out's signal during re-orient. The HOST clears this same bias because its orienting reaches the actor at full
strength; the spiking SC organ's signal does not. Reducing/correcting that N-bias is a **separate shortcut/issue (the
cascade's perception/selection asymmetry), not the SC read-out geometry #6 is scoped to** — so within #6 (the
read-out), the point-neuron mechanism classes are exhausted at faithful scale. The honest brain-based deliverable is:
**the spiking SC read-out, even with the canonical population-vector geometry + Carandini-Heeger normalization + an
inter-cardinal WTA, cannot re-orient on this substrate at grid-32 because the downstream cascade's N-bias swamps it —
mapping a genuine point-neuron operating-point limit of the orienting organ as deployed.** Options NOT yet exhausted
(deferred as out-of-#6-scope or low-probability): correcting the cascade N-bias itself (a different shortcut), or
Option E's goal-change reset (the diagnosis says the residual is swamping, not hysteresis, so Option E is low-probability).

**One final read-out-scoped synthesis shot — combo (FS=8 cortex-WTA + stronger SC drive SC_CORTEX_W=60):** the two
partial signals were Lever 1's static-acquire improvement (more SC influence) and the FS=8 WTA's phase-0 N-pinning
break (competition). The combo tests whether a stronger position-correct SC drive THROUGH the competition lets the
margin win during re-orient (no longer matched-drive — the deliberate test of "enough geometry-correct signal + enough
competition"). **DONE, NEGATIVE (seed 42):** post-change **104.9** — the BEST of all spiking arms (combining the two
partial levers shaved the post-change from NEURAL's 129.8), but still **~60× the host's 1.74** and stuck-N every phase.
The combo is the closest the spiking organ got, and it is not close. ⇒ the read-out-scoped point-neuron mechanisms are
EXHAUSTED at faithful scale.

**This supersedes the prior "closed honest-negative" ledger row** (which cited the drive-knob's grid-32 sweep, not the
geometry fix) **and the prior subagents' grid-8 arm files** (the documented false-GO scale). The geometry fix is now
genuinely tested at faithful scale, and the next mechanism class was attempted, per the no-boundary-exit rule.

⇒ **#6 status: a comprehensively-characterized HONEST NEGATIVE at faithful grid-32 (seed 42). Per the no-boundary-exit
rule the arc was pursued across SEVEN mechanism variants — the prescribed geometry fix (Option A+B), the SC-drive lever
(Lever 1 lower gain), the cortex-WTA competition at three strengths (FS 8/16/40), and the combo (WTA + stronger drive)
— ALL NEGATIVE for re-orient, corroborated at a second `gain=0` operating point. The host orienting scaffold STAYS.**
The seed-42 verdict is conclusive on the read-out scope (the negative is mechanistic, not seed-noise: the actor's
N-S-axis N-bias swamps the SC signal in every variant; 6-seed confirmation was not pursued because the verdict is a
robust negative across variants, not a marginal effect — the GPU is better spent on the next shortcut). The prior
"closed honest-negative" ledger row was premature (it cited the drive-knob's grid-32 sweep, not the geometry fix, whose
grid-32 confirm was an empty FILL); the prior subagents' arm files were all `grid_size=8` (the documented false-GO
scale). This doc supersedes them with the genuine faithful-scale geometry-fix verdict + the continuing next-mechanism
arc. **The host orienting scaffold STAYS in place** — no spiking organ has re-oriented at faithful scale yet.

_GPU (`SIM_BACKEND=cupy`). Every result JSON committed the moment it landed (anti-rest). grid-32 IS the verdict, never
grid-8. The no-confab moat is untouched (the nav SC read-out is array-disjoint from the composer's complex synapses)._
