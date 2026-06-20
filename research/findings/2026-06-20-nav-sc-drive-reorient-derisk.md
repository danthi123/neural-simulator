# Nav SC->cortex drive-strength de-risk — strengthening the orienting drive does NOT restore re-orient-after-goal-change; HONEST NEGATIVE, operating-point FLOOR (2026-06-20)

**Type:** ONE informed cheap-first de-risk of the localized cause from the prior nav loop-closure de-risk (GPU;
the numpy path is bug-blocked for this neural-critic config). NO `sim/` edit (the swept knob `SC_CORTEX_W` is an
existing env var, `g11_bg_runner.py:4433`).
**Pre-registered by:** `research/findings/2026-06-20-nav-loop-closure-derisk.md` (commits `cdb2603d`/`bcb45d38`/`a45629e2`).
**Owner standard:** BRAIN-BASED-ONLY. The deliverable is the honest verdict — a spiking organ that underperforms
the host scaffold IS the deliverable (it maps what the substrate can/can't do on its own).

---

## TL;DR verdict — HONEST NEGATIVE: the gap is an operating-point FLOOR, not a drive-strength deficit

The prior de-risk localized the spiking-SC NO-GO (~58x host, actor partly silent at faithful grid-32) to ONE
load-bearing gap: *the SC orienting drive `sc_map -> cortex_X` (weight `SC_CORTEX_W`, default 18) is too weak to
replace the ~150 pA host Manhattan heuristic as the actor's drive* — the agent navigates EARLY goals fine but cannot
reliably RE-ORIENT after a goal change. The single informed shot is therefore: **strengthen that drive and test
whether re-orient recovers toward the host control.**

**It does NOT.** Sweeping `SC_CORTEX_W` over {18 (current), 60 (mid), 150 (≈ the host's pA equivalent)} — the env
knob, no `sim/` edit — **does not make the spiking-SC arm's post-goal-change re-orient approach the host control**.
At every level the SC arm is ~3–4x the host gate, the post-change re-orient finalQ **saturates** (it stops improving
between the mid and strong levels), and at the strong drive the SC->cortex pooling drives **all four cortical pools
near-UNIFORMLY** rather than sharpening the orienting bias — the actor's action distribution degenerates toward
chance instead of committing harder to the new cardinal.

**⇒ The WHICH classification is the operating-point FLOOR — specifically a non-goal-tracking / under-selective
read-out — NOT a drive deficit and NOT re-targeting-by-lockon.** The faithful grid-32 action distribution is the
clincher: the host re-targets every phase (W-heavy for the far-west goal, E-heavy for the SE goal), while EVERY SC
arm goes **N ~0.45–0.52 in every phase regardless of the goal's location** (it pins itself to the top edge). It does
not point at the *previous* goal's cardinal (that would shift per phase = re-targeting-by-lockon); it points the
**same** way every phase = a `sc_map -> cortex_X` read-out whose output does not track the bump's retinal position at
all. Raising `SC_CORTEX_W` 18→150 changes only the *static-hold* finalQ (sc_w150 phase0 1.54 — a stronger stable bump
holds a single goal) and leaves every re-orient catastrophic (~26–53 vs host 0.5). The drive magnitude is not the
free parameter that closes the gap; the SC->cortex **read-out's selectivity / goal-position-tracking** (the
quadrant-pooling geometry / a sharper WTA / divisive normalization) is.

**This closes #6 (SC orienting) as a CHARACTERIZED honest-negative:** the host Manhattan heuristic stays the
documented scaffold; the spiking superior colliculus is validated for **early-goal orienting only** (it acquires the
first goal — just ~3–4x slower-to-asymptote than the host — but cannot robustly re-orient after a goal change, and
raising its drive does not fix that). The real residual is a **read-out-selectivity** problem (sharpen the
`sc_map -> cortex_X` pooling / add a competitive WTA at the cortical read-out), NOT a drive-magnitude or a
loop-stability or a dendritic-credit-assignment problem. Per the brief, this is a single-hypothesis test, and it is
NOT escalated into a multi-knob search.

---

## What was tested

**Probe:** `research/runners/_nav_sc_drive_reorient_derisk.py` (GPU; `SIM_BACKEND=cupy`). It runs the EXACT failing
`--spiking-sc` kwargs (`enable_spiking_sc` + `enable_spiking_sc_approach` + `spiking_reward_us` + `enable_neural_critic`
+ `spiking_snc` + `heuristic_strength=0`, at the merged het-off SC op-point env values 160/12/3500/40), and per arm:

- **host** — the host-heuristic POSITIVE control (heuristic orienting + host reward, NO spiking SC) — it re-orients.
- **sc_w{18,60,150}** — the spiking-SC arm at `sc_map -> cortex_X` drive = the swept level.

reading, per arm: the per-phase `final_quarter_mean_distance` (the re-orient quality after each goal change — the
4-phase schedule's phase 0 is the initial acquisition, phases 1..3 are post-goal-change re-orients), the gate
(sum of per-phase finalQ, lower=better), and the motor `late_sustain` (does a stronger drive keep the actor firing
through the re-orient, vs the partial-silence the NO-GO showed).

**Anti-cheats used (all four from the brief):**
- **host-heuristic POSITIVE control** anchors the SC-arm degradation (it re-orients cleanly: post-change finalQ ~0.5).
- **the drive SWEEP itself** (is post-change finalQ monotone-toward-host, or saturating/regressing?) distinguishes a
  drive-gap from an operating-point floor — it saturates, which is the floor signature.
- **the per-goal-phase split** (phase 0 early-acquisition vs phases 1..3 post-change) — the localized symptom IS the
  split (early acquires, post-change does not recover).
- **perception NOT stripped** (`enable_visual_cortex` on, warmup honored) — the actor still has its vision drive.

---

## Grid-8/480 smoke (seed 42) — the early read (committed `e7ca4655`)

The cheap grid-8 smoke (seconds, 2 goal phases complete in 480 steps) already shows the saturation + the
de-sharpening:

| arm | phase0 finalQ (acquire) | post-change finalQ (re-orient) | gate (Σ finalQ) | late_sustain |
|---|---|---|---|---|
| **host** (positive control) | **0.496** | **0.500** | **0.996** | 1.000 |
| sc_w18 (current default) | 1.292 | 2.125 | 3.417 | 0.975 |
| sc_w60 (mid) | 1.726 | 1.750 | 3.476 | 0.842 |
| sc_w150 (strong ≈ host pA) | 1.611 | 1.750 | 3.361 | 1.000 |

- The SC arm is **~3.4–3.5x the host gate at EVERY drive level**, even at this easy scale.
- Raising drive 18→150 gives a marginal post-change gain (2.125→1.750) but **saturates** — w60 and w150 are
  identical on post-change (1.750), and `host_over_best_sc_ratio = 0.29` (host is 3.5x better than the best SC arm).
  The gate is **non-monotone** (w60 0.476 worse than w18; w150 marginally best). No level approaches host.
- **The de-sharpening is visible in the action distribution.** Phase-0 action counts (N,E,S,W) by arm:
  host `[58,172,53,167]` (a clean E/W-dominant bias to reach the NE corner) vs sc_w150 `[121,117,105,107]`
  — **near-UNIFORM**. A stronger SC drive does not sharpen the orienting bias; it pushes all four pools past
  threshold roughly equally. (sc_w18 `[150,115,105,80]` is *more* biased than sc_w150 — the stronger drive is
  *less* selective, the floor signature.)

Grid-8 is a weak read (2 phases, easy). The faithful confirm is grid-32.

---

## Grid-32/1800 (seed 42, warmup 600) — the faithful confirm (COMPLETE)

The faithful-scale sweep (the NO-GO's grid + warmup; all 4 goal phases complete; standalone — the prior de-risk
established the structural facts are scale-independent and the grid-32/900 already reproduced the partial-silence):

| arm | w | phase0 finalQ (acquire) | post-change finalQ (re-orient, phases 1/2/3) | Σ post-change | gate (Σ all) | late_sustain |
|---|---|---|---|---|---|---|
| **host** (positive control) | — | **0.690** | **0.50 / 0.50 / 0.50** | **1.50** | **2.19** | **1.000** |
| sc_w18 (current default = the NO-GO) | 18 | 20.86 | 15.42 / 58.42 / 42.25 | 116.1 | 136.95 | 0.438 |
| sc_w60 (mid) | 60 | 12.35 | 30.22 / 56.22 / 44.19 | 130.6 | 142.99 | 0.398 |
| sc_w150 (strong ≈ host pA) | 150 | 1.54 | 26.32 / 52.96 / 31.60 | 110.9 | 112.42 | 0.426 |

**Reproduces the NO-GO and refutes the drive-strength hypothesis at faithful scale:**

- **The host re-orients cleanly through ALL THREE post-change goals** (post-change finalQ ~0.5 each, gate 2.19,
  late_sustain 1.000). Every SC arm is **~50–65x the host gate**; on the post-change re-orient specifically,
  `host_over_best_sc_ratio = 0.0136` — **the host is ~73x better** than the *best* SC arm. The spiking SC simply
  does not re-orient at faithful scale, at any drive level.
- **The sweep is NON-MONOTONE — a stronger drive does NOT close the gap, and the mid level makes it WORSE.**
  Σ post-change goes 116.1 (w18) → **130.6 (w60, worse)** → 110.9 (w150) — it never approaches the host's 1.5, and
  the w150 marginal edge over w18 is ~5% (vs the ~77x gap to host). The summary's `improves_with_drive=True` is a
  misleading first-vs-last artifact (150 < 18 by a hair); the actual trajectory 116→131→111 is the classic
  saturation/floor shape, not a monotone approach.
- **The partial-silence (`late_sustain ~0.40`) does NOT recover with stronger drive** (0.438 → 0.398 → 0.426) — the
  actor stays partly silent regardless, confirming this is not a "the drive wasn't strong enough to keep the actor
  firing" problem.
- **The strong drive helps ONLY static acquisition, never re-orient.** sc_w150's phase0 drops to **1.54** (a stronger
  stable bump *does* help hold a single static goal, ~2x the host's 0.69 — the spiking SC's validated early-goal
  orienting), but every post-change phase stays catastrophic (~26–53). The drive magnitude moves the static-hold
  metric and leaves the re-orient metric broken — exactly an operating-point floor on the read-out, not a drive gap.

### The decisive classification evidence — a non-goal-tracking (stuck-N) read-out, NOT re-targeting-by-lockon

The per-phase action distribution (fraction N/E/S/W) settles the WHICH question unambiguously:

| arm | phase0 (goal NE) | phase1 (goal far-W) | phase2 (goal SW) | phase3 (goal SE) |
|---|---|---|---|---|
| **host** | E .44 / W .37 | **W .49** / E .42 | E .41 / W .41 | **E .53** / W .46 |
| sc_w18 | **N .52** | **N .45** | **N .49** | **N .49** |
| sc_w60 | **N .44** | **N .51** | **N .47** | **N .49** |
| sc_w150 | **N .33** | **N .47** | **N .52** | **N .52** |

- The **host's** action distribution TRACKS the goal — W-heavy when the goal is far-west, E-heavy when it's the SE
  corner. It re-targets every phase.
- **Every SC arm is N-dominated (~0.45–0.52) in EVERY phase, irrespective of where the goal is** (the agent pins
  itself to the top edge, pos row 31). It does not point at the *previous* goal's cardinal (that would shift per
  phase = re-targeting-by-lockon); it points the **same** way (N) regardless of goal = a read-out that does **not
  track the goal's retinal position at all**. (grid-8 read this as "near-uniform"; grid-32 sharpens it to "stuck-N"
  — the same root cause: the `sc_map -> cortex_X` directional selectivity does not respond to the bump's location or
  to drive magnitude.)

⇒ **WHICH gap = operating-point FLOOR (a non-goal-tracking / under-selective read-out), NOT a drive deficit and NOT
re-targeting-by-lockon.** Raising `SC_CORTEX_W` from 18 to 150 changes only the static-hold finalQ; it cannot make
the read-out's output track the goal, so re-orient stays broken at every level.

---

## WHICH gap (the honest-negative classification)

The brief asks to crisply report WHICH residual if the drive doesn't fix re-orient: **re-targeting** (SC stays locked
on the old goal) vs **operating-point floor** (saturates/destabilizes). The faithful grid-32 evidence settles it as
the **operating-point FLOOR**, specifically a **non-goal-tracking / under-selective read-out**:

- **Not re-targeting-by-lockon:** the post-change action counts are not dominated by the *old* phase's winning
  cardinal — across all four phases the SC arm outputs the SAME cardinal (N ~0.45–0.52) regardless of the goal. If it
  were locked on the previous goal, the dominant cardinal would *shift* phase-to-phase tracking the old goals; it
  does not. The read-out output is goal-INVARIANT.
- **Not a drive deficit:** raising `SC_CORTEX_W` 18→60→150 makes the post-change re-orient NON-MONOTONE (116→131→111)
  and never approaches host (best SC arm is ~73x worse on post-change). Stronger drive only moves the *static-hold*
  finalQ (phase0 20.9→12.4→1.5) — it helps hold a single fixed goal but cannot make the output track a *moved* goal.
- **Operating-point floor (read-out selectivity / goal tracking):** the `sc_map -> cortex_X` quadrant-pooling
  read-out's directional output does not respond to the bump's retinal location with enough contrast to override the
  cascade's intrinsic N-bias, and that contrast does not increase with drive magnitude. The pooling weights
  `wv = max(0, ±ddx/ddy)` overlap substantially near the bump centre, so a stronger global drive lifts all four
  pools' input together — the winner's *margin* over the runners-up does not widen with drive; it just over-drives
  the actor (and the partial-silence persists). The free parameter that would close the gap is the read-out's
  **selectivity / goal-position tracking** (a sharper pooling kernel, a competitive WTA at the cortical read-out, or
  divisive normalization), NOT the drive magnitude.

This is consistent with — and sharpens — the prior de-risk's "operating-point family, not a dendrite" framing: the
operating point that matters is the read-out's *selectivity / goal-tracking*, and the drive-strength knob (the one
the prior doc flagged as the single most plausible fix) is now shown to be the wrong lever — closing #6 cleanly as a
characterized honest-negative and pointing the (deferred) follow-on at the read-out geometry, not the drive.

---

## sim/ edit?

**NONE.** `SC_CORTEX_W` is an existing env knob (`g11_bg_runner.py:4433`, default 18.0, explicitly documented as
sweepable). The sweep is env-only; the probe reuses `run_moving_goal_episode` by import. No `sim/` change was needed
or made.

---

## Commits (all on `main`, PATHSPEC, pushed origin + gitea)

- `e7ca4655` — the probe `_nav_sc_drive_reorient_derisk.py` + the grid-8/480 smoke (saturates ~3.5x worse;
  near-uniform action distribution at strong drive).
- `e333d771` — this findings doc (grid-8 + the host grid-32 anchor + the live symptom + the handoff command).
- *(this commit)* — the **grid-32/1800 faithful sweep COMPLETE** (the full SC table + the goal-invariant stuck-N
  action-distribution evidence + the doc finalization).
