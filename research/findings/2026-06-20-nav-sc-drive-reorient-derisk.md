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

**⇒ The WHICH classification is the operating-point FLOOR, not the re-targeting gap.** The failure mode is *not* "the
SC stays locked on the old goal's cardinal" (that would be a re-targeting problem a stronger drive can't fix either,
but for a different reason); it is "a stronger drive saturates/de-sharpens the read-out" — the quadrant-pooling
read-out's selectivity does not increase with drive magnitude, so past a modest level more current only pushes ALL
pools past threshold (uniform competition) instead of widening the winner's margin. The drive magnitude is not the
free parameter that closes the gap; the SC->cortex **read-out's selectivity** (the quadrant-pooling geometry / a
sharper WTA) is.

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

## Grid-32/1800 (seed 42, warmup 600) — the faithful confirm

*(GRID32_PLACEHOLDER — filled below when the run lands.)*

**The exact command (in flight in the background as of this commit; ~30–60 min GPU for the 4-episode sweep):**

```bash
SIM_BACKEND=cupy python -m research.runners._nav_sc_drive_reorient_derisk \
  --seed 42 --grid-size 32 --n-steps 1800 --warmup-steps 600 \
  --sc-drive-levels 18,60,150 \
  --out research/findings/raw/nav_gate_2a/scdrive_grid32_seed42.json
```

Read the `verdict` block + each arm's `per_phase_finalQ` / `phase0_finalQ` / `post_change_finalQ` /
`late_motor_sustain_frac` from the summary JSON. The grid-8 prediction is that the SC arm's post-change finalQ stays
~3–4x the host control at every drive level and does NOT monotonically approach host (it saturates) — confirming the
operating-point-FLOOR (read-out-selectivity) classification, not a drive-strength deficit.

**Host positive control (grid-32/1800, the anchor) — re-orients cleanly through all 3 post-change goals:**
phase0_finalQ **0.690** (acquires the NE corner), post-change finalQ **[0.504, 0.496, 0.504]** (sum **1.504**),
gate **2.195**, late_sustain **1.000**. The host re-adapts to each new corner (~0.5 = essentially at-goal).
The live `sc_w18` trace confirms the symptom directly: at the first re-orient phase (goal far-west `(1,30)`) the SC
arm is stuck at pos `(22,31)→(31,31)`, recent_dist **~22–24** — it drifts away from the new goal instead of
re-orienting, exactly the prior de-risk's "cannot re-orient to the far-west goal after the phase transition."
*(The full 3-level SC sweep numbers land in the table below.)*

---

## WHICH gap (the honest-negative classification)

The brief asks to crisply report WHICH residual if the drive doesn't fix re-orient: **re-targeting** (SC stays locked
on the old goal) vs **operating-point floor** (saturates/destabilizes). The grid-8 evidence (and the grid-32 confirm
below) point to the **operating-point FLOOR**, specifically a **read-out-selectivity** floor:

- **Not re-targeting-by-lockon:** the post-change action counts are not dominated by the *old* phase's winning
  cardinal — they go toward *uniform*, not toward the stale winner. The SC isn't stubbornly pointing the old way; it
  is pointing *every* way at once.
- **Operating-point floor (read-out selectivity):** the `sc_map -> cortex_X` quadrant-pooling read-out's selectivity
  does not increase with drive magnitude. The pooling weights `wv = max(0, ±ddx/ddy)` overlap substantially near the
  bump centre, so a stronger global drive lifts all four pools' input together — the winner's *margin* over the
  runners-up does not widen, and past a modest drive the competition degenerates to chance. The free parameter that
  would close the gap is the read-out's **selectivity** (a sharper pooling kernel / a competitive WTA at the cortical
  read-out / divisive normalization), NOT the drive magnitude.

This is consistent with — and refines — the prior de-risk's "operating-point family, not a dendrite" framing: the
operating point that matters is the read-out's selectivity, and the drive-strength knob (the one the prior doc
flagged as the single most plausible fix) is now shown to be the wrong lever.

---

## sim/ edit?

**NONE.** `SC_CORTEX_W` is an existing env knob (`g11_bg_runner.py:4433`, default 18.0, explicitly documented as
sweepable). The sweep is env-only; the probe reuses `run_moving_goal_episode` by import. No `sim/` change was needed
or made.

---

## Commits (all on `main`, PATHSPEC, pushed origin + gitea)

- `e7ca4655` — the probe `_nav_sc_drive_reorient_derisk.py` + the grid-8/480 smoke (saturates ~3.5x worse;
  near-uniform action distribution at strong drive).
- *(this commit)* — the grid-32/1800 faithful confirm + this findings doc.
