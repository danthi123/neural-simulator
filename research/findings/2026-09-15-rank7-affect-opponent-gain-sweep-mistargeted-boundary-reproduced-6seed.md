---
type: finding
status: contributing
date: 2026-09-15
mechanism: affect-opponent-columnar-convergence
lane: scaffold-retirement
seeds: [42, 43, 44, 100, 101, 102]
verdict: The 2026-09-10 gain sweep swept the WRONG knobs (--to-fs-w/--fs-inh-w are INERT in --opponent mode; all 42
  cells byte-identical) -> it did NOT test Rank-7's named competition-strength lever, but it DID reproduce the
  608a06304 BOUNDARY at 6 seeds (on-substrate recall@FP0 ~0.0318 worst << 0.5 bar at default XINH). Corrected XINH
  sweep queued on the pool.
runner: research/runners/_affect_onsubstrate_noise_robust_convergence_derisk.py
artifacts:
  - research/findings/raw/_affect_gain_sweep/_aggregate_6seed.json
  - research/findings/raw/_affect_gain_sweep/opp_to18_fs8_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_to18_fs27_s42.json
  - research/findings/raw/_affect_gain_sweep/opp_to12_fs15_s42.json
external: NO-EXTERNAL-NEEDED -- reuses the Kang/Watanabe/Pu 2024 PNAS competition-strength framing already banked in
  the Rank-7 arc (608a06304); no new biological claim.
builds_on:
  - research/findings/2026-09-08-affect-opponent-columnar-spiking-worst-case-unchanged-BOUNDARY.md
---

# Rank-7 affect-opponent gain sweep mis-targeted the knobs (boundary reproduced 6-seed; corrected XINH sweep queued)

**One-line.** The owner-operated pool sweep (42 on-substrate cells) meant to test Rank-7's named next rung — retune
the opponent columns' **competition strength** — instead swept `--to-fs-w`/`--fs-inh-w`, which are **inert in
`--opponent` mode** (they drive the shared-FS path, not the opponent cross-inhibition). Every one of the 42 cells is
byte-identical in its metrics, which is the tell. The run is not wasted: it **reproduces the 608a06304 BOUNDARY at 6
seeds** (on-substrate recall@FP0 stays at floor, ~0.0318 worst / 0.201 clean, vs the 0.5 usability bar, at default
XINH gains). The corrected sweep — over `--xinh-exc-w`/`--xinh-inh-w`, the knobs the `--opponent` path actually uses —
is now queued on the pool.

## What the 42 cells show (6 seeds x 7 (to_fs, fs_inh) combos)

<!--derived-->
(6-seed means/derived quantities from the cited per-cell artifacts; per-combo means saved in
`research/findings/raw/_affect_gain_sweep/_aggregate_6seed.json`.)

Mean on-substrate (`--spiking`) `spiking_realistic_worst` = **0.0318 (sd 0.027)**, `spiking_clean_worst` = 0.201,
GO 0/42 — **identical across every (to_fs, fs_inh) combo and every seed**. That invariance is the diagnostic: the
swept parameters had zero effect on the opponent circuit.

## Why they were inert (traced in the runner)

<!--derived-->
(code line references + config constants below, not artifact measurements.)

`--opponent` builds `build_opponent_convergence_bridge`
(`_affect_onsubstrate_noise_robust_convergence_derisk.py:260`), whose cross-inhibition pathways use `a.xinh_exc_w` /
`a.xinh_inh_w` (`:297-303`). `--to-fs-w`/`--fs-inh-w` (TO_FS_W=18 / FS_INH_W=15) drive the SHARED-FS single-assembly
variant, which `--opponent` does not build. So the swept knobs never touched the running circuit. The Rank-7 finding
named the lever precisely — "retune XINH_EXC_W/XINH_INH_W" — and the 2026-09-10 stocker swept the wrong pair because
the XINH defaults were not to hand at stocking time (a stale-parameter lapse: the stocker's own `#checked` reason even
named XINH, but the command swept TO_FS/FS_INH).

## What is still true and useful

<!--derived-->
(6-seed summary + config constants; see the cited artifacts + aggregate.)

At **default XINH** (exc 8.0 / inh 12.0) the on-substrate opponent columnar convergence holds recall@FP0 at floor
across all 6 project seeds — a genuine 6-seed confirmation of the previously single-seed-floor BOUNDARY. The
zero-FP bar is not cleared; the assembly spikes (~30.9/concept, grounding-modulated) and the instrument is valid
(synthetic ceiling 1.0, text ceiling 0.029), so the setup works and the residual is real, not an artifact.

## Corrected sweep queued (no-defer)

<!--derived-->
(planned config values, not artifact measurements.)

On the pool (CPU, `--spiking`): `--xinh-exc-w` in {4,8,16} x `--xinh-inh-w` in {6,12,18,24,36} around defaults
(8.0/12.0), 6 seeds, excluding the default center, written to `opp_xinh_e*_i*`-named files in the same
`_affect_gain_sweep` directory. Primary axis = `xinh_inh_w` (cross-inhibition = competition strength). This is the
actual Rank-7 rung.

## Lesson (for the queue tooling)

A parameter sweep must verify its knob is LIVE in the target code path before spending compute — the same
"one-flag != one-variable / check the default" class already banked in the neural-simulator skill, here as
"a flag can be inert in the mode you're running." Recorded so the corrected sweep confirms non-inertness on cell 1.
Functional read-outs only.
