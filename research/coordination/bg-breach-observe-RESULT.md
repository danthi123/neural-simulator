# BG step-1 breach OBSERVE run — RESULT (batch-1026 (3))

**2026-10-26 12:40 EDT · approved: RULING batch-1026 (3), answering NEEDS `[bg-breach-diag]` ASK ·
runner `research/runners/bg_breach_observe.py` · records
`research/findings/raw/_bg_live_seam/observe/s72_hsb_hi.json` + `s71_hsb_hi.json` (each
`.json.prov.json`-stamped) · tree: branch `research/bg-tonic-loop` (worktree `.claude/worktrees/bg-tonic-loop`)**

## What ran

An exact mirror of a step-1 grid cell (set `stn50_gpe50`, event `hsb_hi`, the event whose aggregate
v range carried −1425.8) with observation added: same env kv (`PYTHONHASHSEED=0`, `SIM_BACKEND=numpy`,
1 thread, `BRAIN_BG_PRESET_CELLS=1`, `BRAIN_BG_BOUNDED_COND=1`, `BRAIN_BG_TONIC_DRIVE=1`,
`BRAIN_BG_TONIC_GPE_PA=50`, `BRAIN_BG_TONIC_STN_PA=50`, neural seam, unique-ignition read, commit
on, `BRAIN_CHAT_SEED=<seed>`), same branch code, same entry `decide_action_neural("hi")`. The
instrument wraps the step function and records per step: whole-network min/max v and min v for each
of 34 regions, the below-−150 mV cells, first-breach read-back (params, flag read-back, currents,
u). Both runs finished in 0.8 s wall, `mem_ok 4`, local (well under the 10-minute
compute-placement line), warnings none.

## THE ANSWER — the breaching region is `str_d2_1` (D2 indirect-pathway striatal MSN)

- **s72: BREACH.** First breach **step 215**, **cell 439 ∈ `str_d2_1`**, the only cell below −150 mV
  at that step (−442.836 mV), one step after the phase mark at 214. The runaway deepens to the
  run's end: **global min −1425.8082275390625 mV at t=849 (final step)** — `identity_match: True`
  against the grid's s72 hsb_hi aggregate (−1425.8). 850 steps recorded × 34 region minima.
- **No other region ever crosses −150 mV:** worst-region table `str_d2_1` −1425.808, next worst
  `str_d2_0` −99.174, then `str_d1_0` −92.904, `str_d1_1` −92.729, `practice_arousal` −77.862 …
  33 of 34 regions stay ≥ −99.2 mV for the whole run.
- **s71 control: NO_BREACH.** Worst region `str_d2_0` −96.983 (in range), deepest at t=729; same
  flags on, 850 steps, warnings none. The blow-up is seed-locked (72) and region-locked (one cell
  of `str_d2_1`).

## The ruling's field list (s72 breaching cell 439)

| field | preset (label read-back) | jittered (live arrays) |
|---|---|---|
| a | 0.01 | **0.00990449357777834** |
| b | −25.0 (`IZH2007_STRIATAL_MSN_D2`) | **−40.13849639892578** |
| C | 50.0 | **24.42342758178711** |
| d | 180.0 | **162.07858276367188** (`cp_izh_d_increment`) |
| c | — | −55.0 (`cp_izh_c_reset`; vr/vt/vpeak/k also recorded) |

- **u (recovery, `cp_recovery_variable_u`): 32395.546875** at read-back.
- **Synaptic current by receptor** (frozen end-state): ampa −5.9e−22 pA (g_e 4.1e−25, E 0), nmda
  0 pA (g 0, E 0), **gaba −12.045 pA (g_i 0.00895, E −80)**.
- **Tonic + external:** breach cell's `cp_external_input_current` = 0.0 pA — striatal cells receive
  no tonic; the tonic rows live in that same array, region means at read-back: `selector_stn` 50.0,
  `gpe_0`/`gpe_1` 50.0, `gpi_0/1` 1000.0, `thal_0/1` 900.0 pA (env-consistent). Env tonic values
  recorded in the record's `env_full`.
- **Ratio/bound flag read-back:** `BRAIN_BG_BOUNDED_COND=1` → `bg_bounded_cond_enabled() True`,
  `bg_preset_cells_enabled() True`, `tonic_drive_enabled() True` — the bounded path was live.

## Notes and limits

- Preset + jittered params are static arrays (state-independent). Receptor currents, u and the
  external array are read at the frozen final state (t=849, the deepest point), not at first breach;
  the per-step v trajectory for all 34 regions is in the record for the full audit.
- The jittered values are what per-neuron heterogeneity ("Applied heterogeneity to 4 parameters")
  produced for this cell at seed 72: C halved (24.4 vs preset 50), b shifted to −40.1 (preset −25),
  a barely moved (0.00990 vs 0.01), d 162.1 (preset 180).
- Rerun (from the bg-tonic-loop worktree): `env PYTHONHASHSEED=0 OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SIM_BACKEND=numpy
  CUDA_VISIBLE_DEVICES= <venv>/bin/python -u -m research.runners.bg_breach_observe --seed 72`.
- For Claude: the fix choice (preset-centre narrowing / per-region clamp floor / bounded-path
  change) can now target `str_d2_1` and this cell's jittered C/b specifically.

## Commit attempt — lane gate BLOCKED (verbatim gate output; NOT waived, per RULING stage2-next-1026)

`git commit` of the runner + 2 records on `research/bg-tonic-loop` returned COMMIT_RC=1:

```
⛔ COMMIT BLOCKED — failure-class gate(s):
   [corpus-check-required] raw/g.json: no direct stamp, but ACCEPTED via fallback evidence in
     /opt/data/cache/scratch/tmpz7zjb_ci/shared.jsonl (checked 999996400.0, query: 'selftest
     fallback probe'). This proves a check happened before the run, not that it was on-topic --
     same strength as the direct stamp.
   consumer-hardware-reference: estimated production spiking-substrate VRAM ~1.344 GiB (N=7002
     neurons, 700200 synapses @ connections_per_neuron=100, x8 safety margin, +1.0 GiB fixed
     overhead) vs the 24 GiB single-consumer-RTX-3090 reference -- 22.66 GiB headroom (5.6% of
     budget used).
     ⛔ lane-starvation        1 problem(s)
           5 of 5 disjoint CPU lanes UNSERVED: A · Affect; B · Curiosity; C · Self/Workspace;
           D · Perception; E · Language.
           They are concurrent with GPU work and cost nothing beside it; leaving them unqueued is
           unused capacity, not prioritisation. Five sat idle 194 min on 2026-07-31 while the
           heartbeat alarmed correctly and was read past.
           FIX: stage one job per idle lane -
             bash tools/pool_queue.sh add '<cmd>' --checked '<what the record says>'
           Or waive (auto-expires in 6h, and the class/evidence is REQUIRED -- free prose is no
             longer accepted, the 2026-09-23 promise-language loophole):
             printf 'CLASS: NO-READY-WORK\nchecked=<what you searched>\n' > research/queue/.lane_waiver
   Deliberate override: git commit --no-verify
```

(The corpus-check and consumer-hardware lines are ACCEPTED advisories; the sole ⛔ is
lane-starvation. Pool state at attempt: `research/queue/pool.queue` 0 lines, nothing running.
The two gate-fix routes were both declined on purpose: a waiver write is forbidden by
RULING stage2-next-1026 (two already written; a third moves the escape to Claude-only by code),
and five honest pool jobs do not exist right now — f1h step 0 (un-parked by the same ruling) and
the dejits attempt 3 requeue are the next-wake candidates for lanes, the diagnoses and this
observe RESULT are driver-lane work, not pool jobs. The observe run itself already completed
locally (0.8 s/run), so requeueing it would be duplicate work. Staged set stays in the worktree,
clean, awaiting either lanes or Claude.)

