---
type: finding
status: live
date: 2026-09-25
lane: A · Affect (affect-marker SETTLE default-flip preconditions)
mechanism: BRAIN_AFFECT_MARKER_SETTLE (default OFF) -- flip precondition 3 (A3, affective GPU timing) and the
  instrument that measures it (research/runners/_affect_marker_settle_gpu_timing.py)
seeds: [42]
seed-waiver: criterion L is a 1-seed latency de-risk at the production default seed 42, as the prereg scopes it. No
  generalisation is claimed, and the verdict is UNDEFINED.
verdict: UNDEFINED (not GO, not NO-GO). The direct cost of the settle WTA is +0.13 s per affective warm turn, inside
  the 0.3 s bound. The whole-turn delta cannot be told apart from the bound because processes differ by up to 4.72 s.
  Precondition 3 stays unmet and SETTLE stays default OFF.
runner: research/runners/_affect_marker_settle_gpu_timing.py
prereg: research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md
artifacts:
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/verdict.json
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/decomposition.json
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/s42/00_off.json
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/s42/01_on.json
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/s42/02_on.json
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/s42/03_off.json
  - research/findings/raw/_affect_marker_settle_gpu_timing/a3/provenance_runs.jsonl
---

# Affect-marker SETTLE, A3 GPU timing: UNDEFINED -- the WTA costs +0.13 s, but 4.72 s of between-process noise hides the whole-turn delta (2026-09-25)

SETTLE (`BRAIN_AFFECT_MARKER_SETTLE`, default OFF) flips only after three preconditions pass (owner rule). This
finding records precondition 3, the affective GPU timing in its amended form (A3 in
`research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md`). The pre-registered rule
reads **UNDEFINED**. That is not a NO-GO and not a GO. Nothing flips.

## The run

- gpu_queue 03:21-04:18 on 2026-09-25, from a clean checkout pinned at d71c1a2c9 (on main). Seed 42. Four fresh
  sequential processes in the order off, on, on, off. `SIM_BACKEND=cupy`, brain `tiny-demo`, the Qwen renderer with
  the LLM on, `rich` at its production default. Every warm turn affective.
- Records: one per process, in run order, under `research/findings/raw/_affect_marker_settle_gpu_timing/a3/s42/`
  (`00_off`, `01_on`, `02_on`, `03_off`, each `.json`); the verdict is
  `research/findings/raw/_affect_marker_settle_gpu_timing/a3/verdict.json`. The five run records (controller plus
  four workers: git_sha d71c1a2c9, git_dirty false, SIM_BACKEND=cupy) are in
  `research/findings/raw/_affect_marker_settle_gpu_timing/a3/provenance_runs.jsonl`. The artifacts were copied
  byte-identical from the run's out-dir in commit 3ceb156ac, which gives the checks.
- Every precondition except resolvability held: all four processes valid (cupy, Qwen renderer, HTTP 200, clean code,
  every warm turn affective with at least one WTA read, every read and every cached reader in the arm's config), two
  processes per arm, balanced order (mean position 1.5 in each arm), one code revision.

## The verdict, by the pre-registered rule

| metric | value (s) |
|---|---|
| M1, warm-turn total, ON minus OFF | -3.4627 |
| M2, WTA time, ON minus OFF | 0.13282 |
| M3, total minus WTA, ON minus OFF (reported) | -3.59698 |
| NOISE, largest within-arm range of process medians | 4.72235 |
| arm total, OFF / ON | 32.94825 / 29.48559 |
| arm WTA, OFF / ON | 0.020225 / 0.153041 |

PASS needs M2 <= 0.3 and M1 + NOISE <= 0.3: M2 holds, but M1 + NOISE = 1.260 s. FAIL needs M2 > 0.3 or <!--derived-->
M1 - NOISE > 0.3: M1 - NOISE = -8.185 s. The result is in neither region, so the rule reads UNDEFINED. <!--derived-->

## Decomposition of the noise

Source: `research/findings/raw/_affect_marker_settle_gpu_timing/a3/decomposition.json`, written by
`--decompose-a3` (pure analysis of the four records; no simulation ran). Build turn = the first turn (neutral text,
brain build, no WTA read, so it does not depend on SETTLE). First affective = the next turn, which builds the WTA
reader (not scored). Warm = the four scored turns.

| process | arm | build turn (s) | first affective (s) | warm median (s) | warm SD (s) | WTA median (s) | WTA reads per warm turn |
|---|---|---|---|---|---|---|---|
| 00 | off | 705.97 | 61.40 | 35.3094 | 4.6413 | 0.021174 | 2, 1, 2, 2 |
| 01 | on | 668.16 | 57.38 | 30.1641 | 1.5307 | 0.158652 | 2, 2, 2, 2 |
| 02 | on | 604.83 | 60.56 | 28.8071 | 1.9780 | 0.147430 | 2, 2, 2, 2 |
| 03 | off | 599.57 | 65.20 | 30.5871 | 1.3151 | 0.019276 | 2, 1, 2, 2 |

**1. The WTA itself is small, stable, and matches its step count.** One WTA read costs 0.0110612 s OFF and 0.0771003 s
ON on average. That is 6.97 times more for 9.75 times the simulated steps (160 steps OFF vs 1560 ON per read). <!--derived-->
Two reads per affective turn at the per-read difference predict 0.1321 s, against the measured M2 of 0.13282 s. <!--derived-->
The ON WTA is 0.52% of the ON warm turn. So M2 is the direct cost of the flag, and it sits 0.16718 s under the bound. <!--derived-->

**2. All of the variance is outside the WTA.** M3 (total minus WTA) is -3.59698 s, the same size as the process
spread. Per Amendment 3's reading of the default code path, the lead SETTLE selects is added after the reply is
rendered and nothing downstream reads it, so SETTLE is not expected to change the render. A3 did not measure that.
M3 is read as noise here, not as a speed-up caused by SETTLE.

**3. The noise is between processes that did the same planned work.** The two processes of an arm had the same
revision, seed, messages and turn order. They also selected the same leads and made the same number of WTA reads at
every warm index (OFF: 2, 1, 2, 2 in both; ON: 2 at every index in both). Yet their warm medians differ by 4.72235 s
(OFF: 35.3094 vs 30.5871) and 1.357 s (ON: 30.1641 vs 28.8071). The within-arm SD of process medians is <!--derived-->
3.33921 s OFF and 0.95957 s ON, pooled 2.45673 s. The pooled value has only 2 degrees of freedom: its 95% interval
is 1.28 to 15.44 s. <!--derived-->

**4. Turn index matters, and the processes share it.** A two-way fit (process x warm-turn index) gives turn-index
means of 32.5014, 30.2074, 28.7455 and 33.8113 s, a range of 5.07 s that all four processes follow. The residual <!--derived-->
turn-level SD is 1.72264 s (9 df). Process 00 alone holds 17.1258 of the 26.7076 residual sum of squares (64.1%); <!--derived-->
without it the residual SD is 0.80345 s (6 df). Process 00 also had the largest turn-to-turn SD (4.6413 s against
1.3151 to 1.9780 s).

**5. The SETTLE-free build turn varied too.** Build turns were 705.97, 668.16, 604.83 and 599.57 s: they fell with
run position, with a spread of 106.4 s (17.7% of the fastest). Across the four processes the build turn and the <!--derived-->
warm median correlate at r = 0.80. With n = 4 and the arm confounded, that is not evidence of anything by itself. <!--derived-->
It is consistent with a machine-level speed that changed during the run (the machine is shared), which is a
hypothesis, not a result.

**6. What the records cannot tell.** The A3 worker recorded wall time, WTA time, reads, lead, level, renderer and
`abstained` per turn. It recorded no render seconds, generated tokens, reply length, CPU time or load average. So
the records cannot say whether the 4.72 s is render length, per-token speed or contention. Two facts limit the
render-length explanation. Every warm reply abstained (16 of 16), and the Qwen mouth decodes greedily and reseeds
before each call (Amendment 3), so an arm's two processes are expected to render the same text. That is expected,
not verified: the reply text was not recorded. Amendment 3's instrument records all of these fields.

## How many processes the A3 instrument would need

**As written, no number.** NOISE is a RANGE of process medians. For n normal draws the expected range is d2(n)
times the SD (d2 = 1.128, 2.059, 3.078 at n = 2, 4, 10). At the pooled SD of 2.45673 s the expected NOISE is <!--derived-->
2.77120 s at n = 2, 5.05842 s at n = 4 and 7.56183 s at n = 10 per arm: it grows with n. If the true delta
equals M2, PASS needs NOISE <= 0.3 - 0.13282 = 0.167 s. That is 16.6 times below the expected range at n = 2. <!--derived-->
The chance that one arm's two-process range falls under 0.167 s is 2 Phi(0.167 / (sqrt(2) x 2.457)) - 1 = 0.038, <!--derived-->
and for both arms 0.0015. FAIL needs M1 - NOISE > 0.3, so in expectation a real cost would need to be <!--derived-->
above about 3 s before A3 could read NO-GO at n = 2. <!--derived-->

**Read as a standard error instead (the best case for a design that compares processes).** The arm difference of
process medians has SE = sigma x sqrt(2 / n) per arm, so n = 2 (z sigma / margin)^2. With sigma = 2.45673 s: <!--derived-->

| margin (s) | z | processes per arm | queue time, both arms, about 14 min per process (h) |
|---|---|---|---|
| 0.3 (if the WTA cost nothing) | 1.645 (one-sided 95%) | 362.88 | 169 <!--derived--> |
| 0.3 | 2.486 (95% + 80% power) | 829.22 | 387 <!--derived--> |
| 0.167 (0.3 minus M2) | 1.645 | 1168.45 | 545 <!--derived--> |
| 0.167 | 2.486 | 2670.08 | 1246 <!--derived--> |

About 14 minutes per process is the A3 run's own rate (57 minutes for 4 processes, of which the build turn is 10 to <!--derived-->
12 minutes). The 2-df sigma moves every count by a factor of 0.271 to 39.5 (its 95% interval, squared). Even the <!--derived-->
lower end, 317 processes per arm at margin 0.167, is about 148 hours of queue. More A3 processes cannot resolve <!--derived-->
the bound at an affordable cost.

**The same arithmetic for a within-process crossover** (Amendment 3's design, where each process runs both arms and
process speed cancels): N scored turns = 4 sigma_turn^2 ((1.645 + 0.842) / 0.167)^2. With sigma_turn = 0.80345 s <!--derived-->
(without process 00) N = 571.155; with 1.72264 s (all four) N = 2625.607. The planned 576 scored turns give a
one-sided 95% half-width of 0.110130 s and 0.236125 s respectively. The second case would read UNDEFINED again.

## What this shows and does not show

- Shows: at seed 42 on this machine, SETTLE's direct cost (the settle WTA reads on an affective warm turn) is
  +0.13282 s, inside the 0.3 s bound. The lever reached the reader in every ON process (every read ran 500/1000 ms).
- Does not show whether the whole warm turn stays within 0.3 s (M1). M1 = -3.4627 s is not evidence that SETTLE
  makes turns faster. It is inside the between-process noise.
- Not a NO-GO: nothing is in the FAIL region. Not a GO: M1 + NOISE is above the bound. One seed, one night, one
  shared machine.
- Side note for the G1 phase-timing prereg (`2026-09-24-production-chat-phase-timing-PREREGISTRATION.md`): its SETTLE
  warm-turn delta compares one process per arm on one warm turn. The A3 within-arm spreads (1.36 to 4.72 s) are 4.5 <!--derived-->
  to 16 times its 0.3 s bound, so a single pair is not expected to resolve it either. <!--derived-->

## Next rung (preregistered, not run)

Amendment 3 in the prereg (commit fd1240f9a, before any new data) replaces the A3 instrument with a within-process
crossover: `--xo-run`, SETTLE toggled per turn inside one process (each arm keeps its own warm reader), 4 processes at
seed 42 with mirrored orientations off, on, on, off, 48 runs of 4 same-arm turns per process in ABBA order, the first
turn of each run a washout, and an OLS on run means with process and run-slot fixed effects. GO iff both one-sided
95% upper bounds (M1, M2) are <= 0.3 s; NO-GO iff either lower bound is > 0.3 s; otherwise UNDEFINED. The instrument
also records render time, generate calls and tokens, the reply without its lead, CPU time, load average, CuPy pool
bytes and max RSS, so a repeat of this noise can be attributed.

`--selftest` (no brain build) passes: 12 A3 cases and 25 Amendment 3 cases, including each failing direction (a
0.58 s WTA -> NO-GO; +1.0 s outside the WTA -> NO-GO; +0.8 s carried into the next turn -> NO-GO; a true 0.30 s cost
-> UNDEFINED, never GO; 5 s turn noise -> UNDEFINED; each lever, validity and balance failure -> UNDEFINED).

The run goes to the GPU queue from a clean checkout pinned at the head of `research/settle-a3-amendment3`
(`<pin>`, with `data/corpus/tinystories.txt` symlinked in). `<a3x>` is
`<pin>/research/findings/raw/_affect_marker_settle_gpu_timing/a3x`, inside the pinned checkout so the provenance door
writes its sidecars (the A3 run wrote to another worktree and got none). Projected about 7.3 hours:

```
cd <pin> && export XDG_RUNTIME_DIR=/run/user/1000; for i in $(seq 1 240); do bash tools/mem_ok.sh 16 4 >/dev/null 2>&1 && break; sleep 60; done; \
  bash tools/mem_ok.sh 16 4 && SIM_BACKEND=cupy OMP_NUM_THREADS=1 bash tools/memcap.sh 20 -- \
  /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._affect_marker_settle_gpu_timing --xo-run --seeds 42 \
  --orient off,on,on,off --runs 48 --run-len 4 --out-dir <a3x> --out <a3x>/verdict.json
```

If it reads UNDEFINED because turn noise stays high, the prereg names the next rung: variance control (a quiet
machine window or CPU isolation), not a smaller bound or a looser rule.

Lever count against this defect (the A3 timing could not resolve the bound): one, the instrument redesign. No
mechanism lever was tried, because the defect is in the measurement.
