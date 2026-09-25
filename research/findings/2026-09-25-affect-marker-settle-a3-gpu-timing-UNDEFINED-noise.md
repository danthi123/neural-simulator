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

**As written, no number.** NOISE is a RANGE of process medians. For n normal draws the expected range of ONE ARM
is d2(n) times the SD (d2 = 1.128, 2.059, 3.078 at n = 2, 4, 10). At the pooled SD of 2.45673 s the expected <!--derived-->
range of a single arm is 2.77120 s at n = 2, 5.05842 s at n = 4 and 7.56183 s at n = 10: it grows with n. <!--derived-->
**Correction (2026-09-25 fix round after independent review): those three numbers are each ONE ARM's expected
range. `decide()`'s own NOISE is the LARGER of the two arms' ranges** (`max over arms of (max - min)`), whose
expectation is higher than either arm's alone. At n = 2 -- two independent range draws from the same pooled SD,
each the absolute difference of an iid normal pair -- numerically integrating E[max(X, Y)] for X, Y iid
half-normal(sigma x sqrt(2)) gives **3.92037 s**, about 1.596 x sigma, not the 2.77120 s of a single arm. <!--derived-->
If the true delta equals M2, PASS needs NOISE <= 0.3 - 0.13282 = 0.167 s, 23.5 times below NOISE's own expected <!--derived-->
value at n = 2. The chance that one arm's two-process range falls under 0.167 s is <!--derived-->
2 Phi(0.167 / (sqrt(2) x 2.457)) - 1 = 0.038, and for both arms (NOISE's own event) 0.0015. FAIL needs <!--derived-->
M1 - NOISE > 0.3, so in expectation a real cost would need to be above about **4.2 s** (0.3 + 3.92037, using <!--derived-->
NOISE's own expectation rather than one arm's 2.77120 s) before A3 could read NO-GO at n = 2. <!--derived-->

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

`--selftest` (no brain build) passes: 12 A3 cases and 30 Amendment 3 cases (26 as originally committed, plus 4 added
in the 2026-09-25 fix round below), including each failing direction (a 0.58 s WTA -> NO-GO; +1.0 s outside the WTA
-> NO-GO; +0.8 s carried into the next turn -> NO-GO; a true 0.30 s cost -> UNDEFINED, never GO; 5 s turn noise ->
UNDEFINED; each lever, validity and balance failure -> UNDEFINED).
A mutation check of the four comparisons in `decide_xo` found that dropping the M2 bound from either region left
all 25 original cases passing, because every M2 case also moved M1. One case now pins M2 on its own (a +0.58 s WTA
with the rest of the ON turn 0.6 s faster, so M1 stays inside the bound -> NO-GO). With it, each of six mutations
(drop U1, L1, U2 or L2; loosen the PASS bound; tighten the FAIL bound) makes the selftest fail. The rule itself is
unchanged.

**Fix round after an independent review (2026-09-25, before any Amendment 3 gate data -- the smoke below is a
pre-flight check, not the gate; see the dated addendum in Amendment 3 of the prereg for the full account).**
Four gaps in the instrument itself, found by reading it adversarially rather than by new data:

1. **The CI width was unpinned.** Every selftest case above checks one draw's SIGN (NO-GO/GO/UNDEFINED), not the
   one-sided 95% bound's WIDTH -- `t = float(stats.t.ppf(1.0 - alpha, df))` in `_fe_fit` could be halved and every
   case would still pass. A calibration case (`_xo_go_rate`, 100 independent synthetic reps at a true whole-turn
   cost exactly at the 0.3 s bound, runs=8) now asserts the GO rate stays <= 10%; it reads 9% on the correct code,
   and the halved-`t` mutant was hand-verified (mutate, rerun, revert, diff back to clean) to push it to 24%.
2. **A washout-exclusion regression had no dedicated test.** The existing carry=0.8 case reads NO-GO whether or not
   washout turns are correctly excluded from scoring (a true M1 of +0.93 s clears 0.3 s either way). A new case
   near the bound (carry=0.25, noise=0.05, true M1 ~0.38 s) also asserts M1's estimate is close to 0.38 s, not just
   its sign.
3. **The warm-up precondition checked only which arm ran, not whether it fully ran.** If either warm-up turn's
   first axis (valence) does not select a word, `expression_lead` returns '' before the arousal axis is read (1
   read, not 2) -- exactly what A3 itself saw at one OFF warm index. That reader's arousal bridge is then built
   LAZILY inside the first SCORED turn of that arm, adding one-time build latency to that arm's early scored turns
   only, which would bias M1 toward GO. `check_process_xo` now requires `n_wta_reads == 2` on both warm-up turns
   (both axes committed, both bridges built) before scoring; two new selftest cases (either warm-up turn, either
   process, under-reading) assert UNDEFINED.
4. **The crossover worker (`_worker_xo`) had never executed** -- no `--xo-worker`/`--xo-run` call appeared in any
   provenance log, so the queued 7.5 h run would have been its first execution, including the
   `_get_warm_qwen_renderer()._fac.model.generate` monkey-patch, a 192-turn single session under `memcap 20`, and
   whether affect stays non-neutral over 96 repeats of each message. This is addressed procedurally, not by a code
   change: a short `--xo-run --orient off,on --runs 4 --run-len 2` smoke against the Qwen renderer under
   `mem_ok`/`memcap`, queued separately before the full run (see below for its result once read).

**HELD (owner, 2026-09-25): the 7.3-7.5 h full run below is NOT queued.** The owner is reconsidering the
prepended affect-marker design itself (it may be retired from replies), independent of what this instrument
would read -- do not queue it until that is decided. The recipe is kept, verbatim, for when it un-holds.

The run goes to the GPU queue from a clean checkout **pinned by SHA, not a branch name** -- an earlier draft of
this section pinned "the head of `research/settle-a3-amendment3`", but a branch name is exactly as stable as
whichever local checkout resolves it, and a stale worktree elsewhere (`b58e4080b`, from a killed session) has
already held that local branch name pointed at an old commit once. **A literal SHA drifted stale here twice
across three fix rounds (8dd9c1ed0, then ce4afac2b) before this round replaced it with `<pin>`/`<a3x>` symbols
(prereg's own style, review LOW 2026-09-25) -- resolve `<pin>`'s SHA at QUEUE TIME, never from a value written in
this document:** `git fetch origin research/settle-a3-amendment3 gitea && git log -1 --format=%H
origin/research/settle-a3-amendment3` (verify `gitea/research/settle-a3-amendment3` reads the SAME SHA --
`push_both.sh` keeps both remotes identical). `<pin>` is
`/home/dant123/Projects/sim/.claude/worktrees/settle-a3x-run-<that SHA>` (a detached-HEAD worktree at it); `<a3x>`
is `<pin>/research/findings/raw/_affect_marker_settle_gpu_timing/a3x`. `data/corpus` is gitignored, so a fresh
worktree needs it symlinked in from the primary checkout (the Qwen renderer reads `data/corpus/tinystories.txt`
at load). Projected about 7.3-7.5 hours. The full recipe, verbatim (worktree, symlink, memory wait, corpus check,
then the queued job):

```
# 1. Pinned worktree (detached HEAD at the exact commit resolved above, not a branch name).
cd /home/dant123/Projects/sim && git fetch origin research/settle-a3-amendment3 && \
  git worktree add --detach <pin> <that SHA>

# 2. Corpus symlink (data/ is gitignored; the Qwen renderer reads data/corpus/tinystories.txt at load).
mkdir -p <pin>/data && ln -s /home/dant123/Projects/sim/data/corpus <pin>/data/corpus

# 3. Queue the run: mem_ok 16 4 wait, before_you_build (corpus-check gate), then the memcap-bounded run.
bash tools/gpu_queue.sh add 'cd <pin> && until bash tools/mem_ok.sh 16 4 >/dev/null 2>&1; do sleep 60; done; \
  bash tools/before_you_build.sh "affect-marker SETTLE A3 whole-turn GPU timing at the 0.3s bound (Amendment 3 within-process crossover)" >/dev/null 2>&1; \
  SIM_BACKEND=cupy OMP_NUM_THREADS=1 bash tools/memcap.sh 20 -- \
  /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._affect_marker_settle_gpu_timing --xo-run --seeds 42 \
  --orient off,on,on,off --runs 48 --run-len 4 --out-dir <a3x> --out <a3x>/verdict.json'
```

Step 3's `before_you_build.sh` call runs inside the pinned worktree, after the memory wait and before the run, so
the provenance door stamps a corpus check less than 24 h old on every artifact `gates/corpus_check_required`
refuses a run of more than 1 h without one; the door reads the pin's own log, not the primary checkout's.

If it reads UNDEFINED because turn noise stays high, the prereg names the next rung: variance control (a quiet
machine window or CPU isolation), not a smaller bound or a looser rule.

Lever count against this defect (the A3 timing could not resolve the bound): one, the instrument redesign. No
mechanism lever was tried, because the defect is in the measurement.

## Smoke read (b21140758, 2 processes, `--orient off,on --runs 4 --run-len 2`) + re-score at this round's HEAD

Re-scored with `--xo-score --raw-dir <smoke a3x_smoke dir> --orient off,on --runs 4 --run-len 2`: **UNDEFINED**,
same as read at b21140758, but now for the PRECISE reason (addendum item 7 above) rather than only "1
process/orientation" (still also true: 1 off + 1 on is short of the required >= 2 per orientation) --
`check_process_xo` flags BOTH processes' only problem as "zero Qwen `model.generate()` calls recorded across
build+warmup+run turns" (verified directly against the raw records: `n_gen_calls`/`gen_calls` are 0/`[]` on every
turn of both `xo_00_off.json` and `xo_01_on.json`, matching `generated_tokens_total_by_arm: {"off": 0, "on": 0}`
in the smoke's own verdict.json diagnostics).

Warm-up read count (Addendum item 6's precondition): **both processes' both warm-up turns read `n_wta_reads == 2`**
(00_off: off-arm 2, on-arm 2; 01_on: on-arm 2, off-arm 2) -- the under-read failure mode A3 hit did not recur here.

Memory extrapolation to 194 turns (linear, from the post-warm-up slope over this smoke's 8 scored turns --
**a short window, 24x shorter than 194, so this rules out only a GROSS per-turn leak, not a slow one**): both
`maxrss_kb` and `cupy_pool_used_bytes` are EXACTLY FLAT from the second warm-up turn through all 8 scored turns in
BOTH processes (00_off: maxrss 11,135,748 KB / pool 3,840,247,296 B unchanged across 9 turns, one 512 B pool
readout noise blip; 01_on: maxrss 11,269,628 KB / pool identical) <!--derived-->, i.e. a measured per-turn slope
of 0 in this window. Extrapolating that (zero) slope to turn 194 predicts NO further growth --
maxrss ~11.1-11.3 GB, cupy pool ~3.84 GB, both comfortably inside `memcap 20`'s 20 GB cap -- but a small per-turn
leak below this window's detection floor (e.g. a few MB/turn) is NOT ruled out by 8 turns; it would total at most
a few hundred MB over 194 turns even so, still well inside the cap. This is a memory-headroom read only, not a
substitute for actually running the full length.
