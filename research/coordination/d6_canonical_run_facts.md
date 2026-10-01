# D6 N=2000 doubled cells: canonical-run facts (TASK 2, research/arcc-b2b-d6-scoring)

Ruling requested by `GAP_CLOSURE_MISSION.md` (2026-09-28 22:29, commit `3bc234f2c`): "which run per doubled
cell is canonical for the D6 score (then re-score if it changes)." This file states what was FOUND, re-scores
both choices, and recommends -- it does not itself change the committed D6 finding
(`research/findings/2026-09-25-d6-capacity-curve-N2000-6seed-CEILING-BETWEEN-50-AND-500.md`, commit
`58316ead8`); the owner rules.

## (a) Does the prereg say which attempt is canonical? No -- it is silent

`research/findings/2026-09-23-d6-capacity-curve-PREREGISTRATION.md` (base text + Amendments A, B, C) defines
per-cell validity (`DEFINED`/`UNDEFINED`, the FREEZE-null/lever/no-probe-write/determinism checks) and the
per-level and per-curve labels, all as a PURE FUNCTION of whatever `(seed, N, arm)` JSON file sits at the
registered path (`s{seed}_N{n}_{arm}.json`). It never anticipates that path being written twice. Searched:
`canonical`, `re-run`/`rerun`, `re-queue`/`requeue`, `duplicate`, `attempt`, `first.land`, `idempot`, `OOM`,
`overwrit` -- the only hit is Amendment A1 ("that job is re-run at the amended revision, and the pre-amendment
file is kept as `s42_N5_HEBB.pre_A1.json`"), which is about a CODE amendment invalidating an old run, not about
two runs of the SAME code landing twice. The determinism check (HEBB_REP, `s{seed}_N{5,50}_HEBB_REP.json`)
tests same-seed reproducibility as a SEPARATE registered arm, not as a rule for resolving an accidental double
run of the SAME arm. **Conclusion: the prereg gives no rule; this is a genuine gap, not a documented policy
this file is re-deriving.**

## What happened, established directly (not inferred)

- `research/queue/pool.queue.claims` (epoch ~1790308090-1790309134, 2026-09-24 23:48 EDT -- matches the
  board's "09-25 00:03" UTC-ish note) re-queued 14 `d6_capacity_curve --worker --n-facts 2000` combos across
  seeds 43/44/100/101, each line's own `--checked` text explaining the root cause: `pool_autodispatch.sh`
  (commit `3308e087c`) under-declared `mem_gb=1` against 4-5 GB actual RSS, so the ORIGINAL dispatch (04:47-
  06:10 that day) "produced NO output file (silently lost to memory pressure)" for these 14 -- that line's own
  belief at requeue time. In fact, for at least 4 of the 14 (`s100`/`s101` x `HEBB`/`FREEZE`), the original
  worker had NOT died -- it was still running under contention and later wrote its own valid output. The
  harvest commit `fe1066f64` (2026-09-25 16:14) committed whichever copy was present at harvest time (the
  "first-landed" run, per the board note); a later `pool-sync` then pulled the re-queued run's output OVER
  those same 4 paths (the "double-run" the board flagged). The overwritten files were restored 2026-09-28
  22:30 to the harvest-commit (`fe1066f64`) content, and the pulled-over (re-queued) copies were kept aside at
  `.claude/worktrees/_d6_duplicate_run_outputs_2026-09-28/research/findings/raw/_d6_capacity_curve/` (4 cells,
  8 files: `s100_N2000_HEBB.json`, `s100_N2000_FREEZE.json`, `s101_N2000_HEBB.json`, `s101_N2000_FREEZE.json`,
  each + `.prov.json`).
- Only these 4 cells have a distinguishable second copy. The other 10 of the 14 requeued combos either never
  had a first-landed file to collide with (the OOM belief was correct for them) or their two attempts produced
  byte-identical output paths with nothing kept to compare -- out of this file's scope; the ruling below is
  about the 4 cells that demonstrably ran twice.
- Currently committed (= "attempt 1", first-landed, restored): `research/findings/raw/_d6_capacity_curve/
  s{100,101}_N2000_{HEBB,FREEZE}.json`. `git diff fe1066f64` on each shows exactly one line changed --
  `cost_acknowledged`, a prose correction to the compute-cost disclosure note, not a scored field. So the
  currently-committed files ARE attempt 1's scored content.
- Kept aside ("attempt 2", the re-queued run that was pulled over and then reverted): the 4 files under
  `_d6_duplicate_run_outputs_2026-09-28/`.

## What differs between attempt 1 and attempt 2, field by field

Compared directly (Python dict equality, not text diff, so key-order never counts as a difference):

| cell | `summary` (recall/abstain/crossed/wrong_word/yn_hit/block_decode_ok/engram_held/mean\_abs\_w/nearmiss/novel/shard-empty fields) | per-probe `taught`/`nearmiss`/`novel` decisions (`cls`/`answer`/`block_decode`/...), all 300 probes | `encode_diag` | cost/timing fields (`encode_s_*`, `query_s_median`, `engram_read_s_*`, `readtime_view_s_per_turn_projected`, `elapsed_s`, `process_time_s`, `node.loadavg_*`, `peak_rss_mb`) |
|---|---|---|---|---|
| `s100_N2000_HEBB` | IDENTICAL | IDENTICAL (0/300 differ) | IDENTICAL | DIFFERENT |
| `s100_N2000_FREEZE` | IDENTICAL | IDENTICAL (0/300 differ) | IDENTICAL | DIFFERENT |
| `s101_N2000_HEBB` | IDENTICAL | IDENTICAL (0/300 differ) | IDENTICAL | DIFFERENT |
| `s101_N2000_FREEZE` | IDENTICAL | IDENTICAL (0/300 differ) | IDENTICAL | DIFFERENT |

Every accuracy/decision field the gate reads (`recall`, `abstain`, `crossed_same_subject`, `crossed_other_fact`,
`wrong_word`, `yn_hit`, `block_decode_ok`, `engram_held`, `mean_abs_w_probed`, `nearmiss_false_accept`,
`novel_false_recall`, and all 300 individual probe decisions per cell) is byte-for-byte identical across the
two attempts, on all 4 cells -- the determinism the prereg's own HEBB_REP check exists to verify held here too,
between two INDEPENDENT worker processes on different pool nodes under different load. Only wall/CPU timing,
RSS and the derived cost fields (`encode_s_median` etc., which scale with pool contention, not with the
substrate) differ -- e.g. `s100_N2000_HEBB`: `encode_s_median` 51.25 s (attempt 1, 31.0 h elapsed under
contention) vs 24.41 s (attempt 2, 21.8 h elapsed); `readtime_view_s_per_turn_projected` 26473 s vs 8165 s.
Both are 800-2600x over the registered `COST_READTIME_MAX_S=10.0` bar and 12-25x over `COST_ENCODE_MAX_S=2.0`.

## (b) Re-scored under both choices with the registered scorer

```
.venv/bin/python -m research.runners.d6_capacity_curve --score \
    --arm-dir research/findings/raw/_d6_capacity_curve \
    --json <attempt1.json>        # attempt 1: the currently-committed, restored files
.venv/bin/python -m research.runners.d6_capacity_curve --score \
    --arm-dir <scratch-dir-with-the-4-cells-swapped-to-attempt-2> \
    --json <attempt2.json>        # attempt 2: the 4 doubled cells replaced with the kept-aside re-queued copies, everything else unchanged
```

Diffed the two `--score` outputs directly (every `level_label_*`, `parity_*`, and `curve_*` key compared for
equality, not just eyeballed):

| output field | attempt 1 (committed) | attempt 2 (re-queued copies swapped in) | changed? |
|---|---|---|---|
| `level_label_HEBB` (all N) | RECALL-HOLDS x4 | RECALL-HOLDS x4 | no |
| `level_label_COPY` (all N) | RECALL-HOLDS x4 | RECALL-HOLDS x4 | no |
| `parity_HEBB_vs_COPY` (all N) | PARITY-BY-CONSTRUCTION x4 | PARITY-BY-CONSTRUCTION x4 | no |
| `curve_HEBB` | RECALL-HOLDS-TO-2000 | RECALL-HOLDS-TO-2000 | no |
| `curve_COPY` | RECALL-HOLDS-TO-2000 | RECALL-HOLDS-TO-2000 | no |
| `level_label_HEBB_cost` | HOLDS,HOLDS,FAILS,FAILS (N=5,50,500,2000) | HOLDS,HOLDS,FAILS,FAILS | no |
| `level_label_COPY_cost` | HOLDS,HOLDS,FAILS,FAILS | HOLDS,HOLDS,FAILS,FAILS | no |
| `curve_HEBB_cost` | CEILING-BETWEEN-50-AND-500 | CEILING-BETWEEN-50-AND-500 | no |
| `curve_COPY_cost` | CEILING-BETWEEN-50-AND-500 | CEILING-BETWEEN-50-AND-500 | no |

**The verdict does not change under either choice.** The only bytes that differ between the two full `--score`
JSON outputs are the raw timing numbers inside the 4 doubled cells' `summary`/`node` blocks (confirmed by a
plain `diff`: every changed line is `encode_s_*`, `query_s_median`, `engram_read_*`, `readtime_view_s_per_turn_
projected`, `elapsed_s`, `process_time_s`, `peak_rss_mb`, `node.loadavg_*`, `node.hostname`, or the `arm_dir`
path itself) -- no `level_label_*`, `parity_*`, or `curve_*` key differs by even one character. This holds
because (i) the cost gate's thresholds are cleared or missed by 1-3 orders of magnitude in BOTH attempts (never
close to the 2.0 s / 10.0 s bars, so contention-driven timing noise cannot flip the label), and (ii) the
accuracy fields the recall/parity gates read are exactly reproduced between the two independent worker runs.

## Recommendation

Because no prereg text governs this and both scored choices are provably equivalent at the verdict level, this
file recommends the rule the prereg's own spirit already implies (DEFINED cells share the fact/probe/threshold
hash across arms; nothing in the design treats compute time as part of the substrate result): **first VALID
landing is canonical** -- the run whose output first satisfied the cell's own validity checks (non-error, exact
N taught, matching hashes), independent of which copy a later sync happened to leave on disk. That is also
what is currently committed (attempt 1, restored 2026-09-28) and what the merged finding `58316ead8` already
scored, so adopting this rule requires no change to it. A "last write wins" rule would be operationally worse
going forward: it rewards whichever attempt a sync race happens to touch last, not which one actually finished
first and validly, and would have picked the SLOWER, more heavily-contended run in this exact case (attempt 2's
lower elapsed time reflects less contention, not a truer measurement -- either could have been the later
write). No change to `research/findings/2026-09-25-d6-capacity-curve-N2000-6seed-CEILING-BETWEEN-50-AND-500.md`
is needed: it already reflects the canonical (first-landed) attempt, and re-scoring against the alternative
choice reproduces every one of its labels exactly.

## What this does not settle

- The other 10 of the 14 originally-requeued combos (seeds 43, 44, and the `COPY` arm at 100/101) are not
  addressed here -- no second copy of any of them was found kept anywhere searched
  (`.claude/worktrees/_d6_duplicate_run_outputs_2026-09-28/` holds only the 4 cells above), so there is nothing
  to compare for them; their committed files are presumed single-attempt.
- This file does not add a general "canonical attempt" rule to the D6 prereg itself (that would be an
  amendment, and the owner rules on whether one is warranted); it answers the specific ruling request on the
  board for the 4 cells that are demonstrably doubled.
- Root-cause prevention (the `mem_gb` under-declaration that caused the false-OOM belief and the requeue in
  the first place) is already logged as fixed per the requeue line's own `--checked` text
  (`pool_autodispatch.sh`'s headroom-reservation fix); not re-verified here.

## Artifacts

- `research/findings/raw/_d6_capacity_curve/s{100,101}_N2000_{HEBB,FREEZE}.json` (+ `.prov.json`) -- attempt 1,
  currently committed, restored 2026-09-28 to `fe1066f64`'s content plus one corrected prose field.
- `.claude/worktrees/_d6_duplicate_run_outputs_2026-09-28/research/findings/raw/_d6_capacity_curve/
  s{100,101}_N2000_{HEBB,FREEZE}.json` (+ `.prov.json`) -- attempt 2, kept aside, NOT committed to the raw
  data directory (scoring against it here used a local scratch copy of the arm-dir, not a change to the
  committed tree).
- The two full `--score` JSON outputs and their `diff` are reproducible with the two commands quoted above;
  not committed here as artifacts (this file's tables are the record).

## RULING (Claude, 2026-09-29)

Adopted: the first valid landing is canonical for each doubled cell. The prereg is silent, the duplicate attempts are
byte-identical on every scored field (only pool-contention timing differs), and the registered scorer returns the same
verdict keys under either choice, so finding 58316ead8 stands unchanged. The kept duplicates stay in
`.claude/worktrees/_d6_duplicate_run_outputs_2026-09-28/` as evidence, never scored.
