---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: per-cell sidecar validity audit of the B2a production-default load-bearing battery (M1
  9db7613296c3d02a161b36fb15da3983188b7902) against its own pre-registration, applying B2b Amendment 1.2's
  validity rule (lb_shard.py aggregate does not check it)
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-production-default-battery-B2a-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json
  - research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json
  - research/findings/raw/_load_bearing/_shards/flipdefaults-thin/aggregate.json
  - research/findings/raw/_b2a0924_cell_validity.json
verdict: FAIL -- R2 (complete) fails. 23 of 28 coverable faculties have an INVALID seed-102 shard, run from a
  dirty, non-pinned local worktree ("b2a-local") instead of the pre-registered M1 git_archive; that is not
  "missing", it is off-registration, and per R2's own text a shard like this makes the faculty UNDEFINED, never a
  pass. R1 (no regression against the flip battery) is UNDEFINED for those same 23 faculties (their true M1
  n_load_bearing over 6 valid seeds is not yet established) and PASSES for the 5 faculties whose full 6-seed run
  is clean. No default flip is licensed by this battery.
---

# B2a scoring: R2 FAILS on a dirty, off-registration seed-102 shard; R1 UNDEFINED where it does

## Setup

Copied the pool-produced shard tree `research/findings/raw/_load_bearing/_shards/b2a0924/` (186 `lb.json` +
sidecars, 6 seeds x 31 sharded rows) into this worktree without moving or deleting anything in the primary
checkout, then re-ran `tools/lb_shard.py aggregate --tag b2a0924 --seeds 42 43 44 100 101 102` here. The result is
byte-identical (via `python3 -m json.tool` on both sides) to the primary checkout's
`research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json`: `robust_core_n 24, union_n 25, mean_fraction
0.949, incomplete_faculties []`. Reproducibility is not in question; what is in question is whether every cell
`lb_shard.py aggregate` folded into that number actually measured M1, which the aggregator itself does not check.

## R2 (complete) — FAIL

`lb_shard.py aggregate` (`tools/lb_shard.py`) computes `dirty_seeds` only from each shard's OWN
`null_control_clean`/`UNRELIABLE` fields, and B2a's own prereg has no per-cell provenance rule (B2b's own
Amendment 1.9 says so explicitly: "B2a's own prereg has no per-cell SHA rule"). Applying B2b Amendment 1.2's rule
instead — every `lb.json.prov.json` and every arm sidecar (`intact_a_*`, `intact_b_*`, `lesion_*[.repN]`) must
record `git_sha` M1 **in full**, `source_kind` `git_archive`, both `source_manifest_verified_at_start/_at_exit`
true, `env.SIM_BACKEND` `numpy`, and no stray `BRAIN_*` key — to all 186 cells (script + compact table:
`research/findings/raw/_b2a0924_cell_validity.json`) finds:

- **145 of 168 coverable (faculty, seed) cells are valid**; **23 are invalid**, and every one of the 23 is at
  **seed 102**.
- The 23 invalid cells' `lb.json.prov.json` (and their arm sidecars) record `git_sha "9db761329"` (the SHORT form,
  not M1 in full), `source_kind: null` (not `git_archive`), `git_dirty: true`, and `argv[0]` under
  `/home/dant123/Projects/sim/.claude/worktrees/b2a-local/...` — a local, uncommitted-changes worktree, not the
  pinned pool/AWS `~/derisk-pool/revisions/<M1>` mechanism the prereg's "Run (fixed)" section specifies. They ran
  early (started between 14:27 and 17:49 on 2026-09-24), well before the properly pinned pool runs for the same
  faculties at every other seed (e.g. `wm-binding-advanced` s100/s101 started 23:05/23:46 the same day, `git_sha`
  M1 in full, `source_kind` `git_archive`).
- Only **5 of 28 coverable faculties are fully valid across all six seeds**: `bg-action-selection`,
  `confidence-forthcomingness`, `open-ended-generation`, `self-initiated-utterance`,
  `vision-identity-spiking-hmax`. The other **23** each have exactly one invalid seed, always 102:
  `affect-coloring`, `affect-drives-response`, `affect-marker-spiking-wta`, `common-ground-drives`,
  `comprehension-learned-animacy-cue`, `comprehension-learned-verb-selects`, `comprehension-monitor`,
  `curiosity-followup`, `da-gated-encoding`, `da-mode-drives-response`, `discourse-register`, `episodic-memory`,
  `gnw-multistep-deliberation`, `metacog-monitor`, `noncontradiction-gate`, `pragmatic-implicature`,
  `prospective-memory`, `reconsolidation`, `source-provenance-honesty`, `surprise-monitor`,
  `swap-drives-response`, `wm-binding-advanced`, `worldmodel-forward` (full per-cell list:
  `research/findings/raw/_b2a0924_cell_validity.json` -> `summary.invalid_faculty_seed_pairs`).
- The dirty local run's *substantive* verdicts (via each `lb.json`'s own `null_control_clean`/verdict fields) are
  not obviously wrong — `lb_shard.py`'s own dirty-seed check reports zero dirty seeds project-wide, and the naive
  per-seed fraction at s102 (0.9615...) ties the other high seeds rather than reading low <!--derived-->. That is
  exactly the failure mode the sidecar rule exists to catch: a cell can look clean on its output and still not be
  a measurement of M1, because `git_dirty: true` means the worktree carried uncommitted changes when the shard
  ran, and `source_kind: null` means it never went through the git-archive pinning the prereg registered.
- Per R2's own text — "a missing or UNRELIABLE shard makes that faculty UNDEFINED, never 0, never a pass" — an
  off-registration shard is not weaker than a missing one; it is the identical failure the s100/open-ended-
  generation torn-queue-line incident already established (`research/FAILURE_LOG.md`, 2026-09-24 row: that cell's
  `git_sha` read `5d10431c6` in `~/derisk-pool/sim`, not M1, and was treated as invalid, moved aside and re-run
  pinned — not scored as delivered). The 23 seed-102 cells here were never caught or re-run. **R2 is not met: 23
  of 28 coverable faculties are incomplete.**

## The one cell the task named — s100/open-ended-generation

Confirmed **VALID**: `research/findings/raw/_load_bearing/_shards/b2a0924/s100/open-ended-generation/lb.json.prov.json`
records `git_sha` M1 in full, `source_kind` `git_archive`, `source_manifest_verified_at_start` and `_at_exit` both
true, `env.SIM_BACKEND` `numpy`, and no `BRAIN_*` key; `started` is `2026-09-24T23:24:31`, consistent with a later
pinned re-run landing after the original off-revision attempt was moved aside per the FAILURE_LOG row. This is the
pinned re-run described in the task, correctly in the tree, and it is one of the 145 valid cells above.

## R1 (no regression against the flip battery) — PASS on 5 faculties, UNDEFINED on 23

Per-faculty `n_load_bearing` in `research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json` against
`research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json` shows **no faculty lower**
across all 28 coverable rows (same 28 faculties in both, no faculty missing either side) — on the numbers as
computed, R1's literal comparison passes everywhere.
But R1 asks whether **M1's** `n_load_bearing` is not lower, and for the 23 faculties above, one of the six inputs
to that count is not a measurement of M1 (previous section). The true M1 count for those 23 is only bounded, not
known, until seed 102 is re-run pinned, so R1 is **UNDEFINED** for them (not FAIL: nothing shows an actual
regression, and the same dirty run's own verdict was "regressed"/load-bearing on every one of the 22 non-zero
ones, i.e. even the untrusted value points toward "no regression" — it just cannot be certified). R1 **PASSES**
outright only for the 5 fully-valid faculties (`bg-action-selection`, `confidence-forthcomingness`,
`open-ended-generation`, `self-initiated-utterance`, `vision-identity-spiking-hmax`), each 6/6 in both battery
aggregates.

## Reported pair (not gated)

As computed from the shard tree, six seeds, no per-cell validity applied (this is what `lb_shard.py aggregate`
outputs, and what the task's pre-registration asks to be reported beside the flip battery's thin headline):

- **b2a0924 (this battery, naive)**: robust core 24, union 25, mean fraction 0.949, SD 0.018.
- **Flip battery thin headline** (`research/findings/raw/_load_bearing/_shards/flipdefaults-thin/aggregate.json`,
  the owner-ratified pair from the prereg):
  mean fraction 0.603, robust core 15 (union 16, SD 0.018).

Caveat that must travel with the 24/0.949 pair: of the 24 robust-core faculties in that count, only 5 have a
fully valid six-seed measurement; the other 19 carry the invalid seed-102 cell described above. The pair is
reported here exactly as pre-registered ("reported, not gated"), not as a certified robust-core-24 result.

## What this licenses

**No default flip is made here** (none was in scope). This battery does not license B2b's own no-regression
comparison to treat B2a's 24/0.949 as a clean baseline for all 28 faculties: B2b Amendment 1.9 already requires
"a B2a cell enters the comparison only if its sidecars meet A1.2 rules 1 to 3 with M1 ... in place of F" — the 23
cells named above fail that rule and must be excluded (or B2a's seed 102 must be re-run pinned first) before
B2b's per-faculty comparison against B2a can treat those 23 faculties as decided. Until then, B2b's comparison
with B2a is well-defined only for the 5 fully-valid faculties; for the other 23 it is pending a clean B2a seed-102
re-run, exactly as A1.9 anticipates for an undefined B2a cell.

## Next action (not performed here — out of this task's scope)

Re-run seed 102 for the 23 named faculties from a properly provisioned, clean M1 git-archive checkout (the same
mechanism the other 145 cells used), landing at the canonical shard path; then re-run
`tools/lb_shard.py aggregate --tag b2a0924 --seeds 42 43 44 100 101 102` and re-check R1/R2. A `research/FAILURE_LOG.md`
row is added alongside this finding naming the class (a local, dirty, non-`git_archive` worktree run entering a
pinned-revision battery's shard tree undetected by `lb_shard.py aggregate`'s own dirty-seed check).
