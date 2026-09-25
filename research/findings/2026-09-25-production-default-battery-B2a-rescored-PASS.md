---
type: finding
status: live
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: re-score of the B2a production-default load-bearing battery (M1 9db7613296c3d02a161b36fb15da3983188b7902)
  against its own pre-registration, after the seed-102 provenance defect
  research/findings/2026-09-25-production-default-battery-B2a-FAIL.md found was closed by a per-cell provenance
  gate in `tools/lb_shard.py aggregate --pin` (merged 0d2642ee6) and the 24 dirty local-worktree cells (23 coverable
  faculties plus one-brain-substrate) were re-run pinned to M1 via git_archive
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-production-default-battery-B2a-PREREGISTRATION.md
supersedes_status_of: research/findings/2026-09-25-production-default-battery-B2a-FAIL.md
artifacts:
  - research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json
  - research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json
  - research/findings/raw/_load_bearing/_shards/flipdefaults-thin/aggregate.json
  - research/findings/raw/_load_bearing/_b2a0924_rescore_0925/rescore_verification.json
verdict: R1 PASS -- no faculty's n_load_bearing is lower than the flip-battery reference, 28/28 coverable faculties
  compared, 0 regressions. R2 PASS -- 168/168 coverable cells verified against the M1 pin (0 invalid, 6
  open-ended-generation cells admitted via the covered-by-parent rule), 0 incomplete faculties. Independently
  re-derived in this task: a read-only re-run of `lb_shard.py aggregate --pin` against a copy of the shard tree
  reproduced the primary checkout's aggregate byte-for-byte. No default flip is licensed by this battery (none was
  in scope, matching the FAIL finding and its own prereg).
---

# B2a re-scored: R1 PASS, R2 PASS after the seed-102 provenance fix

## Why this document exists, and what it does not do

`research/findings/2026-09-25-production-default-battery-B2a-FAIL.md` scored B2a's R2 (complete) criterion FAIL:
23 of 28 coverable faculties had a seed-102 cell that ran from a dirty, non-pinned local worktree instead of the
pre-registered M1 `git_archive` mechanism, undetected by `lb_shard.py aggregate`'s own dirty-seed check. That
finding's verdict was correct for what it inspected at the time and is **not edited here** -- this is a follow-up,
re-scoring the same criteria against what has changed since: a per-cell provenance gate was built into
`lb_shard.py aggregate` itself, the 24 dirty cells were re-run pinned, and an open-ended-generation sidecar gap
(unrelated to the dirty-worktree defect, but also excluding cells from a pinned aggregate) was separately fixed.
This document re-runs the check independently and reports what it finds.

## What changed since the FAIL finding, and what did not

**Changed:**
1. `tools/lb_shard.py aggregate` gained a `--pin <sha>` mode (`bc0a5639f` / merged `0d2642ee6`) that checks every
   coverable cell's sidecars against B2b Amendment 1.2's rule (git_sha in full, `source_kind=git_archive`, both
   manifest-verified flags, `env.SIM_BACKEND=numpy`, no stray `BRAIN_*` key) and excludes/reports any cell that
   fails it, rather than silently folding it in.
2. The 24 dirty local-worktree cells at seed 102 (the 23 coverable faculties the FAIL finding named, plus the
   non-coverable `one-brain-substrate` mechanism-only row) were moved to
   `research/findings/raw/_load_bearing/_b2a0924_attempt1/` and re-run on the pool pinned to M1 via `git_archive`.
3. Independently of the dirty-worktree defect, every seed's `open-ended-generation` cell writes
   `oed_distributional*.json` from the parent process at a point the provenance door's argv scan never saw, so it
   had no sidecar of its own and would have been excluded by rule 1 above on every seed, in every B2a/B2b battery.
   Fixed by `declare_output()` (`5ce76fdfd`) plus a covered-by-parent admission rule requiring the file's content
   to match its cell's own `lb.json` entry and its mtime to fall inside the parent run's window (`56c0465c9`,
   `e755b6791`).
4. The shard tree's `aggregate.json` was re-run with `--pin` at 2026-09-25 03:17 (mtime-stamped), reading
   `provenance.status: "verified"`, `n_cells_checked: 168`, `n_valid: 168`, `n_invalid: 0`, `n_covered_by_parent: 6`,
   `incomplete_faculties: []`.

**Did not change:** the underlying `lb.json` verdict data for the 145 cells the FAIL finding already found valid;
the `flipdefaults-adequate` R1 reference battery (still pre-provenance-gate, still the pre-registered comparison
target, not re-run -- see the caveat below); the FAIL finding's own text; no production default.

## Independent re-verification performed in this task

Working in an isolated worktree, with the primary checkout treated as read-only (nothing there was moved, deleted
or written): `tools/lb_shard.py aggregate --tag b2a0924 --seeds 42 43 44 100 101 102 --pin
9db7613296c3d02a161b36fb15da3983188b7902 --base <primary-checkout>/research/findings/raw/_load_bearing/_shards
--out <scratch>/aggregate_rescore.json`. The result is byte-identical (`diff`, and a full Python dict-equality
check) to the primary checkout's own aggregate at
`research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json` (mtime 2026-09-25 03:17:43, uncommitted at
the time this task started -- the git-tracked copy on `main` still held the pre-provenance-gate version committed
by `50916e654` alongside the FAIL finding, with no `provenance` key at all). This branch's commit of that file
brings the independently-reproduced, pin-verified aggregate into a committed state; it replaces the older
git-tracked copy, which is superseded by the pinned re-run rather than wrong for what it was.
Full re-derivation, per-faculty comparison and the compact per-cell provenance table:
`research/findings/raw/_load_bearing/_b2a0924_rescore_0925/rescore_verification.json`.

## R2 (complete) -- PASS

Applying the same B2b Amendment 1.2 rule the FAIL finding applied, now run by `lb_shard.py aggregate --pin` itself
rather than by hand: all 168 coverable (faculty, seed) cells are valid. 0 are invalid. 6 are admitted via the
covered-by-parent rule (`s42/open-ended-generation`, `s43/open-ended-generation`, `s44/open-ended-generation`,
`s100/open-ended-generation`, `s101/open-ended-generation`, `s102/open-ended-generation` -- every seed's
open-ended-generation cell, not a subset), each verified against its own `lb.json` entry's content and its parent
run's window, not merely assumed clean. `incomplete_faculties` is empty; no faculty is missing a seed. None of the
23 faculties the FAIL finding flagged remain invalid at seed 102 -- each now has 6/6 valid seeds (per-faculty
table below).

## R1 (no regression against the flip battery) -- PASS

Per-faculty `n_load_bearing` in the re-verified `b2a0924` aggregate against
`research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json`: 28/28 coverable faculties
compared (same 28 on both sides), 0 faculties lower. This matches the FAIL finding's own literal-comparison result
(it too found no faculty lower across all 28), but where the FAIL finding could only certify this for 5 of 28
faculties (the other 23 rested on an invalid cell), it now holds for all 28, because all 28 are now built entirely
from valid, pin-verified cells.

### Caveat that travels with the R1 comparison (stated, not fixed here)

The reference battery itself,
`research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/aggregate.json`, predates git_archive pinning:
every one of its 168 coverable cells' sidecars records `git_sha: "unknown"`, `source_kind: null` (confirmed
directly, e.g.
`research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/s42/affect-coloring/lb.json.prov.json`), and
re-running `lb_shard.py aggregate --pin
9db7613296c3d02a161b36fb15da3983188b7902` against that shard tree in this task reads `n_valid: 0, n_invalid: 168`
-- every cell fails the pin check, because none of them were ever pinned to any revision at all, let alone M1. This
was checked in this task and is reported here as a known, pre-existing gap in the R1 reference, not corrected: the
prereg registers R1 against this exact file, and re-deriving `flipdefaults-adequate` at its own revision
(`bd391aa31`, not M1) is out of this task's scope.

## Reported pair (not gated, as pre-registered)

- **b2a0924** (this battery, now pin-verified): robust core 24, union 25, mean fraction 0.949, SD 0.018.
- **Flip battery thin headline**
  (`research/findings/raw/_load_bearing/_shards/flipdefaults-thin/aggregate.json`, the owner-ratified pair from the prereg):
  mean fraction 0.603, robust core 15 (union 16, SD 0.018).

Unlike the FAIL finding, this pair no longer carries the caveat that only 5 of the 24 robust-core faculties have a
fully valid six-seed measurement -- all 24 now do.

## Per-faculty table (28 coverable faculties)

| faculty | n_lb b2a0924 | n_lb flip-adequate ref | regression | seed-102 was invalid (FAIL finding) | now |
|---|---|---|---|---|---|
| affect-coloring | 6 | 6 | no | yes | 6/6 valid |
| affect-drives-response | 6 | 6 | no | yes | 6/6 valid |
| affect-marker-spiking-wta | 4 | 4 | no | yes | 6/6 valid |
| bg-action-selection | 6 | 6 | no | no (always valid) | 6/6 valid |
| common-ground-drives | 6 | 6 | no | yes | 6/6 valid |
| comprehension-learned-animacy-cue | 6 | 6 | no | yes | 6/6 valid |
| comprehension-learned-verb-selects | 6 | 6 | no | yes | 6/6 valid |
| comprehension-monitor | 6 | 6 | no | yes | 6/6 valid |
| confidence-forthcomingness | 6 | 6 | no | no (always valid) | 6/6 valid |
| curiosity-followup | 6 | 6 | no | yes | 6/6 valid |
| da-gated-encoding | 0 | 0 | no | yes | 6/6 valid |
| da-mode-drives-response | 6 | 6 | no | yes | 6/6 valid |
| discourse-register | 6 | 6 | no | yes | 6/6 valid |
| episodic-memory | 6 | 6 | no | yes | 6/6 valid |
| gnw-multistep-deliberation | 6 | 6 | no | yes | 6/6 valid |
| metacog-monitor | 6 | 6 | no | yes | 6/6 valid |
| noncontradiction-gate | 6 | 6 | no | yes | 6/6 valid |
| open-ended-generation | 6 | 6 | no | no (always valid) | 6/6 valid |
| pragmatic-implicature | 6 | 6 | no | yes | 6/6 valid |
| prospective-memory | 6 | 6 | no | yes | 6/6 valid |
| reconsolidation | 6 | 6 | no | yes | 6/6 valid |
| self-initiated-utterance | 6 | 6 | no | no (always valid) | 6/6 valid |
| source-provenance-honesty | 6 | 6 | no | yes | 6/6 valid |
| surprise-monitor | 6 | 6 | no | yes | 6/6 valid |
| swap-drives-response | 0 | 0 | no | yes | 6/6 valid |
| vision-identity-spiking-hmax | 6 | 6 | no | no (always valid) | 6/6 valid |
| wm-binding-advanced | 0 | 0 | no | yes | 6/6 valid |
| worldmodel-forward | 6 | 6 | no | yes | 6/6 valid |

`da-gated-encoding`, `swap-drives-response` and `wm-binding-advanced` are load-bearing on 0/6 seeds in both
batteries (identical, so no regression); `affect-marker-spiking-wta` is 4/6 in both. These four are the reason
`union_n` (25) and `robust_core_n` (24) are below 28, unrelated to the provenance question this document scores.

## What this licenses

**No default flip is made or implied here** (none was in scope for B2a's own prereg or for this task). What this
does license: B2b's own Amendment 1.9 no-regression comparison against B2a, which previously could treat only the
5 always-valid faculties as decided (the FAIL finding's "what this licenses" section), can now treat all 28 B2a
faculties as a clean baseline, since all 28 meet A1.2's per-cell validity rule. The committed
`research/findings/raw/_load_bearing/_shards/b2a0924/aggregate.json` on this branch is the artifact that comparison
should read.

## Next action

None generated by this task specifically (the FAIL finding's own "next action" -- re-run seed 102 pinned and
re-aggregate -- is the work this document confirms landed). B2b's per-faculty comparison against B2a can now
proceed over the full 28-faculty set.
