---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: swap-drives-response (GNW thought-swap drive, webapp/swap_drives_chat.py, default-ON) measured with the pre-registered adequate probe LB_SWAP_DRIVE_PROBE (default OFF); S05 land-6-seed lane, re-verified against the repository rather than against the plan's premise
seeds: [42]
runner: research/runners/load_bearing_fraction.py
artifacts:
  - research/findings/raw/_load_bearing/swap_drive_probe/s42/lb.json
  - research/findings/raw/_load_bearing/swap_drive_probe/s42/lb.json.prov.json
  - research/findings/2026-09-23-swap-drives-adequate-probe-s42-smoke-loadbearing-6seed-staged.md
  - research/findings/2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md
---

# swap-drives-response adequate probe: STATUS check — 1/6 seeds verified in the repository, not a 6-seed result

**This is a status/verification pass, not a new run.** S05's instruction described all six
`research/findings/raw/_load_bearing/swap_drive_probe/s*/lb.json` files as already on disk at git SHA `7f90034df`.
That premise does not hold: only `s42/` exists. This finding (a) re-verifies the seed-42 evidence against the raw
artifact rather than against prose, (b) runs the blocking skeptic question S05 required, and (c) reports honestly
that seeds 43/44/100/101/102 are NOT present as committed artifacts, rather than repeating an unverifiable count.
No production default changes here, and this is not a headline GO — see `docs/TERMS.md` ("GO" requires the gate's
own verdict, at the seed count the prereg's headline rule sets, which is 6/6).

## What was checked before writing anything

`find research/findings/raw/_load_bearing/swap_drive_probe -maxdepth 1 -type d` in this worktree returns only
`s42` and `offcheck`. The same check was repeated against every ref that could plausibly hold the other five
seeds: local branches `research/swap-drives-adequate-probe`, `fixround-swap-drives-adequate-probe`, and `main`;
both remotes' copies (`origin/research/swap-drives-adequate-probe`, `gitea/research/swap-drives-adequate-probe`).
`git ls-tree -r --name-only <ref> | grep swap_drive_probe/s` returns only `s42/*` paths on every one of them. This
worktree also has no `.venv` (per-worktree, expected) and no reachable `pool1`/`pool2`/`~/derisk-pool` — this lane
runs no compute (`compute_lane: agents ... no compute`), so nothing here was re-run; only what already exists in
git was inspected.

**A discrepancy this raises, stated plainly rather than resolved by assertion:** the already-committed staged
finding's "Corrected 2026-09-24 (re-review)" section states seeds 44 and 102 "have since returned `regressed`,
`load_bearing: true`" (3/6 as of 00:40), with 43/100/101 re-queued. No artifact backing the 44/102 claim is
reachable from any ref checked above. This may simply mean those results exist only on a pool node's local disk
and were never committed/synced (consistent with the plan's own S03/S04 pool-sync machinery) — this finding does
not allege the prior text was fabricated, only that it is **currently unverifiable from the repository**, which
is the same standard this finding is being held to. The orchestrator (with live pool/queue access this lane does
not have) should confirm `research/findings/raw/_load_bearing/swap_drive_probe/s44/` and `.../s102/` against
`bash tools/pool_sync.sh` before citing that 3/6 figure anywhere further.

## Re-verifying seed 42 against the raw artifact

Read directly from `research/findings/raw/_load_bearing/swap_drive_probe/s42/lb.json` (`per_faculty[0]`),
provenance `git_sha: 7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`, `git_dirty: false`, `source_kind: git_archive`,
`source_manifest_verified_at_start`/`_at_exit` both `true` — a genuine single-revision, integrity-checked build,
consistent with what the prior finding's amendment A2 already established:

| gate | reading | source |
|---|---|---|
| G1 exercised | yes — `swap_drives` fields present on `sw_switch` in both arms | `lb.json.per_faculty[0].contrast_exercised: true` |
| G2 contrast defined | yes — intact carries a swap trace on `sw_open`/`sw_hold` | `swap_state.intact_a.sw_open/sw_hold` |
| G3 null clean | yes — `intact_a == intact_b` on all 3 turns | `null_control_clean: true`, `null_control_clean_switch_turn: true` |
| G4 contrast clean | yes on the compared fields, **pass-by-construction on 3 of 4** (see below) | `contrast_diffs: []`, `contrast_null_diffs: 0` |
| G5 reply changes | yes — 4 diffs on `sw_switch`: `answer`, `swap_drives.swapped/reason/lead` | `diffs` (4 entries), `treatment_diffs: 4` |
| G6 reproduces | yes | `lesion_reproduced: true` |

`load_bearing: true`, `verdict: regressed` for this one seed — this is the runner's own scorer output on a
genuinely single-revision build, and it is consistent with the withdrawn cross-revision smoke per the prior
finding's A2 correction.

## The blocking skeptic question (S05 item 2): does the probe force the swap regardless of the organ?

**No — not by construction of the swap decision itself, but the probe DOES force the *opportunity* for one, and
the *admission test* is host code, both already declared and neither new here.**

- The three turns (`sw_open`→`sw_hold`→`sw_switch`) are fixed by the probe design, so a topic-change is
  *presented* on every seed — that part is by construction, and the prereg says so ("Both questions are ordinary
  recalls... which is where the server prepends the swap lead").
- Whether the intact brain actually swaps is NOT hardcoded: `research/runners/load_bearing_fraction.py`'s
  `LB_SWAP_DRIVE` branch (`_score_swap_drive`, `_swap_drive_score`) reads whatever `swap_drives.swapped` the live
  handler returns; the prereg's own risk section names the specific failure mode this would produce
  (`mismatch_held_no_swap` on a seed) and states "I do not know the outcome in advance" — i.e. the pre-registered
  design anticipated and would have recorded a non-swap seed as a real (not forced) result. On the one seed
  actually measured, the intact arm did swap; that is one data point, not a guarantee about the other five.
- What IS a declared residual, restated here rather than newly found: the eviction/admission test inside the
  swap circuit is gated on a host slot-inequality comparison (`_slot_for`, per the s42 finding's own text), so
  "the spiking mismatch/eviction decision is real... but the general swap circuit's admission test is a host
  comparison" — this is the s42 finding's own honest residual, not something this pass is discovering fresh.
- **Verdict on Q3: not forced, PASS**, conditional on the pre-existing declared host-admission-test residual,
  which this finding does not re-litigate or clear.

## The falsifiable quantities S05 asked for (per seed)

- **Did the intact arm swap?** Seed 42: yes (`swapped: true`, `reason: topic_change_swap`).
- **Did the lead survive downstream composition?** Seed 42: yes — the final `answer` on `sw_switch` is
  `"Setting the held thread aside — On cat, then — the cat eats the fish"`. Two swap-dependent prefixes compose
  without one overwriting the other: `swap_drives`' own lead ("On cat, then — ") and the GNW global-stop organ's
  prefix ("Setting the held thread aside — "), both reading the same underlying `swapped == True` verdict. Both
  vanish under the lesion (lesion `answer`: `"the cat eats the fish"`, no lead, `reason: mismatch_held_no_swap`).
- **Seeds 43/44/100/101/102:** UNDEFINED here — no committed artifact exists to read these quantities from (see
  discrepancy note above; NOT scored as failing, per `tools/lab`'s "UNDEFINED, not a score of 0" rule).

## Label

Per the prereg's own fix-round amendment A2 (three of the four G4-compared fields — `swapped`/`reason`/`lead` on
the no-swap-due turns — are pass-by-construction; only `answer` on those turns is real evidence, and it too read
0 diff), this row is **near-constructed**, not an unqualified robust-core member, exactly as the s42 finding
already states. This finding does not change that label; it re-confirms it against the raw JSON rather than
against the prose describing it.

## Verdict

**1/6 seeds verified. Not GO. Not NOT-GO. Status: partial, awaiting the outstanding 5 seeds.** The pre-registered
headline rule (`2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md`, "Headline rule, fixed now") requires
all six seeds to read LOAD-BEARING before this faculty can be reported as anything stronger than k/6, and k is
currently unverifiable above 1 from this worktree. `LB_SWAP_DRIVE_PROBE` is NOT added to
`PROBE_SETS['adequate']` by this finding — that registry edit belongs to AG-REG's M1 integration (S08/S22) per
S05 item 5, and doing it here (on a branch that does not carry the flip-branch `FIX_ENV`/`PROBE_SETS` refactor)
would conflict with that lane's own work, exactly as S05 warns.

## Staged job lines for the orchestrator (this lane runs no compute)

Five seeds are needed to reach the pre-registered 6-seed decision: **43, 44, 100, 101, 102**. Template command, at
the current `origin/main` HEAD this branch was cut from (`d7b2a2bb56e7af143c61b926e31f36912434b27d`) — substitute
whatever revision the orchestrator actually provisions (e.g. M1) if it differs, and `<seed>` with each of the
five seed values above, one job per seed, each to its own `s<seed>/` output directory (same layout `s42/` already
uses):

```
cd ~/derisk-pool/revisions/d7b2a2bb56e7af143c61b926e31f36912434b27d && LB_SWAP_DRIVE_PROBE=1 SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.load_bearing_fraction --only swap-drives-response --repeats 2 --seed <seed> --out research/findings/raw/_load_bearing/swap_drive_probe/s<seed>/lb.json
```

`mem_gb`: unmeasured this session (this lane runs no compute — `compute_lane: agents ... no compute`). The prior
lane's own staged run for this identical probe+runner declared `memcap 8`; that number is reused here as a
starting point, not re-measured, and should be confirmed against a real peak-RSS reading before being cited as
`--checked`.

If seeds 44/102 genuinely already completed on a pool node (per the discrepancy note above), only 43/100/101
need dispatching — `bash tools/pool_sync.sh` first to check before re-running work that may already exist.

## Honest residuals (carried over, not new)

- The swap-vs-hold decision is the spiking circuit's; topic extraction and the two lead/stop-prefix strings are
  host articulation templates (declared boundary).
- The eviction/admission test (`_slot_for`) is a host slot-inequality comparison, not a spiking competition —
  this is what Q3's answer above rests on, and it is unchanged by this finding.
- Cross-turn continuity of the held topic is a host label (`held_slot`), inherited from #77.
- Option-C pairing: load-bearing here means load-bearing under an opt-in adequate probe. No production default
  changes.
