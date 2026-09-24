---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: swap-drives-response (GNW thought-swap drive, webapp/swap_drives_chat.py, default-ON) measured with the pre-registered adequate probe LB_SWAP_DRIVE_PROBE (default OFF)
seeds: [42]
runner: research/runners/load_bearing_fraction.py
artifacts:
  - research/findings/raw/_load_bearing/swap_drive_probe/s42/lb.json
  - research/findings/raw/_load_bearing/swap_drive_probe/s42/intact_a_sw_open_sw_hold_sw_switch.json
  - research/findings/raw/_load_bearing/swap_drive_probe/s42/lesion_swap_drives_response.json
  - research/findings/raw/_load_bearing/swap_drive_probe/offcheck/offcheck.json
---

# swap-drives adequate probe: the seed-42 smoke meets every pre-registered gate; 6 seeds are staged (2026-09-23)

This is **one seed**, a smoke. It is not a robust-core claim. Five more seeds are staged on the pool (see
"Staged" below).

Pre-registration: `2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md` (commit `4d203f584`, amendment A1 in
`ebe184fbc`, amendment A2 below — the s42 provenance correction this section now describes).

## What the smoke measured

**Provenance correction (A2).** The first cut of this section described a *local* build of "four arms... under
`tools/memcap.sh 12`", rescored after the A1 fix. That description was wrong for two of the four arms. Per
`.prov.json`: `intact_a` and `intact_b` were built by worker subprocesses at `4d203f584` (19:01/19:15, before A1),
while `lesion` and its rep0 were built by worker subprocesses at `7f90034df` (19:27/19:41, after A1 + the
origin/main merge) — one long-lived parent process kept running while the worktree `HEAD` moved underneath it
(same `SIM_RUN_ID`, different `git_sha` per arm). So the original smoke's treatment comparison (intact vs lesion)
was **cross-revision**, not the single-revision rebuild the finding claimed. A review of the code diff between
`4d203f584` and `7f90034df` found only an import fix, a selftest helper and a no-op merge on this path, so no
confound was actually measured by it — but the finding must say the smoke was cross-revision, and it did not.

**Fix.** Seed 42 is superseded here by the single-revision rebuild that came back from the six-seed pool batch
(same command, same isolated revision as seeds 43/44/100/101/102 — see "Staged" below): all four arms
(`intact_a`, `intact_b`, `lesion`, `lesion.rep0`) were built by `research/runners/onebrain_regression_battery.py
--worker` subprocesses sharing one `SIM_RUN_ID` (`1790209256-3647551`), each `.prov.json` reading `git_sha:
7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`, `git_dirty: false`, `source_kind: git_archive`, with
`source_manifest_verified_at_start`/`_at_exit` both `true` against the same `source_manifest_sha256`
(`e56f6b118de9cf3f8fc120adbc1de3a566229b237c969735c81ae1eb4df39fac`) — a genuinely single-revision, verified
build. The replies, swap states and verdict are **byte-identical** to the withdrawn cross-revision smoke (the
only diff in `lb.json` is a `cuda_visible_devices` reporting field, `""` vs `null`), so the original conclusion
was not, in fact, confounded — but that could only be known once the single-revision result existed. The
committed artifacts below (and the table) are now this corrected, single-revision run.

Artifact: `research/findings/raw/_load_bearing/swap_drive_probe/s42/lb.json`.

| turn | intact reply | lesion reply | swap state (intact / lesion) |
|---|---|---|---|
| `sw_open` | the dog chases the cat | same | first_thought / first_thought |
| `sw_hold` | As for it — the dog chases the cat | same | same_topic_hold / same_topic_hold |
| `sw_switch` | Setting the held thread aside — On cat, then — the cat eats the fish | the cat eats the fish | topic_change_swap / mismatch_held_no_swap |

Gate readings (`_swap_drive_score`):
- G1 exercised: yes.
- G2 contrast defined: yes.
- G3 null clean on all three turns: 0 diffs.
- G4 contrast clean: 0 diffs on `sw_open` and `sw_hold` — **but declare which of the four fields could actually
  have failed.** On these no-swap-due turns the host template sets `reason='first_thought'`/`'same_topic_hold'`
  and `lead=''` deterministically whenever `swapped` is `False`, and `swapped` cannot read `True` when the
  proposed topic equals the incumbent — so `swapped`/`reason`/`lead` are **pass-by-construction** here; only
  `answer` is a field that could have shown a real lesion effect. It did not (0 diff on `answer` too). The
  mechanism itself is **not** unaffected by the lesion on these turns, only the reply is: `mm_peak` drops
  0.0556→0.0056 on `sw_open` and 0.0667→0.0111 on `sw_hold` (both in `swap_state` in `lb.json`), and `boost_max`
  falls with it. The correct statement is **"no reply-level change"** on the contrast turns, not "the lesion
  changes nothing" — the earlier draft of this finding used the latter, over-broad phrasing.
- G5 reply changed: yes. 4 diffs: `answer`, `swapped`, `reason`, `lead`. **What G5 actually tests:** once the
  intact arm swaps (`swapped=True`), the lead string is a deterministic host template of that one boolean, and
  the lesion pins the mismatch-detector's proposal drive to (near-)zero (`boost_max` 0.0111 vs 0.16 intact), so
  once G4's contrast holds, a swap-vs-no-swap reply difference on `sw_switch` is close to guaranteed by
  construction too. The genuinely falsifiable content G5 carries is (a) whether the *intact* integrated brain
  swaps at all on a given seed (it can read `mismatch_held_no_swap` instead — the isolated de-risk's 6/6 was not
  measured inside this battery) and (b) whether the lead or the downstream GNW stop-prefix survives to the final
  `answer` rather than being overwritten by another organ. On this measured seed the spiking mismatch/eviction
  decision is real (a topic-change proposal against an occupied slot), but the general swap circuit's admission
  test is a host slot-inequality (`_slot_for`); that comparison, not new evidence here, is what the swap verdict
  itself is built on.
- G6 reproduced: yes.

Verdict: `regressed`, `load_bearing: true`.

The intact reply carries two swap-dependent prefixes:
- "On cat, then — " is the swap-drives lead.
- "Setting the held thread aside — " is the GNW global-stop organ. It reads `swapped == True` as a hard topic break.

Both vanish under the lesion, because both ride the same neural swap verdict. The lead is `swap_drives`' own output.
The stop prefix is a downstream consumer of the same verdict. The credited effect is "the swap verdict changes the
reply", not "the lead string alone".

The flag state (`LB_SWAP_DRIVE_PROBE=1`) is not in the automatic provenance `env` capture (it records only
`SIM_BACKEND`/`SIM_RUN_ID`), but it is recorded in the per-faculty record itself: `lb.json`'s
`per_faculty[0].swap_drive_probe` reads `true`, and its `note` field names the flag. ON and OFF artifacts can be
told apart from `lb.json` alone without relying on the env capture.

## Byte-identical OFF

Artifact: `research/findings/raw/_load_bearing/swap_drive_probe/offcheck/offcheck.json`.
- **Data.** With `LB_SWAP_DRIVE_PROBE` unset, the flag-off `swap-drives-response` run at seed 42 was built on pool42
  in two isolated revision directories: the pinned pre-change SHA `f35196e66`, and `4d203f584`, whose only diff is
  this change. All four arm files are sha256-identical between them, and so is the per-faculty row (still
  `not-exercised`, as before).
- **Static.** `PROBE_TURNS`, `FACULTY_PROBES`, `FACULTY_LESIONS`, and `turn_group()` for all 49 pinned labels are
  `==` between the pinned tree and the branch HEAD.
- **Sensitivity.** The flag-ON row differs: its turn is `sw_switch`.
- **Coverage limit.** The flag-off data check was run at `4d203f584`, not at the final merge `7f90034df`. Between
  them the branch added a missing import, a selftest helper, and a merge of origin/main. None of these changes
  `load_bearing_fraction.py`'s behaviour with the flag off, and main did not touch the battery files.

## Staged: the six-seed run

- Six pool jobs were queued, one per seed (42 43 44 100 101 102), each running
  `--only swap-drives-response --repeats 2 --seed <s>` with `LB_SWAP_DRIVE_PROBE=1`, numpy backend and
  `memcap 8`, in the isolated revision directory `~/derisk-pool/revisions/7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`
  on pool41 and pool42.
- **Seed 42 has returned** (pool42) and is the corrected result reported above: `regressed`,
  `load_bearing: true`, byte-identical (modulo a reporting field) to the withdrawn cross-revision smoke.
- ~~Seeds 43, 44, 100, 101, 102 are still valid and in flight~~ **Corrected 2026-09-24 (re-review):** that line
  was false when committed. At the pinned revision, seeds 44 and 102 have since returned `regressed`,
  `load_bearing: true` (with seed 42, 3/6). Seeds 43 and 100 had already FINISHED with `arm-build-failed`
  (`load_bearing: null`) during the 2026-09-23 20:50 pool thrash (a missing lesion arm on s43, a missing intact_b
  arm on s100), and seed 101 was killed in the same episode. All three were re-queued 2026-09-24 at the same
  revision `7f90034df`. They are UNDEFINED until they return; none is counted.
- Each seed writes to its own directory, `research/findings/raw/_load_bearing/swap_drive_probe/s<seed>/`.
- The pre-registered headline rule applies: all six seeds must read LOAD-BEARING. Anything less is reported as k/6,
  naming each failing seed's verdict. As of 2026-09-24 00:40: 3/6 load-bearing (42, 44, 102); 43, 100, 101
  re-queued (UNDEFINED); no seed reads not-load-bearing.

## Honest residuals

- **The swap-vs-hold decision is the spiking circuit's.** Topic extraction is host comprehension of the input (the
  declared boundary). The strings "On <topic>, then — " and "Setting the held thread aside — " are host articulation
  templates.
- **Cross-turn continuity of the held topic is a host label** (`held_slot`), inherited from #77.
- **Option-C pairing.** Load-bearing here means load-bearing under the adequate probe with an opt-in flag. The
  shipped default battery still reads this faculty not-exercised, and no production default changed.
