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

This is **one seed**, a smoke. It is not a robust-core claim. The six-seed run is staged on the pool.

Pre-registration: `2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md` (commit `4d203f584`, amendment A1 in
`ebe184fbc`).

## What the smoke measured

Seed 42, numpy backend, `--repeats 2`. Four arms were built locally under `tools/memcap.sh 12`: intact_a, intact_b,
lesion, and a lesion rebuild. The first run crashed in the scorer (amendment A1). The same four arm files were then
rescored with the fixed scorer; `LB_RESUME_SKIP_EXISTING=1` loads an existing arm file instead of rebuilding it.

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
- G4 contrast clean: 0 diffs on `sw_open` and `sw_hold`.
- G5 reply changed: yes. 4 diffs: `answer`, `swapped`, `reason`, `lead`.
- G6 reproduced: yes.

Verdict: `regressed`, `load_bearing: true`.

The intact reply carries two swap-dependent prefixes:
- "On cat, then — " is the swap-drives lead.
- "Setting the held thread aside — " is the GNW global-stop organ. It reads `swapped == True` as a hard topic break.

Both vanish under the lesion, because both ride the same neural swap verdict. The lead is `swap_drives`' own output.
The stop prefix is a downstream consumer of the same verdict. The credited effect is "the swap verdict changes the
reply", not "the lead string alone".

A second copy of the same seed-42 arms was built on pool41 at `4d203f584`. It showed the same per-turn swap state and
replies before its scorer crashed (A1).

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

- Six pool jobs are queued, one per seed (42 43 44 100 101 102). Each runs
  `--only swap-drives-response --repeats 2 --seed <s>` with `LB_SWAP_DRIVE_PROBE=1`, numpy backend and
  `memcap 8`, in the isolated revision directory `~/derisk-pool/revisions/7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`
  on pool41 and pool42.
- Each seed writes to its own directory, `research/findings/raw/_load_bearing/swap_drive_probe/s<seed>/`.
- The pre-registered headline rule applies: all six seeds must read LOAD-BEARING. Anything less is reported as k/6,
  naming each failing seed's verdict.

## Honest residuals

- **The swap-vs-hold decision is the spiking circuit's.** Topic extraction is host comprehension of the input (the
  declared boundary). The strings "On <topic>, then — " and "Setting the held thread aside — " are host articulation
  templates.
- **Cross-turn continuity of the held topic is a host label** (`held_slot`), inherited from #77.
- **Option-C pairing.** Load-bearing here means load-bearing under the adequate probe with an opt-in flag. The
  shipped default battery still reads this faculty not-exercised, and no production default changed.
