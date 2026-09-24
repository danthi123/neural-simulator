---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: an adequate battery probe for swap-drives-response (the GNW thought-swap drive, board #77/#85, webapp/swap_drives_chat.py, default-ON) — LB_SWAP_DRIVE_PROBE (default OFF) remaps the faculty to a held-topic -> same-topic -> competing-topic conversation and scores it with a pre-registered reply + contrast gate
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only (filed before any sw_* arm existed). No result is claimed here.
runner: research/runners/load_bearing_fraction.py
artifacts:
  - research/findings/raw/_load_bearing/_shards/allfixes2/s42/swap-drives-response/intact_a_hold_held.json
  - research/findings/raw/_load_bearing/_shards/allfixes2/s42/swap-drives-response/lb.json
  - research/findings/raw/_swap_drives_chat/verify.json
---

# swap-drives-response adequate probe: PRE-REGISTRATION (filed before any measured run)

**Filed 2026-09-23 in its own commit on branch `research/swap-drives-adequate-probe`, cut from origin/main
`f35196e66`.** No brain arm on the new turns had been built when this was committed. Seeds 42 43 44 100 101 102.

## What was seen when this was written

- The all-fixes 6-seed battery (`2026-09-23-allfixes-adequate-battery-6seed-robust-core-24.md`) read
  `swap-drives-response` **not-exercised on every seed**.
- Why, read from the s42 shard (`research/findings/raw/_load_bearing/_shards/allfixes2/s42/swap-drives-response/`):
  the probe turn `held` ("the wolf watches the owl", after "the fox and the wolf walked in") is answered by the
  role-binding REPAIR short-circuit ("I caught the verb 'watch' ... which of them is the 'watch' done to"). That path
  returns before `webapp/server.py` attaches `swap_drives`, so the fields are absent in both arms. Independently,
  wolf/owl/fox are not tiny-demo KB concepts, so `gnw_thought_swap._extract_topic` returns None on that turn. No swap
  is ever due there.
- The #85 wiring finding (`2026-08-19-swap-drives-chat-load-bearing-GO.md`,
  `research/findings/raw/_swap_drives_chat/verify.json`) measured the lead-on-swap coupling through the real handler
  **with the heavy default organs disabled**, at one seed. It has never been measured inside the integrated battery
  with every default organ on, and never across seeds.
- The GNW continuous-mode lane was parked today (verification claims refuted). This probe does not use it; it
  measures the shipped default-ON `swap_drives` path only.

## The probe (flag `LB_SWAP_DRIVE_PROBE`, default OFF)

Three label-only turns in one fresh session `sw2` (added to `_EXTRA_TURNS`, not to `PROBE_TURNS`):

| turn | message | grounded topic (production extractor) | what should happen |
|---|---|---|---|
| `sw_open` | what does the dog chase | dog | first thought: 'dog' becomes the held topic |
| `sw_hold` | what does the dog chase | dog | same topic: hold, no swap due |
| `sw_switch` | what does the cat eat | cat | salient competing topic: a swap is due |

Both questions are ordinary recalls over boot facts `(dog,chase,cat)` and `(cat,eat,fish)`, so they should reach
the single-fact reply path, which is where the server prepends the swap lead.

The arms are the battery's existing ones:
- intact_a, the treatment reference;
- intact_b, a rebuild at the same seed (the null);
- lesion, with `BRAIN_SWAP_DRIVES_LESION=1`;
- a lesion rebuild (`--repeats 2`).

The lesion is the specific edge the faculty claims: it gives the spiking mismatch/salience detector no proposal
drive (`run_intention_swap(trigger_lesion=True)`). Everything else is byte-identical between arms, and `base_env` is
empty.

Compared fields on every turn: `answer`, `swap_drives.swapped`, `swap_drives.reason`, `swap_drives.lead`.

## The gate (per seed; `_swap_drive_score`, selftested on synthetic inputs)

A seed counts as LOAD-BEARING only if all of these hold:

- **G1 exercised.** The swap-drives fields are present on `sw_switch`. If they are absent in both arms, the seed
  reads `not-exercised`.
- **G2 contrast defined.** The intact arm carries a swap trace on `sw_open` and on `sw_hold`. If not, the seed reads
  `contrast-undefined` (UNDEFINED, not a pass).
- **G3 null clean.** intact_a == intact_b on every compared field of all three turns. If not, `noisy-null-control`.
- **G4 contrast clean.** The lesion does NOT change any compared field on `sw_open` or `sw_hold`, the turns where no
  swap is due. If it does, the seed reads `nonspecific-lesion`: the lesion changes replies with no topic change, so
  the switch-turn diff cannot be credited to the swap.
- **G5 the reply changes.** On `sw_switch`, `answer` differs intact vs lesion. If only the trace fields differ, the
  seed reads `trace-only` and counts as exercised but NOT load-bearing.
- **G6 reproduces.** The lesion rebuild gives the same verdict. If not, `noisy`.

The per-seed result also records each arm's own swap state on every turn: swapped, reason, topic, held topic
before and after, lead, `mm_peak`, `boost_max`. That is the mechanism's state, not an arg-max over a sweep.

**Headline rule, fixed now:** the faculty joins the robust core under this probe only if all six seeds read
LOAD-BEARING. Anything less is reported as k/6, with each failing seed's verdict named.

## Predictions, and how each can fail

- **Intact `sw_switch`: `swapped=True`, reason `topic_change_swap`, answer beginning "On cat, then — ".**
  - The spiking chain decides this, not a host `if`: mismatch fires, the incumbent is evicted, the vacancy gate
    admits the newcomer.
  - It can read `mismatch_held_no_swap` on a seed; the de-risk's 6/6 was measured on the isolated circuit, not in
    this session.
  - The reply can also fail to carry the lead: a short-circuit path, or a downstream organ that rewrites `answer`.
- **Lesion `sw_switch`: `swapped=False`, reason `mismatch_held_no_swap`, answer without the lead.**
- **`sw_open` / `sw_hold`: identical intact vs lesion.**
  - Open risk: the lesion also removes the proposal drive on the match probes (first thought, same-topic hold). If
    that changes the held coalition or the reason string, G4 fails.
  - I do not know the outcome in advance. The contrast is there to catch exactly this.
- **Honest expectation:** G1–G3 and G5 are likely on most seeds, given the #85 wiring and a deterministic numpy
  harness. G4 and cross-seed swap reliability are the open questions.

## What this does NOT claim

- **A pass is "load-bearing under the adequate probe with an opt-in flag"** (the Option-C pairing). It does not
  grow the shipped-default robust core, and no production default changes.
- **Topic extraction is host comprehension** of the input: the declared boundary, the same one the SVO parser
  occupies. The transition string "On <topic>, then — " is a host articulation template. The swap-vs-hold decision
  is the spiking circuit's; what is credited is only that its verdict changes the reply.
- **Inherited #77 residual:** cross-turn continuity of the held coalition is a host label (`held_slot`), re-ignited
  on the swap substrate each turn.

## Byte-identical OFF, asserted in data

`research/runners/_swap_drive_probe_offcheck.py` compares the pinned SHA `f35196e66` with the branch, flag unset:
- **Static.** It materialises `git archive f35196e66` and compares `PROBE_TURNS`, `FACULTY_PROBES`,
  `FACULTY_LESIONS`, and `turn_group()` of every pinned label with `==`.
- **Data.** It runs `--only swap-drives-response --repeats 2` in both trees at seed 42 and compares the sha256 of
  every arm file and the per-faculty row.
- **Sensitivity.** The flag-ON smoke row must differ.

## Commands

Smoke (1 seed, local, one at a time):
`bash tools/mem_ok.sh 12 && LB_SWAP_DRIVE_PROBE=1 SIM_BACKEND=numpy bash tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners.load_bearing_fraction --only swap-drives-response --repeats 2 --seed 42 --out research/findings/raw/_load_bearing/swap_drive_probe/s<seed>/lb.json` (with `<seed>` = 42)

6 seeds (pool, one `--out` directory per seed):
`LB_SWAP_DRIVE_PROBE=1 SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.load_bearing_fraction --only swap-drives-response --repeats 2 --seed <s> --out research/findings/raw/_load_bearing/swap_drive_probe/s<s>/lb.json`

## Amendment log

**A1, 2026-09-23 ~19:25, instrument crash fix. The gate is unchanged.**
- **What broke.** The s42 smoke on pool41 at `4d203f584` built all four arms, then crashed in `_score_swap_drive`
  with `NameError: _get_path`. The helper was never imported from the battery module. The pure-scorer selftests
  did not reach that function.
- **What was seen before the fix.** I read the s42 arm files: intact_a, intact_b and the first lesion arm.
  - Intact `sw_switch`: swapped=True, `topic_change_swap`, and the answer carried "On cat, then — ".
  - Lesion `sw_switch`: swapped=False, `mismatch_held_no_swap`, no lead.
  - `sw_open` and `sw_hold` read the same fields in both arms.
- **What changed.** Only the missing import, plus a selftest that runs `_score_swap_drive` end to end on synthetic
  arms. The rules in `_swap_drive_score`, the fields, the turns and the seeds are unchanged.
- **Consequence for staging.** The six pool jobs staged at `4d203f584` were withdrawn from the queue before any was
  dispatched. They are restaged at the fixed revision, merged with origin/main as provisioning requires.
- **The s42 smoke** is scored by re-running the fixed scorer over the already-built s42 arm files
  (`LB_RESUME_SKIP_EXISTING=1`, which loads an existing arm instead of rebuilding it). The six-seed run rebuilds
  every arm from scratch, s42 included.

**A2, 2026-09-23 (fix round, review key `v2:fc49c6e6f`). Provenance correction: the s42 smoke was cross-revision;
the gate is unchanged.**
- **What was wrong.** The A1 note above says the s42 smoke was "already-built... arm files", implying one build.
  Per each arm's `.prov.json`, `intact_a`/`intact_b` were built at `4d203f584` (before A1) and `lesion`/its rep0
  were built at `7f90034df` (after A1 and the origin/main merge) — the same long-lived worker parent process
  spanned the code change. The finding this preregistration governs stated the smoke's provenance incorrectly
  (as one local build, all four arms at one revision).
- **What was checked.** The code diff between `4d203f584` and `7f90034df` on the arm-building path
  (`load_bearing_fraction.py`, `onebrain_regression_battery.py`) is limited to the A1 import fix, a selftest
  helper, and a no-op merge of origin/main — so no confound was actually measured by the cross-revision build.
  That is a property of this specific diff, not a general excuse for cross-revision comparisons.
- **The fix.** The six-seed pool run (staged at the single pinned revision
  `7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`) includes seed 42. Its four arms all read `git_sha:
  7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`, `git_dirty: false`, `source_kind: git_archive`, one shared
  `SIM_RUN_ID` and a verified `source_manifest_sha256` — a genuine single-revision build. It reproduces the
  withdrawn smoke's verdict, replies and swap state exactly (the only diff is a `cuda_visible_devices` reporting
  field). The finding now reports this rebuilt result as its s42 evidence, with the cross-revision history
  disclosed rather than corrected-away.
- **Also added to the finding (same review round):** which of the four G4-compared fields are pass-by-construction
  (`swapped`/`reason`/`lead` on the no-swap-due turns; only `answer` is real evidence there), a "no reply-level
  change" correction in place of "changes nothing" (mechanism state — `mm_peak`/`boost_max` — does move under the
  lesion on the contrast turns), and a plain statement of what G5 actually tests (a near-guaranteed reply
  difference once G4 holds and the intact arm swaps, so the falsifiable content is whether the intact arm swaps at
  all per seed and whether the lead survives downstream composition, not the reply comparison itself).
- **Gate unchanged.** `_swap_drive_score` and its compared fields are exactly as pre-registered above; this
  amendment is documentation-only (provenance + honesty declarations), not a rule change.
- **Not touched.** The five staged pool jobs for seeds 43/44/100/101/102 at `7f90034dfbf48bfc128e3f9b1c7f7506bb3fe8a6`
  continue unmodified.
