---
type: finding
status: qualified
claim_check: measured
date: 2026-09-25
lane: load-bearing
mechanism: the v3 DA-gated synaptic tag-and-capture ledger (webapp/da_tag_capture.py SynapticTagCaptureLedger)
  wired into the live /api/brain-chat store path and the continuous engine's idle/sleep tick behind
  BRAIN_DA_TAG_CAPTURE (default OFF, webapp/da_tag_capture_chat.py); scored here is the PRODUCTION LTM
  configuration (BRAIN_LTM_SHIP_DEFAULT unset, TieredFactStore wraps the buffer with the wikidata_100k shard,
  `--ltm on` on this runner) -- the flip-deciding arm registered by the prereg's Amendment 3
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md (Amendment 3, commit
  cce3c1dbd -- the exact revision this battery ran at; Amendments 4/5 landed after and are not exercised by
  this code path)
artifacts:
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed42.json (plus its per-arm directory: 11 arms, each
    with a .prov.json sidecar)
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed43.json (plus its per-arm directory: 11 arms, each
    with a .prov.json sidecar)
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed44.json (plus its per-arm directory: 11 arms, each
    with a .prov.json sidecar)
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed100.json (plus its per-arm directory: 11 arms, each
    with a .prov.json sidecar)
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed101.json (plus its per-arm directory: 11 arms, each
    with a .prov.json sidecar)
  - research/findings/raw/_da_tag_capture_chat_ltmon/seed102.json (plus its per-arm directory: 11 arms, each
    with a .prov.json sidecar)
verdict: GO 6/6 seeds under the pre-registered gates (G0, P1, G1-G6, G_isolation_gamma_consistent), scored by
  the runner's own `--aggregate` combine rule, LTM-ON (production tier) arm -- the flip-deciding read Amendment
  3 registered and the prior LTM-off GO finding named as still outstanding. This is a RUNNER-LEVEL GO
  (`docs/TERMS.md`): `da-gated-encoding` remains `wired` but NOT `on-by-default` (BRAIN_DA_TAG_CAPTURE defaults
  OFF in production), so it is not `closed`/`integrated`. No default is flipped by this finding.
---

# DA tag-and-capture wired into chat: 6/6 seeds GO at runner level, PRODUCTION LTM (LTM-on) arm (2026-09-25)

Pre-registration: `2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md`, Amendment 3 (`cce3c1dbd`, branch
`research/da-tag-capture-ltm-on`), which registered the LTM-ON arm as "the flip-deciding read the prior
finding's own 'what this GO does NOT show' section named as still outstanding." This finding scores that arm:
the SAME instrument, seeds, gates and `--aggregate` combine rule as the 2026-09-24 LTM-off GO
(`2026-09-24-da-tag-capture-chat-wire-6seed-GO-runner-level-ltm-off.md`), differing only in `--ltm on`
(`BRAIN_LTM_SHIP_DEFAULT` unset, so `TieredFactStore` wraps the composer's buffer with the routed
`wikidata_100k` LTM shard). Terms follow `docs/TERMS.md`: "GO" is the gate's own verdict; the faculty is
`wired` but not `on-by-default`, so this is a **runner-level GO**, not `closed`/`integrated`.

## Provenance verification (required by this task, done explicitly)

All 66 per-arm artifacts (11 arms x 6 seeds) carry a `.prov.json` sidecar. Verified programmatically against
every sidecar, not spot-checked: `git_sha` == `cce3c1dbdc7bd1f73b6af2c6a5769a6ffd87f944` (Amendment 3's own
commit) on all 66, `source_kind: git_archive` on all 66, `source_manifest_verified_at_start` and
`source_manifest_verified_at_exit` both `true` on all 66, `git_dirty: false` on all 66. Each seed directory
holds exactly 11 arms (`neu_imm_intact`, `neu_night_intact`, `neu_night_lesion`, `neu_night_off_intact` --
Amendment 3's new control -- `sal_imm_intact`, `sal_imm_lesion`, `sal_night_intact_a/b`, `sal_night_lesion`,
`sal_night_off_intact`, `sal_night_off_lesion`), confirming completeness before grading.

**Seeds 43 and 101 (the pool1/AWS-recovery seeds).** Per this task's brief, these were pulled off a stopped AWS
node (pool1) at ~10:05 2026-09-25 after `aws_idle_stop` stopped the instance before syncing (board commit
`8c9c040ab`, "pool1 recovered -- stranded LTM-on seeds 43/101 pulled"). Both read exactly the same as every
other seed on every check above: 11/11 arms present, `git_sha cce3c1dbdc7bd1f73b6af2c6a5769a6ffd87f944` on all
11 sidecars each, `source_kind: git_archive`, both manifest checks `true`, `git_dirty: false`. No arm is missing
or truncated. Seed 42's earlier single-seed artifact (already on disk before this task) is graded here under
the identical rule as the other five -- no separate treatment.

Pool-node liveness (this task's brief): `ssh -n -F research/queue/.pool_ssh_config <host> 'ps -eo etimes,args |
grep _da_tag_capture_chat_probe'` on all four nodes. pool1 and pool2: no matching process (clean; nothing of
this lane still writing). pool41 and pool42: two `_da_tag_capture_chat_probe --family fi ... --ltm off` workers
each, a DIFFERENT lane (`research/findings/raw/_sleep_forgetting_interference`, not `base`/`ltm on`) -- these do
not touch this battery's `--out` directory and do not affect completeness of the artifacts scored here.

No seed-level `.prov.json` sidecar exists for the six top-level `seed<N>.json` combine files (only the 66
per-arm files carry one) -- by this runner's own design: `seed<N>.json` is written by in-process `json.dump`
inside `run_seed`'s own loop over already-provenanced arm artifacts, not by a fresh `-m research.runners.X`
invocation the automatic provenance door instruments. Stated explicitly per this task's brief, not silently
assumed.

## Result (runner's own `grade_seed` / `--aggregate`, LTM-ON)

`.venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --family base --aggregate
research/findings/raw/_da_tag_capture_chat_ltmon` (run against the six artifacts above, copied into this
worktree) reports `n_go: 6`, `verdict: GO`, `diffs_intact_minus_lesion: [1, 1, 1, 1, 1, 1]`,
`diffs_salient_minus_neutral: [1, 1, 1, 1, 1, 1]`, `signflip_p_intact_vs_lesion: 0.015625 <!--derived-->` and
`signflip_p_salient_vs_neutral: 0.015625` <!--derived--> (one-sided exact sign-flip p over 6 seeds; the
combine's own output, reproducible via the command above, is not itself committed here -- its bare top-level
`verdict` string carries no `preconditions` block, the same reason the LTM-off finding gave for not committing
its `aggregate.json`).

| seed | seed_verdict | gamma (all companion-ON arms) | p_max sal_night_intact | p_max sal_night_lesion | p_max neu_night_intact | lesion/intact ratio (G6 < 0.25) |
|---|---|---|---|---|---|---|
| 42 | GO | 32.7735 | 0.02396 | 0.00079 | 0.00089 | 0.033 <!--derived--> |
| 43 | GO | 32.7735 | 0.01845 | 0.0 | 0.0 | 0.0 <!--derived--> |
| 44 | GO | 32.7735 | 0.01962 | 0.00225 | 0.0 | 0.1147 <!--derived--> |
| 100 | GO | 32.7735 | 0.02106 | 0.00020 | 0.0 | 0.0095 <!--derived--> |
| 101 | GO | 32.7735 | 0.01841 | 0.00016 | 0.0 | 0.0089 <!--derived--> |
| 102 | GO | 32.7735 | 0.01359 | 0.00219 | 0.0 | 0.1613 <!--derived--> |

(last column = p_max sal_night_lesion / p_max sal_night_intact, computed from the two preceding columns, both
read off each cited `seed<N>.json`'s `sal_night_intact_a`/`sal_night_lesion` arm's `tag_capture_at_recall.p_max`.)
Every seed passes G0, P1, G1-G6 and `G_isolation_gamma_consistent` (`gamma`/`d1_a_go` identical across every
companion-ON arm at every seed, on every seed -- the Amendment-1 isolation fix holds under LTM-on too);
`n_arm_errors: 0` on all six.

## The transfer Amendment 3 argued from code is now CHECKED, not merely argued -- and matches to the value

Amendment 3's "declared reasoning" predicted the buffer-only measurements would transfer to LTM-on unchanged,
because `TieredFactStore.store()` routes every write to the buffer only and `store_composer()` unwraps to
`.buffer` explicitly, so the LTM shard is inert to this mechanism by construction. Direct arm-by-arm comparison
against the committed LTM-off artifacts (`research/findings/raw/_da_tag_capture_chat/seed42.json`) at seed 42
confirms this for every one of the 10 shared arms: identical `recall_outcome`, identical `tag_capture_at_recall`
(gamma/d1_a_go/p/p_max/n_managed_blocks/external_rescales/external_rewrites all equal), identical `turn_da`
trajectories. The `gamma=32.7735`/p_max table above reproduces the LTM-off finding's own table values exactly,
seed for seed. One honest caveat: this comparison spans two git revisions, not a single-revision on/off toggle
(the LTM-off `seed42.json` was harvested at the post-isolation-fix commit named in that finding's Amendment
2/1 lineage; this LTM-on run pins to `cce3c1dbd`, three commits later). Amendment 4's own root-cause analysis of
those intervening commits found they touch only `webapp/server.py`'s reply-assembly surface (~14 of 19 diff
hunks are unrelated features), not the composer's store synapses or the deterministic ledger scenario (both
independently proven byte-identical there) -- consistent with, though not a substitute for, a same-revision
A/B. The construction-level argument (point 1 above) is the primary basis for the transfer claim; this
byte-identical match is corroborating evidence, not the sole basis.

## Board #227 item (c): does the flip cost an ordinary fact its overnight survival? -- MISSING, confirmed on
## all 6 seeds, under the actual production LTM configuration

Amendment 5 (2026-09-24) measured `ordinary_fact_flip_forgetting: True` on one seed (42), LTM-off only: the
plainly-told neutral fact survives overnight under today's shipped default (`neu_night_off_intact`, `correct`)
but is lost if `BRAIN_DA_TAG_CAPTURE` is flipped on (`neu_night_intact`, `abstain`) -- the mechanism's own
by-design behavioral-tagging selectivity (Moncada & Viola 2007), not a defect. This task's brief describes a
21:15 audit that found item (c) reading MISSING; no separate committed document under that timestamp was found
in this repo's findings/coordination corpus, so this section is grounded in Amendment 5 plus the run below, not
an unlocatable source.

This run extends that result to all 6 seeds AND to the actual production LTM configuration (not just the
LTM-off buffer-only arm): `grade_seed`'s `ordinary_fact_flip_forgetting` reads `True` on every one of the 6
seeds (`neu_night_off_intact` -> `correct`, `neu_night_intact` -> `abstain`, on all 6, per the `gates.outcomes`
field of each cited `seed<N>.json`). **Where item (c) stands: an ordinary fact told once is NOT kept overnight
under the flip, confirmed under LTM-on across all 6 pre-registered seeds** -- this is the same cost Amendment 5
found on one seed at LTM-off, now checked (not assumed) to hold under the actual production LTM tier and to
generalize across seeds. It remains a REPORTED, non-gating field by Amendment 3's own construction (cannot flip
`GO`/`NO-GO`/`UNDEFINED`) and a stated, not hidden, behavior-change cost of the flip.

## What this GO does NOT show

- **Still runner-level, not `closed`/`integrated`** (`docs/TERMS.md`): `BRAIN_DA_TAG_CAPTURE` defaults OFF in
  production; every arm here set it explicitly via `--env`. This GO does not license flipping the default.
- **Byte-identical-OFF is still UNDEFINED at any correctly-derived pin.** Amendment 4 (2026-09-24) found
  `--offcheck --pinned-sha 36a175534` FAILED (`replies_identical: false`), root-caused to the pin itself going
  stale (19 unrelated diff hunks accumulated on `main` since Amendment 2 set it), not to a da-tag-capture
  regression -- but this remains UNDEFINED, not verified, and this task did not re-run `--offcheck`. It concerns
  the OFF path (flag unset entirely), a different code surface from the ON-arm measurements scored here.
- **No adversarial (`verify-go`-style) review of this specific LTM-on artifact set was run as part of this
  scoring task** -- only the runner's own registered gates and the provenance/completeness checks this task's
  brief required.
- **No combined-battery regression check was run.** This task scored `_da_tag_capture_chat_probe`'s own 11-arm
  instrument only; `onebrain_regression_battery.py` appears in every arm's provenance only as the in-process
  worker this probe dispatches through, not as a run of that suite's OWN broader regression battery with
  `BRAIN_DA_TAG_CAPTURE=1` against the other faculties it also covers.
- **This is a synaptic-state GO, not a `consolidation` claim** (`docs/TERMS.md`): the late-phase variable is a
  per-synapse state: no replay path executes here.

## Flip candidacy against the owner's bar (6-seed GO + SOUND review + no regression in the combined battery +
## production-default validation) -- NOT MET; no default is flipped by this finding

1. **6-seed GO** -- MET, twice over: LTM-off (2026-09-24) and LTM-on (this finding), both runner-level GO 6/6.
2. **SOUND review** -- NOT MET for this artifact set. Amendments 1-2 show the review process finding real
   defects on this SAME lane before (the D1-cache confound, the stale-verdict `aggregate()` bug), which is
   why those checks exist in `grade_seed`/`aggregate()` today and pass here -- but no fresh adversarial pass
   was run against these specific LTM-on artifacts in this task, and the byte-identical-OFF check remains
   UNDEFINED (above).
3. **No regression in the combined battery** -- NOT MET. Not run in this task (above).
4. **Production-default validation** -- PARTIALLY MET. This run used the actual production LTM configuration
   (the LTM tier attaches by default), the leg the LTM-off GO explicitly could not speak to. It does NOT
   validate `BRAIN_DA_TAG_CAPTURE` itself running with no explicit flag (every arm here sets it via `--env`),
   so `wired AND on-by-default` is still not demonstrated.

**Conclusion: this is not yet a flip candidate.** Two of four legs remain open (SOUND review of this artifact
set, and a combined-battery no-regression run); leg 4 is only half-closed. Per this task's explicit instruction,
no default is changed here.

## Compute

Ran on the AWS `r7i.4xlarge` CPU pool (per-arm provenance: `POOL_JOB_MEM_GB: "48"`, host `ip-172-31-47-37`),
`SIM_BACKEND=numpy`, one seed per instance-job, pinned to `cce3c1dbd` via `git_archive` (verified above). No
new brain build was run by THIS scoring pass -- it re-grades the six already-landed artifacts via the runner's
own `--aggregate`, after copying them into this worktree (they are untracked in the primary checkout per this
task's rules) and re-verifying provenance programmatically rather than by spot check.
