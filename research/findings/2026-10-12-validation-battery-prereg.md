---
type: finding
status: live
date: 2026-10-12
lane: integration / load-bearing-fraction (the one validation battery for the default flips)
mechanism: PRE-REGISTRATION of the single validation battery at current main (RULINGS speedup-battery,
  tree-c1-flip, c5-g6b-accept-1012, battery-prereg-1012): a B2d-design re-run with each flip's LBF row in the
  registry, one g6b-class C5 pair, one alloff attribution run, and tools/answer_rate.py per flip. Eight flags,
  seven flip units (C1 = BRAIN_CATEGORY_INHERIT + BRAIN_CATEGORY_INHERIT_HOP per RULING c1-flip-config), each
  validated by its row's lesion changing the REPLY at the seed bar registered below.
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed in its own commit before any shard of the battery below was run. No result
  is claimed here.
runner: research/runners/load_bearing_fraction.py (sharded by tools/lb_shard.py; job lines guarded by
  tools/assert_flipped_defaults.py, tools/assert_iw1_defaults.py, tools/assert_iw2_defaults.py;
  scored by tools/b2d_score.py; the C5 pair by research/runners/onebrain_regression_battery.py)
artifacts:
  - research/findings/2026-09-23-flip-validated-fixes-production-default-validation-prereg.md
  - research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md
  - research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md
  - research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md
  - research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md
  - research/findings/2026-10-05-iw2-false-belief-chat-default-on-flip-validated-PROVISIONAL.md
  - research/findings/2026-10-05-iw2-neural-selfid-default-on-flip-validated-PROVISIONAL.md
  - research/findings/2026-10-05-iw2-transitive-chat-default-on-flip-validated-PROVISIONAL.md
  - research/findings/2026-10-12-c5-g6b-replay-fired-pair.md
  - docs/drafts/stage1-c1-inheritance/ATTEMPT2-C1-AMENDMENT-TEXT.md
  - research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt
  - research/findings/raw/_load_bearing/_shards/b2d0929/b2d_score.json
  - research/coordination/LOCAL_LLM_RUNBOOK.md
---

bar-sanity: research/packets/b2d-mouth-renderer/bar_sanity_record_amend2.json
bar-sanity: docs/drafts/stage1-exit/bar_sanity_record_s1_c1_fixed_pathway_lesion.json
bar-sanity: docs/drafts/idle-replay/bar_sanity_record_c5_organ.json
bar-sanity: docs/drafts/idle-replay/bar_sanity_record_c5_battery_amendB.json

# The one validation battery: eight flip flags at current main, PRE-REGISTRATION

**Filed 2026-10-12, in its own commit, before any shard of the battery
below was run.** Seeds 42 43 44 100 101 102. The job lists and output aggregates are committed AFTER this
document, under `research/findings/raw/_load_bearing/_shards/b2d1012/` (stub-arm shards), its `-alloff` sibling
(the attribution run), and `research/findings/raw/_regression_battery/c5val_1012/` (the C5 pair). Nothing in
those directories exists at filing time; every artifact path in the frontmatter exists today and is an INPUT to
this registration.

## 1. Purpose

One run validates every pending default flip at one revision (RULING tree-c1-flip: "the speedup-battery waits for
the C1 flip merge, so ONE run at that main validates 6 flips"; RULING c5-g6b-accept-1012 adds the C5 flip). The
owner's unit of progress (2026-09-29) is a faculty ON BY DEFAULT whose lesion changes the REPLY; this battery is
the reply-level test of that unit for each flip, plus the no-regression and identity checks the flip bar demands
(RULING speedup-battery; the four-legged flip bar of `2026-09-29-iw1-wave-battery-PREREGISTRATION.md` §Why).

## 2. The flip table (current defaults READ from source at main, 2026-10-12; flip owner: Claude)

Every "current default" below is the literal anchor constant or reader at main f734c0c69 as read on 2026-10-12.
An explicit `=0` remains the byte-identical escape everywhere.

| flag | current default at main | anchor (read 2026-10-12) | faculty / row | provenance ruling |
|---|---|---|---|---|
| BRAIN_CATEGORY_INHERIT | ON (flipped) | `webapp/category_inherit_chat.py::_CATEGORY_INHERIT_DEFAULT_ON = True` (flip commit 680a313e8, in main) | `category-inherit` (research/runners/lbf_rows/category_inherit.py) | c1-flip-go, tree-c1-flip, c1-flip-merged (2026-10-12) |
| BRAIN_CATEGORY_INHERIT_HOP | ON when unset (flipped) | `research/runners/category_inherit_organ.py` hop reader (unset -> ON; =0 restores the 09-30 organ) | same row (the flip ships organ + hop) | c1-flip-config: the C1 flip turns on BOTH flags |
| BRAIN_IDLE_REPLAY | **OFF** (pending flip) | `research/runners/idle_replay_organ.py::_IDLE_REPLAY_DEFAULT_ON = False` | no LBF registry row; the g6b-class pair below | c5-g6b-accept-1012: "the C5 default-ON flip now rides the one validation battery with C1" |
| BRAIN_FALSE_BELIEF_CHAT | ON, PROVISIONAL (validation pending) | `webapp/false_belief_chat.py::_FALSE_BELIEF_CHAT_DEFAULT_ON = True` (flip commit 1fb805e9e, now in main) | `tom-false-belief` (research/runners/lbf_rows/tom_false_belief_chat.py) | iw2-flip-hold (owner 2026-10-01) held the flip off main; superseded in fact by the merge — see §9; speedup-battery |
| BRAIN_NEURAL_SELFID | ON, PROVISIONAL | `research/runners/brain_chat_tui.py::_NEURAL_SELFID_DEFAULT_ON = True` | `neural-selfid` (research/runners/lbf_rows/neural_selfid.py) | iw2-selfid: in FLIP only if its section-5 row reads D0 PASS (it did: D0 IDENTICAL, 8 preconditions, the 2026-10-05 finding §1) |
| BRAIN_TRANSITIVE_CHAT | ON, PROVISIONAL | `webapp/reasoning_transitive_chat.py::_TRANSITIVE_CHAT_DEFAULT_ON = True` | `transitive-chat` (research/runners/lbf_rows/reasoning_transitive_chat.py) | speedup-battery; 2026-10-05 finding §1 (R PASS, D0 IDENTICAL) |
| BRAIN_D6_HEBBIAN_STORE (+ its anchored partners BRAIN_D6_ENGRAM_VOCAB, BRAIN_D6_ENGRAM_READTIME) | ON, PROVISIONAL (ONE anchor for all three) | `research/runners/d6_hebbian_store.py::_D6_LEARN_DEFAULT_ON = True` (flip commit cfce97fe1, in main) | `d6-hebbian-write` (research/runners/lbf_rows/d6_hebbian_write.py) | iw1-R: iw1 flip list = d6-hebbian-write, xedge-surprise-worldmodel |
| BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL | ON, PROVISIONAL | `research/runners/onebrain_xedge_surprise_worldmodel_production.py::_XEDGE_SWM_DEFAULT_ON = True` | `xedge-surprise-worldmodel` (research/runners/lbf_rows/xedge_surprise_worldmodel.py) | iw1-R (note: the module's comment text still says "(OFF)" — stale prose; the constant is `True`) |

NOT in the flip list: `wm-focus-bind` (BRAIN_MULTIREF_FOCUS_BIND, `_MULTIREF_FOCUS_BIND_DEFAULT_ON = False` at
main) — held OUT per RULING iw1-wmf-hold / flip-hurts-answers. The iw2 sub-flips
(`_FALSE_BELIEF_SPIKING_WITNESS_DEFAULT_ON`, `_TRANSITIVE_SPIKING_REACH_DEFAULT_ON`) are False and are not part of
this battery. The emo/C4 flip (RULING s2-c4-flip, target 10-20) is not in this battery.

Note on wording: the five speedup-plan flips were "de-risked default-off" candidates, but at current main they are
PROVISIONAL default-ON (the flip commits landed; the flip commits themselves say "Validation pending (shared
prereg steps B, F)"). This battery is that validation: a flip that fails is REVERTED to default-off with a finding.

## 3. Seeds

Gate seeds 42 43 44 100 101 102 only, and only through the registered job lines below (tools/gate_seed_guard.py
resolves the gate seeds to their committed prereg — this document). Dev seed 7 runs are smoke only and carry no
weight. Six seeds before any generalization claim (non-negotiable 6).

## 4. Battery design (one run at current main; smoke first)

**Revision.** Current main containing the C1 flip merge (RULING c1-flip-merged, 2026-10-12: research/c1-flip
merged; category inheritance + hop default-ON). The shard provenance sidecar records the git SHA.

**The B2d re-run (stub arm).** The 2026-09-29 B2d design re-run at the new pin, tag `b2d1012`: the same Step-0
generation as `research/packets/b2d-mouth-renderer/make_jobs.py stub` describes —
`tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2d1012 --no-fixes --probe-set adequate --repeats 2
--root '~/derisk-pool/revisions/<F>' --pin <F>` — numpy, one shard per registry faculty and seed, every job line
prefixed with the defaults guards. The B2d original ran at P=5818dc6 (09-30, before the flips; recorded in
research/findings/raw/_load_bearing/_shards/b2d0929/PIN.txt); it is the reference run. Each flip's LBF row is in
the headline registry at current main (the flip commits lifted them; `category-inherit` likewise carries no
REQUIRED_ENV since the flip), so the same shards measure each flip's row at the production default.

**Defaults guards (job-line prefix, in order).** `.venv/bin/python tools/assert_flipped_defaults.py && `
(the 2026-09-23 guard: the three old anchors True, no BRAIN_* in the environment) plus
`tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel"` and
`tools/assert_iw2_defaults.py` (the wave anchors at their expected set). A job that fails a guard exits before
building a brain. The exact composed prefix is fixed at job-generation time and committed with the JOBS list.

**The alloff attribution run.** Tag `b2d1012-alloff`: the same cells with every flip flag forced `=0` at the same
revision. RULING flip-hurts-answers: "UNMEASURED means not checked: queue the flag-off cells (the alloff run)
first" — so the alloff lines are queued before any answer_rate read.

**The C5 pair (g6b class, six seeds).** `research/runners/onebrain_regression_battery.py --worker`, turn set
emo, bc_a, bc_b, selfinit (the g6b shape: an idle/empty selfinit turn where replay can fire), two sequential arms
per seed: OFF (`BRAIN_IDLE_REPLAY=0`) and default (the flipped ON), both with the flag-independent per-turn RSS
sidecar (`BRAIN_IDLE_REPLAY_LEAKLOG` on the ON side for the organ's own records). One seed at a time, ON and OFF
sequentially, never both at once (RULING c5-off-baseline). Output under
`research/findings/raw/_regression_battery/c5val_1012/`.

**Smoke first** (RULING speedup-smoke-first, amended by smoke-rss-portable): the smallest unit — one probe group,
one seed (dev seed 7) — measured for peak RSS and wall time; the queue line's mem_gb=ceil(1.3 x peak), max_age_h
stamped; if peak x parallelism exceeds host RAM or projected wall time exceeds 24 h, do not launch — one NEEDS
with the numbers. The C5 pair smokes with g6b's turn set at dev seed 7 and must show circuits_built>0 before the
six-seed pair is queued.

**S1/S2 re-measure.** The same shards re-measure S0 (registered trace verdict), S1 (self-describing fields
excluded; the #1 metric) and S2 (reply level: `answer`/`abstained`/`recalled_svo`, docs/TERMS.md `load-bearing`)
per the B2d prereg's scoring rules, against the P=5818dc6 record (`research/findings/raw/_load_bearing/_shards/b2d0929/b2d_score.json`;
S1 0.733<!--derived-->, SD 0.019<!--derived-->, robust core 21; S0 0.843<!--derived-->;
S2 reply-level 0.514<!--derived--> — RULING b2d-merged). The S1/S2 claims of this run go through the verify-go REFUTER-LENS
before any headline document changes (RULING b2d-merged / speedup-battery: "the same run re-measures S1/S2 for
the refute-lens").

**answer_rate per flip** (RULING flip-hurts-answers). For each flip row, `tools/answer_rate.py` on the paired
layout (`--on-tag b2d1012 --off-tag b2d1012-alloff --faculty <row>`; where the alloff cell abstains by design, the
`--off-arm <lesion arm stem>` layout per RULING answer-rate-off). `flags: FLIP-HURTS-ANSWERS` on ANY measured
seed is REPORTED beside the registered verdict, changes no PASS/FAIL, and requires a NEEDS line and a Claude
ruling before that flip stands. `UNMEASURED` seeds are listed, never read as "no harm".

## 5. BARS — registered here, BEFORE any data

- **F1 (the per-flip seed bar; the wave preregs' F1 wording).** For each flipped row
  (`d6-hebbian-write`, `xedge-surprise-worldmodel`, `tom-false-belief`, `neural-selfid`, `transitive-chat`,
  `category-inherit`): **the row is reply-level load-bearing on 6 of 6 seeds** — in every DEFINED cell the
  intact_a reply differs from that row's lesion reply (the lesion of the row's EXTRA_LESIONS; reply fields per
  docs/TERMS.md). A DEFINED cell that is not reply-load-bearing fails the seed. This is the bar RULING
  speedup-battery names ("its row's lesion changing the REPLY at the prereg's seed bar validates it").
- **C1.** (a) The C1 PASS of record is the gate_hop verdict (prereg 2026-09-30 section 13, Amendment 1: the 54
  lines = 9 arms x 6 gate seeds with BRAIN_CATEGORY_INHERIT_HOP=1, scored by
  `python -m research.runners.stage1_c1_category --aggregate research/findings/raw/_stage1_c1_category/gate_hop
  --hop --seeds 42,43,44,100,101,102`; RULING c1-gatehop). It read PASS 6/6 before this prereg (RULING
  c1-flip-go: "with A3 (gate_hop PASS 6/6) the PLAN step-C conditions are met") and is carried, not re-run, here;
  the battery's C1 cells must satisfy the same hop validity preconditions (every arm records both flag values;
  flag-on organ reads record hop:true, else the seed is UNDEFINED for C1). (b) The battery's `category-inherit`
  row meets F1 above (lesion BRAIN_CATEGORY_INHERIT_LESION changing the reply). (c) Per Amendment A2 (C1 section,
  docs/drafts/stage1-c1-inheritance/ATTEMPT2-C1-AMENDMENT-TEXT.md), the counted Stage-1 attempt-2 lesion for
  cat_disc/cat_told is `lesion_inherit_pathway` (in-process patch, applied only if calls>0), bars unchanged from
  the 09-30 prereg: correct 6/6, lesion effect 6/6 reproduced, permuted tracks 6/6. The A2 runner change and its
  bar-sanity re-record are committed before any attempt-2 data and run as Stage-1 attempt 2 (fresh lines at
  current main; RULING pool-hold-1012), not inside this battery's shards.
- **C5.** Per seed (the g6b evidence class, RULING c5-g6b-accept-1012): (1) replay FIRES in ON — at least one
  idle/selfinit turn with circuits_built>0 in the ON sidecar; without it the seed is UNDEFINED ("replay-fired
  precondition"), never a pass; (2) the ON idle-turn reply differs from the OFF reply and is organ-authored
  (the g6b pair: composer=`brain_store`, path=`idle-replay`); active-chat turns must stay byte-identical across
  arms (the g6b diff confined every byte change to the selfinit block); (3) **memory bar: max per-turn ON−OFF
  RSS <= 2 GB** (the tree-c5-memory target; the g6b pair measured +6324 KiB). All three on 6/6 seeds.
- **B1 (no regression, both levels).** For every faculty present in both this run and the P=5818dc6 B2d record,
  n_load_bearing at the new pin is not lower than at P, at S0 and at S2. A drop is attributed by the alloff run
  (the 2026-09-23 design): if the alloff run also drops, the cause is code merged since P, not the flips; if the
  alloff run matches P, the flip caused it (bisect per the wave preregs' decision table if needed).
- **B2 (completeness).** Every coverable cell is DEFINED after the re-run rule: one re-run of a missing/UNDEFINED
  cell at the same revision, the line unchanged, logged before its result is read; still-UNDEFINED cells never
  count as load-bearing or as a pass.
- **F2/F3 (identity, dev seed 7).** `research/runners/iw1_default_identity.py` and
  `research/runners/iw2_default_identity.py` at the pin: `default@F` equals `on:<all flipped>@F` on every script
  turn (the default IS the capability), and `off@F` equals the pre-flip `default` on every turn (the escape still
  reproduces pre-flip behaviour).
- **answer_rate** is REPORTED (not gated) per §4.

## 6. Failure handling

Per RULING speedup-battery: **a flip whose row fails F1 reverts to default-off with a finding** — no retuning on
gate seeds (RULING c1-gatehop: rework on dev seeds only, never retune on gate seeds). C1 additionally follows
c1-gatehop: if the C1 evidence does not hold after one re-run of missing cells, no flip, rework on dev seeds. A
B1 drop caused by a flip reverts that flip (bisecting the set if needed); an interaction that no single bisect
names reverts every flipped packet in the interaction and files NEEDS. UNDEFINED is never a pass (§5 B2). One
RESULT files the per-flip table (flip | F1 | B1 | answer_rate flags | verdict: STANDS / REVERTED / UNDEFINED),
plus the re-measured S0/S1/S2 for the refute-lens.

## 7. Scheduling

- The battery waits for the C1 flip merge (RULING tree-c1-flip) — satisfied at filing: RULING c1-flip-merged
  (2026-10-12) merged research/c1-flip to main, and this prereg is drafted at that main. The run starts only
  after THIS prereg is committed (its own commit first, per the 2026-09-23 precedent).
- Gate seeds flow only through the registered job lines (§4), whose JOBS lists are committed after this document.
- Smoke first (§4) before any multi-seed line is queued. GPU/oversized runs route per RULING speedup-aws: a NEEDS
  [aws-run] ASK with exact command, rev SHA, smoke peak RSS, wall time and estimated hours; Claude launches.
- The pool stays fed meanwhile with dev-seed lane jobs and c5-next(b) (the ~5.3 GB/turn-kind worker retention,
  a separate dev-seed CPU lane job; RULING battery-prereg-1012).

## 8. What this does not claim

- No result. This document registers the run and its bars; every number comes later, in one RESULT filed after
  the shards exist.
- No ownership claim. S2 is a lesion measurement; host-decision share (docs/plans/2026-09-25-prove-who-owns-
  the-computation-PLAN.md) is not measured here.
- No mouth-arm claim. The B2d mouth arm (b2d0929-mouth) is NOT re-run here; this run's S1/S2 are stub-renderer
  numbers, as B2d's headline was.
- No latency gate (speed is secondary; non-negotiable 4) and no long-conversation claim (the B2d turns are
  short sessions; D6 read-time costs are the 2026-09-25 capacity-curve finding's territory).
- No scale or phenomenal claim: the C5 pair's g6b precedent is tiny-demo brain; honesty boundary (non-
  negotiable 5) applies to every reply-level read-out.

## 9. Open questions (UNVERIFIED where marked)

- **(UNVERIFIED) The iw2-flip-hold conflict.** RULING iw2-flip-hold (owner 2026-10-01) keeps the iw2 flip commit
  "on research/integration-wave2 and off main", but at current main the three iw2 anchors read True and the flip
  commit 1fb805e9e is an ancestor of main (checked with git on 2026-10-12). The table reports the code state;
  whether the hold was later lifted by a ruling I could not find is a question for Claude before the run.
- **(UNVERIFIED) The iw1 per-packet step-R verdict files.** The iw1 flip list is recorded as RULING iw1-R in the
  runbook, and the flip commits say "Validation pending (shared prereg steps B, F)"; I did not locate a filed
  iw1 step-R RESULT finding equivalent to the three 2026-10-05 iw2 findings. The d6 row's own capability gate
  (GO 6/6) is cited in its row docstring. If an R artifact exists elsewhere, cite it here before the run.
- **(UNVERIFIED) Exact registry row count at the pin.** The B2d run had 43 rows; the flip commits lifted the iw
  rows and `category-inherit` into the headline registry, so this run's row count is larger. Fix the exact list at
  job-generation time with `research/packets/b2d-mouth-renderer/registry_fingerprint.py` and record it beside the
  JOBS list (the registry_at_H precedent).
- **(UNVERIFIED) Whether a single composed guard prefix passes the queue's line-shape check** (assert_flipped_defaults
  + assert_iw1_defaults + assert_iw2_defaults on one line); if the line-shape gate objects, register the merged
  guard as a small tool in the JOBS commit, before any shard runs.
- Stale docstrings, noted not relied on: `webapp/d6_hebbian_chat.py` and `research/runners/d6_hebbian_store.py::
  hebbian_store_enabled` say "Default OFF" while the anchor is True; `onebrain_xedge_surprise_worldmodel_
  production.py`'s comment says "(OFF)" while `_XEDGE_SWM_DEFAULT_ON = True`. The anchors and the readers govern.

## AMENDMENT B (2026-10-13, filed before any c5val_1012 or b2d1012 shard was read)

**Filed by Claude on 2026-10-13, in its own commit, under RULING battery-amend-B-1013.** It folds in RULINGS
c5-battery-lesion, c5-seed-provenance, lbf-host-input-rows and iw2-transitive-engagement, plus five constraints from
the 2026-10-13 science review: the commit-lesion arm runs at the pin; the exposure arms run at a revision R_B
branched from the pin; check E3 ties R_B to the pin; the C5 memory bar is read inside one revision with
session release OFF; every C5 arm sets BRAIN_CHAT_SEED explicitly. This document has no Amendment A. The "A2" in
section 5 C1 belongs to the 09-30 C1 prereg. B takes its letter from the ruling.

**Revision 1 (2026-10-13, Amendment B review, both lenses FIX_FIRST; filed before any B arm ran and before any
c5val_1012 or b2d1012 shard was read).** First filing a93fc12ea. Revision 1 adds the TAG-LESION and RE-TELL arms
(B.2, B.3), discloses that STAGGERED and ROTATED run the same race (B.3), rules the transitive row a stop-gate row
(B.5), adds the commit-gate-silence condition to B-C5a, the faculty-name host-input detection in `lb_shard.py`, the
all-cells defined count and the host-substitutable report (B.4), the provenance disclosures below, and replaces
R_B (B.1). Every change is stricter or report-only.

**Revision 2 (2026-10-13, re-review FIX_FIRST; still before any B arm ran and before any c5val_1012 or b2d1012
shard was read).** It discloses that a host argmax over the organ's own tag passes every counted clause, so C5
ownership of content selection is reported UNDETERMINED (B.3), names the next method (a race-steer arm, design only),
makes a silent tag-lesion arm undecided (never "moved") in B-C5d, corrects the dev count (69 of 80), records the
re-tell observation as a production gap (B.2), surfaces the stop-gate status in the top-level verdict with S0
governing (B.5), and keeps stop-gate rows out of `lb_shard.py`'s numerator too. Stricter or report-only throughout.

**Stricter only.** No bar is loosened, and no seed, turn or arm is removed. Every bar in section 5 (F1, C1, C5, B1,
B2, F2/F3) stands as written. Each B clause adds a condition, or removes numerator credit or definedness.

**Data state at filing.** The c5val_1012 and b2d1012 shards that landed before this amendment stay data and are
unread. Two workstation dev-seed-7 runs were made for B, under `research/findings/raw/_regression_battery/
c5val_1012_dev/`. (1) The commit-lesion check, selfinit turn only, ran 2026-10-13 02:28 from the uncommitted tree
at bef736ce7, BEFORE the first filing (a93fc12ea); its output was read after that commit. It is a dev check, not
the smoke of record: JOBS 9.1.1 (commitlesion at P, the full pair turns) is. (2) The re-tell smoke of B.3, after
revision 1's runner change and before this revision's commit. Dev seeds carry no weight.

**Stand-in preview disclosure.** During the review a verifier previewed slot outcomes on GATE seeds with a stand-in
idle-replay organ (the organ module alone, synthetic encoding records: no battery shard, no brain). The builder did
not see that preview, and no gate-seed result of it is given here. The bars added by revision 1 were set from DEV
seeds 1-8 only, with the same stand-in
(`docs/drafts/idle-replay/c5_amendB_standin_dev.json`), before any gate data.

### B.1 Revisions

- **P** = f24963bfb05ca535f10f70bf4380fe4b2f68a9ea, the battery pin of the C5 JOBS list.
  `BRAIN_IDLE_REPLAY_COMMIT_LESION` and `BRAIN_IDLE_REPLAY_TAG_LESION` exist at P
  (`research/runners/idle_replay_organ.py:148` and `:144`). The registered OFF and ON arms and B's COMMIT-LESION arm
  run at P.
- **R_B** = 1304d8d6788125c20f3de912d258ea008f34ad89 (branch `claude-build/battery-amend-b-pin`), branched from P. It is P plus one change: the
  label-only C5B session block in `research/runners/onebrain_regression_battery.py` (`_EXTRA_TURNS` gains
  c5st_t1..c5st_t4, c5st_idle, c5rt_t1..c5rt_t4, c5rt_idle, c5re_t1..c5re_t6, c5re_idle). No `sim/`, `webapp/` or
  organ file differs from P, and the PROBE_TURNS roster (26) is unchanged. The same block sits on the scorer branch
  (`claude-build/battery-amend-b`); the scorer reads `C5B_SESSIONS` from it. The first filing's R_B (e70e5c995,
  without the re-tell turns) is superseded; nothing ran on it.
- **Reference for "changes which fact":** the STAGGERED session at R_B against the ROTATED session at R_B (one
  revision, one release mode); the TAG-LESION arm against the STAGGERED arm at R_B. None is compared with the pin's
  ON arm on which fact is voiced.

### B.2 Arms per gate seed (nine; wspool only; one arm in flight globally, section 5 of the JOBS list)

| arm | revision | runner env prefix (after the guards) | turns |
|---|---|---|---|
| off | P | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=… (registered) | emo,bc_a,bc_b,selfinit |
| on | P | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY_LEAKLOG=… (registered) | emo,bc_a,bc_b,selfinit |
| commitlesion | P | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY_COMMIT_LESION=1 BRAIN_IDLE_REPLAY_LEAKLOG=… | emo,bc_a,bc_b,selfinit |
| off_rb | R_B | as off | emo,bc_a,bc_b,selfinit |
| on_rb | R_B | as on | emo,bc_a,bc_b,selfinit |
| stagger | R_B | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY_LEAKLOG=… | c5st_t1,c5st_t2,c5st_t3,c5st_t4,c5st_idle |
| rotated | R_B | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY_LEAKLOG=… | c5rt_t1,c5rt_t2,c5rt_t3,c5rt_t4,c5rt_idle |
| taglesion | R_B | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY_TAG_LESION=1 BRAIN_IDLE_REPLAY_LEAKLOG=… | as stagger |
| retell | R_B | BRAIN_CHAT_SEED=N BRAIN_IDLE_REPLAY_LEAKLOG=… | c5re_t1,…,c5re_t6,c5re_idle |

Facts: F1 "the lynx chases the vole", F2 "the heron hunts the frog", F3 "the yak eats the grass", F4 "the crow
chases the wren"; every word was stored by the tiny-demo brain at dev seed 7 through the same handler. STAGGERED
teaches F1 F2 F3 F4, one per turn, then takes the idle turn. ROTATED is the ruling's "permuted exposure": it teaches
F3 F4 F1 F2 (the order rotated by two), then idles. TAG-LESION is STAGGERED with every engram's excitability set
equal (the organ's own tag lesion). RE-TELL teaches F1..F4, then re-tells two EARLY-slot facts late: c5re_t5
re-tells F2 verbatim, c5re_t6 re-tells F1 with a new patient ("the lynx chases the frog"; the production
reconsolidation path was INTENDED to rewrite a contradicted fact in its own slot; the dev-7 smoke observed it
APPEND instead, lynx chase frog at kb_idx 10 beside lynx chase vole at kb_idx 5), then idles. That observation is a
production gap, recorded here and not fixed by B: a re-told or contradicted fact should update its existing
engram, not duplicate it. Each session is its own fresh
worker build. The code calls ROTATED `rotated` because `gates/attribution_required` reads the word "permuted" in
runner code as a control arm.

Every C5 arm (the registered pair included) sets BRAIN_CHAT_SEED explicitly, carries no other BRAIN_* key than its
row above, and carries no LB_* key: `LB_BATTERY_RELEASE_SESSIONS` stays unset (session release OFF), so the memory
bar keeps measuring production-like accumulation. Files: `c5_s<N>_<arm>.json`, its `.prov.json` and
`c5_s<N>_<arm>_rss.jsonl` in `research/findings/raw/_regression_battery/c5val_1012/`.

### B.3 C5 clauses (scorer `tools/c5_battery_score.py`)

**The confound the review found (disclosed).** The organ lists candidates in store-slot order
(`idle_replay_organ.py:441`), builds its circuit from (K, seed) and draws its noise from (seed, turn, K). STAGGERED
and ROTATED have the same K (9), the same idle turn and the same slot ages, so on one seed they run the SAME race
and differ only in what each slot holds, and the newest fact always sits in the LAST slot. Hence: B-C5b holds by
construction for any brain that picks by slot; B-C5c's 12 sessions are 6 PAIRED outcomes, not 12 draws; and a host
that voices the last slot (newest-appended, i.e. LIFO) and honours the commit lesion read C5 STANDS 50 of 50 in the
review. The first filing's bar-sanity info brain `info_host_fifo_lesion_aware` was mislabelled (it voiced the
newest, last slot: LIFO, not FIFO) and passed 1000 of 1000. It is renamed `newest/LIFO`. Only the TAG-LESION arm and
the RE-TELL arm (newest tag in an EARLY slot) separate recency from the store slot.

**What STANDS means.** "C5 STANDS (PROVISIONAL)" means the idle reply selects by tag recency. It never means "the
spiking race owns content selection": a HOST argmax over the organ's own tag (newest t_enc wins, slot 0 on ties,
silent under the commit lesion; no spiking race) passes every counted clause (300 of 300 in the re-review;
bar-sanity `info_host_tag_argmax` 1.000) and would pass a counted re-tell too. The tag-lesion and re-tell arms
separate the TAG from the store SLOT only, never the race from a host argmax. C5 ownership of content selection is
therefore reported UNDETERMINED (race vs host argmax), not owned; the scorer writes this as `reading`.

**Ruling on the next method (no-defer, 2026-10-13).** The named next method is a counted RACE-STEER arm: during the
idle race, inject drive into a NON-newest candidate's engram population at a level set on dev seeds; a spiking race
should then voice the steered fact, and a tag argmax would not change. Design only (hook, dev-only bar rule, nulls):
`docs/drafts/idle-replay/WORKLOG.md`, "Amendment C candidate". It is not built into B; B ships with the UNDETERMINED
reading.

Bar-sanity record `docs/drafts/idle-replay/bar_sanity_record_c5_battery_amendB.json` (n=1000, synthetic; re-recorded
for revision 2): the perfect brain (which now also models the dev tag-lesion silence, 1 of 80) passes 0.973 and the
noisy perfect brain 0.953; all 16 null brains pass 0.000, including the tag-lesion-inert brain, the always-silent
tag-lesion brain, and the newest/LIFO host (last slot, honours the commit lesion, blind to the tag lesion), which
B-C5d fails. Informational, disclosed: the host tag argmax passes 1.000; a newest/LIFO host that also mimics the tag
lesion with a random pick passes 0.986; a brain whose re-tell voices the last slot passes 0.975, because the re-tell
arm is informational (below). No behavioural clause here excludes those hosts; that is why STANDS reads only
"selects by tag recency" and ownership is UNDETERMINED.

- **B-SEED (validity; RULING c5-seed-provenance).** Every arm's sidecar exists; its `env.BRAIN_CHAT_SEED` equals the
  seed; its git_sha equals P for off/on/commitlesion and R_B for the R_B arms; its BRAIN_* env is the arm's row in
  B.2 (BRAIN_CHAT_SEED and BRAIN_IDLE_REPLAY_LEAKLOG allowed) and it has no LB_* key; no turn body carries `_error`.
  Else the seed is UNDEFINED and no clause reads that arm, so invalid data cannot fail a clause.
- **C5.1, C5.2** unchanged (section 5), read on off and on at P.
- **C5.3** unchanged in substance, with two readings fixed: 2 GB = 2e9 bytes (1953125 KiB, stricter than 2 GiB), and
  the ON−OFF difference is read only between off and on at P (one revision, release mode unset in both). No RSS
  difference is ever read across revisions or across release modes.
- **B-C5a (the commit lesion changes the idle reply).** The lesion reached the organ (`idle_replay.lesions.commit`
  true) and removed the pathway's output: no commit-gate spike (`sum(selection.gate_counts) == 0`, docs/TERMS.md
  "lesion"); else UNDEFINED. The commitlesion idle reply differs from the on idle reply AND is the neutral idle line
  (abstained, n_utt 0, no winner, text exactly `idle_fallback_text()`, "Nothing in particular is surfacing for me to
  bring up right now."). emo/bc_a/bc_b are byte-identical across off, on and commitlesion.
- **E3 (cross-revision identity).** off_rb and on_rb are whole-record identical (every turn body, sorted-key JSON)
  to off and on at P. Else the seed is UNDEFINED, and its R_B session clauses read undecided.
- **B-C5b (exposure changes which fact).** Validity first: each session's idle turn ran the organ unlesioned with
  K >= 1, and every taught fact is a candidate with t_enc >= 1, strictly increasing in teaching order; else
  UNDEFINED. Then each session voices (n_utt 1, composer brain_store, decode_matches_winner) a fact taught in that
  session, and the STAGGERED fact differs from the ROTATED fact. Kept as registered; it holds by construction for a
  slot-picking brain (above), so it carries no evidence against one.
- **B-C5c (pooled; the organ's design prediction).** The most recently taught fact (F4 after STAGGERED, F2 after
  ROTATED) is the voiced fact in all but at most 2 of the 12 exposure sessions. The two sessions of a seed run the
  same race, so the 12 sessions are 6 paired outcomes: the 2 misses allowed may fall on one seed (both sessions) or
  on two seeds (one session each). The two sessions' two newest facts are disjoint ({F4, F3} against {F2, F1}).
- **B-C5d (pooled; the tag lesion moves the voiced fact; counted).** Validity per seed: the tag-lesion idle turn
  records `lesions.tag` true with the other lesions off, equal `x_applied`, and the same candidate store (facts and
  slots) as the intact STAGGERED arm, and a winner that fails to decode makes the seed UNDEFINED. `moved` = the
  tag-lesion VOICED fact differs from the intact STAGGERED voiced fact. A SILENT tag-lesion turn (nothing latched) is
  undecided for B-C5d and never counts as moved. Bar: at least 4 of the 6 seeds MOVED and at most 2 unmoved;
  undecided (C5 UNDEFINED) when silent seeds leave neither settled. Making a silent seed UNDEFINED for the whole seed
  was tried first and made the bar UNREACHABLE at the dev silence rate (bar-sanity perfect 0.920 < 0.95), so silence
  is undecided inside B-C5d only. Set on DEV seeds 1-8 with the stand-in organ at the STAGGERED shape (K 9, idle
  turn 5): the tag lesion moved the voiced fact on 8 of 8 dev battery-noise draws and in 69 of 80 extra dev draws
  under the scorer's definition (5 of 10 on the worst dev seed, 8; 1 of the 80 latched nothing); intact voiced the
  newest fact in 79 of 80. Out-of-sample check by the re-reviewer (stand-in, dev seeds 9-40): moved 31 of 32, newest
  32 of 32.
- **B-C5e (pooled; the re-tell).** Validity per seed: the newest-tagged candidate (unique max t_enc) is a re-told
  fact sitting in an EARLY slot (kb_idx below the last slot); else that seed's re-tell is undecided. Outcome: that
  fact is the voiced fact. Bar: at most 1 miss of 6. Dev seeds 1-8 stand-in: the in-place re-told fact won 80 of 80
  draws (both possible store shapes). **Registered INFORMATIONAL (dev-7 smoke, before gate data).** The re-tell
  smoke (`research/findings/raw/_regression_battery/c5val_1012_dev/c5_s7_retell.json`, run 2026-10-13 08:20 at
  33294dadf, clean tree, under the brain lock; runner identical to R_B's) shows both re-tells APPENDED a store slot:
  F2 re-told verbatim moved to kb_idx 9 (t_enc 5), and F1 re-told with a new patient was stored as a new fact (lynx
  chase frog, kb_idx 10, t_enc 6) beside the untouched lynx chase vole at kb_idx 5. The production reconsolidation
  path was intended to rewrite it in its slot and did not on this route (the production gap of B.2). The newest-
  tagged fact therefore sat in the LAST slot (K 10, voiced lynx chase frog), where recency and a last-slot host
  agree. So B-C5e is reported, never in the verdict (`RETELL_COUNTED = False` in the scorer). Even a counted re-tell
  would separate tag from slot only, not the race from a host tag argmax. The retell arm still runs on every gate
  seed (no arm is removed).

**Verdict precedence.** A failed clause on any seed, or a failed pooled clause: "C5 REVERT-CANDIDATE seed <n>
clause <k>". Else any UNDEFINED seed, or an undecidable counted pooled clause: "C5 UNDEFINED" (RULING s2-c5-val (b):
re-queue that seed's missing arms once at the same revision, then RESULT). Else "C5 STANDS (PROVISIONAL)" with the
`reading` above. No reverting or tuning on gate seeds; Claude reverts on a REVERT-CANDIDATE.

**Seed 42 is NON-BLIND.** The g6b pair ran at seed 42 and was read on 2026-10-12, before this prereg. The RESULT
reports that cell as non-blind; it still counts, and no seed is removed. In addition, the 2026-10-12 "dev-7"
identity smoke (c5-flip-merged-1012's "4/4 whole-record byte-identical") actually ran at seed 42: its prov
(`sim-pool-node/.../1e2f1ddc4.../c5_offcurve_diag/identity_off_at_F.json.prov.json`) records no BRAIN_CHAT_SEED,
and the worker's unset default is 42. So a seed-42 OFF record was also seen before this prereg. That artifact is
committed separately with its prov file (server side).

Scorer command (after all 54 C5 arms land):
`.venv/bin/python tools/c5_battery_score.py --dir research/findings/raw/_regression_battery/c5val_1012
--pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea --rb-pin 1304d8d6788125c20f3de912d258ea008f34ad89
--out research/findings/raw/_regression_battery/c5val_1012/c5_score.json`

### B.4 tom-false-belief is a HOST-INPUT lesion (RULING lbf-host-input-rows)

The row's lesion (`BRAIN_FALSE_BELIEF_LESION`) forces open the witness gate, whose witnessing input is host-parsed;
the belief store's own lesion reads STORE-NOT-LOAD-BEARING on 6/6 gate seeds
(`research/findings/raw/_fb_store/verdict.json`). Rule for b2d1012 and b2d1012-alloff: `tools/b2d_score.py` (default
mode) keeps the row's cells in every level's per-seed denominator but never in a numerator (S0, S1 and S2), never
in robust_core or union, and reports them in each level's `host_input` block. The registry is keyed by faculty name,
so it applies to shards already generated at P (whose lb.json says "neural-lesion"); `tools/lb_shard.py aggregate`
detects the row by the same name registry. Its F1 is reported as "host-input lesion F1 n/6", never as a neural F1.
Whether the BRAIN_FALSE_BELIEF_CHAT flip stands on a host-input F1 is a Claude ruling in the RESULT, not an
automatic STANDS. The row module is retagged `kind="host-input"`; `load_bearing_fraction.py` and `lb_shard.py`
count host-input rows in the denominator only. B1 per-faculty comparisons do not include the row.

Report-only additions (no count moves): `counts.n_defined_all_scored` (DEFINED over every scored cell, host-input
included) beside `counts.n_defined`, which counts the coverable neural-lesion cells INCLUDING the stop-gate cells
of B.5 and excluding host-input cells; and a `host_substitutable` block for
`neural-selfid` (RULING iw2-selfid-answer-rate: its intact reply equals the host router's, so its load-bearing
reading shows the spiking route is used, not that a host could not produce the reply).

### B.5 transitive-chat is a STOP-GATE row (RULING iw2-transitive-engagement; stricter reading ruled 2026-10-13)

Faculty key `reasoning-transitive-chat` (the `transitive-chat` row of section 2). Engagement: a DEFINED cell counts
only when every arm file (intact_a, intact_b, lesion) carries, on the probe turn, a `transitive_trace` record with
route `chase`, a non-empty `chase` hop list, no `error`, no turn `_error`, and `lesioned` true exactly in the lesion
arms. Otherwise the cell reads `UNDEFINED:engagement-<reason>`: B2's re-run rule applies. The 10-05 R data predate
the engagement record (2aac93d43), so their F1 6/6 is UNDEFINED on engagement and is not cited as F1 evidence.

Stop-gate (ruled before any b2d1012 read): the row's cells sit in each level's per-seed denominator and gate the
flip, and are reported in a separate `stop_gate` block; they never count in the S0, S1 or S2 numerators, robust_core
or union, in `tools/b2d_score.py` and in `tools/lb_shard.py aggregate` alike. S0 (the registered trace verdict) is
the governing level for the gate. A failing or UNDEFINED transitive cell at S0 blocks the flip: the top-level
`verdict` then ends "; STOP-GATE BLOCKED at S0: reasoning-transitive-chat", `stop_gate` carries the S0 gate per
seed, and the CLI summary prints it. A passing row is reported as "not blocked", never as "validated", and is not
cited as spiking ownership (the reach sub-flip stays off).

### B.6 Scorers registered before B are unchanged

`tools/iw1_score.py` and `tools/iw2_score.py` call `b2d_score.score(..., amendment_b=False)`, because their preregs
were registered before B and their bars do not change after the fact. In that mode a shard whose lb.json says
"host-input" for a registry row (written after the retag) is scored under the kind it was registered with
("neural-lesion"). `tools/b2d_score.py --no-amendment-b` reproduces the pre-B scorer: its output on the landed
b2d0929 and iw2-flip-1fb805e shards is identical to the pre-B scorer's (checked 2026-10-13).

### B.7 JOBS

The added C5 lines are section 9 of `research/coordination/2026-10-12-validation-battery-JOBS.md`. 9.1 holds the
dev-7 lines, run first: commitlesion at P (the smoke of record), the dev-7 E3 pair (off and on at P, off_rb and
on_rb at R_B) and the four session arms at R_B. 9.2's 42 gate lines (seven per seed: commitlesion at P; off_rb,
on_rb, stagger, rotated, taglesion, retell at R_B) are released only after every 9.1 line lands rc=0 AND dev-7 E3
holds (off_rb and on_rb whole-record identical to off and on). Memory is budgeted per SESSION (the worker builds
one brain per session, about 5.2-5.9 GiB each; the 7-turn dev-7 re-tell session peaked 6.06 GiB): after the 9.1 smokes, the stagger, rotated, taglesion and retell
gate lines are re-stamped mem_gb = ceil(1.3 x that arm's dev-7 peak). R_B must be provisioned on the pool
(`~/derisk-pool/revisions/<R_B>`) before its lines dispatch. This amendment queues nothing.
