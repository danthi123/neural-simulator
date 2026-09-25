---
type: finding
status: live
lane: load-bearing
date: 2026-09-25
mechanism: the two new `research/runners/lbf_rows/learning.py` load-bearing-fraction rows for D5 learn-through-use
  consolidation (`BRAIN_D5_CONSOLIDATE`) and offline sleep-replay (`BRAIN_SLEEP_REPLAY`), scored at seed 7 against
  the pre-registration's own turn groups and the runner's own registered `compare()`/`measure_faculty()` rule.
seeds: [7]
seed-waiver: dev/calibration seed 7 only, exactly as the pre-registration scopes this lane ("Seed 7 is a
  DEV/calibration seed ... no GO/NO-GO capability claim is made here" / "Do NOT run 6-seed evaluations"). This
  document does not claim a 6-seed capability verdict for either row; it scores what the dev-seed data shows.
verdict: UNDEFINED for both rows, not NO-GO. The runner's own `compare()` reads `pass` / `load_bearing=False` for
  d5-consolidate AND sleep-replay (0 treatment diffs, 0 control diffs, reproduced), but `tools.lab.attributable_to`
  reads UNDEFINED on both (0/0 -- no effect in EITHER arm to attribute), because the compared field
  (`episodic.graded_cue.depth_hold`) never populates in ANY arm at ANY turn for either faculty. Root cause: each
  row's TEACH turn (`d5c_teach` = "the wolf chase the rabbit"; `slp_teach2` = "the owl chase the mouse") fails
  comprehension's patient-role disambiguation (`comprehension.comprehended=false`, margin below threshold), so the
  taught topic is never registered as a known referent; the later "you mentioned the wolf" / "you mentioned the
  owl" recall turn then fails `extract_referent()` (the word is not in the known-topics set) and never reaches the
  D5 Hook-A episodic-recall path at all, in EITHER arm. This is a probe-construction precondition failure, not a
  measurement of either mechanism -- confirmed independently by the runner's own `_held_through_tick` diagnostic,
  whose `required=True` lever on d5-consolidate's intact arm RAISES (pre-tick == post-tick == null), which is also
  why the live smoke run died there and never produced sleep-replay's own held-through-tick pair, the knob-off
  byte-identity probe, or the final `smoke_s7.json` summary the pre-registration's Commands section expects.
runner: research/runners/load_bearing_fraction.py (rows from research/runners/lbf_rows/learning.py)
prereg: research/findings/2026-09-24-learning-rows-d5-consolidate-sleep-replay-lbf-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_lbf_rows_learning/intact_a_d5c_teach_d5c_recall1_d5c_tick_d5c_recall2_s7.json
  - research/findings/raw/_lbf_rows_learning/intact_b_d5c_teach_d5c_recall1_d5c_tick_d5c_recall2_s7.json
  - research/findings/raw/_lbf_rows_learning/lesion_d5_consolidate_s7.json
  - research/findings/raw/_lbf_rows_learning/lesion_d5_consolidate_s7.json.rep0
  - research/findings/raw/_lbf_rows_learning/held_intact_d5_consolidate_s7.json
  - research/findings/raw/_lbf_rows_learning/held_lesion_d5_consolidate_s7.json
  - research/findings/raw/_lbf_rows_learning/intact_a_slp_teach1_slp_teach2_slp_teach3_slp_tick_slp_recall_s7.json
  - research/findings/raw/_lbf_rows_learning/intact_b_slp_teach1_slp_teach2_slp_teach3_slp_tick_slp_recall_s7.json
  - research/findings/raw/_lbf_rows_learning/lesion_sleep_replay_s7.json
  - research/findings/raw/_lbf_rows_learning/lesion_sleep_replay_s7.json.rep0
  - research/findings/raw/_lbf_rows_learning/score_seed7.json
external: none new this round (the mechanism itself already carries its 6-seed organ-level GOs cited by the
  pre-registration: 2026-08-21 d5-learn-through-use flip, and the sleep-replay soak
  `research/findings/raw/_sleep_replay_flip/soak_summary_6seed.json`).
---

# d5-consolidate / sleep-replay LBF rows at seed 7: UNDEFINED — the probe's own precondition never held, in either row (2026-09-25)

## 0. Scope of this document

This scores the seed-7 dev/calibration battery the 2026-09-24 pre-registration staged, using data that already
exists on disk (`research/findings/raw/_lbf_rows_learning/`, produced 2026-09-24 14:45-15:29 EDT on the pool at the
pinned revision `46425c2b6b9e01a8d6b488b653d492ac65c2c765`). No new brain build was run to produce this document —
every number below is either read directly from those ten already-produced arm files or recomputed from them with
the runner's own unmodified `compare()` / `_n_decision_diffs()` / `_classify_diffs()` / `tools.lab.attributable_to`
functions (`research/findings/raw/_lbf_rows_learning/score_seed7.json`, this document's own derived artifact).

## 1. Completeness and liveness (checked before scoring, per the task's own requirement)

- **All 10 registered arm files + their `.prov.json` sidecars (20 files) are present** and load cleanly: 4 arms
  per faculty (`intact_a`, `intact_b`, `lesion`, `lesion.rep0`) for BOTH d5-consolidate and sleep-replay, plus the
  `held_intact`/`held_lesion` diagnostic pair for d5-consolidate only. No file is truncated or unreadable.
- **Every `.prov.json` sidecar records the pinned revision in full** (`git_sha: 46425c2b6b9e01a8d6b488b653d492ac65c2c765`,
  `source_kind: git_archive`, `git_dirty: false`) — verified across all 10 sidecars, not sampled.
- **Not still running, anywhere.** `ps -eo etimes,args` on pool1/pool2/pool41/pool42 (read-only, `ssh -n -F
  research/queue/.pool_ssh_config`) shows zero `_lbf_rows_learning_smoke` or `load_bearing_fraction ... d5-consolidate/
  sleep-replay` seed-7 processes (pool1/pool2/pool42 are busy with an UNRELATED 6-seed `b2b0924-base` battery at
  seeds 101/102; pool41 is idle); a local `ps` also finds nothing. The last file write was 2026-09-24 15:29:55 EDT,
  ~22 hours before this check — the run is dead, not paused.
- **The full smoke script (`research/runners/_lbf_rows_learning_smoke.py`) did NOT finish** — `held_intact_sleep_replay_s7.json`
  / `held_lesion_sleep_replay_s7.json` (the sleep-replay half of `_held_through_tick`), the knob-off byte-identity
  probe (`knob_off_a_s7.json`, `knob_off_b_s7.json`), and the final `smoke_s7.json` summary it writes at the very
  end of `main()` are all absent.
  Section 4 below shows exactly where and why it died. **This does not make the registered measurement itself
  incomplete**: `lbf.run(out_dir=..., only=["d5-consolidate","sleep-replay"], repeats=2, seed=7)` — the call that
  produces the ten core arm files `compare()` actually scores — completed for BOTH faculties before the crash; only
  the two DIAGNOSTIC calls that ran afterward (explicitly documented as "NOT part of the registered row's own
  compare()") are missing. So this battery is ripe to score on its registered rows, exactly as flagged.

## 2. Per-gate table (the runner's own registered rule, recomputed from the saved arms)

| Gate (faculty) | turn | compared fields | compare() verdict | treatment_diffs | control_diffs | null_control_clean | lesion.rep0 reproduces | `attributable_to` | registered `load_bearing` |
|---|---|---|---|---|---|---|---|---|---|
| d5-consolidate | `d5c_recall2` | `episodic.graded_cue.depth_hold`, `answer` | `pass` | 0 | 0 | true | true (also `pass`) | UNDEFINED (0/0 — no effect in either arm) | `False` (per the literal rule) |
| sleep-replay | `slp_recall` | `episodic.graded_cue.depth_hold`, `answer` | `pass` | 0 | 0 | true | true (also `pass`) | UNDEFINED (0/0 — no effect in either arm) | `False` (per the literal rule) |

Both rows' `answer` field is identical across `intact_a`/`intact_b`/`lesion`/`lesion.rep0` (all four arms render the
same failed-recall sentence, so `compare()` finds `on_val == off_val` and no diff), and
`episodic.graded_cue.depth_hold` is **absent** (`episodic: null`) in all four arms at the driving turn, for BOTH
faculties (`score_seed7.json`, `graded_field_by_arm`). `_classify_diffs([])` returns `"none"` — there is nothing to
classify, because there is no diff at all, not a diff that happens to be zero.

## 3. Why the driving field never populates (the precondition chain, read directly from the arms)

1. **The teach turn fails comprehension's patient-role disambiguation.** `d5c_teach` ("the wolf chase the rabbit")
   and `slp_teach2` ("the owl chase the mouse") both read `comprehension.comprehended=false`, `abstained=true`, with
   the role margin sitting under threshold (margin ~0.10 vs threshold ~0.20 in both cases — `score_seed7.json`
   `teach_margin`/`teach_threshold`). Contrast this with the SAME turn group's own OTHER two teach facts —
   `slp_teach1` ("the fox chase the hare") and `slp_teach3` ("the hawk chase the vole") — which both read
   `abstained=false` (comprehended cleanly), and with the pre-existing, already-registered `episodic-memory` LB row's
   own driving pair (`epi_store` = "the dog chase the cat", `onebrain_regression_battery.py:113`), which also stores
   cleanly. The two facts these two NEW rows chose to recall from later ("wolf"/"owl") are specifically the two
   role-ambiguous ones; the two facts that DO comprehend (fox/hare, hawk/vole) are not the ones probed.
2. **Because the teach turn never comprehends, "wolf"/"owl" are never registered as a known topic**, so
   `webapp/server.py`'s Hook-A block (`_ep_topics = getattr(chat, "agents_set", None) or _brain_vocab(chat)`) has
   nothing to match against.
3. **`is_referential("you mentioned the wolf")` correctly returns `True`** (`d5_episodic_production_organ.py`'s
   `_REFERENTIAL_RE` includes the literal `you mentioned` alternative — this phrasing is a real trigger, confirmed
   working elsewhere in the same file's own `is_referential`/`extract_referent` pair), **but
   `extract_referent(msg, _ep_topics)` returns `None`** because `"wolf"` is not in `_ep_topics` (`webapp/server.py`
   ~L5499: `if ref is not None:` gates the entire D5 recall block). With `ref is None`, `episodic_info` stays `None`
   for the whole turn — the D5/sleep-replay Hook-A path is never entered, in either arm.
4. **The turn falls through to a different, unrelated fallback** (curiosity's own topic extraction), which is what
   produces the garbled observed reply ("I haven't learned about mentioned yet: what can you tell me about
   mentioned?", `curiosity.topic: "mentioned"` — the literal word "mentioned" is picked up as the topic, not
   "wolf"/"owl"). This confirms the failure is upstream of, not inside, the D5-consolidate/sleep-replay mechanisms
   themselves: neither faculty's own code path is ever reached to lesion.
5. **The runner's own `_held_through_tick` diagnostic independently confirms this is a VOID precondition, not a
   negative reading.** For d5-consolidate, `tools.lab.lever("... pre-tick -> post-tick", pre_i=None, post_i=None,
   required=True)` — required because "if it doesn't [move], the whole probe is void" per the smoke script's own
   comment — **raises `LeverError`** (`score_seed7.json` `held_lever_raises: true`,
   `held_lever_error`: *"did not move (None -> None): both arms are identical, so any A/B over this is VOID"*).
   This is exactly why the live 2026-09-24 run stopped right after d5-consolidate's held-through-tick pair and never
   produced sleep-replay's own diagnostic pair, the knob-off probe, or `smoke_s7.json` (Section 1): the script's own
   internal safety check fired and killed the process, consistent with everything the file timestamps show.

## 4. What this DOES show

- **The LBF row plumbing is wired and reachable end-to-end.** Both rows resolve their turn groups, build fresh
  brains under the pinned revision, and produce a well-formed, deterministic `compare()` verdict (`lesion.rep0`
  reproduces the same verdict as `lesion` for both faculties) — the `research/runners/lbf_rows/learning.py` import
  hook, the `EXTRA_LESIONS`/`EXTRA_PROBES` row shapes, and the turn-group derivation all work as designed.
- **Provenance is clean** — every arm file's sidecar names the pinned revision, `git_archive`, not dirty (Section
  1), so the data is trustworthy as a record of what that revision's code actually produced on this input.

## 5. What this does NOT show

- **Nothing about whether D5-consolidate's or sleep-replay's own BTSP reactivation is load-bearing.** The compared
  field never exists in ANY arm, so `compare()`'s `pass` verdict here means "identical failure mode on both sides
  of the lesion flag," not "the mechanism has no effect." `attributable_to`'s own UNDEFINED read (0 treatment / 0
  control — a null, not a negative) is the correct honest characterization per `tools/lab.py`'s own documented
  degenerate case ("no effect in either arm is a NULL; calling it 0% ... would fabricate an attribution out of an
  absence"), and per this project's standing rule that **UNDEFINED is never scored as 0**. Per `docs/TERMS.md`'s
  `works/solved` condition ("the capability gate passes, not a proxy of it") this cannot be reported as either a
  GO or a clean NO-GO on either mechanism.
- **This is also NOT a capability verdict of any kind at any seed count.** Seed 7 is the pre-registration's own
  declared dev/calibration seed; per this lane's own scope ("Do NOT run 6-seed evaluations") and this project's
  6-seed-validation standard, no generalization is claimed here regardless of the per-gate outcome.
- The two faculties' existing organ-level GOs (2026-08-21 D5 learn-through-use flip; the 2026-08-26 sleep-replay
  6-seed soak, `research/findings/raw/_sleep_replay_flip/soak_summary_6seed.json`) are **untouched by this
  document** — this is a statement about ONE harness's ONE seed-7 realization of a NEW measurement instrument, not
  about the underlying mechanisms those prior GOs already established.

## 6. Next step, per THE LAW (a verdict on the METHOD, not a license to park the CAPABILITY)

The failing element is the turn-group's own fact choice, not the D5-consolidate/sleep-replay mechanism and not the
row-registry plumbing. Concretely, for whoever next revises `research/runners/onebrain_regression_battery.py`'s
`d5c`/`slp` turn groups (`_EXTRA_TURNS`, ~L406-426):

1. **Swap the RECALLED fact to one that this build's comprehension organ actually stores.** `d5c_teach` should use
   an unambiguous predator-prey pair the same way `slp_teach1`/`slp_teach3` and the existing `episodic-memory` row's
   own `epi_store` ("the dog chase the cat") already do, and `slp_recall` should target one of the already-clean
   `slp_teach1`/`slp_teach3` facts (fox or hawk) instead of the ambiguous `slp_teach2` (owl/mouse) — or replace
   `slp_teach2`'s fact with an unambiguous pair too, since the row's own design intent (batch-replay of 3 stored
   episodes) does not require the SPECIFIC fact recalled to be the ambiguous one.
2. **Verify the fix statically first** (no brain build): the pre-existing `episodic-memory` row and `slp_teach1`/
   `slp_teach3` are proof that an unambiguous "X chase Y" pair reaches `comprehended=true` on this build; re-running
   just the teach turn for the replacement fact and checking `comprehension.comprehended` before re-running the
   full held-through-tick pair would have caught this precondition failure for ~1 turn's cost instead of a full
   dead 45-minute run.
3. **Re-run the seed-7 smoke exactly as pre-registered** (`bash tools/mem_ok.sh 10 4 && ... research.runners
   ._lbf_rows_learning_smoke --seed 7`) once the fact swap lands, to get the first genuine numeric reading of
   whether the graded field actually rises intact-vs-lesion — only THEN does a `load_bearing` verdict for either
   row mean anything.
4. This is logged as a newly-noticed failure per the gate matrix (`research/FAILURE_LOG.md`, 2026-09-25 entry) —
   the class (a probe's own teach-fact choice silently fails comprehension's role-disambiguation margin, voiding
   every downstream referential-recall turn in the same session) is not currently caught by any static gate,
   because whether a specific SVO pair clears the comprehension margin is a live-build fact, not a source-tree
   property.

## Derived
<!--derived-->
Section 3's "margin ~0.10 vs threshold ~0.20" are rounded from the exact values in `score_seed7.json`
(`teach_margin: 0.09999999999999998`, `teach_threshold: 0.19999999999999998`, both faculties identical).
