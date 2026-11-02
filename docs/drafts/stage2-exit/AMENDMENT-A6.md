# Stage-2 exit gate, Amendment A6 (DRAFT, for Claude to sign): C2's permuted-control acquisition precondition

status: DRAFT (build agent, 2026-11-01; branch research/stage2-retake-amendments). Not in force until Claude signs it
and the gate's aggregate command passes `--amendment-level A6`.
Ruling (Claude, 2026-11-01): criterion-only variant.
host-decision: none this amendment changes only how the scorer reads a refused teach; it adds no host code that decides a reply
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A6.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A6_fail.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_stage_A6.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_stage_A6_fail.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c1_conv_A6.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c3_ask_A6.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c4_emo_A6.json
bar-sanity: docs/drafts/stage2-exit/bar_sanity_record_s2_c5_inner_A6.json

Amends `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (kept byte-identical, so `PREREG_SHA` stays
`24dac499e`) together with Amendments A3, A4 and A5. Filed after the 2026-10-26 attempt and the 2026-11-01 dev preview,
before the retake's pool lines are emitted. The data read are the 10-26 gate shards (already scored and filed; this
amendment does not re-file them) and the dev preview (dev seeds 1-3, which govern nothing, prereg section 11). Every bar,
counted arm, turn, seed and prediction of every criterion is unchanged. Only the C2 permuted clause gains a precondition.

## Why

At the 10-26 gate, C2 scored FAIL "not exposure-driven" on 0 of 6 seeds. The cause is an acquisition refusal, not a
reasoning result (`research/findings/2026-10-26-stage2-fail-diagnoses.md`, C2 section; the RESULT doc's Claude amendment).
The default-ON D4 comprehension gate refused the first permuted teach `the heron hunts the gnat` on every seed. The teach
reply was "I followed the shape of that, but I don't know the words 'heron' or 'gnat' yet — what do they refer to?",
abstained, and nothing was stored. With hop 1 never stored, the permuted probes could not follow the exposure. The
scorer has no acquisition precondition for C2, unlike A4 (a) for C3, so it scored the refusal as a FAIL of the reasoning
clause. The D4 gate-off arms on dev seeds 1-3 (`research/findings/raw/_stage2_diag/c2_comprehension_gateoff/`) store the
permuted teach, and the conclusion then tracks it (`score_infer(permuted=True)["all"]` True on 3 of 3).

## A6.1 The rule

- **Refused** (`infer_permuted_acquisition` in the runner): the permuted `infer` arm's own reply on a hop teach turn
  (`t_heron_hunt`, `t_frog_eat`, `A6_INFER_HOP_TEACHES`) ABSTAINED. The brain declined to store the teach and said so on
  that turn. The arm's own turn decides, because the question is whether this arm received this exposure.
- **Only a refusal counts, not a missing echo.** A teach reply that did not abstain is scored as received, whatever it
  echoed. A4 (a)'s `teach_acquired` (the triple among `recalled_svo` + `supporting_facts`) is reported per label as
  `carried`, never used to decide. This keeps the exposure-blind brain a FAIL. A permuted arm that answers as if it had
  the intact teaching (no abstain on the teach turn, the intact conclusion on the probe) is still scored, and it fails
  the permuted clause. Reading A4 (a)'s stricter acquisition here would make that brain UNDEFINED; the synthetic
  `null_permuted_inert` brain shows the case (test `test_a6_an_echo_that_drops_the_triple_is_not_a_refusal`).
- **Effect.** With a refused hop teach, the seed's permuted tracking is UNDEFINED (`tracks` None; the pre-A6 read is
  kept as `tracks_as_scored_pre_A6`), never FAIL and never PASS.
- **When the seed becomes UNDEFINED** (the A4 (a) rule, ruling 2026-10-13): only when the permuted clause is what
  decides it. That means intact_a is correct and the chain-route lesion's effect is present and reproduced. If another
  clause already fails, the seed is scored on that clause (FAIL) and the tracking is reported undefined. The stage
  verdict gains the precondition "the infer permuted hop teach received wherever exposure tracking decides a seed
  (amendment A6)", which counts the deciding case only.
- **The stage rule at A6 (Claude's ruling 2026-11-01, the criterion-only variant).** The A6 precondition acts at
  criterion level: C2 is INCOMPLETE. The STAGE verdict is FAIL whenever any counted criterion FAILs. The stage is
  UNDEFINED only when no counted criterion FAILs and at least one is UNDEFINED/INCOMPLETE (or another precondition is
  unmet); otherwise it is PASS as before. Rationale: an undefined C2 must never hide a defined C1 failure. A criterion
  FAILs only on its full set of defined seeds (`aggregate`: fewer defined seeds reads INCOMPLETE). At A6 every
  precondition the stage verdict records is also a per-seed UNDEFINED reason (including the unapplied-lesion rule
  below), so a stage FAIL rests on a criterion whose every seed is defined. The unmet preconditions are still recorded in the aggregate (`preconditions`,
  `undefined_reasons`) and printed by the report; at A6 the aggregate also carries `stage_rule` and `criteria_failed`.
- **An unapplied lesion is an instrument failure (Claude's ruling 2026-11-01, the A6 residual).** At A6 an
  in-process lesion whose patch was never called (`applied` False: the shard's `patch.calls` is 0) adds the reason
  "in-process lesion patch never called (applied False, A6 lesion not applied)" to its seed, so the seed is UNDEFINED
  and that criterion reads INCOMPLETE, never FAIL "not load-bearing". An instrument failure is never evidence that a
  faculty is not load-bearing. Another criterion's defined FAIL (e.g. C1 capability absent) still makes the stage FAIL.
  At A5 the rule is unchanged: the seed stays defined and only the precondition "every in-process lesion applied" is
  unmet (tests `test_a6_a_fail_driven_only_by_an_unapplied_lesion_is_not_a_fail`, `test_a5_keeps_the_unapplied_lesion_reading`,
  `test_a6_a_genuine_c1_fail_still_fails_the_stage`).
- **Report.** At A6 the report prints, per seed, each hop teach's `refused` and `carried`, the tracking as now scored
  and as scored before A6, and the A6 reason. It also prints the stage rule, the criteria that FAILed and the unmet
  preconditions.

## A6.2 How the level is selected (the runner's amendment mechanism)

`AMENDMENT_LEVEL = "A5"` stays the runner's default, and `aggregate(..., amendment_level=None)` reads it. At A5 the
aggregate and the report are byte-identical to the pre-A6 runner (commit `25e360740`) on the 10-26 gate data and on the
dev preview (`a6_real_data_compare.json`: `A5_aggregate_identical_to_pre_A6_runner` and
`A5_report_identical_to_pre_A6_runner` True for both). The 10-26 RESULT therefore re-scores as filed. At A6 the
aggregate adds `amendment_level: "A6"` and the per-seed A6 fields. The retake is scored with:

    .venv/bin/python -m research.runners.stage2_exit_gate --aggregate research/findings/raw/_stage2_exit/gate \
        --seeds 42,43,44,100,101,102 --amendment-level A6 --out research/findings/raw/_stage2_exit/gate/aggregate.json

The level is passed explicitly rather than flipped in the runner. Flipping it would change the runner's sha and void
every bar-sanity record by the `bar_sanity_line` staleness rule. HANDOFF section 8's aggregate line must carry the flag
once this is signed. Without the flag the scorer reads A5, the 10-26 rule, and a refused permuted teach would again be
scored FAIL.

## What the amended scorer returns vs the current one (real data; `a6_real_data_compare.json`)

| data | scorer | stage | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|---|
| 10-26 gate (42-102, pin 9bdb92ee8) | pre-A6 = A5 | FAIL | FAIL | FAIL (0/6 tracked, "not exposure-driven") | FAIL | FAIL | PASS |
| 10-26 gate | A6 (criterion-only) | FAIL (C1, C3, C4 FAIL; A6 precondition 6 and six-seeds 1 unmet, recorded) | FAIL | INCOMPLETE (0 of 6 defined) | FAIL | FAIL | PASS |
| 11-01 dev preview (1-3, 41d76bdbe, 105/105) | pre-A6 = A5 | FAIL | FAIL | FAIL (0/3 tracked) | PASS | PASS | PASS |
| 11-01 dev preview | A6 (criterion-only) | FAIL (C1 FAIL; A6 precondition 3 and six-seeds 1 unmet, recorded) | FAIL | INCOMPLETE (0 of 3 defined) | PASS | PASS | PASS |

On every seed of both data sets the read is the same: `t_heron_hunt` refused True / carried False, `t_frog_eat` refused
False / carried True (`the gnat eats the frog` was stored), intact_a correct, chain-route lesion effect present and
reproduced. So A6 makes every C2 seed UNDEFINED, and the stage is FAIL on C1 (criterion-only rule). The preview's three seed-3 cells that were not committed on the
workstation branch at `25e360740` (emo intact_b, inner intact_a/b) were read from the home server's landed copies
(sha256 in the JSON).

**The ruling (Claude, 2026-11-01): the criterion-only variant.** The draft (commit `de2f61742`) left the stage on the
earned-verdict rule, so at A6 a refused C2 permuted teach made the STAGE UNDEFINED even though C1 FAILs on defined data
(both data sets read UNDEFINED at that draft). The ruling keeps the A6 precondition at criterion level: C2 INCOMPLETE, the
stage FAIL whenever any counted criterion FAILs, UNDEFINED only when none FAILs and at least one is UNDEFINED/INCOMPLETE.
An undefined C2 must never hide a defined C1 failure. `aggregate`'s verdict block implements it at level A6 only; level
A5 keeps the earned-verdict stage rule byte for byte. Either way the retake cannot PASS C2 while the D4 gate refuses
`heron`/`gnat`. A6 only stops that refusal from being reported as a reasoning FAIL. The C2 fix itself is the
acquisition method (stage2-fixes-1026: novel transitives through a learned word-order cue, not a table entry).

## Bar-sanity (n = 1000 per generator, seed 0; `tools/bar_sanity.py`)

Records at the amended scorer (`bar_sanity_s2_common.LEVEL = "A6"`), re-measured at the ruling's runner on
2026-11-01 (the runner's sha moved again; every pass rate is unchanged from the draft's records), and once more after
the unapplied-lesion rule (all nine `_A6` records: every pass rate and n_pass identical; only `scorer_sha256`,
`scorer_files`, `repo_head` and `wall_s` moved). Every pre-existing generator of C1-C5 and the
stage was re-measured, because the runner's sha moved. The A6 code reads only the `infer` permuted arm, so their pass
rates were expected unchanged. Measured: every generator the `_A4` records hold reads the same pass rate in its `_A6`
record (all six records OK). New `infer` generators: `null_permuted_blind_acquired` (takes the permuted teach, answers
the registered conclusion) and `info_permuted_refused` (the capable brain with the 10-26 refusal; reported, never PASS at
either level).

The FAIL claim itself has a spec of its own, `bar_sanity_spec_s2_c2_infer_A6_fail.py`. Its decision is "the aggregate
scores C2 FAIL". The perfect kind is brains that lack a C2 capability: exposure-blind, one-hop, unmarked derivation, an
inert chain-route lesion. The null kind is brains a FAIL would misdescribe: the capable brain with the refused teach, and
the fully capable brain.

| FAIL-claim generator | kind | pre-A6 (A5) | A6 |
|---|---|---|---|
| exposure-blind, teach taken (`perfect`, `perfect_blind_exact`) | perfect | 1.000 | 1.000 |
| one-hop / unmarked derivation / inert chain-route lesion | perfect | 1.000 | 1.000 |
| capable, permuted hop teach refused (`null`, `null_refused_exact`) | null | **1.000 (TOOTHLESS)** | 0.000 |
| fully capable (`null_capable`) | null | 0.000 | 0.000 |

The stage FAIL claim under the criterion-only rule has its own spec, `bar_sanity_spec_s2_stage_A6_fail.py` (decision:
the aggregate's STAGE verdict is FAIL, five groups x six seeds, level A6). Measured at the ruling's runner:

| stage FAIL-claim generator | kind | A6 (criterion-only) |
|---|---|---|
| C1 gap `no_elaboration` + C2 permuted teach refused (`perfect`, `perfect_exact`) | perfect | 1.000 |
| C1 random replies + C2 permuted teach refused (`perfect_c1_random_c2_refused`) | perfect | 1.000 |
| C1 gap `no_elaboration`, C2 teach received (`perfect_c1_gap_c2_received`) | perfect | 1.000 |
| every criterion capable, C2 permuted teach refused (`null`, `null_refused_exact`): stage UNDEFINED | null | 0.000 |
| fully capable (`null_capable`): stage PASS | null | 0.000 |

At the draft runner (`de2f61742`, stage on the earned-verdict rule) the two C1-gap-plus-refusal generators read UNDEFINED,
not FAIL (FAIL on 0 of 5 draws each, an informal check with the draft runner swapped into the spec, not a bar-sanity
record; the C1-gap brain with the teach received read FAIL 5 of 5). The real-data table above shows the same shape.

bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A6_fail_at_A5.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_c2_infer_A4.json
bar-sanity-before: docs/drafts/stage2-exit/bar_sanity_record_s2_stage_A4.json

The A4 records were measured at the pre-A6 runner sha and are stale by sha. At level A5 the scorer is unchanged (byte
identity above), so their reads stand for A5. Cite only the `_A6` records for an A6-scored gate.

## What this does NOT establish

- The criterion-only rule takes a criterion's FAIL at face value once its seeds are defined. The residual the draft
  left (a criterion could FAIL "not load-bearing" on an in-process lesion that was never applied) is closed by the
  unapplied-lesion rule in A6.1: such a seed is UNDEFINED, its criterion INCOMPLETE. No gate shard has `applied` False
  (every counted lesion on 10-26 and the preview is env-flag or patched with calls > 0), so the real-data table is
  unchanged by the rule.

- A6 does not make C2 passable. While the D4 gate refuses the permuted hop-1 teach, every C2 seed whose other clauses
  hold is UNDEFINED.
- The refusal read trusts the teach reply. A brain that abstains on the teach turn but stores the fact anyway would be
  UNDEFINED, not scored. No such reply exists in the data: every refused turn stored nothing (8 facts instead of 9,
  diagnoses doc).
- Only the two hop teaches of the permuted `infer` arm are read. The conv, ask and inner permuted arms keep their
  rules, and ask keeps A4 (a)'s control-arm rule. The intact arm's own acquisition is unchanged: an intact refusal still
  fails `conclusion`/`premise1`, which is a correctness FAIL.
- The C2 conclusion is still read off a host list (`OneBrainComposer.kb`; A5's kb_scramble arms). A6 does not change
  the host-decision share.

## Consequences for the packet

- `PREREG_SHA` is unchanged (`24dac499e`). The prereg file is not touched.
- `GATE_SHA` must contain this file, the A6 runner and `tests/test_stage2_exit_gate_a6.py`
  (`git cat-file -e $SHA:docs/drafts/stage2-exit/AMENDMENT-A6.md`); see AMENDMENT-A7.md (re-pin).
- The gate's pool lines are unchanged by A6 (`tests/test_stage2_exit_gate_a6.py` pins their sha256 against the pre-A6
  runner). No dev line is re-run.
