---
type: preregistration
status: preregistration
date: 2026-10-29
packet: c3-flip
runner: research/runners/stage2_exit_gate.py (groups conv, infer, emo, inner; arm intact_a)
scorer: research/runners/c3_flip_compare.py, tools/answer_rate.py (classify, FLIP-HURTS-ANSWERS)
---
# Curiosity L+P become the production default (PREREGISTRATION draft, 2026-10-29)

flip: yes
host-decision: none - both flags select the curiosity faculty's own neural teaching signals (the topic-stop organ's
follow-up decision and the topic-locked phasic SNc read) inside an existing path; they add no host rule, word list
or table and decide no content.
smoke-run: research/findings/raw/_c3_flip/smoke_verdict.json (reader GO, run_id 1793457559-45080)
smoke-run: research/findings/raw/_c3_flip/smoke_on/ + research/findings/raw/_c3_flip/smoke_off/ (dev seeds
7 and 8, 16 cells, revision e22d5b2b10d3f6724ad52f95f9e65ba31639c373, main >= 8d94cf94e)

smoke-exempt-ruling: c3-flip-smoke-1031
bar-sanity: N/A identity bars: B1-B3 compare ON with unset turn by turn and have no threshold, so no perfect brain fails them; the null case they would pass (flags set, faculty inert) is closed by the ENGAGEMENT precondition of RULING c3-flip-smoke-1031 (section 2)

Drafted by the main driver on Claude's instruction and approved by Claude 2026-10-31 (RULING `c3-flip-prereg-1031`: bars diffed, verbatim) (RULINGS `c3-flip-1029` step 2 and `c3-flip-1029b`), before any
gate-seed run. The bars in section 2 are `c3-flip-1029b`'s own words, asserted as substrings of the runbook entry
when this file was built; Claude diffs the bars before any gate-seed data. The C3 gate that this flip follows PASSED
(research/findings/2026-10-29-c3-topic-stop-gate-RESULT.md); that gate measured the ask group itself, so these bars
are about the REST of the brain. No gate seed (42, 43, 44, 100, 101, 102) has been run for this flip; the only data
below are dev seeds 7 and 8.

## 1 What flips

| flag | old default | new default | anchor |
|---|---|---|---|
| `BRAIN_CURIOSITY_TOPIC_STOP` (L) | OFF (unset reads False) | **ON** | `webapp/curiosity_topic_stop_chat.py:41` `_flag("BRAIN_CURIOSITY_TOPIC_STOP")` |
| `BRAIN_CURIOSITY_PHASIC_DA` (P) | OFF (unset reads False) | **ON** | `webapp/curiosity_topic_stop_chat.py:49` `_flag("BRAIN_CURIOSITY_PHASIC_DA")` |

The flip commit does not exist yet and is not part of this file. It is made only after section 2 passes; unset reads
ON and an explicit `0`/`false`/`no`/`off` stays the escape back to today's behavior, both pinned by tests, with a
dev-7 post-flip smoke where unset reproduces the gate's C intact_a asked vector, then a verify-go panel before the
merge (Pass clause of `c3-flip-1029`, which `c3-flip-1029b` points to).

## 2 Bars, fixed before any gate-seed data

The ruling, verbatim (`research/coordination/LOCAL_LLM_RUNBOOK.md` section 6, `c3-flip-1029b`, which supersedes
`c3-flip-1029`'s bar sentence where they differ):

> Bars, replacing the c3-flip-1029 bar sentence: on the six gate seeds, Stage-2 groups conv, infer, emo and inner, arm intact_a, flags ON vs unset at one revision F: abstained and recalled_svo are identical ON vs OFF on every turn; answer is identical on every turn that did not abstain in both arms; no turn where OFF answered and ON did not (tools.answer_rate.classify, FLIP-HURTS-ANSWERS); every other differing leaf is reported (counts and first path per group), never read as a pass. Pass -> as in c3-flip-1029; anything else -> STOP with the data.

The same bars, split into clauses. Each clause is a word-for-word substring of the ruling; the split changes nothing.

- **SCOPE.** "on the six gate seeds, Stage-2 groups conv, infer, emo and inner, arm intact_a, flags ON vs unset at one revision F"
- **B1.** "abstained and recalled_svo are identical ON vs OFF on every turn"
- **B2.** "answer is identical on every turn that did not abstain in both arms"
- **B3.** "no turn where OFF answered and ON did not (tools.answer_rate.classify, FLIP-HURTS-ANSWERS)"
- **B4.** "every other differing leaf is reported (counts and first path per group), never read as a pass"
- **DEC.** "Pass -> as in c3-flip-1029; anything else -> STOP with the data."

The Pass clause DEC points to, verbatim from `c3-flip-1029`:

> Pass -> the flip commit (unset reads ON, an explicit 0/false/no/off escapes; both pinned by tests; a dev-7 post-flip smoke where unset reproduces the gate's C intact_a asked vector), then a verify-go panel before the merge; anything else -> STOP with the data.

**ENGAGEMENT (precondition, added by Claude before any gate-seed data, RULING `c3-flip-smoke-1031`).** A group whose ON arm carries no `curiosity_topic_stop` trace on any turn of a seed reads UNDEFINED for that seed, never PASS (verify-go lens (a): an identity bar is passed by a flag that does nothing). Dev smoke: traces on/off 8/0, 2/0, 16/0, 2/0.

Scoring of each clause (how it is read; the bar text is above and is not changed by this):

- **B1, B2**: `python -m research.runners.c3_flip_compare --on <ON dir> --off <OFF dir> --seeds 42,43,44,100,101,102
  --out <OUT.json>`; the reader is UNDEFINED unless every cell exists on both sides, the ON sidecars carry both flags
  = 1, the OFF sidecars carry neither, and every sidecar names one revision. Its per-cell PASS on `abstained` and
  `recalled_svo` identity is B1; on `answer` identity for turns that did not abstain in both arms is B2.
- **B3**: `tools/answer_rate.py` classify over the same cells; FLIP-HURTS-ANSWERS reported and never read as a pass.
  The reader's own record of answered on/off per cell is read beside it.
- **B4**: the reader's group lines carry other-leaf-diffs counts and the first differing path per group; they are
  reported in the RESULT and never read as a pass.
- **DEC**: B1 to B4 all hold -> the Pass clause above; anything else -> STOP with the data, naming the clause.

## 3 Protocol (sketch; gate lines are emitted after Claude diffs section 2)

**Gate revision F.** One revision for both arms (the ruling: "flags ON vs unset at one revision F"), pinning
`research/runners/c3_flip_compare.py` and the Stage-2 harness. Gate seeds 42, 43, 44, 100, 101, 102 only, only from
lines emitted against this committed prereg.

**Stage-2 lines.** The `c3-topic-stop-gate-lines.txt` shape (mem 9 / memcap 10, one BLAS thread, frozen harness),
filtered to groups conv, infer, emo, inner, arm intact_a, 6 seeds: ON with `BRAIN_CURIOSITY_TOPIC_STOP=1
BRAIN_CURIOSITY_PHASIC_DA=1` and `--out` under `research/findings/raw/_c3_flip/gate_on/`; OFF with both variables
absent (unset, not "0") and `--out` under `.../gate_off/`. 48 lines. Before `pool_queue.sh` accepts a gate-seed line
it still needs a passing pool canary at F and the prereg check against this file.

## 4 Dev-seed smoke (seeds 7 and 8), recorded before this file

Smoke-first: 16 lines (4 groups x 2 seeds x ON/OFF), arm intact_a, revision `e22d5b2b10d3f6724ad52f95f9e65ba31639c373`
(main >= reader commit 8d94cf94e), landed as `research/findings/raw/_c3_flip/smoke_on/` and `.../smoke_off/`; reader
run `python -m research.runners.c3_flip_compare --on research/findings/raw/_c3_flip/smoke_on --off
research/findings/raw/_c3_flip/smoke_off --seeds 7,8 --out research/findings/raw/_c3_flip/smoke_verdict.json`.
Output, quoted:

> C3 flip compare: GO  (ON research/findings/raw/_c3_flip/smoke_on vs OFF research/findings/raw/_c3_flip/smoke_off)
>   cell 7/conv     PASS  answered on/off 7/7  both-abstained 4  other-leaf-diffs 7
>   cell 7/infer    PASS  answered on/off 9/9  both-abstained 1  other-leaf-diffs 2
>   cell 7/emo      PASS  answered on/off 5/5  both-abstained 8  other-leaf-diffs 16
>   cell 7/inner    PASS  answered on/off 9/9  both-abstained 1  other-leaf-diffs 2
>   cell 8/conv     PASS  answered on/off 7/7  both-abstained 4  other-leaf-diffs 8
>   cell 8/infer    PASS  answered on/off 9/9  both-abstained 1  other-leaf-diffs 3
>   cell 8/emo      PASS  answered on/off 5/5  both-abstained 8  other-leaf-diffs 16
>   cell 8/inner    PASS  answered on/off 9/9  both-abstained 1  other-leaf-diffs 2
>   group conv  2/2 PASS  other-leaf-diffs 15  first {"path": "responses.p_more.curiosity.curiosity_topic_stop", "off": "<absent>", ...}  topic-stop traces on/off 8/0
>   group infer 2/2 PASS  other-leaf-diffs 5  first {"path": "responses.c_prey_chase.curiosity.curiosity_da", ...}  topic-stop traces on/off 2/0
>   group emo   2/2 PASS  other-leaf-diffs 32  first {"path": "responses.s_joy_1.curiosity.curiosity_da", ...}  topic-stop traces on/off 16/0
>   group inner 2/2 PASS  other-leaf-diffs 4  first {"path": "responses.c_mite_all.curiosity.curiosity_da", ...}  topic-stop traces on/off 2/0

Read beside the bars, never as them: on dev seeds every cell passes the four clauses' shapes (no answered-off-and-not-on
turn anywhere; both-abstained turns per cell as listed), the ON sidecar flag set is exact, and the other-leaf diffs are
the expected curiosity-state traces (topic-stop traces present ON, absent OFF). Dev seeds are two seeds, not the six
gate seeds; nothing here is a verdict.

## 5 What this does not claim

- No gate-seed result and no verdict; the smoke is two dev seeds.
- B4's other-leaf diffs (15/5/32/4 on dev) are reported facts, never a pass, on any seed.
- The C3 gate's ask-group PASS is this flip's precondition, not its evidence; these bars test the REST of the brain.
- The curiosity readouts are functional read-outs of the organ's state (CLAUDE.md non-negotiable 5), never a claim
  of felt experience.

## 6 Owed before any gate line is queued

- Claude's diff of section 2's bars against the ruling (explicit: "Claude diffs the bars before any gate-seed data").
- Gate revision F, the emitted lines, and the pool canary at F.
- Answer if any clause wording needs to change; the driver changes nothing in section 2.
