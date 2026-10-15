---
type: finding
status: confirmed-with-caveats
claim_check: measured (tools/s2p_score.py O and R over the landed shards at P=c64f2c70d; Claude verify-go 2026-10-14: CONFIRMED-WITH-CAVEATS, select/O/R re-scored byte-identical)
date: 2026-10-14
lane: load-bearing (the #1 metric, owner 2026-09-19)
mechanism: word-role-commit organ (default OFF behind BRAIN_WORD_ROLE_COMMIT) gate-tested at P=c64f2c70d8232336992aab074f199c36a6560d45: step O operating point, step R reply-level lesion test of the commit stage plus the no-fact control
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_s2p/word-role-commit/c64f2c7/selection.json
  - research/findings/raw/_s2p/word-role-commit/c64f2c7/verdicts/O.json
  - research/findings/raw/_s2p/word-role-commit/c64f2c7/verdicts/R.json
  - research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md
verdict: PROVISIONAL PASS. O: 6/6 seeds all-pass at op=0. R: 6/6 capability cells DEFINED and regressed under the word_role_commit lesion (n_reply_load_bearing 6 of 6); 6/6 no-fact controls control-ok. <!--derived-->
---

# word-role-commit gate at P=c64f2c70d (default OFF behind BRAIN_WORD_ROLE_COMMIT) -- PROVISIONAL

Pre-registration: `research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md`.
Pins: `docs/drafts/spiking-commit/PINS.tsv` row P = c64f2c70d8232336992aab074f199c36a6560d45 (packet branch
`research/word-role-commit`).

## What a PASS lets the finding say (pre-registration section 7, verbatim)

"With the commit flag on, each word's role in the question is committed by spiking neurons: a role gate that bursts
for at most one role, and a role latch that holds itself after the word ends. The host count rule and argmax no
longer decide the parse; the read of which latch is on does not change when its level is halved or doubled. Cutting
only the commit stage, with the role ensembles firing as before, removes the answer." Credit for word-role-parse
becomes FULL only after a flip packet makes `BRAIN_WORD_ROLE_PARSE` and `BRAIN_WORD_ROLE_COMMIT` the default and its
F1 at the default passes with no FLIP-HURTS-ANSWERS; the residuals in section 2 stay named.

## Step O -- operating point (step-O verdict, committed alongside this finding)

Artifact: `research/findings/raw/_s2p/word-role-commit/c64f2c7/verdicts/O.json`.

| seed | state | all_pass |
|---|---|---|
| 42 | DEFINED | true |
| 43 | DEFINED | true |
| 44 | DEFINED | true |
| 100 | DEFINED | true |
| 101 | DEFINED | true |
| 102 | DEFINED | true |

O verdict: PASS, op=0, n_defined 6, n_all_pass 6; precondition (organ cells DEFINED on 6/6) ok.

## Step R -- reply-level load-bearing (step-R verdict, committed alongside this finding)

Artifact: `research/findings/raw/_s2p/word-role-commit/c64f2c7/verdicts/R.json`.

| seed | lesion verdict | intact answer | lesion answer | control (nofact-ctl) |
|---|---|---|---|---|
| 42 | regressed | the brain uses the spikes | abstain (I don't know about that...) | control-ok |
| 43 | regressed | the brain uses the spikes | abstain (I don't know about that...) | control-ok |
| 44 | regressed | the brain uses the spikes | abstain (I don't know about that...) | control-ok |
| 100 | regressed | the brain uses the spikes | abstain (I don't know about that...) | control-ok |
| 101 | regressed | the brain uses the spikes | abstain (I don't know about that...) | control-ok |
| 102 | regressed | the brain uses the spikes | abstain (I don't know about that...) | control-ok |

R verdict: PASS, op=0; capability cells DEFINED 6/6, reply load-bearing 6/6 (changed fields per seed: abstained,
answer, recalled_svo; answer_only false). No-fact control: intact and lesion answers identical on every seed
(control-ok 6/6); the control's intact arm already declines ('drink' never commits, (brain, None) 6/6), so
control-ok holds even if the lesion broke every parse.

## Diagnostics (intact arm, per seed; intact_a_s2wc_use*.json word_role_trace.commit) <!--derived-->

| seed | co_commit | relatch | read_invariant | read x0.5 == x2 | host-rule agree | latched min | never-latched read |
|---|---|---|---|---|---|---|---|
| 42 | 0 | 0 | true | true | true | 95 | 0 |
| 43 | 0 | 0 | true | true | true | 75 | 0 |
| 44 | 0 | 0 | true | true | true | 85 | 0 |
| 100 | 0 | 0 | true | true | true | 89 | 0 |
| 101 | 0 | 0 | true | true | true | 94 | 0 |
| 102 | 0 | 0 | true | true | true | 88 | 0 |

Across seeds: co-commit reads 0 on every seed (no dual-role commits); relatch 0; the latch read is identical when
every latch level is halved or doubled (read_invariant true 6/6); the committed roles agree with the host `_rule`
record on every word of every seed. Latched min 75 across seeds; never-latched slots read 0 in every R trace
(6/6 seeds; step-O levels 0-1 vs latched 71-85). The 79-93 figure previously labeled "unlatched max" was the agent
latch still holding, not an unlatched slot; the commit gate (threshold 18 bursts over the read window) separates
the slots.

## Pre-registration section 4 (report-only) <!--derived-->

Dual-role reads: 17/24 right, 0 wrong, 7 declined (seed 101: 1/4). The exclusion lesion gives 0/24 co-commits, so
"at most one role" comes from the role layer, and the commit gate's cross-inhibition is not shown necessary.
Count-mode baseline at 9ea7411.

## Not yet a live seam

Both flags stay default OFF; production still parses with BridgeParser. Residual host steps at ON: the commit read
(`spiking_commit.py:181-196`; threshold-invariant, not deciding), the slot -> word mapping, the a==v guard, the
self-alias resolution, and ROLE_MIN folded into W_RG.

## Lesion verification (docs/TERMS.md, class-11 "still holds at the moment of measurement")

The manipulation is not a zeroed weight that plasticity could regrow: `BRAIN_WORD_ROLE_COMMIT_LESION=1` closes the
role -> commit-gate transmission gate for the lesion arm, and the flag is read at each turn, so it holds at the
moment of measurement. Verified in the artifacts: every lesion-arm record carries
`word_role_trace.commit_lesioned = true` (6/6 seeds) and every intact-arm record carries false (6/6), with
`route: word_role_commit` in all of them; the organ has no online plasticity on this path (all commit weights
wired, not learned, per pre-registration section 2), so nothing changes between the manipulation and the read.

## Residuals (pre-registration section 2, still named)

training stimuli transcribed from the composer's fact index; self-alias -> 'brain' resolution and the stop-word
strip; the word-form -> assembly dictionary; the latch read (a level read of spike counts, registered
threshold-invariant above); the inter-word reset sparing the latches (a compression of the inter-word interval);
the organ is its own bridge (one-brain debt); all commit weights wired, not learned.

Credit for word-role-parse stays at its current level; FULL credit needs the flip packet (Claude's to write) with
F1 at the default and no FLIP-HURTS-ANSWERS.
