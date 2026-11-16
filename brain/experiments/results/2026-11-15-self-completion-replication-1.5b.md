# Result: self-completion replicates on seeds 300-305 (C0-C3 hold, GO); `self_completion` is now ON by default. Caveats: the gain is day-1 only, C3 is vacuous again, and more unprompted "I have one too" replies appear (not significant)

Prereg: `brain/research/2026-11-15-self-completion-replication-1.5b.md` (C0-C3 as in the registered test, committed
before any run). Run: AWS batch 20261115-1733 (g6.xlarge L4, instance i-09e4fa94c822390fc, terminated after pull), 12
jobs, all rc 0 (`2026-11-15-self-completion-replication-1.5b/run.log`, 22:41-00:37 serial, about 9.5 min each).
Command: `brain.eval.weekly --model 1.5b --shared-sleep --seed <s>` with and without `--self-completion`.
`tools/where_ran.py`: `unknown=12`. The transcripts record no host. The provenance is the batch log, the instance id
and the `/opt/pytorch` paths in the job logs, all prose only. The inputs are cortex weights and a fresh brain, not
`research/findings/raw/`. Report-only: pool CPU fp32, seeds 306-311 (`pool/`). Scorer
`brain/experiments/self_completion_score.py`.

## Registered seeds 300-305 (`table.txt`)

| | day-1 own name | day-1 cat | session-2 keyed (mean) | told (of 4) | "remember my name?" | session-2 own | hippocampus off keyed |
|---|---|---|---|---|---|---|---|
| off | 4/6 | 2/6 | 4.33 | 3.17 | 6/6 | 1/6 | 5.00 |
| on | **6/6** | 4/6 | 4.33 | 3.17 | 6/6 | 1/6 | 5.00 |

Day-1 turn 1, off: "My name is Alex." (s300) and "My name is a very long word..." (s304); the other four say Nova.
On: "My name is Nova." on all six, with "What's yours?" added on five.

- **C0 holds.** The flag is recorded in every on header and in no off header.
- **C1 holds, with no slack.** 6/6 on vs 4/6 off; the bar is >= 5/6 and >= off + 2.
- **C2 holds.** Session-2 keyed means are equal, and the day-1 cat is 4 on vs 2 off.
- **C3 holds vacuously, as the prereg note predicted.** Session 2 (from `## intact` on) is byte-identical off vs on on
  6/6 seeds. After the night "What's your name?" recalls the sister telling in both arms (see
  `2026-11-15-self-recall-after-night-1.5b.md`), so the friend answer, session-2 keyed, told and hippocampus-off say
  nothing about the switch.
- Abstentions ("I don't know about that."): 0 in all 12 transcripts. `tools/answer_rate.py` does not read this
  layout, so the answer-rate lens was checked by grep only.

## Pooled with the earlier runs (24 seeds; same command; L4 for 42-102 and 300-305, CPU fp32 for 200-205 and 306-311)

| seed set | day-1 own off | day-1 own on | day-1 cat off | day-1 cat on | C0-C3 |
|---|---|---|---|---|---|
| 42-102 (registered) | 2/6 | 6/6 | 5/6 | 4/6 | NARROWED (C2 by one seed) |
| 200-205 (report-only) | 2/6 | 6/6 | 3/6 | 3/6 | hold |
| 300-305 (registered) | 4/6 | 6/6 | 2/6 | 4/6 | **GO** |
| 306-311 (report-only) | 5/6 | 6/6 | 1/6 | 3/6 | C1 fails only by the off + 2 bar (off 5/6) |
| all | 13/24 | **24/24** | 11/24 | 14/24 | |

## The cost no criterion reads (post hoc, `confab_count.py`, `confab_count.txt`)

A skeptic found unprompted self-claims in on-arm session 1, for example "I have a red Volvo too" and "I have a cat
too, but her name is Luna" (s300). The count of brain replies in session 1 matching "I have a/an ...", "I also ..." or
"My ... too" (scored answers excluded), over 24 seeds and 192 replies per arm:

- 6 off vs 16 on.
- Per seed: more in the on arm on 6 seeds, more in the off arm on 2, tied on 16. A sign test gives p ≈ 0.29.
- The claims come in runs of 2-3 within one conversation. That reads as a "me too" mode of the cortex, which the off
  arm shows too (s101, s306).

The measure was not registered and the conversations diverge from turn 1, so the count cannot separate the switch from
path noise. Question-backs ("?" in a reply) are 91 off vs 90 on.

## Skeptics (three sonnet subagents, fresh context, lens block verbatim)

- **Instrument: NARROWED.** "The literal C0-C3 pass is real ... C3, the keyed score and the hippocampus-off arm are all
  vacuous again, because session 2 is byte-identical in both arms ... The on arm also shows unscored confabulation in
  session 1."
- **Control: NARROWED.** "Control integrity holds and the day-1 cat/told cost does not recur ... I recommend the flip
  only if it is scoped as a day-1 own-name fix with C3 explicitly untested."
- **Answers: REFUTED (its default when uncertain).** The reasons were the unmeasured question-back "What's yours?" on
  5/6 first replies, a vacuous C3, unknown host, and two tests that pinned the old default.

## Ruling and flip

I flip the switch on by default, as the prereg's GO branch says. The registered bars hold. Day-1 own name is right on
24/24 seeds on vs 13/24 off. No abstentions appear. The confabulation difference is post hoc and not significant, and
the cortex shows the same mode with the switch off.

The scope is narrow. The switch fixes the brain's own name on day 1. It does nothing after the night, because no self
episode is recalled then: session-2 own name is the same in both arms. The flag acts at birth only, so existing homes
keep what they were born with. `--no-self-completion` in the weekly is now the flag-off arm.

Changed files:
- `brain/brain.py`: default `True`.
- `brain/eval/weekly.py`: `--self-completion/--no-self-completion`; unset = the Brain default; the header follows the
  effective value.
- `brain/tests/test_episode_switch.py`: asserts the new default.
- `brain/tests/test_brain_key_switch.py`: the old-Brain comparison pins `self_completion=False`; the S_all similarity
  test reads `route_delta`, which is what routing reads.
- All 92 brain tests pass.

Watch next: the weekly transcript's session-1 "I have one too" count. The weekly scorer should count it from now on.
