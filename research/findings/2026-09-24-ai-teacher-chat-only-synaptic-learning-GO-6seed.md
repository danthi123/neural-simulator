---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: D6-learn-and-grow
mechanism: chat-only AI-teacher social environment (host code, only channel is the /api/brain-chat text handler)
  teaches K=4 invented-noun facts entirely through conversation; retention across a sleep-idle interval with the
  teacher absent is attributable to the brain's own D6 Hebbian synaptic write, per the pre-registered T1-T10 gate
  (amendment 2, including the new SHAM arm)
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-ai-teacher-environment-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_ai_teacher/v1/verdict_6seed.json
  - research/findings/raw/_ai_teacher/v1/s42_*_K4.json
  - research/findings/raw/_ai_teacher/v1/s43_*_K4.json
  - research/findings/raw/_ai_teacher/v1/s44_*_K4.json
  - research/findings/raw/_ai_teacher/v1/s100_*_K4.json
  - research/findings/raw/_ai_teacher/v1/s101_*_K4.json
  - research/findings/raw/_ai_teacher/v1/s102_*_K4.json
verdict: GO 6/6 -- T1-T10 (amendment 2, incl. the SHAM arm/T10) all pass on every registered seed; aggregate
  n_defined=6, n_go=6, n_nogo=0. Not a flip candidate yet -- the combined-battery no-regression run and a
  production-default validation run have not been done.
---

# The chat-only AI teacher's taught facts are carried by the brain's own synaptic write: GO 6/6

## Result

The pre-registered aggregate score (`research/runners/ai_teacher_experiment.py --score-only`, the registered
amendment-2 command) over all seven gated K=4 arms (TEACH, NOTEACH, FREEZE, ZERO, PERM, ERR, SHAM) on every
registered seed reads GO: `research/findings/raw/_ai_teacher/v1/verdict_6seed.json` --
`n_defined=6, n_go=6, n_nogo=0, GO=true, status="GO"`. This closes the amendment-2 UNDEFINED status recorded at
commit `43a337be1` (the SHAM arms landed on pool2 on 2026-09-24, 20:29-21:12 EDT, at commit `d460b4498`; the T1-T9
arms were already in at `c12c0d47e`).

| seed | T1 learn | T2 lesion | T3 freeze | T4 zero | T5 permuted | T6 error | T7 controls | T8 isolation | T9 no-write | T10 sham | seed status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | GO |
| 43 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | GO |
| 44 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | GO |
| 100 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | GO |
| 101 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | GO |
| 102 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | GO |

Every one of the 102 registered preconditions the scorer checked (per-seed arm presence and provenance, plus
per-criterion measured-ness) reads true; none failed.

All seven gated arms plus the K=2/K=8 sweep arms per seed carry clean provenance. The six T1-T9 arms of every seed
are at one revision (`c12c0d47e`, the pre-registration commit itself), `git_dirty=false`, source manifest verified
at both start and exit. SHAM is exempt from the same-revision rule by the pre-registration and lands at
`d460b4498`, a git-verified descendant of `c12c0d47e`, also clean with a verified manifest.

## What the GO actually shows

- Facts taught only through chat (no LTM bundle; no wikidata store reachable) are recalled after a sleep-idle
  interval with the teacher absent (T1), and recall disappears when the teacher never spoke (T2), when the
  conversational write is frozen (T3), and when the taught synapses are zeroed (T4) -- the recall is carried by
  the conversational synaptic write, not a host leak or another per-session buffer.
- The brain learns and holds a counterfactual the teacher asserts, not the vetted ground truth (T5), and it holds
  a teacher's error on the corrupted facts while keeping the clean ones (T6) -- what gets encoded is what the chat
  channel actually said, not something read off a hidden ground-truth source.
- The SHAM control (T10, new in amendment 2) reruns every taught block through the exact ablation code path with
  its own weights written back (no cut), and cuts an equal-size off-target set instead: SHAM loses none of the
  taught facts TEACH recalls. This is the specificity check T4 lacked before amendment 2 -- the recall loss under
  ZERO is specific to cutting the taught synapses, not an artifact of running the ablation machinery itself.
- The isolation guard (T8) and the test-phase write count (T9) hold on every gated arm: nothing outside the chat
  handler reaches the store, and no write happens during the teacher-absent test.
- Learned-content fraction: every seed reaches the K=4 ceiling of four taught facts recovered as new content in
  the test phase (the two build-time controls were already known before the session and are not counted as newly
  learned).

## Not a flip candidate yet

Per the owner's flip bar (6-seed GO, SOUND review, no regression in the combined production-default battery with
the flag ON, and a production-default validation run): the 6-seed GO is now met, and the underlying code was
already re-reviewed SOUND after fix round 2 at the merge (`b342e2571`). Still missing, and not attempted by this
finding: a no-regression run of the combined production-default battery with the AI-teacher / D6 Hebbian-store
flags ON, and a production-default validation run. No default is flipped here.

## Declared residuals (from the pre-registration, unchanged)

Carried over, not re-litigated here: the teacher is template host code (the social environment) judging replies by
a string match; the sleep interval is a host clock advance that runs the server's own idle tick; the D6 store's
own declared residuals (the host-wired instructive pathway, the host phase-lock loop, the W_MAX clamp, one
disjoint block per fact with exact host routing); comprehension routing (the verb lemmatizer and the B3
polar-assertion extractor) is a host rule, not a brain mechanism.

## Honesty

Functional read-outs only: "learns" / "recalls" mean a reply's content changes with a synaptic write, verified by
lesion (T3/T4/T10). No claim of experience.
