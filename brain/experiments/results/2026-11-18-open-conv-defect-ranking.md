# Open-conversation defect ranking, 1.5B, seeds 1100-1105 (read whole, 11-18)

Method: six sonnet readers each read one seed's three transcripts in full: `2026-11-15-open-conv-baseline-1.5b/oc_s*.md`,
`2026-11-16-open-conv-day2-control-1.5b/d2_s*.md`, `2026-11-17-teacher-night-1.5b/oc/*/octn_s*.md`. An opus ranker merged
the reports and a sonnet skeptic re-read the transcripts and the code to refute the top pick. Workflow wf_9f93f347-562.

## Verdict

The two biggest defects are still the two already attacked: **day-2 collapse (K1)** and **invented or wrong-bound facts
(K2)**. Both appear on 6/6 seeds. No new defect of the same size survived the skeptic.

The ranker's pick, "hippocampal retrieval is not cue-selective", was **REFUTED as framed**:
- A gate exists: `brain/hippocampus.py` FAMILIARITY 0.12, MARGIN 0.02. It is weak, so generic recent lines (the
  "novel" line) pass.
- Night replay is strength-weighted over every episode (`sleep.py`), not driven by retrieval.
- K1 persists with the hippocampus off (oc_s1103 hippocampus off: "Do you ever get bored?" x10).
- Hippocampus on vs off is net helpful across seeds: s1105 7/7 vs 5/7; only s1100 baseline and s1103 teacher night
  are hurt.

The "day-2 echo" is a **harness artifact**: the d2 control has no day-1 history. Do not treat it as a brain defect
without re-running it with day 1.

## Smaller real defects (for later)

| Defect | Seeds | Cause |
|---|---|---|
| Weak recall gate lets generic recent lines in (mostly harmless on day 1) | several | hippocampus threshold |
| After the night: shorter replies, lost follow-ups, told names leak into stories ("a lighthouse keeper named Ignatius") | 1100, 1102-1105 | night training |
| Persona confusion: user sentences spoken in the first person, wrong names | 6/6 | mostly the night (K1 family); some cortex style on day 1 |
| Day-1 style: invented experience, "me too" mirroring, no follow-up questions | 6/6 | pretrained cortex |
| A returning user is not recognised | 6/6 | largely cortex |
| Eval gaps: the user's name is never probed; one day-1 recall probe | - | harness |

## What it means for the next step

Three of the post-night defects (collapse, shorter replies, leaked names) are effects of what the **night writes**: next-token
training on about 10 stored user sentences. The levers killed so far were in-context levers (collapse) and self-teacher
levers (invented facts). The night's own training target has had two tries: rehearse (NO-GO) and habituation (fails R4).
The next target is the night's write, judged on whole transcripts; alternatively, a cortex change.
