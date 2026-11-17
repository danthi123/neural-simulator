# Pre-registration: the teacher night (the near-miss guard with the day's tellings in the teacher's context), 1.5B

Written 2026-11-17 before any 1.5B run of this switch. Capability: Metacognition + Sleep consolidation (invented
facts). Lever 4 on invented facts at 1.5B, after the research round (wf_27b711a6-6d0) and M1 (falsified, 11-17).

## Record brief
- The slept 1.5B brain confabulates on nearly every never-told question. On the registered mixed day (banked
  `nmoff`, 6 seeds, arm s) it scores 11.83 of 12: 6.83 told names plus 5.00 new inventions. It declines 0.17.
- In open conversation it invents on 96% (`results/2026-11-17-know-signal-told-vs-never-1.5b.md`).
- **In-context ceiling** (`results/2026-11-17-incontext-abstain.md`). The pretrained 1.5B cortex with the 12 tellings
  in one user turn names 22/24 told rewordings and invents on only 19/48 never-told (4/16 untold attributes). On the
  same 48 questions the slept cortex invents 99%, and a reinstated episode 78%. The cortex has the behaviour; the
  night removes it.
- The M4 near-miss guard (11-14) held the pretrained cortex's answers on its own "my X" cues, but with no context.
  That bare cortex invents 79-86%, so the guard taught inventions (false recall 6.83 -> 3.83, half became inventions).
  M4b dropped told-topic cues because the bare teacher refuses told topics.
- **This method changes one variable relative to M4.** The teacher (replies and KL target) reads the day's tellings
  first: what the user said, in the measured one-turn layout. The student never sees the context. The record's A4
  "distilled reinstatement" and the 11-15 teacher night, with the text teacher instead of the reinstated episode.
- Prerequisites met. Host decision: the declared scaffold row "Teacher night context" (`brain/SCAFFOLDS.md`), whose
  opening and acknowledgement are read by the teacher and are never a target. The frame onsets are M4's scaffold,
  unchanged.
- Expected: confabulation down by at least the in-context gap if distillation transfers to the probe questions.
  Risks:
  - The frame cues rarely name told topics, so untold attributes may stay.
  - The told replay CE may still win on "my X".
  - Told recall may drop, as under M4 (7.17 -> 6.00).
- Code: `Brain(teacher_context=True)` with `near_miss=4, near_miss_frame="F"`; `near_miss.guard_texts(context=)`;
  `sleep.py` (prefix, reply, context) triples. Tests: `brain/tests/test_episode_switch.py` (`teacher_context_*`).
  0.1B smoke: triples carry the six slept tellings, and the eval-topic cue "Who is my aunt?" is dropped.

## Part A: registered mixed day (AWS GPU, six seeds)
- `brain.experiments.mixed_day_probe ... --steps 480 --lr 5e-4 --kl-weight 3 --arms c,s,r --self-q --near-miss 4
  --near-miss-frame F --teacher-context`.
- Seeds 42/43/44/100/101/102 with the banked off arm's slept halves (even/odd/even/odd/even/odd).
- Job file `brain/experiments/jobs/2026-11-17-aws-teacher-night.txt`.
- Off arm: banked `nmoff_s*` (AWS L4, batch 20261114-2031). M4: banked `nmon_s*`.
- All files re-scored (`brain.experiments.rescore`), then `near_miss_table <dir> --on nmtc`.

Criteria on arm s (the registered arm), six-seed means, of 12:
- **D1, confabulation down:** false + invented on never-told is at most 8.00 (off 11.83), and lower than off on at
  least 5/6 seeds.
- **D2, told kept:** told at least 6.67 (off 7.17 minus 0.5) and slept told at least 4.17 (off 4.67 minus 0.5).
- **D3, self:** self 3.00.
- **D4, answer rate:** abstentions on told rows at most off + 0.5 (M4b's N6).

D1-D4 = a LEAD: part B decides any default flip. D1 fails = NO-GO for this method at 1.5B. D1 holds and D2 fails =
NARROWED: report the trade; no flip.

## Part B: open conversation (pool CPU, the real defect)
- `brain.eval.open_conv --model 1.5b --seed S --teacher-night` for seeds 1100-1105 (pool40/41/42, two each).
- Compared against the banked default transcripts of the same seeds (`results/2026-11-15-open-conv-baseline-1.5b/`,
  pool CPU). Brain defaults are unchanged since: every switch added after 11-15 is default OFF.
- The run drops frame cues naming this run's never-told topics (test hygiene).

Read-outs, intact arm, summed over six seeds:
- **O1:** never-told invented (`open_conv.invented`), against the banked rows re-scored with the same mark.
- **O2:** recall_ok.
- **O3:** the most-repeated day-2 reply count per seed.
- Every transcript is read whole; a blind read follows if O1 drops.

A default flip needs:
- Part A LEAD;
- O1 lower on at least 4/6 seeds;
- O2 not lower by more than 1 per seed on average;
- no new collapse (O3 not up on more than 1 seed);
- a skeptic.

## Timebox
Invented facts: KEEP or KILL in `HANDOFF.md` on 11-19.
