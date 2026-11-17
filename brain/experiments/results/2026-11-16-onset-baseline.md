# Result: spontaneous in-context collapse of the pretrained 1.5B cortex (report-only baseline)

`brain/experiments/echo_onset_probe.py` (pool40/41/42, CPU), the pretrained cortex alone (hippocampus and night off,
on a copy of a slept home), open_conv's day-2 scripts of seeds 1112-1120, 1122-1124, 6 sampling seeds each: 72
conversations. Raw: `2026-11-16-onset-baseline/onset_base_pool4N.json`. Counts per conversation (16 turns):
pronoun-normalised echoes, and non-echo replies ending "?" (questions back, role confusion).

- 14 of 72 conversations (19%) collapse: 10 echo the user on > 3 turns (13-15 of 16 once started), 5 ask questions
  back on > 3 turns (12-16), 1 both.
- It depends on the script: 1115 collapses in 5/6 samples, 1113 in 2/6, 1124 in 2/6; 1112, 1114, 1118, 1120, 1122,
  1123 in none. Earlier pair (1121, 1118): 1/12.

So the collapse is a property of the 1.5B cortex in this dialogue format (no memory, no sleep), and any brain-side
method is judged against this rate. Scale is the first prerequisite to check: only 0.1B / 0.4B / 1.5B checkpoints
are local; a larger RWKV-7 chat checkpoint (fits the 24 GB reference GPU) needs a download, which is the owner's
call. Reference arm runnable now: the model family's own chat practice carries the repetition penalty across the
conversation with a slow decay; ours resets it every turn (`brain/cortex.py` speak).

**Carry-penalty reference arm (prereg `brain/research/2026-11-16-carry-penalty-1.5b.md`; `onset_carry_pool4N.json`).**
Same scripts (1113, 1115, 1116, 1117, 1119, 1124) and sampling seeds, cortex alone, penalty counts carried across turns
(decay 0.996): collapse 11/36 vs 14/36 for the paired baseline (the prereg's "12/36" miscounted; 1115 sample 0 is
(4 echoes, 4 questions back), counted collapsed). Not <= 6/36: NO LEAD. It clears 1124 (2 -> 0) and leaves 1115
(4/6 echo), 1113 and 1116 as they were. Read: on 1115 sample 1 the echo starts at turn 1 and holds 13/14 turns. The
penalty covers only tokens the brain said, so the first echo is free, as the prereg expected. Where there is no collapse
the cortex alone invents freely ("Your goldfish is called Goldie."), as expected without memory. `carry_penalty`
stays default OFF.

**Scale trend (`onset_04b_pool4N.json`, `onset_15f_pool4N.json`; `--home fresh`, a newly born home).** Same six
collapse-prone scripts x 6 sampling seeds. Loop = pronoun-normalised echo > 3, or the same six-word reply opening on > 3
of 16 turns (one rule for both sizes; the "?" count alone conflates a clarifying template with role confusion):
0.4B 23/36, 1.5B 14/36. The forms differ: 0.4B rarely echoes (1/36) and loops on a clarifying template ("I'm sorry,
but I don't have access to your sister's location. Can you tell me...?"); 1.5B echoes (10/36) or speaks as the user.
1.5B from a fresh home equals the slept-home arm with the hippocampus and the night off on every count (lesion
check: those lesions remove both). Two sizes only, six scripts: a trend, not a law. It is the prerequisite reading
the rules ask for first: the in-context loop shrinks with cortex size, so the next step is a larger cortex (a
download, the owner's call), before more brain-side levers against it at 1.5B.
All 12 scripts (`onset_04b_rest_pool4N.json` adds 1112, 1114, 1118, 1120, 1122, 1123 at 0.4B): loop (same rule) 0.4B
36/72 (50%), 1.5B 18/72 (25%; this rule also counts same-opening repeats, so it is above the 14/72 echo-or-question
count). Echo > 3: 0.4B 5/72, 1.5B 10/72. Looping halves from 0.4B to 1.5B.
