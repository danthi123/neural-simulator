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
