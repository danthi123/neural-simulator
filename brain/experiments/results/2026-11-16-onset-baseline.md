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
