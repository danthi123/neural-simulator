# Result: open-conversation baseline, 1.5B defaults, seeds 1100-1105. After one night, day 2 collapses into one repeated sentence on 4/6 seeds

Method: `brain/research/2026-11-15-open-conversation-eval.md`; `brain/eval/open_conv.py`, pool CPU fp32. Hosts are in
each transcript header: pool40 node-mxl1462drl ran 1100/1101, pool41 node-dl4g243 1102/1103, pool42 node-dl5d243
1104/1105. All rc 0. Brain defaults throughout: S0 key, gain 0.5, self_completion on, 1.5B sleep 480 steps, lr 5e-4,
KL 3, rank 8. Transcripts: `2026-11-15-open-conv-baseline-1.5b/oc_s*.md`; rows are in `*.rows.json`.

## What the transcripts show

Day 1, before any sleep, is ordinary small talk on all six seeds. The brain is polite and a little generic ("That's a
cute name for a rabbit."), it answers the day-1 question from memory, and its replies never echo the user's turn.

After the night, day 2 is broken on four seeds (count of the most repeated reply, of 16 day-2 turns):

| seed | intact arm | hippocampus-off arm |
|---|---|---|
| 1100 | "You've been told." x11 | varied |
| 1101 | "I went for a long walk by the river." x13 (a day-1 user line) | echoes each user turn verbatim, 14/16 |
| 1102 | "I've been trying to go to bed earlier lately." x13 (a day-1 user line) | the same line x14 |
| 1103 | "I finally finished reading a long novel." x14 (a day-1 user line) | "Do you ever get bored?" x10 |
| 1104 | varied, at most x2 | varied |
| 1105 | varied, at most x2 | varied |

The collapse persists with the hippocampus off (1102, 1103), so it is in the cortex's slow synapses, what the night
wrote, not in recall. The night trains next-token prediction on the stored user sentences: 480 steps on about 10 short
texts, protected only by a KL term on three fixed chat snippets. It learns to produce those sentences.

The counted measures are not informative until this is fixed. Day-2 recall is 0-3 of 3 per kind and swings with the
collapse; intrusions, as counted by key words, stay low because a repeated chat line contains no told key.

## What it means

- The fixed weekly hid this. Its short session 1 led to partial versions: "How are you feeling today?" ->
  "I drive a red Volvo."
- The banked "false recall after sleep 6.83/12" is probably the same defect seen through a probe.
- Every sleep result that counted told answers after the night measured a cortex that had partly lost its speech.
- This is the biggest defect, and it is in the night, not the memory.

Next: the day-2 control on the same slept homes, with the night's synapses off and the hippocampus on
(`brain/eval/open_conv_day2.py`). If replies recover there, the fix is a night that protects the cortex's own
behaviour broadly, for example by rehearsing its own generated conversation, judged on this eval.
