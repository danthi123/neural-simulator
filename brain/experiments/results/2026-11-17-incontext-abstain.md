# Result: the pretrained cortex declines never-told questions when the tellings are in context; the night removes it

Probe `brain/experiments/incontext_abstain_probe.py`. Birth context, then all 12 FACTS told in one user turn with one
acknowledgement, then the 72 fresh M1 questions (abstain-v2 fresh v2 + M1 v3) from that state. Pretrained synapses,
no hippocampus, no sleep, greedy, so deterministic: one run per cortex, and seeds do not apply. Host: pool40 (1.5B,
2.9B) and pool41 (0.4B), CPU fp32. Files are in `2026-11-17-incontext-abstain/pool4*/`. Scored with
`brain/eval/open_conv.invented` (a confident answer, with no "don't know" and no question back).

| cortex | told rewordings named (of 24) | never-told invented (of 48) | untold attributes invented (of 16) | untold topics invented (of 32) | told name reused on never-told |
|---|---|---|---|---|---|
| 0.4B | 8 | 17 | 7 | 10 | 5 |
| 1.5B | 22 | 19 | 4 | 15 | 11 |
| 2.9B | 19 | 13 | 8 | 5 | 1 |

The declines are real replies. At 1.5B: "I don't know." / "I'm sorry, but I don't have that information." /
"I'm not sure. I don't know what my cat likes to eat." / "Your cousin's name is not mentioned in the text." One
counted invention is arguably right ("What did I tell you about my new job?" -> "You said you work at a bakery called
Crumb.").

**The same 48 questions in the slept 1.5B brain** (M1 run, six seeds, `2026-11-17-confab-m1-1.5b/fresh_s*.json`):

| route | never-told invented | untold attributes invented |
|---|---|---|
| slept cortex, nothing reinstated (C) | 285/288 = 99% | 96/96 = 100% |
| reinstated episode, pretrained synapses (H) | 225/288 = 78% | 88/96 = 92% |
| text in context, pretrained (above) | 19/48 = 40% | 4/16 = 25% |

The slept cortex names the told key on 107/144 told rewordings.

## Reading
- **The 1.5B cortex already has the "I was not told that" behaviour when its knowledge is plain text in its
  context.** It is best on exactly the case that defeated every read-out (untold attributes of told topics: 4/16).
- **The reinstated state is a weaker carrier than the text.** H invents 78%.
- **The night removes the behaviour entirely.** C invents 99%. This matches the open-conversation finding (night off
  7/13 inventions, night on 15/15).
- So the defect is in what the night teaches, not in a missing signal. Next-token training on the tellings teaches
  "my X -> a name" for every X. This is the fine-tuning-on-new-facts effect: Gekhman et al. 2024 report that it
  raises hallucination.
- **Next lever** (4th at 1.5B, after the research round): a teacher night. The night trains the adapters to give, with
  no context, what this in-context cortex gives with the day's tellings in context. The targets are the cortex's own
  outputs, not host strings. This is the record's "distilled reinstatement" (A4) and the 11-15 teacher night, with
  the text-in-context cortex as the teacher instead of the reinstated episode (78% invent).
- **Ceiling:** the bar for a learned night at 1.5B is about 40% invention (25% on attributes). Below that would need
  more than the cortex's own in-context judgement.

## Caveats
- One context layout (all 12 in one turn).
- Greedy decoding only.
- A told name reused counts as invented when it is the wrong slot.
- The H/C comparison uses gain 4.0 reinstatement and the 11-13 home (480 steps, KL 3).
