# Result: own-cue frame selection, A1 fails (best frame 2/36 GOOD on DEV), no gate run; the grading rule was too strict

Prereg: `brain/research/2026-11-15-own-cue-frame-1.5b.md`. `brain/experiments/own_cue_frames.py` ran on pool40, 41 and
42. Greedy decoding gave identical results on all three: `frames_*.json`, with the host in each file.

| frame | DEV GOOD (of 36) | user voice | contains key | is a question |
|---|---|---|---|---|
| F0 near-miss OWN | 1 | 1 | 11 | 36 |
| F1 "to check they remembered" | 0 | 0 | 1 | 36 |
| F2 chat frame | 0 | 0 | 0 | 0 (writes emoji) |
| F3 "a question to see if they remember" | 0 | 0 | 5 | 36 |
| F4 M3's SELFQ_PROMPT | 2 | 2 | 5 | 36 |

**A1 fails as registered: no gate run.** The grading rule is wrong, though, and should not be read as "the cortex
cannot write the user's question". The rule required no "you/your". F1 writes good user-voice checks that the rule
rejects: "What did you say my hamster's name was?" and "Do you remember what I told you about my uncle?". It also
leaks the key only 1/36 times.

The bar is not moved after the fact. The teacher night is set aside because the night itself is the larger defect
(`2026-11-15-open-conv-baseline-1.5b.md`). If it is taken up again, F1 needs a re-registered grade that judges the
subject (my X), not the pronoun.
