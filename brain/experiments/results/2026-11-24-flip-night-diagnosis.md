# First-person "flip nights" in slept lives: weights in B, a battery cascade in K-P (2026-11-24, host-only, DEV)

Why: two Fable reviews of the source-hold DRAFT (brain/research/2026-11-24-self-other-tag-prereg-DRAFT.md, BLOCKED) found
K-P's hippocampus-off echo concentrated on whole nights. No model was run; banked DEV lives only (Bg, KP, PRa, PRl, PRx, S1F,
M1 s319-321; AL s322; S1FULL s318; four 7-day s318 lives). Scripts and tables in `2026-11-24-flip-night-diagnosis/`
(`flip.py` -> `flip_out.txt`, `nights.tsv`; refuter's `cascade.py`, `fresh2.py`, `classify.py`, `name.py`, `dump.py`).
Measurer and refuter: Fable. The refuter reproduced every number and overturned the K-P/M1 reading.

## Findings that survive the refuter
- **No keep-bank: the cortex drifts into the user's voice with age, and it stays.** Flip nights (hippocampus-off told rows
  >= 4/8 first person in the user's voice), nights 2-15, 3 seeds: Bg 27/42, S1F 34/42, PRx 28/42, PRl 19/42, PRa 16/42;
  P(flip | previous flip) Bg 20/24. Fresh-state reads (cue ladder, interview forced turns) confirm it is in the weights:
  0.42-0.49 user-voice even on non-flip mornings.
- **The keep-bank (K-P) cuts this.** Strict user-voice nights K-P 5/42, M1 6/42; K-P recovers the next night 5/5.
  Night-15 KL 0.32 (K-P) vs 2.47 (Bg), adapter norm 363 vs 656.
- **K-P's remaining "flip nights" are mostly a conversation cascade, not weights.** The probe battery is one running
  conversation (`run_battery` resets state only for the open prompts). P(first person | previous row first person) = 0.83
  vs 0.17. In K-P/M1, if the first told row (cat) is not first person, later rows are 6.7% first person; if it is, 57%.
  Fresh-state reads on the same mornings: K-P 0.21 on flip nights vs 0.14 on non-flip (Fisher p 0.27); M1 0.19 vs 0.14.
- **Detector holes.** `strict_scorer.is_first_person` fires on any "I", so refusal loops ("I'm sorry, but I don't know what
  you're asking." 8/8, s320 n11) and identity loops ("I'm Dan, and I'm Dan, too.", s321 n9) count as flips. Of K-P's 8 flip
  nights: 4 user voice, 3 refusal loops, 1 identity loop. `is_echo` also scored 4 refusal rows on s320 n12 (true echoes 3,
  not 7). The recall-voice counts use the same rule; they are unchanged here but carry this caveat.
- Not night-locked (flipping lives rise with age n1 1 -> n15 16 of 23), not seed-locked, battery identical across arms.

## Not surviving
- "Global voice, not item-specific" and "bimodal nights" in K-P: the cascade explains both.
- "A lived day with own first-person talk precedes a flip" (K-P 5/10 vs 2/25): of its 5 hits, 3 are refusal nights and
  1 is the stock "my friend" greeting.

## Consequences
- The biggest K-P conversational defect is in-context: one near-threshold first-person (or refusal, or "I'm Dan too")
  reply early in a conversation, which the conversation state then repeats on every later turn. This is the loop onset
  already noted in `brain.py` (`reply_monitor`, `own_gain`). It is a property of talking, not of sleep.
- Counts that treat battery rows as independent (this read, the recall-voice read, TEST v2 bars) overstate n within a
  night. TEST v2 is frozen and is not changed; its read must report per-night clustering beside any row-pooled count.
- The source-hold DRAFT stays BLOCKED: it targets a weight effect K-P already mostly removed.
- Cheapest decisive test (refuter): on pool40, replay K-P s320 to the morning after night 6 and s321 to night 14, then ask
  the battery as banked, each question from a fresh state, and told questions before the name rows. Cascade if the banked
  order flips and fresh state gives <= 1/8; if fresh state flips too, the weights are implicated.

## Cascade test on pool40 (11-24, CPU fp32; M1 night-15 trunk homes; rule fixed before the run)
No per-night K-P homes are banked, so the test used M1's final homes (the night-15 probe weights). pool40 has no GPU: CPU fp32
against the banked A10G fp16 (A matches banked byte-for-byte on 8-14 of 20 rows). Told rows in the user's voice (of 8),
A = banked order, B = each question from a fresh state, C = told questions first. Rule: CASCADE if A >= 4 and B <= 1;
WEIGHTS if B >= 4; UNDEFINED if A does not flip on pool40. Files: `2026-11-24-flip-night-diagnosis/cascade/`.
| seed | arm | banked | A | B | C | verdict |
|---|---|---|---|---|---|---|
| 319 | hippocampus-off | 5 | 5 | 6 | 0 | WEIGHTS |
| 319 | intact | 8 | 8 | 5 | 8 | WEIGHTS |
| 320 | hippocampus-off | 6 | 2 | 1 | 1 | UNDEFINED (flip did not reproduce on CPU) |
| 320 | intact | 4 | 4 | 0 | 3 | CASCADE |
| 321 | both arms | 0 | 0 | 1-2 | 0 | UNDEFINED (no flip) |
- Both mechanisms are real. M1 s319 (the seed whose lived days were full of "X too" replies that lived_key stored) carries
  the user voice in its weights ("My cat is named Pixel" from a fresh state); M1 s320 intact is a conversation cascade.
- So the refuter's "K-P/M1 flips are mostly a cascade" is NARROWED: it holds for s320 intact and does not hold for M1 s319.
  K-P itself is not tested here (no per-night homes). Context still steers even the weight case (s319 hoff C: 0/8).
- One sampled reply per row; near-threshold nights depend on device precision (s320 hoff 6/8 banked vs 2/8 CPU).
