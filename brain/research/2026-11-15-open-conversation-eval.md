# Open-conversation eval (method note, 2026-11-15)

Why: the owner asked on 2026-11-15 whether we were building to pass the tests we wrote. Reading a whole weekly
transcript (seed 301, 1.5B, after one night) showed the larger defect the weekly never scores. Told facts intrude into
unrelated turns:
- "How are you feeling today?" -> "I drive a red Volvo."
- the fox story became "a fox named Marguerite ... a red cat"
- "What do I do for work?" -> "You drive a red Volvo."

The hippocampus-off arm held the better conversation.

What: `brain/eval/open_conv.py` draws a different two-day conversation per seed from a held-out bank
(`brain/eval/open_conv_bank.py`: 40 facts with distinctive keys, plus chat, feeling, story, opinion and knowledge
turns). The bank must never be used to fit, tune or select anything.

Each conversation is scored as a whole, with the arms side by side:
- recall when asked: direct and reworded questions;
- false recall: questions about facts never told;
- intrusions: a told key in a reply to a turn that should not recall;
- needless recalls: the hippocampus returning an episode on such a turn.

The transcript is kept in full and read, not only counted. There is deliberately no single score to optimise.

How it is used:
- It runs every week next to the fixed weekly, on new seeds each time (week number x 100 + 0..5), so no method can
  be tuned to one fixed conversation.
- A change to the brain counts as progress only if it does not worsen intrusions or false recall here while it
  improves recall, and if a reader of the transcripts agrees the conversations got better.

First run: the current defaults, 1.5B, seeds 1100-1105 (pool CPU), as the baseline. Expected: needless recalls and
intrusions are common in the intact arm and rare in the hippocampus-off arm. The first fix to test against it is when
to recall at all; the question bar (0.12 under S0) lets almost any question pull an episode back.
