# Scorer fix: a disclaimer followed by general talk naming the key is not an answer. No banked verdict changes

`strict_scorer.is_disclaimed` (used by `correct_answer`, and by `hippo_gain_refit.klass` as "generic"): the reply
carries a no-knowledge or AI disclaimer ("as an AI language model", "as a digital assistant", "I don't know", "I
don't have personal preferences / access / information") and no sentence naming the key addresses the user, after
service phrases such as "I can tell you that" are set aside. Example credited before, not now: "I'm sorry, but as an
AI language model, I don't have personal preferences or emotions. However, basil is a popular herb...". Still
credited: "I'm not sure, but I think it's probably teal.", "I don't know for sure, but your favourite colour is teal."

A sweep of every credited reply in `brain/experiments/results/` flagged 249 replies. Every distinct one was read by hand
and none answers the question. The one borderline case: "I can tell you that Clover is a soft grey pet." It holds
the told fact but does not address the user.

Re-scored copies (`python -m brain.experiments.rescore <src> <dst>`, originals untouched) of every 11-13/11-14 dir,
each run through its own table. Credits lost to the new rule alone: confidence gate, gain refit 0.1B, live gain,
mixed day, lived live, lived key 0; key default 2 (both in the report-only A2 seeds); S_all contrast 2; speaker
contrast live 1; gain refit 1.5B 6; lived episode 6; speaker contrast 88; speaker contrast 2 144.

| result | registered | re-scored |
|---|---|---|
| gain refit 1.5B | LEAD (dev 58, test 7/7) | LEAD (dev 57, test 7/7) |
| lived episode (oracle) | NOT A LEAD (+7) | NOT A LEAD (dev lived 64 vs tell 57, +7 < 10) |
| speaker contrast 1 (dev c50/c100) | NOT A LEAD | NOT A LEAD (c100 dev 71 -> 63 vs tell 57) |
| speaker contrast 2 (fresh, contrast 3, gain 0.5) | LEAD (84 vs 64) | LEAD (77 vs 62; S3 generic 7 vs bar 7, at the bar) |
| speaker contrast live (S0) | NO-GO | NO-GO (told +0.00, para 2.33 -> 2.17) |
| S_all + contrast live | GO (told 6.83, para 3.50) | GO (told 6.83, para 3.33; Q0-Q3 hold) |
| key default | KEEP S0 | unchanged (no flip in A or B) |

E0 of the lived-episode table pins the 11-13 numbers as scored the old way (58, 7, 7), so on re-scored data it
fails and the table prints VOID by construction. That VOID is not a verdict change: E1 to E3 are as above.

## What it shows

- The contrast partly turns echoes into AI disclaimers, not into answers. In the oracle runs the contrast arms gain
  far more disclaimers than tell (dev gain 0.5: tell 3 generic, contrast 1 11; fresh: tell 2, contrast 3 7).
  Contrast's oracle advantage is therefore smaller than first reported (fresh +20 -> +15; dev c100 +13 -> +6). The
  stored delta d + mu (d - d_self) pushes the cortex toward its assistant persona as well as away from the user's
  voice.
- The live verdicts barely move: in the live unslept probes the disclaimer replies were already rare (2 of 131
  credits).
- The mixed-day S_all + contrast batch now running on AWS ships with the old scorer. Re-score its JSONs with
  `brain.experiments.rescore` before running `mixed_contrast_table`, and report both numbers.

Earlier gap, not changed here: the 11-13 confidence-gate and mixed-day JSONs keep their pre-`is_generic` `ok` fields
(33 credits flip with this rule off), so any new table over those files must re-score them first.
