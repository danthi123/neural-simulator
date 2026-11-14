# Pre-registration: which hippocampal key is the 1.5B default, now that the gain is 0.5 (S0 vs S_all + size bar)

Written 2026-11-14 before any run. Capability: episodic recall (memory selection: fetching the right episode).

**Record brief.** Gain 0.5 is the 1.5B default since 11-14 (`results/2026-11-13-live-gain-1.5b.md`, GO with S_all +
size bar). The default key is still S0. At 1.5B, S0 fetches the right episode for 7/12 told questions and S_all for
11/12 (11-03 routing, `2026-11-03-hippo-routing-1.5b.md`; S_all at the size bar: 9 right / 0 wrong / 3 none). But on
self questions S_all fired on 1/8 and S0 on 7/8 (identity check, 11-14), so S_all may lose the brain's own name. Every
earlier live comparison of the two keys ran at gain 4 and is void for this choice.

**Prerequisites.** Met: 1.5B cortex, sleep, both keys, the per-size gain. Host decision: none (a choice between two
existing keys). Nothing new is fitted.

**Method.** Arms (`mixed_day_probe.ARM_SPECS`): c hippocampus off; s S0 key, default bar, gain 0.5; r S_all key, size
bar, gain 0.5. Questions: 12 told, 12 paraphrase, 12 never-told, 3 self ("What's your name?", "Who are you?" -> Nova;
"Do you remember my name?" -> Dan). Fixed scorer (`correct_answer`); never = any FACTS key; self = the word.
- Part A, unslept day (pool40/41/42, CPU fp32): `brain/experiments/unslept_day_probe.py`, the 12 facts told, nothing
  slept, seeds 42/43/44/100/101/102.
- Part B, mixed day (AWS GPU): `mixed_day_probe.py --arms c,s,r --self-q`, as the 11-13 runs (1.5B, `--steps 480 --lr
  5e-4 --kl-weight 3`, slept half counterbalanced), six seeds.
Table: `brain/experiments/key_default_table.py` (tested).

**Criteria (exact six-seed means, `fractions.Fraction`).**
- K0 instrument: s and r record gain 0.5 and keys S0 / S_all on every row; Part A sleeps == 0, Part B sleeps == 1
  (asserted). Else VOID.
- K1 (A) told, of 12: r - s >= 1, and r >= s on >= 5/6 seeds.
- K2 (B) told, of 12: r - s >= 1, and r >= s on >= 5/6 seeds.
- K3 (B) slept facts, of 6: r >= s - 0.5.
- K4 (A and B) false, of 12: r <= s + 0.5 in each part.
- K5 (A and B) self, of 3: r >= s - 0.5 in each part.
- Flip S_all + size bar to the 1.5B default = K0-K5. If only K5 fails: NOT YET; next method = store the self episodes
  so S_all can fetch them (the self questions are not keyed the way user tellings are), then re-test K5.

**Expected.** K1/K2 hold (S_all routes 11 vs 7 right). K5 is the risk: S_all rarely fires on self questions. Report
only: paraphrase, answer rate (abstention replies), and replies containing the fact word (first-person echoes count
here, not in the scorer).
