# Pre-registration: the confidence gate on a mixed day (slept facts + facts told after the restart, 1.5B)

Written 2026-11-13 before any run. Capability: memory selection (metacognition: feeling of knowing).

**Record brief.** 11-13 (`results/2026-11-13-confidence-gate-1.5b.md`, verdict NARROWED): the slept 1.5B cortex's
`mean_logp` separates its right from wrong answers (AUC 0.765), and gating reinstatement on it removes the cost of
reinstating into a slept cortex (h 9.17 vs f 7.67), but h is indistinguishable from simply switching the hippocampus
off (c 9.33; permutation p ~0.29; a reply-length gate did as well). Reason: every fact had been slept on, so the
hippocampus had almost nothing left to add (oracle 9.67). The skeptics' named next test: a day where only the
hippocampus holds some facts. There a gate must beat BOTH always-off (c) and always-on (f) to earn its place.
Biology: complementary learning systems, recent memories hippocampus-dependent, consolidated ones not.

**Prerequisites.** Met: 1.5B cortex, sleep consolidation, the S_all routing fix (11-05 g arm: unslept episodes
route 9/0 right/wrong). Not tested: the gate inside Brain (offline simulation, exact when M0 holds).
Host decision: none in the brain; the bar is a fixed number carried from the 11-13 run (evaluation choice).

**Method.** `brain/experiments/mixed_day_probe.py`, 1.5B, `--steps 480 --lr 5e-4 --kl-weight 3`, seeds
42/43/44/100/101/102, AWS g5 (A10G). Day 1: six FACTS told; restart = one night; day 2 (same process): the other six
told, saved, no second night (asserted: sleeps == 1 on every question Brain). Slept half = even FACTS indices on
seeds 42/44/101, odd on 43/100/102 (counterbalanced). Per question (12 told, 12 paraphrase, 12 never-told): c
hippocampus off, f S_all key with the 1.5B size bar, c's read-outs as 11-13. Rows carry `set` slept | new.

**Gate (primary, nothing fitted on this run).** h = c's reply if `mean_logp` >= T, else f's reply, with
T = -0.16947272575063316: the bar fitted on all six seeds of 11-13 (`confidence_gate_table.fit`, told + paraphrase).
Table: `brain/experiments/mixed_day_table.py`. Report only: the leave-one-seed-out bar on this run, a reply-length
gate (n_tok >= 6, fitted the same way on 11-13), the oracle.

**Criteria (exact means of 6 seeds, `fractions.Fraction`; told = of 12).**
- M0 instrument: f's reply equals c's on >= 95% of rows where nothing fired (else VOID).
- M1 instrument: on the six NEW told facts, f - c >= 1.5 (the hippocampus must carry them; else VOID: no test).
- M2 gate vs hippocampus off: h - c >= 1, and h >= c on >= 5/6 seeds.
- M3 gate vs always reinstate: h - f >= 0.5, and h >= f on >= 5/6 seeds.
- M4 false recall (never-told, of 12): h <= f + 0.5.
- M5 the signal, not chance: within-seed permutation of `mean_logp` over told + paraphrase rows (fixed bar, 2000
  permutations, RNG 1113), p(told h_perm >= h) < 0.05.
LEAD = M0-M5 hold -> next: the gate inside Brain (switch default OFF), then a live six-seed run. NOT A LEAD otherwise;
the verdict is on gating by `mean_logp` at a fixed bar, not on the capability.

**Expected.** New facts: c near 0 of 6 (the cortex never saw them), f about 2.5 of 6 (11-05 unslept g 4.83 of 12).
Slept facts: c about 4.5-5 of 6, f about 0.8 lower. If c's wrong answers to new facts are confident confabulations,
`mean_logp` stays above T and h collapses to c (M2 fails): that is the 0.1B confabulation result
(`2026-11-03-confabulation-next-methods.md` (b)) showing up at 1.5B, and the next method is cue familiarity.
M3's bar is 0.5, not 1, because only six facts are slept, so the reinstatement cost that h removes is about half
of 11-13's.
