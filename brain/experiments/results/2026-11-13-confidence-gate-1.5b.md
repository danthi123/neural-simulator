# Confidence gate for hippocampal reinstatement at 1.5B, six seeds: registered criteria hold, NARROWED to "no-regret, not better than hippocampus-off"

> **Erratum 2026-11-13 (scorer).** f's credits included general statements about the key word ("Oslo is the capital of Norway."). Re-scored with `strict_scorer.is_generic`: f 7.67 -> 5.67; c and h unchanged; G0-G4 still hold. See `2026-11-13-mixed-day-1.5b.md`, `table_rescored.txt`.

Capability: Metacognition (feeling of knowing) for memory selection after sleep. Pre-registration:
`brain/research/2026-11-13-confidence-gate-1.5b.md` (G0-G4, committed f5c1c1769 before the run). Data:
`results/2026-11-13-confidence-gate-1.5b/cgate_s*.json`, `table.txt`
(`python -m brain.experiments.confidence_gate_table <dir>`). Ran on AWS g5 (A10G), instance i-008140262be021160,
batch 20261113-1126, 16:32-17:49 UTC, about 12 min per seed, `cuda_peak_bytes` 11.37 GB. The JSONs record no host
(`tools/where_ran.py`: unknown); the probe reads no `research/findings/raw` input, so validity does not depend on it.

## Result (of 12 per seed, mean over 42/43/44/100/101/102)

| arm | told | paraphrase | false recall (never-told) |
|---|---|---|---|
| c slept cortex, hippocampus off | 9.33 | 6.00 | 9.17 |
| f always reinstate (S_all, 1.5B bar) | 7.67 | 6.33 | 9.33 |
| h gate: c's reply if mean_logp >= t (LOSO), else f's | 9.17 | 6.67 | 9.33 |

c and f reproduce the 11-05 run per seed exactly (told c 10 8 7 11 9 11, f 7 8 7 9 7 8; fired 38 rows c 33 / f 23).
- G0 instrument (f == c when nothing fired): 155/155, **holds**.
- G1 AUC(mean_logp) c-right vs c-wrong, told + para: pooled 0.765, per-seed mean 0.757, **holds** (seed 44 alone 0.55).
- G2 h >= c - 0.5: **holds** (9.17 vs 9.33). G3 h - f >= 1, h >= f on >= 5/6: **holds** (+1.50, 6/6).
- G4 false h <= f + 0.5: **holds** (9.33 vs 9.33).

## Verification (three independent skeptics: power, instrument, scope) - all three NARROWED

- **h is not distinguishable from c.** Told h - c = -1 answer over 72 rows (1 up, 2 down); told + para 95 vs 92 of
  144 (6 up, 3 down). G3 passes whenever c beats f, so it never tested the gate. A within-seed permutation of
  mean_logp (300, LOSO refit) gives told 8.91 +/- 0.26: the real 9.17 sits at p about 0.29. A gate on reply length
  gives the same told 9.17. Only paraphrase leaves the permutation null (6.67 vs 6.0, p about 0.02), +4 rows net.
- **Little headroom.** An oracle (best of c and f per row) reaches told 9.67: after sleep at 1.5B, reinstatement
  fixes only 2 of 16 rows the slept cortex gets wrong. A gate cannot beat c by much on this bank.
- **The signal is real but untested where it matters.** mean_logp does predict the slept cortex's errors (0.77),
  as at 0.1B (0.74). The unslept case is UNDEFINED: no logp on an unslept cortex was recorded, so it is not shown that
  the bar (about -0.17) would leave the pre-sleep benefit intact (11-05: g 4.83 vs b 2.83).
- **False recall is near ceiling** (about 9 of 12 never-told questions get a learned name in every arm), so G4 has
  no power; never-told questions rarely reach the gate.
- Offline gate, not a live one: on the 15 rows where h takes f's reinstated reply, a live Brain would sample a fresh
  draw (variance, not bias). Leave-one-seed-out shares the same 12 questions across seeds. mean_logp is teacher-forced
  on the re-encoded reply, a proxy of the sampling-time log-prob.

**Verdict.** As registered, LEAD; as verified, NARROWED: the cortex has a usable "my answer is probably right"
signal, and gating on it removes the reinstatement cost, but on this bank it does no better than leaving the
hippocampus off after sleep. No Brain default changes; no gate is built from this alone. The pre-registered bars were
too loose (they compared h only to f); the next registration compares against c and against an unslept arm.

## Next (method)
The gate earns its place only where BOTH stores are needed: a mixed day, where some facts were slept on and others
were told after the last sleep (only the hippocampus has those). Register: 6 slept facts + 6 told after the restart,
arms c / f / h, with the bar fixed from this run (no refit), criteria h > c on the new facts, h >= c - 0.5 on the
slept facts, h > f overall. That one test measures the unslept-side behaviour too.

## Also in this batch: weekly conversation test (0.1B default cortex)
`brain/eval/transcripts/2026-11-13.md` (AWS, not W; RULING w-dedicated-1113). After the restart the hippocampus-off
brain answered 4 of 6 memory questions right, intact 2 of 6 ("Marguerite is Marguerite", "What do I do for work?"
-> "I drive a red Volvo"): the same reinstatement cost after sleep, seen in conversation. Day-7 (11-09) stays missed
by date.
