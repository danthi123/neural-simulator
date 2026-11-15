# Result: weaker-night control for the near-miss guard (Amendment 3), 1.5B, seeds 200-205: SELECTIVE (narrowed)

Prereg: `brain/research/2026-11-14-near-miss-guard-1.5b.md`, Amendment 3 (rule written before the run). Runs: pool40/41/42
CPU, `jobs/2026-11-15-pool-weak-night.sh`, 12 jobs, all rc 0. Compared with the same lane's guard-off / guard-on rows
(`results/2026-11-14-near-miss-guard-1.5b/extra/`), all re-scored. Report-only seeds: this picks the next method; it
is not a GO bar.

## Arm s (S0 key, gain 0.5), six-seed means, of 12

| night | slept told | told (all) | para | false (told fact on a never-told question) | abstentions, never / told rows (of 72) |
|---|---|---|---|---|---|
| normal (off) | 3.83 | 6.50 | 3.33 | 8.83 | 1 / 0 |
| guard (near-miss 4, frame F) | 3.67 | 6.33 | 3.67 | 5.50 | 12 / 9 |
| kl6 (ordinary-text KL 6, no guard) | 4.33 | 7.00 | 4.67 | 10.17 | 0 / 0 |
| st240 (240 steps, no guard) | 3.50 | 6.50 | 4.50 | 9.00 | 0 / 0 |

- The rule: st240 has slept told 3.50 <= the guard's 3.67, and its false recall is 9.00 >= 5.50 + 1.5. **SELECTIVE.**
  - kl6 (slept told above the guard's) has false 10.17, so the NOT SELECTIVE branch does not fire.
  - The doses engaged: adapter norm off about 50-58, kl6 46-53, st240 43-52. Replies differ from off on 39 (kl6)
    and 34 (st240) of 72 never-told rows.
- Per seed, the guard's false recall is below st240's on 6/6: 8 vs 12, 2/8, 6/12, 3/6, 11/12, 3/4.
- A weaker night does not lower false recall at all. kl6 raises it (53 -> 61 rows), and st240 leaves it flat. False
  recall after sleep is not a night-strength trade-off.

## Caveats (fresh sonnet skeptic: NARROWED)

- **The told match is thin.** Slept told summed over six seeds is 22 (guard) vs 21 (st240), so the gate passes by one
  item. st240 matches the guard on told recall and beats it on paraphrase (4.50 vs 3.67).
- **About 60% of the guard's gain is abstention.** Of the guard's 20 fewer intrusions against off, 12 became "I'm sorry,
  I don't have ..." replies. It also abstains on 9 of 72 told rows, where no other night does. Some of the rest are
  invented answers (clean non-told replies: off 18, guard 29).
- `tools/where_ran.py`: unknown=36. All files are `device cpu`, from the pool job script.

## What it means

- The guard does something a weaker night cannot: it removes told-fact intrusions on near-miss questions. The method
  family is worth a second form. That fits the 11-03 plan (CLS interleaving).
- Its costs are also its own:
  - It refuses some told facts. Told-topic cues ("Who is my best friend?") hold the pretrained "I don't have that
    information" reply on topics the night is teaching.
  - It trades intrusions for abstentions and inventions.
- **Next method (M4b):**
  - Keep the frame guard, and drop cues that name a TOLD topic. This is a brain-side rule, its own scaffold row, the
    mirror of the probe's test-hygiene filter.
  - Score confabulation as well as intrusion: an invented answer on a never-told question counts as false.
  - Pre-register on the six registered seeds (AWS) before any run.
