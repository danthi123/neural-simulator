# S_all + speaker contrast on a mixed day with sleep (1.5B, six seeds): NO-GO (X1: told +0.00). No default flip

Pre-registration `brain/research/2026-11-14-sall-contrast-mixed-1.5b.md` (X0-X4, pushed before the run). Data
`results/2026-11-14-sall-contrast-mixed-1.5b/`: AWS g6.xlarge L4, batch 20261114-1229 (`run.log`, 12 jobs rc 0;
the JSONs carry no host field). Contrast off (tellmix) and mu 3 (contmix) paired per seed. `table.txt` is the
re-scored table (`brain.experiments.rescore`); the table as shipped is identical.

| arm | contrast | told (12) | slept (6) | new (6) | para (12) | false (12) | self (3) |
|---|---|---|---|---|---|---|---|
| s S0 | off | 7.17 | 4.67 | 2.50 | 4.67 | 6.83 | 3.00 |
| s S0 | mu 3 | 6.00 | 3.67 | 2.33 | 3.33 | 7.33 | 3.00 |
| r S_all | off | 8.00 | 4.50 | 3.50 | 5.50 | 6.67 | 3.00 |
| r S_all | mu 3 | 8.00 | 4.67 | 3.33 | 5.33 | 7.17 | 3.00 |

X0 holds, **X1 fails** (r told +0.00, needs +1 on 5/6 seeds), X2-X4 hold. **VERDICT (registered): NO-GO.** The
1.5B package stays S0 at gain 0.5; S_all, the size bar and contrast stay OFF.

Instrument check: the contrast-off arms reproduce the key-default mixed day (batch 20261114-0453) exactly (s 7.17,
r 8.00 told), so the batch ran the same brain.

## What it shows

- After a sleep, contrast does nothing for S_all and costs S0 about one told fact and one paraphrase. The unslept
  gain (+1.0 on the registered seeds, +0.5 on 200-205) does not survive into a day with slept facts: the
  reinstatement is no longer the only thing speaking, and on the six facts told after the restart (new) contrast is
  -0.17.
- False recall on the slept cortex (about 7 of 12 never-told questions answered with a told fact) is now the larger
  defect by far. The aunt -> Oslo lure fires on 12/12 runs with or without contrast.
- Method verdict: speaker contrast as a fix for the first-person echo is banked NO-GO in the live brain, joining
  lived episodes and lived_key. The echo is not a storage-format problem a fixed vector edit can solve at 1.5B. The
  capability (answering from the listener's side) stays open; the next lever is the false-recall gate, which the
  slept-day numbers put first.
