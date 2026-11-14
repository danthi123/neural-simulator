# S_all + speaker contrast at mu 1.5 (live 1.5B unslept, six seeds): NOT A LEAD (D3 fails, the aunt lure 5/6)

Pre-registration `brain/research/2026-11-14-sall-contrast-dose-1.5b.md` (D0-D3, commit 0ecbbca2, pushed before the
runs). Data `results/2026-11-14-sall-contrast-dose-1.5b/` (pool40/41/42 CPU fp32, rc 0 all six), re-scored with
`brain.experiments.rescore`; `table_rescored.txt` (its Q lines are the mu 3 prereg's bars, not this one's: Q0
"fails" only because it expects mu 3). Arm r, same seeds:

| contrast | told (12) | para (12) | false (12) | aunt lure (seeds) |
|---|---|---|---|---|
| none (`key-default A`) | 5.83 | 2.17 | 0.83 | 4/6 |
| mu 1.5 | 6.50 | 3.00 | 1.00 | 5/6 |
| mu 3 (`sall-contrast`) | 6.83 | 3.33 | 1.17 | 6/6 |

D0 holds (mu 1.5 recorded, six files). D1 holds (para +0.83 >= +0.67). D2 holds (told 6.50 >= 5.83). **D3 fails**
(lure on 5 of 6 seeds > 4). **VERDICT (registered): NOT A LEAD.** mu 3 stays the candidate for the mixed-day test.

## What it shows

- The effects scale smoothly with mu: told, para and the lure all sit between none and mu 3. No strength buys the
  gain without the lure; lowering mu trades them one for one.
- The lure is not a contrast artefact so much as a routing one. It fires on 4/6 seeds with no contrast at all (S_all
  routes "my aunt" to the grandmother episode, both are relatives with a city). Contrast makes the reinstated episode
  more likely to be spoken. Removing it is the confidence gate's job (does the cortex know whose fact it is?), not a
  dose setting.
