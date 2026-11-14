# Pre-registration: speaker contrast in the live 1.5B brain, unslept day (six seeds, pool)

Written 2026-11-14 before the run. Oracle LEAD (`results/2026-11-14-speaker-contrast-2-1.5b.md`): mu 3 at gain 0.5,
fresh 36 facts 84 vs 64 of 108, echoes 3 vs 16. Live differs: sampled replies, S0 routing (right episode on 36/72
told), never-told and self questions, which the oracle never asked.

**Method.** `unslept_day_probe.py --arms c,s --speaker-contrast 3`, seeds 42/43/44/100/101/102, pool CPU, vs the
recorded tell runs (`results/2026-11-14-lived-episode-live-1.5b/tell_s*.json`, same code path, reproduced part A).
Switch `Brain(speaker_contrast=3.0)` (default 0 = OFF; S0 key; key unchanged, so routing should match tell).
Table `python -m brain.experiments.lived_live_table <dir> contrast` (files `contrast_s*.json`).

**Criteria (arm s, six-seed means).** P0: contrast recorded on every lived run, stored text unchanged. P1 told (of
12): contrast - tell >= 1, contrast >= tell on >= 5/6 seeds. P2 false (never-told, of 12): contrast <= tell + 0.5.
P3 self (of 3): contrast >= tell - 0.5. GO -> mixed-day AWS run with sleep (a night replays the stored text, not the
contrasted change, so sleep is unaffected but the reinstated memory on slept facts is) before any default flip.

**Expected.** Told up by about the oracle gain (+0.19 of cues) times routing (~0.5 right) x 12 = about +1.1: P1 is
uncertain. Report: echoes, right-episode rate (must match tell), answered-given-right-episode.
