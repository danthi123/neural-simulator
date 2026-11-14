# S_all + speaker contrast, live 1.5B unslept day (six seeds): GO as registered, NARROWED by three skeptics

Pre-registration `brain/research/2026-11-14-sall-contrast-1.5b.md` (Q0-Q3, commit d26e862c). Data
`results/2026-11-14-sall-contrast-1.5b/contrast_s*.json` (pool40/41/42; JSON `host` per file, the same node per seed
as part A); baseline `results/2026-11-14-key-default-1.5b/A/unslept_s*.json`; `table.txt`.

| arm r (S_all + size bar, gain 0.5) | told (12) | para (12) | false (12) | self (3) |
|---|---|---|---|---|
| tell (part A) | 5.83 | 2.17 | 0.83 | 1.67 |
| speaker contrast 3 | **6.83** | 3.50 | 1.17 | 1.67 |

Q0 holds (routing identical on every paired row), Q1 holds (+1.00, contrast >= tell on 6/6), Q2, Q3 hold.
**VERDICT (registered): GO.** Arm s (S0) reproduces the S0 contrast run exactly (told 4.83 = 4.83, para 1.50 -> 2.33).
Against the current default (S0, no contrast, 4.83) the package is +2.0 told.

## Skeptics (instrument, control, risk; sonnet, fresh context): all NARROWED

- Told: real but fragile. Per-seed +4, +1, 0, +1, 0, 0 (3 up, 3 tied); seed bootstrap 95% CI +0.17..+2.33; without
  seed 42, +0.40. By hand all 12 told gains are genuine answers ("Your cat is called Pixel."), mostly a My -> Your
  perspective flip of a recalled fact.
- Para: the robust part (5/6 seeds up) but overstated by the scorer: about 5 of the +8 credits are not answers ("Yes,
  there are many foods that contain peanuts ..."); by hand about 2.17 -> 2.67.
- False: +2 genuine wrong-subject intrusions (aunt -> Oslo from the grandmother fact). Abstention-like replies slightly
  up on some seeds (answer_rate.py has no layout for this probe; counted by hand).
- Risk: untested with sleep, after a night, on GPU; route_delta doubles a told episode's bytes (~13 MiB at 1.5B).

Not a default flip. Next (pre-registered, running): `brain/research/2026-11-14-sall-contrast-mixed-1.5b.md`, the mixed
day with sleep on AWS, contrast on and off paired in one batch (an independent six-seed replication of the told gain).
