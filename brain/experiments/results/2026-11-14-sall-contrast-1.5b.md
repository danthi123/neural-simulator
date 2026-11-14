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

## Report-only extra seeds 200-205 (run after the verdict; do not change it)

Same probe (`jobs/2026-11-14-pool-sall-contrast-extra.sh`, pool40/41/42 CPU, `extra/`), against the same seeds
without contrast (`2026-11-14-key-default-1.5b/A2/`). Both re-scored with `brain.experiments.rescore`
(`2026-11-14-disclaimer-rescore.md`). Arm r:

| seeds 200-205 | told (12) | para (12) | false (12) | self (3) |
|---|---|---|---|---|
| S_all, no contrast | 6.00 | 2.33 | 0.33 | 1.67 |
| S_all + contrast 3 | 6.50 | 3.50 | 1.00 | 1.67 |

On these seeds Q1 fails (+0.50; up on 2/6 seeds, 201 and 202) and Q2 fails (+0.67). Over all 12 seeds the told gain
is +0.75 (+6, +9 facts over 12 seeds, both sets carried by two or three seeds). It is not the robust +1 the
registered seeds showed. Para rises on both sets (+1.33, +1.17). False recall is one near-miss lure: "Where does my
aunt live?" answered "Oslo" from the grandmother fact. That happens on 6 of 12 seeds without contrast and 12 of 12
with it; the only other false answer (parrot "Peanut", seed 102) is in both arms. Contrast makes a relative-in-Oslo
fact fire for any relative: the confidence gate's job, and a cost the flip must weigh.
