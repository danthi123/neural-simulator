---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: E · Language
mechanism: the learned open-vocabulary referent lexicon routed into the D6 multi-referent WM organ through the PRODUCTION
  env-flag path (BRAIN_LEARNED_REFERENT_LEXICON / BRAIN_LEARNED_REFERENT_LESION; default OFF)
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-d6-multiref-wm-learned-referent-env-flag-route-PREREGISTERED.md
artifacts:
  - research/findings/raw/_d6_learned_referent_env_flag/s42.json
  - research/findings/raw/_d6_learned_referent_env_flag/s43.json
  - research/findings/raw/_d6_learned_referent_env_flag/s44.json
  - research/findings/raw/_d6_learned_referent_env_flag/s100.json
  - research/findings/raw/_d6_learned_referent_env_flag/s101.json
  - research/findings/raw/_d6_learned_referent_env_flag/s102.json
verdict: GO 6/6 on one recorded input (AMENDMENT 1). R1 route ON in scope 6/6, R2 route OFF 6/6, R3 recovery min per seed at
  least 0.50 (mean 0.76), R4 lesion 0.0 on every seed with the lever moved. Default stays OFF until the flip rule is met.
---

# The learned referent lexicon works through the production route: GO 6/6

## Result

Scored by the pre-registered `score()` over `research/findings/raw/_d6_learned_referent_env_flag/s42.json` and the five sibling
per-seed files (`python -m research.runners._d6_learned_referent_env_flag_derisk --score research/findings/raw/_d6_learned_referent_env_flag`). All six seeds carry the same input
hashes (frame-environment corpus and lexicon corpus both the full 19,971,040-byte tinystories.txt), so `one_input` is true.

<!--derived-->
| seed | R1 (route on) | R2 (route off) | R3 recovered both (intact) | R4 recovered both (lesion) |
|---|---|---|---|---|
| 42 | pass | pass | 0.833 | 0.000 |
| 43 | pass | pass | 0.833 | 0.000 |
| 44 | pass | pass | 0.750 | 0.000 |
| 100 | pass | pass | 0.667 | 0.000 |
| 101 | pass | pass | 0.833 | 0.000 |
| 102 | pass | pass | 0.667 | 0.000 |

The production flag routes a held-out noun ("owl", in no hand list) into the WM organ, the lesion flag removes the learned
lexicon's contribution completely, and with the flag unset the hand route is unchanged.

## Notes

- Seed 42 was re-run under AMENDMENT 1: the first seed-42 artifact (commit f11bcb97f) read a 7.99 MB prefix of the corpus and
  carried no input hash; it is superseded by this run and cannot be pooled by `score()`.
- The six organ seeds share one trained lexicon (get_lexicon() is built at seed 42); this is six organ populations over one
  lexicon, not six independently trained lexicons (declared in the prereg's residuals).
- Next rung: default-ON only through the flip rule (review, no-regression combined battery, production-default run).

## Honesty

Functional read-outs only.
