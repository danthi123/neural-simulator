# Local sleep rule, 0.1B six-seed screen (11-27): NEGATIVE by the pre-written rule

Rule: `brain/research/2026-11-27-local-sleep-rule-prereg-DRAFT.md`, written and reviewed (code review) before any data,
but its pre-data prereg review is not recorded, so it is a DRAFT rule, not a registered one. Pick: `dev_pick.md`
(seed 7, transport feedback, lr 0.01, decay 1e-3; one disclosed tie-break). Runs: pool41 (seeds 42-44) and pool42
(100-102), commit 6a430b8, 0.1B CPU fp32 (host from `six_01b/done_pool*.txt`; the JSONs record no host).

| arm | recall /12 per seed | mean | false recall | drift KL | adapter norm |
|---|---|---|---|---|---|
| no_sleep | 0 0 0 0 0 0 | 0.00 | 0 | 0.000 | 0 |
| gradient | 4 4 4 3 4 4 | 3.83 | 1-4 | 0.028 | 15-17 |
| local | 0 0 1 0 0 0 | 0.17 | 0 | 0.037 | 0.32-0.36 |
| local_shuffled | 0 0 1 0 0 0 | 0.17 | 0 | 0.036 | 0.32-0.36 |
| local_random (lr 0.3) | 0 x6 | 0.00 | 0 | 15.4 | about 3e5 |

Bar: local >= max(0.5 x 3.83, 0 + 2) = 2. Local 0.17. Not UNDEFINED (gradient >= 2, norms finite and nonzero,
shuffled <= 1). NEGATIVE.

Reading: under the drift cap the local rule is nearly inert (final replay loss 2.27 vs gradient 0.17 on s42; local
equals shuffled on every seed). At higher lr (dev sweep) it learned (recall 2/12) but drifted 3.5x the gradient arm. So
this rule trades learning for drift worse than gradient descent; the keep term (local KL error + B decay) is too weak.
local_random: the per-factor update clip works; the norm is sum ||A@B||, quadratic in factors that both drift to the
clip ceiling because random feedback never reduces the error (a method failure at lr 0.3, not a clip bug). Its JSON
`feedback` field logs the CLI flag ("mixed"), not the forced "random": cosmetic.

Skeptic (fresh sonnet, refuter-lens block): SURVIVES, with the DRAFT-status caveat above. Lenses: (a) engaged (norms
> 0, replay loss falls); (b) n/a (no default flip); (c) where_ran unknown=37 (known gap); pool CPU, validity not
host-dependent. Verdict on this rule and pick only, not on local learning (PREREQUISITES: Sleep consolidation).
Next method: a stronger local keep term (the drift is the binding constraint), judged with the 0.4B dev sweep now on
pool40.
