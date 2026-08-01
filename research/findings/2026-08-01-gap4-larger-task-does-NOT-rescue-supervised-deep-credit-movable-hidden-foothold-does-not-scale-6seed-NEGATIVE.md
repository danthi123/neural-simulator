---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/supervised_plateau/tasksize/sizeB_6seed_aggregate.json
---

# gap#4: a LARGER/richer task does NOT rescue supervised deep credit — the movable-hidden foothold does NOT scale, the local-vs-oracle gap WIDENS — 6-seed NEGATIVE (0/6)

<!--derived-->
**One-line verdict:** the direct follow-up the np3 supervised-null finding named — *"does a larger / richer task
let the directed error's train-fit convert to a held-out advantage the small task masks?"* — resolves to a clean
**NO on all 6 seeds**. On task sizeB (n_super=48, n_members=8, n_prop=4, n_obs=14, hidden=64 — larger than the np3
null), the oracle still learns (held-out **0.869**, chance ≈ 0.08333), but BOTH local movable-hidden rules floor at
~0.27 (unsup **0.278**, supervised **0.270**), barely above the frozen reservoir 0.218 and far below oracle. The
directed error does NOT beat unsupervised sharpening (per-seed sup−unsup held-out = +0.019/−0.037/−0.019/+0.037/
+0.009/−0.056, net **−0.008**, up on 3/6); `supervised_beats_both_by_margin` = **0/6**. This settles the np3
residual: supervised deep credit on the movable hidden is **null, not task-limited** — a bigger task makes the
foothold fall *further* behind the oracle, not catch up. Run concurrently with the other roadmap lanes (GPU/cupy).

Artifact: `research/findings/raw/gap4/supervised_plateau/tasksize/sizeB_6seed_aggregate.json` (backend cupy/GPU;
per-seed raw files live beside it in the `sizeB_6seed` directory, each with a provenance sidecar).

## Result — 6 seeds {42,43,44,100,101,102}, task sizeB, chance ≈ 0.08333

<!--derived-->
| read-out (mean held-out) | sizeB (larger) | np3 (original null) | note |
|---|---|---|---|
| oracle (full backprop) | 0.869 | 0.975 | task IS learnable at both scales |
| frozen reservoir | 0.218 | — | the no-credit floor |
| **unsup plateau (local)** | 0.278 | — | barely above frozen, far below oracle |
| **supervised plateau (local)** | 0.270 | — | ≈ unsup, does not beat it |
| supervised plateau TRAIN | 0.700 | 0.80–0.84 | still fits train — credit reaches the movable hidden |
| deep_credit_share unsup | 0.083 | 0.139 | DROPS on the larger task |
| deep_credit_share supervised | 0.080 | 0.108 | DROPS; ≈ unsup |
| `supervised_beats_both_by_margin` | 0/6 | (beats 1/6) | directed error adds nothing |

## What this settles

<!--derived-->
**1. The movable-substrate reframe still HOLDS (the wall is genuinely broken, not re-erected).** The supervised
arm still fits TRAIN (0.700) on the movable plateau hidden — a directed, transport-free credit signal reaches and
moves a deep spiking hidden, exactly as the located-wall finding said was impossible for the tonic-pinned hidden
(stuck at 0.34). That conceptual advance is scale-robust.

**2. But the foothold does NOT scale, and the directed error is genuinely null.** The np3 finding hedged that
held-out might be *"capped by the small/coarse task (k=8), not the credit"* and named a larger task as the test to
tell "null" from "task-limited." The larger task answers it: **null.** The directed error does not beat
unsupervised sharpening at either scale (1/6 at np3, 0/6 at sizeB), and — the decisive part — on the richer task
the local-vs-oracle deep-credit gap **widens** (oracle 0.869 vs local ~0.27; deep_credit_share *drops* 0.139→0.083)
rather than the supervised arm catching up. The train-fit that the directed error buys is overfitting at both
scales; more task does not convert it to generalization.

**3. The residual is now precise and points away from two dead ends.** It is NOT the task size (tested — a bigger
task hurts, not helps) and NOT the supervised-vs-unsupervised distinction (settled — neither local rule closes the
oracle gap). The open gap#4 crux is a *credit mechanism* that lets the movable hidden inherit held-out structure
the way the oracle does — the local plateau-margin covariance rule (unsup) and the fixed-DFA directed error (sup)
both saturate well short of it. No capability abandoned: the movable-hidden foothold is real and the next lever is
a different on-bridge credit mechanism, not a bigger task.

## Anti-cheats (honest caveat — it STRENGTHENS the negative)
<!--derived-->
`anti_cheats_clean` = 4/6: seeds 100 and 43 show partial leakage (shuffle-DFA held-out up to 0.296, supervised-on-
permuted up to 0.315 on the worst seed). This *inflates* the supervised arm on exactly those seeds — and it STILL
does not beat unsupervised (net −0.008 across all 6). A cleaner anti-cheat pass would only widen the negative. The
oracle/frozen/reservoir controls are intact; reproducibility holds per seed.

## Next
Back to the gap#4 crux with the search narrowed: neither a directed fixed-DFA error nor an unsupervised covariance
rule on the movable plateau hidden closes the gap to oracle deep credit, and scaling the task does not rescue
either. The next on-bridge mechanism to test is a genuinely different credit route (e.g. the Deep Feedback Control
arm the np3 finding flagged as still-untested, or a burst-multiplexed dendritic signal), NOT a task-size or
supervised/unsupervised variation — both are now closed on this substrate.
