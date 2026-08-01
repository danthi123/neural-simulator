---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/plastic_plateau/plastic_plateau_6seed_aggregate.json
---

# gap#4: a local UNSUPERVISED transport-free plasticity rule on the MOVABLE plateau hidden beats a frozen random reservoir at the sweet spot — 5/6, anti-cheats 6/6 (the first positive on-bridge local-learning signal; small + unsupervised, NOT directed deep credit)

**One-line verdict:** tonight's sweet-spot finding showed that at `n_prop=3` (forward representable, frozen
reservoir fails) NO credit rule could train the standard tonic-pinned spiking hidden — because the hidden is not
MOVABLE. The deep-research gate's reframe: the project already built a movable, reliable hidden (the
coincidence-plateau reset-read expander) but only ever used it FROZEN. This makes it PLASTIC with a local,
transport-free rule and tests it against the frozen version. Result: the plastic hidden beats the frozen random
one on **5/6 seeds** (pre-registered margin ≥ 0.05), `deep_credit_share` **positive 6/6** (mean 0.139), and
**every anti-cheat holds 6/6**. This is the first time local learning on the on-bridge hidden HELPS at the
operating point where everything else failed. **But the scope is precise and limited (below).**

Artifact: `research/findings/raw/gap4/plastic_plateau/plastic_plateau_6seed_aggregate.json` (backend cupy/GPU).

## Result — 6 seeds {42,43,44,100,101,102}, n_prop=3, epochs 30, N_COL=200

| arm (mean held-out) | value | note |
|---|---|---|
| oracle (fenced backprop) | 0.975 | forward representable (op-point genuine) |
| frozen random RATE reservoir | 0.1 | FAILS (chance 0.167) → the sweet spot is real |
| FROZEN-plateau reservoir | 0.445 | fixed random coincidence map + trained readout |
| **credit-trained plateau** | **0.518** | same columns, input weights PLASTIC |
| **deep_credit_share** | **+0.139** | (credit − frozen)/(oracle − frozen); **positive 6/6** |

Per-seed margin (credit − frozen): +0.056/+0.056/+0.093/+0.037/+0.111/+0.093 → **5/6 clear the 0.05 gate**
(only seed 42 falls short at +0.037). **Anti-cheats hold 6/6:** permuted-label ≈ chance (0.07–0.13, no readout
leakage); plateau/apical lesion ≈ floor (0.11–0.17); reproducibility 1.0 (plateau reliability survives
plasticity); no-transport True (code + runtime).

## The rule (why it is transport-free) + how it moves the hidden

`dW[c,f] = lr · mean_i (margin[i,c] − θ[c]) · pre[i,f]`, where `margin = max(0, cp_v_apical − FLOOR)` is the
reliable plateau margin READ from the spiking bridge, `pre` is local pre-activity, `θ[c]` is a per-column
homeostatic threshold, and each column's afferents are L2-renormalized to their initial norm. It reads **only
local pre-activity × the local plateau margin** — never the forward or readout weights or their transpose (so no
φ′ depth product, no weight transport; verified in code + at runtime + by ordering — plasticity completes before
any readout is fit). It moves the hidden by SHARPENING each column onto the input conjunctions that most reliably
drive its plateau (codon sparsity 0.74 → 0.29, diversity up) — which happens to align with class-relevant
structure, lifting downstream linear separability.

## SCOPE — what this is, and what it is NOT (read before citing)

- **UNSUPERVISED, not directed deep credit.** The hidden update uses NO label and NO output error — it is a local
  plateau-gated Hebbian covariance (representation self-organization), not the supervised error-assignment gap#4
  ultimately wants. This is a real and complementary result (a substrate that self-organizes a useful hidden
  transport-free), but "deep credit" in the directed-error sense is a DISTINCT, stronger claim not made here.
- **SMALL.** credit 0.518 vs frozen 0.445 vs oracle 0.975: the plasticity fills only **~14% of the frozen→oracle
  gap**. It is a robust *improvement over frozen random*, far from *solving* the task.
- **5/6, not 6/6** (seed 42 marginal). A promising direction, not a closure — hence `status: contributing`.

## Why it matters + next

The first positive on-bridge signal on the gap#4 wall, and it validates the reframe: the blocker was a
non-movable hidden, and local learning on a movable one *does* help transport-free. Next: (1) close seed 42 →
6/6 and grow the margin (it is small); (2) the real test — add a SUPERVISED error term to the plateau plasticity
(directed deep credit on the movable hidden) and check `deep_credit_share` rises well past 0.14; (3) the parallel
Deep Feedback Control arm (research gate's fallback) drives the tonic-pinned hidden directly. Rate-level deep
credit is SETTLED — cite, don't re-derive.
