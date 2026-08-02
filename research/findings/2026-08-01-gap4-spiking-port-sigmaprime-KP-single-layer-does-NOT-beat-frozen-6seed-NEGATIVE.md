---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/sigmaprime_6seed_learned.json
  - research/findings/raw/gap4/realspikes/sigmaprime_6seed_fixed.json
---

# gap#4 crux — SPIKING PORT (single-layer): σ′(v−θ) graded credit + KP-learned transport-free feedback does NOT robustly beat a frozen reservoir at 6 seeds — the fragile seed-42 positive did not survive; a weak learned>fixed directional signal only

<!--derived-->
**One-line verdict.** The rate overturn (transport-free chained-FA + σ′ clears the depth-2 ceiling; KP-learned
feedback rescues MNIST depth-4) does NOT transfer to a SINGLE-LAYER real-spikes port at cheap budget. On the
`RealSpikesPlateauExpander` substrate (features → columns → readout, ONE plastic layer, epochs 20, n_col 200),
supervised σ′(v−θ)-gated credit with KP-learned transport-free feedback beats the frozen on-bridge reservoir
only **3/6** (needs 6), mean deep_credit_share **+0.012** (CREDIT 0.340 vs FROZEN 0.321), and the anti-cheats
are not clean on all seeds (`anti_ok False`). The seed-42 smoke positive (learned +0.111, dcs +0.167) was a
FRAGILE single-seed instance — per-seed dcs across 6 seeds is [+0.167, −0.073, −0.286, −0.032, +0.097, +0.200].
No `sim/` edit (subclass of the real-spikes expander; additive runner).

## Result — 6 seeds (42/43/44/100/101/102)

Artifacts: `research/findings/raw/gap4/realspikes/sigmaprime_6seed_learned.json` (learned) and
`research/findings/raw/gap4/realspikes/sigmaprime_6seed_fixed.json` (fixed-random control). Backend numpy/CPU.

<!--derived-->
| arm (single plastic layer, real spikes) | mean CREDIT held-out | dcs>0 | beats-frozen ≥0.05 | mean dcs |
|---|---|---|---|---|
| **KP-learned + σ′(v−θ)** | 0.340 | 3/6 | 3/6 | **+0.012** |
| fixed-random FA + σ′(v−θ) | 0.312 | 3/6 | 1/6 | −0.035 |

Per-seed CREDIT (learned): [0.444, 0.185, 0.241, 0.407, 0.444, 0.315] vs FROZEN [0.333, 0.241, 0.389, 0.426,
0.389, 0.148]. GO gate (learned beats frozen ≥5/6 AND dcs>0 6/6 AND anti clean) FAILS on all three.

## What holds, and what does not

<!--derived-->
**What holds (a real, if weak, directional signal): learned > fixed on the expander.** The KP-learned arm beats
the frozen reservoir on 3/6 seeds vs the fixed-random arm's 1/6, and mean dcs +0.012 vs −0.035 — the same
ORDERING as the rate crux (learned rescues where fixed does not). And two substrate facts are confirmed: (1)
σ′(v−θ) IS computable and genuinely graded on the substrate (98% graded, input-selective, reproducibility 1.0)
— and it is the ONLY usable somatic credit signal because the columns never somatically spike (1-bit event read
degenerate); (2) the credit path is transport-free (asserted in-run, no `self.W`; Y_moved for the learned arm).

**What does NOT hold: it is not a GO.** The learned>fixed margin is +0.047 mean, well inside the seed noise
(per-seed dcs spans −0.286 to +0.20); neither arm clears the frozen reservoir robustly; anti-cheats are not
clean on all seeds. The seed-42 positive was fragile — sensitive to the error-head conditioning (an underfit
head flips it negative).

## Honest scope + the next mechanism (this is a method-negative, not a capability wall)

<!--derived-->
This is a NEGATIVE for the **single-layer, cheap-budget** port, which is expected to be weak: the rate result's
power is at DEPTH (the MNIST depth-4 KP-rescue), and this port has ONE plastic layer — no multi-hop chain — so
it tests only the σ′ + fixed-vs-learned factors, not the depth-crux mechanism the rate result rests on. It does
NOT show the transport-free credit CLASS fails on spikes; it shows this single-layer instance at epochs-20 does
not beat a frozen reservoir that is already at 0.32 (vs oracle 0.97). The distinct value banked: on the
representable-forward expander, fixed-DFA under-delivers vs KP-learned — which informs the adjacent lane-C
(expander + fixed-DFA) that it should carry learned feedback.

**Next mechanism (named, not a wall):** the MULTI-HOP CHAINED spiking port at a REAL budget — chained
transport-free KP-learned feedback across ≥2 plastic layers with the σ′(v−θ) graded read, where the rate result
shows the power lives (depth). Plus the error-head conditioning (standardized margins + inner descent) that the
seed-42 fragility exposed. This single-layer negative maps the boundary point; the depth version is the shot.
