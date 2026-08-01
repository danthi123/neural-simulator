---
type: finding
status: live
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/eprop_noise/_k8_noise_AB_aggregate.json
  - research/findings/raw/eprop_noise/
---

# gap#4: e-prop at K=8 reproduces the deep-credit closure WITH CLEAN PROVENANCE, and the decorrelation-noise hypothesis is REFUTED

**One-line verdict:** the transport-free e-prop rule on the production Izhikevich bridge, population K=8, clean
drive, reaches inherit **0.778 mean (3/3 seeds, 0.741/0.815/0.778)** — near the LIF ceiling — reproducing the
banked K=8 closure that previously had NO recorded knobs. And adding the independent OU/conductance noise the
"√K decorrelation" hypothesis predicted should help **collapses it to 0.197** (a large drop, see the aggregate's `inherit_delta_on_minus_off`). So the
population benefit for e-prop is NOT decorrelation; the closure's correct config is clean drive, now recorded.

Aggregate: `research/findings/raw/eprop_noise/_k8_noise_AB_aggregate.json` (per-seed data embedded; backend
cupy, pool_k=8, epochs 80, train_subsample 160, settle 40; noise-ON cells flagged `below_chance` — the
collapse is the result, not an instrument failure).

## The result — 3 seeds {42,43,44}, K=8, on the real bridge

| arm | inherit per seed | mean | train | permuted |
|---|---|---|---|---|
| **noise-OFF (clean drive)** | 0.741 / 0.815 / 0.778 | **0.778** | 0.93–0.95 | 0.15–0.26 (no leakage) |
| **noise-ON (+ou+cond)** | 0.148 / 0.185 / 0.259 | **0.197** | 0.25–0.28 (collapse) | ~chance |

Chance 0.333. noise-OFF is near the LIF-framework ceiling and **far above the on-bridge plateau of
0.47** <!--derived: quoted from the 2026-07-14 on-bridge under-training finding--> that the whole gap#4 arc
was stuck on. noise-ON collapses BOTH train (0.93→0.25) and inherit — adding the noise destroys the training.

## Why noise HURTS e-prop (the mechanism, and the resolution of the crux question)

The e-prop port DISABLES OU/conductance noise by design — the runner's own constructor comment says they
"make the forward stochastic," and e-prop's forward eligibility traces + membrane surrogate gradients need a
**clean, stationary function of the input** to carry credit. Turn the noise on and the eligibility/surrogate
go noise-dominated, so credit assignment fails and training collapses. The substrate's per-neuron
firing-threshold **heterogeneity** already decorrelates the pool enough for population coding to help at K=8 —
no *added* independent noise is needed or wanted.

**This answers the priority-1 crux question — "why does population coding work for the e-prop net when the
same lever is measurably inert for the BDSP net?" — and the answer is the OPPOSITE of the starting
hypothesis.** It is NOT that e-prop decorrelates via noise (it doesn't; noise hurts it). e-prop reaches
near-ceiling at K=8 on clean, nominally-correlated drive; the BDSP net's flat read-SNR across K (the
2026-07-14 06:55 smoke) is a property of the **BDSP credit rule**, not the pool. The two nets differ in the
credit rule, and e-prop's works with population coding on clean drive while one-step BDSP's does not.

## What this does to the banked closure

It **validates it as-is.** The banked K=8 closure (reported near the LIF ceiling in the biology entry) ran
noise-OFF (ou_noise/cond_noise default False, never
passed True anywhere). The biology entry flagged that unrecorded knob as a possible hidden confound. This A/B
shows the opposite: noise-OFF is the CORRECT config, and turning the knob ON would have made it worse — so the
closure was right, it just lacked provenance. Now it has it: pool_k, epochs, subsample, settle, and the noise
knobs are all recorded in the artifact config.

## Honest scope
3 seeds (the experiment's design), K=8 only. inherit 0.778 is near but not AT the LIF ceiling (0.89) — a small
residual remains. The √K trend {K=1,4,16} noise-OFF is running now to confirm the closure across K (K=1 should
sit near the 0.47 plateau, K=16 tests whether it exceeds K=8). This is on-bridge e-prop, the genuinely-open
07-14 question — NOT the redundant BDSP ceiling crux (still do-not-relaunch).
