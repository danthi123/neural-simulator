# EMERGE-6 (rung-3a) — BOUNDARY (build-informative): the target-based recurrent rule LEARNS the local one-step map but does NOT stabilize autonomous multi-step recall; the crux is a recurrent eligibility trace, not exposure

**2026-07-02 (autonomous; substrate ladder rung 3a — the recurrent sequence cortex, the communication target).** Runner `research/runners/_emerge6_recurrent_microcircuit_seq_derisk.py`; result `research/findings/raw/_emerge6_recurrent_microcircuit_seq.json`. Reuse-by-import (EMERGE-3/5c microcircuit credit machinery); NO `sim/` edit; CPU; run capped at 4 workers (owner gaming — light contention). Multi-seed 42/43/44.

## Why this ran
Rung 2 (EMERGE-5c) established the Sacramento–Senn microcircuit's **active interneuron cancellation** as the noise-robust credit rule (0.981 strict vs Burstprop 0.622). Rung 3 carries that credit rule into a **recurrent** network trained to reproduce a temporal trajectory — the minimal precursor to a sequence-producing (communication) cortex. The rule under test is the Muratore–Capone–Paolucci **target-based** recurrent analogue (the diagonal/local limit of the Capone unified error↔target framework): `Δw ∝ (a* − a) · e_t`, with a forward-only eligibility trace `e = α·e + (1−α)·pre`, **no BPTT, no weight transport** (locality asserted per-arm; `used_transpose` False for all arms, all seeds).

## The task
`make_seq_task`: a periodic sinusoid-superposition trajectory (deterministic, held-out tail). Train the recurrent map on a prefix; then **autonomously free-run** the network from a seed state and measure `recall_heldout` (correlation of the network's own generated continuation vs the true held-out tail). `onestep` = teacher-forced one-step-ahead correlation on the held-out region (is the LOCAL map learnable?).

## Results (N=32, T=140, 600 epochs, lr=0.5, α=0.7; 3 seeds)

| arm | recall_heldout | onestep | reading |
|---|---|---|---|
| **mc_freerun** (scheduled-sampling dynamics-in-loop, PRIMARY) | **−0.249** | **+0.716** | map learnable; autonomous recall DEAD/negative |
| recurrent_microcircuit (naive teacher-forced) | +0.008 | **+0.963** | map cleanly learnable; recall DEAD (exposure bias) |
| hebbian_selforg (no-teaching self-org null) | +0.000 | +0.000 | task does NOT self-organize without credit |
| apical_feedback_lesion (anti-cheat) | +0.025 | −0.015 | credit path load-bearing (kill it → floor) |
| wrong_sign (anti-cheat) | +0.124 | +0.120 | flipped credit → no learning |
| no_teaching_null (anti-cheat) | +0.025 | −0.015 | = floor ✓ |
| shuffled_target (temporal-order anti-cheat) | −0.256 | −0.113 | order destroyed → matches the dead free-run |
| untrained (floor) | +0.025 | −0.015 | floor |

> **CORRECTION 2026-07-31 (wording of the corroboration claim, not the measurement).** `lever-efficacy`
> flagged 18 identical-arm pairs in this run's artifact. Audited: `apical_feedback_lesion`,
> `no_teaching_null` and `untrained` are identical to sixteen digits (`onestep=-0.0698238953499733`), and
> the same holds for the three `eprop_*` arms. **This is NOT a failed manipulation — it is the opposite.**
> A lesion that removes the credit signal entirely produces *no weight update*, so the resulting network
> IS the untrained network, byte for byte. The lesion engaged maximally.
>
> What is wrong is the verdict sentence below: *"lesion + wrong-sign + null all collapse to the floor"*
> reads as THREE independent controls corroborating each other. There are TWO. `wrong_sign` (+0.124/+0.120)
> is genuinely independent and did move; `lesion` and `null` are one condition — no weight update — reached
> two ways, and they equal the floor *by construction* rather than by measurement. The triangulation is
> illusory even though every individual number is correct.
>
> The BOUNDARY verdict itself stands: it rests on the primary arms, `wrong_sign`, and `shuffled_target`.

## Verdict: BOUNDARY (build-informative, NOT a stop; do NOT start the `sim/` port)
The target-based microcircuit credit rule genuinely **learns the local one-step map** — naive teacher-forced `onestep` 0.963, free-run 0.716 (both ≥ 0.70, task-sane), while every non-credit control (hebbian/lesion/null/untrained) sits at ~0 one-step. The credit signal is real and load-bearing (lesion + wrong-sign + null all collapse to the floor). **But autonomous multi-step recall is dead for BOTH training modes:**
- **naive teacher-forced → exposure bias**: trained only on ground-truth prefixes, `recall` collapses to 0.008 when the network is fed its own outputs (the classic teacher-forcing / free-run gap).
- **dynamics-in-loop scheduled sampling → still dead (−0.249)**, and does NOT beat the apical-lesion (0.025) or the hebbian null (0.000). Preserving the map with a gentler free-run learning rate (0.15× the teacher-forced lr — the diagnosed fix for the lr=0.5 destabilization, which had crushed the free-run one-step to −0.010) kept the local map (0.716) but did **not** rescue recall.

So the crux is **NOT exposure** (scheduled sampling addresses exposure and still fails) — it is **credit for the recurrent trajectory**. The shuffled-target control (−0.256, ≈ the dead free-run) confirms the free-run output carries no residual temporal structure to read.

## Diagnosis + the scoped next mechanism (rung-3a iteration 3)
The forward eligibility trace `e = α·e + (1−α)·pre` captures only the instantaneous pre-synaptic activity. It is sufficient for the **single-step** credit (which is why the one-step map is learnable) but it does **not** carry the **recurrent sensitivity** (`∂a_t/∂W` propagated through the recurrence) that multi-step autonomous recall requires — so the trajectory/attractor is never stabilized. This is exactly **scoping risk #2** (`2026-07-02-rung3-recurrent-microcircuit-sequence-scoping.md`). The pre-registered next mechanism:
1. **A proper recurrent (e-prop first-order) eligibility trace** (Bellec 2020) that carries the recurrent Jacobian forward locally — the field-standard local rule for training recurrent spiking networks without BPTT/weight-transport. This is the diagnosed cheap-first next step.
2. **and/or burst-window gating** of the credit (the microcircuit's burst channel gating when the target-based update applies).
3. If a proper eligibility still can't stabilize recall → this is a **recurrent-noise-family limit**, and the shortlist's Urbanczik–Senn population-feedback / NMNC (`references/papers/reports/_SHORTLIST-2025-2026-mechanism-papers-to-acquire.md`) is the next research-gated read.

## Honest scope
- This is a **rate-level** de-risk of the credit rule on a recurrent sequence task (the ladder's cheap-first rung), NOT the spiking substrate build. The spiking noise pass (rung-3b) and the `sim/` two-compartment port (rung-4) remain gated behind a rung-3 GO — **do NOT port**.
- The one-step-map-learnable / recall-dead split is a genuine, multi-seed, anti-cheated characterization of where the local recurrent rule stands — a boundary WITH a specified next mechanism (per the master directive, a boundary is an undiscovered mechanism to work past, not a wall).

## Artifacts
`research/runners/_emerge6_recurrent_microcircuit_seq_derisk.py` (+ `RecurrentMicrocircuitRNN`, scheduled-sampling `free_run` with a 0.15× effective lr, `--max-workers` cap), `research/findings/raw/_emerge6_recurrent_microcircuit_seq.json`. Prior: `2026-07-02-rung3-recurrent-microcircuit-sequence-scoping.md`, `2026-07-02-emerge5c-microcircuit-noise-robust-GO.md`.
