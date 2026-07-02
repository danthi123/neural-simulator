# EMERGE-6b (rung-3a iteration 3) — the proper e-prop first-order eligibility RE-LOCALIZES the wall: credit quality is NOT the bottleneck; autonomous-GENERATION STABILITY is (research gate fires: FORCE / Laje-Buonomano, not another local rule)

**2026-07-02 (autonomous; substrate ladder rung 3a, iteration 3).** Runner `research/runners/_emerge6_recurrent_microcircuit_seq_derisk.py` (extended); result `research/findings/raw/_emerge6_recurrent_microcircuit_seq.json` + log `_emerge6_eprop_run.log`. Reuse-by-import; NO `sim/` edit; CPU; ≤4 workers (owner gaming — light contention). Multi-seed 42/43/44.

## Why this ran
EMERGE-6 (rung-3a, `2026-07-02-emerge6-rung3a-...-BOUNDARY.md`) found the memoryless target-based recurrent rule learns the local one-step map but not autonomous multi-step recall, and diagnosed the scoped next mechanism as a **proper recurrent e-prop first-order eligibility trace** (Bellec 2020) — scoping risk #2. This iteration builds it: a **leaky-integrator** recurrent unit (`u_t = κ·u_{t-1} + W@pre`, `a_t = σ(u_t)`, κ=0.7) with a **per-synapse** eligibility `ε_ji = κ·ε_ji + pre_i` (recurrent sensitivity carried forward through the neuron's own membrane leak), gated by the post pseudo-derivative: `e_ji = σ'(u_j)·ε_ji`, credit `ΔW_ji += (a*_j − a_j)·σ'(u_j)·ε_ji`. **Local by construction** (own leak + own pre-rate + own target error; no `W.T`, no off-diagonal broadcast, no BPTT — `used_transpose` False, all arms/seeds). The memoryless forward arms are preserved byte-identically (κ=0) as the rung-3a baseline; the e-prop family carries its own full anti-cheat panel.

## The decisive 2×2 (N=32, T=140, 600 epochs, lr=0.5, κ=0.7; 3 seeds)

| training | eligibility rule | one-step map | autonomous recall |
|---|---|---|---|
| teacher-forced | memoryless forward | **0.963** | 0.008 (dead — exposure bias) |
| teacher-forced | **leaky e-prop** (`eprop_tf` diagnostic) | **0.695** | −0.031 (dead — exposure bias) |
| free-run (scheduled sampling) | memoryless forward | 0.716 | −0.249 (dead — destabilized) |
| free-run (scheduled sampling) | **leaky e-prop** (`mc_eprop`) | −0.282 | −0.213 (free-run destroys the map) |

Anti-cheats for the e-prop family all intact/load-bearing: eprop-lesion +0.04, eprop-null +0.04, eprop-shuffled −0.22, eprop-wrong-sign ~0; locality asserted for every arm/seed.

## Verdict: BOUNDARY (build-informative) — the wall is RE-LOCALIZED
The `eprop_tf` diagnostic is the load-bearing result. **The leaky e-prop first-order eligibility learns the one-step map teacher-forced (0.695 ≈ the memoryless forward map), so credit QUALITY is NOT the bottleneck** — the proper recurrent eligibility, the scoped fix, does its job on the local map. Yet autonomous recall is **dead for every training mode**:
- **teacher-forced → exposure bias** (recall ~0 for BOTH rules: forward 0.008, e-prop −0.031);
- **dynamics-in-loop free-run → destabilizes the map** (e-prop free-run one-step −0.282; and its recall is dead −0.213; the leaky unit is *more* fragile to free-run than the memoryless forward, because membrane integration compounds the free-run error).

So **two mechanistically distinct local recurrent credit rules (forward eligibility AND proper e-prop first-order eligibility) both fail autonomous trajectory generation the same way.** The bottleneck is not how credit is assigned — it is the **autonomous-generation STABILITY of the learned recurrent dynamical system** (a different problem class). This triggers research-gate condition (f): ≥2 distinct approaches to the same goal have failed → the next move is a **read-only deep-research round on autonomous-generation-stability mechanisms**, NOT another local-rule tweak:
1. **FORCE / recursive-least-squares feedback** (Sussillo–Abbott 2009) — the canonical solution to training a recurrent net to autonomously generate a target trajectory: fast feedback keeps the network *on* the target during training so errors never compound (directly attacks the free-run-destabilization + exposure-bias double-bind).
2. **Laje–Buonomano innate stable trajectories** (2013) — train a reservoir's recurrent weights to make an innate trajectory a stable attractor (biologically grounded neural-sequence stability).
3. **Reservoir + trained feedback readout** (echo-state with output feedback) — the generative dynamics live in a fixed recurrent pool + a trained readout fed back.

## Honest scope
- This is a **build-informative** boundary: the scoped mechanism (e-prop) was built + tested faithfully and it correctly localized where the real wall is. That is the ladder working as intended, not a dead end.
- Rate-level, no spike noise (rung-3b) and NO `sim/` port — both stay gated behind a rung-3a GO. **Do NOT port.**
- κ=0.7 was chosen to balance recurrent-credit depth against over-smoothing the fastest trajectory mode (~28-step period); the diagnostic confirmed the map IS learnable at this κ (0.695), so κ is not the limiter. A κ/lr sweep is not the next move — the generation-stability family is.

## Artifacts
`research/runners/_emerge6_recurrent_microcircuit_seq_derisk.py` (+ leaky-unit `_step`, e-prop `elig` branch, `ARM_SPEC` e-prop family incl. the `eprop_tf` diagnostic, diagnostic-aware verdict), `research/findings/raw/_emerge6_recurrent_microcircuit_seq.json`. Prior: `2026-07-02-emerge6-rung3a-recurrent-microcircuit-sequence-BOUNDARY.md`, `2026-07-02-rung3-recurrent-microcircuit-sequence-scoping.md`.
