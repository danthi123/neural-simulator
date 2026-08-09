---
title: "Fixed-budget prioritized replay bounds per-sleep COMPUTE but FAILS retention at scale (N=50: 0.40 vs full 0.74) — the compute half needs a SUB-LINEAR budget or a compressing generator"
date: 2026-08-09
type: finding
status: contributing
lane: memory-continual-learning
seeds: [42, 43, 44]
---

# Prioritized replay at a FIXED budget k bounds compute but loses coverage at scale — NEGATIVE on the core claim

## Claim

<!--derived-->

To bound per-sleep COMPUTE (the owner's speed concern), each sleep replays only a bounded k≪N subset chosen by a
NEURAL forgetting-risk signal (recall margin read from the readout's own spiking on the reactivated engram;
stochastic risk-weighted, Mattar-Daw 2018). Result (verify CONFIRMED): the per-sleep TRAINING compute IS bounded
(O(k), 48 events/sleep constant across N vs full's 160→400), and prioritized BEATS random-k at N=20 (+0.20, the
neural signal is load-bearing) — **but retention does NOT hold as N grows.** PARTIAL at N=20 (mean 0.85 vs full
0.967, 2/3 within band), and **REFUTED at N=50 (prioritized 0.40 vs full 0.74, −0.34; at N=25 prioritized 0.32 even
LOST to random 0.56).** A fixed k=6 (12% of N=50) is below the coverage floor: you get flat compute OR retention,
not both. **NEGATIVE on "fixed k≪N matches full at equal retention independent of N."**

## Data

<!--derived-->

| | N=20 (k=6, 3-seed) | N=50 (k=6, s42) |
|---|---|---|
| prioritized_k | 0.85 (0.80/1.00/0.75) | **0.40** |
| full O(N) | 0.967 | 0.74 |
| random_k | 0.65 | — (pri lost to random 0.56 @N=25) |

Raws: `research/findings/raw/teacher_loop_prioritized_replay_N20_s42.json`,
`research/findings/raw/teacher_loop_prioritized_replay_N20_s43.json`. Runner + N=50 raw: branch commit ed6101362.

## Read — honest, and it unifies the scalability picture

<!--derived-->

- **The neural priority signal is genuine (not a cheat):** `_select_indices` takes only (net, hippo), no
  future/oracle args; `priority_used_future=False` verified. It reads recall margin from the actual spiking readout
  on the hippocampus's own stored label. And it beats random-k at N=20. So the mechanism is real — it just isn't
  enough at a fixed budget.
- **Two honest confounds** the synth flagged: (a) at N=50 `full` itself only retains 0.74 (the capacity/acquisition
  ceiling — `2026-08-09-capacity-scaling-*SLIPS-at-N100`), so the bounded-k question is ill-posed against a degraded
  full; (b) the priority SELECTION SCAN is still O(N) cheap inference — only the training step is O(k). So even the
  "O(k) compute" headline is partial.
- **The unifying insight:** a fixed budget over INDIVIDUAL facts loses coverage at scale — the SAME failure mode as
  the bounded raw buffer (`0c7531785`). Bounding compute at equal retention needs either a **SUB-LINEAR budget**
  (k ~ log N or a small fixed FRACTION, not a constant) OR — the deeper lever — a **COMPRESSING / compositional
  generator** that rehearses SHARED STRUCTURE sub-linearly, so one replay covers many facts. **The compressing
  generator is the single lever that would unlock BOTH the storage half AND the compute half.**

## The complete scalability map (after tonight's arc)

<!--derived-->

- **Retention:** CLOSED at N=20 (non-forgetting generator matches flat, `0933fdb7a`); OPEN at N=50+ (the capacity
  ceiling — full itself degrades).
- **Storage:** bounded engram store (fixed generator, GO) but a truly COMPRESSING generator (sub-linear params) is
  OPEN — and is the unifying lever.
- **Compute:** fixed-k prioritized replay = NEGATIVE (this); sub-linear k / compressive replay is the live path.
- **Acquisition/capacity at scale:** OPEN — the readout can't cleanly ACQUIRE 50-100 facts (upstream of all).

## Rigor

3-seed N=20, single-seed N=50 (the scaling refutation reproduces byte-identically in verify). Neural signal
substrate-derived (verified, not an oracle); k bounded (per-sleep replay-set ≤ k asserted); cfg.seed byte-identical;
de-clamped bdsp_wmax=1e9; no `sim/` edit; backend numpy.

NEXT (named, ordered): (1) de-confound — re-run N=50 with cortex capacity scaled to N so `full` retains ~1.0, making
the bounded-k question well-posed; (2) SUB-LINEAR k (k ~ log N / a fraction) vs full; (3) the COMPRESSING
compositional generator (unlocks storage + compute together). Grounding: Mattar-Daw 2018, Tse 2007, van de Ven 2020
(all DR-recorded). NO-EXTERNAL-NEEDED: internal scaling de-risk.
