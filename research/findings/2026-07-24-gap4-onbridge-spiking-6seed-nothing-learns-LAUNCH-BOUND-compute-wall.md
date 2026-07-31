---
type: finding
status: superseded
superseded_by: research/findings/2026-07-24-gap4-surpass-POWERED-NO-GO-tonic-pinned-frozen-representation-root-cause.md
date: 2026-07-24
mechanism: gap4-credit
---

# gap#4 on-bridge spiking port — 6-seed: NOTHING learns (even the idealized ceiling); the diagnostic exposed a LAUNCH-BOUND compute wall, not a mechanism failure (2026-07-24)

## Result (6-seed, GPU, `_gap4_onbridge_spiking_selfpredict_derisk`, feasibility-sized epochs=20 / train-subsample=128)
The on-bridge SPIKING port of the gap#4 self-predicting microcircuit (learned vs fixed feedback, transport-free) is
**0/6 GO — because NOTHING learns, including the idealized upper bound.** Per-arm held-out, mean over seeds 42 43 44
100 101 102 (chance 0.167):

| arm | held-out | note |
|---|---|---|
| oracle (host DendriticMLP) | **0.985** | task is learnable ✓ (ceiling valid) |
| transport_ceiling (idealized weight-transport feedback) | **0.139** | **AT CHANCE** — the smoking gun |
| reservoir / fixed_fa / best_learned(kp,micro) | 0.11–0.12 | at chance |

Anti-cheat guards pass (ceiling-guard-fails, ast-no-forward-W, lesion/shuffled collapse) but that is trivial when
every arm is at chance. **This is NOT a "learned feedback didn't beat fixed" result** — the learned-vs-fixed question
is UNANSWERED because the spiking net never learned the task. The transport-ceiling arm uses *perfect* weight-transport
feedback (Y:=Wᵀ); if the spiking forward+credit pipeline worked at all it would climb toward the oracle. It sits at chance.

## Why nothing learns — leading cause: SEVERE under-powering, forced by the substrate's speed
The oracle trained **250 epochs on 1260 examples**; the feasibility-sized spiking config got **20 epochs on 128** —
~**125× less training** — because at ~0.3 s/example the full (epochs 40 / full 1260) config measured out to ~9 days /
6 seeds. So the spiking net was starved of training.

## The diagnostic that killed the arc: the on-bridge net is LAUNCH-BOUND (the real wall)
A focused diagnostic (transport_ceiling only, 80 epochs / 256 subsample, 1 seed) to test "does the idealized ceiling
learn with 4× epochs?" ran **2h8m and was killed** — not because it hung (process state `Rl`, 99.9% CPU, utime
advancing) but because it is **LAUNCH-BOUND: 99.9% CPU / 3% GPU.** The ~2400-neuron / 1.3M-synapse net issues many
tiny CUDA kernels per step, so the CPU kernel-launch overhead dominates and the GPU sits ~idle. ⇒ each arm is ~hours,
and the full learned-vs-fixed comparison (5 arms × 6 seeds × enough epochs to learn) is **computationally infeasible at
a trainable scale on the current substrate.** This is the binding wall, above the under-powering.

## Cross-cutting: the ±5 BDSP weight-clamp may ALSO cap it
gap#5 (finding `2026-07-24-gap5-encode-only-derisk-NEGATIVE-...`, commit 6a9a44c3) found `fused_bdsp_update` returns
`cp.clip(w, bdsp_w_min=-5, bdsp_w_max=5)` **even at lr=0**. gap#4 on-bridge uses the same kernel with `bdsp_w_max=5`,
so every FF weight is bounded to ±5 — plausibly too tight for a 9-way task (the oracle has no such bound). A separate
sim/ clamp-fix task is filed; whether ±5 caps gap#4 is testable once the launch-bound cost is addressed.

## Verdict (per THE LAW — the METHOD is banked, the CAPABILITY stays OPEN)
- The **deep-directed-credit (learned vs fixed feedback) capability is already a RATE GO** (finding
  `2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO.md`), and the separation is spiking-only in
  PRINCIPLE (rate is byte-identical) — so the on-bridge spiking port is the decisive test, and it is currently blocked
  by COMPUTE, not by the mechanism.
- **Method banked NEGATIVE:** the on-bridge spiking port at full task scale is compute-infeasible (launch-bound +
  under-powered). **Capability OPEN.**
- **Surpass paths (ranked, for when this is prioritized):** (1) a **CUDA-graph / masked megakernel** for the on-bridge
  spiking forward (the same fix that made the RF composer's resonate loop ~4× faster — `cfg.enable_rf_cudagraph`) to
  kill the launch-bound overhead; (2) a **smaller task** (fewer classes/supers) the spiking net can learn in feasible
  epochs, so the learned-vs-fixed comparison can actually run; (3) fix the **±5 clamp** (filed) in case it caps learning.
- Speed is secondary per the mission, but "months per de-risk" is not practical — the megakernel or the smaller task is
  the pragmatic surpass. Not a mechanism failure; a substrate-speed wall with a specified way through.

## Files
- 6-seed raw: `research/findings/raw/gap4/onbridge_spiking_seed{42,43,44,100,101,102}.json`
- Design: `research/findings/2026-07-24-gap4-onbridge-spiking-port-DESIGN.md`; runner
  `research/runners/_gap4_onbridge_spiking_selfpredict_derisk.py`
