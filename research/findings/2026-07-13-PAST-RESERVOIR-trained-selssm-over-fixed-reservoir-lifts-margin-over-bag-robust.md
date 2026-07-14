# The TRAINED selective channel over a FIXED reservoir robustly lifts margin-over-bag (~+0.62, 5/5) where the FIXED gate HURT — and the lift is DATA-ROBUST: it holds as the fixed reservoir's own margin decays toward its Ueda-bound (the selective supplies the durable memory the reservoir loses)

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_batched_scale_trained_selssm_derisk.py` · raw `research/findings/raw/_trainedsel_scale/`. numpy (GPU-capable batched reservoir); NO `sim/` edit.
**Status:** ✅ decisive + robust — the LEARNED gate is the scale-critical ingredient (settles the fixed-gate negative). Tractable + GPU-scalable via the batched infra.

## Why

The fixed-gate scale probe was a NEGATIVE (an untrained selective channel HURTS margin-over-bag → the learned gate is required). This trains the gate over a FIXED echo-state reservoir (batched-collected, fast) — cheap (no O(n²) reservoir e-prop) and GPU-scalable toward the validated regime — and asks whether a TRAINED selective lifts margin-over-bag where the fixed one hurt. Everything transport-free (read-out local delta; gate forward eligibility × fixed random feedback, no BPTT/transport); the SAME simple trainer for both arms (fair).

## Result (np=200, V=120, TinyStories; margin-over-bag = bag_ce − arm_ce)

| nt | seed | m_res (res−bag) | m_sel (res+trained-sel − bag) | sel_lift | sel−bigram (aggregate) |
|---|---|---|---|---|---|
| 800 | 42 | +0.153 | +0.753 | **+0.600** | +0.256 |
| 800 | 43 | +0.176 | +0.782 | **+0.606** | +0.293 |
| 800 | 44 | +0.160 | +0.828 | **+0.667** | +0.317 |
| 1600 | 42 | +0.072 | +0.677 | **+0.604** | +0.158 |
| 1600 | 43 | +0.083 | +0.718 | **+0.635** | +0.205 |

- **`sel_lift` ~+0.62 on 5/5 runs** (nt=800 mean +0.624; nt=1600 mean +0.620) — the TRAINED selective decisively + robustly lifts margin-over-bag, where the FIXED gate HURT (−0.076). The LEARNED gate is the ingredient (a fixed hold was noise).
- **The lift is DATA-ROBUST**: it holds ~+0.62 as data grows (nt 800→1600), EVEN THOUGH the fixed reservoir's OWN margin over the bag SHRINKS (m_res +0.16→+0.08 — the reservoir-scale Ueda-bound: the fixed reservoir's dynamics matter less as the bag catches up with more data). ⇒ **the trained selective supplies a durable ~+0.62-nat memory over the bag that the fixed reservoir loses with scale** — the selective compensates for the reservoir's fading-memory decay. This is the mechanism's scale value: where the reservoir's own contribution decays toward the n-gram floor, the learned selective channel keeps adding.

## ⇒ honest read (adversarial-verify + null-discriminator disciplined)

- **Robust decisive claim:** over a fixed reservoir, a TRAINED selective channel adds ~+0.62 nats over the memoryless bag, robustly across seeds AND data-robustly (holding as the reservoir's own margin decays). The LEARNED gate is the scale-critical ingredient. This settles the fixed-gate negative and is realized at the batched-scale-infra level (GPU-scalable).
- **NOT over-claimed:** the aggregate `sel−bigram` SHRINKS with data (+0.29→+0.18) — the bigram is a strong, fast-improving baseline at this tractable (null-discriminator) scale; sel still beats it on aggregate but the margin narrows. Per the a-1 null-discriminator finding + the adversarial-verify shallow-concern lesson, the aggregate-vs-bigram is NOT the deep-tail claim; margin-over-bag is the robust headline. The absolute deep-TAIL-vs-bigram win needs the validated-signal regime (23.7M/V=2000), the named GPU follow-on (this runner is the tractable, GPU-scalable path to it — the reservoir is batched; the gate+read-out training is the cheap part).

## Next
- Scale this runner toward the validated regime (larger V/data on GPU; vectorize the gate+read-out training loop for the V=2000 / large-nt run — the reservoir collection is already batched).
- A by-depth breakdown (does the trained-sel margin-over-bag concentrate at the deep tail, as the joint runner's did?).
- raw `research/findings/raw/_trainedsel_scale/*.json`.
