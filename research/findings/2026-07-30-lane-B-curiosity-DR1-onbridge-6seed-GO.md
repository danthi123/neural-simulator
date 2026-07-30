# Lane B (Curiosity): DR-1 on-bridge is 6/6 seeds GO — every anti-cheat control collapses

Run because the mechanical lane check named it as the one genuinely idle lane (A and C now report their de-risks
as BANKED / awaiting integration, D and E as served). It was the roadmap's `[CPU]` lane B, unserved all session.

**`_curiosity_seek_learn_onbridge_derisk`, seeds 42/43/44/100/101/102: 6/6 GO (ALL GO).** Every one of the seven
pre-registered criteria passed on every seed:

| criterion | seed-42 value |
|---|---|
| (a) `corr(gap, SPIKING-want) >= 0.9` | **+0.991** |
| (b) ask unknown >= 2x known | ratio enormous (known-asks ~0) |
| (c) confidence rises | **+0.55** from a 0.03 floor |
| noisy STOPS (veto) while gap stays high | asks 0.03 → **0.00** while g holds **0.97** (ELP 0.07 <= thr 0.12) |
| lesion | **asks = 0** |
| yoked | mastered 7 vs real 8 |
| permuted | corr **−0.08** |
| moat | True |

**The mechanism:** the `from_novelty` neuromodulator rule drives a spiking ASK pool; a spiking-SNc RPE value
critic learns on the LEARNING-PROGRESS reward. **SNc RPE learn-burst 14.7 Hz vs 0.0 Hz on the noisy condition**,
with learning-progress +0.245 vs +0.003 — so the reward signal genuinely discriminates learnable from noise.
**The veto is the interesting part:** on an unlearnable (noisy) item the agent STOPS asking even though its
novelty/gap signal stays HIGH (0.97) — i.e. it is not merely novelty-seeking, it tracks whether asking is
*paying off*. That is the difference between curiosity and novelty-chasing, and it is the property that makes
this a drive rather than a reflex.

**⇒ ROADMAP POSITION: lane B's de-risk is now BANKED like A and C.** Per the master roadmap, the next step for
all three is **"on-bridge spiking realizations of the Phase-0 GOs + wire into the develop-loop teacher hook"** —
and DR-1 is already on-bridge, so for lane B specifically the remaining work is the **develop-loop wiring**, a
build rather than another de-risk.

**Process note:** this lane sat unserved for an entire session while one arc was worked serially, and produced a
6-seed GO in ~2 minutes once actually launched. It was found only because the lane check prints each idle lane's
literal command.
