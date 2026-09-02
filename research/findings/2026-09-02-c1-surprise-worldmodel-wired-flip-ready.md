---
type: finding
status: live
date: 2026-09-02
mechanism: c1-surprise-worldmodel-production-wiring-flip-verify
board: one-brain integration / cross-edge C1
seeds: [42, 43, 44, 100, 101, 102]
artifact: research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_surprising.json
---

# C1 (D2-surprise → E2-world-model) WIRED into live chat, flip-verify GO — flip-READY, default-OFF pending owner call

**2026-09-02.** The C1 cross-edge — the world-model's own spiking D2 prediction-error unit gating an online update
of the E2 forward model — passed its 6-seed cupy de-risk GO earlier today (research/findings/raw/_crossedge_surprise_worldmodel_6seed.json).
It is now WIRED into the live chat path and flip-verified, advancing it from de-risked to WIRED (the third stage of
the de-risked → WIRED → on-by-default → scaffold-retired bar). Finalized from the isolated C1 build agent's pushed
branch after that agent stalled on a dead background verify (its work was verified clean before merge).

## What landed
- **Production organ** `research/runners/onebrain_xedge_surprise_worldmodel_production.py` — the C1 edge as a
  reusable production module (surprise-gated Hebbian window on the E2 state→pred transition).
- **Live wiring** `webapp/server.py` — behind `BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL` (**default-OFF,
  byte-identical when off**; the update applies only when `_c1_on`). No production behavior changes today.
- **Flip-verify harness** `research/runners/_crossedge_surprise_worldmodel_flip_verify.py` (6/6 numpy self-test GO).

## Flip-verify: ARM B (vary/lesion) is load-bearing per-seed

Artifacts: `research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_surprising.json`,
`research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_expected.json`, and
`research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_surprising_lesion.json`. On seed 44:
- **expected** (no surprise): edge weight flat (1.422), `n_gated=0`, no growth — nothing to learn.
- **surprising** (surprise present): edge GROWS (w 0.0986 → 0.3296, `n_gated=2`, `w_grew=0.231` <!--derived-->) and the
  world-model prediction margin SHIFTS (396.5 → 279.5) — the surprise genuinely updates the forward model.
- **surprising + lesion**: with the edge lesioned, weight stays flat (0.0986) and the margin returns to 396.5 —
  the effect VANISHES on lesion.

So the world-model update is attributable to the learned C1 edge, not a confound (the anti-hollow bar met).

## Status / next
Flip-READY: WIRED default-OFF with a GO flip-verify. The actual default-ON flip (making surprise drive the live
world-model by default) is a separate owner UX call, consistent with the board's other flip-ready faculties. Next
rung after the flip: retire any host forward-model-update path so the row reads scaffold_retired.
