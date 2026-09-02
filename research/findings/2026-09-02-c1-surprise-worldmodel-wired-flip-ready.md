---
type: finding
status: partial
date: 2026-09-02
mechanism: c1-surprise-worldmodel-production-wiring-flip-verify
board: one-brain integration / cross-edge C1
seeds: [42, 43, 44, 100, 101, 102]
artifact: research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_surprising.json
---

# C1 (D2-surprise → E2-world-model) WIRED into live chat + offline flip-verify GO — PARTIAL: real-brain_chat end-to-end ARM A/B/C still owed before flip-ready

**2026-09-02.** The C1 cross-edge — the world-model's own spiking D2 prediction-error unit gating an online update
of the E2 forward model — passed its 6-seed cupy de-risk GO earlier today (research/findings/raw/_crossedge_surprise_worldmodel_6seed.json).
It is now WIRED into the live chat path (default-OFF), and its production organ passes a 6/6 numpy OFFLINE self-test
with a load-bearing vary/lesion, advancing it from de-risked toward WIRED. **HONEST SCOPE: the arm data below is the
production organ's OFFLINE self-test, NOT the end-to-end real-`webapp.server.brain_chat` ARM A/B/C flip-verify** —
that end-to-end pass was interrupted (the controller mistakenly stopped the build agent, misreading a long compute
phase as a stall, just as it was starting the real-brain_chat ARMs). So this is PARTIAL: WIRED + offline-GO, with
the end-to-end flip-verify (ARM A byte-identical-off, ARM B on real traffic, ARM C no-regression through the battery)
the remaining rung before genuinely flip-ready. Two real wiring bugs were found + fixed by the offline self-test
(an `id()`-keyed cache colliding across GC'd organs; a gate calibration that degenerated post-lesion).

## What landed
- **Production organ** `research/runners/onebrain_xedge_surprise_worldmodel_production.py` — the C1 edge as a
  reusable production module (surprise-gated Hebbian window on the E2 state→pred transition).
- **Live wiring** `webapp/server.py` — behind `BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL` (**default-OFF,
  byte-identical when off**; the update applies only when `_c1_on`). No production behavior changes today.
- **Flip-verify harness** `research/runners/_crossedge_surprise_worldmodel_flip_verify.py` (6/6 numpy self-test GO).

## Offline self-test: ARM B (vary/lesion) is load-bearing per-seed

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
PARTIAL: WIRED default-OFF, offline self-test 6/6 GO + offline ARM-B load-bearing. **The remaining rung before
flip-ready is the end-to-end real-`brain_chat` ARM A/B/C flip-verify** (interrupted mid-build) — re-run
`_crossedge_surprise_worldmodel_flip_verify.py` through the real server path, prove ARM A byte-identical-off /
ARM B load-bearing on live traffic / ARM C no-regression on the Phase-1 battery. Only then is the default-ON flip
(an owner UX call) meaningful; after the flip, retire any host forward-model-update path (scaffold_retired).
