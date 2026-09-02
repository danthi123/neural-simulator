---
type: finding
status: live
date: 2026-09-02
mechanism: c1-surprise-worldmodel-production-wiring-flip-verify
board: one-brain integration / cross-edge C1
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: "the NEW end-to-end real-brain_chat ARM A/B/C pass below is single-seed (42, the harness's own fixed
  seed) — a real-handler demonstration, not a generalization claim. The 6-seed evidence for the underlying
  mechanism is the de-risk (research/findings/raw/_crossedge_surprise_worldmodel_6seed.json) and the offline
  production self-test already in this finding (arm_s{42,43,44,100,101,102}_*.json)."
artifacts:
  - research/findings/raw/_crossedge_surprise_worldmodel_flip_verify/numpy_run.json
  - research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_surprising.json
---

# C1 (D2-surprise → E2-world-model) end-to-end real-`brain_chat` ARM A/B/C is GO (FLIP_VERIFY_GO=True) — flip-ready; stays default-OFF pending an owner-gated flip

**2026-09-02.** The remaining rung this finding owed — the end-to-end flip-verify through the REAL
`webapp.server.brain_chat` handler, not the production organ's offline self-test — is now run and GO.
`research/runners/_crossedge_surprise_worldmodel_flip_verify.py` was executed against seed 42, numpy backend,
through the real handler (fresh per-config subprocess builds, exactly as the harness's own docstring specifies),
and all three arms pass:

- **ARM A (byte-identical-off):** 4/4 items match between `A_baseline` (env unset) and `A_off`
  (`BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL="0"` explicit, never popped) — the answer text, abstain, and the
  worldmodel decision fields are identical, and the new `worldmodel.surprise_worldmodel_crossedge` diagnostic key
  is absent in both.
- **ARM B (visible-on-real-traffic + lesion-attributable):** on the real alternating-valence sequence
  (`B_on`), the gate opened on 5 of 8 turns; summed over those 5 turns' own before→after deltas the observed-pool
  weight's cumulative gated growth is `growth_on=0.5299070244655013` (`numpy_run.json`
  `aggregate.arm_B_visible_on_real_traffic.growth_on`) — the weight itself starts the sequence at 0.0128 and, by
  the last gated turn, sits at 0.3609 (it is not monotonic turn-to-turn; two of the eight turns fail to clear the
  gate threshold and contribute no growth, which is why the summed growth exceeds the net first→last change).
  On the SAME gating code with a confirming, never-violating sequence (`B_on_expected`), the gate opened 0 of 8
  times (`growth_expected=0.0`) — selectivity, not a weaker bar. With the cross-edge lesioned
  (`BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL_LESION=1`) on the identical violating sequence (`B_lesion`), the gate
  opened 0 of 8 times and weight growth is exactly 0.0 — the effect VANISHES. `n_hollow=0` (visible AND
  lesion-vanishes together), and the growth is 100% attributable to the manipulation against both the lesion
  control and the expected-arm control (`frac_attributable_vs_lesion=1.0`, `frac_attributable_vs_expected=1.0`).
- **ARM C (no-regression):** `onebrain_regression_battery.run_regression_battery` over the flag ON vs. OFF —
  0/38 registered default-ON faculties regressed (34 exercised, decision-identical between the two arms; 4
  honestly `not-exercised` by this probe set — a documented battery limitation, not a C1 regression).
  `all_pass=True`.

`FLIP_VERIFY_GO = True`. Artifact:
`research/findings/raw/_crossedge_surprise_worldmodel_flip_verify/numpy_run.json` (top-level `preconditions`
list — six `Verdict.require(...)` checks, all `ok: true` — plus `aggregate` carrying the per-arm pass/fail and
`per_worker` carrying every real turn's answer + worldmodel/crossedge fields). Full per-config artifacts
(`w_A_baseline.json`, `w_A_off.json`, `w_B_on.json`, `w_B_on_expected.json`, `w_B_lesion.json`,
`arm_on_BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL.json`, `arm_off_BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL.json`,
`battery_BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL.json`) sit alongside it in the same directory, each with a
`.prov.json` sidecar.

## What this closes vs. the prior PARTIAL

The prior version of this finding (offline self-test only) explicitly named the end-to-end real-`brain_chat`
ARM A/B/C as the remaining rung before flip-ready — that run had been interrupted mid-flight. It turned out the
underlying subprocess orchestration survived the interruption (reparented to `systemd --user` after its parent
session ended) and kept running to completion in the background; this landing reads that completed run's
artifacts rather than re-deriving them. No wiring changed since the offline self-test — same
`research/runners/onebrain_xedge_surprise_worldmodel_production.py`, same `webapp/server.py` call site
(~line 5140-5180).

## What landed (unchanged from the prior finding, restated for continuity)

- **Production organ** `research/runners/onebrain_xedge_surprise_worldmodel_production.py` — the C1 edge as a
  reusable production module (surprise-gated Hebbian window on the E2 state→pred transition).
- **The call site in `webapp/server.py`** reads `xedge_surprise_worldmodel_enabled()` and, when set, applies
  `crossedge_gated_update` after the world-model's own violation read — see the module docstring for the
  ordering note (the lesion must apply before `read_surprise`). Unset/`0`/false/no/off leaves this block a no-op
  and the organ byte-identical to its pre-C1 behaviour, now confirmed by ARM A above, not merely by reading the
  code.
- **Flip-verify harness** `research/runners/_crossedge_surprise_worldmodel_flip_verify.py` — 6/6 numpy self-test
  GO (module-internal) plus, as of today, the real-handler run reported here.

## Offline self-test: ARM B (vary/lesion) is load-bearing per-seed (unchanged, restated)

Artifacts: `research/findings/raw/_onebrain_xedge_surprise_worldmodel_production_arms/arm_s44_surprising.json`,
`arm_s44_expected.json`, `arm_s44_surprising_lesion.json`. On seed 44:
- **expected** (no surprise): edge weight flat (1.422), `n_gated=0`, no growth — nothing to learn.
- **surprising** (surprise present): edge GROWS (w 0.0986 → 0.3296, `n_gated=2`, `w_grew=0.231` <!--derived-->) and
  the world-model prediction margin SHIFTS (396.5 → 279.5) — the surprise genuinely updates the forward model.
- **surprising + lesion**: with the edge lesioned, weight stays flat (0.0986) and the margin returns to 396.5 —
  the effect VANISHES on lesion.

## Status / next

**flip-ready**: end-to-end ARM A/B/C GO through the real `/api/brain-chat` handler, on top of the 6/6-seed
offline self-test and the 6-seed de-risk. The cross-edge stays `BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL`
default-OFF — this finding earns the flip's readiness, it does not spend it: whether to flip
`_c1_on`'s default to `True` in production is a separate, later, owner-gated decision (mirroring how
`onebrain-xedge-d6-comprehension` and `onebrain-xedge-curiosity-d6` were each flipped in their own dedicated
landing, with a `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` row added at THAT time — not this one, since C1 is not
being flipped today). If/when that flip lands: add the ledger row, and retire any host forward-model-update path
this cross-edge would make redundant (scaffold_retired).
