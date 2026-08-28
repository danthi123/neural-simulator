---
type: finding
status: negative
date: 2026-08-28
verdict: The one-brain d6-WM→comprehension cross-edge production-default FLIP is NO-GO — through the REAL /api/brain-chat handler on the production cupy brain, the cross-edge's content-change (repair_role → clarification wording) is NOT visible on any seed (n_visible_grown_focus 0/4, shipped seed 42 not visible), even though it is load-bearing at the ORGAN level (PART 1/2/3 GO). This is the #94-class hollow-on-real-traffic result: the anti-hollow bar caught it. BRAIN_ONEBRAIN_XEDGE stays DEFAULT-OFF. The edge is NOT retracted as a faculty — only the production-default flip is NO-GO, pending a next mechanism that routes the drive across the decision boundary in the full pipeline.
mechanism: verify-then-flip of BRAIN_ONEBRAIN_XEDGE (d6-WM→comprehension cross-edge) — 6-seed cupy through the real handler
lane: onebrain-integration-xedge-flip
artifacts:
  - research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed.json
runner: research/runners/_xedge_flip_production_verify.py
---

> ⚠️ **CORRECTION 2026-08-28 (SUPERSEDED): this NO-GO was an INSTRUMENT ARTIFACT, not a hollow flip.** The verify ran on cupy, where the shared xedge pool BUILD CRASHED silently — `_snapshot_rest` stored the shared-path rest as HOST numpy, `_hard_reset` assigned it into DEVICE cupy arrays → `ValueError: non-scalar numpy.ndarray cannot be used for fill` at `comprehension_production_organ.py:419` — degrading every seed to STANDALONE organs (`pool.primed=false` in the artifact, never checked). So `n_visible_grown_focus=0` measured a CRASH, not hollowness.
> The bug is INVISIBLE on numpy (both sides host), so the numpy-only disambiguation missed it — a cross-backend silent-failure. FIXED backend-correctly (`xp.asarray`, merged `a8c8a2d18`); cupy seed42 now: build SUCCESS, delta 0.0193 vs eps 0.0040, **resolved 144/144, role_differs=True**. A 6-seed cupy re-verify is bumped to the gpu.queue front → autonomous flip if non-hollow 6/6 GO. **The NO-GO verdict below is VOID.**

# One-brain d6-WM→comprehension cross-edge — production-default flip is NO-GO (hollow on real traffic; edge is load-bearing only at the organ level)

Artifact: `research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=cupy`, `b_edge=learn` = both flags on with the in-brain self-supervised edge at its converged weight, run through the REAL `webapp.server.brain_chat` handler on the production tiny-demo+LTM brain).

## The decision (owner-authorized autonomous flip — it did NOT check out)

The owner authorized an autonomous production-default flip of `BRAIN_ONEBRAIN_XEDGE` "if everything checks out". The verify-then-flip harness (`_xedge_flip_production_verify.py`, built + committed on `research/onebrain-xedge-flip-verify-v2`) ran the full gate: ARM-A byte-identical-off · ARM-B load-bearing + VISIBLE-on-real-traffic + lesion-attributable (6-seed) · ARM-C no-regression. **`FLIP_VERIFY_GO = False` → the flip is NOT applied; `BRAIN_ONEBRAIN_XEDGE` + `_LEARN` stay DEFAULT-OFF.**

## Result — the content-change is not visible through the full production pipeline

Aggregate (`worker_problems = []` — clean run, not a crash):

- **`n_visible_grown_focus = 0`** of 4 — on the GROWN-focus seeds (where the through-handler positional focus `CAND_POOLS[0]=w0` maps to a grown role, so the edge SHOULD transmit and change the clarification wording), the `repair_role` with-a-held-referent vs no-held-referent did NOT differ. `role_differs = False` on every grown seed.
- **`shipped_seed42_visible = False`** — the shipped seed shows no visible change.
- `n_correctly_inert_ctrl_focus = 2` of 2 — the 2 control-focus seeds are correctly inert (as designed).
- `n_hollow = 0` — the runner does NOT claim any hollow visible flip (it honestly reports 0 visible rather than a false positive).
- `all_seeds_lesion_revert = True`; byte-identical-off holds (`n_match = 4/4`).

## What this settles — the anti-hollow bar worked

The cross-edge is GO at the ORGAN level (PART 1: drives the judge() margin ~98.5% attributable; PART 2: flips the repair role agent↔patient 5/5 on ambiguous items; PART 3: grows per-turn). But through the FULL real `/api/brain-chat` pipeline (comprehension organ → composer → renderer), that organ-level role-drive does NOT surface as a visible difference in the rendered clarification on real traffic — exactly the #94 confidence-forthcomingness failure class (load-bearing in a harness, invisible on real traffic). Flipping default-ON would have been a hollow flip. The verify-then-flip discipline + the anti-hollow bar correctly prevented it.

## NOT a retraction — the next mechanism (NO-DEFER)

Only the PRODUCTION-DEFAULT FLIP is NO-GO. The one-brain cross-edge remains a valid default-OFF faculty (PART 1/2/3 GO stands at the organ level). The residual to close before the next flip attempt has TWO candidate causes to disambiguate first:

1. **Sub-decision drive doesn't cross the decision boundary in the full pipeline.** PART 1 named a "sub-decision" caveat (the edge drives the margin but doesn't cross the decision boundary); PART 2 claimed to close it at the organ level. This result suggests it re-opens through the FULL handler — the organ's repair_role decision is downstream re-derived by the composer/renderer in a way the edge's margin nudge doesn't reach. Next lever: route the cross-edge drive through the SAME decision the renderer reads (make the edge move the `content_role`/`net_lean` the handler actually consumes, not just the organ's internal margin).
2. **Converged-edge loading nuance** (`learn_wiring_live = False` in the aggregate): confirm the converged edge weight is actually loaded + driving in the handler path (vs a reporting artifact of `set_live_per_turn(False)`). A quick instrumented single-seed handler run (print the loaded edge weight + the pre/post-edge read margin) disambiguates (1) vs (2).

Do (2) first (cheap), then (1) if the edge is confirmed active. Only re-attempt the flip once a grown seed shows `role_differs = True` through the real handler.
