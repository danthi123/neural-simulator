---
type: finding
status: mixed
date: 2026-08-28
verdict: A single-seed (42), NUMPY-backend, instrumented run through the REAL /api/brain-chat handler (same
  b_edge=learn / set_live_per_turn(False) converged-edge config the 6-seed CUPY flip-verify harness used) does
  NOT reproduce either candidate cause named in the 2026-08-28 NO-GO finding. (B) is REFUTED directly — the
  cross-edge weight loaded in the handler path is fully converged (w0->A=16.78, not the ~0.05 baseline). (A) is
  also NOT what this run shows — the converged edge DOES shift the organ's internal read (lesion-attributable)
  AND the shift DOES reach the rendered repair_role through the real handler (content-only 'patient' -> held
  'agent'), contradicting the "drives the organ but not the render" story too. The likely reconciling cause,
  given the code's OWN documented cross-backend-seed trap, is BACKEND-DEPENDENT MARGIN FRAGILITY — the
  wm-resolved delta that must clear `_wm_resolve_eps` is comfortably cleared on this numpy run but apparently
  fails to clear on cupy for most seeds (the original 6-seed result). This is a THIRD, more precise candidate the
  original (A)/(B) framing did not anticipate. NOT a reversal of the NO-GO (still single-seed, single-backend,
  the opposite backend from the original harness) — a disambiguation lead for the next cheap check.
mechanism: instrumented single-seed probe of the d6-WM->comprehension cross-edge through webapp.server.brain_chat,
  b_edge=learn (BRAIN_ONEBRAIN_XEDGE=1 + _LEARN=1, set_live_per_turn(False)), SIM_BACKEND=numpy
lane: onebrain-integration-xedge-flip
artifacts:
  - research/findings/raw/_xedge_flip_disambiguate/seed42.json
runner: research/runners/_xedge_flip_disambiguate.py
---

# xedge flip NO-GO disambiguation — weight IS loaded, edge DOES drive the rendered output (numpy, seed 42); the residual looks like backend-dependent margin fragility, not (A) or (B) as stated

Artifact: `research/findings/raw/_xedge_flip_disambiguate/seed42.json` (seed 42, `SIM_BACKEND=numpy`, the SAME
`b_edge=learn` converged-edge config — `BRAIN_ONEBRAIN_XEDGE=1` + `_LEARN=1`, `set_live_per_turn(False)` — that
`_xedge_flip_production_verify.py`'s ARM B used on cupy across 6 seeds). Runner:
`research/runners/_xedge_flip_disambiguate.py`.

## Why this diagnostic

The 2026-08-28 NO-GO finding
(`2026-08-28-onebrain-xedge-production-default-flip-NO-GO.md`) found `n_visible_grown_focus=0/4` through the real
handler on 6 cupy seeds, with two unresolved candidate causes: **(B)** the converged edge isn't actually loaded/
driving in the handler path (a `learn_wiring_live=False` reporting artifact), or **(A)** the edge drives the
comprehension organ's internal margin but the drive doesn't cross the decision the renderer reads. This diagnostic
instruments a single seed (42) through the real handler to read the three numbers that disambiguate them.

## The three instrumented numbers (seed 42, numpy, b_edge=learn)

**(i) the ACTUAL loaded cross-edge weight in the handler path** — read directly off
`onebrain_xedge_production.get_xedge_pool(42).cross_weights` after the SAME priming sequence the flip-verify
worker runs before importing `webapp.server`:

```
w0->A=16.7788  w0->P=4.3919  w1->A=0.05  w1->P=0.05  w2->A=5.676  w2->P=16.7183
```

Seed 42's `w0` is the grown `p_agent` candidate (matches `_w0_role(42)` in the flip-verify harness). `w0->A =
16.78` is the CONVERGED magnitude (the same build-curriculum ceiling PART 2's own self-test reports), not the
`W0=0.05` ungrown baseline — **(B) is directly refuted**: the weight the handler actually reads is fully grown.

**(ii) the comprehension organ's own read, edge-ON vs edge-lesioned** (direct call to the process-shared
`ComprehensionProductionOrgan.repair_target()` with an explicit `wm_focus='w0'`, matching the region the real
handler's `d6org.current_focus()` resolves to — see (iii)):

| probe | wm_focus | role | content_role | net_lean | wm_resolved | wm_margin |
|---|---|---|---|---|---|---|
| content-only | None | patient | (n/a — content read only) | 0.3888888888888889 | — | — |
| held, edge intact | 'w0' | **agent** | patient | 0.5229166666666667 | **True** | 0.020000000000000018 |
| held, edge lesioned | 'w0' | patient | (n/a) | 0.5256944444444445 | **None** | — |

Edge intact: the held read's `wm_resolved=True` OVERRIDES the content-only role (patient) with the WM-taught role
(agent) — the cross-edge genuinely shifts the organ's internal decision. Lesioning the cross-edge (`pool.
lesion_cross()`, zeroing every `w{k}->sel` synapse) reverts the held read to the content-only role (patient,
`wm_resolved=None`) — the shift is lesion-attributable, confirming the mechanism is real and not some other
co-driven artifact.

**(iii) the resulting repair_role through the REAL `/api/brain-chat` handler**, held vs no-held, on a freshly
re-primed (non-lesioned) pool:

```
d6org.current_focus() after HOLD_TURN (session diag_hd) = 'w0'         (matches (ii)'s explicit probe region)

novisi (fresh session, no hold): repair_role='patient'  answer="...didn't resolve the PATIENT..."
held   (same session, HOLD_TURN then the item):  repair_role='agent'  answer="...didn't resolve the AGENT..."

role_differs = True     answer_differs = True
```

This is the flip-verify harness's own ARM-B visibility criterion (`role_differs`, `answer_differs`), reproduced
directly through `webapp.server.brain_chat` — and on this run it is **visible**: content-only reads "patient",
held reads "agent", exactly the WM-taught tiebreak PART 2/3's organ-level self-tests describe.

## What this rules in/out

- **(B) "edge not loaded" is REFUTED.** The weight in the handler path is the converged ~16.8 magnitude, not the
  0.05 baseline, on both the first priming and a fresh re-prime for the handler test (`iii_reprimed_cross_weights`
  is byte-identical to `i_loaded_cross_weights` — numpy build is deterministic at this seed).
- **(A) "drives the organ but not the render" is NOT what this run shows.** The organ-level shift (ii) DOES reach
  the rendered `repair.role` the harness's own JSON exposes (iii) — `role_differs=True`, `answer_differs=True`.
  Reading `comprehension_production_organ.repair_target()` confirms why: `wm_role` (from `_wm_resolved_role`)
  directly overwrites `base["role"]` when resolved, and `webapp/server.py`'s repair block does
  `repair_info = dict(tgt)` with no further re-derivation before it reaches the JSON `repair.role` field that the
  flip-verify harness reads — there is no separate composer/renderer step between the organ's own decision and
  the field ARM B checks. So a genuine "organ decides but the renderer re-derives it" bug is not evidenced here.

## The residual: BACKEND-DEPENDENT MARGIN FRAGILITY (a third candidate, not A or B as stated)

This numpy/seed-42 run visibly contradicts the original 6-seed CUPY harness's `n_visible_grown_focus=0/4`. The
one uncontrolled variable between them is the backend, and this exact module already documents that backend
changes the SUBSTRATE, not just the runtime: `onebrain_xedge_production.py`'s own docstring calls out the
**CROSS-BACKEND SEED TRAP** — "a numpy-grown weight file is NOT valid for a cupy production build (different RNG
-> different substrate)" — which is why the edge is grown IN-PROCESS on whichever backend will read it, rather
than loaded from a saved artifact. The held read's `wm_margin=0.02` in this run is a THIN number in absolute
terms (though the resolution logic compares it to <!--derived--> `_wm_resolve_eps = max(0.004, 3*|baseline|)`
(the hardcoded floor read from `comprehension_production_organ.py`, not a measured artifact value), not to zero,
so whether it counts as "thin" depends on the baseline this run didn't print). The most parsimonious account
consistent with both results: the SAME mechanism (converged edge, lesion-attributable organ shift) is real on
both backends, but cupy's different neuron realization at "seed 42" pushes the WM-resolved delta below
`_wm_resolve_eps` for most seeds, so `wm_resolved` stays `None` and the content-only role stands — exactly the
0/4 pattern the NO-GO finding measured. This is a MARGIN-ROBUSTNESS problem in the resolution instrument, not a
structural wiring/routing bug — a variant neither (A) nor (B) as literally stated describes.

**This is NOT a reversal of the NO-GO verdict.** The NO-GO's own instrument (6-seed cupy, the actual shipped
backend) is the decisive one; this diagnostic is single-seed numpy, run specifically to keep it CPU/RAM-light per
the tight-bounded-diagnostic scope. `BRAIN_ONEBRAIN_XEDGE`/`_LEARN` stay default-OFF.

## Concrete next lever

Re-run this SAME instrumented script (`research/runners/_xedge_flip_disambiguate.py`) under `SIM_BACKEND=cupy` at
seed 42 — cheap, single-seed, mirrors this run exactly — to confirm or refute the backend-margin-fragility
hypothesis directly: if the cupy run shows the converged weight (i) present but `role_differs=False` at the
handler (iii) with a `wm_resolved` that stays `None` or a delta close to `_wm_resolve_eps` in (ii), that CONFIRMS
the margin is backend-fragile rather than absent, and the fix is to STRENGTHEN the resolution signal (e.g.
increase `_CODRIVE_PARAMS`'s `load_steps`/`hold_steps` so the WM bump is more forcefully re-established before the
balanced read, or widen the credited-step confidence threshold `conf` used to grow the edge) so the delta clears
`_wm_resolve_eps` robustly across backends — NOT a renderer/decision-routing fix (which (A) would have implied,
and which this run's evidence does not support).
