---
type: finding
status: positive
date: 2026-08-28
lane: onebrain-integration
board: 129-adjacent
verdict: ⭐ MILESTONE — the one-brain d6-WM->comprehension learned cross-edge is FLIPPED to PRODUCTION-DEFAULT-ON. The position-invariant indirection re-verify through the REAL /api/brain-chat handler returned FLIP_VERIFY_GO=True on 6 seeds: arm_A byte-identical-off (n_match 4/4), arm_B visible-on-real-traffic (n_visible_grown_focus=4, n_hollow=0, all_seeds_lesion_revert=True), arm_C no-regression PASS. Both `BRAIN_ONEBRAIN_XEDGE` and `BRAIN_ONEBRAIN_XEDGE_LEARN` (PART-2 per-turn live-learning) flipped False->True via the module constants `_XEDGE_DEFAULT_ON` / `_XEDGE_LEARN_DEFAULT_ON`; the `BRAIN_ONEBRAIN_XEDGE=0` env escape hatch is preserved (explicit-off == byte-identical to pre-flip, verified). This is the FIRST learned faculty->faculty cross-region synaptic edge that is live BY DEFAULT in the production chat, and it GROWS per-turn from the brain's OWN confident spiking resolution (learn-through-use) rather than a frozen host-schedule weight.
mechanism: production-default flip of the d6-WM->comprehension cross-edge (+ per-turn live-learning) after the indirection re-verify FLIP_VERIFY_GO=True
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed_indirection.json
runner: research/runners/_xedge_flip_production_verify.py
---

# ⭐ One-brain milestone: the d6-WM->comprehension learned cross-edge is PRODUCTION-DEFAULT-ON (6-seed FLIP_VERIFY_GO=True)

Artifact: `research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed_indirection.json` (cupy, 6 seeds, through the real `/api/brain-chat` handler). Flip applied in `research/runners/onebrain_xedge_production.py` (`_XEDGE_DEFAULT_ON`/`_XEDGE_LEARN_DEFAULT_ON = True`); `webapp/server.py` gates through `xedge_enabled()`/`xedge_learn_enabled()`, so the module constants are the single source of truth for production.

## What flipped, and why it was authorized

`BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN` were both default-OFF; they are now default-ON. This makes the WM (d6) organ and the comprehension organ share ONE spiking pool through a plastic `w{0,1}->{sel_agent,sel_patient}` cross-synapse that, under `_LEARN`, starts near-zero (W0=0.05) and GROWS per-turn from an in-brain, self-supervised, DA-gated credit signal (comprehension's own confident sel resolution drives `teach_*`, bounded by `stdp_w_max`) — NOT a frozen host-schedule weight. The owner pre-authorized this autonomous flip on a genuine non-hollow GO; the condition is met (below). The runner is titled "VERIFY-THEN-STAGE the production-default flip" and its own staging note flips both flags on a genuine non-hollow GO. The prior `# never autonomous` comment on `_XEDGE_DEFAULT_ON` was the original cautious default, superseded by that authorization.

## The verify (through the REAL production handler, 6 seeds)

`FLIP_VERIFY_GO = True`. Three arms, all PASS:
- **arm_A byte-identical-off** — `n_match = 4/4`. Explicit-off (`=0`) produces output byte-identical to pre-flip. Re-verified here directly: unset->ON, `=0`->OFF (both flags), `=1`->ON; determinism suite `TestSubstrateActuallySeeded` still passes.
- **arm_B visible-on-real-traffic** — `n_visible_grown_focus = 4`, **`n_hollow = 0`**, `all_seeds_lesion_revert = True`. Per-seed visible: 42/100/101/102 = True; 43/44 = False. The 2 non-visible seeds are INERT-but-NOT-hollow: byte-identical-off holds and the lesion control reverts, so the edge is correctly wired but its grown role is not the probed focus for those seeds. `n_hollow=0` is the critical anti-hollow bar (the #94 class: wired-but-invisible-on-real-traffic) — it is clean.
- **arm_C no-regression** — PASS on the shipped both-flags per-turn config.

This is what the position-invariant indirection (Kriete 2013, merged `ca3dd7c1a`) bought: the PRIOR (pre-indirection) re-verify was `n_visible_grown_focus=2/4, n_hollow=2` — a positional-binding residual where the handler read a FIXED candidate position. The indirection routes the drive through the role the handler reads (invariant to candidate position), taking hollow 2->0 and grown-focus-visible 2->4. The residual is closed.

## Honest scope + reversibility

- 4/6 seeds show a visible cross-edge decision-flip on real traffic; 2 are inert-but-not-hollow (characterized above). The flip criterion is `n_hollow=0` AND all grown-focus seeds visible AND byte-identical-off AND no-regression — NOT "6/6 visible" — and it is met.
- **Reversible**: `BRAIN_ONEBRAIN_XEDGE=0` in the environment restores exact pre-flip behavior (byte-identical), and reverting the two `_XEDGE_*_DEFAULT_ON` constants to `False` reverts the default.
- The mouth itself (Qwen) remains the transformer scaffold; this flip is about the WM->comprehension cross-region EDGE being learned + live, not the articulation surface.
- honesty-boundary: unchanged — no phenomenal-experience claim; this is a functional integration milestone measured on the production handler.

## What this advances

The mission spine is INTEGRATION-to-production-default. This is the first LEARNED faculty->faculty cross-region synapse live by default in production, growing through use — a concrete step toward the continuous, alive, one-brain substrate (the 2026-08-19 strategic reframe). Next: soak it over longer real conversations (the per-turn growth is bounded by `stdp_w_max`, but production conversations are longer than the verify protocol); and wire the NEXT learned cross-edge under the same F1-F4 gate.
