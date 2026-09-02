---
type: finding
status: go
date: 2026-09-01
mechanism: affective-marker-lateral-inhibition-wta
lane: affect
integration_faculty: affect-marker-spiking-wta
runner: research/runners/_affect_marker_wta_verify.py
artifacts:
  - research/findings/raw/_affect_marker_wta_verify.json
  - research/findings/raw/_affect_marker_wta_verify_postflip_2026-09-01.json
---

# Board #86 (affect-marker spiking WTA) FLIPPED default-ON — auto-flip policy, 6-seed re-verify GO

## Why this doc exists

`2026-08-28-affect-marker-spiking-wta-derisk.md` landed the spiking lateral-inhibition WTA that SELECTS the
#84 affective expression marker (which of "Wonderful"/"Gladly"/"Sure"/"Hm"/"Honestly"/"Frankly" leads the
reply) as an additive, default-OFF de-risk, 6-seed GO on all five measured properties, explicitly deferring
the default-ON decision to "owner review of the affect-path default". The 2026-09-01 auto-flip policy
(`GAP_CLOSURE_MISSION.md`: validated-GO + load-bearing + moat-safe + byte-identical-off + no-regression ->
default-ON, remove owner-gating) applies directly to that deferred decision. This doc: (1) re-verifies the
guard fresh against the CURRENT code (not trusting the 2026-08-28 finding's claim), (2) flips the default,
(3) re-verifies AGAIN post-flip against the new byte-identical-OFF semantics the flip itself introduces.

## Pre-flip reproduction (code default confirmed OFF, guard confirmed GO)

Before touching any code, `webapp/affect_drives_chat.py:152` read
`os.environ.get("BRAIN_AFFECT_MARKER_SPIKING", "0")` — confirmed default OFF, matching the 2026-08-28 finding.
Re-running the existing, UNMODIFIED `research/runners/_affect_marker_wta_verify.py` at the same 6 seeds
{42,43,44,100,101,102} reproduced `status: GO` on all five checks — byte-identical-off (flag unset), load-bearing
(mood sweep selects the matching register, 36/36 rows), lesion collapses to the honest no-marker fallback
(36/36 rows), shuffle anti-cheat (30/36 rows differ from intact), and attribution
(100.0% of the winner-vs-runner-up separation rides the felt-state->assembly drive, 0.0% in the lesioned
control, every seed) — matching `research/findings/raw/_affect_marker_wta_verify.json` (the already-committed
2026-08-28 artifact) number-for-number.

## The flip

`webapp/affect_drives_chat.py`: added `_AFFECT_MARKER_SPIKING_DEFAULT_ON = True` and a
`marker_selection_spiking_off()` helper (explicit-off check: `BRAIN_AFFECT_MARKER_SPIKING` in
`{0,false,no,off,''}`), mirroring the SAME idiom this codebase already uses for its other default-ON anchors
(`_CG_DRIVES_DEFAULT_ON`/`cg_drives_off()` in `webapp/common_ground_drives_chat.py`,
`_ELABORATE_FROM_LTM_DEFAULT_ON` in `research/runners/rich_answer_composer.py`).
`marker_selection_spiking_enabled()` now returns `not marker_selection_spiking_off()` when the anchor is True
— i.e. ON unless the caller explicitly opts out. The two production call sites
(`AffectDrivesWorkspace.observe()` / `.relax_idle()`, `webapp/affect_drives_chat.py:356,401`) already pass
`mood`/`felt_arousal` on every call, so the flip is immediately live on the production `/api/brain-chat`
affect coupling (#84), not a dead flag.

## Post-flip re-verify: the meaning of "byte-identical-off" moves with the flip

Flipping the default changes what "OFF" means: pre-flip, "off" was the unset env var; post-flip, "off" is the
EXPLICIT escape `BRAIN_AFFECT_MARKER_SPIKING=0` (unset now means ON). `_affect_marker_wta_verify.py`'s part
(A) was updated to test the explicit escape instead of unset (mirroring the same escape-based convention
already used by `cg_drives_off()`), then the full 6-seed guard was re-run against the flipped code. Result,
`research/findings/raw/_affect_marker_wta_verify_postflip_2026-09-01.json`:

- **(A) byte-identical-off (explicit escape)**: PASS, 12/12 function rows + 6/6 workspace rows —
  `BRAIN_AFFECT_MARKER_SPIKING=0` reproduces the exact pre-existing `_LEAD_WORD[level]` host-template surface.
- **(B) load-bearing**: PASS, 6/6 seeds.
- **(C) lesion-vanish**: PASS, 36/36 rows.
- **(D) shuffle anti-cheat**: PASS, 30/36 rows differ from intact.
- **(E) attribution**: PASS, min=1.000 max=1.000 (100% of the separation attributable to the felt-state->assembly
  drive, every seed).
- Overall: `=> GO`.

A direct manual check confirms the flip is real, not merely escape-compatible: with the env var fully unset,
`marker_selection_spiking_enabled()` returns `True` and `expression_lead(3, True, mood=0.085,
felt_arousal=0.15, seed=42)` returns `'Wonderful! '` — IDENTICAL to the explicit `BRAIN_AFFECT_MARKER_SPIKING=1`
call, and (at this calibrated register midpoint) also identical to the explicit-off/host-table call, since the
circuit is tuned to reproduce the host's own word at these midpoints (measured in part B).

## Verdict

**GO — FLIPPED.** `_AFFECT_MARKER_SPIKING_DEFAULT_ON = True` in `webapp/affect_drives_chat.py`. All five
guard properties hold 6/6 seeds both before AND after the flip (against the correspondingly-updated
byte-identical-off condition). `BRAIN_AFFECT_MARKER_SPIKING=0` remains the byte-identical escape back to the
pre-existing host `_LEAD_WORD[level]` scaffold.

## Honest residuals (carried over from the 2026-08-28 finding, unchanged by this flip)

1. The `level`/`high_arousal` binning upstream (`mood_to_level`, `_AROUSAL_HIGH`) is still host.
2. Near a register boundary the spiking circuit is honestly LESS decisive than the old hard host threshold
   (reports "no clean winner" -> no lead, where the old binning picked a level crisply) — a genuine behavioral
   difference at (and only at) boundaries, now live on the default turn.
3. The emphasis (arousal) WTA remains a second, separate 2-pool circuit.
