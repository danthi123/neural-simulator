---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Affect-coloring is integrated-HOLLOW for a PROBE reason, not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Sibling of [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md): the SAME class of defect (a real neural read that is load-bearing in isolation but reads hollow on the integrated harness because the default probe never drives it), in the same instrument, resolved the same way — a default-off driving-probe remap, byte-identical when off, no edit to the shared regression battery.
Follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md), which recorded affect-coloring among the hollow faculties: its `BRAIN_AFFECT_LESION` neural cut collapses the ladder differential in isolation, yet lesioning it does not change the reply on the default probe turn.
This finding pins WHY and lands a minimal, honest, default-off instrument fix.

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the affect wiring. Each fact read directly from the code:

1. **The reply IS already colored by the ladder read.** `webapp/server.py` (~L4749-4787) computes the whole `affect_info` block near the TOP of the handler, before any turn-type routing: `valence_sign`, `tone_token`, the `forthcomingness` content plan, the `manner_template`, and `chat._mood_tone_level` ALL flow from `read = organ.read_differential(mood["valence"], lesion=lesion)`. So the neural differential genuinely composes the reply's manner + forthcomingness; it is not computed-then-ignored. That the mechanism drives the reply once given a nonzero mood is independently shown by the existing REAL/SHAM adversarial check (`research/runners/_stageA_full_integration_derisk.py` ~L1972-1982: the tone token goes flat when `affect_out=0` with a nonzero appraisal).

2. **The default probe turn is mood-NEUTRAL, so there is nothing to color.** The affect-coloring probe (`research/runners/onebrain_regression_battery.py` `FACULTY_PROBES`, key `affect-coloring`) is the `well` turn "the wolf bites the apple". `appraise_text` (`research/runners/affect_production_organ.py` ~L104-127) gates mood-moving words on the Warriner salience margin `_STRONG_MARGIN` (~L73/118): "wolf" sits inside the margin (|3.9-5| under the threshold) and "bites"/"apple"/"the" are not in WARRINER at all, so `appraise_text("the wolf bites the apple")` returns n_hits=0. `_update_session_mood` (`webapp/server.py` ~L3505-3514) then HOLDS the prior mood (default 0.0) whenever n_hits==0, and feeds that 0.0 into `read_differential`.

3. **At a 0.0 appraisal the lesion has nothing to bite.** Inside `read_differential`, a 0.0 appraisal is injected through settle/ramp/drive-off/read REGARDLESS of the `lesion` flag — the `affect_out` transmission gate only matters when there is a nonzero differential to clamp. On the default interoceptive-afferent read path (production-default-ON), `lesion=True` clamps `affect_out=0` to give an exactly-0.0 differential (the path's own selftest asserts this), and a 0.0-appraisal intact read is likewise ~baseline. So both the intact and `BRAIN_AFFECT_LESION` arms map to `valence_sign="0"` / `tone_token=""` (`webapp/server.py` ~L4764-4767) — IDENTICAL, and `compare()` verdict `pass` -> NOT load-bearing.

Net: affect-coloring is hollow on this instrument because the probe never gives the ladder a nonzero mood to color. This is the exact defect the file already documents for two siblings: the battery's own note explains why affect-marker-spiking-wta was moved off `well` onto `emo` ("`well`'s mood stays neutral, level 0 ... the field could never discriminate"), and `load_bearing_fraction.py`'s note on affect-drives-response says "the `well` turn is mood-neutral (level 0) intact too, so the acted decision may be unchanged there." affect-coloring reads the SAME Gate-B ladder mechanism but was never given the same remap.

**External grounding (lesion methodology).** The remap is the standard lesion-methodology principle, not an instrument hack: a (virtual) lesion demonstrates a region is NECESSARY only when the manipulation interferes with performance on a task that actually engages the region — a lesion has nothing to reveal on a condition that does not exercise the circuit.
Sliwinska, Vitello & Devlin (2014, JoVE), "Transcranial magnetic stimulation for investigating causal brain-behavioral relationships and their time course" [DOI](https://doi.org/10.3791/51735) (PubMed): "Stimulation that interferes with task performance indicates that the affected brain region is necessary to perform the task normally," and stresses that appropriate experimental/control conditions are what make the causal read valid. <!--derived-->
The mood-neutral `well` turn is the wrong condition for the affect lesion (nothing to interfere with); the mood-engaging `emo` turn is the right one.

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_AFFECT_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the affect-coloring measurement is remapped to the strongly-affective `emo` turn:

- The `emo` turn ("Wonderful! I am so happy and delighted, this is fantastic and amazing!") is ALREADY a self-contained turn in the default `PROBE_TURNS` roster (session `emo`, reset=True, its own single-turn group). So — UNLIKE the episodic fix, which had to add a store->recall pair to `_EXTRA_TURNS` — NO turn is added and the shared regression battery is not touched at all (byte-identical). `turn_group("emo") == ["emo"]`.
- No env-forcing is needed (`base_env` stays `{}`): the Gate-B ladder read runs on numpy for any turn, unlike episodic's cupy-gated BTSP write. Every other faculty keeps its baseline row and `base_env={}` -> byte-identical.
- `measure_faculty` remaps affect-coloring to `("affect-coloring", "emo", ["affect.on", "affect.valence_sign", "affect.tone_token"], False)` only when the flag is set. The lesion is the existing `BRAIN_AFFECT_LESION` neural cut of the `affect_out` readout gate.
- Expected: on `emo`, `appraise_text` hits n_hits=5 strongly-positive words -> the session mood goes non-neutral -> intact reads `valence_sign="+"` (a nonzero positive differential) while the lesion clamps `affect_out=0` -> a 0.0 differential -> `valence_sign="0"` -> `affect.valence_sign` + `affect.tone_token` FLIP -> LOAD-BEARING, with `affect.on` present in both arms (the structural anchor) and a clean null control.

Brain-based-only: the coloring is the genuinely-spiking Koulakov graded-affect ladder read through the `affect_out` transmission gate; host does only the world (the turn text) and the clock. Honesty boundary preserved: `valence_sign` / `tone_token` are FUNCTIONAL read-outs of the neural mood differential, never phenomenal claims; affect colors only MANNER + forthcomingness of an already-moat-verified answer, never a fact or a certainty band.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/affect_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` passes, including three new checks — the `emo` turn is in the default roster, `turn_group("emo") == ["emo"]` (a lone self-contained turn, no dependency chain, no env-forcing), and the affect-coloring lesion flag resolves in source.
The artifact also records the default roster is unchanged (`n_probe_turns_default_roster` = 26, same as the episodic baseline) and the affect-driving turn/group.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (owner gaming window; no local full-brain smokes per the standing rule). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest).
The measured flip + the no-regression of the other faculties are the controller's AWS/local step; the exact `--out` path lives in the runner's docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation).
No-regression is guaranteed by construction when the flag is off (default): the affect remap guard `LB_AFFECT_DRIVE and key == "affect-coloring"` is False, affect-coloring falls back to its baseline `well` row, and no other faculty's env or row is touched (confirmed at import: with the flag unset the row resolves to `well`; with it set only affect-coloring remaps to `emo`, curiosity/episodic/etc. unchanged).

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_AFFECT_DRIVE_PROBE` flag + diagnosis comment, the `measure_faculty` affect remap (no `base_env` forcing), the docstring verify command, 3 new selftest checks, and the selftest-artifact keys (`affect_drive_turn`/`affect_drive_group`/`affect_drive_in_default_roster`).
- `research/runners/onebrain_regression_battery.py` — UNCHANGED (byte-identical): `emo` was already in `PROBE_TURNS`, so no `_EXTRA_TURNS` addition was needed, unlike the episodic fix.
