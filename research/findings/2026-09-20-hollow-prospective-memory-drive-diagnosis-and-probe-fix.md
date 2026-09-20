---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Prospective-memory is integrated-HOLLOW for a PROBE reason, not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Second follow-on to the load-bearing-fraction baseline, after the episodic fix [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md). Prospective-memory is the SAME shape as episodic was: isolated-lesion-load-bearing but integrated-HOLLOW because the integrated probe exercises only ONE half of a two-phase mechanism, and the half that differs under lesion is never reached. This finding pins WHY (statically) and lands a minimal, honest, default-off instrument fix, mirroring the `LB_EPISODIC_DRIVE_PROBE` pattern exactly.

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the reply composer. Three facts, each read directly from the code:

1. **The faculty genuinely drives the reply — on the CUE turn.** `research/runners/_prospective_memory_production_verify.py` rows A (`A_fire_on_cue`, ~L104-113) and C (`C_lesion`, ~L132-166) are a real intact-vs-lesion behavioral contrast: the intact latch fires on the cue turn (`fired=True`, the answer is prepended with "(Reminder ..."), while `BRAIN_PMEM_LESION=1` collapses the held assembly at formation (`held_after_lesion <= 0.02`) so the SAME cue stays silent (`fired=False`, no reminder). The fire is a spiking held x cue coincidence read off `cp_firing_states`, gated by the HELD intention, not a host string match. So the recall READ genuinely composes the reply; it is not computed-then-ignored.

2. **The integrated probe never reaches the cue turn.** `research/runners/onebrain_regression_battery.py` maps prospective-memory to the single FORMATION turn `pmem_form` and compares field `prospective.held`. But `research/runners/prospective_memory_production_organ.py::form_intention()` sets `out = {"held": True, ...}` — a compile-time Python literal, returned UNCONDITIONALLY regardless of the `lesion` argument. The lesion's real effect is a DIFFERENT field, `held_after_lesion` (computed on the formation turn but never placed in the compared field, and not in `FACULTY_PROBES`' field list anyway). So on the formation turn `prospective.held` is True==True across intact and lesion -> `compare()` verdict `pass` -> hollow. The lesion has nothing to bite on that turn.

3. **The design intent is that the lesion shows up on a LATER turn.** `webapp/server.py:5185-5186` documents it directly in-code: `BRAIN_PMEM_LESION=1` -> the latch is zeroed after formation (the held assembly collapses -> the SAME cue does NOT fire -> NO reminder; load-bearing). The default roster (`PROBE_TURNS`) has no follow-up cue turn after `pmem_form` in that session, so `load_bearing_fraction.py`'s per-faculty measurement compares intact vs lesion on a field that is a compile-time constant on that turn -> zero diff -> reported hollow.

Net: prospective-memory is hollow on this instrument because the probe measures the formation half (a constant field) and never runs the cue half (the field that flips under lesion). This is structurally identical to the episodic precedent: the probe exercises only one half of a two-phase mechanism.

**External grounding** (the two-phase structure is not an implementation artifact — it is the biology). According to PubMed, prospective memory is a genuinely two-phase faculty: an intention is ENCODED/formed, then held across intervening activity and RETRIEVED at its target cue, and the cue-triggered SPONTANEOUS RETRIEVAL is a phase distinct from encoding — implementation-intention (if-then) encoding, in Gollwitzer's sense, specifically enhances that later spontaneous retrieval rather than the formation act itself (Rummel, Einstein & Rampey, 2012, *Memory* 20(8):803-17, [DOI](https://doi.org/10.1080/09658211.2012.707214); reviewed in Gollwitzer & Sheeran, 2025, *Annu Rev Psychol* 76:303-328, [DOI](https://doi.org/10.1146/annurev-psych-021524-110536)). The organ implements exactly this: the cue->action binding is LEARNED one-shot at formation (the Gollwitzer implementation-intention) and the load-bearing behavior is the cue-triggered release. So a probe that only exercises formation is measuring the wrong phase by construction — the phase the literature says carries the retrieval effect is the one the default roster never runs. <!--derived-->

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_PMEM_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the prospective-memory measurement is remapped to a FORMATION->CUE pair in one isolated session:

- Two new turns, `pmem_form2` ("remind me to feed the dog when the bird sings") -> `pmem_cue` ("the bird sings"), session `pmem2`, declared formation-first. They live in `_EXTRA_TURNS` in the battery, merged into `_TURN_BY_LABEL` (so the worker resolves them by label) but deliberately NOT in `PROBE_TURNS` — the default roster stays 26 turns, so the regression battery and every flip-verify harness that iterates it are byte-identical (the selftest artifact records the default roster length unchanged).
- `measure_faculty` remaps prospective-memory to turn `pmem_cue` (group `["pmem_form2","pmem_cue"]`) and compares `prospective.fired`. Unlike the episodic fix, NO `base_env` forcing is needed: `BRAIN_PMEM` and `BRAIN_PMEM_HEBBIAN` are both default-ON, so the ordinary intact build learns the cue->action binding one-shot at formation and fires on the cue turn (`fired=True`); the `BRAIN_PMEM_LESION` arm collapses the latch at formation so the cue stays silent (`fired=False`). Every other faculty keeps `base_env={}` -> byte-identical.
- The cue clause "the bird sings" reduces to `["bird","sings"]` (per `_cue_keywords`), and the cue turn reuses that identical clause, so `cue_present()` matches. The cue is a 2-content-token intransitive, so `extract_transitive()` returns None -> `comprehension.judge()` returns None -> the comprehension-repair early-return (server.py:5809) is skipped; the turn is not referential / an expectation query / a stored contradiction, so no other disjoint short-circuit drops `prospective_info` before it attaches to the reply (server.py:6330 / 6639). Both arms therefore attach `prospective.fired` on the cue turn.
- Expected: `pmem_form2` FORMS + one-shot-Hebbian-binds the intention (latch held); `pmem_cue` reads the spiking held x cue coincidence -> intact `fired=True`, lesion `fired=False` -> field flips -> LOAD-BEARING, null-control clean.

Brain-based-only: the HOLD (attractor persistence) and the coincidence-gated RELEASE are the genuinely-spiking substrate; host does only the world (the turn text), the intention/cue text->slot mapping + cue-presence (a declared language/sensory boundary), and the clock. The lesion is the existing `BRAIN_PMEM_LESION` neural cut (the latch zeroed at formation). Honesty boundary preserved: `fired` is a functional coincidence read-out; the reminder text asserts no phenomenal claim, and a non-fire is an honest silence, never a confabulation.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/pmem_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` reports selftest_result PASS, including three new checks — the driving turns resolve by label, `turn_group("pmem_cue") == ["pmem_form2","pmem_cue"]`, and the lesion knob `BRAIN_PMEM_LESION` resolves in source. The artifact also records the default roster is unchanged and the prospective-driving group.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (verify-statically-only; the controller runs the brain build on AWS/local). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the 25-faculty no-regression are the controller's step (runs on any backend — no forced write, `BRAIN_PMEM`/`BRAIN_PMEM_HEBBIAN` are default-ON so the intact arm fires):

```
LB_PMEM_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only prospective-memory --repeats 2 \
    --out <_load_bearing dir>/pmem_drive.json      # exact --out is in the runner docstring
# expect: LOAD-BEARING=1, null-control clean, prospective.fired True(intact) vs False(lesion)
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other 25 unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag (byte-identical to the baseline). The isolated-lesion values quoted in Diagnosis (1) above are read from the `_prospective_memory_production_verify.py` assertions, not measured here — the line carries the derived marker accordingly. <!--derived-->

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_PMEM_DRIVE_PROBE` flag, the `measure_faculty` remap, `turn_group` over `_EXTRA_TURNS`, docstring verify command, 3 selftest checks + the `pmem_drive_group` artifact field.
- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` (`pmem_form2`, `pmem_cue`) merged into `_TURN_BY_LABEL` only; `PROBE_TURNS` unchanged (still 26).
