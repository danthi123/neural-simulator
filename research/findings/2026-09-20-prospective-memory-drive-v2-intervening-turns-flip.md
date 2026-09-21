---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Prospective-memory reads integrated-HOLLOW because the driving probe had NO intervening turns — the held x cue coincidence needs a DELAY to reach its operating point; a formation -> 3-intervening -> cue probe flips it load-bearing (2026-09-20, v2)

Supersedes the diagnosis in `2026-09-20-hollow-prospective-memory-drive-diagnosis-and-probe-fix.md` (branch `research/hollow-prospective-memory-drive`), whose fix — a 2-turn `[pmem_form2, pmem_cue]` driving group — was BRAIN-BUILD verified and FAILED (treat=0, verdict `pass`). That finding correctly identified that the DEFAULT probe (`pmem_form`, comparing the compile-time-constant `prospective.held`) is hollow, but its proposed fix was ALSO hollow for a different, un-diagnosed reason. This finding pins the real reason (measured), lands the corrected probe, and reports the brain-build verdict.

## Why the FIRST fix still read treat=0 (measured, organ-level)

The prior fix compared `prospective.fired` on a group that ran the FORMATION turn IMMEDIATELY followed by the CUE turn — no intervening turns. That does not fire, even INTACT. Driving the production organ directly (`ProspectiveMemoryOrgan`, seed 42, numpy) with the exact probe messages (`FORM="remind me to feed the dog when the bird sings"`, `CUE="the bird sings"`), the intact cue-turn release accumulator `rel_A` ramps with the number of intervening turns held between formation and cue, while the `BRAIN_PMEM_LESION` arm stays flat at the floor (artifact `research/findings/raw/_load_bearing/pmem_intervening_ramp.json`):

| intervening turns | intact rel_A | intact fired (thr 0.2) | lesion rel_A | lesion fired | FLIP |
|---|---|---|---|---|---|
| 0 | 0.1633 | **False** | 0.0428 | False | **no** |
| 1 | 0.2206 | True | 0.0417 | False | yes |
| 2 | 0.2561 | True | 0.0411 | False | yes |
| 3 | 0.3400 | True | 0.0406 | False | yes |
| 4 | 0.3644 | True | 0.0406 | False | yes |
| 5 | 0.3883 | True | 0.0406 | False | yes |

At n=0 the intact accumulator (0.1633) is BELOW the frozen FIRE_THR (0.2), so intact does not fire -> intact `fired=False` == lesion `fired=False` -> `compare()` verdict `pass` -> treat=0 -> hollow. This is exactly the reported failure of the 2-turn probe. At n>=1 the intact arm fires and the lesioned latch stays silent, so `prospective.fired` FLIPS.

This is not a bug and not a tunable — it is the biology of the faculty, and our own de-risk already recorded it: `2026-08-13-prospective-memory-intention-latch-cue-monitor-derisk.md` states "the held intention supplies a subthreshold priming depolarization; only when the CUE arrives on the primed pool does rel_X cross threshold and fire." The intervening turns are the mechanism's companion process (the held-intention priming ramps as the hold is advanced by real competing WM load, `intervening_turn`), which the zero-delay probe replaced with nothing. A zero-delay formation->cue is not a prospective-memory test at all: there is nothing *prospective* (forward-in-time) about a cue that arrives with no delay.

**External grounding** (this is the definition of the faculty, not an implementation artifact). Prospective memory is retrieval of an intention that was ENCODED earlier, HELD across a DELAY filled with intervening ongoing activity, and triggered when its target CUE is later encountered — the multiprocess framework treats the cue-triggered retrieval at the delayed target as the load-bearing phase, distinct from the encoding act (McDaniel & Einstein, 2000, *Applied Cognitive Psychology* 14(7):S127-S144, doi:10.1002/acp.775, https://doi.org/10.1002/acp.775). A probe that puts the cue immediately after formation measures neither the hold nor the delayed retrieval — the exact phase the literature says carries the effect. <!--derived-->

## The fix (built; default-off; byte-identical when off)

Default-off env flag `LB_PMEM_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the prospective-memory measurement is remapped to a formation -> **3 intervening turns** -> cue group in one isolated session, running the NATURAL prospective protocol:

- Five label-only turns in `_EXTRA_TURNS` (session `pmem2`, declared formation-first, NONE in `PROBE_TURNS`): `pmem_form2` ("remind me to feed the dog when the bird sings") -> `pmem_d0` ("what does the cat eat") -> `pmem_d1` ("how is the weather today") -> `pmem_d2` ("tell me about the sky") -> `pmem_cue` ("the bird sings"). The three distractors carry neither cue keyword (`bird`/`sings`) nor a formation phrasing, so each advances the hold (`intervening_turn`) and none fires prematurely; `turn_group("pmem_cue")` resolves to all five in order.
- `measure_faculty` remaps prospective-memory to turn `pmem_cue` and compares `prospective.fired`. NO `base_env` forcing (unlike the episodic fix): `BRAIN_PMEM` and `BRAIN_PMEM_HEBBIAN` are default-ON, so the ordinary intact build learns the cue->action binding one-shot at formation, holds it across the distractors, and fires on the cue turn (`fired=True`); the `BRAIN_PMEM_LESION` arm collapses the latch at formation (`held_after_lesion<=0.02`) so the cue stays silent (`fired=False`). Every other faculty keeps `base_env={}` -> byte-identical.
- 3 intervening turns (not the bare minimum of 1) matches the validated isolated-verify + de-risk protocol (`_prospective_memory_production_verify.py`, fire_on_cue 6/6): intact `rel_A~0.34`, ~70% over threshold — a robustness margin, NOT a knob tuned to barely cross. No substrate parameter (FIRE_THR, homeostat bias, plateau theta) was touched; the flip comes only from running the faculty's own natural protocol.

Brain-based-only: the HOLD (attractor persistence), the per-turn hold advance, and the coincidence-gated RELEASE are the spiking substrate; host does only the world (the turn text), the intention/cue text->slot mapping + cue-presence (a declared language/sensory boundary), and the clock. The lesion is the existing `BRAIN_PMEM_LESION` neural cut (the latch zeroed at formation). Honesty boundary preserved: `fired` is a functional coincidence read-out; a non-fire is an honest silence.

## The brain-build verdict (the required proof; static PASS is NOT sufficient)

Command (numpy CPU, memcapped, the exact HARD-RULE form):

```
SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' LB_PMEM_DRIVE_PROBE=1 tools/memcap.sh 16 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only prospective-memory --repeats 2 \
    --out research/findings/raw/_load_bearing/pmem_drive.json
```

RESULT (real ChatBrain + real `brain_chat` handler, seed 42, stub renderer, no LLM), artifact `research/findings/raw/_load_bearing/pmem_drive.json`:

- **`load_bearing = True`**, verdict `regressed`, `load_bearing_fraction = 1.0` (1/1 exercised).
- The decision field `prospective.fired` FLIPPED: intact `True` (cue turn `rel_A=0.34056`, artifact `research/findings/raw/_load_bearing/intact_a_pmem_form2_pmem_d0_pmem_d1_pmem_d2_pmem_cue.json`; the answer is prepended "(Reminder — you asked me to feed the dog when the bird sings came up, and the bird sings just came up.) ...") vs lesion `False` (`rel_A=0.0411`, no reminder; artifact `research/findings/raw/_load_bearing/lesion_prospective_memory.json`). `treatment_diffs=1`, `change_kind=value`.
- Clean attribution: `control_diffs=0`, `null_control_clean=True` (intact_a == intact_b `research/findings/raw/_load_bearing/intact_b_pmem_form2_pmem_d0_pmem_d1_pmem_d2_pmem_cue.json`, both `fired=True`, `rel_A=0.34056` — the harness is deterministic at seed 42), `attributable_fraction=1.0` (100% of the change is the lesion, 0% in the null control).
- Reproduced: `lesion_reproduced=True` (`--repeats 2`; both lesion rebuilds read `fired=False`, `rel_A=0.0411`; artifact `research/findings/raw/_load_bearing/lesion_prospective_memory.json.rep0`).

So lesioning the brain's prospective contribution (`BRAIN_PMEM_LESION` — the held latch zeroed at formation) provably changes the produced reply (the reminder is delivered intact, absent lesioned) on a natural cue turn. Prospective-memory is now load-bearing on this instrument.

NOTE (transient, for the record, not a caveat on the result): the FIRST brain-build of this same probe returned verdict `noisy`/`load_bearing=None` — its treatment already flipped (`fired` True vs False, `attributable_fraction=1.0`, clean null) but the `--repeats 2` lesion-rebuild subprocess crashed (its output file was never written -> `_spawn_arm` returned None) under heavy CPU contention with a concurrent full `load_bearing` run. Re-run with the contention gone, all four arms built cleanly and the lesion result reproduced. The transient build failure, not a genuine non-reproduction, was the sole cause of the earlier `noisy` verdict.

The static selftest (`--selftest`) PASSES all four pmem-drive checks (turns exist; group is formation->intervening->cue; holds across >=1 intervening turn; lesion knob resolves), but per the first attempt's lesson a static PASS is not the proof — the brain-build verdict above is.

## Byte-identical when off

`LB_PMEM_DRIVE` defaults False; the default probe turn `pmem_form` (comparing `prospective.held`) is unchanged, `PROBE_TURNS` is still 26 turns, and the 5 driving turns are label-only in `_EXTRA_TURNS` (out of the roster the regression battery + every flip-verify harness iterate) -> the whole default path is byte-identical to the 2026-09-19 hollow baseline. Verified statically: `len(PROBE_TURNS)==26`, `turn_group("pmem_form")==["pmem_form"]`, `LB_PMEM_DRIVE is False` with the flag unset.

## Files

- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` gains `pmem_form2`, `pmem_d0`, `pmem_d1`, `pmem_d2`, `pmem_cue` (session `pmem2`, label-only); `PROBE_TURNS` unchanged.
- `research/runners/load_bearing_fraction.py` — the `LB_PMEM_DRIVE_PROBE` flag, the `measure_faculty` remap to `pmem_cue`/`prospective.fired`, docstring verify command, 4 selftest checks + the `pmem_drive_group` artifact field.
- `research/runners/_pmem_intervening_ramp_probe.py` — the organ-level intervening-turn ramp diagnostic (produces the ramp artifact cited above; no full brain build).
