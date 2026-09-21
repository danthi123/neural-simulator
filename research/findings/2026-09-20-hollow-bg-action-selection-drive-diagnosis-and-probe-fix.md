---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# bg-action-selection is integrated-HOLLOW for a PROBE reason (a confounded compared field), not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md) and to the episodic-memory sibling [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md). Same class of bug, a DIFFERENT root cause: the BG selector genuinely DRIVES the reply on its `bgdots` probe, but the ONE field the harness compares (`abstained`) collides with an unrelated abstain path, so the real decision change is invisible to `compare()`. This finding pins WHY (statically) and lands a minimal, honest, default-off instrument fix.

## Diagnosis (proven statically — no brain build needed)

The gap is in the compared FIELD, not the reply composer or the probe turn. Four facts, each read directly from the code:

1. **The reply IS already driven by the BG race, on the RIGHT turn.** The battery row is `("bg-action-selection", "bgdots", ["abstained"], False)` (`research/runners/onebrain_regression_battery.py:272`), and `bgdots` is the message `"..."` (`onebrain_regression_battery.py:75`) — a content-empty turn where STAY-SILENT is a genuine contender. On it, `salience("...")` returns `speak=0, silent=1` (`research/runners/bg_action_selection_production_organ.py:142-150`, zero content tokens), so the STAY-SILENT channel is maximally favored. Intact, `_run_biased_trial` applies the full arousal-gated salience barrage (`bg_action_selection_production_organ.py:162-177`) and the STAY-SILENT channel commits, so `decide_action` returns the commit dict (`organ:282-311`), and `webapp/server.py:4713-4723` SHORT-CIRCUITS the turn with `{"answer": HOLD_TEXT, "abstained": True, ..., "bg_select": {"on": True, ...}}`. The `bg_select` key is written EXACTLY ONCE in the whole server, only inside that short-circuit (`grep '"bg_select"' webapp/server.py` -> `server.py:4718`).

2. **The lesion collapses the race — and the turn falls through, losing the `bg_select` key entirely.** The load-bearing runner lesions with `BRAIN_BG_SELECT_LESION=1` (`research/runners/load_bearing_fraction.py:209`), which `bg_select_lesion()` maps to `"arousal"` (`organ:120-121`). With `lesion="arousal"`, `select_once` sets `arousal=False` (`organ:257`), so the ENTIRE `if arousal:` block (`organ:162-177`) — which supplies BOTH the proposal->D1 barrage AND the salience bias — is skipped; nothing commits, `decide_action` returns None (`organ:301-302`), and the turn falls through past the BG block. It reaches the single-fact path (`webapp/server.py`), where `"..."` has no comprehensible content (no transitive, no SVO) -> `answer, abstained, verified = "I don't know about that.", True, False`, with NO `bg_select` key set.

3. **So the compared field `abstained` is True in BOTH arms — via two independent paths.** Intact: `abstained=True` because the BG HOLD short-circuit fired. Lesion: `abstained=True` because the independent no-content fallback abstained. `compare()`'s only checked field, `abstained`, reads True==True -> verdict `pass` -> NOT load-bearing — even though the answer TEXT differs (HOLD_TEXT vs "I don't know about that.") AND the `bg_select` block's presence/absence differs (a genuine structural diff the current field list never inspects).

4. **This is a field-collision artifact, not a thin probe.** Unlike episodic-memory (which never gave its faculty a memory to recall), `bgdots` already exercises the BG mechanism in its designed STAY-SILENT-favored regime and the lesion already collapses it. The mechanism is load-bearing on the reply; the instrument's (turn, compared-field) pair simply cannot see it, because `abstained` collides with an unrelated abstain path.

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_BG_SELECT_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the bg-action-selection measurement REMAPS its compared field from the confounded top-level `abstained` to the structural `bg_select.on`, keeping the SAME turn, session, and env:

- `measure_faculty` rebinds `row = ("bg-action-selection", "bgdots", ["bg_select.on"], False)` before `turn_group(row[1])`, mirroring the `LB_EPISODIC_DRIVE_PROBE` block exactly, with `base_env` left `{}` (no forced write is needed — `bgdots` already puts the race in its STAY-SILENT-favored regime).
- No new turn or session is added: `bgdots` is already in the default `PROBE_TURNS` roster (unlike the episodic pair, which needed `_EXTRA_TURNS`). So NO change to `onebrain_regression_battery.py` is required, and the default roster stays 26 turns — the regression battery and every flip-verify harness are byte-identical.
- `bg_select.on` is present+True ONLY on the intact short-circuit and absent on the lesioned fallback, so `compare()` sees a field present intact / absent lesioned -> verdict `regressed`, `change_kind` `structural` (the gold-standard robust diff — inherently immune to any RNG-trajectory shift). The NULL control (intact vs intact-rebuild) is unaffected: both intact arms fire the short-circuit -> both set `bg_select.on=True` -> 0 control diffs -> clean null.

Brain-based-only: the SPEAK-vs-STAY-SILENT decision is the genuine two-channel spiking basal-ganglia race (striatal D1 channels, D1->GPi direct-path disinhibition, GPi->thalamus commit burst); host does only the world (the turn text) and the clock. The lesion is the existing `BRAIN_BG_SELECT_LESION` neural cut (the arousal-off / no-barrage control). Honesty boundary preserved: `bg_select.on` is a functional read-out of which channel committed; the HOLD line asserts no phenomenal claim, and a STAY-SILENT hold is an honest "nothing salient to add", never a confabulation.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/bg_select_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` PASSES, including four new checks — `bgdots` is in the default roster, its group is the single-turn `["bgdots"]`, the CONFOUNDED `abstained` field reads `pass` on the two-path-True arms (reproducing the bug), and the REMAPPED `bg_select.on` field reads `regressed` + `structural` (the fix). The artifact also records the default roster is unchanged (`n_probe_turns_default_roster`) and the bg-select driving group/field.

## External grounding (the methodological hazard this fix addresses)

This is a specific instance of a well-established necessity-testing hazard: a neural manipulation (a lesion) can only reveal a faculty's causal contribution if the measured readout is DIAGNOSTIC of the process being manipulated — otherwise a genuinely load-bearing mechanism reads as inert. Krakauer, Ghazanfar, Gomez-Marin, MacIver & Poeppel (2017), "Neuroscience Needs Behavior: Correcting a Reductionist Bias," Neuron 93(3):480-490 (PMID 28182904), make exactly this argument: necessity/sufficiency manipulations require careful behavioral decomposition to choose a readout that actually tracks the component process, or the manipulation's effect is invisible or misattributed. Here `abstained` was the non-diagnostic readout (it collides with an independent no-content abstain path, so the lesion's effect is masked); `bg_select.on` is the diagnostic one (it is set only when the BG commit path fired). The fix changes the readout, not the manipulation — the manipulation (`BRAIN_BG_SELECT_LESION`) was already correct.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (owner gaming; no local full-brain smokes). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the 25-faculty no-regression are the controller's step (numpy is fine — no forced write, unlike episodic):

```
LB_BG_SELECT_DRIVE_PROBE=1 tools/memcap.sh 20 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only bg-action-selection --repeats 2 \
    --out <_load_bearing dir>/bg_select_drive.json   # expect LOAD-BEARING=1, change_kind=structural, null clean
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other 25 unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag (byte-identical to the 16/26 baseline).

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_BG_SELECT_DRIVE_PROBE` flag, the `measure_faculty` field remap, the docstring verify command, 4 selftest checks, the selftest-artifact bg-select group/field record.
- `research/runners/onebrain_regression_battery.py` — UNCHANGED (the `bgdots` driving turn is already in `PROBE_TURNS`).
