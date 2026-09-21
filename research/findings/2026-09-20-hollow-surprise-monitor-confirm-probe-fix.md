---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Surprise-monitor is integrated-HOLLOW for a PROBE reason, not a wiring reason — diagnosis + a default-off CONFIRM-turn probe fix (2026-09-20)

Follow-on to the load-bearing baseline (its tracked per-seed report `research/findings/raw/_load_bearing/load_bearing_s42.json`, where surprise-monitor reads `turn: contra`, `verdict: pass`, `load_bearing: false`, `diffs: []` — integrated-HOLLOW: lesioning its brain contribution does not change the reply on the probe). This is the SAME class the episodic-memory finding closed ([`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md)): an isolated-lesion-load-bearing faculty measured on a probe turn its lesion never bites. It pins WHY and lands a minimal, honest, default-off instrument fix. The measured flip to load-bearing is a brain build and is the controller's AWS/local step (NOT claimed here).

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the reply composer. Three facts, each read directly from the code:

1. **The default probe is a CONTRADICT trial the lesion cannot reach.** The surprise-monitor row (`research/runners/onebrain_regression_battery.py`, `FACULTY_PROBES`) uses turn `contra` ("the dog chase the fish"): (dog,chase) is known from build time with patient 'cat', so 'fish' is a DIFFERENT patient. The dedicated lesion `BRAIN_SURPRISE_LESION` zeroes the block-diagonal `patient_expected->surprise` edges (`surprise_production_organ.py`, `_build_one(lesion=True)` → `_install_block_diagonal(..., "patient_expected", "surprise", ..., 0.0)`). That inhibition, intact, only cancels excitation on the SAME (stored) block — so it only matters on a CONFIRM trial. On CONTRADICT the asserted patient lives in a different block the lesioned inhibition never reached, so the decision is unchanged by the lesion. The organ's own docstring states it directly: "zeroing the patient_expected->surprise prediction edges removes the subtractive inhibition, so CONFIRM fires as high as CONTRADICT — the separation collapses."

2. **The lesion only BITES on a CONFIRM trial.** On CONFIRM (asserted==stored, SHARED block; `surprise_production_organ.py` `read_surprise`, `if str(p_asserted).lower() == str(p_stored).lower(): t = s`), the intact prediction inhibits exactly that block, cancelling the excitation → surprised=False; the lesion removes the cancellation → the block fires → surprised=True. So `surprise.surprised` FLIPS intact-vs-lesion on CONFIRM, and does not on CONTRADICT. This is the canonical prediction-error microcircuit: a top-down prediction delivered by inhibitory interneurons cancels a matching bottom-up input, leaving the prediction-error population quiet on a match; a mismatch (or a REMOVED prediction) leaves it un-inhibited and it fires (Bastos, Usrey, Adams, Mangun, Fries, Friston 2012, "Canonical microcircuits for predictive coding," Neuron 76:695-711, PMID 23177956) — so the lesion's diagnostic signature is precisely a spurious error on a CONFIRM/match trial, not on a mismatch.

3. **The reply IS driven by the field.** `webapp/server.py` computes `sj = sorg.judge(..., lesion=_SO.surprise_lesioned())`, sets `surprise_prefix = _SO.surprise_notice(...)` only when `sj["surprised"]`, and splices `surprise_prefix` into the returned `answer`. So a CONFIRM-turn lesion spuriously prepends "That surprises me — my mismatch monitor fired ..." to a plain restatement — a real user-visible diff the CONTRADICT probe can never expose (both of its arms already carry the notice).

Empirical grounding from the already-produced 6-seed baseline arms (read from the prior run, not measured in this finding): on `contra`, intact and lesion both read surprised=true → no flip → `pass` → not load-bearing. On the `confirm` turn (already in the roster for metacog-monitor) the intact arm reads surprised=false with surprise_hz below threshold `2.629`, while its own `calib.confirm_before_max` `4.398` (the PARTIALLY-inhibited, pre-homeostat confirm rate) already EXCEEDS that threshold — so fully removing the inhibition (the lesion) fires confirm at or above that rate → surprised flips False→True. <!--derived-->

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_SURPRISE_CONFIRM_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the surprise-monitor measurement is remapped from `contra` to the CONFIRM turn `confirm` ("the dog chase the cat"), comparing the decision field `surprise.surprised`:

- The `confirm` turn is ALREADY in `PROBE_TURNS` (it drives metacog-monitor), session `surp`, a single-turn group — so `turn_group("confirm") == ["confirm"]` with NO new turn/session and NO forced-write env. This is strictly simpler than the episodic fix (which needed a store→recall pair in `_EXTRA_TURNS`); the regression battery is not edited at all, so the default roster stays 26 turns and every flip-verify harness is byte-identical.
- No `base_env`: the confirm read is deterministic (homeostat-calibrated at build), so BOTH intact arms read surprised=False → clean null control; only the lesion flips it. Every other faculty keeps `base_env={}`.
- Expected: intact surprised=False (verified present in the prior confirm arm), lesion surprised=True (the un-inhibited confirm block) → the field flips → verdict `regressed` → LOAD-BEARING, null-control clean, change kind `value`.

Brain-based-only: the surprise read is the genuinely-spiking `cp_firing_states[surprise]` off the predictive-coding mismatch circuit; the lesion is the existing `BRAIN_SURPRISE_LESION` neural cut (zeroed prediction edges); host does only the world (the turn text) and the clock. Honesty boundary preserved: `surprised` is a functional threshold read-out on a spiking rate, and the notice ("my mismatch monitor fired") asserts no phenomenal claim.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/surprise_confirm_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` passes, including three new checks — the confirm turn is in the default roster, its group is the single confirm turn, and the `BRAIN_SURPRISE_LESION` cut resolves in source. The artifact also records the default roster is unchanged (`n_probe_turns_default_roster` 26) and the surprise-confirm remap group. A separate capture (stubbing the arm spawner so no brain builds) confirms that with the flag OFF the surprise-monitor arms are `env {}` / turns `["contra"]` (intact) and `{BRAIN_SURPRISE_LESION:"1"}` / `["contra"]` (lesion) — identical to the baseline — and only flip to the `confirm` turn when the flag is ON, with every other faculty (including metacog-monitor, also on the confirm turn) untouched.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (verify statically only, per the task). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the no-regression on the other faculties are the controller's step:

```
LB_SURPRISE_CONFIRM_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only surprise-monitor --repeats 2 \
    --out <_load_bearing dir>/surprise_confirm.json    # exact --out is in the runner docstring
# expect: load-bearing=1, null-control clean, surprise.surprised False(intact) vs True(lesion)
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other faculties unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag (byte-identical to the hollow baseline).

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_SURPRISE_CONFIRM_PROBE` flag, the `measure_faculty` remap to the confirm turn, the docstring verify command, 3 selftest checks, and the selftest artifact fields.
- `research/runners/onebrain_regression_battery.py` — UNCHANGED (the confirm turn already exists in `PROBE_TURNS`); no edit needed, so the default roster stays 26 turns.
