---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Non-contradiction-gate is integrated-HOLLOW for a PROBE reason, not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Second follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md), and a sibling of [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md) — the SAME probe-artifact pattern, a second faculty. The load-bearing-fraction instrument reads `noncontradiction-gate` as HOLLOW (lesioning the spiking polarity recall does not change the reply on its probe turn), even though the gate is isolated-lesion-load-bearing in the 6-seed B3 de-risk (disabling negation storage flips the moat from zero false-accepts to a breach on every seed; the canonical "dog does NOT eat grass" reads "yes" once the negation is gone). This finding pins WHY (static, no brain build), and lands a minimal, honest, default-off instrument fix mirroring the episodic one.

(The B3 6-seed isolated-lesion result above is QUOTED from `research/runners/_burndown_B3_onebrain_negation_moat_derisk.py`'s own GO summary, not measured here — the line carries the derived marker accordingly.) <!--derived-->

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the reply composer. Three facts, each read directly from the code:

1. **The reply IS already driven by the recall gate.** `webapp/server.py` (the B3 block, ~L5941-5954) calls `NonContradictionProductionOrgan.check(chat.inner.is_it_true, msg, lesion=...)`; on `reject=True` it RETURNS EARLY with the rejection message + the whole `noncontradiction` block, and on `reject=False` it falls through to the normal reply with the block still attached. So `reject` genuinely composes the reply (rejection vs normal answer) and the block's fields flip with the spiking `ask_yes_no` verdict — it is not computed-then-ignored.

2. **The default probe never gives the gate a stored belief to contradict.** The `noncontradiction-gate` probe (`research/runners/onebrain_regression_battery.py` `FACULTY_PROBES`) is the single turn `well` = "the wolf bites the apple" — a fresh TEACH of BRAND-NEW vocabulary (the battery's own comment at that file's `prospective-memory` row: "wolf/bite/apple are new vocabulary"). So `ask_yes_no("wolf","bite","apple")` has NO matching stored fact and legitimately reads "unknown" on the INTACT substrate. The lesion (`_RecallShim` with `lesion=True`, `b3_noncontradiction_production_organ.py`) ALSO forces every recall to "unknown". Net: `on`/`reject`/`recalled_yn`/`asserted_polarity` are byte-identical intact vs. lesion (accept / reject=False / "unknown") → `compare()` verdict `pass` → NOT load-bearing. The lesion has nothing to collapse.

3. **The gate genuinely IS wired to a real, recallable stored belief — the probe simply never asserts against one.** The tiny-demo brain stores `(dog,chase,cat)=AFFIRM` at BUILD time (`research/runners/brain_chat_tui.py` `_build*` hear-loop: `for a,v,p in facts: inner.hear(f"{a} {v} {p}", polarity="AFFIRM")` — a build-time store, present on ANY backend, no cupy-gated BTSP write). That is exactly why the battery's pre-existing `confirm` and `metacog` probes already recall "yes" on `(dog,chase,cat)`. A turn that asserts the NEGATED form of that same known fact therefore gives a genuinely different intact-vs-lesion outcome.

Net: the gate is hollow on this instrument because the probe asserts a NOVEL fact (nothing stored → intact recall is "unknown", identical to the lesion), not because the wiring is broken. Same anti-hollow pattern as [`2026-08-19-swap-drives-chat-load-bearing-GO.md`](2026-08-19-swap-drives-chat-load-bearing-GO.md): a real neural read is only load-bearing on a turn that actually exercises it.

**External literature (the load-bearing metric IS lesion-based causal attribution).** This probe-artifact class is the well-known limit of loss-of-function inference: a lesion is informative only relative to a behavioral probe that actually recruits the lesioned component. Jonas & Kording (2017), "Could a Neuroscientist Understand a Microprocessor?", PLoS Comput Biol 13(1):e1005268 (doi:10.1371/journal.pcbi.1005268), demonstrate this directly — lesioning transistors one at a time while running a game "boots-or-not" test misattributes function, because the read is only as good as the probe. The fix here is precisely to run a probe that recruits the stored-belief recall, so the lesion has something to collapse. <!--derived-->

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_NONCONTRADICTION_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the `noncontradiction-gate` measurement is remapped to a single fresh-session turn that asserts the NEGATED form of the boot fact:

- One new turn, `noncontra_neg` = "the dog does not chase the cat", session `ncontra`, reset. It lives in `_EXTRA_TURNS` in the battery, merged into `_TURN_BY_LABEL` (so the worker resolves it by label) but deliberately NOT in `PROBE_TURNS` — the default roster stays 26 turns, so the regression battery and every flip-verify harness that iterates it are unchanged.
- `measure_faculty` remaps `noncontradiction-gate` to that turn and compares `noncontradiction.on / reject / recalled_yn / asserted_polarity / stored_polarity`. UNLIKE the episodic fix, NO forced-write env is needed — the fact is stored unconditionally at boot on every backend — so `base_env` stays `{}` in BOTH arms and the null control is a plain rebuild.
- Expected: the turn parses (through the production organ's own `extract_polar_assertion`) to `(dog,chase,cat,NEGATE)`; intact recalls "yes" → stored=AFFIRM ≠ asserted=NEGATE → `reject=True` (`recalled_yn="yes"`, `stored_polarity="AFFIRM"`); the lesion forces "unknown" → accept (`reject=False`, `recalled_yn="unknown"`, `stored_polarity=None`) → the decision fields flip → LOAD-BEARING, null control clean.

Brain-based-only: the load-bearing element is the genuinely-spiking polarity WTA `ask_yes_no` (`_spiking_select` over `cp_firing_states`); the host does only the world (the turn text) and the clock. The gate boolean `stored != asserted` is the thin host comparison the project already accepts as the no-confab moat. The lesion is the existing `BRAIN_NONCONTRADICTION_LESION` neural bypass. Honesty boundary preserved: `recalled_yn`/`stored_polarity` are functional recall read-outs; the rejection notice asserts a stored polarity, never a phenomenal claim.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/noncontradiction_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` PASSES, including four new checks — the driving turn resolves by label, its group is the single turn `[noncontra_neg]`, the `BRAIN_NONCONTRADICTION_LESION` knob resolves in source, and the probe text parses to `(dog,chase,cat,NEGATE)` (the negated boot fact). The artifact also records the default roster is unchanged (`n_probe_turns_default_roster` = 26) and the driving group/env (empty env). A separate default-off structural exact-compare confirmed: the flag reads False by default, `noncontra_neg` is absent from `PROBE_TURNS` but present in `_TURN_BY_LABEL`, the baseline row stays turn `well`, session `ncontra` never appears in the default roster, and every existing turn-group is unchanged.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (owner directive: the controller verifies on AWS/local; no local full-brain smokes). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the 25-faculty no-regression are the controller's step (numpy is sufficient — no forced write, so a store is not the ~510s/store episodic cost):

```
LB_NONCONTRADICTION_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only noncontradiction-gate --repeats 2 \
    --out <_load_bearing dir>/noncontradiction_drive.json
# expect: LOAD-BEARING=1, null-control clean, noncontradiction.reject True(intact) vs False(lesion)
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other 25 unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag.

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_NONCONTRADICTION_DRIVE` flag, the `measure_faculty` remap (no forced-write env), the `_noncontra_probe_parses_negated_boot_fact` static check, the docstring verify command, 4 selftest checks + 2 artifact fields.
- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` gains `noncontra_neg` (merged into `_TURN_BY_LABEL` only; `PROBE_TURNS` unchanged at 26).
