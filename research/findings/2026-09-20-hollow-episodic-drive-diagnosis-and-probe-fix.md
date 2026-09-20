---
type: finding
status: contributing
lane: load-bearing
date: 2026-09-20
---

# Episodic-memory is integrated-HOLLOW for a PROBE reason, not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Follow-on to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md) (its own cited `_load_bearing` baseline artifact), which recorded episodic-memory as the starkest HOLLOW faculty: isolated-lesion-load-bearing (its ledger `lesion_note` quotes the dAP completion collapsing 0.909->0.000 under `BRAIN_EPISODIC_LESION`) yet integrated-HOLLOW (lesioning it does not change the reply). This finding pins WHY and lands a minimal, honest, default-off instrument fix. Verification of the flip itself is a brain build and is deferred to the controller (owner gaming this machine) — so this is filed `contributing`, not a GO. <!--derived-->

(Those two isolated-collapse values above are QUOTED from the episodic-memory ledger `lesion_note` + the 2026-08-12 isolated-organ verify, not measured here — the line carries the derived marker accordingly.)

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the reply composer. Three facts, each read directly from the code:

1. **The reply IS already driven by the recall gate.** `webapp/server.py` Hook A (referential-recall short-circuit, ~L5253-5299) returns `answer = recall_disclosure(rec, content)`, `abstained = not in_memory`, `verified = in_memory`, and the whole `episodic` block — ALL of which flip with the spiking `in_memory` verdict. When a topic completes, the reply is "Earlier you brought up X ... {content}"; when it does not, it is "I don't recall us discussing X". So the recall READ genuinely composes the reply text; it is not computed-then-ignored.

2. **The probe never gives episodic a memory to recall.** The episodic faculty's probe (`research/runners/onebrain_regression_battery.py` `FACULTY_PROBES`, key `episodic-memory`) is the single turn `episodic` = "did we discuss the dog" on a FRESH session (`("episodic", ..., "epi", True, ...)`, reset=True, no prior turn). With nothing stored, the INTACT recall correctly reads `in_memory=False` (an honest not-in-memory) — the SAME value the lesion produces (an unformed-weights collapse of an assembly that was never formed). Compared field `episodic.in_memory` is False==False -> compare() verdict `pass` -> NOT load-bearing. The lesion has nothing to collapse.

3. **On the numpy probe backend the write is deferred anyway.** Even if a storing turn were added, `EpisodicRecallOrgan.recall` returns `in_memory=False` with reason `no-store-yet` whenever `self.mem is None`, and the BTSP WRITE (Hook B, `webapp/server.py` ~L6174) only runs when `_episodic_store_ok()` is True — which on numpy is False (the write is cupy-gated, ~510s/store on numpy@2000 vs ~seconds on cupy). The load_bearing runner builds numpy arms by default. So without forcing the write, no assembly ever forms on the probe backend and the recall can never read `in_memory=True`.

Net: episodic is hollow on this instrument because the probe (a) never stores before it queries, and (b) runs on a backend where the store is deferred. In cupy production with a real prior turn, the recall gate DOES flip the reply — the instrument simply never constructs that condition. This matches the anti-hollow pattern of [`2026-08-19-swap-drives-chat-load-bearing-GO.md`](2026-08-19-swap-drives-chat-load-bearing-GO.md): a real neural read is only load-bearing on a turn that actually exercises it.

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_EPISODIC_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the episodic-memory measurement is remapped to a STORE->RECALL pair in one isolated session and the BTSP write is forced so it runs on any backend:

- Two new turns, `epi_store` ("the dog chase the cat") -> `epi_recall` ("did we discuss the dog"), session `epi2`, declared store-first. They live in `_EXTRA_TURNS` in the battery, merged into `_TURN_BY_LABEL` (so the worker resolves them by label) but deliberately NOT in `PROBE_TURNS` — the default roster stays 26 turns, so the regression battery and every flip-verify harness that iterates it are byte-identical.
- `measure_faculty` remaps episodic to turn `epi_recall` (group `["epi_store","epi_recall"]`) and injects `BRAIN_EPISODIC_STORE=1` into BOTH the intact and lesion arms (the null control also stores, so both intact arms read `in_memory=True` -> clean null; only the lesion collapses it). Every other faculty keeps `base_env={}` -> byte-identical.
- Expected: `epi_store` BTSP-forms the "dog" CA3 assembly (Hook B verified-SVO write); `epi_recall` completes it -> intact `in_memory=True` (disclosure), lesion `in_memory=False` ("I don't recall") -> field flips -> LOAD-BEARING, null-control clean.

Brain-based-only: the store and recall are the genuinely-spiking BTSP write + dendritic-dAP completion; host does only the world (the turn text) and the clock. The lesion is the existing `BRAIN_EPISODIC_LESION` neural cut (unformed recurrent weights). Honesty boundary preserved: `in_memory` is a functional completion read-out; the disclosure text asserts no phenomenal claim, and a completion failure is an honest abstain, never a confabulation.

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/episodic_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` PASSES, including three new checks — the driving turns resolve by label, `turn_group("epi_recall") == ["epi_store","epi_recall"]`, and the forced-write flag resolves in source. The artifact also records the default roster is unchanged (`n_probe_turns_default_roster`) and the episodic-driving group/env.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (owner gaming; no local full-brain smokes). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the 25-faculty no-regression are the controller's AWS step (cupy strongly preferred so the forced write is ~seconds):

```
SIM_BACKEND=cupy LB_EPISODIC_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only episodic-memory --repeats 2 \
    --out <_load_bearing dir>/episodic_drive.json      # exact --out is in the runner docstring
# expect: LOAD-BEARING=1, null-control clean, episodic.in_memory True(intact) vs False(lesion)
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other 25 unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag (byte-identical to the 16/26 baseline).

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_EPISODIC_DRIVE_PROBE` flag, the `measure_faculty` remap + `base_env` threading, `turn_group` over `_EXTRA_TURNS`, docstring verify command, 3 selftest checks.
- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` (`epi_store`, `epi_recall`) merged into `_TURN_BY_LABEL` only; `PROBE_TURNS` unchanged.
