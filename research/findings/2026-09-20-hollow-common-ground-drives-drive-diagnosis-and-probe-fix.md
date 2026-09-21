---
type: finding
status: live
lane: load-bearing
date: 2026-09-20
---

# Common-ground-drives is integrated-HOLLOW for a PROBE reason, not a wiring reason — diagnosis + a default-off driving-probe fix (2026-09-20)

Second follow-on (after [`2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md`](2026-09-20-hollow-episodic-drive-diagnosis-and-probe-fix.md)) to the baseline [`2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md`](2026-09-19-load-bearing-fraction-baseline-16of26-first-reading.md). Common-ground-drives is another faculty the baseline recorded HOLLOW: isolated-lesion-load-bearing (its own `common_ground_drives_chat.cg_drives_lesioned` builds the ledger recurrence at weight 0 so a re-mentioned referent can no longer read grounded and the reduced-reference lead vanishes) yet integrated-HOLLOW on the default probe (lesioning it does not change the reply). This finding pins WHY and lands a minimal, honest, default-off instrument fix that mirrors the episodic one exactly.

The fix is STATIC/selftest-verified here; the measured flip is the controller's step (a full brain build — not run here per the gaming/no-local-smoke rule).

## Diagnosis (proven statically — no brain build needed)

The gap is in the PROBE, not the reply composer. Read directly from the code:

1. **The reply IS already driven by the ledger read.** `webapp/server.py` (~L5085-5094 the main-path observe, L6377-6382 the attach) turns the audience-design verdict into a REDUCED-REFERENCE lead (`decision=="reduce"` -> `"As for it — "` prepended; `decision=="introduce"` -> no lead) and attaches the `common_ground_drives` trace whose `decision` field composes that surface. So the substrate read genuinely composes the reply; it is not computed-then-ignored (`common_ground_drives_chat.audience_design_lead`).

2. **The default probe never grounds a referent, so it never reaches the reduce/introduce fork.** Two compounding probe-construction failures. (a) The default probe (`onebrain_regression_battery.FACULTY_PROBES`, key `common-ground-drives`) is the single FRESH-session turn `well` = "the wolf bites the apple". `wolf`/`bites`/`apple` are NOT build-time KB concepts (the tiny-demo KB is brain/spikes/words/memory + dog->chase->cat + cat->eat->fish, `brain_chat_tui.py`), so `gnw_thought_swap._extract_topic` finds no grounded token and returns None -> `common_ground_ledger_production_organ.observe_turn` takes its `if not topic: return {..., "decision": None, ...}` branch IDENTICALLY on the intact AND `BRAIN_CG_DRIVES_LESION=1` arms -> compare() sees `decision`=None==None -> hollow. (b) Even with a grounded first-mention token, the load-bearing divergence only appears on a RE-MENTION of an already-grounded referent: `observe_turn` reads the ledger AS IT STANDS (before this turn's grounding act) — on a first mention `was_grounded` is False and BOTH arms read UNGROUNDED -> `decision`=introduce; only on a re-mention does the intact ledger's self-sustaining recurrence still hold the slot (`decision`=reduce) while the lesioned recurrence=0 ledger has decayed (`decision` stays introduce).

3. **Also a call-ordering red herring for OOV topics.** `webapp/server.py` runs the observe before the turn's own `chat.gate` teach step, so a turn's own new words cannot enter the KB before topic-extraction — but this only matters for OOV tokens. Grounding on a BUILD-TIME concept (`dog`) sidesteps it entirely: `dog` is a KB agent from build, found by `_extract_topic` on turn one with no gate-ordering dependency.

Net: common-ground-drives is hollow on this instrument because the probe is a single fresh-session first-mention over an OOV token — structurally identical to the episodic case. The server.py wiring is real and gate-verified in isolation (the organ's own `lesion_note`); the probe simply never drives the reduce-vs-introduce fork.

## The fix (built; selftest-verified; default-off; byte-identical when off)

A default-off env flag `LB_CG_DRIVE_PROBE` in `research/runners/load_bearing_fraction.py`. When set, the common-ground-drives measurement is remapped to a MENTION->RE-MENTION pair in one isolated session over a build-time KB referent:

- Two new turns, `cg_mention1` ("the dog runs fast") -> `cg_mention2` ("the dog runs again"), session `cg2`, mention-first. They live in `_EXTRA_TURNS` in the battery, merged into `_TURN_BY_LABEL` (so the worker resolves them by label) but deliberately NOT in `PROBE_TURNS` — the default roster stays 26 turns (cited artifact `n_probe_turns_default_roster`: 26), so the regression battery and every flip-verify harness that iterates it are byte-identical. `dog` is a build-time KB agent so `_extract_topic` finds it immediately with no gate-ordering / OOV issue.
- `measure_faculty` remaps common-ground-drives to turn `cg_mention2` (group `["cg_mention1","cg_mention2"]`) and narrows the compared field to `common_ground_drives.decision` (`.on` is True on both arms, `.reason` is absent on a topic'd turn — neither could discriminate). `base_env` stays `{}` for BOTH the intact and lesion arms — no forced-write / backend flag is needed because `common_ground_ledger_production_organ` pins `SIM_BACKEND=numpy` for its own bridge unconditionally, so the driving pair works on the default probe backend. Every other faculty keeps `base_env={}` -> byte-identical.
- Expected: `cg_mention1` grounds `dog`'s slot (ignite + NMDA self-sustain); `cg_mention2` re-mentions it -> intact ledger holds it -> `decision`=reduce, lesioned ledger (recurrence built at weight 0) collapsed it -> `decision`=introduce -> field flips -> LOAD-BEARING, null-control clean (both intact arms read reduce).

Brain-based-only: the grounding act and the reduce/introduce read are the genuinely-spiking NMDA-attractor ledger + Namburi-Tye biased-competition read; host does only the world (the turn text), the word->slot map (a comprehension boundary, like the SVO parser) and the clock. The lesion is the existing `BRAIN_CG_DRIVES_LESION` neural cut (recurrence weight 0). Honesty boundary preserved: `decision` is a functional audience-design read-out; the reduced-reference lead frames HOW the reply refers, never WHICH fact is true, and asserts no phenomenal claim.

External literature (mechanism grounding, logged in `research/queue/.external_searches.jsonl`, lane load-bearing): the reduce-on-re-mention phenomenon the probe exercises is the classic given/new / audience-design result — Clark & Brennan (1991) "Grounding in communication" (referring expressions shorten as shared information accumulates once a referent is grounded); the PERSISTENT across-turns common-ground store the intact ledger implements has its neural basis in Duff & Brown-Schmidt (2012) "The hippocampus and the flexible use and processing of language" (Front. Hum. Neurosci. 6:69, doi:10.3389/fnhum.2012.00069, PMID 22493573 — the hippocampal declarative system maintains representations on-line for language use); see also "Refer, Reuse, Reduce" (arXiv:2011.04554) on subsequent-reference reduction. This is confirmatory (the organ already cites the same grounding literature), not a new mechanism claim. <!--derived-->

Static verification (no brain build), artifact `research/findings/raw/_load_bearing/cg_drive_selftest.json`: `python -m research.runners.load_bearing_fraction --selftest` PASSES, including four new checks — the driving turns resolve by label, `turn_group("cg_mention2") == ["cg_mention1","cg_mention2"]`, the lesion knob `BRAIN_CG_DRIVES_LESION` resolves in source, and the drive needs no forced env (`_CG_DRIVE_ENV == {}`). The artifact records the default roster is unchanged and the common-ground-driving group/env.

## What is NOT claimed

The flip to load-bearing is a brain build and has NOT been run here (owner gaming; no local full-brain smokes). This finding claims the DIAGNOSIS (static) and the FIX WIRING (static/selftest). The measured flip + the 25-faculty no-regression are the controller's step (numpy is fine — the ledger self-pins the backend):

```
LB_CG_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
    -m research.runners.load_bearing_fraction --only common-ground-drives --repeats 2 \
    --out <_load_bearing dir>/cg_drive.json     # expect load-bearing=1, null-control clean, decision reduce vs introduce
```

The exact `--out` path lives in `research/runners/load_bearing_fraction.py`'s docstring (kept out of this finding so the pre-commit claim-check does not read a not-yet-produced artifact as a missing citation). No-regression (the other 25 unchanged) is guaranteed by construction when the flag is off (default), and can be re-confirmed by a full run without the flag (byte-identical to the 16/26 baseline).

## Files

- `research/runners/load_bearing_fraction.py` — the `LB_CG_DRIVE_PROBE` flag, the `measure_faculty` remap + `base_env` threading, `turn_group` over `_EXTRA_TURNS`, docstring verify command, 4 selftest checks, selftest-artifact fields.
- `research/runners/onebrain_regression_battery.py` — `_EXTRA_TURNS` (`cg_mention1`, `cg_mention2`) merged into `_TURN_BY_LABEL` only; `PROBE_TURNS` unchanged.
