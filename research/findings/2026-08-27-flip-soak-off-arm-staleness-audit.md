---
type: finding
status: live
date: 2026-08-27
mechanism: flip-soak-off-arm-staleness
lane: integration
seeds: [42]
seed-waiver: A static-code + direct-function audit (does each soak's OFF arm still read OFF given each flag's
  CURRENT production default?), not a stochastic effect-size measurement — every claim is a deterministic
  presence/absence read of a reader function's return value, confirmed by direct import, not inferred from
  prose. A seed population measures nothing here.
artifacts:
  - research/findings/raw/2026-08-27-flip-soak-off-arm-staleness-audit.json
runner: research/runners/_bg_action_selection_flip_soak.py, research/runners/_value_choice_flip_soak.py,
  research/runners/_vision_identity_flip_soak.py, research/runners/_gnw_multistep_reentrant_deliberation_derisk.py,
  research/runners/_continuous_ideation_verify.py
---

# Flip-soak OFF-arm staleness audit: 5 more soaks were comparing ON-vs-ON, fixed; 2 deeper production-default
# conflicts found and flagged separately

**One-line:** the `_spiking_mouth_recall_soak.py` bug (`os.environ.pop(FLAG)` as the OFF arm, silently reading ON
once `FLAG`'s production default flipped) was NOT isolated. Auditing every `research/runners/` soak/verify that
pops a `BRAIN_*` flag against every flag's CURRENT production default found **5 more soaks with the identical
bug**, now fixed the same way (explicit `="0"`, matching the reference fix). ~22 other pop-based soaks were
checked and are genuinely safe (they either already pin explicit `"0"`/`"1"` for the OFF/ON arms, or the flag they
pop for "default" genuinely still defaults OFF/matches). Two additional, DISTINCT production-code bugs were found
along the way — a reader's own default contradicting its docstring, and a flip that is silently INERT — flagged
as follow-ups, not fixed here (out of scope for a soak-only mechanical fix).

## Method

1. `grep -rn 'os\.environ\.pop("BRAIN_' research/runners tests tools` — 27 files, ~45 distinct `BRAIN_*` flags
   popped as a candidate OFF/reset mechanism.
2. For every popped flag, located the reader function actually on that soak's own call path (the organ module's
   own function, `webapp/server.py`'s inline anchor check, or both when they differ) and read/ran it to get the
   CURRENT default — many soaks pop a flag deliberately BECAUSE it is meant to mean "leave at default" (correct
   when the default is what the soak's comment claims), so the check is per-callsite, not per-flag-name.
3. Classified each pop call as either (a) the OFF-arm of an ON-vs-OFF comparison, or (b) inert cleanup/reset with
   no read immediately downstream — only (a) can be stale.
4. Confirmed every classification EMPIRICALLY (`research/findings/raw/2026-08-27-flip-soak-off-arm-staleness-audit.json`):
   imported each reader with the flag genuinely unset and printed its return value, mirroring the reference fix's
   own confirmation method ("confirmed directly: `recall_mouth_enabled()` with the var unset now reads True").

## Full table

`pop`/`unset` = the code relies on the flag being absent from `os.environ`. `explicit-0` = the code writes the
string `"0"` (or equivalent falsy token) itself. AT-RISK = the OFF arm is (a) from step 3 above AND the flag's
current default (per step 4) is ON, so "unset" now reads ON and the comparison collapses to ON-vs-ON.

| runner_file | flag | OFF-arm mechanism | current production default | AT-RISK? |
|---|---|---|---|---|
| `_spiking_mouth_recall_soak.py` | `BRAIN_SPIKING_MOUTH_RECALL` | was `pop` | ON (`_RECALL_MOUTH_DEFAULT_ON`) | **already fixed 2026-08-27** (the seed case; reference pattern) |
| `_bg_action_selection_flip_soak.py` | `BRAIN_BG_SELECT` | was `pop` (6 call sites in `run_handler_no_regression`) | ON (`_BG_SELECT_DEFAULT_ON`, wave-3) | **YES — fixed this session** |
| `_value_choice_flip_soak.py` | `BRAIN_VALUE_CHOICE` | was `pop` (`_set_flags(on=False)`, the ordinary-panel HARD gate) | ON (`_VALUE_CHOICE_DEFAULT_ON`, wave-1/2) | **YES — fixed this session** |
| `_vision_identity_flip_soak.py` | `BRAIN_VISION_IDENTITY` | was `pop` (`run_handler_no_regression`, real-handler path) | ON (`_VISION_IDENTITY_DEFAULT_ON`, wave-1/2, via `webapp.server._vision_identity_on()` — the function this soak's real-handler call actually exercises) | **YES — fixed this session** |
| `_gnw_multistep_reentrant_deliberation_derisk.py` | `BRAIN_GNW_MULTISTEP` | was `pop` (`_gate_multistep_arms` line labeled "# OFF"; `_handler_multistep`'s `off = _arm("ms_off", {})`) | ON (`_GNW_MULTISTEP_DEFAULT_ON`; `webapp/gnw_multistep_deliberation.py`'s own reader also defaults `"1"`) | **YES — fixed this session, 2 call sites** |
| `_continuous_ideation_verify.py` | `BRAIN_CONTINUOUS_IDEATE` | was `pop` (`_integration`, labeled "# (A) OFF") | ON in the ACTUAL code (`ideation_enabled()` reads `os.environ.get(..., "1")`), though the function's own docstring says "Default-OFF anchor" — see "Related discovery 1" below | **YES — fixed this session** |
| `_causal_whatif_production_organ_verify.py` | `BRAIN_CAUSAL` | explicit `"1"`/`"0"`; `pop` only after the comparison (cleanup) | ON | safe |
| `_verify_d5_episodic_organ.py` | `BRAIN_EPISODIC` | direct-function test, explicit `env={}` deliberately expecting `True` | ON | safe (correctly tests the real default) |
| `_gnw_deliberation_wired_verify.py` | `BRAIN_GNW_DELIBERATE` | `pop` = the ON/"default" arm (commented `# default-ON`); OFF arm uses explicit `"0"` | ON | safe |
| `_b3_noncontradiction_organ_verify.py` | `BRAIN_NONCONTRADICTION_GATE` | direct-function test, deliberately expects `True` on unset | ON | safe |
| `_w4_pragmatic_belief_production_verify.py` | `BRAIN_PRAGMATIC` | explicit `"1"`/`"0"` throughout; `pop` only as cleanup | ON | safe |
| `_d6_multiref_wm_production_verify.py` | `BRAIN_MULTIREF` | direct-function test, deliberately expects `True` on unset | ON | safe |
| `_prospective_memory_production_verify.py` | `BRAIN_PMEM` | explicit `"1"`/`"0"` throughout; `pop` only as cleanup | ON | safe |
| `_vocab_agnostic_spiking_generation_production_organ_verify.py` | `BRAIN_SPIKING_DRAW` | `pop` = the ON/"default" arm (organ install check expects `on is True`); OFF arm uses explicit `"0"` | ON | safe |
| `_gateB_repair_production_verify.py` | `BRAIN_REPAIR` | `pop` = the ON/"default" arm (docstring: "Default-ON"); OFF arm uses explicit `"0"` | ON | safe |
| `_dmn_consolidated_selfinit_verify.py` | `BRAIN_SELF_INITIATE_CONSOLIDATE` | `pop` = the "consolidate=True" arm (matches default); `"0"` for the lesion-equivalent arm | ON | safe |
| `_affect_tone_spiking_mouth_fix_verify.py` | `BRAIN_SPIKING_MOUTH_MOOD` | `pop` = the load-bearing intact arm (matches default ON); explicit `"0"` for the lesion arm | ON (`_MOUTH_MOOD_DEFAULT_ON`) | safe (this IS the dedicated fix-verify for the sibling mouth bug) |
| `_generate_channel_wiring_verify.py` | `BRAIN_GENERATE_CHANNEL` | `pop` commented `# default = channel ON`; OFF arm uses explicit `"0"` | ON | safe |
| `_gnw_bus_default_flip_verify.py` | `BRAIN_GNW_BUS`, `BRAIN_GNW_BUS_HOST` | `pop` used as OFF for both | OFF (both; unchanged) | safe |
| `_gnw_bus_shadow_production_verify.py` | `BRAIN_GNW_BUS` | `pop` used as OFF | OFF | safe |
| `_gnw_three_organ_bus_verify.py` | `BRAIN_GNW_3ORGAN` | `pop` used as OFF (matches `three_organ_enabled()`, the function `install_three_organ_gate()` actually calls) | OFF *in the function this soak calls* — see "Related discovery 2" below for a deeper, DIFFERENT bug | safe (for this soak's own claim) |
| `_knowledge_scale_100k_production_verify.py` | `BRAIN_SPARSE_INDEX_RETRIEVAL` | `pop` commented "today's default"; explicit `"1"` tested separately (and shown architecturally inert) | OFF | safe |
| `_comprehension_learned_animacy_wire_verify.py` | `BRAIN_LEARNED_ANIMACY_CUE` | `pop` used as OFF (`_clear_flags`) | OFF | safe |
| `_self_initiated_production_verify.py` | `BRAIN_SELF_INITIATE_LESION` | `pop` = cleanup after explicit `"1"` lesion arm | OFF (lesion, by construction) | safe |
| `_surprise_organ_homeostat_production_verify.py` | `BRAIN_SURPRISE_HOMEOSTAT`, `BRAIN_SURPRISE_LESION` | explicit `"1"`/`"0"` throughout; `pop` only as cleanup | ON / OFF(lesion) | safe |
| `_da_encoding_homeo_trigger.py`, `_da_encoding_138_sleep_trigger_derisk.py`, `_da_encoding_flip_verify.py`, `_da_encoding_wired_verify.py` | `BRAIN_DA_ENCODING`, `BRAIN_DA_ENCODING_SLEEP_TRIGGER`, lesion flags | `pop` for the DA_ENCODING "default" arm (matches ON) or for SLEEP_TRIGGER (matches its own OFF default); explicit `"0"`/`"1"` for the real OFF-arm comparisons — `_da_encoding_wired_verify.py` was PINNED to explicit `BRAIN_DA_ENCODING=0` on 2026-08-25 ahead of its own flip, per its own comment | ON (`BRAIN_DA_ENCODING`) / OFF (`SLEEP_TRIGGER`) | safe |
| `_gnw_global_stop_flip_soak.py` | `BRAIN_GNW_STOP_LESION` only | never touches the main `BRAIN_GNW_STOP` flag at all (organ-level `.run(lesion=...)` calls only) | ON (`_GNW_STOP_DEFAULT_ON`) | not applicable — this soak does not test the flag's on/off state |
| `_continuous_default_flip_soak_cupy.py` | `BRAIN_CONTINUOUS_DRIVES` | `pop`, paired with explicit `BRAIN_CONTINUOUS="0"` in the same OFF arm | ON (`_CONTINUOUS_DRIVES_DEFAULT="1"`) | moot, not at-risk — `webapp/server.py`'s own comment confirms this block is "inert whenever the continuous engine itself is off (`BRAIN_CONTINUOUS=0`)", so `BRAIN_CONTINUOUS_DRIVES`'s value cannot affect this soak's OFF arm regardless |
| `_da_encoding_138_sleep_trigger_derisk.py` (lesion), `_causal_whatif...` (lesion), `_gnw_deliberation_wired_verify.py` (lesion), `_gnw_multistep...` (lesion), `_vision_identity...` (lesion), `_bg_action_selection...` (lesion), `_value_choice...` (lesion), `_d6_multiref...` (lesion), `_pmem...` (lesion), `_spiking_draw...` (lesion), `_noncontradiction...` (lesion), `_pragmatic...` (lesion) | every `*_LESION` flag popped in the files above | `pop` used as the not-lesioned/intact state | OFF (every lesion flag checked defaults to "not lesioned" when unset — a production wave-flip turns a FACULTY on, never a LESION) | safe by construction (out of scope for this bug class: a wave-flip never targets a lesion flag's own default) |
| `_nav_sc_popvector_readout_derisk.py`, `_nav_sc_drive_reorient_derisk.py`, `_merged_neural_reward_validate.py`, `tools/v14_stageB_campaign_supervisor.py`, `tests/test_seam_contracts.py`, `tests/test_g11_bg_runner_flags.py` | `SC_*`, `SIM_BACKEND`, `SIM_NO_PROVENANCE` | `pop` | n/a | out of scope — not `BRAIN_*` faculty flags, no production-default-flip mechanism applies |

## AT-RISK list (fixed this session)

1. `research/runners/_bg_action_selection_flip_soak.py` — flag `BRAIN_BG_SELECT`
2. `research/runners/_value_choice_flip_soak.py` — flag `BRAIN_VALUE_CHOICE`
3. `research/runners/_vision_identity_flip_soak.py` — flag `BRAIN_VISION_IDENTITY`
4. `research/runners/_gnw_multistep_reentrant_deliberation_derisk.py` — flag `BRAIN_GNW_MULTISTEP`
5. `research/runners/_continuous_ideation_verify.py` — flag `BRAIN_CONTINUOUS_IDEATE`

**Fix applied (matches the reference `_spiking_mouth_recall_soak.py::_set_flag` pattern):** each soak's OFF-arm
`os.environ.pop(FLAG, None)` call site is now `os.environ[FLAG] = "0"`, with a one-line comment naming the
default that made `pop` stale. Cleanup-only `pop` calls (no read immediately downstream, e.g. lesion-flag resets,
end-of-function teardown) were deliberately left untouched — converting them would be a no-op (harmless either
way) but changes lines the bug does not touch, and the task scope is the minimal obviously-correct fix.
Confirmed by direct import (see `reader_returns_with_flag_explicit_0` in the JSON artifact): every fixed flag's
`=0` reads False through the exact function the soak's own call path invokes. **None of the 5 soaks were re-run
end-to-end in this session** (that is the natural verification follow-up — this audit is the instrument fix, not
a fresh GO/NO-GO verdict on the underlying faculties).

## Related discovery 1 (NOT fixed here — production reader default contradicts its own docstring)

`webapp/continuous_engine.py::ideation_enabled()`:
```python
def ideation_enabled() -> bool:
    """Default-OFF anchor. `BRAIN_CONTINUOUS_IDEATE` in {1,true,on,yes} arms the between-turn IDEATION mode. ...
    """
    return os.environ.get("BRAIN_CONTINUOUS_IDEATE", "1").strip().lower() in ("1", "true", "on", "yes")
```
The docstring says "Default-OFF anchor" and `webapp/server.py`'s own comment at the call site also says
"DEFAULT-OFF: `ideation_enabled()` reads `BRAIN_CONTINUOUS_IDEATE` default '0'" — but the actual fallback string
passed to `.get()` is `"1"`, which is truthy, so the function returns `True` (ON) when unset (confirmed directly,
see artifact). This means `BRAIN_CONTINUOUS_IDEATE` is currently ON by default in production RIGHT NOW, contrary
to what both the function's own docstring and its caller's comment claim. This looks like a copy-paste default
(most sibling flags in this file are genuinely default-ON and use the `"1"` fallback pattern) rather than an
intentional flip. Resolving it requires a judgment call outside this audit's scope (is "Default-OFF" the correct
intent, in which case the fallback string is a one-character bug, or was this always meant to ship ON and only
the docs never caught up) — flagged as a follow-up, not fixed here.

## Related discovery 2 (NOT fixed here — a 2026-08-21 default-ON flip is currently INERT in production)

`webapp/server.py:4151` (comment: "2026-08-21 FLIPPED default-ON... `BRAIN_GNW_3ORGAN=0` is the byte-identical
escape"):
```python
if os.environ.get("BRAIN_GNW_3ORGAN", "1").strip().lower() in ("1", "true", "on", "yes"):
    ...
    _gnw_3organ_mod.install_three_organ_gate(chat)
```
but `webapp/gnw_three_organ_bus.py::install_three_organ_gate()` immediately re-checks its OWN module-level
`three_organ_enabled()`:
```python
def three_organ_enabled() -> bool:
    """... DEFAULT-OFF (unset -> OFF): ..."""
    return os.environ.get("BRAIN_GNW_3ORGAN", "").strip().lower() in ("1", "true", "on", "yes")

def install_three_organ_gate(chat, *, seed: int = 42) -> bool:
    if not three_organ_enabled():
        return False
    ...
```
So with `BRAIN_GNW_3ORGAN` genuinely unset (the ordinary case): `server.py`'s outer gate reads `"1"` (True) and
enters the block, but `install_three_organ_gate()`'s own inner check reads `""` (False) and returns `False`
immediately — **the 2026-08-21 flip has installed nothing in production since the day it landed**, because the
organ module's own enable-check was never updated to match the anchor that was supposed to flip it. This is the
exact "faculty claims default-ON but is structurally hollow" class the project's own memory flags
("FACULTIES MUST DRIVE, NOT OBSERVE"). Confirmed empirically (see artifact:
`three_organ_enabled()` returns `False` with the flag unset, while `server.py`'s own inline check on the identical
unset state returns `True`). This is a live production correctness bug, not a soak-staleness bug — the soak
(`_gnw_three_organ_bus_verify.py`) is actually *safe* under this audit's narrow definition because its `pop`-based
"OFF" assumption matches the function it invokes; the deeper bug is that this makes the soak's own "flag OFF is a
no-op" claim tautological rather than informative about production. NOT fixed here — the intended repair (make
`three_organ_enabled()`'s own default match `server.py`'s anchor, or route `server.py` through
`three_organ_enabled()` directly instead of duplicating the check) requires understanding intent and is a
production-code change (`webapp/`), out of scope for this mechanical soak-only audit. Flagged as a follow-up task.

## Why this happened at this scale

5 more instances of an identical, already-diagnosed bug class, discovered by mechanically cross-referencing every
soak's popped flag against its actual current reader default, confirms the root cause named in the 2026-08-27
FAILURE_LOG entry: wave-1/2/3 flipped ~12 flags' production defaults from OFF to ON across 3 commits
(`ba02aca94`, `6488c2137`, `75a3a96ee`, and others) that updated the reader/anchor + the ledger, but did NOT
systematically re-check every soak that gates that same flag for a stale `unset==OFF` assumption. The soaks that
survived did so because their authors happened to already write the OFF arm as explicit `"0"`, or wrote the
direct-function tests to assert the CORRECT (then-and-now) default — not because of any enforced convention.

## Follow-up (not built this session, per the 2026-08-27 FAILURE_LOG candidate)

A `tools/gates/` check that flags any `_*_soak.py`/`_*_flip*.py`/`_*_verify.py` under `research/runners/` whose
`os.environ.pop("BRAIN_*", None)` call sits immediately before a read used as the file's own labeled "OFF" arm,
while the flag's owning module's default currently resolves to True, would catch this class going forward. This
audit's own table (which call sites are OFF-arm vs. cleanup, per flag) is exactly the classification such a gate
would need to automate; building it is deferred as a separate task.
