---
type: finding
status: live
date: 2026-08-20
mechanism: gnw-thought-swap-observer-retired
lane: integration
integration_faculty: swap-drives-response
---

# RETIRE SUPERSEDED LEGACY — the #77 GNW thought-swap OBSERVER fallback removed from the live `/api/brain-chat` handler (superseded by the #85 swap-drives DRIVER). Byte-identical-live vs HEAD: all 16 fast+rich production turns unchanged (combined md5 `92f157a1`). One clearly-dead scaffold retired; the scan surfaced no other unambiguous default-path legacy.

**Date:** 2026-08-20 · **Backend:** CPU (numpy) · **No `sim/` edit** (`git diff sim/` empty). · **Scope:** `webapp/server.py` handler wiring + the ledger note only; no faculty logic changed.

## What was retired

`webapp/server.py` `brain_chat` held a mutually-exclusive swap block:

```
if _swap_drives_on():            # board #85, DEFAULT-ON
    swap_drives_chat.observe_turn(...)   # DRIVES the reply (topic-transition lead) + reuses the #77 machinery
elif _GNW_SWAP_DEFAULT_ON or _gnw_swap_flag_on():
    gnw_thought_swap.observe_turn(...)   # board #77 OBSERVER — metadata only, no lead
```

The `elif` is the OLDER board-#77 observe-only swap tracker (`gnw-thought-swap`, `on_by_default: NO`). It is **superseded** by board-#85 `swap-drives-response` (`webapp/swap_drives_chat.py`, `on_by_default: YES`), which is a **strict superset**: `swap_drives_chat.observe_turn` **imports and calls** `gnw_thought_swap.observe_turn` internally (reusing the exact same `ThoughtSwapWorkspace` / `run_intention_swap` / mismatch-detector machinery, with the lesion threaded) and **additionally** makes the neural swap verdict load-bearing (a topic-transition lead). Because swap-drives is default-ON, the `elif` was reachable only when swap-drives was turned OFF (`BRAIN_SWAP_DRIVES=0`) **and** `BRAIN_GNW_SWAP=1` — a review-only combo, never the production default. And even then the observer only ever attached additive metadata; it never changed `answer` or any content field.

**Change:** the `elif` branch is removed, so `swap_drives_chat` is the SOLE swap path (exactly one neural swap per turn, no double-advance of the workspace). `BRAIN_SWAP_DRIVES=0` now simply runs no swap at all — the correct "off". `webapp/gnw_thought_swap.py` is **kept on disk** (still imported by `swap_drives_chat.py`, the live path). `_GNW_SWAP_DEFAULT_ON = False` is **retained unchanged** in `server.py` as the ledger's `default_anchor` for the `gnw-thought-swap` row and it still gates the additive DEFAULT-OFF `gnw_swap` observability key (`BRAIN_GNW_SWAP=1`, which now surfaces swap-drives' own per-turn `_last_gnw_swap` read).

## The byte-identical-live proof (vs HEAD)

The removed `elif` is unreachable while `_swap_drives_on()` is True (the production default), so the live default turn is byte-identical by control flow. Confirmed empirically through the REAL `/api/brain-chat` handler:

- **Config held CONSTANT before + after** the edit: swap-drives DEFAULT-ON, `BRAIN_GNW_SWAP` unset, heavy orthogonal organs isolated. A conversation of **8 turns × 2 return paths** (fast single-fact + rich): establish → hold → topic-change SWAP → hold → SWAP → no-topic hold → SWAP-back → abstain.
- **Determinism:** two pre-edit HEAD runs produced identical `combined_md5 = 92f157a1ac3911c75c994f7197406d5e`.
- **Result:** the post-edit run reproduced `combined_md5 = 92f157a1ac3911c75c994f7197406d5e`; **all 16 per-turn full-response md5s match (0 mismatches)**. The only file-level difference was the harness's own wall-clock `ts` field. Artifacts: `research/findings/raw/_retire_swap_observer/baseline_HEAD.json`, `research/findings/raw/_retire_swap_observer/after_retirement.json`, `research/findings/raw/_retire_swap_observer/byte_identical_proof.json`.
- The observed swap behaviour is intact (topic-change turns SWAP with an `On <topic>, then —` lead; same-topic/no-topic turns HOLD silently; unknown topic abstains) — unchanged from HEAD.

The official swap-drives verify (`research/findings/raw/_swap_drives_chat/verify_swap_drives.py`) is unaffected by this change **by construction**: it disables `BRAIN_GNW_SWAP` (and never turned swap-drives off), so it never exercised the removed observer `elif` — its A/B/C GO (verdict GO at the #85 landing, `research/findings/raw/_swap_drives_chat/verify.json`) is over swap-drives, which is byte-unchanged here (no edit to `swap_drives_chat.py` or `gnw_thought_swap.py`). <!--derived--> (A post-edit re-run of that verify was not captured — an OOM crash interrupted the agent before it landed; the by-construction argument + the 16-turn byte-identical-live proof above are the evidence.)

## Ledger

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` row `gnw-thought-swap` is **KEPT** (the ratchet count `total_faculties: 38` is unchanged) with `on_by_default: NO` unchanged; its `faculty:` note now records that it is SUPERSEDED by `swap-drives-response` and that the observe-only fallback is retired from the default handler. The `default_anchor` (`_GNW_SWAP_DEFAULT_ON`, off_value `False`, count 1) still resolves — production_integration gate Check A/C PASS, selftest PASS.

## The scan (other superseded-and-still-present legacy)

Retired only the UNAMBIGUOUS case (above). The rest of the scan found **nothing else clearly dead in the default path**:

- **(a) RETRACTED findings' code in the default path** — `docs/RETRACTED.md` holds 4 rows, all calibration/attribution corrections of research findings; none names live handler wiring still present in the default path. Nothing to retire.
- **(b) other mutually-exclusive observer/driver pairs** — the swap block was the ONLY `if newer … elif older-observer` in the handler (grep of every `observe_turn` + `elif` in `brain_chat`). `affect-drives-response` and `da-mode-drives-response` are standalone default-on drivers with no dead observer fallback. `affect-coloring` (BRAIN_AFFECT) and `affect-drives-response` (BRAIN_AFFECT_DRIVES) are BOTH default-on and explicitly ORTHOGONAL (prose-manner vs a lead), not a supersede pair — left untouched.
- **(c) `on_by_default: NO` rows superseded by a default-on replacement** — the only such case is `gnw-thought-swap` (retired here). The other `on_by_default: NO` rows (`neural-render`, `self-model-reward-residual`, `perception-motor`, `d5-live-consolidation`, `tiered-knowledge-ltm`) are genuine work-in-progress faculties awaiting their own flip, NOT superseded scaffolds — left untouched.

No ambiguous candidates required an owner decision.
