---
status: live
type: finding
lane: laneC
date: 2026-08-26
integration_faculty: gnw-global-stop
---

# GNW GLOBAL-WORKSPACE STOP reaches the live `brain_chat` handler behind a DEFAULT-OFF flag — a conflict-triggered distributed clear-all that empties the held P1.2 coalition to n_ignited=0 BEFORE the newcomer ignites, so an interrupt gives a CLEAN single-content workspace instead of the recurrence-weakening swap that lets stale content bleed in. Additive, reuse-by-import, NO `sim/` edit; flag-OFF byte-identical; 6/6-seed flip-soak GO. Default-OFF pending the parent's flip after the pool soak. GO.

**Date:** 2026-08-26 · **Backend:** CPU (numpy) · **Verdict:** **GO** (production-organ level, 6/6 seeds) · **No `sim/` edit** (`git diff sim/` empty) · FUNCTIONAL correlate only; NO phenomenal claim.

**Files:** `webapp/gnw_global_stop.py` (NEW — the `_StopWorkspace` warm distributed-workspace holder + `detect_trigger` + `run`/`stop_lead` + `observe_turn`, reuse-by-import), `webapp/server.py` (the `_GNW_STOP_DEFAULT_ON=False` anchor + `_gnw_stop_flag_on()` + one guarded `BEGIN/END`-marked block on EACH of the two return paths, prepending the clearing lead OUTERMOST + attaching the additive `gnw_stop` trace), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `gnw-global-stop`). **Runner:** `research/runners/_gnw_global_stop_flip_soak.py`. **Artifact:** `research/findings/raw/_gnw_global_stop_flip_soak.json`.

**Builds on / reuses:** the de-risk `research/findings/2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md` (the distributed-overwrite GLOBAL-STOP capability, 6/6 GO — a divisively-normalized workspace + a conflict-triggered depression of the shared recurrence drives a co-ignited multi-content workspace to n_ignited=0).
Reuse-by-import from `research/runners/_gnw_distributed_overwrite_workspace_derisk.py` (`build_overwrite_bridge`, `WorkspaceDepression`, `run_conflict_stop`). The trigger reads the already-wired `gnw-deliberation` acc_conflict_gate (`chat._last_gnw_delib`) and the `#85 swap-drives` detector (`chat._last_swap_drives`).

## What it is — the interrupt clear-all the swap could not deliver

The already-wired GNW chain resolves competition per-turn (the ignition bus commits; the acc_conflict_gate abstains on a sustained co-ignition; the #85 swap evicts an incumbent by depressing its OWN recurrence). What it lacked is a DISTRIBUTED clear-all: one control signal driving a co-ignited MULTI-content workspace to n_ignited=0 uniformly.
The de-risk delivered that — a divisively-normalized workspace (a shared inhibitory `norm_pool` returns broad conductance feedback, the shared divisor, so no content sits in a deep isolated self-sufficient basin) plus a conflict-triggered Tsodyks-Markram depression of the SHARED recurrence: because every ignited pattern draws on the shared recurrence, depleting it de-ignites ALL content uniformly. It is gain-withdrawal from a shared resource, not a large external inhibition, so there is no driving-force reversal / rebound.

This organ makes that capability reachable from the live turn. On a strong interrupt / hard topic-break — the gnw-deliberation acc_conflict_gate reporting SUSTAINED multi-candidate co-ignition (`n_ignited >= 2`), OR a hard topic-break the swap detector flags (`swapped == True`) — the held coalition (a stale incumbent + the newcomer == a 2-content conflict) is driven into the distributed workspace and the conflict-triggered depression STOP clears it to n_ignited=0 before the newcomer ignites: a clean single-content workspace, no stale bleed. The clear is the substrate's own global stop, NOT a host state reset (`host_workspace_reset_calls == 0` in the de-risk).

## The coupling (load-bearing, not observe-only)

Mirrors the #85 swap-DRIVES / #84 affect-DRIVES pattern. A CLEAN neural stop (a 2-content conflict `n_pre >= 2` driven to `n_post == 0`) prepends a short clearing lead to the answer (`"Setting the held thread aside — "`), the honest EXPRESSION of the global stop the substrate just performed. No trigger, or a stop that does not reach n_post==0, → NO lead (byte-identical). The FACT after the lead is the SAME gate-matched, moat-verified answer: the stop frames HOW the reply opens, never WHICH fact is true and never whether an unmatched cue abstains. The moat / recall / abstain verdict runs FIRST and unchanged; the content fields (`abstained`, `recalled_svo`, `verified`) are byte-identical with the coupling on or off.

## GO GATE — the 6-seed flip-soak (the flip gate)

`research/runners/_gnw_global_stop_flip_soak.py`, seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism from `cfg.seed`. Per seed it exercises the SAME organ the handler calls (`webapp.gnw_global_stop`). VERDICT: **GO 6/6**. <!--derived-->

| seed | INTACT stop n_pre→n_post | LESION n_pre→n_post | lead-on-stop | lead-vanish-on-lesion | no-trigger→None | determinism | seed_go |
|---|---|---|---|---|---|---|---|
| 42  | 2→0 | 2→2 | yes | yes | yes | yes | GO |
| 43  | 2→0 | 2→2 | yes | yes | yes | yes | GO |
| 44  | 2→0 | 2→3 | yes | yes | yes | yes | GO |
| 100 | 2→0 | 2→2 | yes | yes | yes | yes | GO |
| 101 | 2→0 | 2→2 | yes | yes | yes | yes | GO |
| 102 | 2→0 | 2→2 | yes | yes | yes | yes | GO |

`clean-stop 6/6 · lesion-holds 6/6 · coupling 6/6 · no-trigger byte-identical 6/6 · determinism 6/6`. Flip-gate GO: clean stop on >=5/6 (got 6/6), lesion holds (n_post>=2) on ALL 6, the lead present on a clean stop and absent under the lesion on ALL 6, the no-trigger turn returns None (no key / no lead) on ALL 6, build-twice identical n_post on ALL 6.

## Load-bearing — the de-risk's OWN lesion oracle

The lesion lever `BRAIN_GNW_STOP_LESION=1` ZEROES the shared-resource-depression term (the conflict boost gain → 0), so the conflict-triggered depression never fires.
Organ-level dissociation (seed 42; the same shape on all 6): a two-content workspace under the interrupt goes n_ignited 2→0 with the stop installed (clearing lead present) vs stays 2 co-ignited (n_post=2, the stale content bleeds) with the depression term zeroed → the clean-stop condition (`n_post == 0`) fails → the clearing lead VANISHES and the surface reverts to the byte-identical no-lead baseline.
So the surface change RIDES the SPIKING depression of the shared recurrence, not a host `if interrupt`: zero the depression term and the clearing acknowledgment disappears even though the world input (an interrupt) is unchanged. The newcomer's reply differs with the stop and that difference vanishes under the lesion — the anti-hollow / brain-based proof.

## Flag-OFF byte-identical (by construction)

`BRAIN_GNW_STOP` is DEFAULT-OFF (`_GNW_STOP_DEFAULT_ON=False`; `_gnw_stop_flag_on()` returns False when the env var is unset). With the flag off, the guarded block is fully skipped on both return paths: no workspace is built, no read runs, no `gnw_stop` key is attached, and no clearing lead is prepended → the turn is byte-identical to pre-wiring. When the flag is on but there is no interrupt, `observe_turn` returns None (no key, no lead). The stop workspace also runs on a PRIVATE RNG timeline with the host process-global RNG snapshotted/restored around every read (the #77/#85 footgun), so enabling the organ cannot perturb the downstream RNG-dependent organs — the other response fields stay byte-identical when the parent flips it on.

## Honest residuals (named, not claimed closed)

1. The held coalition is instantiated as host-supplied external drive (a stale incumbent + the newcomer, world/body-legitimate as stimuli, exactly as the swap/deliberation organs drive their workspaces). The CLEAR itself — the drive to n_ignited=0 — is the substrate's own divisive-norm+STD global stop (lesion-proven).
2. The verdict→CLEARING-STRING map is a host conditioned-articulation scaffold (the discourse "mouth"), the owner-sanctioned articulation-crutch pattern (load-bearing on the surface — the lesion collapses the lead). A brain-native spiking discourse-clearing mouth is the next rung.
3. The conflict boost is a host-read margin scaling a neuromodulatory enhancement of the STD (a faithful conflict→neuromodulator effector), host-side until an ACC/BG circuit computes it from synaptic inputs — inherited from the de-risk's remaining-scaffold #2.
4. DEFAULT-OFF: this is a production wire-in behind a reversible flag; the parent flips `_GNW_STOP_DEFAULT_ON` on after the pool soak. This finding reports the organ + wiring + the 6/6 flip-soak, not an on-by-default claim.
5. CO-RESIDENT on its own distributed-workspace bridge (workspace/norm_pool/thal, 440 neurons), not merged onto the recall composer's bridge — rides the one-brain merge.

## Reproduce

`SIM_BACKEND=numpy python -u -m research.runners._gnw_global_stop_flip_soak --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_global_stop_flip_soak.json`
