---
type: finding
status: live
lane: integration
date: 2026-08-26
---

# Basal-ganglia action selection (SPEAK vs STAY-SILENT): a default-OFF production consumer for the Gate-A v2 selector GO, verified load-bearing (GO)

**Status: GO — a production-integration BUILD. The faculty is DEFAULT-OFF (`BRAIN_BG_SELECT`); it is NOT yet on by
default. The parent flips it default-on after the 6-seed pool soak passes.** The claim here is narrow and checked: the
consumer exists, flag-OFF is byte-identical to today, and the ON path is load-bearing (verified by the source finding's
OWN Gate-A anti-cheats — remove shared arousal OR the D1->GPi direct path and the selected-action change vanishes).

**Date:** 2026-08-26 (autonomous, production-integration).
**Faculty:** the Gate-A v2 two-channel spiking basal-ganglia vocal action selector — 4-seed GO
(`research/findings/2026-08-03-neural-vocal-selector-gateA-v2-4seed-GO.md`, runner
`research/runners/_vocal_action_selector_gate.py`).
**Flag:** `BRAIN_BG_SELECT` (default **OFF**; the parent flips default-on after the 6-seed pool soak passes).
**Files:** `research/runners/bg_action_selection_production_organ.py` (the production organ, reuse-by-import), a guarded
BEGIN/END block in `webapp/server.py::brain_chat` (marker `faculty:bg-action-selection`),
`research/runners/_bg_action_selection_flip_soak.py` (the soak gate). **NO `sim/` edit.**

## What was wired

A discrete chat action decision — should the brain SPEAK this turn, or STAY-SILENT (hold, emit nothing salient)? — is
routed through a genuine two-channel basal-ganglia RACE instead of a host `if`. The two candidate actions are the two
striatal channels of the reused Gate-A v2 selector: channel 0 = SPEAK, channel 1 = STAY-SILENT. The composer hands the
selector a per-candidate SALIENCE (SPEAK salience rises with answerable content; STAY-SILENT salience is high only on a
content-empty turn). Two drives combine at the striatum: (1) SHARED practice arousal -> the cortical proposal
populations -> a proposal->D1 barrage that brings BOTH channels' MSNs toward threshold (the enabling drive); (2) the
per-candidate salience -> a modulatory, arousal-gated excitability bias on that candidate's D1 MSN pool (which channel
the shared barrage pushes over first). The FIRST channel whose motor pool crosses the GPi->thalamus disinhibition
commit burst IS the selected action — a real race, **NOT** an argmax over rates. The organ is a reuse-by-import of the
de-risk selector (`build_selector_bridge` + the v2 topology + the `vocal_selector_direct_path` transmission gate).

In the live handler the selector is CONSULTED only on a content-empty turn (STAY-SILENT is a genuine contender there —
a normal content message always favors SPEAK, so the selector is never even called on it), and it SHORT-CIRCUITS the
turn with a brief HOLD line **only** when the BG race COMMITS to STAY-SILENT. A SPEAK commit, a non-commit, or an
ordinary content turn all fall through to the normal path.

**BRAIN-BASED-ONLY boundary** (CLAUDE.md standing standard): host code is legitimate here only for the ENVIRONMENT/body
— the per-candidate SALIENCE is a cortical/neuromodulatory afferent the composer supplies (the same declared boundary
the SVO parser / the vision percept occupy), and surfacing a STAY-SILENT commit as a hold line is the body's
articulation layer. Everything between the salience input and the selected action — the striatal D1 channels, the
D1->GPi direct-path disinhibition, the GPe/STN indirect path, the GPi->thalamus gate, the thalamo-cortical commit
burst, the cross-channel commit inhibition — is neurons/synapses on a real `SimulationBridge` (no numpy argmax). Which
action wins is the substrate's race, not a host max.

## Flag-OFF is byte-identical

`BRAIN_BG_SELECT` unset -> the wiring block's cheap env read is the ONLY thing that runs; it imports nothing and returns
nothing. When ON, the block short-circuits only when `decide_action(msg)` returns a STAY-SILENT commit, and
`decide_action` returns None on any turn whose salience does not make STAY-SILENT the contender (every ordinary content
turn) — WITHOUT building a bridge or touching RNG (the `silent_sal > speak_sal` check short-circuits before the organ
is consulted). So an ordinary content turn is a pure no-op with ZERO code-path difference. Verified deterministically at
the organ boundary: `decide_action("what does the cat eat?")` and three other ordinary messages all return None
(`organ_content_inert=True`), and the tiny-demo handler build is itself deterministic across fresh sessions
(`build_deterministic=True`, a two-fresh-session hash-equal check) — so a flag toggle on a content turn cannot change a
byte.

## LOAD-BEARING (vary the salience -> the action differs; lesion the cascade -> it vanishes, byte-equal to flag-off)

The coupling is salience -> the spiking BG race -> the selected action. The source finding's OWN Gate-A anti-cheats are
the lesion oracle, reused verbatim:

- **VARY the salience** (intact selector, per seed): SPEAK-favored salience (speak=1, silent=0) -> the race commits to
  **SPEAK**; STAY-SILENT-favored (speak=0, silent=1) -> the race commits to **STAY-SILENT**. The majority winner FLIPS
  with the salience on **every one of the 6 seeds** (PART A). Through the production entry point,
  `decide_action("...")` returns a STAY-SILENT commit (`organ_dots_intact=STAY_SILENT`). The winner is the SPIKING race
  outcome, not a host max.
- **LESION: no shared arousal** (`BRAIN_BG_SELECT_LESION=arousal`): remove the shared practice-arousal state -> no
  proposal->D1 barrage AND the arousal-gated salience bias is withdrawn -> the D1 MSNs cannot commit at any salience ->
  `decide_action()` returns None -> the turn falls through to the host path. Commit rate floors to **0.00 on every
  seed** (the source finding's `arousal_is_load_bearing` control, reproduced). At the handler this is byte-identical to
  flag-OFF on the '...' turn (PART B).
- **LESION: no direct path** (`BRAIN_BG_SELECT_LESION=direct_path`): cut the D1->GPi transmission gate -> GPi is never
  inhibited -> the thalamus is never disinhibited -> no commit at any salience. Commit rate floors to **0.00 on every
  seed** (the source finding's `direct_path_is_load_bearing` control, reproduced).

Either lesion collapses the commit, so the selected-action change is attributable to the BG cascade (shared arousal +
the gated direct path), **not** to a host argmax. Aggregate attribution: intact commit **1.000** vs no-arousal
**0.000** (100% attributable) vs no-direct-path **0.000** (100% attributable).

## Selector — 6-seed physiology (the pool soak, PART A)

`SIM_BACKEND=numpy python -m research.runners._bg_action_selection_flip_soak --seeds 42 43 44 100 101 102`
GO gate (per seed): SPEAK-favored commits to SPEAK on >= 0.75 of single races AND STAY-SILENT-favored commits to
STAY-SILENT on >= 0.75 AND the majority winner FLIPS between the two salience conditions AND both lesions floor
(commit rate <= 0.05, either salience). Single-race salience gain 600 pA (calibrated: reliable selection with arousal,
zero commit without it). Reads/aggregates of `research/findings/raw/_bg_action_select_prodflip/soak_summary_6seed.json`.
<!--derived-->

| seed | speak-fav -> SPEAK | silent-fav -> SILENT | flip | no-arousal commit | no-direct-path commit | seed_ok |
|---|---|---|---|---|---|---|
| 42  | 1.00 | 0.92 | yes | 0.00 | 0.00 | yes |
| 43  | 1.00 | 1.00 | yes | 0.00 | 0.00 | yes |
| 44  | 1.00 | 1.00 | yes | 0.00 | 0.00 | yes |
| 100 | 0.83 | 1.00 | yes | 0.00 | 0.00 | yes |
| 101 | 1.00 | 1.00 | yes | 0.00 | 0.00 | yes |
| 102 | 1.00 | 1.00 | yes | 0.00 | 0.00 | yes |
| **agg** | — | — | **6/6** | **0.000** | **0.000** | **6/6** |

**GO:** every seed selects the salience-favored action (>= 0.83) with the winner FLIPPING as salience flips, and BOTH
lesions floor the commit to 0.00 on every seed. Intact commit rate is 1.000 across seeds; both lesions 0.000. The BG
cascade — not a host max — makes the SPEAK-vs-STAY-SILENT choice.

## Soak (the gate the parent runs before flipping default-on)

`SIM_BACKEND=numpy python -m research.runners._bg_action_selection_flip_soak --seeds 42 43 44 100 101 102`
- **PART A** — the selector 6-seed physiology above (pool-friendly numpy; the core gate).
- **PART B** — chat no-regression. Two layers. (1) ORGAN-LEVEL (deterministic, backend-independent — the robust proof;
  captured in `research/findings/raw/_bg_action_select_prodflip/organ_partb.json`):
  `organ_content_inert=True` (four ordinary messages -> `decide_action` None -> byte-identical off), `organ_dots_intact=
  STAY_SILENT` (the '...' turn commits the hold), `organ_dots_arousal_vanish=True` and `organ_dots_direct_vanish=True`
  (either lesion -> None -> the hold vanishes). (2) HANDLER end-to-end through the real `brain_chat` (stub renderer), on
  FRESH sessions per turn so an OFF/ON pair is like-for-like (`build_deterministic=True`): flag-ON == flag-OFF on the
  ordinary content turns (full-JSON equal; no `bg_select` key), flag-ON fires STAY-SILENT on '...', and '...' flag-ON +
  `BRAIN_BG_SELECT_LESION=arousal` == flag-OFF. The runner gates on the organ layer AND (when the build is
  deterministic) the handler layer; it degrades to a reported SKIP on a bare pool node. NOTE: the handler layer is
  COST-HEAVY — each fresh-session turn rebuilds the tiny-demo composer's per-decode sub-bridges (~45-90 s/turn, ~10 min
  for the 13 turns), so PART A + the organ layer are the fast core gate; run `--selector-only` to skip the handler
  layer.

## Honest scope (do NOT overclaim)

- The routed decision is **binary** SPEAK-vs-STAY-SILENT, which is exactly the reused selector's TWO channels. The
  general "choice among N candidate utterance types" would need N competing channels; the de-risk topology is 2-channel,
  so N > 2 is out of scope here (a named extension, not a claim).
- The per-candidate SALIENCE is a host-computed cortical afferent (the declared environment/body boundary), narrow by
  design: STAY-SILENT is a genuine contender ONLY on a content-empty turn. The SELECTION among the candidates is the
  spiking race; the salience input is not.
- The salience bias is **arousal-gated** (striatal UP-state / neuromodulatory gating), which is a design choice with
  biological grounding chosen so shared arousal is strictly load-bearing; the alternative ungated additive bias leaked a
  ~12% commit under the arousal lesion on 3/6 seeds and was rejected.
- This establishes selector physiology + a load-bearing, byte-identical-off production wire-in only. It does NOT
  establish reward learning of WHEN to hold, nor that holding on '...' improves conversation — those are separate.
