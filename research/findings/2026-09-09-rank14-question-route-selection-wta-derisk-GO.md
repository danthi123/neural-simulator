---
status: live
type: finding
lane: comprehension-routing
date: 2026-09-09
integration_faculty: question-route-selection
---

# Question-comprehension ROUTE selection is a spiking N-way lateral-inhibition WTA — retires the host if/elif priority cascade that decides WHICH comprehension construction a question dispatches to. 6/6-seed GO: full parity (66/66 real questions, incl. a deliberately ambiguous overlap item), 100% lesion-collapse to the DEFAULT route, per-pathway independence, and a genuine non-ceiling graded sweep.

**Date:** 2026-09-09 · **Backend:** CPU (numpy) · **Verdict:** **GO** (de-risk level, 6/6 seeds) · **No `sim/` edit** (`git diff sim/` empty) · FUNCTIONAL correlate only; NO phenomenal claim.

**Files:** `research/runners/_rank14_question_route_selection_derisk.py` (NEW — battery generation, `host_route_type`/`host_evidence` ground truth, `evidence_to_currents`, `run_trial`, `evaluate_seed`), `research/biology/question-route-selection-wta.md` (NEW — the biology binding). **Artifacts:** `research/findings/raw/_rank14_question_route_wta_6seed.json` (the 6-seed GO gate, provenance-stamped) + `.prov.json` sidecar.

**Reuses by import (NOT reimplemented):** `research/runners/_affect_marker_wta_derisk.py`'s `_build_bridge`/`_pool_rates` (the ALREADY-GO'd, production-flipped N=6 affect-expression-marker cross-inhibition WTA primitive, here instantiated at `n_pools=4`); `research/runners/_gnw_rung2b_sfa_workspace_eviction_derisk.py`'s `_threshold_hash` (determinism); `research/runners/brain_chat_tui.py`'s `_REL_FRONTED_RE`, `_KB_RELATION_PATTERNS`, `_KB_UNDERSCORED_RELATIONS`, `_kb_relation_phrase`, `_KB_RELATION_IDIOMS` (the host's OWN regex feature-extractors and the 29-relation curated table, reused for feature extraction only — see "What stays host code" below). Scoped by `research/coordination/scaffold_retirement_backlog.md` rank-14 ("NL question-routing host comprehension", MED-HIGH/fresh, first attempt).

## Why rank-14, and why not an already-closed item

The backlog's own exclusion list rules out ranks 1/4/5/6/8/10/12/15/16/20 (retired/flipped this week), rank-9
(metacog confidence, an open wall with 3 prior attempts), rank-13 (de-risked but production-flip failed twice),
rank-17 (anaphor-detector→CA3, landed this session, `1a8152a8a`), the habituation-set retirement, and value-choice
recency (already legitimate). Of what remains, rank-14 is explicitly tagged "MED-HIGH · fresh" — no prior finding
or runner had touched it (confirmed: `git log --all --oneline --grep=route` and `find research/findings
-iname '*question-rout*'` turned up only the 2026-09-01 finding that ADDED the KB-relation regex table itself,
not a retirement of its dispatch — see below). Ranks 2/3 are re-verify/hardening of already-wired mechanisms
rather than a fresh build; rank-7 has two prior attempts and a named BOUNDARY; rank-11/23 are already-closed;
rank-18/19/21/22 are either architecture-scale, SlotBinder-attempted-twice-this-week, owner-deprioritized, or
LOW/dormant. Rank-14 was the only unattempted MED/HIGH item.

## What this retires — the dispatch, not the feature extraction

`ChatBrain._extract_route` (`research/runners/brain_chat_tui.py`) decides, in host Python, WHICH of four
comprehension routes handles an incoming question by testing three regex-shaped special cases in a FIXED
PRIORITY ORDER:

```python
_relf = self._relation_fronted_route(question)          # 'what country is chelsea fc from?'
if _relf is not None: return _relf
_kbrel = self._kb_relation_question_route(question)     # 'where was X born?' — 29-relation curated table
if _kbrel is not None: return _kbrel
if len(content) <= 1:
    _defo = self._definitional_copula_route(question)   # 'what is X?'
    if _defo is not None: return _defo
# ... falls through to the ALREADY-NEURAL generic SVO BridgeParser
```

Each regex TEST is a legitimate matched-filter read of the surface string — the same honesty class as the
ACC/BG STOP-trigger de-risk's `n_ignited`/`mm_peak` afferents (themselves host-computed reads of other organs).
What was NOT neural is the COMBINATION: which construction wins when more than one COULD apply, decided
unconditionally by a fixed textual `if`/`elif` order rather than by the strength of the recognized cue. This
finding retires exactly that combination step.

## The mechanism (generalizes an already-validated primitive; see the biology binding for the full citation chain)

Four small excitatory assemblies — GENERIC, DEFCOP, RELFRONT, KBREL — each with its own dedicated fast-spiking
cross-inhibition sub-pool (24 excitatory + 12 FSI neurons per assembly, 144 neurons total), compete under
mutual/reciprocal lateral inhibition. The circuit builder (`_build_bridge`) and the driven-rate reader
(`_pool_rates`) are imported VERBATIM from `_affect_marker_wta_derisk.py` — the already-GO'd, production-
flipped N=6 affect-expression-marker selector — with `n_pools=4` instead of 6; no new competitive primitive was
written. GENERIC additionally carries a constant BASELINE "elsewhere" current (700 pA, vs. 150 pA "off" and
1350 pA "on" for the three exception channels — the SAME OFF/ON regime the affect-marker circuit itself
calibrated) so it wins whenever none of the three construction-specific regexes match; each exception assembly's
current goes ON only when its OWN host regex genuinely matches, mirroring the Pinker-Ullman words-and-rules
default/elsewhere principle (`dual-route-past-tense-recognition-gated-blocking.md`'s own Kandel 6e p.1373
anchor, reused here). RELFRONT's ON drive carries a small (1.10x) priority tilt over KBREL/DEFCOP, reproducing
the host's own "relf checked first" priority as a genuine drive-strength difference rather than a hidden
tie-break — see the ambiguous-item result below. The winning assembly (rate clears the runner-up by a
dead margin of 0.05, the same units/threshold as the affect-marker circuit) IS the route decision, read off
`cp_firing_states`.

## The battery — real questions, labels computed from the real host regex objects (never hand-asserted)

66 questions, generated programmatically: 8 GENERIC (SVO teaching-shape controls), 6 DEFCOP ("what/who is X"),
7 RELFRONT ("what `<relation>` is X `[prep]`?"), 45 KBREL (29 generic "what is X's `<phrase>`?" templates over
every relation in the shipped `wikidata_core_15k` core's `_KB_UNDERSCORED_RELATIONS` table, plus 17 idiom
examples such as "where was X born?" / "who does X work for?"). Ground truth (`host_route_type`) reproduces
`_extract_route`'s own priority order for the parity comparison ONLY — `_extract_route` itself is never called
for its decision, only its regex objects/tables for feature extraction, and labels are computed from those SAME
imported objects rather than hand-asserted, so a labeling mistake on my part cannot silently pass.

**The deliberately ambiguous item.** "what country is entity_x a citizen of?" genuinely matches BOTH
`_REL_FRONTED_RE` (relation="country") and the `country_of_citizenship` KBREL idiom — confirmed programmatically
(`battery_health_check`'s `dual_matching=True`), not asserted by hand. This is the discriminating, non-ceiling
case: a trivial circuit that just ORs "any evidence present" could not reproduce the host's `relf`-wins priority
here, since two channels have equally strong raw evidence.

## Verification (6 seeds 42/43/44/100/101/102, ONE frozen operating point, no per-seed tuning)

- **PARITY: 66/66 (100%) on every seed** — the intact circuit's winning assembly matches `host_route_type` on the
  full battery, including all 45 KBREL relations/idioms, all 7 RELFRONT items, all 6 DEFCOP items, and all 8
  GENERIC controls.
- **AMBIGUOUS-ITEM PARITY: RELFRONT on every seed** — the deliberately dual-matching item resolves via genuinely
  stronger RELFRONT drive (the 1.10x tilt), reproducing host priority as a real competitive-margin effect.
- **FULL LESION (all three exception pathways forced OFF): 100% of the battery routes to GENERIC on every
  seed** — the substrate's own honest "elsewhere" fallback, matching the host's own all-flags-off behavior.
  `attributable_to("exception-subset parity: intact vs full-exception-lesion")` = **100.0%** on every seed (the
  58-item exception-labeled subset's parity collapses from 1.0 intact to the GENERIC-only floor under lesion).
- **PER-PATHWAY INDEPENDENCE: 100% on every seed for all three pathways** — lesioning the OTHER two exception
  channels leaves each pathway's own battery subset (RELFRONT n=7, KBREL n=45, DEFCOP n=6) still resolving
  correctly; no pathway depends on another.
- **GRADED SWEEP (discriminating, non-ceiling): a genuine monotonic transition on every seed** — scaling a real
  RELFRONT item's ("what country is chelsea fc from?") evidence-driven current from 0 to full drive moves the
  assemblies' rates from GENERIC-favored to RELFRONT-favored, crossing strictly between gain 0.25 and 0.5 on
  every seed (exact per-seed `rate_generic`/`rate_relfront` values in the cited artifact's `sweep` array) —
  demonstrating the readout is a genuine competitive dynamic with real dynamic range, not a boolean gate that
  happens to always land on one side.
- **DETERMINISM:** build-twice-at-one-seed identical seed-derived Izhikevich-parameter hash on all 6 seeds
  (`cfg.seed`, not `actual_seed_used` — the 2026-07-17 trap).

Full per-seed detail (rates, per-question evidence, sweep tables) in
`research/findings/raw/_rank14_question_route_wta_6seed.json`.

## What stays host code (the honesty boundary)

The three regex feature-EXTRACTORS (`_REL_FRONTED_RE`, `_KB_RELATION_PATTERNS`, the definitional-copula shape)
are UNCHANGED and NOT retired by this finding — they remain the same class of scaffold they already were. A
separate, larger residual (named, not claimed closed) is teaching the substrate to recognize these three
construction TYPES from corpus statistics instead of a curated regex/table; this finding retires only the
DISPATCH — which route's evidence wins when more than one construction could apply, and whether the default is
genuinely unopposed when none does.

## Scope and next rungs

This is a DE-RISK (`research/runners/` only) — **not wired into `_extract_route` in production** by this change
(RAM discipline: no integrated brain-chat verify was run; this circuit was measured standalone, <200 neurons,
well under a second per seed). The evidence→current scale (700/1350/150 pA, the 1.10x RELFRONT tilt) is an
untuned de-risk operating-point knob reusing the affect-marker circuit's own calibrated regime, not a
biology-required constant (no `constraints_config` bound, matching that entry's own convention). The named next
rung, if the owner wants this wired, is a production hook mirroring the rank-8/rank-12 pattern (a default-OFF
flag in `_extract_route` reading this circuit's winner, byte-identical-off, then a separate production-flip
verify) — not attempted here per the RAM-discipline / de-risk-only scope of this task.
