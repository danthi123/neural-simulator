---
type: finding
status: qualified
date: 2026-09-09
mechanism: question-route-selection-wta
lane: scaffold-retirement
seeds: [42, 43, 44, 100, 101, 102]
verdict: WIRED-DEFAULT-OFF (focused verify GO; integrated no-regression soak DEFERRED to an AWS-CPU batch)
runner: research/runners/_rank14_qroute_wirein_verify.py
artifacts:
  - research/findings/raw/_rank14_qroute_wirein/verify_6seed.json
external: NO-EXTERNAL-NEEDED -- pure production wiring of the already-GO mechanism de-risk
  (2026-09-09-rank14-question-route-selection-wta-derisk-GO.md, status live); no new biological claim.
---

# Spiking 4-way lateral-inhibition WTA question-route SELECTION wired into `_extract_route`, default-OFF

**Verdict: WIRED, DEFAULT-OFF.** The focused wire-in verify is GO -- `_extract_route`'s route output is
BYTE-IDENTICAL when the flag is off (the pre-existing host `if`/`elif` cascade runs verbatim and the organ is
never built, 0 mismatch across a representative question set), and the spiking WTA DISPATCH is LOAD-BEARING when
on (6/6 project-standard seeds resolve the ambiguous overlap to RELFRONT by drive strength; single-route cases
route correctly; the lesion collapses every route to GENERIC). The **integrated `/api/brain-chat` no-regression
soak is DEFERRED** to an AWS-CPU batch (the dev box is RAM-blocked on the full integrated brain-chat battery and
the owner is gaming -- no heavy local compute); the FLIP to default-ON waits on that verdict. This turns the
6/6-GO mechanism de-risk (`2026-09-09-rank14-question-route-selection-wta-derisk-GO.md`, commit `d63a39ce8`)
from a banked de-risk into an actual production wire-in -- the owner's #1-priority scaffold-retirement step
(the 3rd production wire-in of the session, after novelty `58f7c12c` and anaphor `0213dccb`).

## What was wired

`ChatBrain._extract_route` (`research/runners/brain_chat_tui.py`) decided, in host Python, WHICH of four
comprehension routes handles an incoming question by testing three route feature-extractors in a FIXED PRIORITY
ORDER (`_relation_fronted_route` -> `_kb_relation_question_route` -> `_definitional_copula_route`, guarded by
`len(content) <= 1`) and falling through to the already-neural generic SVO parse when none returned a candidate.
The individual regex/shape TESTS are legitimate matched-filter reads of the surface string; what was NOT neural
is the COMBINATION -- which construction wins when more than one could apply, decided unconditionally by textual
`if`/`elif` order rather than by cue STRENGTH. That DISPATCH is now optionally moved onto the substrate as a
4-way lateral-inhibition WTA, behind ONE new default-OFF flag **`BRAIN_SPIKING_QROUTE`**:

- **`brain_chat_tui.py::ChatBrain._extract_route`** -- the three sequential `if _relf/_kbrel/_defo is not None:
  return` blocks are preserved VERBATIM inside `if not _QROUTE.spiking_qroute_enabled():` (the default path, with
  the lazy short-circuit intact). When the flag is on, all three host candidates are computed and
  `_spiking_route_decision(relf_on, kbrel_on, defcop_on)` picks the winner; the EXACT host priority cascade is
  the fallback on a dead-margin tie or any organ error.
- **`brain_chat_tui.py::ChatBrain._spiking_route_decision`** (new) -- never-raising wrapper: lazily builds the
  per-session organ (seed from `self.agent.seed`), returns the route name or None (host fallback) on ANY error.
- **`research/runners/spiking_qroute_selection_organ.py`** (new; NO `sim/` edit, `git diff sim/` empty). A
  per-session `SpikingQRouteSelectorOrgan` that reuses-by-import the de-risk's OWN drive mapping + operating-point
  constants (`evidence_to_currents`, `ROUTES`, `OFF_PA`/`GENERIC_BASELINE_PA`/`EXCEPTION_ON_PA`/
  `RELFRONT_PRIORITY_TILT`, `WARMUP/WASHOUT/RUN_STEPS`, `DEAD_MARGIN`) and the already-GO'd N-pool cross-inhibition
  primitive (`_affect_marker_wta_derisk._build_bridge`/`_pool_rates`, `n_pools=4`), so the wired mechanism is
  byte-for-byte the de-risked one: four excitatory route assemblies (RELFRONT/KBREL/DEFCOP/GENERIC), each with its
  own FSI cross-inhibition sub-pool, GENERIC carrying a constant baseline "elsewhere" drive.

The organ never runs a regex: `select(relf_on, kbrel_on, defcop_on)` takes the three BOOLEANS the host extractors
already produced. The regex feature EXTRACTION stays host code -- the residual the de-risk + biology binding
already declare (`research/biology/question-route-selection-wta.md`, "Honesty boundary"); only the DISPATCH moves
onto the neurons.

## Fresh-per-decision (a wire-in-specific correctness fix over the de-risk's reuse)

The de-risk's `evaluate_seed` reuses ONE bridge across its whole battery and `_pool_rates` washes out residual
drive at the start of each read. Measured 2026-09-09 during this wire-in: the 40-step washout does NOT fully reset
the Izhikevich adaptation (recovery variable) between reads, and the deliberately AMBIGUOUS RELFRONT/KBREL case
-- resolved by a small ~10% drive tilt -- has a margin tight enough that reuse across an arbitrary question
sequence can shrink it below `DEAD_MARGIN`: on a reused bridge at seed 42 the ambiguous margin collapses from
0.14 (fresh) to 0.009 (a dead-margin tie) after four prior single-route reads, INDEPENDENT of RNG isolation <!--derived--> <!-- reuse-vs-fresh probe, reproducible from the runner; not a committed artifact value -->
(probed both with and without `_isolated` -- identical collapse, so reuse/adaptation, not RNG, is the cause). The
organ therefore builds a FRESH quiescent bridge per decision (the same fresh-substrate-per-decision discipline
`SpikingAnaphorDetectorOrgan` uses), so every route decision is independent and carryover-free -- the correct live
unit for "one decision per question". Cost is trivial (144 neurons, one build + 160 steps per chat turn; speed is
secondary to faithfulness per the mission) and determinism is preserved (every build reseeds from `cfg.seed`).
NOTE: even without this fix the CONTRACT holds -- a dead-margin tie on the ambiguous case falls back to host
priority (RELFRONT first), so the answer is unchanged -- but fresh-per-decision keeps the DECISION load-bearing on
the substrate, which is the deliverable.

## What the focused verify measured (artifact: `research/findings/raw/_rank14_qroute_wirein/verify_6seed.json`)

Reproduce: `SIM_BACKEND=numpy python -u -m research.runners._rank14_qroute_wirein_verify`. LIGHT unit checks only
(tiny <200-neuron nets, sub-second) -- NOT the integrated brain-chat battery. All checks PASS:

- **L1 organ-direct, intact (seed 42):** single-route evidence -> the matching route (RELFRONT/KBREL/DEFCOP);
  no-evidence -> GENERIC; ambiguous (relf+kbrel) -> RELFRONT (margin 0.141). <!--derived--> <!-- margins printed by the cited runner (strings in verify_6seed.json); reproducible -->
- **L1 organ-direct, LESION (seed 42):** every evidence tuple -> GENERIC (the exception pathways' gain forced OFF).
- **L1 ambiguous across 6 seeds (42/43/44/100/101/102):** -> RELFRONT every seed (margins 0.072-0.141), via genuinely stronger drive, not a host tie-break. <!--derived--> <!-- per-seed margins printed by the cited runner (strings in verify_6seed.json); reproducible -->

- **L2 `_extract_route` OFF:** output == host priority cascade AND `decision_calls == 0` / organ never built (the
  byte-identical-off proof) over 5 representative questions.
- **L2 `_extract_route` ON:** returns the winning route's candidate on all 5 (incl. the ambiguous item ->
  RELFRONT's candidate).
- **L2 `_extract_route` ON+LESION:** a relf question reverts to the GENERIC positional-heuristic parse (the exact
  mis-parse RELFRONT was built to fix), NOT the relf candidate -- the load-bearing proof the dispatch drives the
  route.

## Deferred to the AWS-CPU integrated verify (verdict-pending)

The full integrated `/api/brain-chat` no-regression soak is RAM-blocked on the dev box and is NOT run here. The
exact command for the parent to run on an AWS-CPU batch (produces the flip-gating verdict):

```
SIM_BACKEND=cupy BRAIN_SPIKING_QROUTE=1 .venv/bin/python -u -m research.runners.onebrain_regression_battery \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/_rank14_qroute_wirein/<integrated-out>.json
```

It must show: (1) every routed turn's content byte-identical vs `BRAIN_SPIKING_QROUTE=0` on the standard battery
(the WTA reproduces host priority on all non-tie cases, so the chosen route -- and therefore the answer -- is
unchanged), and (2) the spiking route dispatch present + sane end-to-end through the real handler with the flag
on. **Until that lands, the flip to default-ON is NOT taken** and `BRAIN_SPIKING_QROUTE` stays default-OFF
(byte-identical). (If `onebrain_regression_battery` is not the exact battery the parent uses for the routing path,
substitute the standard integrated no-regression runner that exercises `_extract_route` with
`BRAIN_SPIKING_QROUTE=1`; the two required checks are what matter.)

## Honest residuals

- **NOT flipped default-ON.** This finding lands the WIRE-IN; the ledger `comprehension-routing` row stays
  not-retired until the AWS integrated soak returns GO. `status: qualified` (the de-risk finding remains the
  mechanism's one `status: live` answer).
- **The regex feature EXTRACTION stays host code.** This wire-in retires the DISPATCH only -- teaching the
  substrate to recognize the four constructions from corpus statistics rather than a curated regex table is the
  named larger residual (biology binding, "Honesty boundary"), out of scope here.
- **The drive operating point (OFF/BASELINE/ON pA + RELFRONT's ~10% tilt) is the de-risk's untuned knob**, reused
  verbatim; it is not a biology-required constant (de-risk HONEST RESIDUAL #2).
- **FUNCTIONAL correlate only** -- a spiking competitive-selection read, no phenomenal claim.
