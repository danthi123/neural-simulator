---
type: finding
status: positive
date: 2026-08-25
lane: laneC
board: 129
mechanism: source-provenance-opponent honesty wired into known_fact_record/reasoned_fact_record + /api/brain-chat
runner: research/runners/_129_source_provenance_honesty_wirein_derisk.py
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO (6/6) — runner's own verdict GO=True at the >=5/6 bar
artifacts:
  - research/findings/raw/lanes/metacog/_129_source_provenance_honesty_wirein_6seed.json
---

# Board #129 next rung: the perceived-vs-generated provenance monitor is WIRED into the live-chat honesty pathway, additive/default-off, 6-seed GO — the reply text is driven by the live spiking judged label, not a caller claim

<!--derived-->
**Verdict: GO (6/6), runner verdict GO=True at the >=5/6 bar.** The 2026-08-25 de-risked (6-seed GO) #129
perceived-vs-generated source-provenance opponent monitor is now WIRED (default-off) into the conversational
answer path: `BrainConversationalAgent.known_fact_record` (a directly-recalled, taught fact — PERCEIVED) and a
new `BrainConversationalAgent.reasoned_fact_record` (a multi-hop conclusion the brain COMPOSED, not itself any
single stored fact — GENERATED) both route their answer through the live spiking monitor before the reply is
rendered. A GENERATED claim is flagged ("I believe X, but I reasoned that myself rather than being told it
directly."); a PERCEIVED claim renders byte-identical to today's text. The framing is driven by the monitor's
OWN judged label — not the caller's claim about how the fact was obtained — and lesioning the monitor
(plasticity gate held shut at encode, the #129 de-risk's own verified failing-direction anti-cheat) collapses
the discrimination to a silent, non-informative read. Additive + default-off end to end; `webapp/server.py`'s
single-fact `/api/brain-chat` path is also wired (env-flag gated) and reachable through the real HTTP handler.

## The rung being closed

Board #129 (Vikunja #137)'s next rung, verbatim: *"wire the 'I saw this fact vs I inferred/imagined it' read
into the LIVE CHAT honesty pathway so the brain's reply honestly reflects provenance (e.g. hedges or flags a
generated-source claim)."* The de-risk (2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-
6seed-GO.md) itself named this as its explicit next step: *"integrate the provenance read-out into the live
chat honesty pathway as a functional 'I saw this / I inferred this' self-report."* This finding closes that.

## The wire, top to bottom

<!--derived-->
- **`research/runners/source_provenance_honesty.py`** (new, additive): a thin production wrapper around the
  de-risk's own `ProvenanceBrain` / `_judge` — REUSED BY IMPORT (the identical validated substrate, not a
  re-derivation). Adds a deterministic per-key content-pattern generator (`_stable_pattern`, SHA-256-seeded) so
  an arbitrary production fact — not one of the de-risk's 4 fixed calibration pairs — gets a stable episode
  assembly; `SourceProvenanceHonestyMonitor.encode_fact(key, provenance)` (idempotent — a key keeps its FIRST
  provenance) and `.judge_fact(key)` (never fabricates a label for a key never shown to it); and
  `provenance_framed_text(kind, raw_text, label)` — the text transform, driven by `label` alone.
- **`BrainConversationalAgent`** (`research/runners/brain_conversational_agent.py`): two new constructor kwargs
  (`enable_source_provenance_honesty=False`, `source_provenance_honesty_config=None`, mirroring the existing
  `enable_self_schema_honesty` seam exactly — independent axes, either/both/neither may be on). A new
  `_apply_source_provenance(rec, provenance, key)` helper is called from the end of `known_fact_record` (always
  `PROVENANCE_PERCEIVED` — a `known_fact_record` hit is ALWAYS a literal stored fact) and from the new
  `reasoned_fact_record(cue, actions)` (always `PROVENANCE_GENERATED` — wraps `reason_chain`; each HOP is a
  literal stored fact but the composed relation between `cue` and the terminal is not itself any single stored
  fact, so the brain is presenting its OWN inference). The hard moat is unconditionally FIRST in both methods —
  a hard abstain returns before the provenance monitor is even built.
- **`research/runners/source_provenance_production_organ.py`** (new): the process-shared organ wrapper
  (`source_provenance_enabled()` / `source_provenance_lesioned()` / `get_organ()`), mirroring this codebase's
  established `metacog_production_organ.py` / `curiosity_production_organ.py` convention exactly.
- **`webapp/server.py`**: `/api/brain-chat`'s single-fact (`rich=False`) path now reads the organ on every
  turn that gate() matched a fact (`BRAIN_SOURCE_PROVENANCE_HONESTY`, default-OFF), reframes `answer` via
  `provenance_framed_text`, and attaches the judged record as an additive `"provenance"` response key (null
  when off/abstained). `BRAIN_SOURCE_PROVENANCE_HONESTY_LESION` rebuilds the organ under the load-bearing
  lesion. Guarded (`try/except`) so an opt-in honesty read can never crash a turn.

## 6-seed result {42 43 44 100 101 102} — GO (all six PASS)

<!--derived-->
Verified through the REAL `BrainConversationalAgent` class + a real `RFPhasorComposer` (the production
composer) — the same convention this project already uses for the adjacent self_schema_honesty wire-in
(`_laneC_self_schema_honesty_wirein_derisk.py`). Each seed builds 8 PERCEIVED facts + 8 GENERATED (2-hop
composed) conclusions over disjoint vocabulary (16 items/seed, 96 total).

| check | result | reads as |
|---|---|---|
| **default-off byte-identical** | 6/6 seeds | flag off -> `known_fact_record`/`reasoned_fact_record` text unchanged; the monitor is never built (`_source_provenance_monitor is None`) |
| **moat-first** | 6/6 seeds | a hard abstain (unknown cue / dead-end chain) keeps `provenance: None`, text `"I don't know about that."` |
| **un-lesioned battery accuracy** | mean 0.990 (5/6 seeds 1.000, one seed 0.9375) | the judged label matches ground truth (floor 0.90) |
| **text tracks the judged label exactly** | 6/6 seeds | PERCEIVED -> byte-identical text, unflagged; GENERATED -> flagged, never both |
| **no-regression on a correctly-judged perceived fact** | 6/6 seeds | its `answer_text` is byte-identical to the flag-off text |
| **lesion collapses discrimination (d==0 exactly, both pools silent)** | 6/6 seeds, deterministic | the LOAD-BEARING lesion proof (not the noisy tie-broken accuracy — see below) |
| **composes cleanly with `self_schema_honesty`** | 6/6 seeds | both faculties on at once; neither's fields change |

<!--derived-->
**A methodological note on the lesion check.** The lesioned monitor's judged-label *accuracy* is NOT gated —
under the lesion both provenance pools read exactly silent (rate 0.0/0.0), so the "judged label" is a host
tie-break on a fixed RNG stream over a 16-item battery, small-N noisy by construction (one seed's tie-break
happened to agree with truth 12/16 times — a real value the deterministic check below still catches cleanly).
This is the SAME pitfall the #129 de-risk itself named and avoided ("the tie-broken accuracy is small-N noisy
... it is REPORTED, not gated"). The load-bearing, DETERMINISTIC proof is `lesion_d_zero`: `d` and both
`rate_perceived`/`rate_generated` read EXACTLY `0.0` on every item, every seed — no tie-break, no noise —
proving the framing decision is driven by the LEARNED trace (silenced by the lesion), not a hardcoded branch.

## Scope, honesty, and the named residual

<!--derived-->
This is **wired (default-off)**, not production-default or integrated (per `docs/TERMS.md`'s faculty-status
table — all three of wired/on-by-default/scaffold-retired would be required for "integrated"; only the first
holds here, by design — the task asked for default-off). The host boundary (declared, unchanged from the #129
de-risk): WHICH encoding context a fact is taught under is supplied by the caller — a `known_fact_record` hit
is always PERCEIVED, a `reasoned_fact_record` conclusion is always GENERATED — the monitor's readback of which
label a content pattern carries is the genuine spiking read, and it alone decides the framing (proven by the
lesion collapsing it even though the caller's claim is unchanged).

**Named residual (no-defer, next rung):** `webapp/server.py`'s single-fact `/api/brain-chat` path only ever
calls `gate()`, which never returns a composed multi-hop inference — so a live HTTP turn through THAT path can
only ever exercise the PERCEIVED half of the mechanism (still a real, live, every-turn spiking read: the
answer text still depends on the judged label, and a lesioned deployment would flip some perceived facts to
the flagged framing — see `tests/test_webapp_server.py::test_brain_chat_source_provenance_lesion_collapses_
through_the_real_handler`). The GENERATED half is reachable today via `BrainConversationalAgent.
reasoned_fact_record` directly (6-seed GO above) but not yet through a live HTTP turn. Exposing it over HTTP
needs the endpoint to answer SOME turns via `reason_chain`/`chain_of_thought` as a first-class answer channel —
a separate, larger design decision (how does a free-text turn signal "answer via inference"? a "why"-phrased
query? an explicit new request mode?) intentionally left to the owner rather than folded into this rung.

Reproduce:
```
SIM_BACKEND=numpy python -u -m research.runners._129_source_provenance_honesty_wirein_derisk \
  --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/lanes/metacog/_129_source_provenance_honesty_wirein_6seed.json
```
CI guards: `tests/test_source_provenance_honesty_wirein.py` (the `BrainConversationalAgent` seam, fast) and
`tests/test_webapp_server.py::test_brain_chat_source_provenance_*` (three cases: default-off byte-identical,
on reads the live monitor, lesion collapses — through the real `/api/brain-chat` handler).
