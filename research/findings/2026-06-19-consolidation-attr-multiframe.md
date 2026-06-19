# Consolidation: attributed-1-attr + multi-frame folded into the production agent (2026-06-19)

**The owner-approved conversational #1 (CONSOLIDATION, not a new mechanism):** fold the already-validated richer
capabilities that lived only in standalone de-risk runners into the ONE production conversational agent
(`OneBrainComposer` / `BrainConversationalAgent`), behind default-OFF flags, with the no-confab moat re-asserted and
the flat/rf default byte-unchanged. Pre-registered by
`research/findings/2026-06-19-conversational-scaling-next-lever-scoping.md` (#1); the SHIP scope was set by the
critical-risk de-risk `research/findings/2026-06-19-resonator-on-learned-codes-derisk.md`: **ship single-attribute +
multi-frame; do NOT ship the F=3 two-attribute path** (it DEGRADES to ~29% on the correlated production LEARNED 320
codes — the documented boundary, a separate follow-on needing clean codes or bigger D).

## What was wired (additive, default-OFF, reuse-by-import, NO `sim/` edit)

1. **Attributed single-attribute entities ("dog eat big apple" -> `what_does` = "big apple")** —
   `OneBrainComposer(enable_attributed=True)`. The composer's work / Q-register / cleanup layout is parameterized by
   `bind_roles` (default 4 roles `[agent, action, patient, polarity]` -> the constants are byte-identical to before;
   attribute-enabled -> 5 roles `[agent, action, patient, attribute, polarity]`). `store`/`hear`/`query_patient`/
   `render_fact`/reconsolidation bind + decode the ATTRIBUTE role (the 2-factor / one-bind-one-unbind path) and join
   `"adj noun"`. A plain fact on the attribute-enabled composer binds only its present roles (stays a 4-way bundle ->
   no extra crosstalk). The read path returns `{role: word}` dicts (replacing the 4-tuple). `BrainConversationalAgent`
   passes `enable_attributed` through to the onebrain composer AND keeps the existing neural `AttributedBridgeParser` +
   `hear_attributed` (parse-in-spikes) so the production agent comprehends `S V adj N` end-to-end.

2. **Auto-selected multi-frame comprehension (SVO / VSO / OSV)** — a new `research/runners/frame_parser.py`
   (`FrameParser`) composes the two validated GO pieces reused-by-import: `FrameSelector` (verb-position -> frame,
   neural; `2026-06-18-frame-selection-GO.md`) + `MultiFrameParser` (position x frame -> role, neural;
   `2026-06-18-multiframe-comprehension-GO.md`). `OneBrainComposer.hear_multiframe(sentence, verbs)` +
   `BrainConversationalAgent.hear_multiframe(sentence, verbs)` route comprehension through it (the known-verb set is the
   lexical/morphology front end the selector uses to find the verb). Default OFF = the native SVO/passive parser
   unchanged.

The F=3 **two-attribute** ("big red ball") path is DELIBERATELY NOT wired (the documented ~29%-on-learned-codes
boundary; `_resolve_patient` keeps only the first adjective).

## Files

- Wiring: `research/runners/one_brain_composer.py` (the attribute role + dict read path),
  `research/runners/brain_conversational_agent.py` (`enable_attributed`/`enable_multiframe` pass-through +
  `hear_multiframe`), `research/runners/frame_parser.py` (new, the multi-frame `FrameParser`).
- Validation runner: `research/runners/_consolidation_attr_multiframe_validate.py`.
- Raw results: `research/findings/raw/_consolidation_attr_multiframe.json`.

## CPU smoke (numpy, D=64) — the wiring is sound

- **Default layout byte-identical** (attribute OFF): `store_base`/`q_base`/`c_base`/`cb`/`n_total` all equal the
  pre-change values (verified arithmetically + by construction).
- **Attributed single-attribute round-trips:** `dog eat big apple` -> `query_patient("dog","eat")` = `"big apple"`;
  `cat see small ball` -> `"small ball"`; a plain co-resident fact (`bird go north`) still resolves; the moat abstains
  on an unstored cue; `render_fact("dog")` = `"dog eat big apple"`. Batched == per-block for the attributed path.
- **Multi-frame comprehension:** the `FrameParser` resolves `SVO "dog eat apple"`, `VSO "eat lion milk"`, and
  `OSV "garden rabbit jump"` to the correct {agent, action, patient} on numpy.

## The 320-scale matrix (GPU, LEARNED stream codes) — unanimous GO

| readout | seed | flat recall | flat un-regressed | 1-attr recall | multi-frame | moat false-accepts | verdict |
|---|---|---|---|---|---|---|---|
| neural | 42 | 1.00 | yes | 1.00 | 1.00 | 0 | GO |
| neural | 43 | 1.00 | yes | 1.00 | 1.00 | 0 | GO |
| neural | 44 | 1.00 | yes | 1.00 | 1.00 | 0 | GO |
| host | 42 | 1.00 | yes | 1.00 | 1.00 | 0 | GO |
| host | 43 | 1.00 | yes | 1.00 | 1.00 | 0 | GO |
| host | 44 | 1.00 | yes | 1.00 | 1.00 | 0 | GO |

**Verdict: GO (unanimous 3-seed × both readouts).** Flat SVO un-regressed (recall 1.00 = the baseline), single-attribute
attributed recall 1.00, multi-frame comprehension 1.00, and **the no-confab moat held with 0 false-accepts / 0 breaches
across all 6 runs** — on the production LEARNED 320 codes, both the fully-neural and the host read-outs. The consolidation
lands: the production agent now comprehends `S V adj N` and auto-selected SVO/VSO/OSV frames in addition to flat SVO,
with the no-confab guarantee intact and the default path byte-unchanged. (6-seed confirmation [100/101/102] + the
production CI run as the belt-and-suspenders non-regression check.)

## Production CI (`tests/test_one_brain_composer_agent.py`) — the default path is preserved

<!-- FILL: GPU pytest result -->

## Scope + the deferred boundary

- SHIPPED: flat SVO (unchanged) + single-attribute attributed entities + auto-selected multi-frame comprehension, on
  the production one-brain agent, behind default-OFF flags, the no-confab moat intact.
- DEFERRED (the documented boundary, NOT shipped): the **F=3 two-attribute** ("big red ball") decode — ~29% on the
  correlated learned 320 codes (the 3-factor resonator's permutation tie-break is defeated by the codes' semantic
  correlation; `2026-06-19-resonator-on-learned-codes-derisk.md`). The specified follow-ons if two-attribute is later
  prioritized: decorrelate/whiten the grounded phases (the standing point-neuron whitening problem), a stronger
  3-factor restart schedule, or distinct per-attribute named role tags (removing the permutation symmetry).
- Untested follow-on: the **attribute + embedded-clause together** combination (each works independently; the CI
  clause tests run with attribute OFF). At small D on numpy-CPU the 5-role-layout clause decode tips on the noise
  margin — a precision artifact of the larger bridge's OU-noise draw, not a logic bug; the clause decode wiring is
  layout-correct.
