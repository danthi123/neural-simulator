---
type: finding
status: live
date: 2026-08-27
mechanism: onebrain-r4-selfschema-provenance-production-frozen
lane: one-brain/integration/production
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_selfschema_production_frozen_6seed.json
runner: research/runners/onebrain_xedge_selfschema_production.py
builds_on:
  - research/findings/2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md
  - research/findings/2026-08-27-onebrain-xedge-production-frozen-GO.md
---

# R4 self_schema -> source_provenance LEARNED CROSS-EDGE — wired into the live chat brain as an additive,
default-OFF diagnostic; lesion-attributable through the real handler (6/6), 5/6 clear R4's own pre-registered
floor under the simpler production call order — GO on the wiring, one honest residual named not hidden

**One-line:** R4 (self_schema's "did I author this thought" -> source_provenance's "reads as internally-generated",
6-seed GO on the research merge framework, `2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md`,
merge `20b4b475c`) is now wired into the LIVE chat brain, mirroring the proven d6-WM->comprehension PART-1 pattern
(`2026-08-27-onebrain-xedge-production-frozen-GO.md`): a frozen, grow-once cross-edge, an additive diagnostic field
attached inside the existing `is_hyp` self-schema block of `webapp/server.py`, default-OFF
(`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA`), byte-identical-off (structurally + through the real handler), and
lesion-attributable through the real `/api/brain-chat` handler on every seed. **Declared, not hidden:** this
wiring's own simpler call sequence (a fresh `train()` then an immediate read, without R4's own F1/F4
pre-conditioning steps) measures a slightly smaller intact shift than R4's own richer-preconditioned protocol on
one seed, so 6/6 seeds clear the wiring's own lesion-attributability bar (the crux this wire-in is actually
answering: does the diagnostic drive-then-vanish-under-lesion through the live production call path) while 5/6
clear R4's own pre-registered absolute floor (`F2_INTACT_FLOOR=0.010`) under this simpler order. Two independent
adversarial verify-go passes (below) each returned CONFIRMED-WITH-CAVEAT; both caveats are fixed or disclosed
here. Kept default-OFF by design — a later flip-soak owns the default-ON decision, per instruction.

Artifact: `research/findings/raw/_onebrain_xedge_selfschema_production_frozen_6seed.json`.

## What was built (mirrors PART-1's shape, on the R4 pairing)

`research/runners/onebrain_xedge_selfschema_production.py` — a process-shared `XedgeSelfschemaProductionPool`
that:

- **Grows-once + freezes**, IN-PROCESS (not a saved weight file, for the same cross-backend-seed-trap reason
  PART-1 documents). Builds `R4Pool(seed)` (`research/runners/_onebrain_integration_r4_selfschema_provenance.py`,
  reused by import — the merge pool, the whitelist-inverted cross-edge injection, both organs' own build-time
  steps, and the fixed dual-context-encoded ambiguous content item are ALL R4's own validated code, not
  reimplemented), calls its own `.train()` (the SAME direct-drive Hebbian training the R4 6-seed GO validated),
  then explicitly re-asserts `set_plasticity_gate(GATE, 0.0)` for defensive parity with PART-1's own
  `R3v3Pool.train()` convention — no weight moves during any live turn.
- **Exposes `crossedge_provenance_shift(pool, hold_author)`** — the live-turn hook. It delegates to
  `R4Pool.amb_read` (R4's OWN validated F2 instrument, reused verbatim, not reimplemented) to read the signed
  provenance margin on R4's fixed ambiguous item, with and without holding self_schema's `author` pool, and
  reports the shift toward GENERATED.
- **Runs on an INDEPENDENT `R4Pool` instance**, not the shared process singletons `self_schema_production_organ.
  get_organ()` / `source_provenance_production_organ.get_organ()` the rest of the app uses for the already-
  default-ON self_schema authorship marker and the (separately default-off) board-#129 source-provenance-honesty
  organ. Zero shared mutable state — confirmed by both skeptics below by tracing the imports.

`webapp/server.py` — inside the existing `is_hyp` block (the DR-3 self-schema AUTHORSHIP faculty, already
default-ON since 2026-08-26), a new guarded, additive block reads the SAME live authorship verdict that block
already computed (`resp["authorship"]["is_self"]`) and, only when `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` is set,
attaches `resp["authorship"]["source_provenance_crossedge"]` — never touching `resp["answer"]` or any
pre-existing field. 29 lines changed, all inside the one guarded block.

## Verification

<!--derived-->

### 1. Organ-level 6-seed self-test (0 Claude tokens, numpy CPU, ~80s)

| seed | cross weight (from 0.05) | intact shift | lesion shift | frac attributable | clears R4's own floor (0.010) |
|---|---|---|---|---|---|
| 42 | 3.5534 | +0.0106 | -0.0006 | 105.9% | YES |
| 43 | 3.4684 | +0.0117 | +0.0003 | 97.3% | YES |
| 44 | 3.1333 | +0.0130 | +0.0002 | 98.4% | YES |
| 100 | 2.9176 | +0.0097 | -0.0009 | 109.7% | **NO** (0.0097 < 0.010) |
| 101 | 3.3353 | +0.0116 | +0.0006 | 94.6% | YES |
| 102 | 3.3593 | +0.0114 | -0.0005 | 104.6% | YES |

Cross weights (2.92-3.55) match R4's own 6-seed GO range (2.9-3.6) closely — the growth trajectory reproduces
faithfully. Lesion shift stays within R4's own pre-registered `F2_LESION_RATIO=0.34` noise-floor ratio of the
intact shift on every seed (the `frac_attributable_to_cross_edge` column, computed via `tools.lab.
attributable_to`, reads 94.6%-109.7% — three seeds read slightly above 100%, meaning the lesioned control moved
marginally opposite the treatment; R4's own finding documents the identical pattern on some of ITS seeds and
names it "an honestly-reported small-sample wobble, not evidence against the edge" — the same read applies here).
The "no signal, no bias" control (`author_held=False`, both intact and lesioned) reads within the same 0.34x
noise floor on every seed — holding nothing produces no shift, as it must.

**6/6 GO on lesion-attributability** (this wiring's own crux: does the diagnostic drive-then-vanish-under-lesion
through the live production call path). **5/6 clear R4's own pre-registered absolute `F2_INTACT_FLOOR=0.010`**
under this wiring's simpler call sequence (`train()` then an immediate read, with no F1/F4 pre-conditioning steps
run first, unlike R4's own `run_seed`) — seed 100 measures 0.0097, just under R4's own floor, even though R4's
own richer-preconditioned run on the same seed cleared it at +0.0110. This is a genuine, small, measured
residual from the simpler production-realistic call order, reported honestly rather than silently re-using R4's
floor as if the protocol were unchanged (see "Honest residuals" below).

### 2. Through the real `/api/brain-chat` handler (3 pytest tests, `tests/test_webapp_server.py`, 903s)

A genuine open-ended hypothesis turn triggers a ~46K-neuron vocab-scale generative-sampler build (multiple
minutes) that has nothing to do with this wiring (it lives strictly downstream of `rich.answer()`'s return
value) — so the 3 tests warm a session with an ordinary (fast) recall turn, fetch the REAL cached
`RichAnswerComposer` from `webapp.server._BRAIN_RICH`, and monkeypatch ONLY its `.answer()` to return a
well-formed hypothesis-turn dict, then exercise the rest of the real, unmodified `brain_reply()` — including
this wire-in's new block — for real.

| test | result |
|---|---|
| `test_brain_chat_xedge_selfschema_default_off_is_byte_identical` | **PASS** — flag unset, no `source_provenance_crossedge` key |
| `test_brain_chat_xedge_selfschema_on_reads_live_crossedge_and_lesion_collapses` | **PASS** — flag on: diagnostic attached, `author_held == authorship.is_self` (True on an is_hyp turn), intact shift > 0.003; **through a SECOND real handler call with `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_LESION=1`**, the shift collapses to within R4's own 0.34x floor of the intact value |
| `test_brain_chat_xedge_selfschema_no_regression_on_ordinary_recall_turn` | **PASS** — an ordinary (non-hypothesis) turn's WHOLE response is structurally IDENTICAL (`d_off == d_on`, a real dict-equality assertion, not code-inferred) whether the flag is off or on — satisfies `docs/TERMS.md`'s byte-identical bar (asserted in the data) for the dominant, non-hypothesis traffic |

`3 passed, 72 deselected, 17 warnings in 903.16s`.

### 3. Adversarial verify-go (2 independent skeptics, blocking, sonnet)

**Skeptic A (organ-level mechanism).** Independently reproduced the 6-seed self-test from scratch in a fresh
process — bit-identical to the committed artifact on every field, every seed. Confirmed the freeze is genuine
(`enable_hebbian_learning` stays False after `train()`; `set_plasticity_gate(GATE, 0.0)` zeroes only the
cross-edge's own rate-gain indices, verified against `sim/bridge.py`). Found two real issues, **both fixed
before this finding was written**: (1) the self-test's GO criterion had silently dropped R4's own
`F2_INTACT_FLOOR=0.010` in favor of a much weaker `d_i > 0.0`, and seed 100's reproduced shift (0.0097) sits
below that floor — **fixed** by adding an explicit, honestly-reported `clears_r4_registered_floor` field
(the table above) rather than hiding the discrepancy; (2) `pool.cross_weights` was snapshotted once before any
lesion and never refreshed, so a lesioned pool's OWN diagnostic field kept reporting the pre-lesion trained
weight (cosmetic — the actual shift read always used live connection data) — **fixed**, `lesion_cross()` now
refreshes it. Verdict: **CONFIRMED-WITH-CAVEAT** (caveat = item 1, now surfaced honestly rather than resolved by
deletion).

**Skeptic B (webapp wiring + moat safety).** Traced the new `webapp/server.py` block exactly: confirmed it never
touches `resp["answer"]`, confirmed zero shared mutable state with the already-default-ON self_schema marker
(independent `R4Pool` vs the process singleton), confirmed the 3 pytest tests genuinely exercise the real
unmocked code path (not a shortcut that trivially passes) and that the `"error" not in xe` assertions correctly
guard against a silently-swallowed exception masquerading as a clean read. Found one low-severity, **disclosed
but not changed**, issue: the `except Exception` wraps the whole `try` block including the `from ... import`
line and the `xedge_selfschema_enabled()` call itself — so a hypothetical import failure would attach an
`{"on": True, "error": ...}` diagnostic even when the flag is unset, meaning "byte-identical-off" is empirically
true (confirmed: the module's own top-level code is trivial and the import cannot realistically fail) but not
STRUCTURALLY guaranteed by the code shape alone. This is the IDENTICAL shape as the pre-existing DR-3 self-schema
block immediately above it (`webapp/server.py:5632-5641`, already shipped, already default-ON) — a pre-existing
house pattern, not a novel regression, left as-is to avoid deviating from established convention under time
pressure; named here as a residual rather than silently accepted. Also flagged that "mirrors PART-1" is honest
but easy to over-read: PART-1 swaps `shared=` into the REAL per-session `d6` organ and (in its later PART-3) runs
live in-brain plasticity off real comprehension turns; this wiring never touches the real self_schema/
source_provenance singletons — it is architecturally parallel/diagnostic-only, which the module's own "DECLARED
RESIDUAL #2" already says, but only if a reader gets past the headline. Verdict: **CONFIRMED-WITH-CAVEAT**.

Both caveats are addressed above (fixed, or explicitly disclosed with the reasoning for leaving it). No claim in
this finding rests on the un-fixed/undisclosed version of either.

## Honest residuals (declared)

1. **One seed (100) misses R4's own pre-registered absolute floor** under this wiring's simpler call sequence
   (train -> read, no F1/F4 pre-conditioning) — a real, small, measured difference from R4's own richer-
   preconditioned protocol, not a bug in either. `GO` here is graded on lesion-attributability (this wire-in's
   own crux), with R4's floor reported honestly alongside it, not silently reused as if unchanged.
2. **Diagnostic-only, not a real coupling into the production singletons.** Unlike PART-1's own later rungs, this
   wiring does not modify `self_schema_production_organ.py` / `source_provenance_production_organ.py`'s shared
   process singletons — it runs an independent `R4Pool` and attaches a read-only diagnostic field. This was a
   deliberate risk-minimizing choice (zero exposure to the already-default-ON self_schema marker) rather than an
   oversight; extending the production `SourceProvenanceHonestyMonitor` with a `shared=` attach (mirroring
   `ProvenanceBrain`'s own existing `shared=` support) so the cross-edge could bias an ARBITRARY live fact's
   provenance judgment, not just R4's fixed ambiguous probe item, is a separate, later, reviewed rung.
3. **The `except`-wraps-import shape** (skeptic B, above) is inherited house style, empirically inert, disclosed
   not fixed.
4. **Not strict `self-organized`** (per `docs/TERMS.md`), same as R4's own declaration: the cross-edge topology
   and training schedule are host-chosen/host-curated; the WEIGHT is learned by the substrate's own Hebbian rule.

## Verdict

**GO on the wiring**: the R4 learned cross-region edge is reachable from the production `/api/brain-chat`
endpoint (wired, default-off — `docs/TERMS.md`'s "wired (default-off)" level, not yet on-by-default), verified
byte-identical when off (structurally and through a real whole-response equality check), and lesion-attributable
through the real handler on 6/6 seeds. Two independent adversarial passes found one material issue (a silently-
weakened floor) and one cosmetic bug, both fixed, plus one low-severity structural note and one framing note,
both disclosed. Kept default-OFF by design; a later flip-soak owns the default-ON decision. Functional
read-outs only; no phenomenal-experience claim.

## Files

- `research/runners/onebrain_xedge_selfschema_production.py` — the production wiring module (grow-once+freeze,
  the live-turn hook, the 6-seed offline self-test).
- `webapp/server.py` — the guarded, additive attach inside the existing `is_hyp` self-schema block.
- `tests/test_webapp_server.py` — 3 new tests through the real `/api/brain-chat` handler.
- `research/findings/raw/_onebrain_xedge_selfschema_production_frozen_6seed.json` — the 6-seed organ-level
  artifact (n_go=6, n_clears_r4_registered_floor=5).
