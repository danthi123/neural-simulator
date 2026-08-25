---
type: finding
status: live
date: 2026-08-26
mechanism: knowledge-core-default-ltm-flip
lane: integration
integration_faculty: tiered-knowledge-ltm
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: The load-bearing property (byte-identity of the routed tiered store vs the plain-flat unsharded
  oracle) is STRUCTURAL — all of a subject's facts land in one shard over a SHARED codebook, so
  first-match-within-shard == first-match-over-store for that subject, true for ANY seed by construction. Run at
  all 6 mission seeds anyway (per-concept RF codes are seed-dependent) and re-confirmed on main single-seed.
---

# The curated 15k knowledge core ships as the default cortical LTM (board #133)

## Claim
The developed brain now loads a real body of world knowledge on the DEFAULT chat turn, not just when a bundle is
opted in. `webapp/server.py` resolves `BRAIN_LTM_BUNDLE` to the curated 15k wikidata CORE
(`sim-data/knowledge_bundles/wikidata_core_15k`, 15,000 facts, vocab 7,032, 75 shards) whenever the env is unset
and the bundle is present on disk. This is the owner's #1 knowledge priority landing as a production default: the
brain holds + queries an LLM-scale body of facts beside its small conversation working-set, and keeps LEARNING
over it through use (D5 per-topic strength stays in the path) — biological memory, not a static RAG cache.

## What flipped
- `webapp/server.py`: `_LTM_SHIP_DEFAULT_ON = True` (the anchored ship-default) + `_resolve_ltm_bundle()`.
  - `BRAIN_LTM_BUNDLE` unset -> the shipped core (ship-default on AND the bundle dir present; an isdir guard so a
    checkout without the data lake degrades to no-LTM, byte-identical).
  - `BRAIN_LTM_SHIP_DEFAULT=0` (its own on/off knob) OR `BRAIN_LTM_BUNDLE=off`/`0`/`none`/empty -> the pre-flip
    byte-identical no-LTM path (the plain flat composer).
  - `BRAIN_LTM_BUNDLE=<path>` -> that bundle (explicit override, unchanged).
- `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` row `tiered-knowledge-ltm`: `on_by_default: NO -> YES` with a
  machine-checked `default_anchor` (`_LTM_SHIP_DEFAULT_ON` on=1 / off=0, verified). `scaffold_retired` stays NO
  (the agent-hash shard router + the numpy VSA fast path are the standing host residuals).

## Evidence (the gate)
The no-regression SOAK is `go: true`, 6/6 seeds (42/43/44/100/101/102):
`research/findings/raw/_knowledge_core/core_bundle_soak_verdict.json`.
- Byte-identity: 0 mismatches vs the plain-flat UNSHARDED oracle (the routed tiered read returns the SAME answer
  the co-resident store would, over the real curated facts).
- No-confab moat: 0 confabulations (fabricated subjects + a known subject with an action it never has -> abstain).
- First-match recall: 1.0 on every seed.
- Latency: median ~0.41 s warm (vocab 7,032, under the 1 s UX line; a UX knob, not a correctness gate — a snappier
  build is `--top-entities 4000` per the curate runner, deferred).
- Production load path: `load_developed_brain(ltm_bundle=<core>)` installs a `TieredFactStore` and returns LTM +
  buffer recall + moat-abstain GO.
Re-confirmed on `main` (single seed, CPU/numpy, the merged runner code):
`research/findings/raw/_knowledge_core/core_bundle_soak_confirm_main_s42.json`. Curation provenance:
`research/findings/raw/_knowledge_core/curate_report.json`.

## Instrument + control
- Instrument: the soak compares the routed tiered read against a freshly-built plain-flat oracle over the SAME
  facts — the byte-identity check IS the instrument for "no regression from adding the store".
- Control (lesion): with `ltm=None` the `TieredFactStore` is ANSWER-IDENTICAL to the plain buffer (degrade
  identical) -> an LTM answer only ever comes from the installed cortical store, never from decoration; and BOTH
  tiers must abstain for a non-answer (the moat holds through the tier).

## Honesty boundary / residual (no-defer)
Not a phenomenal claim. Standing host residuals (scaffold_retired NO): (1) the shard router is `hash(agent) mod S`
— the faithful version is a learned/spiking cue->sub-population router; (2) the RF composer + moat are the numpy
VSA fast path — the co-resident spiking store is a separate rung; (3) `promote_buffer_to_ltm` (sleep-replay
hippocampal->cortical) is an explicit hook, not replay-driven yet. Next rung for scale: the O(V*D) codebook
cleanup grows with vocabulary (~1.3 s at 100k) -> candidate-restricted cleanup / a learned router; the 100k
bundle is built (`sim-data/knowledge_bundles/wikidata_100k`) for that follow-on.
