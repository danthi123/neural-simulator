---
type: finding
status: live
lane: track1
date: 2026-08-14
mechanism: onebrain-parser-on-pool
---

# OneBrainComposer PARSER joins production pool #1 — Track-1 one-substrate, 6/6 GO, DEFAULT FLIPPED ON

**Date:** 2026-08-14 · **Status:** GO (default flipped ON) · **Backend:** numpy (CPU) · **Seeds:** 42/43/44/100/101/102
**Extends:** `2026-08-14-onebrain-composer-pool1-DEFAULT-FLIP-GO.md` (the RF who/what recall/store joined pool #1 by
default; its named next lever was "the config-COUPLED Hebbian PARSER stayed on a private bridge"). This rung closes
that lever: the parser's INFERENCE now runs ON pool #1 too.

## What landed

`Pool1BoundOneBrainComposer`'s Hebbian PARSER (`BridgeParser`, a spiking Izhikevich conj->role circuit) previously ran
on a PRIVATE full-size bridge because pool #1's config (hebbian_max_weight=45, global homeostasis) and its
whole-bridge `_run_one_simulation_step` were incompatible with the parser (trained at 400, needs fixed vpeak). This
rung binds the parser's `[0:126]` slice (P=6+3*R) onto pool #1's reserved-idle `onebrain_composer` sub-slice
`[rf_base:rf_base+126]` (disjoint from every RF op), so `hear()` comprehends on the ONE shared `cp_membrane_potential_v`
with the D2 surprise + E2 world-model organs. Guarded `BRAIN_PARSER_MERGE` / `_PARSER_IN_POOL1_DEFAULT_ON` (now True).

## The three conflicts, resolved with EXISTING primitives (git diff sim/ EMPTY)

- **A — hebbian_max_weight 45 (pool) vs 400 (parser).** The parse edges live on the SHARED `cp_connections`; the
  surprise/world-model organs run Hebbian on the pool and the ungated clip (bridge.py:9689) would crush the
  transplanted 400-weights to 45 (measured: max parse weight fell to 45.00 before the fix). Resolution: TRANSPLANT the
  standalone-trained edges via `set_pathway_weights(add_missing=True)` (ADDS them without clobbering the region-built
  organ connections) + a PERMANENT per-synapse gain-0 on exactly the 720 parse edges. The gated potentiate/decay/clip
  (9640/9662/9677-9687) leave gain-0 synapses verbatim, so the 400-weights survive every Hebbian step; gain-1
  elsewhere is byte-identical to the un-gated scalar path, so the organs are unperturbed.
- **B — `_run_one_simulation_step` steps ALL neurons.** `MergedSubstrate.parser_isolation` (extends `read_isolation`)
  snapshots+restores the full per-neuron state AND `cp_connections.data`, keep-masking ONLY the parser slice — so
  surprise/world-model + the RF slice are byte-restored; only the parser slice self-evolves.
- **C — global homeostasis vs the parser's fixed vpeak.** `cfg.enable_homeostasis=False` + per-region
  `BrainRegion.enable_homeostasis=True` on surprise+world-model (the diffbuilder pattern, sim/regions.py:171). The
  3-branch spike-select uses adapted thresholds for the two organs and fixed vpeak for the parser slice = standalone.
  Byte-identical for the organs by read (the homeostasis UPDATE runs identically — `cp_homeostasis_update_neuron_mask`
  is None in both configs — and only the SELECT differs, only for the composer slice, which never feeds the organs).

## GO-gate — 6/6 seeds (`_onebrain_composer_pool1_production_verify --parser-on-pool`)

Every seed (42/43/44/100/101/102), ON `BRAIN_PARSER_MERGE=1` vs OFF `=0` (parser private == the STANDALONE parser):
(1) parser ANSWER-identical (hear-decoded fact dicts, active + passive) + correct on the reference; (2) recall +
query_patient byte-identical + correct, moat abstains 0 false-accept; (3) surprise + world-model byte-identical
**max delta 0.00 Hz** + both alive (contradict>>confirm, violated>>expected); (4) genuinely one pool —
`parser.bridge IS pool`, indices ⊂ [rf_base, rf_base+126), same `cp_membrane_potential_v` object as surprise/world-model,
N==v_len, gain-0 on all 720 parse edges + gain-1 else. **Artifact:** `research/findings/raw/_onebrain_parser_pool1_6seed.json`.

Global: **byte-identical-when-off** — `BRAIN_PARSER_MERGE=0` vs UNSET, 2/2 seeds bit-for-bit (recall/qp/parse/
surprise 0.0 Hz/world-model 0.0 Hz/moat), parser-off in both, MEASURED in separate processes. **Determinism 9/9**
(`tests/test_determinism.py` incl. `TestSubstrateActuallySeeded`; substrate seeded via `cfg.seed`). No-regression smoke
(`brain_chat_tui --smoke`, rf path unaffected). All pass ⇒ `_PARSER_IN_POOL1_DEFAULT_ON` FLIPPED True.

## Honest residual (DECLARED)

ANSWER-identity, not raw-firing byte-identity: on the pool the parser is FROZEN (gain-0), so it does not accumulate the
tiny per-`role_of` Hebbian drift (toward 400) the standalone parser does. The argmax role decode is robust to that
drift over bounded panels (weights near-saturated, measured max 355.32), so decoded facts + downstream recall are
byte-identical — but raw firing rates are not bit-identical, and over an unbounded dialog the frozen and drifting
parsers could in principle diverge. Inference-on-pool, not learning-on-pool (the parser trains on its private bridge in
`__init__`; only its INFERENCE runs on the pool). The named next lever for raw byte-identity + native on-pool training
is a guarded per-region `hebbian_max_weight` sim/ feature (mirror per-region homeostasis) — flagged, not built here.
