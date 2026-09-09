---
type: finding
status: live
claim_check: measured
date: 2026-09-09
mechanism: scaffold-retirement production flips — day_33 composer rf->onebrain (RANK-1) + cortical-LTM numpy-KB->substrate-store (RANK-6/#211)
lane: scaffold-retirement (owner 2026-09-04 priority) — retire host shortcuts onto the shared spiking substrate
integration_faculty: one-brain-substrate
seeds: [42]
seed-waiver: >
  This is a NO-REGRESSION + REVERSIBILITY gate on the SINGLE DEPLOYED artifacts (the scale787/day_33 bundle and
  the wikidata_core_15k LTM, both seed 42) — there is one production brain to flip, not a seed family. The measured
  property is a DETERMINISTIC representation/code-path change (spiking onebrain vs host rf composer; substrate-store
  vs numpy-KB), answer-identical on every probe and reproduced identically per run. Flip A's multi-seed / full
  404-fact strict parity + 29-ambiguous-cue no-confab moat is the RANK-1 GO arc
  (2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-GO.md); this gate adds the reversibility leg
  and the substrate-store internal-path parity, neither of which is seed-sensitive.
artifacts:
  - research/findings/raw/_scaffold_retirement_flips_noregression/gate_final.json
  - research/findings/raw/_scaffold_retirement_flips_noregression/flipA_comp_ON.json
  - research/findings/raw/_scaffold_retirement_flips_noregression/flipA_comp_OFF.json
  - research/findings/raw/_scaffold_retirement_flips_noregression/flipB_smoke_ON.json
  - research/findings/raw/_scaffold_retirement_flips_noregression/flipB_smoke_OFF.json
builds_on:
  - research/findings/2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-GO.md
  - research/findings/2026-09-08-rank1-yesno-multiblock-rung-GO.md
  - research/findings/2026-09-08-rank6-substrate-store-cupy-vram-measured-GO.md
  - research/findings/2026-09-02-cupy-scan-vectorize-latency-result-ACCEPT.md
---

# Two scaffold-retirement production flips land: the developed-bundle recall composer is now the spiking onebrain (host closed-form `rf` retired) and the cortical LTM holds facts in the SUBSTRATE (numpy-KB store retired) — no-regression gate GO, both reversible by env

**One-line:** the final no-regression gate for both already-de-risked flips is GO (`gate_final.json` VERDICT: GO, 17/17 checks). Flip A (`_DEVELOPED_COMPOSER_KIND_DEFAULT_OVERRIDE = "onebrain"`) makes every developed bundle recall through the spiking DG-CA3 `Pool1BoundOneBrainComposer` instead of the host closed-form `Pool1BoundComposer` (`rf`), answer-identical to `rf` on the deployed 404-fact scale787/day_33. Flip B (`_ltm_substrate_store_on()`/`_ltm_batched_substrate_scan_on()` default True) makes the cortical LTM hold each fact's composite in synaptic weights (read back by firing) instead of a numpy array cached in `kb`, answer-identical to the numpy-KB store. Both flips are now the production default; both revert with a single env var.

## What flipped (working tree, now production default)

- **Flip A — RANK-1, `webapp/server.py`:** `_DEVELOPED_COMPOSER_KIND_DEFAULT_OVERRIDE` `None` -> `"onebrain"`. Every developed bundle resolves to the spiking `onebrain` composer regardless of its saved manifest `composer_kind` (day_33's is `"rf"`). `BRAIN_COMPOSER_KIND=rf` is the env revert.
- **Flip B — RANK-6/#211, `research/runners/developed_brain_io.py` + `research/runners/rf_phasor_composer.py`:** `_ltm_substrate_store_on()` and `_ltm_batched_substrate_scan_on()` default True, so `load_developed_brain(ltm_bundle=...)` builds the LTM `ShardedPhasorStore` with `enable_substrate_store=True` + `enable_batched_substrate_scan=True`; the composer's env resolver now lets `BRAIN_BATCHED_SUBSTRATE_SCAN=0` force the per-fact loop back. `BRAIN_LTM_SUBSTRATE_STORE=0` / `BRAIN_BATCHED_SUBSTRATE_SCAN=0` are the env reverts.

`git status --short` for the flip is exactly these three files. `webapp/server.py`'s tiny-demo `_load_or_build_ltm_store` reuses the same two flip-B env resolvers, so the tiny-demo +LTM path stays consistent.

## Gate design (cupy, single-tenant GPU, `SIM_BACKEND=cupy`)

Each flip verified ON (production default) vs OFF (env revert) in FRESH subprocesses (the process-global merged-substrate singleton contaminates an in-process composer-family switch). GO = ON works + moat holds + OFF reproduces the pre-flip answers (reversibility).

- **Flip A — through the composer's OWN recall entry point.** The deployed day_33 bundle loaded via `load_developed_brain` exactly as `webapp._build_chat_brain` resolves `composer_kind`; recall driven by `agent.what_does(a,v)` -> `composer.query_patient(a,v)` and `agent.is_it_true(a,v,p)` -> `composer.ask_yes_no` — the SAME calls the handler's recall gate makes (`brain_chat_tui.ChatBrain.gate` -> `_substrate_recall` -> `inner.what_does`; the GNW multistep/deliberation organs only WRAP this verdict). 20 unique-(agent,action) day_33 probes.
- **Flip B — through the INTERNAL query_patient path, NOT the NL parser.** The LTM `ShardedPhasorStore` loaded via the exact production `ShardedPhasorStore.load(..., extra_kwargs=...)` and recall driven by `store.query_patient(a,v)` directly — the path the handler reaches via `TieredFactStore` fall-through. This is deliberate: the LTM holds WIKIDATA-style relations (`instance_of`, `country`) that the TinyStories-trained NL parser cannot reliably phrase, so an NL-path LTM probe fails on PHRASING (a parser limit ORTHOGONAL to the store flip), not on the substrate store. 40 unique-(agent,action) probes over the 15,000-fact store.

## Results (from the committed gate artifacts)

Artifacts: `research/findings/raw/_scaffold_retirement_flips_noregression/gate_final.json` (combined 17/17 verdict), with the phase JSONs `research/findings/raw/_scaffold_retirement_flips_noregression/flipA_comp_ON.json`, `research/findings/raw/_scaffold_retirement_flips_noregression/flipA_comp_OFF.json`, `research/findings/raw/_scaffold_retirement_flips_noregression/flipB_smoke_ON.json`, and `research/findings/raw/_scaffold_retirement_flips_noregression/flipB_smoke_OFF.json` (each with a `.prov.json` sidecar).

**Flip A (day_33 composer).** ON composer type `Pool1BoundOneBrainComposer` (load 269.48s — the 404-fact spiking re-store, no onebrain sidecar on this rf-saved bundle; speed is secondary); OFF `Pool1BoundComposer` (load 1.7s). Recall ON 20/20, OFF 20/20, per-probe answer mismatches 0 (onebrain recall == rf recall on all 20 facts). ON yes/no: stored fact -> `yes`, wrong patient -> `unknown`. ON moat (out-of-store cue): `what_does` -> None (abstains), `is_it_true` -> `unknown` (no confabulation).

**Flip B (cortical LTM).** ON shards report `enable_substrate_store=True` + `enable_batched_substrate_scan=True`; OFF both False. Recall ON 40/40, OFF 40/40, per-fact answer mismatches 0 (substrate-store recall == numpy-KB recall). ON internal moat abstains (None); OFF moat abstains (None). ON load 69.8s (re-imprint 15,000 facts into synaptic handles), warm recall ~497.5ms/query <!--derived--> (query_wall 19.9s / 40).

**Reversibility.** OFF (`rf` + `BRAIN_LTM_SUBSTRATE_STORE=0` + `BRAIN_BATCHED_SUBSTRATE_SCAN=0`) reproduces the pre-flip behavior on every probe: day_33 recall 20/20, yes/no ok, moat holds; LTM recall 40/40, moat holds.

## Scope, bounds, and honest caveats

- **LTM bound (explicit).** The gate used `wikidata_core_15k` (15,000 facts), NOT the deployed default `wikidata_100k` (78,857), to bound substrate-store build time. This is a valid bounded proxy: the RANK-6 latency de-risk already proved substrate-store recall is FLAT in shard-count out to 78,857 (2026-09-08-rank6-substrate-store-cupy-vram-measured-GO.md, ~2.5 GiB VRAM at 100k scale <!--derived-->; the batched-scan drops the per-query resonate count O(K)->O(1), ~505ms vs the ~1189ms numpy-kb median from 2026-09-02-cupy-scan-vectorize-latency-result-ACCEPT.md <!--derived-->). The answer-identity measured here at 15k is a store-representation property (composite in-weights vs in-array), not a scale-dependent one.
- **Flip A recall was tested at the composer's recall ENTRY POINT, not end-to-end through `brain_chat`'s full GNW pipeline.** Reason: the handler's per-session spiking-organ builds (surprise/expectation-RPE training, the content-selection WM bridge, value-choice, bus-shadow) cost tens of minutes on the first turn and are ORTHOGONAL to both flips (identical for `rf` and `onebrain`; they cannot distinguish the flip). `what_does`->`query_patient` IS the exact recall the handler's gate calls. The FULL production handler path (all organs ON) was already independently confirmed to BUILD + run BOTH flips ON cleanly on the bigger wikidata_100k LTM (prior session's live gate: source `developed-brain:...+LTM`, 114,514,560 synapses installed, 0 OOM/crash before it was interrupted for GPU gaming) <!--derived-->.
- **"Answer-identical", not "byte-identical".** This gate did an EXACT compare of recall ANSWERS (per docs/TERMS.md, byte-identical requires a hash/exact-array compare, which this gate did not do). The flip changes the MECHANISM (composer type; store representation) while preserving the recall answer — that mechanism change onto the spiking substrate IS the scaffold-retirement deliverable. The spiking load-bearingness (lesion collapses the answer) is established by the cited RANK-1 arc and `tests/test_rf_phasor_composer.py`, not re-proven here.
- **Single deployed seed (42).** The bundle + LTM are the deployed artifacts (seed 42). Flip A's full 404-fact parity + 29-ambiguous-cue no-confab moat is the RANK-1 GO; this gate corroborates on a 20-probe sample and adds the reversibility leg.

## Bottom line

Both host shortcuts are retired to the shared spiking substrate as the production default, with no recall regression and a one-env-var revert each: developed-bundle recall now runs the spiking DG-CA3 onebrain composer, and the cortical LTM is genuinely memory-in-weights. This is the culmination of the 2026-09-04 scaffold-retirement priority for these two ranks.
