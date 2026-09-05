---
type: finding
status: de-risk-PARTIAL-premise-mis-scoped
date: 2026-09-05
mechanism: scaffold-retirement de-risk (backlog RANK-1) — rebuild the deployed scale787 recall bundle OFF composer_kind='rf' ONTO 'onebrain' (the spiking unbind local_reciprocal_unbind + NEF/Izhikevich cleanup enable_spiking_cleanup, both ON by default in OneBrainComposer)
lane: integration-first (SCAFFOLD-RETIREMENT BACKLOG rank-1)
integration_faculty: content-selection (recall composer)
verdict: PARTIAL / MIS-SCOPED. NO-GO for RANK-1 as written (rebuild the FULL 404-fact scale787 bundle onto a single OneBrainComposer). The block is ARCHITECTURAL, not a tuning wall, so it holds independent of seed. OneBrainComposer is a bounded co-resident BUFFER (hard cap k_max=32); 404 >> 32, so the naive load-override raises "OneBrainComposer store full: k_max=32 reached" on the 33rd fact, and a k_max>=404 rebuild is a ~1.37M-neuron bridge (n_total scales LINEARLY with k_max) — it does not fit the 24 GB RTX-3090 reference and per-query latency at 404 co-resident facts would be minutes (measured ~7-14 s already at 30). The premise "the spiking ops never run on ANY live recall" is ALSO false: the served DEFAULT brain (tiny-demo, 5 facts, fits k_max=32) has run composer_kind='onebrain' since 2026-08-25 (webapp/server.py:_COMPOSER_KIND_DEFAULT). WHERE onebrain FITS, its recall is correct + moat-intact vs the rf oracle (seed 42: 21/21 cues match rf, 0 mismatches; 6-seed robustness verify running, artifact below). The genuine residual — genuinely-SPIKING BULK recall — is a NEW mechanism (a spiking sharded fact store; today's ShardedPhasorStore LTM is HOST FHRR), a substantial build, NOT the "config-flip + rebuild, near-zero risk" RANK-1 claimed. Do NOT flip any production default. RE-SCOPE RANK-1.
artifacts:
  - research/runners/_rank1_composer_bundle_onebrain_derisk.py
  - research/findings/raw/_rank1_composer_bundle_onebrain_derisk_smoke.json
  - research/findings/raw/_rank1_composer_bundle_onebrain_derisk.json
verification: |
  BRAIN_COMPOSER_MERGE=0 python -m research.runners._rank1_composer_bundle_onebrain_derisk
    --bundle bridges/developed/scale787/day_33 --seeds 42,43,44,100,101,102
  (seed-42 mechanism cross-check already captured: recall 21/21 rf==onebrain, 0 mismatches; the full 6-seed
   run adds abstain/moat + VRAM scaling + the aggregate verdict to the .json artifact.)
---

# RANK-1 (scaffold-retirement) — the scale787 -> onebrain flip is MIS-SCOPED: OneBrainComposer is a k_max=32 co-resident BUFFER, not a 404-fact bulk store

RANK-1 called this "the single largest, cheapest reclaim ... NEAR-ZERO risk (config-flip + rebuild)". Checking the
premise against the CURRENT code (the mandated step-1: the map has been wrong before) shows it is mis-scoped. The
mechanism itself is sound and, where it fits, already shipped.

## The premise, checked (four points)

- **P1 — the bundle IS 'rf'. TRUE.** <!--derived--> The deployed `day_33` bundle manifest reads
  `composer_kind: "rf"`, `n_facts=404`, `D=128`, with 788 vocab and 788 grounded codes (all captured in the cited
  smoke artifact). The webapp loads it via
  `load_developed_brain(bundle, composer_kind=_composer_kind_override)`, and `_composer_kind_override` is allowlisted
  to `'slotbinder'` or `None` (webapp/server.py:3892), falling through to the manifest's own `'rf'`. So the scale787
  recall genuinely never runs the spiking ops.

- **P2 — the naive same-bundle flip FAILS.** `OneBrainComposer` has a HARD co-resident cap `k_max=32`
  (one_brain_composer.py:806). It is the recent-conversation BUFFER, not a bulk store. `load_developed_brain(bundle,
  composer_kind='onebrain')` on the 404-fact bundle raises `RuntimeError: OneBrainComposer store full: k_max=32
  reached (shard or raise k_max)` on the 33rd fact.

- **P3 — "never on ANY live recall" is FALSE for the served default.** `webapp/server.py:_COMPOSER_KIND_DEFAULT =
  "onebrain"` (flipped 2026-08-25, commit 135024f70). The out-of-box DEFAULT brain (`tiny-demo`, 5 facts, fits
  k_max=32) ALREADY runs onebrain — the spiking unbind + cleanup DO run on every default recall. scale787 is a
  SELECTABLE developed bundle (from the 787-concept stream-cortex scaling run), not the served default.

- **P4 — a k_max>=404 rebuild does not fit the consumer reference.** `OneBrainComposer.n_total` scales LINEARLY with
  k_max (`n_total = bat_c_base + k_max*cb`, `cb = n_main*V + NP`, one_brain_composer.py:410-417). At k_max=404 that is
  ~1.37M neurons — a ~13x bridge (CLAUDE.md: >100k neurons already need 20 GB+). The de-risk runner measures the VRAM
  slope at k_max in {32,64,128} and extrapolates to 404 vs the 24 GB (24576 MiB) RTX-3090 budget (the
  consumer-hardware-reference principle). Separately, per-query latency is already ~7-14 s at just 30 co-resident
  facts (resonate-and-fire per query); at 404 it would be minutes per turn.

## The deeper reason: there is no SPIKING bulk-fact store

The architecturally-correct home for 404 bulk facts is the routed cortical LTM (`TieredFactStore` = small onebrain
BUFFER + `ShardedPhasorStore`) — how the default brain serves its 15k/100k bulk knowledge. BUT `ShardedPhasorStore`
is built from `RFPhasorComposer` shards — pure HOST FHRR, numpy fast path (`enable_substrate_store=False`, the LTM
default, sharded_phasor_store.py:347), NOT spiking. So today: onebrain (spiking) runs ONLY on the small `<=32`-fact
co-resident buffer; ALL bulk knowledge (15k, 100k, and scale787's 404 if routed to LTM) is served by HOST FHRR.
Making 404-fact recall genuinely spiking therefore requires a NEW mechanism — a spiking sharded fact store — not a
config-flip.

## What IS validated: the onebrain recall mechanism is correct + moat-intact vs rf, WHERE IT FITS

On a FITTING subset (30 of the bundle's own facts, its real vocab + grounded codes), plain OneBrainComposer
(`BRAIN_COMPOSER_MERGE=0`; recall/moat byte-identical to the pool#1-bound production variant per
2026-08-14-onebrain-composer-pool1-DEFAULT-FLIP-GO) vs the rf oracle:

- seeds 42/43/44 (of 42/43/44/100/101/102) each: recall 21/21 cues match rf (`rf_ok=21`, `ob_ok=21`, agreement
  21/21, 0 mismatches); the no-confab moat holds (`rf_confab=0`, `ob_confab=0` on 40 unstored cues); runtime store
  (fresh-cue + a NOVEL word via the vocab_headroom recruit path) recalls correctly. IDENTICAL across all three
  seeds — see `research/findings/raw/_rank1_composer_bundle_onebrain_derisk.json` (the seed-42 cross-check is also
  in `research/findings/raw/_rank1_composer_bundle_onebrain_derisk_smoke.json`). The remaining seeds 100/101/102 +
  the VRAM-scaling feasibility measurement are completing (the runner checkpoints per seed and resumes; the
  ARCHITECTURAL verdict above does not depend on them).

So where onebrain applies (the co-resident buffer), it is correct — and it is already the shipped default for
tiny-demo.

## Verdict

**PARTIAL / MIS-SCOPED.** NO-GO for the RANK-1 flip as written (full 404-fact bundle -> single OneBrainComposer):
blocked by the k_max=32 architecture + VRAM + per-query latency, none of which a rebuild removes. The spiking recall
mechanism is correct where it fits and is already the shipped default for the small buffer. Making 404-fact BULK
recall genuinely spiking is a separate, larger item (a spiking sharded fact store). RANK-1 should be re-scoped and
its "single largest, cheapest reclaim / near-zero risk" framing dropped. No production default is flipped by this
de-risk.
