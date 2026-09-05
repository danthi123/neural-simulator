---
type: finding
status: de-risk-PARTIAL-premise-mis-scoped
date: 2026-09-05
mechanism: scaffold-retirement de-risk (backlog RANK-1) — rebuild the deployed scale787 recall bundle OFF composer_kind='rf' ONTO 'onebrain' (the spiking unbind local_reciprocal_unbind + NEF/Izhikevich cleanup enable_spiking_cleanup, both ON by default in OneBrainComposer)
lane: integration-first (SCAFFOLD-RETIREMENT BACKLOG rank-1)
integration_faculty: content-selection (recall composer)
verdict: PARTIAL / MIS-SCOPED. The onebrain recall MECHANISM is a validated GO — 6/6 seeds (42/43/44/100/101/102), onebrain recall == the rf oracle (21/21 cues EACH seed, 0 mismatch), no-confab moat intact (0 confab / 40 unstored cues, rf also 0), runtime store (incl. the vocab_headroom recruit path) OK — at BUFFER scale (30 co-resident facts). NO-GO for the RANK-1-as-written flip (rebuild the FULL 404-fact scale787 bundle onto ONE OneBrainComposer), but the blocker is NOT VRAM (my first analytic guess was WRONG, refuted by direct measurement): a k_max=404 build is 1,374,504 neurons yet only ~563 MiB (the RF/phasor coresident bridge is sparse, not a dense Izhikevich net), builds in 49 s, stores 404 facts in 30 s — fits the 24 GB 3090 easily. The real blocker is PER-QUERY LATENCY: ~114-150 s per recall at 404 co-resident facts (two runs: 114 s standalone probe, 150 s in-runner; vs ~6 s at 30) because the resonate-and-fire cleanup scans ALL k_max blocks (O(k_max)/query) — which is exactly why OneBrainComposer.k_max defaults to 32 (a small, fast buffer). A ~2-minute recall per live turn is not a shippable flip. The genuinely-scalable spiking-bulk-recall path is a SHARDED spiking store (per-query O(K/shards)); today's LTM ShardedPhasorStore is host-FHRR, so that is a NEW mechanism, not the "config-flip + rebuild, near-zero risk" RANK-1 claimed. P3: the served DEFAULT brain (tiny-demo, 5 facts) already runs onebrain since 2026-08-25. Do NOT flip any production default; RE-SCOPE RANK-1 to a sharded spiking store.
artifacts:
  - research/runners/_rank1_composer_bundle_onebrain_derisk.py
  - research/findings/raw/_rank1_composer_bundle_onebrain_derisk.json
  - research/findings/raw/_rank1_composer_bundle_onebrain_derisk_kmax404_probe.json
  - research/findings/raw/_rank1_composer_bundle_onebrain_derisk_smoke.json
verification: |
  BRAIN_COMPOSER_MERGE=0 python -m research.runners._rank1_composer_bundle_onebrain_derisk
    --bundle bridges/developed/scale787/day_33 --seeds 42,43,44,100,101,102 --attempt-full
  (6/6 seeds: onebrain==rf 21/21 recall, 0 moat confab, store OK, at 30 co-resident facts; the k_max=404
   full-scale probe: 1.37M neurons, build ~46-49s, store ~28-31s, VRAM ~294-563 MiB, per_query ~114-150s
   [two runs] -> latency_ok=false -> verdict PARTIAL/MIS-SCOPED.)
---

# RANK-1 (scaffold-retirement) — the scale787 -> onebrain flip is MIS-SCOPED: OneBrainComposer is a k_max=32 co-resident BUFFER, not a 404-fact bulk store; the wall is per-query LATENCY, not VRAM

RANK-1 called this "the single largest, cheapest reclaim ... NEAR-ZERO risk (config-flip + rebuild)". Checking the
premise against the CURRENT code (the mandated step-1: the map has been wrong before), plus DIRECT measurement at
full 404-fact scale, shows it is mis-scoped. The mechanism itself is sound and, where it fits, already shipped.

## The premise, checked (four points)

- **P1 — the bundle IS 'rf'. TRUE.** <!--derived--> The deployed `day_33` bundle manifest reads
  `composer_kind: "rf"`, `n_facts=404`, `D=128`, with 788 vocab and 788 grounded codes (all captured in the cited
  artifacts). The webapp loads it via `load_developed_brain(bundle, composer_kind=_composer_kind_override)`, and
  `_composer_kind_override` is allowlisted to `'slotbinder'` or `None` (webapp/server.py:3892), falling through to
  the manifest's own `'rf'`. So the scale787 recall genuinely never runs the spiking ops.

- **P2 — the naive same-bundle flip FAILS.** `OneBrainComposer` has a HARD co-resident cap `k_max=32`
  (one_brain_composer.py:806). It is the recent-conversation BUFFER, not a bulk store. `load_developed_brain(bundle,
  composer_kind='onebrain')` on the 404-fact bundle raises `RuntimeError: OneBrainComposer store full: k_max=32
  reached (shard or raise k_max)` on the 33rd fact.

- **P3 — "never on ANY live recall" is FALSE for the served default.** `webapp/server.py:_COMPOSER_KIND_DEFAULT =
  "onebrain"` (flipped 2026-08-25, commit 135024f70). The out-of-box DEFAULT brain (`tiny-demo`, 5 facts, fits
  k_max=32) ALREADY runs onebrain — the spiking unbind + cleanup DO run on every default recall. scale787 is a
  SELECTABLE developed bundle (from the 787-concept stream-cortex scaling run), not the served default.

- **P4 — a k_max=404 rebuild FITS the consumer reference; the wall is LATENCY (measured — my analytic guess was
  WRONG).** `n_total` does scale linearly with k_max (k_max=404 -> 1,374,504 neurons), and I first guessed
  ">100k neurons need 20 GB+" (CLAUDE.md). DIRECT measurement REFUTES that: the OneBrainComposer bridge is an
  RF/phasor coresident substrate (structured/sparse), NOT a dense Izhikevich net. k_max=404: build 48.6 s, VRAM
  ~563 MiB (nvsmi delta) / ~260 MiB (cupy pool); storing all 404 facts = 30.5 s (0.076 s/fact). Well within 24 GB.
  The residual cost is PER-QUERY LATENCY: the resonate-and-fire cleanup scans all co-resident blocks
  (O(k_max)/query), so a recall is ~6 s at 30 co-resident facts and **~114-150 s at 404** (two independent runs).
  That ~2-minute-per-turn cost (not VRAM) is why k_max is a small buffer, and why the naive full-co-resident flip
  is a NO-GO for live use.

## The deeper reason: there is no SPIKING bulk-fact store

The architecturally-correct home for 404 bulk facts is the routed cortical LTM (`TieredFactStore` = small onebrain
BUFFER + `ShardedPhasorStore`) — how the default brain serves its 15k/100k bulk knowledge — because a SHARDED store
routes each query to one shard (per-query O(K/shards), not O(K)). BUT `ShardedPhasorStore` is built from
`RFPhasorComposer` shards — pure HOST FHRR, numpy fast path (`enable_substrate_store=False`, the LTM default,
sharded_phasor_store.py:347), NOT spiking. So today: onebrain (spiking) runs only on the small `<=32`-fact buffer;
all bulk knowledge is served by HOST FHRR. Making 404-fact recall genuinely spiking AND interactive therefore
requires a NEW mechanism — a SHARDED spiking fact store (onebrain shards / making ShardedPhasorStore's shards
onebrain) — not a config-flip.

## What IS validated: the onebrain recall mechanism is correct + moat-intact vs rf (6/6 seeds), WHERE IT FITS

Plain OneBrainComposer (`BRAIN_COMPOSER_MERGE=0`; recall/moat byte-identical to the pool#1-bound production variant
per 2026-08-14-onebrain-composer-pool1-DEFAULT-FLIP-GO) vs the rf oracle, on a FITTING subset (30 of the bundle's
own facts, its real vocab + grounded codes), seeds 42/43/44/100/101/102 — see
`research/findings/raw/_rank1_composer_bundle_onebrain_derisk.json`:

- EVERY seed: recall 21/21 cues match rf (`rf_ok=21`, `ob_ok=21`, agreement 21/21, 0 mismatches); the no-confab
  moat holds (`rf_confab=0`, `ob_confab=0` on 40 unstored cues); runtime store (fresh-cue + a NOVEL word via the
  vocab_headroom recruit path) recalls correctly. `attributable_to` (tools.lab) on recall errors and moat confabs
  is UNDEFINED (both arms ~0) — the composer flip introduces NO recall/moat error where the facts FIT the buffer.

So where onebrain applies (the co-resident buffer), it is correct — and it is already the shipped default for
tiny-demo.

## Verify-go note: the automated verdict said GO; the latency probe overturned it

The runner's first pass (mechanism + VRAM only, no latency) printed `verdict: GO`. Adversarial scrutiny — building
k_max=404 and TIMING a real recall — refuted it: ~114-150 s/query. The runner now measures full-scale latency under
`--attempt-full` and gates the full-flip verdict on it, so the automated verdict (`PARTIAL / MIS-SCOPED`,
`full_flip_feasible=false`, `latency_ok=false`, `per_query_s_full=150.14`) matches the corrected one.

## Verdict

**PARTIAL / MIS-SCOPED.** The onebrain recall mechanism is a validated GO at buffer scale (6/6, moat intact) and is
already the shipped default for tiny-demo. NO-GO for the RANK-1 flip as written (full 404-fact bundle -> one
OneBrainComposer): VRAM fits, but per-query latency is ~114-150 s at 404 co-resident (the O(k_max) scan — why k_max
is 32). Making 404-fact BULK recall genuinely spiking AND interactive is a separate, larger item (a SHARDED spiking
fact store). RANK-1 should be re-scoped and its "single largest, cheapest reclaim / near-zero risk" framing dropped.
No production default is flipped by this de-risk.
