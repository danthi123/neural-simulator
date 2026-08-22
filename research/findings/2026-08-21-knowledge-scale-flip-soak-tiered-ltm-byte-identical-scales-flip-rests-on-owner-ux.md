---
type: finding
status: contributing
date: 2026-08-21
mechanism: knowledge-scale-tiered-ltm-flip-soak
lane: integration
integration_faculty: tiered-knowledge-ltm
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: The load-bearing claim (byte-identity of the routed store vs the unsharded store) is a STRUCTURAL
  property of agent co-location — all of a subject's facts land in one shard over a SHARED codebook, so
  first-match-within-shard == first-match-over-store for that subject — true for ANY seed BY CONSTRUCTION. Run at
  all 6 mission seeds anyway (the per-concept RF codes are seed-dependent) and it is GO on all 6.
instrument: research/runners/_knowledge_scale_flip_soak.py — builds an LTM (closed-form bulk bind) at N well past
  the k_max=32 cap, installs a TieredFactStore as a REAL BrainConversationalAgent's composer, and checks through
  the agent's own recall methods (what_does / is_it_true): byte-identity vs a plain-flat unsharded oracle,
  no-confab moat, recall-at-scale, warm routed latency, the ltm=None degrade, and the REAL production load path
  (load_developed_brain(ltm_bundle=<persisted store>) == BRAIN_LTM_BUNDLE), with a tools.verdict.Verdict.
runner: research/runners/_knowledge_scale_flip_soak.py
external: NO-EXTERNAL-NEEDED — composes in-repo validated stores (RFPhasorComposer buffer + ShardedPhasorStore LTM
  + the closed-form bulk bind) behind the composer API; the measurement is internal.
artifacts:
  - research/findings/raw/_knowledge_scale_flip_soak_verdict.json
---
# The knowledge-scale flip's no-regression soak is GO — the tiered/sharded LTM is answer-identical to the unsharded store and scales past the k_max cap over 6 seeds; the default-on flip now rests on an owner-UX decision, not a technical wall

Artifact: `research/findings/raw/_knowledge_scale_flip_soak_verdict.json` (GO).

**One line.** The ledger row `tiered-knowledge-ltm` (`on_by_default: NO`) says the `BRAIN_LTM_BUNDLE` default-on
flip is "awaiting ... a soak/no-regression" + "an owner-UX flip". This closes the **soak/no-regression** half at 6
seeds and at scale: turning the tiered knowledge store on changes NO answer (byte-identical to the unsharded
store), the no-confab moat is intact, and 100k facts load + recall correctly. The remaining gate is the **owner-UX
product decision** (which knowledge bundle ships as the default) — not a technical wall.

## Verify-first correction (what the flip actually is on `main`)

The task was framed around an in-flight `BRAIN_SHARDED_STORE` / `ShardedPhasorStore.from_existing_composer` wiring
(branch `research/knowledge-scale-sharding`, commit dc938a50). That branch is **NOT on `main`** and is **superseded**
by `main`'s **`TieredFactStore`** (buffer + routed `ShardedPhasorStore` LTM), wired opt-in as **`BRAIN_LTM_BUNDLE`**
(findings 2026-08-20-sharded-fact-store-... + 2026-08-20-tiered-fact-store-... + 2026-08-21-closed-form-bulk-bind-...).
This soak validates the tiered store that is actually on `main`; the superseded branch was not merged.

**There is no "fact-cap constant to raise."** The k_max=32 cap is the co-resident spiking WORKING-SET (the buffer),
and the tiered architecture already lifts it: bulk knowledge lives in the **uncapped** sharded LTM while the buffer
keeps only the recent working set. Raising `OneBrainComposer.k_max` would grow the spiking substrate LINEARLY
(`n_total ≈ k_max·(block + n_roles·D + n_main·V + NP)` — ~24M neurons at k_max=2500, V≈3k) and is the WRONG lever;
the query wall is removed by sharding (O(K)→O(K/S)) and the build wall by the closed-form bulk bind (~356-670×).

## What the soak proves (6 seeds, numpy CPU, D=128)

Through the REAL `BrainConversationalAgent.what_does / is_it_true`, with a `TieredFactStore(buffer, ltm)` installed
as the agent's composer (the exact production shape):

| property | result |
|---|---|
| **byte-identity** vs the plain-flat unsharded oracle (exact answer compare), N=1000 (31× cap, 6 seeds) + N=2000 (62× cap, seed 42) | **0 mismatches** (`what_does` + `is_it_true`) |
| **no-confab moat** (unknown agent / unknown action → abstain), all cells | **0 confabulations**; abstains identically to the oracle |
| **recall at scale** (live answer == the ordered first-match), all N incl. 100k | **1.0** on all 6 seeds |
| **scale / k_max lifted**: N=20000 + N=100000 facts all loaded (total_facts==N), buffer holds ≤ few | **loaded**; buffer ≤ 8 (cap lifted) |
| **ltm=None degrade** = answer-identical to the plain buffer (= BRAIN_LTM_BUNDLE unset = today) | **identical** (0 mismatches) |
| **production load path** `load_developed_brain(ltm_bundle=<persisted store>)` (the exact BRAIN_LTM_BUNDLE path) | **GO** — LTM recall + buffer recall + moat, through the agent |
| **verdict** | **GO (6/6 seeds)** |

**The byte-identity guarantee (the moat).** Agent co-location: every fact ABOUT a subject lands in one shard
(`hash(agent) mod S`) over a shared codebook, so first-match within that shard == first-match over the whole store
for that subject. Routing changes no answer. Verified in the data by exact compare (0 mismatches), not inferred.

## Latency is characterized, not a no-regression gate (a newly-measured O(V·D) term)

<!--derived-->
Warm routed-recall median (heavily-loaded CPU during the run): **~130 ms @ N=1000 · ~250 ms @ N=2000 · ~510 ms @
N=20000 · ~1.3 s @ N=100000**. Routed recall is **sub-second through ~20k** and stays **tractable (~1.3 s) at
100k** — vs the *minutes* an unsharded O(K) scan would cost at 100k (that IS the wall sharding removes; the plain
oracle is ~2.2 s/query at K=1000 alone). The rise above 1 s at 100k is a real, newly-characterized term: the
per-query codebook **cleanup is O(V·D)** in the VOCABULARY V (distinct concepts), which here grows with the fact
count (a pathological 100k-distinct-entity synthetic KB). It is a UX/perf property, **not** a no-regression failure
(answers are byte-identical either way), so it does not gate the soak — but it IS the next technical rung for a
100k+-distinct-entity default (a candidate-restricted cleanup, or the learned/spiking cue→sub-population router that
narrows the cleanup). For a moderate default bundle (≤ ~20k distinct entities) recall is sub-second today.

## Implementation note for whoever ships the default bundle

`load_developed_brain`'s facts-only BUILD PATH calls `build_ltm_from_facts(...)` **without `fast=True`** (the neural
resonate bind, ~52 ms/fact → a 100k facts-only bundle is ~1.5 h to load). The flip should ship a **PERSISTED**
sharded store: build once offline with `fast=True` (closed-form, recall-identical) + `ShardedPhasorStore.save`;
`load_developed_brain` then reloads it in seconds via `ShardedPhasorStore.load` (the manifest.json fast path). An
optional additive win is to pass `fast=True` in the facts-only build path (recall-identical per the closed-form
finding), removing the build wall there too.

## Verdict on the flip: NO-GO autonomously — the residual is an owner decision, not a wall

The **soak/no-regression gate is GREEN** (6/6). The `BRAIN_LTM_BUNDLE` default-on flip is **not** taken here because
it needs two OWNER/PRODUCT decisions the ledger reserves (`on_by_default: NO` note, "Any large-KB default needs the
owner-UX flip"): **(1)** which knowledge bundle ships as the default — none exists in-repo today (the raw
`wikidata5m` corpus is on disk but not built into a shippable bundle), and pointing `BRAIN_LTM_BUNDLE` at a large KB
by default changes what the brain knows by default + repo size + load time; **(2)** whether to keep an unset→off
escape (a large-KB default breaks the current "unset → byte-identical" contract; a separate on/off knob would be
needed). Both are product calls. The k_max cap, the O(K) query wall, and the build wall are already removed on
`main`; with a GREEN soak the flip is a one-decision step for the owner.

## Honest scope (brain-based-only)

- The router `hash(agent) mod S` is a DECLARED host scaffold (ledger `scaffold_retired: NO`); the in-shard FHRR
  recall + the no-confab moat are the genuine reads. The faithful version is a learned/spiking cue→sub-population
  router (hippocampal indexing; Teyler & Rudy 2007) — which would ALSO narrow the O(V) cleanup above.
- The closed-form bulk bind is a declared bulk TEACHER-LOAD optimisation (recall-identical to the neural resonate,
  measured); the QUERY / recall (the cognition) stays fully neural (resonate unbind + cleanup), unchanged.
- Byte-identity is verified at tractable N (the unsharded oracle is O(K) — the wall itself); the property is
  structural, so it holds at any N by construction. NOT a phenomenal claim.
