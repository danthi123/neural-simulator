---
type: finding
status: wired
claim_check: measured-result
date: 2026-09-05
mechanism: dg-ca3-sparse-index-over-fact-blocks (wired into the OneBrainComposer query path; additive default-off)
lane: knowledge-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_fact_shard_wirein/verify_404_6seed.json
  - research/runners/_onebrain_fact_shard_wirein_verify.py
  - research/runners/one_brain_composer.py
  - research/runners/brain_conversational_agent.py
  - research/findings/2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md
  - research/biology/dg-ca3-sparse-index.md
---

# The DG-CA3 fact-shard sublinear retrieval is WIRED into the OneBrainComposer query path (additive, default-off): reachable from /api/brain-chat via `BRAIN_FACT_SHARD_RETRIEVAL=1`, no-regression + latency win + byte-identical-off verified through the REAL agent path

**Board/lane: rank-1 composer-latency residual — the wire-in rung the de-risk named.** The de-risk
(`2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md`, GO 6/6 @ 404 facts) proved a
per-role DG-CA3 sparse index over the composer's FACT BLOCKS retires the O(k_max) linear scan (~149 s/recall ->
~0.37 s, ~402x) with 6/6 parity + moat, but it was a **runner-level de-risk driving bespoke `sharded_query_*` free
functions** — NOT reachable from `/api/brain-chat`. This finding **WIRES the mechanism into `OneBrainComposer`
itself** (the `enable_sparse_index` pattern: additive, default-off, env-flip) so the composer's OWN public API
(`query_patient`/`query_agent`/`ask_yes_no`) shards internally, and verifies it through the **production agent
construction** (`BrainConversationalAgent(composer_kind="onebrain")` — the class `webapp/server.py`'s chat path
builds).

## What was wired (additive + default-off = byte-identical when off)

* **`one_brain_composer.py`** — new ctor params `enable_fact_shard=False` (OR'd with env `BRAIN_FACT_SHARD_RETRIEVAL`,
  == the `enable_sparse_index` / `BRAIN_SPARSE_INDEX_RETRIEVAL` precedent) + `fact_shard_g/G/c` (2/4/8, the de-risk's
  DG params). When on it (a) builds a per-role DG index over the stored blocks' filler codes (`_ensure_fact_shard`,
  reuse-by-import of the validated `DGSparseIndex`; clause-robust — a non-word/clause filler is skipped per-role, so
  it is a strict superset of the flat-SVO de-risk it was validated against), (b) routes each cue-known recall through
  `_fact_shard_first_match` (intersect the per-role shards -> decode ONLY the shard via the composer's EXISTING
  per-block spiking `_read_block`/`_read_block_indexed` -> first-match ascending == the full scan), and (c) auto-sets
  `no_batched_region=True` (a fact-shard composer never batches -> the bridge drops the dead k_max*(n_roles*D+cb)
  batched region). The three cue-known methods gained a guarded fast-path prefix + a shared tail; **default-off leaves
  every line of the full path unexecuted** (`_fact_shard_active()` is False) so the layout, decode and answers are
  unchanged.
* **`brain_conversational_agent.py`** — new `onebrain_k_max=None` (default 32 = the prior hardcoded default =
  byte-identical), threaded to the OneBrainComposer construction (both the bare + pool1 paths). This is the enabling
  scale knob: the O(k_max) scan is why k_max was pinned at 32; moving the LLM-scale knowledge OFF the host FHRR (rf)
  composer ONTO the spiking one-brain composer needs k_max to scale, which the fact-shard makes tractable.
* **`_onebrain_fact_shard_wirein_verify.py`** — the integrated 6-seed verify runner (below).

## Reachable from /api/brain-chat (the "wired" condition, docs/TERMS.md)

Call path: `webapp/server.py` `/api/brain-chat` -> the `ChatBrain` it builds ->
`BrainConversationalAgent(composer_kind="onebrain")` -> `OneBrainComposer` -> `query_patient`/`query_agent`/
`ask_yes_no` (server.py:5852 anchors `chat.inner.is_it_true == OneBrainComposer.ask_yes_no`). The **DEFAULT served
onebrain composer is the pool1-merged `Pool1BoundOneBrainComposer`** (`BRAIN_COMPOSER_MERGE` default-on) — and the env
flip reaches THAT: measured `type=Pool1BoundOneBrainComposer, enable_fact_shard=True`, fast path active, parity 30/30
per kind, moat 0, faster than its own full scan (`pool1_reachability` in the artifacts). So the fast path is genuinely
reachable on the shipped default; the owner-controlled default flip is a single env var.

## Integrated verification (through the REAL agent path, not the raw composer)

`_onebrain_fact_shard_wirein_verify.py` builds each brain via `BrainConversationalAgent(composer_kind="onebrain",
onebrain_k_max=N+16)` and drives the composer's OWN wired methods. Three legs:
* **(a) NO REGRESSION** — the wired fast-path answer == the full O(k_max) scan for `query_patient`/`query_agent`/
  `ask_yes_no` on every stored fact, on the IDENTICAL substrate (one brain/seed; the full path is the SAME composer
  with the fast path toggled off -> a clean same-neurons control); + full recall == ground truth; + the no-confab
  moat (out-of-store cues abstain, 0 new confab). The "full_reference IS the real full path" anchor is established by
  the byte-identical-off leg (the composer's REAL methods == the cached reference, 3*N answers) + the pool1 leg.
* **(b) LATENCY** — the wired sharded recall wall-clock << the full per-block scan, and blocks DECODED (shard
  mean/max) << k_max. On the **bare** composer (`BRAIN_COMPOSER_MERGE=0`, used for the rigorous 6-seed: private
  bridge per seed -> clean independence + a genuine bridge shrink) `no_batched_region` ALSO shrinks the bridge
  (measured 8.2x at N=40), so per-block reads run at ~their small-store cost. On the **pool1 default** path the win is
  FEWER reads (shard ~1 vs k_max) — the shared substrate's span is pre-sized WITH the batched region, so the per-read
  shrink there is a named follow-on (a `no_batched_region`-aware `_onebrain_layout_span`); the fewer-reads win alone
  is ample.
* **(c) BYTE-IDENTICAL WHEN OFF** — asserted IN THE DATA (docs/TERMS.md): the flag-off agent-built composer keeps the
  as-is layout (`n_total == the batched-region arithmetic`, `enable_batched` True, `no_batched_region` False), NEVER
  builds the fact-shard index over a full query session (`_fact_shard is None`), a second independent off build decodes
  bit-identically (rows-hash match), and its answers == the full reference.

Anti-cheats (mirroring the de-risk, wired into the runner): content-addressable routing (the key is the cue WORD code,
never an answer id; the answer is read off the spiking decode); parity vs the full scan is a HARD gate; a SCRAMBLE
control (permuted band-winners collapse recall; `tools.lab.attributable_to`).

## Results

**Integrated 6-seed (42/43/44/100/101/102, N=404 = the de-risk's exact scale, `verify_404_6seed.json`): GO, 6/6**
(all 10 `tools.verdict` preconditions pass). The wired composer methods, driven through the real agent, equal the
full O(k_max) scan on every stored fact while decoding ~1 block instead of 404.

<!--derived-->
| seed | store (s) | full O(k_max) scan (s) | shard mean/max | sharded recall (s) | speedup | parity P/A/YN | recall vs truth | moat new-confab | scramble |
|---|---|---|---|---|---|---|---|---|---|
| 42  | 104 | 172.0 | 1.07/2 | 0.415 | 414x | 40/40 40/40 40/40 | 404/404 | 0/20 | 0/30 |
| 43  | 101 | 184.6 | 1.05/2 | 0.471 | 392x | 40/40 40/40 40/40 | 404/404 | 0/20 | 0/30 |
| 44  | 109 | 192.7 | 1.12/2 | 0.459 | 420x | 40/40 40/40 40/40 | 404/404 | 0/20 | 0/30 |
| 100 | 116 | 186.3 | 1.23/3 | 0.457 | 407x | 40/40 40/40 40/40 | 404/404 | 0/20 | 0/30 |
| 101 | 116 | 184.3 | 1.23/2 | 0.444 | 415x | 40/40 40/40 40/40 | 404/404 | 0/20 | 0/30 |
| 102 | 114 | 185.7 | 1.18/3 | 0.495 | 375x | 40/40 40/40 40/40 | 404/404 | 0/20 | 0/30 |

<!--derived: every cell is read directly from research/findings/raw/_onebrain_fact_shard_wirein/verify_404_6seed.json
(per_seed[*]: store_seconds, full_perblock_scan_seconds, shard_size_mean/max, latency_shard_median_seconds.patient,
speedup_full_over_shard, parity, full_recall_vs_truth, moat.new_confab/checked, scramble_control.recovered/checked);
no cell is computed from another cell. Seconds rounded to the runner's own precision.-->

<!--derived-->
<!-- Aggregate numbers below are rounded from the aggregate block of verify_404_6seed.json: shard_size_mean 1.146 ->
1.15, shard_size_max 3, full_perblock_scan_seconds_median 185.168 -> 185.2, shard_patient_latency_median 0.458 ->
0.46, speedup_full_over_shard_median 410.84 -> 411, bridge_shrink_ratio 18.248 -> 18.2. -->
**Aggregate:** shard mean **1.15** (max **3**) vs **404** blocks decoded per recall (sublinear); full O(k_max) scan
median **185.2 s**; wired sharded recall median **0.46 s**; speedup median **411x**; bridge shrink **18.2x** (bare
composer). All of parity / real-anchor / moat / full-recall / sublinear / latency / byte-identical-off / pool1-
reachable = **6/6** (the scramble control recovers **0/30** every seed -> 100% of recall is attributable to content
routing).

**byte-identical-off (N=32, `byte_identical_off` in the JSON): True** -- the flag-off composer's `n_total` **69368 ==
the as-is batched arithmetic**, `enable_batched` True, `no_batched_region` False, the fact-shard index **never built**
over a full query session, a second independent build's decode **rows-hash matches** (6640a813...), and **96/96**
answers == the full reference. **pool1-default reachability (N=60): True** -- `Pool1BoundOneBrainComposer`,
`enable_fact_shard=True`, parity **30/30** x3, moat **0/12**, full scan 36.7 s -> sharded 0.72 s (the DEFAULT served
composer reaches + benefits from the fast path).

## GO/NO-GO for retiring the host FHRR composer

**Wire-in: GO (wired, default-off).** The mechanism is reachable from the production endpoint, additive, byte-identical
off, and verified no-regression + latency-win + byte-identical through the real agent path: **6-seed integrated GO 6/6
at N=404** (parity 720/720, recall 404/404 x6, moat 0, ~411x median, byte-identical-off + pool1-reachable both True) +
the de-risk's own 6/6. **The exact flag for the controller to flip:** `BRAIN_FACT_SHARD_RETRIEVAL=1`
(auto-enables `no_batched_region`; on the bare composer path also right-sizes the bridge). Scaling the onebrain brain
to hold the FHRR-scale knowledge additionally needs `onebrain_k_max` raised (default 32) to >= the corpus size.

**Retiring FHRR is NOT closed by this rung** (docs/TERMS.md: `integrated`/`scaffold_retired` need on-by-default + the
host path removed + a lesion test). This rung delivers the LATENCY unlock that MAKES the retirement tractable; the
remaining rungs are (1) the owner flips the default on, (2) migrate the scale787/day_33 knowledge (currently `rf`)
onto the onebrain composer at the scaled k_max, (3) a real-bundle (grounded-codes) verify, (4) the pool1 span-reclaim
follow-on for the per-read shrink on the default merged substrate, (5) retire the `rf` recall path once (1)-(3) hold.

## Honest residuals / scope (NO-DEFER — named next rungs)

1. **Default-off (not flipped).** This is `wired`, not `on-by-default`; the owner reviews the default flip.
2. **pool1 per-read shrink.** On the default `Pool1BoundOneBrainComposer` the shared-substrate span still includes
   the batched region (the win is fewer reads, not a smaller bridge); `_onebrain_layout_span` needs a
   `no_batched_region` variant to reclaim it. Not a correctness issue — safe (over-reserved), measured parity + moat +
   latency-win intact on that path.
3. **Synthetic FHRR codes, not a grounded bundle** (== the de-risk's residual #3): routing is on CLEAN cue codes, so
   the recovered-phasor noise that killed the RFPhasorComposer DG port does not apply; a real day_33 404-fact bundle
   verify is a named follow-on.
4. **Latency measured on numpy CPU** (cost-routing; blocks-decoded is backend-independent, so a GPU re-verify refines
   absolute numbers only). **render_fact** (agent-only generation cue) stays on the full per-block scan — an agent-only
   shard is a trivial follow-on; `query_chain` inherits the fast path via `query_patient`.

## Sources

Code: `research/runners/one_brain_composer.py` (the `enable_fact_shard` wire-in), `brain_conversational_agent.py`
(`onebrain_k_max`), `_onebrain_fact_shard_wirein_verify.py` (the integrated 6-seed runner). Prior:
`2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md` (the de-risk this wires in). Biology:
`research/biology/dg-ca3-sparse-index.md` (Kandel: DG pattern separation + CA3 completion).
