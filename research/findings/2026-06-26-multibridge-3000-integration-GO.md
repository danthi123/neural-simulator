# Multi-bridge 3000-concept integration — GO (RoutedComposer wired into the first-chat console)

**Date:** 2026-06-26
**Verdict:** **GO.** The `RoutedComposer` (N-shard composer facade) is built, wired into the first-chat console
behind `--shards N`, and validated on the trained 3,000-concept brain. Multi-bridge **beats single-bridge on the
rare-knowledge tail** (the load-bearing claim), **preserves the no-confab moat (0 FA, incl. cross-shard)**, is
**~2× faster per turn**, **passes the permuted-routing anti-cheat**, and the console rubric is **10/10
regression-free**. No `sim/` edit, no `rf_phasor_composer.py` edit.

Design: `research/findings/2026-06-26-multibridge-deep-knowledge-design.md`.
De-risk reused: `research/runners/_multibridge_stage0_derisk.py` (Stage-0 GO, numpy-CPU).

---

## What was built

1. **`research/runners/routed_composer.py` — `RoutedComposer`.** Wraps N `RFPhasorComposer` shards (each over a
   disjoint ~V/N-concept vocab + its grounded codes) behind a host `word2shard` router (the proven
   `g20_multibridge` pattern). Every cleanup ranges over ~V/N concepts, not the union. Presents the SAME composer
   API the DiscursiveTurn/proposer/agent consume — `store` (agent-anchored + bounded cross-shard codebook
   extension, option 2a), `query_patient`/`query_agent`/`ask_yes_no`/`query_chain`/`render_fact`/
   `update_on_mismatch`/`count_facts`/`elaborate`/`unbind`, `.words`/`.concepts`/`.kb` union views, and
   `_assoc_graph()` over the UNION kb (so the discuss channel sees cross-shard relatedness). Reuse-by-import of the
   VERIFIED Stage-0 blocks (`load_brain`, `split_shards`, `build_shard_composers`, `store_facts_routed`). Shard
   policy: `domain` (g20-category bands, design §2.6) or `partition` (disjoint random split).

2. **`first_chat_console.py` — `--shards N` + `--shard-by {domain,partition}`.** `build_brain_on_codes(shards=1)`
   = today's single composer, **byte-unchanged** (regression-free); `shards>1` builds a `RoutedComposer` over the
   SAME loaded grounded codes. The router is invisible to the DiscursiveTurn/proposer/agent/`audit_moat`.

3. **`research/findings/raw/_facts3000.json` — 2,191 corpus-attested SVO facts** on the 3,000 vocab. Extracted
   with spaCy (`_corpus_svo_extract`) from BOTH corpora and combined (`_combine_facts`):
   - TinyStories: **917** facts (narrative: `boy go park` 169×, `bird fly sky` 63×);
   - Simple-Wiki: **1279** facts (encyclopedic: `people speak language` 41×, `people live area` 40×).

4. **`_multibridge_3000_validate.py`** — the validation sweep (reuses the Stage-0 measurement functions).

---

## The numbers (the 3,000-concept brain, `brain3000pos_w7000.npz_seed42.npz`, D=128, numpy-CPU)

### Fact count
**2,191** distinct corpus-attested facts in `_facts3000.json` (1,013 with unambiguous who/what cues after dedup).

### The crowding IS real — and it lives on the RARE TAIL (the regime "discuss almost anything" needs)
`research/findings/raw/_multibridge_3000_rare_tail.json` — recall on facts drawn from the LOW-count tail:

| tail slice (count ≤ 2) | single-bridge recall | **3-shard recall** | worst shard | 3-shard FA | Δ |
|---|---|---|---|---|---|
| last 40  | 0.900 | **0.950** | 0.923 | 0 | **+0.050** |
| last 80  | 0.887 | **0.975** | 0.962 | 0 | **+0.088** |
| last 120 | 0.875 | **0.950** | 0.944 | 0 | **+0.075** |
| last 200 | 0.875 | **0.955** | 0.947 | 0 | **+0.080** |

- Single-bridge rare-tail recall sits at **0.875–0.900** — exactly the design's predicted crowding (the 0.875
  matches the design's documented 2012-on-one-bridge number). Cleanup confusability rises with codebook density at
  fixed D=128, and the rare/low-frequency facts are where it bites.
- **3-shard recall is a robust 0.950–0.975 — +5 to +9pp**, crossing the 0.95 bar where the single bridge fails it,
  with the worst shard ≥0.92 and **FA = 0 on every slice**. This is the headline: sharding lifts rare-knowledge
  recall above bar.

### Frequent facts: at-ceiling parity (sharding can't help where single is already perfect)
On the top-40 frequent facts: single **1.000**, 3-shard **0.975** (a small cross-shard option-2a cost). The
single bridge is NOT crowded by the frequent facts at D=128 — so the recall benefit is specifically a RARE-TAIL
phenomenon. (This is why the top-N validation sweep below, which draws the most-frequent facts, shows single at
ceiling and understates the benefit — the honest reconciliation.)

### Validation sweep (top-N facts, `_multibridge_3000_validate.json`) — speed + moat + anti-cheat
At 48 top-frequency facts (single already at recall ceiling here):

| config | recall | worst shard | moat FA | t/q | permuted recall (FA) |
|---|---|---|---|---|---|
| single-bridge 3000 | 1.000 | — | 0 | 127.2 ms | — |
| N=3 partition (1000/1000/1000) | 0.979 | 0.952 | 0 | **56.1 ms** | 0.125 (FA 0) |
| N=3 domain (1804/746/450) | 0.979 | 0.968 | 0 | 68.0 ms | 0.115 (FA 0) |
| N=4 partition (750×4) | 1.000 | 1.000 | 0 | **42.5 ms** | 0.094 (FA 0) |

### Moat false-accepts: **0 everywhere**
Single, every shard count, every policy, on absent_what + absent_who + cross-shard-absent cues. The no-confab moat
is fully preserved cross-shard (unknown concept → router abstains; present-but-unstored → shard composer abstains;
never-stored cross-shard cue → abstains, not spuriously matched via the option-2a extension).

### Per-turn time (console, full DiscursiveTurn pipeline)
Median non-first-touch turn: single **2026 ms → 3-shard 1063 ms (~1.9×)**; opinion 2027→1063 ms; known-fact
3686→2628 ms. (The first touch of a topic pays a one-time ~43 s `propose_candidates_about` resonate that is then
cached — present in BOTH single and sharded, a pre-existing DiscursiveTurn characteristic independent of sharding.)

### Console rubric: **10/10, MIXED, moat 0 leaks, VERDICT PASS** (3-shard, `--shard-by partition`)
Regression-free: single-bridge rubric is also **10/10 PASS**. The full chat UX is preserved at 3,000 concepts on
the sharded composer — certain known-facts (`boy goes park`, `girl goes mom`, `bird flies sky`), flagged
hypotheses (the (N)/(D) discuss channels fire), graceful abstention on unknown words, phatic. Build: 23/24 facts
recall correctly via what_does (vs single 24/24 — the 1-fact gap is the cross-shard option-2a cleanup cost; the
rubric still passes 10/10 because known-fact prompts draw from the correctly-recalled subset).

### Discuss-richness
The (D) channel surfaces multiple adjacent flagged facts per topic — e.g. `what is world?` → depth 3, 3 flagged
adjacent propositions (vs the design's described thinned-to-1 single-bridge-at-2012 case). The shared PPMI graph
(209,354 co-occurrence scenes, 2,998/3,000 concepts connected) + the union `_assoc_graph` give cross-shard
adjacency, so the discuss arena is not narrowed by sharding.

### Anti-cheat: permuted-routing control PASSES
Store each fact on the WRONG shard (agent_shard+1) but query with the TRUE router → recall **collapses to
0.09–0.24** (~chance) while **FA stays 0** on every config. The routing is load-bearing AND the moat is not
routing-dependent (a wrongly-routed query abstains, never confabulates). The analogue of the project's
permuted-label controls.

---

## Honest scope / open risks

1. **The recall benefit is a RARE-TAIL phenomenon at D=128.** On frequent facts the single 3000-concept bridge is
   already at the recall ceiling, so sharding's recall value shows specifically on the low-frequency tail (where a
   "discuss almost anything" brain spends most of its knowledge). The original top-N validation sweep masked this;
   the tail sweep is the honest measurement. The speed + moat + anti-cheat wins are uniform.
2. **Cross-shard facts (option 2a) carry a small frequent-fact cost** (1.000 → 0.975/0.979): a handful of
   cross-shard patients clean up to near-neighbors. The design's flagged soft spot (§2.5/§6.2). At N=4 it
   disappears (recall 1.000) — more, smaller shards mitigate. Domain sharding has fewer cross-shard facts per
   shard but imbalanced sizes (1804/746/450 at N=3); partition gives balanced 1000/1000/1000. **Partition is the
   recommended default** (balanced cleanup load, the robust tail-recall win); domain is available for semantic
   coherence.
3. **The taxonomy ceiling (design §1.3, §6.1) is the real #1-goal bottleneck.** Multi-bridge buys recall+speed
   headroom to HOLD many concepts; reaching a discuss-almost-anything vocab ALSO needs a grown word→category spec
   beyond the g20 taxonomy. Separate, higher-leverage work item.
4. **A neural router is deferred** (design §2.3, §6.6): the host `word2shard` index is legitimate bookkeeping (the
   `g20_multibridge` precedent), not cognition — the comprehend/recall/abstain/discuss stay neural inside each
   shard. A future spiking shard-selector is the fully-on-substrate form; not a #1-goal blocker.

---

## Reproduce

```bash
# Step 1 — facts on the 3000 vocab (already produced -> research/findings/raw/_facts3000.json)
SIM_BACKEND=numpy python -m research.runners._corpus_svo_extract \
  --npz bridges/firstchat/brain3000pos_w7000.npz_seed42.npz --corpus data/corpus/tinystories.txt \
  --max-sentences 200000 --min-count 2 --out research/findings/raw/_tinystories_svo_3000.json
SIM_BACKEND=numpy python -m research.runners._corpus_svo_extract \
  --npz bridges/firstchat/brain3000pos_w7000.npz_seed42.npz --corpus data/corpus/simplewiki.txt \
  --max-sentences 200000 --min-count 2 --out research/findings/raw/_simplewiki_svo_3000.json
SIM_BACKEND=numpy python -m research.runners._combine_facts \
  --inputs research/findings/raw/_tinystories_svo_3000.json research/findings/raw/_simplewiki_svo_3000.json \
  --out research/findings/raw/_facts3000.json

# Step 2 — RoutedComposer self-check
SIM_BACKEND=numpy python -m research.runners.routed_composer --n-shards 3

# Step 4 — validation sweep (recall/moat/time/anti-cheat) + console demo + rubric
SIM_BACKEND=numpy python -m research.runners._multibridge_3000_validate \
  --facts-json research/findings/raw/_facts3000.json --n-facts 48 --n-shards 3 --sweep-shards 1,2,3,4
SIM_BACKEND=numpy python -m research.runners.first_chat_console \
  --brain bridges/firstchat/brain3000pos_w7000.npz_seed42.npz --shards 3 --shard-by partition \
  --facts-json research/findings/raw/_facts3000.json --rubric
```

**Bottom line:** the multi-bridge integration is production-ready and GO. It lifts rare-knowledge recall above the
0.95 bar (0.875→0.95+), halves per-turn latency, and keeps the no-confab moat at 0 FA — the recall+speed headroom
the deep-knowledge goal needs, with the moat intact.
