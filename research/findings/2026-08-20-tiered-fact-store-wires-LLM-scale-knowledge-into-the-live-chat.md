---
type: finding
status: contributing
date: 2026-08-20
mechanism: tiered-knowledge-ltm-integration
lane: integration
integration_faculty: tiered-knowledge-ltm
seeds: [42, 43, 44]
seed-waiver: A drop-in CORRECTNESS + capacity de-risk of a routing/composition wrapper (recall / moat / recency /
  latency / k_max-lift measured through the real agent), not a stochastic effect size; run at 3 seeds anyway because
  the per-concept RF codes are seed-dependent, and it is GO on all 3.
instrument: research/runners/_knowledge_tiered_livechat_derisk.py — installs a TieredFactStore as a real
  BrainConversationalAgent's `composer` (bulk facts in a ShardedPhasorStore LTM, conversation facts in the flat
  buffer) and exercises recall / teach-in-conversation / recency-shadow / moat / k_max-lift / ltm=None-degrade
  THROUGH the agent's public methods (what_does / who_does / is_it_true), with a tools.verdict.Verdict.
runner: research/runners/_knowledge_tiered_livechat_derisk.py
external: NO-EXTERNAL-NEEDED — composes two in-repo validated stores (RFPhasorComposer buffer + the de-risked
  ShardedPhasorStore LTM) behind the composer API; the hippocampal-working-set / cortical-semantic split is the
  biological motif, the measurement is internal.
artifacts:
  - research/findings/raw/_knowledge_tiered_livechat/seed42.json
  - research/findings/raw/_knowledge_tiered_livechat/seed43.json
  - research/findings/raw/_knowledge_tiered_livechat/seed44.json
---
# The tiered fact store wires LLM-scale knowledge into the LIVE CHAT — bulk knowledge queryable at sub-second latency beside the conversation working-set, moat intact, k_max=32 cap lifted

Artifact: research/findings/raw/_knowledge_tiered_livechat/seed42.json (GO; 43/44 GO).

**One line.** The owner's #1 priority is to teach the sim-brain the fundamental knowledge an LLM has, then interact
with it daily and have it learn through that. The sharded fact-store already removed the O(K) query wall
([[2026-08-20-sharded-fact-store-removes-the-O-K-query-wall-knowledge-scales-to-LLM-scale]]); what remained was
*wiring it into the live conversation* past the k_max=32 co-resident working-set cap. This builds the biological
hippocampal-buffer / cortical-LTM split as a **transparent drop-in for the live agent's fact store** and wires it,
opt-in + default-off, into the production load path. The brain now answers over a large body of knowledge in the
same chat where it learns new facts, with the no-confab moat preserved.

## The build (`TieredFactStore`, reuse-by-import, NO `sim/` edit)
A small flat composer stays the active-conversation **BUFFER** (recent, conversation-taught facts — the k_max
working set, unchanged); a **`ShardedPhasorStore`** is the cortical **LTM** (bulk knowledge; a routed query touches
only ONE shard ~K/S facts → sub-second at ANY K). `TieredFactStore` implements the exact composer READ+WRITE API
the live path uses (`store` / `query_patient` / `query_agent` / `ask_yes_no` / `query_chain` / `chain_of_thought` /
`render_fact`) and delegates every other attribute to the buffer, so `agent.composer = TieredFactStore(buffer, ltm)`
is a transparent substitution. A **read checks the buffer, then falls through to the routed LTM shard on an abstain**
(None / "unknown"); a **write goes to the buffer** (the recent working set). The tiers self-consistently encode by
WORD (the buffer's fact-scan string-compares UNBOUND stored words to the query — it never encodes the query into its
codebook, so a bulk-knowledge query simply misses the buffer and falls through, no shared codebook needed). With
`ltm=None` it is byte-identical to the plain buffer (the safe default).

## Production wiring — additive, opt-in, DEFAULT-OFF (= byte-identical)
`developed_brain_io.load_developed_brain(..., ltm_bundle=<path>)` builds the LTM from a separate bundle's facts and
installs the tiered store; `webapp/server.py` exposes it as `BRAIN_LTM_BUNDLE` (unset → the plain flat composer,
byte-for-byte). So a running chat server gains the LTM by pointing one env var at a knowledge bundle — no other
change to the turn pipeline, and the moat/recall/abstain paths are unchanged.

## The 3-seed verdict (numpy CPU, N=5000 bulk facts, D=128) — all GO
<!--derived-->
Through the REAL `BrainConversationalAgent` public methods, N=5000 knowledge facts in the LTM + a handful of
conversation facts in the buffer:
- **KNOWLEDGE RECALL** (LTM tier, via `agent.what_does`): **50/50 (recall 1.000) on all 3 seeds** — the LTM is genuinely consulted on a buffer miss.
- **SUB-SECOND** routed LTM query at 5000 facts: **446–495 ms/query** (3 seeds) — vs the tens-of-seconds a 5000-fact flat scan would cost.
- **TEACH-IN-CONVERSATION**: a fact taught mid-chat answers and landed in the BUFFER, leaving the LTM untouched.
- **RECENCY**: a buffer fact SHADOWS a contradicting LTM fact about the same cue (recent working-set wins).
- **MOAT**: every unknown subject abstains (None / "unknown") — BOTH tiers must abstain for a non-answer.
- **k_max LIFTED**: N=5000 (>> 32) knowledge facts answer correctly while the co-resident BUFFER holds ≤ a few facts
  — bulk knowledge lives in the uncapped sharded LTM, so the working-set cap no longer bounds knowledge.
- **DEGRADE**: with `ltm=None` the tiered store is answer-identical to the plain buffer (the byte-safe default).

## End-to-end through the WIRED production load path with REAL knowledge
Loading a tiny developed brain (conversation facts "dog chases cat", "otter caught clam") with the real 2413-fact
Wikidata bundle as `ltm_bundle`, queried through the agent: `aardvark isa → mammal` (LTM), `dog chases → cat`
(buffer), `snarklebee isa → None` (moat), `elephant isa mammal → yes`, `elephant isa fish → unknown` (honest
abstain, not a false "no"). The seam works with a real knowledge body, not just synthetic facts.

## Honest scope
The router is the ShardedPhasorStore's declared HOST agent-hash scaffold (the faithful version is a learned/spiking
cue→sub-population router). Cross-TIER multi-hop chains are not merged (each chain runs within one tier; the LTM
holds the bulk knowledge graph where chains live). Attributed patients collapse to their noun in the bulk LTM load
(simple SVO is preserved; adjective binding is a buffer/teach path feature). `promote_buffer_to_ltm()` is an
explicit consolidation hook (the sleep-replay hippocampal→cortical transfer), NOT auto-invoked in v1. NEXT: load a
100k+ real-fact bundle into the LTM (the rate-limited Wikidata fetch is in flight) and flip `BRAIN_LTM_BUNDLE` on by
default once a soak confirms no regression — the daily-teachable, knowledge-rich brain.
