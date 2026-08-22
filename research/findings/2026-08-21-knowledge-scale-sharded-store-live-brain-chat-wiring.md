---
type: finding
status: contributing
date: 2026-08-21
mechanism: knowledge-scale-sharded-fact-store-live-brain-chat-wiring
lane: integration
---

# Knowledge-scale: wire the agent-routed sharded fact-store into the LIVE brain_chat recall path (opt-in, default-OFF)

**Board #66 (owner #1) / #127.** The de-risked, scale-confirmed agent-routed FHRR sharded fact-store
(`research/runners/sharded_phasor_store.py`; GO to K=10000 at 18.8x — finding
`2026-08-21-knowledge-scale-fhrr-fact-store-sharding-routed-capacity-GO`) is now WIRED into the live `webapp/server.py`
`brain_chat` recall path behind an opt-in flag, ADDITIVELY. Default-OFF is byte-identical to today; when set, the
chat's fact recall routes through the sharded (scale-capable) store instead of the single-store composer.

## What was wired (additive, default-OFF)

1. **`ShardedPhasorStore` is now a drop-in composer** (`research/runners/sharded_phasor_store.py`, additive):
   - a `composer_factory` ctor arg (default `RFPhasorComposer` — the de-risk is UNCHANGED);
   - read-only `kb` (aggregated over shards) + `words`/`concepts`/`roles`/`pol_words` passthroughs, because the live
     chat reads `composer.kb` (`ChatBrain._refresh_facts`, `agent.elaborate`) and `composer.words` as a duck;
   - `ShardedPhasorStore.from_existing_composer(composer, n_shards=16)` — builds a sharded store that SHARES the
     source composer's codebook object (identical phasor codes, incl. grounded overrides + runtime-grown words) and
     RE-HOMES the source's already-stored facts routed by `hash(agent) mod S`, in order (per-agent first-match
     preserved). Handles a plain, attributed, or Clause patient + a bound polarity. Works for a bare
     `RFPhasorComposer` OR a `OneBrainComposer` (whose codebook lives on its inner `.comp`, whose `kb` carries the
     fact-dicts).

2. **`BRAIN_SHARDED_STORE` flag in `webapp/server.py`** (`_sharded_store_enabled` + `_maybe_shard_composer`, called
   once in `_build_chat_brain` before the `ChatBrain` is constructed, for all three brain sources). Unset → returns
   the agent UNTOUCHED (imports nothing, does no work: the single-store path is byte-unchanged). Set → replaces the
   `BrainConversationalAgent.composer` with `ShardedPhasorStore.from_existing_composer(...)` and refreshes the agent's
   cached `_composer_has_hear`. `BRAIN_SHARDED_STORE_SHARDS` (default 16) sets S. Any sharding failure is caught and
   logged, leaving the single-store composer in place — an opt-in capacity substrate NEVER breaks the chat.

## Why byte-identical holds

Agent co-location: every fact ABOUT a subject lands in ONE shard (`hash(agent) mod S`), and all shards share the
source's codebook, so first-match WITHIN a subject's shard == first-match over the whole store for that subject. Every
agent-cued read (`query_patient`, `ask_yes_no`, `render_fact`, each `chain_of_thought`/`query_chain` hop) returns the
identical answer while scanning ~K/S facts. The one cue lacking the agent (reverse `query_agent`) fans out to all shards.

## Verification

`research/runners/_knowledge_scale_sharded_store_live_wiring_verify.py`. Two independent byte-identical checks + moat.

Results artifact: `research/findings/raw/_knowledge_scale_sharded_live_wiring_verdict.json` (verdict GO, 9
preconditions, all met).

seed-waiver: the GO claim is byte-identical ROUTING — the sharded store returns the same answer as the single store
for every agent-cued read, and the moat abstains identically. This is a STRUCTURAL property of agent co-location (all
of a subject's facts land in one shard by `hash(agent) mod S` over a SHARED codebook, so first-match-within-shard ==
first-match-over-store for that subject), true for ANY codebook seed BY CONSTRUCTION, not a statistical outcome — the
same waiver the parent sharding finding carries. There is nothing seed-distributional to average; a mismatch would be a
routing/codebook bug, which the 0-mismatch battery rules out at the tested seed and every seed by the same construction.

**PART A — store level (load-bearing), reproduced LOCALLY (SIM_BACKEND=numpy):** a single `RFPhasorComposer` vs a
`ShardedPhasorStore.from_existing_composer(single, n_shards=16)` over a 38-fact battery (36 distinct agents,
multi-fact agents, a bound NEGATE, an attributed patient):

| check | result |
|---|---|
| facts re-homed (single vs sharded) | 38 vs 38 |
| forward recall `query_patient` (all stored cues) | 37 checked, 0 mismatch |
| `ask_yes_no` (incl. NEGATE) | 37 checked, 0 mismatch |
| `render_fact` | 37 checked, 0 mismatch |
| `query_chain` + `chain_of_thought` | 6 checked, 0 mismatch |
| no-confab moat (unknown cues abstain identically) | 11 checked, 11 abstain, 0 confab |
| **verdict** | **GO (0 mismatches)** |

**Server swap function — reproduced LOCALLY (real `webapp.server._maybe_shard_composer`, BRAIN_SHARDED_STORE=1),
`part_b_swap_fn` in the artifact:** on a populated agent the composer is swapped `RFPhasorComposer` →
`ShardedPhasorStore`, forward recall is byte-identical (37 checked, 0 mismatch — france→country, france→paris,
gold→element, dog→cat), the moat is identical (8 checked, 8 abstain — dragon→None), and the `.kb`/`.words` drop-in
surface works (kb 38, words 73). GO. This exercises the ACTUAL server function with NO full brain load (composers only).

**PART B — live wiring end-to-end (real tiny-demo brain), QUEUED:** the same script's `part_b_live_brain` builds a real
`BrainConversationalAgent` (tiny-demo, composer_kind=rf), applies the server swap, and asserts the AGENT's own recall
methods (`what_does`/`is_it_true`/`describe`) are byte-identical before vs after the swap + the moat holds. This loads a
brain, so per the one-brain-load safety rule it is QUEUED via `tools/gpu_queue.sh` (SIM_BACKEND=numpy), NOT run
directly (locally it is `SKIP_LIVE_BRAIN`, marked skipped in the artifact). The queue was PAUSED (owner) at submit
time; on resume the full run OVERWRITES the artifact with the live-brain result. The load-bearing byte-identical claim
(Part A + the server swap function) is already GO locally.

## Honest scope (brain-based-only)

- The router `hash(agent) mod S` is a DECLARED host scaffold (the finding above carries it as a disabled precondition;
  the faithful version is a learned/spiking cue→shard router — hippocampal indexing theory). The reads INSIDE each
  shard remain the genuine FHRR recall + the genuine no-confab moat.
- Shards default to the numpy RF fast-path (the de-risked, scale-capable store). The live default composer is the
  spiking `OneBrainComposer`, which is itself VALIDATED byte-identical to that RF numpy oracle, so routing a OneBrain
  source's facts through RF shards preserves the production answer while removing the O(K) scan wall. Sharding the
  exact production spiking class (`composer_factory=type(composer)`) is faithful but pays an S-fold build/VRAM cost
  (a future rung, infeasible S-wide today). The teaching path (`hear`) after a flag-ON swap re-parses with the agent's
  own parser (the sharded store carries no `hear`); the RECALL path — the load-bearing one — is byte-identical.
- **No production default is flipped.** `BRAIN_SHARDED_STORE` is default-OFF and byte-unchanged. Flipping it default-ON
  is handed back to the owner (the end-to-end Part B queue verdict should land first).

## What this unblocks / next rungs on #66

The live chat can now recall from an LLM-scale-capable store on one env flag, byte-identically and moat-safe. Next:
(1) land the Part B end-to-end queue verdict (on queue resume); (2) the learned/spiking cue→shard router (burn down
the host-hash scaffold); (3) grow the live store past the current `k_max=32` cap toward the 2413-fact Wikidata bundle
(commit 44a34b8c) and beyond, measuring routed latency + load balance at scale; (4) owner decision on default-ON.
