# Multi-bridge deep-knowledge design — recall+speed-preserving scaling beyond one stream-cortex bridge

**Date:** 2026-06-26
**Type:** DESIGN-INPUT (read-only on code; this doc is the deliverable). The controller trust-but-verifies the
load-bearing claims (flagged ⚠️VERIFY) and builds from this.
**Goal (owner #1):** DEEP knowledge — "discuss almost anything" — i.e. MANY more concepts than the working
1,454-concept first-chat brain, WITHOUT the recall + per-turn-speed cost that growing one bridge incurs.
**Context:** `project_deep_knowledge_brain_fluency_build.md`, `project_communicable_brain_not_rag.md`. A week of
training is fine; staging matters; the BRAIN does the wording; all local.

---

## 1. Diagnosis — WHY one bridge fails to grow, located precisely in code

### 1.1 The two confirmed crowding sources (both scale with the number of concepts on ONE bridge)

The first-chat brain is ONE `RFPhasorComposer` (`rf_phasor_composer.py`) holding `D=128` phasor codes for the whole
vocab, wrapped by the `DiscursiveTurn`/`CommunicableTurn`/`GenerativeReplayProposer`/`BrainConversationalAgent`
stack (`first_chat_console.py:190` builds the composer; `:205` the agent; `:209` the proposer; `:230` the
CommunicableTurn — **all four share the ONE composer instance `comp`**). Verified this session: 1,454 concepts →
recall 0.958, discuss-channel ≈4 adjacent facts, fast; 2,012 concepts → recall **0.875**, discuss thinned to **1**
adjacent fact, **~3× slower/turn**. The two mechanisms, located in code:

**(A) The composer CLEANUP codebook scans the full vocab.** Every `unbind` resolves a recovered phasor to a word by
a matched-filter argmax over **`self.words`** (the entire vocab):
- `_cleanup(self, rec_phases, words=None)` → `words = words if words is not None else self.words`;
  `sims = [mean cos over rec vs self.concepts[w] for w in words]` (`rf_phasor_composer.py:381-386`).
- batched: `_cleanup_all(self, rec, words=None)` builds `cb = stack(codebook over words)`,
  `sims = (rec_z @ conj(cb).T).real`, `argmax` (`:423-433`) — **O(K_facts × V × D)** per query.
- `_scan_first_match(**cue_roles)` (the who/what store scan, `:435-444`) calls `_cleanup_all` over **the full
  vocab** for EACH cue role across ALL stored composites.

So a who/what query cost ≈ `n_facts × n_cue_roles × V × D`. Doubling V (1454→2012) directly inflates this. More
importantly for QUALITY: a larger codebook = more near-neighbour candidates competing in the argmax, so a
phasor recovered at D=128 fidelity is more likely mis-cleaned to a wrong-but-close word → **recall 0.958 → 0.875**.
This is the documented FHRR cleanup behaviour — cleanup confusability rises with codebook density at fixed D
(CLAUDE.md "composer is a principled idealization"; the D=128 read is lossy, published 0.958 is who+what).

**(B) The proposer's contradiction gate resonates over the store, and that resonate cleans up over the full vocab.**
The discuss channel (the (N)/(D) adjacency) runs `CommunicableTurn.propose_candidates_about`
(`_communicable_turn_stageA_derisk.py:311`), which for each sampled candidate calls
`self.proposer._contradicts(a,ac,p)` (`:346`) → `composer.ask_yes_no(a,ac,p)`
(`_genfrontier_b2_generative_replay_derisk.py:201`) → a full store scan + cleanup. `_cand_cap` (`:353`,
console `:234` = 16) bounds the ACCEPTED count but each accepted/rejected candidate still pays a store-scan
resonate. As facts/vocab grow, each resonate is slower AND the candidate pool's PPMI neighbourhood is noisier, so
fewer survive the plausibility+non-contradiction filters → **discuss thins 4 → 1**.

**NOTE — the proposer's role pools are NOT the crowding source.** `GenerativeReplayProposer.agents/actions/patients`
are derived from the STORED facts only (`allf = stored_facts + negated_facts`, `:169-172`), i.e. ~tens of words for
24 facts — they do NOT scale with vocab. The crowd is the composer cleanup (A) and the store-scan resonate it
drives (B). ⚠️VERIFY: a quick profiler line on a 2,012 vs 1,454 query confirming the cleanup/`ask_yes_no` time
dominates (expected: cleanup matvec + the 208-step resonate are >90%, the latency-arc profile already says so).

### 1.2 A SECOND, independent problem hiding inside "2,012": the extra 558 words are the bad ones

This must be stated honestly or the design will chase the wrong fix. The trainer
`_curriculum_step1_320_real_corpus.py` derives its vocab from the g20 taxonomy by corpus frequency under a
`--vocab-filter`:
- `content` (the DEFAULT, entity+verb domains) is **corpus-capped at 1,454** — exactly the working brain.
- `all` (the full 32-cluster taxonomy) is **2,012** — the working 1,454 PLUS 558 distributionally-FLAT
  adjective/function/emotion words.

The trainer's OWN docstring (`:40-46`, the vocab-filter fix) says those flat words "co-occur with everything →
near-uniform codes that **homogenize the entity codes too**." So `brainALL` (2,012) is partly worse because of (A)
codebook crowding AND partly because the 558 flat words **degrade the shared code space** — they are the words the
content filter was created to keep OUT of the target set (and IN as context hubs). ⚠️VERIFY (cheap): measure recall
on `brain1454` (content, 1454) vs a hypothetical "first 1,454 content words split across 2 bridges" — if 2×727
content concepts recalls ≥0.95 while `brainALL` 2,012 sits at 0.875, the crowding fix is validated separately from
the flat-word confound.

### 1.3 The HARD ceiling is the taxonomy, not the corpus (load-bearing for "discuss almost anything")

Measured this session: with `tinystories+simplewiki(+wikitext)`, `--vocab-filter content` tops out at **1,454**
and `all` at **2,012** — and adding wikitext does NOT raise it. The cause: the ceiling is the **g20 taxonomy word
lists themselves** — `CONTENT_G20_DOMAINS` contains **1,472** words; all 32 g20 domains contain **2,048**
(`g20_vocab_spec_2048.ALL_CLUSTERS_2048`). The trainer only admits words that are members of this hand-curated
taxonomy (`derive_curriculum_from_corpus`: `candidates = list(word2cat)`, `:296`; content adds
`_is_content_word`, `:301`). **So "many MORE concepts" is gated by the taxonomy, not the number of bridges.**
Multi-bridge gives recall+speed headroom to HOLD ~5,000+ concepts at quality — but to actually REACH a
discuss-almost-anything vocab we must ALSO grow the taxonomy (a corpus-mined word→category spec). This is a
separate work item the design flags as the real bottleneck for the owner's #1 goal; the multi-bridge mechanism is
necessary but not sufficient on its own.

---

## 2. The design — N stream-cortex bridges, per-bridge cleanup, a routing layer

### 2.1 The core decision: per-bridge cleanup with a routing layer on top (NOT one shared phasor space)

**Recommendation: strictly per-bridge cleanup + a host vocab→bridge routing map. Do NOT try to make N bridges
share ONE phasor space.** Answering KEY QUESTION (1):

- A shared projection across bridges is POSSIBLE in principle — each `StreamCortexBridge` builds its phasor codes
  via a fixed random complex projection `proj: (D, n_hub)` seeded by `seed*7+3` (`_curriculum_step1_320_real_corpus.py:406-407`).
  Two bridges with the SAME seed + SAME hub set would share `proj`, so codes would live in one phasor space.
- BUT it does NOT solve the problem and it REINTRODUCES it. Even with a shared projection, the cleanup argmax must
  range over whatever codebook you give it; the only thing that buys recall+speed back is **a SMALLER codebook per
  cleanup**. If all N bridges shared one space AND you cleaned up over the union, you are back to V=5,000 crowding.
  And cross-bridge code COMPARABILITY is not needed for who/what recall (each fact's three roles live within ONE
  bridge's concepts — see §2.5 on cross-bridge facts).
- Decisive: the composer cleanup ALREADY accepts a codebook subset — `_cleanup(..., words=None)`,
  `unbind(..., words=None)`, `_cleanup_all(..., words=None)` (`rf_phasor_composer.py:381,388,423`). The
  per-bridge design needs NO `sim/` edit and NO composer edit to restrict the codebook; it just constructs one
  composer per bridge over that bridge's ~1,000-word vocab. **This is the cheapest correct mechanism.**

So: **each bridge is a self-contained `RFPhasorComposer` over ~1,000 concepts** (the size where recall ≥0.95 and
per-turn stays fast — to be pinned by §4 Stage 1). Cleanup is per-bridge by construction. A thin router decides
WHICH composer answers a query.

### 2.2 Data structures

```
ConceptBridge:                       # one stream-cortex shard, mirrors first_chat_console.build_brain_on_codes
    name           : str             # e.g. "shard0_animals_food", "shard1_objects_verbs"
    vocab          : list[str]       # this shard's ~1,000 concepts (DISJOINT across shards)
    grounded       : {word: phases[D]}   # this shard's stream-LEARNED codes (its own .npz)
    cat_ids/names  : per-shard category labels
    comp           : RFPhasorComposer(D=128, vocab=this shard's vocab, grounded_codes=this shard's codes)
    # NOTE: comp.kb (the stored SVO facts) is per-shard; a fact lives on the shard that owns its agent.

MultiBridgeBrain:
    shards         : list[ConceptBridge]
    word2shard     : {word: shard_index}     # the routing map (host bookkeeping; see §2.3)
    P, row         : the PPMI association graph over the UNION vocab (one graph; built once from the corpus)
    # one shared PPMI graph is fine + desirable: the (N)/(D) discuss adjacency is cross-shard relatedness, and
    # PPMI is O(V^2) sparse but built ONCE at load, not per-turn — it does not crowd cleanup.
```

The PPMI graph (`build_real_cooccurrence` + `build_plausibility`, `first_chat_console.py:169-172`) is built over
the WHOLE union vocab, exactly as today — it is the relatedness substrate the proposer/discuss use, and it is
read-only at chat time, so it is not a crowding source. Only the CLEANUP is sharded.

### 2.3 Cross-bridge ROUTING — how a query finds the bridge that owns a word

Answering KEY QUESTION (2): **a host `word2shard` dict is the right router, and it is legitimate** — it is
environment/bookkeeping (which neurons hold which concept), NOT cognition (the brain still does
comprehend/recall/abstain/discuss in spikes). This is exactly the precedent set by the validated
`g20_multibridge.py`: `find_member_for_word` / `find_member_for_pair` (`:256-272`) are plain host dict lookups over
`m.vocab_set`, and CLAUDE.md ships that as the "160/320-concept sparse ensemble." The parser front-end and the
composer recall/abstain remain neural; routing is the index, not the thought.

Routing rules (mirroring `g20_multibridge.dispatch`, `:459`):
- **who/what query `(agent, action)`** → the shard that owns `agent` (the fact lives where its agent's code lives,
  because `store(agent, action, patient)` binds all three into ONE composite on the agent's shard — see §2.5).
  `shard = word2shard[agent]`; if `agent` not in any shard vocab → **abstain** (unknown word, the moat).
- **`who (action, patient)` query** → query the shard that owns `patient` first; if it abstains, this is a
  genuine cross-shard case (agent on shard A, patient named in the query) — see §2.5 for the bounded handling.
- **discuss/topic `X`** → `shard = word2shard[X]` for the topic's own facts; the (N)/(D) adjacency uses the SHARED
  PPMI graph (cross-shard relatedness) but proposes candidates whose words resolve via `word2shard` per candidate.
- **unknown word** (not in any `word2shard`) → the graceful non-fabrication the console already emits
  (`first_chat_console.py:406-410`).

A NEURAL router is possible but unnecessary now (and is a later brain-based-purity item, not a #1-goal blocker):
a small spiking gate region whose input is the cue word's code and whose output selects a shard — but the
who/what answer + abstention are ALREADY neural inside each shard's composer, so the host index does no cognition
the brain should be doing. Defer it; flag it in §6.

### 2.4 The per-bridge cleanup, concretely (no `sim/` edit, no composer edit)

Each shard's `comp = RFPhasorComposer(D=128, vocab=shard.vocab, grounded_codes=shard.grounded)`. Because
`comp.words = sorted(shard.vocab)` (~1,000), EVERY cleanup inside that composer (`_cleanup`, `unbind`,
`_cleanup_all`, `_scan_first_match`) ranges over ~1,000 candidates, NOT the union. Recall fidelity and per-query
matvec cost are both restored to the single-1,000-bridge regime — the regime measured GO. No code change to the
composer is required: this falls out of constructing N composers each over its own vocab.

### 2.5 Cross-bridge FACTS — where a fact lives, and the one honest limit

A stored fact `store(a, v, p)` binds agent+action+patient into ONE composite phasor (`comp._encode`,
`rf_phasor_composer.py:261`) held on ONE composer. The clean rule: **a fact lives on the shard that owns its
AGENT.** Recall `query_patient(a, v)` routes to `word2shard[a]` and the patient is decoded by that shard's cleanup
— which therefore needs `p`'s code in its codebook. Two cases:

1. **Same-shard fact** (a, v, p all on shard A): trivial — store + recall entirely within A's composer. The SVO
   fact-generator should PREFER these (draw agent, action, patient from the SAME shard's vocab). The corpus-mined
   facts (`_corpus_svo_extract`) can be partitioned per-shard so the bulk of facts are same-shard.
2. **Cross-shard fact** (agent on A, patient on B): the composite is on A, but A's cleanup codebook does not
   contain `p`'s code → A cannot decode the patient. TWO clean options, in order of preference:
   - **(2a) co-store the patient's code into A's codebook for that fact.** A composer's codebook is just its
     `concepts` dict; A can hold `p`'s grounded code as an extra cleanup entry (a small per-shard codebook
     extension for the handful of cross-shard patients A's facts reference). This keeps recall fully neural +
     in-shard. The codebook grows only by the number of distinct cross-shard patients referenced, not by the full
     union — crowding stays bounded. **Recommended.**
   - **(2b) store cross-shard facts on a dedicated small "bridge fact" composer** whose vocab is the union of the
     concepts those facts touch (like `g20_multibridge`'s cross-bridge engram tags). Simpler but a second codebook
     to route to.

The HONEST LIMIT (flag for the owner): the SVO recall mechanism is cleanest when facts are agent-anchored to one
shard. A discuss-almost-anything brain that freely relates concepts ACROSS shards in single facts will lean on
(2a)'s per-shard codebook extension; if that extension itself grows large for a hub shard, crowding partially
returns FOR THAT SHARD. Mitigation: shard by SEMANTIC domain (animals, food, objects, verbs, …) so most facts are
within-domain (within-shard) — the g20 32-cluster structure is already a domain partition (§2.6).

### 2.6 Sharding policy — by semantic domain (reuse the g20 cluster structure)

Shard the ~1,472 content concepts (and, later, a grown taxonomy) by g20 DOMAIN, not by frequency or arbitrarily.
`g20_vocab_spec_2048.ALL_CLUSTERS_2048` already groups concepts into 32 co-occurrence-coherent clusters (64 words
each). Grouping clusters into ~1,000-word shards (e.g. shard0 = animals+food+body+people+nature; shard1 =
objects+vehicles+buildings+clothing+tools; shard2 = the verb domains + …) means:
- most SVO facts are within-domain → same-shard (§2.5 case 1);
- each shard's stream cortex learns a clean within-domain code space (it ALSO improves generalization — the
  domain is the gen reference);
- routing is by `word2shard[word]` = which domain-shard owns the cluster.

This directly reuses the validated `_curriculum_step1_320_real_corpus` trainer with a `--vocab-filter` /
domain-subset selection per shard.

### 2.7 Console + DiscursiveTurn integration

Answering KEY QUESTION (3): **wrap N composers behind a thin multi-bridge facade that presents the SAME composer
API the DiscursiveTurn already consumes — do NOT add an N-bridge mode inside `OneBrainComposer` or rewrite the
DiscursiveTurn.** The whole DiscursiveTurn/CommunicableTurn/proposer/agent stack talks to ONE object via a small
surface: `store`, `query_patient`, `query_agent`, `ask_yes_no`, `unbind`, `elaborate`, `kb`, `concepts`, `words`
(see `first_chat_console.py:190-260`; `CommunicableTurn.__init__` takes `comp`; the proposer takes `composer`; the
agent takes `composer`). Build a `RoutedComposer` that implements that surface by dispatching to the right shard:

```
class RoutedComposer:                    # the SAME API the DiscursiveTurn/proposer/agent expect
    def __init__(self, shards, word2shard): ...
    def store(self, a, v, p, polarity=None):
        self.shards[self.word2shard[a]].comp.store(a, v, p, polarity)   # agent-anchored (+ §2.5 codebook ext)
    def query_patient(self, a, v, order_fn=None):
        s = self.word2shard.get(a)
        return None if s is None else self.shards[s].comp.query_patient(a, v, order_fn)   # abstain if unknown agent
    def query_agent(self, v, p): ...      # route by patient's shard (+ the §2.5 cross-shard fallback)
    def ask_yes_no(self, a, v, p): ...    # route by agent; abstain (return "unknown") if agent unknown
    def elaborate(self, topic): ...       # route by topic's shard; the assoc graph is per-shard kb
    @property
    def concepts(self): return self._union_concepts   # a read-only union view (for context_code/_phase_cos)
    @property
    def words(self): return self._union_words          # union for callers that introspect vocab; cleanup never uses this
    kb = property(lambda self: [f for s in self.shards for f in s.comp.kb])   # union of stored facts (for _assoc_graph)
```

Then `build_brain_on_codes` (the console builder) loads N `.npz` files, builds N `ConceptBridge`s, builds
`word2shard`, builds `RoutedComposer`, and passes it to `BrainConversationalAgent(composer=routed)`,
`GenerativeReplayProposer(routed, …)`, `CommunicableTurn(routed, …)` UNCHANGED. The router is invisible to the
DiscursiveTurn — it sees one composer with the moat intact (each shard's `query_*` returns `None`/`"unknown"` on a
miss; `word2shard.get` returns `None` for an unknown word → abstain). The `_load_real_facts`/`_make_svo_facts`
fact-generator partitions facts per shard (prefer same-shard, §2.5).

The moat audit (`first_chat_console.audit_moat`, `:439`) is unchanged: a CERTAIN proposition must be in the
union `kb`; a FLAGGED proposition must be hedged + not in `kb` + a `what_does` on it must abstain — all read
through the `RoutedComposer` surface.

### 2.8 What is reused verbatim vs new

- **Reused verbatim (NO edit):** `RFPhasorComposer` (its `words=`-subset cleanup is the whole mechanism);
  `_curriculum_step1_320_real_corpus` (the per-shard trainer, with a domain/vocab subset arg);
  `DiscursiveTurn`/`CommunicableTurn`/`GenerativeReplayProposer`/`BrainConversationalAgent` (they consume the
  composer surface); `build_real_cooccurrence`/`build_plausibility` (the shared PPMI graph).
- **New (small, all host/runner — NO `sim/` edit):** `RoutedComposer` (the dispatch facade, ~120 lines);
  `build_brain_on_codes` gains an N-`.npz` path + `word2shard` construction; the per-shard fact partition; the
  g20-domain → shard-vocab grouping helper. The `g20_multibridge.py` routing/dispatch logic is the proven
  template to copy from (`find_member_for_word`, the per-bridge load loop, the dispatch routing).

---

## 3. The mechanism it most resembles (precedent)

`g20_multibridge.py --sparse` is the VALIDATED precedent (CLAUDE.md "160/320-concept sparse-distributed ensemble
SHIPPED"): N bridges each holding a 32–64-concept disjoint vocab, a host `word2bridge` router
(`find_member_for_word`/`_pair`), per-bridge recall (`recall_rates`), cross-bridge association via tag-name search
across bridges (`query_concept`, `:394`). **What transfers:** the architecture (disjoint per-bridge vocab + host
routing + per-bridge recall + cross-bridge facts) transfers DIRECTLY. **What does NOT transfer:** the
representation. `g20_multibridge` uses **sparse-distributed (Kanerva-SDM) engram codes + firing-rate readout +
tag-name multitag** recall (`stim_recall_sparse_rates`, `encode_pair_engram_sparse`); the first-chat brain uses
**stream-cortex PHASOR codes + the FHRR composer's unbind/cleanup** recall. So we take g20's ROUTING SKELETON but
keep the phasor composer as the per-bridge recall engine — the `RoutedComposer` above is exactly that
substitution. The two are API-compatible at the routing layer (both "find the bridge for the word, ask it,
aggregate"), which is why the integration is a thin facade, not a rewrite.

---

## 4. Staged cheap-first de-risk plan (each stage independently checkable)

Order is cheapest→most-expensive; each stage gates the next. Reuse existing runners where possible.

### Stage 0 — DIAGNOSTIC, ~minutes, no training (separate the two confounds of §1.2)
Confirm the crowding is the codebook, not just the flat words. On the EXISTING artifacts:
build a `RFPhasorComposer(vocab=brain1454.vocab, grounded=brain1454.grounded)` and one over
`brainALL`(2012); store the SAME 24 facts (drawn from the 1,454 overlap); measure who/what recall + time/query.
**Expected / GO:** 1,454 ≈0.958 fast; 2,012 ≈0.875 slow — reproduces the session result, locating it in the
composer cleanup, not the discuss stack. Then the decisive split: take the brain1454 codes, SPLIT the 1,454 words
into two disjoint ~727 shards, build two composers, route the 24 facts by agent-shard, measure recall+time. **GO
bar:** per-shard recall ≥0.95 AND aggregate time/query ≤ the single-1,454 time (cleanup is now over ~727, so it
should be FASTER). This proves per-bridge cleanup preserves recall+speed using ZERO new training and ZERO new
codes — the cheapest possible proof of the core mechanism. ⚠️This is the load-bearing de-risk.

### Stage 1 — pin the per-bridge sweet-spot size, ~1 short train, reuse `_curriculum_step1_320_real_corpus`
We already KNOW 1,454 content concepts on one bridge gives 0.958. Find the largest per-shard V that holds
recall ≥0.95 at D=128 and stays fast: run the existing trainer at V ∈ {1000, 1200, 1454} (it already sweeps via
`--n-concepts`; `--save-codes`), reading off `recall` + per-query time from its `measure_recall_and_moat`.
**GO bar:** identify V* (likely ~1,000–1,200 for margin) where recall ≥0.95 and time/query is in the fast band.
This is the shard size for the build. (We have the 1,454 data point already; this just confirms the knee.)

### Stage 2 — TWO real shards end-to-end, ~2 trains (≈ a few hours each at GPU scale, local)
Partition the g20 content domains into TWO disjoint ~V* shards (§2.6). Train each shard with the existing trainer
(`--save-codes shard0.npz`, `shard1.npz`) on `tinystories+simplewiki`. Build `RoutedComposer` over both, build
`word2shard`, partition the corpus-extracted facts per shard (prefer same-shard). **GO bars (the heart of the
design, mirrors the single-bridge bars):**
- per-shard who/what recall ≥0.95 (each shard's `measure_recall_and_moat`);
- aggregate moat 0 false-accepts over BOTH shards + cross-shard absent cues (the moat is per-shard + the
  `word2shard.get → None` abstain) — **HARD STOP if any false-accept**;
- ~2×V* ≈ 2,000 distinct concepts at recall ≥0.95 (vs the single-bridge 2,012's 0.875) — the headline number;
- per-turn time ≤ the single-1,454 turn (each query cleans up over ONE shard).

### Stage 3 — the DiscursiveTurn console on the 2-shard brain, ~minutes (no training)
Run `first_chat_console.py --demo`/`--rubric` against the `RoutedComposer`. **GO bar:** the 10-prompt rubric
≥8/10, moat 0 leaks, discuss channel restored to ≥3 adjacent facts (vs the thinned 1 at 2,012-on-one-bridge),
mixed-type span. This proves the full chat experience is preserved at 2× concepts.

### Stage 4 — scale to N shards (≈ a week of staged training, local, cumulative)
Once 2 shards are GO, add shards one at a time (the trainer + `--save-codes` is the cumulative staging the owner
asked for — `project_deep_knowledge_brain_fluency_build`: "STAGE so a week of training is cumulative"). Each new
shard is an independent train + an entry in `word2shard`; no retrain of existing shards. **GO bar at each add:**
the new shard's recall ≥0.95, the moat stays 0-FA, per-turn time stays flat (it scales with the ROUTED shard's V,
not the union — the whole point). Concurrently (separate work item, §6): grow the taxonomy so N shards can hold
NEW concepts beyond the 2,048 g20 ceiling.

---

## 5. Anti-cheat controls (the load-bearing invariants)

1. **Moat = 0 false-accepts, preserved and tested cross-shard.** Each shard's composer already abstains on a miss
   (`query_patient → None`); the router abstains on an unknown word (`word2shard.get → None`). The test MUST
   include (a) never-stored cues whose agent IS in a shard (the composer must abstain), AND (b) cues whose agent
   is in NO shard (the router must abstain), AND (c) a cross-shard cue `(agent∈A, patient∈B)` that was never
   stored (must abstain, not spuriously match via the §2.5 codebook extension). A single confident answer on any
   absent cue is a HARD STOP. Reuse `measure_recall_and_moat`'s absent-cue construction
   (`_make_svo_facts → absent_what/absent_who`) per shard + a new cross-shard absent set.
2. **Recall ≥0.95 PER BRIDGE** (not just aggregate) — report each shard's recall separately so a weak shard
   cannot hide behind strong ones (the per-seed-not-pooled discipline, `feedback_6seed_validation`).
3. **Routing-correctness control.** Assert `word2shard` is a partition: every concept maps to exactly one shard
   (disjoint vocabs), and every stored fact's agent's shard actually CONTAINS the fact's composite. A
   PERMUTED-routing control (route each query to a WRONG shard) MUST collapse recall to ~chance and MUST NOT raise
   false-accepts above 0 — proving the router is load-bearing and the moat is not routing-dependent (a wrongly
   routed query should abstain, never confabulate). This is the analogue of the permuted-mapping controls used
   throughout the project (CLAUDE.md "permuted-label control").
4. **Flat-word confound isolation** (§1.2): the comparison must be content-vs-content. Compare 2×V* CONTENT shards
   to the single-bridge CONTENT 1,454 (0.958), NOT to `brainALL` 2,012 (which is confounded by the 558 flat
   words). The headline "2,000 concepts at ≥0.95" must use content concepts on both sides.
5. **No `sim/` edit** — the whole mechanism is composer-codebook-subset (already supported) + a host router. If a
   `sim/` edit is ever proposed, it needs the byte-level review the owner requires (`feedback_brain_based_only_standard`).
6. **Frozen-brain control carries over** (per shard): the trainer's plasticity-OFF control
   (`_curriculum_step1_320_real_corpus` frozen-brain) must still show competence does not rise without learning —
   each shard's codes are LEARNED, not smuggled.

---

## 6. Honest open risks

1. **The taxonomy ceiling is the real #1-goal bottleneck (§1.3).** Multi-bridge buys recall+speed headroom to
   HOLD many concepts, but the trainer only admits g20-taxonomy words (≤1,472 content / 2,048 total). "Discuss
   almost anything" needs a much larger word→category spec mined from the corpus (the corpus has far more unique
   types — `simplewiki` alone is 143 MB). Growing the taxonomy is a SEPARATE work item, arguably the higher-leverage
   one for deep knowledge; multi-bridge is necessary but not sufficient. Recommend scoping it in parallel.
2. **Cross-shard facts (§2.5) are the soft spot.** Agent-anchoring keeps recall in-shard, but a brain that freely
   relates concepts across domains in single facts leans on per-shard codebook extension; a hub shard could
   re-crowd. Domain-sharding mitigates (most facts within-domain) but a "general" shard (verbs, abstract relations)
   may attract many cross-shard references. ⚠️Measure the codebook-extension size per shard at Stage 2; if a shard's
   extension approaches its native vocab, re-shard or use the dedicated bridge-fact composer (2b).
3. **The discuss (N)/(D) channel spans shards via the shared PPMI graph, but the proposer's `_contradicts`
   resonates on a SHARD composer.** A candidate whose words live on different shards needs its non-contradiction
   check routed correctly (check on the agent's shard). This is straightforward but must be wired in the
   `RoutedComposer.ask_yes_no` — verify the discuss channel still surfaces cross-domain adjacency (e.g. "dog" on
   the animal shard relating to "ball" on the object shard) at Stage 3.
4. **D=128 fidelity is the underlying lossy constraint.** Per-bridge cleanup restores recall by SHRINKING the
   codebook at FIXED D; it does not raise D. If a per-shard V* still shows recall <0.95 at D=128, the lever is a
   smaller shard (more bridges), not more concepts/bridge. The sweet-spot V* (Stage 1) is the real knob.
5. **Latency of N composers at chat time.** Each turn touches only the ROUTED shard's composer (good), but the
   discuss channel + multi-hop may touch several shards' composers in one turn. Per-turn time should still be flat
   in N (a turn routes to O(1) shards for recall; the discuss candidate set is PPMI-bounded), but ⚠️measure at
   Stage 3 that an N=5 brain's turn is not N× a single shard's turn.
6. **A neural router is deferred** (§2.3). It is brain-based-purity debt (the host index is bookkeeping, defensible
   by the §2.3 precedent + `feedback_brain_based_only_standard`'s environment/body boundary), but a future
   spiking shard-selector is the fully-on-substrate form. Not a #1-goal blocker; flag it on the shortcut inventory.

---

## 7. Summary for the controller

- **Diagnosis:** growing one bridge fails because the FHRR composer's CLEANUP argmax ranges over the FULL vocab
  (`rf_phasor_composer.py:381-444`) — bigger codebook = more cleanup confusion (recall 0.958→0.875) + more matvec
  (3× slower) + a thinner discuss channel (the proposer's `ask_yes_no` store-scan resonates more, surfaces fewer).
  The proposer's role pools do NOT scale (fact-scoped). A SECOND confound: the 2,012 vocab includes 558 flat words
  the trainer deliberately excludes from `content` (they homogenize codes). And the hard vocab ceiling is the g20
  TAXONOMY (1,472 content / 2,048 total), not the corpus.
- **Design:** N stream-cortex shards, each a `RFPhasorComposer` over ~V*≈1,000 DISJOINT domain-grouped concepts
  (per-bridge cleanup falls out of `comp.words` being the shard vocab — the composer ALREADY takes `words=`
  subsets, NO `sim/` or composer edit). A host `word2shard` router (legitimate bookkeeping, the proven
  `g20_multibridge` pattern) picks the shard; facts are agent-anchored; cross-shard facts use a bounded per-shard
  codebook extension. A thin `RoutedComposer` facade presents the existing composer API so the
  DiscursiveTurn/proposer/agent stack is reused UNCHANGED.
- **Cheapest de-risk (Stage 0):** SPLIT the existing brain1454 codes into 2×727 shards, route the same 24 facts by
  agent-shard, show per-shard recall ≥0.95 at ≤ the single-bridge time — ZERO new training, proves the core
  mechanism. Then 2 real domain shards (Stage 2) for ~2,000 concepts at ≥0.95 vs the single-bridge 2,012's 0.875.
- **Anti-cheat:** moat 0-FA tested cross-shard (incl. unknown-agent + cross-shard-absent), recall per-bridge,
  permuted-routing control collapses recall without raising false-accepts, content-vs-content comparison,
  no `sim/` edit.
- **Biggest honest risk:** reaching a discuss-almost-anything vocab ALSO needs a grown taxonomy — multi-bridge
  removes the recall+speed wall but not the 2,048-word taxonomy ceiling.
```
