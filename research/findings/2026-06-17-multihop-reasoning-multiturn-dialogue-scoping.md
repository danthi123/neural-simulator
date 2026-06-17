# Multi-hop reasoning + multi-turn dialogue — deep-research scoping (2026-06-17)

**Status:** READ-ONLY scoping doc. Gates whether/how we build the next pre-registered
conversational frontier (multi-hop relational inference + multi-turn discourse). No code
written, no heavy experiments run. Follows the standing "deep research + catalog review
FIRST at new directions" workflow.

**One-line bottom line:** the *clean* multi-hop path (separate-fact storage + **iterated
single-hop unbind**, where the unbound filler becomes the next hop's cue) is reachable on
the current spiking substrate for **2–3 hops** before the unbind/cleanup SNR floor bites;
the *naive* path (multitag spreading / superposed associations) is the one that produced the
retracted result and **must be excluded by anti-cheat**, not measured. Multi-turn dialogue
needs a real **discourse-state working-memory buffer carried across turns** — the substrate
machinery (NMDA loop-attractor WM) exists but is currently invoked single-shot per call.

---

## Diagnosis

### What "multi-hop reasoning" actually requires, computationally

A transitive/relational query — *"what does the thing the dog chases eat?"* over the facts
`dog chase cat` and `cat eat fish` — is a **two-step pointer-chase**:

1. **Hop 1:** find the fact whose agent = `dog`, unbind its `patient` role → `cat`.
2. **Hop 2:** use `cat` as the new cue: find the fact whose agent = `cat`, unbind its
   `patient` (or `action`) role → `fish`.

So multi-hop reasoning is **iterated single-hop retrieval where each hop's *output*
becomes the next hop's *cue***. The hard part is not the algebra of one hop (the project
already does that, validated multi-seed — `compose_relational_memory_demo.py`,
`rf_phasor_composer.query_*`). The hard part is that **error compounds across hops**, and
there are two qualitatively different ways to implement the chain, with very different
failure modes:

- **(i) Pointer-chase over SEPARATELY-stored facts (clean).** Each fact is its own bound
  composite vector `agent⊗dog ⊕ action⊗chase ⊕ patient⊗cat`, kept in its own slot
  (the project's KB is a Python list of separate composites — `RFPhasorComposer.kb`). A hop
  is: unbind one role from the matching composite, **clean up to the nearest discrete
  concept** (snap back to a noise-free codebook vector), then re-cue. Because cleanup
  *re-discretizes* between hops, the SNR does **not** integrate multiplicatively — each hop
  pays one unbind's worth of noise, and a *successful* cleanup resets the signal to a clean
  codebook vector before the next hop. The chain dies only when a single hop's unbind SNR
  drops below the cleanup margin.
- **(ii) Spreading activation over a SUPERPOSED association memory (leaky).** All
  pairwise associations live in one recurrent weight matrix (or one multitag superposition);
  cueing `dog` lets activation spread `dog → cat → fish` along learned edges. This is what
  `compose_concept_chain_test.py` did. It **conflates hop-1 and hop-2** (a 2nd-degree
  neighbour just shows up with a weaker score than a 1st-degree neighbour), it has **no role
  structure** (it answers "what is associated with dog?", not "what does the thing dog
  chases eat?"), and it is the mechanism that produced the retracted 90%.

### Why the substrate's current single-hop bind/unbind doesn't already give multi-hop for free

Two distinct walls:

**(a) The unbind/cleanup SNR wall (the per-hop noise floor).** In a vector-symbolic
("VSA"/HRR) code, unbinding is an *approximate* inverse: `unbind(bind(r,f) ⊕ other, r) =
f + crosstalk`, where the crosstalk grows with the number of bundled role-filler pairs and
shrinks with dimension `D`. HRR/FHRR memory degrades roughly **linearly in the number of
bound items** unless `D` is raised or redundancy added (Plate 1995; Frady-Sommer 2019; and
the 2024 factorizer-noise analysis, Kymn et al., *On the Role of Noise in Factorizers*,
arXiv:2412.00354). The project's production composer runs at **D = 128** (the
`brain_conversational_agent` default; `consolidated_320_conversation_demo.py` line 62), with
facts stored **separately** precisely because superposing many facts into one vector "degrades
— the multi-hop wall" (the `compose_relational_memory_demo.py` docstring states this
verbatim). Each hop re-incurs that one-fact unbind noise; the question the de-risk must
answer empirically is **how many clean re-discretizations you get before a hop misses**.

**(b) The discourse-state-across-turns gap.** Single-hop Q&A is stateless: each
`what_does`/`who_does`/`is_it_true` call resolves against the whole KB and returns. There is
no carried context — *"what about the cat?"* (an anaphoric follow-up) has nothing to resolve
"the cat" against, and a 2-hop query's *intermediate result* (`cat`) is not held anywhere to
feed hop 2. The substrate **has** the right machinery for this (an NMDA loop-attractor
working-memory buffer — `content_selection_spiking.SpikingLoopContextBuffer`, validated to
hold a fading multi-concept set across updates), but the conversational agent invokes the
spreading controller **single-shot per `elaborate()` call** with a fresh WM reset each time
(`SpikingSpreadingController.turn_latency` calls `_reset_wm()` every probe). Multi-turn
coherence requires *not* resetting between turns and binding the discourse referents into
that buffer.

### Honest reconciliation with the retracted transitive-inference result

The 2026-05-14 retraction (`2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`)
is load-bearing context, and it cuts two ways:

- The **"90% transitive inference"** claim was a **measurement artifact** — a module-level
  monkey-patch built the eval bridge with a *mismatched architecture* (28 pools loaded with
  16-pool weights), producing pseudo-random firing that scored 90% across seeds in the same
  direction. Corrected, the same chain test scored **1/4 (25%) on seed 42** — i.e. chance.
- **Separately and more subtly**, the follow-up forensic
  (`2026-05-14-multitag-spurious-are-2nd-degree.md`) found that *even with correct
  architecture*, the **multitag spreading mechanism** DOES leak 2nd-degree neighbours into
  the top-N (e.g. `cat → cold → river` surfaces `river` for cue `cat`). The doc optimistically
  reads this as "transitive inference may be real after all." **This scoping doc explicitly
  rejects that read as a future trap:** soft spreading that surfaces a 2nd-degree neighbour is
  mechanism (ii) — it is *exactly* what a non-reasoning lookup-with-smearing baseline also
  does, it has no role structure, and it cannot answer a relational query whose two hops use
  different roles. A result that this baseline also passes is **not** a result. Mechanism (ii)
  is therefore a **memorization-floor control**, not a candidate.

So the genuine, honest claim to test is narrow and specific: **does the role-structured
pointer-chase over separately-stored facts (mechanism (i)) answer a held-out relational query
that a same-data spreading baseline CANNOT** — and how many hops does it survive.

---

## Ranked, biologically-grounded options

### Option 1 (RECOMMENDED) — Iterated single-hop unbind + attractor cleanup over the existing FHRR composer KB (pointer-chase)

- **Biology:** Eichenbaum–Cohen **relational memory / "memory space"** — catalog **D.02**
  ("networks via overlapping events allowing flexible inference (e.g., transitive)"); Kandel
  6e Ch 52 pp 1301–1302; Dusek & Eichenbaum 1997; Bunsey & Eichenbaum 1996. The cognitive-science
  consensus (Eichenbaum) is that the hippocampus stores **discrete events with the relations
  among them** and supports inference by **traversing the relational network at retrieval**,
  *not* by pre-computing a value scale — lesioned animals **learn the premise pairs but fail
  the inference** (the search-confirmed Dusek/Eichenbaum result). "Traverse the network at
  retrieval" = iterate single-hop retrievals. The **between-hop cleanup** is the CA3/cortical
  attractor snapping a noisy recall back to a stored pattern (catalog **D.05** CA3 autoassociator,
  Marr 1971; Kandel Ch 54 pp 1342, 1360–1361) — the project's validated NEF/attractor cleanup.
- **Mechanism:** add a `query_chain(cue, [role1, role2, …])` loop to the composer: `x ← cue`;
  for each role in the chain, `x ← cleanup(unbind(matching_fact(x), role))`; return `x`. The
  per-hop primitives **already exist and are validated**: `RFPhasorComposer._unbind_phases`
  (conj diagonal complex synapse on resonate-and-fire neurons) + `_cleanup` (spiking NEF
  cleanup when `enable_spiking_cleanup=True`, else numpy argmax) + the cue-match loop in
  `query_agent`/`query_patient`. The abstention moat composes naturally — if any hop's
  matching-fact lookup finds nothing, the chain returns `None` (no confabulated answer).
- **Spiking-native?** **Yes, end-to-end** at `enable_spiking_cleanup=True`: unbind is a complex
  synaptic matvec on RF neurons; cleanup is a spiking matched-filter + Izhikevich WTA. The only
  host code is the loop control (which fact-slot to probe next) — and that is legitimate
  "which assembly fired" routing, not cognition. **This is the lowest-new-code, highest-
  substrate-fidelity option, and it directly reuses the production composer.**

### Option 2 — CA3 heteroassociative pointer-chase (substrate-learned edges) via `LearnedAssocGraph`

- **Biology:** catalog **D.05** (CA3 recurrent autoassociator) + **D.02** (relational binding).
  The CA3 recurrent collateral net is a Marr/Treves–Rolls autoassociator; a *hetero*associative
  variant (`A → B` directed Hebbian edges) pattern-completes one hop, and iterating the
  completion is the multi-hop traversal. O&N (Ch 4.8, pp 222–230) emphasize the autoassociator
  is **theta-paced sequential** — successive theta cycles complete successive nodes along a
  path — which is the biological substrate for "chase the pointer one hop per cycle."
- **Mechanism:** `research/runners/learned_assoc_graph.py` already builds a sparse-coded pool
  with a **plastic excitatory recurrent** that *learns* concept→concept edges by Hebbian
  co-firing (`store_fact` co-fires a fact's concepts; `graph()` reads back the learned weights),
  validated to match the co-occurrence oracle multi-seed. To make it **multi-hop AND
  role-aware** (avoiding the mechanism-(ii) trap), store **directed, role-labelled** edges
  (separate recurrent sub-matrices per role, or edges tagged by which role-pair they came from)
  and drive the chain one completion at a time with a clean reset between hops.
- **Spiking-native?** **Yes** — it is a real `SimulationBridge` with Hebbian-learned recurrent
  weights; the completion is spiking pattern-completion. **Caveat:** as currently written its
  `graph()` is an *undirected co-occurrence* read (it does NOT preserve role/direction), so
  used naively it **degenerates into mechanism (ii)** and would fail anti-cheat. It needs the
  role-labelled-edge extension before it can answer a true relational chain. Higher build cost
  than Option 1, but it is the more biologically complete "relational network in CA3."

### Option 3 — NMDA loop-attractor working memory holds the chain's intermediate state + discourse state (the multi-turn half)

- **Biology:** persistent-activity working memory — catalog **G** cluster (Working memory / PFC,
  recurrent attractor dynamics); **Wang 2002** NMDA reverberatory attractor (the project's own
  dlPFC config, used for the navigation accumulator and the dialogue planner). Theta–gamma
  multiplexing (catalog **N.15**, Lisman & Idiart 1995) is the deeper biological account of a
  **~7±2-item** multiplexed buffer — the substrate for holding several discourse referents at
  once.
- **Mechanism:** carry one persistent `SpikingLoopContextBuffer` across turns (do **not** reset
  it between turns); bind the current referents (subject of the last sentence, the intermediate
  hop result, the open question) into it; resolve anaphora ("the cat", "it") by reading the
  held set. For multi-hop specifically, the **intermediate filler from hop 1 is written into the
  buffer and read as the cue for hop 2** — making the chain's working state genuinely neural
  rather than a Python variable.
- **Spiking-native?** **Yes** — `SpikingLoopContextBuffer` / `SpikingController` are validated
  spiking WM (6/6-seed multi-concept hold). This is **necessary for multi-turn dialogue** and
  **complementary to** (not a substitute for) Options 1/2 for the reasoning chain. It is the
  natural Phase-2 once single-shot multi-hop is de-risked.

### Option 4 — TEM-style factorised structural/relational code (the generalizing, high-ceiling, high-cost option)

- **Biology:** the **Tolman–Eichenbaum Machine** (Whittington, Muller, Mark, Chen, Barry,
  Burgess, Behrens, *Cell* 2020) — factorise a **structural** code (the relation graph,
  entorhinal grid-like) from a **sensory** code (the items, hippocampal place/landmark-like),
  bind them conjunctively, and **generalize the structure to new items** — the model that
  unifies spatial navigation and relational inference and explicitly does transitive inference
  by *path integration over an abstract relational graph*. A **spiking TEM** now exists
  (biorxiv **2025.10.16.682754**, *The Spiking Tolman–Eichenbaum Machine*) — STDP + theta-
  modulated input, grid + place + phase precession emerge in spikes — establishing this is
  realizable on a spiking substrate.
- **Mechanism:** learn a factorised relational code so that the *relation* `chase`/`eat` is a
  reusable transition operator, and multi-hop = composing transition operators over the abstract
  graph. This is the only option that buys **generalization of the relational schema to
  never-seen items** (true relational reasoning, not per-fact lookup).
- **Spiking-native?** In principle yes (spiking TEM exists), but this is a **months-scale
  research build** requiring a learned structural code — comparable in scope to the deferred
  dendritic-cortex rewrite. **Not a cheap-first candidate; the strategic end-state**, flagged for
  the roadmap if Options 1–3 hit a generalization ceiling.

### Option 5 — Extend the dlPFC spreading-activation Control to depth > 1 (REJECTED as a primary mechanism; useful only as the baseline)

- **Biology:** spreading activation (Collins & Loftus 1975) over an association graph — the
  project's `SpikingSpreadingController`.
- **Why rejected:** this is **mechanism (ii)**. Its own code documents the fatal honest
  boundary: latency coding "naturally ranks DIRECT associates before INDIRECT ones, and an
  indirect concept (reached via a direct one) **can never out-race its own upstream**"
  (`content_selection_spiking.py`, `relevance_by_latency` docstring), and the *rate* read
  "over-spreads multi-hop and **loses topic focus**." It has no role structure, so it cannot
  answer a relational query that uses different roles on each hop, and it is exactly the
  baseline that the retracted result rode. **Keep it as the memorization-floor control, not as
  the reasoning engine.**

---

## What existing project machinery is reusable

| Need | Reuse | File / API |
|---|---|---|
| Single-hop bind/unbind on spikes | `RFPhasorComposer._bind / _unbind_phases / _cleanup` (RF complex synapses + spiking NEF cleanup) | `research/runners/rf_phasor_composer.py` |
| Cue-match retrieval loop + **abstention moat** | `query_agent / query_patient / ask_yes_no` (loop over separate `kb` composites, return `None` on no match) | `rf_phasor_composer.py` |
| Production agent surface to extend | `BrainConversationalAgent.what_does / who_does / is_it_true / describe / elaborate` | `research/runners/brain_conversational_agent.py` |
| Separate-fact KB (correct, non-superposed) | `composer.kb` (list of per-fact composites); `compose_relational_memory_demo.py` cue-based unbind | `rf_phasor_composer.py`, `compose_relational_memory_demo.py` |
| Substrate-learned concept→concept edges (CA3 autoassoc) | `LearnedAssocGraph.store_fact / graph` (Hebbian recurrent) — **needs role-labelled-edge extension** | `research/runners/learned_assoc_graph.py`, `_D_sparse_heteroassoc.py` |
| Spiking discourse-state WM across turns | `SpikingLoopContextBuffer`, `SpikingController`, `SpikingSpreadingController.turn_latency` | `research/runners/content_selection_spiking.py` |
| Engram tag / stimulate (alt. fact-binding) | `bridge.start_engram_recording / commit_engram_tag / stimulate_tag` | `sim/bridge.py:3154–3320` |
| Production grounded codes at scale | `consolidated_320_conversation_demo.py` (D=128, 320-concept stream-learned codes) | `research/runners/consolidated_320_conversation_demo.py` |
| The retracted leaky mechanism (now a CONTROL) | `compose_concept_chain_test.py` (spreading), `multitag_transitive_eval.py` (2nd-degree leak) | both under `research/runners/` |

**Net:** Option 1's reasoning loop is **~30 lines on top of the already-validated composer**
— a `query_chain` method plus a falsification harness. No `sim/` edit anticipated for the
cheap-first probe.

---

## Recommended cheap-first de-risk (the decisive falsification probe, BEFORE any build)

**Run on CPU/numpy** (`SIM_BACKEND=numpy`), using the existing `RFPhasorComposer` numpy fast
path (the spiking-cleanup parity is already established multi-seed; numpy is the right cheap
instrument to map the **hop-depth SNR curve** without GPU cost). A `query_chain` shim can be
written as a few lines around the existing `_unbind_phases` + `_cleanup` + `kb` loop.

### The probe

1. Build a small relational KB of **separate** SVO facts forming explicit chains, e.g.
   `dog chase cat`, `cat eat fish`, `fish swim river`, … (and several independent chains so
   there is a real population, not one path). Use the project's own concept codes.
2. **2-hop query:** `query_chain(dog, [patient, patient])` should return `fish`
   (`dog --chase--> cat`, `cat --eat--> fish`). Generalize to `query_chain(cue, roles)` for a
   role list of length `k`.
3. **Sweep hop depth `k = 1..5`** and **dimension `D ∈ {128, 256, 512}`**. For each `(k, D)`
   record the fraction of held-out chains answered correctly.

### GO / BOUNDARY / NEGATIVE thresholds (must beat the controls below, multi-seed ≥ 3 seeds for the cheap probe, then 6 for the build gate)

- **GO:** at `D = 128` (production), **2-hop accuracy ≥ 0.90** on held-out chains AND
  **strictly above** the memorization-floor spreading baseline by a clear margin (e.g. ≥ 0.5
  absolute), AND the permuted-relation control collapses to chance, AND the lesion collapses
  it. (Chance for a `k`-hop query over `V` concepts is `1/V` per the final cleanup; with V≈16
  in the probe, chance ≈ 0.06.) Report the **depth at which accuracy crosses 0.5** as the
  honest "how many hops" number.
- **BOUNDARY:** 2-hop works (≥ 0.90) but **3-hop falls below 0.5 at D=128** and only recovers
  by raising `D` (e.g. 3-hop needs D≥256). This is the *expected* and still-publishable outcome
  — it **maps the SNR/cleanup depth limit precisely** ("reachable to 2 hops on the point-neuron
  substrate at production D; deeper chains need higher D / redundancy"). An honest
  depth-limited result IS the deliverable.
- **NEGATIVE:** even **2-hop ≤ the spreading baseline** at all D (the role-structured chase
  buys nothing over leaky spreading), OR the permuted-relation control does **not** collapse
  (meaning the "chain" is reading co-occurrence, not relations). Then multi-hop is the next
  genuine wall and the recommendation flips to Option 4 (TEM) as a research program.

---

## Anti-cheat controls (LOAD-BEARING — these exist *because* of the retraction)

Every one of these is mandatory; the retraction happened precisely because a result was
celebrated without them. A multi-hop number that any of these defeats is **not a result**.

1. **Memorization-floor / spreading baseline (the decisive one).** Run the SAME facts through
   the **leaky spreading mechanism** — `SpikingSpreadingController` (or the multitag
   `multitag_transitive_eval` path) — and a **pure lookup table that returns the highest
   co-occurrence neighbour**. If this non-reasoning baseline answers the 2-hop query as well as
   the pointer-chase, the pointer-chase has demonstrated **nothing**. The pointer-chase MUST
   beat it. (This is the formal version of "soft graph traversal surfaces 2nd-degree
   neighbours" — `2026-05-14-multitag-spurious-are-2nd-degree.md` — which is the trap.)
2. **Permuted-relation control.** Re-bind the facts under a **random permutation of the
   relation/role labels** (or shuffle which patient goes with which agent), holding the concept
   set fixed. Real relational inference must **collapse to chance** under this permutation
   (the answer depends on *which* relations chain, not on the concepts being present). If
   accuracy survives the permutation, the model is reading concept co-occurrence, not relations
   — the exact failure mode of the retracted result, restated as a control. (Mirror of the
   project's standing permuted-label control, `permuted_label_check.py`.)
3. **Held-out chains never trained.** The tested chains (`dog→cat→fish`) must use **premise
   pairs the model stored** but a **2-hop combination it was never given as a direct fact**
   (no `dog … fish` fact). Train on the edges, test on the *composition*. (This is the
   Eichenbaum design exactly: premise pairs trained, inference on indirectly-related items.)
   Crucially, also include **held-out chains whose 2-hop endpoint is NOT a 1st-degree
   co-occurrence neighbour of the cue**, so a spreading baseline cannot reach it — this is what
   separates genuine 2-hop from smearing.
4. **Lesion / ablation that should collapse the capability.** Sever the between-hop re-cue
   (feed hop-1's output as a **random concept** instead of the cleaned filler) — accuracy must
   drop to chance, proving the chain is load-bearing. Equivalently, **disable the cleanup
   between hops** and show the chain dies (proving cleanup is the mechanism that resets SNR).
5. **Abstention / no-confab moat preserved.** A chain whose intermediate hop has **no matching
   fact** must return `None`, not a confabulated terminal concept. Verify the moat survives at
   every hop (an unstored `cue` and a broken mid-chain both abstain).

---

## Honest reality check — is multi-hop reasoning reachable on this point-neuron substrate?

**Two hops: yes, with high confidence. Three+ hops: a mapped SNR boundary, not a hard wall —
and multi-turn dialogue is reachable but unbuilt.** The pointer-chase (Option 1) sidesteps the
fundamental superposition limit by *re-discretizing between hops*: cleanup snaps each
intermediate recall back to a clean codebook vector, so error does not integrate
multiplicatively — the chain pays one unbind's noise per hop and survives as long as each
single hop clears the cleanup margin. The project **already** does one such hop at D=128
multi-seed (the validated relational-memory demo), so 2 hops is a short, well-grounded
extension. The genuine limit is the **per-hop unbind/cleanup SNR**: HRR/FHRR retrieval degrades
~linearly with bundled-item count and improves with dimension (Plate 1995; Frady-Sommer 2019;
Kymn et al. 2024 factorizer-noise analysis), so deep chains (4–5 hops) at the production D=128
will likely cross the cleanup floor — the de-risk's job is to *measure exactly where*, and a
depth-limited "reachable to N hops at D=128" is itself the scientific deliverable (it
characterizes what the point-neuron substrate can do unaided). The **true** wall for *general*
relational reasoning — generalizing a relation to never-seen items, not just chasing pointers
between stored facts — is Option 4 (a TEM-style factorised structural code), which is
months-scale and the strategic end-state, not a near-term build. Multi-**turn** dialogue is a
separable, lower-risk add (Option 3): the spiking NMDA loop-attractor WM is validated; it
simply needs to be carried across turns rather than reset per call. **Recommendation: build
Option 1's `query_chain` + the five anti-cheat controls as the cheap-first numpy probe; if GO
(2-hop beats all controls), promote to a 6-seed GPU gate and add Option 3 for multi-turn; if
the spreading baseline ties it, multi-hop is the next genuine wall and we escalate to the TEM
research track.**

---

### Sources

Catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`): **D.01** (declarative/
episodic), **D.02** (Eichenbaum–Cohen relational binding / "memory space" — *names transitive
inference*), **D.03** (trisynaptic loop), **D.05** (CA3 autoassociator, Marr 1971), **G** cluster
(working memory / PFC attractor), **N.15** (theta–gamma multiplexed WM buffer, Lisman & Idiart
1995). Kandel 6e Ch 52 pp 1301–1302, Ch 54 pp 1342, 1360–1361.

Literature: Dusek & Eichenbaum 1997 / Bunsey & Eichenbaum 1996 (hippocampus required for
transitive inference, not for premise-pair learning); Whittington et al., *The Tolman–Eichenbaum
Machine*, Cell 2020 (S0092-8674(20)31388-X); *The Spiking Tolman–Eichenbaum Machine*, bioRxiv
2025.10.16.682754; Plate 1995 (HRR); Frady & Sommer 2019 (FHRR resonate-and-fire); Kymn et al.
2024, *On the Role of Noise in Factorizers for Disentangling Distributed Representations*
(arXiv:2412.00354); Collins & Loftus 1975 (spreading activation).

Project prior art: `research/findings/2026-05-14-CRITICAL-bug-compose-concept-architecture-mismatch.md`
(retraction), `2026-05-14-multitag-spurious-are-2nd-degree.md` (the 2nd-degree-leak trap),
`research/runners/{rf_phasor_composer,compose_relational_memory_demo,learned_assoc_graph,content_selection_spiking,multitag_transitive_eval,compose_concept_chain_test}.py`.
