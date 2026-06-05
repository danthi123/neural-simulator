# Cheat D — the dialogue-planning association graph is a Python dict, not learned in the substrate: biology-grounded conversion research

**Date:** 2026-06-05
**Type:** research / conversion plan (no code shipped here — the deliverable is this analysis)
**Status:** plan + honest difficulty assessment. The recommended mechanism is **TRACTABLE for storage,
PARTIAL for cue→associate spreading** (the project has already measured the exact partial: see §6).

---

## 0. The shortcut, precisely stated

The conversational agent's **dialogue planning** ("what on-topic concept to bring up next") runs on a
genuinely spiking dlPFC: `SpikingSpreadingController` (`research/runners/content_selection_spiking.py`,
class at line 278) holds the discourse context in a real cortico-PFC loop-attractor working memory on a
`SimulationBridge` and computes relevance by **spreading spikes** through inter-assembly synapses
(`turn_latency`, line 371; `relevance_by_latency`, line 389). That part is **not** a cheat — it was
validated 6/6 multi-seed (`2026-06-03-content-selection-milestone3-spiking-relevance-VALIDATED.md`).

The cheat is **the source of the associations it spreads over.** The association graph is a **Python
dict** `{concept: {concept: weight}}` recomputed every call from co-occurrence in the agent's Python
knowledge base:

- `brain_conversational_agent.py::_assoc_graph` (line 204) and the identical
  `rf_phasor_composer.py::_assoc_graph` (line 219): iterate `self.composer.kb` (a Python list of facts),
  and for every fact add `+1.0` to `graph[x][y]` for each ordered pair of the fact's
  agent/action/patient. Pure Python, O(facts) each call.
- `elaborate` (line 218 / 231) then hands that dict to `SpikingSpreadingController(graph)`, whose
  `_install_graph_edges` (line 315) writes the weights into the bridge's `c2d` pathway:
  `set_pathway_weights("c2d", ..., weights = graph[A][B] * edge_scale, add_missing=True)`.

So the *representation* during spreading is synaptic (good), but the *content* — **which concepts
associate with which, and how strongly** — is computed in Python from a Python store and **stamped onto
the synapses by hand each call.** The biology-grounded target: those concept-concept association weights
should be **learned, Hebbian, and held in the substrate** — concepts that co-occur in stored facts
should *wire together* through experience, and the dlPFC should spread over the **learned** synaptic
associations, never a recomputed dict.

The most-integrated version (`unified_brain_bridge.py`, Step 3 `enable_dlpfc`, lines 82–257) puts the
dlPFC assemblies on the **one unified bridge** — but it *still* calls `_install_graph_edges` to stamp
`graph[A][B]` onto the pre-allocated edges (line 115). The cheat persists end-to-end.

---

## 1. The biology this shortcut violates

### 1.1 "Neurons that fire together wire together" — Hebbian co-occurrence IS the association
`docs/biology.md` §"How the brain learns (plasticity)" (lines 239–296): Hebbian learning (Hebb 1949),
implemented as STDP (A-before-B within ~20 ms → LTP; Bi & Poo 1998 *J. Neurosci.* 18:10464; Song et al.
2000 *Nat. Neurosci.* 3:919; Caporale & Dan 2008 *Annu. Rev. Neurosci.* 31:25). **A concept-concept
association is exactly a Hebbian-strengthened synapse between two concept assemblies that fired together
during an episode.** Building that association by counting co-occurrences in a Python list and stamping a
weight is a *non-biological re-implementation of the very thing STDP is for.* The brain does not keep an
external co-occurrence table; the count lives *in the synaptic weight*, accumulated one episode at a time.

### 1.2 Cortical association areas + the semantic hub — distributed Hebbian assemblies
Concept-concept associations in the brain are cortico-cortical. Kandel 6e Ch 67 (implicit/associative
memory) and the anterior-temporal-lobe **semantic hub** literature (Patterson-Lambon Ralph hub-and-spoke;
Pobric-Jefferies-Lambon Ralph 2010 *PNAS* 107:2717, "Coherent concepts are computed in the anterior
temporal lobes"; Lambon Ralph et al. 2017): the ATL hub binds modality-specific "spoke" features into
coherent concepts, and **associations between concepts are learned synaptic links among these distributed
assemblies.** The directly-relevant computational model is **Garagnani & Pulvermüller's
neurobiologically-constrained spiking cortex** (Tomasello, Garagnani, Wennekers, Pulvermüller 2018,
*"A Neurobiologically Constrained Cortex Model of Semantic Grounding With Spiking Neurons and Brain-Like
Connectivity"*, Front. Comput. Neurosci. / PMC6232424): word-forms and their semantically-related object/
action representations become **linked via Hebbian (Artola-Singer) plasticity across perisylvian, visual,
and motor regions**, forming distributed **cell assemblies**; once learned, presenting one pattern
"triggers distributed circuit reactivation across all associated areas" — i.e. **associative spreading is
an emergent property of the learned synaptic strengths, not an external graph.** This is almost exactly
the mechanism the cheat should be converted to, and it is the project's own catalog lineage (Pulvermüller
G.20 distributed cortical word ensembles, already used for the 320-concept tier).

### 1.3 Hippocampal CA3 autoassociation — the canonical "learn associations from co-occurrence" circuit
`docs/biology.md` §"Memory: hippocampus and replay" (lines 380–433): CA3 is a **pattern completer** —
heavy recurrent collaterals, each pyramidal contacting ~5% of the others, **synaptically modifiable**, so
partial cues retrieve full memories (Marr 1971 *Phil. Trans. R. Soc. B* 262:23, "Simple memory: a theory
for archicortex"; McClelland-McNaughton-O'Reilly 1995 *Psychol. Rev.*). The formal account is **Treves &
Rolls** (1994 *Hippocampus* 4:374; Rolls 2013 *Front. Cell. Neurosci.* 7:98, "A quantitative theory of the
functions of the hippocampal CA3 network in memory"): CA3 is a single **autoassociation network** whose
**recurrent-collateral synapses are associatively (Hebbian) modified** so that *"arbitrary associations
between inputs originating from very different parts of the cerebral cortex"* are formed and recalled by
completion; **"any place could be associated with any object, and the object recalled with a spatial cue,
or the place with an object cue"** — i.e. **bidirectional concept-concept association learned into the
recurrent weights, retrieved by cue.** The number of retrievable patterns scales with the number of
modifiable recurrent synapses / sparseness. **This is the textbook version of exactly what we want: store
"apple co-occurred with big" by Hebbian-strengthening the apple↔big recurrent synapses; later, cue
"apple" → completion lights "big".**

### 1.4 Spreading activation has a neural-attractor account
The dlPFC's spreading-activation read is itself well-grounded: Collins & Loftus 1975 *Psychol. Rev.* 82:407
(spreading-activation theory of semantic processing) and Anderson's ACT-R (1983) formalize activation
propagating along associative links and decaying with distance — and the **neural** realization is an
attractor network with latching dynamics (Lerner, Bentin & Shriki 2012 *Cogn. Sci.* 36:1339, "Spreading
Activation in an Attractor Network With Latching Dynamics: Automatic Semantic Priming Revisited"): concept
**attractors**, **Hebbian-learned links between them**, and activation that **latches on one attractor and
hops to an associated one** — which is *precisely* `SpikingSpreadingController`'s loop-attractor +
inter-assembly-synapse design. So the spreading mechanism is already faithful; **only the link weights need
to come from learning instead of a dict.**

**Bottom line of §1:** the cheat replaces a *learned synaptic associative memory* (CA3 recurrent
autoassociator / Garagnani-Pulvermüller Hebbian cortical assemblies / Lerner attractor-with-latching) with
a *recomputed Python co-occurrence table.* The conversion is to learn the concept-concept weights by
Hebbian co-occurrence and let the dlPFC spread over them.

---

## 2. This is the SAME mechanism the project already has (reuse, don't reinvent)

The key finding of this research: **the biology-grounded resolution is not new infrastructure — the project
has already built and validated the substrate associative memory.** Three existing pieces line up:

### 2.1 Engram-tag co-occurrence store (catalog D.14, Tonegawa) — ALREADY a substrate co-occurrence memory
`sim/bridge.py` engram API (`start_engram_recording` 2593, `commit_engram_tag` 2622, `stimulate_tag` 2707,
`get_engram_tag_indices` 2759). The **engram-stim-recall** finding
(`2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`) is *exactly the target mechanism, already
working:* encode a (concept-A, concept-B) co-occurrence by driving both concept pools and committing a
top-K engram tag spanning both → **stimulating the tag reactivates BOTH concepts 87.5% multi-seed (5 seeds
× 8 pairs)**, vs 8.3% chance. The **multitag cue retrieval** finding
(`2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md`) indexes those tags by cue word and aggregates:
**90% FULL / 100% PARTIAL multi-seed**. *This is Hebbian co-occurrence stored in the substrate*: the tag is
the set of neurons that fired together for the (A,B) episode; recall is reactivating that ensemble. **The
association graph the dlPFC needs can be the substrate's own engram tags** — and the dialogue-Control plan
already says so (see §3).

### 2.2 CA3 autoassociator (catalog D.13) — ALREADY validated for completion
`2026-05-11-P1-trisynaptic-loop-validation.md`: the `build_biological_brain_regions(
enable_hippocampus_consolidation=True)` trisynaptic loop (`ec/dg/ca3/ca1` + plastic `ca3→ca3` recurrent)
passes **D.13 pattern completion (CA3 cos 0.748 from a 50% partial cue)** and **D.12 separation**. This is
the Treves-Rolls/Marr autoassociator running on the bridge. If concept assemblies are imprinted into CA3
and co-occurring concepts are Hebbian-linked in the recurrent collaterals, a cue concept completes to its
associate — *the §1.3 biology, in this codebase, already.*

### 2.3 The dlPFC spreading Control (validated) — ALREADY reads associations as synaptic spread
`SpikingSpreadingController` already spreads over **synaptic** associations (the `c2d` edges). It even ran
its decisive eval **on the project's real learned associations** (the multitag concept graph, §"Also
RESOLVES 5/5 on the project's REAL learned associations" in the milestone-3 finding) — *but the edge
weights were still set from the dict.* So three-quarters of the conversion is done; the missing quarter is
**where the per-edge weight comes from.**

**Therefore the conversion is an integration, not an invention:** route the dlPFC's association weights
from the substrate's existing learned co-occurrence store (engram tags and/or CA3 recurrent weights)
instead of from `_assoc_graph`'s Python recompute.

---

## 3. The plan already names this fix

`docs/plans/2026-06-03-content-selection-dialogue-control-implementation.md` (the dialogue-Control
implementation plan), **Key design correction (lines 24–27) and Task 8 (lines 335–356)**, explicitly states
the intended biology-grounded source: *"'relevance' is computed from the substrate's **learned
associations** — an association graph derived from the stored engram tags / KB facts (e.g., the tag
`apple_big` means apple is associated with big)"*, with `build_association_graph(tag_names)` reading
`bridge.list_engram_tags()`. And it flags the residual cheat to remove: *"Strengths default to 1.0 (a later
refinement can use retrieval cosine scores)."*

So the **cheat-D fix is the completion of that already-scoped refinement**: (a) the *edge set* comes from
the substrate's learned engram tags (not a Python co-occurrence recompute), and (b) the *edge weight* comes
from a substrate read (engram stim-recall cosine / CA3 completion strength), not a hard-coded `1.0`/`+1.0`
count.

---

## 4. The recommended on-bridge spiking realization

Two layered options. **Option A** is the smallest faithful step and reuses the most validated machinery;
**Option B** is the deeper "fully Hebbian-learned recurrent autoassociator" that maximizes biological
fidelity. Recommend **A first** (it is mostly an integration of shipped, multi-seed-validated parts), then
B as the fidelity stretch.

### Option A (recommended first): learned-engram-derived association weights — "the graph is the substrate's tags"
Replace `_assoc_graph`'s Python recompute with a read of the **substrate's own learned co-occurrence
store**, then keep the validated dlPFC spreading exactly as-is.

1. **Store co-occurrence in the substrate as it happens, not in a dict.** When the agent stores a fact
   (`hear`/`store`), for each co-occurring concept pair (agent/action, action/patient, agent/patient) run
   the **already-validated engram encoding**: drive both concept pools (the `--balanced-teacher-pA 500` +
   `encoding-steps 500` recipe from `2026-05-14-engram-stim-recall…`) with `start_engram_recording(
   f"{a}__{b}")` → `commit_engram_tag(top_k=…)`. The co-occurrence now lives in the substrate as a tagged
   ensemble (Tonegawa D.14), **not** as `graph[a][b]+=1`. Repeated co-occurrence is repeated encoding →
   stronger/larger tag (the Hebbian "fire together → wire together" accumulation, in the substrate).
2. **Derive the dlPFC edge weight from a substrate read, not a count.** For each candidate edge A→B,
   `stimulate_tag("A__B")`, run a short window, read the associate's reactivation strength (the engram
   stim-recall signal — lang_output cosine or pool-firing). That **substrate-measured** strength is the
   `c2d` edge weight. The dlPFC then spreads over weights that are *reads of learned synapses*, closing the
   `1.0`-default residual. (`build_association_graph` already has the hook; this swaps its constant for a
   real stim-recall read.)
3. **Keep the validated spreading.** `SpikingSpreadingController.turn_latency` / `relevance_by_latency`
   are unchanged (6/6 validated). On `unified_brain_bridge`, the dlPFC assemblies are already on the bridge;
   only the *weight source* feeding `_install_graph_edges` changes.

**Why A is the right first step:** every sub-part is *already shipped and multi-seed-validated* — engram
encoding (87.5%), multitag cue indexing (90%/100%), and the dlPFC spread (6/6). The conversion is wiring
"where the weight comes from," and it makes the association memory genuinely substrate-held: the
co-occurrence is in tagged ensembles, the strength is a spiking read. **No `sim/` edits** (engram API +
`set_pathway_weights` already exist; reuse-by-import in the runner, mirroring how `B-substrate-store`
imprinted a fact into static weights without touching `sim/`).

**Honest caveat for A:** the *edge weight* is a substrate read but the **edge SET** is still enumerated by
the runner (which pairs to encode comes from the facts). That is acceptable and biological — encoding *is*
driven by which concepts co-occur in experience — but it is "learned strength over an experience-driven
edge set," not yet "synapses self-organized with zero bookkeeping." Option B removes the last bookkeeping.

### Option B (fidelity stretch): online Hebbian/STDP concept-concept synapses (CA3-style recurrent autoassociator)
Make the concept-concept weights **learned online by STDP**, never stamped:

1. Put the concept assemblies on a **recurrent** region with `plastic_internal=True` (the CA3 recurrent
   collateral pattern from `build_biological_brain_regions(enable_hippocampus_consolidation=True)`, P1.D.13
   validated). The dlPFC's `c2d` loop is made **plastic** (`RegionPathway(..., plastic=True)`,
   `enable_hebbian_learning`/STDP on, `stdp_w_max` raised per the CLAUDE.md soft-bound gotcha).
2. **Learning = co-firing during fact storage.** When a fact is stored, co-drive its concept assemblies
   together for an encoding window; STDP (A-before-B → LTP) strengthens the inter-assembly synapses
   *automatically* — the Hebbian outer-product / Treves-Rolls modification, in spikes. No `graph[a][b]`
   anywhere; the association literally *is* the grown synapse. Repeated co-occurrence → repeated co-firing
   → stronger synapse (graded strength emerges, not a hand-set `1.0`).
3. **Spreading = pattern completion.** Cue concept A → recurrent collaterals complete to associate B
   (Marr/Treves-Rolls), read by the dlPFC's existing latency spread. This is `relevance_by_latency` over
   *grown* edges instead of *stamped* edges.

**Why B is the stretch, not the first step:** it inherits the project's known hard edges — (i) STDP weight
growth from zero-init reaching functional magnitude in a bounded encoding window is the exact failure that
killed the v16 direct-pathway compose arc (`CLAUDE.md` "v16 compose pathway is essentially silent"; the
STDP cold-start problem); (ii) the heteroassociative **cue→associate** read is the project's measured weak
spot (§6). B is the *correct* biology, but it needs the de-risk in §5 before committing.

---

## 5. The smallest de-risk test

**Pre-registered question:** *Can a cue concept, driven alone, spread to its associate through
substrate-held (learned, not stamped) concept-concept weights, cleanly enough for the dlPFC's
latency read to pick the associate — at parity with the current Python-dict spread?*

**Cheapest decisive probe (Option-A path, ~1 GPU run, reuse-by-import, no `sim/` edits):**
1. Load a validated v16 concept-pool bridge (the `2026-05-14` recipe) with ≥3 known co-occurrence pairs
   (e.g. `apple__big`, `apple__cat`, `dog__small`) **already encoded as engram tags** (the shipped
   `compose_concept_engram` encoder).
2. Build the dlPFC `c2d` edges **two ways** on the same assemblies: (i) the current Python-dict weights
   (the oracle), and (ii) the **substrate-read** weights — for each pair, `stimulate_tag` → read the
   associate reactivation strength → use that as the edge weight.
3. Run `SpikingSpreadingController.relevance_by_latency("apple")` for both. **PASS** iff the
   substrate-read graph picks the **same** direct associates (big/cat earliest, dog-cluster never), i.e.
   the learned-weight spread reproduces the dict-weight spread (the `B-substrate-store` parity bar:
   identical winner, multi-seed 42/43/44).

**Why this is the right de-risk:** it isolates the *only* thing that changes (weight source) on the
*already-validated* spreading machinery, and it is graded — it directly reports whether the substrate read
is faithful enough, with the exact parity criterion (`2026-06-05-B-substrate-store-fidelity-GO.md`) the
project used for the analogous memory de-risk. If A passes, ship A and write the GO. If the **cue→associate
direction** is too weak (the §6 risk), that is the signal to invest in Option B's recurrent completion (or
the consolidation/SWR strengthening below) rather than guess.

**Option-B de-risk (only if pursued):** one pair, STDP on a plastic `c2d` loop, co-fire for N encoding
events, then drive the cue alone and check the associate's first-spike latency drops below the
unrelated-concept floor *as a function of N* — i.e. does the grown synapse ever reach functional magnitude
before interference? (This is the v16-compose cold-start question, re-asked for dialogue edges.)

---

## 6. HONEST difficulty

**Storage of the associations in the substrate: TRACTABLE (largely already done).** Holding concept-concept
co-occurrence as engram tags is validated at 87.5% (stim-recall) / 90%-FULL (multitag), multi-seed. The
"associative memory is a Python structure" cheat is, for **storage**, a wiring change away from being
substrate-held, reusing shipped APIs with no `sim/` edits. Option A is mostly integration of validated
parts.

**Cue→associate *spreading* over the learned weights for clean dialogue selection: PARTIAL — and the
project has already measured the exact partial.** The decisive honest number is in
`2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md`: **co-stimulation** of a tag reactivates both
concepts at 87.5%, but **cue-only associative recall** (drive A *alone*, expect B in top-3) is **27.5%
multi-seed — barely above the 20% chance floor.** The heteroassociative-memory literature predicts exactly
this asymmetry: clean cue→associate completion needs **sparse population codes** (capacity ~0.7 bit/synapse
in heteroassociation; Knoblauch-Palm; "population sparseness determines strength of Hebbian plasticity")
and enough modifiable synapses per cell (Treves-Rolls capacity ∝ recurrent-synapses / sparseness). The v16
concept pools are not optimally sparse for this, and the project's STDP cross-pool compose attempts went
NEGATIVE for the same cold-start reason. So:

- The dlPFC's **latency spread itself** is robust (6/6) — *given* edges of the right weights.
- Whether **substrate-read** weights (Option A) preserve that robustness is the §5 de-risk — *plausible*
  because the read is of a strong (87.5%) co-stim signal, not the weak (27.5%) cue-only signal, but
  **unverified** (it has not been run; this is research, not a result).
- A **fully online-STDP-learned** recurrent autoassociator (Option B) is the *most* faithful but is the
  *least* de-risked, inheriting both the STDP cold-start hazard and the sparse-coding capacity constraint.

**Net assessment:** **TRACTABLE → ship Option A behind the §5 de-risk** (substrate-held storage + substrate-read
edge weights, reusing engram + dlPFC machinery, no `sim/` edits). **PARTIAL → flag honestly** that
cue-directional associative strength is the project's known soft spot, so the de-risk's parity gate is
load-bearing; do **not** claim "the association memory is now learned in the substrate" until the §5 probe
passes multi-seed at dict-parity. **HARD (defer)** the fully-online-STDP recurrent version (Option B) until
A's de-risk shows whether the simpler substrate-read suffices — and if cue→associate strength is the
blocker, the principled lever is the project's **SWR sleep-replay consolidation** (catalog D.13/D.19,
`consolidation_trainer.run_concept_replay_phase`): replaying co-occurring concept ensembles strengthens the
association into cortex over cycles (McClelland-McNaughton-O'Reilly 1995), the biological route to lifting a
weak heteroassociation into a reliable one — rather than hand-tuning weights.

---

## 7. Concrete next step (one line)

Run the §5 Option-A de-risk: build the dlPFC `c2d` edges from **engram stim-recall reads** of an already-
tagged v16 bridge and check `relevance_by_latency` picks the same direct associates as the Python-dict
oracle (parity, seeds 42/43/44). GO → wire the substrate-read into `build_association_graph` /
`_assoc_graph` (runner-only, no `sim/` edits) and the association graph is genuinely substrate-held;
NEGATIVE on the cue direction → escalate to SWR-consolidation strengthening (Option B path).

---

## Citations

**Project (reuse):**
- `research/runners/content_selection_spiking.py` (`SpikingSpreadingController` L278, `turn_latency` L371,
  `relevance_by_latency` L389, `_install_graph_edges` L315) — the validated spiking dlPFC spread.
- `research/runners/brain_conversational_agent.py` `_assoc_graph` L204 / `elaborate` L218;
  `rf_phasor_composer.py` `_assoc_graph` L219 / `elaborate` L231 — the cheat.
- `research/runners/unified_brain_bridge.py` L82–257, L115 — cheat persists on the unified bridge.
- `sim/bridge.py` engram API L2593/2622/2707/2759; `set_pathway_weights` L2305.
- `research/findings/2026-06-03-content-selection-milestone3-spiking-relevance-VALIDATED.md` (6/6 spread;
  "Graph is installed, not learned" scope, L102–106; runs on real learned associations, L236–240).
- `research/findings/2026-05-14-engram-stim-recall-multi-seed-VALIDATED.md` (87.5% co-stim / **27.5%
  cue-only** — the honest boundary).
- `research/findings/2026-05-14-multitag-cue-retrieval-90pct-VALIDATED.md` (90% FULL / 100% PARTIAL).
- `research/findings/2026-05-11-P1-trisynaptic-loop-validation.md` (CA3 D.13 completion 0.748).
- `research/findings/2026-06-05-B-substrate-store-fidelity-GO.md` (the parity-de-risk template; no `sim/` edits).
- `docs/plans/2026-06-03-content-selection-dialogue-control-implementation.md` L24–27, L335–356 (the fix is
  already named: learned-engram-derived graph; "1.0 default → cosine scores" refinement).

**Biology / catalog / textbook:**
- `docs/biology.md` L239–296 (Hebbian/STDP), L380–433 (hippocampus/CA3 completion/replay).
- Kandel 6e Ch 54 pp 1340–1342, 1357–1361 (trisynaptic loop, DG separation, CA3 completion); Ch 67
  (implicit/associative memory). Marr 1971 *Phil. Trans. R. Soc. B* 262:23.
- Catalog D.12 (pattern separation), D.13 (pattern completion, Marr autoassociator), D.14 (Tonegawa engram
  cells), D.19 (concept replay), G.20 (Pulvermüller distributed cortical word ensembles).

**Literature (web):**
- Treves & Rolls 1994 *Hippocampus* 4:374; Rolls 2013 *Front. Cell. Neurosci.* 7:98 — CA3 autoassociation,
  modifiable recurrent collaterals store arbitrary bidirectional cortical associations, recall by cue
  completion; capacity ∝ recurrent synapses / sparseness.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3691555/
- Tomasello, Garagnani, Wennekers & Pulvermüller 2018, *A Neurobiologically Constrained Cortex Model of
  Semantic Grounding With Spiking Neurons and Brain-Like Connectivity* (PMC6232424) — Hebbian (Artola-
  Singer) learning forms distributed concept cell assemblies; presenting one reactivates associated areas
  (associative spread emergent from learned weights). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6232424/
- Collins & Loftus 1975 *Psychol. Rev.* 82:407 — spreading-activation theory of semantic processing;
  Anderson ACT-R (1983) spreading activation. https://www.researchgate.net/publication/200045115
- Lerner, Bentin & Shriki 2012 *Cogn. Sci.* 36:1339 — *Spreading Activation in an Attractor Network With
  Latching Dynamics* (concept attractors + Hebbian links + latching = the dlPFC's exact design).
  https://onlinelibrary.wiley.com/doi/10.1111/cogs.12007
- Pobric, Jefferies & Lambon Ralph 2010 *PNAS* 107:2717 — coherent concepts computed in ATL semantic hub.
  https://pnas.org/content/107/6/2717
- Heteroassociative capacity / sparse coding: Knoblauch & Palm (iterative retrieval, block coding;
  ~0.7 bit/synapse heteroassociation); "Population sparseness determines strength of Hebbian plasticity for
  maximal memory lifetime in associative networks" (bioRxiv 2025).
  https://www.biorxiv.org/content/10.1101/2025.06.16.659837v2.full
