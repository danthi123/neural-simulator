---
type: finding
status: contributing
date: 2026-06-21
---

# A13 — the dialogue-planning association graph (host dict by default): on-substrate-close scoping (2026-06-21)

**Type:** read-only deep-research + scoping (this doc is the only write). No code edits.

**Trigger.** The definitive shortcut inventory (`2026-06-21-shortcut-inventory-definitive.md`, `ddc3b8db`) surfaced
A13 — a previously-untracked residual — under the owner's 2026-06-21 bar (close ALL host shortcuts AND run the one
brain fully-spiking end-to-end, compatible with future neuromorphic hardware). Associative recall for dialogue
planning is a cognitive operation that should be neural, not a host Python dictionary. This doc pins down the exact
residual, decides whether it is a genuine shortcut or a host index over already-neural recall, ranks the cheapest
on-substrate closes, and recommends a cheap-first de-risk.

**One-line answer.** The SELECTION/spreading is already genuinely spiking; the residual is the association GRAPH
CONTENT (which concepts relate, and how strongly) being recomputed as a Python `{concept: {concept: weight}}` dict
from the agent's fact list on the default path. A fully-built, multi-seed-validated on-substrate replacement
(`LearnedAssocGraph`, the Hebbian CA3-recurrent that LEARNS the co-occurrence) **already exists and is wired into the
agent behind `enable_learned_assoc` (default OFF)**. The cheapest close is a **default flip** (plus a one-line plumb
through the two production demos). It is a genuine criterion-2 (structure) residual, NOT criterion-1 — the op is
spiking — and it is **partly already-neural with a host structure-source**, not a fully host shortcut.

---

## 1. The exact host residual (file : function + what is a dict vs already-neural + size)

### 1.1 What runs the dialogue plan today

`elaborate(topic)` is the "bring up the next on-topic concept" dialogue-planning entry point. Two things happen:

1. **Build the association graph** — `_assoc_graph()` returns a Python dict `{concept: {concept: weight}}` where each
   stored fact's agent/action/patient concepts co-occur (+1.0 per ordered pair). Clause patients are skipped (their
   inner concepts are structural). **This is the host residual.**
2. **Spread + select over that graph** — `SpikingSpreadingController(graph).turn_latency([topic])` holds the discourse
   context in a real cortico-PFC loop-attractor working memory on a `SimulationBridge` and computes relevance by
   spreading spikes through inter-assembly synapses; the earliest-first-spike (1-hop) candidate is the selection.
   **This is genuinely spiking** (criterion-1 YES).

### 1.2 Exactly where the dict is, and how big

| Surface | File : symbol | What is a host dict | What is already neural |
|---|---|---|---|
| **Agent (the production `elaborate` path)** | `research/runners/brain_conversational_agent.py` : `_assoc_graph` (line 451), `elaborate` (line 471) | `_assoc_graph` recomputes the `{concept:{concept:weight}}` dict from `self.composer.kb` (a Python list of fact dicts) on the default path (line 461–469) | the spread/selection (`SpikingSpreadingController`, on a bridge); AND, behind the flag, the graph itself (see §1.3) |
| **rf / onebrain composer (a parallel `elaborate` the agent does NOT call)** | `research/runners/rf_phasor_composer.py` : `_assoc_graph` (657), `elaborate` (669) | same dict recompute from `self.kb` (660–667); **no `enable_learned_assoc` branch here** (0 occurrences) | the spread/selection only |
| **The spreading Control (structure injection)** | `research/runners/content_selection_spiking.py` : `SpikingSpreadingController._install_graph_edges` (315) | reads the `graph` dict and stamps each edge as `set_pathway_weights("c2d", weights = graph[A][B] * edge_scale, add_missing=True)` (315–329) — i.e. the synaptic structure is host-computed-and-injected from the dict each call | the spreading op itself (loop-attractor WM, latency read) is spiking |
| **Inhibition-of-return (a separate, minor residual on the same path)** | `content_selection.py` : `SaidTrace` (58) | a small numpy decay dict (`activation`/`mark`/`step`) that excludes a just-said concept for several turns | — (the module itself flags spike-frequency-adaptation as the "Milestone-3b" step) |

**Size of the dict (criterion-2 structure).** O(facts) to build; the graph has one node per concept that appears in a
stored fact and an edge for every co-occurring pair within a fact. It is recomputed on **every** `elaborate` call (the
Control caches by the graph CONTENT, so a new fact rebuilds the Control). It IS built from the agent's own stored facts
— it is not an external hand-authored knowledge base; it is a co-occurrence count over what the agent heard. So the
"shortcut" is specifically: *the brain does not keep an external co-occurrence table; that count should live in synaptic
weight, accumulated one episode at a time* (the cheat-D research doc, §1.1, citing Hebb 1949 / Bi-Poo 1998).

### 1.3 The opt-in replacement already exists and is wired (agent only)

`research/runners/learned_assoc_graph.py` : `LearnedAssocGraph` — a substrate-learned concept-concept association
memory (the CA3 / Treves-Rolls autoassociator), reusing `_D_sparse_heteroassoc` (the RESOLVED sparse heteroassociative
memory, `2026-06-05-D-cue-recall-RESOLVED-sparse-heteroassoc.md`). NO `sim/` edits.

- Concepts = sparse K-of-N patterns in a pool with a **plastic excitatory recurrent** (zero-init).
- `store_fact(concept_list)` co-fires the fact's concept patterns → the recurrent **grows** the pairwise co-occurrence
  by Hebbian co-fire (NOT set) (lines 31–51: gate plasticity ON, drive 30 cycles × ~15 steps, gate OFF).
- `graph(thresh)` **reads the learned recurrent weights** back into a `{concept:{concept:weight}}` dict (53–66).

**Wiring (agent, already present):** the constructor builds it when `enable_learned_assoc=True`
(`brain_conversational_agent.py:229–233`); every `hear()` path calls `store_fact(...)` (lines 321, 344, 371, 407); and
`_assoc_graph()` returns `self._learned_assoc.graph()` when present (459–460). So on the agent the entire learn-from-
facts → read-learned-graph → spread loop is wired; it just defaults OFF.

**Validation (already done).** `tests/test_brain_conversational_agent.py:test_learned_assoc_graph_agent` (164, GPU-only):
with `enable_learned_assoc=True`, the substrate-learned graph is non-None and `elaborate("dog")` returns a true
co-occurring associate. The module's own `main()` parity harness compares the learned graph to the Python co-occurrence
oracle (edges recovered, top-associate match). The inventory records the multi-seed result as **24/24 edges, 9/9 top
associate**. The underlying sparse heteroassociator is multi-seed clean + anti-cheat-clean (permuted-encoding follows
the encoding → genuinely learned, `2026-06-05-D-cue-recall-RESOLVED`).

---

## 2. Genuine shortcut, or host index over neural recall?

**It is a genuine criterion-2 (structure) residual, NOT criterion-1 (op).** Precisely:

- **Criterion 1 (runtime-spiking op): already YES.** The dialogue selection — the relevance computation and the choice
  — runs as spreading spikes through assemblies on a real bridge (`SpikingSpreadingController`, validated 6/6 multi-seed,
  `2026-06-03-content-selection-milestone3-spiking-relevance-VALIDATED.md`). This half is not a shortcut.
- **Criterion 2 (hardware-portable structure): HOST-DESIGNED by default.** The association weights the spread runs over
  are computed by a host Python formula (count co-occurrences in `kb`) and injected via `set_pathway_weights`. For a
  neuromorphic port, a host would have to compute the co-occurrence table and stamp the synapses — exactly the
  "spiking-at-runtime, host-designed-structure" case the owner's hardware-port lesson targets.

So it is **partly already-neural with a host structure-source** — the honest framing is "host index over neural recall"
on the SELECTION side, but a genuine host computation on the STRUCTURE side. The replacement closes criterion 2: with
`LearnedAssocGraph`, the weights are *learned by Hebbian co-fire on the substrate*, so the structure self-organizes from
experience rather than from a host formula.

**One residual nuance to be honest about (does NOT block the close).** Even with `enable_learned_assoc=True`,
`graph()` reads the learned recurrent into a Python dict, which `_install_graph_edges` then re-injects into a *second*
bridge (the dlPFC spread bridge) via `set_pathway_weights`. So the close moves the *content computation* (which concepts
relate + the weights) onto the substrate (the real win), but a host dict still mediates the hand-off between the
learned-association bridge and the spread bridge, and the spread bridge's edges are still stamped. The fully-host-free
end-state is the learned associations and the spread living on ONE bridge with no dict hand-off (the `unified_brain_bridge`
Step-3 direction, §3 close (c)). The flip is the large, cheap, validated first step; the single-bridge fold is the
deeper follow-on.

---

## 3. Ranked cheap-first on-substrate closes

### (a) RECOMMENDED — flip `enable_learned_assoc` to default-on (agent), plumb it through the two production demos

- **What:** make the agent's default the substrate-learned graph; plumb the flag through `consolidated_320_conversation_demo.py`
  (constructs `BrainConversationalAgent` at line 126 without the flag → currently host dict on the flagship path) and
  `MultiTurnAgent` (`multi_turn_agent.py:47–56`, does not expose/pass it).
- **Reusable machinery:** `LearnedAssocGraph` + `_D_sparse_heteroassoc` (both built, no `sim/` edits) + the existing
  agent wiring (constructor + every `hear()` + `_assoc_graph`) + the existing test.
- **Cheap-first de-risk:** run `test_learned_assoc_graph_agent` (exists) at multi-seed on GPU AND the
  `consolidated_320` demo with the flag on; assert `elaborate(topic)` returns a true co-occurring associate (parity vs
  the host-dict oracle) on the validated topics, the no-confab moat holds (`what_does`/`render_fact`/`reason_chain`
  abstain `is None` unchanged — `elaborate`/the graph never touches the abstention path), and a lesion control
  (no `store_fact` → empty learned graph → `elaborate` returns `None`) + a no-learning control collapse. Wall-clock
  note below.
- **Anti-cheat:** spiking-assoc plan == host-dict plan on the validated topics; moat preserved; lesion/no-learning
  collapses; (already-precedented) permuted-encoding follows the encoding.
- **`sim/` edit?** NO. Default flip + a flag plumb in two demos.
- **Honest cost / scope:** `store_fact` builds a separate ~1800-neuron bridge once and runs ~450 sim steps per `hear()`
  call (30 cycles × ~15 steps). That is real per-turn wall-clock the host dict does not pay, and the learned bridge is a
  SEPARATE bridge (not the one brain) with a dict hand-off into the spread bridge (§2 nuance). GPU-only (the bridge build
  + steps); keep a `False` escape for the numpy-CPU + test-oracle path (mirrors the `enable_spiking_cleanup` /
  `local_reciprocal_unbind` pattern). This is the validated, lowest-risk close and the one the inventory recommends.

### (b) ALSO — give the rf/onebrain composer's own `elaborate` the same learned path (close the parallel surface)

- **What:** the composer-level `_assoc_graph`/`elaborate` (`rf_phasor_composer.py:657/669`, inherited by onebrain) has
  NO `enable_learned_assoc` branch (0 occurrences). The agent does not call it (it has its own `elaborate`), so
  production is covered by (a). But a caller invoking `composer.elaborate(...)` directly still hits the host dict.
- **Reusable machinery:** the SAME `LearnedAssocGraph`; mirror the agent's wiring (build in the composer ctor; call
  `store_fact` in the composer's store paths; read in the composer's `_assoc_graph`).
- **De-risk / anti-cheat:** same parity + moat + lesion gate, asserted on the composer surface.
- **`sim/` edit?** NO. **Priority:** lower than (a) — completeness, not the production path.

### (c) DEEPER (follow-on, not cheap) — fold the learned associations + the spread onto ONE bridge (no dict hand-off)

- **What:** put the learned-association recurrent and the dlPFC spread assemblies on the SAME bridge so there is no
  `graph()`-dict → `_install_graph_edges` re-injection — the spread reads the learned synapses directly. This is the
  `unified_brain_bridge.py` Step-3 direction the cheat-D research doc flagged (the most-integrated path still calls
  `_install_graph_edges` today, lines 109–119).
- **Reusable machinery:** `unified_brain_bridge` (the union/Step-3 dlPFC join) + the learned recurrent.
- **De-risk / anti-cheat:** parity vs (a) on the validated topics; moat preserved; the spread reads learned (not
  stamped) edges (assert no `set_pathway_weights` at `elaborate` time).
- **`sim/` edit?** Likely NO (reuse-by-import), but it is a real integration build (layout faithfulness, plasticity
  isolation of the spread bridge), not a flag flip. **Priority:** the genuine fully-host-free end-state; sequence after
  (a)/(b).

### (d) ADJACENT (separate, minor) — make inhibition-of-return spiking (`SaidTrace` → SFA)

- The just-said exclusion is a numpy decay dict; the module flags spike-frequency-adaptation on the selected assembly as
  the documented "Milestone-3b" step. Tangential to A13 (which is the GRAPH), tiny, listed for completeness.

---

## 4. Recommended cheap-first de-risk

**Flip `enable_learned_assoc` default-on at the agent + plumb the flag through `consolidated_320_conversation_demo.py`
and `MultiTurnAgent` (close (a)); de-risk by running `test_learned_assoc_graph_agent` multi-seed on GPU AND the 320
demo with the flag on, asserting (i) `elaborate` parity with the host-dict oracle on the validated topics, (ii) the
no-confab moat unchanged (`is None` abstentions verbatim), (iii) a lesion (no `store_fact`) + no-learning control
collapse.** This is the lowest-risk, already-validated, no-`sim/`-edit close. Carry the honest cost (per-turn ~450
steps on a separate ~1800-neuron GPU bridge) and keep a `False` numpy-CPU/test-oracle escape. Then optionally do (b)
for the composer surface; sequence (c) — the single-bridge fold — as the deeper fully-host-free follow-on.

---

## 5. Honest framing + downstream dependency

- **Genuine residual vs host-index:** criterion-1 (the SELECTION op) is already spiking and validated; the residual is
  criterion-2 (the graph STRUCTURE) being a host co-occurrence dict on the default path. So A13 is **partly already-
  neural with a host structure-source**, not a fully host shortcut. The replacement (Hebbian-learn the co-occurrence on
  the substrate) closes criterion 2.
- **Size:** small in code (a default flip + a flag plumb in two files); the substrate replacement is fully built and
  tested. The real cost is wall-clock per `hear()` (a separate ~1800-neuron bridge, ~450 steps/fact), GPU-only.
- **Self-organized vs host-designed (the owner's hardware-port lesson):** with the flip, the association weights are
  LEARNED by Hebbian co-fire (self-organized from the fact stream) — a genuine criterion-2 improvement. The remaining
  nuance is that the learned graph is read into a dict and re-stamped onto a separate spread bridge; the fully-host-free
  end-state (one bridge, no dict hand-off, no per-call stamping) is close (c).
- **Moat:** the no-confab abstention moat is **independent of `elaborate`/the association graph** — `what_does`,
  `render_fact`, `query_*`, and `reason_chain` abstain on the cue-match/familiarity path, which the graph never touches.
  The de-risk asserts the moat is byte-unchanged. The moat is NOT weakened by any close here.
- **Downstream dependency:** `elaborate` feeds dialogue planning (the demos' "on-topic elaborate" line) and is reused by
  the multi-turn / multi-hop / multi-referent dialogue runners. The flip changes the SOURCE of the associations, not the
  selection mechanism or the API, so downstream consumers are unaffected aside from the parity-validated associate
  choice and the per-call wall-clock. No interaction with the FHRR bind, the cleanup, or the parser.

**Net:** A13 is a real but inexpensive criterion-2 close — a default-on flip of an already-built, multi-seed-validated,
no-`sim/`-edit substrate-learned association memory, behind a numpy-CPU escape — with the single-bridge fold as the
deeper fully-host-free follow-on.

---

## Sources verified (read-only this session)

- `research/findings/2026-06-21-shortcut-inventory-definitive.md` (A13 row + Tier-1 #6 + completeness note).
- `research/runners/brain_conversational_agent.py` (`_assoc_graph` 451, `elaborate` 471, ctor 229–233, `hear` store_fact
  321/344/371/407).
- `research/runners/learned_assoc_graph.py` (the full `LearnedAssocGraph`); `research/runners/_D_sparse_heteroassoc.py`
  (`build` n_pool=2000/1500, the learned recurrent).
- `research/runners/content_selection_spiking.py` (`SpikingSpreadingController` 278, `_install_graph_edges` 315,
  `turn_latency` 371); `research/runners/content_selection.py` (`SaidTrace` 58).
- `research/runners/rf_phasor_composer.py` (`_assoc_graph` 657, `elaborate` 669 — no learned branch);
  `research/runners/one_brain_composer.py` (`kb` only, no `elaborate`).
- `research/runners/consolidated_320_conversation_demo.py` (126, 190 — agent built without the flag);
  `research/runners/multi_turn_agent.py` (47–56 — flag not exposed); `research/runners/unified_brain_bridge.py` (84–119,
  293–294 — Step-3 still stamps).
- `tests/test_brain_conversational_agent.py` (`test_learned_assoc_graph_agent` 164, `test_dialogue_planning_elaborate`
  55, cache-invalidate 75).
- `research/findings/2026-06-05-cheat-D-associative-graph-research.md` (the conversion plan: Option A/B, §6 honest
  difficulty); `research/findings/2026-06-05-D-cue-recall-RESOLVED-sparse-heteroassoc.md` (the underlying RESOLVED
  heteroassociator).
