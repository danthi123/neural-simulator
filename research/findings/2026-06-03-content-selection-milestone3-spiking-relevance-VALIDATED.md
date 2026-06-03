# Content-selection Milestone 3 — spiking relevance (spreading activation) VALIDATED (6/6 multi-seed)

**Date:** 2026-06-03
**Status:** ✅ VALIDATED — `SpikingSpreadingController` coherence 6/6 seeds (12/12 conditions).
**Module:** `research/runners/content_selection_spiking.py` (`SpikingSpreadingController`)
**Builds on:** the seed-robustness resolution
(`2026-06-03-content-selection-milestone2-seed-robustness-RESOLVED.md`) — same clean-dynamics WM.

## One-line result

The PFC "Control" **relevance computation is now done in spikes**: the association graph is embodied
as inter-assembly synapses, driving the discourse context *spreads activation* to associated concept
assemblies, and the most-active candidate assembly **is** the selection. This is the faithful spiking
analogue of the structured relevance sum used in Milestones 1–2, and it is **coherent across all 6
seeds tested (42–47), both topics each = 12/12 conditions**.

## What changed vs Milestone 2

| | relevance computation | working memory |
|---|---|---|
| **M1 (structured)** | numpy `Σ_c context[c]·graph[c][cand]` | structured ContextBuffer |
| **M2 (spiking WM)** | numpy relevance | **spiking** loop-attractor WM |
| **M3 (this)** | **spiking** spreading activation | **spiking** loop-attractor WM |

In M3 the *whole* selection loop is spiking: spiking working memory holds the discourse context, and
spiking spreading activation through learned-style associative synapses computes the relevance. Only
inhibition-of-return remains structured (the `SaidTrace`); making it spiking (spike-frequency
adaptation on the selected assembly) is the documented Milestone-3b step.

## Mechanism

For each association A→B with strength `w`, install `cortex_A → dlpfc_B` synapses at weight `w·scale`
(`_install_graph_edges`). When concept A fires, it drives B's dlPFC assembly, whose within-concept
attractor then sustains B. Because **only designed associations get a synaptic path**, spreading stays
on the association graph and never leaks into unrelated concepts — clean by construction.

`turn(user_concepts)`: drive the context into the spiking WM → spreading activation lights associated
assemblies → read per-assembly firing (= spiking relevance) → winner-take-all over the unsaid
candidates → mark said. The clean-dynamics config (`internal_density=0`, `enable_ou=False`) from the M2
resolution keeps the multi-concept hold exact, so spreading is not corrupted by spurious states.

## Validation

Cheap-first probe (driving "apple"): apple's cluster lights up (big/cat/hot ≈ 0.32) while the unrelated
dog-cluster stays at **0.00** — the spreading reproduces the relevance ranking, seed-robustly (both
seeds, both edge scales tested).

Full controller, 6 seeds × 2 topics:

| seed | apple topic | dog topic |
|---|---|---|
| 42 | hot, cat, hot ✅ | cold, small, cold ✅ |
| 43 | big, cat, big ✅ | cold, small, cold ✅ |
| 44 | hot, big, hot ✅ | small, cold, small ✅ |
| 45 | cat, big, cat ✅ | river, cold, small ✅ |
| 46 | cat, big, cat ✅ | cold, small, cold ✅ |
| 47 | big, hot, big ✅ | cold, river, cold ✅ |

**12/12 conditions coherent** (every pick stays within the driven topic's cluster).

```bash
python -c "from research.runners.content_selection_spiking import SpikingSpreadingController; \
from research.runners.content_selection import build_association_graph; \
g=build_association_graph(['apple_big','apple_cat','dog_small','dog_river','cat_hot','river_cold','big_hot','small_cold']); \
cl={'apple':{'big','cat','hot'},'dog':{'small','river','cold'}}; \
[print(s, all(all(c in cl[t] for c in [SpikingSpreadingController(g,seed=s).turn([t]) for _ in range(3)]) for t in ['apple','dog'])) for s in [42,43,44,45,46,47]]"
```

## Honest scope

- **Repetition / inhibition-of-return is still structured:** with `SaidTrace` decay 0.6, picks can
  alternate (e.g. hot, cat, hot) — coherent (all in-cluster) but not strictly non-repeating across 3
  turns. The structured M1 controller gives clean non-repetition; making inhibition-of-return *spiking*
  (M3b) is the documented next step.
  - **M3b cheap-probe (2026-06-03): hyperpolarizing-fatigue approach REFUTED.** Applying targeted
    negative ("fatigue") current to a recently-selected, latched assembly to silence it for the next
    relevance read does **not** work: firing *increased* (hot 0.395 → 0.490 at amt=6000) instead of
    going silent. Cause = the `IZH2007_HIPPO_PYRAMIDAL` **rebound** dynamics (hyperpolarization-activated
    currents → rebound depolarization) — a latched hippocampal-pyramidal attractor cannot be silenced by
    hyperpolarizing it. So spiking inhibition-of-return needs a different mechanism. The principled path
    (pre-registered) connects to this project's validated **latency/rank-order coding** insight: read the
    *transient* spread (first-spike latency) rather than the sustained latch, so a fatigued (slower-to-
    respond) assembly loses the transient WTA race — a real read-path redesign, not a one-line fatigue.
  - **Latency read VALIDATED as a richer relevance signal (shipped: `relevance_by_latency`).** Reading
    first-spike latency of the spread encodes graph DISTANCE in spike timing, seed-robustly (3/3 seeds,
    fresh-bridge probe): driving "apple" -> direct associates big/cat earliest (~8-19 steps), the 2-hop
    hot later (~16-20), unrelated dog-cluster NEVER fires. Strictly more informative than the rate read
    (which gives all in-cluster roughly equal); the faithful spiking analogue of spreading-activation with
    distance = latency.
  - **Fully-spiking inhibition-of-return has THREE obstacles (honest open sub-problem):** (1) REBOUND —
    hyperpolarizing a latched assembly raises its firing, so a recently-selected concept can't be silenced
    by inhibition; (2) DIRECT-BEFORE-INDIRECT — latency ranks direct associates before indirect ones, and
    an indirect concept reached via a direct one can never out-race its own upstream, so latency-fatigue
    cycles among direct neighbours (big<->cat) and cannot reach a 2-hop concept (hot) by delay alone
    (full-cluster coverage still needs exclusion, the structured SaidTrace); (3) RESET — `_reset_wm`
    returns v/u/conductances/firing to rest but does NOT clear the in-flight synaptic delay buffers / slow
    NMDA state, so repeated latency probes on one bridge are contaminated (a 2nd apple probe fires faster;
    a dog probe lights the apple-cluster first). Net: the validated deliverable is M3 spiking relevance
    (rate read, 6/6) + structured SaidTrace inhibition-of-return, plus the latency read as a richer
    single-probe relevance; the fully-spiking multi-turn inhibition-of-return is a precisely-characterized
    open sub-problem, not a one-session drop-in.
- **Graph is installed, not learned:** the association synapses are set from the known graph (`w·scale`),
  not learned from experience — same scope caveat as the set (not learned) attractor weights.
- **Graph is installed, not learned (substrate):** the validation uses synthetic multi-cluster
  association graphs (each cluster a 4-cycle, no cross-cluster edges) at increasing size. Scaling to a
  larger *learned-association* substrate (a trained tagged engram bridge) is the richer-eval next step.

## Scaling (synthetic multi-cluster graphs)

The spiking content-selection scales past the original 8-concept toy:

All at the validated default `edge_scale=60` (strict criterion: all 3 turns non-`None` *and* in-cluster):

| concepts | clusters | result | note |
|---|---|---|---|
| 8 | 2 | **12/12** (6 seeds × 2) | the headline base case |
| 16 | 4 | **12/12** (3 seeds × 4) | strict — clean after the `edge_scale` fix below |
| 24 | 6 | **12/12** (2 seeds × 6) | strict — clean at `edge_scale=60` |

**Clean strict at every scale (8 → 16 → 24 concepts, all 12/12 conditions).**

**The `edge_scale` fix (default 20 → 60).** At the original `edge_scale=20`, 16-concept was 11/12
strict — one within-cluster `None` (seed 42, apple). Diagnosed: a *designed* associate failed to latch
(apple lit only `pear`; `plum`/`grape` stayed at 0.0), a seed-dependent sub-threshold spread-failure (the
inverse of the M2 spurious-state issue). Fix: stronger spread — `edge_scale=60` lights *every* designed
associate (apple → pear 0.33, plum 0.30, grape 0.32) with **no off-topic risk** (there are no
cross-cluster edges, so stronger spread stays strictly in-cluster). Re-validated: 8-concept 6/6 (no
regression) and 16-concept 12/12 strict. So both failure directions are now handled: spurious states
(M2: clean dynamics) and missed associates (M3: sufficient spread).

**The load-bearing coherence property — never picking an off-topic concept — holds at every scale
tested**, and with `edge_scale=60` the within-cluster `None` is eliminated too. The spiking
content-selection's *topic discipline* is robust to 3× the original vocabulary.

The bridge grows with the vocabulary (`n = max(600, 60·V)` neurons per region, one 50×50 attractor +
the association edges per concept), so larger graphs are a GPU-scale concern; the synthetic-cluster
sweep maps where coherence holds before investing in a large learned substrate.

## Connected (realistic) graphs — the latency mode (`turn_latency`), and M3b obstacle 3 resolved

The clean-cluster sweeps above use graphs with **no cross-cluster edges**, so spreading stays local. On a
**richly-connected** graph (the M1 eval's 27-node multi-topic web: weather/animal/fruit/music with
cross-links), the default rate-read `turn()` **over-spreads**: activation diffuses *multi-hop* through the
connected graph and loses topic focus (e.g. `rain → storm, dog, tree`). The structured M1 relevance is
1-hop weighted, so it stays focused; the spiking *rate* read is effectively multi-hop diffusion.

**The latency read is the focused, faithful fix.** Direct (1-hop) neighbours fire *first*; distant
concepts fire later (multi-hop latency). Reading first-spike latency therefore recovers the 1-hop focus:
on the connected graph the earliest-latency pick is a **direct neighbour for 6/6 topics**. Shipped as
`SpikingSpreadingController.turn_latency()` (relevance = first-spike latency; inhibition-of-return = the
structured SaidTrace). Validated:

| graph | `turn()` (rate) | `turn_latency()` |
|---|---|---|
| clean clusters (8c) | 4/4 in-cluster | 4/4 in-cluster (no regression) |
| connected web (6 topics × 3 turns) | over-spreads off-topic | **18/18 chains within the 2-hop topic region (3 seeds × 6 topics)** |

**M3b obstacle 3 (clean inter-probe reset) is substantially resolved.** Multi-turn latency selection
*first* drifted off-topic on turns 2-3 because the best-effort `_reset_wm` left in-flight state. Diagnosed
the missing arrays — `cp_prev_firing_states`, `cp_refractory_timers`, and the synaptic
`cp_synapse_pulse_timers`/`cp_synapse_pulse_progress` (delayed transmission carried between probes) — and
cleared them in a fuller `_reset_wm`. With the fuller reset, repeated latency probes are clean and the
connected-graph chains stay on-topic 6/6. (Obstacles 1 *rebound* and 2 *direct-before-indirect* are
sidestepped by `turn_latency`: it uses the SaidTrace for inhibition-of-return rather than silencing a
latched assembly, and the SaidTrace exclusion lets the chain move past direct neighbours.) What remains of
"fully spiking" is making the SaidTrace itself spiking; the **relevance + working memory + inter-probe
reset are now all spiking and robust on realistic connected graphs.**

## Demonstration — a conversation on the faithful spiking Control

The validated spiking Control is wired into the interactive `DialogueAgent` (dependency injection:
`DialogueAgent(graph, controller=SpikingSpreadingController(graph))`, or `dialogue_agent.py --repl
--spiking`). A scripted multi-turn conversation, *every elaboration computed by spreading spikes through
the spiking working memory* (seed 42, 8-concept graph):

```
user : apple        agent: hot          (apple's cluster: big/cat/hot)
user : more         agent: cat
user : more         agent: hot
user : is apple related to big?   agent: Yes -- apple and big are associated (strength 1.0)
user : dog          agent: small        (clean topic shift -> dog's cluster: small/river/cold)
user : more         agent: river
user : more         agent: small
```

**Topic-shift contamination, caught and fixed.** Without intervention the shift to "dog" resurfaced
"apple" (the prior topic's assemblies stay *latched* in the persistent spiking WM and bleed into the new
topic). Fix: on an explicit topic shift the agent calls the spiking controller's `_reset_wm()` (clears
v/u/conductances/firing) before refocusing — the disjoint new-topic spread then dominates. With the reset,
the shift is clean (dog → small/river/small, all in-cluster). The structured `DialogueAgent` default
handles shifts via its decaying context buffer; the spiking backend handles them via reset-on-shift.
(This is the *benign* face of the same persistent-latch property whose *full* clearing — for fully-spiking
inhibition-of-return — remains the M3b open sub-problem: best-effort `_reset_wm` suffices for a
disjoint-topic switch but not for within-topic per-turn suppression.)

## Bottom line

The faithful brain-analogue content-selection Control is now demonstrated **end-to-end in spikes**:
spiking cortico-PFC loop-attractor working memory holds the discourse context, and spiking spreading
activation through associative synapses computes relevance — coherent and seed-robust (6/6). Combined
with the M2 seed-robustness resolution, the spiking conversation substrate (what to hold + what to say
next) is validated at full spiking fidelity for the selection, with inhibition-of-return and learned
associations as the clearly-scoped remaining faithfulness steps.
