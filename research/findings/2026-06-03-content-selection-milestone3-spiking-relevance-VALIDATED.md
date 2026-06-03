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
- **Graph is installed, not learned:** the association synapses are set from the known graph (`w·scale`),
  not learned from experience — same scope caveat as the set (not learned) attractor weights.
- **Small toy substrate:** an 8-concept association graph. Scaling to a larger learned-association
  substrate (a trained tagged bridge) is the richer-eval next step.

## Bottom line

The faithful brain-analogue content-selection Control is now demonstrated **end-to-end in spikes**:
spiking cortico-PFC loop-attractor working memory holds the discourse context, and spiking spreading
activation through associative synapses computes relevance — coherent and seed-robust (6/6). Combined
with the M2 seed-robustness resolution, the spiking conversation substrate (what to hold + what to say
next) is validated at full spiking fidelity for the selection, with inhibition-of-return and learned
associations as the clearly-scoped remaining faithfulness steps.
