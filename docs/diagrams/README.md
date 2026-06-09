# Simulated-brain architecture diagrams

Honest, **as-implemented** flowcharts of the whole simulated brain — every
region type, every distinct pathway, the direction and *nature* of each
signal, and an explicit faithful-vs-shortcut layer. Generated with
[Graphviz](https://graphviz.org/) from the source `.dot` files in this folder.

The brain is not one fixed network — it is **assembled per configuration** from
a declarative region/pathway grammar (`sim/regions.py`). Two builder functions
assemble almost the entire architecture, gated by opt-in flags. The diagrams
reflect the **maximal** configuration (every flag on); a given run builds a
subset.

| Diagram | What it shows | Files |
|---|---|---|
| **Master map** | Cluster-level overview: the 12 subsystems, the main signal arteries, the two config-scoped supergroups on one engine, the honesty legend. Start here. | [`brain_master.svg`](brain_master.svg) · [`.png`](brain_master.png) · [`.dot`](brain_master.dot) |
| **Navigation brain** | Exhaustive: every region + every distinct pathway of `build_bg_brain_regions()` — the basal-ganglia action-selection cascade, spiking-SNc actor-critic + neural value critic, thalamus/TRN, superior colliculus, cerebellum, hippocampus, dlPFC. | [`brain_navigation.svg`](brain_navigation.svg) · [`.png`](brain_navigation.png) · [`.dot`](brain_navigation.dot) |
| **Conversational brain** | Exhaustive: every region + every distinct pathway of `build_biological_brain_regions()` — language I/O, Wernicke pools, semantic cortex, Broca, concept pools, multimodal hub, hippocampal consolidation, dlPFC verb working memory. | [`brain_conversational.svg`](brain_conversational.svg) · [`.png`](brain_conversational.png) · [`.dot`](brain_conversational.dot) |

> The two detail graphs are **dense by design** — the goal was every distinct
> pathway drawn separately, not bundled. Open the **SVG** to zoom; the master
> map is the readable overview.

## How to read the diagrams

**Per-action / per-concept pools are collapsed.** The navigation cascade is a
`×4` template over the four actions (N/E/S/W); concept pools are `×4 names` per
kind. Each is drawn **once** with a `×N` badge; cross-action arcs are shown once
and labelled (e.g. "cross-action WTA (X→Y≠X)").

### Signal nature (the primary channel — arrow points downstream)

| Glyph | Meaning |
|---|---|
| `──▶` (black) | **excitatory** — glutamate (AMPA, with NMDA sharing the same presynaptic event) |
| <code>──⊣</code> (dark red, tee head) | **fast inhibition** — GABA_A, chloride (E ≈ −75 mV; striatal MSN/SNc overrides −60/−55) |
| <code>- -⊣</code> (purple, dashed tee) | **slow inhibition** — GABA_B / GIRK potassium (E_K = −90 mV). One signature edge: `striosome_value → snc`, the value subtraction r − V |
| `──▶` (green) | **NMDA-recurrent** self-excitation — slow integration / working-memory bistability (`⟲` self-loops) |
| `····◆` (gold, dotted) | **neuromodulator broadcast** — dopamine. A concentration scalar, *not* a synapse |

### Node / edge status

- **Solid border** = always built (the CORE). **Thick border** = an inhibitory region (low excitatory fraction).
- **Dashed border / edge** = opt-in (flag-gated), or a weaker / feedback pathway.
- **⊟** on an edge = a runtime gate (transmission or plasticity); the gate name is in the label.
- **⚠** = a documented **shortcut**. Two kinds, distinguished:
  - *host-rendered input* (sensory codes drawn by the environment) — **legitimate**: the world renders the agent's input, the body acts on its output.
  - *collapsed / phenomenological stand-in* (e.g. GPi+SNr merged, A9+VTA merged, per-action cortex labeling, a 250-cell cerebellar granule layer) — a genuine reduction.
- **✗ gray dashed** (conversational graph) = a composition pathway that is present **structurally** but is a **NEGATIVE / BOUNDARY** result (v12/v15/v16/v18) — wired, but **not a working capability**.

### Substrate-wide shortcuts (apply almost everywhere — annotated once)

Point neurons (no dendrites) · uniform one-step conduction delay · GABA_A by
default (GABA_B only on the one opt-in edge) · neuromodulators as global scalars
· Izhikevich point-neuron default (HH/AdEx available) · forward-Euler
integration. The full list (SH-1…SH-14) and per-node/edge shortcuts are in the
extraction spec:
[`research/findings/2026-06-09-brain-architecture-flowchart-spec.md`](../../research/findings/2026-06-09-brain-architecture-flowchart-spec.md).

The **BRAIN-BASED-ONLY** standard (owner directive, `CLAUDE.md`): even where a
host-side computation is biologically correct, it is a shortcut if the *brain*
isn't doing it. Host code is legitimate only for the **environment** (world
state + sensory rendering) and the **body** (acting on motor output); everything
between sensation and action is meant to be neurons and synapses. The diagrams
mark host-computed cognition distinctly from host-rendered environment/body.

## Scientific accuracy caveat

These diagrams are accurate **to the degree the biology is implemented in this
simulator** — not to the degree real brains are organized. They are an honest
map of the code, including its reductions, not a textbook of neuroanatomy.

## Regenerating

Requires Graphviz (`dot`). On Windows: `winget install Graphviz.Graphviz`.

```bash
cd docs/diagrams
for f in brain_master brain_navigation brain_conversational; do
  dot -Tsvg          "$f.dot" -o "$f.svg"
  dot -Tpng -Gdpi=140 "$f.dot" -o "$f.png"
done
```

The `.dot` sources are the source of truth; edit them and re-render. They are
authored from the as-implemented extraction spec (linked above), which in turn
was read directly from `sim/regions.py`, `research/runners/g11_bg_runner.py`,
`research/runners/text_minimal_isolation.py`, `sim/neuromodulators.py`,
`sim/kernels.py`, and `sim/config.py`.
