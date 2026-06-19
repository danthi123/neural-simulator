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

Two current-state headlines the diagrams now show: **(1)** navigation decides
which way to move by a **spiking race** (an evidence accumulator → an all-or-none
commit burst) — this is the **default**, and the old hand-coded `argmax`
read-out is retired (kept only as an opt-in oracle); **(2)** the conversational
diagram now foregrounds the **production pipeline** (`OneBrainComposer`) — a
learned-from-conversation 320-concept cortex, a sentence parser, a
resonate-and-fire phasor composer with a persistent fact store, and a
**no-confabulation safeguard** (the answer step says "I don't know" instead of
guessing) — above the earlier builder's region inventory.

The two supergroups are now (a) **consolidated** — navigation, the conversational
parser, the dlPFC, and the resonate-and-fire composer run as disjoint slices on
*one* `SimulationBridge` — and (b) joined by **three validated functional cross-brain
routes** (each 6/6-seed GO), all drawn on the master map:

- **(A) language → action** — the `command_route` transmission gate on a learned
  `language_input → cortex_{N,E,S,W}` route, opened by the conversational parser's
  firing, so a spoken instruction steers the navigation body (spoken-instruction
  navigation).
- **(B) perception → memory (recall)** — a perceived object's `cortex_it` ("what"
  stream) ensemble is engram-tagged in-episode, then neural reactivation reads it
  through a trained `cortex_it → language_output` route (`it_to_lang`), recalling
  what was seen (navigate-to-see-then-answer). This is RECALL, not composition.
- **(C) compose-perceived** — the LIVE `cortex_it` spiking *rate* of a perceived
  object is mapped by a fixed complex projection M into a unit *phasor* (⚠ host
  arithmetic on the substrate's own live rate), so the percept enters the composer's
  bind/bundle/unbind algebra — dissolving the rate-vs-phasor wall for perceived-object
  facts (navigate-to-compose-then-answer).

The master map also draws a co-resident, opt-in **generalization stack**
(`gen_perception` → `gen_concept` (NMDA) → `gen_fact`): a novel object perceived
through the Gabor/V1 front end is recognised by *category*, the path to generalize
across *similar* concepts on the point-neuron substrate (no dendritic rewrite needed).

These cross-brain routes and the generalization stack are added by specific runners
**on top of** a builder (`nav_conv_merged_bridge.py`, the `navigate_to_*` runners, the
`_genfrontier_*` de-risks), so they appear only on the master map; the two exhaustive
detail graphs stay scoped to a single builder's own region/pathway output.

| Diagram | What it shows | Files |
|---|---|---|
| **Master map** | Cluster-level overview, in plain language: the **one brain** (navigation + conversation as separate neuron groups on one network, one update loop), the main signal arteries, the navigation **spiking decision** (the move emerges from a spiking race — now the default, the host shortcut retired), the production **conversational pipeline** (learned word-meaning cortex → sentence parser → composer + fact store → recall & answer with the **no-confabulation safeguard** → reply), the **three** validated cross-brain routes (A a spoken word steers the body; B navigate to *see* then *recall*; C bind a *perceived* object into a fact), the co-resident generalization stack, the honesty legend. Start here. | [`brain_master.svg`](brain_master.svg) · [`.png`](brain_master.png) · [`.dot`](brain_master.dot) |
| **Navigation brain** | Exhaustive: every region + every distinct pathway of `build_bg_brain_regions()` — the basal-ganglia action-selection cascade, the spiking superior-colliculus orienting reflex (sc_retina→sc_map→cortex), the spiking actor-critic (reward_us → SNc; striosome value critic → SNc via GABA_B), thalamus/TRN, **the accumulate→commit decision layer — now the DEFAULT read-out (the move emerges from the spiking race; the host argmax is retired, kept as the opt-in oracle)**, cerebellum, hippocampus, dlPFC. | [`brain_navigation.svg`](brain_navigation.svg) · [`.png`](brain_navigation.png) · [`.dot`](brain_navigation.dot) |
| **Conversational brain** | **Two layers.** *Top* — the **production conversation pipeline** (`OneBrainComposer`, on the shared one brain): a learned-from-conversation 320-concept cortex → an on-bridge sentence parser (who-did-what; flexible word orders) → a resonate-and-fire phasor composer that binds words into facts (attributed objects, negation, embedded clauses; ~10–20× faster) + a persistent fact store → recall & answer with the no-confabulation safeguard → a word-ordered spiking reply + dialogue planning. *Below* — the underlying **region inventory** of `build_biological_brain_regions()` (a reference: language I/O, Wernicke pools, semantic cortex, Broca, concept pools, multimodal hub, hippocampal consolidation, dlPFC verb working memory; not all of it is on the production path). | [`brain_conversational.svg`](brain_conversational.svg) · [`.png`](brain_conversational.png) · [`.dot`](brain_conversational.dot) |

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
- **⌂** = the **host I/O boundary** — host-rendered sensory input (raw pixels: retina/image) or host-decoded motor output (which word/action the body emitted). This is **legitimate, not a shortcut**: under the brain-based-only standard the world is allowed to render the agent's input and the body to act on its output. (Distinguished from ⚠ with its own glyph precisely so a legitimate boundary is never read as a cheat.)
- **⚠** = a documented **shortcut** — a genuine reduction, of two sub-kinds: a *collapsed / phenomenological stand-in* (GPi+SNr merged, A9+VTA merged, per-action cortex labeling, a 250-cell cerebellar granule layer), or *host-computed cognition* (e.g. the place/goal **coordinate** codes handed to the navigator — the agent is given its position instead of perceiving it). Both are things the brain itself is meant to do.
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
  dot -Tsvg              "$f.dot" -o "$f.svg"   # SVG: viewers do per-glyph font fallback
  dot -Tpng:gdiplus -Gdpi=140 "$f.dot" -o "$f.png"   # Windows: gdiplus for symbol glyphs
done
```

**Font note (important).** The diagrams use box-drawing and symbol glyphs (⚠ ⊣ ━ ▶
⊟ ⟲ ◆ …) in `Helvetica`, which Arial-on-Windows lacks. The default Cairo PNG
renderer leaves them as empty boxes showing the code point (e.g. `26A0`, `2501`)
unless a Unicode-complete fallback font (e.g. *DejaVu Sans*) is available to
fontconfig. Two fixes: on **Windows**, render PNGs with `-Tpng:gdiplus` (GDI+ does
per-glyph font-linking, so only the missing symbols substitute and the Helvetica
body text is preserved — used above); on **Linux/macOS**, the default
`-Tpng`/Cairo works once *DejaVu Sans* is installed (it ships with most Graphviz
packages). The SVGs name `Helvetica` and rely on the viewer's own font fallback,
so they render correctly in browsers regardless.

The `.dot` sources are the source of truth; edit them and re-render. They are
authored from the as-implemented extraction spec (linked above), which in turn
was read directly from `sim/regions.py`, `research/runners/g11_bg_runner.py`,
`research/runners/text_minimal_isolation.py`, `sim/neuromodulators.py`,
`sim/kernels.py`, and `sim/config.py`.
