# OpenWorm — what they do, what we can learn

**Date:** 2026-05-04
**Source:** openworm.org + linked subprojects (live fetch)
**TL;DR:** OpenWorm is a 12-year community project simulating one *C. elegans* (302 neurons, known connectome) at compartmental-HH fidelity. They're not a learning-systems project — they're a fidelity / reproducibility project. The two things genuinely worth borrowing are (1) the **BioParameter `(name, value, source, certainty)` provenance pattern** from c302 to retrofit our 375-entry feature catalog, and (2) **NeuroML2 export** of network topology so our Izhikevich/HH/AdEx regions become loadable in their tool ecosystem (NEURON, Brian, NetPyNE) for cross-validation. Their cross-engine validation framework (OMV) is the right model for our spike-timing regression tests.

## Project at a glance

OpenWorm has been running since ~2011 as a volunteer open-source community building a virtual *C. elegans* — 302 neurons, ~7000 chemical + electrical synapses, full connectome from Sulston/White. Stack is Python + Java + C++/OpenCL, MIT licensed across ~40 GitHub repos under `github.com/openworm` [1, 2]. The flagship Docker stack ([1]) wires nervous-system simulation (c302 → NEURON/jNeuroML) into a body-physics simulator (Sibernetic, PCISPH on OpenCL [3]) and outputs locomotion videos. They have ~7 active repos with recent commits, a Slack community, four 2018 papers in Phil. Trans. R. Soc. B that are the canonical references [4]. Latest c302 release v0.12.0 was March 2026 [5].

Scale: **1 worm, 302 cells, no learning, behavioral validation against worm-tracker databases**. Our project: **~5000 Izhikevich neurons, 50+ regions, learning + RL, no ground-truth connectome**. The overlap is biological parameter discipline + reproducibility infrastructure, not architecture or scientific question.

## Architecture

**Simulation engines (multi-backend by design):**
- c302 generates `.nml` (NeuroML 2 XML) + LEMS simulation specs, then dispatches to a backend simulator [5, 6]
- Backends: jNeuroML (Java), pyNeuroML, NEURON, Brian, NetPyNE, MOOSE — same model file runs on all [7]
- Sibernetic (separate process) reads voltages from c302 over a Python bridge, drives muscle activation in PCISPH fluid [3]

**Neuron models supported in NeuroML2 schema [8]:**
- `<iafCell>` (integrate-and-fire — c302 parameter sets A, B, C use this)
- `<izhikevich2007Cell>` (9-param Izhikevich — what *we* use)
- `<adExIaFCell>` (Brette-Gerstner AdEx — what *we* also use)
- `<fitzHughNagumoCell>` (FHN)
- `<ionChannelHH>` + `<gateHHrates>` (compartmental HH — c302 parameter set D)

**Synapse models in NeuroML2:** `<alphaSynapse>`, `<expTwoSynapse>`, `<blockingPlasticSynapse>`, `<doubleSynapse>` [8]. Plasticity rules are *not* a first-class NeuroML primitive — STDP / Hebbian require LEMS extensions.

**Data formats:**
- `.nml` — NeuroML 2 XML for cells + topology
- LEMS XML for simulation parameters
- WCON / Tracker-Commons (worm-tracker JSON) for behavioral data [9]
- VTK for body-physics output (Paraview) [3]
- They explicitly *don't* use NWB (Neurodata Without Borders), HDF5 checkpoints, or numpy `.npz` — heavier on standards-track XML, lighter on binary [10]

**Visualization:** Geppetto — Java OSGi backend (Spring + Maven + Eclipse Virgo) + THREE.js/WebGL front-end + WebSocket JSON messaging [11]. Heavy stack. Our DearPyGUI + raw OpenGL is far simpler.

## Biology approach

**Fidelity:** Compartmental Hodgkin-Huxley is the target. They discretize each neuron into multiple compartments, preserving 3D position from anatomical reconstruction [12]. NeuroML maps directly to HH ODEs. Most of c302 today actually runs simpler `iafCell` integrate-and-fire (parameter sets A/B/C) because the HH parameter sets aren't fully tuned [5]. So in practice it's *aspirationally* HH, *operationally* IaF for most runs.

**Neurotransmitters / channels:** The **ChannelWorm2** subproject hand-curates ion channel models from electrophysiology papers. Glutamate, GABA, ACh are represented at the channel/conductance level, not as a declared neuromodulator subsystem like our `sim/neuromodulators.py`.

**Learning rules:** **None.** OpenWorm models a fully-developed adult worm with fixed weights. Plasticity, STDP, reward, eligibility traces, dopamine — *all absent*. Their "Optimization Engine" subproject [13] uses genetic algorithms (HeuristicWorm, Bionet, both C++) to *fit* synaptic weights to electrophysiology, but at design-time, not as runtime plasticity. **This is the single largest scope difference with us.**

**Connectome:** Sulston/White (1986) C. elegans connectome — ~6400 chemical synapses, ~900 gap junctions, fully enumerated. Stored in `owmeta` with literature provenance [14]. Our cortex has no equivalent ground truth, which is why we use density / motif / WS connectivity generators in `sim/connectivity.py`.

**The critical bioparameter pattern.** c302's `BioParameter` class is a small but meaningful primitive [15]:

```python
BioParameter(name, value, source, certainty)
# e.g. ("neuron_leak_cond_density", "0.005 mS_per_cm2", "BlindGuess", "0.1")
# or   ("chem_exc_syn_decay", "5 ms", "BlindGuess", "0.1")
```

Every parameter carries a `source` (paper citation or `"BlindGuess"`) and a `certainty` ∈ [0,1]. This is the right discipline for our `references/feature-catalog.md` 375 entries — currently we cite, but we don't tag certainty consistently per parameter, and our runner kwargs don't propagate that metadata.

## Tooling + reproducibility

**CI:** GitHub Actions matrix [16]. c302 tests `[python 3.10, 3.13] × [jNeuroML, jNeuroML_NEURON, jNeuroML_validate]` on `ubuntu-latest`, with `fail-fast: false`. Our test suite is broader (40 files) but single-engine. Their multi-engine matrix catches simulator-specific bugs; ours can't.

**OMV (OSB Model Validation):** [17] cross-simulator regression framework. You write a `.mep` file declaring "expected spike times" + a `.omt` file (YAML) saying "run this LEMS sim on engine X, compare voltage trace at neuron Y". OMV runs the sim and compares against tolerances. Same model file, multiple engines, identical expected output. **This is the missing piece in our test suite** — we have unit tests on kernels and findings docs on full runs, but no regression tests that pin "AVAL fires at t=12.3ms ± 0.5" across versions.

**Parameter sweeps:** c302 uses a `paramoverride` CLI flag and `parameters_*.py` files. Crude — modify Python file, reinstall package, regenerate. No declarative sweep framework. Our `research/experiment_runner.py` (YAML-driven, with `result_aggregator.py`) is significantly more sophisticated than anything in their stack.

**Multi-seed:** Not addressed in any OpenWorm docs I could find. Their optimization engine uses GA populations, but the documentation [13] doesn't mention seed management or n-seed validation. Our 6-seed validation discipline (per `MEMORY.md`) is more rigorous than what's documented on their side.

**Docker stack:** They ship a single Dockerfile [1] that pins versions of c302, Sibernetic, NEURON, jNeuroML — runs anywhere with Docker. We have no equivalent; reproducing our work currently requires `pip install -r requirements.txt` + a CUDA GPU, which is not portable.

**Findings format:** **None visible.** Their `docs.openworm.org` has no findings/results/experiments section. Negative results are not documented anywhere I could locate [10]. Our `research/findings/` (93+ docs, both positive and negative) is markedly better-organized than their public research record. They use GitHub issues + Slack for ongoing discussion; results live in publications (4 papers in 2018, sparse since [4]).

**owmeta — data + provenance layer:** [14, 18] Python API over RDF/SPARQL that links biological claims to literature. Pattern:

```python
e = evctx(Evidence)(key="Sulston83", reference=doc)
avdl = dctx(Neuron)(name="AVDL")
avdl.lineageName("AB alaaapalr")
e.supports(dctx.rdf_object)
```

Heavy infrastructure (RDF triple store + Prolog rules — codebase is 37% Prolog [18]). Last release v0.12.2 was October 2020 — appears semi-dormant. The *idea* (versioned, provenance-tagged biological claims) is right; the *implementation* (RDF + Prolog) is way too heavy for us to adopt.

## What we could borrow (recommendations)

Prioritized by effort × payoff. Concrete enough to file as tasks.

**1. Adopt the `BioParameter(name, value, source, certainty)` discipline.** [HIGH payoff, MEDIUM effort]
Retrofit `references/feature-catalog.md` so each of the 375+ entries has explicit `source` (Kandel ch.X / Pulvermuller-2001 / etc) and `certainty` ∈ {high, medium, low, blindguess}. Tag config defaults (`stdp_a_plus = 0.012` etc) with the source they came from in a sidecar table. When we set runner CLI kwargs, log the certainty alongside. Right now `MEMORY.md` flags "BlindGuess" implicitly via `# tuned, no biology source` comments — this would make it auditable. Direct port of c302's `bioparameters.py` pattern [15].

**2. Add a NeuroML2 exporter for our region topology.** [HIGH payoff, MEDIUM-HIGH effort]
Write `sim/neuroml_export.py` that emits one `.nml` per region, plus a top-level network XML stitching pathways. NeuroML2 natively supports `izhikevich2007Cell` and `adExIaFCell` [8] — we wouldn't need to translate models, just dump dataclass fields into XML elements. Payoff: anyone with NEURON or Brian or NetPyNE can run our networks, and we get free cross-validation against established simulators. Limitation: NeuroML2 has no first-class STDP, so our plasticity stays opaque to importers — but that's fine for forward dynamics tests. Use `pyNeuroML` to validate exports against the v2.3 XSD schema [7].

**3. Adopt OMV-style cross-engine spike-timing regression tests.** [MEDIUM payoff, MEDIUM effort]
Build `tests/test_spike_regression.py` that pins specific neuron firing times for a few canonical configs (G v2.5 + K v2 at seed 42, BG cascade probe at seed 42, etc) — like OMV's `.mep` files [17]. Run on every commit. Catches numerical drift bugs like the 2026-04-25 HH per-gate Q10 fix that silently changed all HH dynamics. Doesn't need cross-simulator support; just self-consistency over time. Tolerances: ±1 ms per spike, ±5% on rate over 1s window.

**4. Borrow OpenWorm's matrix CI.** [LOW payoff, LOW effort]
Add Python-version × CuPy-version matrix to our GitHub Actions. They test `[3.10, 3.13]`. We currently pin a single pair. Modest win but cheap.

**5. Ship a reference Docker image.** [MEDIUM payoff, MEDIUM-HIGH effort]
Wrap our stack in a CUDA-enabled Dockerfile pinning Python, CuPy, dearpygui, OpenGL drivers. Their precedent [1] shows it's doable for a multi-language stack. Reproducibility win for the inevitable grad student / collaborator who wants to rerun a finding from `research/findings/`. Ours would be CUDA-only (theirs is CPU-runnable via OpenCL), so smaller audience, but still valuable.

**6. Steal `c302`'s "parameter sets A/B/C/D as named tiers" naming.** [LOW payoff, LOW effort]
Rather than ad-hoc CLI flags accreting (`--enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-cluster-a-closed-loop ...`), name canonical configurations: `flagship-A` = current G v2.5+K v2, `flagship-B` = D-stack, `cheats-allowed` = old 4.41 config. Keep the underlying flags but expose named presets as `--config-tier`. Less typing, easier reproduction in findings docs. Doesn't replace the flag system, just sits above it.

## What we should NOT borrow

- **Do NOT adopt NeuroML as our primary internal format.** XML round-tripping our 50+ region configs every step would crater performance. Use NeuroML for export only, internal stays CuPy + dataclasses.
- **Do NOT adopt Geppetto for visualization.** Java + OSGi + WebSocket + THREE.js is enormous overkill for our needs and the wrong language stack. Our DearPyGUI + raw OpenGL viewer is good enough; don't migrate.
- **Do NOT adopt owmeta's RDF/Prolog stack.** The `(name, value, source, certainty)` *idea* is great; the RDF triple store + Prolog reasoner is wildly disproportionate for our scale. A flat JSON sidecar with the bioparameter tuples is sufficient.
- **Do NOT adopt LEMS for simulation specs.** LEMS XML is meant for declarative ODE descriptions in jLEMS — useful when you don't have a simulator. We have a simulator. Sticking with Python configs is correct.
- **Do NOT switch from CuPy/CUDA to OpenCL.** Sibernetic uses OpenCL for portability, which costs them ~2× perf vs native CUDA. Our 7-8× speedup stack relies on CUDA-specific features (CUBLAS deterministic paths, fused kernels, masked updates). No reason to give that up.
- **Do NOT mirror their findings format.** They essentially don't have one. Our `research/findings/*.md` discipline (positives *and* negatives, dated, seeded, reproducible commands) is genuinely better than what they publish. Keep it.
- **Do NOT adopt jNeuroML / NEURON as the primary simulator.** Single-CPU NEURON would lose ~50× performance against our CuPy. They use it because their network is 302 neurons; ours is 5000 with plasticity at every synapse.

## Collaboration paths

- **Slack:** invitation requires signup as contributor at openworm.org/contacts.html [19]. Channels include General, Geppetto, Movement Validation, Muscle model, ChannelWorm.
- **Mailing list:** They've moved off mailing lists onto Slack [19]. info@openworm.org for general inquiries.
- **GitHub issues** on relevant repos (`openworm/c302`, `openworm/owmeta`) are active and triaged. Adding NeuroML2 support to our stack would naturally produce upstream-able test models.
- **Most useful contacts:** Padraig Gleeson (c302 + NeuroML), Matteo Cantarelli (Geppetto + co-founder). Both authored the canonical 2018 papers [4].
- **Realistic collaboration angle:** if we publish G v2.5 + K v2 results, the natural connection is "GPU-accelerated cortical learning system, exporter compatible with NeuroML" — gives them a data point on a non-worm system using their format. We're unlikely to contribute connectome data; we *can* contribute Izhikevich-2007 + AdEx parameter set examples optimized for navigation tasks.
- **Key papers to cite if we publish:** Sarma et al. 2018 (overview), Gleeson et al. 2018 (c302), Cantarelli et al. 2018 (Geppetto), Szigeti et al. 2014 (founding paper) [4].

## Sources

1. https://github.com/openworm/OpenWorm — main Docker stack repo
2. https://github.com/openworm — GitHub org listing
3. https://github.com/openworm/sibernetic — PCISPH body simulator
4. https://openworm.org/publications.html — canonical publication list (Phil. Trans. R. Soc. B 2018, Frontiers 2014)
5. https://github.com/openworm/c302 — c302 framework repo + recent v0.12.0 release
6. https://docs.openworm.org/Projects/c302/ — c302 docs page
7. https://docs.neuroml.org/Userdocs/Software/Software.html — NeuroML tool ecosystem
8. https://docs.neuroml.org/Userdocs/NeuroMLv2.html — NeuroML2 schema (cell types, synapse types)
9. https://docs.openworm.org/Projects/worm-movement/ — WCON / Tracker-Commons / open-worm-analysis-toolbox
10. https://docs.openworm.org/ — top-level docs (no findings section)
11. https://docs.openworm.org/Projects/geppetto/ — Geppetto Java + WebGL stack
12. https://docs.openworm.org/modeling/ — compartmental HH approach
13. https://docs.openworm.org/Projects/optimization/ — HeuristicWorm / Bionet GA optimizers
14. https://github.com/openworm/owmeta — owmeta data + provenance layer
15. https://raw.githubusercontent.com/openworm/c302/master/c302/bioparameters.py — `BioParameter(name, value, source, certainty)` class
16. https://github.com/openworm/c302/blob/master/.github/workflows/ci.yml — c302 CI matrix
17. https://github.com/OpenSourceBrain/osb-model-validation — OMV cross-engine validation framework
18. https://owmeta.readthedocs.io — owmeta docs
19. https://docs.openworm.org/community/ — contribution / Slack signup
20. https://docs.openworm.org/Projects/ — full subproject inventory (12 projects)
