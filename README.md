# Neural Simulator

**A GPU-accelerated simulator for biologically realistic spiking neural networks —
and a research program building one brain out of them.**

Networks are made of model neurons that communicate with discrete pulses ("spikes")
over time, the way real neurons do. Learning is local: connections change from
spike timing (Hebb 1949) and from a dopamine-like reward signal (Schultz 1998).
There is no backpropagation, no supervised label, no symbolic optimizer. It runs on
CUDA via CuPy, or on any CPU via NumPy, with an optional real-time 3D view.

**The north-star:** a *single* simulated brain that navigates a world **and**
converses genuinely — reasoning to its own conclusions, with an emotionally
coloured world-model and a working sense of what it does and does not know. The
long-range wager is the emergentist one: mind emerges from emulating a brain
completely and faithfully enough. So progress is measured by the faithfulness of
the biology, not by a benchmark score. Everything between sensing and acting must
be neurons and synapses; ordinary code is allowed only for the world and the body.

**One caveat governs every claim below.** Where this project reports "emotion" or
"self-awareness", it means a *measured functional correlate*. It never claims the
brain feels anything.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Backend](https://img.shields.io/badge/backend-CuPy%20(CUDA)%20%2F%20NumPy%20(CPU)-orange.svg)
![Status](https://img.shields.io/badge/status-active%20research-yellow.svg)

**Jump to:** [How to read a status here](#how-to-read-a-status-here) ·
[What works today](#what-works-today) ·
[What is not solved](#what-is-not-solved) ·
[Current frontier](#current-frontier) ·
[How to run it](#how-to-run-it) ·
[Architecture](#architecture) ·
[Where to look for detail](#where-to-look-for-detail)

---

## How to read a status here

This project's single largest cost is re-deriving work already done, so results are
kept with an explicit status rather than a headline. Every classified finding in
`research/findings/` declares one in its frontmatter:

| status | meaning |
|---|---|
| **live** | the central claim stands as written |
| **contributing** | real, but it feeds a larger result rather than standing alone |
| **qualified** | holds only inside a stated scope or operating point |
| **corrected** | stands after a correction was applied to the claim |
| **superseded** | a later result replaced it |
| **retracted** | the central claim died |

Measured now (`grep -h '^status:' research/findings/*.md | sort | uniq -c`):

```
93 live   87 contributing   53 qualified   27 corrected   16 superseded   7 retracted
```

**190 of the 283 classified findings — 67% — carry a qualifier rather than plain
`live`.** The remaining 1,562 findings are unclassified and their status is
unmeasured. Read the table below with that in mind: *nothing here is a finished
feature*, and several once-headline results have been withdrawn by this project's
own controls.

---

## What works today

| Capability | Honest status |
|---|---|
| **The engine** — spiking dynamics, region/pathway framework, plasticity rules, checkpointing, 3D view, GPU **and** CPU backends | **Mature.** 43 modules / 22,759 lines; 478 test modules; 47 region profiles. A 2026-07-31 sweep fixed 10 confirmed defects, two long-standing: spatial networks above 15,000 neurons raised `NameError` and had never run, and `set_plasticity_gate()` addressed the wrong synapses after a connectivity rebuild. Regression 53/53 on CuPy afterwards. |
| **Vision-based navigation to a moving goal** | **Works; the popular description of it was withdrawn.** The shortcut-closed configuration — no goal coordinates, no agent coordinates, no hand-coded heuristic, no distance-based reward — scores 4.08 ± 0.49 over 6 seeds (p=0.00045, 30.6% over baseline). The more-quoted 2.97 / 2.57 "visual cortex only" figures were measured with that heuristic **at full strength**; the description is retracted (2026-07-16) and the visual pathway's own contribution is unquantified. The move decision is a spiking accumulator racing to a commit burst, default in the library, at 1.16× the host-argmax baseline. ⚠️ the plasticity-gate defect above is live in this runner, so "frozen pathway" claims made through it are suspect. |
| **Grounded question-answering, on the spiking substrate** | **Works inside a bounded scope.** Parses active and passive sentences into who-did-what-to-whom, stores facts, answers who/what and yes/no including negation, and inherits properties across categories with exceptions — those at 6 seeds. Transitive inference is 6-seed off-brain but 3-seed on spikes; multi-hop chaining is 3-seed and rides the fixed algebra; multi-turn pronoun tracking is 3-seed. Those three sit below this project's own 6-seed bar. |
| **Abstention instead of fabrication** | **Works, at 3 seeds.** The learned familiarity gate agrees with the host abstention decision on 168/168 cues across 3 seeds, with zero breaches, and the separation collapses when its learned weights are cut. The fluent generator is never called when the brain abstains. 3 seeds, not 6. |
| **Continual word–concept memory** | **Works at a few-hundred concepts.** Distinct codes recall at 1.000, any-bank 0.992 at 6 seeds, with old facts intact while new ones are learned. |
| **Categories and meaning learned from experience** | **Works, 6 seeds.** Word meanings from co-occurrence; categories and simple taxonomies discovered unsupervised; category structure that also emerges from *seeing* objects (a pixel-scramble control collapses it). Recalling a fact about a *newly seen* object still routes through a hybrid path, not one all-spiking path. |
| **Emotional colouring of concepts** | **Split, and one summary line was corrected.** Off-brain concept tagging holds at 6 seeds (held-out r = +0.811). The on-brain affect-state region is **qualified**: its own artifact reads `"GO": false` at 2 of 6 seeds, and a board line calling it a 6-seed GO was corrected on 2026-07-31. Its mood is a measured **ratchet** — it rises on 3/3 seeds and never comes back down. |
| **Self-model read-out** | **6 seeds, off-bridge, on the NumPy backend.** A small region reads and reports the brain's own attention, confidence, and whether a thought was its own; cutting its access to the real internal state collapses the reports. Not yet running inside the develop-loop, and not yet reproduced on GPU. |
| **Curiosity from uncertainty** | **6/6 on-bridge, with every anti-cheat control collapsing** — it asks about what it can learn and declines to chase unlearnable noise. The finding carries a `qualified` status. |
| **Reading replay *direction* in spikes** | **Arc complete; one headline withdrawn the same month.** Single-trial order accuracy 0.969 (chance 0.500) on GPU, 6 seeds × 16 trials, with the relay-bypass control at chance. Separately, the place-field headline tuned to maximise `circ_dW` turned out to measure how *concentrated* weight increments are, not *where* — position-shuffling moved it 1.3%, p=0.42 — and is withdrawn. The distinct field-quality configuration survived a stricter position-only null at 5.05× (p=0.0025, 6/6), above the ideal-field oracle's 4.53×, and stands. |
| **Fluent English phrasing** | **A temporary external scaffold — and the project's biggest open gap.** A ~21M-parameter locally-trained transformer supplies *wording only*, behind the abstention gate; the brain decides what is true and whether to answer. An 88.6M model's forward pass has been re-run as spiking neurons matching the conventional version at perplexity ratio 0.9999999 — validated, and deliberately not deployed. Nothing in the cloud is in the runtime loop. |

---

## What is not solved

Named plainly, because each has a documented method that failed:

- **Open-ended fluent prose from the brain's own circuitry** — the largest gap, and
  field-wide at this data scale: a from-scratch spiking language model *and* a full
  transformer both lose to a well-tuned word-predictor on a few million words.
- **Deep multi-layer credit assignment on spikes** — the smooth-rate version trains
  across 6 seeds; the on-spikes port is a **powered NO-GO** tested to 40 epochs
  (credit vanishes ~1600× over depth, hidden representation tonic-pinned).
- **A learned binder replacing the fixed algebra** — single-attribute binding is a
  GO on spikes; multi-attribute bundling from scratch is tested-negative on point
  neurons (0.193 additive / 0.056 learned-linear against 0.989 for the fixed rule).
- **Choosing among several remembered referents for a bare "it"** — contested in our
  own record. A spiking biased-competition de-risk is 5/6 seeds with its controls at
  6/6, but on NumPy and default-off (`qualified`); the later 6-seed write-up meant to
  confirm it is **retracted** (a non-spiking re-run handed the answer it claimed to
  derive); and a 2026-07-17 audit still lists the capability as 0/3, having missed the
  de-risk. Treat it as unsettled until re-run on GPU.
- **Episodic recall by hippocampal pattern completion** — retracted as a drive
  artifact. Recall by content is available another way, so conversation is not blocked.
- **Consolidating composed memories into cortex** — slot allocation is retired *as a
  method*, not as a capability: it ceilings at 8–12 facts across three formulations
  at 6 seeds. Re-routed to a sparse distributed store.
- **V1 orientation selectivity self-organizing on the bridge** — the 6/6 negative was
  measured on a population that never fired, so it is void, not a mechanism result.

---

## Current frontier

The live resume point is the STATE OF THE PROJECT block in
[`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md); the forward plan is
[`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md).
As of 2026-07-31:

1. **Deep credit on spikes (the crux)** — running, 8 parallel cells; the roadmap's
   single load-bearing dependency. Its central arm, transport-free learned feedback,
   had never actually executed until an efficacy assertion caught two arms agreeing
   to five decimals.
2. **Credit on top of the plateau-expanded forward** — the highest-value experiment
   not yet run. The forward half was surpassed at 6 seeds; that expander has never
   been combined with the credit runner.
3. **Eviction for the affect ratchet** — GABA-B or slow after-hyperpolarization,
   both already in the engine and default-off.
4. **Region-scoping the one-brain state restore**, which today wipes whole-bridge
   state and erases co-resident mood every word.
5. **Record hygiene** — 30 stale citations, 11 artifacts flagged for identical
   experimental arms, 229 plans asserting results outside every gate.

### How the record keeps itself honest

Correctness is enforced by checks that fail loudly, not by discipline. The
failure→gate matrix ([`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md))
maps 23 known failure classes to modules; 17 block on a path that cannot be avoided.
On their first pass over this repo they blocked eight things written by their own
author in one session, and found 40 identical experimental-arm pairs across 11
banked artifacts, three "NEGATIVE" verdicts whose arms all sat below chance, and a
wrong number in an already-published table.

Two more rules are checked in CI — [`docs/TERMS.md`](docs/TERMS.md) (one term, one
meaning, with a code condition per load-bearing word) and
[`docs/WRITING.md`](docs/WRITING.md). Run both: `.venv/bin/python tools/check_docs.py`.

---

## How to run it

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator
pip install -r requirements.txt        # CuPy + DearPyGUI + PyOpenGL + h5py + scipy
pip install -r requirements-dev.txt    # optional: pytest and friends
python neural-simulator.py             # GUI with live 3D visualization
```

Full setup and troubleshooting: [QUICKSTART.md](QUICKSTART.md).

**No NVIDIA GPU?** One environment variable swaps CuPy for NumPy — slower, same
numbers:

```bash
SIM_BACKEND=numpy python -m research.runners.chat_demo --seed 43
```

**Talk to it.** An interactive shell that saves its trained brain so the next
session loads in seconds:

```bash
# First time: train, then save (~6-20 min depending on vocabulary)
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --save-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5

# Later: load and chat (~30 sec)
python -m research.runners.chat_repl --mode tier1 --seed 43 \
    --load-bridge simulation_checkpoints_h5/repl_tier1.simstate.h5
```

```
> remember the dog is big
  OK, I'll remember dog is big.
> is the apple small?
  I don't know. I haven't been told.
> who ate the apple?
  Dog did.
```

Every conversational demo: [`docs/CHAT-DEMO-GUIDE.md`](docs/CHAT-DEMO-GUIDE.md).

**Watch it navigate.** This is the shortcut-closed configuration — the one whose
4.08 ± 0.49 is quoted above. It has no goal coordinates, no agent coordinates, no
hand-coded heuristic, and no distance-based reward:

```bash
python -m research.runners.g11_bg_runner --moving-goal \
    --enable-place-goal-readout --learned-perception --enable-dlpfc-wm \
    --beacon-perception --beacon-replaces-goal \
    --cue-reflex --cue-reflex-replaces-heuristic \
    --enable-landmark-sensor --landmarks-replace-place \
    --sensed-reward --enable-msn-lateral-inhibition \
    --adaptive-da --adaptive-da-ema-decay-negative 0.7 \
    --curriculum --curriculum-warmup-steps 600 \
    --seed 42 --n-steps 1800
```

Drop the four shortcut-closing flag groups and `--heuristic-strength` returns to
its default of 1.0, which drives the motor cortex directly from a goal-coordinate
comparison. Check the flags before quoting a score.

**Tests, benchmarks, sweeps:**

```bash
pytest tests/ -v
python benchmark.py --quick                       # GPU throughput
python run_benchmarks.py --benchmark stdp-timing  # biological validation (Bi & Poo)
python -m research.experiment_runner experiments/biology_sweep.yaml
python -m research.result_aggregator --config biology
```

---

## Architecture

The GUI thread and the simulation thread are isolated and talk over lock-free
queues. The central object is the **`SimulationBridge`**, which owns all neuron and
synapse state as GPU arrays and advances the network one step at a time: synaptic
currents → noise → neuron-model update → plasticity → visualization → recording.
Connectivity is sparse, so memory grows with actual synapses, not neurons-squared.
Regions and the pathways between them are *declared as data* and wired
automatically.

| Package | What's in it |
|---|---|
| `sim/` | The engine — bridge, neuron/plasticity kernels, region framework, neuromodulators, connectivity, checkpointing |
| `viz/` · `ui/` | OpenGL 3D renderer; DearPyGUI panels and the config round-trip |
| `experiment/` | Stimulus injection, neuron groups, readout, training protocols |
| `research/runners/` | 1,332 headless experiment scripts |
| `research/findings/` | 1,845 dated write-ups — negative results included, by design |
| `tools/gates/` | 16 gate modules — the checks that refuse a bad result at commit or dispatch |
| `webapp/` | Web dashboard for launching runs and watching them live |

Measured on one RTX 3090 (24 GB): 1K–10K neurons runs anywhere including CPU;
10K–100K is the everyday research range; 100K+ needs ~20 GB VRAM. The GPU path is
roughly 4–50× the CPU path and numerically equivalent. Time step is 0.5 ms, tightened
automatically to 0.05 ms for full Hodgkin–Huxley biophysics.

Diagrams: [overview](docs/diagrams/brain_architecture_current.md) ·
[every region and pathway](docs/diagrams/brain_architecture_detailed.md).

### Known limitations

- **Not a large language model**, and not trying to be. The contribution is
  integrity — no shortcuts between sensing and acting, no fabrication,
  self-contained at runtime — not fluency parity.
- **Doing everything in neurons costs speed.** A fully-spiking conversation is much
  slower than the same pipeline with the retired ordinary-code shortcuts.
- **Scale.** Thousands of neurons per region against far more in biology, and far
  fewer training examples than a developing brain sees.
- **Simplifications.** No developmental pruning or cortical-layer formation; no
  protein-synthesis-dependent slow consolidation tier; no per-pathway conduction
  delays.
- **Research software.** APIs change. Not peer-reviewed, and not for clinical use.

---

## Where to look for detail

| You want… | Read |
|---|---|
| The live resume point and pending work | [`GAP_CLOSURE_MISSION.md`](GAP_CLOSURE_MISSION.md) → *STATE OF THE PROJECT* |
| The plain-language development path, stage by stage | [`ROADMAP.md`](ROADMAP.md) |
| The engineer-level plan: every faculty, every wall, its named biological surpass | [`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md) |
| What each result actually shows | `research/findings/` (dated; check the `status:` frontmatter first) |
| Which failure classes are mechanically prevented | [`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md) |
| What died, and what replaced it | [`docs/RETRACTED.md`](docs/RETRACTED.md) |
| What a load-bearing word is allowed to mean | [`docs/TERMS.md`](docs/TERMS.md) |
| The modelled biology, in plain language | [`docs/biology.md`](docs/biology.md) |
| Install, prerequisites, first run | [QUICKSTART.md](QUICKSTART.md) |
| How to contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **History** — the development narrative, superseded arcs, retired designs | [`docs/project-history-archive.md`](docs/project-history-archive.md) and [CHANGELOG.md](CHANGELOG.md). Searchable: `.venv-rag/bin/python tools/rag/rag_search.py "<question>" 5 --corpus doc` |

---

## Cite

```bibtex
@software{neural_simulator,
  title  = {Neural Simulator: a GPU-accelerated biologically realistic
            spiking neural network simulator with real-time 3D visualization},
  author = {Thiberge, Daniel},
  year   = {2026},
  url    = {https://github.com/danthi123/neural-simulator}
}
```

MIT — see [LICENSE](LICENSE). Mirrors:
[GitHub](https://github.com/danthi123/neural-simulator) ·
[Gitea](https://git.dant123.com/dant123/neural-simulator).
