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

**Jump to:** [What it can do](#what-it-can-do-today) ·
[What it can't do](#what-it-cant-do-yet) ·
[How to run it](#how-to-run-it) ·
[Architecture](#architecture) ·
[Limitations](#known-limitations) ·
[Further reading](#further-reading)

---

## What "works" means here

Claims below are deliberately unflattering. Unless a row says otherwise, a
capability has been reproduced across **six random seeds**, with control conditions
designed to break it — if switching the mechanism off doesn't damage the result,
the result doesn't count. Several once-headline claims here were withdrawn by
exactly those controls, and the withdrawals are stated inline rather than quietly
removed.

Nothing below is a finished feature. This is active research code, and the honest
caveats are part of the result.

---

## What it can do today

| Capability | Where it stands |
|---|---|
| **The engine** — spiking dynamics, region/pathway framework, plasticity rules, checkpointing, 3D view, GPU **and** CPU backends | **Mature.** 43 modules / 22,759 lines; 478 test modules; 47 region profiles. A 2026-07-31 sweep fixed 10 confirmed defects, two long-standing: spatial networks above 15,000 neurons raised `NameError` and had never run, and `set_plasticity_gate()` addressed the wrong synapses after a connectivity rebuild. Regression 53/53 on CuPy afterwards. |
| **Vision-based navigation to a moving goal** | **Works — but the widely-quoted description of it was withdrawn.** With every shortcut closed (the agent is given no goal coordinates, no coordinates of its own, no hand-written steering rule, and no reward computed from distance) it scores 4.08 ± 0.49 across 6 seeds, 30.6% above baseline. The more-quoted "visual cortex only" figures of 2.97 / 2.57 were actually measured with that hand-written steering rule still switched on at full strength, so that description was retracted in July 2026 and the visual pathway's own contribution is currently unmeasured. Which way to move is decided by competing populations of neurons racing each other to fire, not by picking the largest number in code. |
| **Answering questions about things it was told** | **Works within a narrow scope.** It parses active and passive sentences into who-did-what-to-whom, remembers facts, answers *who*/*what* and yes/no questions including negatives, and infers properties down a category hierarchy with exceptions — all at 6 seeds. Three related abilities are weaker: transitive inference ("A > B, B > C, so A > C"), chaining several facts together, and tracking pronouns across turns each hold at only 3 seeds, which is below this project's own bar. |
| **Saying "I don't know" instead of making something up** | **Works, at 3 seeds.** A learned familiarity signal decides whether the brain actually knows something. It matched the correct decision on all 168 test cues with no failures, and the separation disappears when the learned part is removed — so it is really doing the work. The language generator is never invoked when the brain declines to answer. |
| **Remembering new words and concepts without forgetting old ones** | **Works at a few hundred concepts**, 6 seeds, with previously learned facts still intact as new ones arrive. |
| **Learning categories and word meanings from experience** | **Works, 6 seeds.** Word meanings come from which words occur together; categories and simple hierarchies are discovered without being told; and category structure also forms from *seeing* objects, which collapses if the images are scrambled. Recalling a fact about a newly *seen* object still passes partly through ordinary code rather than staying entirely in neurons. |
| **Emotional colouring of concepts** | **Partly working, and one summary claim was corrected.** Tagging concepts with emotional value works at 6 seeds when done in ordinary code. Doing it inside the simulated brain is weaker — it fails on 2 of 6 seeds, and an internal note describing it as a clean 6-seed result was corrected on 2026-07-31. Its mood also only ever goes *up*: on 3 of 3 seeds it rises and never returns to baseline, because the mechanism that should bring it back down is not implemented yet. |
| **Reporting on its own internal state** | **6 seeds, but only on the CPU backend and outside the main loop.** A small region reads and reports where the brain's attention is, how confident it is, and whether a thought originated internally; cut its access to the real internal state and the reports fall apart. Not yet wired into the running system, and not yet reproduced on GPU. |
| **Curiosity** | **6 of 6 seeds inside the simulated brain**, with every control condition collapsing as it should — it asks about things it can actually learn from, and declines to chase noise it cannot learn. Holds within a stated scope. |
| **Reading the *direction* of memory replay from spikes** | **Complete; one related headline was withdrawn the same month.** It recovers whether a remembered sequence is replaying forwards or backwards on a single trial at 0.969 accuracy (chance 0.500), 6 seeds, with the bypass control at chance. Separately, a place-cell headline from the same arc turned out to be measuring how *concentrated* the synaptic changes were rather than *where* they were — shuffling the positions barely moved it (1.3%, p=0.42) — so that claim was withdrawn. A different configuration survived the stricter test at 5.05× and stands. |
| **Fluent English phrasing** | **A temporary external scaffold — and the project's biggest open gap.** A ~21M-parameter locally-trained transformer supplies *wording only*, behind the abstention gate; the brain decides what is true and whether to answer. An 88.6M model's forward pass has been re-run as spiking neurons matching the conventional version at perplexity ratio 0.9999999 — validated, and deliberately not deployed. Nothing in the cloud is in the runtime loop. |

---

## What it can't do yet

Stated plainly, because in each case a specific approach was tried and documented as
having failed:

- **Writing open-ended fluent prose from its own circuitry.** The biggest gap. At
  this amount of training data the problem is field-wide, not specific to this
  project: a from-scratch spiking language model *and* a conventional transformer
  both lose to a simple well-tuned word-predictor on a few million words.
- **Learning across many layers of neurons.** A smoothed, non-spiking version of the
  learning rule trains fine at 6 seeds. The spiking version does not learn at all,
  and the cause is measured: the correction signal shrinks by roughly 1600× as it
  passes back through the layers, and the middle layers stop changing.
- **Learning to combine several properties of an object at once.** Binding one
  property works in spiking neurons. Learning to bundle several from scratch does
  not — 0.19 and 0.06 against 0.99 for the hand-written rule it would replace.
- **Working out which of several remembered things a bare "it" refers to.** Our own
  record disagrees with itself here: one spiking experiment reaches 5 of 6 seeds
  with clean controls, a later write-up meant to confirm it was retracted (the
  re-run quietly answered the question in ordinary code instead of deriving it), and
  an audit still lists the ability as absent because it missed the first result.
  Treat this as unsettled.
- **Recalling a whole memory from a fragment**, the way the hippocampus is thought
  to. The claimed result was withdrawn — it turned out to be an artefact of how the
  neurons were driven. Recall by content works another way, so conversation is not
  blocked by this.
- **Moving assembled memories into long-term cortical storage.** One approach —
  allocating memories to slots — is retired: it stops scaling past 8–12 facts across
  three different formulations. The capability is not abandoned, only that method;
  work has moved to a distributed store.
- **Growing orientation-selective vision cells on their own.** The apparent negative
  result is void rather than informative: it was measured on a population of neurons
  that never actually fired.

---

## What's being worked on now

The main effort is getting learning to work across many layers of spiking neurons —
almost everything else depends on it. Alongside that: giving the emotional state a
way to come back down, and moving assembled memories into long-term storage by a
method that scales.

### A note on how results are checked

Because the most expensive mistake in a project like this is believing a result that
isn't real, correctness here is enforced by automated checks that block a change
rather than by care alone. They refuse claims whose numbers don't appear in the
underlying data, experiments whose control condition was accidentally identical to
the test condition, and results reported from a single random seed. On their first
run over this repository they rejected eight things written by their own author, and
found a wrong number in an already-published table.

Contributors: see [CONTRIBUTING.md](CONTRIBUTING.md) for how this works in practice.

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

## Further reading

**Start here**

| You want… | Read |
|---|---|
| Install, prerequisites, first run | [QUICKSTART.md](QUICKSTART.md) |
| The biology being modelled, in plain language | [`docs/biology.md`](docs/biology.md) |
| Where the project is heading, stage by stage | [`ROADMAP.md`](ROADMAP.md) |
| The conversational demos | [`docs/CHAT-DEMO-GUIDE.md`](docs/CHAT-DEMO-GUIDE.md) |
| How to contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |
| What changed and when | [CHANGELOG.md](CHANGELOG.md) |

**Working on the research itself**

Individual experiments live in `research/findings/`, dated, with negative and
withdrawn results kept rather than deleted — each declares its own status, and
[`docs/RETRACTED.md`](docs/RETRACTED.md) lists what died and what replaced it. The
detailed engineering plan is
[`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md`](docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md),
the automated checks are described in
[`docs/FAILURE_GATE_MATRIX.md`](docs/FAILURE_GATE_MATRIX.md), and the earlier
development narrative is in
[`docs/project-history-archive.md`](docs/project-history-archive.md).

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
