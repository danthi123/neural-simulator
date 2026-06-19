# Hardware Acceleration & Neuromorphic Computing — Long-Term Vision and Roadmap

**Date:** 2026-06-18
**Status:** VISION / STRATEGY document. **This is not a current build task.** It is a durable reference for a
future phase — *once the simulator is production-ready and the spiking substrate is settled, how do we turbocharge
the substrate?* The project is presently mid-arc on the "one brain" consolidation (moving every cognitive
computation onto one shared spiking `SimulationBridge`); this document is the "what comes after the software is
mature" reference, to be revisited then.
**Scope:** the field of **neuromorphic computing** (brain-inspired hardware), why this simulator is an unusually
good fit for it, the full spectrum of substrates from today's GPU to physical analog neurons, and a pragmatic,
honestly-sequenced ladder for getting there. Written to be read cold by a future visitor with no memory of this
session.
**Audience:** the project owner and any future agent picking up the hardware-acceleration thread.

---

## 0. Executive summary

The simulator is a GPU-accelerated spiking neural network engine (CuPy / NVIDIA CUDA). It already exhibits the
three properties that define neuromorphic computing's natural workload: it is **spiking** (information is carried
by discrete events, not dense tensors), **sparse** (only a small fraction of neurons fire on any step), and
**event-shaped** (computation is irregular and driven by which spikes arrive where). And — the decisive clue from
the owner's RTX-3090 profiling (2026-06-17) — its real-time wall is **per-operation latency / kernel-dispatch
overhead** (the resonate loop is ~98% kernel-launch, the GPU ~99% idle), **not** memory (VRAM) and **not** raw
arithmetic throughput (FLOPs). That is precisely the regime where a conventional GPU is fighting its own nature
(dense, clocked, batch-oriented) and where neuromorphic hardware — event-driven, no global-clock dispatch,
compute co-located with memory — is purpose-built to win.

The recommended path is a **ladder of increasing specialization**, each rung gated on the previous one paying off
and on the science being settled enough to justify the loss of flexibility:

1. **GPU (today)** — flexible, mature, the development and validation substrate.
2. **Software kernel fusion / CUDA graphs (the current engineering arc)** — nearly free, large win, attacks the
   exact launch-bound bottleneck the profile found. Do this first; it may suffice for years.
3. **FPGA prototype** — validate an event-driven dataflow in real hardware without a chip tape-out; the realistic
   first *hardware* step for a solo researcher.
4. **Flexible digital neuromorphic chips (SpiNNaker2, Loihi 2)** — real, shipping, software-mappable silicon
   accessible through research-access programs; keeps the rich neuron zoo because the neuron model is still
   programmable.
5. **Custom ASIC co-designed for this sim's models** — the ultimate digital efficiency, but only once the models
   *freeze*; it is a multi-million-dollar, multi-year, platform-scale undertaking.
6. **Analog / physical neuromorphic (BrainScaleS, memristor crossbars; photonic / spintronic further out)** — the
   moonshot with the highest ceiling, where the neuron's dynamics *are* the device physics. Most aligned with the
   project's artificial-life and biology-translatable-insight goals, and the bleeding edge of engineering risk.

The single most important strategic insight: **the faithful, instrumented simulator being built now is not thrown
away when hardware arrives — it IS the specification and the golden reference** against which any custom hardware
is co-designed and verified.

A summary comparison table is in §8.

---

## 1. Framing: this is neuromorphic computing, and this sim is unusually well-suited to it

**Neuromorphic computing** is the engineering field that builds computer hardware whose architecture and physics
are modeled on the nervous system, rather than on the conventional stored-program ("von Neumann") computer. The
term was coined by **Carver Mead** at Caltech in the late 1980s; his 1989 book *Analog VLSI and Neural Systems*
laid the foundation, including the famous "silicon retina" and "silicon cochlea" — sensor circuits whose analog
electronics reproduced the *physics* of biological sensory transduction rather than digitizing and number-crunching
it. (Mead 1990, "Neuromorphic electronic systems"; the silicon retina work with Misha Mahowald.) The field's
defining bet is that a great deal of the brain's efficiency comes not from cleverer algorithms but from a
fundamentally different *architecture*.

Two architectural facts about the conventional computer motivate the whole field:

- **The von Neumann bottleneck / "memory wall."** In a standard CPU or GPU, the processing units and the memory
  are physically separate, and every computation requires shuttling data back and forth across that gap. For
  large neural workloads this data movement, not the arithmetic, dominates both time and energy. The human brain,
  by contrast, *co-locates* computation and memory — a synapse both stores a weight and performs the
  multiplication — and runs the entire system on roughly **20 watts**, performing on the order of 10^18
  synaptic operations per second. No conventional machine comes within orders of magnitude of that efficiency on
  brain-like workloads.
- **The clocked, dense, batch nature of the GPU.** A GPU is a throughput machine: it is fastest when it can apply
  the *same* dense operation to a large batch of data in lockstep under a global clock. Sparse, irregular,
  event-driven computation — where the work to be done depends on which few neurons happened to spike — is
  exactly the workload that *defeats* this model. The hardware ends up mostly idle, waiting on a serial stream of
  small operations.

**Why this particular simulator fits the neuromorphic sweet spot so well** — three reasons, the third being the
decisive one:

1. **It is already spiking and sparse.** The engine (`sim/`) simulates large networks of biological neuron models
   that communicate by discrete spikes through sparse (CSR) synaptic matrices. Activity is sparse by design. This
   is the *native* data model of neuromorphic hardware; most efforts to run conventional deep networks on
   neuromorphic chips must first painfully *convert* a dense network into a spiking one. This project starts where
   they have to arrive.
2. **Its computation is event-shaped.** The interesting work — which motor pool wins, which concept assembly
   completes, which gate opens — is determined by the pattern of spikes, not by a fixed dense matmul schedule.
3. **Its measured bottleneck is dispatch latency, not FLOPs or VRAM.** This is the load-bearing argument. The
   owner's 2026-06-17 RTX-3090 profile
   (`research/findings/2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md`) found that one
   conversational composer operation costs ~160 ms, of which **97.7% is the 208-step resonate loop**, which
   issues roughly **3,000–4,000 sequential tiny CUDA kernel launches** per operation (~15–20 launches/step ×
   208 steps). At ~40–50 µs of launch overhead each, the GPU sits **~99% idle**, waiting on the launch stream.
   The literature corroborates the per-launch cost: kernel launch overhead is on the order of **3 µs**, and
   launch-plus-synchronization is **5–20 µs** per kernel (Characterizing CPU-Induced Slowdowns in Multi-GPU LLM
   Inference, arXiv:2603.22774; the PyTorch CUDA-graphs work). This is the textbook signature of a *launch-bound*
   workload — many tiny, sequential, irregular operations — which is the precise workload neuromorphic hardware
   exists to accelerate. A neuromorphic chip has no host-side launch stream: each neuron updates locally and
   asynchronously, in or near the memory that holds its state, and spikes propagate as self-routing events. The
   bottleneck the GPU suffers from *structurally does not exist* on the right substrate.

So the project is not contemplating neuromorphic hardware because it is fashionable; it is contemplating it
because the simulator's workload is the canonical neuromorphic workload, and its measured pain is the canonical
neuromorphic pain.

**The load-bearing tension (carry this through the whole document).** This simulator runs an unusually *rich zoo*
of neuron and synapse models — see §2 — and most neuromorphic silicon achieves its efficiency by *fixing* the
neuron model in hardware. The richness that makes the sim scientifically valuable is exactly what makes it hard
to map onto fixed-function chips without losing fidelity. This tension is the reason the roadmap is a *ladder*
and not a leap: we do not freeze rich, still-evolving dynamics into silicon until the science that uses them is
settled.

---

## 2. The model zoo — the central constraint on hardware choice

Before discussing substrates, it is worth stating plainly what the hardware would have to reproduce. The engine
(`sim/enums.py`, `sim/kernels.py`, `sim/bridge.py`) supports, among others:

- **Izhikevich 2007** (9-parameter) and legacy Izhikevich — the workhorse spiking model.
- **Hodgkin–Huxley** with temperature-dependent, per-gate Q10 kinetics, plus extended currents: persistent sodium
  (NaP), low-threshold calcium (CaT), the h-current (Ih), and an M-current.
- **Adaptive Exponential Integrate-and-Fire (AdEx)** (Brette & Gerstner 2005) with multiple phenotype presets.
- **Resonate-and-fire complex-phasor neurons** (`NeuronModel.RESONATE_AND_FIRE`, defined in `sim/enums.py`,
  driven by `rf_kick` / `rf_resonate_steps` / `rf_set_complex_weights` / `_rf_advance_one` in `sim/bridge.py`):
  the neuron state is a complex number Z = re + i·im that rotates by exp(λ + iω) each step; a zero-crossing of the
  imaginary part is a spike whose *phase* carries information. This is the substrate for the project's
  phase-coded vector-symbolic ("FHRR" — Fourier Holographic Reduced Representation) composer.
- **Complex synapses** — the bind/unbind/bundle operations of the composer are performed *through* complex-valued
  synaptic weights (`cp_rf_w_re` / `cp_rf_w_im`), a sparse complex matrix-vector product.
- **NMDA receptors** with voltage-dependent Mg²⁺ block; **GABA_B / GIRK** slow inhibition; **short-term plasticity
  (STP)** (Tsodyks–Markram); **spike-timing-dependent plasticity (STDP)** with soft bounds; **eligibility traces**
  for three-factor (reward-modulated) learning; **homeostatic** firing-rate regulation.

No single neuromorphic chip on the market natively implements all of this. The practical consequence, expanded
per-substrate below, is the sequencing rule: **flexible substrates first** (which can run any of these in software
or microcode, just slower or less efficiently), and **fixed substrates only for the parts of the model that have
stopped changing.**

A pleasant surprise from the 2026 hardware survey (§4–§5): the *programmable* digital neuromorphic chips and the
analog AdEx chips are far better matched to this zoo than one might fear — Loihi 2 natively lists Izhikevich and
resonate-and-fire among its supported models and exposes graded (integer-valued) spikes and three-factor learning;
BrainScaleS-2 implements AdEx in analog and has demonstrated NMDA-plateau dendrites. The richness is a constraint,
but not a wall.

---

## 3. The spectrum of substrates: flexibility versus efficiency

Hardware for neural computation forms a continuous trade-off axis. At one end, maximum *flexibility* (you can run
any model, change it freely, but pay an efficiency tax). At the other, maximum *efficiency* (orders of magnitude
less energy and latency, but the design bakes in assumptions that are expensive or impossible to change). Moving
along the axis trades the ability to change the computation for the efficiency of executing a fixed one.

```
 MORE FLEXIBLE / GENERAL                                    MORE EFFICIENT / SPECIALIZED
 <-------------------------------------------------------------------------------------->
 CPU --- GPU --- FPGA --- manycore RISC-V --- digital neuromorphic ASIC --- mixed-signal/analog --- physical
 (today) (today)  (reconfig. (programmable    (Loihi2 / TrueNorth /          (BrainScaleS;        (memristor
                   logic)     cores; SpiNNaker) NorthPole; fixed models)      analog circuits)     crossbar;
                                                                                                   photonic;
                                                                                                   spintronic)
```

- **CPU** — fully general, serial; far too slow for large networks. Already left behind.
- **GPU** — massively parallel dense arithmetic; the project's current substrate. Wrong shape for sparse,
  launch-bound spiking, but unbeatable for development flexibility and maturity.
- **FPGA (Field-Programmable Gate Array)** — a chip of reconfigurable logic you can wire into a custom digital
  circuit. You design the dataflow (e.g., an event-driven spike router) and it executes in true parallel hardware
  with deterministic low latency. Reprogrammable, so no fabrication needed — the bridge between software and
  custom silicon.
- **Manycore RISC-V / SpiNNaker-class** — many small general-purpose processor cores, each with local memory,
  each simulating a handful of neurons, all communicating by spike packets. Fully programmable (any neuron model)
  but architecturally matched to spiking by being many-small-with-local-memory rather than few-big-with-shared-memory.
- **Digital neuromorphic ASIC** — a fixed-function chip (an Application-Specific Integrated Circuit) whose digital
  circuits implement a specific spiking-neuron computation in event-driven, in/near-memory fashion. Loihi 2,
  TrueNorth, NorthPole. 100–1000× more energy-efficient than a GPU on the right workload, at the cost of model
  fixity.
- **Mixed-signal / analog neuromorphic** — the neuron's differential equations are emulated by *analog* circuits
  (a real capacitor as the membrane), with digital periphery for routing and configuration. BrainScaleS. The
  dynamics are the circuit's physics; there is no time-stepping, so it runs far faster than real time.
- **Physical / emerging-device** — the computation *is* a physical process in a novel material: memristor/RRAM
  crossbars (the weight is a device conductance; the multiply is Ohm's law), photonics (light through interferometer
  meshes), spintronics (magnetic textures). The ultimate co-location of memory and compute.

The rest of the document walks the three substrate families the owner independently named — programmable boards
(§4), custom digital ASICs (§5), and analog/physical materials (§6) — then gives the recommended sequencing (§7),
a comparison table (§8), and an honest realism check (§9).

---

## 4. Idea 1 — RISC-V / programmable manycore boards: the win is *architecture*, not single-core speed

The owner's first idea was low-level programmable boards, possibly RISC-V, specialized for the sim. The crucial
reframing: **the advantage is not faster individual cores — it is a fundamentally better *architecture* for sparse
event-driven computation.**

**The architectural pattern.** Instead of one or a few powerful cores sharing a single large memory (the von
Neumann shape that creates the memory wall), use **many small cores, each with its own small local memory**. Map
neurons onto cores — a few hundred or thousand neurons per core. A neuron's state and its incoming synaptic weights
live in the *local* memory of its core, so the update never crosses the memory wall. When a neuron spikes, the core
emits a small **spike packet** onto an on-chip network; the network routes it to the cores hosting the
post-synaptic neurons, which add the synaptic effect locally. This routing scheme — sending only the *address* of
the neuron that fired, and only when it fires — is called **Address-Event Representation (AER)**, a foundational
neuromorphic idea (Mahowald, early 1990s): communication is sparse and event-driven, exactly mirroring biology,
where an axon is silent until its neuron fires.

**Why RISC-V specifically.** RISC-V is an *open* instruction-set architecture: anyone may design a processor that
implements it and, critically, may add **custom instruction extensions**. For this sim that is the lever. The
inner loop of any neuron update is a small fixed computation (integrate inputs, decay, check threshold, maybe spike).
On a general core that is many instructions; with a custom instruction it can become *one*. Concretely:

- A single-cycle **leaky-integrate-and-fire update** instruction.
- A **complex-synapse rotate** instruction for the resonate-and-fire phasors — the `exp(λ + iω)` rotation that the
  208-step loop performs thousands of times. Folding that into one custom instruction directly attacks the
  project's measured bottleneck *in hardware*.
- An **STDP / eligibility-trace update** instruction for on-chip learning.

This is an active, real research area, which de-risks the concept:

- **SpiNNaker / SpiNNaker2** (University of Manchester, led by **Steve Furber**, the original ARM architect, with
  TU Dresden) is the existence proof at scale. The first-generation SpiNNaker machine reached **over one million
  ARM cores** (1,036,800 cores across 1,200 boards) and can emulate on the order of a billion neurons; because the
  cores are general-purpose, it can run **any neuron model in software** — including this project's full zoo,
  HH and resonate-and-fire and all — at the cost of being slower per-neuron than fixed-function silicon. SpiNNaker2
  (22 nm FDSOI, 152 processing elements per chip, scaling toward a 10-million-core system) adds hardware
  accelerators for common operations and supports event-based machine learning. SpiNNaker is the canonical
  "programmable neuromorphic" platform: maximum model flexibility, real chips, accessible via the EBRAINS research
  infrastructure. (Furber et al.; arXiv:1911.02385, "SpiNNaker 2: A 10 Million Core Processor System".)
- **Custom RISC-V SNN cores** are being published steadily: *Polaris 23* (RISC-V custom SNN instruction
  extension + SIMD LIF with backprop-STDP), *SNAP-V* (a RISC-V SoC issuing custom SNN instructions to an
  event-driven pipeline), *IzhiRISC-V* (a custom ISA extension specifically for the **Izhikevich** model — directly
  relevant, since Izhikevich is this sim's workhorse), and *FeNN* (a RISC-V *vector* processor for SNNs). One
  reported a **72× speedup** from adding a tightly-coupled neuromorphic coprocessor versus the bare RISC-V core.
  (J. Supercomputing, "Polaris 23"; arXiv:2603.11939 "SNAP-V"; arXiv:2508.12846 / "IzhiRISC-V"; arXiv:2506.11760
  "FeNN".)

**The realistic on-ramp** is *not* to design a RISC-V chip. It is to **prototype the event-driven dataflow on an
FPGA first** (§7, rung 3): the architecture — neurons-to-tiles, local memory, AER spike routing — can be expressed
as a reconfigurable circuit, validated, and measured against the GPU for latency, with **no tape-out and no
fabrication cost**. The literature is consistent that FPGAs already beat GPUs on *sparse, event-driven* SNN
workloads on both latency and power, precisely because they exploit the sparsity the GPU cannot (see §7).

---

## 5. Idea 2 — a custom ASIC that beats a generic GPU: shipping reality, with one hard catch

The owner's second idea was a custom chip (ASIC) that outperforms a generic GPU. This is not speculative — digital
neuromorphic ASICs are a shipping reality, and on the right workload they beat GPUs by **100–1000×** in energy and
often dramatically in latency.

**The shipping chips and their numbers (2024–2026):**

- **Intel Loihi 2** (Intel 4 / 7 nm). Up to **1 million neurons** and **120 million synapses** per chip across
  ~120–150 neuromorphic cores; each core co-locates local SRAM for neuron state and weights (compute-in-memory),
  with a 2-D mesh routing spike packets. Crucially for *this* project, **Loihi 2's neuron model is programmable in
  microcode** — fixed-point "assembly" the user writes to define arbitrary spiking dynamics — and it **natively
  lists LIF, Izhikevich, resonate-and-fire, and Hopf-oscillator models**, supports **graded (integer-valued)
  spikes** up to 32-bit payloads, and supports **programmable three-factor learning rules** (STDP, Oja's rule,
  reward-modulated). That is a strikingly good match to the sim's Izhikevich + resonate-and-fire + eligibility-trace
  stack. Intel's **Hala Point** system networks 1,152 Loihi 2 chips into **~1.15 billion neurons / 128 billion
  synapses** across 140,544 cores at a maximum **~2,600 W** — and reports order-of-magnitude energy advantages on
  suitable workloads. (Intel Newsroom, "Intel Builds World's Largest Neuromorphic System"; HPCwire 2024-04-24;
  Open Neuromorphic, "A Look at Loihi 2".)
- **IBM TrueNorth** (28 nm, 2014). 5.4 billion transistors, 4,096 cores, **1 million neurons**, **256 million
  synapses**, running real-time neural networks at **~65 mW** (≈400 billion synaptic-operations-per-second per
  watt). The pioneering large digital neuromorphic chip, but with a *fixed* LIF-style neuron — the cautionary
  example of model fixity. (Merolla et al., *Science* 2014.)
- **IBM NorthPole** (12 nm, 2023). Not strictly spiking — it is a brain-*inspired* inference accelerator — but it is
  the sharpest demonstration of the in-memory principle: it **eliminates off-chip memory entirely**, keeping the
  whole model on-chip and "appearing externally as a memory chip." On a 3-billion-parameter model it ran inference
  **~47× faster** than the next most energy-efficient GPU and at **~73× higher energy efficiency** than the
  lowest-latency GPU. (Modha et al., *Science* 2023, "Neural inference at the frontier of energy, space, and time.")
- **SpiNNaker2** and **BrainScaleS** (covered in §4 and §6) round out the family — SpiNNaker2 as the programmable
  digital option, BrainScaleS as the analog one.

**The efficiency mechanisms** (why 100–1000×): event-driven execution (power is spent only when a spike actually
occurs — between spikes, the silicon is genuinely idle, not spinning); **in-/near-memory weights** (the synaptic
weight is read and applied where it is stored, killing the von Neumann round-trip); **no wasted dense matmul** (a
GPU computes the whole weight matrix even where activity is zero; a neuromorphic chip touches only the synapses of
neurons that actually fired). For a sparse spiking network this is the difference between paying for the whole grid
and paying only for the lit cells.

**THE HARD CATCH — model fixity.** Most neuromorphic ASICs achieve their efficiency by *fixing* the neuron model
in silicon. TrueNorth fixes LIF. Even Loihi 2's celebrated programmability is microcode over a *fixed-point*
arithmetic substrate optimized for certain dynamics — it does Izhikevich and resonate-and-fire well, but a faithful
**Hodgkin–Huxley** neuron (continuous conductances, four gating variables, temperature-dependent kinetics) or the
sim's **NMDA Mg²⁺-block / GABA_B** conductances would have to be *approximated*, with attendant fidelity loss that
matters for a project whose deliverable is *biological faithfulness*, not just function. This is the load-bearing
tension from §1, made concrete.

**The sequencing rule that follows.** Do **not** freeze the sim's rich, still-evolving dynamics into a custom ASIC
until the science that depends on them is settled. The flexible escape hatch is **SpiNNaker (programmable cores):
any model, just slower**. A true custom ASIC *co-designed for this sim's specific models* is the ultimate digital
endpoint — but it is a **multi-million-dollar, multi-year** effort that only makes sense once (a) the model set has
stabilized and (b) the project has become enough of a *platform* (many users, or a settled scientific instrument)
to justify the spend. That gating is in §7.

---

## 6. Idea 3 — custom analog materials / physical neurons: the most profound, and the most aligned with the goal

The owner's third idea — custom analog materials that replicate neuron properties in an analog fashion — is the
deepest of the three, and the one most aligned with the project's stated north star (artificial life with a proper
brain analogue, and biology-translatable insight). Here the line between *simulating* a brain and *building* one
genuinely blurs.

### 6.1 Analog neuron circuits — the dynamics ARE the physics (BrainScaleS)

In a digital simulator (including every chip above), the neuron's differential equations are *solved numerically*,
step by step, by arithmetic. In an **analog neuromorphic** system they are not solved at all — they are *physically
embodied*. The membrane potential is the voltage on a real **capacitor**; the leak is a real resistor; the synaptic
input is a real current. The equation `C dV/dt = -g(V - E) + I` is simply *what the circuit does*, continuously, in
real physical time. There is **no time-step `dt`** and **no per-step kernel launch** — the very things that bottleneck
the GPU do not exist as concepts.

The flagship is **BrainScaleS / BrainScaleS-2** (Heidelberg University, architect **Johannes Schemmel**, the
Electronic Vision(s) group):

- It implements the **AdEx** (adaptive-exponential integrate-and-fire) neuron — *which this sim already supports* —
  in **analog circuits**. A single BrainScaleS-2 chip has **512 analog AdEx neuron compartments** and **131,072
  plastic synapses**, with local in-synapse circuitry for STDP measurement and a local plasticity processing unit
  that can run learning rules *while the network runs*. (arXiv:2003.11996, "Accelerated Analog Neuromorphic
  Computing"; the BrainScaleS-2 hybrid-plasticity papers.)
- Because the dynamics ride on the native scale of on-chip capacitances and conductances, BrainScaleS runs at
  **~1000× biological real-time** (a tunable acceleration factor of 10³, and up to ~10⁴ in some regimes). The
  staggering implication for a *learning* project: experiments that would take a week of wall-clock at biological
  speed run in minutes; **years of simulated learning can be compressed into days.** For a project whose central
  difficulty is the wall-clock cost of training brain-faithful learning, this is the single most exciting hardware
  property in the entire landscape.
- And it is not limited to point neurons: BrainScaleS-2 has demonstrated **multi-compartment neurons with NMDA-
  and calcium-based non-linear dendrites** — sodium spikes, calcium spikes, **NMDA plateau potentials** in apical
  and thin dendrites (Schemmel, Kriener et al., "An accelerated analog neuromorphic hardware system emulating
  NMDA- and calcium-based non-linear dendrites"). This directly addresses two of the project's deepest open
  problems noted elsewhere in the codebase: the point-neuron limit on decorrelation/whitening (the
  Mikulasch–Priesemann limit) and the deferred dendritic-substrate rewrite. **Analog dendritic hardware could be
  the substrate on which the dendritic computations this project keeps hitting a wall on simply happen for free.**

### 6.2 Memristor / RRAM crossbars — compute *in* memory, the von Neumann bottleneck simply gone

The most direct attack on the memory wall is the **memristor crossbar** (memristor = "memory resistor"; in practice
**RRAM**, resistive RAM, or phase-change memory, PCM). A memristor is a two-terminal device whose electrical
conductance can be set and held — a programmable, non-volatile resistor. Arrange them in a grid (crossbar) and
something remarkable happens, a piece of physics doing linear algebra:

- Store a weight matrix as the **conductances** of the devices.
- Apply the input vector as **voltages** on the rows.
- By **Ohm's law** (current = voltage × conductance), each device outputs a current that is the product of its
  input and its stored weight — the multiply happens *in the device*.
- By **Kirchhoff's current law**, the currents on each column **sum** automatically as they merge on the wire — the
  accumulate happens *on the wire*.

So a full vector-matrix multiply — the dominant cost of any neural network — is performed in **one physical step,
in O(1) time, inside the memory itself.** There is no data movement: the weight never leaves the device, because
the device *is* the computation. This is the literal end of the von Neumann bottleneck for the matmul. The 2023–2024
literature shows real progress: PCM-based analog in-memory chips running ResNet and LSTM workloads with on-chip
digital periphery; oxide-RRAM crossbars doing in-memory vector-matrix multiply at >98% device yield; and methods to
program crossbar conductances to "arbitrarily high precision" for analog computing (*Science* 2024, "Programming
memristor arrays with arbitrarily high precision"; APL Machine Learning perspective 2023; ACS Applied Electronic
Materials 2024).

**The variability inversion — the bug that becomes a feature.** Every analog device is slightly different — device
mismatch, thermal noise, drift, finite precision. For a deterministic digital design this *variability* is the
enemy, the thing precision engineering fights. For a **brain** model it is the *opposite*: real neurons *are*
heterogeneous and noisy, and that heterogeneity is functionally important (it decorrelates, it regularizes, it
broadens dynamic range). An analog substrate's "imperfection" is biological realism delivered for free. The
project's own findings repeatedly note that biological noise and heterogeneity are features, not bugs — analog
hardware *is* that thesis made physical.

### 6.3 The philosophical and scientific point for THIS project

Two reasons analog/physical neuromorphic is not just another accelerator for *this* project specifically:

1. **It blurs "simulating" into "building."** A digital simulation *represents* a neuron with numbers. An analog
   circuit *is* a system with the same dynamics, governed by the same kind of physical laws (capacitive
   integration, conductance-mediated current). For an **artificial-life** goal — the project's defining aim — a
   physical analog neuron is not a detour on the way to a simulation; it is a plausible *destination*. Building an
   artificial creature whose "neurons" are real physical dynamical systems is closer to the goal than any number of
   floating-point updates.
2. **The biology-translatable loop closes.** This is the subtle, high-value argument. The project's other north
   star is that insights from the sim should translate back to insights about *real* brains. A digital GPU
   simulation can cheat physics in ways biology cannot: it has a global clock, unlimited precision, free
   long-range communication, no energy budget. **Analog neuromorphic hardware faces the *same constraints biology
   does** — device noise, component mismatch, strict locality of communication, a real energy budget, no global
   synchronizing clock. Therefore, porting the sim onto analog hardware would *force* the design to confront the
   exact problems evolution had to solve, and the solutions it finds would be *biologically meaningful* in a way a
   GPU solution might not be. The hardware stops being merely an accelerator and becomes a **scientific instrument
   for biology-translatable insight.** A learning rule that works under real analog noise and locality tells you
   something about why real synapses work the way they do; a learning rule that works only with perfect digital
   precision and free global communication may not.

### 6.4 Further-out substrates (flag as speculative)

- **Photonic neuromorphic** — neurons and synapses built from light. Synaptic weighting via Mach-Zehnder
  interferometer meshes or spectral slicing; spiking neurons via resonant-tunneling diodes, saturable absorbers,
  or superconducting-nanowire single-photon detectors. The draw is *speed* and *bandwidth*: optical signals
  propagate at light speed with little heat, and a 40,000-neuron spiking photonic network (trained with latency /
  rank-order coding to exploit sparsity) has been demonstrated. Still early; integration density, optical loss,
  and the electronic-optical interface are the open problems. (arXiv:2509.01262 review; IOPscience *Neuromorphic
  Computing and Engineering* 2025.)
- **Spintronic** — neurons and synapses from magnetic textures (e.g. skyrmions). Skyrmion-based SNNs have shown
  ~2× lower programming energy than CMOS for low-power pattern recognition. Even earlier-stage than photonics.

Both are worth *watching*, not betting on. They belong in this document for completeness and for the future
visitor's situational awareness, explicitly flagged as research-frontier substrates with no near-term path for a
solo project.

---

## 7. The pragmatic sequencing ladder (the recommendation)

The transitions are gated; each rung must earn the next. The guiding principle: **exhaust the cheap, flexible wins
before spending on the expensive, rigid ones; and never specialize the hardware ahead of the science.**

### Rung 0 — GPU (today)
The current substrate. CuPy / CUDA, an RTX 3090. Flexible, mature, the home of all development and validation. The
ladder begins here and **the GPU remains the development and golden-reference substrate at every later rung** (§7.5).

### Rung 1 — Software kernel fusion / CUDA graphs (the current engineering arc) — DO THIS FIRST
This is **already the active plan** (`docs/plans/2026-06-17-resonate-cudagraph-refactor-design.md`) and it is
nearly free relative to any hardware. It attacks the launch-bound bottleneck the profile found, *on the existing
GPU*, by collapsing the ~3,000–4,000 sequential per-op kernel launches into ~one **CUDA-graph** replay (a CUDA
graph records a sequence of kernels once and replays the whole batch with a single launch, eliminating per-kernel
host overhead). The de-risk prototype already measured **11× per op** (107 ms → 9.8 ms) by making the resonate
step pure-GPU (device-side counter, pre-allocated buffers, the structured composer matvec as an elementwise
gather-scale rather than a library call). Stacked with batching the knowledge-base scan and indexing the fact
store, a conversational turn plausibly drops from ~0.8 s to ~10–25 ms — **real-time, with no hardware change at
all.** *This rung may well suffice for years.* Only pursue hardware rungs once software fusion is exhausted and a
*further* step-change is genuinely needed.

**Gate to rung 2:** software fusion is implemented and exhausted, and a further large speedup (or a large
energy/parallelism win) is still wanted — e.g. for massively parallel training, always-on embodied real-time
operation, or untethered/edge deployment where a GPU's power budget is prohibitive.

### Rung 2 — FPGA prototype (the realistic first *hardware* step)
An FPGA lets you build a custom **event-driven digital dataflow** — the neurons-to-tiles + local-memory + AER
spike-routing architecture of §4 — and run it in real reconfigurable hardware, **without a chip tape-out**. It is
the bridge between software and silicon: you validate the architecture, measure real latency against the GPU, and
learn what a custom design would need, all for the cost of a dev board (hundreds to a few thousand dollars) and
engineering time. The literature is consistent that FPGAs **already beat GPUs on sparse, event-driven SNN
workloads** on latency and power — one demonstrated real-time SNN speech recognition at ~70% less power than a top
GPU; others report tens-of-× throughput gains — precisely because they exploit the sparsity the GPU's dense
clocked model cannot (Spiker and the SNN-on-FPGA survey, ScienceDirect S0893608025001352; PeerJ CS-3077). This is
the **realistic next hardware step for a solo researcher** and the natural place to prototype the custom-instruction
ideas (a one-cycle LIF update, a complex-synapse rotate) before any thought of a chip.

**Gate to rung 3:** the FPGA prototype validates the event-driven dataflow and quantifies the win, *and* the desired
scale (neuron count) or the desire for a turnkey software stack exceeds what hand-rolling on an FPGA delivers.

### Rung 3 — Flexible digital neuromorphic chips (SpiNNaker2, Loihi 2)
Real, shipping silicon, **software-mappable**, accessible **through research-access programs** rather than purchase
(Intel's Neuromorphic Research Community for Loihi 2; the EBRAINS infrastructure for SpiNNaker2 and BrainScaleS).
The key property that puts these *above* a custom ASIC on the ladder despite being real chips: **the neuron model
is still programmable** (SpiNNaker in full software; Loihi 2 in microcode), so the sim's **rich zoo survives** —
SpiNNaker can run HH and resonate-and-fire and everything else in software; Loihi 2 runs Izhikevich and
resonate-and-fire and graded spikes and three-factor learning natively. This is where the project gets
neuromorphic energy/latency benefits **without yet sacrificing model fidelity.** The honest cost is the porting
effort: mapping the sim onto the chip's tools and memory model, and accepting the dynamics that don't map cleanly
must be approximated or left on the GPU.

**Gate to rung 4:** the model set has *stabilized* (the science using it is settled), the chosen neuromorphic
platform's flexibility has become the bottleneck (you are paying an efficiency tax to keep programmability you no
longer need), **and** the project has the scale/funding/platform-status to justify a custom design.

### Rung 4 — Custom ASIC co-designed for this sim's models
The ultimate *digital* efficiency: a chip whose fixed-function circuits implement *exactly* this sim's neuron and
synapse models, nothing more, nothing less. Maximum energy efficiency and latency for this specific workload. The
cost is the catch from §5: **multi-million-dollar, multi-year, and rigid** — you are betting the models won't
change, because changing them means a new chip. Only rational once the science has frozen and the project is a
platform. For most plausible futures of a solo/small research project, this rung is aspirational.

### Rung 5 — Analog / physical neuromorphic (the moonshot)
BrainScaleS-class analog (accessible via EBRAINS research access *today*, notably — so the *analog* rung is
reachable for experimentation earlier than a custom *digital* ASIC) and, further out, custom memristor crossbars,
photonics, spintronics. The highest ceiling (1000× real-time learning; the von Neumann bottleneck gone; noise-as-
feature; the artificial-life and biology-translatable-insight payoffs of §6.3). Also the bleeding edge of
engineering risk: analog precision, programmability, device variability, and the immaturity of the tooling. The
destination, not the next step — but a destination genuinely worth naming, because for *this* project's goals it is
arguably the *right* one, not merely the fastest one.

### 7.5 The cross-cutting insight: the sim as golden reference
At **every** rung, the faithful, instrumented simulator built now serves three roles that mean **the software is
never wasted**:
1. **The specification.** The hardware's job is "do what the sim does." The sim *defines* the target behavior — the
   neuron equations, the learning rules, the network architecture — in executable, inspectable form.
2. **The validation reference.** Any hardware (FPGA, neuromorphic chip, analog) is verified by checking that it
   reproduces the sim's behavior on the same inputs — exactly as the project already gates the CUDA-graph refactor
   with bit-identity tests (`tests/test_rf_*`) against the reference loop. The same discipline scales to hardware:
   the sim is the oracle.
3. **The co-design tool.** You explore *what the hardware should be* by experimenting in the flexible sim first
   (which approximations are acceptable? which dynamics are load-bearing? where is fidelity worth silicon area?),
   then build hardware to match the answers.

This is why building the software faithfully now is not in tension with a hardware future — **it IS the hardware
spec, written in advance.** Every hour spent making the sim correct and well-instrumented is an hour of hardware
specification banked.

---

## 8. Comparison table

Ratings are deliberately coarse (Low / Medium / High and orders of magnitude) and reflect the project's specific
situation — a solo/small research effort whose deliverable is biology-faithful artificial life. "Fit for this sim"
weighs *model-zoo flexibility* heavily, because that is this project's binding constraint. "Solo-realistic" asks
whether a single researcher could actually get hands on it in a reasonable horizon.

| Substrate | Flexibility (model freedom) | Efficiency vs GPU (energy/latency on sparse SNN) | Maturity | Fit for THIS sim's model zoo | Realistic for a solo researcher | Representative systems |
|---|---|---|---|---|---|---|
| **CPU** | Highest | ~0.01–0.1× (worse) | Highest | Runs anything; far too slow at scale | Already have it | — |
| **GPU (today)** | High | 1× (baseline; launch-bound on this workload) | Highest | Runs the full zoo; the current home | Already have it (RTX 3090) | CuPy / CUDA |
| **GPU + CUDA-graph fusion** | High | ~10–100× on the launch-bound op (no HW change) | High (in progress) | Identical zoo — same code, fused | **Yes — already the active arc** | `enable_rf_cudagraph` |
| **FPGA** | Medium-High (you design the circuit) | ~10–50× latency/power on sparse SNN | High | Good — custom event-driven dataflow; can host any model you wire | **Yes — dev board + time; the realistic first HW step** | Xilinx/AMD, Intel/Altera; "Spiker" |
| **Manycore RISC-V / SpiNNaker** | High (any model, in software) | ~10–100× (architecture-matched; per-core slower) | Medium-High (SpiNNaker2 real, research access) | **Excellent — full zoo survives in software**; custom ISA can 1-cycle the LIF/phasor steps | Partial — SpiNNaker2 via EBRAINS; a custom board is a project | SpiNNaker / SpiNNaker2 (Furber) |
| **Digital neuromorphic ASIC** | Low–Medium (Loihi 2 microcode-programmable; TrueNorth fixed) | ~100–1000× | Medium (Loihi 2 real, research access) | Mixed — **Loihi 2 natively does Izhikevich + resonate-and-fire + graded spikes + 3-factor**; HH/NMDA/GABA_B need approximation | Partial — Loihi 2 via Intel's research community; *building* one is not solo-scale | Loihi 2, TrueNorth, NorthPole, SpiNNaker2 |
| **Mixed-signal / analog** | Low (fixed analog model, tunable params) | ~1000× real-time + very low energy | Medium (BrainScaleS real, research access) | Strong on the parts it covers — **analog AdEx (sim has it) + NMDA-plateau dendrites demonstrated**; not arbitrary HH | Partial — BrainScaleS via EBRAINS for *experiments*; fabricating custom analog is lab-scale | BrainScaleS / BrainScaleS-2 (Schemmel) |
| **Memristor / RRAM crossbar** | Low (matmul-in-memory primitive) | ~100–1000× (von Neumann bottleneck eliminated) | Low–Medium (lab demos, small arrays) | Synaptic matmul-in-memory is a strong fit; full neuron dynamics need surrounding circuitry | No (research-frontier; fabrication) | PCM/RRAM crossbar demos |
| **Photonic** | Low–Medium | Very high speed/bandwidth; energy TBD | Low (early demos) | Speculative for this sim's dynamics | No (research frontier) | MZI-mesh + RTD/SNSPD SNNs |
| **Spintronic** | Low | ~2× lower programming energy (early) | Low (earliest) | Speculative | No (research frontier) | Skyrmion-based SNNs |

**How to read it.** The cheap, high-fit, solo-realistic rows are at the top (CUDA-graph fusion, FPGA). The
high-efficiency rows (ASIC, analog, memristor) trade away exactly the model-flexibility this project most needs,
which is why they sit later on the ladder and gate on the science settling. The pleasant surprise is the
"Fit" column for the two flagship research platforms — **Loihi 2** (Izhikevich + resonate-and-fire + graded spikes
+ three-factor, natively) and **BrainScaleS-2** (analog AdEx + NMDA-plateau dendrites) — both of which map onto
this sim's zoo far better than the general "neuromorphic = fixed LIF" reputation would suggest, and both reachable
for *experiments* through research-access programs without a fabrication budget.

---

## 9. Honest realism check

A forward-looking document should be ambitious; it should also be honest about what is cheap and near versus
expensive and far. In rough order of increasing cost/risk and decreasing immediacy:

- **Software wins come first and are nearly free.** The CUDA-graph / kernel-fusion arc (rung 1) is already in
  flight, already de-risked at 11×, requires no hardware, and plausibly delivers real-time conversation on the
  *existing* RTX 3090. **This is the correct and only near-term priority.** Everything in this document past rung 1
  is explicitly *future* work, contingent on the software path being exhausted and a further step-change being
  genuinely needed. Do not let the excitement of silicon distract from the fact that the measured bottleneck is
  software-fixable today.
- **FPGA is the realistic next hardware step** for a solo researcher: a few hundred to a few thousand dollars for a
  board, plus a real but bounded learning curve in hardware description. It de-risks every later rung and is where
  the architectural ideas (event-driven dataflow, custom instructions) get their first real test. It is the highest
  hardware rung a solo project can fully *own*.
- **Commercial neuromorphic chips (Loihi 2, SpiNNaker2, BrainScaleS) are accessible via research programs, not the
  store.** Intel's Neuromorphic Research Community gives academic/research access to Loihi 2; EBRAINS provides
  remote access to SpiNNaker2 and BrainScaleS-2. This means the project could, in principle, *experiment* on
  world-class neuromorphic hardware — including the *analog* BrainScaleS — without owning or fabricating anything,
  contingent on a research-access application. The cost is the porting effort and the constraint of the platforms'
  tooling and model limits.
- **A full custom ASIC is product-scale or well-funded-lab-scale, full stop.** Millions of dollars, years of
  effort, a team, and a frozen model set. It is the right *endpoint* for a project that has become a platform; it
  is not a thing a solo researcher tapes out. Naming it as the digital endpoint is correct; planning to build it
  soon is not.
- **Analog / physical is the moonshot — highest ceiling, bleeding-edge engineering.** The payoffs (1000×
  real-time learning, the von Neumann bottleneck gone, noise-as-feature, the artificial-life and
  biology-translatable-insight alignment) are real and, for this project's specific goals, arguably make it the
  *right* destination rather than merely the fastest one. But the engineering is genuinely hard: analog precision
  and calibration, the difficulty of *programming* analog substrates, device variability and drift, and immature
  tooling. *Experimenting* on BrainScaleS via EBRAINS is reachable; *building* custom analog or memristor hardware
  is a research program in its own right. Treat it as the inspiring far horizon it is, and revisit when the rungs
  below it have been climbed.

**Trade-offs to keep in view at every rung.** More efficiency buys less flexibility — and this project's binding
constraint is flexibility, because its value is model fidelity and its models are still evolving. The whole point
of the ladder is to *defer* the flexibility-for-efficiency trade until the science is settled enough that it stops
hurting. Until then, the flexible substrates (GPU, FPGA, programmable manycore) are not a compromise — they are the
*correct* choice, and the right place to keep the work.

---

## 10. Bottom line

This simulator is an unusually good candidate for neuromorphic acceleration because it is already spiking, sparse,
and event-shaped, and because its measured bottleneck — launch-bound dispatch latency, not VRAM or FLOPs — is the
exact pain neuromorphic hardware exists to remove. The right path is a **gated ladder**: exhaust the nearly-free
software fusion first (already in flight, ~11× proven, plausibly real-time on today's GPU); then, only if a further
step-change is needed, prototype the event-driven dataflow on an **FPGA** (the realistic first hardware step);
then run on **flexible digital neuromorphic chips** (SpiNNaker2 / Loihi 2, via research access, model zoo
preserved); reserve a **custom ASIC** for a settled-science, platform-scale future; and hold **analog / physical
neuromorphic** (BrainScaleS, memristor crossbars) as the moonshot — the substrate where the neuron's dynamics *are*
the device physics, where the von Neumann bottleneck simply ceases to exist, where device noise becomes biological
realism for free, and where, uniquely among the options, the hardware faces the same constraints biology does and
so becomes a scientific instrument for the project's biology-translatable goals, not merely an accelerator.

Through all of it, the faithful, instrumented simulator built now is **the specification and the golden reference**
for whatever hardware comes later. The software is not a stopgap to be discarded when the silicon arrives — it is
the silicon's blueprint, written in advance.

---

## Appendix A — Sources

Chip specifications and architectures:
- Intel Loihi 2 — Intel Newsroom, "Intel Builds World's Largest Neuromorphic System" (Hala Point); HPCwire,
  2024-04-24; Open Neuromorphic, "A Look at Loihi 2"; Intel "Loihi 2 / Lava" briefs. (1M neurons, 120M synapses;
  microcode-programmable neuron models incl. Izhikevich + resonate-and-fire; graded spikes; three-factor learning.)
- IBM TrueNorth — Merolla et al., *Science* 345:6197 (2014). (1M neurons, 256M synapses, 65 mW.)
- IBM NorthPole — Modha et al., *Science* 382:6668 (2023), "Neural inference at the frontier of energy, space, and
  time." (~47× faster, ~73× more energy-efficient than comparable GPUs; off-chip memory eliminated.)
- SpiNNaker / SpiNNaker2 — Furber et al.; arXiv:1911.02385, "SpiNNaker 2: A 10 Million Core Processor System for
  Brain Simulation and Machine Learning"; Open Neuromorphic, "A Look at SpiNNaker 2." (>1M ARM cores;
  software-programmable any neuron model.)
- BrainScaleS / BrainScaleS-2 — Schemmel et al.; arXiv:2003.11996, "Accelerated Analog Neuromorphic Computing";
  Schemmel, Kriener et al., "An accelerated analog neuromorphic hardware system emulating NMDA- and calcium-based
  non-linear dendrites"; BrainScaleS-2 hybrid-plasticity papers; EBRAINS BrainScaleS-2 platform notes. (512 analog
  AdEx neurons, 131k plastic synapses, ~1000× real-time, multi-compartment NMDA-plateau dendrites.)

Foundations and learning:
- Carver Mead, *Analog VLSI and Neural Systems* (1989); Mead, "Neuromorphic electronic systems," *Proc. IEEE* 78
  (1990). The silicon retina (with Misha Mahowald). Address-Event Representation.
- Indiveri et al., "Neuromorphic Silicon Neuron Circuits," *Frontiers in Neuroscience* 5:73 (2011).
- Bellec, Scherr, Maass et al., "A solution to the learning dilemma for recurrent networks of spiking neurons"
  (e-prop / eligibility propagation), *Nature Communications* 11:3625 (2020); "E-prop on SpiNNaker 2," *Frontiers
  in Neuroscience* 16:1018006 (2022).

RISC-V / FPGA SNN acceleration:
- "Polaris 23," *The Journal of Supercomputing* (2024); "SNAP-V," arXiv:2603.11939; "IzhiRISC-V," arXiv:2508.12846;
  "FeNN: A RISC-V vector processor for SNN acceleration," arXiv:2506.11760.
- "Spiker: an FPGA-optimized hardware acceleration for SNNs"; "Spiking neural networks on FPGA: a survey,"
  ScienceDirect S0893608025001352; PeerJ Computer Science CS-3077.

Memristor / in-memory and emerging substrates:
- "Programming memristor arrays with arbitrarily high precision for analog computing," *Science* (2024);
  "Bring memristive in-memory computing into general-purpose machine learning," *APL Machine Learning* 1:040901
  (2023); ACS Applied Electronic Materials (2024), RRAM crossbar VMM at >98% yield.
- "Integrated photonic neuromorphic computing," arXiv:2509.01262; spiking photonic network of 40,000 neurons,
  *Neuromorphic Computing and Engineering* (IOPscience, 2025); skyrmion-based SNNs (spintronic).

GPU launch overhead (the bottleneck framing):
- "Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference," arXiv:2603.22774 (kernel launch ~3 µs;
  launch+sync 5–20 µs); "Accelerating PyTorch with CUDA Graphs," PyTorch blog; "Boosting Performance of Iterative
  Applications on GPUs: Kernel Batching with CUDA Graphs," arXiv:2501.09398.

Project-internal grounding:
- `research/findings/2026-06-17-scaling-profile-3090-latency-is-the-wall-not-vram.md` — the decisive profile:
  ~160 ms/op, 97.7% the 208-step resonate loop, ~3–4k sequential kernel launches, GPU ~99% idle, launch-bound.
- `docs/plans/2026-06-17-resonate-cudagraph-refactor-design.md` — the CUDA-graph fusion arc (rung 1) and its 11×
  de-risk prototype.
- `sim/enums.py`, `sim/kernels.py`, `sim/bridge.py` — the neuron-model zoo (Izhikevich, HH, AdEx,
  `RESONATE_AND_FIRE`, complex synapses, NMDA, GABA_B/GIRK, STP, STDP, eligibility, homeostasis).
- Owner memory: `feedback_prioritize_orchestration_overhead` (latency is the wall; local-only at small-LLM scale),
  `project_actual_goal_artificial_life_brain_analogue` (artificial life + biology-translatable insight),
  `feedback_move_everything_to_shared_spiking_substrate` (the "one brain" consolidation context this doc post-dates).

*Note:* the emerging-substrate items (memristor crossbars, photonic, spintronic) reflect lab-scale demonstrations,
not commercial products, and should be read as research-frontier indicators rather than available technology.
