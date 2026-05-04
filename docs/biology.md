# What Biology We Model

This is a tour of the real neuroscience inside the simulator, written
for someone curious but not necessarily an expert. Each section
describes a real brain mechanism, what it does, and how we capture it.

The full encyclopedia of mechanisms with per-claim citations is in
[`references/feature-catalog.md`](../references/feature-catalog.md)
on the `catalog-build` branch — ~375 entries each grounded in a
specific page range of one of the source texts below. This doc is
the curated subset that matters most for understanding the simulator.

## Source texts

The catalog and simulator are anchored in 13 primary references:

**Primary textbook:**
- Kandel, E. R., Koester, J. D., Mack, S. H., & Siegelbaum, S. A. (eds.) (2021).
  *Principles of Neural Science*, 6th ed. McGraw-Hill, 1500 pp.
  Cited throughout this doc as "Kandel 6e Ch X".

**Specialty references for specific brain systems:**
- **Basal ganglia:**
  - Bolam, J. P., et al. (2000). "Synaptic organisation of the basal ganglia." *J. Anat.* 196:527–542.
  - Tepper, J. M., & Koós, T. (2017). "GABAergic Interneurons of the Striatum." *Handbook of Behavioral Neuroscience*.
  - Tepper, J. M., et al. (2018). "Heterogeneity and Diversity of Striatal GABAergic Interneurons: Update 2018." *Front. Neuroanat.* 12:91.
  - Tepper, J. M., Abercrombie, E. D., & Bolam, J. P. (eds.) (2007). *GABA and the Basal Ganglia: From Molecules to Systems*. *Progress in Brain Research* Vol. 160 (cited as "PBR-160").
- **Cerebellum:**
  - Marr, D. (1969). "A theory of cerebellar cortex." *J. Physiol.* 202:437–470.
  - Albus, J. S. (1971). "A theory of cerebellar function." *Math. Biosci.* 10:25–61.
  - Hesslow, G. (2013). "Classical conditioning of motor responses." *Neural Networks* 47:81–87.
- **Hippocampus + cognitive map:**
  - O'Keefe, J. & Nadel, L. (1978). *The Hippocampus as a Cognitive Map*. Oxford University Press.
- **Brain rhythms:**
  - Buzsáki, G. (2006). *Rhythms of the Brain*. Oxford University Press.
- **Reward / reinforcement learning:**
  - Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
  - Schultz, W. (1998). "Predictive reward signal of dopamine neurons." *J. Neurophysiol.* 80:1–27.
  - Hollerman, J. R. & Schultz, W. (1998). "Dopamine neurons report an error in the temporal prediction of reward." *Nat. Neurosci.* 1:304–309.
  - Schultz, W. (2016). "Dopamine reward prediction-error signalling: a two-component response." *Nat. Rev. Neurosci.* 17:183–195.

**Language additions (project-internal `language-mechanisms-additions.md`):**
- Pulvermüller, F. (1999, 2005). Distributed action-word neural ensembles.
- Hagoort, P. (2014). Memory-Unification-Control framework. *Curr. Opin. Neurobiol.* 28:136–141.
- Tomasello, M. (2003). *Constructing a Language*. Harvard.
- Indefrey, P. (2011). Neurochronometry of word production. *Front. Psychol.* 2:255.
- Hickok, G., & Poeppel, D. (2007). Dual-stream model. *Nat. Rev. Neurosci.* 8(5):393–402.
- Friederici, A. D. (2017). *Language in Our Brain*. MIT Press.

**What "biology-grounded" means here:**
1. Every architectural decision in the simulator (region size, connectivity
   pattern, neuron type, plasticity rule) is traceable to a specific
   citation in the catalog.
2. Where simulation pragmatics force simplification (e.g., 5 neurons per
   sub-pool vs real cortex's millions), that's documented as a known
   limitation, not hidden.
3. We **never** add a non-biological shortcut to make the agent perform
   better. If the agent fails at a task, that failure is real — the
   biology may simply not be sufficient at our scale.

---

## The big picture: why model biology at all?

Modern AI uses gradient descent — a powerful but biologically
implausible optimization technique. Real brains have no equivalent of
backpropagation through time. They learn from local rules.

If we want to understand how brains actually learn, we have to use
the rules brains actually use. That's the constraint this project
takes seriously.

The trade-off: biology-faithful models are computationally expensive
and far less accurate than gradient-descent models. Gain: we learn
something about brains.

---

## Neurons, the basic unit

Real neurons are tiny electrical cells. They sum incoming signals,
fire a brief electrical pulse (the "spike") when the sum exceeds a
threshold, then briefly rest before firing again.

We model this using **Izhikevich's 2007 model** — a 9-parameter
equation (Izhikevich 2007, *Dynamical Systems in Neuroscience*, MIT
Press) that captures the essential firing behaviors of real cortical
neurons (Kandel 6e Ch 8 covers the membrane biophysics; Ch 14 covers
the cell-type taxonomy). There are 30+ "presets" for different cell
types:

- **Regular spiking pyramidals** — the "main" cortical neurons,
  ~80% of cortex (Kandel 6e Ch 14, "Neurons in the cerebral cortex")
- **Fast-spiking interneurons** — parvalbumin-positive cells that
  suppress others, fire 4× faster than pyramidals (Tepper-Koós 2017
  for the striatal homologs; Markram et al. 2004 for cortical FS)
- **Bursting cells** — fire in clumps of 2-4 spikes; cortical layer 5
  pyramidals (Connors & Gutnick 1990) and thalamic relay cells (Sherman
  2001)
- **Dopamine neurons** — fire phasically on reward, tonically at rest
  (Schultz 1998 — directly modeled)

For full biophysical detail when needed (e.g., for slow-time-scale
dynamics), we also support **Hodgkin-Huxley** (Hodgkin & Huxley 1952,
Kandel 6e Ch 10) and **Adaptive Exponential** (Brette & Gerstner
2005) neuron models.

Real brains have hundreds of cell types; we cover the main 30+ relevant
to the cortex/BG/thalamus/cerebellum loop.

---

## How the eye works (vision)

### Retina

The retina is a 2D sheet of light-sensitive cells. Two main channels:
- **ON cells** — fire when light is brighter at center than surround
  (detects "this region is bright")
- **OFF cells** — fire when surround is brighter than center
  (detects "this region is dark")

Real biology: Hubel & Wiesel demonstrated this center-surround
organization in 1950s-60s, leading to the 1981 Nobel Prize.

In our simulator:
- 32×32 grid of ON cells + 32×32 grid of OFF cells = 2048 retina neurons
- Each neuron's firing rate proportional to local image intensity

### V1 — primary visual cortex

V1 cells respond to **oriented edges**. A horizontal edge fires
horizontal-tuned cells; vertical edges fire vertical-tuned cells.

This emerges from "Gabor filters" — sinusoidal patterns multiplied by
Gaussian envelopes. Real V1 develops Gabor-like receptive fields in
the first weeks of life from natural visual experience (Olshausen &
Field 1996; Kandel 6e Ch 23 "Low-level visual processing").

In our simulator:
- 1024 V1 simple cells with Gabor-pre-tuned receptive fields (8 orientations
  × 2 spatial frequencies × 8×8 spatial grid). The pre-tuning skips a
  developmental learning phase the real V1 goes through, but post-init
  STDP can still refine connections to higher cortex.
- 512 V1 complex cells that pool nearby simple cells (orientation-
  invariant within a small region — same edge but allowing slight
  position shift; Hubel-Wiesel 1962)

### V2 and IT

Higher visual areas combine V1 features into more complex shapes,
then objects (Kandel 6e Ch 24-25 "Intermediate-level" and "High-level
visual processing"):
- **V2** — combines edges into corners, junctions, simple shapes
- **IT** (Inferotemporal cortex) — full object recognition; sparse-
  distributed concept representations (Quian Quiroga et al. 2005, *Nature*
  435:1102–1107, "the Jennifer Aniston neuron")

In our simulator:
- 256 V2 neurons receive from V1
- 64 IT neurons receive from V2
- Both refine via STDP during navigation training

---

## How the brain decides (action selection)

The basal ganglia is the brain's "action arbiter". It takes multiple
candidate actions from cortex, picks one to execute, suppresses the
others.

### The cascade

```
Cortex (action options)
   ↓
Striatum (D1 = "go" pathway, D2 = "no-go" pathway)
   ↓
Globus Pallidus (GPi = output, GPe = intermediate)
   ↓
Thalamus (relays selected action back to cortex)
   ↓
Motor cortex (executes the action)
```

Real biology — this is the canonical "direct/indirect pathway" model:
- Albin, R. L., Young, A. B., & Penney, J. B. (1989). "The functional
  anatomy of basal ganglia disorders." *Trends Neurosci.* 12(10):366–375.
- Mink, J. W. (1996). "The basal ganglia: focused selection and
  inhibition of competing motor programs." *Prog. Neurobiol.* 50(4):
  381–425.
- Kandel 6e Ch 38 "The Basal Ganglia" pp. 932–960.
- Anatomy detail: Bolam et al. 2000, *J. Anat.* 196:527–542 (canonical
  synaptic organization).

Damage at different points produces different diseases (Kandel 6e Ch 39):
- Loss of dopamine (substantia nigra) → Parkinson's (can't initiate movement)
- Excess striatal D2 activity → Tourette's (can't suppress unwanted movements)
- Striatal cell loss (caudate atrophy) → Huntington's (chorea)

In our simulator: 4 parallel channels (one per direction) of
cortex → striatum → GPi → thalamus → motor cortex. The chosen action
emerges from which channel "wins" the competition.

### Closed loop

Real BG sends thalamic output BACK to the cortex that started the
selection. This creates positive feedback for the chosen action and
helps maintain consistent action over short delays. We model this with
explicit `thal_X → cortex_X` and `cortex_X → stn` (hyperdirect)
pathways. Anatomical reference: Nambu, A., Tokuno, H., & Takada, M.
(2002). "Functional significance of the cortico-subthalamo-pallidal
'hyperdirect' pathway." *Neurosci. Res.* 43(2):111–117 — covered in
Kandel 6e Ch 38 pp. 941–946.

### Striatal microcircuit

The striatum has multiple cell types (Tepper-Koós 2017 §IV; Tepper et
al. 2018):
- **D1-MSNs** (~47% of striatum) — direct pathway, "go" signal
- **D2-MSNs** (~47%) — indirect pathway, "no-go" signal
- **PV-FSI** (1-2%) — parvalbumin fast-spiking interneurons, fast
  feedforward inhibition; the dominant inhibitory force on MSNs
  (Bolam-2000 Fig 3E; Bevan et al. 1998)

The D1/D2 asymmetry is biologically critical: dopamine excites D1 but
inhibits D2. This means the same dopamine signal both "go for chosen
action" AND "don't go for unchosen actions" simultaneously (Albin/Young
1989; Kandel 6e Ch 38 pp. 941–946).

We model this with separate D1 and D2 pools per direction, each with
correct dopamine receptor signs. Striatal interneuron taxonomy
(8 known classes per Tepper-2018) is partially modeled — currently
only PV-FSIs (the dominant feedforward inhibitor); other classes
(NPY-LTS, ChI/TAN, CR, etc.) are deferrable extensions.

---

## How the brain learns (plasticity)

### "Neurons that fire together wire together"

This is **Hebbian learning**, proposed by Donald Hebb in 1949
(*The Organization of Behavior*, Wiley). It's the foundation of all
biological learning.

In real synapses, this is implemented as **spike-timing-dependent
plasticity (STDP)**:
- If neuron A fires *before* neuron B (within ~20ms), the A→B
  connection gets stronger (LTP, Long-Term Potentiation)
- If A fires *after* B, the connection gets weaker (LTD)

First measured by Bi, G. Q. & Poo, M. M. (1998). "Synaptic modifications
in cultured hippocampal neurons." *J. Neurosci.* 18(24):10464–10472.
Comprehensive review: Caporale, N. & Dan, Y. (2008). "Spike timing-
dependent plasticity: a Hebbian learning rule." *Annu. Rev. Neurosci.*
31:25–46. Kandel 6e Ch 67-68 covers cellular and molecular mechanisms.

We implement exactly this rule with the asymmetric STDP window
(τ_+ ≈ τ_- ≈ 20 ms; A_+ slightly larger than A_- per Song et al. 2000
*Nat. Neurosci.* 3:919–926, "Competitive Hebbian learning").

### Reward changes everything

Pure STDP would learn random correlations. The brain solves this with
**dopamine** as a third factor — gating which spike-pair correlations
get reinforced.

Foundational work:
- Schultz, W. (1998). "Predictive reward signal of dopamine neurons."
  *J. Neurophysiol.* 80:1–27.
- Hollerman, J. R. & Schultz, W. (1998). "Dopamine neurons report an
  error in the temporal prediction of reward during learning."
  *Nat. Neurosci.* 1:304–309.
- Schultz, W. (2016). "Dopamine reward prediction-error signalling: a
  two-component response." *Nat. Rev. Neurosci.* 17:183–195.
- Computational/algorithmic side: Sutton & Barto (2018) Ch 14
  "Psychology" + Ch 15 "Neuroscience".

The mechanism: when dopamine fires, recently-active synapses get a
"tag" (eligibility trace) that decays over ~500ms. When dopamine
arrives during this window, those tagged synapses get extra LTP.

This is **three-factor learning**: pre-synaptic spike × post-synaptic
spike × dopamine = weight change. Computational survey: Frémaux, N. &
Gerstner, W. (2016). "Neuromodulated spike-timing-dependent plasticity,
and theory of three-factor learning rules." *Front. Neural Circuits*
9:85.

In our simulator: every plastic synapse has an eligibility trace that
decays exponentially with τ ≈ 500 ms. Reward (positive or negative)
multiplies STDP events that happened recently. Dopamine is
implemented as a global modulator with explicit reward injection
during training; per-action DA channels are also available
("compartmentalized DA", Cluster C v2 in our codebase).

### Working memory needs special wiring

To keep firing AFTER input stops (e.g., remembering a goal), neurons
need a special property called "bistability".

Foundational work:
- Wang, X. J. (1999). "Synaptic basis of cortical persistent activity:
  the importance of NMDA receptors to working memory." *J. Neurosci.*
  19(21):9587–9603.
- Wang, X. J. (2002). "Probabilistic decision making by slow
  reverberation in cortical circuits." *Neuron* 36(5):955–968.
- Goldman-Rakic, P. S. (1995). "Cellular basis of working memory."
  *Neuron* 14(3):477–485.
- Kandel 6e Ch 60 "Working memory" pp. 1330-1336.

NMDA receptors are like a magnetically-locked door:
- Need glutamate to be there (input arriving)
- AND need the cell to be already-depolarized (Mg²⁺ block must be
  removed by depolarization to let Ca²⁺ flow)
- When both conditions met, calcium rushes in
- This sustains depolarization, enabling continued firing

Our simulator implements per-region NMDA receptor activation: the
cortex pools and PFC have NMDA-mediated bistability enabled (Cluster G
v2.5 — Wang 2002 calibration), so they can sustain "goal in mind"
representations across delays. Striatum and cerebellum stay
NMDA-disabled (matching their actual biology — striatal MSNs use AMPA
predominantly per PBR-160 Ch 6 Wilson; cerebellar Purkinje use
metabotropic glutamate plus AMPA per Marr 1969 / Albus 1971).

---

## How the brain learns words (language)

Real human language uses two main pathways from auditory cortex
(Hickok-Poeppel 2007):

- **Dorsal stream**: word sound → motor cortex (for speaking, repetition)
  - Goes through arcuate fasciculus, Wernicke's → Broca's
- **Ventral stream**: word sound → semantic cortex (for understanding)
  - Goes through inferior longitudinal fasciculus, anterior temporal lobe

We model the dorsal stream as the "PFC bypass": `language_input →
motor_X` direct connections. This is biology-grounded but simplified
(real Broca involves much more than direct motor wiring).

### Pulvermüller's distributed action-word neurons

Friedemann Pulvermüller (1999, 2005) proposed that action words like
"kick" or "grasp" are stored in **distributed cortical ensembles**
that overlap with the motor cortex regions that execute the action.
Brain imaging confirms: hearing "kick" activates the leg motor cortex
specifically.

Our current architecture uses 4 separate motor pools (motor_N, motor_E,
motor_S, motor_W). This is a SIMPLIFICATION of Pulvermüller — real
biology has overlapping representations.

### What we found

After many architectural variations:
- Pure STDP + reward modulation produces 28.5% W→A accuracy across 6
  seeds (p=0.027 vs 25% chance) — but a permuted-label control test
  (2026-05-03) showed this is structure above chance, NOT aligned
  word→action learning. Across 25 prior eval files, 0/25 had the true
  labeled mapping ranked best of 24 permutations; best-permutation
  scores cluster at 30-37% but the orientation is randomly seeded,
  not task-aligned.
- The current architecture has cascade-driven structural noise that
  yields some 28-33%-accurate mapping per seed, but the mapping is
  arbitrary, not learned. See
  `research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`.
- The minimal-isolation test (2026-05-04) falsified the
  cascade-as-cause hypothesis (mean 16.7% at 3 seeds, BELOW chance);
  the cascade was a weak dampener on seed-dependent random structure,
  not its source.

Currently testing biology-grounded fixes (topographic prior per
Pulvermüller 2001-2003, PV-FS lateral inhibition between motor pools
per Vogels 2011) to see if real word→action learning emerges.

---

## Memory: hippocampus and replay

The hippocampus has three main regions in series (Kandel 6e Ch 54
"Internally Generated Cognition" + Ch 67 "Implicit memory"):
- **Dentate Gyrus (DG)** — pattern separator. Sparse activation (~3-5%
  of granule cells active per pattern) ensures similar inputs produce
  highly different DG patterns. Reference: Marr, D. (1971). "Simple
  memory: a theory for archicortex." *Phil. Trans. R. Soc. Lond. B*
  262(841):23–81.
- **CA3** — pattern completer. Heavy recurrent connectivity (each
  pyramidal connects to ~5% of others) creates attractor states;
  partial cues retrieve full memories. Reference: McClelland et al.
  1995 *Psychol. Rev.* "Why there are complementary learning systems
  in the hippocampus and neocortex."
- **CA1** — readout. Compares CA3-completed memory against current
  cortical input via direct EC bypass, projects to cortex via
  subiculum. Reference: O'Keefe & Nadel (1978) *The Hippocampus as a
  Cognitive Map* (Oxford University Press).

Damage to hippocampus → no new declarative memories (the famous case
of Henry Molaison, "H.M." — Scoville & Milner 1957 *J. Neurol.
Neurosurg. Psychiatry* 20:11–21; covered in Kandel 6e Ch 67).

### Sharp-wave-ripples and replay

During sleep, CA3 generates "sharp-wave-ripples" (SWRs) — brief
high-frequency (140-200 Hz) bursts that replay recent waking
experiences in fast-forward (10-20× compressed time).

Foundational work:
- Wilson, M. A. & McNaughton, B. L. (1994). "Reactivation of
  hippocampal ensemble memories during sleep." *Science* 265(5172):
  676–679.
- Buzsáki, G. (1986). "Hippocampal sharp waves: their origin and
  significance." *Brain Res.* 398:242–252.
- Buzsáki, G. (2006). *Rhythms of the Brain* (Oxford University Press)
  — comprehensive treatment.
- Girardeau, G., et al. (2009). "Selective suppression of hippocampal
  ripples impairs spatial memory." *Nat. Neurosci.* 12(10):1222–1223
  — causal evidence that SWRs are required for consolidation.

Replay is critical for consolidation per the **complementary learning
systems** framework (McClelland, McNaughton, & O'Reilly 1995):
experiences are fast-replayed to cortex during sleep, strengthening
recently-learned associations in long-term cortical memory while the
original hippocampal trace fades.

We have hippocampus + SWR infrastructure (Cluster D v1 trisynaptic
loop, and Cluster D v2 SWR-gated CA3 plasticity). Currently being
integrated with text I/O training — Phase 3 of `text_train_curriculum.py`
runs SWR-style replay of recent (token, action) tuples post-training.

---

## What we don't model

The simulator captures the major biology relevant to perception,
action selection, working memory, and reward learning at the
millisecond-to-seconds timescale. What's NOT modeled:

### Sub-cellular dynamics
- Individual ion channels (we use lumped Izhikevich/HH models)
- Calcium spikes in dendrites
- Glia (astrocytes, microglia, oligodendrocytes)
- Neurovascular coupling

### Long-time-scale plasticity
- Protein-synthesis-dependent late-LTP (Frey & Morris 1997)
  → Can't model multi-day consolidation
- Structural plasticity in cortex (synapse formation/elimination)
- Adult neurogenesis in DG

### Whole-body biology
- Embodiment beyond direction-of-movement (no muscles, no body)
- Neuroendocrine signals (cortisol, oxytocin, etc.)
- Sleep stages and circadian rhythms (basic NREM/REM only)

### Higher cognition
- Theory of mind
- Compositional language (only single-token mappings)
- Logical reasoning
- Abstract concepts

These are areas where real biology far exceeds what we capture. They're
also areas where modeling them at the spike level is genuinely
unsolved — fundamental research questions remain.

---

## Reading roadmap

If you want to go deeper:

**For the curious non-expert:**
- Read the rest of this document, then explore [README.md](../README.md)
- Try the GUI: `python neural-simulator.py`

**For neuroscience students:**
- Kandel et al. *Principles of Neural Science* (6th ed) — the bible
- Buzsáki *Rhythms of the Brain* (2006) — sleep, replay, oscillations
- Tepper-Bolam-Abercrombie *Basal Ganglia VIII* (PBR 160) — for BG depth

**For computational neuroscience researchers:**
- Catalog: [`references/feature-catalog.md`](../references/feature-catalog.md)
  (catalog-build branch) — every mechanism with citations
- Architecture: [CURRENT-STATE.md](CURRENT-STATE.md)
- Specific design decisions: [`docs/plans/`](plans/)

**For software engineers wanting to extend the codebase:**
- [CLAUDE.md](../CLAUDE.md) — project conventions and structure
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to add features
- [tests/](../tests/) — examples of biology validation
