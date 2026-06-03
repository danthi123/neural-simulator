# Deep research — how the field gets past our generative-conversation wall (and what is genuinely untried)

**Date:** 2026-06-03
**Type:** Literature/field research synthesis (5 parallel research threads, web-sourced, cited).
**Trigger:** Owner asked for deep research into our generative-conversation boundary "in relation to what
we've tried and how others get past it," after the in-project check found a ~10-way-triangulated ceiling.

## TL;DR

The research strongly **validates that our walls are the field's walls** (independent reproduction of the
state of the art's limits — a confidence signal, not a mistake), AND surfaces **two genuinely-new,
biology-faithful, cheap-first-testable mechanisms our ~10 prior arcs did NOT cover**:

1. **A dedicated resonator-network DECODER + noise injection** — the field's universal fix for exactly our
   "composition works in algebra, fails in the substrate" + "non-vacuity degrades as the KB grows" walls.
   *Most biology-faithful, directly targets our characterized wall, cheap to probe, public code, and
   spiking stochasticity may supply the key ingredient for free.*
2. **Thalamocortical dynamical gating** (Logiaco-Abbott-Escola 2021) — the field's leading *circuit-level
   generative* sequencing mechanism, **genuinely distinct from all ten we tried**, and **partly already in
   our architecture** (the BG→thalamus cascade). The biggest research bet; motor-validated, not yet
   language-validated.

Plus the honest scale boundary (general open conversation needs ~360M+ params + trillions of tokens →
fundamentally beyond from-scratch biology-faithful single-GPU) and its documented escape (retrieval-augment
or distill a small generator; our validated VSA retrieval + abstention is the hard, distinctive half).

---

## Our wall, precisely (as the field would name it)

- **"Composition works in VSA algebra but fails in the learned spiking substrate"** = the **decode / clean-up
  / factorization** problem. Unbinding yields a *noisy* vector; recovering the intended atom is a separate,
  hard, iterative search. Algebraic exactness says nothing about whether recovery converges.
- **"Non-vacuity degrades as the knowledge base grows"** = the **crosstalk / superposition-capacity** law.
  Frady & Sommer: ~½ bit/neuron; retrieval SNR ∝ √D/√M (dimension D, M superposed items). It is a *theorem*,
  not a bug. (Frady, Kleyko & Sommer 2018, *Neural Computation*.)
- **"Spiking BPTT LM got WORSE with scale (overfit)"** = a **named phenomenon in the SNN field**: SpikeGPT's
  own paper reports train-BPC dropping while test-BPC stalls — "potentially suffering from over-fitting."
  Cause: surrogate-gradient mismatch + vanishing temporal gradients + limited-data overfitting.

## Thread 1 — Biologically-grounded / VSA cognitive architectures (Spaun, HRR/HDC)

**Verdict: this tradition does structured retrieval / Q&A / binding, NOT open generation — at any scale.**
- Spaun (Eliasmith 2012, 2.5M spiking neurons) outputs *drawn digits*, not utterances; its "novel" output is
  numeric induction over a tiny fixed symbol space. SPA "question answering" is unbind→clean-up-to-nearest-
  known-symbol — *structurally incapable of emitting outside its vocabulary*. **Spaun confirms our ceiling.**
- HDC/VSA surveys (Kleyko-Rachkovskij-Osipov-Rahimi 2023) name the inverse op "decoding/parsing/retrieval" —
  **"generation" is conspicuously absent.** NVSA (Hersche et al., *Nat. Mach. Intell.* 2023) — the strongest
  "generative VSA" — is closed-vocabulary answer-completion over a fixed attribute dictionary.
- The one paper asking our exact question — **"Bridging Cognitive Architectures and Generative Models with
  VSA" (AAAI-SS)** — concludes in writing: **VSA binding is "fundamentally associative rather than
  generative"; open generation requires COUPLING VSA to a separate neural generator** that samples under
  VSA constraints. *This is the field telling us the generation wall is real and the workaround is hybrid.*

## Thread 2 — Spiking language models

**Verdict: no direct-trained spiking net at single-GPU scale produces coherent open conversation; the field's
remedy is ANN-to-SNN CONVERSION, not better direct training.**
- SpikeGPT (45M/216M, direct surrogate-grad BPTT) reaches GPT-2-small-tier perplexity on *easy* corpora,
  makes **no coherence claims**, and **reports our exact overfitting-at-scale phenomenon**.
- The capable spiking LMs are **conversions/distillations** of pretrained transformers (SpikeLLM 7-70B from
  LLaMA; SpikingBERT/SpikeLM distill BERT/BART; FAS matches OPT-7B). Capability is *borrowed from the ANN*.
- To scale direct training at all, every system **softens the binary spike** (RWKV recurrence, SSM duality,
  integer/elastic activations). **Our finding = the field's finding.** TinyStories shows coherence-at-small-
  scale is a *data/curriculum* problem (10M-param ANN → coherent children's stories) — never shown for spikes.

## Thread 3 — Neuroscience of novel generation (THE genuinely-distinct mechanism)

**Verdict: there IS a leading, circuit-level, generative mechanism distinct from all ten we tried:
THALAMOCORTICAL DYNAMICAL GATING.** (Logiaco, Abbott & Escola 2021, *Cell Reports*; Halassa-lab thalamus
program: Schmitt 2017 *Nature*, Rikhye 2018 *Nat. Neurosci.*)
- **Mechanism:** recurrent cortex stores **no sequence**; it is a fixed *dynamic pattern-generator* with many
  intrinsic dynamical modes. Basal ganglia select the next motif by **disinhibiting a dedicated thalamic
  subset**, which imposes a **rank-one (low-rank) perturbation on cortical effective connectivity**,
  switching cortex into the mode that executes that motif. A **shared generic "preparatory" loop** drives
  cortex to a previous-motif-independent state → **any-to-any transitions with NO transition-specific
  learning.** Add a motif by recruiting unused thalamic neurons (no interference).
- **Why distinct from our 10:** vs HVC synfire chain (order hard-coded in the chain — here cortex stores no
  order); vs replay (replays stored — here constructs online); vs theta-gamma WM (maintains items — here
  switches generative dynamics); vs predictive coding (infers — here generates); vs VSA binding (algebra —
  here dynamical-systems gating). **It is compositional by construction.**
- **CRITICAL — it is partly already ours:** our g11 BG cascade `cortex_X → str_D1_X → gpi_X → thal_X →
  motor_X` IS qualitatively the Logiaco selection circuit. **What we are missing:** (a) a *fixed recurrent
  cortical pattern-GENERATOR* whose dynamical MODE is switched by thalamic low-rank perturbation (our motor/
  concept pools are *static attractors*, not reconfigurable dynamical generators), and (b) a *shared generic
  preparatory loop* for transitions. The research note diagnoses this as the likely reason **our compose-
  pathway went silent — we grew static synaptic weights instead of gating dynamics.**
- **Honest caveat:** validated for MOTOR sequencing; language application is a reasonable extrapolation, not
  an established result. No consensus spiking-circuit model of sentence generation exists (Matchin & Hickok
  2020 say so outright). The cognitive consensus is frame-and-slot + Hagoort Unification (which our content-
  selection Control already mirrors at the planning level).
- **Companion prior:** TEM (Whittington 2020) factorize-structure-from-content + the 2025 result that
  compositional generalization needs **linear + near-ORTHOGONAL** role codes (not just disentangled) — backs
  our orthogonal_drive_pattern / Kanerva-SDM instinct; the lever is orthogonality of *role slots*.

## Thread 4 — Compositional generalization & the algebra-vs-substrate fix (THE adoptable decoder)

**Verdict: the universal fix for "algebra works, substrate fails" is — DON'T make the learned forward pass be
the decoder. Insert a dedicated, structured DECODER between unbind and emission.**
- **Resonator networks** (Frady, Kent, Olshausen & Sommer 2020, *Neural Computation*) factorize bound
  composites by iterating unbind↔clean-up; they **outperform gradient descent / ALS** at this combinatorial
  search — i.e., a naively-learned substrate (effectively gradient descent) is *worse* at decode than the
  purpose-built factorizer. **This is the single most relevant external result to our wall.**
- **Noise injection** (Kymn et al. 2024) breaks the resonator's spurious limit cycles and **extends
  operational capacity ≥50×** — and **spiking stochasticity is a natural noise source** (we may get this for
  free). Demonstrated on spiking/neuromorphic hardware (Langenegger/Renner *Nat. Mach. Intell.* 2024).
- **"More knowledge → worse composition" remedy:** sparse high-D codes (Kanerva SDM — we do this) + **SEPARATE
  / modular stores with pattern separation** (DG orthogonalization / our multi-bridge route — scale by
  *number of stores*, not depth of one superposition: the capacity-safe LINEAR direction) + resonator+noise
  decoder. Pushes the degradation point out 1-2 orders; the trade-off remains a theorem.
- **MLC** (Lake & Baroni, *Nature* 2023): meta-learning for compositionality — **1.4M params, single GPU, no
  pretraining**, human-level systematic generalization. Transferable principle (architecture-agnostic):
  **meta-train the decoder on an episodic stream of (study-examples → query) tasks with re-randomized
  fillers**, so it learns to *unbind-and-emit in general* rather than memorizing one KB. (Honest limits:
  in-distribution-bound; fails length extrapolation; "lacks a mechanism for emitting new symbols.")
- **CPG** (Klinger 2023): SCAN from **14 examples** via a symbolic scaffold + tiny learned modules — extreme
  data-efficiency, but supplies the grammar by hand.

## Thread 5 — The honest scaling reality

- No hard minimum-scale cliff (loss is smooth; "emergence" is largely a metric artifact, Schaeffer 2023). But
  competent **general** conversation empirically needs **~360M-500M params + TRILLIONS of tokens** (Qwen2.5-
  0.5B, SmolLM2-360M). A single RTX 3090 can host/fine-tune but **cannot pretrain to competence**.
- **BabyLM (10-100M words)** buys strong grammar (BLiMP) but **explicitly NOT generation or world knowledge**
  ("current systems do not learn world knowledge within 100M words").
- **Documented escape:** retrieval-augment a small generator (RETRO: 25× smaller matches larger; RETRO-LI:
  works at small sparse DBs — *our regime*) and/or distill conversation from a teacher (BabyLlama: 58M beats
  its teachers on 10M words). **Our validated VSA retrieval + no-confabulation abstention IS the hard,
  distinctive half** most RAG systems lack. Trade: the *generator* becomes a conventional small net (relax
  biology for the generator), OR restrict the domain (TinyStories regime → small models become coherent).

---

## The convergent meta-finding (a confidence signal)

Our project independently reproduced the field's exact walls: the VSA-doesn't-generate consensus (AAAI),
the spiking-LM-overfits-at-scale phenomenon (SpikeGPT), and the crosstalk-capacity theorem (Frady-Sommer).
**We are not doing something wrong — we hit real, characterized walls.** That is the scientific value the
owner's framing predicted ("honest negatives under strict biology ARE the deliverable").

## What is genuinely untried + adoptable (the actual "how others get past it")

| # | Mechanism | Gets past which of OUR walls | Biology-faithful? | Cheap-first testable? | Status |
|---|---|---|---|---|---|
| A | **Resonator decoder + noise injection** | algebra-works-substrate-fails; capacity-degrades | YES (spiking resonators exist; spike noise = the ingredient) | YES (numpy/spiking resonator vs our current decode; measure capacity w/ + w/o noise) | strongest near-term lever |
| B | **Thalamocortical dynamical gating** | the *generator* gap (novel sequencing) | YES (BG→thal already ours) | partial (cheap probe: does thalamic low-rank gating of a fixed recurrent cortical RNN produce arbitrary motif orderings?) | biggest research bet; motor-validated only |
| C | **Hybrid: VSA-retrieval + small RA/distilled generator** | general open conversation | partial (generation delegated) | YES (wire our 320-VSA retrieval as RAG memory for a small generator) | pragmatic; relaxes biology for the generator |
| D | **MLC meta-learning + TEM orthogonal-role factorization** | data-efficient compositional generalization | principle is architecture-agnostic | YES (meta-train decoder on re-randomized-filler episodes) | a training-protocol change to A |

## Recommended next step (cheap-first, owner to confirm)

**Probe A first** — it is the most biology-faithful, directly targets our *characterized* wall, is cheap, has
public reference code, and our spiking noise may supply the key ingredient for free. Concretely: a numpy/
spiking **resonator-network decoder** placed between unbind and emission on our FHRR substrate, with a
pre-registered three-state gate measuring (i) does it recover bound composites where our current decode
fails, (ii) does **noise injection** extend decodable capacity (the ≥50× claim) on OUR codes, (iii) the
control (no resonator / no noise) reproduces the failure. If it RESOLVES → it is the missing decode stage
that may lift the compositional ceiling at far less than out-of-scope cost. If BOUNDARY → an honest negative
that further tightens the characterized wall.

B (thalamocortical gating) is the higher-ceiling but higher-cost generative bet, and notably the diagnosis
that "we grew static weights instead of gating dynamics" is a *specific, falsifiable* explanation for our
silent compose-pathway — worth a cheap probe after A. C is the pragmatic route to actual conversation if the
owner will accept a delegated (non-spiking) generator conditioned on our validated grounded memory.

## Key citations

- Eliasmith et al. 2012, *Science* (Spaun). Kleyko-Rachkovskij-Osipov-Rahimi 2023, *ACM CSUR* (HDC/VSA survey).
  Hersche et al. 2023, *Nat. Mach. Intell.* (NVSA). "Bridging Cognitive Architectures & Generative Models with
  VSA," AAAI-SS.
- Zhu et al. 2023 (SpikeGPT, arXiv:2302.13939 — the overfitting admission). Xing et al. 2024 (SpikeLLM,
  2407.04752). Eldan & Li 2023 (TinyStories, 2305.07759).
- **Logiaco, Abbott & Escola 2021, *Cell Reports* (thalamocortical flexible sequencing — the distinct
  generator).** Schmitt et al. 2017 *Nature*; Rikhye et al. 2018 *Nat. Neurosci.* Whittington et al. 2020,
  *Cell* (TEM). arXiv:2501.18797 (composition needs linear+orthogonal codes).
- **Frady, Kent, Olshausen & Sommer 2020, *Neural Computation* (resonator networks).** Kymn et al. 2024,
  arXiv:2412.00354 (noise ≥50× capacity). Frady, Kleyko & Sommer 2018 (crosstalk capacity theory).
- Lake & Baroni 2023, *Nature* (MLC, 1.4M params). Klinger et al. 2023, arXiv:2309.16467 (CPG, 14 examples).
- Hoffmann et al. 2022 (Chinchilla). Schaeffer et al. 2023, NeurIPS (emergence "mirage"). BabyLM findings
  2023/2024. RETRO-LI, arXiv:2410.00004. Timiryasov & Tastet 2023 (BabyLlama).
</content>
