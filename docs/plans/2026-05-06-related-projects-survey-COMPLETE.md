# Related projects survey — comprehensive findings

**Status:** All 4 research agents reported. ~25 projects/papers cataloged.

---

## Top 3 most actionable for current Tier 2.1 synonym WTA problem

These directly address the failure mode v1/v2 hit (STDP winner-take-all
when 2 input codes compete for same motor target).

### Pick 1: Heterosynaptic LTD (Tomasello / Pulvermüller / Felix)

- **Source:** Felix → NEST brain-constrained model (2026 NEST port)
  https://link.springer.com/article/10.1007/s11571-026-10415-5
- **Mechanism:** When post-V is high but pre-input is below threshold,
  the inactive synapse weakens. Coupled with standard STDP LTP, this
  prevents WTA: both synonyms' synapses get balanced down equally
  during off-trials, then strengthened equally during paired trials.
- **Why it fixes our problem:** v1/v2 showed that "north" trials only
  strengthen "north" synapses; "up" synapses drift via decay but don't
  see explicit LTD. Heterosynaptic LTD applies LTD to "up" synapses
  during "north" trials, balancing them. Same for vice versa.
- **Implementation cost:** ~3-5 days. Add het-LTD term to STDP kernel
  in `sim/kernels.py`. Gated by post-V > threshold AND pre-input <
  sub-threshold-active. Eq. B4 in Tomasello 2018 Frontiers Comp Neuro
  is the formal rule.
- **Risk:** moderate — modifies the core STDP kernel. Need careful
  testing to ensure existing Tier 1 result still holds.

### Pick 2: Two-stage curriculum (Tomasello fast mapping 2023)

- **Source:** https://academic.oup.com/cercor/article/33/11/6872/7048899
- **Mechanism:** Pre-shape language_input + motor pools INDEPENDENTLY
  first (drive each region alone for N events). Once each side has
  stable cell assemblies, ONE paired event suffices for binding.
  Compared to direct paired training (40-100 events for one mapping),
  this is far more data-efficient.
- **Why it might help synonyms:** Each synonym gets independent
  pre-shaping. Then one paired event with motor target binds both
  synonyms equally because both have well-formed assemblies that
  compete equally for the single co-firing event.
- **Implementation cost:** ~2 days. Add `--pre-shape-events N` flag
  to bio_three_factor: solo drive of each region for N events before
  paired training begins.
- **Risk:** low — just changes training schedule, no rule change.

### Pick 3: BTSP — Behavioral Timescale Synaptic Plasticity (Bittner & Magee)

- **Source:** https://github.com/neurosutras/BTSP (Magee lab)
- **Mechanism:** Single dendritic plateau potentiates synapses active
  within a **~2-second symmetric window** (vs STDP's 50ms). One-shot
  place-field formation in CA1; experimentally validated.
- **Why it might help:** STDP's 50ms window is fragile — pre/post
  must be tightly coincident. BTSP's 2-second window is much more
  permissive; teacher-driven plateaus can potentiate any active
  synapse in a 2-second envelope. Could enable **one-shot vocabulary
  entries** (one teacher episode → one stable mapping).
- **Implementation cost:** ~3-5 days. Add `cp_btsp_plateau` array
  triggered by motor teacher current; potentiates active language→motor
  synapses within 2-second window.
- **Risk:** moderate — new plasticity kernel.

## Top 3 most actionable for Tier 2.2/2.3 compositional language

### Pick 4: TEM tensor-product binding (Whittington 2020)

- **Source:** https://github.com/jbakermans/torch_tem
- **Mechanism:** verb⊗noun stored as outer product in CA3-analog
  memory. Retrieved via Hebbian "bind → store outer product → recall
  with partial cue" primitive.
- **Application:** Tier 2.3 verb+noun phrases. "go" + "north" =
  outer-product pattern in CA3. Sleep-time replay strengthens the
  binding via Hebbian consolidation.
- **Implementation cost:** ~1-2 weeks (Tier 2.3 budget).

### Pick 5: CMR-Replay context-driven retrieval (Schapiro 2024)

- **Source:** https://github.com/schapirolab/CMR-replay
- **Mechanism:** Drifting context vector probes memory; retrieved
  item updates context. Context-driven replay produces forward/
  reverse/clustered patterns naturally.
- **Application:** Sleep consolidation of Tier 2.3 phrase bindings.
  PFC working-memory state during phrase trial = "context"; replay
  during NREM-like windows reactivates phrase patterns.
- **Implementation cost:** ~1 week.

### Pick 6: Brain-Inspired (Generative) Replay (van de Ven 2020)

- **Source:** https://github.com/GMvandeVen/brain-inspired-replay
- **Mechanism:** Generator (CA3-analog) produces synthetic samples
  during sleep; classifier (cortex) trained on real-new + generated-old.
- **Application:** Tier 2.3 compositional generalization. Don't
  replay literal phrase trials — train CA3 as generator over
  (verb, noun) joint distribution. During sleep, sample novel valid
  combinations to push into PFC↔motor cortex.
- **Implementation cost:** ~2 weeks. Bigger lift, higher reward
  (compositional generalization to unseen phrase pairs).

## Architecture upgrade candidates (longer term)

### Burstprop (Payeur et al. 2021)

- **Source:** https://github.com/jordan-g/Burstprop
- **Mechanism:** Two-compartment pyramidal neurons. Singlets carry
  feedforward; bursts carry top-down error. Burst probability gates
  STDP polarity (LTP/LTD).
- **Status in our project:** Compatible with our shipped dendritic-
  learning design doc. Maps directly onto pyramidal apical-basal
  anatomy. Demonstrated at ImageNet scale.
- **Cost if pursued:** ~1.5 mo (replaces dendritic-learning Week 1+
  scope but with concrete reference implementation).

### dendritic_balance (Priesemann lab)

- **Source:** https://github.com/Priesemann-Group/dendritic_balance
  (PNAS 2023)
- **Mechanism:** Each dendritic branch self-balances drive. Local
  dendritic prediction error drives learning. Fully local; no global
  error broadcast.
- **Status:** Cleanest reference impl of apical-basal compartmental
  learning. Could port their dendritic-balance loss to our CuPy
  kernels rather than implementing from scratch.

### Axon (emer/axon) — closest match to our architecture

- **Source:** https://github.com/emer/axon
- **Highlights:**
  - Same Izhikevich/AdEx neuron family
  - Explicit BG (D1/D2 MSNs + GPi + STN), PFC working memory,
    hippocampus, motor/sensory cortices
  - **Rubicon model** of goal-driven motivated cognition
  - **PVLV** (Primary Value, Learned Value) gives biology-grounded
    sensed/learned reward dissociation — directly applicable to
    closing our cheat-5 reward signal
  - GPU via WGPU/Vulkan
- **Idea to steal:** PVLV separation of US/CS reward channels.
  Replaces our scalar `current_reward_signal` with anatomically-
  grounded primary + learned reward streams.

### CARLsim 6 — same neuron family, much bigger scale

- **Source:** https://github.com/UCI-CARL/CARLsim6
- **Highlights:** Izhikevich neurons + CUDA. Demonstrated **8.6M
  neurons / 0.48B synapses on 4 GPUs** with 60× single-thread CPU.
- **Idea to steal:** Experience-Dependent Axonal Plasticity
  (per-axon delay learning, not just weight). Our cortex→striatum→
  thalamus cascade has fixed delays; learnable delays could
  improve D1/D2 timing windows.

### BrainCog BDM-SNN — tri-pathway action selection

- **Source:** https://github.com/BrainCog-X/Brain-Cog
- **Highlights:** Explicit PFC→BG→thalamus→premotor with all three
  pathways (direct, indirect, hyperdirect) co-trained. E/I separation
  inside BG nuclei.
- **Idea to steal:** Tri-pathway action selection. Our cascade has
  STN but it's monolithic per-action. BDM-SNN's tri-pathway design
  could give rapid stop/no-go behavior for phase-2 goal-change
  benchmarks (cheat-5 multi-goal).

## Replay/sequence learning (Tier 2.3 + future)

### Ecker 2022 — emergent SWRs from recurrent matrix

- **Source:** https://github.com/Abolfazl-Alipour/ca3netReplay
- **Highlights:** SWRs spontaneously emerge from trained recurrent
  matrix during quiescence. No scheduled trigger. Symmetric STDP +
  PV cross-action FSI inhibition (~150 Hz ripples).
- **Apply to us:** Replace our Cluster D v2 scheduled `ca3_swr_burst`
  14% duty-cycle gate with emergent SWRs. We already have PV cross-
  action FSIs in Cluster B — same circuit motif.

### Haga & Fukai 2018 — reverse replay for goal-directed RL

- **Source:** https://github.com/TatsuyaHaga/reversereplaymodel_codes
- **Highlights:** Symmetric STDP + STD/ADP breaks forward/reverse
  symmetry. Reverse replay at reward locations strengthens forward
  goal-directed paths.
- **Apply to us:** We have STP machinery already. Gating CA3→CA3 STD
  during SWR + reward-anchored reverse replay = biologically grounded
  alternative to our scheduled D v2.

### biodreaming — Atari Pong via spiking world-model

- **Source:** https://github.com/cristianocapone/biodreaming
- **Highlights:** "Awake" trains policy on env; "dreaming" trains same
  policy on rollouts from learned spiking world-model. e-prop-style.
- **Apply to us:** Plugs into our existing SWR scaffold. Add a
  one-step prediction RNN; do reward-modulated rollouts during sleep.

## Learning rule candidates (beyond 3-factor)

| Rule | Repo | Scale | Biology score | Best for |
|---|---|---|---|---|
| e-prop (Bellec 2020) | IGITUGraz/eligibility_propagation | ~1k neurons, TIMIT | 7/10 | Online BPTT replacement |
| SuperSpike (Zenke 2018) | fzenke/pub2018superspike | small SNN | 8/10 | Voltage-graded plasticity |
| Burstprop (Payeur 2021) | jordan-g/Burstprop | ImageNet | 9/10 | Direct dendritic learning replacement |
| Predictive Coding | RobertRosenbaum/Torch2PC | MNIST/CIFAR | 8/10 | Hierarchical local error |
| DFA (Nøkland 2016) | lightonai/dfa-scales-... | NLP/transformer | 6/10 | Cheap test of "fixed random feedback" |
| BTSP (Magee 2024) | neurosutras/BTSP | small | 10/10 | One-shot vocab via 2-sec window |
| Equilibrium Prop | bscellier/Towards-... | small | 7/10 | Energy-based |
| GAIT-prop | nasiryahm/GAIT-prop | small | 7/10 | Target-prop variant |

## Synthesis — what to do if v3 cofire fails

(v3 looking like a fail at ~25% accuracy on partial eval as of 02:50)

**v4: scale-up** (already pre-staged) — fastest test, lowest risk.
**v5: heterosynaptic LTD** — most directly addresses WTA. ~3-5 days.
**v6: two-stage curriculum** — pre-shape language + motor before paired. ~2 days.
**v7: BTSP one-shot** — replaces STDP timing fragility with 2-sec window. ~3-5 days.

If v4 (scale-up) doesn't fix synonyms, the order of follow-on tries:
1. v6 (two-stage curriculum) — cheapest, biggest expected hit
2. v7 (BTSP) — if v6 doesn't fix, change rule
3. v5 (het-LTD) — if v7 doesn't fix, change rule deeper
4. Pivot to cluster motor pools (Nord-inspired, ~1 week)

Tier 2.2 (object binding) and Tier 2.3 (compositional) have ample
references; not blocked.

Architecture-level future work (Burstprop, dendritic_balance, Axon
PVLV/Rubicon) are months-scale investments.

## Citations

All sources documented above. ~25 projects total surveyed across
embodied language, brain-inspired RL, biology-plausible learning
rules, and hippocampal replay.
