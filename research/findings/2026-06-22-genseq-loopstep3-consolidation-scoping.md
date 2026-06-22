# Generative-sequence frontier (Spine A) — LOOP-STEP 3 (CONSOLIDATE the FULL spiking generator onto the ONE bridge) scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings/literature scoping for loop-step 3 of the generative-sequence
> frontier — *consolidate the converted spiking `TinyGPT` (Gen-F) onto the ONE `SimulationBridge` as a co-resident
> spiking slice*, the C1 "ends fully-spiking on one bridge" goal, for the **FULL generator** (attention + LN + GELU +
> MLP + embeddings + head), not just step-0's single positive-weight one-hot layer. **NO `sim/` edits, NO experiments,
> NO GPU.** Single deliverable = this doc. Every load-bearing project claim re-verified against the repo (file:line).
> SOTA bounded by a fresh June-2026 spiking-transformer / neuromorphic-attention literature pass (verified to primary
> sources). Builds on — does not re-derive — the parent frontier scoping
> (`2026-06-22-generative-sequence-frontier-scoping.md`), the convert scoping
> (`2026-06-22-genseq-convert-scoping.md`), step-0 GO (`2026-06-22-genseq-step0-C1-consolidation-GO.md`), and convert GO
> (`2026-06-22-genseq-convert-GO-spiking-generates.md`). **This is a SCOPING/DECISION doc, NOT a brain-based result and
> NOT a commitment to build.** The controller should trust-but-verify the **[VERIFY]** items, push, and present before
> building.

---

## 0. One-paragraph answer (the rest is the evidence)

**The crux (Q1 — attention on a fixed-connectivity bridge) has a clean, low-`sim/`-edit answer, and it is NOT "emulate
softmax."** Every linear op of `TinyGPT` (token/pos embeddings, the 4× MLP `Linear(d,4d)→GELU→Linear(4d,d)`, the Q/K/V
and output projections, the LM head) is a static weight matmul that installs directly onto bridge synapses via
`inject_explicit_wiring` (step-0 proved the install + spike + 0.92 fidelity for one such matrix). The ONLY genuinely
data-dependent op is **self-attention** — softmax(QKᵀ/√d) is a per-token-pair routing weight, not a fixed synaptic
matrix. The decisive finding from the SOTA: **the only spiking LM at scale (SpikingBrain-7B) does not run softmax in
spikes at all — it replaces attention with a gated-linear / sliding-window form whose running-state recurrence maps
*directly* onto a recurrent spiking accumulator**, and the spike-native vision transformers (Spike-driven Transformer,
SDSA) replace softmax with **binary coincidence-AND + column-sum + threshold-gated masking of V** — a multiply-free
primitive the bridge *already implements and uses* (the `coincidence_detector` dendritic-AND path,
`bridge.py:2640-2662` + the `_phaseB_onebrain_sequencer` per-word coincidence-AND match). So attention on the bridge is
**not** an open research wall: it is a **small, additive, guarded `sim/` step-block** (one of two ranked forms — a gated
linear-attention accumulator, or an SDSA-style coincidence-mask), justified, byte-reviewable, and default-off in the
transmission-gate/dendritic-gain precedent style. **Q2 (signed weights)** is real but mechanically solved by the
bridge's existing E/I split (`bridge.py:6084-6126`) — split each trained projection into an excitatory + an inhibitory
source population, the documented ANN→SNN signed-weight convention; the step-0 "positive-only" residual is a
probe-shortcut, not a bridge limit. **Q3 (multi-layer fidelity)** is the largest *uncertainty* (step-0 measured 1 layer
+ one-hot; 4-layer real-activation error accumulation is unmeasured) and gets the first, cheapest de-risk. **Q4
(dataflow/timing)** is a layered feed-forward wave over bridge ticks (N rate-window ticks per layer per token), the
exact pattern the convert-GO ran in PyTorch unrolled-over-T and the project's serial-order runners already prototype.
**Verdict: feasible; the full-generator consolidation needs ONE bounded `sim/` edit (the attention step-block) plus the
already-de-risked install/calibration machinery; the cheapest-first ladder de-risks the two NO-`sim/`-edit residuals
(multi-layer + signed E/I, on an MLP-only slice) BEFORE the attention edit, so the edit is only committed once the rest
is proven.**

---

## 1. Where this step sits, and the ONE thing the prior scopings did NOT analyze

The parent frontier scoping (`2026-06-22-generative-sequence-frontier-scoping.md` §4.2) named the C1 hard part as the
**LIF↔Izhikevich/AdEx/RF dynamics gap** — and step-0 RESOLVED that gap for a single feedforward layer (a global
synaptic-gain calibration recovers 947 active hidden, Spearman 0.918 vs the off-bridge LIF, specificity margin 0.893 —
`2026-06-22-genseq-step0-C1-consolidation-GO.md` §Result). The convert scoping
(`2026-06-22-genseq-convert-scoping.md` §0-1) then established that Gen-F is a *standard PyTorch decoder-only GPT*
(`tiny_transformer.py:35`), and convert-GO (`2026-06-22-genseq-convert-GO-spiking-generates.md`) confirmed the FULL
spiking `TinyGPT` generates coherent novel text **in PyTorch** at T=32 (+6% ppl), with `nn.MultiheadAttention`
hand-reimplemented (split Q/K/V, per-head SDPA, causal mask) so the internal softmax could be rate-quantized.

**But all three stopped short of the load-bearing question for THIS step:** the parent scoping's C1 analysis assumed the
generator was a **stacked-LIF MLP** (the `cortex_pretraining.py` net — the failed/overfit `cortex_10M_seed42.npz`
step-0 loaded), which has *no attention*. The actual generator Spine A converts is the **attention-based `TinyGPT`**.
So the parent's "LIF dynamics gap" is a *necessary* condition that step-0 met, but it is **not sufficient** for the full
generator: a transformer also requires running **data-dependent attention** on a fixed-connectivity substrate. That is
the genuinely new crux this scoping answers (Q1), and it was never in scope before. The convert-GO ran attention in
PyTorch SDPA — it was explicit (`§Next`) that the on-bridge realization of attention is "deferred to this consolidation
step." This doc is that step's scoping.

**One inherited correction the controller should carry:** Gen-F is **3.45M params**, not the "~6M" the older finding
rounds to (verified two ways in `2026-06-22-genseq-convert-scoping.md` §1, table row "Params (REALIZED)"). Does not
change any conclusion.

---

## 2. Q1 — ATTENTION ON A FIXED-CONNECTIVITY SPIKING BRIDGE (the crux)

### 2.1 Why this is the hard op (and the three sub-quantities)

A transformer block is `x + Attn(LN(x)); x + MLP(LN(x))`. Decomposed by data-dependence:

| Sub-op | Data-dependent? | Maps to a fixed synaptic matrix? |
|---|---|---|
| token/pos embeddings, Q/K/V projections, output projection, MLP `Linear`s, LM head | NO (fixed weights) | **YES** — static synapses, step-0 proved it (one matrix, 0.92 fidelity) |
| `LayerNorm` (per-token mean/var normalize) | partly (per-token statistics) | NO — but a feed-forward divisive-normalization circuit (the bridge HAS one) |
| `GELU` (pointwise nonlinearity) | NO (pointwise) | the neuron's own f-I curve / a calibrated activation, like step-0's gain |
| **softmax(QKᵀ/√d) · V** (self-attention) | **YES — the routing weight A_ij depends on the current token activations** | **NO — this is the crux** |

Attention has **two** nested data-dependent multiplies the bridge's fixed CSR cannot express directly: (i) the score
`S_ij = q_i · k_j` (a multiply of two activation vectors), and (ii) the readout `out_i = Σ_j A_ij v_j` (a multiply of
the data-dependent weight `A_ij` by `v_j`). On a fixed-connectivity point-neuron substrate, *neither* is a static
synaptic weight — both are runtime products of live activations. THIS is what "fixed-connectivity bridge can't do
attention naïvely" means.

### 2.2 What the SOTA actually does (fresh June-2026 pass, verified to primary sources)

The spiking-transformer / neuromorphic field has converged on **three** families; the key fact is that **the capable
LM-scale systems do NOT compute softmax in spikes** — they change the *attention operator* so it stops being
data-dependent O(N²) routing:

1. **Spike-native attention — Spikformer SSA + Spike-driven SDSA** (Zhou et al. ICLR 2023, arXiv 2209.15425; Yao et al.
   NeurIPS 2023, arXiv 2307.01694). They DROP softmax (spikes are non-negative, so the non-negativity softmax enforces
   is free). **SDSA = `SN(Σ_c(Q_S ⊗ K_S)) ⊗ V_S`** where `⊗` is the **Hadamard (element-wise AND of binary spikes)** and
   `Σ_c` is a column-sum — i.e. **binary coincidence-AND → integer column-sum → threshold to a binary gate → mask V**.
   *No real-valued multiply; the multiply is replaced by binary AND + addition.* O(N·D), not O(N²). Reported cost:
   ImageNet 77.1% (vs ANN ~80–87%), **vision only** — never shown at word-LM scale.

2. **Linearized attention — SpikingBrain-7B + gated linear attention** (CAS/BICLab 2025, arXiv 2509.05276;
   Katharopoulos et al. 2020, arXiv 2006.16236). The only spiking LM at scale **replaces attention with Gated Linear
   Attention (GLA) + a thin local sliding-window**, whose autoregressive recurrence is a **running matrix-valued state**:
   `S_t = diag(g_t)⊙S_{t-1} + k_tᵀ v_t`, `o_t = q_t S_t`. This is **O(N), O(1) state/step, and maps DIRECTLY onto a
   recurrent spiking accumulator** (S_t = a slowly-updated synaptic/membrane buffer the substrate updates every tick).
   SpikingBrain upcycled Qwen2.5-7B, **reused the pretrained Q/K/V weights verbatim**, and recovered ~90% of the source
   on ~2% of from-scratch compute. The small-scale recall gap of pure linear attention (Hedgehog, ICLR 2024 arXiv
   2402.04347: ~9.6 ppl at 125M from scratch) is **fixed by a thin local exact window** (BASED, ICLR 2024 arXiv
   2402.18668: linear + a ~64-token exact-softmax window ≈ matches full attention; SpikingBrain uses LA+SWA 1:1).

3. **Exact-softmax conversion — SpikeZIP-TF / division-neuron operators** (You et al. ICML 2024, arXiv 2406.03470;
   plug-and-play spiking operators 2026, arXiv 2605.20289). KEEP exact softmax, emulate it with an auxiliary spiking
   circuit over a T-step window (accumulate-apply-difference, or a "division neuron group" decoding the quotient by
   counting how many ordered-threshold neurons fired). **Stays genuinely O(N²) data-dependent**, needs a quantized ANN
   first and T=16–128 steps. Most faithful (<1% drop), heaviest, worst fit for a fixed-connectivity substrate.

**The multiply primitive the field relies on, and that the bridge ALREADY HAS:** rate×rate multiply is unreliable on
point neurons (the project's documented limit), but **binary-spike coincidence-AND is exact and free** — which is
*precisely why* SDSA reformulates attention to need only AND + addition. And the bridge already implements
coincidence-AND: the `coincidence_detector=True` pathway flag routes synapses through a **dendritic-coincidence plateau**
(`bridge.py:2640-2662`, `cp_coincidence_synapse_mask`, the supralinear `>= coincidence_k_threshold` switch), and the
project ALREADY USES per-word coincidence-AND as a data-dependent *match* gate in `_phaseB_onebrain_sequencer_derisk.py`
(lines 26-27: "a per-word COINCIDENCE-AND … `coinc[w]` needs BOTH the cue line AND the decoded line on word w, each
alone subthreshold"). So the AND primitive SDSA needs is on-substrate and validated.

### 2.3 The ranked Q1 options for THIS substrate (one allowed `sim/` step-block, minimized)

| Rank | Option | Mechanism on the bridge | `sim/` edit? | Cost / risk |
|---|---|---|---|---|
| **#1** | **Gated linear-attention accumulator** (SpikingBrain/Katharopoulos/GLA) | Q/K/V/out projections install as static synapses (step-0 path). A **new step-block** maintains the running state `S_t` (a per-channel state array updated by the gated outer-product `g·S_{t-1} + φ(k_t)v_tᵀ`) and reads `out_t = φ(q_t)ᵀS_t`. Rides the substrate's natural temporal accumulation; **no N×N matrix, no rate×rate multiply** if φ(k),v feed a per-channel buffer (the read is a matvec against the live state). | **YES — small, additive, guarded, default-off.** One step-block (the gated outer-product accumulator + read), mirroring the transmission-gate / dendritic-gain / graded-plateau additive-block precedents. | Lowest kernel footprint; ~2–10 ppl small-scale, recoverable with a thin local window. **BUT** it requires *re-architecting Gen-F's softmax attention into GLA* — i.e. it's a different attention operator than the trained `nn.MultiheadAttention`, so Gen-F's weights are NOT verbatim-reusable; a conversion/finetune step is needed. Highest *capability* fidelity to a recurrent substrate, lowest *weight-reuse* fidelity to Gen-F. |
| **#2** | **SDSA-style coincidence-mask attention** | After Q/K/V projections (static synapses), an additive step-block does the SDSA op: binary `Q_S ⊗ K_S` (= the existing coincidence-AND, `cp_coincidence_synapse_mask`) → column-sum → threshold → a binary gate masking V (= the existing `transmission_gate` per-synapse current scaling, `bridge.py:5988-5994`, driven by the gate spikes). **Reuses two primitives the bridge already has** (coincidence-AND + transmission-gate); the step-block is the column-sum-and-threshold glue. | **YES — small**, or possibly **NONE** if expressed purely as a coincidence pathway + a result-driven transmission gate (the `_phaseB_onebrain_sequencer` pattern, which is NO `sim/` edit). The column-sum-and-threshold is the only candidate kernel addition. | Multiply-free, uses native coincidence. **BUT** SDSA's "attention" degenerates to a per-channel input-gated selector on V — it is barely token-to-token routing; vision-proven, *unproven at word-LM* (a real risk for a generative LM that needs genuine token-mixing). Lower capability ceiling than #1. |
| **#3** | **Exact-softmax conversion step-block** (SpikeZIP-TF / division-neuron) | Keep Gen-F's `nn.MultiheadAttention` weights verbatim; a step-block emulates softmax over a T-step window (the convert-GO already built this in PyTorch — `2026-06-22-genseq-convert-GO`: "rate-quantized to T levels over calibrated ranges"). | **YES — larger.** A T-step accumulate-apply-difference / ordered-threshold division block; the heaviest of the three. | **Verbatim Gen-F weight reuse (highest fidelity to the proven generator)**, <3% ppl per convert-GO. **BUT** stays O(N²), needs T=16–32 steps, the largest kernel; and the convert-GO already proved this works *in PyTorch* — porting that exact spiking-softmax to a bridge step-block is the most faithful but most code. |

**The honest Q1 ranking depends on a fork the owner sets:** *weight-reuse fidelity to Gen-F* (→ #3, port the
already-validated PyTorch spiking-softmax) **vs** *substrate-naturalness + smallest kernel* (→ #1, GLA accumulator, but
re-architect+finetune attention). **#2 is the cheapest-edit middle** (reuses coincidence + transmission-gate) but has
the lowest capability ceiling and is unproven for generation. **Is a `sim/` edit required? YES for the full
attention-based generator** — there is no way to express data-dependent attention purely through the fixed CSR without
*some* runtime step-block. The minimal honest edit is **one additive, guarded, default-off step-block** (any of the
three), exactly the owner-OK'd "justified + byte-reviewed" `sim/` edit category, and exactly the precedent set by the
transmission-gate (2026-06-03), the coincidence plateau (2026-06-09), the graded-plateau (2026-06-20), and the RF
megakernel mask (2026-06-18) — all additive, all default-off-byte-identical, all `sim/`.

> **[VERIFY — most load-bearing]** That option #2 can be expressed with **NO** `sim/` edit (pure coincidence-pathway +
> result-driven transmission-gate, the `_phaseB_onebrain_sequencer` pattern) — if so, the cheapest attention de-risk is
> NO-edit. The controller should confirm whether the column-sum-and-threshold can be realized as an Izhikevich
> subnetwork on `cp_connections` (a fixed pooling layer + a threshold neuron) rather than a kernel. The sequencer runner
> strongly suggests YES, but it does per-word match, not the full QKᵀ column-sum — the difference is the open question.

---

## 3. Q2 — SIGNED WEIGHTS via the bridge's E/I machinery

**The step-0 "positive weights only" residual is a probe shortcut, not a bridge limit.** The step-0 probe explicitly
dropped negative weights (`_genseq_step0_bridge_load_probe.py:204-211`: "Keep only positive weights for a pure-
excitatory test … Negative-weight handling is a named downstream conversion concern"). A trained transformer's
projections are dense signed matrices, so signed routing is mandatory for the real generator.

**The bridge already routes signed E/I**, verified in the forward step (`bridge.py:6084-6126`): when
`enable_inhibitory_neurons` and `cp_traits` are set, the firing vector is split into excitatory and inhibitory sources
(`exc_fired_prev = prev_fired_float * (~is_inhibitory_neuron_output)`, `inhib_fired_prev = ... * is_inhibitory`), and the
two are matvec'd separately into `g_e` (with `propagation_strength`) and `g_i` (with `inhibitory_propagation_strength`),
where `g_i` drives toward `E_inh = -75 mV` (hyperpolarizing). The inhibitory trait is set per-source-neuron, and
`inject_explicit_wiring` already supports flipping a population to inhibitory via `output_inhibitory_indices`
(`bridge.py:2743-2752`).

**The standard ANN→SNN signed-weight convention maps cleanly onto this:** split each source feature into a **positive
channel (excitatory neuron)** and a **negative channel (inhibitory neuron)**; a weight `W_ij > 0` wires the excitatory
copy, `W_ij < 0` wires the inhibitory copy (with `|W_ij|`). This is the textbook signed-weight realization in
conductance-based SNNs and is exactly what the bridge's trait-split + dual-conductance step implements — **no `sim/`
edit**, reuse-only. The cost is a ~2× neuron count for the signed layers (each feature gets an E and an I copy), which
at d=256/L=4 is trivially within budget.

**One named subtlety (real, but mechanical):** the bridge's E/I is **per-source-neuron**, not per-synapse — a single
source neuron is *either* excitatory *or* inhibitory for ALL its outgoing synapses. A transformer weight matrix has both
signs in the same row (one source feeds some targets positively, others negatively). The split-channel convention
handles this: the source feature is *duplicated* into an E-copy and an I-copy (both driven identically by the upstream
activation), and the downstream wiring chooses which copy each signed weight reads. **Q2 verdict: solved by the existing
E/I split + the split-channel ANN→SNN convention; no `sim/` edit; the only cost is feature duplication. This is the
first thing to de-risk alongside multi-layer (ladder step #1), since step-0 never tested it.**

> **[VERIFY]** That the dual-conductance E/I (driving-force-dependent `g·(V−E)`, not a pure linear sum) reproduces a
> *linear* signed matmul closely enough. The trained transformer sums `Σ W_ij x_j` linearly; the bridge's synaptic
> current is `g·(V−E)` (sub-linear near reversal). Step-0's global-gain calibration handled the excitatory case (0.92);
> the signed case needs the same calibration to confirm the E/I balance reproduces the signed sum. Measured directly in
> ladder step #1.

---

## 4. Q3 — MULTI-LAYER FIDELITY (the largest uncertainty)

**Step-0 measured ONE layer with a ONE-HOT input** (`_genseq_step0_bridge_load_probe.py:277-281`: "layer 0 (input 66 →
hidden 2048) … a one-hot char drives exactly one input row of W"). It did NOT measure: (a) **error accumulation across
4 stacked layers**, where layer L's spiking-rate approximation error feeds layer L+1's input; (b) **real (non-one-hot)
dense activations** — a one-hot input is the easiest possible case (one active weight column); a mid-network layer sees
a dense graded activation vector where the rate-code approximation is harder; (c) **the LayerNorm + residual structure**
that couples the layers (a transformer is pre-norm residual, not a plain MLP stack).

**This is the genuine open uncertainty of the consolidation.** The SOTA conversion literature reports *whole-network*
perplexity (GPT-2-small ~10% ppl naïve, ≤3% with LAS/MBE — `2026-06-22-genseq-convert-scoping.md` §0), which IS a
multi-layer end-to-end number — so the field's evidence says 4-layer transformer conversion holds at ≤3% ppl. **BUT
that evidence is for the PyTorch spiking forward (which convert-GO reproduced at +6% ppl, T=32), NOT for the BRIDGE's
Izhikevich/AdEx/RF dynamics.** Step-0 proved one layer survives the bridge dynamics at 0.92; whether 4 layers + signed
E/I + the attention block compound to a usable end-to-end generation is **unmeasured and is the load-bearing de-risk**.

**Project machinery available:** the layer-by-layer fidelity metric step-0 already built (`offbridge_layer_rates` vs
`onbridge_post_rates`, Spearman/Pearson/top-k — `_genseq_step0_bridge_load_probe.py:76-111`) extends trivially to
stage-2/3/4 (drive layer L's measured on-bridge rates into layer L+1, compare to the off-bridge `forward_unroll` at
that layer). The per-layer threshold-balance (the SOTA's standard refinement; step-0 used a single global gain and
flagged "per-layer threshold-balance is the standard refinement" — §Honest caveats (c)) is the cheap fix if a single
global gain doesn't hold across 4 layers.

**Q3 verdict: the LARGEST uncertainty, gets the FIRST cheap de-risk (ladder step #1 — extend step-0 to multi-layer +
signed on an MLP-only slice, NO attention yet). The hypothesis (per the SOTA's ≤3% whole-net conversion + step-0's
per-layer 0.92) is that it holds with per-layer threshold-balance; it is a hypothesis, not a result.**

---

## 5. Q4 — DATAFLOW / TIMING (feed-forward wave over bridge ticks)

**The mismatch:** the bridge steps ALL neurons every tick (`_run_one_simulation_step`), a recurrent dynamical system; a
transformer is a **feed-forward per-token forward pass** (layer 0 → 1 → 2 → 3 → head, for each token position). How does
a per-token forward map onto bridge stepping?

**The answer (three mechanisms, all already in the project's vocabulary):**

1. **Rate-window-per-layer (the convert-GO's T, ported to ticks).** The convert-GO ran the spiking forward "over T
   steps" (T=32 the GO knob — `2026-06-22-genseq-convert-GO` §Result), where each nonlinear op is "rate-quantized to T
   levels." On the bridge this is: drive the input layer for a rate-window of `T` ticks, let the layered synapses carry
   the spike-rate wave forward, and read the head's spike-rate after `~L·(window)` ticks. Step-0 already does exactly
   this for one layer (`onbridge_post_rates`: warmup + n_steps, count spikes — `_genseq_step0_bridge_load_probe.py:229-261`,
   with `n_steps=64, warmup=16`). The whole forward is `L` such windows pipelined as a wave (the layered disjoint-slice
   structure means each tick advances all layers' current state, so after the pipeline fills, one token's logits emerge
   per window).

2. **Layered disjoint slices = a natural feed-forward wave.** Because the generator installs as disjoint index slices
   (layer 0 neurons, layer 1 neurons, … — the roadmap-step-2 co-residence pattern, `2026-06-22-frontier-scoping` §4.1),
   and synapses only connect layer L → layer L+1 (feed-forward, no back-edges), the bridge's all-neuron step IS a
   layered wave: each tick, layer L's spikes drive layer L+1's conductances, which spike next tick. After `L` ticks the
   wave reaches the head. This is automatic from the wiring topology — no special scheduling, no `sim/` edit (the same
   way nav+conv+composer co-reside as disjoint slices).

3. **Per-token autoregression = re-drive the input slice per token.** A decoder LM generates one token at a time
   (`TinyGPT.forward` is called per growing context). On the bridge: drive the token+positional input slice with the
   current token, run the L-layer wave, read the head's argmax-over-spike-rate, append, repeat. The KV-cache /
   attention-over-past-tokens is what the Q1 attention block handles (the GLA running-state #1 *is* the KV-cache as a
   persistent accumulator state — it naturally carries across tokens, which is the recurrent substrate's strength).

**Project prior art for the wave:** the serial-order runners (`_phaseB_onebrain_sequencer_derisk.py`, the
`neural_serial_order_renderer` CQ generator, the `_genseq_convert_derisk` unroll-over-T) all already prototype
multi-stage spiking propagation with rate-window reads. **Q4 verdict: a layered feed-forward wave over rate-window
ticks (the convert-GO's T realized as bridge ticks), automatic from the disjoint-slice feed-forward topology; the GLA
accumulator (#1) handles the cross-token KV-cache as a persistent state. NO `sim/` edit for the dataflow itself (only
the attention block, Q1).**

---

## 6. THE CHEAPEST-FIRST DE-RISK LADDER

> **Principle:** de-risk the TWO NO-`sim/`-edit residuals (multi-layer + signed E/I, Q3+Q2) on an **MLP-only slice
> FIRST**, BEFORE the attention `sim/` edit (Q1) — so the edit is only committed once everything around it is proven.
> Every step has a GO/NO-GO. Cheapest-first; CuPy for decisive runs, numpy for tiny smoke
> (`feedback_gpu_not_numpy`); ≥6 seeds for variable claims (`feedback_6seed_validation`); the no-confab moat asserted
> intact at every step (the generator generates; the conversational retrieval/moat layer still abstains).

| # | Step | Scale / cost | What it PROVES | `sim/` edit? | GO / NO-GO |
|---|---|---|---|---|---|
| **1** | **Multi-layer + SIGNED on an MLP-only slice** (the two named step-0 residuals, NO attention). Install the 2-layer MLP sub-block of ONE Gen-F transformer block (`Linear(d,4d)→GELU→Linear(4d,d)`, ~256→1024→256) onto the bridge as **signed E/I split-channel** populations (excitatory + inhibitory copies, Q2), drive with REAL (non-one-hot) activations sampled from the off-bridge forward, extend step-0's per-layer Spearman metric to stage-2, with **per-layer threshold-balance**. | **hours**, 1×3090 (or numpy smoke), NO training | (a) signed E/I reproduces a signed linear matmul (Q2); (b) 2-layer fidelity holds with real dense activations + error accumulation (Q3 — the largest uncertainty); (c) per-layer threshold-balance is the right refinement. | **NONE** (reuse-only: `inject_explicit_wiring` + E/I split + `output_inhibitory_indices`). | **GO** if stage-2 Spearman ≥ ~0.8 (step-0's per-layer bar) AND the signed E/I tracks the off-bridge signed sum (specificity margin holds) → multi-layer + signed are NO-`sim/`-edit-solved, proceed to attention. **NO-GO** if signed E/I or stage-2 fidelity collapses → fix calibration / per-layer balance BEFORE any attention work (cheap to learn now). |
| **2** | **The ATTENTION-mechanism de-risk** (Q1, the crux). Build the chosen attention form on a SMALL attention slice (1 head, short context, the trained Q/K/V/out weights). **Try the NO-`sim/`-edit option #2 FIRST** (coincidence-pathway + result-driven transmission-gate, the `_phaseB_onebrain_sequencer` pattern) — does an SDSA-style coincidence-mask reproduce the off-bridge attention output above chance? If NO-edit option #2 fails the generation bar, escalate to the **additive guarded step-block** (option #1 GLA accumulator or option #3 spiking-softmax port). | **hours–day**, 1×3090 | the data-dependent attention op runs on the bridge: on-bridge attention output tracks the off-bridge SDPA on a held-out calibration set within the conversion tolerance. | **NONE for option #2 attempt; ONE additive/guarded/default-off step-block** if it must escalate to #1 or #3 (the owner-OK'd, byte-reviewed category; transmission-gate/coincidence/graded-plateau/RF-megakernel precedent). | **GO** if on-bridge attention reproduces off-bridge attention rank-order on the calibration set (the genuine token-mixing test — NOT just "spikes"). **NO-GO** if coincidence-mask degenerates (SDSA's known per-channel-selector failure mode for generation) → escalate to the step-block; if the step-block also fails → the attention-conversion fidelity work (raise T, finetune) before the full forward. |
| **3** | **The FULL generator forward on the ONE bridge** (C1 complete). Install all 4 blocks (embeddings + signed MLPs + attention blocks + LN + head) as a co-resident disjoint slice; run the per-token autoregressive wave (Q4); compare on-bridge generation to off-bridge (the convert-GO's PyTorch spiking generation) within the conversion tolerance; run the **byte-unmodified Gen-F gate** (held-out ppl < vocab, word-shuffle, verbatim-copy, novelty); assert the conversational no-confab moat byte-intact (`test_nav_conv_step2b_coresident`). | **days**, 1×3090 | C1: the **full attention-based generator generates coherent novel text AS SPIKES on the ONE bridge**, co-resident with conv/nav, moat intact. | **NONE beyond step #2's one block** (reuse the install + the one attention step-block). | **GO** if on-bridge generation matches off-bridge within ~10–20% ppl (the SOTA conversion tolerance, widened for the bridge dynamics gap) AND clears the Gen-F gate AND the moat asserts byte-intact. **NO-GO / HONEST NEGATIVE** if multi-layer + attention + signed compound past the tolerance → the post-conversion surrogate-grad-on-bridge finetune (the deeper guarded edit, parent scoping §4.2 option 3) OR the honest "fidelity wall" finding (a real deliverable). |

**Why this order:** step #1 settles the two residuals step-0 explicitly deferred (signed + multi-layer) with **zero
`sim/` edit** — if they fail, the attention edit is moot and we've spent only hours. Step #2 isolates the single crux
(attention) and tries the NO-edit form first, committing the `sim/` edit only if forced. Step #3 is the integration,
gated on both. This is the cheapest-first gate discipline: each step's NO-GO routes to a cheaper fix before the next
investment.

---

## 7. THE RANKED PLAN + the `sim/`-EDIT VERDICT + GO/NO-GO per step

**Ranked plan (cheapest-first, = the ladder):** (1) multi-layer + signed MLP-only slice [NO edit, hours] → (2)
attention-mechanism de-risk, NO-edit option #2 first then escalate [≤1 additive block, hours–day] → (3) full-generator
forward + Gen-F gate + moat [reuse, days]. Spine A's loop-step 3 is COMPLETE at step #3 GO.

**The `sim/`-edit verdict (the load-bearing call):**

- **Steps #1 (signed + multi-layer) and Q4 (dataflow) need NO `sim/` edit** — reuse-only (`inject_explicit_wiring`, the
  E/I split, `output_inhibitory_indices`, disjoint-slice co-residence, rate-window reads). These are the two residuals
  step-0 named, and both are mechanically solved by existing machinery.
- **Step #2 (attention) needs, at most, ONE small additive guarded default-off `sim/` step-block** — and possibly
  ZERO if the SDSA coincidence-mask (option #2) can be expressed as an Izhikevich subnetwork on `cp_connections` +
  a result-driven transmission-gate (the `_phaseB_onebrain_sequencer` pattern; **[VERIFY]**). The minimal edit, if
  needed, is the owner-OK'd category: additive, default-off-byte-identical, byte-reviewable, in the exact precedent of
  the transmission-gate (2026-06-03), coincidence-plateau (2026-06-09), graded-plateau (2026-06-20), and RF-megakernel
  mask (2026-06-18). **There is NO way to run data-dependent attention on a fixed CSR without *some* runtime step-block
  — so for the full attention-based generator, the honest answer is "one bounded `sim/` edit is likely required, of the
  already-precedented additive-guarded kind."**
- **The escalation edit (parent §4.2 option 3 — surrogate-grad-on-bridge finetune)** is the fallback ONLY if step #3's
  end-to-end fidelity fails the tolerance; it is the deeper edit (a LIF/AdEx-LIF forward consistent with `bptt_snn_gpu`)
  and is NOT on the critical path unless the cheap calibration loses too much.

**GO/NO-GO per step:** as tabled in §6 — each is a decisive gate. The whole step is GO when #3 is GO (full generator
generates coherent novel text as spikes on the one bridge, within conversion tolerance, moat byte-intact). An honest
NEGATIVE at #3 (the multi-layer + attention + signed dynamics gap compounds past tolerance) is itself the deliverable —
a measured "fidelity wall for transformer consolidation on the point-neuron substrate," routing to the surrogate-grad
finetune or a documented limit.

**The genuine fork the owner should set (surfaced, not chosen):** the attention option (#1 GLA / #2 SDSA / #3
spiking-softmax) trades *Gen-F weight-reuse fidelity* (→ #3, port the already-validated convert-GO spiking-softmax,
verbatim weights, largest edit) against *substrate-naturalness + smallest edit* (→ #1 GLA accumulator, re-architect +
finetune attention, smallest kernel but Gen-F weights not verbatim). #2 is the cheapest-edit middle with the lowest
capability ceiling. **Recommendation: in step #2, try #2 (NO-edit) first as the cheap probe; if it clears the
token-mixing bar, it's the cheapest path; if not, escalate to #3 (port the proven convert-GO softmax — highest fidelity
to the already-GO generator) rather than #1 (which adds a re-architecture+finetune on top of the edit).** This is a
genuine fork; the owner sets it.

---

## 8. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / source read):**
- **The crux is attention, and the prior scopings did NOT analyze it on the bridge** — parent scoping §4.2 analyzed a
  stacked-LIF MLP (no attention); convert-GO ran attention in PyTorch SDPA and deferred on-bridge attention to "this
  consolidation step" (`2026-06-22-genseq-convert-GO-spiking-generates.md` §Next, read in full).
- **Gen-F IS `nn.MultiheadAttention` + LN + GELU** (`sim/tiny_transformer.py:15,25-32,44-64`, read in full) — the
  attention is real, the convert-GO confirms it generates spiking at T=32.
- **The bridge E/I split** (`bridge.py:6084-6126`) routes excitatory/inhibitory sources to `g_e`/`g_i` separately;
  `inject_explicit_wiring` supports `output_inhibitory_indices` (`bridge.py:2743-2752`) — Q2's machinery, verified.
- **The bridge HAS coincidence-AND** (`coincidence_detector` flag → `cp_coincidence_synapse_mask` → dendritic plateau,
  `bridge.py:2640-2662`, `2449,2461-2462,2474`) and the **transmission-gate** (per-synapse current scaling at runtime,
  `bridge.py:5988-5994, 3179-3203`) — the two primitives SDSA attention (#2) needs, verified.
- **The project ALREADY uses per-word coincidence-AND as a data-dependent match + BG control-flow conditioned on
  spiking results** (`_phaseB_onebrain_sequencer_derisk.py:18-40`, read) — the strongest evidence the AND-based attention
  primitive is on-substrate and the NO-edit option #2 is plausible.
- **The graded analog path** (drive conductance from source membrane potential continuously, `bridge.py:6128-6175`) and
  **the divisive-normalization circuit** (`bridge.py:6179-6199`) — relevant to LayerNorm-on-bridge (a feed-forward
  divisive normalize), verified to exist.
- **The RF complex synapse multiply** (`rf_set_complex_weights`/`_rf_advance_one`, `bridge.py:5691-5747`) — a genuine
  multiplicative primitive (complex matvec = phase sum), available if a phasor-coded attention variant is ever wanted;
  noted, not on the critical path (Gen-F is rate-coded, not phasor).
- **Step-0's residuals** (positive-weight-only `_genseq_step0_bridge_load_probe.py:204-211`; one-hot single-layer
  `:277-281`; global-gain not per-layer `:298-335`) — the exact gaps Q2/Q3 close, verified in the probe source.
- **The dataflow primitive** (`onbridge_post_rates` rate-window-and-count, `:229-261`) — Q4's per-layer wave, verified.
- **SOTA** (fresh June-2026 pass, primary sources verified by the deep-research sub-investigation): SDSA = binary
  Hadamard-AND + column-sum + mask (Spike-driven Transformer, arXiv 2307.01694, Eq. 13); SpikingBrain-7B = GLA + SWA,
  reuses Qwen Q/K/V verbatim, ~90% recovery on ~2% compute (arXiv 2509.05276); GLA/linear-attention running-state =
  recurrent accumulator (Katharopoulos arXiv 2006.16236); small-scale linear-attention recall gap fixed by a thin local
  window (Hedgehog arXiv 2402.04347, BASED arXiv 2402.18668); exact-softmax conversion keeps O(N²) (SpikeZIP-TF arXiv
  2406.03470; division-neuron operators arXiv 2605.20289); coincidence-AND is the reliable point-neuron multiply.

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing]** That SDSA-style coincidence-mask attention (option #2) can be expressed with **NO**
   `sim/` edit (pure coincidence-pathway + result-driven transmission-gate, the sequencer pattern) — if YES, the
   cheapest attention de-risk is NO-edit. The sequencer does per-word match, not the full QKᵀ column-sum; whether the
   column-sum-and-threshold realizes as an Izhikevich subnetwork vs a kernel is the open question (ladder step #2).
2. **[VERIFY — the largest uncertainty]** That 4-layer + signed E/I + the attention block compound to a usable
   end-to-end generation on the BRIDGE dynamics (not just the PyTorch spiking forward convert-GO proved). Step-0 proved
   1 layer at 0.92; multi-layer error accumulation on Izhikevich/AdEx/RF is unmeasured (ladder step #1 + #3 measure it).
3. **[VERIFY — Q2 linearity]** That the dual-conductance E/I (`g·(V−E)`, sub-linear near reversal) reproduces a *linear*
   signed matmul closely enough after global/per-layer gain calibration (ladder step #1).
4. **[VERIFY — the fork]** Whether the owner wants attention option #2 (cheapest edit, lowest ceiling), #3 (verbatim
   Gen-F weights, port the proven convert-GO softmax, largest edit), or #1 (smallest kernel but re-architect+finetune).
   This sets the `sim/`-edit scope and is a genuine fork the owner should confirm.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/findings/2026-06-22-generative-sequence-frontier-scoping.md` (parent §4.2 LIF gap, §6 loop; treated generator as stacked-LIF MLP — read in full).
- `research/findings/2026-06-22-genseq-convert-scoping.md` (Gen-F = standard GPT, 3.45M, ≤3% convert cost — §0-1 read).
- `research/findings/2026-06-22-genseq-convert-GO-spiking-generates.md` (FULL spiking TinyGPT generates at T=32 in PyTorch; attention reimplemented; on-bridge deferred to THIS step — read in full).
- `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md` (1 layer installs + spikes + 0.92; positive-only + one-hot + global-gain residuals — read in full).
- `sim/tiny_transformer.py` (`TinyGPT`, `nn.MultiheadAttention`, LN, GELU — `:11-64`, read in full).
- `sim/bridge.py`: `inject_explicit_wiring` (`:2393`, plan format incl. `coincidence_detector`/`graded`/`transmission_gate` `:2400-2475`); E/I split (`:6084-6126`); coincidence plateau (`:2640-2662`); transmission-gate (`:5988-5994`, `:3179-3203`); graded analog path (`:6128-6175`); divisive-norm (`:6179-6199`); RF complex synapses (`:5691-5747`); `output_inhibitory_indices` (`:2743-2752`).
- `research/runners/_genseq_step0_bridge_load_probe.py` (positive-only `:204-211`; one-hot single-layer `:277-281`; gain sweep `:298-335`; rate-window read `:229-261`; per-layer metric `:76-111`).
- `research/runners/_phaseB_onebrain_sequencer_derisk.py` (per-word coincidence-AND data-dependent match + BG control-flow on spiking results, NO `sim/` edit — `:18-40`, read).

### Current literature (June 2026 pass, primary sources verified)
- **Spike-driven Transformer / SDSA** — Yao et al., NeurIPS 2023, arXiv 2307.01694 (SDSA = binary Hadamard-AND + column-sum + mask, Eq. 13; ImageNet 77.1%; vision only). Spikformer SSA — Zhou et al., ICLR 2023, arXiv 2209.15425.
- **SpikingBrain-7B** — CAS/BICLab 2025, arXiv 2509.05276 (GLA + SWA; reuses Qwen Q/K/V verbatim; ~90% recovery on ~2% compute; the only spiking LM at scale, does NOT run softmax in spikes).
- **Linear attention as recurrent state** — Katharopoulos et al. 2020, arXiv 2006.16236 (running-state RNN form; maps to a recurrent spiking accumulator). SpikeGPT — arXiv 2302.13939 (RWKV-recurrence spiking LM).
- **Small-scale recall fix** — Hedgehog, Zhang et al. ICLR 2024, arXiv 2402.04347 (~9.6 ppl gap, learned feature map); BASED, Arora et al. ICLR 2024, arXiv 2402.18668 (linear + thin local exact window ≈ full attention).
- **Exact-softmax conversion (stays O(N²))** — SpikeZIP-TF, You et al. ICML 2024, arXiv 2406.03470; plug-and-play spiking operators (division-neuron softmax), 2026, arXiv 2605.20289.
- **ANN→SNN transformer conversion (C1 tolerance)** — LAS arXiv 2505.09659; training-free MBE arXiv 2508.07710; GPT-2 conversion ~5–12% cosine / ~10% ppl naïve, ≤3% with LAS/MBE.
