# Generative-sequence frontier (Spine A) — SURPASSING the multi-layer DENSE consolidation RATE-SATURATION WALL — scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings re-verification at a CONFIRMED BOUNDARY (the standing research
> gate fired — `_genseq_loopstep3_multilayer` returned NEGATIVE, "rate-saturation wall," the rate-code boundary family
> the project has hit AND RESOLVED before). **NO `sim/` edits, NO experiments, NO GPU.** Single deliverable = this doc.
> Applies the project's SURPASS discipline: (1) ISOLATE + QUANTIFY the genuine residual, (2) REFRAME via real biology,
> (3) RANK cheap-first surpass mechanisms, (4) verdict surpassable-and-how-cheaply vs genuinely-irreducible. Builds on —
> does not re-derive — the consolidation scoping (`2026-06-22-genseq-loopstep3-consolidation-scoping.md`), step-0 GO
> (`2026-06-22-genseq-step0-C1-consolidation-GO.md`), and the NEGATIVE finding
> (`2026-06-22-genseq-loopstep3-multilayer-NEGATIVE-rate-saturation.md`). Every load-bearing claim re-verified against
> the repo (file:line). The controller should trust-but-verify the **[VERIFY]** items, push, and present before building.

---

## 0. One-paragraph answer (the rest is the evidence)

**The multi-layer dense consolidation is SURPASSABLE, the fix is CHEAP, NO `sim/` edit, and it is the *same* mechanism
the project already built to defeat this exact rate-code wall.** The genuine residual is **NOT** the signed split (layer-0
carries Spearman 0.32 = well above chance, so signed E/I routing is *faithful, only soft-degraded*) and **NOT** the
fan-in per se — it is **rate-readout saturation**: the runner reads the on-bridge SPIKE-RATE (`cp_firing_states`), which
is hard-capped at **0.5** by the 1-step refractory at `dt=1ms` (`bridge.py:6616`, `refractory_period_steps=1`), while
the off-bridge LIF reference runs to **1.0** and carries graded structure at **off_mean_rate ≈ 0.43**. With ~1000+
supra-threshold dense sources, every target pins at the 0.5 ceiling → the per-neuron signed-sum variation that should
rank the targets is crushed into a 1-bit "all firing" code → rank destroyed (cumulative 0.009). **The fix is to stop
reading the saturating spike-rate and read the GRADED / analog signal the bridge ALREADY transmits** — and the bridge
has a per-pathway **`graded=True`** transmission mode (`RegionPathway.graded`, `regions.py:355-372`; step block
`bridge.py:6128-6175`) that drives the next layer from the SOURCE's **continuous membrane state** `a_cont =
clip((v−rest)/scale, 0, 1)` instead of its binary spikes — **bypassing the refractory rate ceiling entirely**, with E/I
routing preserved, byte-identical-when-off, and **built for precisely this purpose** (its own code comment:
*"a SPIKING inhibitory pool cannot linearly track the population mean … but its graded membrane state can"* —
`bridge.py:6132-6134`). That is **#1, NO `sim/` edit**. The documented complementary lift is **population coding** (the
47%→100%-of-host rate-code-wall lift table, `2026-06-15-…GO.md` lines 106-109) — N neurons/feature so the population-mean
read does not saturate; also NO `sim/` edit. **Verdict: the dense-MLP consolidation is surpassable cheaply** — the
top-ranked de-risk is re-running ladder-step-#1 with `graded=True` on the hidden→hidden pathways and reading the graded
analog output (hours, no edit, no training). The point-neuron *spiking* limit is real (Mikulasch-Priesemann: rate codes
can't whiten/encode-densely pre-spike) — but the bridge's graded path is *exactly* the project's sanctioned analog
escape from it, and it has a prior GO defeating this same wall.

---

## 1. ISOLATE the genuine residual (SURPASS move 1) — it is the RATE READOUT, not the signed split, not fan-in

Three candidate causes were named in the prompt; the JSON pins which one is load-bearing.

### 1.1 The signed split is NOT the residual — it is faithful, only soft-degraded

Layer-0 (the signed E/I split-channel, one-hot driven) carries **Spearman 0.321 / Pearson 0.384 / top-k 0.563**
(`_genseq_loopstep3_multilayer.json` `per_layer_fidelity[0]`). The step-0 *positive-only* single layer was 0.918, so the
signed split **degraded 0.92 → 0.32** — a real loss, but **0.32 is ~10σ above chance** for a 2048-D rank correlation, and
per-input it is consistent (0.22 / 0.18 / 0.17 / 0.48 / 0.47 / 0.42 across the 6 probe chars). **The signed routing
WORKS** — the E/I split-channel reproduces a signed matmul's rank well enough to keep most of the structure. The 0.92→0.32
soft-degrade has two known, separable, *non-fatal* contributors (both calibration, neither a wall):
- **The `g·(V−E)` driving-force non-linearity** (the [VERIFY] the consolidation scoping flagged, §3): the bridge's
  synaptic current is `g·(V−E)`, sub-linear near reversal, vs the trained net's exact linear `Σ W_ij x_j`. The I-channel
  hyperpolarizes toward `E_inh=−75 mV` while the E-channel depolarizes toward `E_e=0 mV`; at the operating membrane these
  driving forces are asymmetric (the runner even pre-scales `i_gain` by the driving-force ratio, `:223-225`), but the
  *product* with the moving `V` is still only approximately linear.
- **Even at layer 0 the on-bridge rate is already mildly compressed** — `on_max_rate=0.75` vs off `1.0`
  (`per_layer_fidelity[0]`), i.e. the saturation is *beginning* even one-hot driven; it is just not yet total.

⇒ **Signed split: a soft, calibratable degrade, not the wall.** (Worth a separate cheap recovery — §2 lever (a2) — but it
is NOT what destroys the cumulative score.)

### 1.2 The residual IS the rate-readout saturation — quantified from the JSON

The collapse is **abrupt and total at the first dense hidden→hidden stage**, and the saturation diagnostic the runner
explicitly logged is the smoking gun:

| block | input | Spearman | on_mean_rate | on_max_rate | off_mean_rate | off_max_rate |
|---|---|---|---|---|---|---|
| 0 (66→2048, one-hot, signed) | sparse | **0.321** | 0.178 | **0.75** | 0.370 | 1.0 |
| 1 (2048→2048, dense) | dense | **−0.019** (chance) | **0.500** | **0.500 ← PINNED** | 0.436 | 1.0 |
| 2 (2048→2048, dense) | dense | **0.009** (chance) | **0.500** | **0.500 ← PINNED** | 0.432 | 1.0 |

(`per_layer_fidelity`, `cumulative_mean_spearman=0.009`, `specificity_margin=0.000`.)

The mechanism, pinned three ways:
1. **The ceiling is structural and exactly 0.5.** At `dt_ms=1.0` + `refractory_period_steps=1` (`runner :184,:215`), a
   fired neuron is forced into 1 refractory step (`bridge.py:6616` `new_refractory_for_fired = refractory_period_steps−1`
   = 0, but the fire-gate `not_in_refractory` at `:6583,:6601` blocks the immediately-following step) → a neuron can fire
   at most every **other** ms → **max spike-rate 0.5**. The off-bridge `bptt_snn.LIFLayer` has no refractory and fires up
   to **1.0** every step. So on/off are reading two different-ceiling instruments, and the on-bridge one's ceiling is half.
2. **Every dense-block target is driven supra-threshold, so EVERY one pins at 0.5.** `on_mean_rate = on_max_rate = 0.500`
   at blocks 1 and 2 — the mean EQUALS the max, i.e. **all 2048 targets fire at the ceiling**, identically. The
   per-neuron net-input *differences* that should rank the targets (the signed sum) all map above threshold, so they all
   produce the same saturated 0.5 → the rank information is quantized away. (`on_active = 2048/2048` at every dense block
   across the whole gain sweep once it fires — `global_gain_sweep` gain≥64.)
3. **The cheap calibration levers provably cannot un-pin it** (already exhausted in the NEGATIVE): global-gain 0.25→256×
   (block-1 jumps 0→2048 active with nothing in between — there is no gain where a *graded fraction* fires, because once
   any dense sum clears threshold, ~all do); per-block geometric-bisection threshold-balance drove block-1's incoming
   scale to the **0.2 lower bound** and it STILL showed 2048/2048 active (`per_layer_balance_calibration_log` block 1, all
   six probes `on_active=2048`); an I/E-ratio sweep ≤16× extra inhibition — block-1 stays pinned. **No amount of scalar
   re-balancing recovers a graded code from a 1-bit saturated readout.**

### 1.3 The one signal that SURVIVES (and tells us the wiring is fine) — top-k overlap

Crucial nuance the verdict must carry: **`top-k overlap` stays ~0.57 at the saturated blocks** (block 1 = 0.566, block 2
= 0.574; `per_layer_fidelity`). Chance top-k overlap for k≈1150 of 2048 is ≈ k/N ≈ 0.56 — so on a first read this is
"just chance." BUT note it does NOT collapse below chance, and the *identity* of which units are active is set by the
(faithful) signed wiring; what's destroyed is purely the **ordinal rank among the active set** (because they all pin to
0.5). **This confirms the wiring/signed-routing is intact and the loss is purely a readout-quantization loss** — exactly
the failure mode a graded (de-saturating) readout fixes. The single non-saturating layer (step-0, 0.92; and here layer-0
at 0.32 because it is only *mildly* saturated at 0.75) prove a non-saturating stage carries rank; the dense hidden stage
fails ONLY because its readout saturates.

**Isolated residual (the answer to SURPASS move 1):** the load-bearing failure is **rate-readout saturation at the
refractory ceiling under dense supra-threshold fan-in** — a *readout/representation* limit, NOT the signed split (faithful)
and NOT the fan-in topology itself (the fan-in causes the supra-threshold drive, but the *information loss* happens at the
0.5 spike-rate quantization, which a graded/analog or population readout does not suffer).

---

## 2. RANK the cheapest resolutions (cheapest-first) — each grounded in a PRIOR PROJECT WIN or SOTA

> Ordering principle: cheapest-first = NO-`sim/`-edit + NO-training first; each grounded in a verified prior win or
> primary-source SOTA; each with mechanism → grounding → cheapest de-risk → `sim/`-edit scope → expected fidelity ceiling.

### (a) GRADED / sub-threshold ANALOG transmission — **#1, NO `sim/` edit, built for THIS wall** ✅

- **Mechanism on the bridge:** flip the hidden→hidden (and signed) pathways to **`graded=True`**. The graded step block
  (`bridge.py:6128-6175`) drives each downstream conductance from the SOURCE neuron's **continuous membrane state**
  `a_cont = clip((v − graded_source_rest_mV) / graded_source_scale_mV, 0, 1)` (default rest −65 mV, scale 15 mV) — a
  saturating *sub-threshold* analog readout — **instead of its binary spikes**. These synapses are *removed from the
  spike matvec* and transmit gradedly (`bridge.py:6030-6037` builds `_graded_src_data` = the weights masked by
  `cp_graded_synapse_mask`, and the spike path skips them — mirroring the NMDA AMPA-suppression precedent). **E/I routing
  is preserved** (a graded inhibitory source still feeds `g_i` with `inhibitory_propagation_strength`, an excitatory
  source feeds `g_e` — `:6152-6175`), so the signed split-channel still works, only continuous-valued. **The readout
  must also be graded/analog** — read each layer's `a_cont` (the membrane-derived analog), NOT its spike count, so the
  off-bridge↔on-bridge comparison is analog↔analog (off-bridge `forward_unroll` already exposes the pre-spike
  membrane/activation, so the metric extends trivially — `bptt_snn.forward_unroll` returns the layer state, not only
  spikes).
- **Why it dissolves the wall:** the 0.5 refractory rate cap simply **does not apply** — `a_cont` is a continuous
  [0,1]-valued membrane readout with no quantization and no refractory ceiling, so a dense supra-threshold sum produces a
  *graded* `a_cont` near 1.0 that still **ranks** by net input (the saturation is a smooth `clip`, with the full graded
  range below it preserved — and `graded_source_scale_mV` is tunable to keep the operating point off the clip).
- **Prior-project-win grounding (verified):** the graded path was **built to defeat this exact rate-code wall**. Its code
  comment (`bridge.py:6132-6134`): *"This bypasses the spike threshold: a SPIKING inhibitory pool cannot linearly track
  the population mean (depol block makes its spikes anti-track it), but its graded membrane state can — the common-mode
  removal (whitening) a learned cortex needs."* It is the retina/horizontal-cell graded-release analogue
  (`regions.py:355-372`, *"the retina's horizontal-cell graded release"*; Kandel 6e Ch 22, graded potentials). It is the
  **same family of fix** the project's standing-practice cites — the Mikulasch-Priesemann point-neuron limit ("whitening
  is analog/pre-spike in biology") was resolved for the conversational cortex by moving the common-mode op into the
  analog stage. **The graded path IS that analog stage, on the bridge.**
- **Cheapest de-risk:** re-run the EXISTING ladder-step-#1 runner with the hidden→hidden + signed pathways tagged
  `graded=True` and the readout switched to the membrane-analog `a_cont` (the runner already builds wiring dicts —
  `:286-297` — so adding `"graded": True` to the block-1/block-2 entries + reading `cp_membrane_potential_v`-derived
  `a_cont` instead of `cp_firing_states` is a runner change, **NO `sim/` edit**). **Hours, 1×3090, no training.**
- **`sim/`-edit scope:** **NONE.** Reuse-only (`inject_explicit_wiring` + the per-pathway `graded` flag + the existing
  graded step block + `output_inhibitory_indices`).
- **Expected fidelity ceiling:** **HIGH** — this is the natural realization of a linear ANN layer on the substrate
  (analog membrane = the linear pre-activation; the `clip` is the only non-linearity, matching GELU's soft saturation
  roughly). Hypothesis: cumulative analog-Spearman recovers toward the step-0 single-layer 0.9 band per stage. The one
  honest caveat: `a_cont`'s `clip` upper-saturation at exactly 1.0 could re-introduce a *milder* saturation if the dense
  sum drives `v` far past `rest + scale`; the cheap mitigation is to widen `graded_source_scale_mV` and/or scale the
  incoming weights so the operating point sits in the graded band — a calibration the runner can sweep (and which, unlike
  the spike-rate ceiling, *exists* because the graded readout has headroom the 0.5 cap did not).

### (b) POPULATION coding — **#2, NO `sim/` edit, the documented 47%→100% rate-code-wall LIFT** ✅

- **Mechanism on the bridge:** represent each ANN feature with **N neurons** (a population), drive the whole population,
  and read the **population-mean spike-rate** per feature. Population averaging cancels the per-neuron quantization noise
  that bounds the single-neuron read; the population mean of many 0/0.5 spikers is a graded value with sub-0.5 resolution.
- **Prior-project-win grounding (verified — the decisive number):** the project's **documented rate-code-wall lift**
  (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` §"The population lift", lines 104-109): on the
  REAL spiking bridge, single-neuron read = **47%** of host; **n_per=8 → 100%**, **n_per=16 → 103%**, **n_per=32 → 108%**.
  *"the single-neuron ~50% plateau was indeed the documented rate-code SNR wall, and the population code lifts the
  read-out to the host reference"* (line 111-112). This is **the same ~50% plateau** as the 0.5 ceiling here, with the
  SAME fix. Also used as the generalization read-out (`2026-06-16-generalization-graded-propagation.md`, candidate 2 =
  population pooling).
- **Cheapest de-risk:** wrap each MLP feature in `n_per` neurons (the runner's `Layout` already maps feature→index; widen
  each feature to a block of `n_per`, replicate the incoming/outgoing wiring, read the block-mean rate). **Hours, no
  training, no edit.** Cheaper to *combine* with (a): population-of-graded is the belt-and-suspenders.
- **`sim/`-edit scope:** **NONE.** Pure wiring replication + a block-mean readout (the established population-code idiom).
- **Expected fidelity ceiling:** **HIGH for the readout** (100-108% of host in the prior win) — BUT note population mean
  of spikes still tops out at 0.5 *per neuron*, so under *fully*-saturating dense drive every population mean could pin at
  0.5 too. **The population lift works when the population spans a graded firing fraction** (some neurons in the pop fire,
  some don't, by heterogeneity/threshold spread). The prior win had that (the Hebbian codes were sparse). Under the
  dense, uniformly-supra-threshold drive HERE, the population needs **threshold/excitability heterogeneity** across its
  N neurons so a *graded fraction* fires — which the bridge supplies (`cfg.heterogeneity_distributions`, the het-band).
  ⇒ population coding is **#2, complementary to (a)**, and strongest when the population has heterogeneous thresholds so
  the dense drive recruits a graded fraction rather than all-or-none.

### (c) SURROGATE-GRAD-on-bridge finetune — **#3, the deeper guarded `sim/` use, only if (a)+(b) miss** ⚠️

- **Mechanism:** re-learn the on-bridge layer weights (or per-layer gains/thresholds) so the *spiking* forward — WITH the
  refractory ceiling — matches the teacher, via surrogate-gradient BPTT through the bridge dynamics. The infra exists:
  `sim/surrogate_grad.py` (ATan + fast_sigmoid surrogate), `sim/bptt_snn.py` (numpy reference forward+backward unroll
  with hard-reset surrogate + recurrent chain rule), `sim/bptt_snn_gpu.py` (CuPy port, validated fp32-equivalent — the
  CLAUDE.md Phase 2.1/2.2 wins, "loss 14.1→2.24, 84% reduction").
- **Why it could help where calibration can't:** scalar gain/threshold-balance can only *shift* the operating point; a
  *learned* per-weight adjustment can spread the dense sums so a graded firing *fraction* survives the ceiling (learn an
  effectively sparser/contrast-enhanced code that the rate readout CAN rank). This is the parent consolidation scoping's
  named §4.2 fallback ("surrogate-grad-on-bridge finetune … re-learn the on-bridge weights to compensate for the spiking
  dynamics").
- **Cheapest de-risk:** finetune ONLY the dense hidden layers' incoming weights against the off-bridge teacher rates on
  the 6-char calibration set (small, the net is frozen elsewhere). **Day-scale; a learning step.**
- **`sim/`-edit scope:** the deeper guarded edit — a LIF/AdEx-as-LIF forward consistent with `bptt_snn_gpu` exposed for
  the finetune loop (the parent scoping flags this is "NOT on the critical path unless the cheap calibration loses too
  much"). It is heavier than (a)/(b) and should be reached ONLY after they are measured.
- **Expected fidelity ceiling:** highest *capability* fidelity (learns around the ceiling) but **does NOT keep Gen-F /
  the trained net's weights verbatim** — it changes them — and is the most code. **Lower priority than (a)/(b) because it
  trades the cheapest property (no-edit, no-train, weight-faithful) for capability.**

### (d) Lower fan-in / sparsification / non-saturating (temporal-phase) code — **#4, partial, situational** ⚠️

- **Mechanism:** reduce the supra-threshold dense fan-in so the targets don't all pin — e.g. k-WTA / top-k masking of each
  layer's active set (the bridge has `transmission_gate` per-synapse current scaling, `bridge.py:5988-5994`, and
  coincidence/threshold gating), OR encode information in spike *phase/timing* (the RF resonate-and-fire complex-synapse
  path, `bridge.py:5691-5747`, `NeuronModel.RESONATE_AND_FIRE`) where info is in PHASE not rate (no rate ceiling at all).
- **Grounding:** sparsification is standard ANN→SNN (sparse codes don't saturate); the phase code is the project's own
  FHRR composer substrate (info in phase, "no common mode, no rate ceiling"). SOTA SDSA (Spike-driven Transformer, arXiv
  2307.01694) explicitly relies on *sparse binary* spikes to keep the coincidence-AND meaningful.
- **Cheapest de-risk:** add a top-k gate per hidden layer (keep the trained activation's top-k, mask the rest) and re-read
  the rate — does a sparse active set rank-recover? Cheap, NO edit (reuse `transmission_gate` or a host top-k mask in the
  runner).
- **`sim/`-edit scope:** NONE for a host top-k mask; small for a learned/dynamic gate.
- **Expected fidelity ceiling:** PARTIAL — a transformer MLP's hidden activation is genuinely *dense* (GELU rarely zeroes
  most units), so aggressively sparsifying loses real signal; this is a mitigation, not a clean fix. The **phase-code
  reframe is a bigger pivot** (re-encode the whole layer as phasors) — high-fidelity in principle (RF is the project's
  proven phase substrate) but it re-architects the generator's representation (Gen-F is rate-coded, not phasor) and is a
  *much* larger change than (a). Park unless (a)/(b)/(c) all miss.

### (e) REFRAME — sidestep dense-MLP saturation via the GLA-accumulator / linear-attention running-state path — **a structural reframe, not a calibration** ⚠️→ strategic

- **The reframe (SURPASS move 2 at the architecture level):** the dense-MLP saturation is a property of consolidating a
  **stacked feed-forward MLP** that fires *all units supra-threshold per token*. The consolidation scoping
  (`2026-06-22-genseq-loopstep3-consolidation-scoping.md` §2.3) already established that the **capable spiking-LM SOTA
  does NOT run dense per-token matmuls as rate-saturating spike layers** — SpikingBrain-7B (arXiv 2509.05276) replaces
  attention with a **Gated Linear Attention running-state accumulator** `S_t = diag(g_t)⊙S_{t-1} + k_tᵀv_t`, which is a
  **slowly-updated synaptic/membrane buffer** (a *graded accumulator*, NOT a per-step saturating spike layer); SDSA
  (arXiv 2307.01694) replaces the dense multiply with **binary coincidence-AND + integer column-sum** (no rate readout to
  saturate). **Both sidestep the rate-saturation wall by construction** — the accumulator is read as analog state, the
  coincidence-AND as a binary gate, neither as a saturating dense spike-rate.
- **Why it is relevant HERE:** the wall is specifically the **dense hidden→hidden MLP stage**. If the on-bridge generator
  keeps the *linear* ops as **graded accumulators** (= option (a) generalized: the layer's output IS the analog buffer
  state, never a spike-rate readout), and only the genuinely-needed nonlinearity passes through spikes, the dense
  saturation never arises. This is the deepest correct framing: **the bridge's strength is analog temporal accumulation
  (membrane/synaptic state), and consolidation should read analog state for linear ops, reserving spikes for the
  threshold/gating ops** — which is exactly what (a) graded-transmission does locally, and what GLA does globally.
- **`sim/`-edit scope:** the GLA accumulator step-block is the consolidation scoping's ranked attention option #1 (a
  small additive guarded default-off block); for the MLP it reduces to (a) (graded transmission, NO edit). The reframe's
  value is **strategic**: it says the cheap fix (a) is not a hack but the *correct* substrate-native representation, and
  it aligns the MLP fix with the attention plan (both go analog-state for linears).
- **Expected fidelity ceiling:** HIGH and substrate-natural (this is what the only-spiking-LM-at-scale does). Reach for
  the explicit GLA block at the *attention* step; for the MLP, (a) already realizes the reframe.

---

## 3. The cheapest-first de-risk of the top-ranked resolution + GO/NO-GO

**Top-ranked = (a) GRADED / analog transmission + readout** (and free to stack (b) population coding). Both are
NO-`sim/`-edit, NO-training, hours-scale, and (a) is *literally* the project's built-for-this-wall mechanism.

**The de-risk (cheapest, decisive):**
1. Re-run the EXISTING `_genseq_loopstep3_multilayer_signed_derisk.py` machinery, changed in the RUNNER only:
   - tag the block-1 and block-2 (dense hidden→hidden) wiring entries with `"graded": True` (and the signed copies too);
   - read each block's output as the membrane-derived **analog** `a_cont = clip((v−rest)/scale, 0, 1)` (the same signal
     the graded path transmits), NOT `cp_firing_states`;
   - compare to the off-bridge `forward_unroll` **layer ACTIVATION/membrane** (analog↔analog), not the spike rate;
   - sweep `graded_source_scale_mV` (the operating-point knob with headroom) to keep the dense sum off the `clip` ceiling.
2. Keep the existing anti-cheat (matched vs mismatched cross-input specificity on the final stage) and the per-layer +
   cumulative metric. Add the saturation diagnostic on `a_cont` (it should NOT pin: `a_cont` mean < max, graded).
3. CuPy for the decisive run (`feedback_gpu_not_numpy`); ≥6 seeds only if it becomes a variable claim (the first
   smoke is a single decisive existence run — does the analog readout recover rank at all).

**GO / NO-GO:**
- **GO** if cumulative analog-Spearman recovers to **≥ ~0.8** across the 3 stacked blocks with the signed graded wiring,
  AND the specificity margin re-opens (matched ≫ mismatched), AND `a_cont` is graded (not pinned). ⇒ the dense-MLP
  consolidation is NO-`sim/`-edit-SOLVED via the graded path; proceed to stack population coding for margin and then to
  the attention step (the consolidation scoping's ladder step #2).
- **PARTIAL** if it recovers above chance but < 0.8 (e.g. the `a_cont` `clip` re-saturates under the densest sums) →
  stack (b) population coding with threshold heterogeneity (the het-band recruits a graded firing fraction), and/or widen
  the graded operating point; re-measure. If still < 0.8 → escalate to (c) surrogate-grad finetune of the dense layers.
- **NO-GO (the honest fidelity wall)** ONLY if BOTH (a) graded-analog AND (b) population coding AND (c) surrogate-grad
  finetune fail to recover usable multi-layer rank → THEN the multi-layer dense consolidation is a genuine point-neuron
  boundary (and §4's "convert + P2-knowledge stand; full on-bridge generator is the precisely-characterized boundary"
  verdict holds). **This is not expected** given the graded path has a prior GO defeating this exact wall.

---

## 4. Honest verdict (SURPASS move 4) — SURPASSABLE-and-cheaply

**The multi-layer dense consolidation is SURPASSABLE, cheaply, with NO `sim/` edit, via a mechanism the project already
built and already proved against this exact wall.** Specifically:

- **The genuine residual is rate-readout saturation** (the 0.5 refractory spike-rate ceiling under dense supra-threshold
  fan-in), NOT the signed split (faithful at 0.32) and NOT the fan-in topology (which merely *causes* the supra-threshold
  drive; the information loss is the 1-bit spike-rate quantization). The wiring is intact (top-k overlap holds at chance,
  not below; the loss is purely ordinal-rank-among-the-active-set).
- **The cheapest surpass is the bridge's GRADED / analog transmission** (`RegionPathway.graded=True`, the
  horizontal-cell common-mode-removal mechanism, `bridge.py:6128-6175` / `regions.py:355-372`) read as the analog
  membrane signal — bypassing the refractory ceiling entirely, E/I-preserving, byte-identical-when-off, NO `sim/` edit.
  It is the project's **sanctioned escape from the Mikulasch-Priesemann point-neuron limit** (decorrelation/dense-coding
  is an analog/pre-spike computation a *spiking* point neuron can't do — but the bridge's graded path *is* that analog
  stage, and it has a prior GO: `2026-06-16-generalization-graded-propagation.md` used `graded=True` alongside the
  population/NMDA lifts to propagate a category-correct signal as the analog/spiking readout). The documented
  complementary lift is **population coding** (47%→100%-of-host, `2026-06-15-…GO.md`).
- **Therefore the NEGATIVE is correctly scoped as "the CHEAPEST (rate-coded spike readout, no-edit) path does not work"
  — NOT "consolidation is impossible."** The cheap-fixes-exhausted in the NEGATIVE were all *scalar calibrations of the
  saturating spike readout* (global gain, threshold-balance, I/E ratio) — none of which can un-quantize a 1-bit code. The
  next lever is not another calibration; it is **changing the readout from saturating-spike-rate to graded-analog** (the
  thing the bridge was extended to do). That lever was simply not in the ladder-step-#1 runner (it reads `cp_firing_states`).
- **The point-neuron *spiking* boundary is real and respected:** a point neuron's *binary spike train* cannot, at
  `dt`/refractory resolution, carry a high-dynamic-range dense linear sum — that IS the rate-code/Mikulasch-Priesemann
  wall, and the NEGATIVE is an honest, well-characterized measurement of it for the *spike-rate readout*. The SURPASS is
  not to deny the boundary but to **read the analog substrate the bridge already exposes** (membrane state / population
  mean / accumulator buffer), which is precisely how biology (retina horizontal cells) and the SOTA (GLA running-state
  accumulators) sidestep it. The deeper reframe (e) — keep *linear* ops as analog accumulators, reserve spikes for
  threshold/gating ops, the GLA/SDSA pattern — is the correct architecture and aligns the MLP fix with the attention step.

**Bottom line for the controller:** do NOT accept the multi-layer-dense consolidation as a boundary. Re-run ladder-step-#1
with `graded=True` + analog readout (hours, no edit, no GPU-architecture risk) as the cheapest-first de-risk; GO if
cumulative analog-Spearman ≥ ~0.8. The spiking-CONVERT GO + the P2-knowledge GO stand regardless; and IF — contrary to the
graded path's prior GO — (a)+(b)+(c) all miss, THEN and only then is "the full on-bridge dense generator is the
precisely-characterized point-neuron boundary" the honest deliverable.

---

## 5. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / JSON read):**
- **The residual is rate saturation, pinned at 0.5** — `_genseq_loopstep3_multilayer.json` `per_layer_fidelity`: blocks 1
  & 2 `on_mean_rate == on_max_rate == 0.500`, Spearman −0.019 / 0.009; layer 0 `on_max_rate 0.75`, Spearman 0.321
  (signed split faithful, not chance). `cumulative_mean_spearman 0.009`, `specificity_margin 0.000`.
- **The 0.5 ceiling is the 1-step refractory at dt=1ms** — `bridge.py:6583,6601,6616` (`not_in_refractory` gate +
  `refractory_period_steps` timer); runner sets `dt_ms=1.0`, `refractory_period_steps=1` (`:184,:215`). Off-bridge LIF has
  no refractory → max rate 1.0 (`off_max_rate 1.0` in JSON; `bptt_snn.LIFLayer`).
- **The runner reads SPIKES not membrane** — `onbridge_block_output_rates` counts `bridge.cp_firing_states`
  (`:339-350`), never `cp_membrane_potential_v`. This is exactly the readout the graded fix replaces.
- **The cheap levers were calibrations of the spike readout** — global gain sweep (`global_gain_sweep`), per-block
  geometric-bisection threshold-balance (`per_layer_balance_calibration_log`, block 1 all six probes `on_active 2048` at
  scale down to 0.0215), I/E ratio — all in the JSON; none un-pin block 1.
- **The bridge HAS graded analog transmission, built for this wall** — `RegionPathway.graded` (`regions.py:355-372`,
  *"transmits with GRADED (analog, non-spiking) release"*, the horizontal-cell analogue); the step block
  (`bridge.py:6128-6175`, `a_cont = clip((v−rest)/scale,0,1)`, E/I-preserving, *"a SPIKING inhibitory pool cannot linearly
  track the population mean … but its graded membrane state can"*); the mask build + spike-matvec removal
  (`bridge.py:2665-2682`, `:6030-6037`).
- **Population coding is the documented 47%→100%-of-host lift** — `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`
  §"The population lift", lines 104-114: single 47%, n_per=8 → 100%, 16 → 103%, 32 → 108%; *"the single-neuron ~50%
  plateau was indeed the documented rate-code SNR wall, and the population code lifts the read-out to the host reference."*
- **The graded path has a prior GO defeating the same wall** — `2026-06-16-generalization-graded-propagation.md`
  (candidate 3 graded transmission propagates 69/cue; candidate-independent; NMDA-integration + population are the project's
  standard "slow-NMDA integration + population code" lift; NO `sim/` edit).
- **The surrogate-grad infra exists** — `sim/surrogate_grad.py`, `sim/bptt_snn.py`, `sim/bptt_snn_gpu.py` (the parent
  scoping's §4.2 fallback; CLAUDE.md Phase 2.1/2.2 validation).
- **The reframe is the SOTA** — `2026-06-22-genseq-loopstep3-consolidation-scoping.md` §2.3 (GLA running-state accumulator
  = SpikingBrain-7B arXiv 2509.05276; SDSA binary-AND + column-sum = arXiv 2307.01694), re-read this pass.

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing]** That `a_cont`'s upper `clip` at 1.0 does NOT itself re-saturate under the *densest*
   hidden sums (the graded path was validated for *sparse* perception drive + common-mode removal, not for a dense
   transformer MLP layer's full-fan-in activation). The cheap de-risk (§3) measures this directly and sweeps
   `graded_source_scale_mV`; the mitigation if it does (population + heterogeneity, §2(b)) is in hand. **This is the one
   genuine uncertainty** — the graded mechanism is built and proven, but proven on a *different* (sparser) drive regime.
2. **[VERIFY]** That the off-bridge `bptt_snn.forward_unroll` exposes a per-layer ANALOG activation/membrane (not only
   spikes) to compare against `a_cont` analog↔analog. (`forward_unroll` returns layer state in the GO probe; confirm it
   surfaces the pre-spike membrane, else read the off-bridge analog from the layer's pre-threshold potential.)
3. **[VERIFY — fidelity ceiling]** That graded-analog multi-layer cumulative Spearman actually reaches ≥0.8 across 3
   stacked blocks WITH the signed split (the §1.1 `g·(V−E)` non-linearity stacks across analog layers too) — the de-risk
   measures it; the per-layer gain/scale calibration (now with graded headroom) is the cheap recovery if a single
   operating point doesn't hold across layers.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/findings/2026-06-22-genseq-loopstep3-multilayer-NEGATIVE-rate-saturation.md` + `research/findings/raw/_genseq_loopstep3_multilayer.json` (the boundary; per-layer saturation pinned at 0.5, signed layer-0 at 0.321, exhausted scalar calibrations — read in full).
- `research/runners/_genseq_loopstep3_multilayer_signed_derisk.py` (reads `cp_firing_states` `:339-350`; signed E/I split-channel wiring `:246-302`; dt=1.0/refrac=1 `:184,:215`; global-gain + per-block threshold-balance phases `:396-477` — read in full).
- `sim/bridge.py`: graded analog transmission step block (`:6128-6175`, `a_cont` membrane readout, E/I-preserving); graded mask build + spike-matvec removal (`:2665-2682`, `:6030-6037`); refractory fire-gate + timer (`:6583,6601,6616`); divisive-norm (`:6179-6199`); transmission-gate (`:5988-5994`); RF complex synapses (`:5691-5747`).
- `sim/regions.py`: `RegionPathway.graded` (`:355-372`, horizontal-cell graded release); `graded_lateral` / `input_mean_adapt` / `input_divisive_norm` analog circuits (`:206-247`).
- `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (the population lift table 47%→108%, lines 104-114 — the documented rate-code-wall lift — read).
- `research/findings/2026-06-16-generalization-graded-propagation.md` (graded transmission + NMDA + population propagate a category signal as spikes/analog; prior GO over the same wall family — read in full).
- `research/findings/2026-06-22-genseq-loopstep3-consolidation-scoping.md` (parent ladder; §2.3 GLA/SDSA reframe; §4.2 surrogate-grad fallback — read in full prior).
- `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md` (single-layer 0.918; positive-only + one-hot residuals — read).
- `sim/surrogate_grad.py`, `sim/bptt_snn.py`, `sim/bptt_snn_gpu.py` (surrogate-grad-on-bridge finetune infra — confirmed present).

### Current literature / biology (verified via the parent scoping's primary-source pass)
- **GLA running-state accumulator / linear attention** — SpikingBrain-7B, CAS/BICLab 2025, arXiv 2509.05276 (the only spiking LM at scale; replaces dense per-token matmul-as-spikes with a graded running-state accumulator). Katharopoulos et al. 2020, arXiv 2006.16236 (linear attention = recurrent accumulator).
- **Sparse-binary spike attention (no rate readout to saturate)** — Spike-driven Transformer / SDSA, Yao et al., NeurIPS 2023, arXiv 2307.01694 (binary Hadamard-AND + integer column-sum + threshold mask).
- **Point-neuron analog/pre-spike limit (the boundary the graded path escapes)** — Mikulasch-Priesemann point-neuron limit (project standing-practice: decorrelation/dense-coding/whitening is an analog/pre-spike dendritic computation a *spiking* point neuron cannot do; the bridge's graded membrane path is the on-substrate analog stage). Kandel 6e Ch 22 (retinal horizontal/bipolar graded potentials — the biology the `graded` pathway models).
