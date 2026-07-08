# Research gate — surpassing the on-bridge-spiking deep-credit "does-not-train" boundary: the fix is POPULATION CODING (K neurons/logical unit, pooled event rate), a proven project lever + the way Payeur-2021 Burstprop was *defined*

**Date:** 2026-07-07
**Type:** READ-ONLY deep-research + reference-catalog gate (no build, no `sim/` edit, no GPU job). Mechanism-surpass gate per the standing practice: the on-bridge-doesn't-train is the next mechanism to FIND, not a wall.
**Boundary under review:** `2026-07-07-deep-credit-onbridge-spiking-6seed-does-not-train-at-cheap-scale.md` — the depth-2 two-compartment spiking net on ONE `SimulationBridge` does NOT train the compositional-generalization task at CPU-smoke scale (6-seed, two configs, all arms BELOW chance 0.333, `trains_at_all=0/6`); the fenced-backprop RATE oracle reaches 1.0 (task is learnable); the microcircuit does not beat plain FA on spikes (delta ~0).
**Runner:** `research/runners/_semantic_inheritance_onbridge_spiking_derisk.py` (`OnBridgeBDSPNet`).

---

## 1. DIAGNOSIS — the true residual, isolated and quantified

### 1a. The read the net actually uses is a SINGLE-NEURON spike-count event rate
`OnBridgeBDSPNet` lays out the depth-2 net as slices `[input | H1 | H2 | out]` on one bridge, **one neuron per logical unit** (H1/H2 are `hidden=40` neurons = 40 *distinct* logical units, not 40 neurons pooled into fewer units). The forward activation each layer reads is the committed per-neuron low-pass EVENT rate `cp_bdsp_E`, sliced per neuron:

- forward (`_forward_spiking`, runner L212–214): `E = to_host(cp_bdsp_E)`; `acts[li] = E[slices[li]]` → **each hidden "unit" is ONE neuron's `E`**.
- the output logits (`_train_one` L273; `_logits` L216–218): `E_out = acts[-1]` → the class score is **each output neuron's single `E`** through a softmax.
- the descending clean credit (L280–301): `v_api = e_upper @ Y`, then `phi = E*(1−E)` — `E` is the **single-neuron** event rate; `phi` is the sigmoid-derivative of one noisy neuron.
- the Burstprop arm's descending credit (L291–294): `e_upper = (Bm − Pbarm·Em)` read straight off `cp_bdsp_B/Pbar/E` for the hidden **single-neuron** slice — a per-single-neuron burst FRACTION.

### 1b. That single-neuron `E` is finite-spike Poisson-noise-dominated at the operating rate
`cp_bdsp_E` is an exponential low-pass with `bdsp_rate_tau = 0.90` (`sim/config.py:255`; `sim/bridge.py:7176–7183`): `r *= 0.9; r[event] += 0.1`. Effective integration window ≈ `1/(1−0.9) = 10` steps (the settle is 25/40 steps, but the low-pass forgets older steps → only ~last 10 contribute). The runner's own tonic drive targets an event rate **~0.05–0.10** (`OnBridgeBDSPNet.__init__` comment, L127). Expected events in the effective window and the resulting coefficient of variation (independently computed):

| operating event rate p | expected events in window (~10 steps) | single-neuron CV(E) | CV with K=8 pool | CV with K=16 pool |
|---|---|---|---|---|
| 0.03 | 0.30 | **1.83** | 0.65 | 0.46 |
| 0.05 | 0.50 | **1.41** | 0.50 | 0.35 |
| 0.10 | 1.00 | **1.00** | 0.35 | 0.25 |
| 0.15 | 1.50 | **0.82** | 0.29 | 0.20 |

⇒ at the actual operating point the single-neuron `E` has **CV ≈ 1.0–1.4** — the forward activation, the softmax logits, AND the descending credit are **noise-dominated per example**. The rate reference reads a clean sigmoid scalar `soma∈(0,1)` (CV 0); the bridge substitutes a ~unit-CV single-spike estimate of it. This is the documented **point-neuron single-neuron rate-code SNR wall** (the project's own recurring wall; Mikulasch-Priesemann-family — see §2c on why this is the READ-OUT variant, NOT the pre-spike whitening variant).

### 1c. It is a CREDIT/READ-SNR failure, not a dead-plumbing or drive-scale failure — MEASURED
From the committed raw (`research/findings/raw/_semantic_onbridge_fair_seed42.json`): `ff_weight_moved_fa = 7013.7` (weights move a LOT), `ff_weight_moves = True`, yet `trains_at_all = False` (plain-FA 0.222 / burstprop 0.333 / microcircuit 0.074, chance 0.333), `oracle_inherit = 1.0`. Weights move hard but in **noise-dominated directions** → no learning. Signature of a credit/read-SNR limit, not a plumbing or drive bug. The task is learnable (oracle 1.0). This is the residual to lift.

### 1d. The precise diagnostic to run FIRST (SURPASS move a — isolate forward-SNR vs credit-SNR)
The one measurement that decides everything (and guards the central anti-cheat in §4): on a trained net, correlate the on-bridge single-neuron spiking read against the rate-reference activation for the SAME input, per layer:
- `corr(E_spiking[H1], soma_rate[H1])` and `corr(E_spiking[H2], soma_rate[H2])` (forward-activation fidelity), and
- `corr(descending_credit_spiking, descending_credit_rate)` per hidden layer (credit fidelity).
The runner already holds both worlds (the `DendriticMLP` rate oracle is built in `run_seed`, and the spiking `acts` are exposed via `hidden_rep`). Expected at CV≈1: both correlations LOW (~0.2–0.4), rising toward the rate reference as the pool size K grows — the direct, quantified fingerprint of the read-SNR wall and the direct readout of the population lift. **This is the diagnostic the gate names; run it before/with the population de-risk.**

---

## 2. REFRAME — how real cortex / the SNN field carries a usable rate through spiking layers

### 2a. It is a POPULATION, not a single neuron (catalog + Kandel, first-hand)
- **Catalog E.03 "Population coding & vector averaging"** (Kandel 6e Ch 17 p ~458–464): "A stimulus parameter … is represented by the *distribution of activity* across many broadly-tuned neurons; downstream vector sum or Bayesian decoding extracts the value. **Robust to noise and single-neuron loss.**" This is the exact property the runner throws away by using 1 neuron/unit.
- **Catalog H.17 "Population vector coding (Georgopoulos)"** (Kandel 6e Ch 34 p 825–840): the population vector — `Σ rᵢ·θᵢ` weighted by firing rate — "predicts reach direction on a **trial-by-trial basis**" where a single M1 neuron cannot. Trial-by-trial (= per-example) reliability from pooling is exactly the regime the deep-credit net needs (one apical pattern per example).
- The brain does NOT read cognition off one Poisson neuron; it averages a population. The deep-credit read must too.

### 2b. The project's OWN rule family (Burstprop / BurstCCN) was DEFINED at the ENSEMBLE (population) level — the runner's single-neuron read is a departure from the source method
The load-bearing external anchors:
- **Payeur, Guerguiev, Zenke, Richards, Naud 2021, *Nature Neuroscience* "Burst-dependent synaptic plasticity can coordinate learning in a hierarchy of cortical areas":** "In an ensemble of pyramidal neurons, the inputs to the perisomatic and distal apical dendritic regions can be distinctly encoded using the event rate computed *across the ensemble of cells* and the percentage of events *in the ensemble* that are bursts (the 'burst probability')." i.e. Burstprop's event rate and burst probability are **ENSEMBLE quantities**; the spiking version requires each ensemble to compute a moving average of its burst probability (the MNIST/CIFAR/ImageNet results are the *rate-coded* spiking version, "an order of magnitude fewer neurons" but still ensemble-per-unit).
- **BurstCCN — Greedy, Zhu, Mellor, Costa, NeurIPS 2022, "Single-phase deep learning in cortico-cortical networks" (arXiv:2206.11769) — the CLOSEST published analogue to the project's exact rule (burst-multiplexing, single-phase, transport-free feedback).** Decisive: **"The burst probability of an ensemble is … the ratio of the event rate (e) and burst rate (b): p = b/e"** — a `b/e` ratio **does not exist at a single neuron at an instant; it is an ensemble-only statistic.** Their *spiking* XOR net used **5 populations × 500 individual neurons each = 500 neurons per logical unit** (sparse p=0.05, balanced E/I Poisson), each example presented **8 s** (≈80 steps at dt=0.1 s) with an **eligibility low-pass τ_pre=0.1 s**. Tellingly, they demonstrate the *spiking* BurstCCN only on **XOR**; **MNIST/CIFAR are the RATE-based implementation** — i.e. the rate model IS the ensemble limit, and dropping below it (to spikes) is exactly where the finite-spike-noise problem appears and is paid for with **500 neurons/unit + long windows.**
- **Neftci, Mostafa, Zenke 2019, "Surrogate Gradient Learning in SNNs" (IEEE SPM 36(6):51–63, arXiv:1901.09948) — states the problem AND the two cheapest cures verbatim:** "precise estimation of firing rates requires averaging over a number of spikes. Such averaging requires either **relatively high firing rates or long averaging times** because several repeats are needed to average out discretization noise. This problem can be **partially addressed by spatial averaging over large populations of spiking neurons. However, this may require the use of larger neuron numbers.**"

⇒ a burst FRACTION over many cells is a low-variance estimate; over ONE cell it is `Binomial(k,p)/k` with `k`≈0–1 spikes = maximal variance. The on-bridge runner reads event rate / burst fraction per single neuron (§1a) → **it is running Burstprop/BurstCCN off-spec.** Population coding is not an add-on; it is **how the method was specified to be read** — and the field's canonical review names it as the fix.

### 2c. This is the READ-OUT variance wall, NOT the pre-spike whitening wall — so it is CHEAP (a crucial framing correction)
The compositional-semantics rate finding already flagged "**decorrelation is a red herring**" for this task. The Mikulasch-Priesemann point-neuron limit the project documents (Mikulasch et al. *Trends Neurosci.* 2023, PMID 36577388; PNAS 2021, doi 10.1073/pnas.2021925118; see `2026-06-06-*decorrelation*`, `2026-06-11-option-B-whitening*`) bites *whitening / off-diagonal decorrelation*, which is a **pre-spike analog dendritic** computation a point neuron cannot do. Population averaging of `E` is **post-spike, feed-forward, and local** — it removes finite-spike *variance*, not cross-neuron *covariance*. It sidesteps the whitening wall entirely. So the fix is in the cheap regime (post-spike pooling), not the months-scale dendritic-rewrite regime.

---

## 3. RANKED cheap-first SURPASS mechanisms

Each: what · cite · why it lifts SNR · cheapest de-risk reusing `_semantic_inheritance_onbridge_spiking_derisk.py` · transport-free/single-phase preserved.

### #1 — POPULATION CODING: K neurons per logical unit, pooled event rate (RECOMMENDED)
- **What:** represent each of the H logical hidden units (and each output class) as a **pool of K neurons**; the layer activation is the **mean `E` over the pool** (and the descending credit / burst fraction is the pool-mean). Concretely: widen each slice from `hidden` to `hidden·K` neurons; add a fixed block-pooling read so `act[unit] = mean(E[pool_of_unit])`; the fixed-random feedback `Y` maps between *pooled-unit* spaces (unchanged structure, size H not H·K). The per-synapse BDSP kernel is untouched (it already moves each of the H·K neurons' weights); pooling is a read/credit-side average.
- **Cite:** BurstCCN (Greedy-Costa 2022, **500 neurons/unit**, `p=b/e` ensemble-only) + Payeur 2021 (ensemble event-rate/burst-prob) — the project's own rule family REQUIRES it (§2b); Neftci-Mostafa-Zenke 2019 ("spatial averaging over large populations" as THE cure); **PopSAN (Tang, Kim, Panda, Michmizos et al., CoRL 2020, arXiv:2010.09635): K=10/dimension is the standard working point, performance MONOTONE in K∈{2,3,5,10}, larger K = "redundant representations" = variance averaging**; catalog E.03 / H.17 (population vector, "robust to noise and single-neuron loss", trial-by-trial reliability); PNAS 2010 "Optimal population coding by noisy spiking neurons"; arXiv:2301.07275 (multi-compartment + population encoding, deep SNN RL).
- **Why it lifts SNR:** pooling K independent neurons cuts the read/credit CV by ~√K (Poisson variance averaging; RMS coding error ∝ √(tuning-width/K)). At the operating rate K=8 takes CV 1.0–1.4 → 0.35–0.50; K=16 → 0.25–0.35 (§1b table). Combined with a longer window T the variance falls as ~1/(K·T) (K and T multiply) — the regime where credit becomes usable through 2 layers.
- **Cheapest de-risk (reuse the runner):** add a `--pool-k` knob; widen slices to `hidden·K`; pool `E`/credit block-mean per unit; RE-RUN the exact 6-seed CPU-smoke at K∈{1,8,16} (K=1 == the current negative = the causal control). Predicted lift: 47%→~100% by K≈8, mirroring the 2026-06-15 precedent (§below). **Escalation guide from the source method:** if K≈8–16 still sits at chance on spikes, BurstCCN's own working K is **500/unit** — scale K (and window T) toward there before concluding a wall. CPU, minutes; the held-out/permuted/lesion/oracle controls are already wired.
- **Transport-free / single-phase:** YES — pooling is a local read average; `Y` stays a separate seed stream; no phase added.
- **DIRECT PROJECT PRECEDENTS (two, decisive):**
  1. `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` — the SAME single-neuron `E`-read wall, on the SAME substrate, LIFTED by exactly this: single-neuron read plateaued at **47% of host**; "each concept gets `n_per` neurons; drive the whole concept-population, average the `n_per×n_per` learned-weight sub-block → M_pop. **Population averaging cancels the per-synapse spiking noise that bounds the single-neuron read-out.**" Result: **K=8 → 100%, K=16 → 103%, K=32 → 108%; saturates by ~8 neurons/concept.**
  2. `2026-06-15-off-diagonal-red-herring-ppmi-local-normalization-reaches-host.md` (CYCLE 91) — a SECOND independent instance: single-neuron PPMI read **20% of host** → **16 neurons/dim 66% → 32 neurons/dim (+ window 80) = 94% of host.** Fidelity scales monotonically with population size — "the on-bridge cortex path is viable with a **population-rate code** (the most brain-faithful encoding for a graded value)."
  Together these are the strongest evidence that #1 is the fix, is cheap, and saturates at a small K.

### #2 — LONGER TEMPORAL AVERAGING / low-pass eligibility read (more settle steps / slower `bdsp_rate_tau`) — STACK WITH #1
- **What:** integrate `E` over a longer window T (raise `settle_steps`; and/or raise `bdsp_rate_tau` 0.90→0.97 → effective window 10→~33 steps); read the burst/event rate off a low-pass-filtered trace, not a raw short-window count.
- **Cite:** variance ∝ 1/T for a rate estimate (standard); every burst-credit paper that runs on spikes pairs the ensemble with a LONG window + eligibility low-pass — **BurstCCN 8 s/example, τ_pre=0.1 s; Payeur/Naud neuromorphic ports 100 ms/example; e-prop's `ẑ` filtered presynaptic trace + leaky-integrator readout** (Bellec et al. 2020, Nat. Commun.); Neftci-2019 ("long averaging times" as the co-cure); the runner's own settle 30→80 de-risk (in flight per the boundary doc); the project's 2026-06-01 GATE2 (temporal integration over K=16 obs lifted recognition 0.50→0.893).
- **Why:** cuts CV by ~√(T/T₀); with #1 the variance falls as ~1/(K·T) (K and T multiply — this is the field's recipe). CAVEAT: EMERGE-5's S-sweep showed **temporal budget on ONE neuron alone does not recover accuracy at low burst rate** (`2026-07-02-emerge5b`) — because there the bottleneck was credit *structure*. So #2 is the **stack partner for #1 and the matched anti-cheat control** (§4.2: independent-neuron pooling must beat equal-total-spike single-neuron temporal averaging), NOT a standalone fix.
- **De-risk:** `--settle-steps 80 --bdsp-rate-tau 0.97`, same 6-seed. Cheap; compose WITH #1.
- **Transport-free / single-phase:** YES.

### #3 — GRADED / FILTERED READ instead of raw spike-count `E`
- **What:** read a graded/filtered quantity (the apical/soma membrane `cp_v_apical`/`cp_membrane_potential_v` low-pass, or a filtered event trace) as the activation, instead of the raw counted event rate `E`.
- **Cite:** **the SG-SNN default IS a graded read** — Neftci-Mostafa-Zenke 2019: "the output layer consisted of leaky integrators that did not spike," loss on the max/sum of membrane potential, NOT a spike count; **e-prop** builds its credit from `filtered ẑ trace × smooth membrane pseudo-derivative × leaky readout` (never a single spike count); **DCLL** (Kaiser-Mostafa-Neftci 2020) forms local rate costs from *filtered spike-based basis functions* with fixed random local projections (a transport-free single-phase deep-credit cousin of this exact setup); the project's own graded-read lineage (the 88.6M spiking-forward graded read; EMERGE graded-vs-spike-count reads).
- **Why:** a graded/filtered potential is not finite-spike-quantized → far lower CV. HONEST purity note: reading a graded membrane as the "activation" is a *partial retreat* from pure spike-count coding (the membrane is a real neural variable, not a host computation — defensible, but note it). The committed BDSP kernel is wired to `E`/`B`; swapping the read is a larger change to the mechanism's spirit than pooling. Keep as the *combineable* fallback if #1+#2 leave residual.
- **De-risk:** a `--read {event,membrane}` arm reading the pooled soma potential; same 6-seed. No `sim/` edit (read existing arrays).
- **Transport-free / single-phase:** YES.

### #4 — PER-LAYER NORMALIZATION / HOMEOSTASIS (threshold-balancing) — the cheap force-multiplier the project already has
- **What:** per-layer rate normalization / adaptive-threshold homeostasis so each hidden layer sits at a target mean `E` (avoids the sub-threshold dead-`E`→`phi'≈0`→no-credit and the saturated regimes), stabilizing the *operating point* the pooled read averages over.
- **Cite:** **BNTT (Kim-Panda, Front. Neurosci. 2021, arXiv:2010.01729) + tdBN (Zheng AAAI 2021)** — per-layer normalization / threshold-balancing directly targets "a **large variation of forward activation and backward gradients**" and lets deep SNNs "train **stably from scratch**" at **25–50 timesteps** (vs 2500 for conversion), "computationally inexpensive," no auxiliary net; **this maps to homeostatic threshold adaptation the bridge ALREADY has** (`fused_homeostasis_update`, threshold adaptation, input-mean adaptation); the project's spike-frequency-adaptation + feedforward-inhibition normalization (`2026-06-16-biologization-sweep`).
- **Why:** does not by itself cut per-neuron variance, but keeps every layer in the informative `E`-band so a *short*-window rate is informative and the descending `phi'(E)` credit doesn't die — a **necessary enabler for depth + a force-multiplier on #1/#2** (the runner already adds tonic drive ad-hoc for exactly this reason; homeostasis is the principled version).
- **De-risk:** enable per-layer target-`E` homeostatic gain; same 6-seed, layered on #1.
- **Transport-free / single-phase:** YES.

### #5 — SPARSE EXPANSION RECODE (Marr-Albus granule codon) for decorrelation
- **What:** insert a fixed sparse random-expansion layer (K-of-N codon, F.12) so the representation is high-dimensional + sparse + decorrelated before the credit read.
- **Cite:** catalog **F.12** "Codon representation — sparse expansion recoding via granule layer" (Marr 1969 §3.0 p444; Albus 1971 §IV.A p41–42); catalog **F.02** (granule/PF divergent code); **D.12** (DG expansion recoding, pattern separation). EMERGE-35's spiking Marr-codon pooler is the project's built precedent.
- **Why:** decorrelation + sparsity make a *single* read more linearly separable — but (a) this targets *representational overlap*, not finite-spike *read variance* (the actual residual per §1c), and (b) sparse codes fire *fewer* spikes/read → can WORSEN single-neuron CV unless pooled. Highest-value as a *representation* upgrade, not the direct SNR fix. Deprioritized for THIS boundary.
- **De-risk:** a fixed MF→GC expansion slice before H1; same 6-seed. More architecture; lower priority.
- **Transport-free / single-phase:** YES (fixed random expansion).

---

## 4. VERDICT + the anti-cheats it needs

### The #1 recommended mechanism + its cheapest de-risk
**POPULATION CODING — K neurons per logical unit, pooled event rate — is the #1 fix.** It (a) is the direct lift for the *measured* residual (single-neuron read/credit variance, CV≈1.0–1.4 → √K reduction), (b) is exactly how Payeur-2021 Burstprop was *defined* (ensemble event-rate + burst-probability), (c) sidesteps the Mikulasch-Priesemann whitening wall (post-spike averaging, not pre-spike decorrelation), and (d) has a **direct, quantified, same-substrate project precedent** (2026-06-15: single-neuron 47% → K=8 pool 100%, saturates by ~8/unit).

**Cheapest de-risk (CPU, reuse `_semantic_inheritance_onbridge_spiking_derisk.py`):**
1. Add `--pool-k K`; widen each hidden/output slice to `hidden·K` neurons; read `act[unit] = mean(E[pool])` and inject/descend the pool-mean credit; `Y` stays H×H (pooled-unit space). No `sim/` edit (widen the slice layout + block-pool the read; the BDSP kernel already moves all H·K synapses).
2. Run the **existing** 6-seed CPU-smoke at **K ∈ {1, 8, 16}** (K=1 == the committed negative = the causal control). Add the §1d `corr(E_spiking, soma_rate)` per-layer diagnostic at each K.
3. GO bar: at K≈8, `trains_at_all=True` on ≥5/6 seeds and best-arm held-out clears the 1-layer floor by a real margin — i.e. the population read makes on-bridge deep credit TRAIN. Then (and only then) re-read the microcircuit-vs-FA contrast on the pooled, now-usable credit.

### The anti-cheats it needs (load-bearing — the population-vs-more-samples trap is central)
1. **The K=1 causal control** (== the current 0/6 negative). If K=8 trains and K=1 does not on the *same* net/task/seed, the lift is the pooling.
2. **Pooling ≠ "just more spikes" (the EMERGE-5b refutation guard).** EMERGE-5b/5c found "naive population-averaging is mathematically identical to raising the sample budget S" and the S-sweep FAILED to recover accuracy at low burst rate — because there the bottleneck was *credit STRUCTURE*, not read variance. RECONCILE explicitly: run **temporal-averaging (#2, longer T = more samples on ONE neuron) as a matched control against spatial pooling (#1, K neurons)**. If ONLY spatial pooling of *independent* neurons recovers training (and equal-total-spike temporal averaging on one neuron does not), the lift is genuine *independent-sample* variance reduction (the 2026-06-15 regime), not the already-refuted "more samples" that EMERGE-5b ruled out. **This is the decisive anti-cheat: it distinguishes a read-variance residual (population fixes) from a credit-structure residual (population does NOT fix — the microcircuit clean-channel is needed).** The §1d correlation diagnostic operationalizes it: population should raise `corr(E_spiking, soma_rate)`; if it doesn't, the residual is structural.
3. **Held-out / no-leakage preserved** (reuse verbatim): memorization control ≈ chance (0.000), permuted → chance, member-id-only below chance — pooling must NOT create a leakage path (it shouldn't; it only averages the same units).
4. **Depth still load-bearing:** the 1-layer floor (also pooled at K) must still UNDERFIT held-out inheritance — i.e. K helps the *deep* net compose, it doesn't trivially solve the task shallowly.
5. **No weight transport:** `Y` stays a separate seed stream in pooled-unit space (H×H), never a forward `W`/`Wᵀ`; single-phase preserved (pooling is a within-step read average).
6. **`ff_weight_moves` sanity:** weights already move (7013); the read is whether they now move in *signal* directions (held-out clears floor), not merely more.

### Honest scope / what this gate does NOT claim
- Population coding fixes the **read/credit SNR** residual. If de-risk anti-cheat #2 shows the residual is *credit-structure* (temporal-matched control also fails, correlation doesn't rise with K), then the fix is the **microcircuit clean-error channel** (`2026-07-07-D1-microcircuit-noise-robust*`: clean-error FA credit is batch-robust where raw burst-fraction credit is fragile) — i.e. descend the pooled *clean* `v_api` rather than the pooled *burst fraction*, which the runner's plain-FA/microcircuit arms already do. The most likely outcome (given ff-moves-but-noisy + single-neuron reads on all three channels) is that **population coding of a CLEAN-error-descended credit** (pool #1 on the plain-FA/microcircuit arm, not the burst arm) is what trains — the two levers compose.
- This is a SCALE/instrument surpass (per the owner's standing correction: a cheap-scale non-training is an instrument limit, not a mechanism wall). The disciplined next step is the K-sweep CPU de-risk with anti-cheat #2, BEFORE any GPU scale-up.

---

## Files
- Boundary: `research/findings/2026-07-07-deep-credit-onbridge-spiking-6seed-does-not-train-at-cheap-scale.md`; runner `research/runners/_semantic_inheritance_onbridge_spiking_derisk.py`; raw `research/findings/raw/_semantic_onbridge_fair_seed{42,43,44,100,101,102}.json`.
- Direct precedent (the lift): `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md` (single-neuron 47% → K=8 population 100%).
- Credit-vs-read diagnostics: `research/findings/2026-07-02-emerge5b-credit-vs-readout-PARTIAL.md`, `-emerge5c-microcircuit-noise-robust-GO.md`, `2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`.
- Rate-task GO: `research/findings/2026-07-07-deep-credit-real-task-compositional-semantics-GO.md`.
- Substrate mechanism: `sim/bridge.py` (BDSP block ~L7120–7220, `cp_bdsp_E/B/Pbar`, `fused_bdsp_update`); `sim/config.py:255` (`bdsp_rate_tau=0.90`); `sim/dendritic_mlp.py` (rate oracle).
- Catalog: E.03 (Kandel 6e Ch 17 p458–464), H.17 (Ch 34 p825–840, Georgopoulos 1986), F.12 / F.02 / D.12 (Marr 1969 §3.0 p444; Albus 1971 §IV.A p41–42).
- External (subagent-verified, several from fetched full text):
  - **BurstCCN — Greedy, Zhu, Mellor, Costa, NeurIPS 2022, "Single-phase deep learning in cortico-cortical networks," arXiv:2206.11769** — the closest analogue to the project's rule; `p=b/e` ensemble-only burst probability; spiking XOR net = **500 neurons/unit**, 8 s/example, τ_pre=0.1 s; MNIST/CIFAR are the RATE version.
  - **Payeur, Guerguiev, Zenke, Richards, Naud 2021, *Nat. Neurosci.* 24:1010–1019** (ensemble event-rate/burst-probability; spiking needs a burst-probability moving average).
  - **PopSAN — Tang, Kim, Kozdon, Panda, Michmizos et al., CoRL 2020, arXiv:2010.09635** (K=10/dimension standard; monotone in K∈{2,3,5,10}); arXiv:2301.07275 (multi-compartment + population encoding, deep SNN RL).
  - **Neftci, Mostafa, Zenke 2019, "Surrogate Gradient Learning in SNNs," IEEE SPM 36(6):51–63, arXiv:1901.09948** — states the problem + names spatial-population-averaging AND long-averaging-times as the cures; leaky-integrator graded readout; DCLL (Kaiser-Mostafa-Neftci 2020) local rate cost from filtered spike bases.
  - **Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass 2020, *Nat. Commun.* 11:3625 (e-prop)** — filtered presynaptic trace ẑ × pseudo-derivative × leaky readout (low-variance credit).
  - **BNTT — Kim & Panda, Front. Neurosci. 2021, arXiv:2010.01729** + tdBN (Zheng AAAI 2021) — per-layer normalization / threshold-balancing, trains deep SNNs from scratch at 25–50 timesteps, controls forward/backward variance.
  - PNAS 2010 "Optimal population coding by noisy spiking neurons"; Cayco-Gajic, Clopath, Silver 2017 *Nat. Commun.* (sparse expansion decorrelates code GEOMETRY, not per-unit spike variance — why F.12 is NOT the fix for this residual); Stuck & Naud 2023/2025 (neuromorphic burstprop ports, 100 ms/example).
  - Mikulasch et al. *Trends Neurosci.* 2023 (PMID 36577388); PNAS 2021 (doi 10.1073/pnas.2021925118) — the point-neuron pre-spike-whitening limit (the wall population coding SIDESTEPS).
