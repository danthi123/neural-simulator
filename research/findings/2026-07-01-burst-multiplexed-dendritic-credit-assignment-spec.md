---
type: finding
status: contributing
date: 2026-07-01
mechanism: deep-credit
---

# Burst-multiplexed & dendritic-microcircuit credit assignment — a digitization SPEC for EMERGE-1b

**2026-07-01 (read-only deep-research subagent; NO code edited, NO experiment run).** The prior gate
(`2026-07-01-emerge1-deep-dendritic-representation-BOUNDARY.md`) showed VANILLA feedback alignment (fixed-random
top-down `B` + the committed Urbanczik-Senn local rule, `sim/dendritic_mlp.py`) **memorizes** a depth-2 task
(train->1.0) but does **not generalize** through depth (held-out ~= chance ~0.58), while the fenced backprop oracle
generalizes (0.95). Per the owner's master directive, that BOUNDARY is the *start* of research, not the end: vanilla FA
is the WEAKEST plausible mechanism, the brain demonstrably credit-assigns through depth, so a faithful biological
mechanism exists and must be digitized. This doc specifies the two prime candidates precisely enough to implement in
numpy, deltas them from the existing code, recommends the ONE to build first (EMERGE-1b), and gives the GO/anti-cheat
design + honest expected outcome.

**The unifying reframe of the wall.** Vanilla FA fails at depth for TWO compounding reasons the literature names
explicitly: (i) the fixed-random feedback `B` is *not aligned* with the forward path, so the projected "error" is a
random rotation of the true error at every hidden layer (alignment must be *earned*, and vanilla FA earns it poorly
past one hidden layer); and (ii) the projected error is *not linearized* through the layer nonlinearities — it does not
carry the transfer-function-derivative chain that backprop's `D_l = diag(phi'(u_l))` factors supply. **Both prime
candidates below attack exactly these two failure modes** — one by *learning the feedback weights to align them* (the
microcircuit's interneuron does this), the other by a *recurrent linearization of the credit signal* + a burst channel
that keeps the credit signal from corrupting the feedforward pass (burstprop). This is why they generalize where vanilla
FA memorizes.

---

## MECHANISM 1 — Burst-multiplexed plasticity (Payeur, Guerguiev, Zenke, Richards & Naud, *Nat Neurosci* 2021; "Burstprop")

**Sources:** Payeur et al. 2021, Nat Neurosci 24:1-10 (PubMed 34728832 is the Author-Correction; primary DOI
10.1038/s41593-021-00857-x; preprint bioRxiv 2020.03.30.015511). Multiplexing substrate: Naud & Sprekeler,
*Sparse bursts optimize information transmission in a multiplexed neural code*, PNAS 2018. Rate-model formalization &
scaling critique: Greedy, Zhang, Najafi, Bengio, Richards, Costa, *Single-phase deep learning in cortico-cortical
networks* ("BurstCCN"), NeurIPS 2022 (arXiv 2206.11769). Neuromorphic spiking re-derivation: *A burst-dependent
algorithm for neuromorphic on-chip learning*, Neuromorph. Comput. Eng. 2025 (IOP 2634-4386/adb511).

### 1.1 What multiplexing means (the two channels)

A pyramidal neuron (or an *ensemble* of them) carries TWO simultaneous rate signals on the same axon:

- **Event rate `E`** = rate of *events*, where an event is either an isolated spike OR the first spike of a burst.
  `E` carries the **feedforward** (bottom-up) signal — the network's actual activation / output.
- **Burst probability `P`** = fraction of events that become **bursts** (a burst = >=2 spikes with ISI < threshold,
  counted at the second spike). `P` carries the **top-down credit/teaching** signal.
- **Burst rate `B = P * E`** is what a downstream/plasticity read-out actually sees.

The apical dendrite is the demultiplexer's control knob: **apical (top-down) input sets `P`**, while **basal
(bottom-up) input sets `E`** (via the soma). Because `P` and `E` are separable channels on the same spike train, a
top-down error signal can steer plasticity WITHOUT corrupting the feedforward computation it rides on. (Naud-Sprekeler
2018 proves short, sparse bursts are near-optimal for this multiplexing.)

### 1.2 The burst-probability function (the credit channel)

Let `v_api,i` be the apical-dendrite potential of unit `i` (driven by top-down feedback + local dendritic
regenerative activity). Burst probability is a monotone squashing of the apical drive around a **baseline**:

```
    P_i = sigmoid( beta * v_api,i )                                   (M1.1)
```

Key property, load-bearing for the no-confab / no-spurious-learning behavior: **in the absence of any teaching signal
the network must sit at a fixed baseline burst probability `P0`** (e.g. `P0 = 0.5` if `v_api = 0`), so that the
plasticity rule (M1.2) produces ZERO net weight change at rest. Feedback that says "increase this unit's output" pushes
`v_api` up -> `P` above `P0` -> LTP; feedback that says "decrease" pushes `P` below `P0` -> LTD. (Greedy 2022, Payeur
2021: "in the absence of a teaching signal ... a baseline level of bursting such that no weight changes occur.")

### 1.3 The burst-dependent plasticity rule (BDSP) — the exact update

For a synapse from presynaptic unit `j` to postsynaptic unit `i`:

```
    dw_ij/dt = eta * [ B_i(t) - P_bar_i(t) * E_i(t) ] * E_tilde_j(t)      (M1.2)
```

- `B_i(t)`     = postsynaptic burst rate (= `P_i * E_i`).
- `E_i(t)`     = postsynaptic event rate.
- `P_bar_i(t)` = a slow **moving average of the postsynaptic burst probability** (low-pass filter of `P_i`); it is the
  learned per-unit *baseline* the instantaneous burst rate is compared against.
- `E_tilde_j(t)` = a filtered presynaptic event train (a short presynaptic **eligibility trace**).
- The bracket `[B_i - P_bar_i * E_i] = E_i * (P_i - P_bar_i)` is the **change in burst probability from baseline**,
  gated by activity. So: *potentiate when this unit is bursting MORE than its recent baseline given how active it is,
  depress when less.* This is a three-factor, fully LOCAL rule (post burst rate + post event rate + pre trace); no unit
  ever reads another unit's weights.

**Rate-model form** (what you implement first — the deterministic BurstCCN limit, Greedy 2022): with per-unit event
rate `e_i = phi(basal drive)` and burst prob `p_i = sigmoid(beta * v_api,i)` and baseline `p0`,

```
    b_i    = p_i * e_i                                                  (M1.3)
    dW_ff_ij ∝  ( b_i - p0 * e_i ) * e_j^pre  =  e_i*(p_i - p0) * e_j^pre  (M1.4)
```

i.e. the presynaptic *event rate* times the postsynaptic *deviation of burst rate from baseline*.

### 1.4 How top-down credit reaches the apical dendrite (no weight transport)

Feedback connections from layer `l+1` project onto the **apical** compartment of layer `l` through a feedback matrix
`Y_l` that is **independent of the forward weights `W_l`** (fixed-random -> feedback alignment; or slowly learned).
The apical potential of layer `l` is driven by the *burst signal* of the layer above:

```
    v_api,l  =  Y_l * ( b_{l+1} - b_{l+1}^baseline )   (top-down burst-coded error)   (M1.5)
```

Because `b_{l+1} - baseline` encodes "how wrong layer `l+1` is," and `Y_l` maps it back down, `v_api,l` becomes a
**burst-coded local error** at layer `l`, which sets `P_l` (via M1.1), which drives BDSP (M1.2) at layer `l`'s
feedforward synapses. Iterating this down the stack propagates credit — the burst channel is a *biological realization
of the backward pass*, and the event channel (the forward pass) is unperturbed because it lives on a separate
(rate vs burst-probability) channel of the same axons.

### 1.5 Short-term plasticity demultiplexes the two channels (the biological read-out)

A downstream synapse recovers ONE of the two channels depending on its short-term dynamics (Naud-Sprekeler 2018;
Payeur 2021):

- **Short-term DEPRESSION (STD)** synapses -> transmit the **event rate `E`** (the feedforward signal).
- **Short-term FACILITATION (STF)** synapses -> transmit the **burst rate `B`** (the credit signal).

So the feedforward pathway uses depressing synapses (reads `E`) and the credit/plasticity pathway uses facilitating
synapses (reads `B`) — the SAME axon feeds both without cross-talk. (For EMERGE-1b's rate model this is not literally
required — you compute `e_i` and `b_i` directly — but it is the biological justification the owner's standard demands,
and a later `sim/`-substrate spiking version WOULD instantiate STD vs STF synapses. Cite it; don't fake it.)

### 1.6 What it achieves that plain FA doesn't; and the honest depth/scale limits

- **Achievement:** the recurrent burst dynamics + the moving-average baseline **linearize the credit signal** and
  **align the feedback**, so BDSP approximates the loss gradient through depth far better than static FA. Concretely
  Payeur 2021 reports (rate + spiking): MNIST test error **~1.1%** for Burstprop == backprop == FA (MNIST is too easy
  to separate them); **CIFAR-10** with several hidden/conv layers where Burstprop clearly tracks backprop and beats
  fixed-feedback FA; and **ImageNet** top-5 error **~56.1%** — "much better than keeping feedback weights fixed [FA],
  and much closer to full gradient descent." THIS is the depth-scaling evidence vanilla FA lacks: on hard/deep problems
  burstprop moves toward backprop while FA stalls.
- **Honest limits (where IT breaks):** (a) The ImageNet gap to backprop is still large — Payeur attribute it to being
  **unable to use the recurrent (credit-linearizing) input at convolutional layers due to memory limits**, which
  degraded feedback linearity at early layers. So the *depth benefit depends on the recurrent linearization actually
  running at every layer.* (b) The clean theory assumes an **ensemble** (many neurons share an event/burst code so `P`
  is a well-estimated fraction); single-neuron burst estimates are noisy. (c) Greedy 2022 (BurstCCN) is in part a
  *critique*: the original burstprop's feedback linearization/alignment is imperfect and they add explicit
  cortico-cortical feedback learning to fix it — i.e. burstprop is better-than-FA but not backprop-exact, and the
  residual gap is the feedback-alignment quality.

---

## MECHANISM 2 — Dendritic microcircuit / self-predicting network (Sacramento, Costa, Bengio & Senn, NeurIPS 2018)

**Source (equations extracted verbatim from the paper):** Sacramento, Costa, Bengio, Senn, *Dendritic cortical
microcircuits approximate the backpropagation algorithm*, NeurIPS 2018 (arXiv 1810.11393; ar5iv HTML). Follow-on that
extends it single-phase to deeper nets + names its scale limits: Greedy et al. 2022 (BurstCCN, NeurIPS).

### 2.1 The circuit (per hidden layer `k`)

Two populations per layer: **pyramidal** cells `P` (two dendritic compartments + soma) and lateral **SST-like
interneurons** `I` (one dendrite + soma). Wiring:
- bottom-up `W^{PP}_{k,k-1}`  : layer `k-1` pyramids -> layer `k` pyramidal **basal** dendrite.
- top-down `W^{PP}_{k,k+1}`   : layer `k+1` pyramids -> layer `k` pyramidal **apical** dendrite (FIXED-random or slowly
  learned; the feedback path).
- lateral `W^{IP}_{k,k}`      : layer `k` pyramids -> layer `k` interneuron dendrite (the interneuron *learns to mimic
  the top-down feedback*).
- lateral `W^{PI}_{k,k}`      : layer `k` interneurons -> layer `k` pyramidal **apical** dendrite (cancels the top-down
  prediction).
- teaching nudge: layer `k+1` pyramid soma weakly nudges the layer `k` interneuron soma (conductance `g_som`).

### 2.2 Membrane dynamics (exact equations)

Pyramidal soma (eq. 1):
```
    d/dt u^P_k = -g_lk * u^P_k + g_B*(v^P_{B,k} - u^P_k) + g_A*(v^P_{A,k} - u^P_k) + sigma*xi(t)   (M2.1)
```
Interneuron soma (eq. 2):
```
    d/dt u^I_k = -g_lk * u^I_k + g_D*(v^I_k - u^I_k) + i^I_k(t) + sigma*xi(t)                        (M2.2)
      with teaching current   i^I_k = g_som*(u^P_{k+1} - u^I_k)                                       (M2.2b)
```
Dendritic potentials (instantaneous functions of presynaptic rates `r = phi(u)`):
```
    v^P_{B,k} = W^{PP}_{k,k-1} * phi(u^P_{k-1})                       (basal / feedforward)           (M2.3)
    v^P_{A,k} = W^{PP}_{k,k+1} * phi(u^P_{k+1}) + W^{PI}_{k,k} * phi(u^I_k)   (apical = top-down + interneuron)  (M2.4)
    v^I_k     = W^{IP}_{k,k}   * phi(u^P_k)                            (interneuron dendrite)          (M2.5)
```
Rate function `phi`: logistic `phi(u)=1/(1+e^-u)` (MNIST) or soft-ReLU `phi(u)=gamma*log(1+e^{beta(u-theta)})`.

### 2.3 The three plasticity rules (all LOCAL dendritic-prediction-error rules; form `eta*(phi(u)-phi(v_hat))*r_pre`)

Bottom-up pyramidal (eq. 7) — soma vs *attenuated basal* prediction:
```
    d/dt W^{PP}_{k,k-1} = eta^{PP} * ( phi(u^P_k) - phi(v_hat^P_{B,k}) ) * phi(u^P_{k-1})^T           (M2.6)
      with  v_hat^P_{B,k} = [ g_B / (g_lk + g_B + g_A) ] * v^P_{B,k}   (dendritic attenuation factor)  (M2.6b)
```
Pyramid->interneuron (eq. 8) — interneuron learns to predict its own soma from its dendrite:
```
    d/dt W^{IP}_{k,k} = eta^{IP} * ( phi(u^I_k) - phi(v_hat^I_k) ) * phi(u^P_k)^T                       (M2.7)
      with  v_hat^I_k = [ g_D / (g_lk + g_D) ] * v^I_k                                                  (M2.7b)
```
Interneuron->pyramid apical (eq. 9) — inhibitory homeostatic rule that drives the apical toward its rest value
`v_rest` (=0), i.e. **learns to silence the apical when there is no external teaching**:
```
    d/dt W^{PI}_{k,k} = eta^{PI} * ( v_rest - v^P_{A,k} ) * phi(u^I_k)^T   ,  v_rest = 0                (M2.8)
```
(Optional, eq. 10: top-down `W^{PP}_{k,k+1}` can be *learned* to minimize an inverse-reconstruction loss ->
target-propagation flavor; or left FIXED-random -> feedback-alignment flavor.)

### 2.4 The self-predicting condition + the local error

At convergence the interneuron circuit must **cancel the top-down feedback in the absence of a target** — the
"self-predicting" fixed point (eqs. 11-12):
```
    W^{PI*}_{k,k} = - W^{PP}_{k,k+1}                                                                    (M2.9)
    W^{IP*}_{k,k} = [ (g_B + g_lk) / (g_B + g_A + g_lk) ] * W^{PP}_{k+1,k}                              (M2.10)
```
When (M2.9)-(M2.10) hold and NO teaching signal is applied, the apical potential is **zero**: the interneuron's
inhibition exactly cancels the top-down excitation. When a teaching signal nudges the OUTPUT layer, the mismatch
propagates *up the apical dendrites* as a nonzero local error. Under self-prediction the apical potential reduces to
(supp. eq. 16):
```
    v^P_{A,k} = W^{PP}_{k,k+1} * [ phi(u^P_{k+1}) - phi(u^I_k) ] = W^{PP}_{k,k+1} * e_{k+1}             (M2.11)
      where  e_{k+1} = phi(u^P_{k+1}) - phi(u^I_k)  is the layer-(k+1) prediction error.
```
This `v^P_{A,k}` is the **local, dendritically-computed error signal** that drives the bottom-up rule (M2.6) — it plays
the role of backprop's `delta_k`.

### 2.5 Why it credit-assigns through depth; no weight transport; honest limits

- **No weight transport:** the top-down `W^{PP}_{k,k+1}` are a *separate physical pathway* from the bottom-up
  `W^{PP}_{k,k-1}`; they are fixed-random (feedback alignment) or slowly learned (target-prop), never copied from the
  forward weights. The *interneuron* is what makes it work better than raw FA: by learning `W^{IP}` and `W^{PI}` to
  self-predict, it forces the apical to encode a genuine cancellation-based error rather than a raw random projection.
- **Backprop approximation (weak-feedback limit `lambda -> 0`, `lambda = g_som/(g_lk+g_B+g_som)`):** the paper proves
  the pyramidal soma equals feedforward activation + `lambda^{N-k+1} * (product of D_l W_{l,l+1}) * output_error`, i.e.
  the update matches backprop to leading order in `lambda`, with *arbitrary (non-symmetric) top-down weights* — this is
  feedback alignment made gradient-faithful by the microcircuit.
- **Runs continuously (single relaxation), no separate forward/backward phases** (unlike Guerguiev 2017 / equilibrium
  prop) — a plausibility win.
- **Honest limits (where IT breaks):** demonstrated up to **two hidden layers** (784-500-500-10 MNIST, **1.96% test
  error** vs backprop 1.53%; a 30-50-10 nonlinear-regression toy; a 784-1000-10 with plastic top-down at 2.48%). The
  paper does NOT demonstrate CIFAR/ImageNet or many-layer depth; the gradient match is only *leading order in
  `lambda`* (weak feedback), so strong-feedback / very-deep regimes lose the guarantee; and the self-predicting state
  must be *maintained* (the interneuron learning must keep pace) or the apical error is corrupted. Greedy 2022 flags the
  microcircuit's feedback as a scale bottleneck and rebuilds it.

---

## DELTA FROM EXISTING CODE

**What exists today** (all read in full):
- `sim/dendritic_mlp.py` — `DendriticMLP(sizes, seed)`: a deep sigmoid MLP; per hidden layer a forward `W[li]` + a
  **fixed-random per-layer feedback `B[li]` of shape `(n_out, hidden)`** (set once from seed, never learned, never
  derived from `W` — genuine no-weight-transport FA). `train_step(X,y,mode,lr)` computes the output error
  `e = softmax(logits) - onehot(y)` and for each hidden layer `li` forms `ap = e @ B[li]` (the top-down projected
  error), then the local update `base = a_prev.T @ (ap * a_l*(1-a_l))` (this is exactly the Urbanczik-Senn
  apical-driven form, M-analog). Modes: `local_correct` (FA), `local_wrongsign`, `global_scalar`, `permuted`,
  `oracle` (fenced hand-derived backprop, measurement only). Mean-over-batch + heavy-ball momentum optimizer applied
  identically to every mode. GPU-pluggable, deterministic seeded init, `hidden_grad_alignment(X,y)` measurement.
- `sim/dendritic_plasticity.py` — `urbanczik_senn_update(...)`: the committed LOCAL rule; with `apical_signal` supplied
  it returns `dw = outer(pre, lr*gate*(apical_signal * soma*(1-soma)))`.
- `sim/dendritic_neuron.py` — `DendriticLayer`: a two-compartment (basal/apical) spiking-ish unit with fixed-random
  `B_apical`, Larkum BAC threshold-lowering. Single layer; not wired into a deep stack.
- `sim/predictive_coding.py` — `PredictiveCoder`: a Rao-Ballard next-token predictor (unrelated to spatial credit
  assignment; not reusable here except as a style reference).

**The precise gap.** `DendriticMLP` already IS the vanilla-FA deep learner and already has the two things both
candidates need: (1) a separate fixed-random feedback path `B` per layer (no weight transport), and (2) a per-layer
apical-error injection point (`ap = e @ B[li]`). What it LACKS, per candidate:

### Delta to add MECHANISM 1 (Burstprop) — the SMALLER delta
Add to a `DendriticMLP`-sibling (do NOT mutate the shipped class):
1. **A burst-probability read-out per hidden unit.** Forward as usual to get event rates `e_l = a_l = sigmoid(...)`.
   For the credit pass, compute apical drive per layer from the layer ABOVE's burst-coded error and set
   `p_l = sigmoid(beta * v_api,l)` with baseline `p0=0.5` at `v_api=0` (M1.1). At the OUTPUT layer, `v_api,N` is set
   directly by the loss (target minus output), so `p_N - p0 ∝ -(softmax(logits)-onehot(y))` (the top-down teaching).
2. **A top-down burst-error recursion** (M1.5): `v_api,l = (e_{l+1}*(p_{l+1}-p0)) @ B[l]` — REUSE the existing fixed
   `B[l]` as the feedback matrix `Y_l`. (Optionally add the recurrent *linearization* term — a within-layer relaxation
   that multiplies the incoming error by the local `phi'` — this is the piece Payeur credit for closing the FA gap;
   start WITHOUT it to see the bare burst channel, then add it as the "linearized" arm.)
3. **The BDSP feedforward update** (M1.4): `dW_l = e_{l-1}^{pre .T} @ ( e_l*(p_l - p_bar_l) )`, with `p_bar_l` a slow
   EMA of `p_l` (initialize `p_bar=p0` so the first step is unbiased). This REPLACES the `base = a_prev.T @ (ap*...)`
   line for the `burst` mode only.
   - Net new state: `p_bar_l` EMA buffers (one per hidden layer), `beta`, `p0`. ~30-50 lines. No `sim/` edit needed.

### Delta to add MECHANISM 2 (microcircuit) — the LARGER delta
1. **An interneuron population per hidden layer** with its own soma/dendrite states and TWO new weight matrices
   `W_IP[l]` (pyr->int) and `W_PI[l]` (int->pyr apical), plus the fixed/slow top-down `W_PP_td[l]` (reuse `B[l]`).
2. **A relaxation step**: iterate (M2.1)-(M2.5) to (approximate) steady state per input (or use the paper's
   closed-form steady-state in the rate limit — much cheaper for numpy) to get `u^P_l`, `u^I_l`, and the apical error
   `v^P_{A,l}` (M2.11).
3. **THREE plasticity rules** (M2.6, M2.7, M2.8) instead of one, plus initialization at/near the self-predicting
   condition (M2.9-M2.10) OR letting `W_IP`/`W_PI` learn into it (the honest version — but then the interneuron
   learning must converge, adding an inner loop / more epochs).
   - Net new state: interneuron activations + `W_IP`,`W_PI` per layer, `g_B,g_A,g_D,g_lk,g_som` constants, a relaxation
     loop. ~120-200 lines. Still numpy, still no `sim/` edit for the rate version.

---

## RECOMMENDATION — digitize MECHANISM 1 (Burstprop) FIRST for EMERGE-1b

**Why Burstprop over the microcircuit as the first probe:**
1. **Smallest faithful delta on the EXISTING harness.** It reuses `DendriticMLP`'s fixed-random `B` verbatim as the
   feedback matrix, adds a burst-probability read-out + an EMA baseline + one changed update line. The microcircuit
   needs an entire interneuron population, two extra weight matrices, a relaxation loop, and self-predicting init/inner
   learning — 3-4x the code and 3-4x the ways to get it subtly wrong (and its own convergence to babysit).
2. **It is the mechanism with published DEPTH-SCALING evidence.** Burstprop is the ONLY one of the two demonstrated to
   *approach backprop on hard/deep problems where FA stalls* (CIFAR-10 multi-layer; ImageNet top-5 ~56.1% "much closer
   to gradient descent than fixed feedback"). The microcircuit's published ceiling is 2-hidden-layer MNIST (1.96%) —
   it does not itself demonstrate the depth win we are testing for. Since EMERGE-1b's whole question is "does a
   *faithful* rule generalize through depth where vanilla FA memorized," the candidate with the depth evidence is the
   right first bet.
3. **Faithfulness is high and clean.** BDSP is a genuine three-factor LOCAL rule (post burst rate, post event rate, pre
   trace) grounded in real pyramidal burst physiology (Larkum BAC = catalog G.02; Naud-Sprekeler multiplexing) — it
   sits squarely inside the owner's BRAIN-BASED-ONLY standard, and the apical->burst-probability control is exactly the
   apical-basal coincidence the catalog names.

**Minimal EMERGE-1b implementation plan (numpy first, reuse-by-import, NO `sim/` edit):**
- New runner `research/runners/_emerge1b_burstprop_credit_derisk.py` + a new **learner** class next to `DendriticMLP`
  (either a sibling class `BurstpropMLP` in a NEW module `research/runners/_burstprop_mlp.py`, or a `mode="burst"`
  branch — but a sibling class is cleaner because BDSP needs extra per-layer state). It subclasses/mirrors
  `DendriticMLP` so ALL the harness plumbing (`_forward`, `accuracy`, `loss`, `hidden_grad_alignment`, the
  mean-batch+momentum optimizer, the seeded no-weight-transport init) is reused unchanged.
- Implement (M1.1) `p=sigmoid(beta*v_api)`, (M1.5) `v_api,l = (e_{l+1}*(p_{l+1}-p0)) @ B[l]` cascaded top-down from the
  output error, (M1.4) `dW_l = e_{l-1}.T @ (e_l*(p_l - p_bar_l))`, EMA `p_bar`. Start with `beta=1.0`, `p0=0.5`.
- **Two arms of the mechanism** to separate "burst channel alone" from "burst + linearization": `burst_bare` (no
  recurrent linearization) and `burst_linearized` (multiply the descending error by local `phi'(u_l)=a_l*(1-a_l)` at
  each hop — the piece Payeur say closes the FA gap). This directly tests WHICH part buys the depth generalization.
- **Reuse the EMERGE-1 task + splits + arms VERBATIM** (`make_task`, the depth-2 threshold-of-5-pair-XORs over 10 bits,
  65/35 held-out split, the linear latent probe, `hidden_grad_alignment`). Same 3 seeds (42/43/44), same
  epochs/lr/hidden defaults; add a 900-ep/lr-0.7 confirm like EMERGE-1 did.
- Cost: CPU, minutes. If GO at toy scale, the honest follow-on is a **larger/deeper** version (more XOR levels or small
  vision) to confirm the depth benefit persists — still numpy, still no `sim/` edit.
- **Where a `sim/` spiking version would later be needed (allowed, faithful):** to make burstprop *fully spiking on the
  one-brain substrate*, a later build would add real bursts to a two-compartment `NeuronModel` (event = first spike,
  burst = ISI<thresh) and STD-vs-STF synapses to demultiplex `E` vs `B` (Naud-Sprekeler). That is the months-scale
  substrate step and is OUT OF SCOPE for EMERGE-1b — the rate-model numpy probe must GO first (else the substrate build
  is moot, exactly the EMERGE-1 build-saving logic).

---

## GO / ANTI-CHEAT DESIGN for EMERGE-1b (mirror EMERGE-1 exactly, add the burstprop-specific self-checks)

**GO (multi-seed, seeds 42/43/44):** the faithful burst rule **GENERALIZES through depth** where vanilla FA memorized:
- `heldout(burstprop) >= 0.75` AND `> heldout(vanilla_FA=local_correct) + 0.10` (the decisive contrast: same net, same
  fixed `B`, only the rule changed) AND `> apical_lesion_floor + 0.10` AND `> chance + margin`.
- **Structure emerges:** the level-1 XOR linear probe `>= 0.70` and `> apical_lesion probe + 0.10`; and/or
  `hidden_grad_alignment` climbs during training (FA alignment earned).
- **Train-vs-heldout gap SHRINKS** relative to vanilla FA (vanilla FA drove train->1.0 with heldout~0.58; a GO burst
  rule should generalize, not just memorize — check heldout tracks train, not a widening gap).

**Anti-cheats (each must hold, else NO-GO / artifact):**
1. **Apical/feedback lesion collapses it.** Set `B[l]=0` (or force `v_api=0` so `p_l≡p0`): the burst rule must fall to
   the no-credit floor (~single-layer/chance) and the probe to ~0.5. Proves the top-down burst channel is load-bearing,
   not the forward pass alone. (Reuse EMERGE-1's `apical_lesion` arm.)
2. **Wrong-sign anti-learns.** Flip the sign of `(p_l - p_bar_l)` (or of `B`): held-out must be <= chance+0.05 (drive
   credit the wrong way -> below chance). Proves the *sign/content* of the burst-coded error matters, not just its
   presence.
3. **Baseline / no-teaching null (burstprop-specific, the moat analog).** With the target detached (`v_api,N=0` ->
   `p_l≡p0` everywhere), the BDSP update `[b_i - p_bar_i*e_i]` must be ~0 and produce **no net learning** (held-out
   stays ~chance, weights ~unchanged). This is the "no spurious learning at rest" check — the direct test that `p0`
   is set correctly. A failure here = the rule learns garbage with no teacher = a broken/cheating implementation.
4. **Memorization floor / permuted-label control.** Shuffle `y` -> the rule can only memorize; held-out must be
   ~chance while train can rise. Confirms held-out measures generalization, not leakage.
5. **Oracle ceiling confirms task-learnability.** The fenced backprop oracle arm must reach `>=0.80` held-out (EMERGE-1
   got 0.95) — else INCONCLUSIVE (task/config bug), not a mechanism verdict.
6. **No-weight-transport self-check.** Assert `B[l]` is never written after init and is never a function of any
   `W[l]` (byte-check `B` unchanged across `train_step`; assert `id(B)`/values independent of `W`). Burstprop uses the
   SAME fixed `B` vanilla FA used, so this is inherited — but assert it explicitly so the GO cannot be a
   weight-transport artifact.
7. **The decisive within-net contrast is the headline.** vanilla-FA and burstprop share the identical network, seed,
   init, fixed `B`, task, splits, and optimizer — the ONLY difference is the plasticity rule. So a burstprop-vs-FA
   held-out gap is attributable to the rule, not to any confound. Report it as the primary number.

**Honest expected outcome (calibrated, not optimistic).** This is a genuine open question and a BOUNDARY is a real
possibility — but the prior is **meaningfully better than for vanilla FA**, and this is worth running:
- **Most likely: a PARTIAL/qualified GO.** Burstprop's published depth advantage is real but *modest and
  regime-dependent* (it needs the recurrent linearization running; on ImageNet without it the gap to backprop stayed
  large). On the depth-2 XOR toy the `burst_linearized` arm has a fair chance to clear held-out ~0.75-0.85 (a clear win
  over FA's 0.58) while `burst_bare` may only partially beat FA — which would itself be the informative result:
  *the linearization, not the burst channel per se, is what buys depth-generalization.*
- **Plausible: BOUNDARY.** At this tiny scale with a single hidden width, the ensemble assumption (well-estimated `P`)
  is weakest and the depth-2 task may be too small to reveal the benefit; burstprop could memorize like FA. That is
  still build-saving and sharpens the map (it would say: the faithful *rate* rule also needs scale/ensemble, pointing
  to the microcircuit as arm 2 or to the honest re-scope).
- **Either way it is decision-useful:** GO -> localizes the months-scale substrate build to the burst two-compartment +
  STD/STF substrate (and promotes the microcircuit to a follow-on comparison). BOUNDARY -> the honest posture is the
  EMERGE-1 re-scope (backprop-trained components do deep-emergent structure; the spiking brain does grounding /
  continual memory / the no-confab moat / embodiment), now *doubly* evidenced (both the weakest AND a strong plausible
  local rule failed at depth on our substrate/scale).
- **If Burstprop is a clean GO, run Mechanism 2 (microcircuit) as the confirming second arm** — it is the more
  gradient-faithful mechanism (proven backprop-approximation) and a GO there would show the depth-generalization is
  robust across two independent faithful mechanisms, not a burstprop artifact.

---

## Sources (cited)
- Payeur, Guerguiev, Zenke, Richards, Naud. *Burst-dependent synaptic plasticity can coordinate learning in
  hierarchical circuits.* Nat Neurosci 2021. DOI 10.1038/s41593-021-00857-x; preprint bioRxiv 2020.03.30.015511;
  Author Correction PubMed 34728832 (DOI 10.1038/s41593-021-00970-x). [BDSP rule M1.2, multiplexing, MNIST 1.1% /
  CIFAR / ImageNet top-5 56.1%.]
- Naud & Sprekeler. *Sparse bursts optimize information transmission in a multiplexed neural code.* PNAS 2018.
  [Event rate = feedforward / burst prob = feedback; STD decodes events, STF decodes bursts.]
- Sacramento, Costa, Bengio, Senn. *Dendritic cortical microcircuits approximate the backpropagation algorithm.*
  NeurIPS 2018; arXiv 1810.11393. [Eqs. M2.1-M2.11 extracted verbatim; MNIST 784-500-500-10 = 1.96% test error.]
- Greedy, Zhang, Najafi, Bengio, Richards, Costa. *Single-phase deep learning in cortico-cortical networks*
  (BurstCCN). NeurIPS 2022; arXiv 2206.11769. [Rate-model formalization + scale critique of both prior models.]
- *A burst-dependent algorithm for neuromorphic on-chip learning of spiking neural networks.* Neuromorph. Comput. Eng.
  2025 (IOP 2634-4386/adb511). [Spiking BDSP with moving-average burst probability; baseline-bursting-> no weight
  change.]
- Urbanczik & Senn 2014; Guerguiev, Lillicrap, Richards 2017 (the segregated-dendrite FA precedents already in
  `sim/dendritic_plasticity.py` / `sim/dendritic_neuron.py`).
- Catalog `sim-catalog/references/feature-catalog.md` G.02 (active dendrites — Larkum BAC apical-basal coincidence ->
  bursts; Kandel 6e Ch 13 pp 293-298).
- A 2026 follow-on worth tracking (not required for EMERGE-1b): *Cell-type-specific cortical feedback coordinates
  hierarchical credit assignment*, bioRxiv 2026.06.16.732595.

**Do NOT commit — the controller reviews + commits.**
