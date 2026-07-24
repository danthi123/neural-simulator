# gap#4 on-bridge SPIKING self-predicting microcircuit — DESIGN (build-ahead; GPU run deferred until the 3090 frees) (2026-07-24)

**Status: DESIGN + SKELETON + construct-smoke ONLY.** No GPU run launched (the 3090 is busy). This doc + the
skeleton `research/runners/_gap4_onbridge_spiking_selfpredict_derisk.py` are ready to launch the decisive multi-seed
GPU de-risk the instant a lane frees. **Owner review requested before the full run.**

---

## 1. What this closes, and why it is the decisive next de-risk

gap#4 = deep directed credit assignment via the Sacramento-Senn (2018) self-predicting microcircuit: an apical
dendrite that is **silent-when-correct**, a **PLASTIC feedback/interneuron weight W^PI**, **NO weight transport**
(biologically-plausible feedback), and **LEARNED feedback (Kolen-Pollack / Akrout weight-mirror) beating FIXED
random feedback (feedback alignment)**.

The CPU numpy-RATE phase is a **6/6-seed GO** (`_gap4_learned_microcircuit_selfpredict_derisk`, finding
`2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO.md`): credit ≫ reservoir (0.694 vs 0.117), the
apical-silent-when-correct property is **EARNED** (silent_ratio 0.087 with plastic W^PI from a noisy init, selfpred_cos
0.89 → W^PI learned Y; vs frozen-noisy W^PI ratio 1.004), all anti-cheat guards pass.

**The load-bearing limitation (flag G2, verbatim from the CPU finding):** at the numpy RATE reference the LEARNED arms
(kp, plastic-Eq.9 micro) are **accuracy-byte-identical to plain fixed-random feedback-alignment** — the
learned/interneuron machinery is *inert on the feedforward weights at rate*. **The learned-vs-fixed SEPARATION only
appears on SPIKES**, where the burst-coded, finite-sample, saturating credit degrades a fixed-random projection's
alignment in a way a learned/aligned projection resists. **⇒ the on-bridge SPIKING run is the one that actually tests
the gap#4 hypothesis.** That is exactly why this port is the decisive next step, and why it must be on the real
`SimulationBridge`, not a rate reference.

---

## 2. The substrate is already built — this is a RUNNER, not a `sim/` edit

The committed, additive/default-off `sim/` machinery for a two-compartment spiking self-predicting microcircuit
**already exists** and is byte-identical to today when its flags are off:

| Committed flag / array (in `sim/config.py`, consumed in `sim/bridge.py:_run_one_simulation_step`) | Role |
|---|---|
| `enable_bdsp` | two-compartment spiking pyramidal: basal soma (`cp_membrane_potential_v`) + apical (`cp_v_apical`); event rate `cp_bdsp_E`, burst rate `cp_bdsp_B`, burst-prob `cp_bdsp_P`, EMA baseline `cp_bdsp_Pbar`; the FF rule `dw = eta·Etilde_pre·(B − Pbar·E)` moves plastic `cp_connections` edges via `fused_bdsp_update`. |
| `cp_bdsp_apical_drive` | runner-set per-neuron top-down apical current → integrated into `cp_v_apical` (the credit channel). |
| `enable_bdsp_microcircuit` + `cp_bdsp_int_drive` | the SST-interneuron cancellation: the bridge integrates **`(apical_drive − int_drive)`** into `cp_v_apical` (bridge.py ~7914-7917), so P/B ride on the CANCELLED clean error. Unreached (byte-identical) unless the flag is on AND both arrays are set. |
| `enable_bdsp_graded_credit` | swaps the credit factor from the noisy measured burst `B` to the graded expectation `E·P` (kernel identity `B − Pbar·E == E·(P − Pbar)`): the low-variance clean-error credit that rides the apical directly, and the moat holds EXACTLY at rest (`P == Pbar`). |
| `bdsp_apical_couples_soma` / `bdsp_apical_soma_g` | electrotonic apical→soma coupling so a top-down apical raises the MEASURED burst rate B (only needed for the measured-B path; the graded path does not need it). |
| `bdsp_apical_bistable` (+ `coincidence_plateau_self_regen`, `apical_kir_g`) | gap#4 bistable apical: latch/hold the teaching error across the eligibility window. |

**Reuse-by-import base:** `research/runners/_semantic_inheritance_onbridge_spiking_derisk.py` already builds a **depth-2
two-compartment spiking net on ONE bridge** (input(9 features) → H1 → H2 → out(k classes)) with population coding
(`pool_k`), fixed-random Y credit descent, `int_drive` wiring, and the full anti-cheat battery — for its own arms
(`plain_fa` / `burstprop` / `microcircuit`). Its `OnBridgeBDSPNet` solves the hard on-bridge problems (the
graded-feature drive, the population-coded read that lifts the finite-spike-noise wall, the per-example online credit
pass). **The gap#4 port SUBCLASSES it** and adds the gap#4-specific feedback modes + the genuinely-new plastic-Eq.9
W^PI. **NO `sim/` edit is required** (see §8).

---

## 3. Architecture (the gap#4 net on one bridge)

Logical layout on ONE `SimulationBridge`, reusing `OnBridgeBDSPNet`:

```
 input (n_features logical units, graded feature drive)
   │  ff_0  (PLASTIC cp_connections, moved by fused_bdsp_update)
   ▼
 H1  (hidden, two-compartment BDSP neurons)
   │  ff_1  (PLASTIC)
   ▼
 H2  (hidden, the TOP hidden layer — carries the plastic W^PI interneuron cancellation for the micro arm)
   │  ff_2  (PLASTIC)
   ▼
 out (k class units — argmax over pooled event rate = the prediction; the output "has target access")
```

- **Depth = 2 hidden layers** (`n_hidden_layers=2`), matching the CPU reference's `--deep-layers 2`. The CPU op-point
  precheck showed depth-2 fixed-FA has *no gap* to close **at rate**; the gap#4 finding's plan is explicit that on the
  bridge the **spiking SPARSITY** at depth-2 is what degrades fixed-FA to ≤ reservoir (the analogue of rate-depth-3),
  giving the learned feedback a gap to close. So depth-2 is correct; the degradation comes from spikes, not from adding
  rate-depth.
- **Population coding** (`pool_k` neurons per logical unit): each logical unit = a contiguous block of K bridge
  neurons; the layer activation is the block-mean `cp_bdsp_E`; the credit is broadcast to all K. K=1 is byte-identical
  to the single-neuron net. K≥8 is the validated fix for the single-neuron finite-spike-noise wall.
- **Task:** `make_task_semantic_inheritance` (imported verbatim) — the XOR-over-pool compositional-inheritance task,
  the ONLY instrument the record proves is genuinely depth-required AND transport-free-learnable. GO metric =
  **held-out INHERITANCE accuracy** on `idx["inh_idx"]` (a NOVEL member of a taught superordinate; requires composing
  member→super→property across the 2 hidden layers). 9 continuous standardized features → graded input current.

### Apical vs basal mapping (the two-compartment substrate)
- **Basal = the feedforward channel.** The bottom-up drive (input features / lower-layer spikes through the plastic FF
  `cp_connections`) charges the soma `cp_membrane_potential_v`; the neuron spikes; its EVENT rate `cp_bdsp_E` is the
  layer activation. Invariant to the apical (the multiplexing invariant).
- **Apical = the top-down credit channel.** The runner sets `cp_bdsp_apical_drive[neuron]` = the descending credit for
  that neuron; the bridge integrates it into `cp_v_apical`; `P = sigmoid(β·scale·(v_apical − E_rest) + logit(p0))`
  rides the apical. At rest (apical = 0 ⇒ v_apical = E_rest) `P == Pbar == p0` ⇒ the credit factor `E·(P − Pbar) == 0`
  ⇒ **the P0 no-spurious-learning moat**. The apical raises/lowers P → the graded credit `E·(P − Pbar)` signs LTP/LTD
  per postsynaptic neuron, and the committed FF kernel moves that neuron's incoming plastic weights.

**⇒ The credit that changes the deep feedforward weights is carried by the spiking substrate** (each postsynaptic
layer's apical-modulated burst/graded deviation drives `fused_bdsp_update` over `cp_connections`). See §7 for the
honest boundary on what is host-side.

---

## 4. The five arms (like-for-like: same bridge, task, seed, init; only the feedback rule differs)

| arm | feedback rule | what it tests |
|---|---|---|
| `reservoir` | hidden apical = 0 (both H1,H2 frozen at random init: apical off ⇒ dev = 0 ⇒ no FF move); only the H2→out readout learns (output apical on). | the credit-INDEPENDENT baseline — the floor the learned feedback must climb above. |
| `fixed_fa` | fixed-random Y feedback, full sequential-FA descent, graded BDSP FF plasticity on spikes. | the fixed-feedback credit — **the thing the learned arms must beat** (the G2 control). |
| `kp` | Kolen-Pollack LEARNED feedback: `dY[k] = kp_lr·outer(e_above, E_k) − kp_decay·Y[k]` (Akrout weight-mirror; aligns Yᵀ→W). TRANSPORT-FREE: reads only the descending error + the layer's event rate + Y; never a forward weight. | learned feedback fixing the credit **DIRECTION**. |
| `micro` | plastic-Eq.9 self-predicting **interneuron W^PI** at the top hidden layer H2, init NOISY, learned by the local self-prediction rule toward the fixed point W^PI == Y (apical-silent EARNED). Delivered on-bridge as `apical_drive[H2] = onehot(y)@Y` and `int_drive[H2] = softmax@W^PI` → the bridge integrates `drive − int_drive` = `(onehot − softmax)@Y` = the clean FA credit, **silent when correct**. | the **genuinely-new** build: learned feedback via the noise-robust interneuron cancellation (Sacramento-Senn M2.11). |
| `transport_ceiling` | Y := (pooled forward W)ᵀ each step (weight transport ≈ backprop). | the labeled CHEAT upper bound; **its no-weight-transport guard MUST FAIL** (that failure is the proof the guard works). |

**How W^PI is realized as an on-bridge plastic pathway (micro arm):** it is a runner-held logical-unit matrix
(shape = Y[top].shape = `(k_classes, H2_units)`) whose value is projected into the physical `cp_bdsp_int_drive` array
(broadcast to the K-pool) each credit pass. It is LEARNED by the local, transport-free self-prediction update
`dW^PI = +wpi_lr·(src_pred_predᵀ @ (src_pred_pred @ (Y − W^PI)))` (reads only the interneuron rate `src_pred`,
= softmax of the net's own readout, plus Y and W^PI — never a forward weight). The `int_drive` it produces is a
genuine physical cancellation current the committed `enable_bdsp_microcircuit` block subtracts on the bridge. This is
the on-substrate M2.7/M2.8 interneuron loop realized runner-side per the D1 build spec (which explicitly keeps the
interneuron layer-structured on the runner side to minimize the `sim/` surface; the burst detector + the
apical-drive/int-drive subtraction run in `sim/`).

**"Apical-silent-when-correct" read from spikes:** with the plastic W^PI, on a CORRECT sample `src_pred ≈ onehot(y)`
⇒ the injected effective apical `drive − int_drive ≈ (onehot − softmax)@Y ≈ 0` ⇒ `cp_v_apical` stays near `E_rest`
⇒ P ≈ Pbar ⇒ the graded credit ≈ 0 (silent). On an INCORRECT sample the residual is large (loud). We measure
`silent_ratio = mean|effective apical|_correct / mean|effective apical|_incorrect` (both the injected drive and the
read-back `cp_v_apical − E_rest` on the H2 slice) and `selfpred_cos(W^PI, Y)`. **EARNED-silent** ⇒ ratio ≪ 1 with a
plastic W^PI (cos → 1) vs ratio ≈ 1 with a frozen-noisy W^PI (cos ≈ 0) — the on-bridge analogue of the CPU
observable.

**Graded credit is ON by default** (`enable_bdsp_graded_credit=True`): it is the on-bridge realization of the
clean-error (M2.6-somatic-analog) credit the CPU MicroNet uses, and it keeps the moat exact and avoids the
"apical-decoupled-from-soma" boundary the measured-B path hits (`_d1_onbridge_learn_to_accuracy` BOUNDARY verdict).
The measured-B + `bdsp_apical_couples_soma` path is retained as an opt-in fallback.

---

## 5. The exact GO-gate + anti-cheat controls

**GO-gate (the decisive spiking claim):**
> **best(kp, micro) held-out inheritance > fixed_fa on spikes, by a real margin, ≥ 5/6 seeds** — in a regime where
> **fixed_fa ≤ reservoir + margin** (the spiking FA-wall the learned feedback is meant to close), with the
> task-validity gate (oracle ≥ 0.80, single-layer floor ≈ chance) and ALL anti-cheats passing.

If no reachable spiking regime yields both (fixed_fa degraded to ≤ reservoir) AND (learned climbs back above), the
honest verdict is **"scale-frontier at this budget"** — a characterized boundary, itself the deliverable, NOT a
license to abandon the capability.

**Anti-cheat controls (built in; they are the point):**
1. **fixed-FA control** — the `fixed_fa` arm IS the baseline the learned arms must beat (the G2 separation test).
2. **weight-transport lesion / ceiling** — `transport_ceiling` (Y := Wᵀ): its `no_weight_transport()` guard MUST
   FAIL (detecting Y == Wᵀ). Plus the structural guards on kp/micro (AST: the KP/W^PI update methods never read a
   forward-W array; Y from a separate RNG stream, never byte-equal to a forward block/its transpose).
3. **shuffle** — (a) shuffled TRAINING target, eval on TRUE labels → chance (no leak); (b) `shufE` directed-credit
   scramble (permute the descending error across the batch in the HIDDEN path only) → the credit collapses to chance.
4. **no-plasticity frozen** — `bdsp_learning_rate = 0` (or apical-lesion: zero ALL apical) → held-out at chance AND
   the hidden FF weights barely move (the P0 moat).
5. **apical-silent EARNED (Sacramento dissociation)** — plastic W^PI silent_ratio ≪ frozen-noisy W^PI silent_ratio,
   with selfpred_cos(W^PI, Y) → 1 (plastic) vs ≈ 0 (frozen). The genuinely-new property, read on the bridge.
6. **memorization control** (from the task) — `memctrl_idx` (untaught super, novel class) must stay ≈ chance = no
   per-super leakage.

**Seed discipline (a real bug — read before trusting any number):** set **`cfg.seed`** (the field the bridge reads).
`actual_seed_used` SEEDS NOTHING. Hash `cp_neuron_firing_thresholds` across two builds at the same seed and confirm
byte-identical before trusting seed-to-seed comparisons (the deep-credit arc was confounded by exactly this for
months). The reused `OnBridgeBDSPNet` already sets `cfg.seed` correctly; the skeleton's construct-smoke asserts the
two-build threshold identity.

---

## 6. The exact command for the full GPU de-risk (run once a lane frees)

```bash
# 6-seed decisive on-bridge SPIKING run (one process per seed; SIM_BACKEND=cupy REQUIRED — numpy is CI-smoke only).
for s in 42 43 44 100 101 102; do
  SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._gap4_onbridge_spiking_selfpredict_derisk \
      --seeds $s --arms reservoir fixed_fa kp micro transport_ceiling \
      --hidden 64 --pool-k 16 --n-hidden-layers 2 --epochs 40 \
      --graded-credit --wpi-plastic --wpi-init noisy --assert-no-transport \
      --out research/findings/raw/gap4/onbridge_spiking_seed$s.json &
done; wait
# then aggregate the 6 per-seed JSONs (best(kp,micro) > fixed_fa on >=5/6 seeds, fixed_fa <= reservoir, guards pass).
```

Op-point note: if fixed_fa does NOT degrade to ≤ reservoir at `--pool-k 16`, sweep `--pool-k {8, 4}` and/or
`--in-current-pA` DOWN (sparser spiking ⇒ noisier credit ⇒ fixed-FA degrades) to find the FA-wall regime BEFORE
concluding; a healthy oracle/ceiling must hold at that operating point (else the regime is confounded, not a wall).

---

## 7. HONEST SCOPE — what is on-substrate vs host (the brain-based boundary)

- **ON the spiking substrate (the actual deep credit assignment):** the three feedforward weight matrices
  (`cp_connections` edges of ff_0/ff_1/ff_2) are moved by the committed `fused_bdsp_update` kernel, driven by each
  postsynaptic layer's apical-modulated graded/burst deviation `E·(P − Pbar)`. The forward pass is spiking
  (features → graded current → spikes → event rate). The interneuron cancellation is delivered as a physical bridge
  current `cp_bdsp_int_drive` subtracted in the committed block. **The credit that changes the weights is carried by
  neurons/synapses.**
- **Host-side (the documented shortcut, identical to every D1/EMERGE reference and `stage_a_bridge_learns`):** the
  credit PROJECTION — the fixed-random Y feedback matrix-multiply, the sequential descent `e_upper → next layer`, the
  W^PI cancellation VALUE `softmax @ W^PI`, and the KP/W^PI weight updates — is computed host-side over the bridge's
  read-back rates. There is NO host BACKPROP (no `Wᵀ` chain; the descent uses fixed-random / learned-but-transport-free
  feedback, and each update is a LOCAL outer product of neural activities). The transport-freeness is structural
  (AST-asserted) and the projection is exactly the numpy reference's `v_api = e_upper @ Y`.
- **The deepest follow-on (flagged, NOT in this de-risk):** a fully-on-substrate credit projection — a real spiking
  interneuron population computing `W^PI @ u^I` in spikes through real apical-feedback synapses, and the descent
  realized as top-down synaptic pathways — is the genuine "credit carried entirely by neurons/synapses" endpoint. The
  gap#4 DECISIVE claim (learned transport-free feedback beats fixed feedback on spikes) does **not** require that
  endpoint: the feedforward credit IS on-substrate, and the learned-feedback learning IS local + transport-free. This
  de-risk answers whether the learning-rule hypothesis holds on spikes; the fully-neural projection is the next arc.

**An honest negative (learned == fixed on spikes at every reachable op point) IS the scientific deliverable** — it maps
what the point-neuron spiking substrate can/can't do with this credit family, and points at the fully-neural projection
or the dendritic substrate as the next mechanism. A wall is a verdict on a METHOD, never a license to abandon the
CAPABILITY.

---

## 8. Why NO `sim/` edit is needed (and the bar if one ever were)

Every mechanism this port needs is a committed, additive, default-off `sim/` flag (§2). The reused base injects wiring
via `inject_explicit_wiring`, sets `cp_bdsp_apical_drive` / `cp_bdsp_int_drive` directly, and reads
`cp_bdsp_E`/`cp_bdsp_B`/`cp_bdsp_P` — all existing, public bridge state. The gap#4-specific logic (the KP update, the
plastic-Eq.9 W^PI, the arm dispatch, the apical-silent read, the GO-gate) is **runner-side only**. If a future refinement
DID want a `sim/` change (e.g. an on-substrate interneuron population), the bar is: strictly additive, default-off,
byte-identical when off (assert with a numpy 2000-trial + a cupy 2.048M-element `max|diff| == 0` check, as the existing
BDSP flags do) — and it would fire the research gate first.

---

## 9. Open design questions (for owner review before the full run)

1. **Which spiking degradation actually opens the FA-wall at depth-2?** The plan bets on population-coded sparsity
   (`pool_k` / input drive) degrading fixed-FA's alignment to ≤ reservoir. If the sweep in §6 finds NO such regime with
   a healthy ceiling, the honest verdict is "scale-frontier at this budget" — do we then (a) go deeper (3 hidden layers,
   accepting the confound the rate depth-3 showed), or (b) push straight to the fully-neural projection / dendritic
   substrate? (Recommend: report the boundary + sweep first; decide after.)
2. **DFA vs sequential-FA descent.** The skeleton uses the CPU reference's SEQUENTIAL feedback-alignment descent
   (`e_upper → next layer`, reading each hidden layer's event rate) to match the validated rate reference exactly. A
   direct-feedback-alignment variant (each layer's apical = a direct fixed-random projection of the TOP error, no
   descent, no intermediate read-back) is *more* local / host-lighter and a cleaner "no host backward pass." Worth an
   A/B? (The gap#4 CPU reference is sequential-FA, so sequential is the faithful default; DFA is a follow-on.)
3. **Feature→current encoding.** The skeleton uses the base runner's graded linear encoding (`in_bias + in_current·f`,
   clipped ≥ 0) of the 9 standardized continuous features. A signed ON/OFF two-population encoding (pos/neg rectified)
   may carry the ±1 XOR-pair structure more faithfully. Encoding is a tuning lever for the accuracy run; flagged, not
   yet swept.
4. **W^PI placement.** The CPU reference places the plastic W^PI at the TOP hidden layer only (lower layers plain FA).
   The skeleton mirrors that. Placing a W^PI at every hidden layer (a full interneuron column) is a richer test but
   adds host machinery; deferred unless the top-layer-only micro arm shows the earned-silent property but no accuracy
   separation.
5. **`--wpi-init noisy` vs `fixedpoint`.** noisy (default) is the load-bearing test (silence must be EARNED). fixedpoint
   (W^PI := Y at init) is the positive control (silent from step 0). Both are wired; the full run reports both.

---

## 10. Files

- **This design:** `research/findings/2026-07-24-gap4-onbridge-spiking-port-DESIGN.md`
- **Skeleton runner (construct-smoke-only for now):** `research/runners/_gap4_onbridge_spiking_selfpredict_derisk.py`
- **Reuse-by-import:** `_semantic_inheritance_onbridge_spiking_derisk.OnBridgeBDSPNet` (the on-bridge depth-2 spiking
  base), `_semantic_inheritance_deep_credit_derisk.{make_task_semantic_inheritance, _acc_on}` (task + metric),
  `sim.dendritic_mlp.DendriticMLP` (the fenced backprop oracle ceiling).
- **CPU-rate reference (the GO this ports):** `_gap4_learned_microcircuit_selfpredict_derisk.py` +
  `2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO.md`.
- **NO `sim/` edit.** All BDSP machinery is the committed default-off flags in `sim/config.py` + `sim/bridge.py`.
