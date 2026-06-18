# Scoping: realizing the learned binder's local read-out rule ON the spiking substrate

**Date:** 2026-06-17
**Type:** read-only deep-research + reference-catalog scope (no code, no experiments). Per the standing
"deep research + catalog review FIRST" directive, before committing build/GPU resources to convert the last
host shortcut in the on-bridge learned binder.
**Subject:** the binder's read-out decoder weights are still learned off-substrate by a host gradient method
(Adam / a host numpy delta rule). This scopes realizing that local delta rule **in spikes**, via real synaptic
plasticity, on the point-neuron `SimulationBridge`.

---

## 0. One-line summary

The read-out decoder's *learning rule* is already proven biologically plausible (a local
`presynaptic-rate × per-output-error` delta rule = the Neural Engineering Framework principle, 6-seed GO,
`2026-06-17-localrule-readout-NEF-GO.md`) but is **computed in numpy**; the crux of moving it on-substrate is the
**third factor** — the delta rule needs a *per-output* error `err_j = target_j − actual_j` (a different scalar per
read-out neuron), whereas the bridge's three-factor pipeline multiplies eligibility by a **single global**
neuromodulator scalar (`current_reward_signal`). The recommended path delivers that per-output teaching signal as
a **cerebellar climbing-fiber-style one-teacher-per-output-neuron** signal (catalog F.04/F.05, Albus's rule
`Δw_i = −η·pf_i·cf_burst` is the delta rule), reusing the bridge's **already-present per-synapse third-factor hook**
`cp_per_synapse_reward_override` (the same array the Cluster-F-v2 CF-gated-LTD path already drives).

---

## 1. Diagnosis — what is host vs substrate today, and why the per-output error is the crux

### 1.1 The binder read-out, precisely

The on-bridge learned spiking binder (`_phaseB_onbridge_learned_composer_derisk.py`, finding
`2026-06-17-onbridge-learned-composer-step2-GO.md`) composes a subject-verb-object fact and answers who/what by
unbind + cleanup. Its decomposition:

| piece | what it is | brain-based today? |
|---|---|---|
| bind / bundle | `role_pm1 ⊗ (filler @ W_F)` summed, driven onto LIF ON/OFF populations; read as spike **rates** | **YES — real LIF spikes** (`_phaseB_onbridge_bind_nonlinearity_derisk.py:69` `lif_onoff`, ON/OFF populations `bind_pos`/`bind_neg`) |
| stored composite | the spiking-read signed rate vector | **YES — spike rates** |
| unbind | `composite ⊗ role_pm1` (the fixed ±1 self-inverse) | YES (elementwise channel-swap; the ±1 role is fixed, not learned) |
| **read-out cleanup `W_O`** | `act @ W_O` → `[D_in]`, then nearest-codebook `argmax` | **NO — numpy; weights `W_O` trained off-substrate** |
| filler projection `W_F` | `filler @ W_F` (encoder) | weights off-substrate, BUT the NEF result proves it can be **fixed-random** (no learning needed) — so it is not the target |

The read-out `W_O` is the residual host shortcut. The NEF de-risk (`_phaseB_localrule_readout_derisk.py:55`
`LocalRuleBinder`) already replaced the *learning algorithm* for `W_O`: fixing `W_F` to a random projection and
learning **only** `W_O` by the local Widrow-Hoff/LMS delta rule reaches **1.000 = 103 % of Adam, 6/6 seeds**. The
rule, verbatim from that runner (`:84`):

```
err = est − filler_target            # [D_in]  per-output error (one entry per read-out output dim)
W_O −= lr * (outer(act, err) + λ·W_O) # act = unbind pre-activation [D_h]; outer(act,err) = pre × post-error
```

So the algorithm question is **closed**. The residual is purely: realize `W_O −= lr·outer(act, err)` **as a
synaptic weight update driven by spikes**, not a numpy `outer`.

### 1.2 Why the per-output error is the crux

Map the delta rule onto a real synapse `i` from presynaptic read-out-input neuron `p(i)` to postsynaptic read-out
**output** neuron `j(i)`:

```
Δw_i  =  −lr · act_{p(i)} · err_{j(i)}
       =  −lr · (presynaptic activity)   · (post-synaptic OUTPUT-SPECIFIC error)
            └── factor 1 (pre) ──────┘     └── factor 3 (the teaching signal) ──┘
```

- **Factor 1 (presynaptic activity `act_{p(i)}`)** is exactly what the bridge's eligibility trace already
  captures: `cp_eligibility_trace[i]` accumulates the STDP/Hebbian weight change at synapse `i`, which is a
  pre×post coincidence signal (`bridge.py:6721-6723`). A pre-rate-driven trace is standard three-factor.
- **Factor 3 (`err_{j(i)}`)** is the problem. It is **per output neuron `j`** — a *different* scalar for every
  read-out output. The bridge's three-factor rule (`bridge.py:6765-6905`, the "C2 Reward-Modulated Plasticity"
  block) multiplies the eligibility trace by ONE global scalar:

  ```
  weight_updates = effective_reward_lr · effective_signal · cp_eligibility_trace[:nnz]   # bridge.py:6880
  #                                       └── effective_signal = current_reward_signal − reward_baseline, ONE number
  ```

  A single global dopamine-like scalar **cannot** carry `target_j − actual_j` for each output neuron
  independently. This is the precise gap. (It is the same gap the existing `SUPERVISED_TARGET` training mode
  hits: `experiment/training.py:201-221` computes a per-group rate error but then **collapses it to
  `mean_error`** and writes that one number to `current_reward_signal:221` — a global scalar again.)

**The crux question:** what is the biologically-faithful mechanism for delivering a *per-output* supervised
teaching/error signal on a **point-neuron** substrate (no dendrites), such that the resulting
`presynaptic-rate × per-output-error` synaptic update learns the read-out decoder to host-numpy parity?

A favorable structural fact: the bridge **already has the per-synapse third-factor channel built** —
`cp_per_synapse_reward_override` (`bridge.py:423, 2596, 6866-6878`). When set, it replaces the global
`effective_signal` with a **per-synapse** value:

```
weight_updates = effective_reward_lr · cp_per_synapse_reward_override[:nnz] · cp_eligibility_trace[:nnz]  # bridge.py:6874-6878
```

It is initialized to `None` and **never written inside `sim/`** — it is a pre-built hook (added for the E.3
batched-replica framework, and used by the Cluster-F-v2 CF-gated-LTD runner path, see §3). To deliver a per-output
error, a teaching mechanism need only populate `cp_per_synapse_reward_override[i] = err_{j(i)}` for the read-out
synapses (broadcast the per-output error onto each synapse landing on that output). **No new `sim/` plasticity
machinery is required for the application of the signal** — only the *generation* of `err_j`, which is the
brain-based question.

---

## 2. Ranked biologically-grounded options for the per-output teaching mechanism

Ranked by feasibility-on-a-point-neuron-substrate × directness-of-mapping × reuse. For each: biological grounding
(catalog/Kandel/literature), the map to the read-out decoder, point-neuron feasibility, and the brain-based-only
classification.

### Option A (RECOMMENDED) — Cerebellar climbing-fiber-style one-teacher-per-output-neuron

**Biology.** The cerebellum's learning rule is the textbook per-output supervised mechanism: **one climbing fiber
(CF) per Purkinje cell** (catalog **F.04**, Kandel 6e Ch 37 pp 920-925; Marr 1969 §1; one CF wraps one PC's
dendrites). The CF carries that PC's **error/teaching signal**, and coincident parallel-fiber (PF) activity + CF
event drives **PF→PC LTD** (catalog **F.05**, Marr-Albus-Ito; Kandel 6e Ch 37 pp 922-925). The catalog states
Albus's explicit weight rule verbatim (F.05, p2265-2269): on each CF burst, every active PF synapse `i` is
decreased by `Δw_i = −η · pf_i · cf_burst`. **This is structurally the delta rule** — `pf_i` is the presynaptic
activity (factor 1), `cf_burst` is the per-PC teaching signal (factor 3), and Albus's η is the learning rate. The
read-out's `−lr · act · err_j` is Albus's rule with a *graded, signed* CF rather than a binary burst.

**Map to the read-out.** Each read-out **output neuron** `j` (the Purkinje analogue) gets a dedicated teaching
input (the CF analogue) carrying `err_j = target_j − actual_j`. The read-out-input neurons (the PF/granule
analogue) supply factor 1 via the eligibility trace. The synapse-level update
`cp_per_synapse_reward_override[i] = err_{j(i)}` is exactly the CF-gated per-synapse plasticity, applied through
the existing per-synapse third-factor channel.

**Point-neuron feasibility — HIGH.** The catalog (F.04 supplemental, Hesslow & Yeo 2002) notes the CF is a
*massive* depolarisation, "not just a single EPSP", and that "a simulator that models the CF as a single
point-synapse must amplify it ~10-fold". A point-neuron model represents the CF teaching event as a strong
external current / a one-per-output teaching afferent — no dendritic Ca²⁺ plateau is *required* for the
plasticity gate; the project already models exactly this as "the CF as the reward-sign gate" (g11 Cluster-F path,
§3). The graded-error variant is the only enrichment (graded/signed instead of CF-binary), which is admissible —
graded climbing-fiber error coding is supported by modern cerebellar literature (e.g. graded CF responses scaling
with error magnitude). **This is the most direct map and the only option with a precedent already wired on the
bridge.**

**Brain-based-only classification.** The teaching SIGNAL `err_j = target_j − actual_j` computed by a host
formula is a **documented teaching SCAFFOLD** (the innate-reflex-teaches-a-learned-circuit pattern the project
already uses for N5/N9). The genuine end-state is `err_j` produced by **neurons** — `actual_j` is the read-out
output neuron's own firing rate (already neural), and `target_j` is supplied by a teaching population; the
subtraction `target − actual` is then realized neurally (Option B's error neurons, or feedforward inhibition of
the teacher by the actual). Flag: the host subtraction must be converted; the *structure* (one teacher per
output) is the genuinely-neural cerebellar mechanism.

### Option B — Predictive-coding error neurons (a paired error population computing target − prediction per output)

**Biology.** Predictive coding (Rao & Ballard 1999; Bastos et al. 2012 canonical microcircuit; Keller & Mrsic-
Flogel 2018) posits, for each represented variable, a dedicated **error neuron** whose firing encodes
`prediction − sensory` (or `target − actual`). The subtraction is realized neurally: the error neuron receives
excitation from the target/teaching input and **subtractive inhibition** from the prediction/actual (or vice
versa). The catalog has no dedicated predictive-coding entry, but the substrate primitives exist (the
inhibitory-subtraction motif, GABA_B graded subtraction the project already uses for the SNc value subtraction,
`bridge.py:6222-6227`).

**Map to the read-out.** Add an **error population** of `D_in` neurons (one per output dim), each computing
`err_j = target_j − actual_j` as a neural rate (excited by `target_j`, subtractively inhibited by the read-out
output `actual_j`). The error neuron's rate becomes the per-output third factor:
`cp_per_synapse_reward_override[i] = (error-neuron j(i) rate, signed)`.

**Point-neuron feasibility — MEDIUM-HIGH.** Subtraction by a rate code requires a **signed** difference, which a
non-negative firing rate cannot represent directly — this is the SAME common-mode/opponency issue the project
documented as a rate-code SNR wall (the FHRR pivot, the SNc value-subtraction circuit). The project's *existing*
neural answer is a **bipolar / two-channel** error (a positive-error neuron `relu(target−actual)` + a
negative-error neuron `relu(actual−target)`), exactly the ON/OFF scheme the binder bind already uses
(`bind_pos`/`bind_neg`). So the error population reuses the ON/OFF pattern. This is more new machinery than Option
A (a whole error population + its subtractive wiring), and is best framed as the **neural realization of Option A's
teaching scaffold** rather than a competing option.

**Brain-based-only classification.** The cleanest — the error itself is neural by construction. It is Option A
with the host subtraction *replaced* by neurons. Recommend as the **Phase-2 conversion target** once Option A's
scaffolded version is GO.

### Option C — A three-factor rule with a per-output (not global) modulatory signal

**Biology.** Compartmentalized / spatially-addressed neuromodulation: rather than one broadcast dopamine scalar,
distinct modulatory channels target distinct output pools (catalog C-cluster, the project's Cluster-C-v2
per-action DA `dopamine_{N,E,S,W}`, `bridge.py:6809-6818`). Generalized to read-out: one modulatory channel per
output neuron carrying `err_j`.

**Map to the read-out.** Identical *delivery* to Option A (per-synapse third factor), but justified by
"per-output neuromodulation" rather than "per-output climbing fiber". The bridge has the precedent: Cluster-C-v2's
`compute_per_synapse_da_signal` (`bridge.py:6815`) already produces a per-synapse DA value from action tags.

**Point-neuron feasibility — HIGH but biologically weaker.** Mechanically trivial (same channel as A). But
neuromodulators are **diffuse/volume-transmitted** — a per-*output-neuron* dopamine channel is biologically
implausible at single-cell granularity (DA does not address individual postsynaptic cells with independent
scalars). This option is mechanically equivalent to A but **less faithful**: the cerebellar CF *is* the biology's
answer to "per-output supervised teaching", whereas per-cell neuromodulation is not. Ranked below A for fidelity;
useful as the *implementation* lens (it confirms the per-synapse channel is the right wiring) but cite F.04/F.05 as
the grounding, not per-cell DA.

### Option D — Supervised current-injection teaching (clamp each output to its target during a teaching phase)

**Biology.** Teacher-forcing / target-clamping: during a teaching window, drive each output neuron toward its
target rate by external current, so plain Hebbian `pre × post` realizes the delta rule (because with the post
clamped to target, `Δw ∝ pre · target`; alternating with a free phase whose `Δw ∝ −pre · actual` yields the
contrastive `pre·(target − actual)`). This is the **contrastive-Hebbian / clamped-vs-free** family (Movellan
1990; the wake-sleep / equilibrium-propagation lineage). The project's `SUPERVISED_TARGET` mode + teacher-current
machinery is the nearest existing tool.

**Map to the read-out.** Phase 1 (clamp): inject `cp_external_input_current` into each output neuron `j`
proportional to `target_j`, run, let Hebbian/STDP eligibility accumulate `pre·target`. Phase 2 (free): no clamp,
eligibility accumulates `pre·actual`; apply with opposite sign. The net weight change approximates
`pre·(target − actual)` = the delta rule. No per-synapse error array needed — the per-output specificity comes
from the **clamp current** (each output clamped to its own target).

**Point-neuron feasibility — HIGH (no signed-rate problem), but indirect.** Avoids the signed-error issue (the
sign comes from the two-phase subtraction, not a single rate). Reuses the most existing machinery
(`cp_external_input_current` injection at `bridge.py:5963`, `stimulate_tag`/`set_token_drive`, the
`SUPERVISED_TARGET` mode). Downsides: two-phase protocols are slower and the contrastive approximation is looser
than the exact delta rule; clamping is a strong intervention (must verify the clamp current is in the linear LIF
band so `post ≈ target`). **Ranked second** — the cheapest to *try* (most reuse, no new array semantics), and a
strong fallback if the per-synapse-error route (A) shows an SNR problem. It is also a clean teaching SCAFFOLD: the
target current is host-supplied (legitimate as a teaching scaffold), and the learning is genuine spiking Hebbian
plasticity.

### Ranking summary

1. **A — climbing-fiber per-output teaching** (most faithful, most direct map, has an on-bridge precedent, reuses
   `cp_per_synapse_reward_override`). Host-computed `err_j` is a scaffold → convert to B.
2. **D — supervised current-injection / contrastive-Hebbian** (most reuse, no signed-rate issue, cheapest to
   try; looser approximation). Strong fallback / cheapest first probe.
3. **B — predictive-coding error neurons** (the *neural realization* of A's scaffold; the Phase-2 conversion that
   makes the error itself neural; more new machinery).
4. **C — per-output neuromodulation** (mechanically = A's delivery, but biologically weak at single-cell
   granularity; use only as the wiring lens, cite F.04/F.05 for grounding).

---

## 3. Reusable existing project machinery (exact anchors)

Everything needed to *apply* a per-output third factor already exists on the bridge. Named, with file:line:

**The per-synapse third-factor channel (the key reuse — this IS the per-output delivery mechanism):**
- `sim/bridge.py:423` and `:2596` — `self.cp_per_synapse_reward_override = None` (declared, init'd to None).
- `sim/bridge.py:6866-6878` — when set, the three-factor update uses it **instead of** the global scalar:
  `weight_updates = effective_reward_lr · cp_per_synapse_reward_override[:actual_nnz] · cp_eligibility_trace[:actual_nnz]`.
  This is the exact insertion point for `err_{j(i)}`. **Never written inside `sim/`** — a pre-built hook.
- **Precedent that already drives it:** `research/runners/g11_bg_runner.py:7079-7095` (Cluster-F-v2 CF-gated LTD)
  builds `override = cp.full(nnz, delivered_reward)` then sets a *different* per-synapse value on a tagged synapse
  subset (`override[cerebellum_pf_pc_mask] = cf_signal`) and assigns `bridge.cp_per_synapse_reward_override =
  override`. **This is a working, shipped example of a per-synapse (effectively per-output-group) teaching signal
  routed through the bridge with NO `sim/` edit.** The read-out rule generalizes this from a binary CF gate to a
  graded per-output `err_j`.

**The three-factor / eligibility pipeline (factor 1 = presynaptic activity):**
- `sim/bridge.py:6765-6905` — the "C2: Reward-Modulated Plasticity (Three-Factor Learning)" block (the whole
  rule). `:6880` is the global-scalar default path; `:6874-6878` the per-synapse override path.
- `sim/bridge.py:6721-6723` — eligibility accumulation: `cp_eligibility_trace[stdp_active_indices] +=
  weight_changes` (SIGNED, preserves LTP/LTD direction). This is factor 1.
- `sim/bridge.py:469, 690, 6769` — `cp_eligibility_trace` allocation + `fused_eligibility_trace_decay`.
- `sim/kernels.py:355` — `fused_eligibility_trace_decay(trace, decay_factor)`.
- `sim/kernels.py:313` — `fused_stdp_weight_update(...)` (the pre-post coincidence that feeds eligibility).
- `sim/config.py` — `current_reward_signal`, `reward_baseline`, `reward_learning_rate`, `reward_eligibility_tau_ms`,
  `enable_reward_modulation` (the knobs; the global-scalar path).

**Per-pathway gating (to confine learning to the read-out, and protect the no-confab moat):**
- `sim/bridge.py:3033` `set_plasticity_gate(name, value)` + `cp_plasticity_rate_gain` array
  (`:6698-6701, 6886-6887, 6915-6918`) — freeze all weights EXCEPT the read-out (gate the read-out pathway open,
  everything else closed). The "5a characterized gap" note (`:6910-6918`) and its clip-gating fix are relevant —
  raise `stdp_w_max`/`hebbian_max_weight` above the frozen conversational weights so the ungated clips don't move
  them (the de-risk-5a mitigation already documented).
- `sim/bridge.py:3059` `set_transmission_gate(name, value)` + `cp_transmission_gain` — route the teaching
  afferent's CURRENT on/off (open only during the teaching phase) without touching weights.
- `sim/regions.py` — `RegionPathway(plasticity_gate=..., transmission_gate=...)`; declare the read-out as its own
  gated pathway and a separate teaching afferent pathway.

**Teacher / external-current injection (Option D, and driving the CF/teaching afferent for A/B):**
- `sim/bridge.py:5963` — `total_input_current_pA = synaptic_current_I_syn_pA + self.cp_external_input_current`
  (external current is added to the membrane drive — the injection path).
- `sim/bridge.py:3268` `stimulate_tag(name, drive_pA, additive=...)` and `:2756` `set_token_drive(...)` — write
  `cp_external_input_current` at chosen indices. `:3304-3310` `clear_tag_drive`. These clamp/drive output (or
  teacher) neurons.
- `research/runners/_phaseB_onbridge_bind_nonlinearity_derisk.py:69-85` `lif_onoff(...)` — the canonical pattern
  for driving a population by external current and reading its per-neuron spike **rate** (the read-out's `actual_j`
  is read exactly this way; `RUN_STEPS`, `cp_firing_states` accumulation).

**The existing supervised mode (shows the gap, and is partially reusable):**
- `experiment/training.py:201-221` `_apply_supervised_error` — computes per-group `target − actual` BUT collapses
  to `mean_error` → `current_reward_signal` (`:221`). **This is the global-scalar limitation made concrete.** The
  fix is to NOT collapse: write the per-output errors into `cp_per_synapse_reward_override` instead. `sim/config.py:852-853`
  `target_rates_per_group`, `supervised_error_gain` are the existing config surface.

**The cerebellar substrate already partially built (Option A grounding in code):**
- `research/runners/g11_bg_runner.py:670-673, 1365, 2342-2410` — the Cluster-F builder declares
  `granule_layer / purkinje_X / inferior_olive`, the `cerebellum_pf_pc` plastic pathway (`:2384`,
  `plasticity_gate="cerebellum_pf_pc"`), and `inferior_olive → purkinje` (the climbing fiber, `:2405`).
- `research/runners/g11_bg_runner.py:6970-6979` (v1: CF teaching = reward<0 sign) and `:7079-7095` (v2: per-synapse
  CF-gated LTD via the override). `:4649-4668` caches the `cerebellum_pf_pc` synapse mask for the override.
- `sim/replicas.py:179` references the `cerebellum_pf_pc` gate. `sim/enums.py:285-302, 531-532` the Purkinje/granule
  HH presets. `sim/profiles.py:55-60` `CEREBELLAR_CORTEX_SIMPLE`.

**The host reference + on-bridge binder (what the probe must match / extend):**
- `research/runners/_phaseB_localrule_readout_derisk.py:55-89` `LocalRuleBinder` — the **host numpy delta-rule
  reference** (the upper-bracket / parity target). The rule at `:84`.
- `research/runners/_phaseB_onbridge_learned_composer_derisk.py:47-110` `OnBridgeLearnedComposer` — the current
  on-bridge binder read-out path (store/query/abstain). The read-out `W_O` to be learned on-substrate is
  `self.binder.W_O` (the numpy projection at `:96`).
- `research/runners/_phaseB_fixed_role_learned_filler_bundling_derisk.py:65-136` `FixedRoleLearnedFillerBinder` —
  defines `W_O` and the numpy delta/Adam updates.

---

## 4. Recommended cheap-first de-risk

**Goal:** falsify-or-support "the bridge's three-factor plasticity (the per-synapse third-factor channel) can
learn the read-out decoder `W_O` from spikes to host-numpy-delta parity."

### Recommended probe (cheapest meaningful, two stages)

**Stage 0 — numpy/CPU equivalence proof (a few minutes, no GPU, no `sim/` edit):** before any bridge run, show
the **per-synapse delivery is algebraically the delta rule**. Take the host `LocalRuleBinder`'s exact update and
re-express it as the bridge would apply it: a per-synapse third factor `o_i = err_{j(i)}` (broadcast the
post-error onto each read-out synapse) times a presynaptic-activity eligibility `e_i = act_{p(i)}`, summed —
i.e. `ΔW_O[p,j] = −lr · Σ (e_i o_i over synapses (p→j)) = −lr · act_p · err_j`. Run the NEF arm twice: once with
the native `outer(act, err)` and once assembled from the `e_i·o_i` per-synapse form. They must be **bit-identical**
(it is the same arithmetic). GO/NO-GO: identical → the bridge's `cp_per_synapse_reward_override · cp_eligibility_trace`
product **is** the delta rule, so the only remaining risk is spiking SNR, not the rule. This is a pure-numpy
sanity gate that de-risks the whole concept for ~0 cost.

**Stage 1 — tiny-GPU on-bridge read-out, the actual de-risk:** on a small bridge (D_h=64, F=16, the existing
de-risk scale; ~1-2 GPU min/seed like the bind-nonlinearity probe), wire a minimal read-out:
- a `readout_in` population (the unbind pre-activation `act`, driven by external current exactly as
  `lif_onoff` does) → a plastic `readout` pathway → a `readout_out` population of `D_in` output neurons (`actual_j`
  read as spike rate);
- each output neuron `j` gets a teaching signal `err_j = target_j − actual_j` (Stage-1 scaffold: host-computed
  from the read-out's own spike rate `actual_j` and the known target code, **flagged as a scaffold**);
- per training step: read `actual_j`, compute `err_j`, set `cp_per_synapse_reward_override[i] = err_{j(i)}` for
  the read-out synapses, let eligibility carry `act` (factor 1), let the bridge's three-factor block apply
  `Δw_i = lr · err_{j(i)} · e_i`. Gate every other pathway frozen (`set_plasticity_gate`), raise
  `stdp_w_max`/`hebbian_max_weight` above design weights (the de-risk-5a clip mitigation).
- After training, read held-out who/what recall through the on-bridge-learned `W_O`.

Use Option A's per-synapse delivery (the recommended path). If Stage-1 SNR is poor, the **fallback is Option D**
(contrastive current-clamp), which avoids the signed-rate issue — note it in the probe as the branch.

### GO criterion

- **Primary:** on-substrate-learned read-out held-out recall **≥ 0.85 ×** the host-numpy-delta read-out
  (`LocalRuleBinder`), at **≥ 5/6 seeds** (seeds 42/43/44/100/101/102). (Mirrors the established
  `ob ≥ 0.75·numpy` bind-nonlinearity bar but tightened to the 0.85 the local-rule de-risk used; the binder's
  who/what is a 1/F=0.062-chance task so 0.85× of a ~1.0 host is a strong bar.)
- **Systematicity preserved:** single-binding held-out (generalization to never-seen role-filler combos) holds
  (≥ 0.6 × train, as in the existing systematicity protocol) — the on-substrate read-out must still generalize,
  not memorize.

### Anti-cheat controls (all required)

1. **Lesion the teaching signal → collapse.** Zero `cp_per_synapse_reward_override` (no `err_j` delivered) for
   the read-out synapses; with no third factor the read-out must **not** learn — recall collapses to the
   no-learning floor. (The direct analogue of the de-risk's `lesion` arm and the cerebellum's IO-lesion control.)
2. **Permuted / scrambled teaching → collapse.** Deliver `err_j` to the **wrong** output neuron (a derangement of
   the output index map), so factor 3 is decorrelated from factor 1. Learning must collapse to ~floor (a real
   error-correcting rule cannot learn from a misaddressed teacher; this is the `permuted-role` control's analogue
   and rules out "any third factor drives spurious structure").
3. **Brackets:** the **host-numpy-delta `LocalRuleBinder`** (and/or host-Adam) is the **UPPER** bound; a
   **no-learning FLOOR** (`W_O` frozen at init / random) brackets the result. The on-substrate result must sit
   between, near the upper.
4. **No-confab abstention moat stays intact.** The moat is **structural** in the composer — it is the
   iterate-and-match retrieval returning `None` when no stored composite's unbound cue matches the query
   (`_phaseB_onbridge_learned_composer_derisk.py:100-110`), and the familiarity separation (known vs novel
   confidence). It depends on **which concept the read-out decodes**, not on *how* `W_O` was trained. The probe
   MUST re-run the never-stored-cue abstention (must stay 6/6 clean) and permuted-cue clean check on the
   on-substrate-learned read-out, and the GO is void if either regresses. Per the standing rule the moat is never
   weakened by this change (and per the 2026-06-17 owner note it is a PLUS, not a hard gate — keep it where free,
   which here is free since abstention is not read-out-training-dependent).
5. **Scaffold honesty flag.** The Stage-1 `err_j` is host-computed (a teaching scaffold). The finding must label
   it as such and name the conversion (Option B's neural error population: `actual_j` is already the output
   neuron's rate; add a teaching population for `target_j` and realize `target − actual` by subtractive
   inhibition / an ON/OFF bipolar error pair). The GO is "the *rule* runs in spikes via real synaptic plasticity";
   the residual scaffold is the error *generation*, scoped as the next step.

---

## 5. Catalog cross-reference (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`, accessible)

Directly relevant entries (read in full where cited):

- **F.04 — Climbing fiber: inferior-olive single-cell teaching signal** (catalog L2183-2237; Kandel 6e Ch 37 pp
  920-925; Marr 1969 §1; Albus 1971 §II.E/§IV.C; Hesslow & Yeo 2002). **One CF per Purkinje cell** = the per-output
  teaching mechanism. Supplemental note: a point-neuron CF must be amplified ~10× / modelled as a strong teaching
  current (directly informs Option A's point-neuron feasibility). The Albus CS=MF / US=CF / CR=PC-pause framing
  "maps cleanly onto our existing reward/training engine".
- **F.05 — PF→PC LTD (Marr-Albus-Ito): sign-flipped, CF-gated plasticity** (catalog L2238-2301; Kandel 6e Ch 37
  pp 922-925). States Albus's explicit rule `Δw_i = −η · pf_i · cf_burst` (p2265-2269) = **the delta rule**
  (pre × per-output-teaching). Notes `fused_stdp_weight_update` is timing-Hebbian, not CF-gated — a CF-gated update
  would need either a per-synapse third factor (the `cp_per_synapse_reward_override` route this scope recommends)
  or a new kernel. Sign discrepancy (Marr LTP vs Albus LTD) is about *sign*, not structure — the read-out's signed
  graded error subsumes both.
- **O.03 — DA modulation of corticostriatal plasticity = three-factor rule** (catalog L4772-4783): "eligibility-
  trace × `current_reward_signal` × STDP machinery in `sim/bridge.py`" — the implemented three-factor pipeline the
  read-out reuses (Option A/C delivery). **C.22 / O.02** (Schultz RPE, L907-921, L4754) ground the (global) DA
  scalar that the per-output channel generalizes.
- **C.29 — Eligibility traces and TD(λ)** (catalog L583-589): the credit-assignment / eligibility machinery
  (factor 1) the read-out's presynaptic-activity term reuses. Cluster-C-v2 **per-action DA** (catalog C cluster;
  `compute_per_synapse_da_signal`) is the spatially-addressed-modulation precedent for Option C.
- **Cluster F status** (catalog L30, L3127-3136, L3870): "presets exist; no circuit" — closing F needs a runner
  wiring granule/PF/Purkinje/CF/DCN + a CF-gated PF→PC LTD. The read-out de-risk is a **minimal slice** of that
  (one teacher per output + the per-synapse-error update) and would be the first on-bridge realization of the
  F.04/F.05 rule.
- **Dendrites (B.x, catalog L2646-2652)** — apical/basal coincidence, Larkum two-layer; NMDA plateaus. Relevant
  only to argue the **point-neuron constraint holds**: the cerebellar per-output rule does **not** require
  dendritic compartments (the CF gate is a per-cell teaching event, representable on a point neuron), so the
  default-off two-compartment "D2" neuron is **not** the unlocker here. (Were one to want the genuinely-dendritic
  CF Ca²⁺-plateau coincidence, D2 would be the substrate — but it is unnecessary for parity.)

**NEF / predictive-coding:** no dedicated catalog entry. NEF grounding is in-project (the NEF-cleanup finding +
`2026-06-17-localrule-readout-NEF-GO.md`; Eliasmith-Anderson). Predictive-coding grounding is literature
(Rao & Ballard 1999; Bastos et al. 2012; Keller & Mrsic-Flogel 2018) — flagged for Option B.

**Kandel 6e** is present as a PDF (`references/textbooks/kandel-pns-6e/full-book.pdf` in the catalog worktree; the
in-repo `E:\Documents\Projects\sim\references\textbooks\` path was NOT found — only the catalog-worktree copy
exists). The catalog citations above (Ch 37 pp 920-925) are the load-bearing Kandel references for the cerebellar
mechanism; I did not page the PDF (the catalog entries quote the relevant passages directly).

---

## 6. Standing-constraint compliance summary

- **Brain-based-only:** the *delivery* (per-synapse third factor) and *application* (eligibility × per-output
  error) are synaptic. The per-output error `err_j` is, in Stage 1, a host-computed teaching **scaffold**
  (explicitly flagged); its neural conversion (Option B error neurons; `actual_j` already neural) is the named
  next step. Host code stays within "environment/body + teaching scaffold".
- **No-confab moat:** structural (retrieval-side), independent of read-out training; the probe re-asserts it and
  voids GO on any regression. Never weakened.
- **Point-neuron substrate:** the recommended path needs no dendrites; the constraint holds (argued via F.04/F.05
  + the dendrite entries). D2 two-compartment neuron is **not** invoked.
- **6-seed validation:** GO requires ≥5/6 of seeds 42/43/44/100/101/102.
- **No `sim/` edit needed for the application** (the `cp_per_synapse_reward_override` hook already exists and is
  already driven by the shipped Cluster-F-v2 path); the read-out de-risk is reuse-by-import like every prior
  on-bridge step.

---

## 7. Bottom line

The learning-rule question is closed (local delta rule = NEF, 6-seed GO). The on-substrate question reduces to
**delivering a per-output teaching signal**, and the bridge **already has the channel** (`cp_per_synapse_reward_override`,
shipped + exercised by the Cluster-F-v2 CF path). The biologically-faithful per-output teacher is the **cerebellar
climbing fiber** (F.04/F.05; Albus's `Δw_i = −η·pf_i·cf_burst` *is* the delta rule). Recommended de-risk: a numpy
bit-equivalence gate (Stage 0) then a tiny-GPU on-bridge read-out (Stage 1) learned via the per-synapse third
factor, GO at ≥0.85× the host-numpy-delta read-out, 5/6 seeds, with lesion + permuted-teaching + bracket controls
and the abstention moat re-asserted. The host-computed error is a flagged teaching scaffold; its neural conversion
(predictive-coding ON/OFF error neurons, Option B) is the follow-on that makes the teaching signal itself neural.
