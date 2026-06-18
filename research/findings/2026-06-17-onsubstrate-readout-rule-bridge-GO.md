# The binder read-out is learned by REAL synaptic plasticity ON the bridge — 6-seed unanimous GO

**Date:** 2026-06-17 (the biology-faithfulness frontier: remove the binder's last host-training shortcut, ON the substrate)
**Status:** **GO, 6 seeds unanimous** (42, 43, 44, 100, 101, 102). The learned binder's read-out decoder `W_O` is
learned by the bridge's **own three-factor synaptic plasticity** — `weight_update = learning_rate ·
per_output_error · presynaptic_eligibility` — NOT a host optimizer (no Adam, no host numpy delta update). NO new
protected `sim/` mechanism: it reuses the **already-present, default-None** per-synapse third-factor channel
`cp_per_synapse_reward_override` (`bridge.py:6866-6878`, the same array the shipped Cluster-F-v2 climbing-fiber
path drives). GPU.
**Runner:** `research/runners/_phaseB_onsubstrate_readout_bridge_derisk.py`
**Raw:** `research/findings/raw/_phaseB_onsubstrate_readout_bridge.json`

## Context — the last host-training shortcut

CYCLE 153 (`2026-06-17-localrule-readout-NEF-GO.md`) proved the read-out is learnable by a biologically-plausible
LOCAL delta rule (Widrow-Hoff/LMS, the Neural-Engineering-Framework principle) — but in numpy. CYCLE 157
(`2026-06-17-onsubstrate-localrule-spikerate-derisk` track) proved the delta rule survives a realistic spiking-rate
input code (mechanism-agnostic). This step closes the loop: realize the rule **in real synaptic plasticity on the
spiking `SimulationBridge`**, so the read-out weights are learned in the substrate, not numpy.

The crux (deep-research scope `2026-06-17-onsubstrate-local-readout-rule-scoping.md`, controller-verified): the
delta rule needs a **per-output** error `err_j = target_j − est_j` (a different teaching scalar per read-out
neuron) — the cerebellar climbing-fiber form (Albus `Δw_i = −η·pf_i·cf_burst`) — whereas the bridge's standard
three-factor block multiplies eligibility by a single GLOBAL neuromodulator scalar. The scope found the bridge
**already has the per-output channel**: `cp_per_synapse_reward_override`.

## Result — STAGE 0 (linchpin) + STAGE 1 (learn), 6 seeds, D_h=64

**STAGE 0 — the bridge applies the delta rule exactly.** A single reward-modulation step with eligibility = a
known presynaptic vector and `cp_per_synapse_reward_override` = a known per-output error yields
`ΔW = learning_rate · outer(err, pre)` to **float precision (max abs err ~4e-8, rel ~8e-8) on all 6 seeds**.
⇒ `weight_update = lr · override[synapse] · eligibility[synapse] = lr · err_post · pre` is the delta rule per
synapse, confirmed on the real bridge.

**STAGE 1 — iterating it learns the decoder to host parity.**

| seed | on-bridge held-out | host (numpy delta, same budget) | scrambled-teaching |
|---|---|---|---|
| 42 | 1.000 | 0.867 | 0.000 |
| 43 | 1.000 | 0.822 | 0.167 |
| 44 | 1.000 | 1.000 | 0.000 |
| 100 | 1.000 | 0.897 | 0.143 |
| 101 | 1.000 | 0.944 | 0.000 |
| 102 | 1.000 | 0.859 | 0.000 |
| **mean** | **1.000** | **0.898** | **0.052** |

- **on-bridge held-out 1.000 = 111% of the host numpy delta rule**, 6/6 ≥ 0.85× host, ≫ memorization-floor 0.000.
- **Systematicity preserved:** the held-out test generalizes to never-trained role-filler combinations (it is not
  memorization; mem-floor 0.000).
- **Anti-cheat collapses:** scrambling the per-output teaching error (so `err_j` no longer addresses output `j`)
  drops recall to 0.052 ≈ chance — the per-output teaching signal is **load-bearing**, not noise.

## Brain-based-only classification (honest scope)

- **NOW brain-based:** the read-out weight learning is **real synaptic plasticity** — the bridge's three-factor
  reward-modulated update at each synapse, gated by a per-output teaching signal (the cerebellar climbing-fiber
  form). No host optimizer touches the weights.
- **Still a host teaching SCAFFOLD:** the per-output error `err_j = target_j − est_j` is computed by a host formula
  and written into `cp_per_synapse_reward_override` (the innate-teacher-teaches-a-learned-circuit pattern). The
  named next biologization is to make that error NEURAL — predictive-coding error neurons (a paired population
  computing `target − prediction` per output, scope Option B) or a climbing-fiber teaching population. The
  *structure* (one teacher per output) is biology-faithful; the subtraction is the residual scaffold.
- **Two modelling simplifications** (documented, the further on-substrate escalations): (1) the eligibility is
  injected as the presynaptic rate (the cerebellar PF-activity form, eligibility ∝ pre) rather than emerging from
  STDP pre-post coincidence — valid because with STDP off nothing overwrites it, and the cerebellar rule is exactly
  `pf_activity × cf_error`; the fully-spike-driven eligibility (drive both populations) is a further step. (2) the
  read-out `est` is read linearly from the bridge's own learned weights (`W @ rate`), matching the production
  composer's linear cleanup; a spiking (nonlinear) read-out population is a separate escalation. (3) the signed
  decoder uses signed weights (the Dale's-law-respecting exc/inh split is the follow-on).

## The four bridge-plumbing gotchas fixed (for future on-bridge plasticity work)

1. **CSR convention:** `cp_connections` is `[pre, post]` (row = presynaptic, col = postsynaptic) — the matvec is
   `rate_post = rate_pre @ C`. Setting per-synapse eligibility/override requires `pre_of = coo.row`, `post_of =
   coo.col` (the opposite of the intuitive `[post, pre]`). A swap silently writes zeros (disjoint index sets).
2. **Dale's-law weight clip:** the reward update clips to `[hebbian_min_weight, hebbian_max_weight]` when STDP is
   off (`bridge.py:6908-6920`); the default min is ≥0, which RECTIFIES the signed decoder's negative updates to 0.
   Set `hebbian_min_weight`/`stdp_w_min` negative to allow the signed decoder (the est is read in numpy, so the
   forward-current sign constraint does not apply; the exc/inh-split realization is the Dale's-law follow-on).
3. **Zero-weight pathway:** a `weight_mean=0.0` plastic pathway creates no usable synapses (the framework falls
   back to default connectivity). Use a tiny non-zero init (0.01) and let the rule grow it.
4. **Eligibility is internally managed:** with STDP ON the eligibility is overwritten from spike coincidence each
   step (`bridge.py:6721-6723`); with STDP OFF an injected eligibility persists (decay ~1 at large tau). For the
   injected-pre form, keep STDP off.

## Reproduce
```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onsubstrate_readout_bridge_derisk \
    --dh 64 --seeds 42,43,44,100,101,102
```

⇒ Together with the brain-based forward path (binding/retrieval/abstention on real spikes, Steps 1-2 GO) and the
local-rule result (CYCLE 153) and the spiking-rate robustness (CYCLE 157), the on-bridge learned binder's read-out
is now learned by **real synaptic plasticity** — the last host-TRAINING shortcut for the read-out is removed on the
substrate. The remaining residual is neuralising the per-output teaching error (a host scaffold today).
