# Spiking-realization research gate (scoping, controller-verified): the R3 "learn the INPUT projection on a FIXED reservoir" long-range mechanism realizes ON SPIKES with the COMMITTED `enable_bdsp` rule — NO new `sim/` mechanism — by pivoting the existing on-bridge BDSP runner from learning W_rec (the wrong bottleneck) to learning W_in

**Date:** 2026-07-12
**Type:** research-gate / build scoping (read-only; a subagent design, controller-verified against `sim/bridge.py` + the reference runners).
**Verdict:** the biology-legal long-range mechanism validated at rate level (R3 reframe: fixed random reservoir + e-prop-learned INPUT projection W_in + learned-local feedback + local read-out reaches ~78% of the frozen-reservoir→full-BPTT gap) is realizable on a real `SimulationBridge` **with the ALREADY-COMMITTED `enable_bdsp` burst-multiplexed rule** — reuse-by-import, no `sim/` edit.

## The load-bearing claim — VERIFIED (`sim/bridge.py:7247-7273`)
The committed BDSP update is `dw = η · Ẽ_pre · (B_post − P̄_post·E_post)`, per plastic synapse over `cp_connections`, gated by the plastic mask + `cp_plasticity_rate_gain` (a frozen pathway is untouched). For a plastic **`input_token_region → reservoir`** pathway:
- `Ẽ_pre = cp_bdsp_E[coo.row]` = the presynaptic (input-token) EVENT rate = the **input eligibility** (a decaying trace of `1[x_t==v]`).
- `B_post − P̄·E`, driven by `cp_bdsp_apical_drive[reservoir_j] = k·(δ@Y)_j` (fixed-random Y, own RNG stream) = the **per-reservoir-neuron broadcast credit** `L_j = (δ@Bfb)_j`.
- ⇒ `dw[input_v → res_j] = η · E[v] · L_j` = **exactly "input-synapse eligibility × broadcast feedback"** = the rate rule's `W_in[j,v] += lr·L_j·e_in[j,v]` (`_emerge_stream_eprop_lm_derisk.py::train_eprop_learn_win`), on spikes, NO weight transport (Y is host-held; the host computing `δ@Y` and setting the apical is the legitimate fixed-random credit-projection wiring, as every D1/EMERGE reference does; the WEIGHT CHANGE is the committed kernel's = the brain's job).

## The pivot (why this is the RIGHT build)
The existing on-bridge BDSP reservoir runner (`_emerge_reservoir_lm_onbridge_bdsp_derisk.py`) learns **W_rec** — the bottleneck the R3 reframe PROVED is wrong (training W_rec is counterproductive; that on-bridge W_rec run boundaried at 2/6 seed-variable). **Pivot to learning W_in on a FROZEN reservoir** (`plastic_internal=False`; only `input→reservoir` plastic). W_in's rate-level margin is LARGE + STABLE (+4 nats deep, monotone) vs W_rec's small + unstable → a robust spiking win is far more likely.

## Reusable pieces (import, don't rebuild)
- **Fixed spiking reservoir:** `_emerge82_onbridge_lsm_derisk.py::OnBridgeLSM` (recurrent Izhikevich `BrainRegion`, `plastic_internal=False`; read-out = pool spike-counts from `cp_firing_states`; EMERGE-61 wash-out between sentences; n_pool~300, ~12 steps/token, numpy-CPU).
- **Plastic input→region BDSP template + apical injection + fixed-random Y (no transport):** `_d1_onbridge_learn_to_accuracy_derisk.py::OnBridgeBDSPNet` (`input→hidden` plastic, `cp_bdsp_apical_drive = k·(δ@Y)`, `Y = RandomState(seed+9973)`, the `no_weight_transport` assert, the `apical_coupling_diag` B_rises check).
- **Committed rule + wall-#1 fix:** `enable_bdsp` + `bdsp_apical_couples_soma=True` + `bdsp_apical_soma_g` (routes apical→soma so RS neurons burst → directed credit; default-off byte-identical) + `bdsp_w_max` above the forward design weight.
- **Long-range task + baselines + read-out:** `_emerge81_spiking_memory_depth_derisk.py` (variable-distance distal-cue generator) + `_emerge78_reservoir_form_to_role_derisk.py` (ridge read-out, gov/symwin baselines).

## The cheapest-first de-risk (build `research/runners/_reslm_onbridge_learn_win_derisk.py`)
Fork `_emerge_reservoir_lm_onbridge_bdsp_derisk.py`; (a) reservoir `plastic_internal=False`; (b) add a spiking **input region** + `plastic=True` **`input→reservoir`** pathway so BDSP learns **W_in**; (c) task = **K-cue distal-cue delayed-decode** (`[CUE_k]·filler×d·[QUERY]`; K large enough that a fixed-random W_in COLLIDES the cues; d within the reservoir's fading-memory depth) — this isolates the R3 claim: the fixed recurrence HOLDS the cue (EMERGE-81), but only a LEARNED input embedding makes it DECODABLE after mixing.
- **ONE variable:** fixed-random W_in vs BDSP-learned W_in (reservoir/read-out/task/seeds identical). Read-out W_out trained cleanly per-arm (isolates W_in).
- **Arms:** `fixed_win` (control) · `learn_win` (the piece) · `apical_lesion` (apical=0 → W_in moves only by undirected drift → must ≈ fixed_win) · `wrong_sign` (apical=−k·(δ@Y) → must anti-learn) · `rate_reference` (numpy `train_eprop_learn_win` / BPTT_frozen_wrec on the same task = ceiling + headroom).
- **Anti-cheats:** no-weight-transport (Y own RNG, asserted) · input-lesion → chance · distal-cue scramble → chance · `B_rises` (apical raises the MEASURED burst rate, else no directed credit).
- **GATE (6-seed):** GO if `learn_win − fixed_win ≥ +0.10` decode acc on ≥5/6 AND apical_lesion≈fixed_win AND wrong_sign anti-learns AND scramble at chance AND within the rate reference AND no transport AND B_rises. BOUNDARY (name it, don't force) if the machinery is clean but the margin is seed-variable (as the W_rec run was) — still a valid deliverable mapping the rate→spike gap on the RIGHT bottleneck.

## Ranked next rungs
1. Learn W_in on spikes (host ridge read-out) — THIS de-risk. 2. Read-out on spikes (BDSP `reservoir→output` pathway, or the A→W read-out). 3. KP feedback on spikes (host Y += lr·(rᵀδ)/batch → W_outᵀ). 4. Less-coarse spiking credit (graded burst-fraction read; more bursts/token) for the FA-variance residual. 5. Scale + co-residence on the shared nav/conv bridge (EMERGE-87) = the one-brain integration replacing the transformer scaffold's long-range capability.

## Files
Design verified against `sim/bridge.py:7247-7273` (BDSP dw), the reference runners above. Realizes `2026-07-11-R3-REFRAME-...md` on spikes. NO `sim/` edit anticipated.
