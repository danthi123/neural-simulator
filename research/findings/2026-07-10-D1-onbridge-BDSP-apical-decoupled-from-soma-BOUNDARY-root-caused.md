# D1 on-bridge — the committed `enable_bdsp` rule does NOT learn a map to accuracy on a real bridge: the apical is **decoupled from the soma's measured bursts**. Verified boundary, root-caused, surpass named.

**Date:** 2026-07-10
**Runner:** `research/runners/_d1_onbridge_learn_to_accuracy_derisk.py` (built this cycle; numpy CPU smoke; NO `sim/` edit).
**Verdict:** BOUNDARY — verified on D1's own Stage-A smokes, root-caused by reading `sim/bridge.py`, with a
biologically-grounded surpass named.

## The question this closes
D1 deferred its own fully-on-bridge learning-to-accuracy run (`2026-07-07-D1-microcircuit-...clears-bar-on-spikes.md`,
line 37: *"The 0.964 is the NUMPY REFERENCE... the fully-on-bridge spiking multi-seed net remains the controller's GPU
run (not yet demonstrated)."*). The register's transition-learning-on-bridge depends on it, so it is the honest
cheapest-first rung. **The answer, on the documented path, is NO — and the reason is specific and fixable.**

## What the runner establishes (built + wired correctly)
A 3-region spiking feedforward net (input → hidden → output) on one `SimulationBridge`, plastic `input→hidden` +
`hidden→output` learned by the committed `enable_bdsp` kernel, a fixed-random apical feedback delivering the top error.
It is correctly wired: the task is valid (numpy oracle **0.989** ≥ 0.80; single-layer floor 0.510 ≈ chance → the task
genuinely needs the hidden layer), and there is **no weight transport** (asserted). But held-out accuracy stays at chance
and the **moat is inverted, not held**.

## The root cause (verified two ways)
Driving `cp_bdsp_apical_drive` raises the burst-**probability** read `P` (0.30 → 1.00) but **NOT the measured burst rate
`B`** (0.000 → 0.000). The committed feedforward update is `dw ∝ etilde · (B − Pbar·E)` — it uses the **measured** `B`,
which is set by the soma and is apical-independent. So the apical delivers no directed credit, and the moat inverts
(credit-condition weight-drift ≈ lesion-condition drift).

**Verified on D1's OWN validated Stage-A smokes, in current code:**
- `stage_a_bridge_detector(42)`: **B_rises = False** (B_rest 0.198 → B_apical 0.189), while P_rises = True (0.30 → 1.00).
- `stage_a_bridge_learns(42)`: **moat_smaller = False** (dw_credit 22.2 **<** dw_moat 26.8 — inverted).

So this is not an artifact of the new runner. It reproduces on the exact smokes D1 cited as validation.

**Read from `sim/bridge.py` directly (not from a citation):** there are **two writers of `cp_v_apical`**.
- The **coincidence / two-compartment-dAP block** (`~6507`) integrates `v_apical` from the coincidence current AND applies
  the electrotonic soma coupling `total_input_current_pA += apical_g_couple·(v_apical − v_soma)` (`~6510`) — this is the
  term that lets apical depolarize the soma and raise real bursts. It runs only under `enable_two_compartment_dap` /
  `enable_coincidence_detection`.
- The **`enable_bdsp` block** (`~7186`) writes `v_apical` **only** to compute `P = sigmoid(β·scale·(v_apical − E_rest))`
  — the burst-probability *read* — and never applies the soma-coupling term.

So on the pure `enable_bdsp` path the apical raises `P` (the read) but never depolarizes the soma, so measured `B` is flat,
so the FF rule gets no directed credit. **The electrotonic apical→soma coupling exists in the substrate; it is simply not
wired to the BDSP apical drive.**

## A documentation defect this corrects
D1's Stage-A″ was reported as validating the on-bridge moat (*"with the apical silenced the change is ~absent"*). In
current code `stage_a_bridge_learns` returns `moat_smaller = False`. Whether the code drifted since D1 or the claim was
always operating-point-fragile, **the committed on-bridge FF learning does not hold the moat**, and D1's own corrections
#2/#3 already half-conceded this (the on-bridge FF update is not what carries the numpy 0.964 accuracy — that is the
runner-side M2.6 somatic rule the committed kernel does not implement). This finding makes that explicit and measured.

## ⇒ the surpass (research-gated; a faithful `sim/` edit, not a config tweak)
The apical must raise **real** bursts for the committed FF rule to get directed credit. The mechanism already exists — the
two-compartment electrotonic coupling — it just needs the BDSP apical drive routed through it: `apical↑ → v_apical↑ →
gc·(v_apical − v_soma) into the soma → more somatic spikes → more measured bursts B → directed credit → the moat holds`.
This is a genuine `sim/` edit (there are two writers of `cp_v_apical`; the BDSP drive must feed the *coupled* apical, or
the coupling term must be added to the BDSP block), and it is exactly the protected-module edit the project sanctions: a
faithful biological mechanism (apical→soma electrotonic coupling — real dendritic biology, and this project's documented
"top lever for emergent capability"), additive and guarded.

**Cheapest-first de-risk before the edit:** confirm the coupling mechanism actually raises `B` — drive a coincidence
pathway (which charges the coupled `v_apical`) with `enable_two_compartment_dap`, and check `B` rises with drive. If it
does, the surpass is "route `cp_bdsp_apical_drive` into that coupled apical," and the edit is minimal.

## Honest scope
- CPU smoke, small scale — the wiring + boundary are the deliverable, not a tuned number.
- The runner self-reports `apical_couples_to_bursts`, so any config/edit that restores coupling shows immediately.
- `--task parity` and `--task emerge1` both supported; `sim/` byte-clean (`git status --porcelain -- sim/` empty).

## Files
`research/runners/_d1_onbridge_learn_to_accuracy_derisk.py`; the deferred question's source
`2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`; `sim/bridge.py` (~6507/6510 coupling,
~7186 BDSP apical); `sim/kernels.py: fused_bdsp_update`.
