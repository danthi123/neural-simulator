---
type: plan
status: live
date: 2026-06-06
---

# Graded LGN decorrelation stage — design (the biology-faithful on-substrate whitening) — 2026-06-06

> **For the owner:** this design REQUIRES a protected `sim/` edit (a graded recurrent-inhibition term). It is presented
> for approval BEFORE any `sim/` change. The edit is additive + opt-in (default off) — zero effect on existing runs.

## Goal
Realize the validated whitening — the regularized local rule that composes at 100%, 6/6 at the algorithm level
(`2026-06-06-option1-local-learning-whitening-VALIDATED-6seed.md`) — **on the substrate, in the GRADED/analog domain**,
replacing the numpy-ZCA shortcut. This is the project-goal-meeting path (decorrelation as real neural dynamics on the
brain analogue, not an off-bridge matrix op) that the owner pointed at.

## Why graded, not a spiking lateral (the load-bearing biology — settles where the whitening belongs)
Three independent results converge:
1. **Deep research (2026-06-06):** whitening's variance-equalization is a GRADED, analog, pre-spike operation — the
   retina/LGN efficient-coding stage. Realizing it as graded is biology-FAITHFUL, not a cheat.
2. **The opponency-wall finding (2026-06-05):** common-mode removal MUST happen in the graded/analog stage before
   spikes — rate codes provably cannot subtract a small signed common mode.
3. **The on-bridge SPIKING realization (2026-06-06 subagent, BOUNDARY):** a shared-FS spiking lateral does GLOBAL gain
   control, not pairwise whitening (composition stuck at the raw floor, guards green so it is a genuine representational
   limit). The whitening does NOT belong in the recurrent spiking inhibition.
→ The whitening belongs in a GRADED pre-cortical stage. This design puts it there.

## Substrate findings (from the capability check)
- Neuron models are all SPIKING (Izhikevich/HH/AdEx/Resonate). No graded/rate/linear non-spiking model.
- Graded STATE exists + is read/writable: `cp_membrane_potential_v`, `cp_conductance_g_e/g_i/g_nmda`,
  `cp_total_input_current`, `cp_external_input_current` (the graded drive domain).
- Synapses are SPIKE-DRIVEN (conductance increments on a presynaptic spike) — so a graded *lateral* (analog activity
  drives the inhibition, not spikes) is NOT expressible with existing mechanisms → the one needed `sim/` edit.

## Architecture
A graded LGN decorrelation region sits between the grounded drive (CIFAR V1 codes, Track A) and the spiking
cortex/composer:
1. **Drive:** the grounded code enters as `cp_external_input_current` on the LGN region (graded, as today).
2. **Graded recurrent inhibition (the new piece):** each LGN neuron's membrane receives, PRE-SPIKE, an inhibitory term
   `−Σ_j M_ij · a_j` where `a_j` is the *graded* activity of LGN neuron j (its rectified sub-threshold membrane /
   instantaneous drive), NOT its spikes. This is the analog center-surround that does the precise pairwise subtraction
   the spiking lateral could not.
3. **Plastic M (the validated rule):** `ΔM_ij ∝ ⟨a_i a_j⟩ − δ_ij − λ M_ij` (anti-Hebbian on graded co-activity +
   identity target + weight-decay) — the exact rule proven at the algorithm level. Updated from the graded activities.
4. **Read-out → cortex:** the whitened graded LGN activity drives the spiking cortex/composer (its codes → the agent).

## The required `sim/` edit (minimal, additive, opt-in — the approval ask)
A per-region GRADED LATERAL INHIBITION term in the membrane update:
- Config: `enable_graded_lateral: bool = False` + a per-region opt-in (mirrors the `transmission_gate` /
  `enable_brain_region_framework` opt-in pattern — additive, default off, zero effect on every existing run/test).
- Bridge: in `_run_one_simulation_step`, for the flagged region, before the spike threshold, add
  `−(M @ a)` to that region's `cp_total_input_current` (or membrane), where `a = relu(graded activity)` and `M` is a
  plastic per-region lateral matrix (a new small dense `cp_graded_lateral_M` for the region, K×K, K≈300).
- Learning: the `ΔM ∝ ⟨a a^T⟩ − I − λM` update on `cp_graded_lateral_M`, gated by the existing
  `enable_hebbian_learning` + `hebbian_weight_decay` plumbing where possible (reuse), or a small dedicated update.
- Scope: ONE region, dense K×K (K≈300 → 90k floats, trivial). Guarded so it is a no-op unless the flag + the region
  are set. Izhikevich/HH/AdEx paths byte-unchanged when off.

This is the smallest faithful realization: it adds the analog (graded, pairwise) lateral inhibition the retina/LGN
uses, which the spike-driven synapse cannot express, and nothing else.

## Validation plan (the rigor — same controls/guards that caught 5 false positives this session)
- **GATE ON COMPOSITION** (the agent benchmark), NEVER coherence (it misled 3×).
- **Controls** bracket: RAW grounded (~66.7% floor), CONCEPT-whiten (~100% target). Setup-invalid if these are off.
- **Guards** every run: LGN graded activity (not silent / not blown up), `M` norm (bounded), and a **no-lateral
  baseline** (M disabled) to attribute any lift to the learned graded lateral specifically.
- **Multi-seed 6** before any GO. Heavy GPU runs SEQUENTIAL (the OOM lesson).
- **GO** = graded LGN whitening composes ~100% (matches the rate model) with bounded M + alive LGN. **BOUNDARY** =
  the honest limit + where it breaks (the graded lateral still can't do pairwise? the read-out to cortex re-spikes
  away the gain? — diagnose precisely).

## Honest scope + fallbacks
- This is the biology-FAITHFUL, on-substrate realization. It needs the one `sim/` edit above (owner approval).
- The validated SCIENCE (a local rule composes, 6/6) is unchanged regardless of outcome.
- If the graded lateral also hits a limit, the documented fallbacks remain: the upstream graded stage modeled by
  numpy ZCA (research-confirmed faithful) feeding `grounded_codes`, or a structured multi-interneuron inhibitory layer
  (a larger build). Both are real options; this design is the most faithful + smallest-`sim`-edit first attempt.

## Build sequence (after approval)
1. Add the opt-in config flags + the `cp_graded_lateral_M` array + the guarded graded-inhibition term + the update
   (the `sim/` edit; flagged, reviewed, protected paths byte-unchanged when off; add a focused test that it is a no-op
   when off and that M learns + stays bounded when on).
2. A runner: grounded CIFAR codes → the graded LGN region → cortex/composer; controls + guards + composition gate.
3. 1-seed validate (controls + guards), then 6-seed. GO or BOUNDARY finding, honest.
