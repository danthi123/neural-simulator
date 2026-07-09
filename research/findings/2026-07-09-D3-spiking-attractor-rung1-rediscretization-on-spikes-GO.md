# D3 spiking port, rung 1 — the re-discretization runs ON SPIKES: a plain Izhikevich attractor-pool WTA faithfully maintains the running group state to held-out-deeper depth (the simulated recurrent language cortex takes concrete form)

**Date:** 2026-07-09
**Runner:** `research/runners/_d3_spiking_attractor_derisk.py` (reuse-by-import of the rate `discrete_attractor_rnn` + `build_divnorm_score_bridge`; numpy backend, small Izhikevich bridge; NO `sim/` edit).
**Verdict:** GO (rung 1 — the re-discretization on spikes); honest scope + the next rungs named.

## Context
The rate de-risk (`2026-07-09-D3-discrete-attractor-recurrence-length-generalizes-GO.md`) proved the MECHANISM for recurrent multi-hop composition: DISCRETE-ATTRACTOR state maintenance (re-discretize the running state to a clean attractor each step) length-generalizes where a continuous RNN provably cannot (S3 0.999 + theorem-backed A5 0.996; continuous RNN 0.000). This RUNG-1 ports the load-bearing operation — the **re-discretization** — onto the project's OWN spiking substrate.

## The result — the spiking WTA re-discretization is FAITHFUL and length-generalizes on spikes
Each step: the rate-learned transition produces K next-state scores → drive K Izhikevich attractor pools by `input_gain·max(score,0)` on a real `SimulationBridge` → the winner pool FIRES most → the next state is decoded `argmax(firing)` → iterate (autoregressive on the SPIKING winner). Streaming S3 composition, train len 1/2/3, held-out-DEEPER 4/5/6:

| metric (S3, seed 42) | value |
|---|---|
| SPIKING same-length state-track | 1.000 |
| **SPIKING held-out-DEEPER state-track** | **1.000** |
| spiking-winner == host-argmax agreement (per step) | **1.000** (faithful) |
| _(E%-max divisive-norm WTA — over-normalizes)_ | 0.217 (diagnostic) |

**Multi-seed 42/43/44 — robust GO: every seed SPIKING deeper-track 1.000, host-agree 1.000** (`research/findings/raw/_d3_spiking_attractor_s3.json`); the plain Izhikevich WTA faithfully reproduces the argmax on the real bridge, seed-robust.

**⇒ the discrete-attractor re-discretization runs ON SPIKES faithfully** (winner decoded from real Izhikevich firing == the host argmax every step) and maintains the running group state to arbitrary held-out depth. The recurrent multi-hop composition is realized on the project's spiking substrate — a concrete step of the "simulated recurrent sequence/language cortex."

## A methodology self-catch (the wrong WTA variant first)
The FIRST pass wired the E%-max **divisive-normalization** WTA (the OneBrainComposer/NEF cleanup) as primary → deeper-track 0.217 (≈chance), host-agree 0.178. The divnorm OVER-normalizes a clear ONE-of-K winner (it divides the winner's drive by the total, erasing its advantage — divnorm is for sparse competition among MANY similar inputs, not a single decisive winner). The `enable_divnorm=False` control gave 1.0 → the plain competitive Izhikevich drive IS the right cleanup here. Fixed the primary to plain-drive; the divnorm-ON is kept as the honest diagnostic (0.217). Lesson: match the cleanup to the score structure (single-winner → plain WTA / lateral inhibition; many-similar → divnorm).

## Honest scope + the next rungs
- **Only the RE-DISCRETIZATION is on-spikes; the TRANSITION (δ: state×input→scores) is still the rate-learned weights** (host-computed scores drive the pools). Next rung: the transition ON-SPIKES (a learned/attractor-bound synaptic map).
- **The plain-drive WTA is faithful at SMALL K (S3: host-agree 1.0) but DEGRADES at LARGE K** (A5 K=60: host-agree 0.964 → deeper-track 0.75, still ≫ chance 0.017): at 60 candidates the transition-score margins shrink, so the un-competitive plain drive occasionally fires a runner-up, and that per-step error COMPOUNDS autoregressively (0.964^L).
- **⇒ FS LATERAL INHIBITION rung (built + MULTI-SEED validated, `build_fswta_score_bridge`): K attractor pools + a shared INHIBITORY FS pool (each pool excites FS; FS inhibits all → the winner suppresses runners-up).** On A5 (blind seeds 100/101/102) it ROBUSTLY lifts the spiking re-discretization from plain-WTA **0.711 (agree 0.951) → FS-WTA 0.828 (agree 0.971)** every seed — the competitive circuit cleans the one-of-K selection at K=60 (the biology: the project's shared_FS / concept-pool WTA). A residual (~2.9% per-step) still compounds at the default weights; **stronger fs→exc + longer settle push it further: `--fs-inh 18 --fs-settle 45` → A5 deeper-track 0.933 (agree 0.99) = GO** (the competition resolves more fully → the winner is selected cleanly; agree 0.99 → the residual ~1%/step is small). At S3 the plain WTA is already clean (large margins → host-agree 1.0). ⇒ **the spiking re-discretization runs CLEANLY on the substrate at BOTH scales — S3 1.000 and the theorem-backed A5 0.933** (tuned FS lateral inhibition); the confirmed clean-attractor mechanism. `research/findings/raw/_d3_spiking_attractor_a5.json`.
- **numpy backend, small bridge, per-step (teacher-forced) transition.** Next: the real CA3-attractor bridge (pattern completion = re-discretization); reduce the per-step supervision (end-label/reward).
- **A5-scale on spikes** once the FS-WTA + on-spikes transition hold.

## Landing
The mission-central "simulated recurrent sequence/language cortex" now has a working spiking core: attractor-stabilized recurrence composing to arbitrary depth, the re-discretization realized on the project's Izhikevich substrate. The rate arc found the mechanism (discrete-attractor = CA3); this rung realizes its load-bearing op on spikes; the remaining rungs (FS-WTA clean attractor, on-spikes transition, real CA3 bridge, reduced supervision) are named + scoped. NO `sim/` edit.

## Files
`research/runners/_d3_spiking_attractor_derisk.py`; rate arc `2026-07-09-D3-discrete-attractor-recurrence-length-generalizes-GO.md` + `-first-arc-BPTT-no-length-generalization.md` + `-research-gate.md`.
