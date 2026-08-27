---
type: finding
status: partial
date: 2026-08-27
mechanism: onebrain-integration-r3-spiking-dopamine-credit
lane: one-brain / integration / emergence-bar
artifacts:
  - research/findings/raw/_onebrain_integration_r3_spiking_dopamine_credit_6seed.json
runner: research/runners/_onebrain_integration_r3_spiking_dopamine_credit.py
---

# One-brain INTEGRATION R3 — the cross-edge credit signal becomes a SPIKING dopamine population (R2-a residual closed at the mechanism level), but the clean integration into the full gate is NOT yet achieved — PARTIAL (UNDEFINED on the full gate — precondition failed)

**One-line:** R3 closes R2-a's one declared residual AT THE MECHANISM LEVEL — the third-factor credit's VALUE is
now the firing of a co-located SPIKING dopamine/coincidence population (an SNc-style AND-gate reading the network's
OWN resolved WTA decision `sel_agent`/`sel_patient` in coincidence with a teacher-confirmation drive), fed to the
engine's native `dopamine` NeuromodulatorConfig (`from_region_firing`, an EXISTING primitive — **NO `sim/` edit**),
which `sim/bridge.py`'s C2 reward-modulated-STDP block automatically prefers over the host `current_reward_signal`
(never set away from 0.0 in this runner). The DOPAMINE-LESION control confirms it is load-bearing: zeroing the
coincidence synapses makes the credit vanish and NO learning occurs — **6/6 seeds**. BUT the full functional gate is
NOT passed: **0/6 seeds** clear all of F1-F4 + lesion-recovers-migration + the emergence controls.

## Per-arm, 6-seed (42/43/44/100/101/102), numpy CPU

Artifact: `research/findings/raw/_onebrain_integration_r3_spiking_dopamine_credit_6seed.json`.

| arm | result | reading |
|---|---|---|
| F1 faculty-still-works | **6/6** | comprehension still comprehends/abstains |
| F2 vary-then-lesion | **0/6** | the cross-edge does NOT cleanly shift the target read + collapse on lesion |
| F3 no-runaway | **6/6** | rates in band |
| F4 moat | **6/6** | held |
| lesion-recovers-migration | **0/6** | the intact run does NOT recover the byte-identical base connectivity |
| emergence: R3a three-factor | **5/6** | intact selective / removed inert / shuffled degraded (mostly holds) |
| emergence: R3 DOPAMINE-LESION | **6/6** | **the credit is genuinely carried by the spiking population** |
| emergence: no-corruption-intact | **0/6** | the intact `da_credit` wiring CORRUPTS the base pool |

## What this means (honest)

**Closed (the R2-a residual, at the mechanism level):** the credit signal's value is no longer a host-delivered
scalar — it is a spiking dopamine population's own firing, and the 6/6 dopamine-lesion control proves that firing is
what drives the learning (not host bookkeeping). This is the emergence-bar burn-down R2-a named.

**Open (the integration):** the current realization does NOT pass the full gate. The `no-corruption-intact 0/6` +
`lesion-recovers-migration 0/6` + `F2 0/6` triad points at one root cause — the added `da_credit` organ's fixed
coincidence synapses (`sel/teach -> snc`, `w=2.0`) perturb the SHARED merge pool's connectivity, breaking the
migration byte-identity invariant and swamping the small vary-then-lesion signal. So the spiking-credit MECHANISM is
sound and load-bearing, but wiring it into the merge pool WITHOUT corrupting the migration (e.g. an isolated/masked
`da_credit` sub-pool, or forming its coincidence synapses by learning rather than fixed injection) is the genuine
open residual — the next R3 rung, not a wall.

Banked as a PARTIAL (the mechanism + its load-bearing spiking-credit proof are the deliverable; the clean
integration is the named next step). Functional read-outs only; no phenomenal-experience claim.
