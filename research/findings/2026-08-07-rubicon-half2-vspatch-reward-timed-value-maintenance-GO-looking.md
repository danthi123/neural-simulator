# Rubicon HALF-2 — a reward-TIMED (VSPatch) DA-gated rule MAINTAINS the held-goal value across the delay where scope-all DA-STDP collapses it; reward-CONTINGENT — GO-looking (3-seed smoke)

**Date:** 2026-08-07
**Type:** BUILD + numpy 3-seed smoke (de-risk). NO `sim/` edit (only research-runner code; the plasticity_gate + neuromodulator machinery already exist in `sim/`). Foreground, one process.
**Runner:** `research/runners/_rubicon_vspatch_reward_timed_derisk.py` (extends `_rubicon_delayed_credit_derisk.py` by import + additive default-OFF build flags).
**Artifact:** `research/findings/raw/rubicon_delayed_credit/vspatch_smoke_3seed.json` (carries the 8-precondition `preconditions` block + provenance sidecar).
**Builds on:** `2026-08-07-rubicon-delayed-credit-maintained-goal-bridge-PARTIAL.md` (HALF-1 bridge, PARENT-VERIFIED 6-seed).
**Adoption source:** `research/findings/raw/_landscape_adoption_plan_axon_rubicon.md` #2 (Astera/Axon "Rubicon" — PVLV/VSPatch).

---

## 0. One-paragraph result

HALF-1 (the NEURAL maintained-goal delay bridge) is a 6-seed GO. HALF-2 was NO: plain scope-all DA-STDP DEPRESSES the saturated held-goal->value synapse (168 Hz < the 452 Hz structural floor), netting to LTD. This de-risk builds the Rubicon/PVLV VSPatch component — a reward-TIME-gated potentiation — and runs it head-to-head. **HALF-2 is now GO-looking at 3 seeds (8/8 preconditions):** the reward-timed rule MAINTAINS the held-goal value across the 200-step gap at **456 Hz ~ the 472 Hz no-learning floor**, where the scope-all DA-STDP rule collapses it to **168 Hz**; and the maintenance is **reward-CONTINGENT** — omitting the US (goal held, no reward) lets the value decay to **71 Hz**. The reward-window plasticity gate is NEURAL and reward-selective (open at the US = 1.00, shut in the gap = 0.004, driven by the reward population's spiking through the neuromodulator subsystem — no host if-reward flag). HALF-1 stays intact (PFC holds 340 Hz vs the recur=0 lesion 8 Hz; the decayed-trace arm expresses 0 Hz; gap external drive 0; host reward scalar 0). **Two honest scope limits, both load-bearing: (1) this is reward-contingent MAINTENANCE of STRUCTURALLY-available value, NOT building value from zero — the D1-MSN value cell has no learn-from-below window (swept). (2) The load-bearing fix is the reward-TIME DA SIGN, not the gate per se: the gate is validated-neural but nearly redundant with the reward-timed DA here (removing it changes 456 -> 449 Hz).**

---

## 1. RE-ANCHOR (what was already built vs the genuine un-built step)

Verified against our record before building (RAG + grep; drift-#12 discipline). A reward-WINDOW-gated / VSPatch potentiation is UN-built: the only "reward window" hits are incidental phrases; the N9 TD cue-shift is TD *timing* via a decayed eligibility trace; the reward-modulated three-factor STDP that exists IS the scope-all rule that FAILED at HALF-2. So the reward-time gate + the reward-time DA sign are the genuine step. (Reused by import: `build_core` / `run_condition` from `_rubicon_delayed_credit_derisk`, extended with additive default-OFF flags: `vspatch_gate`, `reward_coactivity`, `da_from_reward_us`, `omit_reward`, `yoke_reward`.)

---

## 2. The head-to-head (means over 3 seeds; long gap = 200 steps = the informative window)

| arm | held-goal value across the delay | note |
|---|---|---|
| FLOOR (no-learning) | **471.9 Hz** | structural value (the synapse is already strong) |
| **VSPATCH (reward-timed, gated)** | **456.5 Hz** | **MAINTAINED at ~floor — the LTD is prevented** |
| COACT-nogate (same rule, NO gate) | 448.9 Hz | the gate adds little here (see §4.2) |
| SCOPE-ALL DA-STDP (the failing HALF-2 rule) | **167.8 Hz** | the documented DEPRESSION (nets to LTD) |
| OMIT (goal held, reward ABSENT) | **70.8 Hz** | CONTINGENCY: value LOST without reward |
| YOKED (reward, no held goal) | 243.2 Hz | 2nd contingency (noisy; PFC=0, not the clean test) |
| DECAYED (recur=0 bridge lesion) | 0.0 Hz | HALF-1 intact: decayed trace expresses no value |

Reward-window gate telemetry (VSPATCH arm): open at the US = **1.000**, shut in the gap = **0.004**. PFC hold across the gap: maintained **340 Hz** vs decayed(recur=0) **8 Hz**. `gap_ext_drive_max = 0`, `host_reward_signal = 0`.

---

## 3. Anti-cheat outcomes

- **(a) the reward-window gate is NEURAL/temporal — PASS.** The gate value is a spiking-driven neuromodulator concentration (`from_region_firing` on the reward-US population, feeding a per-pathway `plasticity_gate` through the NM subsystem each step). It OPENS at the US (1.00) and is SHUT in the gap (0.004). No host `if reward: potentiate` flag wraps the update.
- **(b) the potentiation is reward-CONTINGENT — PASS.** OMIT (goal held across the gap, US ABSENT) does NOT maintain the value: it decays to 71 Hz vs the paired 456 Hz (84.5% of the maintained value is attributable to reward being present). So it is credit, not a freeze/clock.
- **(c) HALF-1 bridge intact — PASS.** recur>0 holds the goal (340 Hz) vs the recur=0 lesion (8 Hz); the decayed-trace arm expresses 0 Hz; `gap_ext==0`; `host_reward==0`. The maintained goal, not the task, does the bridging.
- **(d) 6-seed** — this is a 3-seed smoke; the 6-seed command is in §5 (NOT run here).

---

## 4. Honest scope + the localized residuals (what the wall actually was, and what it now is)

### 4.1 The load-bearing fix was the reward-TIME DA SIGN (an operating-point trap), diagnosed

The original HALF-2 depression was a dopamine operating-point trap. The reward-mod update is `dw = lr * da_signal * eligibility`, with `da_signal = DA_conc - DA_baseline`. On this substrate the SNc-signed dopamine sits BELOW its 0.5 baseline at reward — the SNc does not clear its tonic firing threshold — so `da_signal < 0` at reward and EVERY DA-gated update is LTD-signed. That is why scope-all DA-STDP collapses the value. The fix (`da_from_reward_us`): a clean reward-TIME DA burst (baseline 0, `from_region_firing` on the US population) makes `da_signal >= 0` at reward -> LTP-signed. This is dopamine's reward burst (Schultz); the value subtraction -V remains synaptic at the SNc (strio->snc GABA_B), so HALF-1's critic is untouched. This is the project's deepest-lesson pattern exactly: the OPERATING POINT (DA baseline vs the SNc reward-burst magnitude) was implicit, and the proxy dominated the measurement.

### 4.2 The reward-window GATE is validated-neural but nearly REDUNDANT with the reward-timed DA here

Removing the gate (COACT-nogate) changes the maintained value only 456 -> 449 Hz. On this substrate the reward-TIME DA is itself reward-gated (no reward -> no DA burst -> no potentiation), so the extra reward-window plasticity gate is largely redundant for the value outcome. The gate is still built + validated as a neural reward-window signal (§3a) and is a faithful VSPatch element; it is simply not the load-bearing factor for THIS rescue. Reported honestly, not headlined as the cause.

### 4.3 This is reward-contingent MAINTENANCE, NOT build-from-zero (the D1-MSN has no learn-from-below window)

The held-goal->value synapse starts strong, so the value is STRUCTURALLY available (472 Hz). Attempts to make the value LEARN FROM A WEAK synapse failed across a wide sweep (initial weight 0.5-2.4, US-teacher 0-6, both DA signs, coactivity scale 0.05-1.5, lr 0.1-0.5, 45-60 trials): the D1-MSN value cell is a hard-threshold (UP/DOWN) unit — below a weight it is SILENT (no post-activity -> no coactivity eligibility -> no credit: a dead zone), above it the value is saturated (no headroom). So the demonstrated capability is reward-contingent MAINTENANCE of delayed value across the gap (preventing the scope-all LTD), not building value from nothing.

### 4.4 A hard constraint discovered: `enable_stdp` is load-bearing for the maintained-goal bridge

`stdp_off` (enable_stdp=False) collapses the PFC hold 341 -> 5 Hz (verified). Pair-STDP cannot be removed on this substrate, so the reward-timed rule must OFFSET the whole-trial STDP LTD, not replace it. (The residual companion process for 4.3 is a value-cell excitability homeostasis / dynamic range — the named next mechanism, a research frontier, not a tuning knob.)

---

## 5. 6-seed validation command (FOR THE PARENT — not run here)

```bash
SIM_BACKEND=numpy PYTHONPATH=$PWD /home/dant123/Projects/sim/.venv/bin/python \
    -m research.runners._rubicon_vspatch_reward_timed_derisk --seeds 42,43,44,100,101,102 \
    --gap-long 200 \
    --out research/findings/raw/rubicon_delayed_credit/vspatch_val_6seed.json
```

GO requires, at >=5/6 seeds: rescue (vspatch > 1.5x scope-all AND vspatch >= 0.85x floor), reward-contingency (omit < 0.6x paired), neural gating (us gate > 0.3 and > 2x the gap gate), and HALF-1 intact (pfc_hold > 3x decayed, decayed value ~0, gap_ext==0, host_reward==0).

---

## 6. Citations

**Project record:** `2026-08-07-rubicon-delayed-credit-maintained-goal-bridge-PARTIAL.md` (HALF-1, PARENT-VERIFIED 6-seed); `2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (spiking reward-timing, decayed-trace bridge); `2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md` (the decayed-trace HALF failed). **Reused code:** `_rubicon_delayed_credit_derisk.build_core` / `run_condition` (the neural limbic core + the maintained-goal bridge + the head-to-head harness). **Engine mechanism (already in `sim/`):** `RegionPathway.plasticity_gate` + `ModulatorTarget(target_type="plasticity_gate", scope="gate:<name>")` (per-pathway reward-window gate); `reward_eligibility_from_coactivity` (DA-gated coactivity eligibility). **Adoption source:** `research/findings/raw/_landscape_adoption_plan_axon_rubicon.md` #2. **Literature:** Mollick et al. 2020 (PVLV/VSPatch reward-timing); Schultz-Dayan-Montague 1997, Schultz 1998 (DA reward-prediction-error, the burst-and-dip); Wang 2002 (NMDA persistent activity, the bridge).

_BUILD + numpy 3-seed smoke. NO `sim/` edit. A smoke is not a verdict — the GO-looking claim needs the §5 6-seed run. Honest scope: reward-contingent MAINTENANCE of structurally-available delayed value (the scope-all LTD is prevented), NOT build-from-zero; the load-bearing fix is the reward-TIME DA sign, the reward-window gate is validated-neural but redundant with it here._
