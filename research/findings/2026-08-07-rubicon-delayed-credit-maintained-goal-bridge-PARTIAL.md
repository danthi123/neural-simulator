# Rubicon delayed-credit — a NEURAL maintained-goal bridges the delay where a decayed trace fails; the LEARNED half (DA-STDP potentiation) does not — PARTIAL (3-seed smoke)

**Date:** 2026-08-07
**Type:** BUILD + numpy 3-seed smoke (de-risk). NO `sim/` edit. Foreground, one process.
**Runner:** `research/runners/_rubicon_delayed_credit_derisk.py`
**Artifact:** `research/findings/raw/rubicon_delayed_credit/smoke_3seed.json` (carries the `preconditions` block).
**Adoption source:** `research/findings/raw/_landscape_adoption_plan_axon_rubicon.md` #2 (Astera/Axon "Rubicon").

---

## 0. One-paragraph result

The Astera "Rubicon" delayed-credit mechanism decomposes into two halves: (HALF 1) the CS->US delay is bridged by
a MAINTAINED GOAL (PT/PFC recurrent-NMDA sustained activity), so credit attaches to the HELD goal rather than a
decayed stimulus trace; (HALF 2) VSPatch learns the reward TIMING so the RPE is correctly placed. Wired onto the
numpy spiking limbic core (`cue -> striosome_value -> snc <- reward_us`, delta=r-V neural via GABA_B) extended with
a recurrent slow-NMDA PFC pool (Wang 2002 persistent activity, the `_d3_persistent_slot` mechanism) as the delay
bridge, and run HEAD-TO-HEAD against a decayed-trace control (the SAME network with the PFC recurrence lesioned,
`recur=0`): **HALF 1 is GO-looking on spikes, 3/3 seeds** — across a 200-step (~200 ms) CS-free gap the held goal
sustains at 340 Hz and drives the critic to express value (452 Hz no-learning), while the decayed-trace control's
PFC collapses to 8 Hz and expresses EXACTLY 0 Hz of value (100% of the effect is attributable to the maintained
goal; the same dissociation holds at the short 20-step gap). All anti-cheats pass: the bridge is neural (external
drive to cue+PFC is identically 0 during the gap — no host-held variable), recurrence-dependent (lesion it ->
collapse), and the RPE is neural (`current_reward_signal=0`; r is synaptic). **HALF 2 is NO on this substrate** —
plain DA-gated STDP does NOT potentiate the held-goal->value synapse across the delay; it DEPRESSES it (trained
value 168 Hz < the no-learning structural floor 452 Hz), so the delayed value expression is STRUCTURAL, not
learned. **Verdict: PARTIAL / OVERALL NO-GO.** The maintained-goal bridge — the load-bearing Rubicon prerequisite,
and the exact thing the DECAYED-trace R4 task (`2026-06-27-navcloseout-R4-...-NEGATIVE`) lacked — is validated on
spikes and worth banking. The genuine un-solved residual is localized to the reward-TIMED potentiation rule
(Rubicon's VSPatch): a saturated-rate held-goal synapse under plain three-factor DA-STDP nets to LTD, so a
reward-window-gated potentiation is genuinely needed, not optional.

---

## 1. RE-ANCHOR (what was already built vs the genuine un-built step)

Per the owner-flagged drift-#12 discipline, verified against our record before building:

| Piece | Status in our record | Role here |
|---|---|---|
| N9 TD cue-shift (`snc_stageb_critic_probe --td-csc`) | **GO** (`2026-06-10-N9-TD-cue-shift-A-CSC-GO`), spiking SNc, r=-0.80/-0.77/-0.89 | reward-timing on spikes, but its CS->US gap is bridged by a DECAYED eligibility trace + world-clocked CSC taps (short gaps) |
| PFC-WM recurrent region (`g11_bg_runner --pfc`) | **GO** on nav (`2026-04-27-pfc-working-memory`); delayed-response Stage-2 **PARTIAL/confounded** (p=0.51) | the maintained-goal substrate — but never wired AS a delay bridge into credit assignment |
| R4 delayed-reward value task | **BUILT + NEGATIVE (2-seed)** (`2026-06-27-navcloseout-R4-...-NEGATIVE`) | used the DECAYED eligibility-trace bridge (`reward_eligibility_tau_ms=500`), NO maintained goal |
| `_d3_persistent_slot_derisk` | recurrent slow-NMDA HOLD validated; `recur=0` = the no-recurrence-fails control | the reusable maintained-goal mechanism + its exact control |

**The genuine un-built step:** R4 already tested the DECAYED-trace approach and it failed. NO ONE had wired the
PFC-WM maintained goal AS the delay bridge and run it HEAD-TO-HEAD against the decayed trace on the neural
credit-assignment substrate. That is this de-risk. (This is NOT re-deriving R4 or N9 — it is the maintained-goal
arm that neither built.)

---

## 2. The head-to-head (means over 3 seeds; long gap = 200 steps is the informative window)

| metric | maintained-goal (recur=25, nmda_slow) | decayed-trace (recur=0) |
|---|---|---|
| PFC hold across the gap | **340 Hz** (sustains) | **8 Hz** (collapses) |
| value expressed across the delay, NO-learning (structural availability) | **452 Hz** | **0 Hz** |
| value expressed across the delay, TRAINED | 168 Hz | 0 Hz |
| short-gap (20-step) value, trained | 108 Hz | 0 Hz |

- **HALF 1 (maintained-goal BRIDGE) — GO-looking, 3/3 seeds.** The held goal makes value AVAILABLE across the delay
  (452 Hz) where the decayed trace gives EXACTLY 0 Hz. `attributable_to` = 100% (0% is present in the control).
  This is the load-bearing prerequisite for delayed credit and the exact thing R4's decayed bridge lacked.
- **HALF 2 (LEARNED credit) — NO.** Training (DA-gated three-factor STDP) moves the held-goal value DOWN, 452 ->
  168 Hz — it DEPRESSES the synapse rather than potentiating it. So the value expression is structural, not a
  learned reward prediction. `unpaired` = 453 Hz ~= no-learning, confirming the residual value is not
  contingency-driven.

---

## 3. Anti-cheat outcomes

- **(a) the maintained goal is NEURAL — PASS.** During the gap the external input to `cue`+`pfc` is identically 0
  (`gap_ext_drive_max == 0.0`, asserted every trial); the bridging firing is the PFC's own recurrent slow-NMDA
  activity. It is recurrence-dependent: lesioning the recurrence (`recur=0`) collapses the hold 340 -> 8 Hz and
  the value 452 -> 0 Hz. Not a host-held variable.
- **(b) the RPE/timing is NEURAL — PASS.** `current_reward_signal == 0` (no host scalar); r enters synaptically via
  `reward_us -> snc`; the value -V subtracts via the GABA_B/GIRK conductance; DA = the SNc firing delta. No host TD
  chain.
- **(c) the decayed-trace control FAILS where the maintained-goal wins — PASS.** 0 Hz vs 452/168 Hz at BOTH the
  long (200) and short (20) gap, 3/3 seeds. The maintained goal — not the task — does the bridging.
- **(d) 6-seed** — this is a 3-seed smoke; the 6-seed command is in §5 (NOT run here).

---

## 4. Honest scope + the localized residual (what the wall actually is)

This is a PARTIAL, and the residual is specific, not a capability to abandon (THE LAW). HALF 2 fails because the
held-goal->value synapse fires at a saturated rate (PFC hold ~340 Hz, structural strio ~452 Hz), so a plain
three-factor DA-STDP nets to LTD over the trial — there is no silent-then-potentiated learning signal, because the
value is already structurally available before any learning. Rubicon's OWN second half is the fix it names:
**VSPatch learns the reward TIMING and gates a correctly-placed potentiation** (a reward-window-locked plasticity
event), which plain scope=all DA-STDP does not provide. This matches the project's deepest lesson: the real system
runs an INTERACTING process (a reward-timed potentiation gate) alongside the maintained goal, and replacing it with
plain DA-STDP lets the proxy (the saturated structural drive) dominate. The maintained-goal BRIDGE is the banked
positive; the reward-timed VSPatch potentiation is the named next method — the adoption plan itself flags
"study `sims/pvlv`/`bgventral`/`pfcmaint` first". A secondary instrument note: the `predicted`/`unpredicted` SNc
burst readout was noisy/uninterpretable here (predicted > unpredicted) and was NOT used for the verdict; the clean
metric is the anticipatory value expression across the gap.

---

## 5. 6-seed validation command (FOR THE PARENT — not run here)

```bash
SIM_BACKEND=numpy PYTHONPATH=$PWD /home/dant123/Projects/sim/.venv/bin/python \
    -m research.runners._rubicon_delayed_credit_derisk --seeds 42,43,44,100,101,102 \
    --gap-short 20 --gap-long 200 \
    --out research/findings/raw/rubicon_delayed_credit/val_6seed.json
```

The HALF-1 (maintained-goal bridge) GO-looking claim is what a 6-seed run would confirm/deny: bridge_go requires
PFC hold > 3x the lesioned control, structural value > 5x the decayed control, gap_ext==0, host_reward==0, at >=5/6
seeds. HALF 2 (learned credit) is already NO at 3-seed and is a separate research thread (VSPatch reward-timing),
not a seed-count question.

---

## 6. Citations

**Project record:** `2026-06-27-navcloseout-R4-delayed-reward-value-task-NEGATIVE.md` (the decayed-trace bridge
failed) + `...-SCOPED.md` (the task spec); `2026-06-21-shortcut9-B4-delayed-reward-value-task-scoping.md` (the
trace-conditioning 2x2 + the load-bearing gate); `2026-06-10-N9-TD-cue-shift-A-CSC-GO.md` (spiking reward-timing);
`2026-04-27-pfc-working-memory.md` + `2026-04-27-pfc-stage2-delayed-response.md` (the PFC-WM maintained goal).
**Reused code:** `_limbic_core_rpe_battery_derisk.build_limbic_core` (the neural delta=r-V core);
`_d3_persistent_slot_derisk` (the recurrent slow-NMDA HOLD + the recur=0 control).
**Adoption source:** `research/findings/raw/_landscape_adoption_plan_axon_rubicon.md` #2.
**Literature (via the adoption plan / B4 scoping):** Wang 2002 (NMDA-dependent persistent activity);
Amit-Brunel 1997; Schultz-Dayan-Montague 1997 (RPE); the Rubicon/Axon PVLV/VSPatch reward-timing.

_BUILD + numpy 3-seed smoke. NO `sim/` edit. A smoke is not a verdict — the HALF-1 bridge GO-looking claim needs
the §5 6-seed run; HALF 2 (learned credit) is a NO-GO here and a separate VSPatch thread._

## ✅ PARENT-VERIFIED (6-seed, `research/findings/raw/rubicon_delayed_credit/val_6seed.json`)
The HALF-1 maintained-goal bridge holds at 6 seeds: `bridge_go=True`, `pfc_hold_maintained` 340 Hz vs
`pfc_hold_decayed` 8 Hz, decayed-trace value 0.0 at BOTH the 20-step and 200-step gaps,
`attributable_maintained_vs_decayed=1.0`. `learned_credit_go=False` (unchanged — the reward-timed VSPatch
potentiation is the named next component). Net: the neural maintained-goal DELAY BRIDGE is a 6-seed GO-looking
result — the load-bearing prerequisite our earlier R4 decayed-trace attempt lacked — and the first concrete
adoption win from the 2026-08-07 landscape survey (Rubicon). NEXT = a reward-window-gated (VSPatch) potentiation
rule to replace the scope-all DA-STDP that over-depresses the saturated held-goal→value synapse.
