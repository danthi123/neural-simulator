# gap#5 on-spikes ordered replay — the 3 failed external-inhibition approaches were a HOLD-vs-PUSH CATEGORY ERROR; the correct mechanism is INTRINSIC FATIGUE (spike-frequency adaptation / STD), 5/5-angle unanimous deep-research

**2026-07-22, deep-research workflow (5 parallel angles + synthesis, 1.18M subagent tokens), coexisting with the fluency
training.** After THREE on-spikes attempts to impose replay ORDER by external inhibition all struggled — (1) crude fixed
soma-silence of the just-fired assembly (seed-42-only, over-suppresses), (2) gamma-rhythm FS-basket feedback inhibition
(preserves reactivation but ordering 0-1/3), (3) theta-ramped GLOBAL inhibition (KILLED all firing, active=[0,0,0]) — the
research gate fired (≥2 approaches to one goal failing). The workflow was UNANIMOUS (5/5 angles, zero dissent) and both
firsthand trust-but-verify checks confirmed it.

## The diagnosis — TWO independent conclusions, both against the current design
1. **BISTABLE HOLD is the WRONG representation for the replay READ.** Every canonical/coded CA3 replay model (Ecker 2022
   eLife e71850 — OUR exact substrate class: spiking CA3, learned weights, github KaliLab/ca3net; Haga-Fukai 2018;
   Romani-Tsodyks 2015; Chenkov-Sprekeler-Kempter 2017; Schmutz-Gerstner-Schwalger 2022) represents a replayed item as a
   TRANSIENT / metastable MOVING BUMP, NEVER a self-sustaining attractor. The DURABLE memory lives in the WEIGHTS; the
   activation is a transient READ of them. Our self-regenerating dendritic plateau + KIR latch is precisely the "stationary
   bump" these papers must BREAK to get replay. **Ecker's ablation is the smoking gun:** remove spike-frequency adaptation
   (AdExpIF→ExpIF) → "a STATIONARY rather than a moving bump" and NO replay — **bit-for-bit our co-ignition symptom
   active=[3,3,3], forward_frac~0.33.** Our OWN repo already proved this twice: `_gap5_sequence_replay_derisk.py:18-20`
   says a TRANSIENT plateau (self_regen=0) during the chain sweep is LOAD-BEARING (bistable latch ON → symmetric/reverse
   links form); and EMERGE-85/86 (our only WORKING spiking ordered read) uses a NON-bistable representation.
2. **The A→B transition is INTRINSIC (fatigue), NOT external (inhibition).** In every model the hand-off is driven by slow
   fatigue of the just-active population — cellular spike-frequency ADAPTATION (fast, tau_w~85ms, Ecker) and/or
   Tsodyks-Markram SHORT-TERM DEPRESSION (slower, tau_rec~200-800ms). Feedback inhibition (PV basket) sets ripple/gamma
   SYNCHRONY and rate, NOT direction (Ecker: blocking PVBC→PVBC removes ripples but "only minor effects on sequence
   replay"). This EXACTLY explains our 3 failures: gamma-FS preserved reactivation but never ordered (inhibition
   synchronizes, doesn't sequence — as predicted); fixed soma-silence is a brittle hand-tuned stand-in for what adaptation
   does automatically + self-scaled; the theta-ramp reset every latch (the textbook consequence of periodic strong
   inhibition on a bistable attractor). **The HOLD-vs-PUSH tension is not a tuning problem — it is the KNOWN, expected
   artifact of storing items as bistable latches and then trying to sequence them externally.**

## The corollary that resolves the +1.3-asymmetry red herring
Under intrinsic fatigue the just-fired assembly is the MOST-fatigued → removed from competition (Chenkov "refractory
adaptation solves transition disambiguation"). Reverse is blocked by the departed assembly's OWN fresh fatigue, so
forward-vs-reverse is NO LONGER a raw-weight contest — the tiny +1.3 asymmetry need not beat the ~8 noise; order rides the
ROBUST adjacent-forward(143) >> skip(22) structure. **This is also exactly why the numpy reference works:**
`np.fill_diagonal(Wm,0)` REMOVES the hold and adds explicit self-avoidance — adaptation is the SPIKING realization of both.
So the numpy-vs-spiking gap is NOT a hidden algorithmic cheat; the ONE genuine residual risk is soma-vs-dendrite: somatic
u-adaptation cannot de-latch a self-regenerating DENDRITIC plateau — which is why the build ALSO makes the plateau
transient during the read (the de-latch is load-bearing).

## The ranked build (top of 4; NO `sim/` edit; all knobs already exposed)
1. **[TOP] Transient-plateau + Izhikevich spike-frequency adaptation** (Ecker 2022 on our substrate): de-latch
   `coincidence_plateau_self_regen`→0 (read live at `bridge.py:7399`), crank `cp_izh_d_increment` (per-spike u-kick, Ecker
   AdEx b~207pA analog) + slow `cp_izh_a` (tau_u>gamma) on the CA3-exc slice; the stored BTSP forward chain drives the
   next; keep weak Poisson (finite-size noise triggers each hop at N=3); NO external inhibition.
2. Tsodyks-Markram SHORT-TERM DEPRESSION on the E→E path (`enable_per_type_stp`, stp_tau_d 200-400ms, stp_U 0.4-0.6) —
   co-equal synaptic-fatigue route, releases the assembly where somatic adaptation (soma-vs-dendrite) might not.
3. Decaying-ADP graded hold + E%-max gamma WTA (keeps periodic inhibition but on a graded sub-threshold memory that rides
   through the dip) — only if pure fatigue under-orders.
4. **[LAST, cheat-prone]** the EMERGE-85/86 address-indexed WM buffer — REJECTED as the sequencer: its ordering is a
   Python slot-index on the SEPARATE RF phasor substrate, no transition dynamics, bypasses the learned CA3 weights (the
   numpy-reference cheat re-imported). **My firsthand read of `ordered_position_wm.py` confirmed this independently** — it
   orders ALGEBRAICALLY via `bind(item,position)`/`unbind`, not by spike timing; the synthesis correctly ranks it last.

## Anti-cheats (the GO gate, 6-seed)
GO: forward_frac ≥ 1.5× chance (0.33) AND per_asm_active ~[1,1,1] (NOT [3,3,3]). Load-bearing controls: **ADAPTATION-LESION**
(d_increment→0 MUST collapse to co-ignition = Ecker's ExpIF control, proving adaptation not inhibition is the driver);
**DE-LATCH-load-bearing** (bistable ON MUST degrade); **SHUFFLED-CHAIN** (→chance, order rides weights); **REVERSE-CHAIN**
(replay reverses); **NO-NOISE acid** (→0, the exact test that validated RANK-1); **DWELL-SCALES-WITH-tau_u** (parametric
causal signature); **NUMPY-REFERENCE GUARD** (no host per-step assembly silence / argmax-next in the loop — order emerges
from the substrate's own u-fatigue + stored weights).

## Status
Runner `_gap5_intrinsic_fatigue_replay_derisk.py` built (4 arms: INTRINSIC vs ADAPT-LESION vs LATCH-ON vs NO-NOISE);
seed-42 calibration run in flight (d_abs/a_abs/self_regen sweep to follow). This is the single UNTESTED branch, needs zero
`sim/` machinery, and — unlike external inhibition — never has to reset the latch cycle-by-cycle (the failure mode that
killed the theta-ramp). Workflow script:
`workflows/scripts/gap5-onspikes-replay-mechanism-research-wf_b56d418a-abb.js`; full synthesis in the task output.
