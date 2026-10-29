# cs-race f1h — INPUTS: intrinsic-persistence mechanisms (READ-ONLY round)

**2026-10-22.** Executed per RULING `cs-race-f1g-0c-1022` part 1. No new runs, no new
bars; the frozen packet is untouched. This document gathers the code + record inputs for
the intrinsic arm. Part 4 of the ruling: it contains NO recommendation — the design is
Claude's call.

## 1. Which intrinsic-persistence mechanisms the engine has

### 1.1 Transient + self-regenerating (held) coincidence plateau
- Flags/defaults: `enable_coincidence_detection=False` (`sim/config.py:437`);
  `coincidence_gain=2.0` (:530); `coincidence_plateau_strength=80.0` (:531);
  `coincidence_plateau_self_regen=0.0` (**OFF**, :537); `coincidence_plateau_v_hold=-35.0`
  (:538); `coincidence_plateau_v_hold_k=0.2` (:539); `coincidence_tau_decay_ms=80.0`
  (:540); `coincidence_tau_rise_ms=2.0` (:541).
- Trigger: synaptic coincidence on the coincidence-routed distal synapses (all-or-none
  switch, gain 2; conductance increment strength 80/step). The sustain term
  `g_regen · mg_block(v) · sigmoid(k_v·(v − v_hold))` replenishes the slow reservoir
  while the compartment sits above `v_hold`, so the plateau HOLDS after the volley ends.
- Kernel: `fused_coincidence_plateau`, `sim/kernels.py:1368-1389` (sustain :1375-1380;
  v-gate sigmoid :1388-1389 — below `v_hold` the sustain switches off → passive decay).
  Bridge: `sim/bridge.py:9620-9622`; the per-step/numpy-path twin :10667-10673.
- Validating finding: `research/findings/2026-07-18-gap5-dendritic-bistability-offline-IV-validated-KIR-needed.md`
  — offline I-V validated, KIR-needed (verdict in the filename); CI
  `tests/test_dendritic_bistability.py` (triad + bifurcation + default byte-identity).
- On cs_race's Izhikevich numpy path: the SAME bridge step runs there
  (`b._run_one_simulation_step` at `cs_race_f1.py:799`; backend from
  `sim.backend.get_backend()` at `sim/bridge.py:43-45` — numpy under `SIM_BACKEND=numpy`),
  so the kernel code is on that path when flags are on. **Inert in the probe today**:
  `enable_coincidence_detection` is False and `cs_race_f1.py` wires no coincidence /
  apical / bdsp / ssm pathway (grep of the runner: none) → there are no distal synapses
  to trigger it, and with the two-compartment flag off the plateau would inject somatically.

### 1.2 KIR down-state stabilizer
- Flags/defaults: `apical_kir_g=0.0` (**OFF**), `apical_kir_E_K=-90.0`,
  `apical_kir_vhalf=-50.0`, `apical_kir_k=8.0` (`sim/config.py:565-568`).
- Trigger: voltage, not input — an inward-rectifier K conductance
  `g = apical_kir_g / (1+exp((v−vhalf)/k))`, HIGH when hyperpolarized and LOW when
  depolarized (Sanders-2013 "perfect couple"), so it anchors a silent down state without
  fighting the up state. Requires the apical ODE (§1.3).
- Validating finding: the same gap5 finding — a linear leak leaves a knife-edge bistable
  band, KIR opens a wide one (g_res 2–14 all bistable), CI `test_dendritic_bistability`.
- cs_race path: inert — the apical ODE is not allocated while
  `enable_two_compartment_dap=False` (§1.3).

### 1.3 Two-compartment dAP (real apical voltage)
- Flags/defaults: `enable_two_compartment_dap=False` (`sim/config.py:549`);
  `apical_tau_ms=15.0` (:550); `apical_R=0.15` pA→mV (:551); `apical_g_couple=1.0`
  soma→apical back-coupling (:552); `apical_g_couple_to_soma=-1.0` sentinel → symmetric;
  ≥0 overrides the forward read (:559); `apical_E_rest=-65.0` (:560).
- Trigger: plateau current charges `cp_v_apical`; only
  `apical_g_couple_to_soma · (v_apical − v_soma)` reaches the soma, so a full apical
  plateau stays sub-threshold there — no somatic runaway (config comment :542-548).
- Validating findings: `research/findings/2026-07-02-emerge10-stageAprime-two-compartment-dap-GO.md`
  (**GO**) + `2026-07-02-emerge10-rung4-stageA-risk1-confirmed-need-two-compartment.md`
  (risk: single-compartment somatic injection confirmed as a shortcut) +
  `2026-07-02-emerge10-stageAprime-two-compartment-must-sense-apical-voltage.md`.
- cs_race path: flag False → the byte-identical somatic default — inert.

### 1.4 BDSP apical soma-coupling (+ bistable apical)
- Flags/defaults: `bdsp_apical_couples_soma=False` (`sim/config.py:782`; sibling
  params `bdsp_apical_soma_g=0.004` etc. :784-788); `bdsp_apical_bistable=False` (:791);
  both need `enable_bdsp` and BDSP wiring.
- Trigger: with the coupling flag ON, the BDSP two-compartment pyramidal's apical state
  feeds the soma (Na plateau-burst coupling; config comment :777-781); `bistable` applies
  the self-regen sustain + KIR to the BDSP apical (the configuration the gap4 demo used).
- Validating findings: `research/findings/2026-07-19-gap5-bdsp-apical-soma-coupling-PARTIAL-surpass-the-named-fix-was-a-missing-flag.md`
  (**PARTIAL**, and the named fix was a missing flag) +
  `2026-07-10-D1-apical-soma-coupling-adaptive-rate-bound-with-a-PARTIAL-surpass.md`
  (**PARTIAL** — rate bound).
- cs_race path: flags off and no BDSP wiring in the runner → inert.

### 1.5 The graded slow "dendritic-plateau/calcium" state at `sim/config.py:586`
- Flags/defaults: `enable_selective_ssm_state=False` (`sim/config.py:594`);
  `ssm_k_leak=0.06` (:595). The runner injects `cp_ssm_inject` / `cp_ssm_shunt` per step;
  update `s = lam_eff·s + (1−lam_eff)·inject`, `lam_eff = clip(1 − ssm_k_leak·(1+shunt), 0, 1)`
  — a per-neuron slow graded leaky integrator whose leak is set by an INPUT-DRIVEN SHUNT
  (the comment at :583-592 calls it the dendritic-plateau/calcium graded state).
- Trigger: graded, input-driven. Hold: at zero shunt `lam_eff ≈ 0.94` → HOLDS with no
  further input; a shunt conductance steers/releases it.
- Validating findings: `research/findings/2026-07-13-PAST-RESERVOIR-RUNG4b-i-onbridge-shunt-primitive-membrane-too-fast-slow-state-needed.md`
  (**NEEDED**) → `2026-07-13-PAST-RESERVOIR-RUNG4b-ii-onbridge-slow-SSM-state-GO-additive-sim-edit.md`
  (**GO**, additive sim edit); companion `2026-07-13-PAST-RESERVOIR-RUNG4a-conductance-shunt-input-modulated-leak-survives.md`.
- cs_race path: flag False → state never allocated (config comment :590) → inert; when ON
  it is a per-neuron state in the shared bridge step (backend-agnostic, so it would run on
  the numpy path).

### 1.6 ADP / slow-calcium current — NOT PRESENT
- `grep 'ADP|afterdepolar|slow_cal|ical_slow' sim/config.py` → no hits: the engine has no
  slow depolarizing ADP-like current for the engram cells. Nearest existing slow currents:
  (a) `snr_calcium_*` — an SNr-region calcium state (influx 0.01, decay tau 80 ms,
  baseline 0, SK half 0.5; `sim/config.py:155-158`) — real but SNr-specific, not a
  pyramidal ADP; (b) the Izhikevich cell's `u` — slow **afterhyperpolarizing** adaptation
  (AHP), default RS params (`sim/config.py:19` area) — slow but hyperpolarizing.

## 2. What the RECORDS say about held depolarization (value / duration / terminator)

- **Transient plateau (sustain off)**: rise 2 ms, decay tau 80 ms
  (`sim/config.py:540-541`; Major-Larkum-Schiller NMDA-spike scale 50–100 ms) — a
  transient ends ≈80 ms after its volley.
- **Held plateau, single-cell latch-and-hold probe** (gap5 finding, real kernel): at cue
  end v = −1.7 mV; **held −6.3 mV at 250 probe steps after input removal**; no-self-regen
  control −80.9 (decays — sustain load-bearing); no-cue −81.6 (silent). Bifurcation at
  `self_regen ≈ 0.8` (below → decays, above → holds).
- **Held plateau on substrate** (gap4 BTSP 6-seed GO,
  `2026-07-18-gap4-BTSP-onbridge-behavioral-timescale-GO-6seed-the-keystone-runs-on-the-brain.md`):
  a BRIEF `cp_bdsp_apical_drive` pulse latches `v_apical` at **−24.2 mV** (v_hold −35),
  **seconds-long** hold (the BTSP pre-eligibility window is seconds; held_dw ≈110 vs
  transient −65 → 8.4×, 6/6 seeds); transient control decayed to −65.
- **What ends a held plateau (code)**: v dips below `v_hold` → the sigmoid v-gate ≈ 0 →
  sustain off (`sim/kernels.py:1388-1389`) → passive apical decay (`apical_tau_ms` 15)
  plus the KIR pull-down; with no perturbation the up state is an attractor and holds
  indefinitely (gap5 triad).
- **Production reads of the same quantity**: `2026-08-14-self-initiated-all-basins-ignite-PARTIAL.md`
  — apical latched far above `plateau_v_hold=-35`; the d5 graded-apical-read findings
  (2026-08-20/21, incl. the flip GO) surface `depth_hold` = mean held `max(v_apical−v_hold, 0)`.
- **SSM graded state**: holds at `lam_eff ≈ 0.94/step` at zero shunt; the input shunt
  steers/releases it (Rung4b-ii GO).

## 3. Why C3 is 0/8 where C1 passes (from the code), and what k0 measures

- **C1** = one 150 ms pulse (`MENTION_MS`, `cs_race_f1.py:90`; schedule :767-769), rate
  window `[pulse_end, pulse_end + 650)` (:781-782); band `c_lo, c_hi, quiet = 2.0, 20.0, 1.0`
  (:1587).
- **C3** = two 150 ms pulses with the 150 ms `GAP_B_MS` between
  (`cs_race_f1.py:752-755`), window `[word2_on + 300, word2_on + 650)` (:783-785) — i.e.
  it starts **150 ms AFTER word2's pulse ends**, unlike C1 which starts at pulse end.
  Pass = for ALL pairs `(("vole","frog"),("frog","vole"),("lynx","grass"))` (:1588):
  `word2 ∈ [2,20] Hz AND word1 < 1.0 Hz` in that window (:1639-1641).
- **Failure modes visible in the committed 150 pA grid**
  (`research/findings/raw/_cs_race/f1g/cal150_wmm20_*.json`):
  (a) *word2 below band* — the post-pulse spike tail is sparse and front-loaded, so the
  150 ms dead strip in front of the window empties it (seed 41: c1 = 3.08 Hz in-band,
  pair2 word2 = 0.36, pair3 = 0.0 → c3 False);
  (b) *word2 above band* on over-driven cells (26.79–28.93 Hz) — the same cells whose c1
  is also out of band (seeds 42: 32.12, 46: 30.0, 47: 27.12 → those fail c1 too);
  (c) *word1 not silent* — at W_MM 20 the first word's trace rings into word2's window
  (seed 46 pair1 word1 = 4.29 Hz; seed 48 pair3 = 27.86 Hz ≥ quiet 1.0).
  Because the check is `all(pairs)`, one bad pair or one ringing word1 fails the whole
  cell — so C3 is 0/8 even on cells where the single-word C1 passes 5/8.
- **k0 (C5b)** = `clause_b_per_k[0]` = the cal entry's `total_exc_current_pa_per_k[0]`:
  the per-slot mean of `g_e · (E_e − V)` with `E_e = 0 mV`, over slot-0 (vole) ENGRAM
  cells in absolute `[100, 200)` ms of the race (compute `cs_race_f1.py:706-711`;
  formula note :639-642). Per the race schedule (encode words 50 ms each from t=0,
  `ENCODE_MS=60`, `GAP_A_MS=800` — :87-88, protocol :380-405) that window is the third
  heard vole word (100–150 ms) plus the first 50 ms of GAP_A: **the excitatory synaptic
  current the heard vole words actually drove into the vole engram during hearing**.
- **Verdict: CIRCUIT, not instrument.** `k0 = 0.0` means no excitatory spikes reached
  slot-0 engram cells in that window — consistent with the `MENT_PA=40` sub-rheobase
  drive (step-0 probe: 0 spikes in a 150 ms pulse). The instrument is demonstrably live:
  the identical field reads `k0b = 57.23 / 37.33 / 49.10` pA in the committed 150 pA grid
  (cal150_wmm20_wim20 seeds 42/45/47) — it reports current when the circuit delivers it.
  Honest oddity, recorded as observation only: clause-a's `clause_a_per_k[0]` reads exactly
  0.0 on all 8 cells at 150 pA even though its encode schedule matches clause-b (they
  differ by rng draw +51 vs +52 and clause-a has no later mention turn); protocol
  difference does not obviously explain it — the raw per-ms rasters for [100,200) in both
  clauses would settle it.

## 4. Recommendation

None (ruling part 4). The intrinsic-persistence arm design is Claude's call; this file is
the input set.
