# R-iii research gate — the two-compartment dAP for CA3 pattern completion: can the sim's EXISTING `enable_two_compartment_dap` reach a completing regime by parameter choice alone, or is a `sim/` forward-dAP-current the named change?

**Date:** 2026-07-08
**Type:** READ-ONLY deep-research gate (no `sim/` edit, no sweep, no build). Findings only.
**Boundary being surpassed:** on the POINT-neuron substrate a partial CA3 cue does NOT reactivate held-out ensemble members even with a hand-installed strong attractor (`2026-07-08-riii-onsubstrate-point-neuron-completion-limit-decisive-with-installed-attractor.md`); the minimal numpy surpass (`...-dendritic-completion-surpass-cheap-first-GO.md`, 0.89 vs 0.26) used an ABSTRACTED count-threshold plateau. Named hypothesis: a genuine two-compartment dAP with a thin high-input-resistance dendrite where a few CLUSTERED coincident recurrent inputs fire a local regenerative dAP that reliably drives the soma.

---

## EXECUTIVE SUMMARY (the single most-likely-to-work cheap-first rung)

1. The sim's two-compartment path (`enable_two_compartment_dap`, `bridge.py:6478-6509`) is **NOT purely passive**: the apical compartment `cp_v_apical` carries a genuine **regenerative NMDA element** — the plateau current is `I = g_eff·mg_block(v_apical)·(E_e−v_apical)` (`kernels.py:277`) and `mg_block` is the Jahr-Stevens voltage positive-feedback loop (`kernels.py:275`), i.e. the biological NMDA spike (Kandel 6e Ch 13 p 297).
2. What the apical **lacks vs the model the project cites** (Bouhadjar-Diesmann 2022) is an **active FORWARD dAP current into the soma**. The sim delivers only the *passive, symmetric electrotonic* term `apical_g_couple·(v_apical−v_soma)` (`bridge.py:6506-6507`), which Kandel (p 297) and the detailed CA3 model (Humphries-Mellor) both say is *attenuated to a very small somatic response*. Bouhadjar-Diesmann instead **inject a fixed `I_dAP = 200 pA` plateau current into the soma for `τ_dAP = 60 ms`** once ≥ γ=5 coincident synapses cross `θ_dAP` — the "predictive" near-threshold drive.
3. **Verdict on parameter-only reachability: PLAUSIBLE-BUT-UNTESTED-AT-THE-RIGHT-OPERATING-POINT, honestly on the edge.** The apical *can* ignite an NMDA plateau with a high `apical_R` (thin high-R dendrite → large local ΔV per unit current — Humphries-Mellor R²≈0.8), but the passive delivery ceiling is `≈ apical_g_couple·(v_apical_plateau − v_soma) ≈ gc·50 pA` at the self-limiting plateau voltage (~−15 mV), and raising `gc` DRAINS the apical (the symmetric coupling merges the compartments → dilution). CYCLE-1066/1067 never swept this — it ran the DEFAULT attenuating `apical_R=0.15`, `apical_g_couple=1.0`.
4. **THE cheap-first rung (Rung 1):** on the EXISTING installed-attractor harness `_riii_onsubstrate_readout_test.py` (already exposes `--two-comp --apical-R --apical-gc --k-thresh --plateau-strength --mg`), turn two-comp ON and sweep, ONE variable at a time, **`apical_R` up (10–50×, the thin-dendrite loop gain), then `apical_gc` (1→3, the delivery), high `plateau_strength` (140→300), and calibrate `k_thresh` in WEIGHT units** to the measured held-vs-non gap. Gate: held-out completion > 0.30, non-ensemble < 0.20, FLAT control < 0.20, AND linear (coincidence-OFF) still fails — the harness prints exactly this verdict.
5. **Honest odds + the named `sim/` change if Rung 1's passive ceiling blocks it (Rung 2):** the biology weight (attenuated distal→soma delivery; Bouhadjar injects a *forward* current) says the robust fix is a **small, guarded, default-off `sim/` addition: on apical-plateau/`c_weighted` crossing `θ_apical`, inject a FIXED forward `I_dAP` (~200 pA, ~60 ms) into the soma, decoupled from the symmetric `apical_g_couple` back-coupling** — the exact Bouhadjar-Diesmann dAP the project already cites. It is NOT "an apical Na-spike+reset" (a plateau is not a sodium spike); it is the forward plateau→soma current Kandel p 298 describes ("when the plateau potential arrives at the cell body, it can trigger a brief burst"). Byte-identical when off (mirrors the existing guarded blocks).

**⇒ Run Rung 1 first (cheap, zero `sim/` edit, exactly the untested regime). Expect it to get close and possibly complete on some seeds; if the passive-coupling ceiling holds, Rung 2 (forward `I_dAP` injection) is the precise, biology-grounded `sim/` change — implementing the cited Bouhadjar-Diesmann dAP that the current passive coupling only approximates.**

---

## PART 1 — the sim's existing two-compartment machinery, read in depth

### 1a. The apical ODE and the soma coupling (`bridge.py:6478-6509`)
The guarded two-compartment block, verbatim structure:
```
6478  _two_comp = getattr(cfg, "enable_two_compartment_dap", False)
6480  cp_v_apical  <- lazily allocated to apical_E_rest (-65) on first use
6482  _v_plateau = self.cp_v_apical if _two_comp else self.cp_membrane_potential_v
6483  I_coincidence = fused_coincidence_plateau(... _v_plateau ...)   # plateau reads the APICAL V when 2-comp
6502  _dv = -(v_apical - Er) + R*I_coincidence + gc*(v_soma - v_apical)   # apical leaky-membrane ODE
6504  cp_v_apical += (dt/tau) * _dv
6506  total_input_current_pA += gc*(v_apical - v_soma)   # ONLY the attenuated electrotonic coupling reaches the soma
6508-9 else: total_input_current_pA += I_coincidence     # single-comp: plateau current added to the soma directly
```
Key structural facts:
- **The apical is a leaky membrane with NO spike + reset.** There is no `if v_apical > theta: v_apical = reset` anywhere. The apical cannot emit an all-or-none sodium-like dendritic spike.
- **But the apical is NOT passive** — it carries the NMDA-spike regenerative loop through `mg_block(v_apical)` (see 1b). With the plateau regenerating on the apical (small, high-R compartment when `apical_R` is large), `R·I_coincidence` gives a large *local* ΔV, so the loop can ignite locally even while the soma stays at rest. This is exactly the biophysical advantage the finding hypothesized.
- **The soma delivery is the SYMMETRIC electrotonic term `gc·(v_apical − v_soma)`.** Its maximum is bounded by `gc·(v_apical − v_soma)`, and `v_apical` self-limits near the NMDA plateau voltage (~−15 mV, see 1c), so realistic delivery ≈ `gc·50 pA`. Raising `gc` also raises the *back*-coupling `gc·(v_soma − v_apical)` into the apical (line 6502), which DRAINS the apical toward `v_soma` — the classic two-compartment coupling trade-off (as `gc→∞` the compartments merge = back to a diluted point neuron). This tension is why "just raise gc" does not straightforwardly work.
- **Anti-runaway is genuine and load-bearing:** because the plateau `g_inc` is triggered by the input COUNT/weight switch (1b), a somatic spike backpropagating via the coupling does NOT retrigger the plateau — so there is no somatic→apical→somatic runaway. This is a real strength to preserve in any Rung-2 change.

### 1b. The plateau kernel `fused_coincidence_plateau` (`kernels.py:253-278`)
```
268  g_inc = plateau_strength / (1.0 + exp(-gain*(c_count - k_thresh)))   # ALL-OR-NONE switch on the coincident COUNT/weight
270-273 dual-exponential plateau conductance (g_slow - g_rise), 80ms/2ms kinetics
275  mg_block = 1.0 / (1.0 + (mg_conc/3.57) * exp(-0.062 * v))            # Jahr-Stevens voltage gate (regenerative)
277  I = g_eff * mg_block * (E_e - v)                                     # plateau current toward E_e = 0 mV
```
- The **count/weight threshold `k_thresh`** is the dAP trigger: `g_inc → plateau_strength` when the coincident drive crosses `k_thresh`, `→0` below it. In the R-iii harness `coincidence_weighted_drive=True` (`_riii_ca3_coincidence_completion_derisk.py:58`), so the drive is `c_weighted = Σⱼ w_eff,j·xⱼ` and **`k_thresh` is in WEIGHT units** (config comment `config.py:290-292`). This is the specificity gate: set it between the non-member drive and the held-out drive.
- The **`mg_block` term is the regenerative NMDA element.** At rest it is ~94% blocked and it opens with voltage:

  | v (mV) | mg_block (mg=1.0) | I ∝ mg_block·(0−v) |
  |---|---|---|
  | −65 | 0.060 | 3.9 |
  | −40 | 0.230 | 9.2 |
  | −20 | 0.508 | 10.2 (peak) |
  | 0   | 0.781 | 0 (self-limits at E_e) |

  So the plateau current PEAKS near −20 mV and self-limits toward `E_e=0` — the biological NMDA plateau. **The "regenerates only once depolarized" problem the prior finding named is real: at `v=−65` the plateau is 94% Mg-blocked.** In single-comp mode the plateau reads `v_soma=−65` → throttled to 6% and diluted across the whole soma → the CYCLE-1066 −46.5 mV sub-threshold read. In two-comp mode the plateau reads `v_apical`, and a high `apical_R` lets `R·I` push `v_apical` up into the unblock range → ignition. **This is precisely why two-comp + high `apical_R` is the untested regime that could bootstrap.**

### 1c. The config defaults (quoted, `config.py`)
```
128  syn_reversal_potential_e: float = 0.0        # E_e (plateau reversal), 0 mV
137  nmda_mg_concentration:    float = 1.0        # [Mg2+] mM
173  enable_coincidence_detection: bool = False
174  coincidence_k_threshold:  float = 6.0        # trigger count (WEIGHT units when weighted_drive)
202  coincidence_gain:         float = 2.0        # all-or-none switch slope (~0.88/0.12 at K±1)
203  coincidence_plateau_strength: float = 80.0   # regenerative plateau conductance scale
204  coincidence_tau_decay_ms: float = 80.0       # NMDA-spike 50-100ms tail
213  enable_two_compartment_dap: bool = False
214  apical_tau_ms:   float = 15.0
215  apical_R:        float = 0.15                # plateau-current -> apical-voltage scale (pA -> mV)  << THE thin-dendrite lever, default LOW
216  apical_g_couple: float = 1.0                # electrotonic soma<->apical coupling (attenuation to soma)
217  apical_E_rest:   float = -65.0
292  coincidence_weighted_drive: bool = False
```
**`apical_R=0.15` is a LOW default** — a "fat" low-resistance dendrite giving only 0.15 mV of local depolarization per pA of plateau current. The thin-high-R biology (Humphries-Mellor) wants this an order of magnitude larger. **`apical_g_couple=1.0`** is the symmetric coupling. These two defaults are exactly what CYCLE-1066/1067 ran, and are why the finding said the current coupling "SUPPRESSED completion."

### 1d. The other apical machinery (context, not the completion path)
- `sim/dendritic_neuron.py` (`DendriticLayer`, Larkum BAC) is a **numpy rate-reference**, not on-bridge. Its BAC model is *threshold-modulation* (`effective_threshold = theta_high − apical_gain·|apical_drive|`, line 44-47) — the apical LOWERS the soma threshold rather than driving a current. This is the Bouhadjar "predictive/priming" form, a useful Rung-2/3 alternative but a different module.
- `cp_v_apical` is REUSED by Burstprop/BDSP (`bridge.py:7130-7168`) as the top-down credit compartment (fixed-random apical feedback). Irrelevant to completion but confirms `cp_v_apical` is a real per-neuron leaky membrane voltage the two-comp path already owns.
- The GRADED sibling `fused_graded_dendritic_plateau` (`kernels.py:281-330`) is a value read-out (smooth logistic), not the all-or-none completion switch — not the completion path.

**Honest structural conclusion (Part 1):** the current two-compartment impl has (i) a count/weight-threshold dAP trigger (= Bouhadjar-Diesmann's `θ_dAP`, present), (ii) a regenerative NMDA-Mg plateau on a separate apical compartment (= Kandel's NMDA spike, present, ignitable via `apical_R`), but (iii) delivers to the soma ONLY the small, symmetric, passive electrotonic coupling — it LACKS the active *forward* dAP current that both the biology and the cited Bouhadjar-Diesmann model use. Whether (iii) can be pushed to a completing regime by `apical_R`/`apical_gc`/`plateau_strength` alone is **empirically untested at the correct operating point** and is Rung 1; the biology weight suggests it is on the edge and the forward-current `sim/` change (Rung 2) is the robust fix.

---

## PART 2 — the biology, read in depth

### 2a. Kandel 6e Ch 13 "Synaptic Integration in the CNS", pp 297-298 (read in full-text)
- **The NMDA spike is the Mg-unblock positive-feedback loop (p 297):** *"Moderate synaptic stimuli … lead to expulsion of Mg2+ from a fraction of NMDA receptors. As these receptors begin to conduct … they produce a further depolarization that leads to even greater unblocking of Mg2+, increasing further the size of the NMDA receptor EPSC, resulting in even greater depolarization. In some instances, this leads to a local regenerative depolarization, referred to as an NMDA spike. Such NMDA spikes are purely local events."* ⇒ the sim's `mg_block(v_apical)` loop IS this mechanism; the requirement of an "intermediate level of depolarization" to START it is the bootstrap the high-`apical_R` regime addresses.
- **Distal input is tiny at the soma UNLESS a regenerative dendritic spike carries it (p 297):** *"excitatory synaptic inputs to the distal dendrites usually produce only a very small depolarizing response at the soma, due to electronic decay along the dendritic cable."* ⇒ this VALIDATES the sim's attenuating `gc·(v_apical−v_soma)` as biophysically correct for *passive* distal input — and tells us that a completing signal must be an *active* regenerative event, not a larger passive current.
- **The completing event is a plateau that ARRIVES at the soma and drives a BURST (p 298, Larkum et al 1999 BAC):** *"the backpropagating spike summates with the distal EPSP to trigger a long-lasting type of dendritic spike called a plateau potential, which depends on … voltage-gated Ca2+ channels and NMDA receptors. When the plateau potential arrives at the cell body, it can trigger a brief burst of action potentials"* (3+ spikes at ≤100 Hz). ⇒ the biological delivery is an ACTIVE forward event (plateau → somatic burst), which the sim's passive coupling does not model. In Larkum's canonical experiment the plateau is triggered by distal+proximal PAIRING; pure autoassociative completion (no separate proximal) needs the distal cluster alone to ignite the NMDA spike — the harder regime, hence the thin-high-R lever.

Catalog anchor: **G.02 "Active dendrites — local computation, dendritic spikes"** (Kandel 6e Ch 13 p 293-298): *"NMDA spikes (NMDAR-driven plateau potentials) … nonlinear summation (cluster of inputs on one branch ≫ scattered inputs on many branches) … Larkum's two-layer model."* Sim status listed "missing (single-compartment everywhere)" — this arc is exactly the addition. Pattern-completion anchor: **D.13 / D.05** (Kandel 6e Ch 54 pp 1357, 1360-1361; O'Keefe-Nadel 1978 pp 209-215, 224-227): CA3 recurrent autoassociator; CA3-NMDA-KO mice fail with 2/4 cues removed (partial-cue completion is NMDA-dependent — consistent with the NMDA-spike being the completing nonlinearity).

### 2b. Bouhadjar, Wouters, Diesmann, Tetzlaff 2022 (PLoS Comp Biol 18(6):e1010233) — the exact cited dAP model
Read from PMC9273101. Their dAP is:
- **Trigger = a THRESHOLD ON THE DENDRITIC CURRENT (count/sum of coincident mature synapses), NOT a Mg-voltage gate:** *"if the dendritic current I_ED exceeds a threshold θ_dAP, it is instantly set to the dAP plateau current I_dAP,"* with `θ_dAP` *"chosen such that the co-activation of γ neurons … reliably triggers a dAP,"* **γ = 5**. ⇒ this MATCHES the sim's count/weight switch (`g_inc` at `k_thresh`) and matches CYCLE-1065's abstracted count-threshold. It is the SAME mechanism the sim already has for the *trigger*.
- **Delivery = a FIXED FORWARD PLATEAU CURRENT INTO THE SOMA:** `I_dAP = 200 pA`, `τ_dAP = 60 ms`, `θ_dAP = 59 pA`. *"This plateau current leads to a long lasting depolarization of the soma."* ⇒ this is the ACTIVE forward event the sim's passive `gc` coupling does NOT provide. **This is the crux difference and the target of Rung 2.**
- **The dAP alone PRIMES, does not fire (predictive state):** it *"can speed up somatic … firing, provided the time interval … is in the right range"* — the feedforward input then fires the predictive cell first. ⇒ IMPORTANT nuance for the completion task: pure autoassociative recall of a held-out member (no separate feedforward) asks the dAP to do MORE than Bouhadjar (fire solo), so completion needs either a stronger-than-predictive `I_dAP`, OR (biologically faithful) the dAP + summed recurrent convergence + tonic excitability (Rung 3).

### 2c. Humphries, Mellor et al. — "Acetylcholine boosts dendritic NMDA spikes in a CA3 pyramidal neuron model" (Neuroscience 2022; PMC7614718) — two-compartment/detailed CA3 NMDA-spike + pattern completion
Read from PMC7614718. Directly grounds the `apical_R`/`k_thresh` levers:
- **Synapse count for an NMDA spike depends on branch input resistance:** proximal SR (recurrent associational/commissural) branches need **~10.3 ± 0.9** clustered synapses; distal high-R SLM branches need only **~2.5 ± 0.4** — a ~4× swing, and **local input resistance predicts the threshold (R²≈0.80)**: *"thin, high-resistance dendrites achieve larger voltage responses from the same synaptic input."* ⇒ VALIDATES the sim's `apical_R` as the correct lever: raising it lowers the effective synapse count needed to ignite (the ~5 recurrent inputs a held-out member gets can suffice on a thin-enough branch).
- **But the very distal high-R branches deliver LESS to the soma:** *"the amplitude of the dendritic NMDA response increased with distance … whereas the opposite was true for somatic responses."* ⇒ VALIDATES the delivery tension: thin high-R dendrites ignite easily but their passive somatic footprint is small — precisely why an ACTIVE forward current (Rung 2), not just a bigger passive coupling, is the robust completion fix.
- **ACh lowers the threshold ~20% (blocks K currents) to facilitate pattern completion** — the biological analog of lowering `k_thresh` / adding a tonic excitability boost (Rung 3, an ACh-like disinhibition/depolarization).

**Honest synthesis (Part 2):** the sim already has the two biologically-correct *ingredients of the trigger* (count-threshold dAP = Bouhadjar; NMDA-Mg regeneration = Kandel) and the correct *thinning lever* (`apical_R` = Humphries-Mellor high-R). The missing piece, unanimous across all three sources, is the **ACTIVE FORWARD delivery of the plateau to the soma** — Kandel's "plateau arrives at the soma → burst," Bouhadjar's fixed `I_dAP=200 pA` into the soma. The sim substitutes a passive, symmetric, attenuating electrotonic coupling for it.

---

## PART 3 — external engineering / comp-neuro literature: the dAP→soma coupling regime + specificity

- **Two-compartment attractor completion (general):** CA3 attractor pattern-completion (Rolls-Treves autoassociator) works because a partial cue drives *many* recurrent synapses onto each recalled cell; a recalled cell integrates convergent recurrent input above a threshold. On a POINT soma the convergence must be linearly supra-threshold (fails at sparse cue — the R-iii boundary). The two-compartment/dendritic-nonlinearity resolution (Poirazi-Mel two-layer; Kastellakis clustering; Humphries-Mellor NMDA-spike) is: cluster the co-active recurrent inputs on one thin branch → an NMDA spike there → forward drive to the soma. **The specificity (Marr / Rolls-Treves completeness-vs-specificity) is enforced at the branch threshold**: a held-out MEMBER clusters its same-ensemble inputs (they co-fired at storage) → crosses the branch threshold; a NON-member's inputs scatter → no branch crosses → stays silent (this is the CYCLE-1065 minimal-model result: 0.89 completion, 0.00 non-stored). In the sim this maps to `k_thresh` on the weighted drive: set it above the non-member drive and below the member drive.
- **The dAP→soma coupling regime without runaway (all-fire):** the completion literature's stable regime is (i) a HARD branch/count threshold for the trigger (specificity — only clustered members ignite), (ii) a STRONG but BOUNDED forward current per ignition (completeness — an ignited member reliably fires), (iii) NO voltage-retrigger from the soma side (no runaway). Bouhadjar-Diesmann's fixed `I_dAP` (bounded, forward, count-triggered) satisfies all three; the sim's count-triggered `g_inc` + anti-runaway design (1a) already gives (i) and (iii); what it under-provides is (ii) — the passive `gc·(v_apical−v_soma)` is not a strong bounded forward current. **Feedback inhibition** (an ensemble-level interneuron the R-iii harness can add) is the biological knob that prevents all-fire while allowing completeness — the completeness/specificity balance is set jointly by `k_thresh` (per-cell gate) and inhibition (network gate), not by the coupling gain alone.
- **Convergence matters (Rung 3 grounding):** the installed-attractor harness gives each held-out member ~5 within-ensemble recurrent inputs (ens_size=20, 50% cue, density 0.5) — right at Bouhadjar's γ=5, the *predictive-only* floor. Rolls-Treves completion in vivo rides on larger convergence; raising the cue fraction / `ca3_density` (more coincident recurrent synapses per held-out member) is a cheap, biology-faithful rung that reduces the burden on the dАP to "fire solo."

---

## THE RANKED, CHEAP-FIRST DE-RISK LADDER (all on `_riii_onsubstrate_readout_test.py`, one variable per rung)

The harness already threads every needed knob into `_build` (`enable_coincidence_detection`, `coincidence_weighted_drive`, `coincidence_k_threshold`, `coincidence_plateau_strength`, `enable_two_compartment_dap`, `apical_R`, `apical_g_couple`, `nmda_mg_concentration`) — **NO `sim/` edit for Rungs 0/1/3.** Standing anti-cheats printed by the harness every rung: **(A) LINEAR** (coincidence OFF) fails at the SAME installed attractor (non-linearity is load-bearing, not more inputs); **(B) SPECIFICITY** (non-ensemble completion < 0.20 — completeness is not all-fire runaway); **(C) FLAT control** (no installed attractor, all `w_low`) → nothing completes < 0.20 (completion rides the installed structure). GO gate: PLATEAU held-out > 0.30 AND non-ens < 0.20 AND flat < 0.20 AND linear held-out < 0.30, 6-seed.

**Rung 0 — CALIBRATE `k_thresh` to the real weighted-drive gap (no completion claim).** Use the sibling `_riii_ca3_coincidence_completion_derisk.py`'s `_cdrive_for_cue` probe (lines 137-153) which reports `held_cdrive` vs `nonstored_cdrive` — the ACTUAL weighted coincident drive on the installed attractor. Set `k_thresh` (weight units) BETWEEN them. This makes (B) specificity a per-cell property before any completeness sweep. Variable: `k_thresh`.

**Rung 1 — TWO-COMP + THIN HIGH-R APICAL sweep (the single most-likely cheap-first, the untested regime).** `--two-comp`, calibrated `k_thresh`, then sweep in this order, one at a time:
  1. **`apical_R`** ↑ from 0.15 → {1, 5, 15, 50} — the thin-dendrite loop gain (Humphries-Mellor high-R → large local ΔV → Mg ignition). *Primary lever.*
  2. **`apical_gc`** {1, 2, 3} — the delivery vs back-drain trade-off (watch for the drain-collapse as gc rises).
  3. **`plateau_strength`** {140, 220, 300} — the `g_inc` magnitude feeding `R·I`.
  4. **`mg`** {1.0, 0.5, 0.3} — a modest Mg-opening to assist ignition (NOT a substitute; the CYCLE-1066 single-comp mg sweep was null because it read the resting soma — here it reads the apical).
  Expected: ignition succeeds (v_apical → ~−15 mV plateau) for high `apical_R`; the open question is whether `gc·(v_apical−v_soma) ≈ gc·50 pA` clears the soma rheobase before the back-drain collapses the plateau. HONEST: on the edge — may complete on some seeds, may plateau just below (the CYCLE-1066 −46.5 mV was close). If GO 6-seed with (A)(B)(C) → done, NO `sim/` edit.

**Rung 2 — (only if Rung 1's passive ceiling blocks it) the named `sim/` change: a FORWARD dAP current (Bouhadjar-Diesmann I_dAP).** Add a guarded, default-off branch to the two-comp block (`bridge.py` ~6505): when the apical plateau conductance (equivalently `c_weighted ≥ k_thresh`, the count trigger) crosses `θ_apical`, inject a FIXED forward current `I_dAP` (default ~200 pA) into `total_input_current_pA` for `τ_dAP` (~60 ms), **decoupled from the symmetric `apical_g_couple`** so it is a forward-only active event (Kandel "plateau arrives at soma → burst"; Bouhadjar `I_dAP=200 pA`/`τ_dAP=60 ms`). Preserve the anti-runaway property (trigger stays the input-count switch, never the soma voltage). Byte-identical when off (mirror the existing `getattr(cfg,...,False)` guards, e.g. `enable_forward_dap`). Then re-run Rung-1's harness with it on. Anti-cheats unchanged; ADD: the forward `I_dAP` must be gated by the SAME `k_thresh` (a non-member below threshold gets ZERO forward current — specificity preserved by construction). ONE new mechanism class → fires the research-gate's "new mechanism" clause, but it is the minimal delta and directly implements the cited model.

**Rung 3 — (cheap, biology-faithful, composes with Rung 1 or 2) relax "dAP fires solo" toward Rolls-Treves convergence.** Raise the cue fraction (0.5 → 0.6-0.7) and/or `ca3_density` so each held-out member receives MORE coincident recurrent inputs (Bouhadjar's dAP is predictive at γ=5; more convergence lets the summed recurrent + dAP cross), and/or add a small tonic excitability boost (ACh-like disinhibition, Humphries-Mellor −20% threshold) to the ensemble during recall. Anti-cheat: the tonic boost alone (coincidence OFF) must NOT complete (else it is a generic drive artifact, not completion) — this is the (A) linear control with the boost on.

---

## The precise answer to the gate's question (a)

**Can the CURRENT two-compartment impl reach a completing regime by parameter choice ALONE?** Honest read: **possibly, but on the edge, and empirically untested at the right operating point** — the apical is genuinely regenerative (not passive), and a high `apical_R` (thin-dendrite) should ignite an NMDA plateau from the ~5 coincident recurrent inputs, but the delivery is the passive symmetric `apical_g_couple·(v_apical−v_soma)` whose realistic ceiling (~`gc·50 pA`, drained by back-coupling as `gc` rises) may not clear the soma rheobase. **The specific regime to try (Rung 1): `--two-comp`, `apical_R` 15-50 (up from 0.15), `apical_gc` 1-3, `plateau_strength` 140-300, `k_thresh` calibrated to the measured weighted-drive gap, `mg` 0.3-1.0.** CYCLE-1066/1067 ran the DEFAULT `apical_R=0.15`/`apical_gc=1.0` and (correctly) found it suppressive — this sweep is the genuinely untested corner.

**If Rung 1 fails, the specific `sim/` change (Rung 2):** a guarded, default-off **forward dAP plateau current** — on the count-threshold crossing, inject a fixed `I_dAP` (~200 pA, ~60 ms) forward into the soma, decoupled from the symmetric back-coupling — i.e. implement the Bouhadjar-Diesmann 2022 dAP (`I_dAP`/`τ_dAP`/`θ_dAP`) the project already cites, which Kandel p 298 grounds ("plateau arrives at the cell body → burst"). NOT an apical Na-spike+reset (a plateau ≠ a sodium spike). This is additive/byte-identical-when-off and preserves the existing anti-runaway design.

---

## Files & citations
- **Sim:** `sim/bridge.py:6432-6509` (coincidence + two-comp blocks), `6478/6482/6502-6507` (apical ODE + soma coupling); `sim/kernels.py:253-278` (`fused_coincidence_plateau`), `281-330` (graded sibling); `sim/config.py:128,137,173-217,292` (defaults quoted above); `sim/dendritic_neuron.py` (numpy Larkum-BAC reference).
- **Harness (no `sim/` edit):** `research/runners/_riii_onsubstrate_readout_test.py` (installed-attractor read-out; `--two-comp/--apical-R/--apical-gc/--k-thresh/--plateau-strength/--mg`); `research/runners/_riii_ca3_coincidence_completion_derisk.py:28-70` (`_build`, sets the flags), `:137-153` (`_cdrive_for_cue` calibration probe).
- **Biology:** Kandel 6e Ch 13 pp 293-298 (NMDA spike p 297; plateau→soma burst p 298; distal electrotonic decay p 297) = catalog **G.02**; Kandel 6e Ch 54 pp 1357,1360-1361 + O'Keefe-Nadel 1978 pp 209-215,224-227 = catalog **D.13/D.05** (CA3 completion, NMDA-dependent). Bouhadjar, Wouters, Diesmann, Tetzlaff 2022, PLoS Comput Biol 18(6):e1010233 (dAP: `θ_dAP`, `I_dAP=200 pA`, `τ_dAP=60 ms`, γ=5; predictive state). Humphries, Mellor et al., "Acetylcholine boosts dendritic NMDA spikes in a CA3 pyramidal neuron model," Neuroscience 2022, PMC7614718 (branch input-resistance ↔ synapse threshold R²≈0.80; ~10 SR / ~2.5 SLM synapses; distal NMDA big-in-dendrite/attenuated-at-soma; ACh −20% threshold for completion). Poirazi-Mel 2003 (two-layer subunit); Kastellakis-Poirazi (clustering); Rolls-Treves (completeness-vs-specificity).
- **Prior R-iii:** `2026-07-08-riii-onsubstrate-point-neuron-completion-limit-decisive-with-installed-attractor.md`, `2026-07-08-riii-dendritic-completion-surpass-cheap-first-GO.md`.
