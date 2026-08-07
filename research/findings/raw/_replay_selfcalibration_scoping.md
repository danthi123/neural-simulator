# Replay consolidation #4: emergent homeostatic self-calibration scoping (2026-08-07)

Research round for the v6 multiseed NO-GO (`2026-08-06-replay-cortical-consolidation-v6-multiseed-NO-GO-operating-point-overfit`).
No code yet; pre-build spec with a cheap STEP-0 closability gate. Transfer: this session's gateb FIX C intrinsic
homeostat (`_vocal_gateb_stage2j_intrinsic_homeostasis.py`) self-calibrated a per-seed set-point on this substrate.

## The wall + the host-tuned operating point + the per-seed quantity it can't track
v6 (learned CA1→cortex reinstatement + SFA one-of-N eviction + order-STDP) is per-seed GO on calib 412/413 but
MULTISEED NO-GO on dev 414/415/410 (retest false-recall 0.46–0.50 vs 0.15 ceiling; order margin ~0). STDP-OFF
isolation is decisive: false-recall STILL 0.44–0.50 with the order term off ⇒ **the load-bearing failure is the
INTERFERENCE-CONTROL operating point, not the order-STDP.** The operating point is a VECTOR of ABSOLUTE-unit gains
frozen on 412/413: SFA `d=180`/`a=0.02`, FS gains (`fs_to_target=44`, `target_to_fs=120`, `ca1_to_fs=40`,
`n_target_fs=12`), `cortical_target_recurrent=24`, reinstatement weights, STDP a±. **Per-seed quantity it can't
track: the per-brain E/I balance / competition working point at retest** — the SFA one-of-N eviction only silences
the unsupported (interfering, `cue_overlap=3/24`) assembly faster than the recurrently-supported correct one IF the
absolute gains match the seed's firing regime (set by cfg.seed Izhikevich thresholds `bridge.py:1508` + random
assembly/overlap membership + accumulated absolute weights). Classic seed-dependent WTA/attractor finickiness.

## Why 2-seed tuning overfits
Absolute-unit gains searched to place false<0.15 on the 412/413 E/I regime; a fresh threshold/assembly draw shifts
the regime → the same absolute eviction is mis-scaled → interfering assembly survives → false-recall→0.5. The v5+SFA
sweep already showed `false≤0.15 AND order-margin≥+0.01` unsatisfiable at a single fixed `d` even across 412/413 — a
fixed scalar operating point doesn't span even 2 seeds.

## Proposed mechanism: slow homeostatic INTEGRAL controller to a shared REGIME set-point
Replace the fixed absolute interference gains with `gain_{t+1} = gain_t + K_I·(S_measured − S*)` on a label-free
settling window over UNCUED replay, where `gain` = FS inhibitory gain (`fs_to_target_weight`) and/or SFA `d`, and `S`
= a **label-free WTA-sparsity / E-I statistic** of the `cortical_target` pop (participation ratio, or top-k/total-rate
concentration) — computed WITHOUT assembly labels/seed/false-recall metric. Frozen across seeds = the SET-POINT `S*`
+ loop gain `K_I` (one bio constant + one controller gain), NOT the absolute gain. A hotter-wrong-assembly seed
simply accrues more inhibition/SFA until the SAME regime `S*` is reached → dissolves the absolute-gain-vs-per-seed-
regime mismatch. Biology: O'Leary/Marder 2014 (homeostasis IS integral feedback control to an activity set-point);
Turrigiano 2011 (inhibitory scaling → E/I set-point); Desai 1999 (intrinsic-excitability homeostasis, the FIX C
precedent). ⛔ **CRITICAL CONSTRAINT (from the v8 source-monitor NO-GO, re-confirmed in gap#3 this session): target
the competition REGIME (WTA sparsity / E-I balance), NEVER per-assembly firing-RATE equalization** — a firing-rate
set-point EQUALIZES correct-vs-wrong rates and compresses the discrimination below floor at every operating point
(the exact same conservation/equalization trap #3 hit). Regulate the dynamical regime that lets one-of-N resolve.

## De-risk
**STEP 0 (near-zero build, decides closability BEFORE any controller):** instrument the frozen v6 runner to log, per
dev seed 414/415/410, the label-free sparsity statistic `S` alongside the scored false-recall. If a monotone relation
holds (one-winner regime ⇒ low false-recall), the set-point EXISTS → a controller will work. If false-recall stays
high where S looks one-winner → the label-free statistic is insufficient → deeper boundary, precisely named. One
instrumented replay of the existing runner.
**STEP 1 (if Step 0 passes):** v7 runner, replace ONE fixed interference gain (start `fs_to_target_weight=44`, the
direct WTA knob, OR SFA `d`) with the settling-window integral controller to `S*`; freeze everything else in v6.
Anti-cheats proving SELF-calibration (not host-set): (a) controller input ONLY label-free cortical_target activity on
uncued replay — code-asserted to exclude seed / correct-vs-wrong identity / the false-recall metric; (b) **converged
gain MUST differ across seeds** (identical ⇒ effectively host-set = FAIL); (c) `S*`,`K_I` declared+frozen before dev;
(d) fixed-gain reference = the existing v6 multiseed NO-GO (attribute the win to the loop); (e) **scrambled-input
negative** (feed the loop noise) must NOT generalize. GO = **6/6 multiseed incl. sealed held-out 417/418/419** (false
≤0.15 AND order margin ≥+0.01 AND all 4 causal lesions→0 AND memory-selective reinstatement) — NOT 2 seeds.

## Honest closability
Plausibly closable (FIX C self-calibrated per-seed this session; the 2026-06-11 LGE-homeostasis GO removed a
hand-picked operating point; O'Leary 2014 makes integral-control-to-a-set-point principled). REAL RISK: the v8
source-monitor NO-GO is a same-class counterexample (rate set-point compressed discrimination) — the proposal
survives ONLY if a label-free REGIME statistic predicts false-recall across seeds WITHOUT equalizing rates (Step 0
tests exactly this). Also: a SCALAR controller may lower false-recall but re-break the order margin (v5+SFA false-vs-
order trade-off along d) → escalate to a 2-element CO-REGULATOR (SFA-vs-inhibition), still a legitimate homeostatic
surpass. Net: Step 0 first — if the set-point predicts the score, closes with a single-knob integral homeostat (high
confidence); else the boundary is precisely named and the surpass escalates to a vector controller (no capability deferred).
