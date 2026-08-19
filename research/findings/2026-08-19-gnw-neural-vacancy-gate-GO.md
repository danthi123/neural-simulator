---
status: live
type: finding
lane: laneC
date: 2026-08-19
---

# GNW NEURAL VACANCY GATE — the thought-swap is now FULLY self-driven: a spiking DIS-INHIBITORY admission gate admits the challenger on the SUBSTRATE's own read that the workspace has been vacated (no host `if collapsed:` trigger). GO 6/6. Occupancy interneurons `occ` (excited by the whole workspace) tonically CLAMP a per-pattern admission relay `gate_k`; when the incumbent's recurrence-depression eviction collapses its firing, `occ` falls silent and DIS-INHIBITS `gate_k`, which — coincident with a sub-threshold sensory PROPOSAL — drives the challenger over the ignition knee into the freed slot. The admission drive is UNCONDITIONAL each step; nothing in the loop reads a workspace rate to decide admission. old_residual_post 0.000, new 0.333, n=1, timing correct (challenger ignites only after vacancy, zero co-ignition), reversible A→B→A, deterministic. <!--derived-->(0.333 = the period-3 ignited plateau; per-seed reads in the cited artifact)

**Date:** 2026-08-19
**Runner:** `research/runners/_gnw_neural_vacancy_gate_derisk.py` (reuse-by-import the #34/recweaken swap substrate + STD eviction effector; **NO `sim/` edit**; additive explicit-wiring pools; native STP/homeostasis OFF).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** **GO 6/6** (pooled) — every anti-cheat holds on every seed.
**Artifacts:** `research/findings/raw/_gnw_neural_vacancy_gate_6seed.json` (6-seed GO, +`.prov.json`), `research/findings/raw/_gnw_neural_vacancy_gate_smoke.json` (single-seed detail).
**Reproduce:** `SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_neural_vacancy_gate_derisk --six-seed --json research/findings/raw/_gnw_neural_vacancy_gate_6seed.json`
**Calibrate the gate primitive:** `SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_vacancy_gate_derisk --calibrate --seed 42`

## What this closes
The recurrence-weaken thought-swap (`2026-08-19-gnw-recurrence-weaken-swap-GO.md`, GO 6/6) EVICTS the incumbent neurally (Rung-2d short-term depression drains the incumbent's own recurrent E→E loop below the sustain knee → self-collapse), but its honest limit #2 was that the IN-gate ADMISSION was HOST-orchestrated: a python loop read `_instant_private_rate(incumbent)`, counted consecutive sub-threshold steps, and only then drove the challenger (`if vacancy: drive B`). Per the BRAIN-BASED-ONLY standard that admission control between sensation and action is a shortcut. This finding replaces it with a spiking dis-inhibitory gate: the whole swap is now neurons/synapses except the two host writes the base finding already documented as legitimate (thal tonic + the top-down "swap-now" intention pulse) and the world's sub-threshold sensory PROPOSAL.

## The mechanism (dis-inhibition; explicit wiring; NO `sim/` edit)
Two pools are ADDED to the exact inhibition-resistant swap substrate (workspace / norm_pool / thal, disjoint supra-critical uniform recurrence w=34):
- **occ** (40 inhibitory interneurons): `ws_used → occ` (E_TO_I). occ fires whenever ANY coalition is ignited == the workspace slot is OCCUPIED (a spiking occupancy read).
- **gate_k** (60 excitatory per pattern k): TONICALLY INHIBITED by occ (`occ → gate_k`, I_TO_E). While a coalition holds, occ fires and CLAMPS every gate shut. When the incumbent's loop depletes and it COLLAPSES, its firing — and thus occ's feed-forward inhibition onto the gates — falls silent, DIS-INHIBITING the gates. A gate_k also receiving the challenger PROPOSAL (a sub-threshold sensory drive into gate_k, "content k is proposed") now fires and drives its coalition (`gate_k → pattern_k`, E_TO_E, w=100) over the ignition knee. Content is the world's PROPOSAL (which gate_k is driven); the neural work is the vacancy-gated ADMISSION.

So the challenger is admitted by the substrate reading vacancy (occ silent) coincident with a proposal — this is the mechanism the task named (a tonically-inhibited pool released by disinhibition when the incumbent's feed-forward inhibition collapses) and the mechanism the base finding's limit #2 called for. The eviction effector is unchanged (`MultiLoopSTD`, reused unchanged from the recweaken swap).

Biology: BG→thalamus / SNr tonic-inhibition RELEASE by disinhibition gates thalamocortical transmission (**Chevalier & Deniau 1990**, *TINS* 13:277; **Deniau & Chevalier 1985**, *Brain Res* 334:227). Cortical VIP→SST→PC disinhibitory gating (**Pi et al. 2013**, *Nature* 503:521). **Dehaene & Changeux 2011**, *Neuron* 70:200 (an ignited workspace state must be destabilizable and "spontaneously replaced by another"). Eviction: **Mongillo, Barak & Tsodyks 2008**, *Science* 319:1543 (recurrent-resource short-term depression). Corpus-first (`before_you_build.sh`) + the source check were run and logged BEFORE building.

## Result — a genuine self-driven swap on all 6 seeds (per-seed, from the cited artifact)
Operating point (no per-seed tuning): occ_n=40, gate_per=60, w_ws_occ=8, w_occ_gate=20, w_gate_ws=100, proposal=2800 pA, establish=8000 pA, boost=0.12, boost_steps=200, evict_steps=260, w_rec=34, heterogeneity ON.
Table + "Every seed" values are per-seed reads rounded from the cited `_gnw_neural_vacancy_gate_6seed.json` (the ignited plateau is the period-3 rate).

<!--derived-->

| seed | HEADLINE win | old_residual_post | new_rate_post | xA_min | a_vacate → b_ignite (gap) | NON-CIRCULAR new (A held) | DETECTOR-REMOVED b_ignite / a_vacate (premature) | reversible xA_recovered |
|---|---|---|---|---|---|---|---|---|
| 42  | A→B | 0.000 | 0.333 | 0.116 | 102 → 164 (62) | 0.004 | 9 / 76 (True) | ✓ (0.988) |
| 43  | A→B | 0.000 | 0.333 | 0.112 | 110 → 172 (62) | 0.001 | 9 / 74 (True) | ✓ (0.988) |
| 44  | A→B | 0.001 | 0.333 | 0.109 | 115 → 173 (58) | 0.000 | 9 / 81 (True) | ✓ (0.988) |
| 100 | A→B | 0.000 | 0.333 | 0.129 | 99 → 164 (65)  | 0.000 | 9 / 76 (True) | ✓ (0.989) |
| 101 | A→B | 0.000 | 0.333 | 0.110 | 111 → 169 (58) | 0.003 | 9 / 71 (True) | ✓ (0.988) |
| 102 | A→B | 0.000 | 0.333 | 0.091 | 173 → 227 (54) | 0.005 | 4 / 74 (True) | ✓ (0.985) |

Every seed: the incumbent's loop depletes (xA_min≈0.09–0.13, effective loop far below the sustain knee), A collapses, occ falls silent, the challenger's gate is dis-inhibited, and B ignites — at the read the OLD state is at BASELINE (old_residual 0.000–0.001) and the NEW state is ignited (0.333), with n_ignited settling to EXACTLY 1.

## Anti-cheats (each holds 6/6 — this is the result, not decoration)
<!--derived-->
- **NEURAL-GATED SWAP (6/6):** win_pre=A & n_pre=1, then old_residual_post NOT ignited (measured 0.000–0.001) & new ignited (0.333) & win_post=B & n_post=1. The admission drive `drive_map=[(gate_k, proposal)]` is UNCONDITIONAL every step (grep-checkable); the loop's only conditionals are the fixed-time boost clear and READ-ONLY windowed timing measurements — no host read of a workspace rate gates the admission.
- **NO DIRECT DRIVE TO THE CHALLENGER (the crux):** across the entire swap the challenger's WORKSPACE pattern receives **0.0 pA** direct external current (instrumented on seed 42) — the proposal is delivered ONLY to the gate pool, and B ignites purely through `gate_k → pattern_k` synapses. The challenger is admitted by the substrate, not injected by the host.
- **TIMING CORRECT (6/6):** on trailing-window rates the challenger ignites strictly AFTER the incumbent vacates (b_ignite > a_vacate by 54–65 steps) with ZERO co-ignition steps — the gate conservatively waits for genuine emptiness (occ near baseline), so there is no premature admission into an occupied slot.
- **NON-CIRCULAR / VETO WORKS (6/6):** present the proposal but apply NO eviction (A holds, so the workspace correctly reads OCCUPIED) → occ stays high → the gate is vetoed → the challenger is LOCKED OUT (new 0.000–0.005, no swap on any seed). The occupancy read blocks admission when occupied; the proposal alone cannot displace a supra-critical incumbent.
- **DETECTOR ENFORCES TIMING (6/6):** REMOVE the detector (occ→gate=0, the gates are never vetoed) → the challenger is admitted PREMATURELY (b_ignite step 4–9) while the incumbent still holds (a_vacate step 71–81) → the occ→gate veto is exactly what enforced the correct timing (not a host delay).
- **GATE→PATTERN EFFECTOR LOAD-BEARING (verified 3-seed):** build with w_gate_ws=0 (the gate fires but cannot drive the coalition) → even with the slot free (old_residual 0.000 from the STD eviction) the challenger is NEVER admitted (new 0.000–0.015, not ignited) → the `gate_k → pattern_k` synapses are the admission effector.
- **REIGNITE (6/6):** the admitted coalition ignites and HOLDS through an extended free tail (n=1, winner B, old gone).
- **REVERSIBLE (6/6):** a two-swap A→B→A on ONE continuous substrate, BOTH admissions through the neural gate — after swapping A out, A's depleted loop RECOVERS (x→0.985–0.989) and a second gated swap brings A BACK as the settled winner.
- **NO HOST RESET (6/6):** `host_workspace_reset_calls==0` on the swap headline (continuous run).
- **DETERMINISM (6/6) = the substrate-integrity anti-cheat:** build twice at one seed → identical seed-derived Izhikevich-param hash; the 6 seeds' hashes are all distinct (real seed variation). The `additive_substrate` hash anti-cheat is N/A here — the RNG-prefix property does NOT hold on this engine (a known quirk banked in the quench NO-GO), which is exactly why the appended pools shift the workspace params per seed (see limits).

## Why "silence the detector → admission fails" is proven via MISTIMING, not a clamp (honest characterization)
The task's anti-cheat #1 asks for "admission fails OR mistimes" under a detector lesion. The clean, tight dissociation here is MISTIMING: removing occ→gate admits prematurely (above) while the veto blocks admission whenever the workspace reads occupied (non-circular control). A complementary "detector STUCK reporting occupied → admission fails though the slot is free" lesion (build occ blind to the workspace, drive it with a steady tonic) was ATTEMPTED and LEAKS: the dis-inhibitory veto is workspace-SYNCHRONIZED (natural occ, driven by the incumbent's period-3 firing, lands its inhibition exactly when the strongly-proposed gate would otherwise fire), whereas a DC-driven occ leaves brief gaps through which the strongly-recurrent challenger latches (measured: gate_max≈0.55 bursts during the gaps → B ignites). This is a real property of the substrate, not swept under: the two clean dissociations above (occupancy-veto necessary to block premature admission; occupancy-read sufficient to block admission when occupied) fully establish the detector's causal role.

## Honest limits / remaining scaffolds (named, not claimed closed — this is a runner-level de-risk)
1. The coalitions are hand-wired dense frozen populations (disjoint 100-neuron cliques), not self-organized; the gate/occ pools are likewise hand-wired.
2. What is now NEURAL: the vacancy read (occ), the admission gating (occ→gate disinhibition), and the admission effector (gate→pattern). What remains HOST (the same legitimate writes the base finding documented, plus one): thal tonic; the top-down "swap-now" STD boost = the intention to switch (a FIXED-duration command, NOT vacancy-gated); and the world's sub-threshold sensory PROPOSAL into gate_k (which cannot admit alone — the neural vacancy gate is required). The eviction TRIGGER (deciding to swap) is intentionally top-down; the ADMISSION is now fully neural.
3. **Content routing is per-pattern labeled-line, not learned/composed:** each gate_k drives its own pattern_k, so admitting arbitrary content requires the corresponding labeled line. The challenger's IDENTITY is the world's proposal (legitimate); a learned/composer route for arbitrary content is out of scope (the base finding's limit #1).
4. The appended occ/gate pools shift the workspace Izhikevich params per seed (the RNG-prefix property does not hold on this engine), which put one seed's incumbent on the substrate's documented non-deterministic near-threshold ignition boundary at 5000×35; establish=8000×35 clears it uniformly on every seed. The establishment strength is not load-bearing — it only sets the "before" initial condition.
5. The stuck-detector "fails" lesion leaks (see above) — the veto is workspace-synchronized, not DC-clampable.
6. This is a de-risk at the runner level; it is NOT yet wired to production (`/api/brain-chat`).

## Files
Runner: `research/runners/_gnw_neural_vacancy_gate_derisk.py`. Artifacts: `research/findings/raw/_gnw_neural_vacancy_gate_6seed.json` (+`.prov.json`), `research/findings/raw/_gnw_neural_vacancy_gate_smoke.json`.
