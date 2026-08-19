---
status: live
type: finding
lane: laneC
date: 2026-08-19
---

# GNW single-move THOUGHT-SWAP — recurrence-weakening self-eviction composed with a vacancy-gated IN-gate swaps one held workspace coalition DIRECTLY for another: GO 6/6. The incumbent is evicted by short-term DEPRESSION of its OWN recurrent loop (it self-collapses below the sustain knee), the freed slot admits the challenger, and the OLD state drops to BASELINE (old_residual 0.000, beating the 0.333 quench residual) while the NEW state ignites — n stays 1, no host reset, reversible (A->B->A), deterministic. The quench/STN/BG-gate inhibition-resistance wall is surpassed by the mechanism those NO-GOs NAMED: attack the RECURRENCE, not the soma. <!--derived-->(0.000 = per-seed old_residual_post; 0.333 = the quench NO-GO's period-3 plateau, quoted.)

**Date:** 2026-08-19
**Runner:** `research/runners/_gnw_recurrence_weaken_swap_derisk.py` (composition of two proven halves, both reuse-by-import; **NO `sim/` edit**; additive; native STP/homeostasis OFF).
The IN-gate substrate is `build_swap_bridge` (from `_gnw_active_overwrite_derisk.py`) in its SUPRA-CRITICAL disjoint config — overlap=0, uniform recurrence w=34, NO WTA, divisive-norm `norm_pool` + tonic `thal` — the EXACT inhibition-resistant substrate of the quench-evict / active-overwrite / BG-gate NO-GOs.
The eviction effector is `RecurrenceDepression` (the same Mongillo-Barak-Tsodyks STD as Rung-2d's `RecurrentSTD`, already built for this substrate), here TARGETED per coalition to the INCUMBENT's own recurrent E->E loop (`MultiLoopSTD`, target_units = the incumbent pattern; 9900 loop synapses/coalition = the full 100-neuron clique).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** **GO 6/6** (pooled) — a genuine single-move content SWAP on every seed with all anti-cheats holding.
**Artifacts:** `research/findings/raw/_gnw_recweaken_swap_6seed.json` (6-seed GO, +`.prov.json`), `research/findings/raw/_gnw_recweaken_swap_robustness.json` (the operating-window edge map).
**Reproduce:** `SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_recurrence_weaken_swap_derisk --six-seed --json research/findings/raw/_gnw_recweaken_swap_6seed.json`
**Builds on / cites:**
- The EVICTION mechanism (proven): `research/findings/2026-08-18-gnw-rung2d-weakenable-recurrence-GO.md` (STD depression on a coalition's OWN E->E loop makes the attractor dynamically weakenable -> it self-evicts below the sustain knee; GO 6/6). This finding APPLIES that to the swap substrate's incumbent + composes it with the IN-gate.
- The wall it surpasses: `research/findings/2026-08-19-gnw-quench-evict-overwrite-NOGO.md` (a transient open-loop FS quench does NOT evict a self-sufficient workspace attractor: a supra-critical recurrent clique is INHIBITION-RESISTANT; the wall is recurrence self-sufficiency, not the brake shape). Also `2026-08-18-gnw-stn-stop-veto-NOGO`, `2026-08-18-gnw-bg-thalamus-gate-overwrite-NOGO`, `2026-08-18-gnw-active-overwrite-NOGO`, `2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY` — all converged on the same recurrence-self-sufficiency wall; this finding closes it.
- The IN-gate (admit-6/6 half): `2026-08-18-gnw-bg-thalamus-gate-overwrite-NOGO` (a gate admits the challenger; the unsolved half was EVICTING the incumbent).
Biology: **Mongillo, Barak & Tsodyks 2008**, *Science* 319:1543 (recurrent resources x deplete u*x per spike, recover with tau_D). **Dehaene & Changeux 2011**, *Neuron* 70:200 (an ignited workspace state must be destabilizable and "spontaneously replaced by another"). **Compte, Brunel, Goldman-Rakic & Wang 2000**, *Cereb. Cortex* 10:910 (persistent-activity termination). Corpus-first (`before_you_build.sh`) + the source check were run and logged BEFORE building.

## The single move (composition)
A GNW thought-swap = replace the coalition the workspace holds with a new one. The workspace already ADMITS a new coalition (drive the challenger pattern -> it ignites; the quench NO-GO confirmed "B reaches 0.333 whenever driven"). <!--derived-->(0.333 quoted from the quench NO-GO.) The unsolved half was EVICTING the OLD held coalition — every prior lever (STN veto, BG-thal gate, active-overwrite WTA/STD, affect quench) failed because a supra-critical recurrent clique is inhibition-resistant. The named fix (Rung-2d): don't fight the incumbent with an external brake — make its OWN recurrent loop sub-critical so it collapses on its own.

The SINGLE MOVE, triggered at one moment: (1) engage short-term depression on the INCUMBENT's own recurrent E->E loop (a transient boost to the per-spike utilization U of A's loop synapses) -> A's own sustained firing depletes its loop resources x below the sustain knee -> A COLLAPSES to rest; (2) the collapse (A's private core below rate 0.05 for 12 consecutive steps) OPENS the IN-gate: after a short settle the challenger volley is admitted into the VACATED workspace and B ignites. Baseline U=0, so a held coalition NEVER self-depletes (it holds indefinitely until swapped — the inhibition-resistant incumbent); the depression BOOST is the "swap-now" trigger.

## Why the IN-gate is vacancy-gated (and why that makes the dissociation clean, not circular)
A real IN-gate does not fire content into an OCCUPIED workspace. So the challenger is admitted only once the incumbent's collapse is confirmed (a spiking vacancy read). This makes the recurrence-weakening LOAD-BEARING by the biology: no depression -> no collapse -> the gate never opens -> the incumbent holds. The circularity worry ("of course there's no swap if you never drive B") is REBUTTED by the UNGATED/forced control below: even FORCE-driving the challenger volley onto the held (un-depleted) incumbent produces NO swap on any seed (0/6) — the incumbent out-competes the volley through its intact recurrence. So the gate is not hiding a swap that would otherwise happen; the eviction is genuinely necessary.

## Result — a genuine single-move swap on all 6 seeds (per-seed, from the cited artifact)
Operating point (no per-seed tuning): boost=0.12, evict_steps<=260 (exits at the confirmed vacancy), chal_pa=5000, U_baseline=0, tau_D=250, w_rec=34, vacancy_thresh=0.05, vacancy_confirm=12, settle_gap=25, b_drive=35, heterogeneity ON.
<!--derived-->

| seed | HEADLINE win | xA_min (loop=34·x) | old_residual_post | new_rate_post | vacancy@step | LESION vacancy / old_res | FORCED (ungated) swap / old_res | reversible (xA_recovered) |
|---|---|---|---|---|---|---|---|---|
| 42  | A→B | 0.139 (4.7) | 0.000 | 0.333 | 152 | False / 0.333 | False / 0.333 | ✓ (0.988) |
| 43  | A→B | 0.140 (4.8) | 0.001 | 0.333 | 173 | False / 0.333 | False / 0.333 | ✓ (0.988) |
| 44  | A→B | 0.152 (5.2) | 0.000 | 0.333 | 148 | False / 0.333 | False / 0.333 | ✓ (0.988) |
| 100 | A→B | 0.138 (4.7) | 0.000 | 0.333 | 174 | False / 0.333 | False / 0.333 | ✓ (0.988) |
| 101 | A→B | 0.138 (4.7) | 0.000 | 0.333 | 168 | False / 0.333 | False / 0.333 | ✓ (0.988) |
| 102 | A→B | 0.135 (4.6) | 0.000 | 0.333 | 157 | False / 0.333 | False / 0.333 | ✓ (0.987) |

Every seed: the incumbent's loop depletes to x≈0.14 (effective loop weight ~4.7, far below the ~22-24 sustain knee), A collapses, the gate opens ~step 150-174, B ignites; at the read the OLD state is at BASELINE (old_residual 0.000-0.001 — it BEATS the 0.333 period-3 plateau the quench NO-GO left) and the NEW state is ignited (0.333), with n_ignited settling to EXACTLY 1 (not 0 = a stop, not 2 = co-ignition).

## Anti-cheats (each holds 6/6 — this is the result, not decoration)
<!--derived-->
- **SWAP (6/6):** win_pre=A & n_pre=1, then old_residual_post NOT ignited (< the 0.167 ignite threshold; measured 0.000-0.001) & new ignited & win_post=B & n_post=1.
- **RECURRENCE-WEAKENING LOAD-BEARING (6/6) — the NON-CIRCULAR evidence is the UNGATED control, and the gate now REQUIRES it:** the gated lesion alone (no depression -> no collapse -> gate never opens -> no swap) would be circular on its own, so the decisive control is the UNGATED/forced one.
  FORCE the SAME challenger volley onto the HELD (un-depleted) incumbent (`b_driven=True` verified on all 6): it fails to swap on every seed (0/6), old_residual stays on the plateau (0.333), the challenger never even ignites (new ~0.001-0.007).
  Only A's state (held vs STD-evicted) differs between this control and the headline, so the recurrence weakening — not the drive — is what clears the slot. `pooled_go` REQUIRES ungated-forced-swap==0 (consistent with the quench/active-overwrite NO-GOs: divisive-norm competition ALONE cannot swap a supra-critical incumbent).
- **REIGNITE (6/6):** the NEW coalition ignites and HOLDS through an extended free tail (n=1, winner B, old gone).
- **REVERSIBLE (6/6):** a two-swap A→B→A on ONE continuous substrate — after swapping A out, A's depleted loop RECOVERS (x→0.988) during the free-run, and a second swap brings A BACK as the settled winner (n=1). Short-term depression is TRANSIENT, not a permanent lesion; the third swap works.
- **NO HOST RESET (6/6):** `host_workspace_reset_calls==0` on the swap headline — a continuous run; the only host writes are external stimulus drive + the swap-trigger boost (the swap command). The eviction is the synaptic depression, not a "clear the workspace" call.
- **DETERMINISM (6/6) = the substrate-integrity anti-cheat:** the anti-cheat proper builds twice at one seed -> identical seed-derived Izhikevich-param hash (substrate-BUILD determinism; the 6 seeds' hashes are all distinct = real seed variation, not a frozen substrate).
  Separately, a full 6-seed RE-RUN was empirically **byte-identical** (0 numeric mismatches across every headline/lesion/reversibility field) — this holds because the arms share one bridge + one OU stream in a FIXED order; it is full-trajectory reproducibility of that fixed sequence, not a claim that the OU realization is irrelevant (and OU 30 pA << the 5000 pA drives, so the qualitative GO is robust to it — consistent xA_min / vacancy_step across 6 independent streams).
  NOTE: the `additive_substrate` hash anti-cheat is N/A here — NO pool is appended to `build_swap_bridge`, so nothing perturbs the base Izhikevich RNG draw (the RNG-prefix property does not hold on this engine — a known quirk banked in the quench NO-GO).

## Robustness — a genuine operating WINDOW, not a knife-edge (edge map artifact)
<!--derived-->
The single-swap + lesion-hold + load-bearing criteria hold **6/6 at EVERY edge tried**; the two-swap reversibility is the more demanding one. Edges (6-seed each): boost=0.09 -> swap 6/6, reversible 4/6 (weaker depression: swap2 less reliable); boost=0.18 -> full GO 6/6; chal_pa=4000 -> GO 5/6 (a weaker challenger still latches — the gate guarantees a vacated slot); evict_steps=180 -> swap 6/6 but reversible 0/6 (too short for the second swap's collapse+recovery). The GO point (boost=0.12, evict<=260, chal=5000) sits with margin inside boost∈[0.12,0.18], chal_pa∈[4000,5000], evict>=~220.

## Why it works where inhibition could not (the diagnosis)
The quench/STN/BG-gate NO-GOs applied SOMA-level feedback inhibition to a self-sufficient recurrent clique: g_i could double and the incumbent stayed on the plateau, because its recurrent excitation was intact and self-sustaining (the wall was recurrence self-sufficiency).
This mechanism attacks the RECURRENCE itself: depleting the incumbent's own loop resources slides its self-drive DOWN through the Rung-1 hysteresis loop until it crosses the lower (sustain) knee, and the ignited branch ceases to exist — the assembly falls all-or-none to rest.
Once the incumbent has vacated, the challenger enters a genuinely free slot (no competition), so a strong short volley latches cleanly and the divisive normalization then keeps the evicted coalition down.
This is CLAUDE.md's companion-process lesson realized twice over: the prior levers replaced the incumbent's DYNAMIC recurrent efficacy with a static frozen weight AND fired content into an occupied workspace; restoring the dynamic (depression) variable AND gating admission on vacancy is what makes the marginally-stable, replaceable state Dehaene-Changeux require.

## A real bug found and fixed mid-arc (banked)
`RecurrenceDepression` snapshots its `base` recurrent weights from `cp_connections.data` AT CONSTRUCTION. An STD instance built AFTER a prior arm had run captured DEPRESSED (too-low) base weights -> an underpowered incumbent loop that spontaneously collapsed even in the lesion (a false "eviction without depression"). Fix: construct ALL `MultiLoopSTD` instances up-front on the freshly-built substrate, before any arm runs (so each captures the true base=34). With the fix, the lesion and the ungated control both hold 6/6 — the earlier apparent "competition displaces the incumbent" for 2 seeds was entirely this artifact; the real substrate is uniformly inhibition/competition-resistant.

## Honest limits / remaining scaffolds (named, not claimed closed — this is a runner-level de-risk)
1. The coalitions are hand-wired dense frozen populations (disjoint 100-neuron cliques), not self-organized.
2. The EVICTION is substrate-based (short-term depression writing base*x into the incumbent's OWN recurrent synapses), but the IN-gate ADMISSION is host-orchestrated: the STD variable is host-computed and written into the loop weights each step (a faithful in-runner model; the engine's native global STP is a banked foot-gun), and the vacancy detector, settle timing, swap-trigger boost, and challenger injection are host control logic reading `_instant_private_rate` (world/body-legitimate stimuli + a top-down swap command). So the incumbent's departure is neural; the "admit the next thought" decision is not yet. A later rung routes the STD through a per-pathway substrate STP and drives the vacancy read + swap trigger from spiking control/salience organs (a neural IN-gate is the real target).
3. The "single move" is one TRIGGERED operation spanning the collapse+admit (~200 ms: ~150 steps depression collapse + 25 settle + 35 admit), not an instantaneous replacement. This is the faithful biological timescale (slow-but-faithful is in scope), not a shortcut.
4. This is a de-risk at the runner level; it is NOT yet wired to production (`/api/brain-chat`).

## Files
Runner: `research/runners/_gnw_recurrence_weaken_swap_derisk.py`. Artifacts: `research/findings/raw/_gnw_recweaken_swap_6seed.json` (+`.prov.json`), `research/findings/raw/_gnw_recweaken_swap_robustness.json`.
