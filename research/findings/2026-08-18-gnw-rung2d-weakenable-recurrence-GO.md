---
status: live
type: finding
lane: laneC
date: 2026-08-18
---

# GNW Rung-2d — dynamically-weakenable recurrence (short-term synaptic DEPRESSION on the E->E loop) opens the "empty metastable window": GO (6/6 seeds — a sustained incumbent SELF-EVICTS via recurrent depression, an unignited window opens with NO host reset, and NEW content re-ignites in the freed workspace; the Rung-2c frozen-recurrence BOUNDARY is surpassed by the mechanism Rung-2c NAMED — Mongillo-Barak-Tsodyks 2008 STD on the recurrent excitatory synapses)

**Date:** 2026-08-18
**Runner:** `research/runners/_gnw_rung2d_weakenable_recurrence_derisk.py` (FORK of the Rung-2c runner; reuse-by-import of the Rung-2c workspace substrate `build_disinhibition_bridge` + the Rung-1/2/2b assembly-loop / ignition instruments — the STD eviction effector is added IN-RUNNER as a host-computed Tsodyks-Markram depression variable on the recurrent E->E synapse weights, **NO `sim/` edit**).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** a decisive, well-anti-cheated **6/6 GO** — short-term synaptic DEPRESSION on the workspace's recurrent excitatory (E->E) synapses makes the ignited attractor DYNAMICALLY WEAKENABLE, so a long-held incumbent depletes its own recurrent drive below the sustain knee and COLLAPSES to rest (the empty metastable window), and a new content then IGNITES in the freed workspace. This is a de-risk GO at the runner level (the assemblies remain hand-wired — a named scaffold below); it is NOT yet wired to production.
**Builds on / cites:** surpasses the Rung-2c BOUNDARY `research/findings/2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY.md` (a salience-gated dis-inhibition pulse cannot evict a FROZEN-weight recurrent attractor; it diagnosed the frozen recurrence as the missing companion process and NAMED this exact next lever).
Mechanism: **Mongillo, Barak & Tsodyks 2008**, *Science* 319:1543, "Synaptic Theory of Working Memory" (recurrent synapses carry a resources variable x, depleted by u*x per spike, recovering with tau_D). Metastability requirement: **Dehaene & Changeux 2011**, *Neuron* 70(2):200-227 (an ignited workspace state must be "destabilized" and "spontaneously replaced by another").
`NO-EXTERNAL-NEEDED:` the mechanism, its operating point, and the residual are fixed by the corpus (the Rung-2c BOUNDARY named the lever) + the cited source.

## What was built (the effector Rung-2c named)
Each recurrent E->E (assembly-loop) synapse carries a Tsodyks-Markram resources variable x (depression) and utilization u (facilitation, off in the GO op). Effective loop efficacy = base_weight * x; per presynaptic spike x depletes by u*x; x recovers toward 1 with tau_D. u modulates the per-spike depletion, NOT the resting efficacy, so **at rest x=1 the attractor weight is IDENTICAL to Rung-2c's frozen 30** and the Rung-1 ignition knee is unchanged.
The variable is host-tracked per presynaptic assembly neuron and written back into the loop synapse weights (`cp_connections.data`) each step BEFORE the substrate transmits — it is the synaptic-DEPRESSION mechanism applied to the substrate's own recurrent synapses, not a host state manipulation. The engine's native global STP stays OFF (banked foot-gun 2026-08-01); this targets ONLY the E->E recurrence with Mongillo params.
Frozen operating point (no per-seed tuning): U=0.02, tau_D=300 ms, no facilitation, drive_inc=drive_chal=5000 pA, hold=400 steps, fs_to_ws=16, ou=40, heterogeneity on, attractor_weight=30, dt=1.0.

## The instrument (the "unignited window between two ignitions" Rung-2c's BOUNDARY asked for)
A single CONTINUOUS run (NO `_restore_state` mid-run): (1) drive A -> A ignites & HOLDS; (2) hold with zero drive -> A's recurrent x depletes -> A drops below the sustain knee and COLLAPSES to rest BEFORE any challenger arrives; (3) drive B into the freed workspace -> B ignites; (4) free tail -> settled winner. A trailing-window (15-step) per-assembly rate detects: `a_ignites_holds` (A ignited in the early hold), `a_self_evicts` (A un-ignited at the END of the hold, before B is driven), `empty_window` (>= 20 consecutive steps with A off AND B off), `b_ignites` (B ignited at the free-tail end), `a_evicted_final` (A off at the end). GO iff all five, on >= 5/6 seeds.

## Result — the empty metastable window opens on all 6 seeds

The per-seed values below are rounded from the cited 6-seed artifact `research/findings/raw/_gnw_rung2d_6seed.json` (full precision lives in its `per_seed[*].metastability_on`).
<!--derived-->
Every seed shows the full ignited(A) -> EMPTY -> ignited(B) handover. A holds ~290 steps, then its recurrent x falls to ~0.43 (loop weight ~13, far below the ~22-24 sustain knee) and A collapses to a genuine REST state (windowed rate 0.000 — not merely sub-threshold). An 82-118 step EMPTY window opens (both assemblies silent), then B (fresh x=1) ignites to the period-3 plateau while A stays silent.

| seed | A holds | A self-evicts (before B) | empty window (steps) | B re-ignites (plateau) | A silent at end | xA_min | GO |
|---|---|---|---|---|---|---|---|
| 42  | ✓ | ✓ (rate 0.000 @ hold-end) | ✓ 106 | ✓ (rB 0.316) | ✓ (rA 0.000) | 0.429 | GO |
| 43  | ✓ | ✓ | ✓ 107 | ✓ (rB 0.291) | ✓ | 0.463 | GO |
| 44  | ✓ | ✓ | ✓ 118 | ✓ (rB 0.318) | ✓ | 0.464 | GO |
| 100 | ✓ | ✓ | ✓ 100 | ✓ (rB 0.307) | ✓ | 0.431 | GO |
| 101 | ✓ | ✓ | ✓ 82  | ✓ (rB 0.328) | ✓ | 0.424 | GO |
| 102 | ✓ | ✓ | ✓ 107 | ✓ (rB 0.211) | ✓ | 0.430 | GO |

`n_go = 6/6`; the ignited plateau is the Rung-1 period-3 rate ~1/3 (ignite threshold 0.167); every empty window (82-118) is far above the 20-step minimum.

## Anti-cheats (each held in the data — this is a real GO, not an artifact)
- **STD is LOAD-BEARING (the eviction is the depression):** the SAME run with x frozen to 1 (`freeze_x=True`) — no depletion — reproduces the Rung-2c frozen-recurrence BOUNDARY on all 6 seeds: A ignites & HOLDS through the entire hold window, `a_self_evicts=False`, `empty_window=False`, B fails to take over (`b_ignites=False`), lesion GO=False. `all_std_load_bearing=True`. The freeze holds throughout by construction (x pinned; plasticity off, so nothing regrows it).
- **BYTE-IDENTICAL when the STD layer is off** (asserted in the data, all 6): the STD-OFF metastable A/B spike-count timecourse is hash-identical to the freeze-x=1 timecourse (`hash_off == hash_frozen`), proving the weight-write machinery at x=1 is a provable no-op. Separately, the seeded substrate params match the pre-edit Rung-2c build EXACTLY across two separate processes (substrate hash `d9b12db6...` from the Rung-2c runner's build == the Rung-2d fork's build) — the fork reuses the identical Rung-2c substrate.
- **NO host shortcut:** `host_workspace_reset_calls == 0` on all 6 — the whole ignite/hold/evict/re-ignite sequence is one continuous free-run; the eviction is produced by the synaptic depression on the substrate, not by any host "clear the workspace" call. `std_weight_writes_on = 570` (one per step; the depression is applied every step).
- **Determinism:** build-twice at one seed -> identical hash of the seed-derived Izhikevich params (heterogeneity seeded from `cfg.seed`, NOT `actual_seed_used`) on all 6.

## Why STD opens the window where inhibition could not (the diagnosis)
Rung-2c proved somatic/feedforward inhibition cannot grade down a FROZEN dense recurrent attractor: below the inhibition that would evict, both co-ignite; above it, neither ignites; the metastable middle is empty. STD attacks the OTHER factor — the recurrence itself.
Sustained self-reverberation continuously depletes the loop's resources x, so the effective self-drive slides DOWN through the Rung-1 hysteresis loop; when it crosses the lower (sustain) knee the ignited branch ceases to exist and the assembly falls all-or-none to the REST branch (Rung-1 bistability makes the collapse sharp, not a graded plateau). Once at rest the assembly stops firing, x recovers, but the assembly stays silent (rest is stable without drive) — a clean empty window.
This is exactly CLAUDE.md's companion-process lesson realized: Rung-2c had replaced the incumbent's DYNAMIC recurrent efficacy with a STATIC frozen weight; restoring the dynamic (depression) variable is what creates the marginally-stable state Dehaene-Changeux require.

## NOT the banked "STP annihilates" negative
The 2026-08-01 STP-annihilates bank was STP on a SINGLE self-exciting pool with no competitor at a non-Mongillo operating point. Here STP is DEPRESSION on the E->E RECURRENCE of a COMPETITIVE two-assembly workspace, at U/tau_D tuned so the LOOP depletes (evictable-yet-holds) rather than the SOMA (Rung-2b intrinsic SFA, which killed the neuron before it yielded). Resting x=1 keeps the attractor at full frozen strength, so the incumbent still HOLDS; only sustained use weakens it.

## Residual / robustness (honest characterization — the operating point is a genuine window, not a knife-edge)
The edge sweep below is the 6-seed artifact `research/findings/raw/_gnw_rung2d_robustness.json` (the frozen op + two edge ops).
<!--derived-->
The GO operating point is a real metastability WINDOW that requires balance, and the anti-cheat sweeps map its edges (this IS the mechanism, not overfitting): (a) U=0.025 (stronger depression) -> the same 5000-pA B drive itself depletes B's fresh loop below the sustain knee during B's 35-step ignition pulse, so B fails to latch (2/6); (b) tau_D=250 (faster recovery) -> A's x recovers fast enough to RE-IGNITE after collapse, so the empty window shrinks below 20 steps (1/6).
The GO point (U=0.02, tau_D=300) sits comfortably between: empty windows 82-118 steps, B latches on all 6, with margin. The one delicacy is symmetric drive (B driven at 5000 = A's drive, not stronger) — a stronger B drive over-depletes B's own loop during ignition.
Named next levers if a wider window is wanted: add FACILITATION (u dynamics, already implemented behind `--facilitation`) to protect a freshly-driven assembly's first spikes; or make B's ignition drive briefer; or add a slow adaptation current as a parallel eviction timescale. None are needed for this GO.

## Do-NOT-retread (banked)
Rung-2c banked: GABA_B eviction KILLED; STP annihilates on a SINGLE self-exciting pool; intrinsic Izhikevich SFA self-extinguishes (Rung-2b); a salience-gated dis-inhibition PULSE cannot evict a frozen attractor (Rung-2c). **NOW BANKED POSITIVE:** short-term DEPRESSION on the E->E recurrence of a COMPETITIVE workspace, at U=0.02/tau_D=300, DOES open the empty metastable window (self-eviction + re-ignition, 6/6) with STD load-bearing and no host reset — this is the working eviction effector the Rung-2c BOUNDARY named.

## Remaining scaffolds (named, not claimed closed — this is a runner-level de-risk)
1. **The assemblies + per-slot cross-inhibition are hand-wired** (dense fixed-weight populations), not self-organized.
2. **The STD variable is host-computed and written into the loop weights each step** — a faithful in-runner model of the synaptic mechanism (the engine's native per-synapse STP is a banked global foot-gun); a later rung routes it through a per-pathway substrate STP on the E->E recurrence only.
3. **Content A/B and their drive timing are host-supplied external drive** (world/body-legitimate as stimuli); an emergent salience/value organ selecting what ignites is a later rung.
4. **This is a de-risk, not wired to production** — the metastable workspace is not yet reachable from `/api/brain-chat`.

## Files
Runner: `research/runners/_gnw_rung2d_weakenable_recurrence_derisk.py`. 6-seed GO artifact: `research/findings/raw/_gnw_rung2d_6seed.json` (with a `.prov.json` provenance sidecar). Reproduce: `SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2d_weakenable_recurrence_derisk --six-seed --recurrent-std --json research/findings/raw/_gnw_rung2d_6seed.json`.
