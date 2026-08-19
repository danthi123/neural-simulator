---
status: live
type: finding
lane: laneC
date: 2026-08-18
---

# GNW active OVERWRITE — a clean content SWAP (replace ignited incumbent A with challenger B while n_ignited stays 1) is a characterized NO-GO on the distributed workspace: a structural BREAK-IN vs LOCKOUT catch-22 (any mechanism strong enough to hold A as a stable single-content workspace also LOCKS OUT B before it can trigger eviction; any mechanism that lets B break in settles into a stable n=2 CO-IGNITION). Selectivity (unchallenged-hold) 6/6; both boundary horns quantified; substrate anti-cheats pass; next lever named.

**Date:** 2026-08-18
**Runner:** `research/runners/_gnw_active_overwrite_derisk.py` (FORK of `_gnw_distributed_overwrite_workspace_derisk.py`; reuse-by-import of the Rung-1 ignition / Rung-2 competition instruments + the Rung-2b determinism hash + the Rung-2c dense-population + the _p1_2 full snapshot/restore + the distributed-overwrite base build for the byte-identical anchor. Adds a SPLIT recurrence (sub-critical private core), a target-restricted Tsodyks-Markram depression, per-slot LATERAL WTA inhibition, and a self-limiting DYNAMIC eviction gate — all IN-RUNNER, **NO `sim/` edit**).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** **NO-GO** — the clean swap (SWITCH: A->B with n_ignited settling to exactly 1) is achieved on **0/6** seeds by any of the three named levers alone, their composition, or the added WTA + self-limiting-gate mechanism. SELECTIVITY (an unchallenged confident A commit holds n=1, winner A) holds **6/6**. The failure is a mapped substrate property, not a tuning miss.
**Artifact:** `research/findings/raw/_gnw_active_overwrite_6seed.json` (+ `.prov.json` sidecar).
**Builds on / cites:** the parent PARTIAL `research/findings/2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md` (global STOP GO 6/6; active-overwrite 0/6, NAMED the three levers implemented here); the Rung-2d GO `research/findings/2026-08-18-gnw-rung2d-weakenable-recurrence-GO.md` (STD self-eviction = a SLOW clear-then-reload overwrite, the achievable fallback below); the Rung-2c BOUNDARY `research/findings/2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY.md` (inhibition alone cannot evict a frozen attractor); the STN-veto NO-GO `research/findings/2026-08-18-gnw-stn-stop-veto-NOGO.md`.
Biology: **Carandini & Heeger 2012**, *Nat Rev Neurosci* 13:51 (divisive normalization); **Mongillo, Barak & Tsodyks 2008**, *Science* 319:1543 (short-term depression, resources x depleted u*x/spike, recover tau_D); **Dehaene & Changeux 2011**, *Neuron* 70:200 (an ignited state must be destabilizable and "spontaneously replaced by another"); **Frank 2006** / **Aron & Poldrack 2006** (hyperdirect/lateral inhibition, inherited from the STN chain). External-literature check done at this wall (logged, lane C): **O'Reilly & Frank 2006**, *Neural Comput* 18:283 (PBWM basal-ganglia selective gating displaces PFC WM states with new info via an external gate); **Lundqvist/Lisman** attractor WM (theta-nested gamma multi-item phase segregation, PMC5039104) — both name the next lever below.

## The question and the three named levers (the parent's own next-levers)
The distributed divisively-normalized workspace can STOP (conflict-triggered depression of the SHARED recurrence -> n=0, GO 6/6), EVICT from within (Rung-2d STD), and SELECT (STN sensor). The residual is ACTIVE OVERWRITE: replace ignited A with challenger B while n_ignited stays ~1 — a genuine content SWAP (A's identity gone, B's delivered, n stays 1), NOT co-ignition n->2, NOT a clear-then-reload. The parent named three composable levers, all implemented + tested here:
1. **timed self-eviction** — drive B concurrently while A's shared-recurrence STD depletes, timed so A self-evicts as B's drive is present;
2. **larger overlap + a SUB-CRITICAL private core** — split the recurrence so each pattern's private core cannot self-sustain alone (verified: private-alone rate 0.000, full-pattern 0.333 rounded, sub_critical 6/6) while the full pattern holds; <!--derived-->
3. **use-driven depression targeting ONLY the shared A-B neurons** — a Tsodyks-Markram STD restricted to the A/B overlap.
Beyond the three, the strongest additional lever was also built and tested: per-slot **LATERAL WTA inhibition** (a rising winner suppresses its rivals) + a **self-limiting DYNAMIC gate** (the depletion boost is ON only while A and B are BOTH ignited, OFF the moment A drops out, so the challenger's recurrence recovers and holds once it wins). The asymmetry meant to evict A specifically: A rides the recurrence alone, B has external drive, so depleting recurrence starves A first.

## The result — 0/6 clean swaps; a structural BREAK-IN vs LOCKOUT catch-22
The per-seed values below are from the cited 6-seed artifact.
<!--derived-->

| seed | headline (disjoint+WTA+dyngate) win, n | lever2 n_post | lever3 n_post | SELECTIVITY (unchallenged A) | clean swap |
|---|---|---|---|---|---|
| 42  | 0->0, 1->1 (empty 0)  | 2 | 2 | hold n=1 | NO |
| 43  | 0->0, 1->1 (empty 0)  | 2 | 2 | hold n=1 | NO |
| 44  | 0->0, 1->1 (empty 47) | 2 | 2 | hold n=1 | NO |
| 100 | 0->0, 1->1 (empty 0)  | 2 | 2 | hold n=1 | NO |
| 101 | 0->0, 1->1 (empty 47) | 2 | 2 | hold n=1 | NO |
| 102 | 0->0, 1->1 (empty 47) | 2 | 2 | hold n=1 | NO |

**clean-swap = 0/6; selectivity (unchallenged-hold) = 6/6.** The two failure modes are the two horns of one catch-22:
- **CO-IGNITION horn** (the three named levers over overlapping patterns; lever2/lever3 n_post=2 on all 6): the challenger B DOES break in, but the drive-sustained incumbent A does NOT leave -> a stable n=2 co-ignition. Traced directly: while B is driven the workspace holds `AB AB AB ...` and settles at n=2. Depleting the shared/full recurrence to x~0.4 does not evict A, because the thalamic tonic support AND B's own external drive keep the shared units firing, so A rides the partially-depleted recurrence (this reproduces the Rung-2c "inhibition/depletion cannot grade down a driven attractor" horn from the distributed side).
- **LOCKOUT horn** (the disjoint substrate + WTA lateral inhibition strong enough to give a selective single-content workspace; headline win 0->0 on all 6): A's lateral inhibition + the global divisive normalization suppress B BEFORE it can break in, so the self-limiting gate never sees co-activity, never fires, and A holds (n stays 1 as A). Raising WTA to force the eviction instead destabilizes the solo-A hold (selectivity fails at wta_w>=4 on the fragile seed 101), never producing a swap.

A clean swap needs B to break in JUST enough to trigger eviction, then A to leave and B to hold — a knife-edge. A single seed-42 noise realization produced it once; it did not survive heterogeneity or a re-draw (0/6). The tension is structural, not a missed operating point: it was swept across WTA in {0,3,4,5,6,8,12}, overlap in {0,12,15,20,30}, w_priv (sub-critical depth), baseline U, and conflict boost. Every robust outcome is either HOLD or CO-IGNITION.

## Anti-cheats (each held on all 6 — the NO-GO is real, not an instrument failure)
- **SUBSTRATE-DRIVEN, no host poke:** the headline is a CONTINUOUS run — `host_workspace_reset_calls == 0` and `host_content_swap_calls == 0` on all 6 (the only host writes are the external stimulus drive = world/body-legitimate). The "A holds" / "co-ignition" outcomes are substrate dynamics, not a host content manipulation.
- **BYTE-IDENTICAL substrate (separate process):** with the overwrite path FLAG-OFF (wta_w=0, base overlap 15, uniform recurrence) the seed-derived Izhikevich params hash EQUALS the distributed-overwrite base build at the same seed (`d9b12db6...`, verified across two separate processes; it also matches the Rung-2d cited substrate hash) — the overwrite additions are purely additive wiring/regions, not a different random draw. The headline build (disjoint + WTA pools, 530 neurons) is a distinct ADDITIVE substrate whose first 440 neurons are the same seeded draw.
- **DETERMINISM:** build twice at one seed -> identical substrate hash (heterogeneity seeded from `cfg.seed`, NOT `actual_seed_used`), all 6, and identical across separate processes.
- **LEVER-2 KNOB VERIFIED:** the sub-critical private core is real — driving only a private core does NOT ignite it (rate 0.000) while the full pattern does (0.333 rounded); `sub_critical_private` 6/6. So lever 2 was genuinely exercised; it still cannot deliver a swap (the incumbent co-ignites through the shared pool). <!--derived-->
- The STD/WTA "load-bearing" causal checks are reported as N/A (`std_load_bearing`/`wta_load_bearing` = False): there is no swap to attribute, so a lesion cannot abolish one. This is stated, not hidden — a load-bearing claim without an achieved effect would be an overclaim.

## The achievable fallback (banked at Rung-2d, NOT re-claimed here)
A SLOW overwrite — A self-evicts via STD-depletion of its recurrence, a transient EMPTY workspace opens, then B re-ignites in the freed slot (settled n=1, winner=B) — IS achievable and is the Rung-2d GO. But it transits an EMPTY window (a STOP then a reload), which is exactly the "clear-then-reload" mode the clean-swap gate excludes. On THIS runner's concurrent-B protocol even that did not appear (0/6 switched-identity): driving B during A's decline pushes the workspace into the co-ignition or lockout horn rather than the empty-window handover. So the substrate offers STOP, co-access (n=2), and a SLOW clear-then-reload — but not a clean n-stays-1 SWAP.

## Why this is a genuine boundary (the deepest lesson, per CLAUDE.md) — the missing companion process is an EXTERNAL GATE
The external-literature check (logged, lane C) resolves the "what does the real system run alongside this?" question decisively: the biological workspace does NOT resolve incumbent-vs-challenger by intrinsic rate competition over a shared resource at all — that is precisely the catch-22 our levers live in. It uses an EXTERNAL, dedicated gate. Two convergent mechanisms in the literature:
- **Basal-ganglia selective OUTPUT gating (PBWM; O'Reilly & Frank 2006, *Neural Comput* 18:283):** the BG-thalamus opens a per-slot gate that DISPLACES the prefrontal activity state with new information; the model explicitly notes that "when gating demands switch... a transient period of conflict ensues in which the previous and currently relevant information are competing" — i.e. the swap is arbitrated by a gate that actively DISINHIBITS the challenger's slot (and can inhibit the incumbent's), not by the two attractors fighting over one resource. This is exactly the missing companion process: an external actor that opens the slot.
- **Theta/gamma phase multiplexing (Lundqvist/Lisman attractor WM; theta-nested gamma cyclic reactivation, PMC5039104):** multiple items coexist by occupying DIFFERENT theta phases, so a challenger need never evict an incumbent from the same instantaneous resource — removing the break-in/lockout competition entirely.
Our levers all attack the same instantaneous competition the catch-22 lives in; biology sidesteps it with a gate and/or phase segregation.

## Next lever (named, not deferred — a wall is a verdict on a METHOD)
1. **A BG-thalamus SELECTIVE GATE (PBWM; O'Reilly & Frank 2006):** wire a per-slot gate that, on a challenger's afferent volley, phasically DISINHIBITS the challenger's thalamic drive AND inhibits the incumbent's slot — an EXTERNAL arbiter that opens the slot for B and closes it for A, rather than asking the two attractors to compete. Buildable in-runner from the existing thal + per-slot inh pools (make thal->ws slot-specific and gate it). This directly attacks the catch-22 root (no intrinsic competition to lose).
2. **Theta-phase multiplexing** (Lundqvist/Lisman; SPEAR/Hasselmo, Lisman-Idiart): segregate incumbent and challenger into DIFFERENT theta phases so B occupies a phase while A is phase-suppressed. Buildable as a phasic drive-and-inhibition schedule on the existing pools.
3. **Novelty/mismatch -> phasic targeted hyperpolarization of the incumbent**, timed to the challenger's afferent volley — a comparator (afferent-B present AND a DIFFERENT content ignited) driving a brief, incumbent-specific inhibitory transient that opens the slot exactly as B arrives.
Lever 1 (the gate) is the primary target BEFORE re-attempting the competitive levers, which are now banked as the boundary.

## Do-NOT-retread (banked)
On the distributed divisively-normalized workspace, a clean active-overwrite SWAP (n stays ~1) is a **structural NO-GO** via the break-in/lockout catch-22.
(a) The three named levers (timed self-eviction / larger-overlap sub-critical private core / targeted shared-A-B STD) all settle into a stable **n=2 CO-IGNITION** — a driven incumbent, propped by the thalamic tonic + the challenger's own drive, does not vacate a depleted-but-still-driven shared recurrence.
(b) Adding per-slot **LATERAL WTA** strong enough to give single-content selectivity **LOCKS OUT** the challenger before it can co-activate (so no eviction fires), and stronger WTA destabilizes the solo-A hold instead of swapping.
(c) The clean swap is a noise-fragile knife-edge (1 fluke, 0/6 robust). Selectivity (unchallenged-hold) is 6/6 and all substrate anti-cheats pass, so this is a mapped substrate property.
**BANKED NEGATIVE.** The achievable content replacement remains the Rung-2d SLOW clear-then-reload (via an empty window), not an n-stays-1 swap.

## Remaining scaffolds (named, not claimed closed)
1. The K cliques + norm_pool + thal + WTA pools are hand-wired dense frozen pools, not self-organized.
2. The STD is host-computed and written into the recurrence weights each step (a faithful in-runner model, the Rung-2d pattern; native global STP stays OFF). The self-limiting gate reads the workspace's own per-slot ignition (a host-side conflict effector, as in the parent).
3. Content and drive timing are host-supplied external drive (world/body-legitimate as stimuli).
4. This is a de-risk, not wired to production — the distributed workspace + overwrite path is not reachable from `/api/brain-chat`.

## Files
Runner: `research/runners/_gnw_active_overwrite_derisk.py`. 6-seed artifact: `research/findings/raw/_gnw_active_overwrite_6seed.json` (+ `.prov.json` sidecar). Reproduce: `SIM_BACKEND=numpy python -u -m research.runners._gnw_active_overwrite_derisk --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_active_overwrite_6seed.json`. Smoke: add `--smoke --seed 42 --wta-grid 3 5 --cb-grid 0.18 0.28`.
