---
status: live
type: finding
lane: laneC
date: 2026-08-18
---

# GNW active OVERWRITE via a per-slot BG-THALAMUS GATE — the external gate SOLVES the catch-22's LOCKOUT horn (a conflict-gated striatal-Go disinhibition ADMITS the challenger past the divisive-normalisation lockout the WTA could not, 6/6) but the CO-IGNITION horn REMAINS a characterised NO-GO: on the supra-critical recurrence substrate the incumbent is thalamus-INDEPENDENT, so the gate's eviction (brute per-slot inhibition) is inhibition-resistant (Rung-2c) and n settles at 2, not 1. Selectivity 6/6; all substrate anti-cheats pass; the eviction-by-loop-closure next lever needs sub-critical (thalamus-dependent) maintenance — prototyped, single-content maintenance achieved, the eviction-vs-self-Go robustness is the residual.

**Date:** 2026-08-18
**Runner:** `research/runners/_gnw_bg_thalamus_gate_overwrite_derisk.py` (FORK of `_gnw_active_overwrite_derisk.py`; reuse-by-import of the workspace geometry + the STN conflict-sensor read + the Rung-2b determinism hash + the Rung-2c dense-population + the _p1_2 snapshot/restore + the active-overwrite base build for the byte-identical anchor. Adds the per-slot BG-thalamus gate: thal_k relay held by a tonic gpi_k, a striatal-Go str_k that disinhibits it, and a gate_inh_k eviction interneuron — all IN-RUNNER, **NO `sim/` edit**).
**Backend:** CPU (numpy, BLAS-capped=2). **Seeds:** 42/43/44/100/101/102. **Verdict:** **NO-GO** — a clean SWITCH (identity A->B, n_ignited settling to exactly 1) is achieved on **0/6** seeds. **SELECTIVITY** (an unchallenged confident A holds n=1, winner A) holds **6/6**. **ADVANCE:** the disinhibition ADMITS the challenger (solves the LOCKOUT horn) on **6/6**. **RESIDUAL:** the incumbent is never evicted (co-ignition horn) on **0/6** — n settles at 2.
**Artifact:** `research/findings/raw/_gnw_bg_thalamus_gate_6seed.json` (+ `.prov.json` sidecar).
**Builds on / cites:** the parent NO-GO `research/findings/2026-08-18-gnw-active-overwrite-NOGO.md` (the break-in/lockout catch-22; NAMED the BG-thalamus gate as lever 1, implemented here); the Rung-2c BOUNDARY `research/findings/2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY.md` (inhibition alone cannot evict a frozen attractor — the wall the eviction arm hits); the STN-veto GO `research/findings/2026-08-18-gnw-rung-stn-stop-veto...` (the conflict sensor + OU-async, reused); the genuine-disinhibition resolution `research/findings/2026-06-04-cheat2-genuine-bg-disinhibition-RESOLVED.md` (opening a thalamic gate by DIRECT current is a cheat — here the disinhibition is genuine, driven through a striatal-Go pool that silences the tonic GPi).
Biology: **O'Reilly & Frank 2006**, *Neural Comput* 18:283 (PBWM basal-ganglia selective output gating; a per-stripe gate DISPLACES the prefrontal state with new information, disinhibiting the challenger's thalamus and re-inhibiting the incumbent's); **Frank 2006** / **Aron & Poldrack 2006** (hyperdirect/lateral inhibition, inherited from the STN chain); **Lundqvist/Lisman** (theta-nested gamma multi-item phase segregation, PMC5039104, the alternative next lever). External-literature check logged, lane C.

## The question and the gate (the parent's own next-lever 1)
The parent NO-GO proved a clean single-slot SWAP is a substrate catch-22 by INTRINSIC competition: any lever that lets B break in settles into a stable n=2 CO-IGNITION; any per-slot WTA strong enough for single-content selectivity LOCKS OUT B before it can co-activate (or destabilises solo-A). Its external-lit conclusion: biology does not arbitrate incumbent-vs-challenger by intrinsic rate competition over a shared resource; it uses an EXTERNAL, dedicated gate (PBWM) that DISINHIBITS the challenger's thalamic drive and INHIBITS the incumbent's slot. This runner builds that gate.

MECHANISM (per-slot k; all explicit wiring, dense frozen pools; NO `sim/` edit):
- workspace: K=3 DISJOINT recurrent patterns (w=34, supra-critical — a pattern self-sustains on recurrence alone, verified: no thalamic tonic needed) + a divisive-normalisation norm_pool + a uniform content-neutral workspace baseline tonic (replaces the base substrate's shared-thal support removed for per-slot relays; makes a marginal slot's recurrence robustly supra-critical across seeds — WITHOUT it seed-100 slot-0 is sub-critical and selectivity is UNDEFINED).
- per-slot BG-thalamus relay: thal_k (tonic drive, WANTS to relay) is held SILENT by a tonically-firing gpi_k (gpi_k -> thal_k inhibitory). A striatal-Go str_k, when driven, INHIBITS gpi_k -> DISINHIBITS thal_k -> thal_k -> slot_k (releases the challenger's thalamic drive). The genuine disinhibition, not a direct thalamic-current cheat.
- per-slot eviction: gate_inh_k (inhibitory), when driven, INHIBITS slot_k's assembly (closes the incumbent slot).
- THE GATE SIGNAL is the MISMATCH comparator (the STN next-lever #3): i_gate = conflict_gain * volley * incumbent_rate * scale, where volley = a challenger afferent volley is present and incumbent_rate = the strongest held non-challenger slot. This is signal-driven from the substrate's afferent + cortical state; 0 at no volley (selectivity) or no incumbent (scramble). (A cortical-co-activation margin read DEADLOCKED: B cannot co-activate to create conflict before the gate admits it — the mismatch read breaks that bootstrap.)

## The result — the gate SOLVES the lockout horn, the co-ignition horn REMAINS (0/6 clean swap)
The per-seed values below are rounded from the cited 6-seed artifact (`research/findings/raw/_gnw_bg_thalamus_gate_6seed.json`, per_seed[].advance / .residual / .go_gate).
<!--derived-->

| seed | SELECTIVITY (unchallenged A) | ADVANCE: B rate locked-out -> admitted (disinhibition) | headline n_pre->n_post | evict-sweep n_post (gate_inh x1/x2/x4) | clean swap |
|---|---|---|---|---|---|
| 42  | hold n=1 | 0.0040 -> 0.3333 (admitted) | 1 -> 2 (co-ignition) | 2 / 2 / 2 | NO |
| 43  | hold n=1 | 0.0027 -> 0.3333 (admitted) | 1 -> 2 (co-ignition) | 2 / 2 / 2 | NO |
| 44  | hold n=1 | 0.0000 -> 0.3333 (admitted) | 1 -> 3 (co-ignition + a rebound slot) | 2 / 2 / 2 | NO |
| 100 | hold n=1 | 0.0087 -> 0.3333 (admitted) | 1 -> 2 (co-ignition) | 2 / 2 / 2 | NO |
| 101 | hold n=1 | 0.0013 -> 0.3333 (admitted) | 1 -> 2 (co-ignition) | 2 / 2 / 2 | NO |
| 102 | hold n=1 | 0.0033 -> 0.3333 (admitted) | 1 -> 2 (co-ignition) | 2 / 2 / 2 | NO |

**clean-swap = 0/6; SELECTIVITY = 6/6; disinhibition-admits-challenger = 6/6; incumbent-ever-evicted = 0/6.**

- **THE ADVANCE (the LOCKOUT horn is SOLVED).** The parent's lockout horn was: any inhibition strong enough for single-content selectivity suppresses the challenger BEFORE it can co-activate. The external gate defeats this. A challenger B driven at 5000 pA is suppressed to ~0.00-0.009 by the divisive normalisation (the lockout) when driven ALONE; the moment the gate's disinhibition releases thal_B (str_B -> silence gpi_B -> thal_B relays, measured thal_B rate ~0.40 during the gate), B breaks in to the full ignited plateau (0.333) on all 6 seeds. The WTA could not do this (it locked B out); the external gate ADMITS it. This is the qualitative difference the parent finding predicted, now demonstrated on the substrate.
- **THE RESIDUAL (the CO-IGNITION horn REMAINS).** With B admitted, the gate must now EVICT the incumbent A to reach n=1. On the supra-critical recurrence substrate A self-sustains INDEPENDENT of the thalamus, so the gate's eviction (gate_inh_k -> slot_k, a per-slot inhibition) hits the Rung-2c inhibition-resistance wall directly: in the committed sweep at gate_inh x1/x2/x4 A is NEVER evicted (evict-sweep n_post stays 2 on all 6), and wider scratch probing (higher weight and drive, during both confident-A and genuine co-ignition) reproduced only the two Rung-2c failure modes — weak inhibition leaves A holding, strong inhibition destabilises A UPWARD or rebound-ignites a quiescent slot. n settles at 2 (co-ignition), never 1.

The gate thus CONVERTS the catch-22: it removes the lockout horn (B is admitted by an external arbiter, not fought for) and isolates the residual to a SINGLE, precisely-characterised difficulty — evicting a thalamus-independent (supra-critical) incumbent.

## Anti-cheats (each held on all 6 — the NO-GO is real, not an instrument failure)
- **SUBSTRATE-DRIVEN, no host poke:** the headline is a CONTINUOUS run — `host_workspace_reset_calls == 0` and `host_content_swap_calls == 0` on all 6 (the only host writes are the external stimulus drive + the content-neutral baseline tonics = world/body-legitimate).
- **SIGNAL-DRIVEN gate:** i_gate is 0 at zero afferent volley and rises when a volley arrives on a held incumbent (measured `i_gate_by_amp` = [0, ~3200, ...] across a challenger-drive sweep) — `gate_signal_driven` 6/6. A margin-SCRAMBLE (feed "no incumbent held" = 0 to the mismatch read) keeps the gate silent (0 fired steps) even though a content IS held — `scramble_breaks_swap` 6/6 — proving the cortical incumbent read gates the current.
- **DISINHIBITION load-bearing for the break-in (before/after):** B is LOCKED OUT (rate ~0.00) without the disinhibition and ADMITTED (0.333) with it, all 6 — the disinhibition arm is causally responsible for admitting the challenger.
- **BYTE-IDENTICAL base substrate:** the gate-OFF build (the active-overwrite base) hash EQUALS the DIST-OVERWRITE base build hash at the same seed, all 6 — the gate additions are additive wiring, not a different random draw.
- **DETERMINISM:** build the gate-ON substrate twice at one seed -> identical seed-derived Izhikevich-param hash (heterogeneity seeded from `cfg.seed`, NOT `actual_seed_used`), all 6.
- The **gate_inh load-bearing** causal check for the SWAP is reported N/A (`inh_load_bearing` = False): there is no clean swap to attribute, so a lesion cannot abolish one. This is stated, not hidden — a load-bearing claim without an achieved swap would be an overclaim (the parent's discipline).

## Why the co-ignition horn is a genuine boundary here, and the next lever (named + prototyped)
The eviction cannot use brute cortical inhibition (Rung-2c: a frozen supra-critical attractor is inhibition-resistant — hold, or destabilise/rebound). The PBWM gate evicts NOT by inhibiting the cortex but by CLOSING the incumbent's thalamocortical LOOP (withdrawing thalamic support) — which only works if maintenance DEPENDS on that loop (sub-critical recurrence). On our supra-critical substrate the incumbent needs no thalamus, so loop-closure has nothing to withdraw. This is the exact mismatch that leaves the co-ignition horn open.

**Next lever 1 — eviction-by-loop-closure on a THALAMUS-DEPENDENT substrate (prototyped, in scratch):** a sub-critical recurrence + a disinhibition-maintained thalamocortical loop (slot -> str -> silence gpi -> thal -> slot) + the norm brake DOES maintain a single content (A held, others down, on seeds 42/43/44).
The residual is robustness: (a) the loop sits near the criticality knee (OU-noise-dependent survival) so the sub-/supra-critical margin must be widened, and (b) the eviction (drive gpi_incumbent, or silence str_incumbent) does not yet overpower the incumbent's OWN cortico-striatal Go once it is at full plateau — the self-Go re-opens the loop.
Concrete next steps: widen the criticality margin; make the eviction target the incumbent's Go with the norm-inhibition-from-B assisting; sequence the gate (evict-then-admit) rather than concurrent. This is a tuning residual, not a mechanism gap.

**Next lever 2 — theta-phase segregation (Lundqvist/Lisman):** occupy the challenger and incumbent on DIFFERENT theta phases so a challenger never needs to evict an incumbent from the same instantaneous resource — removing the eviction requirement entirely. Buildable as a phasic drive/inhibition schedule on the existing pools.

## Do-NOT-retread (banked)
On the supra-critical distributed workspace, the per-slot BG-thalamus gate:
(a) SOLVES the lockout horn — the striatal-Go disinhibition robustly ADMITS the challenger past the divisive-normalisation lockout (B ~0.00 locked-out -> 0.333 admitted, 6/6); the WTA could not (it locked B out).
(b) Does NOT solve the co-ignition horn — the gate's per-slot inhibition CANNOT evict the thalamus-independent (supra-critical) incumbent (Rung-2c inhibition-resistance: hold, or destabilise/rebound), swept x1/x2/x4 and beyond; n settles at 2 on 0/6.
(c) SELECTIVITY is 6/6 and all substrate anti-cheats pass, so this is a mapped substrate property.
**BANKED NEGATIVE.** The clean n-stays-1 SWAP remains a NO-GO on this substrate; the eviction requires thalamus-DEPENDENT maintenance (sub-critical recurrence) so the gate can evict by loop-closure, or theta-phase segregation to avoid eviction — the two named next levers.

## Remaining scaffolds (named, not claimed closed)
1. The K patterns + norm_pool + per-slot thal/gpi/str/gate_inh are hand-wired dense frozen pools, not self-organised.
2. The gate current is a host-computed conflict read (the mismatch comparator) injected into the striatal-Go / eviction pools (the reused STN-sensor pattern); the disinhibition/inhibition then play out ON the substrate. Native STP/homeostasis stay OFF (frozen-weight foot-guns).
3. Content and drive timing are host-supplied external drive (world/body-legitimate as stimuli); the workspace baseline tonic is a content-neutral uniform bias.
4. This is a de-risk, not wired to production — the distributed workspace + gate path is not reachable from `/api/brain-chat`.

## Files
Runner: `research/runners/_gnw_bg_thalamus_gate_overwrite_derisk.py`. 6-seed artifact: `research/findings/raw/_gnw_bg_thalamus_gate_6seed.json` (+ `.prov.json` sidecar). Reproduce (single invocation, serial; the committed artifact was merged from two parallel 3-seed groups but this yields the same result): `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 SIM_BACKEND=numpy python -u -m research.runners._gnw_bg_thalamus_gate_overwrite_derisk --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_bg_thalamus_gate_6seed.json`. Smoke: add `--smoke --seed 42 --gate-inh-grid 14 40`.
