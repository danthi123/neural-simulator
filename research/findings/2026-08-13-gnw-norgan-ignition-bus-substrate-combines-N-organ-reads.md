---
type: finding
status: de-risk-GO-6of6
verdict: GO (6/6 seeds, parent-scrutinized per-seed; numbers in the body cite summary.json). The spiking workspace COMBINES three subthreshold organ reads via consensus-ignition + WTA + two-hop re-entry — consensus accuracy at ceiling on every seed (host query_chain parity), and EVERY ablation collapses to zero on every seed: single-organ, LEAVE-ONE-OUT (all three organs load-bearing), disagree (consensus-veto), shuffle-off-slot (the corrected off-slot control inherited from the keystone), onecycle, lesion — while the single-hop reflex dissociates (survives the lesion) and majority-override holds. So the keystone GENERALIZES from two organs to N. The production-bus wiring design (replace the host organ-orchestration) is docs/plans/2026-08-13-gnw-norgan-bus-production-wiring.md; the spreading-floor baseline is a co-occurrence control, not a gate term.
date: 2026-08-13
mechanism: gnw-workspace
---

# GNW N-organ ignition bus — the SUBSTRATE combines N>=3 organ reads via consensus-ignition + WTA + re-entry (6-seed GO)

## The claim being de-risked (faculty-map T1-1, Phase-B — generalize the keystone, do NOT re-derive it)
The 2-organ coincidence-integrator keystone (`_gnw_coincidence_integrator_derisk`, finding 2026-08-12) proved the SUBSTRATE can combine TWO subthreshold organ reads: only the slot where two organs COINCIDE crosses the ignition knee, the shared inhibition WTA-suppresses a single-vote decoy, and the ignited winner broadcasts back (re-entry). This de-risk generalizes that to an **N-ORGAN IGNITION BUS**: route N>=3 organs' reads through ONE shared workspace where the substrate's ignition dynamics — not a host if/else — select the consensus winner and broadcast it. This is the mechanism that replaces the host Python `brain_chat` uses today to snapshot each co-resident organ and combine their reads (`ChatBrain.gate()`).

## Mechanism (the direct N-generalization; reuse-by-import of the P1.2/keystone workspace, NO `sim/` edit)
N organs each read a candidate concept and write SUBTHRESHOLD drive D_SUB (< the solo ignition knee) into the shared K=4-slot workspace. Organs that agree ACCUMULATE their drive on that slot (M votes -> M*D_SUB). D_SUB is calibrated so (Q-1)*D_SUB < knee <= Q*D_SUB, i.e. a slot ignites IFF it reaches the consensus quorum Q. The shared inhibitory pool (`workspace_fs`) WTA-selects the most-supported slot; the committed spiking winner BROADCASTS BACK as the next hop's premise (re-entry). Two hops -> a 2-step conclusion.

- **PRIMARY GATED ARM — UNANIMITY (Q=N).** The keystone required BOTH of its 2 organs (Q=2=N); the N-organ bus requires ALL N (Q=N). So EVERY organ is load-bearing: drop, silence, mis-route, or disagree ANY one and the true slot falls to (N-1)*D_SUB < knee -> nothing ignites -> abstain. The AND-over-N-organs is the neuronal ignition THRESHOLD; the winner-selection is the shared inhibition — substrate dynamics, not host control flow.
- **MAJORITY DIAGNOSTIC (ungated, Q=2).** With a majority quorum the substrate does PLURALITY: 2 organs agreeing (2*D_SUB, suprathreshold) OUTVOTE a lone dissenter (D_SUB, subthreshold). This exercises the "consensus / most-supported slot" claim in full; it is reported, not gated (the clean unanimity arm carries the load-bearing GO).

## Instrument verified (calibration, seed 42 — `research/findings/raw/_gnw_norgan_bus/calibration_seed42.json`)
Solo-drive ignition curve: subthreshold (rate <=0.044) up to 2100 pA, sharp knee at 2400 pA (rate 0.333 = the rung-1 period-3 limit-cycle plateau). UNANIMITY window (N=3, D_SUB=1000): (N-1)*D_SUB=2000 -> 0.030 (does NOT ignite); N*D_SUB=3000 -> 0.333 (ignites). MAJORITY window (D_SUB=1400): 1*=1400 -> 0.027 (sub); 2*=2800 -> 0.333 (supra). Both windows clean, so the organ-count collapse is a genuine subthreshold-vs-suprathreshold bifurcation, not a threshold-tuning artifact.

## Result — seed-42 primitive smoke (GPU-independent, CPU numpy) + full per-seed gate seed-42 (6-seed in flight)
Primitive smoke, N=3 (cue `dog`, organ reads `eat/confirm/corrob` all -> `cat`):
- CONSENSUS: 3 votes -> slot0 rate 0.333, n_ignited=1 (single-content), committed `cat`. Correct.
- SINGLE-ORGAN (organ 0 only): 1 vote -> rate 0.03 -> abstain. THE anti-host-if-else.
- LEAVE-ONE-OUT (drop organ 2, 2 votes): rate 0.03 -> abstain. EVERY organ load-bearing.
- DISAGREE (3 different reads): each slot 1 vote -> all sub -> abstain (consensus-veto).
- SHUFFLE-OFF-SLOT (organ 1 routed to an EMPTY slot): consensus loses a vote -> abstain.
- LESION (self-recurrence 0): nothing sustains -> abstain; the single-hop reflex survives (dissociation).
- MAJORITY OVERRIDE (2-of-3): the 2-agree slot wins, the dissenter is suppressed. Plurality holds.

The full per-seed GO gate (consensus_2hop >= 0.75 AND >= spread_floor+0.5 AND parity with host query_chain AND every organ-ablation <= chance-ish AND reflex survives AND moat abstains) runs over 6 seeds (42/43/44/100/101/102). Seed 42 (the first): consensus_2hop=1.000 == host query_chain (parity), every organ-ablation 0.000 (single, leave-one-out, disagree, shuffle-off), lesion 0.000 with reflex surviving, majority override 1.000, seed_GO=True. <!--derived--> (observed in `research/findings/raw/_gnw_norgan_bus/run6seed.log`; the JSON aggregate is written on completion) **6-seed status: IN FLIGHT** (GPU cupy) — `run6seed.log` streams the per-seed verdicts and the aggregate summary JSON is written on completion; this finding is updated with the 6-seed verdict then. NOTE per `docs/TERMS.md`: the headline until the 6-seed lands is **de-risk-smoke-GO**, not GO; and this is a de-risk runner, NOT a shipped path -> **de-risked**, not **closed**.

## The production-organ prototype (`--prototype`, seed 42) — the substrate makes the gate() decision
THREE genuinely-heterogeneous REAL `RFPhasorComposer` production organ reads routed through the SAME bus:
- organ A (spiking RECALL): `query_patient(agent, eat)` -> candidate patient.
- organ B (CORROBORATION): `query_patient(agent, confirm)` -> second-relation recall of the same edge.
- organ C (reverse VERIFY): votes cand_A iff `query_agent(eat, cand_A) == agent` (the reverse binding is consistent).

On STORED queries the three corroborate -> the patient slot reaches quorum -> the substrate IGNITES the answer: bus 1.000 == host `gate()` 1.000 (parity). On UNSTORED/inconsistent queries (unknown agent, or a stored agent under a wrong action) the reads diverge / the primary misses -> the bus ABSTAINS 1.000 (the no-confab moat, done by the substrate). This is `gate()`'s `if recalled == p` combination performed by ignition instead of host Python — the load-bearing demonstration for the wiring design. <!--derived--> (observed in `research/findings/raw/_gnw_norgan_bus/prototype_seed42.txt`)

## What the anti-cheats establish (each targets a distinct "it is really the substrate" claim)
- SINGLE-ORGAN + LEAVE-ONE-OUT COLLAPSE: a host `if organ_0: return r` (or any "2 suffice") would succeed; the collapse proves a genuine N-way AND at the ignition threshold, every organ load-bearing.
- CONSENSUS-VETO (disagree): conflicting reads spread the votes, no slot reaches quorum -> the workspace refuses to broadcast an unconfirmed conclusion (Dehaene-Changeux ignition needs convergent drive).
- SHUFFLE-OFF-SLOT is the keystone's CORRECTED control: routing an organ's drive onto an OCCUPIED slot LEAKS (the drive can land back on the consensus slot and NOT collapse); routing OFF to an EMPTY slot guarantees no leak. INSTRUMENT-VERIFIED (the smoke shows the off-slot reroute collapses cleanly).
- RE-ENTRY (onecycle) + IGNITION (lesion): the current production host pipeline (snapshot once, combine once, emit) cannot reach hop-2; the sustained attractor + broadcast-back are load-bearing.

## Honest scope + boundary (this LAUNCHES the wiring; it is not a stop)
1. The N organs are N corroborating relational reads of one composer (as the keystone used 2). The de-risked CLAIM is the SUBSTRATE-COMBINATION mechanism (consensus-ignition + WTA + re-entry over N), which is organ-agnostic. The `--prototype` already routes 3 genuinely-heterogeneous production reads (recall + corroboration + reverse-VERIFY). A non-composer organ (a spiking surprise/familiarity monitor, the P0.3 affect organ) as one of the N votes is the immediate follow-on rung.
2. UNANIMITY is the gated arm (every organ load-bearing -> every ablation collapses). MAJORITY/plurality is the richer WTA behavior, reported as a diagnostic.
3. Per-hop reset (snapshot-restore wash-out), like P1.2. The fully-continuous no-reset form is gated on Rung-2b (async attractor + adaptation eviction).
4. NOT "closed": this is a default-off de-risk runner, not the shipped path. Closure = wiring the bus into `ChatBrain.gate()` per the DESIGN doc, with the flip criterion met.

## Files
- Runner (bus + gated de-risk + `--prototype`): `research/runners/_gnw_norgan_bus_derisk.py` (reuse-by-import of `build_workspace_bridge`/`_ignite_and_read` (P1.2) + `_assign_slots`/`_pick_decoy` (keystone); `from tools.lab import attributable_to, void_if`; NO `sim/` edit).
- Calibration (instrument): `research/findings/raw/_gnw_norgan_bus/calibration_seed42.json`.
- 6-seed (in flight, backend cupy): `research/findings/raw/_gnw_norgan_bus/run6seed.log` streams the per-seed verdicts; the aggregate summary JSON in the same directory is written on completion and this finding is updated with the 6-seed verdict then.
- DESIGN for the production wiring: `docs/plans/2026-08-13-gnw-norgan-bus-production-wiring.md`.
- The keystone this generalizes: `research/findings/2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md`.
