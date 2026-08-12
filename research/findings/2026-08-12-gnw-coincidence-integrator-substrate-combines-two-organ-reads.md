---
type: finding
status: de-risk-GO-seed42-6seed-in-flight
date: 2026-08-12
mechanism: gnw-workspace
---

# GNW coincidence-integrator — the SUBSTRATE combines two subthreshold organ reads via coincidence-ignition + re-entry (seed-42 GO, clean; 6-seed in flight)

## The claim being de-risked (the T1-1 missing rung, NOT a re-derivation of P1.2)
The 2026-08-12 faculty-map audit names a WIRED Global-Neuronal-Workspace bus the highest-leverage next build, and calls the GNW stack "multiple de-risked GOs, unwired". The corpus already de-risks: spiking IGNITION (rung-1, 6-seed GO), MUTUAL EXCLUSION / single-content access (rung-2, 6-seed GO), BROADCAST + report==reasoning (rung-3/3b/3c/4), and RE-ENTRY over 3 hops (P1.2, 6-seed GO). What is NOT de-risked is the mission's exact load-bearing question: can the SUBSTRATE'S ignition dynamics COMBINE >=2 organ reads (an AND/consensus), rather than a host if/else?

**Why P1.2 does not already answer this.** P1.2's workspace only ever re-ignites ONE candidate: `query_patient` returns a single answer driven at IGNITE_PA (3.3x the random distractors), so a SINGLE organ read already crosses the ignition threshold alone. The workspace there is a RELAY for the composer's answer; the combining is host-side. This de-risk removes that: each organ writes SUBTHRESHOLD drive, so no single read can ignite — only the coincidence of two can. The ignition threshold IS the integrator.

## Mechanism (reuse of the P1.2/rung-1 spiking workspace; NO `sim/` edit)
Two organs write D_SUB=1400 pA (below the measured solo ignition knee ~2400 pA) into the shared K=4-slot workspace: organ R (RECALL, relation EAT) -> slot(r); organ C (CONFIRM, relation SEE over the same edges) -> slot(c); plus a single-vote DECOY at D_SUB. In a consistent world r==c, so that slot receives 2*D_SUB=2800 pA -> it alone crosses the knee, the shared inhibitory pool (`workspace_fs`) WTA-suppresses the decoy, the NMDA attractor sustains it, and the committed spiking winner BROADCASTS BACK as the next hop's premise. Two hops -> a 2-step conclusion ch[0]->ch[2]. The AND-over-organs is the neuronal threshold; the WTA is the shared inhibition — the substrate's dynamics, not host control flow.

## Instrument verified (calibration, seed 42 — `research/findings/raw/_gnw_coincidence_integrator/calibration_seed42.json`)
Solo-drive ignition curve: subthreshold (rate <0.05) up to 1800 pA, sharp knee at 2400 pA (rate 0.333 = the rung-1 period-3 limit-cycle plateau). D_SUB=1400 solo -> 0.027 (does NOT ignite); 2*D_SUB=2800 -> 0.333 (ignites). D_SUB sits cleanly in the coincidence window, so the single-organ collapse is a genuine subthreshold-vs-suprathreshold bifurcation, not a threshold-tuning artifact.

## Result (seed 42, GPU cupy, full per-seed gate + JSON; 6-seed in flight)
`coincidence_2hop_acc=1.000` = the host one-shot `query_chain=1.000` (parity — same conclusion, synaptic path), spreading floor 0.000. EVERY ablation of the synaptic mechanism collapses to 0.000: R-only, C-only (single organ subthreshold — THE anti-host-if-else), disagree (permuted CONFIRM -> consensus-veto, the workspace withholds), shuffle (organ C off-target), onecycle (single-shot reaches only hop-1), lesion (no ignition). The single-hop recall reflex survives the lesion at 1.000 (the dissociation keystone). Moat abstains on unstored cue + chain over-run. Single-content access (mutual exclusion) 1.000 at every committed hop. `attributable_to`: 100% of the 2-hop success needs BOTH organs (0% present in the best single-organ control).

## What the anti-cheats establish (each targets a distinct "it's really the substrate" claim)
- SINGLE-ORGAN COLLAPSE (R-only=0, C-only=0): a host `if organ_R: return r` would succeed; the collapse proves the combination is the workspace's ignition threshold, not a host read.
- CONSENSUS-VETO (disagree=0): conflicting organ reads fail to ignite -> the workspace refuses to broadcast an unconfirmed conclusion (a convergent-evidence gate, Dehaene-Changeux ignition needs convergent drive).
- RE-ENTRY (onecycle=0): the current PRODUCTION host pipeline (snapshot organs once, combine once, emit) cannot reach hop-2; the broadcast-back re-entry is load-bearing.
- IGNITION (lesion=0) + SHUFFLE=0: the sustained attractor and the congruence (not slot position) are both load-bearing.

## Honest scope + boundary (this LAUNCHES the next rung; it is not a stop)
1. **Both organ reads come from the composer** (recall organ under two relations = two evidence streams). This is a deliberate simplification: the de-risked CLAIM is the SUBSTRATE-COMBINATION mechanism (coincidence-ignition + consensus-veto + re-entry), which is organ-agnostic — any two organs that write subthreshold drive to the bus integrate identically. A genuinely distinct second organ (a spiking surprise/familiarity monitor, or the P0.3 affect/value organ) is the immediate next rung.
2. **Consistent-world r==c is by construction** in the intact arm. The decisiveness is in the CONTROLS: single-organ + disagreement collapse prove one read alone never ignites and conflicting reads withhold. The workspace is not just "re-igniting a doubled drive" — no single organ read suffices.
3. **Per-hop reset (snapshot-restore wash-out), like P1.2.** The fully-continuous no-reset form is gated on Rung-2b (async attractor + adaptation-based eviction), still unbuilt.
4. **NOT "closed" (per docs/TERMS.md):** this is a default-off de-risk runner, not the shipped production path. Closure = wiring the bus into `webapp/server.py`'s host organ-orchestration (see below).

## The honest path to WIRING the bus into production
Today the production one-brain (`research/runners/brain_chat_tui.py::ChatBrain`, orchestrated by `webapp/server.py::_build_chat_brain`) is a HOST pipeline: organs (recall/composer, moat, renderer) are read and combined by Python. Wiring the bus means: (a) each organ writes its read as a subthreshold drive vector into a persistent shared `workspace` region (this de-risk shows the write+integrate primitive); (b) the host per-turn combine step is replaced by ignition (the substrate decides the broadcast content); (c) a re-entrant cycle feeds the ignited partial conclusion back as the next premise; (d) metacog/conflict (ACC) reads `n_ignited`/disagreement to gate an extra cycle or raise the abstain threshold; (e) a hyperdirect STN->GPi STOP-SIGNAL as the veto effector. This de-risk covers (a)+(b)+(c) as a mechanism; (d)+(e) and the genuinely-distinct second organ are the next rungs, and the byte-identical additive-default-off integration into `ChatBrain` is the closure step.

## Files
- Runner: `research/runners/_gnw_coincidence_integrator_derisk.py` (reuse-by-import of P1.2 `build_workspace_bridge`/`_ignite_and_read`; `from tools.lab import attributable_to, void_if`; NO `sim/` edit).
- Calibration (ignition knee): `research/findings/raw/_gnw_coincidence_integrator/calibration_seed42.json`.
- Seed-42 GO (full per-seed gate): `research/findings/raw/_gnw_coincidence_integrator/smoke_seed42.json`.
- 6-seed (in flight, pid 834799, backend cupy): outputs land in the `research/findings/raw/_gnw_coincidence_integrator/` dir (the aggregate `summary.json` + its run log); this finding will be updated with the 6-seed verdict on completion.
