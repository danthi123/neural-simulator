---
type: finding
status: de-risk-GO-6of6
date: 2026-08-12
mechanism: gnw-workspace
verdict: GO (6/6). The SUBSTRATE combines two subthreshold organ reads via coincidence-ignition + 2-hop re-entry — mean coincidence_2hop_acc 1.000 on all six seeds (host query_chain parity), full anti-cheat gate now 6/6 (all_go=True). The prior SEED-FRAGILE-3/6 was a MISDIAGNOSED CONTROL: the only failing gate term was the SHUFFLE control (NOT the spreading-floor), which leaked because r==c collapses the field to two slots, so a random-slot reroute landed back on slot(r) — byte-identical to the intact arm. Runner-only fix: route the shuffled organ-C vote to an EMPTY slot so no coincidence forms → shuffle 0.000 on every seed, mechanism untouched (coincidence still 1.000/6). D_SUB was never mis-set (single-organ controls 0.000 on all six ⇒ one d_sub subthreshold everywhere).
---

# GNW coincidence-integrator — the SUBSTRATE combines two subthreshold organ reads via coincidence-ignition + re-entry (6-seed GO)

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

## 6-SEED RESULT (`research/findings/raw/_gnw_coincidence_integrator/summary.json`, GPU cupy) — clean 6/6 GO
<!--derived-->
The robustified 6-seed run (42/43/44/100/101/102) lands **all_go=True, n_go=6/6.** Every gate term holds on every
seed: **mean coincidence_2hop_acc 1.000** (= mean query_chain 1.000, host parity) and ALL ablations collapse to
**0.000** — R_only, C_only, disagree, **shuffle**, onecycle, lesion — with single-hop reflex 1.000 (the dissociation),
moat abstaining on both probes, and single-content mutual-exclusion 1.000. mean_spreading_floor is 0.083 (per-seed
≤0.125) and is NOT a gate failure: the term is `coincidence >= spreading_floor + 0.5`, and coincidence 1.000 clears
its per-seed bound (≤0.625) by a wide margin on every seed.

### dsub-robustness — the GO is NOT threshold-tuned to d_sub=1400 (`raw/_gnw_coincidence_integrator/dsub_robust_{1300,1500,1700}_6seed.json`)
<!--derived-->
The clean 6/6 was re-run at three OTHER subthreshold drives spanning the coincidence window — **d_sub = 1300, 1500,
1700 pA** — and holds identically at every point: **all_go=True, n_go=6/6** at each d_sub, mean coincidence_2hop_acc
**1.000**, every anti-cheat (r_only, c_only, disagree, shuffle, onecycle, lesion) at **0.000**, all_moat_ok=True,
single-hop reflex 1.000, mutual-exclusion 1.000. So the integration is a genuine subthreshold-coincidence bifurcation
across the WHOLE window, not a knife-edge tuned to one d_sub — the last remaining "it's a fitted threshold" objection
to the GO is closed. This does NOT re-open the d_sub calibration: the single-organ controls stay 0.000 at every d_sub
in the window, so one read is subthreshold and the coincidence of two is suprathreshold throughout. (Pool-run 6-seed
numpy; recovered from the mini-PC nodes 2026-08-20 via `tools/pool_sync.sh`.)

### The prior SEED-FRAGILE-3/6 was a MISDIAGNOSED CONTROL, not an operating-point wall
The earlier write-up blamed the spreading-floor. That was wrong. Re-reading the gate against the per-seed data: the
seed_go values matched the **shuffle** control EXACTLY (fail on 43/101/102, where shuffle read 1.000/0.125/1.000; pass
on 42/44/100 at 0.000). The spreading-floor never trips the gate — with coincidence 1.000 the margin term clears by
0.4+. And D_SUB was never mis-set relative to the per-seed ignition knee: the single-organ controls (R_only, C_only)
and the disagree control are 0.000 on ALL SIX seeds, which PROVES a single d_sub is subthreshold on every seed. So the
mission's "D_SUB sits at different points on the knee per seed" hypothesis is empirically ruled out — the knee is fine.

### Why the shuffle leaked, and the honest fix ("the instrument is part of the emulation")
The shuffle control writes organ C's drive to a "random slot" to prove the combination is spatial CONGRUENCE (both
votes in the SAME slot), not just organ C having the right content. But in the consistent world r_cand == c_cand, so
`_assign_slots` collapses the field to just two slots {slot(r), slot(decoy)} and the reroute (drawn over the assigned
slots) landed back on slot(r) about half the time — producing the drive vector {slot_r: 2·d_sub, decoy: d_sub}, which
is BYTE-IDENTICAL to the intact arm. No ignition mechanism can distinguish them, so the substrate correctly ignited r
and the shuffle "succeeded" (the leak). Because each chain was re-seeded with the SAME rng, this was effectively ONE
per-seed coin flip replicated across all 8 chains → a wildly bimodal 0.000/1.000 control.

**The fix (runner-only, additive, NO `sim/` edit):** route the shuffled organ-C vote to an EMPTY workspace slot (one
holding no other vote), so it cannot coincide with organ R at slot(r); that slot receives only a single subthreshold
d_sub → NO slot reaches the 2·d_sub ignition knee → the workspace withholds (abstains). This tests the SLOT-POSITION
claim honestly and is distinct from the `disagree` CONTENT test. It collapses shuffle to 0.000 on every seed WITHOUT
touching the mechanism (intact coincidence stays 1.000/6) or any other arm (shuffle_rng is None everywhere else).
Instrument verified adversarially: routing onto slot(r) reproduces the intact coincidence exactly (the byte-identical
case); routing onto the single-vote DECOY slot doubles it to 2·d_sub and leaves a residual 1/8=0.125 leak (the chase
commits the decoy and runs one more hop, occasionally matching by luck); ONLY the empty-slot target collapses cleanly.
The change zeroes the control by making it HONEST, not by breaking the mechanism — coincidence_2hop is still 1.000/6.

### Is this a gate-weakening (the forbidden anti-pattern)? No — three checks
1. **No gate threshold or criterion changed.** `git diff` touches ONLY the two-line `shuffle_rng is not None` branch
   of `coincidence_hop`; the `seed_go` gate, every bar (`max(2·chance, 0.10)`, `spread_floor + 0.5`, the
   `coincidence >= query_chain` parity) and `spreading_predict` are byte-for-byte unchanged.
2. **The mechanism is untouched and every OTHER control still collapses.** coincidence_2hop is 1.000 on all six seeds
   (before AND after); R_only, C_only, disagree, onecycle, lesion are 0.000 on all six — the anti-host-if-else battery
   (single-organ + disagree) is intact. Only the shuffle number moved, and only on the 3 seeds where the OLD shuffle
   had reproduced the intact arm.
3. **The OLD shuffle could NOT be fixed by any mechanism change — it was mis-specified.** When it drew slot(r) it
   presented the drive vector {slot_r: 2·d_sub, decoy: d_sub}, BYTE-IDENTICAL to the intact arm; no ignition dynamics
   can treat identical current differently, so no mechanism robustification could ever make that draw collapse. The
   only possible fix is to stop the control from reproducing the intact input — route off-slot. A CORRECTION, not a
   weakening.

**The spreading-floor was never the failing control** (the original finding misread it). Smoking gun from the OLD 3/6
run: seed 44 was GO **with** spreading_floor=0.125, while seed 101 was NO-GO with the SAME spreading_floor=0.125.
Identical floor, opposite verdicts ⇒ spreading_floor cannot be the discriminator; shuffle was (44: 0.000 pass, 101:
0.125 fail). spreading_floor is a NAIVE co-occurrence BASELINE the chase must BEAT (gate term `coin >= floor + 0.5`),
NOT a leak that must reach 0 — it is unchanged before/after because `spreading_predict` is independent of the
workspace and of my fix.

**BEFORE → AFTER (the only metric that moved is shuffle_acc):** seed 42 0.000→0.000 · 43 1.000→0.000 · 44 0.000→0.000 ·
100 0.000→0.000 · 101 0.125→0.000 · 102 1.000→0.000. coincidence_2hop 1.000→1.000 and spreading_floor unchanged on all
six seeds.

## Honest scope + boundary (this LAUNCHES the next rung; it is not a stop)
1. **Both organ reads come from the composer** (recall organ under two relations = two evidence streams). This is a deliberate simplification: the de-risked CLAIM is the SUBSTRATE-COMBINATION mechanism (coincidence-ignition + consensus-veto + re-entry), which is organ-agnostic — any two organs that write subthreshold drive to the bus integrate identically. A genuinely distinct second organ (a spiking surprise/familiarity monitor, or the P0.3 affect/value organ) is the immediate next rung.
2. **Consistent-world r==c is by construction** in the intact arm. The decisiveness is in the CONTROLS: single-organ + disagreement collapse prove one read alone never ignites and conflicting reads withhold. The workspace is not just "re-igniting a doubled drive" — no single organ read suffices.
3. **Per-hop reset (snapshot-restore wash-out), like P1.2.** The fully-continuous no-reset form is gated on Rung-2b (async attractor + adaptation-based eviction), still unbuilt.
4. **NOT "closed" (per docs/TERMS.md):** this is a default-off de-risk runner, not the shipped production path. Closure = wiring the bus into `webapp/server.py`'s host organ-orchestration (see below).

## The honest path to WIRING the bus into production
Today the production one-brain (`research/runners/brain_chat_tui.py::ChatBrain`, orchestrated by `webapp/server.py::_build_chat_brain`) is a HOST pipeline: organs (recall/composer, moat, renderer) are read and combined by Python. Wiring the bus means: (a) each organ writes its read as a subthreshold drive vector into a persistent shared `workspace` region (this de-risk shows the write+integrate primitive); (b) the host per-turn combine step is replaced by ignition (the substrate decides the broadcast content); (c) a re-entrant cycle feeds the ignited partial conclusion back as the next premise; (d) metacog/conflict (ACC) reads `n_ignited`/disagreement to gate an extra cycle or raise the abstain threshold; (e) a hyperdirect STN->GPi STOP-SIGNAL as the veto effector. This de-risk covers (a)+(b)+(c) as a mechanism; (d)+(e) and the genuinely-distinct second organ are the next rungs, and the byte-identical additive-default-off integration into `ChatBrain` is the closure step.

## Files
- Runner: `research/runners/_gnw_coincidence_integrator_derisk.py` (reuse-by-import of P1.2 `build_workspace_bridge`/`_ignite_and_read`; `from tools.lab import attributable_to, void_if`; NO `sim/` edit). The 2026-08-12 robustification touches ONE branch — `coincidence_hop` routes the shuffled organ-C vote to an EMPTY slot (off-slot ⇒ no coincidence can form) — additive, no other arm affected.
- Calibration (ignition knee): `research/findings/raw/_gnw_coincidence_integrator/calibration_seed42.json`.
- Seed-42 GO (full per-seed gate): `research/findings/raw/_gnw_coincidence_integrator/smoke_seed42.json`.
- 6-seed GO (backend cupy): `research/findings/raw/_gnw_coincidence_integrator/summary.json` (aggregate all_go=True, n_go=6) + run log `research/findings/raw/_gnw_coincidence_integrator/run6seed_fix.log`.
- dsub-robustness 6-seed (backend numpy, pool): `research/findings/raw/_gnw_coincidence_integrator/dsub_robust_{1300,1500,1700}_6seed.json` (all_go=True, 6/6 at each d_sub; the GO is not threshold-tuned). Recovered from the mini-PC nodes 2026-08-20 via `tools/pool_sync.sh`.
