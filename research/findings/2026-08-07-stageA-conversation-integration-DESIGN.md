---
type: finding
status: contributing
date: 2026-08-07
mechanism: stageA-open-ended-conversation-integration
lane: E-language
---

# Stage A — the co-resident conversation-integration DESIGN (orchestrated scope + adversarial premortem, 2026-08-07)

Product of the `stage-a-conversation-integration-design` workflow (7 agents: 5 research-gate-first workstream
scopes → 1 integration architect → 1 adversarial skeptic). Builds on the faculty-state scoping
(`2026-08-07-open-ended-conversation-faculty-state-and-blockers-SCOPING.md`). The adversarial skeptic returned
**REVISE**; the corrections are folded in below (they change the first build). Read-only design; no build yet.

## The substrate decision (code-verified)
Unify on the MERGED nav+conv `SimulationBridge` hosting **`CoResidentOneBrainComposer`** (`nav_conv_merged_bridge.py`,
IS-A `OneBrainComposer`), driven via **`MergedNavConvAgent`** — the ONLY substrate that already carries the
co-resident-region-append harness (`co_resident_limbic/drive`, `internal_density=0`, per-region homeostasis,
appended-LAST byte-unchanged indices) AND the shared `neuromodulator_manager`, with a proven default-OFF
byte-identical wire-up precedent (the DA-salience gate, `2026-06-18-DA-salience-gate-production-wireup-GO`). The
brain's OWN faculties become co-resident slices; the 267M WKV LM is NEVER a slice (scaffold-quarantine below).

## The integration contract — 7 shared seams
1. **g_eff (abstention/speak margin)** — one scalar at the read ops. Composition LAW (fixed):
   `cue_match_moat (HARD floor) < honesty_floor < affect+DA modulation`. Affect/DA only modulate talkativeness on
   candidates that already cleared moat + honesty; neither ever touches the cue-match moat.
2. **One speak/silence WTA arbiter** — a single competitive-queuing 3-way {volunteer | ask | stay-silent}; affect-arousal,
   curiosity ask-drive, and answer-readiness all FEED it; one winner/turn. ⚠️ this arbiter does NOT exist yet → it is a BUILD.
3. **The certainty band {assert, hedge, soft_abstain, MOAT}** — WRITTEN by metacog-honesty (route the calibrated
   monitor, not a recall-score), READ by curiosity (low→ask), affect (colors tone WITHIN the band), generation
   (per-proposition after VERIFY). A HARD floor, not a tone knob.
4. **novelty_signal + group:ask pool** — curiosity's shared fields; every candidate probe washes-out/resets NM (DR-1 drift protocol).
5. **One shared DA broadcast** — all appraisal/reward mods (affect, teacher, curiosity) group-scoped `excitability_drive`,
   never `scope=all`; one summed reward bus (or the bistable affect latch saturates).
6. **store ↔ spiking-readout split** — facts live in the host VSA store (moat/VERIFY read it); a spiking-taught concept
   MUST register its spiking-readout confidence into the moat/honesty path, else it is invisible to VERIFY (FM6, deepest un-derisked).
7. **per-faculty RNG isolation** — every faculty gets a dedicated `np.random.default_rng(seed+offset)`; all read-only
   measurement forwards snapshot/restore global state (the teacher-loop seed-46 instrument bug, generalized to all five). PREREQUISITE.

## Conflict pre-mortem — the failure modes + guards
- **FM1 affect/free-gen bypass the moat** → cue-match is a hard floor affect never reaches; free-gen is a separate opt-in
  turn with post-hoc per-SVO VERIFY (a WEAKER placement → distinct label, never "moat GO").
- **FM2 free-gen breaks the honesty read** → honesty runs PER-PROPOSITION after the spiking re-parse; sub-clausal claims are a declared residual leak.
- **FM3 curiosity breaks turn-taking** → hedge + ask both feed the ONE arbiter; couple into one utterance or the WTA picks one winner.
- **FM4 affect inflates an assert past a low honesty band** → honesty floor applied AFTER affect as a hard floor on ASSERTION; anti-cheat: yoked high-arousal mis-colors tone but must NOT flip abstain→assert.
- **FM5 double-counted reward saturates the affect latch** → one shared summed reward bus (seam 5).
- **FM6 taught fact invisible to the moat** → taught items register spiking-readout confidence into the moat/honesty path.
- **FM7 shared-RNG cross-contamination** (deepest, prerequisite) → per-faculty RNG + snapshot/restore (seam 7).
- **FM8 frozen-monitor operating-point drift as the store grows** → pin the trace/feature schema; recalibrate per session.

## ⚠️ THE CRUX (adversarial correction) — the honesty floor is the WEAKEST link, not an assumed input
The whole stack composes UNDER the honesty floor, but that floor is currently **3/6 PARTIAL behavior** (the metacog
monitor's 6/6 is *discrimination*; the honesty wire-in routes a *recall-score*, not the calibrated monitor —
`2026-08-07-laneC-neural-abstain-hedge...PARTIAL`, `2026-08-02-laneC-...honesty-wirein-PARTIAL`). **Relabel it PARTIAL
everywhere; the Stage-A crux is LIFTING it to 6/6 CO-RESIDENT — lead with it, not with the comfortable affect win.**

## REVISED build order + first build
- **STEP 0 (prerequisite):** one shared substrate + RNG isolation; prove default-OFF byte-identity with a null co-resident slice.
- **STEP 1 (the crux — build FIRST):** the honesty floor — route the CALIBRATED metacog monitor (not the recall-score)
  through `meta_schema→self_schema→assert/veto`; the certainty band; the g_eff composition law; the shared 3-way speak/silence WTA arbiter (a BUILD). GO: default-off identity, hard-moat 475/475, honesty behavior measured on the real bridge (target 6/6, HONESTLY reported), lift the 3/6.
- **FIRST BUILD = STEP 0 + STEP 1 fused** (revised from the architect's Step0+Step2). Lead with the floor the whole stack rests on.
- **STEP 2 affect-coloring** — composes UNDER the real honesty floor; adds the FM4 anti-cheat (yoked high-arousal must not flip abstain→assert). Declare the host-fed-appraisal shortcut + the bistable-latch (binary, not graded) coloring as explicit honest-negatives.
- **STEP 3 curiosity wh-question emission** — a BUILD (the A→W question spell is a scope comment, not code; on-bridge LP memory is 1/6 fragile), feeding the shared arbiter; content words from the brain's OWN naming-map spike decode (NOT WKV).
- **STEP 4 grounded free-gen** — opt-in, scaffold-LABELED, weaker (post-hoc-verify) moat; per-proposition honesty.
- **STEP 5 interactive teacher-loop (P2.1)** — corrects the brain's OWN emergent producer (never distills WKV into a co-resident weight); requires the Step-0 RNG fix.

## Scaffold quarantine (owner steer, enforced)
The 267M WKV LM stays TEACHER/SCAFFOLD-only: never a co-resident slice; content words come from the brain's own
naming-map spike decode (WKV used only as a fixed articulatory alphabet); free-gen (Step 4) is the ONE place WKV
free-generation runs — default-off, distinct "post-hoc-verify" label, a ledgered burn-down item; the teacher-loop
corrects the brain's OWN producer and free-gen transcripts are distillation FUEL that GROWS it, then the scaffold retires.

## Open risks (declared)
The five faculties have never co-run on one substrate (the whole design is an untested integration hypothesis);
VRAM ceiling for 5 co-resident slices + the multi-turn buffer + e-prop; whether the aff_* append survives with no
`sim/` edit; frozen-monitor drift (FM8); fragile on-bridge LP memory (Step 3 may fall back to a host-TD tracker,
declared honest-negative); the store↔spiking-readout reconciliation (FM6) is the deepest un-derisked piece.
