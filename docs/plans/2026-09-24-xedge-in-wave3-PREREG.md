# Pre-registration: BRAIN_XEDGE_IN_WAVE3 — the d6 w{k}->sel cross-edge grown inside the production merged pool

**Written 2026-09-24 (~13:05 EDT), after the S02 identity probe and one build smoke, before any gate run below.**
Branch `research/xedge-wave3-reconcile`; the build commit is `58eaecd5c`. The gate constants in
`research/runners/_xedge_in_wave3_verify.py` are copied from §3 of this document. Any later change to a gate goes
in the amendment log (§6), with a timestamp and a list of every result artifact already seen at that time.

## 1. What was already observed before this document

- S02 identity probe, seed 42, pool1, main-equivalent revision `8e1bbc470`, shipped defaults
  (`research/findings/raw/_xedge_wave3_probe/s42.json`): CONFIRMED-SEVERED.
  - (a) `comprehension_production_organ.get_organ(42) is get_xedge_pool(42).comp_organ` = False.
  - (b) the per-session d6 organ's `shared=` is `get_merged_cortical_pool(42)` = False; it is the separate xedge pool.
  - `xedge_enabled()` = True, and the Wave-3 pool is default ON.
- One flag-ON build smoke, dev seed 7, pool1 (scratch; not committed as evidence).
  - The pool built with 7098 neurons (7002 Wave-3 + 96 da_credit).
  - The six candidate masks hold 480 synapses each.
  - One credited live turn (focus = p_agent = w2) moved `w2->A` from 0.05 to 0.4632. Every other candidate edge
    stayed at 0.05.
  - Peak RSS 0.36 GB. The holder build took ~87 s, one amb_read ~19 s and one credited turn ~23 s.
- No other-organ read, leak test or load-bearing selftest has been run on the flag-ON pool.

## 2. Claim under test

With `BRAIN_XEDGE_IN_WAVE3=1`, the learned d6 w{k}->sel cross-edge lives on the same spiking pool as:
- the live comprehension organ;
- every per-session d6 organ.

So a held WM referent can drive the production comprehension read through a real synapse, and the per-turn
credit can grow that synapse, without changing any other organ's reads.

The flag is DEFAULT OFF. With it unset, the code paths are those of main. The two routing points and the d6
`load()` hook each check the flag or an attribute that only the flag-ON pool carries.

## 3. Gates (dev seed 7 tonight; seeds 42/43/44/100/101/102 untouched until the orchestrator runs the battery)

All comparisons are EXACT (tolerance 0.0) unless stated otherwise.

**G0 — routing reconciled.** The S02 probe (`research.runners._xedge_wave3_identity_probe`) is run at seed 7 with the
flag ON. It must read RECONCILED, with BOTH (a) and (b) True. With the flag OFF on this branch, it must read
CONFIRMED-SEVERED, with (a) and (b) False. That matches main's booleans, so the flag-OFF routing is unchanged.

**G1 — other-organ read identity (the substrate-change test).**
- Instrument: `_onebrain_wave1_organread_verify._isolated_reads` over the 11 Wave-3 descriptors, the same instrument
  as the Wave-3 organ-read GO.
- Arm OFF: the plain `get_wave3_pool(7)`, in a fresh process.
- Arm ON: the flag-ON pool `build_wave3_xedge_pool(7)`, in a fresh process. It is read at two points:
  - **ON-build**: right after the pool, the xedge holder and the production comprehension organ are built.
  - **ON-exercised**: after `N_EXERCISE_TURNS = 6` credited live turns (alternating agent/patient discourse with the
    focus on p_agent / p_patient), 2 per-session d6 loads, and 2 comprehension `judge` + `repair_target` calls
    with a focus held.
- PASS: for each of the 9 organs other than the cross-edge endpoints (surprise, worldmodel, metacog, pragmatic,
  source_provenance, self_schema, curiosity, causal_whatif, prospective_memory), max |Δread| == 0.0 and the answer is
  equal, in BOTH ON states vs OFF.
- The endpoints (d6_multiref_wm, comprehension) are reported with the same statistics. Their isolated reads hold
  no focus, so they are EXPECTED to be identical too. A deviation there is reported as a named defect, but it is not
  the G1 verdict.
- **If G1 fails, A4 is a SUBSTRATE CHANGE.** Every pooled row must then be re-measured, and there is no flip tonight.

**G2 — interleaved two-session leak test.**
- Session A holds {wolf, dog}. Session B holds {cat, bird}.
- Each session runs the same script:
  1. a two-referent message;
  2. a hold-query;
  3. a comprehension `judge` and a `repair_target` on a content-ambiguous battery item, both with the session's own
     `current_focus()`;
  4. the hold-query again.
- Three fresh processes are run: A alone, B alone, and A/B interleaved turn by turn (A1 B1 A2 B2 ...).
- **G2-transient** (no credit steps; the gating arm):
  - Every per-turn output of B in the interleaved run must equal B-alone field for field, and the same for A.
    The compared fields are the d6 judge dict, the comprehension judge dict, the repair_target dict and the
    hold-query readout.
  - No output of B may name a referent of A, and the reverse.
  - The session guard must have fired: `n_session_resets > 0` in the interleaved run and == 0 in the alone runs.
- **G2-learning** (informational, not gating):
  - The same three runs, with `credit_live_turn_from_comprehension` after each comprehended turn.
  - The learned edge is process-shared BY DESIGN (one brain learns from every conversation), so an interleaved-
    vs-alone difference here is expected.
  - Reported: the per-field diffs and the cross-edge weight trajectories.

**G3 — load-bearing selftest, seed 7, pool2** (run by `_xedge_in_wave3_verify --selftest`).
- The flag-ON holder grows its edge with the in-pool PART-2 build curriculum (`grow_live_selfsupervised`, 80 turns,
  `set_live_per_turn(False)`). This is the same credit atom as the per-turn path, and the same protocol as the
  2026-08-27 live-learning 6-seed GO.
- Then `onebrain_xedge_production._selftest_loadbearing(holder, 7)` runs.
- PASS = its own criterion `lesion_attributable`: max |dNet| intact > 1e-3 AND max |dNet| after the lesion < 1e-9.
- Also reported, not gating:
  - the same selftest on the fresh (W0=0.05) flag-ON holder;
  - `_selftest_livelearn(holder, 7)` (decision flips intact vs lesioned), run after growing.

**Verdict wording.** One dev seed is a de-risk. The write-up says "de-risk (seed 7)", never GO. The 6-seed
battery belongs to the orchestrator; the job lines are staged in the lane report.

## 4. Declared host residuals (none added by this lane)

- The positional focus `CAND_POOLS[0]` and the candidate topology are carried from the shipped xedge.
- The host-timed 400 pA co-drive of the focus pool before the comprehension read is carried from the shipped xedge.
- The teach drive onto teach_{agent,patient} is a teacher/environment input (R3). WHICH discourse is presented is
  host scaffold. The credit VALUE and DIRECTION are read off the brain's own amb_read.
- The session-ownership token (which session last wrote the d6 slice) is host bookkeeping over the WORLD boundary:
  which conversation is talking. It computes nothing between sensation and action. It only triggers a hard reset
  of transient activity.

## 5. What this lane does not claim

- It does not claim the reply follows the held referent's CONTENT; the focus is positional.
- It makes no production-default claim. The flag stays OFF, and a flip needs the orchestrator's 6-seed battery,
  a review and the no-regression arm.
- It makes no claim about the affect-pool combination. `xedge_in_wave3_enabled()` is False while
  `BRAIN_ONEBRAIN_AFFECT_POOL=1`.

## 6. Amendment log

(none)
