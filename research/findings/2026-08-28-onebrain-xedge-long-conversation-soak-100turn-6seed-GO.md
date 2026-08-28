---
type: finding
status: live
date: 2026-08-28
mechanism: onebrain-xedge-per-turn-live-plasticity-long-soak
lane: onebrain-integration
board: 177-adjacent
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_long_soak_6seed.json
runner: research/runners/onebrain_xedge_long_conversation_soak.py
builds_on: research/findings/2026-08-27-onebrain-xedge-per-turn-live-plasticity-GO.md
builds_on2: research/findings/2026-08-28-onebrain-xedge-production-default-flipped-ON-6seed-GO.md
---

# One-brain xedge PRODUCTION-DEFAULT-ON safety soak: the per-turn cross-edge stays bounded, non-degrading, and
sustained-load-bearing across a 100-turn conversation, on 6/6 seeds — STRENGTHENS the fe1911f2 flip (GO, not a
revert signal)

**One-line:** `BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN` were flipped PRODUCTION-DEFAULT-ON on
2026-08-28 (commit `fe1911f2`), making the d6-WM->comprehension cross-edge grow ONE in-brain self-supervised
credited step PER REAL CHAT TURN. That flip's own verify protocol (`2026-08-27-onebrain-xedge-per-turn-live-
plasticity-GO.md`) was 24 turns / 3 seeds — the finding named its own residual explicitly: "production
conversations are longer than the verify protocol." This soak extends that protocol to 100 turns / 6 seeds and
adds four checks the 24-turn verify never ran (full-trajectory boundedness, early-vs-late comprehension
quality, sustained load-bearing at turn 60 AND the final turn, and a teach-then-distract catastrophic-
interference probe). Result: **GO on 6/6 seeds** — the flip is SAFE over a conversation ~4x longer than its own
verify protocol, on every check run. Artifact:
`research/findings/raw/_onebrain_xedge_long_soak_6seed.json`.

## VERIFY-FIRST (mandatory, run before building)

`git log --all --oneline | grep -iE "xedge|per-turn|live-plasticity|soak"` and
`.venv-rag/bin/python tools/rag/rag_search.py "xedge per-turn live plasticity long conversation soak stability"
5 --corpus finding` + `bash tools/before_you_build.sh "..."` were run first (the corpus-check log recorded the
query, stamped `corpus_check_fresh: true` into this soak's provenance sidecar). Confirmed: no >=50-turn soak
existed. The only prior per-turn protocol
is the cited 24-turn/3-seed PART-3 finding, and its own "what this advances" section names the exact residual
this soak closes: "soak it over longer real conversations (the per-turn growth is bounded by `stdp_w_max`, but
production conversations are longer than the verify protocol)."

## The soak design

`research/runners/onebrain_xedge_long_conversation_soak.py`, `_soak_seed`, reuses the PART-3 credit atom
UNCHANGED (`onebrain_xedge_production.XedgeProductionPool.credit_live_turn` -> `_credit_turn_step`: read the
brain's OWN frozen `amb_read` resolution, and IFF confident drive `teach_{resolved}` for one DA-gated credited
episode, gate opened for exactly that step then re-frozen) — no new learning rule, no host label. Per seed, FOUR
100-turn sessions on the SAME per-turn-live pool construction PART 3 verified:

1. **agent_session** — teach the AGENT role (present agent-leaning cues) on the focus pool (`w0`) for all 100
   turns.
2. **patient_session** — teach the PATIENT role on `w0` for all 100 turns (the opposite-direction control).
3. **lesion_session** — the IDENTICAL agent-teaching credit path with the plasticity gate left FROZEN (no
   weight can accumulate) — the load-bearing lesion PART 3 also ran, now for the full 100 turns.
4. **interference_session** (NEW) — teach the AGENT role on `w0` for `TEACH_TURNS=30` turns (empirically past
   the WM-resolved-read establishment threshold measured on a smoke run: 0/5 resolved at turn 10, 5/5 by turn
   15-24), THEN spend turns 31-100 (70 turns) on UNRELATED per-turn credited activity that holds a DIFFERENT
   candidate pool (`w1`, alternating agent/patient content) — `w0` receives no further direct teaching — while
   re-probing `w0`'s taught edge/read throughout, to ask whether intervening unrelated chat activity erodes an
   earlier turn's teaching (a stability-plasticity / catastrophic-interference test PART 3 never ran).

Checkpointed probes (never credit — pure frozen reads) at turns {10, 30, 60, 100}: (a) the xedge-focus
content-cancelled WM-resolved balanced margin (the EXACT quantity PART 3's `_wm_resolved_role` thresholds) plus
the decision-level resolved-role rate through the REAL `repair_target` path on 5 held-out ambiguous items; (b)
a GENERAL comprehension check via `corg.judge(...).comprehended` on 5 WELL-FORMED (content-decisive,
WM-independent) items — a control that should stay flat/high regardless of what the cross-edge does, catching
the failure mode of per-turn plasticity poisoning comprehension broadly, not just the taught item. An
`attributable_to` call (from `tools.lab`, matching the codebase's own attribution discipline) forces the
taught-vs-lesion weight-growth subtraction rather than just reporting both arms.

CPU/numpy (`SIM_BACKEND=numpy`), seeded via `cfg.seed` (the same R3Pool-seeded-construction path PART 2/3
verified), 6 seeds run SEQUENTIALLY in one process (memory-budgeted per the task's own instruction — peak RSS
observed ~155MB, `free -m` never dropped below ~23GB available). Total wall time ~50 minutes for all 6 seeds.

## Results

**1. Boundedness (full trajectory, all 4 sessions, all 6 seeds).** Global min/max across every recorded weight
(all 4 sessions x 100 turns x 6 seeds) = **0.05 / 16.1886** — well under `stdp_w_max`=20 (F3), no runaway, no
collapse. Per-seed agent/patient/lesion final weights:

| seed | agent w0->A (start->final, min/max) | patient w0->P (start->final) | lesion w0->A (frozen gate) | frac attributable to plasticity (vs lesion) |
|---|---|---|---|---|
| 42  | 0.05 -> 15.6403 (min 0.05, max 15.6403) | 0.05 -> 15.6310 | 0.05 -> 0.05 (unchanged) | 100.0% |
| 43  | 0.05 -> 16.1886 (min 0.05, max 16.1886) | 0.05 -> 16.1886 | 0.05 -> 0.05 (unchanged) | 100.0% |
| 44  | 0.05 -> 15.4831 (min 0.05, max 15.4831) | 0.05 -> 15.4829 | 0.05 -> 0.05 (unchanged) | 100.0% |
| 100 | 0.05 -> 16.0319 (min 0.05, max 16.0319) | 0.05 -> 16.0318 | 0.05 -> 0.05 (unchanged) | 100.0% |
| 101 | 0.05 -> 16.0734 (min 0.05, max 16.0734) | 0.05 -> 16.0734 | 0.05 -> 0.05 (unchanged) | 100.0% |
| 102 | 0.05 -> 15.9895 (min 0.05, max 15.9895) | 0.05 -> 15.9890 | 0.05 -> 0.05 (unchanged) | 100.0% |

Every seed's weight trajectory is monotone, decelerating toward the soft bound (the same shape as PART-3's
24-turn curve, just carried ~4x further) — no seed shows a late-conversation runaway or a collapse back toward
W0. The lesion arm (identical credit path, gate frozen) stays at EXACTLY 0.05 on all 6 seeds, so
`attributable_to` reads 100.0% of the weight growth as owned by the per-turn plasticity, 0% present in the
frozen-gate control, on every seed.

**2. No-degradation-over-turns (early vs late, within the SAME 100-turn session).** Two references were used
deliberately: turn=10 for the GENERAL (WM-independent) comprehension check (no establishment threshold
applies), and turn=30 (`TEACH_TURNS`, past the empirically measured establishment point) for the xedge-focus
quality check — comparing pre-establishment noise to late quality would have been meaningless. Both held on
all 6 seeds:

- **xedge-focus quality**: the WM-resolved balanced margin is established (|margin|>eps, >=3/5 resolved) by
  turn 30 on every seed for BOTH the agent-taught and patient-taught sessions, and REMAINS established (same
  sign, still >=3/5 resolved) at the final turn — no seed shows a late-conversation collapse. Decision-level
  read rate: **5/5 ambiguous items resolved at the final turn, on BOTH the agent-taught and patient-taught
  session, on ALL 6 seeds** (e.g. seed 42: agent margin -0.0003(t10, unestablished) -> +0.0107(t30) ->
  +0.0092(t100); patient margin -0.0012(t10) -> -0.0118(t30) -> -0.0074(t100) — opposite-signed throughout,
  never crossing).
- **general (WM-independent) comprehension**: `comprehended` rate on 5 well-formed, content-decisive items
  stayed EXACTLY 3/5 -> 3/5 (early vs late) on every seed of every session — flat, no drop. The per-turn
  cross-edge plasticity does not spill over into degrading comprehension on items that never depend on the WM
  focus at all.

**3. Sustained load-bearing at turn 60 AND the final turn (not just turn 24).** On ALL 6 seeds: the
content-cancelled WM-resolved margin at the turn-60 checkpoint is positive (agent-taught) / negative
(patient-taught) past eps, AND the same holds at the final turn (100) — `sustained_load_bearing_turn60` and
`sustained_load_bearing_final` both `True` on 6/6. The role-flip effect PART 3 measured at turn 24 is not a
transient: it persists to turn 60 and turn 100 with essentially unchanged magnitude (e.g. seed 44: agent margin
+0.0096(t30) -> +0.0109(t100); patient -0.0105(t30) -> -0.0120(t100) — if anything slightly SHARPENING, not
fading).

**4. Drift / catastrophic-interference (teach-then-distract).** Teach the AGENT role on `w0` for 30 turns
(established: |margin|>eps AND >=3/5 resolved on every seed), then run 70 turns of UNRELATED credited activity
on a DIFFERENT candidate pool (`w1`) while re-probing `w0`. Result on ALL 6 seeds: **`w_drift` = exactly
`+0.0000`** — `w0`'s taught weight is bit-for-bit identical immediately post-teaching (turn 30) and after 70
turns of unrelated distraction (turn 100). `margin_drift` is `+0.0000` to `+0.0001` (noise-floor) on every
seed, and `taught_role_preserved_after_distraction=True` on 6/6. The taught role is not eroded by intervening
unrelated conversation turns.

**Honest reading of the drift check.** This is a genuinely clean, reproducible zero — but it is a WEAKER claim
than "the mechanism is robust to noisy interference," and the finding says so explicitly. The engine's
`_hard_reset` between episodes drives membrane potential/recovery variables to a deterministic REST snapshot
and clears eligibility traces/last-spike-times; the `w1`-focused distractor episodes never drive `w0`'s
candidate-pool neurons at all (the R2/R3 `_episode` primitive loads+holds ONLY the named pool). With this
substrate's OU background-noise term effectively silent for this circuit (no spontaneous pre-synaptic firing
observed on the un-driven pool across 70 distractor turns, 6/6 seeds), there is no source of stray coincident
activity for the shared `wm_to_sel_r2` plasticity gate to act on even though it opens globally during every
distractor credited episode. So the exact-zero drift reflects the mechanism's ARCHITECTURAL isolation (per-
episode hard-reset + only the actively-held pool fires) under the PRODUCTION configuration actually shipped —
a real and relevant result — but this soak did NOT exercise a noisier substrate configuration where background
spontaneous firing could, in principle, produce a small stray coincidence on the untouched synapse. That
remains an open residual, named rather than papered over.

## VERDICT: GO — 6/6 seeds, STRENGTHENS the fe1911f2 flip

**GO = bounded_F3 AND grew_from_baseline AND lesion_did_not_grow AND no_degradation_over_turns AND
sustained_load_bearing_turn60 AND sustained_load_bearing_final**, all True on **6/6 seeds** (42, 43, 44, 100,
101, 102). This is **not a revert signal** — every check the flip's own residual named (a conversation
substantially longer than the 24-turn verify protocol) came back clean: the cross-edge does not run away, does
not collapse, does not silently degrade general comprehension, keeps signing the taught role deep into a
100-turn conversation, and an established teaching episode is not overwritten by 70 turns of unrelated
intervening activity. The production-default-ON flip (`BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN`,
commit `fe1911f2`) is CONFIRMED SAFE at this conversation length; no action needed on the flip itself.

## Declared residuals (honest, not deferred)

1. **100 turns is still short of an unbounded chat session.** The task asked for ">=60 turns (ideally 100)";
   this soak ran 100. A genuinely open-ended conversation (thousands of turns) was not tested, and the weight
   trajectory is still visibly approaching (not yet flat against) `stdp_w_max`=20 at turn 100 (final weights
   ~15.5-16.2) — an even-longer soak would show whether it saturates cleanly or needs a slower approach curve.
2. **The drift check's exact-zero result is bounded by this substrate's near-silent background noise for this
   circuit** (see "Honest reading" above) — it demonstrates architectural stability-plasticity under the
   SHIPPED configuration, not robustness to a noisier one.
3. **CPU/numpy only**, matching the task's memory/GPU-availability constraint (GPU busy). A faithful cupy
   real-handler soak through the actual `/api/brain-chat` production path (like the flip's own
   `_xedge_flip_production_verify` 6-seed cupy verify) with the mouth (Qwen) in the loop is the natural next
   rung, carried forward unchanged from the flip finding's own "Next" pointer.
4. **Semantic referent->pool binding stays a declared residual**, unchanged from PART 2/3 and the flip finding
   — the focus pool `w0` is a positional proxy, not a semantic discourse-role binding. This soak did not touch
   that residual; it only extends the per-turn plasticity's SAFETY envelope in time.
