---
type: finding
status: design
claim_check: synthesis
date: 2026-09-05
mechanism: cross-cutting research-hygiene re-audit of research/coordination/scaffold_retirement_backlog.md
  (drift #12, stale-pointer prevention) + a prioritized, diversified READY-NEXT-RUNGS list across faculties,
  explicitly excluding further affect-gate work and the mouth (both well-served this session already)
lane: coordination / cross-faculty planning
verdict: >
  PLANNING / hygiene only — no new measurement, no science verdict changed. 13 stale or missing status pointers
  in the scaffold-retirement backlog were corrected in place (research/coordination/scaffold_retirement_backlog.md),
  each citing the finding that supersedes it: 2 items were DONE-but-listed-as-ready (rank-4, rank-5, both flipped
  to production default-ON), 1 item was a straight mis-scope (rank-11, already integrated), 1 was a stale
  "not flipped" claim now false (rank-12, since flipped), 1 was a falsified "ready" example (rank-24's d5
  depth_hold direct-read, NO-GO), 1 was a mislabeled rank (the old "Rank-20 worldmodel-state WTA" line, which
  matches no finding or ranked item), and 6 more moved from "fresh"/"partial" to a concrete de-risk verdict
  (ranks 1, 2, 9, 10, 15, 16) without the backlog reflecting it. See the corrected doc for the full per-rank
  citations. Separately, a diversified prioritized ready-next-rungs list is below, ranked by leverage-on-
  genuine-conversation times readiness, spanning composer/memory, value/reward, knowledge-capacity, curiosity,
  introspection, self-schema and perception — explicitly excluding affect-gate work and the mouth.
artifacts:
  - research/findings/raw/_onebrain_fact_shard_wirein/verify_404_6seed.json
  - research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partA.json
  - research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partB.json
  - research/findings/raw/_shared_salience_prodflip/verify_AB.json
  - research/findings/raw/appraisal_interoceptive_ladder/_production_flip_verify.json
  - research/findings/raw/_gnw_stop_trigger_production_flip_verify.json
  - research/findings/raw/_rank6_knowledge_core_substrate_write_derisk/full_run.json
  - research/findings/raw/_curiosity_graded_novelty_derisk.json
  - research/findings/raw/_metacog_spiking_recall_margin_derisk/6seed_results.json
  - research/findings/raw/_selfschema_neural_turnclass/soak_6seed.json
  - research/findings/raw/_da_write_gain_spiking/6seed.json
  - research/findings/raw/_value_choice_neural_context/verify_6seed.json
  - research/findings/raw/_rank23_vision_cluster_spiking_wta.json
  - research/findings/raw/_rank24_quick_flips/d5_direct_read_6seed.json
external: NO-EXTERNAL-NEEDED — this is a hygiene/planning re-audit of the project's OWN existing findings against
  its OWN coordination doc, not a new biological or engineering question.
---

# Scaffold-retirement backlog hygiene pass + a diversified prioritized ready-next-rungs list (excluding affect-gate and the mouth)

## Why this exists

`research/coordination/scaffold_retirement_backlog.md` was produced earlier today (`6c63136ff`) and has one
append-only STATUS UPDATES section maintained by three later commits (ranks 6, 8, 12). Between then and now,
**16 more findings landed against named ranks in that same backlog** (ranks 1, 2, 4, 5, 9, 10, 11, 12-update,
15, 16, 20, 23, 24) without the backlog being told. Per CLAUDE.md's own drift-#12 discipline ("stale pointers are
the #1 cause of re-deriving concluded work"), this is exactly the failure mode being guarded against: a reader
of the backlog today would see rank-4 and rank-5 still listed under "READY TO BUILD NOW" and spend an agent
re-deriving work that is already merged and running in production, or would see rank-12's status update say
"NOT flipped default-on" when the code now says otherwise.

## What was corrected (see the doc itself for the full text; this is the index)

| rank | backlog said | now confirmed | citation |
|---|---|---|---|
| 1 | "shortest path" step, still framed as fully open | literal rebuild still NOT done (verified: `composer_kind` defaults to `'rf'`, `webapp/server.py:3925` still describes scale787 as `'rf'`) — but the retrieval-latency sub-blocker is RESOLVED | `2026-09-05-onebrain-fact-shard-wirein-production-composer.md` |
| 2 | "ready to build: thread + re-verify" | Part A (plumbing) is GO/wired-default-off; Part B (at-scale) is UNDEFINED (honest recall-incompleteness); its own named fix (sharded composer wire-in) has SINCE LANDED, unblocking the re-verify | `2026-09-05-rank2-integrated-loop-webapp-thread-derisk.md`, `...-at-scale-answer-preservation-UNDEFINED-recall-incomplete.md`, `2026-09-05-onebrain-fact-shard-wirein-production-composer.md` |
| 4 | listed under "READY TO BUILD NOW" | FLIPPED TO PRODUCTION DEFAULT-ON, merged to the checked-out main (verified: `shared_salience_enabled()` returns `True` unset; flip commit `04cbd1bec` is an ancestor of HEAD) | `2026-09-05-shared-salience-afferent-production-default-flip-GO.md` |
| 5 | listed under "READY TO BUILD NOW" ("the readiest") | FLIPPED TO PRODUCTION DEFAULT-ON (verified: `appraisal_interoceptive_enabled()` returns `True` unset) | `2026-09-05-gateB-appraisal-interoceptive-production-flip-GO.md` |
| 9 | "fresh" | de-risked PARTIAL, 6/6 lesion-load-bearing, residual = precision in the ambiguous middle band | `2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md` |
| 10 | "de-risked start" | full 6/6-seed GO de-risk, default-OFF | `2026-09-05-rank10-curiosity-graded-novelty-familiarity-scaffold-derisk-GO.md` |
| 11 | "fresh, self-referential" | MIS-SCOPED — already neural, wired, production DEFAULT-ON since 2026-08-19/20, re-verified byte-identical against TODAY's code | `2026-09-05-rank11-topic-swap-scaffold-backlog-item-already-integrated.md` |
| 12 (status update) | "NOT flipped default-on (owner call)" | now FALSE — flipped to DEFAULT-ON later the same session, 6/6-seed GO | `2026-09-05-gnw-stop-trigger-accbg-circuit-PRODUCTION-FLIP-GO.md` |
| 15 | "fresh" | de-risked PARTIAL 5/6, one characterized miss | `2026-09-05-selfschema-authorship-neural-turnclass-derisk-PARTIAL.md` |
| 16 | "partial: homeostasis-half retired" | BOTH halves now de-risked (the remaining write-magnitude leaf is GO, default-OFF) | `2026-09-05-da-write-gain-spiking-derisk-GO.md` |
| 20 (READY-list label) | "Rank-20 worldmodel-state WTA" | mislabeled — rank-20 is value-choice reward-context (matches no "worldmodel-state WTA" finding), now de-risked GO default-OFF | `2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md` |
| 24 (READY-list example) | "d5 depth_hold direct-read" cited as ready-to-build | NO-GO, falsified: `depth_hold` is not comparable across topics | `2026-09-05-d5-depth-hold-direct-read-NO-GO-cross-topic-baseline-not-comparable.md` |

All 13 corrections are recorded append-only in `research/coordination/scaffold_retirement_backlog.md`'s new
"HYGIENE RE-AUDIT, 2026-09-05 (later cycle)" section, plus a rewritten "READY TO BUILD NOW" paragraph that no
longer lists the two done items or the falsified example. **No science verdict was changed or invented — every
correction quotes a pre-existing, already-committed finding's own frontmatter/verdict**, e.g. rank-2's own
production-scale artifact `research/findings/raw/_rank2_integrated_loop_webapp_thread_derisk_partB.json`
(`overall_go: False`, 17-26/35 answers exactly matching per seed) and rank-4's production-flip artifact
`research/findings/raw/_shared_salience_prodflip/verify_AB.json` (6-seed default-change correctness), both
re-read directly (not re-cited from a headline) before writing the corrections above.

## Diversified prioritized READY-NEXT-RUNGS (excludes affect-gate work and the mouth — both well-served this session)

Ranked by **leverage on genuine-conversation × readiness**. "Ready" means a runnable/buildable step with a clear
GO-bar, not a research question still needing a new idea.

**1. Memory/composer — Rank-2 at-scale re-verify, NOW UNBLOCKED. [GPU-shaped]** Leverage: very high — this is
recall-completeness in the exact spiking recall path the deployed conversation would use. The V=320/K=32 6-seed
re-verify (`_rank2_integrated_loop_webapp_thread_derisk.py --skip-mechanical`) came back UNDEFINED with an
honest recall-incompleteness gap (patient matches 24-31/35, but 6-14/35 cues get an "unknown" the host would
answer), and the finding named its own fix: re-run AFTER the DG-CA3 sharded composer is wired to the live path.
That wire-in (`BRAIN_FACT_SHARD_RETRIEVAL`) landed later the same session. Re-running the identical 6-seed
battery with the fact-shard flag on is the single most decisive next measurement in the memory lane — either the
recall gap closes (unblocking rank-2's own flip) or it doesn't, which sharpens where the residual actually
lives. GO-bar: `seedXX_answer_identical` clears on a clear majority of seeds where it currently fails on all six.

**2. Memory/composer — Rank-1 bundle rebuild (rf → onebrain). [CPU-shaped, possibly GPU for the build step at scale]**
Leverage: MAXIMAL per the backlog's own original ranking (every recall/store/abstain in production routes
through this bundle). Readiness: now HIGH — the retrieval-latency objection to running onebrain at production
scale is resolved (item 1 above), so the actual rebuild + a no-regression parity check against the current
`rf`-composer bundle's answers is the concrete next step, not a fresh de-risk. GO-bar: the rebuilt bundle answers
identically (or with a characterized, honest delta) to the current `rf` bundle on the production fact set, with
the fact-shard flag available (default-off, opt-in) for when rank-2's re-verify clears.

**3. Knowledge-capacity/persistence — Rank-6's `ShardedPhasorStore.save()` pickle bug. [CPU-shaped, pure engineering]**
Leverage: high — this is the concrete, previously-undocumented reason the 78k-fact LTM core excludes the
already-validated synaptic-weight write path (`enable_substrate_store=True`, GO 6/6 seeds, 50.26 KB/fact ≈ 3.78 GB
projected at full scale — affordable). Readiness: very high — it is a scoped, reproduced `TypeError: cannot
pickle 'mappingproxy'`, not an open research question; fixing it is what actually unlocks scaling the
already-GO'd write path past the current default-off ceiling. See
`research/findings/2026-09-05-rank6-knowledge-core-substrate-write-scaled-derisk-mixed.md`.

**4. Value/reward — Rank-16 + Rank-20 production-flip verify. [CPU-shaped, ready now]** Leverage: medium — closes
the last host-arithmetic leaves in the DA/value pathway (write-gain leaf, value-choice context). Readiness: very
high — both are already 6/6-seed GO'd de-risks; the only remaining step is the exact production-flip-and-verify
recipe already executed four times this session for ranks 4, 5, 8 and 12 (flip the default, re-run the 6-seed
no-regression + load-bearing-not-hollow battery through the real production organs). See
`research/findings/2026-09-05-da-write-gain-spiking-derisk-GO.md` and
`research/findings/2026-09-05-value-choice-real-critic-neural-salience-context-6seed-GO.md`.

**5. Curiosity — Rank-10 production-flip verify. [CPU-shaped, ready now]** Leverage: medium — a graded novelty
signal (vs. today's `NOVEL_SIGNAL=0.95` constant on every abstain) should make curiosity-driven follow-ups
sensitive to which specific topic is actually novel, a direct driver of engagement quality in extended
conversation. Readiness: very high — same known flip-and-verify recipe as item 4. See
`research/findings/2026-09-05-rank10-curiosity-graded-novelty-familiarity-scaffold-derisk-GO.md`.

**6. Introspection/self-model (metacognition) — Rank-9's ambiguous-middle-band residual. [Needs-scoping]**
Leverage: medium — this is the evidence feeding the honesty-hedge, a named deliverable of the north-star's
"honesty boundary." Readiness: NOT a quick flip — the de-risk's own window-size sweep already failed to resolve
the ~50%-agreement ambiguous-middle-band gap, so per CLAUDE.md's wall-reframe ("what companion process did we
replace with a constant?"), the next lever is likely a genuinely different read (e.g., an accumulation-to-bound
process over the recall competition rather than a single-snapshot margin) — a real research question, not an
engineering task. See `research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md`.

**7. Self-schema/authorship — Rank-15's one characterized miss. [CPU-shaped, narrow, ready now]** Leverage:
low-medium (a narrow capability: self-authored vs. recalled classification). Readiness: high but small-scope —
the single miss (a generated candidate sharing 2/3 role-fillers with a co-resident stored fact, reading weak
novelty by construction) is precisely characterized, so a bundled-cue novelty fix (e.g. penalize shared-role-
filler overlap directly) is a scoped, testable lever, not an open question. See
`research/findings/2026-09-05-selfschema-authorship-neural-turnclass-derisk-PARTIAL.md`.

**8. Perception/vision — Rank-23's grouping-decision WTA. [Needs-scoping, lowest priority]** Leverage: low right
now — the finding is explicit that there is **no live consumer**: the EMERGE-36 bird/fish pipeline this de-risks
is not wired into any production faculty today. Readiness for the mechanism itself is high (GO, 6/6 seeds), but
the genuinely-ready next step is not more de-risking — it is deciding whether/where this pipeline should become a
live consumer at all before investing further. See
`research/findings/2026-09-05-rank23-vision-cluster-grouping-decision-spiking-wta-derisk-GO.md`.

## Explicitly excluded (well-served this session, per the task brief)

Affect-gate rank-7 (grounded-experience-stream / noise-robust homeostatic convergence / spiking-port boundary —
already has a de-risk GO and a banked spiking-port surpass this session) and the mouth (broad-domain fluency —
lever already decided, token-scaling train running). Neither needs another rung queued right now.
