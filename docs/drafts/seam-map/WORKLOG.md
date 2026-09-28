# WORKLOG — host-decision seam map (runbook task 1)

Branch: `research/draft-seam-map` (from main @ 411782014). Task: trace ONE default chat turn
(`/api/brain-chat`, `webapp/server.py:4526`) in order; one table row per place Python decides
something from neural output. Facts only; "unsure" where not certain. Output:
`docs/drafts/seam-map/host-decision-seam-map.md`.

- 2026-09-25 23:58  Branch created from main @ 411782014. Entry point: `@app.post("/api/brain-chat")`
  at webapp/server.py:4526 (`def brain_chat`). The handler does request parsing + brain build/cache,
  then delegates the whole turn to `brain_reply(chat, req, source, cache_key)` (server.py:4628) — the
  shared full-faculty pipeline (also used by the TUI + OpenAI shim; see webapp/brain_reply.py).
- 2026-09-25 23:59  Read server.py 4526-4826. Structure of the default turn so far: (1) handler:
  renderer default, `_pin_bridge_backend()`, session cache key, mark session active, optional reset,
  build/cache ChatBrain, reload persisted learning; (2) `brain_reply` installs, in order, gate
  WRAPPERS around `chat.gate`: GNW ignition bus (gnw_bus_shadow, default), 2-organ coincidence bus
  (gnw_two_organ_bus, default-on), 3-organ consensus (gnw_three_organ_bus, flipped default-on),
  deliberation gate (gnw_deliberation, default-on), value-driven choice (value_choice_production_organ,
  default-on), multi-step chase deliberation (gnw_multistep_deliberation, default-on), OEG (default-OFF),
  composer read-only trace flag, then swap-drives topic swap (swap_drives_chat, default-on). The
  installation blocks are wiring (try/except install_X(chat)); the host decisions live inside the
  wrapped functions and in the rest of `brain_reply` (gate call, render, rich path, response assembly).
- 2026-09-25 23:59  Parallel work (not this branch): full test suite (runbook task 5) launched
  2026-09-25 ~23:52, PID 648822, log /home/dant123/Projects/sim-scratch/suite_run.log,
  junitxml /home/dant123/Projects/sim-scratch/suite.xml, command:
  `bash tools/memcap.sh 12 -- env SIM_NO_PROVENANCE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -p no:cacheprovider --continue-on-collection-errors --junitxml=/home/dant123/Projects/sim-scratch/suite.xml -rfE tests`.
  UNSURE: nothing yet; triage happens on `research/draft-test-triage` after it completes (~02:00).
- 2026-09-26 ~00:01 (prior session)  Wrote PART 1 of the map to `host-decision-seam-map.md` (35 KB): the
  complete trace-order line; the 57-row Part 1 table (default content turn, execution order, exact verified
  line numbers, "unsure" markers); the D1-D12 disjoint-classes table; the default-OFF blocks table; the
  single-fact mirror note (6775-7132); and the Part 2 skeleton (components A-E with file paths). NOT committed
  and NOT logged at the time (this entry is the retroactive log).
- 2026-09-26 14:59  RESUMED. Start-of-session: `bash tools/status.sh` → arcc_awake_completion 6/6 READY,
  b2b_base 256/258 WAITING (DO NOT HARVEST — Claude's). Checked the arcc harvest: the board line is ALREADY
  committed (411782014, "2026-09-25 23:40 HARVEST", GAP_CLOSURE_MISSION.md:245-248); only the RAW FILES commit
  was deferred (waiver budget exhausted until ~16:00 Sat). Re-ran the registered aggregate
  (`.venv/bin/python -m research.runners._da_tag_capture_chat_probe --family arcc --aggregate
  research/findings/raw/_awake_replay_completion`) → unchanged: verdict GO 6/6, signflip p=0.015625,
  n_completion_load_bearing=1. The raw files are untracked (`?? research/findings/raw/_awake_replay_completion/`
  in the MAIN checkout). ACTION: commit those raw files at/after 16:00 Sat when the waiver budget frees (needs a
  NO-READY-WORK waiver; Markdown-only is exempt but .json is not). NOT doing it at 14:59 (budget still closed).
- 2026-09-26 15:0x  Committing Part 1 now (Markdown-only → exempt from the idle-compute check). Then Part 2:
  the called components. FOCUS (context-limited local model, conserving Claude): A = `ChatBrain` core
  (`research/runners/brain_chat_tui.py:717` gate/gate_extract/render/what_does/is_it_true/_spiking_select) and
  B = `RichAnswerComposer.answer` (`research/runners/rich_answer_composer.py:303`) — the winner-selection /
  commitment / candidate-generation seams the host-share instrument (task 2) needs. C = the GNW gate-wrapper
  stack (consensus/veto/commit). D (observe_turn lead mappers) + E (organ mapping/text fns) = host compositions
  already pointed at by Part 1's "Part 2" refs; fill with the specific mapping + line, mark unsure where the
  read is not yet traced.
- 2026-09-26 14:15 (Claude): the GPU fell off the bus at ~11:07 during the commit above; Claude committed Part 1 on the local model's behalf (Markdown only). Resume at Part 2 (A = ChatBrain core ... E) per the entry above.
- 2026-09-26 20:38  RESUMED after compaction. Re-read the runbook in full (FIRST-EVERY-TIME): task 0 ("Pending
  commit from Claude") is now DONE (runbook line 138: "DONE 2026-09-26 16:10 by Claude (ae60029b0; b629d1a35)");
  the arcc per-seed raw files are committed (b629d1a35), only aggregate.json stays for Claude's scoring. `git -C
  /home/dant123/Projects/sim rev-parse --abbrev-ref HEAD` = main. `bash tools/status.sh`: GPU idle, pool queue 0,
  no READY battery (b2b_base is Claude's, arcc 0/6 WAITING), AWS $0.40, local-llm active. Nothing to harvest or
  queue. Wrote Part 2 section A (the `ChatBrain` core) into host-decision-seam-map.md: 12 rows A1-A12 in
  execution order as `gate()` runs (open-ended routing 811, in-loop acquisition 816, anaphora 819, comprehension
  1514, spiking recall 1521, miss-only fallbacks 1528-1573, honest-abstain 1613, gate abstain handling 828, host
  router candidate 852, **the `if recalled == p` commitment seam 857-868 = the host-share instrument target**,
  render+verify 1642, spiking-mouth 1648). Verified location facts: `what_does`/`is_it_true` are on the INNER
  `BrainConversationalAgent` (brain_conversational_agent.py:772/783), not ChatBrain; `_spiking_select` is on the
  composer (one_brain_composer.py:956) → section B. `what_does` → `composer.query_patient(a,v)` (the spiking
  recall, returns patient or None = the no-confab moat). UNSURE markers kept on A2/A3/A6/A11 (host-vs-neural
  split of comprehension, anaphora, fallbacks, render). Next: section B (`RichAnswerComposer.answer`,
  rich_answer_composer.py:303).
- 2026-09-26 20:5x  Wrote Part 2 section B (`RichAnswerComposer.answer`, rich_answer_composer.py:303) into
  host-decision-seam-map.md: 13 rows B1-B13 in execution order as `answer()` runs. Verified line facts against the
  source: B1 `_is_followup` (1001, helper 219-230); B2 `if not facts: abstain` (1003-1008); B3 single-HypothesisSVO
  flagged-guess (1014-1018); B4 `if not kept: abstain` (1020-1024); B5 `direct = self._direct_fact(question); if
  direct is None: return None,[]` (647-649, the moat); B6 topic `p if (p in agents_set) else a` (664-668); B7
  chain+elab+`_dedup(...)[:max_sentences]` cap (671-674); B8 `resolve_compositional_chain` FIRST then `chat.gate`
  (463-468); B9 `_chain_facts` spiking hop + host corner-turn `own[0]` (503-520); B10 `_planner.ordered_associates`
  dlPFC rank (540 / 578-579, neural); B11 `composer.elaborate` + edge-weight fallback (613-623); B12 per-fact
  `_render_one_verified` keep/drop (826-832); B13 `_verify_claim_set`/`_verify` (987-991). Added a "Note on the
  moat": B2/B4/B5/B12/B13 are the same no-confab moat at different altitudes (neural verdict → host maps to
  abstain/drop, never injects a fact); the only genuinely host-authored candidate-generation in B is B9's
  corner-turn (`own[0]`) and B11's edge-weight fallback, both fire only after the spiking selection missed and both
  draw from the brain's own stored facts (never the LLM). Fixed a duplicate B skeleton bullet left over from the
  section A edit (the bullet now appears once, at 205 as C). UNSURE/partial markers kept: B3/B6/B7/B8/B9/B11 are
  PARTLY (host-vs-neural split). Next: section C (the GNW gate-wrapper stack, server.py:4650-4769).
- 2026-09-26 17:1x  Wrote Part 2 section C (the GNW gate-wrapper stack) into host-decision-seam-map.md: 25 rows
  C1-C25 grouped by wrapper outermost→innermost (multistep C1-C4, value_choice C5-C9, deliberation C10-C13,
  three_organ C14-C21, two_organ fallback C22-C23, bus_shadow innermost-fallback C24-C25), each in execution order
  with verified file:line. Verified the architectural facts against server.py + the six wrapper sources: install
  order bus_shadow(4652)→two_organ(4679)→three_organ(4700)→deliberation(4721)→value_choice(4745)→multistep(4765),
  last installed = outermost; execution order multistep→value_choice→deliberation→three_organ→[two_organ→bus_shadow→
  host gate]; three_organ is the ACTIVE default author (two_organ/bus_shadow are fallbacks reached only on re-check
  fail or throw). KEY FINDING (mechanism=fact, fires?=UNSURE): `install_deliberation_gate` (gnw_deliberation.py:237)
  passes `getattr(chat, "_last_gnw_bus", None)` into `deliberate_svo`, which gates on `authored_by == "bus"`
  (gnw_deliberation.py:179), but `chat._last_gnw_bus` is set ONLY by gnw_bus_shadow (gnw_bus_shadow.py:533) —
  three_organ sets `chat._last_three_organ` instead (gnw_three_organ_bus.py:425) and on the default path never
  invokes the inner bus, so `_last_gnw_bus` is stale/None on a default turn → the deliberation conflict gate
  (C10-C13) is likely inert on the default three_organ path (a candidate host-share-instrument target). Added a
  "Note on the moat across C": the abstain rows (C2/C3/C13/C21 + the committed-is-None branches in C18/C23/C25)
  are the same no-confab moat at different altitudes; the genuinely host-authored candidate-generation in C is
  C1 chase-parse, C5 (agent,action) extraction, C7 engagement context, C12 equal-drive/retry, C14 real-vocab
  membership, C17 vote-vector assembly, C24 miss-only identity retry — each fires only around a spiking read and
  draws from the brain's own store/vocab (never the LLM). `check_docs.py` PASSES (W1 0, W2 0). Next: section D
  (observe_turn lead mappers) + section E (organ mapping / text fns).
- 2026-09-26 17:2x  Wrote Part 2 sections D + E (the final two), completing the map. Section D (the observe_turn
  lead mappers): 5 rows D1-D5 — swap_drives (D1, transition_lead "On %s, then — "), affect_drives (D2, appraise_text
  host + #81 ladder read), da_mode (D3, engagement mix → spiking SNc → DA level → mode → _SUFFIX), gnw_global_stop
  (D4, detect_trigger host gate + spiking clear-all + stop_lead), continuous_engine recent_wander (D5, consume-once
  bookkeeping over the selfinit organ's tick-time spiking selection). All five = the conditioned-articulation
  pattern (spiking verdict → host-composed lead/suffix that frames the MANNER, never the fact). Section E (the organ
  mapping/text fns): 3 sub-types, 28 rows E1-E28. E.1 host parses (E1-E7, input boundary: is_causal_query/extract_cue,
  extract_topic, extract_assertion, parse_intention, is_feel_query, appraise_text — declared sensory-encoding
  boundary, the DECISION is the downstream spiking/moat read). E.2 host threshold/compare read-outs (E8-E14:
  tone_level/content_plan/manner_template_for, **mean_role_confidence = HOST ARITHMETIC over matched-filter scores by
  default — only opt-in margin_spiking is spiking (BRAIN_METACOG_SPIKING_MARGIN default OFF)**, evidence_from_role_conf,
  apply_cap, reach_plan). E.3 host text-composition read-outs (E15-E28: recall_disclosure, expectation_readout,
  surprise_notice, pragmatic_notice, rejection_message, clarification_question, read_author/authorship_marker,
  reminder/acknowledgement_text, hedge_prefix, followup_question, comprehension judge, d3 maybe_answer, silent_wm,
  d6 judge). KEY host-share-instrument targets flagged in the "Note on E": (1) E11/E12/E13 the metacog confidence
  read is host-owned by default; (2) the E.1 parses gate the turn class (legitimate host, but the comprehension-gate
  biologizing target). `check_docs.py` PASSES (W1 0, W2 0). MAP COMPLETE (Part 1 57 rows + Part 2 A1-A12/B1-B13/
  C1-C25/D1-D5/E1-E28). Ready to commit (Markdown-only → exempt from the idle-compute check).
- 2026-09-26 17:3x  COMMITTED Part 2 as e1e4e0ad0 (2 files, 188 insertions; the pre-commit ⚠️ lines — lever-efficacy
  40 / throughput 1 / stale-pointer 1 — are ADVISORY, the commit landed). Working tree clean. PENDING PUSH: the runbook
  workflow for this branch is `git push origin research/draft-seam-map` + the same for `gitea` (never merge, never to
  main). The push has NOT gone out yet — the Bash safety classifier kept timing out on the network action (4 retries
  17:2x-17:3x; read-only git works, the push could not be assessed). RETRY THE PUSH when the classifier is back:
  `git push origin research/draft-seam-map && git push gitea research/draft-seam-map` (run from this worktree). The
  local commit e1e4e0ad0 is safe either way. TASK 1 (the seam map) is COMPLETE; the next runbook task is #2 (the
  host-share instrument, `research/runners/_draft_host_share_probe.py` + a test) — NOT started (a fresh build; left
  for a full-context session, and the metacog host-arithmetic finding E11/E12/E13 is the natural first seam to probe).
- 2026-09-26 17:5x  PUSHED to BOTH remotes (classifier back): `git push origin research/draft-seam-map`
  (d98494a2a..e1e4e0ad0) and `git push gitea research/draft-seam-map` (same range). The PENDING PUSH above is
  RESOLVED. Branch `research/draft-seam-map` is now on origin + gitea, ready for Claude's review. TASK 1 (the
  host-decision seam map) is COMPLETE: 57-row Part 1 + Part 2 A1-A12 / B1-B13 / C1-C25 / D1-D5 / E1-E28, all
  verified against source, check_docs PASSES. Next runbook task = #2 (host-share instrument,
  `research/runners/_draft_host_share_probe.py` + a test) — NOT started; natural first seam = the metacog
  host-arithmetic confidence read (E11/E12/E13, `mean_role_confidence` is host arithmetic over matched-filter
  scores by default; only opt-in `BRAIN_METACOG_SPIKING_MARGIN` is spiking).

## Runbook task 2 — host-share instrument, first seam

- 2026-09-26 18:0x  STARTED task 2. Seam chosen: `metacog_production_organ.mean_role_confidence` (a
  COMMITMENT-STRENGTH seam: host arithmetic over the composer's per-role matched-filter scores ->
  `evidence_from_role_conf` -> `MetacogProductionOrgan.judge` -> the `confident` bool = the reply's assert-vs-hedge
  commitment). Chose it over the `if recalled == p` commitment seam because it is the cleanest host-arithmetic seam
  AND there is a reusable composer-direct build pattern (`_metacog_spiking_recall_margin_derisk.py`, ~40x faster
  than the full chat stack; the mechanism is entirely internal to composer + metacog organ).
- 2026-09-26 18:0x  BUILT `research/runners/_draft_host_share_probe.py` (NEW file, no sim//webapp/ edit; imports +
  wraps production code). Reuses the composer-direct pattern: `OneBrainComposer(seed, trace=True, vocab=VOCAB)` +
  `hear` 5 dev facts + `query_patient` -> `last_trace`. Sets `BRAIN_METACOG_SPIKING_MARGIN=1` so each role chip
  carries BOTH `margin` (host) AND `margin_spiking` (spiking) -> one query serves both arms. Records the three
  plan-section-2 measurements per query: (a) DECODING = host_mrc (the host chain, never prefers margin_spiking) +
  the per-role fields; (b) SUBSTITUTION = downstream `confident` at host_mrc vs at neural_mrc (mean_role_confidence
  with the flag on, prefers margin_spiking); (c) INTERVENTION = hold the trace fixed, sweep host_mrc over MRC_GRID,
  read `confident` at each (monotonicity + host-controllability). host_share_indicator = fraction of answered rows
  where the host's mrc choice controls `confident` AND the WTA transmits it monotonically.
- 2026-09-26 18:0x  SMOKE (dev seed 7, `bash tools/memcap.sh 8 -- env SIM_BACKEND=numpy ... --seeds 7`): 5/5
  answered, decoding_corr=0.769, host_share=1.0. Wrote `research/findings/raw/_draft_host_share_probe/
  dev_seed7_smoke.json`.
- 2026-09-26 18:0x  DEV SWEEP (seeds 1-7, memcap 8, ~89s): 35/35 answered. Per-seed decoding corr
  [0.969,0.93,0.859,0.062,0.946,0.767,0.769] (pooled 0.838; SEED 4 IS AN OUTLIER at 0.062 -- the host read does not
  track the neural read there). host_mrc 0.78+-0.045, neural_mrc 0.869+-0.054 (both HIGH). SUBSTITUTION
  confident_changed = 0/35 (a NO-OP on this CLEAN-fact panel: both confidences sit above the confident threshold, so
  the commitment is confident either way -- a PANEL LIMITATION, not a bug). host_share = 1.0 on all 35 rows (the
  intervention is a single 0->1 transition at evidence 0.5 on every row -> the spiking WTA faithfully transmits the
  host's evidence -> the host owns the commitment at this seam). Wrote `dev_sweep.json`.
- 2026-09-26 18:0x  TEST `tests/test_draft_host_share_probe.py` (dev seed 7, numpy/CPU): 2 tests --
  (1) the three measurements are recorded (host + neural reads present, per-role fields finite, both substitution
  arms map to a confident bool); (2) the intervention is monotonic (n_transitions <= 1 -- the core claim).
  `2 passed in 15.8s`.
- FINDINGS (DEV, not a verdict): (1) host OWNED the commitment at this seam on the dev panel (host_share 1.0, 35/35
  monotonic); (2) the host's arithmetic mostly DECODES the neural read (pooled corr 0.838) but DIVERGES on seed 4
  (0.062) -- worth a follow-up (is seed 4's host read a degenerate case?); (3) the substitution is a NO-OP on a
  clean panel. NEXT STEP (clear, not a NEEDS-CLAUDE): add a DEGRADED panel (reuse `_noise` from
  `_emergent_graceful_degradation_derisk` on the store conns, a small sigma sweep) so the host vs neural confidences
  STRADDLE the threshold and the substitution can show a real commitment change. NOT yet built (context budget).
- 2026-09-26 19:0x  BUILT the DEGRADED panel (the NEXT STEP above). `research/runners/_draft_host_share_probe.py`
  gains: `_noise(store_conns, sigma, rng)` (mirrors `_emergent_graceful_degradation_derisk._noise` verbatim -- complex
  Gaussian jitter on the store phasors; sigma 0 = the intact store), `SIGMA_GRID = [0,0.3,0.6,0.9,1.2,1.5,1.75,2.0,
  2.25,2.5]`, and `run_seed_degraded(seed)` (per sigma: apply `_noise` to a copy of the store, re-query the 5 dev
  facts, record host_mrc / neural_mrc / host_confident / neural_confident / confident_changed, then RESTORE the intact
  store before the next level -- the perturbation never accumulates). A `--degraded` CLI flag runs this panel instead
  of the clean one. No sim//webapp/ edit (the probe sets `comp.store_conns` on a COPY and restores it, exactly the
  derisk's `_sweep` pattern; verified empirically that a single-query read re-reads `store_conns` each call, so the
  perturbation reaches the read without touching `_store_dirty`).
- 2026-09-26 19:0x  SMOKE (dev seed 7, `--degraded`): the clean panel (sigma 0-1.5) is a NO-OP (straddle=0); the
  straddle appears at sigma 1.75 (2 facts: the host's lower mrc maps to hedge, the neural's higher mrc maps to assert).
  Wrote `dev_seed7_degraded.json`.
- 2026-09-26 19:0x  DEV SWEEP (seeds 1-7, memcap 8, `--degraded`, ~3 min): EVERY seed produces a NON-EMPTY straddle
  band, all in the DEGRADED regime, none at the clean panel: seed1 [1.5,2.5], seed2 [1.5,2.25], seed3 [1.5], seed4
  [1.5,2.0,2.5], seed5 [1.75,2.25], seed6 [1.5,1.75], seed7 [1.75]. The clean panel (sigma 0-1.2) is a NO-OP on all 7
  (both arms confident). Wrote `dev_degraded_sweep.json`.
- 2026-09-26 19:0x  TEST: added `test_degraded_panel_straddles` (dev seed 7): the clean panel (sigma 0) answers all 5
  facts with n_straddle=0 (the NO-OP), AND the degraded panel produces a non-empty straddle band that excludes sigma 0
  (a real commitment change in the degraded regime), AND every straddling row is a genuine host-vs-neural divergence
  (both confident bools present and unequal). `3 passed in 34.9s`.
- FINDINGS (DEV, not a verdict): the DEGRADED panel closes the clean-panel NO-OP. As the store degrades, the host and
  the neural mrcs fall and cross the metacog confident-threshold at DIFFERENT points, so the two arms STRADDLE and the
  substitution (host decision -> neural decision) flips the downstream commitment (hedge<->assert) on a band of sigma
  levels. The band is NARROW (1-3 sigma levels per seed) and sits in the degraded regime (sigma ~1.5-2.5): the host
  and neural mrcs both fall with sigma, so they straddle the threshold only in a window before the facts abstain. The
  straddle direction is per-fact (the host is usually the less-confident read, but not always) -- the invariant is the
  DIVERGENCE, not which arm is higher. This is a DEV screen (dev seeds 1-7, NOT the 6 gate seeds); it shows the
  substitution CAN be load-bearing, it is not a 6-seed generalization claim.

## The tone-selection seam (next rung): the N=5 tonic-firing wall (2026-09-27 → 2026-09-28)

- 2026-09-27  The next rung of the seam (the tone-selection seam, `webapp.affect_tone_selection.select_by_mood`)
  is the N-channel basal-ganglia action selector (`research/runners/_tone_bg_nchannel_selector.py`). It commits
  cleanly at N=2 but NOT at N=5: at the default tonic (gpi=1000/thal=900) every commit pool fires tonically
  (~9000 spikes each) so no clean winner emerges. Finding:
  `research/findings/2026-09-27-tone-bg-nchannel-tonic-firing-wall.md`.
- 2026-09-28  The wall localised to the GPi's unfaithful near-silence (pathway trace: the D1 pool differentiates
  perfectly, but the GPi fires only 8-28 spikes so its differentiation cannot modulate the thalamus). The no-defer
  lever: raise ONLY the GPi tonic (the real GPi fires at a ~20-50 Hz tonic). A clean N=5 winner emerges at
  gpi=400-600 (thal=45) on the dev seed (v_held=+0.9). Fixed operating point: gpi=500, thal=45.
- 2026-09-28  SIX-SEED VERDICT (non-negotiable #6, raw `research/findings/raw/_gpi_fix_6seed.json`): the dev seed
  was optimistic. Mean intact commit 0.40 (range 0.15-0.69), 0/6 go (the 0.9 commit bar is not met). BUT the
  anti-cheat is rock-solid on every seed: lesioning EITHER the arousal drive OR the direct path drops commit to
  0.0 in all six seeds -> 100% of the selection is owned by the spiking race, 0% by the host (the 2026-09-25
  "who owns the computation" answer for this seam). The winner is affect-dependent and mood-congruent, reliably at
  the high-contrast moods (v_held=-0.9/0.0/+0.9); the residual is the low-contrast moods (v_held=±0.45, salience
  contrast 0.05) plus seed-to-seed sensitivity of the operating point.
- 2026-09-28  VERDICT (honesty boundary): the GPi-tonic method REPLACES the wall's method (the capability is
  present and 100% neural-attributable) but does NOT close the wall to the de-risk bar (0/6 go, mean commit 0.40
  < 0.9). No-defer: the capability is kept; the GPi-tonic lever is banked as a real (partial) fix; the next lever
  is the STN->GPi competition loop + post-commit reset (the model has the regions `selector_stn`/`selector_reset`;
  the real selector runs them alongside the GPi tonic — the "missing companion process" question).
- NOTE: the main-arc summary docs (GAP_CLOSURE_MISSION.md / ROADMAP.md / AUTONOMOUS_STATE.md) are 11 commits behind
  main on this branch; the N=5 verdict is recorded here + in the finding doc and folds into the main arc's walls
  ledger at merge time (not edited on the stale base, to avoid a conflict).
