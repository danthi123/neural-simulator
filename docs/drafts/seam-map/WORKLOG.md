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
