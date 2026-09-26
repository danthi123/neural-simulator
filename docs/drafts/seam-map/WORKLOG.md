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
