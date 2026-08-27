---
type: finding
status: live
date: 2026-08-27
mechanism: shared full-faculty turn pipeline — the faculty-DRIVE coupling sequence (affect_drives / swap_drives / GNW N-organ + 2-/3-organ ignition buses / value-choice / multistep / metacog / curiosity / surprise / episodic / world-model / prospective / pragmatic / ...) extracted from webapp/server.py's brain_chat HTTP handler into ONE module-level function webapp.server.brain_reply(chat, req, source, cache_key), re-exported via the new webapp/brain_reply.py so every surface reaches the SAME pipeline
lane: architecture (one-brain integration — a single faculty-drive pipeline behind the webapp handler, the OpenAI shim, and the standalone TUI)
verdict: DONE — the couplings lived only in the webapp handler, so the standalone TUI (research/runners/brain_chat_tui, SIM_BACKEND=... python -m research.runners.brain_chat_tui) gave the recall+moat CORE but missed every coupling; the OpenAI shim already rode brain_chat. Extracted the handler body VERBATIM into a module-level brain_reply (pure extract-method, zero reindentation, zero return-site edits) + a thin webapp/brain_reply.py surface (reply_over_chat for a caller-built ChatBrain). GATE MET: /api/brain-chat is byte-identical before vs after across 12 representative turns (self, knowledge, affect +/-, assertion/surprise, topic-swap, anaphora, abstain/curiosity, prospective, feel-readout) at production-default flags. TUI now routed through the shared pipeline: on identical turns it fires metacog/affect_drives/curiosity/da_drives/worldmodel/swap couplings that ChatBrain.answer never produced (answers differ on 3/3 probed turns), and the affect_drives lead 'Honestly — ' VANISHES under BRAIN_AFFECT_DRIVES_LESION=1 (load-bearing, not hollow). Shim still answers /v1/chat/completions (object=chat.completion, content non-empty + carries the da suffix -> rides the shared path, reasoning_content present). Additive/refactor only, NO sim/ edit, no behavior change to production.
artifacts:
  - research/findings/raw/2026-08-27-shared-faculty-pipeline-verify.json
---

# One shared faculty pipeline: the TUI + the OpenAI shim + the webapp now run the IDENTICAL coupling sequence (webapp byte-identical)

Artifact: `research/findings/raw/2026-08-27-shared-faculty-pipeline-verify.json`.

The faculty-DRIVE couplings were wired ONLY into `webapp/server.py`'s `brain_chat` handler, so the standalone TUI — which talks to the shared core `ChatBrain` directly — got recall + the moat but NONE of the couplings (affect/swap/metacog/curiosity/surprise/world-model/...). The shim (`/v1/chat/completions`) already delegated to `brain_chat`, so it was fine.

Fix: extract the handler's turn body VERBATIM into a module-level `webapp.server.brain_reply(chat, req, source, cache_key)` (the handler is now request-parse + brain build/cache, then `return brain_reply(...)`). The extraction is a pure move — both are top-level functions with the same 4-space body indent, so no reindentation and no return-site edits were needed; the webapp path returns the identical `JSONResponse`. A new `webapp/brain_reply.py` is the thin, documented import surface: `reply_over_chat(chat, msg, ...)` runs the pipeline on a ChatBrain the caller already built (the TUI's entry), decoding to the payload dict exactly as the shim does.

Byte-identity gate (the hard one): 12 representative turns through `/api/brain-chat` at production defaults were byte-for-byte identical before vs after. The TUI's `run_repl` now routes each turn through `reply_over_chat` (falling back to the local core if the webapp package is unavailable) — so it exercises the WHOLE brain, proven by couplings that fire where `ChatBrain.answer` produced a bare answer, and by the affect lead vanishing under lesion.

Standing DISCIPLINE (recorded in `webapp/brain_reply.py`): a faculty coupling belongs in the SHARED `brain_reply` pipeline, NEVER inline in a request handler — wiring it into the handler alone silently regresses the TUI + the shim back to a partial brain.
