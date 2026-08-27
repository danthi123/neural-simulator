"""THE ONE SHARED FULL-FACULTY TURN PIPELINE — import surface + conveniences (2026-08-27).

The faculty-DRIVE couplings (affect_drives / swap_drives / the GNW N-organ + 2-/3-organ ignition buses /
confidence-forthcoming / metacog / curiosity / surprise / episodic / world-model / prospective / ... ) used to be
wired ONLY into `webapp/server.py`'s `brain_chat` HTTP handler, so a caller that talked to the CORE `ChatBrain`
directly — the standalone TUI `research/runners/brain_chat_tui.py` — got the recall+moat core but MISSED every
coupling. The pipeline is now the module-level `webapp.server.brain_reply(chat, req, source, cache_key)`, and this
module is the thin, documented surface every OTHER caller reaches it through, so the FastAPI endpoint, the OpenAI
`/v1/chat/completions` shim, and the standalone TUI all run the IDENTICAL coupling sequence.

⭐ DISCIPLINE (standing rule going forward): a faculty coupling belongs in the SHARED pipeline
(`webapp.server.brain_reply`), NEVER inline in a request handler. Adding a coupling to the handler alone silently
regresses the TUI + the shim back to a partial brain. Wire it in the shared pipeline so every surface gets it at
once.

The pipeline body LIVES in `webapp/server.py` because it is deeply bound to that module's ~40 session-state dicts +
organ-cache helpers; keeping it there made the extraction a pure move (byte-identical webapp path). `server.brain_reply`
returns the same Starlette `JSONResponse` the handler always returned; the helpers here decode it to the response
PAYLOAD DICT (exactly as the OpenAI shim already does), which is what a non-HTTP caller (the TUI) wants.
"""
from __future__ import annotations

import json as _json


def payload_of(response) -> dict:
    """Decode a Starlette JSONResponse (what `webapp.server.brain_reply` / `brain_chat` return) to its payload
    dict — the SAME extraction the OpenAI shim performs (`json.loads(bytes(r.body))`)."""
    return _json.loads(bytes(response.body))


def make_request(message, *, session="default", brain="tiny-demo", renderer=None,
                 rich=None, reset=False, percept=None):
    """Build a BrainChatRequest without the caller importing webapp.server's request model directly."""
    from webapp.server import BrainChatRequest
    return BrainChatRequest(session=session, message=message, brain=brain, renderer=renderer,
                            rich=rich, reset=reset, percept=percept)


def brain_reply(chat, req, source, cache_key) -> dict:
    """Run the FULL faculty-drive pipeline on an already-built ChatBrain and return the response PAYLOAD DICT.

    Delegates to `webapp.server.brain_reply` (the single shared pipeline the `/api/brain-chat` handler runs) and
    decodes its JSONResponse to a dict. `req` is a BrainChatRequest (see `make_request`); `cache_key` keys the
    per-session organ/state dicts."""
    from webapp.server import brain_reply as _impl
    return payload_of(_impl(chat, req, source, cache_key))


def turn(req) -> dict:
    """Build/reuse the session-cached ChatBrain for `req` and run the full pipeline; returns the payload dict.

    The dict form of the `/api/brain-chat` handler (`webapp.server.brain_chat`)."""
    from webapp.server import brain_chat as _handler
    return payload_of(_handler(req))


def reply_over_chat(chat, message, *, source="tui-brain", brain="tiny-demo", renderer=None,
                    rich=False, session="tui-session", percept=None) -> dict:
    """Run the shared full-faculty pipeline over a ChatBrain the CALLER already built (the TUI's entry point).

    Returns the response payload dict (answer + every faculty-coupling field). The couplings' per-session state
    lives under a stable `cache_key` derived from (session, brain, renderer) — the SAME shape server.py uses."""
    rname = renderer if renderer is not None else "raw"
    req = make_request(message, session=session, brain=brain, renderer=rname, rich=rich, percept=percept)
    cache_key = (session, brain, rname)
    return brain_reply(chat, req, source, cache_key)
