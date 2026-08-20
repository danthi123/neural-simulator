"""OpenAI-API-compatible shim for the sim brain (2026-08-19 reframe — the two-surface UX).

Exposes the developed sim-brain over the OpenAI `/v1/chat/completions` + `/v1/models` contract so ANY standard LLM
client (Open WebUI, LibreChat, etc.) can talk to it with no custom UI. Two surfaces ride the one response:
  * the BRAIN'S REPLY  -> `choices[0].message.content`   (the conversation surface: person <-> brain)
  * the INTERNAL MONOLOGUE -> `choices[0].message.reasoning_content`  (the brain's live internal state, which modern
    clients render in a collapsible "thinking" panel — visible ALONGSIDE the reply by default, as requested).

THE MONOLOGUE IS AN HONEST FUNCTIONAL READ-OUT (the 2026-08-19 state-fidelity honesty boundary made VISIBLE): it
reports the brain's REAL internal spiking-derived state each turn (mood, decision-confidence, surprise, the held
topic, curiosity, a topic-swap, a prospective intention) as functional statements — NEVER a claim of phenomenal
experience. It is the transparency surface for "its words track its state." It surfaces the per-turn faculty state
today; once the continuous background-tick engine lands, the same read-out also shows BETWEEN-turn activity
(wandering, evolving mood, consolidation).

This is host TRANSPORT only (request/response marshalling + formatting) — it computes NO cognition; every field it
reports is read from `brain_chat`'s existing spiking-faculty metadata. No `sim/` edit; additive; the existing
`/api/brain-chat` endpoint is untouched.
"""
from __future__ import annotations

import time


def _fmt_num(x, nd=2):
    try:
        return ("%+.*f" % (nd, float(x)))
    except Exception:
        return str(x)


def format_internal_monologue(resp: dict, inner_life: list | None = None) -> str:
    """Turn brain_chat's response metadata into an honest functional internal-state read-out (the 'thinking' stream).

    Every line is a functional statement of a real internal signal, never a phenomenal claim. Missing faculties are
    simply omitted (a faculty that did not fire this turn contributes nothing). `inner_life` (from the continuous-
    state engine) is the between-turn activity that happened WHILE the user was away — surfaced first so you see the
    brain was 'alive between questions'."""
    lines = []

    # CONTINUOUS-STATE ENGINE: what happened between turns (the brain running while idle)
    if inner_life:
        last = inner_life[-1]
        n = len(inner_life)
        note = last.get("note") or "state evolved"
        lines.append("while you were away (%d idle tick%s): %s" % (n, "" if n == 1 else "s", note))

    # AFFECT / mood (Gate-B affect ladder + the #84 tone drive)
    aff = resp.get("affect") or {}
    ad = resp.get("affect_drives") or {}
    if isinstance(aff, dict) and aff.get("on") and "error" not in aff:
        diff = aff.get("differential")
        sign = aff.get("valence_sign")
        val = aff.get("appraisal_valence")
        aro = aff.get("appraisal_arousal")
        mood_word = {"+": "positive", "-": "negative", "0": "neutral"}.get(sign, "neutral")
        seg = "mood reads %s" % mood_word
        if val is not None:
            seg += " (valence %s, arousal %s)" % (_fmt_num(val), _fmt_num(aro))
        if diff is not None:
            seg += " [spiking differential %s]" % _fmt_num(diff, 3)
        lines.append(seg)
    if isinstance(ad, dict) and ad.get("lead"):
        lines.append("felt state is coloring my tone: \"%s\"" % str(ad.get("lead")).strip())

    # METACOG confidence (E1 balance-of-evidence)
    mc = resp.get("metacog") or {}
    if isinstance(mc, dict) and mc.get("on"):
        conf = "confident" if mc.get("confident") else "low-confidence"
        bal = mc.get("balance")
        seg = "my decision-margin reads this recall as %s" % conf
        if bal is not None:
            seg += " (margin %s vs threshold %s)" % (_fmt_num(bal, 3), _fmt_num(mc.get("threshold"), 3))
        lines.append(seg)

    # SURPRISE / expectation-violation (D2)
    su = resp.get("surprise") or {}
    if isinstance(su, dict) and su.get("surprised"):
        lines.append("that violates what I'd stored — my surprise unit fired")

    # HELD TOPIC / thought (thought-swap workspace + WM)
    sw = resp.get("gnw_swap") or {}
    if isinstance(sw, dict) and sw.get("swapped"):
        lines.append("a more salient input arrived — I swapped the thought I was holding")
    held = resp.get("held_topic") or resp.get("wm_referents")
    if held:
        lines.append("holding in mind: %s" % (", ".join(map(str, held)) if isinstance(held, (list, tuple)) else str(held)))

    # WORLD-MODEL forward prediction (E2)
    wm = resp.get("worldmodel") or {}
    if isinstance(wm, dict) and wm.get("on") and wm.get("pred_sign") is not None:
        lines.append("my forward model expected a %s next" % {"+": "positive", "-": "negative", "0": "neutral"}.get(str(wm.get("pred_sign")), "neutral"))

    # CURIOSITY (D3)
    cu = resp.get("curiosity") or {}
    if isinstance(cu, dict) and cu.get("craved"):
        lines.append("this is a novel topic — my curiosity drive wants to ask a follow-up")

    # PROSPECTIVE intention (BA10 latch)
    pm = resp.get("prospective") or {}
    if isinstance(pm, dict) and (pm.get("fired") or pm.get("latched")):
        lines.append("prospective intention: %s" % (pm.get("note") or ("fired a held reminder" if pm.get("fired") else "latched a deferred intention")))

    # ABSTAIN / moat (the honesty core)
    if resp.get("abstained"):
        lines.append("I have no grounded trace for this — I'm declining rather than confabulating")
    elif resp.get("hypothesis"):
        lines.append("I'm volunteering this as a guess, flagged (not a stored fact)")
    elif resp.get("recalled_svo"):
        lines.append("recalled from my store: %s" % " ".join(map(str, resp.get("recalled_svo"))))

    if not lines:
        lines.append("(no faculty fired strongly this turn)")
    return "\n".join("· " + ln for ln in lines)


def _last_user_message(messages) -> str:
    for m in reversed(messages or []):
        if (m or {}).get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):  # OpenAI content-parts form
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            return str(c or "")
    return ""


MODEL_ID = "sim-brain"


def models_list() -> dict:
    """OpenAI /v1/models — advertise the sim brain as one model so clients can select it."""
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "neural-simulator"}
    ]}


def chat_completion_object(reply: str, reasoning: str, model: str, created: int) -> dict:
    """OpenAI non-streaming ChatCompletion. `reasoning_content` carries the internal monologue (the 'thinking' panel)."""
    return {
        "id": "simbrain-%d" % created,
        "object": "chat.completion",
        "created": created,
        "model": model or MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply, "reasoning_content": reasoning},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def stream_chunks(reply: str, reasoning: str, model: str, created: int):
    """OpenAI streaming (SSE) generator. Emits the internal monologue on the `reasoning_content` delta channel FIRST
    (so the client's thinking panel fills), then the reply on `content`, then the terminator. The brain produces the
    whole reply at once, so we stream it as a small number of deltas — valid SSE that Open WebUI et al. render as a
    thinking section followed by the answer. Clients that ignore `reasoning_content` still get a correct reply."""
    import json as _json
    cid = "simbrain-%d" % created

    def _chunk(delta, finish=None):
        return "data: " + _json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": model or MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    yield _chunk({"role": "assistant"})
    if reasoning:
        # chunk the monologue by line so the thinking panel streams line-by-line
        for ln in reasoning.split("\n"):
            yield _chunk({"reasoning_content": ln + "\n"})
    yield _chunk({"content": reply})
    yield _chunk({}, finish="stop")
    yield "data: [DONE]\n\n"
