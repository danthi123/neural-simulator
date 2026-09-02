"""VERIFY: R1 rung 2 -- an open-ended (BRAIN_OPEN_ENDED) turn now runs the shared pipeline's PER-TURN SESSION-STATE
WRITERS, closing the rest of the "open-ended bypasses the shared pipeline's session-state writes" debt (board #199).

WHAT CHANGED (webapp/server.py, inside the existing `if os.environ.get("BRAIN_OPEN_ENDED", ...)` guard, strictly
BETWEEN rung-1's D5 episodic write and the pre-existing `return _safe_json_response(_oe_resp, ...)`): after building
`_oe_resp`, ADDITIONALLY run the SAME per-turn faculty state-writers a NORMAL turn runs below the open-ended block --
affect-drives (`affect_drives_chat.observe_turn`, the #84 felt body-state EMA on `chat._affect_drives_workspace`),
affective-ToM, DA-mode, common-ground (`common_ground_drives_chat.observe_turn`, the per-referent audience-design
ledger, cache_key-keyed), and the D3 discourse register's per-turn FOLD (`d3_..._organ.note_turn`, part i only). Each
call is flag-gated + try/excepted, and its returned tone/reference LEAD is DISCARDED (state-write side effect ONLY),
so `_oe_resp` / the free-talk surface stay byte-identical. `_SESSION_MOOD` (~4539) + the D5 episodic write (rung-1)
already ran for an open-ended turn; this adds the remaining named writers.

THIS RUNNER PROVES, CHEAPLY (no real Qwen -- the warm-Qwen loader + `open_ended_chat.answer_turn` are monkeypatched;
numpy backend), FOUR things, each PATCHED-vs-ORIGINAL (git stash of webapp/server.py ONLY):

  (1) BYTE-IDENTICAL WHEN OFF. `BRAIN_OPEN_ENDED` unset -> two SEPARATE processes (patched file, and the pre-change
      file via `git stash`) call `brain_reply` on the SAME single-fact turn; the JSON responses diff byte-for-byte.
      Structurally guaranteed (every new line sits inside the already-existing `if BRAIN_OPEN_ENDED truthy` block).
  (2) OPEN-ENDED SURFACE PRESERVED. `BRAIN_OPEN_ENDED=1` -> the open-ended JSON response is byte-identical patched
      vs original (the new code only adds STATE-WRITE side effects after `_oe_resp` is built).
  (3) THE STATE WRITES NOW HAPPEN (affect-drives + common-ground). Two open-ended turns re-mentioning a referent
      ("I'm thrilled about the dog!") drive the changed `brain_reply`; AFTER, the #84 body-state workspace exists
      with a body-state MOVED off neutral, and the common-ground organ for that cache_key shows n_turns=2 (+ a
      grounded slot). The SAME two turns on the ORIGINAL file leave NO workspace (affect-drives never ran) and the
      organ at n_turns=0 (never observed) -- the load-bearing before/after.
  (4) THE DISCOURSE FOLD NOW HAPPENS. One open-ended discourse-clause turn ("dog chase cat"), with a real spiking
      event register attached to `chat.agent`, drives the changed `brain_reply`; AFTER, `_SESSION_DISCOURSE[cache_key]`
      shows `heard_any=True` (the clause folded into the register). On the ORIGINAL file the key is absent (the fold
      never ran).

  python -m research.runners._open_ended_pipeline_state_r2_verify off          # -> RESPONSE:: line
  python -m research.runners._open_ended_pipeline_state_r2_verify on           # -> RESPONSE:: + AFFECT_STATE:: + CG_STATE::
  python -m research.runners._open_ended_pipeline_state_r2_verify on_discourse # -> RESPONSE:: + DISCOURSE_STATE::
  python -m research.runners._open_ended_pipeline_state_r2_verify report       # -> the full before/after artifact
"""
from __future__ import annotations
import json, os, sys, time, types
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "off")
# Switch OFF every faculty that runs in the 4200-4600 PREFIX (before the open-ended block) so a MOCK chat never trips
# an unrelated faculty's attribute expectations -- EXACTLY as rung-1's verify did. The rung-2 TARGETS (affect-drives,
# common-ground, discourse) are enabled per-phase below; none of them run in the prefix (they sit BELOW the open-ended
# block, so on an open-ended turn ONLY the new state-write block reaches them).
for _k, _v in {
    "BRAIN_AFFECT": "0", "BRAIN_AFFECT_DRIVES": "0", "BRAIN_CG_DRIVES": "0", "BRAIN_DISCOURSE_REGISTER": "0",
    "BRAIN_AFFECTIVE_TOM": "0", "BRAIN_DA_DRIVES": "0", "BRAIN_GNW_2ORGAN": "0", "BRAIN_GNW_3ORGAN": "0",
    "BRAIN_GNW_BUS": "0", "BRAIN_VALUE_CHOICE": "0", "BRAIN_SWAP_DRIVES": "0", "BRAIN_SELF_INITIATE": "0",
    "BRAIN_VISION_IDENTITY": "0", "BRAIN_BG_SELECT": "0", "BRAIN_GNW_MULTISTEP": "0",
}.items():
    os.environ.setdefault(_k, _v)
import logging
logging.disable(logging.INFO)
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _decode(resp) -> dict:
    body = resp.body
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    return json.loads(body)


def _mock_chat(agent=None):
    c = types.SimpleNamespace()
    c.agents_set = {"dog", "cat", "brain"}
    c.actions_set = {"chase", "eat", "use", "learn", "store", "run"}
    c.patients_set = {"cat", "fish", "spikes", "words", "memory"}
    c.renderer = None
    c.inner = None
    c.agent = agent
    c.gate = lambda question: None
    c._brain_chat_source = "mock"
    return c


def _install_fake_qwen(WS, OE):
    class _FakeFac:
        pass

    class _FakeWarmRenderer:
        _fac = _FakeFac()

    WS._get_warm_qwen_renderer = lambda: _FakeWarmRenderer()

    def _fake_answer_turn(msg, warm_faculty, valence, arousal, *, ltm_bundle, brain_bundle, chat=None,
                          seed: int = 42, max_new_tokens: int = 110) -> dict:
        return {
            "answer": "Dogs chase cats around here.", "raw": "Dogs chase cats around here.",
            "filtered": "Dogs chase cats around here.", "topic": "dog", "known": True,
            "facts": [["dog", "chase", "cat"]], "n_sentences": 1, "gen_seconds": 0.0,
            "gen_time_honesty_used": False, "gen_time_trace": None, "generator": "qwen",
            "wkv_mouth_used": False, "fact_clause_used": False,
            "state": {"valence": float(valence), "arousal": float(arousal), "familiarity": 0.9,
                      "novelty": 0.1, "curiosity": 0.53},
        }

    OE.answer_turn = _fake_answer_turn


def run_off():
    import webapp.server as WS
    os.environ.pop("BRAIN_OPEN_ENDED", None)
    cache_key = ("_r2verify_off_session", "tiny-demo", "raw")
    chat = _mock_chat()
    req = WS.BrainChatRequest(session=cache_key[0], message="what does the dog chase?",
                              brain=cache_key[1], renderer=cache_key[2], rich=False)
    resp = WS.brain_reply(chat, req, "mock", cache_key)
    print("RESPONSE::" + json.dumps(_decode(resp), sort_keys=True))


def run_on():
    import webapp.server as WS
    from webapp import open_ended_chat as OE
    os.environ["BRAIN_OPEN_ENDED"] = "1"
    os.environ["BRAIN_AFFECT_DRIVES"] = "1"    # rung-2 target: #84 felt body-state EMA
    os.environ["BRAIN_CG_DRIVES"] = "1"        # rung-2 target: per-referent common-ground ledger
    _install_fake_qwen(WS, OE)

    cache_key = ("_r2verify_on_session", "tiny-demo", "raw")
    chat = _mock_chat()
    # two open-ended turns, SAME referent 'dog', an AFFECTIVE message so the #84 body-state genuinely moves off neutral
    for _ in range(2):
        req = WS.BrainChatRequest(session=cache_key[0], message="I'm thrilled about the dog!",
                                  brain=cache_key[1], renderer=cache_key[2], rich=False)
        resp = WS.brain_reply(chat, req, "mock", cache_key)
    print("RESPONSE::" + json.dumps(_decode(resp), sort_keys=True))

    # ── read back #84 affect-drives body-state (attached to chat) ──
    ws = getattr(chat, "_affect_drives_workspace", None)
    print("AFFECT_STATE::" + json.dumps({
        "workspace_exists": ws is not None,
        "body_h": getattr(ws, "h", None), "body_a": getattr(ws, "a", None),
        "moved_off_neutral": bool(ws is not None and (abs(getattr(ws, "h", 0.5) - 0.5) > 1e-6
                                                       or abs(getattr(ws, "a", 0.0)) > 1e-6)),
    }))
    # ── read back the common-ground ledger organ for the SAME cache_key ──
    import research.runners.common_ground_ledger_production_organ as CGL
    org = CGL.get_organ(cache_key, seed=42, lesion=False)
    print("CG_STATE::" + json.dumps({
        "n_turns": int(getattr(org, "n_turns", -1)),
        "grounded_slots": len(getattr(org, "_grounded_slots", []) or []),
    }))


def run_on_discourse():
    import webapp.server as WS
    from webapp import open_ended_chat as OE
    os.environ["BRAIN_OPEN_ENDED"] = "1"
    os.environ.pop("BRAIN_DISCOURSE_REGISTER", None)   # default-ON
    os.environ["BRAIN_AFFECT_DRIVES"] = "0"
    os.environ["BRAIN_CG_DRIVES"] = "0"
    _install_fake_qwen(WS, OE)

    # build a REAL spiking discourse register and attach it to chat.agent (what the ChatBrain's MultiTurnAgent carries)
    import research.runners.d3_discourse_event_register_production_organ as DR
    reg = DR.make_discourse_register(["dog", "cat", "maria", "sam"], seed=42)
    agent = types.SimpleNamespace(_event_register=reg)
    cache_key = ("_r2verify_disc_session", "tiny-demo", "raw")
    chat = _mock_chat(agent=agent)
    req = WS.BrainChatRequest(session=cache_key[0], message="dog chase cat",   # a 3-token SVO discourse clause
                              brain=cache_key[1], renderer=cache_key[2], rich=False)
    resp = WS.brain_reply(chat, req, "mock", cache_key)
    print("RESPONSE::" + json.dumps(_decode(resp), sort_keys=True))

    dstate = WS._SESSION_DISCOURSE.get(cache_key)
    print("DISCOURSE_STATE::" + json.dumps({
        "session_key_present": dstate is not None,
        "heard_any": bool(dstate.get("heard_any")) if dstate else False,
        "who_now": (reg.who_agent() if hasattr(reg, "who_agent") else None),
    }))


def _run_subprocess(phase: str, extra_env: dict | None = None) -> str:
    import subprocess
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run([sys.executable, "-u", "-m",
                        "research.runners._open_ended_pipeline_state_r2_verify", phase],
                       cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


def _extract(tag: str, text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(tag):
            return line[len(tag):]
    return None


def report():
    import subprocess
    from tools.verdict import Verdict
    from tools.lab import attributable_to

    def _git(*args):
        subprocess.run(["git", *args], cwd=str(_REPO), check=True, capture_output=True, text=True)

    # ---- OFF: byte-identical response, patched vs original --------------------------------------------------------
    off_after = _run_subprocess("off")
    resp_off_after = _extract("RESPONSE::", off_after)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        off_before = _run_subprocess("off")
    finally:
        _git("stash", "pop")
    resp_off_before = _extract("RESPONSE::", off_before)
    off_byte_identical = bool(resp_off_after) and (resp_off_after == resp_off_before)

    # ---- ON (affect-drives + common-ground): surface byte-identical + the state writes now fire -------------------
    on_after = _run_subprocess("on")
    resp_on_after = _extract("RESPONSE::", on_after)
    affect_after = _extract("AFFECT_STATE::", on_after)
    cg_after = _extract("CG_STATE::", on_after)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        on_before = _run_subprocess("on")
    finally:
        _git("stash", "pop")
    resp_on_before = _extract("RESPONSE::", on_before)
    affect_before = _extract("AFFECT_STATE::", on_before)
    cg_before = _extract("CG_STATE::", on_before)

    # ---- ON (discourse fold) -------------------------------------------------------------------------------------
    disc_after_out = _run_subprocess("on_discourse")
    resp_disc_after = _extract("RESPONSE::", disc_after_out)
    disc_after = _extract("DISCOURSE_STATE::", disc_after_out)
    _git("stash", "push", "--", "webapp/server.py")
    try:
        disc_before_out = _run_subprocess("on_discourse")
    finally:
        _git("stash", "pop")
    disc_before = _extract("DISCOURSE_STATE::", disc_before_out)

    aff_a = json.loads(affect_after) if affect_after else {}
    aff_b = json.loads(affect_before) if affect_before else {}
    cg_a = json.loads(cg_after) if cg_after else {}
    cg_b = json.loads(cg_before) if cg_before else {}
    dsc_a = json.loads(disc_after) if disc_after else {}
    dsc_b = json.loads(disc_before) if disc_before else {}

    on_surface_byte_identical = bool(resp_on_after) and (resp_on_after == resp_on_before)
    disc_surface_ok = bool(resp_disc_after)   # the discourse turn still returns a valid open-ended surface
    affect_now_writes = (aff_a.get("moved_off_neutral") is True and aff_b.get("workspace_exists") is False)
    cg_now_writes = (int(cg_a.get("n_turns", -1)) == 2 and int(cg_b.get("n_turns", -1)) == 0)
    disc_now_writes = (dsc_a.get("heard_any") is True and dsc_b.get("heard_any") is False)

    art = {
        "probe": "open_ended_pipeline_state_r2_rung2",
        "off_path": {"byte_identical": off_byte_identical,
                     "response_before": resp_off_before, "response_after": resp_off_after},
        "on_path_affect_cg": {"response_byte_identical": on_surface_byte_identical,
                              "affect_before": aff_b, "affect_after": aff_a,
                              "cg_before": cg_b, "cg_after": cg_a},
        "on_path_discourse": {"surface_ok": disc_surface_ok,
                              "discourse_before": dsc_b, "discourse_after": dsc_a},
    }
    v = Verdict("R1 rung 2: an open-ended turn now runs the shared pipeline's per-turn SESSION-STATE writers; "
                "the OFF path + the open-ended surface are unperturbed")
    v.require("OFF path (BRAIN_OPEN_ENDED unset): brain_reply's JSON response is byte-identical, patched vs "
              "original server.py, through the REAL single-fact pipeline", off_byte_identical, expect=True)
    v.require("ON path: the open-ended JSON response surface is byte-identical patched vs original (the new code "
              "only adds state-write side effects after _oe_resp is built)", on_surface_byte_identical, expect=True)
    v.require("ON path, affect-drives: PATCHED leaves a #84 body-state workspace MOVED off neutral; ORIGINAL "
              "leaves NO workspace (the faculty never ran on the open-ended turn)", affect_now_writes, expect=True)
    v.require("ON path, common-ground: PATCHED leaves the ledger organ at n_turns=2 for the cache_key; ORIGINAL "
              "at n_turns=0 (never observed)", cg_now_writes, expect=True)
    v.require("ON path, discourse: PATCHED folds the clause -> _SESSION_DISCOURSE[cache_key].heard_any=True; "
              "ORIGINAL leaves it False (the fold never ran)", disc_now_writes, expect=True)
    v.control("affect-drives body-state MOVED off neutral after the open-ended turn: PATCHED vs ORIGINAL",
              treatment=1.0 if aff_a.get("moved_off_neutral") else 0.0,
              control=1.0 if aff_b.get("moved_off_neutral") else 0.0, min_separation=0.5,
              note="the affect body-state write is attributable to this change")
    v.control("common-ground ledger n_turns after two open-ended turns: PATCHED vs ORIGINAL",
              treatment=float(cg_a.get("n_turns", 0)), control=float(cg_b.get("n_turns", 0)), min_separation=1.5,
              note="the ledger write is attributable to this change")
    # ATTRIBUTION (tools.lab): whose is the state-write difference? The control here is the ORIGINAL server.py
    # (this change reverted), which holds EVERYTHING else fixed and varies ONLY the rung-2 block -- so the write
    # is fully attributable to it. (Both arms were measured; this makes the subtraction explicit, not implied.)
    art["attribution"] = {
        "affect_moved_off_neutral": attributable_to("affect body-state write (patched vs original server.py)",
                                                     1.0 if aff_a.get("moved_off_neutral") else 0.0,
                                                     1.0 if aff_b.get("moved_off_neutral") else 0.0),
        "cg_n_turns": attributable_to("common-ground ledger n_turns (patched vs original server.py)",
                                      float(cg_a.get("n_turns", 0)), float(cg_b.get("n_turns", 0))),
    }
    go = (off_byte_identical and on_surface_byte_identical and affect_now_writes and cg_now_writes
          and disc_now_writes and disc_surface_ok)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["preconditions"] = decided.get("preconditions", [])
    art["GO"] = bool(go)
    out = _REPO / "research" / "findings" / "raw" / "2026-09-02-open-ended-pipeline-state-r2-rung2-verify.json"
    out.write_text(json.dumps(art, indent=1))
    print(json.dumps({"off_byte_identical": off_byte_identical,
                      "on_surface_byte_identical": on_surface_byte_identical,
                      "affect_now_writes": affect_now_writes, "cg_now_writes": cg_now_writes,
                      "disc_now_writes": disc_now_writes, "GO": go}, indent=1))
    print(f"wrote {out} -> {decided['status']}")
    return decided["status"]


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "off"
    if phase == "off":
        run_off()
    elif phase == "on":
        run_on()
    elif phase == "on_discourse":
        run_on_discourse()
    elif phase == "report":
        report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; pass 'off', 'on', 'on_discourse', or 'report'")


if __name__ == "__main__":
    main()
