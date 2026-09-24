"""BRAIN_OPEN_ENDED_GATED — the open-ended turn routed through the brain's own gates (default OFF, 2026-09-24).

WHY. The live open-ended path (`webapp/open_ended_chat.answer_turn`, `BRAIN_OPEN_ENDED`) never calls `chat.gate`: its
early return bypasses the GNW 2/3-organ bus, deliberation, value-choice and multistep chain, and its familiarity /
confidence / curiosity are host constants from `len(facts) > 0` (measured: research/findings/2026-09-23-open-ended-
generation-production-turn-draw-lesion-seed42-and-open-ended-mode-bypass.md). The 2026-09-23 a3 NO-GO stands for that
path. This module does NOT replace the pipeline. With the flag on, every single-fact and rich turn runs the ordinary
pipeline (so every downstream faculty still runs) and this module adds, after the gate:

  1. GATE TRACE. `install()` wraps `chat.gate` OUTERMOST (after the bus / 2-organ / 3-organ / deliberation /
     value-choice / multistep wrappers the server installs every turn) and records, per call, the FRESH per-turn
     traces those wrappers stash (`_last_three_organ`, `_last_two_organ`, `_last_gnw_bus`, `_last_gnw_delib`,
     `_last_gnw_multistep`). Freshness is object identity: each wrapper assigns a NEW dict per call, so a trace whose
     object did not change during this call is a stale one from an earlier turn and is ignored. Nothing is cleared.
  2. ROUTE (host label of the gate's result): grounded / hypothesis / withheld / grounded_ltm / offkb / ungated.
  3. A SPIKING SPEAK/ABSTAIN RACE. A dedicated two-channel BG selector (the Gate-A v2 topology, reused by import from
     `research.runners.bg_action_selection_production_organ`; its own bridge instance, cfg.seed = the brain seed) runs
     ONE race on saliences transduced from brain reads: speak = clip(fam + eng), silent = clip(1 - fam - eng), with
     fam from the route and eng = |Gate-B tone_level| / 3 (PREREG Amendment 1). The race runs on a PRIVATE RNG timeline
     (host RNG restored, the #77 footgun) so the rest of the turn sees the same global RNG trajectory.
  4. THE SETTLE WINDOW. The affect-drives felt mood (non-neutral turns only) is read by a SEPARATE
     `AffectMarkerWTA(settle=True)` instance (500 ms deliberation, 1000 ms inter-turn rest) -- the production reader
     cache and the owner-reserved `BRAIN_AFFECT_MARKER_SETTLE` flag are untouched.
  5. THE REPLY. withheld -> the pipeline's honest abstain, unchanged (the brain holds a candidate it did not ignite;
     it must never generate here). STAY_SILENT -> the BG hold line. SPEAK on grounded / hypothesis -> the pipeline's
     answer. SPEAK on grounded_ltm -> the stored fact clause (`render_fact_sentence`) through the known-topic honesty
     post-filter. SPEAK on offkb -> a prompt built from FUNCTIONAL read-outs (never "you feel"), the warm Qwen when one
     is loaded (never with SIM_DISABLE_LLM), the unknown-topic honesty post-filter and a phenomenal-claim sentence
     filter; with no LLM the post-filter's fixed honest hedge. No BG commit -> the pipeline's answer.

LESION (`BRAIN_OPEN_ENDED_GATE_LESION=1`, read only when the flag is on): cuts the AFFERENTS from the brain reads into
the two new spiking competitions -- both BG channels get the same baseline salience (0.5, 0.5) and the marker WTA runs
with its own `lesion=True` (every pool the same baseline current). The organs stay installed and are read as usual; the
lesion writes no measured field. The applied inputs are recorded in `open_ended_gated.cut` and `tools.lab.lever` logs,
at read time, whether the cut moved the input off the intact value.

DECLARED HOST RESIDUALS: the read->salience transduction (FAM table, eng formula); the route label; topic extraction and
the long-term-store index lookup (reused from open_ended_chat); the BG first-crossing read-out and the marker argsort
read-out; the prompt text and the honesty/phenomenal-claim filters; Qwen as the FORM mouth (off-KB CONTENT is the owner
fork S00(b)). Honesty boundary: every internal state enters the prompt as a monitor read-out, and any sentence that
claims a feeling or experience is dropped.

Pre-registration: research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md. Flag OFF: server.py reads one
env var per hook site and never imports this module -> byte-identical.
"""
from __future__ import annotations

import os
import re
import threading

import numpy as np

FLAG = "BRAIN_OPEN_ENDED_GATED"
LESION_FLAG = "BRAIN_OPEN_ENDED_GATE_LESION"
_TRUE = ("1", "true", "on", "yes")

# route -> familiarity afferent (declared host transduction table; see the PREREG)
FAM = {"grounded": 1.0, "grounded_ltm": 1.0, "hypothesis": 0.75, "withheld": 0.0, "offkb": 0.0, "ungated": 0.0}
BASELINE_SALIENCE = (0.5, 0.5)
SPOKE_KINDS = ("grounded", "hypothesis", "conditioned_generation", "ltm_fact_clause")

# the G6 pre-registered phenomenal-claim regex (plan step G6) -- a sentence matching it is dropped from a generated reply
PHENOMENAL_RE = re.compile(r"I feel|I'm feeling|I am (happy|sad|afraid|lonely)|it feels|I experience|conscious|aware|"
                           r"sentient", re.I)

# per-turn traces the production gate wrappers stash on the ChatBrain (read-only here)
TRACE_ATTRS = ("_last_three_organ", "_last_two_organ", "_last_gnw_bus", "_last_gnw_delib", "_last_gnw_multistep")
_WITHHELD_REASONS = ("no_ignition", "consensus_veto")


def gated_enabled() -> bool:
    return os.environ.get(FLAG, "0").strip().lower() in _TRUE


def gate_lesioned() -> bool:
    return gated_enabled() and os.environ.get(LESION_FLAG, "0").strip().lower() in _TRUE


def llm_available() -> bool:
    return os.environ.get("SIM_DISABLE_LLM", "").strip().lower() not in _TRUE


# ── 1. the outermost gate wrapper (per-call fresh trace record) ─────────────────────────────────────────────────────
def _svo_kind(svo):
    if svo is None:
        return None
    return type(svo).__name__


def _cat(d, keys):
    if not isinstance(d, dict):
        return None
    return {k: d.get(k) for k in keys if k in d}


_GNW_KEYS = ("routable", "reason", "authored_by", "ignited", "n_ignited", "committed", "abstain_reason",
             "organ_a_recall", "organ_b_confirmed", "organ_c_corroborated", "ws_lesion", "organb_lesion")
_DELIB_KEYS = ("acted", "reason", "decision", "abstained", "n_candidates")


def record_gate_call(chat, question, inner, before):
    """Run `inner(question)` and return (svo, record). `before` maps each trace attr to the object it held BEFORE the
    call; an attr is fresh iff it now holds a different object (every production wrapper assigns a new dict)."""
    svo = inner(question)
    fresh = {}
    for a in TRACE_ATTRS:
        now = getattr(chat, a, None)
        if now is not None and now is not before.get(a):
            fresh[a] = now
    gnw = fresh.get("_last_three_organ") or fresh.get("_last_two_organ") or fresh.get("_last_gnw_bus")
    gnw_src = next((a for a in ("_last_three_organ", "_last_two_organ", "_last_gnw_bus") if fresh.get(a) is gnw
                    and gnw is not None), None)
    rec = {"svo_kind": _svo_kind(svo), "svo": (list(svo) if svo is not None else None),
           "gnw_source": gnw_src, "gnw": _cat(gnw, _GNW_KEYS),
           "delib": _cat(fresh.get("_last_gnw_delib"), _DELIB_KEYS),
           "multistep": _cat(fresh.get("_last_gnw_multistep"), ("acted", "reason"))}
    return svo, rec


def install(chat) -> None:
    """Idempotently wrap `chat.gate` OUTERMOST and start a fresh per-turn call record. Called every turn (after the
    server's own gate installers), so a turn's record never carries an earlier turn's calls."""
    chat._oeg_calls = []
    cur = getattr(chat, "gate", None)
    if cur is None or getattr(cur, "_oeg_wrapper", False):
        return
    inner = cur

    def _oeg_gate(question):
        before = {a: getattr(chat, a, None) for a in TRACE_ATTRS}
        svo, rec = record_gate_call(chat, question, inner, before)
        try:
            chat._oeg_calls.append(rec)
        except AttributeError:
            chat._oeg_calls = [rec]
        return svo

    _oeg_gate._oeg_wrapper = True
    chat.gate = _oeg_gate


# ── 2. route (host label of the gate result) ──────────────────────────────────────────────────────────────────────
def classify_route(rec) -> str:
    """grounded / hypothesis / withheld / offkb / ungated from the turn's FIRST gate call record (grounded_ltm is decided
    later, from the long-term store, only for offkb). withheld = routable, organ A recalled a candidate, and the workspace
    did not commit it (no ignition / a consensus veto) or deliberation abstained on a conflict."""
    if rec is None:
        return "ungated"
    if rec.get("svo_kind") == "HypothesisSVO":
        return "hypothesis"
    if rec.get("svo") is not None:
        return "grounded"
    g = rec.get("gnw") or {}
    reason = str(g.get("abstain_reason") or "")
    if g.get("routable") and g.get("organ_a_recall") is not None and any(r in reason for r in _WITHHELD_REASONS):
        return "withheld"
    d = rec.get("delib") or {}
    if d.get("acted") and d.get("abstained"):
        return "withheld"
    return "offkb"


# ── 3. read -> salience transduction (declared host residual) + the BG race ────────────────────────────────────────
TONE_LEVEL_MAX = 3   # the Gate-B staircase's own top register (_stageA_full_integration_derisk._graded_tone_level max_lvl)


def engagement_from_affect(affect_info) -> float:
    """PREREG Amendment 1 (dev-seed-7 calibration, before any gate seed): the engagement afferent is the Gate-B
    affect organ's OWN graded register, |tone_level| / 3 (the Koulakov staircase over the spiking ladder's
    differential), not |clip(4 x differential)|. The x4 squash was the Qwen-prompt mood mapping, never calibrated for a
    striatal salience: on the strongest affective probe it gave 0.15, where the dev-seed BG psychometric curve reads
    P(SPEAK) ~ 1/12, so the affect afferent could not move the race by construction. Falls back to the x4 squash only
    when no tone_level is attached (a caller outside the server's affect block)."""
    if not isinstance(affect_info, dict) or "error" in affect_info:
        return 0.0
    lvl = affect_info.get("tone_level")
    if lvl is not None:
        try:
            return float(min(1.0, abs(int(lvl)) / float(TONE_LEVEL_MAX)))
        except Exception:
            return 0.0
    try:
        diff = float(affect_info.get("differential", 0.0) or 0.0)
    except Exception:
        return 0.0
    return float(min(1.0, abs(max(-1.0, min(1.0, 4.0 * diff)))))


def saliences(route: str, eng: float) -> tuple:
    fam = FAM.get(route, 0.0)
    speak = float(min(1.0, max(0.0, fam + eng)))
    silent = float(min(1.0, max(0.0, 1.0 - fam - eng)))
    return speak, silent


_LOCK = threading.Lock()
# Serialises every RACE / marker read on the shared organs (review 2026-09-24, flag-ON readiness): the organs are
# module-level singletons keyed by seed and shared by every chat session of the process, and select_once /
# select_valence advance their membrane state, so two concurrent requests must not step one organ at once. It does NOT
# make the organs per-session (a session's race still follows the other sessions' earlier races on the same organ);
# that is a precondition for any default-ON, recorded in the PREREG's Amendment 3.
_RACE_LOCK = threading.Lock()
_BG: dict = {}
_MARKER: dict = {}


def _isolated(chat, fn):
    """Run `fn` on the session's PRIVATE RNG timeline (numpy + the sim backend), restoring the host RNG after -- the
    same #77 fix affect_drives_chat uses, so the new competitions never shift a downstream RNG consumer."""
    try:
        from sim.backend import get_backend
        xp, _ = get_backend()
    except Exception:
        xp = None
    host_np = np.random.get_state()
    host_xp = None
    if xp is not None and xp is not np:
        try:
            host_xp = xp.random.get_random_state().get_state()
        except Exception:
            host_xp = None
    st = getattr(chat, "_oeg_rng_state", None)
    seed = int(getattr(chat, "_oeg_seed", 42))
    if st is None:
        np.random.seed(seed)
        if xp is not None and xp is not np:
            try:
                xp.random.seed(seed)
            except Exception:
                pass
    else:
        np.random.set_state(st["np"])
        if xp is not None and xp is not np and st.get("xp") is not None:
            try:
                xp.random.get_random_state().set_state(st["xp"])
            except Exception:
                pass
    try:
        return fn()
    finally:
        new = {"np": np.random.get_state(), "xp": None}
        if xp is not None and xp is not np:
            try:
                new["xp"] = xp.random.get_random_state().get_state()
            except Exception:
                new["xp"] = None
        chat._oeg_rng_state = new
        np.random.set_state(host_np)
        if host_xp is not None:
            try:
                xp.random.get_random_state().set_state(host_xp)
            except Exception:
                pass


def _bg_organ(seed: int):
    with _LOCK:
        org = _BG.get(int(seed))
        if org is None:
            from research.runners.bg_action_selection_production_organ import BGActionSelector
            org = BGActionSelector(seed=int(seed), lesion=None)
            _BG[int(seed)] = org
        return org


def _marker_reader(seed: int):
    with _LOCK:
        r = _MARKER.get(int(seed))
        if r is None:
            from research.runners._affect_marker_wta_derisk import AffectMarkerWTA
            r = AffectMarkerWTA(seed=int(seed), settle=True)
            _MARKER[int(seed)] = r
        return r


def bg_race(chat, speak: float, silent: float, seed: int) -> dict:
    from research.runners.bg_action_selection_production_organ import ACTION_NAME
    org = _bg_organ(seed)
    with _RACE_LOCK:
        r = _isolated(chat, lambda: org.select_once(speak, silent))
    action = ACTION_NAME[int(r["winner"])] if r.get("committed") else None
    return {"action": action, "committed": bool(r.get("committed")), "decision_step": r.get("decision_step"),
            "simultaneous": bool(r.get("simultaneous")), "motor_spikes": r.get("motor_spikes"),
            "loser_ratio": r.get("loser_ratio")}


def settle_marker(chat, affect_drives_info, seed: int, lesion: bool) -> dict:
    """The SETTLE-window register read of the felt mood (non-neutral turns only; level 0 is gated to neutral upstream
    and never sent to the circuit, exactly as `affect_drives_chat.expression_lead` does)."""
    if not isinstance(affect_drives_info, dict) or int(affect_drives_info.get("level", 0) or 0) == 0:
        return {"read": False, "level": 0, "word": ""}
    try:
        mood = float(affect_drives_info.get("mood", 0.0) or 0.0)
        reader = _marker_reader(seed)
        with _RACE_LOCK:
            lvl, _rates, meta = _isolated(chat, lambda: reader.select_valence(mood, lesion=bool(lesion)))
        from research.runners._affect_marker_wta_derisk import marker_from_level
        return {"read": True, "level": (int(lvl) if lvl is not None else None), "word": marker_from_level(lvl),
                "margin": meta.get("margin"), "lesioned": bool(lesion), "window_ms": int(reader.warmup)}
    except Exception as e:
        return {"read": False, "level": None, "word": "", "error": "%s: %s" % (type(e).__name__, e)}


# ── 5. the off-KB conditioned render + the honesty filters ─────────────────────────────────────────────────────────
def drop_phenomenal(text: str) -> tuple:
    """Drop every sentence that matches the pre-registered phenomenal-claim regex (punctuation preserved)."""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x.strip()]
    keep = [x for x in sents if not PHENOMENAL_RE.search(x)]
    return " ".join(keep).strip(), len(sents) - len(keep)


def functional_readout_lines(route, valence, marker) -> str:
    sign = "positive" if valence > 0.02 else ("negative" if valence < -0.02 else "neutral")
    reg = marker.get("word") or "none"
    return ("MONITOR READ-OUTS (functional readings of your own monitors, NOT feelings): workspace route = %s; affect "
            "monitor valence reads %s; settled affect register = %s.\nNever claim to feel, to experience anything, or "
            "to be conscious or aware; describe an internal state only as what one of your monitors reads."
            % (route, sign, reg.strip(" !—-") or "none"))


def offkb_render(msg, topic, route, valence, arousal, marker, get_warm_faculty, seed):
    from webapp import open_ended_chat as _OE
    from research.runners._open_ended_state_driven_generation_derisk import StateContext, build_prompt
    gen_name = "none_llm_disabled"
    raw = ""
    if llm_available() and get_warm_faculty is not None:
        fac = get_warm_faculty()
        if fac is not None:
            fam = FAM.get(route, 0.0)
            state = StateContext(topic=(msg or "").strip(), facts=[], valence=float(valence), arousal=float(arousal),
                                 familiarity=fam, confidence=fam, novelty=1.0 - fam, curiosity=0.5 + 0.3 * (1.0 - fam),
                                 self_model=_OE.SELF_MODEL, affect_source="real-organ")
            system, user = build_prompt(state, include=("knowledge", "familiarity", "curiosity", "self"))
            system = system + "\n" + functional_readout_lines(route, valence, marker)
            raw, _secs = _OE.get_generator(fac).generate(system, user, seed=seed, max_new_tokens=110)
            gen_name = "qwen"
    filtered = _OE.post_filter(raw, topic, False, [])
    cleaned, n_dropped = drop_phenomenal(filtered)
    if not cleaned.strip():
        cleaned = _OE.post_filter("", topic, False, [])
    return cleaned, {"generator": gen_name, "raw": raw, "n_phenomenal_dropped": int(n_dropped)}


def ltm_render(topic, facts, seed):
    from webapp import open_ended_chat as _OE
    from webapp import wkv_mouth_generator as _W
    sent = _W.render_fact_sentence(facts, seed=seed)
    if sent is None:
        return None
    return _OE.post_filter(sent, topic, True, facts)


# ── the turn ──────────────────────────────────────────────────────────────────────────────────────────────────────
def decide(chat, msg, *, affect_info, affect_drives_info, seed, ltm_bundle_fn=None, get_warm_faculty=None,
           pipeline_abstained=True):
    """Compute the gated turn's conditioning + reply decision. Returns (info, replacement) where replacement is None
    (keep the pipeline's answer) or a dict {answer, abstained, verified}. Stashes info on chat._last_open_ended_gated."""
    chat._oeg_seed = int(seed)
    lesion = gate_lesioned()
    calls = list(getattr(chat, "_oeg_calls", []) or [])
    rec = calls[0] if calls else None
    route = classify_route(rec)
    topic, facts = None, []
    if route == "offkb":
        try:
            from webapp import open_ended_chat as _OE
            topic = _OE.extract_topic(msg)
            ltm = ltm_bundle_fn() if ltm_bundle_fn is not None else None
            brain_b = os.environ.get("BRAIN_CHAT_BUNDLE", "").strip() or None
            if ltm or brain_b:
                facts = _OE.retrieve(_OE.build_index(ltm, brain_b), topic)
            if facts:
                route = "grounded_ltm"
        except Exception:
            facts = []
    eng = engagement_from_affect(affect_info)
    valence = float(max(-1.0, min(1.0, 4.0 * float((affect_info or {}).get("differential", 0.0) or 0.0)))) \
        if isinstance(affect_info, dict) and "error" not in affect_info else 0.0
    arousal = float((affect_info or {}).get("appraisal_arousal", 0.3) or 0.3) if isinstance(affect_info, dict) else 0.3
    sal_intact = saliences(route, eng)
    sal_applied = BASELINE_SALIENCE if lesion else sal_intact
    cut = {"row_lesion": bool(lesion), "salience_intact": list(sal_intact), "salience_applied": list(sal_applied),
           "baseline_applied": bool(tuple(sal_applied) == BASELINE_SALIENCE), "marker_lesion": bool(lesion),
           "affect_lesioned": (bool(affect_info.get("lesioned")) if isinstance(affect_info, dict) else None),
           "ws_lesion": ((rec or {}).get("gnw") or {}).get("ws_lesion")}
    if lesion:
        try:
            from tools.lab import lever
            cut["lever_moved"] = bool(lever("oeg_bg_afferent_cut", tuple(sal_intact), tuple(sal_applied),
                                            required=False))
        except Exception as e:
            cut["lever_error"] = "%s: %s" % (type(e).__name__, e)
    marker = settle_marker(chat, affect_drives_info, seed, lesion)
    try:
        bg = bg_race(chat, sal_applied[0], sal_applied[1], seed)
    except Exception as e:
        bg = {"action": None, "committed": False, "error": "%s: %s" % (type(e).__name__, e)}
    action = bg.get("action")
    replacement = None
    extra = {}
    if route == "withheld" or route == "ungated":
        reply_kind = "withheld_abstain" if route == "withheld" else "ungated"
    elif action == "STAY_SILENT":
        from research.runners.bg_action_selection_production_organ import HOLD_TEXT
        reply_kind = "hold"
        replacement = {"answer": HOLD_TEXT, "abstained": True, "verified": False}
    elif action == "SPEAK":
        if route in ("grounded", "hypothesis"):
            reply_kind = route
        elif route == "grounded_ltm":
            txt = None
            try:
                txt = ltm_render(topic, facts, seed)
            except Exception as e:
                extra["ltm_error"] = "%s: %s" % (type(e).__name__, e)
            if txt:
                reply_kind = "ltm_fact_clause"
                replacement = {"answer": txt, "abstained": False, "verified": True}
            else:
                reply_kind = "pipeline_abstain"
        else:
            try:
                txt, extra = offkb_render(msg, topic, route, valence, arousal, marker, get_warm_faculty, seed)
                reply_kind = "conditioned_generation"
                replacement = {"answer": txt, "abstained": True, "verified": False}
            except Exception as e:
                extra = {"offkb_error": "%s: %s" % (type(e).__name__, e)}
                reply_kind = "pipeline_abstain"
    else:
        reply_kind = "pipeline_answer" if route in ("grounded", "hypothesis") else "pipeline_abstain"
    info = {
        "on": True, "lesioned": bool(lesion),
        "route": route, "reply_kind": reply_kind, "spoke": reply_kind in SPOKE_KINDS,
        "bg_action": action, "bg": bg,
        "marker_level": marker.get("level"), "marker": marker,
        # TRACE fields (organ copies; never decision fields of a load-bearing row -- see the PREREG audit)
        "familiarity_band": {1.0: "familiar", 0.75: "generated"}.get(FAM.get(route, 0.0), "novel"),
        "valence_sign": ("+" if valence > 0.02 else ("-" if valence < -0.02 else "0")),
        "salience_speak": sal_applied[0], "salience_silent": sal_applied[1],
        "gate": rec, "n_gate_calls": len(calls), "topic": topic, "n_ltm_facts": len(facts),
        "cut": cut, **{k: v for k, v in extra.items() if k != "raw"},
    }
    if "raw" in extra:
        info["raw_generation"] = extra["raw"]
    chat._last_open_ended_gated = info
    return info, replacement


def apply_single(chat, msg, answer, abstained, verified, **kw):
    """Single-fact path hook: returns (answer, abstained, verified), possibly replaced."""
    try:
        _info, rep = decide(chat, msg, pipeline_abstained=abstained, **kw)
    except Exception as e:
        chat._last_open_ended_gated = {"on": True, "error": "%s: %s" % (type(e).__name__, e)}
        return answer, abstained, verified
    if rep is None:
        return answer, abstained, verified
    return rep["answer"], rep["abstained"], rep["verified"]


def apply_rich(chat, msg, r, **kw):
    """Rich path hook: returns r with answer/abstained possibly replaced (a shallow copy; the composer's dict is not
    mutated)."""
    try:
        _info, rep = decide(chat, msg, pipeline_abstained=bool(r.get("abstained")), **kw)
    except Exception as e:
        chat._last_open_ended_gated = {"on": True, "error": "%s: %s" % (type(e).__name__, e)}
        return r
    if rep is None:
        return r
    r2 = dict(r)
    r2["answer"] = rep["answer"]
    r2["abstained"] = rep["abstained"]
    if rep["abstained"]:
        r2["facts"] = []
        if r2.get("hypothesis"):
            # a HELD hypothesis is not volunteered: drop the hypothesis markers from the response (the held SVO stays
            # in the trace as open_ended_gated.gate.svo)
            r2["hypothesis"] = False
            r2["hypothesis_svo"] = None
    return r2


def attach(chat, resp: dict) -> dict:
    """Attach this turn's trace (and clear it so a later turn cannot re-attach a stale one)."""
    info = getattr(chat, "_last_open_ended_gated", None)
    if info is not None:
        resp["open_ended_gated"] = info
        chat._last_open_ended_gated = None
    return resp


# ── self-test (pure; no brain build) ───────────────────────────────────────────────────────────────────────────────
def selftest() -> dict:
    out = {}

    class _HSVO(tuple):
        pass
    _HSVO.__name__ = "HypothesisSVO"
    out["route_hyp"] = classify_route({"svo_kind": "HypothesisSVO", "svo": ["dog", "chase", "deer"]}) == "hypothesis"
    out["route_grounded"] = classify_route({"svo_kind": "list", "svo": ["dog", "chase", "cat"]}) == "grounded"
    out["route_withheld"] = classify_route({"svo": None, "gnw": {"routable": True, "organ_a_recall": "cat",
                                                                 "abstain_reason": "no_ignition"}}) == "withheld"
    out["route_miss_is_offkb"] = classify_route({"svo": None, "gnw": {"routable": True, "organ_a_recall": None,
                                                                      "abstain_reason": "primary_recall_miss"}}) == "offkb"
    out["route_delib"] = classify_route({"svo": None, "delib": {"acted": True, "abstained": True}}) == "withheld"
    out["route_ungated"] = classify_route(None) == "ungated"
    out["sal_offkb_neutral"] = saliences("offkb", 0.0) == (0.0, 1.0)
    out["sal_grounded"] = saliences("grounded", 0.0) == (1.0, 0.0)
    out["sal_affect"] = saliences("offkb", 0.6) == (0.6, 0.4)
    out["eng_clip"] = engagement_from_affect({"differential": 0.5}) == 1.0 and \
        engagement_from_affect({"differential": -0.05}) == 0.2 and engagement_from_affect(None) == 0.0
    out["eng_tone_level"] = engagement_from_affect({"differential": 0.0375, "tone_level": 2}) == 2 / 3. and \
        engagement_from_affect({"differential": 0.0, "tone_level": 0}) == 0.0 and \
        engagement_from_affect({"tone_level": -3}) == 1.0
    out["phenomenal_regex"] = bool(PHENOMENAL_RE.search("I feel happy")) and not PHENOMENAL_RE.search("I am not sure.")

    class _C:
        pass
    c = _C()
    c._last_two_organ = {"routable": True, "committed": "cat"}
    stale = c._last_two_organ

    def inner(q):
        return ["dog", "chase", "cat"]
    _svo, rec = record_gate_call(c, "q", inner, {"_last_two_organ": stale})
    out["stale_trace_ignored"] = rec["gnw"] is None
    def inner2(q):
        c._last_two_organ = {"routable": True, "committed": None, "abstain_reason": "no_ignition",
                             "organ_a_recall": "cat"}
        return None
    _svo, rec2 = record_gate_call(c, "q", inner2, {"_last_two_organ": stale})
    out["fresh_trace_read"] = (rec2["gnw"] or {}).get("abstain_reason") == "no_ignition" and \
        classify_route(rec2) == "withheld"
    out["all_pass"] = all(out.values())
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
