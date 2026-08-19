"""Verify the board-#79 DA-MODE-DRIVES-THE-RESPONSE wiring THROUGH the real /api/brain-chat handler (in-process).
The anti-hollow-integration proof: the #76 spiking DA-mode (rest/focus/arousal) is LOAD-BEARING on the live turn.
(A) the MODE TRACKS the conversation (a dull/greeting opening reads REST; an engaging/novel exchange reads
    FOCUS/AROUSAL; the engagement suffix appears iff the mode is focus/arousal).
(B) the MODE DRIVES the response: message held FIXED, INDUCE the DA mode (focus vs rest via the SNc afferent) ->
    the reply's engagement suffix DIFFERS, and that difference VANISHES under the neural lesion (SNc nucleus
    silenced) -> the coupling rides the SPIKING SNc->DA read, not a host `if engagement>x`. Content byte-identical.
(C) NO-REGRESSION on content: the moat/recall/abstain verdict is byte-identical off-vs-on; the mode changes only
    the surface (an optional engagement suffix), never a fact; a rest/neutral ON-minus-key == OFF byte-identical.
Runs all three in ONE process (shares the heavy first-turn warmup). Usage: python verify_da_drives.py"""
import os, json, hashlib, subprocess, time, random
import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

# ISOLATE THE FACULTY FOR A TRACTABLE IN-PROCESS VERIFY. The OTHER default-on organs (Gate-B affect, the affect-DRIVES
# lead #84, the swap-DRIVES lead #85, worldmodel, surprise, metacog, comprehension, ...) are ORTHOGONAL to da-drives
# and make each turn slow; disabling them (a consistent baseline across ALL arms) keeps the SAME /api/brain-chat
# handler + the SAME recall/moat core while dropping per-turn cost. da-drives reads its own #76 substrate + APPENDS an
# engagement suffix regardless of the others, so this isolation cannot change any da-drives verdict. In particular
# BRAIN_AFFECT_DRIVES=0 + BRAIN_SWAP_DRIVES=0 so neither lead co-mingles with the DA suffix in the answer string.
if os.environ.get("DA_VERIFY_FULL_CONFIG", "0").strip().lower() not in ("1", "true", "on", "yes"):
    for _k in ("BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE",
               "BRAIN_METACOG", "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF",
               "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE",
               "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER", "BRAIN_GNW_SWAP"):
        os.environ[_k] = "0"

from webapp.server import brain_chat, BrainChatRequest  # the REAL handler

_ART = os.environ.get("DA_DRIVES_JSON", "research/findings/raw/_da_drives_chat/verify.json")
_RENDERER = os.environ.get("DA_VERIFY_RENDERER", "stub")   # GPU-free deterministic surface
_RESULTS = {"runner": "verify_da_drives (in-process /api/brain-chat)", "backend": os.environ.get("SIM_BACKEND"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "part_a": {}, "part_b": {}, "part_c": {}}
try:
    _RESULTS["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    _RESULTS["git_sha"] = None


def turn(session, message, reset=False, brain="tiny-demo"):
    # rich=False -> the single-fact path (chat.gate + chat.render), STATELESS + fast. The da-drives suffix is wired
    # into BOTH return paths; this exercises the fast one. renderer=stub -> GPU-free deterministic answer surface.
    resp = brain_chat(BrainChatRequest(session=session, message=message, brain=brain, reset=reset, rich=False,
                                       renderer=_RENDERER))
    return json.loads(bytes(resp.body))


def _md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _content(d):
    """The honesty-floor content fields (must be mode-INVARIANT): abstain/recall/verify verdict + source/brain."""
    return {k: d.get(k) for k in ("abstained", "recalled_svo", "verified", "brain", "source", "rich")}


def _seed_all(s):
    np.random.seed(s); random.seed(s)
    try:
        import cupy as cp
        cp.random.seed(s)
    except Exception:
        pass


def _clear_env():
    for k in ("BRAIN_DA_DRIVES", "BRAIN_DA_DRIVES_LESION", "BRAIN_DA_DRIVES_INDUCE"):
        os.environ.pop(k, None)


def _suffix_of(d):
    return (d.get("da_drives") or {}).get("lead", "") or ""


def _mode_of(d):
    return (d.get("da_drives") or {}).get("mode", "")


def _da_of(d):
    return float((d.get("da_drives") or {}).get("da_level", 0.0) or 0.0)


# ── (A) the MODE TRACKS the conversation: a dull/greeting opening reads REST (no suffix); an engaging/novel exchange
#    reads FOCUS/AROUSAL (+ the engagement suffix). "engaged" = mode in {focus,arousal}; "rest" = mode rest, no suffix.
CONV = [
    ("hi",                                                        "rest"),     # content-free -> e=0 -> REST
    ("ok",                                                        "rest"),     # content-free -> HOLD 0 -> REST
    ("what does the dog chase",                                   "engaged"),  # novel content -> FOCUS
    ("what colour is the sky ocean mountain forest river",        "engaged"),  # more novel/rich -> FOCUS/AROUSAL
    ("photosynthesis chloroplast thylakoid electron transport",   "engaged"),  # very novel+rich -> AROUSAL
]


def part_a():
    print("=" * 96)
    print("(A) THE DA MODE TRACKS THE CONVERSATION through /api/brain-chat  (BRAIN_DA_DRIVES default-on)")
    print("=" * 96)
    _clear_env()
    rows, ok = [], True
    das_rest, das_engaged, modes = [], [], []
    for i, (msg, kind) in enumerate(CONV):
        _seed_all(200 + i)
        d = turn("conv_a", msg, reset=(i == 0))
        dd = d.get("da_drives") or {}
        mode = dd.get("mode", "")
        suffix = dd.get("lead", "")
        da = float(dd.get("da_level", 0.0) or 0.0)
        ans = d.get("answer", "") or ""
        engaged = mode in ("focus", "arousal")
        suffix_iff_engaged = (bool(suffix) == engaged)
        suffix_appended = ans.endswith(suffix) if suffix else True
        if kind == "rest":
            row_ok = (mode == "rest") and (suffix == "") and suffix_iff_engaged
            das_rest.append(da)
        else:  # engaged
            row_ok = engaged and bool(suffix) and suffix_iff_engaged and suffix_appended
            das_engaged.append(da)
        modes.append(mode)
        ok = ok and row_ok
        rows.append({"i": i, "msg": msg, "kind": kind, "mode": mode, "da_level": da,
                     "ema_engagement": dd.get("ema_engagement"), "afferent_pA": dd.get("afferent_pA"),
                     "suffix": suffix, "answer": ans, "row_ok": row_ok})
        print("  [%d] %-52r kind=%-8s mode=%-8s DA=%.3f suffix=%-40r %s"
              % (i, msg[:52], kind, mode, da, suffix, "OK" if row_ok else "**MISMATCH**"))
    mode_moves = (len(das_rest) >= 1 and len(das_engaged) >= 1)
    reached_arousal = ("arousal" in modes)
    monotone = (min(das_engaged) > max(das_rest)) if (das_engaged and das_rest) else False
    suffix_iff_all = all((bool(r["suffix"]) == (r["mode"] in ("focus", "arousal"))) for r in rows)
    print("\n  modes=%s   rest-DA<=%.3f   engaged-DA>=%.3f   reached_arousal=%s   suffix-iff-engaged(all)=%s"
          % (modes, max(das_rest) if das_rest else -1, min(das_engaged) if das_engaged else -1,
             reached_arousal, suffix_iff_all))
    a_ok = ok and mode_moves and monotone and reached_arousal and suffix_iff_all
    print("  (A) %s" % ("PASS" if a_ok else "FAIL"))
    _RESULTS["part_a"] = {"pass": bool(a_ok), "modes": modes, "mode_moves": bool(mode_moves),
                          "monotone_engaged_gt_rest": bool(monotone), "reached_arousal": bool(reached_arousal),
                          "suffix_iff_engaged_all": bool(suffix_iff_all),
                          "max_rest_da": max(das_rest) if das_rest else None,
                          "min_engaged_da": min(das_engaged) if das_engaged else None, "rows": rows}
    return a_ok


# ── (B) the MODE DRIVES the response (the crux): MESSAGE FIXED, INDUCE the DA mode (focus vs rest) -> the engagement
#    suffix differs; the neural lesion (SNc nucleus silenced) collapses the level -> REST -> the suffix VANISHES. ──
FIXED_PROBE = "what does the dog chase?"   # a real recall on tiny-demo (a stable answered fact)
INDUCE_FOCUS = "1300"    # SNc afferent (pA) -> self-produced DA high -> AROUSAL/FOCUS band -> suffix present
INDUCE_REST = "100"      # SNc afferent (pA) -> self-produced DA low  -> REST band -> no suffix


def _induced_probe(session, induce_pa, lesion):
    # Fresh session (reset=True) so the workspace is clean -> no cross-arm leak (the #84 persistence lesson). The
    # INDUCE afferent drives the SNc directly with the MESSAGE HELD FIXED; the neural SNc->DA read still runs.
    _seed_all(7); _clear_env()
    os.environ["BRAIN_DA_DRIVES_INDUCE"] = str(induce_pa)
    if lesion:
        os.environ["BRAIN_DA_DRIVES_LESION"] = "1"
    d = turn(session, FIXED_PROBE, reset=True)
    _clear_env()
    return d


def part_b():
    print("\n" + "=" * 96)
    print("(B) THE DA MODE DRIVES THE RESPONSE — message FIXED, induce focus vs rest; lesion collapses the difference")
    print("=" * 96)
    # INTACT: SAME fixed probe, two INDUCED modes. focus-induce -> suffix present; rest-induce -> no suffix. All arms
    # REUSE one session "b" (each _induced_probe resets it -> only one ChatBrain resident at a time: the #84 memory lesson).
    d_focus = _induced_probe("b", INDUCE_FOCUS, lesion=False)
    d_rest = _induced_probe("b", INDUCE_REST, lesion=False)
    suf_focus, suf_rest = _suffix_of(d_focus), _suffix_of(d_rest)
    ans_focus, ans_rest = d_focus.get("answer", "") or "", d_rest.get("answer", "") or ""
    mode_focus, mode_rest = _mode_of(d_focus), _mode_of(d_rest)
    da_focus, da_rest = _da_of(d_focus), _da_of(d_rest)
    # the fact under the suffix (strip the appended suffix from the end) must equal the rest-arm answer.
    base_focus = ans_focus[:-len(suf_focus)] if suf_focus and ans_focus.endswith(suf_focus) else ans_focus
    intact_diff = (ans_focus != ans_rest) and (suf_focus != suf_rest)
    focus_suffix_ok = bool(suf_focus) and mode_focus in ("focus", "arousal")
    rest_no_suffix = (suf_rest == "") and (mode_rest in ("rest", "neutral"))
    base_same = (base_focus == ans_rest)
    content_same = (_md5(_content(d_focus)) == _md5(_content(d_rest)))
    print("  INTACT  focus-induce(%spA): mode=%-8s DA=%.3f suffix=%-40r ans=%r"
          % (INDUCE_FOCUS, mode_focus, da_focus, suf_focus, ans_focus[:56]))
    print("          rest-induce(%spA):  mode=%-8s DA=%.3f suffix=%-40r ans=%r"
          % (INDUCE_REST, mode_rest, da_rest, suf_rest, ans_rest[:56]))
    print("          intact_diff=%s focus_suffix_ok=%s rest_no_suffix=%s base_identical=%s content_identical=%s"
          % (intact_diff, focus_suffix_ok, rest_no_suffix, base_same, content_same))
    # LESION: silence the SNc nucleus -> the SAME focus-induce afferent can NO LONGER raise DA -> mode REST ->
    # the suffix VANISHES -> the answer reverts to the byte-identical no-suffix base. So the suffix RIDES the SNc read.
    d_focusL = _induced_probe("b", INDUCE_FOCUS, lesion=True)
    suf_focusL = _suffix_of(d_focusL)
    ans_focusL = d_focusL.get("answer", "") or ""
    mode_focusL = _mode_of(d_focusL)
    da_focusL = _da_of(d_focusL)
    lesion_suffix_gone = (suf_focusL == "")
    lesion_rest_mode = (mode_focusL in ("rest", "neutral"))
    lesion_equals_base = (ans_focusL == base_focus == ans_rest)
    print("  LESION  focus-induce(%spA)+SNc-silenced: mode=%-8s DA=%.3f suffix=%-8r ans=%r"
          % (INDUCE_FOCUS, mode_focusL, da_focusL, suf_focusL, ans_focusL[:56]))
    print("          suffix_gone=%s rest_mode=%s ==base=%s" % (lesion_suffix_gone, lesion_rest_mode, lesion_equals_base))
    _clear_env()
    b_ok = (intact_diff and focus_suffix_ok and rest_no_suffix and base_same and content_same
            and lesion_suffix_gone and lesion_rest_mode and lesion_equals_base)
    print("  (B) %s   [intact engagement suffix PRESENT and it VANISHES under the neural SNc-lesion]"
          % ("PASS" if b_ok else "FAIL"))
    _RESULTS["part_b"] = {"pass": bool(b_ok), "intact_diff": bool(intact_diff), "suffix_focus": suf_focus,
                          "suffix_rest": suf_rest, "ans_focus": ans_focus, "ans_rest": ans_rest,
                          "mode_focus": mode_focus, "mode_rest": mode_rest, "da_focus": da_focus, "da_rest": da_rest,
                          "focus_suffix_ok": bool(focus_suffix_ok), "rest_no_suffix": bool(rest_no_suffix),
                          "base_identical": bool(base_same), "content_identical": bool(content_same),
                          "lesion_suffix_gone": bool(lesion_suffix_gone), "lesion_rest_mode": bool(lesion_rest_mode),
                          "lesion_equals_base": bool(lesion_equals_base), "mode_lesion": mode_focusL,
                          "da_lesion": da_focusL}
    return b_ok


# ── (C) NO-REGRESSION: content mode-INVARIANT + byte-identical-off. Two panels. ──────────────────────────────────────
# C1 (NEUTRAL byte-identity) runs a REST panel (content-free/greeting turns -> mode rest -> no suffix every turn) so
# ON-minus-key == OFF. C2 (content-invariance under an ACTIVE mode) shows the content fields are identical off vs a
# focus-induce vs a rest-induce (only the suffix differs, the fact does not).
C1_PANEL = ["hi", "ok", "no"]   # content-free tokens (all filtered / stoplisted) -> engagement 0 -> REST -> no suffix


def part_c():
    print("\n" + "=" * 96)
    print("(C) NO-REGRESSION — content mode-invariant + byte-identical-off")
    print("=" * 96)
    any_off_key = False
    all_on_key = True
    neutral_byte_ok = True
    c1_rows = []
    print("  -- C1 rest byte-identity (a rest/neutral turn adds no suffix) --")
    print("  %-20s | %-9s %-9s | keys | ON-mode ON-suffix | byte" % ("message", "OFF md5", "ON md5"))
    first = True
    for i, msg in enumerate(C1_PANEL):
        _seed_all(1300 + i); _clear_env(); os.environ["BRAIN_DA_DRIVES"] = "0"
        d_off = turn("c1_off", msg, reset=first)
        _seed_all(1300 + i); _clear_env()   # default-on
        d_on = turn("c1_on", msg, reset=first)
        first = False
        has_off_key = "da_drives" in d_off
        has_on_key = "da_drives" in d_on
        any_off_key = any_off_key or has_off_key
        all_on_key = all_on_key and has_on_key
        on_suffix = _suffix_of(d_on)
        on_mode = _mode_of(d_on)
        d_on_stripped = {k: v for k, v in d_on.items() if k != "da_drives"}
        h_off, h_on_strip = _md5(d_off), _md5(d_on_stripped)
        byte_ok = (h_off == h_on_strip)
        neutral_byte_ok = neutral_byte_ok and byte_ok and (not has_off_key) and has_on_key and (on_suffix == "")
        c1_rows.append({"i": i, "msg": msg, "off_md5": h_off, "on_stripped_md5": h_on_strip, "byte_ok": byte_ok,
                        "off_has_key": has_off_key, "on_has_key": has_on_key, "on_suffix": on_suffix,
                        "on_mode": on_mode, "off_answer": d_off.get("answer"), "on_answer": d_on.get("answer")})
        print("  %-20s | %-9s %-9s | off=%s on=%s | %-6s %-8r | %s"
              % (msg[:20], h_off[:8], h_on_strip[:8], has_off_key, has_on_key, on_mode, on_suffix,
                 "IDENTICAL" if byte_ok else "**DIFFERS**"))
    # C2 content-invariance under an ACTIVE mode: the SAME fixed probe, off vs focus-induce vs rest-induce -> the
    # CONTENT fields (abstain/recall/verify) are identical (the suffix differs, the fact does not).
    content_invariant = True
    c2_rows = []
    print("  -- C2 content-invariance under an ACTIVE mode (content unchanged, only the engagement suffix) --")
    _seed_all(55); _clear_env(); os.environ["BRAIN_DA_DRIVES"] = "0"
    d_off = turn("c2", FIXED_PROBE, reset=True)
    _clear_env()
    d_fc = _induced_probe("c2", INDUCE_FOCUS, lesion=False)   # focus-induce (suffix present)
    d_rt = _induced_probe("c2", INDUCE_REST, lesion=False)    # rest-induce (no suffix)
    cm_off, cm_fc, cm_rt = _md5(_content(d_off)), _md5(_content(d_fc)), _md5(_content(d_rt))
    cinv = (len({cm_off, cm_fc, cm_rt}) == 1)
    content_invariant = content_invariant and cinv
    c2_rows.append({"msg": FIXED_PROBE, "content_md5": cm_off, "content_invariant": cinv,
                    "off_answer": d_off.get("answer"), "focus_answer": d_fc.get("answer"),
                    "rest_answer": d_rt.get("answer"), "focus_suffix": _suffix_of(d_fc),
                    "rest_suffix": _suffix_of(d_rt)})
    print("  %-30s | content-md5 {off,focus,rest}=%s | focus-suffix=%r | %s"
          % (FIXED_PROBE[:30], cm_off[:8], _suffix_of(d_fc), "OK" if cinv else "**REGRESS**"))
    _clear_env()
    print("\n  OFF ever carried da_drives key: %s (want False)" % any_off_key)
    print("  ON  always carried da_drives key: %s (want True)" % all_on_key)
    print("  rest ON-minus-key == OFF (byte) every turn: %s" % neutral_byte_ok)
    print("  content fields invariant across {off,focus,rest}: %s" % content_invariant)
    c_ok = neutral_byte_ok and content_invariant and (not any_off_key) and all_on_key
    print("  (C) %s" % ("PASS" if c_ok else "FAIL"))
    _RESULTS["part_c"] = {"pass": bool(c_ok), "off_ever_has_key": bool(any_off_key), "on_always_has_key": bool(all_on_key),
                          "neutral_byte_identical_all": bool(neutral_byte_ok), "content_invariant_all": bool(content_invariant),
                          "c1_rows": c1_rows, "c2_rows": c2_rows}
    return c_ok


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    c = part_c()
    preconditions = [
        {"name": "(A) the DA mode tracks the conversation (rest opening, engaged/novel reaches arousal, suffix iff engaged)",
         "ok": bool(_RESULTS["part_a"].get("pass"))},
        {"name": "(B) message-fixed: inducing focus vs rest changes the reply suffix, with content identical",
         "ok": bool(_RESULTS["part_b"].get("intact_diff") and _RESULTS["part_b"].get("content_identical")
                    and _RESULTS["part_b"].get("base_identical"))},
        {"name": "(B) the neural lesion (SNc nucleus silenced) collapses the DA level -> the engagement suffix VANISHES -> == base",
         "ok": bool(_RESULTS["part_b"].get("lesion_suffix_gone") and _RESULTS["part_b"].get("lesion_rest_mode")
                    and _RESULTS["part_b"].get("lesion_equals_base"))},
        {"name": "(C) content fields mode-invariant across {off,focus,rest}",
         "ok": bool(_RESULTS["part_c"].get("content_invariant_all"))},
        {"name": "(C) byte-identical-off: OFF never carries the key; a rest ON-minus-key == OFF",
         "ok": bool(_RESULTS["part_c"].get("neutral_byte_identical_all")
                    and not _RESULTS["part_c"].get("off_ever_has_key"))},
    ]
    _RESULTS["preconditions"] = preconditions
    all_pre = all(p["ok"] for p in preconditions)
    verdict = "GO" if (a and b and c and all_pre) else "NO-GO"
    _RESULTS["verdict"] = verdict
    os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
    with open(_ART, "w") as f:
        json.dump(_RESULTS, f, indent=2, default=str)
    print("\n" + "=" * 96)
    print("VERDICT  (A) tracks=%s  (B) drives+lesion=%s  (C) no-regression=%s  => %s" % (a, b, c, verdict))
    for p in preconditions:
        print("   [%s] %s" % ("PASS" if p["ok"] else "FAIL", p["name"]))
    print("wrote %s" % _ART)
    print("=" * 96)
    raise SystemExit(0 if (a and b and c) else 1)
