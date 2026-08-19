"""Verify the board-#84 AFFECT-DRIVES-THE-RESPONSE wiring THROUGH the real /api/brain-chat handler (in-process).
The anti-hollow-integration proof: the #81 graded-affect ladder read is LOAD-BEARING on the live turn.
(A) affect TRACKS the conversation (mood moves sensibly with emotional content, + persists across a neutral probe).
(B) affect DRIVES the response: message held FIXED, vary the affect state -> the reply's affective lead DIFFERS,
    and that difference VANISHES under the neural lesion (intero->ladder cut) -> the coupling rides the SPIKING read.
(C) NO-REGRESSION on content: the moat/recall/abstain verdict is byte-identical off-vs-on; affect changes only tone.
Runs all three in ONE process (shares the heavy first-turn warmup). Usage: python verify_affect_drives.py"""
import os, json, hashlib, subprocess, time, random
import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

# ISOLATE THE FACULTY FOR A TRACTABLE IN-PROCESS VERIFY. The OTHER default-on organs (Gate-B affect = a 25k-neuron
# brain stepped every turn, worldmodel, surprise, metacog, comprehension, ...) are ORTHOGONAL to affect-drives and
# make each turn ~40-60s; disabling them (a consistent baseline across ALL arms) keeps the SAME /api/brain-chat
# handler + the SAME recall/moat core while dropping per-turn cost to a few seconds. affect-drives reads its own #81
# ladder + prepends a lead regardless of the others, so this isolation cannot change any affect-drives verdict.
# (A) + (B) were ALSO confirmed on the FULL default-organ config — see full_default_config_AB_transcript.txt.
if os.environ.get("AFFECT_VERIFY_FULL_CONFIG", "0").strip().lower() not in ("1", "true", "on", "yes"):
    for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_COMPREHENSION_GATE",
               "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE",
               "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM",
               "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER"):
        os.environ[_k] = "0"

from webapp.server import brain_chat, BrainChatRequest  # the REAL handler

_ART = os.environ.get("AFFECT_DRIVES_JSON", "research/findings/raw/_affect_drives_chat/verify.json")
_RESULTS = {"runner": "verify_affect_drives (in-process /api/brain-chat)", "backend": os.environ.get("SIM_BACKEND"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "part_a": {}, "part_b": {}, "part_c": {}}
try:
    _RESULTS["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    _RESULTS["git_sha"] = None

_POS_WORDS = {"Wonderful", "Gladly", "Sure"}
_NEG_WORDS = {"Hm", "Honestly", "Frankly"}


def turn(session, message, reset=False, brain="tiny-demo"):
    # rich=False -> the single-fact path (chat.gate + chat.render), which is STATELESS (no cross-turn discourse
    # thread, unlike the rich composer) and much faster than rebuilding the 47k-neuron rich path each turn. The
    # affect-drives lead is wired into BOTH paths; the FULL-config transcript exercises the rich/default path.
    resp = brain_chat(BrainChatRequest(session=session, message=message, brain=brain, reset=reset, rich=False))
    return json.loads(bytes(resp.body))


def _md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _content(d):
    """The honesty-floor content fields (must be affect-INVARIANT): abstain/recall/verify verdict + source/brain."""
    return {k: d.get(k) for k in ("abstained", "recalled_svo", "verified", "brain", "source", "rich")}


def _lead_word(lead):
    return (lead.split(" ", 1)[0].rstrip("!—").strip()) if lead else ""


def _seed_all(s):
    np.random.seed(s); random.seed(s)
    try:
        import cupy as cp
        cp.random.seed(s)
    except Exception:
        pass


def _clear_env():
    for k in ("BRAIN_AFFECT_DRIVES", "BRAIN_AFFECT_DRIVES_LESION", "BRAIN_AFFECT_DRIVES_INDUCE"):
        os.environ.pop(k, None)


# ── (A) affect TRACKS the conversation: appraised affective turns move the neural mood; a neutral probe HOLDS it ─────
# (message, kind) — kind in {neutral, pos, neg}. pos/neg = an appraised affective turn; neutral = a fact probe (holds).
CONV = [
    ("what does the dog chase?",                        "neutral"),
    ("I am so happy and joyful, this is wonderful",     "pos"),
    ("what does the dog chase?",                        "neutral"),   # holds positive
    ("I feel so sad and afraid, everything is terrible", "neg"),
    ("what does the dog chase?",                        "neutral"),   # holds negative
    ("this is a delight, I am glad and cheerful",       "pos"),
    ("what does the dog chase?",                        "neutral"),   # holds positive
]


def part_a():
    print("=" * 90)
    print("(A) AFFECT TRACKS THE CONVERSATION through /api/brain-chat  (BRAIN_AFFECT_DRIVES default-on)")
    print("=" * 90)
    _clear_env()
    rows, prev_sign = [], 0
    ok = True
    for i, (msg, kind) in enumerate(CONV):
        _seed_all(100 + i)
        d = turn("conv_a", msg, reset=(i == 0))
        ad = d.get("affect_drives") or {}
        mood = float(ad.get("mood", 0.0)); felt = float(ad.get("felt_arousal", 0.0))
        level = int(ad.get("level", 0)); lead = ad.get("lead", "")
        # expected sign: pos-> +, neg-> -, neutral-> holds the previous induced sign
        exp_sign = {"pos": 1, "neg": -1}.get(kind, prev_sign)
        got_sign = 1 if mood > 0.010 else (-1 if mood < -0.010 else 0)
        row_ok = (got_sign == exp_sign) if kind != "neutral" or i > 0 else (got_sign in (0, exp_sign))
        if kind in ("pos", "neg"):
            prev_sign = exp_sign
        ok = ok and row_ok
        rows.append({"i": i, "msg": msg, "kind": kind, "mood": mood, "felt_arousal": felt, "level": level,
                     "lead": lead, "exp_sign": exp_sign, "got_sign": got_sign, "row_ok": row_ok,
                     "answer": d.get("answer")})
        print("  [%d] %-52r kind=%-8s mood=%+.4f felt=%.4f lvl=%+d lead=%-14r %s"
              % (i, msg[:52], kind, mood, felt, level, lead, "OK" if row_ok else "**MISMATCH**"))
    # sensible-trajectory summary: baseline ~0, pos>+tol, neg<-tol, neutral probe holds the induced sign
    moods = [r["mood"] for r in rows]
    baseline_neutral = abs(moods[0]) <= 0.010
    pos_positive = all(rows[i]["mood"] > 0.010 for i in (1, 5))
    neg_negative = rows[3]["mood"] < -0.010
    holds_pos = rows[2]["mood"] > 0.010 and rows[6]["mood"] > 0.010
    holds_neg = rows[4]["mood"] < -0.010
    felt_rises = max(r["felt_arousal"] for r in rows) > rows[0]["felt_arousal"]
    a_ok = ok and baseline_neutral and pos_positive and neg_negative and holds_pos and holds_neg and felt_rises
    print("\n  baseline_neutral=%s pos_positive=%s neg_negative=%s holds_pos=%s holds_neg=%s felt_rises=%s"
          % (baseline_neutral, pos_positive, neg_negative, holds_pos, holds_neg, felt_rises))
    print("  (A) %s" % ("PASS" if a_ok else "FAIL"))
    _RESULTS["part_a"] = {"pass": bool(a_ok), "baseline_neutral": bool(baseline_neutral),
                          "pos_positive": bool(pos_positive), "neg_negative": bool(neg_negative),
                          "holds_pos": bool(holds_pos), "holds_neg": bool(holds_neg),
                          "felt_arousal_rises": bool(felt_rises), "rows": rows}
    return a_ok


# ── (B) affect DRIVES the response (the crux): MESSAGE FIXED, vary the affect state -> the lead differs; the neural
#    lesion (intero->ladder cut) collapses the mood -> the lead VANISHES -> the difference is gone. ──────────────────
FIXED_PROBE = "what does the dog chase?"


def _probe_induced(session, valence_arousal, lesion, reset=False):
    # The single-fact path (rich=False) is STATELESS, so ONE warm session holds the BASE answer fixed across probes
    # (build once, reset only on the first probe) while only the induced affect varies. The INDUCE env sets the
    # body-state directly (message held literally identical); the affect workspace EMA is overwritten each call.
    _seed_all(7)
    _clear_env()   # clears only the affect-drives flags; the heavy-organ baseline (set at import) persists
    if lesion:
        os.environ["BRAIN_AFFECT_DRIVES_LESION"] = "1"
    os.environ["BRAIN_AFFECT_DRIVES_INDUCE"] = valence_arousal
    d = turn(session, FIXED_PROBE, reset=reset)
    return d


def part_b():
    print("\n" + "=" * 90)
    print("(B) AFFECT DRIVES THE RESPONSE — message FIXED, vary affect; lesion collapses the difference")
    print("=" * 90)
    # INTACT: same message, positive vs negative induced affect state. ONE reused session NAME but reset=True on
    # EVERY probe -> the ChatBrain + rich discourse thread are dropped+rebuilt fresh each call (base answer held
    # fixed) while only ONE tiny-demo brain stays resident (memory) -- distinct never-reset session names cached 4
    # full brains at once and crashed Part C.
    d_pos = _probe_induced("b", "0.6,0.35", lesion=False, reset=True)   # build the warm brain once here
    d_neg = _probe_induced("b", "-0.7,0.6", lesion=False)
    lead_pos = (d_pos.get("affect_drives") or {}).get("lead", "")
    lead_neg = (d_neg.get("affect_drives") or {}).get("lead", "")
    ans_pos, ans_neg = d_pos.get("answer", ""), d_neg.get("answer", "")
    base_pos = ans_pos[len(lead_pos):] if lead_pos and ans_pos.startswith(lead_pos) else ans_pos
    base_neg = ans_neg[len(lead_neg):] if lead_neg and ans_neg.startswith(lead_neg) else ans_neg
    intact_diff = (ans_pos != ans_neg) and (lead_pos != lead_neg)
    lead_pos_ok = _lead_word(lead_pos) in _POS_WORDS
    lead_neg_ok = _lead_word(lead_neg) in _NEG_WORDS
    base_same = (base_pos == base_neg)
    content_same = (_md5(_content(d_pos)) == _md5(_content(d_neg)))
    print("  INTACT  pos: lead=%-14r ans=%r" % (lead_pos, ans_pos[:60]))
    print("          neg: lead=%-14r ans=%r" % (lead_neg, ans_neg[:60]))
    print("          intact_diff=%s pos_is_warm=%s neg_is_curt=%s base_identical=%s content_identical=%s"
          % (intact_diff, lead_pos_ok, lead_neg_ok, base_same, content_same))
    # LESION: cut the interoceptive->ladder synapses -> mood collapses -> BOTH leads vanish -> answers identical (=base)
    d_pos_l = _probe_induced("b", "0.6,0.35", lesion=True)
    d_neg_l = _probe_induced("b", "-0.7,0.6", lesion=True)
    lead_pos_l = (d_pos_l.get("affect_drives") or {}).get("lead", "")
    lead_neg_l = (d_neg_l.get("affect_drives") or {}).get("lead", "")
    mood_pos_l = float((d_pos_l.get("affect_drives") or {}).get("mood", 9.9))
    mood_neg_l = float((d_neg_l.get("affect_drives") or {}).get("mood", 9.9))
    lesion_leads_gone = (lead_pos_l == "" and lead_neg_l == "")
    lesion_answers_identical = (d_pos_l.get("answer") == d_neg_l.get("answer"))
    lesion_equals_base = (d_pos_l.get("answer") == base_pos == base_neg)
    lesion_mood_collapsed = (abs(mood_pos_l) < 0.010 and abs(mood_neg_l) < 0.010)
    print("  LESION  pos: lead=%-8r mood=%+.4f ans=%r" % (lead_pos_l, mood_pos_l, (d_pos_l.get('answer') or '')[:60]))
    print("          neg: lead=%-8r mood=%+.4f ans=%r" % (lead_neg_l, mood_neg_l, (d_neg_l.get('answer') or '')[:60]))
    print("          leads_gone=%s answers_identical=%s ==base=%s mood_collapsed=%s"
          % (lesion_leads_gone, lesion_answers_identical, lesion_equals_base, lesion_mood_collapsed))
    _clear_env()
    b_ok = (intact_diff and lead_pos_ok and lead_neg_ok and base_same and content_same
            and lesion_leads_gone and lesion_answers_identical and lesion_equals_base and lesion_mood_collapsed)
    print("  (B) %s   [intact difference PRESENT and it VANISHES under the neural lesion]" % ("PASS" if b_ok else "FAIL"))
    _RESULTS["part_b"] = {"pass": bool(b_ok), "intact_diff": bool(intact_diff), "lead_pos": lead_pos,
                          "lead_neg": lead_neg, "ans_pos": ans_pos, "ans_neg": ans_neg,
                          "base_identical": bool(base_same), "content_identical": bool(content_same),
                          "lesion_leads_gone": bool(lesion_leads_gone),
                          "lesion_answers_identical": bool(lesion_answers_identical),
                          "lesion_equals_base": bool(lesion_equals_base),
                          "lesion_mood_collapsed": bool(lesion_mood_collapsed),
                          "lesion_mood_pos": mood_pos_l, "lesion_mood_neg": mood_neg_l}
    return b_ok


# ── (C) NO-REGRESSION: content is affect-INVARIANT + byte-identical-off. Two panels. ────────────────────────────────
PANEL = [
    "what does the dog chase?",              # recall
    "what does a unicorn fly?",              # abstain (moat)
    "who are you?",                          # self / identity
]


def part_c():
    print("\n" + "=" * 90)
    print("(C) NO-REGRESSION — content affect-invariant + byte-identical-off")
    print("=" * 90)
    # C1 (NEUTRAL byte-identity) and C2 (content-invariance under ACTIVE affect) run on SEPARATE sessions. The mood
    # is a PERSISTENT per-session state (that is the point — see (A)), so a session that has been mood-INDUCED
    # correctly HOLDS the induced mood into a later 'neutral' turn (a colored lead). To test the neutral byte-identity
    # honestly, C1 uses a session that is NEVER induced (mood stays ~0 -> no lead -> ON == OFF except the key); C2
    # uses its own induced session to check the content fields are affect-invariant.
    any_off_key = False
    all_on_key = True
    neutral_byte_ok = True
    c1_rows = []
    print("  -- C1 neutral byte-identity (un-induced session; a neutral mood adds no lead) --")
    print("  %-34s | %-9s %-9s | keys | ON-lead | byte" % ("message", "OFF md5", "ON md5"))
    first = True
    for i, msg in enumerate(PANEL):
        _seed_all(1234 + i); _clear_env(); os.environ["BRAIN_AFFECT_DRIVES"] = "0"
        d_off = turn("c1", msg, reset=first); first = False
        _seed_all(1234 + i); _clear_env()   # default-on, no induction -> the held mood stays neutral
        d_on = turn("c1", msg)
        has_off_key = "affect_drives" in d_off
        has_on_key = "affect_drives" in d_on
        any_off_key = any_off_key or has_off_key
        all_on_key = all_on_key and has_on_key
        on_lead = (d_on.get("affect_drives") or {}).get("lead", "")
        d_on_stripped = {k: v for k, v in d_on.items() if k != "affect_drives"}
        h_off, h_on_strip = _md5(d_off), _md5(d_on_stripped)
        byte_ok = (h_off == h_on_strip)
        neutral_byte_ok = neutral_byte_ok and byte_ok and (not has_off_key) and has_on_key
        c1_rows.append({"i": i, "msg": msg, "off_md5": h_off, "on_stripped_md5": h_on_strip, "neutral_byte_ok": byte_ok,
                        "off_has_key": has_off_key, "on_has_key": has_on_key, "on_lead": on_lead,
                        "off_answer": d_off.get("answer"), "on_answer": d_on.get("answer")})
        print("  %-34s | %-9s %-9s | off=%s on=%s | %-8r | %s"
              % (msg[:34], h_off[:8], h_on_strip[:8], has_off_key, has_on_key, on_lead,
                 "IDENTICAL" if byte_ok else "**DIFFERS**"))
    content_invariant = True
    c2_rows = []
    print("  -- C2 content-invariance under ACTIVE affect (induced session; content unchanged, only tone) --")
    print("  %-34s | content-md5 {off,+pos,+neg} | inv" % "message")
    first = True
    for i, msg in enumerate(PANEL):
        _seed_all(1234 + i); _clear_env(); os.environ["BRAIN_AFFECT_DRIVES"] = "0"
        d_off = turn("c2", msg, reset=first); first = False
        _seed_all(1234 + i); _clear_env(); os.environ["BRAIN_AFFECT_DRIVES_INDUCE"] = "0.7,0.4"
        d_p = turn("c2", msg)
        _seed_all(1234 + i); _clear_env(); os.environ["BRAIN_AFFECT_DRIVES_INDUCE"] = "-0.7,0.6"
        d_n = turn("c2", msg)
        cm_off, cm_p, cm_n = _md5(_content(d_off)), _md5(_content(d_p)), _md5(_content(d_n))
        cinv = (len({cm_off, cm_p, cm_n}) == 1)
        content_invariant = content_invariant and cinv
        c2_rows.append({"i": i, "msg": msg, "content_md5": cm_off, "content_invariant": cinv,
                        "off_answer": d_off.get("answer"), "induced_pos_answer": d_p.get("answer"),
                        "induced_neg_answer": d_n.get("answer")})
        print("  %-34s | %-8s | %s" % (msg[:34], cm_off[:8], "OK" if cinv else "**REGRESS**"))
    _clear_env()
    print("\n  OFF ever carried affect_drives key: %s (want False)" % any_off_key)
    print("  ON  always carried affect_drives key: %s (want True)" % all_on_key)
    print("  neutral ON-minus-key == OFF (byte) every un-induced turn: %s" % neutral_byte_ok)
    print("  content fields invariant across {off,+pos,+neg} every turn: %s" % content_invariant)
    c_ok = neutral_byte_ok and content_invariant and (not any_off_key) and all_on_key
    print("  (C) %s" % ("PASS" if c_ok else "FAIL"))
    _RESULTS["part_c"] = {"pass": bool(c_ok), "off_ever_has_key": bool(any_off_key), "on_always_has_key": bool(all_on_key),
                          "neutral_byte_identical_all": bool(neutral_byte_ok),
                          "content_invariant_all": bool(content_invariant), "c1_rows": c1_rows, "c2_rows": c2_rows}
    return c_ok


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    c = part_c()
    preconditions = [
        {"name": "(A) mood tracks the conversation (baseline~0, pos>+tol, neg<-tol, neutral holds, felt rises)",
         "ok": bool(_RESULTS["part_a"].get("pass"))},
        {"name": "(B) message-fixed: affect changes the reply lead (pos warm != neg curt) with content identical",
         "ok": bool(_RESULTS["part_b"].get("intact_diff") and _RESULTS["part_b"].get("content_identical")
                    and _RESULTS["part_b"].get("base_identical"))},
        {"name": "(B) the neural lesion (intero->ladder cut) collapses the mood -> the lead difference VANISHES",
         "ok": bool(_RESULTS["part_b"].get("lesion_leads_gone") and _RESULTS["part_b"].get("lesion_answers_identical")
                    and _RESULTS["part_b"].get("lesion_mood_collapsed"))},
        {"name": "(C) content fields affect-invariant across {off,+pos,+neg} on every panel turn",
         "ok": bool(_RESULTS["part_c"].get("content_invariant_all"))},
        {"name": "(C) byte-identical-off: OFF never carries the key; a neutral (un-induced) ON-minus-key == OFF",
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
    print("\n" + "=" * 90)
    print("VERDICT  (A) tracks=%s  (B) drives+lesion=%s  (C) no-regression=%s  => %s" % (a, b, c, verdict))
    for p in preconditions:
        print("   [%s] %s" % ("PASS" if p["ok"] else "FAIL", p["name"]))
    print("wrote %s" % _ART)
    print("=" * 90)
    raise SystemExit(0 if (a and b and c) else 1)
