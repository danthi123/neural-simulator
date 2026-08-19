"""Verify the board-#85 SWAP-DRIVES-THE-RESPONSE wiring THROUGH the real /api/brain-chat handler (in-process).
The anti-hollow-integration proof: the #77 neural thought-swap verdict is LOAD-BEARING on the live turn.
(A) the swap TRACKS the conversation (topic-change swaps, same-topic holds; the transition lead appears iff swapped).
(B) the swap DRIVES the response: message held FIXED, vary the held-topic CONTEXT (swap vs hold) -> the reply's
    transition lead DIFFERS, and that difference VANISHES under the neural lesion (mismatch detector silenced) ->
    the coupling rides the SPIKING swap read, not a host `if topic_changed`. Content byte-identical throughout.
(C) NO-REGRESSION on content: the moat/recall/abstain verdict is byte-identical off-vs-on; the swap changes only
    the surface framing (an optional transition lead), never a fact; a no-swap ON-minus-key == OFF byte-identical.
Runs all three in ONE process (shares the heavy first-turn warmup). Usage: python verify_swap_drives.py"""
import os, json, hashlib, subprocess, time, random
import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

# ISOLATE THE FACULTY FOR A TRACTABLE IN-PROCESS VERIFY. The OTHER default-on organs (Gate-B affect, the affect-DRIVES
# lead, worldmodel, surprise, metacog, comprehension, ...) are ORTHOGONAL to swap-drives and make each turn slow;
# disabling them (a consistent baseline across ALL arms) keeps the SAME /api/brain-chat handler + the SAME recall/moat
# core while dropping per-turn cost. swap-drives reads its own #77 workspace + prepends a transition lead regardless of
# the others, so this isolation cannot change any swap-drives verdict. In particular BRAIN_AFFECT_DRIVES=0 so the
# affect lead never co-mingles with the swap lead in the answer string (the two leads are orthogonal + independently on).
if os.environ.get("SWAP_VERIFY_FULL_CONFIG", "0").strip().lower() not in ("1", "true", "on", "yes"):
    for _k in ("BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG",
               "BRAIN_COMPREHENSION_GATE", "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF",
               "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE", "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE",
               "BRAIN_RECONSOLIDATION", "BRAIN_PMEM", "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER", "BRAIN_GNW_SWAP"):
        os.environ[_k] = "0"

from webapp.server import brain_chat, BrainChatRequest  # the REAL handler

_ART = os.environ.get("SWAP_DRIVES_JSON", "research/findings/raw/_swap_drives_chat/verify.json")
_RESULTS = {"runner": "verify_swap_drives (in-process /api/brain-chat)", "backend": os.environ.get("SIM_BACKEND"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "part_a": {}, "part_b": {}, "part_c": {}}
try:
    _RESULTS["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    _RESULTS["git_sha"] = None


def turn(session, message, reset=False, brain="tiny-demo"):
    # rich=False -> the single-fact path (chat.gate + chat.render), STATELESS (no cross-turn discourse thread) and much
    # faster than the rich composer. The swap-drives lead is wired into BOTH return paths; this exercises the fast one.
    resp = brain_chat(BrainChatRequest(session=session, message=message, brain=brain, reset=reset, rich=False))
    return json.loads(bytes(resp.body))


def _md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _content(d):
    """The honesty-floor content fields (must be swap-INVARIANT): abstain/recall/verify verdict + source/brain."""
    return {k: d.get(k) for k in ("abstained", "recalled_svo", "verified", "brain", "source", "rich")}


def _seed_all(s):
    np.random.seed(s); random.seed(s)
    try:
        import cupy as cp
        cp.random.seed(s)
    except Exception:
        pass


def _clear_env():
    for k in ("BRAIN_SWAP_DRIVES", "BRAIN_SWAP_DRIVES_LESION"):
        os.environ.pop(k, None)


# ── (A) the swap TRACKS the conversation: topic-change turns SWAP (+ a transition lead); same-topic/no-topic HOLD ─────
# (message, kind) — kind in {establish, change, hold}. change -> expect swapped True + a lead; establish/hold -> False + no lead.
CONV = [
    ("what does the dog chase?",   "establish"),  # first thought -> hold 'dog', NO lead
    ("what does the dog chase?",   "hold"),        # same topic 'dog' -> no swap, NO lead
    ("what does the brain use?",   "change"),      # 'brain' != 'dog' -> SWAP -> lead "On brain, then — "
    ("what does the brain store?", "hold"),        # same topic 'brain' -> no swap, NO lead
    ("what does the cat eat?",     "change"),      # 'cat' -> SWAP -> lead "On cat, then — "
    ("tell me more",               "hold"),        # no new grounded topic -> hold 'cat', NO lead
    ("what does the dog chase?",   "change"),      # 'dog' -> SWAP -> lead "On dog, then — "
]


def part_a():
    print("=" * 92)
    print("(A) THE SWAP TRACKS THE CONVERSATION through /api/brain-chat  (BRAIN_SWAP_DRIVES default-on)")
    print("=" * 92)
    _clear_env()
    n_change = n_change_swap = n_hold = n_hold_swap = 0
    rows, ok = [], True
    for i, (msg, kind) in enumerate(CONV):
        _seed_all(100 + i)
        d = turn("conv_a", msg, reset=(i == 0))
        sd = d.get("swap_drives") or {}
        sw = bool(sd.get("swapped"))
        held = sd.get("held_topic")
        lead = sd.get("lead", "")
        ans = d.get("answer", "") or ""
        exp_swap = (kind == "change")
        # the lead must be present IFF the neural swap fired, must name the (new) held topic, and must be prepended.
        lead_present = bool(lead)
        lead_names_topic = (held in lead) if (sw and held) else True
        lead_prepended = ans.startswith(lead) if lead else True
        row_ok = (sw == exp_swap) and (lead_present == sw) and lead_names_topic and lead_prepended
        if kind == "change":
            n_change += 1; n_change_swap += sw
        elif kind == "hold":
            n_hold += 1; n_hold_swap += sw
        ok = ok and row_ok
        rows.append({"i": i, "msg": msg, "kind": kind, "swapped": sw, "held_topic": held, "lead": lead,
                     "evicted_topic": sd.get("evicted_topic"), "reason": sd.get("reason"),
                     "mm_peak": sd.get("mm_peak"), "boost_max": sd.get("boost_max"),
                     "answer": ans, "row_ok": row_ok})
        print("  [%d] %-30r kind=%-9s swapped=%-5s held=%-6s lead=%-18r %s"
              % (i, msg, kind, sw, held, lead, "OK" if row_ok else "**MISMATCH**"))
    change_rate = n_change_swap / max(1, n_change)
    hold_rate = n_hold_swap / max(1, n_hold)
    leads_on_swaps = all((bool(r["lead"]) == r["swapped"]) for r in rows)
    print("\n  SWAP RATE  topic-change=%.2f (%d/%d)   same-topic/hold=%.2f (%d/%d)   lead-iff-swap=%s"
          % (change_rate, n_change_swap, n_change, hold_rate, n_hold_swap, n_hold, leads_on_swaps))
    a_ok = ok and change_rate == 1.0 and hold_rate == 0.0 and leads_on_swaps
    print("  (A) %s" % ("PASS" if a_ok else "FAIL"))
    _RESULTS["part_a"] = {"pass": bool(a_ok), "topic_change_swap_rate": change_rate, "same_topic_swap_rate": hold_rate,
                          "lead_iff_swap": bool(leads_on_swaps), "n_change": n_change, "n_change_swap": n_change_swap,
                          "n_hold": n_hold, "n_hold_swap": n_hold_swap, "rows": rows}
    return a_ok


# ── (B) the swap DRIVES the response (the crux): MESSAGE FIXED, vary the held-topic CONTEXT (swap vs hold) -> the
#    transition lead differs; the neural lesion (mismatch detector silenced) collapses the swap -> the lead VANISHES. ──
FIXED_PROBE = "what does the dog chase?"   # topic 'dog'


def _establish_then_probe(session, establish_msg, lesion):
    # turn1 establishes the held topic CLEANLY (drives-on, NO lesion); turn2 is the FIXED probe (optionally lesioned).
    # Fresh session (reset=True on turn1) so the held-topic register is clean -> no cross-arm leak (the #84 lesson).
    _seed_all(7); _clear_env()
    turn(session, establish_msg, reset=True)             # establish (first_thought -> hold, no lead)
    _seed_all(7)
    if lesion:
        os.environ["BRAIN_SWAP_DRIVES_LESION"] = "1"
    d = turn(session, FIXED_PROBE)                        # the fixed probe -> the neural swap decides
    _clear_env()
    return d


def _lead_of(d):
    return (d.get("swap_drives") or {}).get("lead", "") or ""


def part_b():
    print("\n" + "=" * 92)
    print("(B) THE SWAP DRIVES THE RESPONSE — message FIXED, vary held-topic context; lesion collapses the difference")
    print("=" * 92)
    # INTACT: SAME fixed probe 'what does the dog chase?', two held-topic contexts.
    #   swap-context: held='cat' established first -> 'dog' is a MISMATCH -> SWAP -> "On dog, then — " lead.
    #   hold-context: held='dog' established first -> 'dog' MATCHES -> HOLD -> no lead.
    # All three arms REUSE one session "b" (each _establish_then_probe resets it -> only one ChatBrain resident at a
    # time: the #84 memory lesson -- distinct never-reset session names cached several full brains and crashed).
    d_swap = _establish_then_probe("b", "what does the cat eat?", lesion=False)
    d_hold = _establish_then_probe("b", "what does the dog chase?", lesion=False)
    lead_swap, lead_hold = _lead_of(d_swap), _lead_of(d_hold)
    ans_swap, ans_hold = d_swap.get("answer", "") or "", d_hold.get("answer", "") or ""
    sw_swap = bool((d_swap.get("swap_drives") or {}).get("swapped"))
    sw_hold = bool((d_hold.get("swap_drives") or {}).get("swapped"))
    base_swap = ans_swap[len(lead_swap):] if lead_swap and ans_swap.startswith(lead_swap) else ans_swap
    intact_diff = (ans_swap != ans_hold) and (lead_swap != lead_hold)
    lead_swap_ok = lead_swap.startswith("On dog") and lead_swap.strip().endswith("then —")
    hold_no_lead = (lead_hold == "")
    base_same = (base_swap == ans_hold)   # the fact under the lead == the hold-arm answer (no lead)
    content_same = (_md5(_content(d_swap)) == _md5(_content(d_hold)))
    print("  INTACT  swap-ctx(held=cat): swapped=%s lead=%-18r ans=%r" % (sw_swap, lead_swap, ans_swap[:64]))
    print("          hold-ctx(held=dog): swapped=%s lead=%-18r ans=%r" % (sw_hold, lead_hold, ans_hold[:64]))
    print("          intact_diff=%s swap_lead_ok=%s hold_no_lead=%s base_identical=%s content_identical=%s"
          % (intact_diff, lead_swap_ok, hold_no_lead, base_same, content_same))
    # LESION: silence the mismatch detector -> the SAME swap-context 'dog' can NO LONGER swap -> the lead VANISHES ->
    # the answer reverts to the byte-identical no-lead base. So the transition lead RIDES the spiking mm read.
    d_swapL = _establish_then_probe("b", "what does the cat eat?", lesion=True)
    lead_swapL = _lead_of(d_swapL)
    ans_swapL = d_swapL.get("answer", "") or ""
    sw_swapL = bool((d_swapL.get("swap_drives") or {}).get("swapped"))
    reason_swapL = (d_swapL.get("swap_drives") or {}).get("reason")
    mm_swapL = float((d_swapL.get("swap_drives") or {}).get("mm_peak", 9.9) or 9.9)
    lesion_lead_gone = (lead_swapL == "")
    lesion_no_swap = (sw_swapL is False)
    lesion_equals_base = (ans_swapL == base_swap == ans_hold)
    print("  LESION  swap-ctx(held=cat)+mm-silenced: swapped=%s reason=%s mm_peak=%.3f lead=%-8r ans=%r"
          % (sw_swapL, reason_swapL, mm_swapL, lead_swapL, ans_swapL[:64]))
    print("          lead_gone=%s no_swap=%s ==base=%s" % (lesion_lead_gone, lesion_no_swap, lesion_equals_base))
    _clear_env()
    b_ok = (intact_diff and lead_swap_ok and sw_swap and (not sw_hold) and hold_no_lead and base_same
            and content_same and lesion_lead_gone and lesion_no_swap and lesion_equals_base)
    print("  (B) %s   [intact transition lead PRESENT and it VANISHES under the neural mm-lesion]" % ("PASS" if b_ok else "FAIL"))
    _RESULTS["part_b"] = {"pass": bool(b_ok), "intact_diff": bool(intact_diff), "lead_swap": lead_swap,
                          "lead_hold": lead_hold, "ans_swap": ans_swap, "ans_hold": ans_hold,
                          "swapped_swap_ctx": bool(sw_swap), "swapped_hold_ctx": bool(sw_hold),
                          "swap_lead_ok": bool(lead_swap_ok), "hold_no_lead": bool(hold_no_lead),
                          "base_identical": bool(base_same), "content_identical": bool(content_same),
                          "lesion_lead_gone": bool(lesion_lead_gone), "lesion_no_swap": bool(lesion_no_swap),
                          "lesion_equals_base": bool(lesion_equals_base), "lesion_reason": reason_swapL,
                          "lesion_mm_peak": mm_swapL}
    return b_ok


# ── (C) NO-REGRESSION: content swap-INVARIANT + byte-identical-off. Two panels. ──────────────────────────────────────
# C1 (NEUTRAL byte-identity) runs a NO-SWAP panel (establish, hold, abstain -> no lead every turn) so ON-minus-key ==
# OFF. C2 (content-invariance under an ACTIVE swap) shows the content fields are identical whether or not a swap fires.
C1_PANEL = [
    "what does the dog chase?",   # establish 'dog' (first_thought -> no lead)
    "what does the dog chase?",   # same 'dog' -> hold -> no lead
    "what does a unicorn fly?",   # unicorn not grounded -> no topic -> hold -> no lead; abstain (moat)
]


def part_c():
    print("\n" + "=" * 92)
    print("(C) NO-REGRESSION — content swap-invariant + byte-identical-off")
    print("=" * 92)
    any_off_key = False
    all_on_key = True
    neutral_byte_ok = True
    c1_rows = []
    print("  -- C1 no-swap byte-identity (a no-swap turn adds no lead) --")
    print("  %-34s | %-9s %-9s | keys | ON-lead | byte" % ("message", "OFF md5", "ON md5"))
    first = True
    for i, msg in enumerate(C1_PANEL):
        _seed_all(1234 + i); _clear_env(); os.environ["BRAIN_SWAP_DRIVES"] = "0"
        d_off = turn("c1_off", msg, reset=first)
        _seed_all(1234 + i); _clear_env()   # default-on
        d_on = turn("c1_on", msg, reset=first)
        first = False
        has_off_key = "swap_drives" in d_off
        has_on_key = "swap_drives" in d_on
        any_off_key = any_off_key or has_off_key
        all_on_key = all_on_key and has_on_key
        on_lead = (d_on.get("swap_drives") or {}).get("lead", "")
        d_on_stripped = {k: v for k, v in d_on.items() if k != "swap_drives"}
        h_off, h_on_strip = _md5(d_off), _md5(d_on_stripped)
        byte_ok = (h_off == h_on_strip)
        neutral_byte_ok = neutral_byte_ok and byte_ok and (not has_off_key) and has_on_key and (on_lead == "")
        c1_rows.append({"i": i, "msg": msg, "off_md5": h_off, "on_stripped_md5": h_on_strip, "byte_ok": byte_ok,
                        "off_has_key": has_off_key, "on_has_key": has_on_key, "on_lead": on_lead,
                        "off_answer": d_off.get("answer"), "on_answer": d_on.get("answer")})
        print("  %-34s | %-9s %-9s | off=%s on=%s | %-8r | %s"
              % (msg[:34], h_off[:8], h_on_strip[:8], has_off_key, has_on_key, on_lead,
                 "IDENTICAL" if byte_ok else "**DIFFERS**"))
    # C2 content-invariance under an ACTIVE swap: the SAME fixed probe, off vs a swap-context vs a hold-context ->
    # the CONTENT fields (abstain/recall/verify) are identical (the lead differs, the fact does not).
    content_invariant = True
    c2_rows = []
    print("  -- C2 content-invariance under an ACTIVE swap (content unchanged, only the transition lead) --")
    _seed_all(55); _clear_env(); os.environ["BRAIN_SWAP_DRIVES"] = "0"
    d_off = turn("c2", FIXED_PROBE, reset=True)
    _clear_env()
    d_sw = _establish_then_probe("c2", "what does the cat eat?", lesion=False)     # swap-context (lead present)
    d_hd = _establish_then_probe("c2", "what does the dog chase?", lesion=False)   # hold-context (no lead)
    cm_off, cm_sw, cm_hd = _md5(_content(d_off)), _md5(_content(d_sw)), _md5(_content(d_hd))
    cinv = (len({cm_off, cm_sw, cm_hd}) == 1)
    content_invariant = content_invariant and cinv
    c2_rows.append({"msg": FIXED_PROBE, "content_md5": cm_off, "content_invariant": cinv,
                    "off_answer": d_off.get("answer"), "swap_answer": d_sw.get("answer"),
                    "hold_answer": d_hd.get("answer"), "swap_lead": _lead_of(d_sw), "hold_lead": _lead_of(d_hd)})
    print("  %-34s | content-md5 {off,swap,hold}=%s | swap-lead=%r | %s"
          % (FIXED_PROBE[:34], cm_off[:8], _lead_of(d_sw), "OK" if cinv else "**REGRESS**"))
    _clear_env()
    print("\n  OFF ever carried swap_drives key: %s (want False)" % any_off_key)
    print("  ON  always carried swap_drives key: %s (want True)" % all_on_key)
    print("  no-swap ON-minus-key == OFF (byte) every turn: %s" % neutral_byte_ok)
    print("  content fields invariant across {off,swap,hold}: %s" % content_invariant)
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
        {"name": "(A) the swap tracks the conversation (change swap-rate 1.00, hold 0.00, lead iff swap)",
         "ok": bool(_RESULTS["part_a"].get("pass"))},
        {"name": "(B) message-fixed: the held-topic context changes the reply lead (swap 'On dog' != hold '') with content identical",
         "ok": bool(_RESULTS["part_b"].get("intact_diff") and _RESULTS["part_b"].get("content_identical")
                    and _RESULTS["part_b"].get("base_identical"))},
        {"name": "(B) the neural lesion (mismatch detector silenced) collapses the swap -> the transition lead VANISHES -> == base",
         "ok": bool(_RESULTS["part_b"].get("lesion_lead_gone") and _RESULTS["part_b"].get("lesion_no_swap")
                    and _RESULTS["part_b"].get("lesion_equals_base"))},
        {"name": "(C) content fields swap-invariant across {off,swap,hold}",
         "ok": bool(_RESULTS["part_c"].get("content_invariant_all"))},
        {"name": "(C) byte-identical-off: OFF never carries the key; a no-swap ON-minus-key == OFF",
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
    print("\n" + "=" * 92)
    print("VERDICT  (A) tracks=%s  (B) drives+lesion=%s  (C) no-regression=%s  => %s" % (a, b, c, verdict))
    for p in preconditions:
        print("   [%s] %s" % ("PASS" if p["ok"] else "FAIL", p["name"]))
    print("wrote %s" % _ART)
    print("=" * 92)
    raise SystemExit(0 if (a and b and c) else 1)
