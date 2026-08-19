"""Verify the GNW thought-swap wiring THROUGH the real /api/brain-chat handler (in-process).
(A) continuous-conversation functional + swap-rate; (B) byte-identical-off + no-regression md5 panel.
Runs both in ONE process (shares the heavy first-turn warmup). Usage: python _verify_swap.py"""
import os, json, hashlib, subprocess, time, random
import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

from webapp.server import brain_chat, BrainChatRequest  # the REAL handler

_ART = os.environ.get("SWAP_VERIFY_JSON", "research/findings/raw/_gnw_swap_chat/verify.json")
_RESULTS = {"runner": "_verify_swap (in-process /api/brain-chat)", "backend": os.environ.get("SIM_BACKEND"),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "part_a": {}, "part_b": {}}
try:
    _RESULTS["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    _RESULTS["git_sha"] = None


def turn(session, message, reset=False, brain="tiny-demo"):
    resp = brain_chat(BrainChatRequest(session=session, message=message, brain=brain, reset=reset))
    return json.loads(bytes(resp.body))


def _md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _seed_all(s):
    """Seed every process-global RNG so a turn is deterministic. Some base-system turns (a curiosity-augmented ABSTAIN,
    a no-topic follow-up) sample a follow-up off the global RNG and are non-deterministic across runs REGARDLESS of the
    swap; controlling the RNG isolates whether ENABLING the swap changes the answer (the no-regression claim)."""
    np.random.seed(s)
    random.seed(s)
    try:
        import cupy as cp
        cp.random.seed(s)
    except Exception:
        pass


# ── (A) continuous conversation: topic-change turns SWAP; same-topic turns HOLD ─────────────────────────────────────
# (message, kind) — kind in {establish, change, hold}. change -> expect swapped True; establish/hold -> expect False.
CONV = [
    ("what does the dog chase?",  "establish"),   # first thought -> hold 'dog'
    ("what does the dog chase?",  "hold"),        # same topic 'dog' -> no swap
    ("what does the brain use?",  "change"),      # 'brain' != 'dog' -> SWAP
    ("what does the brain store?","hold"),        # same topic 'brain' -> no swap
    ("what does the cat eat?",    "change"),      # 'cat' -> SWAP
    ("tell me more",              "hold"),        # no new grounded topic -> hold 'cat'
    ("what does the dog chase?",  "change"),      # 'dog' -> SWAP back
]


def part_a():
    print("=" * 80)
    print("(A) CONTINUOUS CONVERSATION through /api/brain-chat  (BRAIN_GNW_SWAP=1)")
    print("=" * 80)
    os.environ["BRAIN_GNW_SWAP"] = "1"
    n_change = n_change_swap = n_hold = n_hold_swap = 0
    ok = True
    rows = []
    for i, (msg, kind) in enumerate(CONV):
        d = turn("conv", msg, reset=(i == 0))
        s = d.get("gnw_swap") or {}
        sw = bool(s.get("swapped"))
        held = s.get("held_topic")
        exp_swap = (kind == "change")
        row_ok = (sw == exp_swap)
        if kind == "change":
            n_change += 1; n_change_swap += sw
        elif kind == "hold":
            n_hold += 1; n_hold_swap += sw
        ok = ok and row_ok
        rows.append({"i": i, "msg": msg, "kind": kind, "answer": d.get("answer"), "swapped": sw,
                     "held_topic": held, "evicted_topic": s.get("evicted_topic"),
                     "n_ignited_post": s.get("n_ignited_post"), "new_rate_post": s.get("new_rate_post"),
                     "held_rate_post": s.get("held_rate_post"), "old_residual_post": s.get("old_residual_post"),
                     "mm_peak": s.get("mm_peak"), "boost_max": s.get("boost_max"), "reason": s.get("reason"),
                     "row_ok": row_ok})
        print("  [%d] %-28r kind=%-9s swapped=%-5s held=%-6s evicted=%-6s n_ign=%s new=%.3f old=%.3f  %s"
              % (i, msg, kind, sw, held, s.get("evicted_topic"),
                 s.get("n_ignited_post"), s.get("new_rate_post") or s.get("held_rate_post") or 0.0,
                 s.get("old_residual_post") or 0.0, "OK" if row_ok else "**MISMATCH**"))
    change_rate = n_change_swap / max(1, n_change)
    hold_rate = n_hold_swap / max(1, n_hold)
    print("\n  SWAP RATE  topic-change=%.2f (%d/%d)   same-topic/hold=%.2f (%d/%d)"
          % (change_rate, n_change_swap, n_change, hold_rate, n_hold_swap, n_hold))
    a_ok = ok and change_rate == 1.0 and hold_rate == 0.0
    print("  (A) %s" % ("PASS" if a_ok else "FAIL"))
    _RESULTS["part_a"] = {"pass": a_ok, "topic_change_swap_rate": change_rate, "same_topic_swap_rate": hold_rate,
                          "n_change": n_change, "n_change_swap": n_change_swap, "n_hold": n_hold,
                          "n_hold_swap": n_hold_swap, "rows": rows}
    return a_ok


# ── (B) byte-identical-off + no-regression: same panel, interleaved OFF vs ON, md5 compared ─────────────────────────
PANEL = [
    "what does the dog chase?",              # recall (topic dog)
    "what does the brain use?",              # recall (topic change brain)
    "what does a unicorn fly?",              # abstain (moat) + topic none
    "what does the cat eat all the way?",    # multi-step chase form (topic cat)
    "who are you?",                          # self / identity
    "tell me more",                          # follow-up (no topic)
]


def part_b():
    print("\n" + "=" * 80)
    print("(B) BYTE-IDENTICAL-OFF + NO-REGRESSION  (same panel, OFF vs ON, md5)")
    print("=" * 80)
    any_off_key = False
    all_on_key = True
    regress_ok = True
    rows = []
    print("  %-40s | %-10s %-10s %-10s | key?" % ("message", "OFF md5", "ON md5", "ON\\swap"))
    for i, msg in enumerate(PANEL):
        # seed identically before the OFF and the ON turn so both see the same global RNG start; then any difference is
        # ONLY the swap wiring (the swap itself also restores the host RNG internally — belt and suspenders).
        _seed_all(1234 + i)
        os.environ["BRAIN_GNW_SWAP"] = "0"
        d_off = turn("p_off", msg, reset=(i == 0))
        _seed_all(1234 + i)
        os.environ["BRAIN_GNW_SWAP"] = "1"
        d_on = turn("p_on", msg, reset=(i == 0))
        has_off_key = "gnw_swap" in d_off
        has_on_key = "gnw_swap" in d_on
        any_off_key = any_off_key or has_off_key
        all_on_key = all_on_key and has_on_key
        d_on_stripped = {k: v for k, v in d_on.items() if k != "gnw_swap"}
        h_off, h_on, h_on_strip = _md5(d_off), _md5(d_on), _md5(d_on_stripped)
        match = (h_off == h_on_strip)
        regress_ok = regress_ok and match and (not has_off_key) and has_on_key
        rows.append({"i": i, "msg": msg, "off_md5": h_off, "on_md5": h_on, "on_minus_swap_md5": h_on_strip,
                     "off_has_key": has_off_key, "on_has_key": has_on_key, "identical": match,
                     "off_answer": d_off.get("answer"), "on_answer": d_on.get("answer")})
        print("  %-40s | %-10s %-10s %-10s | off=%s on=%s  %s"
              % (msg[:40], h_off[:8], h_on[:8], h_on_strip[:8], has_off_key, has_on_key,
                 "IDENTICAL" if match else "**DIFFERS**"))
    print("\n  OFF response ever carried gnw_swap key: %s (want False)" % any_off_key)
    print("  ON  response always carried gnw_swap key: %s (want True)" % all_on_key)
    print("  ON-minus-gnw_swap == OFF on every panel turn: %s" % regress_ok)
    print("  (B) %s" % ("PASS" if regress_ok else "FAIL"))
    _RESULTS["part_b"] = {"pass": regress_ok, "off_ever_has_key": any_off_key, "on_always_has_key": all_on_key,
                          "on_minus_swap_equals_off_all_turns": regress_ok, "rows": rows}
    return regress_ok


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    pa, pb = _RESULTS.get("part_a", {}), _RESULTS.get("part_b", {})
    preconditions = [
        {"name": "functional: topic-change swap rate == 1.00 through the real handler",
         "ok": bool(pa.get("topic_change_swap_rate") == 1.0)},
        {"name": "functional: same-topic/no-topic swap rate == 0.00 through the real handler",
         "ok": bool(pa.get("same_topic_swap_rate") == 0.0)},
        {"name": "byte-identical: OFF response never carries a gnw_swap key",
         "ok": bool(pb.get("off_ever_has_key") is False)},
        {"name": "no-regression: ON-minus-gnw_swap == OFF (md5) on every panel turn",
         "ok": bool(pb.get("on_minus_swap_equals_off_all_turns"))},
        {"name": "ON response always carries the additive gnw_swap key",
         "ok": bool(pb.get("on_always_has_key"))},
    ]
    _RESULTS["preconditions"] = preconditions
    all_pre_ok = all(p["ok"] for p in preconditions)
    verdict = "GO" if (a and b and all_pre_ok) else "NO-GO"
    _RESULTS["verdict"] = verdict
    os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
    with open(_ART, "w") as f:
        json.dump(_RESULTS, f, indent=2, default=str)
    print("\n" + "=" * 80)
    print("VERDICT  (A) functional+rate=%s   (B) byte-identical+no-regression=%s   => %s" % (a, b, verdict))
    print("wrote %s" % _ART)
    print("=" * 80)
    raise SystemExit(0 if (a and b) else 1)
