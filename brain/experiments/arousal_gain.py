# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md (section 19: iteration 6 pre-registered; build plan 2026-11-18-arousal-gain-plan.md)
# Expected-fail: the heard-turn gain may amplify echo or invented meanings rather than real uptake of the event (plan A, most likely failure)
"""Mood iteration 6: arousal as encoding gain on the heard turn (pre-registration: section 19 of the mood-state design).

Nothing here changes the brain beyond `Brain(arousal_gain=SPAN, arousal_pin=...)`; no host word is ever added to what it
hears or says. Subcommands:
  calibrate  (MU, SD) of the drive from V5_DEV neutral events + 12 ordinary sentences read from birth
  smoke      S1 split vs joint read (logits <= 1e-5, 20 greedy turns identical; else the OFF-split fallback),
             S2 LESION == OFF, S3 filler gain <= 1.1 on >= 90% of filler turns (dev bank trace)
  trace      per-turn drive / arousal / gain of the ON arm on a bank (greedy, no sampling)
  pilot      seed 42, SPAN 1, OFF / ON / CONSTANT on the dev bank; the pilot gate
  dev        seeds 42/43/44, SPAN in {0.5, 1, 2}; SPAN choice, futility and floor (dev rule, no readers)
  test       the six arms on the test bank at a frozen --span, seeds 42/43/44/100/101/102, 2 samples
  export     the reader batch (T2 and T4 replies; shuffled; arm and class hidden) + an audit key
  table      criteria C0-C5, the claim-name rule and the decision branch from run JSONs + a labels file

Arms (same sampling seed per conversation and turn): OFF (switch off, or the split path with no gain when the S1
fallback applies), ON, LESION (g pinned at 1, arousal still computed), SHUFFLED (the g sequence of the matched-pair
conversation of the other class, same seed and sample), CONSTANT (the mean ON g of that seed on every turn),
CURRENT (TAU = 1, no carry-over; decides only the claim's name).

  python -m brain.experiments.arousal_gain smoke --model 0.1b --threads 4 --out <json>
  python -m brain.experiments.arousal_gain test --model 1.5b --span 1 --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import tempfile
import time

import numpy as np
import torch

from brain.affect import calibrate_surprise
from brain.brain import BIRTH, Brain, _is_copy, _words
from brain.cortex import Cortex, resolve_model
from brain.experiments.mood_probe import NEU_EVENTS_V5_DEV, provenance

METHOD = "arousal_gain"
ARMS = ["OFF", "ON", "LESION", "SHUFFLED", "CONSTANT", "CURRENT"]
SEEDS_TEST = [42, 43, 44, 100, 101, 102]
SEEDS_DEV = [42, 43, 44]
SPANS_DEV = [0.5, 1.0, 2.0]
T4 = "How's your day going?"
FILLERS = ["It has been a busy week.", "I need to go to the shop later."]   # drives picked low at 0.1B (< MU), before any run

# 12 ordinary sentences for the (MU, SD) calibration (with V5_DEV neutral events). Disjoint from the stage-0 banks and
# from both conversation banks (tests/test_arousal_gain.py).
CAL_SENTENCES = [
    "The post office closes at five o'clock.", "I walked down to the river after lunch.",
    "She made a pot of coffee for the meeting.", "The train to the city runs every twenty minutes.",
    "We watered the plants by the back door.", "He folded the towels and put them away.",
    "The library has a quiet room on the second floor.", "It was warm enough to leave the window open.",
    "They painted the fence over the weekend.", "I need to charge my phone before we leave.",
    "The market is busy on Saturday mornings.", "She reads the newspaper at the kitchen table.",
]

# (frame with {}, common item, coined item). A conversation is T1 = the frame filled, T2 and T3 = FILLERS, T4 = T4.
# Coined words are new to every earlier bank in the repo (novelty_stage0, open_conv_bank, inquiry_gate_dev, mood banks).
DEV_FRAMES = [
    ("Yesterday my aunt gave me a {} for my birthday.", "scarf", "snudgeloom"),
    ("My landlord keeps a {} in the hallway.", "bicycle", "frabbitron"),
    ("I spent the afternoon fixing the {} in our kitchen.", "radio", "dolvimeter"),
    ("The woman next door sells {} at the weekend market.", "cookies", "pelvoncakes"),
    ("My brother just adopted a {} from the shelter.", "rabbit", "grobbet"),
    ("We ate {} at the party last night.", "sandwiches", "krundlewaffs"),
]
TEST_FRAMES = [
    ("My colleague brought a {} to the office today.", "salad", "zimbrolet"),
    ("I just bought a new {} for the living room.", "lamp", "trelvanda"),
    ("My cousin plays the {} at weddings.", "violin", "quorvain"),
    ("There is a {} on the shelf above my desk.", "clock", "jumbleflint"),
    ("My grandfather collects old {}.", "coins", "splenticks"),
    ("We watched a film about a {} last weekend.", "castle", "vorquelle"),
    ("My friend repainted her front door in {}.", "blue", "zalquinth"),
    ("I carry a {} in my bag when it rains.", "umbrella", "plondrel"),
    ("The shop on the corner sells {}.", "newspapers", "brindlecorks"),
    ("My daughter is learning to ride a {}.", "bicycle", "scootavane"),
    ("Our teacher gave us a {} to look after.", "plant", "wibblethorn"),
    ("I found a {} under the sofa this morning.", "sock", "gruntlekin"),
]


def make_bank(frames, name):
    """Conversations in matched pairs: pair i has a surprising (coined item, "S") and a mundane (common item, "M")."""
    bank = []
    for i, (frame, common, coined) in enumerate(frames):
        for cls, item in (("S", coined), ("M", common)):
            bank.append({"id": f"{name}{i:02d}{cls}", "pair": i, "cls": cls, "item": item,
                         "turns": [frame.format(item), *FILLERS, T4]})
    return bank


BANKS = {"dev": make_bank(DEV_FRAMES, "d"), "test": make_bank(TEST_FRAMES, "t")}


# ---------------------------------------------------------------- frozen mechanical measures
def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 3 else w


def carries(reply: str, item: str) -> bool:
    """The reply names the T1 item (coined word or event noun), whole word, plural-tolerant."""
    t = _stem(item.lower())
    return any(_stem(w) == t for w in _words(reply))


def leaks(reply: str) -> bool:
    return "User:" in reply


def fluency_nll(cortex, base, heard: str, reply: str):
    """Mean per-token NLL (nats) of the reply under the birth cortex, given only the turn it answers."""
    if not reply.strip():
        return None
    prompt = f"User: {heard}\n\nAssistant:"
    n_prompt = len(cortex.pipe.encode(prompt))
    nll = cortex.read_nll(f"{prompt} {reply}\n\n", base)[2]
    tail = nll[n_prompt - 1:]
    return float(tail.mean()) if len(tail) else None


# ---------------------------------------------------------------- running conversations
def provenance_full(threads):
    return provenance(threads) | {"host": socket.gethostname()}


def turn_seed(seed, conv_i, sample, turn):
    return (seed * 100003 + conv_i * 1009 + sample * 101 + turn) % (2 ** 31)


def reseed(sd):
    random.seed(sd)
    np.random.seed(sd % 2 ** 32)
    torch.manual_seed(sd)


def wrap_speak(b, max_tokens, greedy):
    """Cap / make greedy the brain's mouth for the harness (no words involved; the cortex still speaks)."""
    orig = b.cortex.speak

    def speak(logits, state, **kw):
        kw = {**kw, "max_tokens": max_tokens, **({"temperature": 0} if greedy else {})}
        return orig(logits, state, **kw)

    b.cortex.speak = speak


def brain_kwargs(arm, span, off_split):
    if arm == "OFF":
        return {"arousal_gain": 0.0} if off_split else {}
    if arm == "LESION":
        return {"arousal_gain": span, "arousal_pin": True}
    return {"arousal_gain": span}


def make_brain(home, cortex, calib, arm, span, off_split, max_tokens, greedy=False):
    b = Brain(home, cortex=cortex, sleep_on_restart=False, arousal_calibration=(calib["mu"], calib["sd"]),
              **brain_kwargs(arm, span, off_split))
    if arm == "CURRENT":
        b.arousal.TAU_AROUSAL = 1.0          # no carry-over
    wrap_speak(b, max_tokens, greedy)
    return b


def run_conv(cortex, base, conv, conv_i, arm, seed, sample, span, calib, off_split, max_tokens, scratch,
             gains=None, fluency=True):
    """One conversation in a fresh brain. Returns one row per turn."""
    with tempfile.TemporaryDirectory(dir=scratch) as home:
        b = make_brain(home, cortex, calib, arm, span, off_split, max_tokens)
        if gains is not None:
            b.arousal_gains = list(gains)
        rows = []
        for ti, text in enumerate(conv["turns"]):
            reseed(turn_seed(seed, conv_i, sample, ti))
            earlier = list(b._recent_replies)
            reply, info = b.hear(text)
            ar = info.get("arousal") or {}
            rows.append({
                "arm": arm, "seed": seed, "sample": sample, "conv": conv["id"], "cls": conv["cls"], "pair": conv["pair"],
                "turn": ti + 1, "t1": conv["turns"][0], "text": text, "reply": reply, "item": conv["item"],
                "gain": ar.get("encode_gain", 1.0), "arousal": ar.get("arousal"), "surprise": ar.get("surprise"),
                "carry": carries(reply, conv["item"]), "copy": _is_copy(reply, [text] + earlier),
                "q": "?" in reply, "leak": leaks(reply),
                "nll": fluency_nll(cortex, base, text, reply) if fluency else None})
    return rows


def run_arms(cortex, base, bank, arms, seeds, samples, span, calib, off_split, max_tokens, scratch, log=print,
             fluency=True, sink=None):
    """All arms over a bank. ON first (SHUFFLED and CONSTANT replay its gains). `sink(rows)` is called after each seed."""
    order = [a for a in ARMS if a in arms]
    t0, rows, const = time.time(), [], {}
    for seed in seeds:
        seed_rows = []
        on = {}
        for arm in order:
            for ci, conv in enumerate(bank):
                for k in range(samples):
                    gains = None
                    if arm == "SHUFFLED":
                        mate = [c for c in bank if c["pair"] == conv["pair"] and c["cls"] != conv["cls"]][0]
                        gains = on[(mate["id"], k)]
                    elif arm == "CONSTANT":
                        gains = [const[(seed, k)]] * len(conv["turns"])
                    r = run_conv(cortex, base, conv, ci, arm, seed, k, span, calib, off_split, max_tokens, scratch,
                                 gains=gains, fluency=fluency)
                    if arm == "ON":
                        on[(conv["id"], k)] = [x["gain"] for x in r]
                    seed_rows += r
            if arm == "ON":
                for k in range(samples):
                    gs = [g for (cid, kk), v in on.items() if kk == k for g in v]
                    const[(seed, k)] = float(np.mean(gs))
            log(f"[{METHOD}] seed {seed} arm {arm} done {time.time() - t0:.0f}s")
        rows += seed_rows
        for r in seed_rows:
            r["span"] = span
        if sink:
            sink(rows, {f"{s}_{k}": v for (s, k), v in const.items()})
    return rows, {f"{s}_{k}": v for (s, k), v in const.items()}


# ---------------------------------------------------------------- statistics
def sel(rows, arm=None, cls=None, turns=None, seed=None, span=None):
    return [r for r in rows if (arm is None or r["arm"] == arm) and (cls is None or r["cls"] == cls)
            and (turns is None or r["turn"] in turns) and (seed is None or r["seed"] == seed)
            and (span is None or r.get("span") == span)]


def conv_means(rows, key):
    """{conv id: mean of key over its rows} (bool -> float; None skipped)."""
    acc = {}
    for r in rows:
        v = key(r) if callable(key) else r[key]
        if v is not None:
            acc.setdefault(r["conv"], []).append(float(v))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def paired(a, b):
    """Per-conversation differences a - b over the conversations in both."""
    ks = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in ks])


def boot_ci(d, n=4000, seed=0):
    d = np.asarray(d, float)
    if len(d) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"mean": float(d.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5)),
            "n": int(len(d))}


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def pooled(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


# ---------------------------------------------------------------- setup, calibration, smoke checks
def setup(a):
    torch.set_num_threads(a.threads)
    cortex = Cortex(resolve_model(a.model))
    _, base = cortex.read(BIRTH)
    if getattr(a, "calib", None):
        j = json.load(open(a.calib))
        calib = j.get("calibration", j)
    else:
        calib = calibration(cortex, base)
    os.makedirs(a.scratch, exist_ok=True)
    return cortex, base, calib


def calibration(cortex, base):
    texts = list(NEU_EVENTS_V5_DEV) + CAL_SENTENCES
    mu, sd, xs = calibrate_surprise(cortex, base, texts)
    return {"mu": mu, "sd": sd, "n": len(texts), "texts": texts, "drives": xs, "model": cortex.model_name}


def smoke_turns():
    """20 user turns: the dev T1s (alternating coined / common), the fillers, the probe, five ordinary sentences."""
    t1 = [c["turns"][0] for c in BANKS["dev"]]
    return (t1 + FILLERS + [T4] + CAL_SENTENCES[:5])[:20]


def s1_check(cortex, base, calib, turns=20, max_tokens=60, scratch="_scratch/arousal_homes"):
    """S1: the split read (read_nll of "User: ..." then the reply prompt) against the joint read the brain does today:
    last logits within 1e-5 at every turn and the greedy replies identical over `turns` turns. Also S2: LESION is
    byte-identical to the split OFF."""
    os.makedirs(scratch, exist_ok=True)
    ts = smoke_turns()[:turns]
    out = {"turns": len(ts), "max_logit_diff": 0.0, "max_state_diff": 0.0, "replies_identical": True,
           "diff_turns": [], "same_text_read_nll_vs_read": None}
    with tempfile.TemporaryDirectory(dir=scratch) as h1, tempfile.TemporaryDirectory(dir=scratch) as h2, \
            tempfile.TemporaryDirectory(dir=scratch) as h3:
        joint = make_brain(h1, cortex, calib, "OFF", 1.0, False, max_tokens, greedy=True)
        split = make_brain(h2, cortex, calib, "OFF", 1.0, True, max_tokens, greedy=True)
        les = make_brain(h3, cortex, calib, "LESION", 1.0, False, max_tokens, greedy=True)
        l_a, s_a = cortex.read("User: Hello there.\n\nAssistant:", base)
        l_b, s_b, _ = cortex.read_nll("User: Hello there.\n\nAssistant:", base)
        out["same_text_read_nll_vs_read"] = float((l_a - l_b).abs().max())
        out["lesion_equals_split_off"] = True
        for i, t in enumerate(ts):
            st = joint.state
            lj, sj = cortex.read(f"User: {t}\n\nAssistant:", st)
            _, su, _ = cortex.read_nll(f"User: {t}\n\n", st)
            ls, ss = cortex.read("Assistant:", su)
            d = float((lj - ls).abs().max())
            ds = max(float((x - y).abs().max()) for x, y in zip(sj, ss))
            out["max_logit_diff"] = max(out["max_logit_diff"], d)
            out["max_state_diff"] = max(out["max_state_diff"], ds)
            rj, _ = joint.hear(t)
            rs, _ = split.hear(t)
            rl, _ = les.hear(t)
            if rj != rs:
                out["replies_identical"] = False
                out["diff_turns"].append(i)
            if rs != rl or not all(torch.equal(x, y) for x, y in zip(split.state, les.state)):
                out["lesion_equals_split_off"] = False
    out["S1_pass"] = bool(out["max_logit_diff"] <= 1e-5 and out["replies_identical"])
    out["off_split"] = not out["S1_pass"]    # fallback: every arm, OFF included, uses the split path
    return out


def trace_bank(cortex, base, calib, bank, span, max_tokens, scratch, greedy=True):
    """The ON arm over a bank, one greedy pass: per turn drive, arousal, gain."""
    rows = []
    for ci, conv in enumerate(bank):
        with tempfile.TemporaryDirectory(dir=scratch) as home:
            b = make_brain(home, cortex, calib, "ON", span, False, max_tokens, greedy=greedy)
            for ti, text in enumerate(conv["turns"]):
                reseed(turn_seed(0, ci, 0, ti))
                reply, info = b.hear(text)
                ar = info["arousal"]
                rows.append({"conv": conv["id"], "cls": conv["cls"], "turn": ti + 1, "text": text, "reply": reply,
                             "surprise": ar["surprise"], "arousal": ar["arousal"], "gain": ar["encode_gain"]})
    return rows


def trace_summary(rows):
    fill = [r for r in rows if r["turn"] in (2, 3)]
    a1 = {c: [r["arousal"] for r in rows if r["conv"] == c and r["turn"] == 1][0] for c in {r["conv"] for r in rows}}
    cls = {r["conv"]: r["cls"] for r in rows}
    s = [v for c, v in a1.items() if cls[c] == "S"]
    m = [v for c, v in a1.items() if cls[c] == "M"]
    frac = float(np.mean([r["gain"] <= 1.1 for r in fill])) if fill else None
    part = lambda f: (float(np.mean([r["gain"] <= 1.1 for r in fill if f(r)])) if any(f(r) for r in fill) else None)  # noqa: E731
    return {"filler_gain_le_1.1": frac, "S3_pass": bool(frac is not None and frac >= 0.9),
            # the frozen criterion pools every filler turn; by construction a surprising T1 leaves arousal/TAU on T2
            "filler_gain_le_1.1_by": {"mundane": part(lambda r: r["cls"] == "M"), "surprising": part(lambda r: r["cls"] == "S"),
                                      "T2": part(lambda r: r["turn"] == 2), "T3": part(lambda r: r["turn"] == 3),
                                      "surprising_T2": part(lambda r: r["cls"] == "S" and r["turn"] == 2)},
            "auroc_arousal_after_T1_S_vs_M": auroc(s, m),
            "mean_gain_by_turn_S": {t: float(np.mean([r["gain"] for r in rows if r["cls"] == "S" and r["turn"] == t]))
                                    for t in (1, 2, 3, 4)},
            "mean_gain_by_turn_M": {t: float(np.mean([r["gain"] for r in rows if r["cls"] == "M" and r["turn"] == t]))
                                    for t in (1, 2, 3, 4)},
            "mean_drive_by_turn_S": {t: float(np.mean([r["surprise"] for r in rows if r["cls"] == "S" and r["turn"] == t]))
                                     for t in (1, 2, 3, 4)},
            "mean_drive_by_turn_M": {t: float(np.mean([r["surprise"] for r in rows if r["cls"] == "M" and r["turn"] == t]))
                                     for t in (1, 2, 3, 4)}}


# ---------------------------------------------------------------- gates (pilot, dev)
def arm_stats(rows, arm, span=None):
    rs = sel(rows, arm=arm, span=span)
    s24 = sel(rows, arm=arm, cls="S", turns=(2, 4), span=span)
    return {"carry_S_T2T4": pooled(s24, "carry"), "carry_S_T2": pooled(sel(rows, arm=arm, cls="S", turns=(2,), span=span), "carry"),
            "echo": pooled(rs, "copy"), "questions": pooled(rs, "q"), "leaks": pooled(rs, "leak"), "nll": pooled(rs, "nll"),
            "n_replies": len(rs)}


def carry_diff(rows, a, b, span=None):
    ca = conv_means(sel(rows, arm=a, cls="S", turns=(2, 4), span=span), "carry")
    cb = conv_means(sel(rows, arm=b, cls="S", turns=(2, 4), span=span), "carry")
    return boot_ci(paired(ca, cb))


def pilot_gate(rows):
    on, con, off = (arm_stats(rows, x) for x in ("ON", "CONSTANT", "OFF"))
    d = carry_diff(rows, "ON", "CONSTANT")
    echo_rise = (on["echo"] or 0.0) - (off["echo"] or 0.0)
    ok = d["mean"] is not None and d["mean"] >= 0.10 and echo_rise <= 0.05
    return {"carry_ON_minus_CONSTANT_T2T4_surprising": d, "echo_rise_ON_minus_OFF": echo_rise,
            "gate": "GO" if ok else "NO-GO for the method (branch iii)", "arms": {"OFF": off, "ON": on, "CONSTANT": con}}


def dev_decision(rows, spans):
    """Choose SPAN: max carry ON - CONSTANT subject to fluency ratio <= 1.15 and echo rise <= 0.05; futility < +0.10;
    floor: LESION T2 carry on surprising conversations >= 0.15."""
    off_nll = pooled(sel(rows, arm="OFF"), "nll")
    les = sel(rows, arm="LESION", cls="S", turns=(2,))
    floor = pooled(les, "carry")
    per = {}
    for sp in spans:
        d = carry_diff(rows, "ON", "CONSTANT", span=sp)
        on = arm_stats(rows, "ON", sp)
        ratio = (on["nll"] / off_nll) if on["nll"] and off_nll else None
        echo_rise = (on["echo"] or 0.0) - (pooled(sel(rows, arm="OFF"), "copy") or 0.0)
        per[str(sp)] = {"carry_ON_minus_CONSTANT": d, "fluency_ratio_vs_OFF": ratio, "echo_rise_vs_OFF": echo_rise,
                        "admissible": bool(ratio is not None and ratio <= 1.15 and echo_rise <= 0.05)}
    adm = [sp for sp in spans if per[str(sp)]["admissible"] and per[str(sp)]["carry_ON_minus_CONSTANT"]["mean"] is not None]
    best = max(adm, key=lambda sp: per[str(sp)]["carry_ON_minus_CONSTANT"]["mean"]) if adm else None
    out = {"per_span": per, "chosen_span": best, "lesion_T2_carry_surprising": floor,
           "floor_ok": bool(floor is not None and floor >= 0.15)}
    if not out["floor_ok"]:
        out["verdict"] = "BANK FAILED the floor: reframe once, do not tune SPAN"
    elif best is None:
        out["verdict"] = "NO-GO: no admissible SPAN (branch iii)"
    elif per[str(best)]["carry_ON_minus_CONSTANT"]["mean"] < 0.10:
        out["verdict"] = "STOP: futility, best SPAN gives < +0.10 pooled (branch iii)"
    else:
        out["verdict"] = f"GO to test at SPAN {best}"
    return out


# ---------------------------------------------------------------- subcommands
def _common(ap, seeds):
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, seeds)))
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--calib", default=None, help="JSON with mu/sd (a calibrate output); default: computed here")
    ap.add_argument("--scratch", default="_scratch/arousal_homes")
    ap.add_argument("--off-split", choices=["auto", "yes", "no"], default="auto",
                    help="OFF arm on the split path (S1 fallback). auto: run S1 first")
    ap.add_argument("--s1-turns", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N conversations of the bank (smoke runs only)")
    ap.add_argument("--no-fluency", action="store_true")


def _off_split(a, cortex, base, calib):
    if a.off_split != "auto":
        return a.off_split == "yes", None
    s1 = s1_check(cortex, base, calib, turns=a.s1_turns, max_tokens=min(a.max_tokens, 60), scratch=a.scratch)
    return s1["off_split"], s1


def _bank(a, name):
    b = BANKS[name]
    if a.limit:    # keep matched pairs together
        b = b[:a.limit]
    return b


def _write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, default=float)


def _header(a, cortex, calib, off_split, s1):
    return {"method": METHOD, "cmd": a.cmd, "model": cortex.model_name, "calibration": calib, "off_split": off_split,
            "s1": s1, "provenance": provenance_full(a.threads), "max_tokens": a.max_tokens, "samples": a.samples}


def cmd_calibrate(a):
    cortex, base, calib = setup(a)
    _write(a.out, {"calibration": calib, "provenance": provenance_full(a.threads), "model": cortex.model_name})
    print(json.dumps({k: calib[k] for k in ("mu", "sd", "n")}))


def cmd_smoke(a):
    t0 = time.time()
    cortex, base, calib = setup(a)
    s1 = s1_check(cortex, base, calib, turns=a.s1_turns, max_tokens=min(a.max_tokens, 60), scratch=a.scratch)
    rows = trace_bank(cortex, base, calib, _bank(a, "dev"), a.span, min(a.max_tokens, 40), a.scratch)
    tr = trace_summary(rows)
    out = _header(a, cortex, calib, s1["off_split"], s1) | {"S1": s1, "S2_lesion_equals_off_split": s1["lesion_equals_split_off"],
                                                              "S3": tr, "seconds": round(time.time() - t0, 1)}
    _write(a.out, out)
    print(json.dumps({"S1_pass": s1["S1_pass"], "max_logit_diff": s1["max_logit_diff"],
                      "replies_identical": s1["replies_identical"], "off_split": s1["off_split"],
                      "S2": s1["lesion_equals_split_off"], "S3_pass": tr["S3_pass"],
                      "filler_gain_le_1.1": tr["filler_gain_le_1.1"], "auroc": tr["auroc_arousal_after_T1_S_vs_M"]}))


def cmd_trace(a):
    cortex, base, calib = setup(a)
    rows = trace_bank(cortex, base, calib, _bank(a, a.bank), a.span, a.max_tokens, a.scratch)
    tr = trace_summary(rows)
    _write(a.out, _header(a, cortex, calib, None, None) | {"bank": a.bank, "span": a.span, "rows": rows, "summary": tr})
    for r in rows:
        print(f"{r['conv']} T{r['turn']} drive {r['surprise']:.2f} arousal {r['arousal']:.3f} gain {r['gain']:.3f}  {r['text'][:50]}")
    print(json.dumps(tr))


def _run_and_write(a, bank_name, arms, spans, extra):
    cortex, base, calib = setup(a)
    off_split, s1 = _off_split(a, cortex, base, calib)
    bank = _bank(a, bank_name)
    seeds = [int(x) for x in a.seeds.split(",")]
    hdr = _header(a, cortex, calib, off_split, s1) | {"bank": bank_name, "arms": arms, "seeds": seeds, "spans": spans,
                                                       "conversations": [c["id"] for c in bank]} | extra
    rows, consts = [], {}
    t0 = time.time()
    for i, sp in enumerate(spans):
        arms_sp = arms if i == 0 else [x for x in arms if x not in ("OFF", "LESION")]   # span-independent arms once

        def sink(r, c, sp=sp):
            _write(a.out, hdr | {"rows": rows + r, "const_gain": consts | {f"{sp}:{k}": v for k, v in c.items()},
                                 "partial": True})
        r, c = run_arms(cortex, base, bank, arms_sp, seeds, a.samples, sp, calib, off_split, a.max_tokens, a.scratch,
                        fluency=not a.no_fluency, sink=sink)
        rows += r
        consts |= {f"{sp}:{k}": v for k, v in c.items()}
    hdr["seconds"] = round(time.time() - t0, 1)
    return hdr, rows, consts


def cmd_pilot(a):
    hdr, rows, consts = _run_and_write(a, "dev", ["OFF", "ON", "CONSTANT"], [a.span], {"seeds_note": "pilot"})
    gate = pilot_gate(rows)
    _write(a.out, hdr | {"rows": rows, "const_gain": consts, "pilot": gate})
    print(json.dumps(gate, default=float))


def cmd_dev(a):
    spans = [float(x) for x in a.spans.split(",")]
    hdr, rows, consts = _run_and_write(a, "dev", ["OFF", "ON", "LESION", "CONSTANT"], spans, {})
    dec = dev_decision(rows, spans)
    _write(a.out, hdr | {"rows": rows, "const_gain": consts, "dev": dec})
    print(json.dumps(dec, default=float))


def cmd_test(a):
    arms = [x for x in a.arms.split(",")]
    hdr, rows, consts = _run_and_write(a, "test", arms, [a.span], {"frozen_span": a.span})
    _write(a.out, hdr | {"rows": rows, "const_gain": consts})
    print(json.dumps({arm: arm_stats(rows, arm) for arm in arms}, default=float))


# ---------------------------------------------------------------- reader batch and the criteria table
def item_id(t1, reply):
    return hashlib.sha1((t1 + "\x00" + reply).encode()).hexdigest()[:10]


def load_rows(paths):
    rows, meta = [], []
    for p in paths:
        j = json.load(open(p))
        rows += [r | {"file": os.path.basename(p)} for r in j["rows"]]
        meta.append({k: j.get(k) for k in ("model", "frozen_span", "provenance", "calibration", "off_split")})
    return rows, meta


def cmd_export(a):
    rows, _ = load_rows(a.runs)
    items, key = {}, {}
    for r in rows:
        if r["turn"] not in (2, 4) or not r["reply"].strip():
            continue
        i = item_id(r["t1"], r["reply"])
        items[i] = {"id": i, "t1": r["t1"], "reply": r["reply"]}
        key.setdefault(i, []).append([r["arm"], r["cls"], r["seed"], r["sample"], r["conv"], r["turn"]])
    order = list(items)
    random.Random(a.seed).shuffle(order)
    batch = [items[i] for i in order]
    _write(a.out, batch)
    _write(a.key, key)
    print(json.dumps({"items": len(batch), "rows_covered": sum(len(v) for v in key.values())}))


def cohen_kappa(x, y):
    x, y = np.asarray(x, int), np.asarray(y, int)
    if not len(x):
        return None
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def label_value(labels, rows_item, field):
    """Mean of the readers' labels of one item (0, 0.5 or 1); None if unlabelled."""
    ls = labels.get(rows_item)
    if not ls:
        return None
    v = [float(bool(x.get(field, False))) for x in ls]
    return sum(v) / len(v)


def table(rows, labels, overlap_min=40):
    out = {}
    on_rows = sel(rows, arm="ON")
    # reader agreement on (a)
    both = [i for i, ls in labels.items() if len(ls) >= 2]
    kap = cohen_kappa([bool(labels[i][0].get("a")) for i in both], [bool(labels[i][1].get("a")) for i in both])
    out["reader_agreement"] = {"items_both_readers": len(both), "kappa_a": kap,
                               "ok": bool(kap is not None and kap >= 0.6 and len(both) >= overlap_min)}
    for r in rows:
        i = item_id(r["t1"], r["reply"])
        for f in ("a", "b", "c"):
            r[f"lab_{f}"] = label_value(labels, i, f) if r["turn"] in (2, 4) else None
        r["echo"] = 1.0 if r["copy"] else r.get("lab_b")
    T24 = (2, 4)

    def cmean(arm, cls, key, turns=T24):
        return conv_means(sel(rows, arm=arm, cls=cls, turns=turns), key)

    # C0
    a1 = {}
    for r in on_rows:
        if r["turn"] == 1:
            a1.setdefault(r["conv"], (r["cls"], []))[1].append(r["arousal"])
    sv = [np.mean(v) for c, v in a1.values() if c == "S"]
    mv = [np.mean(v) for c, v in a1.values() if c == "M"]
    au = auroc(sv, mv)
    fill = [r["gain"] for r in on_rows if r["turn"] in (2, 3)]
    ff = float(np.mean([g <= 1.1 for g in fill])) if fill else None
    out["C0"] = {"auroc": au, "filler_gain_le_1.1": ff, "pass": bool(au is not None and au >= 0.90 and ff is not None and ff >= 0.90)}
    # C1
    d1 = boot_ci(paired(cmean("ON", "S", "lab_a"), cmean("LESION", "S", "lab_a")))
    seeds = sorted({r["seed"] for r in on_rows})
    per_seed = {}
    for s in seeds:
        x = conv_means([r for r in sel(rows, arm="ON", cls="S", turns=T24, seed=s)], "lab_a")
        y = conv_means([r for r in sel(rows, arm="LESION", cls="S", turns=T24, seed=s)], "lab_a")
        dd = paired(x, y)
        per_seed[s] = float(dd.mean()) if len(dd) else None
    npos = sum(1 for v in per_seed.values() if v is not None and v > 0)
    out["C1"] = {"ON_minus_LESION_a": d1, "per_seed": per_seed, "seeds_positive": npos, "seeds": len(seeds),
                 "pass": bool(d1["mean"] is not None and d1["mean"] >= 0.15 and d1["lo"] > 0 and npos >= min(5, len(seeds)))}
    # C2
    d_sh = boot_ci(paired(cmean("ON", "S", "lab_a"), cmean("SHUFFLED", "S", "lab_a")))
    d_co = boot_ci(paired(cmean("ON", "S", "lab_a"), cmean("CONSTANT", "S", "lab_a")))
    out["C2"] = {"ON_minus_SHUFFLED": d_sh, "ON_minus_CONSTANT": d_co,
                 "pass": bool(all(d["mean"] is not None and d["mean"] >= 0.10 and d["lo"] > 0 for d in (d_sh, d_co)))}
    # C3
    d3 = boot_ci(paired(cmean("ON", "M", "lab_a"), cmean("LESION", "M", "lab_a")))
    out["C3"] = {"ON_minus_LESION_a_mundane": d3, "pass": bool(d3["mean"] is not None and abs(d3["mean"]) <= 0.05)}
    # C4
    les = {(r["seed"], r["sample"], r["conv"], r["turn"]): r for r in sel(rows, arm="LESION")}
    hi, lo = [], []
    for r in on_rows:
        l = les.get((r["seed"], r["sample"], r["conv"], r["turn"]))
        if l is None:
            continue
        differs = r["reply"] != l["reply"]
        if r["gain"] >= 1.2:
            hi.append(differs)
        elif r["gain"] < 1.05:
            lo.append(differs)
    fh = float(np.mean(hi)) if hi else None
    fl = float(np.mean(lo)) if lo else None
    out["C4"] = {"differ_g_ge_1.2": fh, "n_hi": len(hi), "differ_g_lt_1.05": fl, "n_lo": len(lo),
                 "pass": bool(fh is not None and fh >= 0.5 and (fl is None or fl <= 0.10))}
    # C5
    def diff(key, turns=None, cls=None):
        return boot_ci(paired(conv_means(sel(rows, arm="ON", cls=cls, turns=turns), key),
                              conv_means(sel(rows, arm="LESION", cls=cls, turns=turns), key)))
    echo, inv, qb, lk = diff("echo", T24), diff("lab_c", T24), diff("q"), diff("leak")
    nl_on, nl_les = pooled(sel(rows, arm="ON"), "nll"), pooled(sel(rows, arm="LESION"), "nll")
    ratio = nl_on / nl_les if nl_on and nl_les else None
    out["C5"] = {"echo": echo, "invents": inv, "questions_back": qb, "user_leaks": lk, "fluency_ratio_ON_over_LESION": ratio,
                 "pass": bool(all(d["mean"] is not None for d in (echo, inv, qb, lk)) and echo["mean"] <= 0.05
                              and inv["mean"] <= 0.05 and qb["mean"] <= 0.05 and lk["mean"] <= 0.0
                              and ratio is not None and ratio <= 1.15)}
    # name
    cur = sel(rows, arm="CURRENT")
    if cur:
        dn = boot_ci(paired(cmean("ON", "S", "lab_a"), cmean("CURRENT", "S", "lab_a")))
        out["name"] = {"ON_minus_CURRENT_a": dn,
                       "claim": "arousal state sets encoding gain" if dn["mean"] is not None and dn["mean"] >= 0.05
                       else "surprise-gated encoding"}
    else:
        out["name"] = {"claim": "surprise-gated encoding (no CURRENT arm in these runs)"}
    # branch
    c = {k: out[k]["pass"] for k in ("C0", "C1", "C2", "C3", "C4", "C5")}
    if not out["reader_agreement"]["ok"]:
        br = "INCONCLUSIVE (reader agreement: kappa on (a) < 0.6 or overlap < 40 items)"
    elif not c["C0"]:
        br = "(iv) C0 fails: the drive unit is wrong; re-calibrate on dev only"
    elif not c["C5"]:
        br = "(iii) C5 fails: NO-GO for flat turn-level gain; next per-token gain on high-NLL tokens"
    elif c["C1"] and not c["C2"]:
        br = "(ii) C1 holds, C2 fails: plain heard-turn gain works, arousal timing not shown; NO-GO for the arousal claim"
    elif all(c.values()):
        br = f"(i) GO: arousal shapes replies (lesion-proven), named: {out['name']['claim']} (then land-result, fresh skeptic)"
    else:
        br = "NO-GO: " + ", ".join(k for k, v in c.items() if not v) + " fail (no branch named in section 19 for this pattern)"
    out["criteria"], out["branch"] = c, br
    return out


def cmd_table(a):
    rows, meta = load_rows(a.runs)
    labels = json.load(open(a.labels))
    res = table(rows, labels) | {"runs": a.runs, "run_meta": meta}
    if a.out:
        _write(a.out, res)
    print(json.dumps({k: res[k] for k in ("reader_agreement", "criteria", "branch")}, indent=1, default=float))
    for k in ("C0", "C1", "C2", "C3", "C4", "C5", "name"):
        print(k, json.dumps(res[k], default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate"); _common(p, []); p.set_defaults(f=cmd_calibrate)
    p = sub.add_parser("smoke"); _common(p, []); p.add_argument("--span", type=float, default=1.0); p.set_defaults(f=cmd_smoke)
    p = sub.add_parser("trace"); _common(p, []); p.add_argument("--span", type=float, default=1.0)
    p.add_argument("--bank", choices=["dev", "test"], default="dev"); p.set_defaults(f=cmd_trace)
    p = sub.add_parser("pilot"); _common(p, [42]); p.add_argument("--span", type=float, default=1.0)
    p.set_defaults(f=cmd_pilot, samples=1)
    p = sub.add_parser("dev"); _common(p, SEEDS_DEV); p.add_argument("--spans", default=",".join(map(str, SPANS_DEV)))
    p.set_defaults(f=cmd_dev)
    p = sub.add_parser("test"); _common(p, SEEDS_TEST); p.add_argument("--span", type=float, required=True)
    p.add_argument("--arms", default=",".join(ARMS)); p.set_defaults(f=cmd_test)
    p = sub.add_parser("export"); p.add_argument("runs", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--key", required=True); p.add_argument("--seed", type=int, default=19); p.set_defaults(f=cmd_export)
    p = sub.add_parser("table"); p.add_argument("runs", nargs="+"); p.add_argument("--labels", required=True)
    p.add_argument("--out", default=None); p.set_defaults(f=cmd_table)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
