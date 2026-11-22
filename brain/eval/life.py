# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: schema transfer and cued recall by context at 1.5B (design section 2)
"""A multi-week simulated life with a nightly sleep (design 2026-11-19, Stage 0 = L1-L7 on the DEV split, report-only).

python -m brain.eval.life --model 1.5b --device cpu --seed 310 --days 7 --split dev --full-probe-nights 1,4,7 --nosleep-arm \
    --homes ~/brain-lane/workLIFE_s310 --out ~/brain-lane/outLIFE/life_s310 [--resume] [--key S0|S_all --key-bar slept] [--inquiry auto|none]

One trunk brain lives a day (09:00 sim clock), then `save()` + `close()` = the night, then the world clock jumps to 09:00
the next day. Each morning the battery (design section 6) runs on COPIES of the trunk home (shutil.copytree on disk, opened
with sleep_on_restart=False and `assert not pending()`), so no probe changes the life. Arms: intact and hippocampus-off every
night; consolidated-off and both-off on --full-probe-nights. --nosleep-arm lives the same life with sleep_on_restart=False.
Outputs <out>.json (rows, nights, ledger, host, seconds per phase) and <out>.md (every lived day, every probe arm in full).

Stage 0 deviations/TODOs (the design's later stages): no transfer scoring (the incident turns are lived, their lp forced-choice
read-out is a TODO), no span-NLL/log-odds, no pruned-store copy, no blind reading. Interviews: --interview (design 5 honest self-report, brain/eval/life_interview.py). L-prunedA: --pruned-a (design 4/5 savings, Stage 1). The TEST split is wired (11-20, sha256-guarded).
Day 1 is weekly.SESSION_1 verbatim (sanity anchor) so the A set is told on L2 rather than L1-L2.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import random
import re
import shutil
import socket
import sys
import time
from collections import Counter

import numpy as np
import torch

from brain import clock
from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
import hashlib

from brain.eval import life_bank as LB
from brain.eval import life_bank as _DEV_BANK
from brain.eval.open_conv import invented
from brain.eval.weekly import SESSION_1
from brain.experiments.strict_scorer import correct_answer

STEP_HOURS = 14           # the night starts at 23:00 sim time; the next day starts at 09:00
H_REMENTION_DAYS = (3, 5, 8, 11, 14)
E_DAYS = (3, 4, 5, 6, 8, 9)
D_DAYS = (3, 5, 8, 10, 13, 15)
ARMS = {"intact": (False, False), "hippocampus-off": (True, False),
        "consolidated-off": (False, True), "both-off": (True, True)}
LIGHT_ARMS = ("intact", "hippocampus-off")
FROZEN = "intact-frozen"   # --frozen-probe: an intact copy with Hippocampus.FREEZE_USE, so the battery cannot boost itself
TODO = [
        "pruned-store copy on full nights", "evening same-day recall in the hippocampus-off arm", "gist slots (E gist_q)"]

TEST_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_bank_test.sha256")
TEST2_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_bank_test2.sha256")   # TEST v2 (design 2026-11-22)
SPLITS = ("dev", "test", "test2")


def use_split(split: str):
    """Point the module-level bank LB at the DEV or the TEST split. TEST is frozen: its sha256 must equal the recorded hash
    (design section 3), so an edited TEST file cannot be run silently."""
    global LB
    if split == "dev":
        LB = _DEV_BANK
        return LB
    if split not in ("test", "test2"):
        raise ValueError(f"split must be dev, test or test2, not {split!r}")
    if split == "test":
        from brain.eval import life_bank_test as T
        hfile = TEST_HASH_FILE
    else:   # TEST v2 (design 2026-11-22 section 4): frozen by its own hash file, committed before the first v2 run
        hfile = TEST2_HASH_FILE
        if not os.path.exists(hfile):
            raise RuntimeError(f"TEST v2 split is not frozen: the hash file {hfile} is missing (write it with sha256sum "
                               "brain/eval/life_bank_test2.py at the freeze, design 2026-11-22 section 4)")
        from brain.eval import life_bank_test2 as T
    want = open(hfile).read().split()[0]
    got = hashlib.sha256(open(T.__file__, "rb").read()).hexdigest()
    if got != want:
        raise RuntimeError(f"TEST split changed: sha256 {got[:12]} != recorded {want[:12]} ({hfile})")
    LB = T
    return LB


# ---------------------------------------------------------------- world clock
T0 = calendar.timegm((2026, 11, 23, 9, 0, 0))   # a Monday, 09:00


def day_start(d: int, weekend: bool = False) -> float:
    """Sim time of 09:00 on lived day d. weekend=True adds the +48 h jump after every fifth lived day (Stage 1)."""
    off = (d - 1) + (2 * ((d - 1) // 5) if weekend else 0)
    return float(T0 + 86400 * off)


class SimClock:
    """A ticking world clock: each now() advances 1 ms, so an episode made after the last night is strictly later."""
    def __init__(self, base: float = T0):
        self.base, self.n = float(base), 0

    def now(self) -> float:
        self.n += 1
        return self.base + self.n * 1e-3

    def to(self, t: float):
        self.base, self.n = float(t), 0


def _reseed(seed: int, tag: int):
    s = seed * 1000 + tag
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


# ---------------------------------------------------------------- the life script
def build_life(seed: int, days: int, split: str = "dev"):
    """-> (schedule, ledger). schedule = [{"day": d, "turns": [{"kind", "text", "item"}]}]; ledger = {item_id: {cls, told_days,
    keys, telling, variant}}. Deterministic per seed, and day d does not depend on `days`. Day 1 is weekly.SESSION_1."""
    use_split(split)
    rng = random.Random(seed)
    ab = list(LB.AB)
    rng.shuffle(ab)
    ab = ab[:16]   # the 16-item A/B mini-pool (DEV has exactly 16; TEST samples 16 of 48 per seed)
    A, B = (ab[:8], ab[8:]) if seed % 2 == 0 else (ab[8:], ab[:8])
    Es, Fs, Ds, small = list(LB.E), list(LB.F), list(LB.D), list(LB.SMALL_TALK)
    for x in (Es, Fs, Ds, small):
        rng.shuffle(x)
    var = {"sysadmin": "X" if seed % 2 == 0 else "Y"}
    var["home repair"] = "Y" if var["sysadmin"] == "X" else "X"
    inc = {dom: rng.sample(LB.INCIDENTS[dom], 4) for dom in LB.INCIDENTS}
    ledger: dict = {}

    def told(item, d):
        e = ledger.setdefault(item.id, {"cls": item.cls, "told_days": [], "keys": list(item.keys), "telling": item.telling,
                                        "variant": None, "dup_of": item.dup_of})
        e["told_days"].append(d)

    sched = []
    for d in range(1, days + 1):
        r = random.Random(seed * 1000 + d)
        turns = []

        def T(kind, text, item=None):
            turns.append({"kind": kind, "text": text, "item": item})
        if d == 1:
            hid = {SESSION_1[1]: "admin", SESSION_1[2]: "pixel", SESSION_1[4]: "marguerite", SESSION_1[5]: "volvo"}
            for i, t in enumerate(SESSION_1):
                kind = "greet" if i == 0 else "tell" if t in hid else "sameday" if i == 6 else "small"
                T(kind, t, hid.get(t) if kind == "tell" else "pixel" if kind == "sameday" else None)
            for h in ("dan", "nova", "admin", "pixel", "marguerite", "volvo"):
                told(LB.ALL[h], 1)
            sched.append({"day": d, "turns": turns})
            continue
        tell_items = []
        if d == 2:
            tell_items += A
        if d == 11:
            tell_items += A + B   # A' (the re-telling) and the fresh set B
        if d in E_DAYS:
            tell_items.append(Es[E_DAYS.index(d) % len(Es)])
        if d in D_DAYS:
            tell_items.append(Ds[D_DAYS.index(d) % len(Ds)])
        nf = [x for x in range(2 * (d - 2), 2 * (d - 2) + 2) if x < len(Fs)]   # 2 fillers a day while the 18-item pool lasts
        tell_items += [Fs[x] for x in nf]
        tell = [("tell", it.telling, it.id) for it in tell_items]
        for dom, days_ in LB.INCIDENT_DAYS.items():
            if d in days_:
                k = days_.index(d)
                text = inc[dom][k][var[dom]]
                iid = f"inc:{dom}:{inc[dom][k]['sub']}"
                ledger.setdefault(iid, {"cls": "T", "told_days": [], "keys": [], "telling": text, "variant": var[dom],
                                        "dup_of": None})["told_days"].append(d)
                tell.append(("incident", text, iid))
        r.shuffle(tell)
        for it in tell_items:
            told(it, d)
        T("greet", LB.GREETINGS[r.randrange(len(LB.GREETINGS))])
        st = [small.pop(0) if small else None for _ in range(2)]
        small.extend(x for x in st if x)    # recycled after a day (the pool is 14)
        for i, (k, text, iid) in enumerate(tell):
            T(k, text, iid)
            if i % 2 == 1 and st:
                T("small", st.pop(0))
        for s in st:
            T("small", s)
        first = next((it for it in tell_items), None)
        if first is not None:
            T("sameday", first.direct, first.id)
        if d in H_REMENTION_DAYS:
            k = H_REMENTION_DAYS.index(d)
            for h in (LB.H[(2 * k) % 6], LB.H[(2 * k + 1) % 6]):
                T("remention", h.direct if r.random() < 0.5 else h.reworded, h.id)
        sched.append({"day": d, "turns": turns})
    return sched, ledger


def told_keys(ledger: dict, night: int) -> list:
    """Every key word of items told on or before lived day `night`."""
    return sorted({k for e in ledger.values() if e["told_days"] and min(e["told_days"]) <= night for k in e["keys"]})


# ---------------------------------------------------------------- scoring
def _has(text: str, keys) -> bool:
    return any(re.search(rf"\b{re.escape(k)}\b", text, re.I) for k in keys)


def mark_item(item, text: str, reply: str):
    """Strict correct-perspective scorer for a question about a bank item; None when there is no item."""
    if item is None:
        return None
    return any(correct_answer(reply, k, text, item.telling) for k in item.keys)


def episode_is_source(item, recalled: str | None) -> bool:
    """The fetched episode is the item's own telling (matched on episode text); H name/nova have no telling: their
    birth episode holds the key."""
    if not recalled:
        return False
    if item.telling is None:
        return _has(recalled, item.keys)
    return recalled.strip() == f"User: {item.telling}".strip() or item.telling in recalled


def battery(seed: int, night: int, ledger: dict, exclude=None) -> list:
    """The morning battery after night `night` (design 6): opens (fresh state each), then a conversation of 6 H + <=2 rotating
    direct questions by age, 2 near-duplicate frames and 3 never-told questions. exclude (--age-ladder only, design
    2026-11-22 section 2): item ids left out of the rotating draw and the D near-duplicate pool; None = unchanged."""
    rng = random.Random(seed * 100003 + night)
    items = LB.ALL
    opens = LB.OPEN if len(LB.OPEN) <= 6 else random.Random(seed + 29).sample(LB.OPEN, 6)   # TEST: 6 of 12 per seed
    qs = [{"kind": f"open-{k}", "text": t, "item": None, "slot": slot} for k, t, slot in opens]
    qs.append({"kind": "greet", "text": "Good morning! How are you?", "item": None})
    qs += [{"kind": "H", "text": h.direct, "item": h.id} for h in LB.H]
    old = sorted((i for i, e in ledger.items() if e["cls"] in ("AB", "E", "F") and min(e["told_days"]) <= night),
                 key=lambda i: (min(ledger[i]["told_days"]), i))
    if exclude is not None:
        ex = set(exclude)
        old = [i for i in old if i not in ex]
    if old:
        half = max(len(old) // 2, 1)
        for pool in (old[:half], old[half:] or old[:half]):
            pick = rng.choice(pool)
            qs.append({"kind": "rotating", "text": items[pick].direct, "item": pick})
    if exclude is None:
        dpool = [x.id for x in random.Random(seed).sample(LB.D, 2)]
    else:
        dfree = [x for x in LB.D if x.id not in ex]
        dpool = [x.id for x in random.Random(seed).sample(dfree, min(2, len(dfree)))]
    for i in dpool:
        told_ = i in ledger and min(ledger[i]["told_days"]) <= night
        qs.append({"kind": "neardup", "text": items[i].direct, "item": i, "told": told_})
    for nid, text, keys in random.Random(seed + 17).sample(LB.NEVER, 3):
        qs.append({"kind": "never", "text": text, "item": nid, "keys": list(keys)})
    return qs


def run_battery(b: Brain, qs: list, ledger: dict, night: int) -> list:
    """Ask the battery of an open Brain; returns the rows (reply, fetched episode, marks)."""
    keys = told_keys(ledger, night)
    rows = []
    for q in qs:
        if q["kind"].startswith("open-") or q["kind"] == "ab":
            b.state = [x.clone() for x in b.base]         # each open prompt (and --pruned-a AB question) starts fresh
        reply, info = b.hear(q["text"])
        rec = info.get("recalled")
        row = {"kind": q["kind"], "you": q["text"], "brain": reply, "recalled": rec, "item": q.get("item")}
        if "fetch" in info:     # only with the M-use gate or the freeze on
            row["fetch"] = info["fetch"]
        if q["kind"] == "greet":
            pass
        elif q["kind"] in ("H", "rotating") or (q["kind"] == "neardup" and q.get("told")):
            it = LB.ALL[q["item"]]
            row["ok"] = bool(mark_item(it, q["text"], reply))
            row["fetched"] = rec is not None
            row["wrong_fetch"] = rec is not None and not episode_is_source(it, rec)
            row["right_episode"] = episode_is_source(it, rec)
            if q["kind"] == "neardup":
                dup = LB.ALL[it.dup_of]
                row["named_dup_of_key"] = _has(reply, dup.keys)
        elif q["kind"] == "neardup":   # frame never told yet: any told key is a false answer
            row["told"] = False
            row["fetched"] = rec is not None
            row["false"] = _has(reply, [k for k in keys if not _has(q["text"], [k])])
            row["named_dup_of_key"] = _has(reply, LB.ALL[LB.ALL[q["item"]].dup_of].keys)
        elif q["kind"] == "never":
            row["fetched"] = rec is not None
            row["false"] = _has(reply, [k for k in keys if not _has(q["text"], [k])])
            row["invented"] = invented(reply)
        elif q["kind"] == "ab":     # --pruned-a: the A/B recall battery (strict scorer only; provisional until the reader)
            row["ok"] = bool(mark_item(LB.ALL[q["item"]], q["text"], reply))
            row["set"] = q["set"]
        elif q["kind"].startswith("open-"):
            row["intrusion"] = _has(reply, [k for k in keys if not _has(q["text"], [k])])
            row["needless"] = rec is not None
            if q.get("slot"):
                row["named_fox"] = bool(re.search(r"\bfox (?:named|called)\s+\w+|\bnamed\s+\w+", reply, re.I))
        rows.append(row)
    return rows


def _full_logits(cx, ids, state):
    """Per-token logits of `ids` read from `state`, through the consolidated synapses when they are active (the same
    path as LiveCortex.forward, full output). Never mutates `state`."""
    st = [x.clone() for x in state]
    with torch.no_grad():
        if getattr(cx, "active", False):
            p = cx.plastic
            dt = p.z["emb.weight"].dtype
            was = p.enabled
            p.enabled = True
            out = p(ids, [x.to(dt) for x in st])[0]
            p.enabled = was
        else:
            out = getattr(cx, "raw", cx).model.forward(ids, st, full_output=True)[0]
    return out[None] if out.dim() == 1 else out   # a 1-token read comes back as one row


def option_lp(b: Brain, question: str, option: str, fetch: bool = True) -> float:
    """Mean-token log-prob of `option` as the brain's reply to `question` (design 3.5 read-out 1), from a fresh
    conversation, with the hippocampal fetch and reinstatement exactly as in Brain.hear (the copy's arm lesions apply).
    Read-only on the reply path; the fetch may change episode use counters, so it runs on a probe copy after the
    battery. Host read-out (eval), no text reaches the brain."""
    b.state = [x.clone() for x in b.base]
    ep, _ = b.hippocampus.recall(b.cortex, b.base, question, question=True) if fetch else (None, 0.0)
    active = b.hippocampus.reinstate(b.cortex, b.state, ep) if ep is not None else b.state
    logits, st = b.cortex.read(f"User: {question}\n\n{b._asst}", active)
    ids = b.cortex.pipe.encode(" " + option.strip())
    first = torch.log_softmax(logits.float(), -1)[ids[0]]
    lps = [float(first)]
    if len(ids) > 1:
        full = torch.log_softmax(_full_logits(b.cortex, ids[:-1], st).float(), -1)
        lps += [float(full[t, ids[t + 1]]) for t in range(len(ids) - 1)]
    return sum(lps) / len(lps)


def answer_span(item) -> str:
    """The correct answer span: the first key as it is written in the telling (case kept), e.g. "Pixel"."""
    for k in item.keys:
        m = re.search(rf"\b{re.escape(k)}\b", item.telling or "", re.I)
        if m:
            return m.group(0)
    return item.keys[0]


def span_read(b: Brain, ledger: dict, night: int, seed: int, classes=("AB", "E", "F", "D")) -> list:
    """Design section 5 continuous read-outs for every item told by `night` (AB, E, F, D): span-NLL = minus the mean-token
    log-prob of the answer span after `User: <direct>\n\nAssistant:` from a fresh state, adapters as the arm sets them,
    no reinstatement; log-odds = that log-prob minus the mean log-prob of 3 same-slot foils (answer spans of other
    items in the bank, drawn per seed and item). age = night minus the last lived day the item was told. `classes` only
    filters which told items are read; the foil pool is always the four classes, so a restricted read has the same foils."""
    pool: dict = {}
    for it in LB.ALL.values():
        if it.telling and it.cls in ("AB", "E", "F", "D"):
            pool.setdefault(it.slot, []).append(it)
    rows = []
    for iid, e in sorted(ledger.items()):
        if e["cls"] not in classes:
            continue
        told = [d for d in e["told_days"] if d <= night]
        if not told:
            continue
        it = LB.ALL[iid]
        same = [x for x in pool.get(it.slot, []) if x.id != iid]
        foils = random.Random(f"{seed}:{iid}").sample(same, min(3, len(same)))
        lp = option_lp(b, it.direct, answer_span(it), fetch=False)
        lpf = [option_lp(b, it.direct, answer_span(f), fetch=False) for f in foils]
        rows.append({"item": iid, "cls": e["cls"], "age": night - max(told), "n_told": len(told),
                     "span_nll": round(-lp, 4), "logodds": round(lp - sum(lpf) / len(lpf), 4) if lpf else None,
                     "foils": [f.id for f in foils]})
    return rows


def _domain_diffs(b: Brain) -> dict:
    out = {}
    for dom, incs in LB.INCIDENTS.items():
        out[dom] = [round(option_lp(b, i["new"], i["opt"]["X"]) - option_lp(b, i["new"], i["opt"]["Y"]), 4)
                    for i in incs if "opt" in i]
    return out


def _T(per: dict, sgn: int):
    d = {dom: (sum(v) / len(v) if v else None) for dom, v in per.items()}
    T = None if None in (d.get("sysadmin"), d.get("home repair")) else round(sgn * (d["sysadmin"] - d["home repair"]), 4)
    return T, {k: (None if v is None else round(v, 4)) for k, v in d.items()}


def transfer_read(b: Brain, ledger: dict, night: int) -> dict:
    """T_life (design 3.5): per domain, the mean over its never-told incident questions of lp(X option) - lp(Y option);
    T = s * (d_sysadmin - d_home), s = +1 when sysadmin lives in variant X. k = incidents of the domain told by `night`.
    T0 is the same read on the same copy with both lesions on (the pretrained cortex, no fetch): the options' own
    fluency differs by domain, so within one life T carries an offset that cancels only across seed parity (0.1B smoke
    11-20: T0 = 0.58 before any incident). dT = T - T0 is the per-life read-out."""
    var, k = {}, {}
    for iid, e in ledger.items():
        if e["cls"] == "T":
            dom = iid.split(":")[1]
            var[dom] = e["variant"]
            k[dom] = k.get(dom, 0) + sum(1 for t in e["told_days"] if t <= night)
    sgn = 1 if var.get("sysadmin") == "X" else -1
    per = _domain_diffs(b)
    h_on, c_on = b.hippocampus.enabled, b.cortex.consolidated_on
    b.set_lesions(hippocampus_off=True, consolidated_off=True)
    per0 = _domain_diffs(b)
    b.hippocampus.enabled, b.cortex.consolidated_on = h_on, c_on
    T, d = _T(per, sgn)
    T0, d0 = _T(per0, sgn)
    return {"T": T, "T0": T0, "dT": None if None in (T, T0) else round(T - T0, 4), "d": d, "d0": d0, "per": per,
            "k": k, "variant": var}


def collapse(rows: list) -> float:
    """Share of the most repeated reply among the replies of a battery (0.4+ = collapse, design 5)."""
    rep = [re.sub(r"\s+", " ", r["brain"].strip().lower()) for r in rows]
    return round(max(Counter(rep).values()) / len(rep), 3) if rep else 0.0


def summarize(rows: list) -> dict:
    def n(p):
        return sum(1 for r in rows if p(r))
    src = [r for r in rows if "wrong_fetch" in r]
    opens = [r for r in rows if r["kind"].startswith("open-")]
    nev = [r for r in rows if r["kind"] == "never"]
    return {"H_right": n(lambda r: r["kind"] == "H" and r.get("ok")), "H_n": n(lambda r: r["kind"] == "H"),
            "direct_right": n(lambda r: r["kind"] in ("H", "rotating") and r.get("ok")),
            "direct_n": n(lambda r: r["kind"] in ("H", "rotating")),
            "src_n": len(src), "fetched": sum(1 for r in src if r["fetched"]), "wrong_fetch": sum(1 for r in src if r["wrong_fetch"]),
            "neardup_right": n(lambda r: r["kind"] == "neardup" and r.get("ok")),
            "neardup_named_other": n(lambda r: r["kind"] == "neardup" and r.get("named_dup_of_key")),
            "never_n": len(nev), "never_false": sum(1 for r in nev if r["false"]), "never_invented": sum(1 for r in nev if r["invented"]),
            "open_n": len(opens), "open_intrusions": sum(1 for r in opens if r["intrusion"]),
            "needless_recall": sum(1 for r in opens if r["needless"]), "collapse": collapse(rows)}


# ---------------------------------------------------------------- living and sleeping
def _line(r: dict) -> str:
    tag = "" if r.get("ok") is None else (" ✅" if r["ok"] else " ❌")
    tag += " WRONG-FETCH" if r.get("wrong_fetch") else ""
    tag += " FALSE" if r.get("false") else ""
    tag += " INTRUSION" if r.get("intrusion") else ""
    tag += " invented" if r.get("invented") else ""
    rec = f" _(recalled: {r['recalled']})_" if r.get("recalled") else ""
    return f"- [{r['kind']}] **you:** {r['you']}\n  **brain:** {r['brain']}{tag}{rec}"


def replay_share(b: Brain, ledger: dict, now: float) -> dict:
    """Expected replay steps per class tonight: steps * sum(w) / total(w), the weights the night uses."""
    cls_of = {e["telling"]: e["cls"] for e in ledger.values() if e["telling"]}
    w = {}
    for e in b._night_episodes():
        c = cls_of.get(e.text.removeprefix("User: ").strip(), "other")
        w[c] = w.get(c, 0.0) + max(b.hippocampus.effective_strength(e, now), 1e-3)
    tot = sum(w.values()) or 1.0
    steps = b.sleep_cfg["steps"]
    return {c: round(steps * v / tot, 2) for c, v in sorted(w.items())}


def replay_priority_record(b: Brain, ledger: dict) -> dict | None:
    """D2-PR night read-out from the weights the night actually used (Brain.last_replay_weights); None when the switch
    is off or nothing was replayed, so a flags-off night record is unchanged."""
    rows = getattr(b, "last_replay_weights", None)
    if not rows:
        return None
    cls_of = {e["telling"]: e["cls"] for e in ledger.values() if e["telling"]}
    by = {}
    for r in rows:
        c = "permanent" if r["permanent"] else cls_of.get(r["text"].removeprefix("User: ").strip(), "other")
        by[c] = by.get(c, 0.0) + r["share"]
    sh = sorted((r["share"] for r in rows), reverse=True)
    # the same night under today's weighting (effective strength alone): a within-night reference for the engagement
    # read-out (prereg 3.1 compares against arm B, whose JSON keeps no per-episode weights)
    st = [max(r["strength"], 1e-3) for r in rows]
    tot = sum(st) or 1.0
    ssh = sorted((x / tot for x in st), reverse=True)
    return {"shares_by_class": {c: round(v, 4) for c, v in sorted(by.items())},
            "permanent_share": round(sum(r["share"] for r in rows if r["permanent"]), 4),
            "top6_share": round(sum(sh[:6]), 4), "max_share": round(sh[0], 4),
            "strength_permanent_share": round(sum(x / tot for x, r in zip(st, rows) if r["permanent"]), 4),
            "strength_top6_share": round(sum(ssh[:6]), 4), "strength_max_share": round(ssh[0], 4),
            "mean_loss": round(sum(r["loss"] for r in rows) / len(rows), 4), "n": len(rows)}


def live_day(home: str, cortex, day: dict, ledger: dict, sim: SimClock, seed: int, brain_kw: dict, sleeps: bool,
             on_lived=None, weekend: bool = False, extras: bool = False) -> tuple[list, dict]:
    """Live one day on the trunk home from 09:00, then save and (if sleeps) close = the night. Returns (rows, night_info)."""
    d = day["day"]
    sim.to(day_start(d, weekend))
    _reseed(seed, d)
    t0 = time.time()
    log = []
    b = Brain(home, cortex=cortex, sleep_log=log.append, **brain_kw)
    rows = []
    for t in day["turns"]:
        reply, info = b.hear(t["text"])
        row = {"day": d, "kind": t["kind"], "you": t["text"], "brain": reply, "recalled": info.get("recalled"), "item": t["item"]}
        if t["kind"] in ("sameday", "remention"):
            row["ok"] = bool(mark_item(LB.ALL[t["item"]], t["text"], reply))
        if info.get("inquiry"):
            row["inquiry"] = True
        if "fetch" in info:     # only with the M-use gate on
            row["fetch"] = info["fetch"]
        rows.append(row)
    b.save()
    live_s = round(time.time() - t0, 1)
    sim.to(day_start(d, weekend) + STEP_HOURS * 3600)         # evening: the night
    if on_lived:
        on_lived(rows, live_s)
    info = {"night": d, "live_seconds": live_s, "episodes_before": len(b.hippocampus.episodes),
            "pending": len(b.pending()), "replay_share": replay_share(b, ledger, sim.now())}
    t1 = time.time()
    b.close() if sleeps else b.release()    # the night (close = sleep once if pending, save, release)
    info["seconds"] = round(time.time() - t1, 1) if sleeps else 0.0
    info["slept"] = sleeps
    info["sleep_log"] = log[:1] + log[-1:]
    rp = replay_priority_record(b, ledger) if brain_kw.get("replay_priority") is not None else None
    if rp is not None:
        info["replay_priority"] = rp
    if brain_kw.get("keep_bank") is not None and getattr(b, "last_keep_record", None):
        info["keep_bank"] = b.last_keep_record
        if "kept_pos_share" in b.last_keep_record:   # K-P-U only (prereg 2026-11-23-kp-identity section 1)
            info["kept_pos_share"] = b.last_keep_record["kept_pos_share"]
    if extras:   # M-use / M-scale read-outs, present only when one of the new flags is on
        info.update({k: b.meta[k] for k in ("adapter_budget", "norm_pre_night", "downscale_g") if k in b.meta})
    return rows, info


def night_after_cut(home, cortex, day_no, sim, seed, brain_kw, weekend=False, ledger=None) -> dict:
    """Resume of a day cut between save and the night: Brain.__init__ re-sleeps what is pending (same sleep seed)."""
    sim.to(day_start(day_no, weekend) + STEP_HOURS * 3600)
    t1 = time.time()
    log = []
    b = Brain(home, cortex=cortex, sleep_log=log.append, **brain_kw)
    n = len(b.pending())
    b.save()
    b.release()
    out = {"night": day_no, "seconds": round(time.time() - t1, 1), "resumed_night": True, "pending_after_open": n,
           "slept": True, "sleep_log": log[:1] + log[-1:]}
    rp = replay_priority_record(b, ledger) if brain_kw.get("replay_priority") is not None and ledger is not None else None
    if rp is not None:
        out["replay_priority"] = rp
    if brain_kw.get("keep_bank") is not None and getattr(b, "last_keep_record", None):
        out["keep_bank"] = b.last_keep_record
        if "kept_pos_share" in b.last_keep_record:   # K-P-U only (prereg 2026-11-23-kp-identity section 1)
            out["kept_pos_share"] = b.last_keep_record["kept_pos_share"]
    return out


def probe(home: str, cortex, night: int, arms, qs: list, ledger: dict, sim: SimClock, seed: int, brain_kw: dict,
          work: str, check_pending: bool = True, weekend: bool = False, extras: bool = False, transfer: bool = False,
          span: bool = False, span_classes=None, tag: str | None = None, age_ladder: dict | None = None) -> list:
    """The morning battery on a disk copy of the trunk home, one arm at a time. The copy is opened with
    sleep_on_restart=False; the trunk is never opened here. span_classes restricts the span read's items (foils
    unchanged); tag marks an extra probe (--pruned-a's trunk AB recall) that is not one of the night's arms.
    age_ladder (--age-ladder only): the plan; the arm's due ladder questions run on the same copy after the battery,
    transfer and span reads, each from a fresh conversation, into rec["age_ladder"] (never rec["rows"])."""
    out = []
    for arm in arms:
        h_off, c_off = ARMS.get(arm, (False, False))   # the frozen arm is an intact copy
        cp = os.path.join(work, f"probe_L{night}_{arm}")
        shutil.rmtree(cp, ignore_errors=True)
        shutil.copytree(home, cp)
        t0 = time.time()
        sim.to(day_start(night + 1, weekend))
        _reseed(seed, 500 + night)
        kw = {**brain_kw, "sleep_on_restart": False}
        b = Brain(cp, cortex=cortex, lock=True, **kw)
        try:
            if check_pending:
                assert not b.pending(), "the trunk has an unslept day: a probe copy must start from a slept home"
            b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
            if arm == FROZEN:
                b.hippocampus.FREEZE_USE = True
            if extras:
                strengths = [e.strength for e in b.hippocampus.episodes if not e.permanent]
                max_strength = round(max(strengths), 3) if strengths else 0.0
            rows = run_battery(b, qs, ledger, night)
            tr = transfer_read(b, ledger, night) if transfer else None   # after the battery: its fetches cannot touch it
            sp = (span_read(b, ledger, night, seed, **({"classes": tuple(span_classes)} if span_classes else {}))
                  if span else None)     # no fetch: the copy's synapses only
            p = b.cortex.plastic
            rec = {"night": night, "arm": arm, "rows": rows, "summary": summarize(rows), "seconds": round(time.time() - t0, 1),
                   "store": len(b.hippocampus.episodes), "adapter_norm": round(p.adapter_norm(), 4) if p is not None else 0.0,
                   "host": socket.gethostname()}
            if extras:
                rec["max_strength_open"] = max_strength
                rec["factor_norms"] = [round(x, 4) for x in p.factor_norms()] if p is not None else [0.0, 0.0]
            if tr is not None:
                rec["transfer"] = tr
            if sp is not None:
                rec["span"] = sp
            if tag:
                rec["tag"] = tag
            if age_ladder is not None:
                from brain.eval import life_age_ladder as LA
                if arm in LA.arms_for(age_ladder, night):
                    t1 = time.time()
                    rec["age_ladder"] = LA.run_age_ladder(b, age_ladder, night, ledger, LB, sim,
                                                          day_start(night + 1, weekend), seed, sys.modules[__name__])
                    rec["age_ladder_seconds"] = round(time.time() - t1, 1)
        finally:
            b.release()
            shutil.rmtree(cp, ignore_errors=True)
        out.append(rec)
    return out


# ---------------------------------------------------------------- L-prunedA (--pruned-a, design 4 and 5 savings)
AB_BRANCH_LINE = "prunedA"
AB_TAG = "ab-recall"     # the trunk's extra hippocampus-off AB recall probe on nights bd..bd+2


def ab_sets(ledger: dict, bd: int) -> tuple[list, list]:
    """From the ledger, never by hand: A = AB items first told before day bd (build_life re-tells them on L11 as A');
    B = AB items first told on bd. Sorted ids."""
    ab = {i: e for i, e in ledger.items() if e["cls"] == "AB" and e["told_days"]}
    return (sorted(i for i, e in ab.items() if min(e["told_days"]) < bd),
            sorted(i for i, e in ab.items() if min(e["told_days"]) == bd))


def _told_head(text: str) -> str:
    """The user's part of a stored episode ("User: <telling>" before the brain's reply)."""
    return text.split("\n\nAssistant:", 1)[0].strip()


def prune_episodes(b: Brain, ledger: dict, a_ids: list) -> dict:
    """Delete every A telling's episode from an open Brain (a COPY). An episode is an A episode only if its user part
    IS the telling ("User: <telling>", the reply cut off), it is not the brain's own words and not permanent: an
    episode whose reply echoes an A telling is kept (it is counted in kept_mentions, a diagnostic). The self-belief
    index in meta is remapped by object identity (dropped if that episode went)."""
    eps = b.hippocampus.episodes
    heads = {i: f"User: {ledger[i]['telling']}".strip() for i in a_ids}
    belief = b.self_belief()
    per = {i: 0 for i in a_ids}
    keep, gone = [], []
    for e in eps:
        hit = None
        if e.source != "self" and not e.permanent:
            h = _told_head(e.text)
            hit = next((i for i in a_ids if h == heads[i]), None)
        if hit is None:
            keep.append(e)
        else:
            per[hit] += 1
            gone.append(e)
    b.hippocampus.episodes = keep
    if belief is not None:
        idx = next((k for k, e in enumerate(keep) if e is belief), None)
        if idx is None:
            b.meta.pop("self_belief_unknown", None)
        else:
            b.meta["self_belief_unknown"] = idx
    kept = sorted(i for i in a_ids if any(ledger[i]["telling"] in e.text for e in keep))
    return {"per_item": per, "pruned": len(gone), "store_before": len(eps), "store_after": len(keep),
            "kept_mentions": kept, "mismatch": any(n != 1 for n in per.values()),
            "_files": [getattr(e, "_file", None) for e in gone]}


def _du(root: str) -> int:
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(root) for f in fs)


def make_pruned_branch(trunk: str, dst: str, ledger: dict, a_ids: list, cortex, brain_kw: dict, sim: SimClock,
                       seed: int, bd: int, weekend: bool = False) -> dict:
    """L-prunedA (design 4): a copytree of the trunk at L<bd> 09:00 (after night bd-1) with every A episode deleted, a
    host operation on the COPY like a lesion. The trunk is only read. last_sleep, sleeps and the adapters are untouched,
    so the branch's night seeds equal the trunk's. Built in dst.build, then moved to dst."""
    build = dst + ".build"
    shutil.rmtree(build, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    need = 2 * _du(trunk)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(dst))).free
    if free < need:
        raise RuntimeError(f"--pruned-a: {free} bytes free, the branch needs about {need}")
    shutil.copytree(trunk, build)
    sim.to(day_start(bd, weekend))
    _reseed(seed, 900 + bd)
    b = Brain(build, cortex=cortex, lock=True, **{**brain_kw, "sleep_on_restart": False})
    try:
        assert not b.pending(), "the branch must start from a slept trunk"
        rep = prune_episodes(b, ledger, a_ids)
        b.save()
    finally:
        b.release()
    for f in rep.pop("_files"):
        if f:
            p = os.path.join(build, "episodes", f)
            if os.path.exists(p):
                os.remove(p)
    os.replace(build, dst)
    return rep


def ab_questions(ledger: dict, bd: int) -> list:
    """The A/B recall battery: each item's direct question from a fresh state, A first, then B (sorted ids)."""
    a_ids, b_ids = ab_sets(ledger, bd)
    return [{"kind": "ab", "text": LB.ALL[i].direct, "item": i, "set": s}
            for s, ids in (("A", a_ids), ("B", b_ids)) for i in ids]


def sameday_ab(sched: list, ledger: dict, bd: int) -> list:
    """A/B items asked by a live same-day question up to day bd (build_life asks the day's first telling, which is
    A[0] on L2 and on L11): the sensitivity rows leave them out (retrieval practice inside the life)."""
    a_ids, b_ids = ab_sets(ledger, bd)
    return sorted({t["item"] for day in sched if day["day"] <= bd for t in day["turns"]
                   if t["kind"] == "sameday" and t["item"] in set(a_ids) | set(b_ids)})


def _mean(v):
    return round(sum(v) / len(v), 4) if v else None


def savings_read(span_rows, rows, a_ids, b_ids, exclude=()) -> dict:
    """G = mean log-odds(A') - mean log-odds(B') and dR = recall(A') - recall(B') from one copy's AB span read and AB
    recall rows. Per-item values are keyed by item id (pooling across seeds is done from the JSON). The *_excl fields
    repeat G and dR without the `exclude` items (the same-day-asked item). None where a side is empty."""
    lo = {r["item"]: r["logodds"] for r in (span_rows or []) if r.get("logodds") is not None}
    nll = {r["item"]: r["span_nll"] for r in (span_rows or [])}
    ok = {r["item"]: bool(r.get("ok")) for r in (rows or []) if r.get("kind") == "ab"}

    def g(ex):
        a = [lo[i] for i in a_ids if i in lo and i not in ex]
        b_ = [lo[i] for i in b_ids if i in lo and i not in ex]
        return None if not a or not b_ else round(sum(a) / len(a) - sum(b_) / len(b_), 4)

    def dr(ex):
        a = [i for i in a_ids if i in ok and i not in ex]
        b_ = [i for i in b_ids if i in ok and i not in ex]
        if not a or not b_:
            return None, None, None
        ka, kb = sum(ok[i] for i in a), sum(ok[i] for i in b_)
        return ka, kb, ka - kb
    ka, kb, d = dr(())
    out = {"G": g(()), "logodds": {"A": {i: lo[i] for i in a_ids if i in lo}, "B": {i: lo[i] for i in b_ids if i in lo}},
           "nll_A": _mean([nll[i] for i in a_ids if i in nll]), "nll_B": _mean([nll[i] for i in b_ids if i in nll]),
           "recall": {"A": {i: ok[i] for i in a_ids if i in ok}, "B": {i: ok[i] for i in b_ids if i in ok}},
           "recall_A": ka, "recall_B": kb, "dR": d, "n_A": len(a_ids), "n_B": len(b_ids)}
    ex = [i for i in exclude if i in set(a_ids) | set(b_ids)]
    out.update(excluded=ex, G_excl=g(set(ex)), dR_excl=dr(set(ex))[2])
    return out


def g_minus_null(off: dict, null: dict, exclude=()) -> float | None:
    """Null-subtracted savings: mean over A of (log-odds off - log-odds both-off) minus the same over B, per item. The
    pretrained cortex already rates the A and B items unequally (s318 DEV rehearsal: both-off G +3.4), so raw G mixes
    item difficulty with savings; the per-item subtraction removes it. Items missing on either copy are left out."""
    if not off or not null:
        return None
    lo, ln = off.get("logodds") or {}, null.get("logodds") or {}

    def side(k):
        a, b = lo.get(k) or {}, ln.get(k) or {}
        return [a[i] - b[i] for i in a if i in b and i not in exclude]
    da, db = side("A"), side("B")
    return None if not da or not db else round(sum(da) / len(da) - sum(db) / len(db), 4)


def savings_table(res: dict) -> dict:
    """{night: {"prunedA": {arm: savings_read}, "trunk": {"hippocampus-off": savings_read}}} for nights bd..bd+2. The
    both-off null is probed once (branch night bd) and reused for the later nights (it holds no memory). The trunk
    row uses the trunk's own hippocampus-off span read (same foils) and its tagged AB recall probe."""
    pa = res["pruned_a"]
    bd, a_ids, b_ids, ex = pa["day"], pa["A"], pa["B"], pa.get("sameday_items", [])

    def find(line, night, arm, tag=None):
        return next((p for p in res["probes"] if p["line"] == line and p["night"] == night and p["arm"] == arm
                     and p.get("tag") == tag), None)
    out = {}
    null = find(AB_BRANCH_LINE, bd, "both-off")
    for n in range(bd, bd + 3):
        row: dict = {}
        off = find(AB_BRANCH_LINE, n, "hippocampus-off")
        if off is not None:
            row["prunedA"] = {"hippocampus-off": savings_read(off.get("span"), off["rows"], a_ids, b_ids, ex)}
            if null is not None:
                row["prunedA"]["both-off"] = {**savings_read(null.get("span"), null["rows"], a_ids, b_ids, ex),
                                              "from_night": bd}
        tsp, trc = find("trunk", n, "hippocampus-off"), find("trunk", n, "hippocampus-off", AB_TAG)
        if tsp is not None or trc is not None:
            span_rows = [r for r in (tsp or {}).get("span", []) or [] if r["cls"] == "AB"]
            row["trunk"] = {"hippocampus-off": savings_read(span_rows, (trc or {}).get("rows", []), a_ids, b_ids, ex)}
        if null is not None:    # null-subtracted G (primary, design 13): the same both-off null for every line
            nr = savings_read(null.get("span"), null["rows"], a_ids, b_ids, ex)
            for part in ("prunedA", "trunk"):
                o = row.get(part, {}).get("hippocampus-off")
                if o is not None:
                    o["G_null_sub"] = g_minus_null(o, nr)
                    o["G_null_sub_excl"] = g_minus_null(o, nr, set(o.get("excluded") or ()))
                    o["dR_null_sub"] = None if o.get("dR") is None or nr.get("dR") is None else o["dR"] - nr["dR"]
        if row:
            out[str(n)] = row
    return out


# ---------------------------------------------------------------- outputs and state
def _atomic(path: str, text: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)


def _lg_md(n: dict) -> str:
    g = n.get("lang_guard")      # --lang-guard only
    if not g:
        return ""
    return f"; KL {g['kl']}; NLL on/off {g['nll_on']}/{g['nll_off']}"


def write_outputs(res: dict, out: str):
    if "report" in res.get("pruned_a", {}):     # --pruned-a only: refreshed on every write
        res["pruned_a"]["savings"] = savings_table(res)
    if "cue_ladder" in res:     # --cue-ladder only: refreshed on every write
        from brain.eval import life_ladder
        res["cue_summary"] = life_ladder.cue_summary(res)
    _atomic(out + ".json", json.dumps(res, indent=1))
    L = [f"# Multi-week life, seed {res['seed']} ({res['split']} split, {res['days']} days)",
         f"\n_cortex: {res.get('model')}; device: {res.get('device')}; host: {res['host']}; Brain kwargs: {res['brain_kw']};"
         f" full-probe nights {res['full_probe_nights']}; world clock from {time.strftime('%Y-%m-%d %H:%M', time.gmtime(T0))}_"]
    for line in ("trunk", "nosleep", AB_BRANCH_LINE):
        lived = [x for x in res["lived"] if x["line"] == line]
        if not lived:
            continue
        if line == AB_BRANCH_LINE:
            bd = res["pruned_a"]["day"]
            L.append(f"\n## L-prunedA (trunk copy at L{bd} 09:00 with all A episodes deleted; lives L{bd}-L{bd + 2})")
        else:
            L.append(f"\n## {'life (nightly sleep)' if line == 'trunk' else 'L-nosleep (same life, never sleeps)'}")
        for x in lived:
            L.append(f"\n### L{x['day']} (lived)\n")
            L += [_line(r) for r in x["rows"]]
            n = next((n for n in res["nights"] if n["line"] == line and n["night"] == x["day"]), None)
            if n:
                L.append(f"\n_night {n['night']}: {'slept' if n['slept'] else 'no sleep'}, {n['seconds']} s; store {n.get('store')}; "
                         f"replay share {n.get('replay_share')}; adapter norm {n.get('adapter_norm')}{_lg_md(n)}_")
            for p in (p for p in res["probes"] if p["line"] == line and p["night"] == x["day"]):
                s = p["summary"]
                L.append(f"\n#### morning after night {p['night']}, arm: {p['arm']}{' (' + p['tag'] + ')' if p.get('tag') else ''}\n")
                L += [_line(r) for r in p["rows"]]
                if p.get("tag") or line == AB_BRANCH_LINE:
                    ab = [r for r in p["rows"] if r["kind"] == "ab"]
                    na, nb = (sum(1 for r in ab if r["set"] == k) for k in "AB")
                    ka, kb = (sum(1 for r in ab if r["set"] == k and r["ok"]) for k in "AB")
                    L.append(f"\n_AB recall A {ka}/{na}, B {kb}/{nb}; span read {len(p.get('span') or [])} items; "
                             f"store {p['store']}; adapter norm {p['adapter_norm']}; {p['seconds']} s_")
                    continue
                L.append(f"\n_H {s['H_right']}/{s['H_n']}; direct {s['direct_right']}/{s['direct_n']}; wrong fetch {s['wrong_fetch']}/{s['src_n']}; "
                         f"never-told false {s['never_false']}/{s['never_n']}; open intrusions {s['open_intrusions']}/{s['open_n']}; "
                         f"collapse {s['collapse']}; store {p['store']}; adapter norm {p['adapter_norm']}; {p['seconds']} s_")
    if "report" in res.get("pruned_a", {}):
        L += _savings_md(res)
    if "cue_ladder" in res:     # --cue-ladder only: the cue block's own section
        from brain.eval import life_ladder
        L += life_ladder.cue_md(res)
    if "interview" in res:      # --interview only: the interview block's own section
        from brain.eval import life_interview
        L += life_interview.interview_md(res)
    if "age_ladder" in res:     # --age-ladder only: the ladder block's own section
        from brain.eval import life_age_ladder
        L += life_age_ladder.ladder_md(res)
    L.append("\n## TODO (not in Stage 0)\n\n" + "\n".join(f"- {t}" for t in res["todo"]))
    _atomic(out + ".md", "\n".join(L) + "\n")


def _savings_md(res: dict) -> list:
    pa = res["pruned_a"]
    r = pa["report"]

    def f(x):
        return "-" if x is None else f"{x:+.3f}" if isinstance(x, float) else f"{x:+d}" if isinstance(x, int) else str(x)
    L = [f"\n## Savings (L-prunedA)\n",
         f"_A (told L<{pa['day']}, re-told L{pa['day']}) {pa['A']}; B (told L{pa['day']}) {pa['B']}; pruned {r['pruned']} episodes "
         f"(store {r['store_before']} -> {r['store_after']}), per item {r['per_item']}; mismatch {r['mismatch']}; "
         f"kept mentions {r['kept_mentions']}. Sensitivity rows leave out the same-day-asked {pa.get('sameday_items')}._\n",
         "| night | G-null prunedA (primary) | G-null excl | G prunedA off | G excl | G both-off (null) | dR prunedA off | dR excl | G-null trunk | G trunk off | G trunk excl | dR trunk | dR trunk excl |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, row in sorted(pa.get("savings", {}).items(), key=lambda kv: int(kv[0])):
        po, pb = row.get("prunedA", {}).get("hippocampus-off", {}), row.get("prunedA", {}).get("both-off", {})
        t = row.get("trunk", {}).get("hippocampus-off", {})
        L.append(f"| {n} | {f(po.get('G_null_sub'))} | {f(po.get('G_null_sub_excl'))} | {f(po.get('G'))} | {f(po.get('G_excl'))} | "
                 f"{f(pb.get('G'))} | {f(po.get('dR'))} | {f(po.get('dR_excl'))} | {f(t.get('G_null_sub'))} | "
                 f"{f(t.get('G'))} | {f(t.get('G_excl'))} | {f(t.get('dR'))} | {f(t.get('dR_excl'))} |")
    L.append("\n_G and dR are provisional until the blind reader (strict scorer only) and valid only if P0 (forgetting) "
             "is VALID; no verdict here._")
    return L


def run(cortex, seed: int, days: int, split: str, homes: str, out: str, full_nights=(), nosleep_arm: bool = False,
        resume: bool = False, brain_kw: dict | None = None, weekend: bool = False, device: str = "cpu", log=print,
        frozen_probe: bool = False, transfer: bool = False, span: bool = False, pruned_a: bool = False,
        pruned_a_day: int = 11, cue_ladder: bool = False, interview: bool = False, interview_nights=None,
        lang_guard: bool = False, age_ladder: bool = False) -> dict:
    """pruned_a (--pruned-a, needs span and days >= pruned_a_day + 2): at L<bd> 09:00 the trunk is copied to the
    L-prunedA branch with every A episode deleted; the branch lives L<bd>..L<bd+2> after the other lines, and each
    branch morning runs the AB recall battery + an AB span read on hippocampus-off (both-off once, the null). The
    trunk adds a tagged hippocampus-off AB recall probe on nights bd..bd+2 (the unpruned read-out)."""
    homes, out = os.path.expanduser(homes), os.path.expanduser(out)
    brain_kw = {"sleep_seed": seed, **(brain_kw or {})}
    extras = frozen_probe or brain_kw.get("use_gate") is not None or brain_kw.get("sleep_downscale") is not None \
        or brain_kw.get("rank") is not None
    state_path = os.path.join(homes, "life_state.json")
    sched, ledger = build_life(seed, days, split)
    if resume and os.path.exists(state_path) and os.path.exists(out + ".json"):
        with open(state_path) as fh:
            state = json.load(fh)
        with open(out + ".json") as fh:
            res = json.load(fh)
        log(f"resume: {state}")
        saved = res.get("pruned_a")      # the --pruned-a config is the one the life was started with
        if saved is not None:
            if pruned_a and saved["day"] != pruned_a_day:
                raise ValueError(f"resume: this life has --pruned-a-day {saved['day']}, not {pruned_a_day}")
            pruned_a, pruned_a_day = True, saved["day"]
        elif pruned_a:
            raise ValueError("resume: this life was started without --pruned-a")
        if res.get("cue_ladder") is not None:     # the --cue-ladder config is the one the life was started with
            cue_ladder = True
        elif cue_ladder:
            raise ValueError("resume: this life was started without --cue-ladder")
        if res.get("interview") is not None:      # the --interview config is the one the life was started with
            if interview and interview_nights is not None and \
                    sorted(int(n) for n in interview_nights) != res["interview"]["nights"]:
                raise ValueError(f"resume: this life has --interview-nights {res['interview']['nights']}, "
                                 f"not {sorted(interview_nights)}")
            interview, interview_nights = True, res["interview"]["nights"]
        elif interview:
            raise ValueError("resume: this life was started without --interview")
        saved_rp = (res.get("brain_kw") or {}).get("replay_priority")    # the --replay-priority is the one the life was started with
        if brain_kw.get("replay_priority") not in (None, saved_rp):
            raise ValueError(f"resume: this life has replay_priority {saved_rp!r}, not {brain_kw['replay_priority']!r}")
        if saved_rp is not None:
            brain_kw["replay_priority"] = saved_rp
        saved_kb = (res.get("brain_kw") or {}).get("keep_bank")    # the --keep-bank is the one the life was started with
        if brain_kw.get("keep_bank") != saved_kb:
            raise ValueError(f"resume: this life has keep_bank {saved_kb!r}, not {brain_kw.get('keep_bank')!r}")
        if res.get("lang_guard") is not None:     # the --lang-guard flag is the one the life was started with
            lang_guard = True
        elif lang_guard:
            raise ValueError("resume: this life was started without --lang-guard")
        if res.get("age_ladder") is not None:     # the --age-ladder flag is the one the life was started with
            age_ladder = True
        elif age_ladder:
            raise ValueError("resume: this life was started without --age-ladder")
    else:
        shutil.rmtree(homes, ignore_errors=True)
        os.makedirs(homes, exist_ok=True)
        state = {"seed": seed, "arms": {}}
        res = {"seed": seed, "split": split, "days": days, "model": getattr(cortex, "model_name", None), "device": device,
               "host": socket.gethostname(), "brain_kw": brain_kw, "full_probe_nights": sorted(full_nights),
               "ledger": ledger, "schedule": sched, "lived": [], "nights": [], "probes": [], "todo": TODO}
        if getattr(cortex, "strategy", None) == "cuda fp32":
            res["strategy"] = "cuda fp32"      # recorded only off the default (default JSON unchanged)
        if frozen_probe:
            res["frozen_probe"] = True
        if transfer:
            res["transfer"] = True
        if span:
            res["span"] = True
    ladder_ex = None
    if age_ladder:      # design 2026-11-22 section 2: the plan is re-derived and must equal the saved one
        from brain.eval import life_age_ladder
        lplan = life_age_ladder.age_ladder_plan(seed, days, sched, ledger, LB)
        if res.get("age_ladder") is not None and res["age_ladder"] != json.loads(json.dumps(lplan)):
            raise ValueError("resume: the --age-ladder plan differs from the one this life was started with")
        res["age_ladder"] = lplan
        ladder_ex = lplan["exclude"]
        miss = [n for n in lplan["arm_nights"]["both-off"] if n not in full_nights]
        if miss:
            log(f"--age-ladder: nights {miss} are not --full-probe-nights, so they have no both-off ladder rows")
    bkw = {} if ladder_ex is None else {"exclude": ladder_ex}
    if cue_ladder:      # the plan is re-derived and must equal the saved one (same bank, cue texts and horizon)
        from brain.eval import life_ladder
        cues = life_ladder.load_cues(split)
        cfg = {"plan": life_ladder.ladder_plan(seed, days, sched, ledger, split, weekend, **bkw),
               "config": {"cues_sha256": life_ladder.cues_sha(split), "split": split, "weekend": weekend,
                          "arms": list(LIGHT_ARMS), "full_night_arms": list(life_ladder.FULL_EXTRA)}}
        saved = res.get("cue_ladder")
        if saved is not None and saved != cfg:
            raise ValueError("resume: the --cue-ladder plan or config differs from the one this life was started with "
                             "(another --days, bank or cue-text file)")
        res["cue_ladder"] = cfg
        res.setdefault("cue_probes", [])
        res["todo"] = [t for t in res["todo"] if not t.startswith("gist slots")]
    if interview:       # the config (nights, arms, n, spec hash) must equal the saved one on a resume
        from brain.eval import life_interview
        life_interview.load_spec()
        icfg = life_interview.config(interview_nights if interview_nights is not None else
                                     life_interview.SP.NIGHTS_DEFAULT)
        if res.get("interview") is not None and res["interview"] != icfg:
            raise ValueError("resume: the --interview config or spec differs from the one this life was started with")
        res["interview"] = icfg
        res.setdefault("interview_probes", [])
    if lang_guard:      # the held-out set is frozen (hash-checked) and must be the one the life was started with
        from brain.eval import life_heldout, life_langguard
        lcfg = {"heldout_sha256": life_heldout.verify(), "n_texts": len(life_heldout.HELDOUT)}
        if res.get("lang_guard") is not None and res["lang_guard"] != lcfg:
            raise ValueError("resume: the --lang-guard held-out set differs from the one this life was started with")
        res["lang_guard"] = lcfg
    bd = pruned_a_day
    if pruned_a:
        if not (span and days >= bd + 2):
            raise ValueError(f"--pruned-a needs --span and --days >= --pruned-a-day + 2 ({bd + 2})")
        res.setdefault("pruned_a", {"day": bd})
        a_ids, b_ids = ab_sets(ledger, bd)
        ab_qs = ab_questions(ledger, bd)
    sim = SimClock()
    clock.set(sim.now)
    try:
        for line in ("trunk",) + (("nosleep",) if nosleep_arm else ()) + ((AB_BRANCH_LINE,) if pruned_a else ()):
            sleeps = line in ("trunk", AB_BRANCH_LINE)
            kw = brain_kw if sleeps else {**brain_kw, "sleep_on_restart": False}
            trunk = os.path.join(homes, line)
            work = os.path.join(homes, f"probes_{line}")
            snap = os.path.join(homes, f"snap_{line}")
            if line == AB_BRANCH_LINE and line not in state["arms"]:
                raise RuntimeError("--pruned-a: the branch was never built (the trunk did not reach its day)")
            st = state["arms"].setdefault(line, {"done": 0, "phase": "idle"})

            def save_state():
                _atomic(state_path, json.dumps(state, indent=1))
            for day in sched:
                d = day["day"]
                if d <= st["done"] or (line == AB_BRANCH_LINE and d > bd + 2):
                    continue
                if (line == "trunk" and pruned_a and d == bd and st["phase"] == "idle"
                        and (AB_BRANCH_LINE not in state["arms"] or "report" not in res["pruned_a"])):
                    # L-prunedA: built from the slept trunk at L<bd> 09:00; res is written before the state, so a cut
                    # anywhere here leaves the trunk idle at bd-1 and the branch is rebuilt from scratch.
                    log(f"[{AB_BRANCH_LINE}] building the branch at L{bd} 09:00")
                    state["arms"].pop(AB_BRANCH_LINE, None)
                    for k in ("lived", "nights", "probes"):
                        res[k] = [x for x in res[k] if x["line"] != AB_BRANCH_LINE]
                    for x in ("", ".build"):
                        shutil.rmtree(os.path.join(homes, AB_BRANCH_LINE) + x, ignore_errors=True)
                    rep = make_pruned_branch(trunk, os.path.join(homes, AB_BRANCH_LINE), ledger, a_ids, cortex, kw,
                                             sim, seed, bd, weekend)
                    res["pruned_a"].update(A=a_ids, B=b_ids, sameday_items=sameday_ab(sched, ledger, bd), report=rep)
                    write_outputs(res, out)
                    state["arms"][AB_BRANCH_LINE] = {"done": bd - 1, "phase": "idle", "pruned": True}
                    save_state()
                if st["phase"] == "idle":
                    shutil.rmtree(snap, ignore_errors=True)
                    if os.path.exists(trunk):
                        shutil.copytree(trunk, snap)
                    st.update(phase="day_start", cur=d)
                    save_state()
                    resumed = False
                else:
                    resumed = True
                if st["phase"] == "day_start":
                    if resumed:   # the cut day is rerun from its saved start
                        shutil.rmtree(trunk, ignore_errors=True)
                        if os.path.exists(snap):
                            shutil.copytree(snap, trunk)
                        res["lived"] = [x for x in res["lived"] if not (x["line"] == line and x["day"] == d)]
                        res["nights"] = [x for x in res["nights"] if not (x["line"] == line and x["night"] == d)]
                    log(f"[{line}] L{d} living")

                    def mark_lived(rows, live_s, line=line, d=d):
                        res["lived"].append({"line": line, "day": d, "rows": rows, "seconds": live_s})
                        st["phase"] = "lived"
                        save_state()
                        write_outputs(res, out)
                    _, ninfo = live_day(trunk, cortex, day, ledger, sim, seed, kw, sleeps, on_lived=mark_lived, weekend=weekend,
                                        **({"extras": True} if extras else {}))
                elif st["phase"] == "lived":
                    log(f"[{line}] L{d} resuming the night")
                    ninfo = night_after_cut(trunk, cortex, d, sim, seed, kw, weekend, ledger) if sleeps else {"night": d, "seconds": 0.0, "slept": False}
                else:   # "slept": the night is done, redo the probes
                    ninfo = None
                if ninfo is not None:
                    ninfo["line"] = line
                    res["nights"].append(ninfo)
                    st["phase"] = "slept"
                    save_state()
                    write_outputs(res, out)
                ninfo = next(n for n in res["nights"] if n["line"] == line and n["night"] == d)
                last = d == days
                arms = (LIGHT_ARMS if d not in full_nights else tuple(ARMS)) if sleeps else (LIGHT_ARMS if last else ())
                if frozen_probe and arms:
                    arms = arms + (FROZEN,)
                res["probes"] = [p for p in res["probes"] if not (p["line"] == line and p["night"] == d)]
                if line == AB_BRANCH_LINE:
                    # The branch runs only the AB probes (no battery, frozen, transfer or full-night arms): the AB recall
                    # battery + the AB span read on hippocampus-off every morning, both-off (the null) once at night bd.
                    arms = ()
                    for p in probe(trunk, cortex, d, ("hippocampus-off",) + (("both-off",) if d == bd else ()), ab_qs,
                                   ledger, sim, seed, kw, work, weekend=weekend, extras=extras, span=True,
                                   span_classes=("AB",)):
                        p["line"] = line
                        res["probes"].append(p)
                    ip = next(p for p in res["probes"] if p["line"] == line and p["night"] == d)
                    ninfo.update(store=ip["store"], adapter_norm=ip["adapter_norm"])
                if arms:
                    qs = battery(seed, d, ledger, **bkw)
                    lad = {"age_ladder": res["age_ladder"]} if age_ladder and line == "trunk" else {}
                    for p in probe(trunk, cortex, d, arms, qs, ledger, sim, seed, kw, work, check_pending=sleeps, weekend=weekend,
                                   extras=extras, transfer=transfer, span=span, **lad):
                        p["line"] = line
                        res["probes"].append(p)
                    ip = next(p for p in res["probes"] if p["line"] == line and p["night"] == d and p["arm"] == "intact")
                    ninfo.update(store=ip["store"], adapter_norm=ip["adapter_norm"])
                if cue_ladder and line == "trunk":    # its own disk copy per arm, after the battery copies are gone
                    res["cue_probes"] = [p for p in res["cue_probes"] if p["night"] != d]
                    cells = [c for c in res["cue_ladder"]["plan"]["cells"] if c["night"] == d]
                    if cells:
                        carms = LIGHT_ARMS + (life_ladder.FULL_EXTRA if d in full_nights else ())
                        res["cue_probes"] += life_ladder.cue_probe(trunk, cortex, d, carms, cells, ledger, sim, seed, kw,
                                                                   work, cues, weekend=weekend, bank=LB)
                if interview and line == "trunk":     # its own disk copy per arm, after the battery copies are gone
                    res["interview_probes"] = [p for p in res["interview_probes"] if p["night"] != d]
                    if d in res["interview"]["nights"]:
                        res["interview_probes"] += life_interview.interview_probe(
                            trunk, cortex, d, tuple(res["interview"]["arms"]), ledger, sim, seed, kw, work,
                            battery(seed, d, ledger, **bkw), LB, sys.modules[__name__], weekend=weekend)
                if lang_guard and line == "trunk" and "lang_guard" not in ninfo:   # every trunk night, once
                    ninfo["lang_guard"] = life_langguard.night_read(trunk, cortex, d, sim, kw, work, weekend=weekend)
                if line == "trunk" and pruned_a and bd <= d <= bd + 2:   # the unpruned AB recall (span: the normal probe)
                    for p in probe(trunk, cortex, d, ("hippocampus-off",), ab_qs, ledger, sim, seed, kw, work,
                                   weekend=weekend, extras=extras, tag=AB_TAG):
                        p["line"] = line
                        res["probes"].append(p)
                st.update(done=d, phase="idle")
                save_state()
                write_outputs(res, out)
                log(f"[{line}] L{d} done: night {ninfo['seconds']} s")
    finally:
        clock.set(None)
    write_outputs(res, out)
    return res


def main(argv=None, cortex: Cortex | None = None, brain_kw: dict | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cuda-precision", choices=("fp16", "fp32"), default="fp16",
                    help="speaking-path precision on --device cuda (default fp16, unchanged); fp32 matches CPU numerics")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--split", default="dev", choices=SPLITS)
    ap.add_argument("--full-probe-nights", default="1,4,7")
    ap.add_argument("--nosleep-arm", action="store_true")
    ap.add_argument("--weekend", action="store_true", help="Stage 1: +48 h world-clock jump after every fifth lived day")
    ap.add_argument("--homes", required=True)
    ap.add_argument("--out", required=True, help="prefix: writes <out>.json and <out>.md")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--key", default=None)
    ap.add_argument("--key-bar", default=None)
    ap.add_argument("--use-gate", action="store_true", help="M-use: Brain(use_gate='raw_win')")
    ap.add_argument("--sleep-downscale", type=float, default=None, help="M-scale: Brain(sleep_downscale=K), K 1.0 pre-registered")
    ap.add_argument("--replay-priority", default=None, choices=("strength_x_loss", "loss", "loss_a"),
                    help="D2-PR: Brain(replay_priority=...), loss-prioritised replay (brain/research/2026-11-21-replay-priority-prereg.md); "
                         "loss_a = D2-PR2 tempered, L_e**0.6 (brain/research/2026-11-22-replay-priority-2-prereg.md)")
    ap.add_argument("--keep-bank", default=None, choices=("prior", "prior_u", "prior_0"),
                    help="K-P: Brain(keep_bank='prior'), the sleep keep set drawn from the cortex's own prior (brain/research/2026-11-21-sleep-language-keep-prereg.md); "
                         "prior_u = K-P-U (keep KL on User-turn positions only), prior_0 = K-P-0 (no frame) (brain/research/2026-11-23-kp-identity-prereg.md)")
    ap.add_argument("--rank", default=None, choices=("raw",), help="R-raw: Brain(rank='raw')")
    ap.add_argument("--downscale-factors", default=None, choices=("B", "AB"), help="M-scale-3 variant: AB = sqrt(g) on A and B")
    ap.add_argument("--downscale-when", default=None, choices=("before", "end"), help="M-scale-E: end = the cut after replay")
    ap.add_argument("--frozen-probe", action="store_true", help="add an intact-frozen arm (FREEZE_USE) to every morning battery")
    ap.add_argument("--span", action="store_true", help="span-NLL + log-odds vs 3 foils for every told item (design 5)")
    ap.add_argument("--transfer", action="store_true", help="T_life forced-choice read-out (design 3.5) on every probe copy")
    ap.add_argument("--pruned-a", action="store_true", help="L-prunedA savings branch (design 4/5); needs --span")
    ap.add_argument("--pruned-a-day", type=int, default=11, help="the day the branch is cut from the trunk (A' and B told)")
    ap.add_argument("--cue-ladder", action="store_true",
                    help="design 5 cue ladder (C0-C4) on E/F items at +1/+3/+7/+14 nights, own copy per arm per morning")
    ap.add_argument("--interview", action="store_true",
                    help="design 5 honest-self-report interview (intact, hippocampus-off, both-off), own copy per arm")
    ap.add_argument("--interview-nights", default=None,
                    help="mornings of the interview (with --interview; unset = 5,10,15, or the saved nights on --resume)")
    ap.add_argument("--lang-guard", action="store_true",
                    help="design 5 language guard: KL and NLL of the adapters-on cortex vs pretrained on a frozen held-out set, every trunk night")
    ap.add_argument("--age-ladder", action="store_true",
                    help="design 2026-11-22 age-ladder sub-battery (fixed old-age rows, slot floor, FM-v2) into probe['age_ladder']")
    ap.add_argument("--inquiry", default=None, help="auto | none | a bar in nats; unset = the Brain default")
    a = ap.parse_args(argv)
    kw = dict(brain_kw or {})
    if a.key:
        kw["key"] = a.key
    if a.key_bar:
        kw["key_bar"] = a.key_bar
    if a.use_gate:
        kw["use_gate"] = "raw_win"
    if a.sleep_downscale is not None:
        kw["sleep_downscale"] = a.sleep_downscale
    if a.replay_priority:
        kw["replay_priority"] = a.replay_priority
    if a.keep_bank:
        kw["keep_bank"] = a.keep_bank
    if a.rank:
        kw["rank"] = a.rank
    if a.downscale_factors:
        kw["downscale_factors"] = a.downscale_factors
    if a.downscale_when:
        kw["downscale_when"] = a.downscale_when
    if a.inquiry is not None:
        kw["inquiry"] = None if a.inquiry.lower() == "none" else "auto" if a.inquiry.lower() == "auto" else float(a.inquiry)
    cortex = cortex or Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision=a.cuda_precision)
    full = {int(x) for x in a.full_probe_nights.split(",") if x.strip()}
    res = run(cortex, a.seed, a.days, a.split, a.homes, a.out, full, a.nosleep_arm, a.resume, kw, a.weekend, a.device, frozen_probe=a.frozen_probe,
              transfer=a.transfer, span=a.span, **({"pruned_a": True, "pruned_a_day": a.pruned_a_day} if a.pruned_a else {}),
              **({"cue_ladder": True} if a.cue_ladder else {}), **({"lang_guard": True} if a.lang_guard else {}),
              **({"age_ladder": True} if a.age_ladder else {}),
              **({"interview": True, "interview_nights": None if a.interview_nights is None else
                    [int(x) for x in a.interview_nights.split(",") if x.strip()]} if a.interview else {}))
    print(f"wrote {a.out}.json and {a.out}.md; nights {[n['seconds'] for n in res['nights'] if n['line'] == 'trunk']}")
    return res


if __name__ == "__main__":
    main()
