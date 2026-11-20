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
read-out is a TODO), no span-NLL/log-odds, no pruned-store copy, no interviews, no blind reading, no L-prunedA. The TEST split is wired (11-20, sha256-guarded).
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
TODO = ["transfer forced-choice lp read-out (incident turns are lived, not scored)", "span-NLL / log-odds",
        "pruned-store copy on full nights", "evening same-day recall in the hippocampus-off arm", "gist slots (E gist_q)"]

TEST_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_bank_test.sha256")


def use_split(split: str):
    """Point the module-level bank LB at the DEV or the TEST split. TEST is frozen: its sha256 must equal the recorded hash
    (design section 3), so an edited TEST file cannot be run silently."""
    global LB
    if split == "dev":
        LB = _DEV_BANK
        return LB
    if split != "test":
        raise ValueError(f"split must be dev or test, not {split!r}")
    from brain.eval import life_bank_test as T
    want = open(TEST_HASH_FILE).read().split()[0]
    got = hashlib.sha256(open(T.__file__, "rb").read()).hexdigest()
    if got != want:
        raise RuntimeError(f"TEST split changed: sha256 {got[:12]} != recorded {want[:12]} ({TEST_HASH_FILE})")
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


def battery(seed: int, night: int, ledger: dict) -> list:
    """The morning battery after night `night` (design 6): opens (fresh state each), then a conversation of 6 H + <=2 rotating
    direct questions by age, 2 near-duplicate frames and 3 never-told questions."""
    rng = random.Random(seed * 100003 + night)
    items = LB.ALL
    opens = LB.OPEN if len(LB.OPEN) <= 6 else random.Random(seed + 29).sample(LB.OPEN, 6)   # TEST: 6 of 12 per seed
    qs = [{"kind": f"open-{k}", "text": t, "item": None, "slot": slot} for k, t, slot in opens]
    qs.append({"kind": "greet", "text": "Good morning! How are you?", "item": None})
    qs += [{"kind": "H", "text": h.direct, "item": h.id} for h in LB.H]
    old = sorted((i for i, e in ledger.items() if e["cls"] in ("AB", "E", "F") and min(e["told_days"]) <= night),
                 key=lambda i: (min(ledger[i]["told_days"]), i))
    if old:
        half = max(len(old) // 2, 1)
        for pool in (old[:half], old[half:] or old[:half]):
            pick = rng.choice(pool)
            qs.append({"kind": "rotating", "text": items[pick].direct, "item": pick})
    dpool = [x.id for x in random.Random(seed).sample(LB.D, 2)]
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
        if q["kind"].startswith("open-"):
            b.state = [x.clone() for x in b.base]         # each open prompt starts a fresh conversation
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
        elif q["kind"].startswith("open-"):
            row["intrusion"] = _has(reply, [k for k in keys if not _has(q["text"], [k])])
            row["needless"] = rec is not None
            if q.get("slot"):
                row["named_fox"] = bool(re.search(r"\bfox (?:named|called)\s+\w+|\bnamed\s+\w+", reply, re.I))
        rows.append(row)
    return rows


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
    if extras:   # M-use / M-scale read-outs, present only when one of the new flags is on
        info.update({k: b.meta[k] for k in ("adapter_budget", "norm_pre_night", "downscale_g") if k in b.meta})
    return rows, info


def night_after_cut(home, cortex, day_no, sim, seed, brain_kw, weekend=False) -> dict:
    """Resume of a day cut between save and the night: Brain.__init__ re-sleeps what is pending (same sleep seed)."""
    sim.to(day_start(day_no, weekend) + STEP_HOURS * 3600)
    t1 = time.time()
    log = []
    b = Brain(home, cortex=cortex, sleep_log=log.append, **brain_kw)
    n = len(b.pending())
    b.save()
    b.release()
    return {"night": day_no, "seconds": round(time.time() - t1, 1), "resumed_night": True, "pending_after_open": n,
            "slept": True, "sleep_log": log[:1] + log[-1:]}


def probe(home: str, cortex, night: int, arms, qs: list, ledger: dict, sim: SimClock, seed: int, brain_kw: dict,
          work: str, check_pending: bool = True, weekend: bool = False, extras: bool = False) -> list:
    """The morning battery on a disk copy of the trunk home, one arm at a time. The copy is opened with
    sleep_on_restart=False; the trunk is never opened here."""
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
            p = b.cortex.plastic
            rec = {"night": night, "arm": arm, "rows": rows, "summary": summarize(rows), "seconds": round(time.time() - t0, 1),
                   "store": len(b.hippocampus.episodes), "adapter_norm": round(p.adapter_norm(), 4) if p is not None else 0.0,
                   "host": socket.gethostname()}
            if extras:
                rec["max_strength_open"] = max_strength
                rec["factor_norms"] = [round(x, 4) for x in p.factor_norms()] if p is not None else [0.0, 0.0]
        finally:
            b.release()
            shutil.rmtree(cp, ignore_errors=True)
        out.append(rec)
    return out


# ---------------------------------------------------------------- outputs and state
def _atomic(path: str, text: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(text)
    os.replace(path + ".tmp", path)


def write_outputs(res: dict, out: str):
    _atomic(out + ".json", json.dumps(res, indent=1))
    L = [f"# Multi-week life, seed {res['seed']} ({res['split']} split, {res['days']} days)",
         f"\n_cortex: {res.get('model')}; device: {res.get('device')}; host: {res['host']}; Brain kwargs: {res['brain_kw']};"
         f" full-probe nights {res['full_probe_nights']}; world clock from {time.strftime('%Y-%m-%d %H:%M', time.gmtime(T0))}_"]
    for line in ("trunk", "nosleep"):
        lived = [x for x in res["lived"] if x["line"] == line]
        if not lived:
            continue
        L.append(f"\n## {'life (nightly sleep)' if line == 'trunk' else 'L-nosleep (same life, never sleeps)'}")
        for x in lived:
            L.append(f"\n### L{x['day']} (lived)\n")
            L += [_line(r) for r in x["rows"]]
            n = next((n for n in res["nights"] if n["line"] == line and n["night"] == x["day"]), None)
            if n:
                L.append(f"\n_night {n['night']}: {'slept' if n['slept'] else 'no sleep'}, {n['seconds']} s; store {n.get('store')}; "
                         f"replay share {n.get('replay_share')}; adapter norm {n.get('adapter_norm')}_")
            for p in (p for p in res["probes"] if p["line"] == line and p["night"] == x["day"]):
                s = p["summary"]
                L.append(f"\n#### morning after night {p['night']}, arm: {p['arm']}\n")
                L += [_line(r) for r in p["rows"]]
                L.append(f"\n_H {s['H_right']}/{s['H_n']}; direct {s['direct_right']}/{s['direct_n']}; wrong fetch {s['wrong_fetch']}/{s['src_n']}; "
                         f"never-told false {s['never_false']}/{s['never_n']}; open intrusions {s['open_intrusions']}/{s['open_n']}; "
                         f"collapse {s['collapse']}; store {p['store']}; adapter norm {p['adapter_norm']}; {p['seconds']} s_")
    L.append("\n## TODO (not in Stage 0)\n\n" + "\n".join(f"- {t}" for t in res["todo"]))
    _atomic(out + ".md", "\n".join(L) + "\n")


def run(cortex, seed: int, days: int, split: str, homes: str, out: str, full_nights=(), nosleep_arm: bool = False,
        resume: bool = False, brain_kw: dict | None = None, weekend: bool = False, device: str = "cpu", log=print,
        frozen_probe: bool = False) -> dict:
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
    else:
        shutil.rmtree(homes, ignore_errors=True)
        os.makedirs(homes, exist_ok=True)
        state = {"seed": seed, "arms": {}}
        res = {"seed": seed, "split": split, "days": days, "model": getattr(cortex, "model_name", None), "device": device,
               "host": socket.gethostname(), "brain_kw": brain_kw, "full_probe_nights": sorted(full_nights),
               "ledger": ledger, "schedule": sched, "lived": [], "nights": [], "probes": [], "todo": TODO}
        if frozen_probe:
            res["frozen_probe"] = True
    sim = SimClock()
    clock.set(sim.now)
    try:
        for line in ("trunk",) + (("nosleep",) if nosleep_arm else ()):
            sleeps = line == "trunk"
            kw = brain_kw if sleeps else {**brain_kw, "sleep_on_restart": False}
            trunk = os.path.join(homes, line)
            work = os.path.join(homes, f"probes_{line}")
            snap = os.path.join(homes, f"snap_{line}")
            st = state["arms"].setdefault(line, {"done": 0, "phase": "idle"})

            def save_state():
                _atomic(state_path, json.dumps(state, indent=1))
            for day in sched:
                d = day["day"]
                if d <= st["done"]:
                    continue
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
                    ninfo = night_after_cut(trunk, cortex, d, sim, seed, kw, weekend) if sleeps else {"night": d, "seconds": 0.0, "slept": False}
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
                if arms:
                    qs = battery(seed, d, ledger)
                    for p in probe(trunk, cortex, d, arms, qs, ledger, sim, seed, kw, work, check_pending=sleeps, weekend=weekend,
                                   extras=extras):
                        p["line"] = line
                        res["probes"].append(p)
                    ip = next(p for p in res["probes"] if p["line"] == line and p["night"] == d and p["arm"] == "intact")
                    ninfo.update(store=ip["store"], adapter_norm=ip["adapter_norm"])
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--split", default="dev")
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
    ap.add_argument("--rank", default=None, choices=("raw",), help="R-raw: Brain(rank='raw')")
    ap.add_argument("--downscale-factors", default=None, choices=("B", "AB"), help="M-scale-3 variant: AB = sqrt(g) on A and B")
    ap.add_argument("--frozen-probe", action="store_true", help="add an intact-frozen arm (FREEZE_USE) to every morning battery")
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
    if a.rank:
        kw["rank"] = a.rank
    if a.downscale_factors:
        kw["downscale_factors"] = a.downscale_factors
    if a.inquiry is not None:
        kw["inquiry"] = None if a.inquiry.lower() == "none" else "auto" if a.inquiry.lower() == "auto" else float(a.inquiry)
    cortex = cortex or Cortex(model_path=resolve_model(a.model), device=a.device)
    full = {int(x) for x in a.full_probe_nights.split(",") if x.strip()}
    res = run(cortex, a.seed, a.days, a.split, a.homes, a.out, full, a.nosleep_arm, a.resume, kw, a.weekend, a.device, frozen_probe=a.frozen_probe)
    print(f"wrote {a.out}.json and {a.out}.md; nights {[n['seconds'] for n in res['nights'] if n['line'] == 'trunk']}")
    return res


if __name__ == "__main__":
    main()
