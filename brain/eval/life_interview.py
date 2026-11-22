# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: honest self-report at 1.5B (claims may not track answerability; report-only until the scorer runs)
"""The interview block (design section 5, honest self-report) for brain/eval/life.py --interview.

On each interview morning (--interview-nights, default 5,10,15, trunk line only) every arm of ARMS (intact,
hippocampus-off, both-off) gets its OWN fresh disk copy of the slept trunk, opened after the battery copies are gone
and deleted after use; the trunk is never opened. Normal retrieval (no FREEZE_USE). The wording, the bucket rule and
n per bucket live in brain/eval/life_interview_spec.py, whose sha256 is checked on load.

Units: per selected item KNOW then FORCED, then the three GENERIC questions; every unit is one turn from a fresh
conversation, and every episode's strength / uses / last_used, the store list, the recent replies, the reply-monitor
log, the turn counter and the arousal state are snapshotted before the turn and restored after it, so units are
independent (use_restored logs whether the turn had changed anything). No claim extraction here: the replies go to a
sonnet extractor and brain/eval/life_interview_read.py (host only) scores them.
"""
from __future__ import annotations

import copy
import hashlib
import os
import random
import shutil
import socket
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.eval import life_interview_spec as SP
from brain.eval.open_conv import invented

SPEC_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_interview_spec.sha256")


def spec_sha() -> str:
    return hashlib.sha256(open(SP.__file__, "rb").read()).hexdigest()


def load_spec():
    """The spec module; its sha256 must equal the recorded one (fail closed)."""
    want = open(SPEC_HASH_FILE).read().split()[0]
    got = spec_sha()
    if got != want:
        raise RuntimeError(f"interview spec changed: sha256 {got[:12]} != recorded {want[:12]} ({SPEC_HASH_FILE})")
    return SP


def config(nights) -> dict:
    """What res["interview"] records (and a resume must match)."""
    return {"nights": sorted(int(n) for n in nights), "arms": list(SP.ARMS), "n": dict(SP.N),
            "recent_max_age": SP.RECENT_MAX_AGE, "old_min_age": SP.OLD_MIN_AGE, "spec": SP.VERSION,
            "spec_sha256": spec_sha()}


def _age(e: dict, night: int):
    told = [d for d in e["told_days"] if d <= night]
    return (night - max(told), told) if told else (None, [])


def select(seed: int, night: int, ledger: dict, battery_qs: list, bank) -> dict:
    """The night's items per bucket (pure, deterministic; the same in every arm). battery_qs = that morning's battery."""
    asked = {q["item"] for q in battery_qs if q["kind"] in ("rotating", "neardup", "never") and q.get("item")}
    pool = {"recent": [], "old": []}
    ages = {}
    for iid in sorted(ledger):
        e = ledger[iid]
        if e["cls"] not in SP.POOL_CLASSES or iid in asked:
            continue
        age, told = _age(e, night)
        if age is None:
            continue
        ages[iid] = (age, told)
        if age <= SP.RECENT_MAX_AGE:
            pool["recent"].append(iid)
        elif age >= SP.OLD_MIN_AGE:
            pool["old"].append(iid)
    pool["never"] = sorted(nid for nid, _, _ in bank.NEVER if nid not in asked)
    picks = {b: sorted(random.Random(f"interview:{seed}:{night}:{b}").sample(pool[b], min(SP.N[b], len(pool[b]))))
             for b in SP.BUCKETS}
    return {"night": night, "picks": picks, "pool_sizes": {b: len(pool[b]) for b in SP.BUCKETS},
            "empty": sorted(b for b in SP.BUCKETS if not picks[b]), "excluded_battery": sorted(asked),
            "old_min_age": SP.OLD_MIN_AGE, "recent_max_age": SP.RECENT_MAX_AGE,
            "ages": {i: ages[i][0] for b in ("recent", "old") for i in picks[b]},
            "told_days": {i: ages[i][1] for b in ("recent", "old") for i in picks[b]}}


def units(sel: dict, bank) -> list:
    """[{unit, kind, bucket, item, text}] in order: per item KNOW then FORCED, then the generic questions."""
    never = {nid: (q, keys) for nid, q, keys in bank.NEVER}
    out = []
    for b in SP.BUCKETS:
        for iid in sel["picks"][b]:
            if b == "never":
                q = never[iid][0]
                tp = getattr(bank, "NEVER_TOPICS", SP.NEVER_TOPICS)[q]   # TEST v2 carries its own topics
            else:
                it = bank.ALL[iid]
                q, tp = it.direct, SP.topic(it.partial)
            out.append({"unit": f"{iid}:know", "kind": "know", "bucket": b, "item": iid,
                        "text": SP.KNOW_TEMPLATE.format(topic=tp), "topic": tp})
            out.append({"unit": f"{iid}:forced", "kind": "forced", "bucket": b, "item": iid, "text": q})
    for k, q in SP.GENERIC:
        out.append({"unit": f"generic:{k}", "kind": k, "bucket": None, "item": None, "text": q})
    return out


def _turn_seed(seed: int, night: int, unit: str) -> int:
    return int(hashlib.sha256(f"interview:{seed}:{night}:{unit}".encode()).hexdigest()[:8], 16)


def turn(b: Brain, text: str, sim, t_morning: float, seed: int, night: int, unit: str):
    """One independent turn: fresh conversation, reseeded, every mutable trace restored after. -> (reply, info, changed)."""
    hp = b.hippocampus
    eps = list(hp.episodes)
    vals = [(e.strength, e.uses, e.last_used) for e in eps]
    recent, mon, turns = list(b._recent_replies), len(getattr(b, "monitor_log", [])), b.meta.get("turns")
    arousal = copy.deepcopy(b.arousal) if getattr(b, "arousal", None) is not None else None
    occ = copy.deepcopy(getattr(b, "_occ", None))
    b.state = [x.clone() for x in b.base]
    b._recent_replies = []
    sim.to(t_morning)
    s = _turn_seed(seed, night, unit)
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    try:
        reply, info = b.hear(text)
    finally:
        changed = len(hp.episodes) != len(eps) or any((e.strength, e.uses, e.last_used) != v for e, v in zip(eps, vals))
        hp.episodes = eps
        for e, (st, u, lu) in zip(eps, vals):
            e.strength, e.uses, e.last_used = st, u, lu
        b._recent_replies = recent
        if hasattr(b, "monitor_log"):
            del b.monitor_log[mon:]
        if turns is not None:
            b.meta["turns"] = turns
        if arousal is not None:
            b.arousal = arousal
        if occ is not None:
            b._occ = occ
        b.state = [x.clone() for x in b.base]
    return reply, info, changed


def run_units(b: Brain, us: list, sel: dict, ledger: dict, sim, t_morning: float, seed: int, night: int, bank,
              L) -> list:
    """Ask every unit of an open copy. L = the running life module (scorers, told_keys)."""
    keys = L.told_keys(ledger, night)
    never = {nid: keys_ for nid, _, keys_ in bank.NEVER}
    rows = []
    for u in us:
        reply, info, changed = turn(b, u["text"], sim, t_morning, seed, night, u["unit"])
        rec = info.get("recalled")
        row = {**u, "brain": reply, "recalled": rec, "fetched": rec is not None, "use_restored": changed}
        row["you"] = row.pop("text")
        if u["item"] is not None and u["bucket"] != "never":
            it = bank.ALL[u["item"]]
            row["age"] = sel["ages"][u["item"]]
            row["right_episode"] = L.episode_is_source(it, rec)
            row["wrong_fetch"] = rec is not None and not row["right_episode"]
            if u["kind"] == "forced":
                row["ok"] = bool(L.mark_item(it, u["text"], reply))
            else:   # the answer named in a KNOW reply (the question never contains it)
                row["key_hit"] = L._has(reply, it.keys)
        elif u["bucket"] == "never":
            row["false"] = L._has(reply, [k for k in keys if not L._has(u["text"], [k])])
            row["invented"] = invented(reply)
            if u["kind"] == "forced":
                row["answerable"] = False
            row["never_keys"] = list(never[u["item"]])
        else:
            row["intrusion"] = L._has(reply, [k for k in keys if not L._has(u["text"], [k])])
        rows.append(row)
    return rows


def week_ledger(ledger: dict, night: int) -> list:
    """Items told in the last 5 lived days up to the night (report-only provenance for 'this week')."""
    return sorted(i for i, e in ledger.items() if any(night - 4 <= d <= night for d in e["told_days"]))


def interview_probe(home: str, cortex, night: int, arms, ledger: dict, sim, seed: int, brain_kw: dict, work: str,
                    battery_qs: list, bank, L, weekend: bool = False, check_pending: bool = True) -> list:
    """The interview of one morning: one fresh disk copy of the trunk per arm (never the battery's copy)."""
    load_spec()
    sel = select(seed, night, ledger, battery_qs, bank)
    us = units(sel, bank)
    out = []
    for arm in arms:
        h_off, c_off = L.ARMS[arm]
        cp = os.path.join(work, f"interview_L{night}_{arm}")
        shutil.rmtree(cp, ignore_errors=True)
        shutil.copytree(home, cp)
        t0 = time.time()
        t_m = L.day_start(night + 1, weekend)
        sim.to(t_m)
        b = Brain(cp, cortex=cortex, lock=True, **{**brain_kw, "sleep_on_restart": False})
        try:
            if check_pending:
                assert not b.pending(), "the trunk has an unslept day: an interview copy must start from a slept home"
            b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
            store0 = len(b.hippocampus.episodes)
            rows = run_units(b, us, sel, ledger, sim, t_m, seed, night, bank, L)
            rec = {"night": night, "arm": arm, "selection": sel, "rows": rows, "week_ledger": week_ledger(ledger, night),
                   "replies": len(rows), "store": store0, "store_after": len(b.hippocampus.episodes),
                   "seconds": round(time.time() - t0, 1), "host": socket.gethostname()}
        finally:
            b.release()
            shutil.rmtree(cp, ignore_errors=True)
        out.append(rec)
    return out


def interview_md(res: dict) -> list:
    cfg = res["interview"]
    M = ["\n## Interviews (--interview, design 5 honest self-report)\n",
         f"_Nights {cfg['nights']}; arms {cfg['arms']}; n {cfg['n']}; recent age <= {cfg['recent_max_age']}, old age >= "
         f"{cfg['old_min_age']} (an empty bucket stays empty); spec {cfg['spec']} sha256 {cfg['spec_sha256'][:12]}. Each "
         "unit is one turn from a fresh conversation on the arm's own disk copy, every episode trace restored after it. "
         "Claims are scored post hoc (sonnet extractor + life_interview_read.py); nothing is scored here._"]
    for p in res.get("interview_probes", []):
        sel = p["selection"]
        M.append(f"\n#### interview, morning after night {p['night']}, arm: {p['arm']} ({p['seconds']} s)\n")
        M.append(f"_picks {sel['picks']}; pool sizes {sel['pool_sizes']}; empty buckets {sel['empty'] or 'none'}; "
                 f"excluded (battery) {sel['excluded_battery']}_\n")
        for r in p["rows"]:
            tag = "" if r.get("ok") is None else (" ✅" if r["ok"] else " ❌")
            tag += " WRONG-FETCH" if r.get("wrong_fetch") else ""
            tag += " FALSE" if r.get("false") else ""
            tag += " invented" if r.get("invented") else ""
            age = f", age {r['age']}" if r.get("age") is not None else ""
            lab = f"{r['kind']} {r['bucket']} {r['item']}{age}" if r["item"] else r["kind"]
            recd = f" _(recalled: {r['recalled']})_" if r.get("recalled") else ""
            M.append(f"- [{lab}] **you:** {r['you']}\n  **brain:** {r['brain']}{tag}{recd}")
    return M
