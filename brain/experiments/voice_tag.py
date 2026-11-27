# Capability: Self and identity
# Record: brain/research/2026-11-26-source-tag-prereg-DRAFT.md
# Expected-fail: the self/other source tag is missing (PREREQUISITES Self and identity, NOT-READY); a GO ~20% (judge)
"""Source-tag screen (DEV, offline, CPU): re-sleep a life's last night with a learned user-channel tag, then ask every
told item's own probe question, fresh state per question.

The last night N of a KR life is re-slept from night N-1's slow store (meta, adapters, keep bank) with night N's
episodes. The arms are:
  A0 (no tag), TAG (a learned vector on "User:" tokens, brain/source_tag.py), SHAM (a fixed random vector with the
  same norm as TAG's). The batteries are A0, TAG, LESION (TAG's adapters, tag removed), SHAM, plus A0snap (the banked
  night-N adapters).
Each battery is asked hippocampus-off (primary) and intact. The hippocampus-off battery also records the onset
log-odds, log(p(" You") + p(" Your")) - logp(" My"), at the first answer token (R2). brain.py, sleep.py and cortex.py are untouched; the
tag lives in PlasticCortex."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import socket
import time

import torch

from brain import clock
from brain import source_tag as ST
from brain.experiments import cascade_screen as CS
from brain.experiments import routing_replay as RR
from brain.experiments import voice_split as VS

NIGHT = 14
CONDS = (("hoff", True, False), ("intact", False, False))
SLEPT_FILES = ("meta.json", "cortex_adapters.pt", "keep_bank.json")
BOOT, BOOT_SEED = 2000, 20261127
ENGAGE_MIN_KL, ENGAGE_RATIO = 0.01, 2.0


def told_questions(res, night=NIGHT):
    """Every told item's probe question from the banked batteries of nights 1..night (deduplicated, in first order)."""
    led, seen, out = res["ledger"], set(), []
    for n in range(1, night + 1):
        for q, r in zip(RR._qs(RR.banked_rows(res, n)), RR.banked_rows(res, n)):
            if q.get("item") and CS.is_told(r, led) and q["text"] not in seen:
                seen.add(q["text"])
                out.append(q)
    return out


def _open(snap, files_from, home, cortex, brain_kw, trunk):
    """open_snapshot on `snap`, with the files named in files_from ({name: path}) copied over the snapshot's."""
    orig = RR.build_home

    def bh(s, h, t=None, with_deltas=False):
        rows = orig(s, h, t, with_deltas)
        for name, path in files_from.items():
            shutil.copy2(path, os.path.join(h, name))
        return rows
    RR.build_home = bh
    try:
        return RR.open_snapshot(snap, home, cortex, brain_kw, trunk=trunk, with_deltas=True)
    finally:
        RR.build_home = orig


def _state(p):
    return {k: v.detach().cpu().clone() for k, v in p.state_dict().items() if k != "source_tag"}


def resleep(res, snaps, trunk, cortex, work, mode, vec=None, layer=None, night=NIGHT, log=print):
    """Night `night` slept again from night-1's store. mode: "none" | "learn" (zero init) | "fixed" (vec)."""
    seed = res["seed"]
    prev, cur = os.path.join(snaps, f"n{night - 1}"), os.path.join(snaps, f"n{night}")
    when = json.load(open(os.path.join(cur, "meta.json")))["last_sleep"]
    home = os.path.join(work, f"s{seed}_sleep_{mode}")
    clock.set(lambda: when)
    t0 = time.time()
    b = _open(cur, {f: os.path.join(prev, f) for f in SLEPT_FILES if os.path.exists(os.path.join(prev, f))}, home,
              cortex, dict(res["brain_kw"]), trunk)
    try:
        assert b.meta.get("sleeps") == night - 1, b.meta.get("sleeps")
        b.hippocampus.FREEZE_USE = False
        p = b._plastic()
        if mode != "none":
            ST.attach(p, torch.zeros(p.C) if mode == "learn" else vec, layer, learn=mode == "learn")
        losses = b.sleep()
        rec = {"mode": mode, "losses": losses, "adapter_norm": p.adapter_norm(), "seconds": round(time.time() - t0, 1),
               "state": _state(p), "tag": None if p.source_tag is None else p.source_tag.detach().cpu().clone(),
               "layer": layer, "n_episodes": len(b.hippocampus.episodes)}
    finally:
        b.release()
        shutil.rmtree(home, ignore_errors=True)
        clock.set(None)
    log(seed, "sleep", mode, rec["seconds"], "adapter_norm %.4f" % rec["adapter_norm"],
        "tag_norm", None if rec["tag"] is None else round(float(rec["tag"].norm()), 4))
    return rec


def reproduction(state, snap_adapters):
    """A0 re-slept on this host against the banked night (slept on another device): relative difference and cosine."""
    blob = torch.load(snap_adapters, map_location="cpu", weights_only=False)["adapters"]
    a = torch.cat([state[k].flatten().double() for k in sorted(blob)])
    b = torch.cat([blob[k].flatten().double() for k in sorted(blob)])
    return {"rel_diff": float((a - b).norm() / b.norm()), "cosine": float(a @ b / (a.norm() * b.norm()))}


def _onset_ids(pipe):
    """(second-person first tokens, user-voice first token): ([" You", " Your"], " My"), each one token (R2)."""
    ids = [pipe.encode(w) for w in (" You", " Your", " My")]
    assert all(len(i) == 1 for i in ids), ids
    return [ids[0][0], ids[1][0]], ids[2][0]


def battery(res, snaps, trunk, cortex, work, qs, adapters, vec, layer, tag_name, night=NIGHT, conds=CONDS, weekend=True):
    """Ask every told question, fresh state each, with the given adapters (a saved state file) and tag (or None)."""
    from brain.eval import life
    seed, led = res["seed"], res["ledger"]
    cur = os.path.join(snaps, f"n{night}")
    sim = life.SimClock()
    clock.set(sim.now)
    out = {}
    try:
        for cond, h_off, c_off in conds:
            home = os.path.join(work, f"s{seed}_b_{tag_name}_{cond}")
            t0 = time.time()
            sim.to(life.day_start(night + 1, weekend))
            life._reseed(seed, 500 + night)
            b = _open(cur, {"cortex_adapters.pt": adapters} if adapters else {}, home, cortex, dict(res["brain_kw"]), trunk)
            try:
                b.hippocampus.FREEZE_USE = False
                if vec is not None:
                    ST.attach(b._plastic(), vec, layer, learn=False)
                b.cortex.refresh()
                b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
                yid, mid = _onset_ids(b.cortex.pipe)
                rows = []
                for q in qs:
                    onset = None
                    if h_off:
                        with torch.no_grad():
                            lg, _ = b.cortex.read(f"User: {q['text']}\n\n{b._asst}", [x.clone() for x in b.base])
                            lp = torch.log_softmax(lg.float(), -1)
                            onset = float(torch.logsumexp(lp[yid], 0) - lp[mid])
                    b.state = [x.clone() for x in b.base]
                    life._reseed(seed, 500 + night)   # paired draws: every arm samples each reply from the same RNG state
                    r = life.run_battery(b, [q], led, night)[0]
                    rows.append({**{k: r.get(k) for k in VS.KEEP}, "onset": onset})
            finally:
                b.release()
                shutil.rmtree(home, ignore_errors=True)
            out[cond] = {"rows": rows, **VS.voice(rows, led), "seconds": round(time.time() - t0, 1)}
    finally:
        clock.set(None)
    return out


def onset_kl(res, snaps, trunk, cortex, work, qs, adapters, vec, layer, name, night=NIGHT):
    """Engagement: KL(tag on || tag off) of the first answer token, hippocampus off, per told question."""
    seed = res["seed"]
    home = os.path.join(work, f"s{seed}_kl_{name}")
    b = _open(os.path.join(snaps, f"n{night}"), {"cortex_adapters.pt": adapters}, home, cortex, dict(res["brain_kw"]),
              trunk)
    try:
        b.cortex.refresh()
        p = b._plastic()
        kls = []
        for q in qs:
            text = f"User: {q['text']}\n\n{b._asst}"
            with torch.no_grad():
                ST.detach(p)
                off = torch.log_softmax(b.cortex.read(text, [x.clone() for x in b.base])[0].float(), -1)
                ST.attach(p, vec, layer, learn=False)
                on = torch.log_softmax(b.cortex.read(text, [x.clone() for x in b.base])[0].float(), -1)
            kls.append(float((on.exp() * (on - off)).sum()))
        return kls
    finally:
        b.release()
        shutil.rmtree(home, ignore_errors=True)


def run(res, snaps, trunk, cortex, work, out_path=None, night=NIGHT, steps=None, log=print, smoke=False):
    os.makedirs(work, exist_ok=True)
    seed = res["seed"]
    qs = told_questions(res, night)
    L = cortex.n_layers // 2
    assert L >= 1, cortex.n_layers
    R = {"seed": seed, "host": socket.gethostname(), "device": str(getattr(cortex, "strategy", "cpu")).split()[0], "strategy": str(getattr(cortex, "strategy", "?")), "night": night,
         "layer": L, "n_questions": len(qs), "sleeps": {}, "batteries": {}}

    def save():
        if out_path:
            json.dump(R, open(out_path + ".tmp", "w"), indent=1)
            os.replace(out_path + ".tmp", out_path)

    def keep(name, rec):
        path = os.path.join(work, f"s{seed}_adapters_{name}.pt")
        torch.save({"rank": 8, "layers": None, "adapters": rec["state"]}, path)
        R["sleeps"][name] = {k: v for k, v in rec.items() if k not in ("state", "tag")}
        R["sleeps"][name]["tag_norm"] = None if rec["tag"] is None else float(rec["tag"].norm())
        return path

    a0 = resleep(res, snaps, trunk, cortex, work, "none", night=night, log=log)
    a0_path = keep("A0", a0)
    R["reproduction"] = reproduction(a0["state"], os.path.join(snaps, f"n{night}", "cortex_adapters.pt"))
    log(seed, "reproduction", R["reproduction"])
    save()
    plan = [("A0snap", os.path.join(snaps, f"n{night}", "cortex_adapters.pt"), None, (CONDS[0],)),
            ("A0", a0_path, None, CONDS)]
    if not smoke:
        tg = resleep(res, snaps, trunk, cortex, work, "learn", layer=L, night=night, log=log)
        tg_path = keep("TAG", tg)
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        d = torch.randn(tg["tag"].shape, generator=g, dtype=torch.float64)
        sham = (d / d.norm() * float(tg["tag"].norm())).to(tg["tag"].dtype)
        sh = resleep(res, snaps, trunk, cortex, work, "fixed", vec=sham, layer=L, night=night, log=log)
        sh_path = keep("SHAM", sh)
        R["sham_norm"] = float(sham.norm())
        torch.save({"tag": tg["tag"], "sham": sham, "layer": L}, os.path.join(work, f"s{seed}_tags.pt"))
        save()
        R["engagement"] = {"TAG": onset_kl(res, snaps, trunk, cortex, work, qs, tg_path, tg["tag"], L, "TAG", night),
                           "SHAM": onset_kl(res, snaps, trunk, cortex, work, qs, sh_path, sham, L, "SHAM", night)}
        save()
        plan += [("TAG", tg_path, tg["tag"], CONDS), ("LESION", tg_path, None, CONDS), ("SHAM", sh_path, sham, CONDS)]
    for name, path, vec, conds in plan:
        R["batteries"][name] = battery(res, snaps, trunk, cortex, work, qs, path, vec, L, name, night, conds)
        log(seed, name, {c: (v["uv_knows"], v["knows"], v["seconds"]) for c, v in R["batteries"][name].items()})
        save()
    return R


# ------------------------------------------------------------------ scoring (rule: the prereg, fixed before data)
def _cells(outs, ledgers, arm, cond, f):
    """{(seed, item, question): value} for told rows, f(row, ledger) -> value or None."""
    c = {}
    for o in outs:
        led = ledgers[o["seed"]]
        for r in o["batteries"][arm][cond]["rows"]:
            if r.get("item") and CS.is_told(r, led):
                v = f(r, led)
                if v is not None:
                    c[(o["seed"], r["item"], r["you"])] = v
    return c


def _boot_diff(a, b, draws=BOOT, seed=BOOT_SEED):
    keys = sorted(set(a) & set(b))
    if not keys:
        return {"diff": None, "ci": None, "n": 0}
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(draws))
    return {"diff": sum(d) / len(d), "ci": [ms[int(0.025 * draws)], ms[int(0.975 * draws) - 1]], "n": len(keys)}


def _rate(outs, ledgers, arm, cond):
    n = k = u = rf = 0
    for o in outs:
        led = ledgers[o["seed"]]
        for r in o["batteries"][arm][cond]["rows"]:
            if not (r.get("item") and CS.is_told(r, led)):
                continue
            n += 1
            kn = VS.knows(r, led)
            k += kn
            u += bool(kn and CS.own2(r["brain"]))
            rf += bool(CS.refusal(r["brain"]))
    return {"told": n, "knows": k, "uv_knows": u, "p_uv": (u / k) if k else None, "refusal": rf}


def table(outs, ledgers):
    onset = lambda r, led: r.get("onset")
    T = {"reproduction": {o["seed"]: o.get("reproduction") for o in outs}}
    rates = {a: {c: _rate(outs, ledgers, a, c) for c, _, _ in CONDS if c in outs[0]["batteries"][a]}
             for a in outs[0]["batteries"]}
    T["rates"] = rates
    ons = {a: _cells(outs, ledgers, a, "hoff", onset) for a in outs[0]["batteries"]}
    T["onset_TAG-A0"] = _boot_diff(ons["TAG"], ons["A0"])
    T["onset_TAG-SHAM"] = _boot_diff(ons["TAG"], ons["SHAM"])
    T["onset_LESION-A0"] = _boot_diff(ons["LESION"], ons["A0"])
    T["onset_A0-A0snap"] = _boot_diff(ons["A0"], ons["A0snap"])
    kt = [x for o in outs for x in o["engagement"]["TAG"]]
    ks = [x for o in outs for x in o["engagement"]["SHAM"]]
    T["engagement"] = {"TAG_mean_kl": sum(kt) / len(kt), "SHAM_mean_kl": sum(ks) / len(ks)}
    eng = T["engagement"]["TAG_mean_kl"] >= max(ENGAGE_MIN_KL, ENGAGE_RATIO * T["engagement"]["SHAM_mean_kl"])
    T["engaged"] = eng
    per_seed = {}
    for o in outs:
        one = [o]
        per_seed[o["seed"]] = {a: _rate(one, ledgers, a, "hoff")["p_uv"] for a in ("A0", "TAG", "SHAM")}
    T["per_seed_p_uv"] = per_seed
    h = {a: rates[a]["hoff"] for a in rates}
    bad_repro = [s for s, r in T["reproduction"].items() if r is None or r["rel_diff"] > 0.5]
    if bad_repro or h["A0"]["knows"] < 10:
        T["verdict"], T["why"] = "UNDEFINED", f"reproduction {bad_repro} or A0 knows {h['A0']['knows']} < 10"
    elif not eng:
        T["verdict"], T["why"] = "NO-ENGAGEMENT", "the learned tag moves the onset no more than the bar"
    else:
        o = T["onset_TAG-A0"]
        go = (o["ci"][0] > 0
              and h["TAG"]["p_uv"] is not None and h["A0"]["p_uv"] is not None
              and h["TAG"]["p_uv"] - h["A0"]["p_uv"] <= -0.10
              and T["onset_TAG-SHAM"]["ci"][0] > 0
              and abs(T["onset_LESION-A0"]["diff"]) < abs(o["diff"])
              and h["TAG"]["knows"] >= 0.9 * h["A0"]["knows"]
              and (h["TAG"]["refusal"] - h["A0"]["refusal"]) / max(h["A0"]["told"], 1) < 0.10
              and sum(1 for s in per_seed.values() if s["TAG"] is not None and s["A0"] is not None
                      and s["TAG"] < s["A0"]) >= 2)
        if go:
            T["verdict"], T["why"] = "GO", "screen bars met"
        elif o["ci"][0] <= 0:
            T["verdict"], T["why"] = "NEGATIVE", "engaged, onset CI includes 0 or below"
        else:
            T["verdict"], T["why"] = "MIXED", "onset moved; another bar failed"
    return T


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--night", type=int, default=NIGHT)
    r.add_argument("--smoke", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--lives", nargs="+", required=True)
    t.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex
            cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
        R = run(res, a.snaps, a.trunk, cortex, a.work, out_path=a.out, night=a.night, smoke=a.smoke)
        json.dump(R, open(a.out, "w"), indent=1)
    else:
        outs = [json.load(open(p)) for p in a.outs]
        ledgers = {}
        for p in a.lives:
            lr = json.load(open(p))
            ledgers[lr["seed"]] = lr["ledger"]
        T = table(outs, ledgers)
        print(json.dumps(T, indent=1))
        if a.out:
            json.dump(T, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
