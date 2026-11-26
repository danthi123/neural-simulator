# Capability: Self and identity
# Record: brain/research/2026-11-26-voice-ceiling-prereg.md (prereg); brain/experiments/results/2026-11-26-voice-split-read.md; brain/experiments/results/2026-11-17-teacher-night-1.5b.md (the killed teacher night)
# Expected-fail: yes for the cross-item transfer (missing prerequisite: self/other source tag); the in-item ceiling is expected to move
"""Voice ceiling: put the pretrained cortex's in-context answer (one item's telling read first) into the slept weights
by a KL on the reply positions of that item's own (oracle) probe questions, then ask the night's whole battery from a
fresh state per question, hippocampus off. DEV ceiling on K-P's KR night snapshots; no Brain change, no flip path.
Arms: A0 (no training), T_<x> (teacher read x's telling), S_<x> (sham: teacher without context), L_pixel (lesion:
pixel's cues, volvo's telling).
  python -m brain.experiments.voice_ceil run --life lifeKR_s<seed>.json --snaps <dir> --trunk <trunk> --out <json>
  python -m brain.experiments.voice_ceil table <outs...> --vs <vs_s*.json...> --lives <lifeKR...> --out <json>
"""
import argparse
import json
import os
import random
import shutil
import socket
import time

import torch
import torch.nn.functional as F

from brain import clock
from brain.experiments import cascade_screen as CS
from brain.experiments import routing_replay as RR
from brain.experiments import voice_split as VS

NIGHTS = (6, 10, 14)
NAME_ITEMS = ("pixel", "marguerite")
ARMS = (("A0", None, None), ("T_pixel", "pixel", "pixel"), ("T_marguerite", "marguerite", "marguerite"),
        ("S_pixel", "pixel", None), ("S_marguerite", "marguerite", None), ("L_pixel", "pixel", "volvo"))
STEPS, LR, KL_WEIGHT = 120, 5e-4, 3.0      # the 1.5B sleep row (brain/brain.py SLEEP_DEFAULTS)
ACK = "Thanks for telling me. I'll keep that in mind."   # Brain.TEACHER_ACK (declared scaffold, read by the teacher only)
GEN_SEED = 20261127


def cues_for(res, item, ledger):
    """The distinct probe questions about `item` over every night of the banked battery (oracle cues, flagged)."""
    out = []
    for n in sorted({p["night"] for p in res["probes"] if p["arm"] == RR.FROZEN and p.get("line") == "trunk"}):
        try:
            rows = RR.banked_rows(res, n)
        except ValueError:
            continue
        out += [r["you"] for r in rows if r.get("item") == item and CS.is_told(r, ledger)]
    return list(dict.fromkeys(out))


def context_for(ledger, item):
    return f"User: {ledger[item]['telling']}\n\nAssistant: {ACK}\n\n"


def teacher_targets(b, cues, ctx):
    """(prefix, reply, context) triples, or (prefix, reply) pairs for the sham, from the pretrained cortex (adapters off,
    birth state); greedy, 24 tokens (near_miss.guard_texts)."""
    from brain.near_miss import guard_texts
    return guard_texts(b.raw_cortex, b.base, cues, split=True, context=ctx)


def distill(b, targets, steps=STEPS, lr=LR, kl_weight=KL_WEIGHT, seed=0):
    """KL on the targets' reply positions (student reads prefix + reply, no context) plus a keep KL on the life's own
    keep-bank texts held to TONIGHT's slept cortex (so the rest of what sleep built stays). No replay CE.
    Returns per-step (guard, keep) losses."""
    plastic = b._plastic()
    pipe = b.raw_cortex.pipe
    base = [s.float() for s in b.base]
    bank = b._keep_bank()
    rng = random.Random(seed)
    generic = rng.sample(bank, min(64, len(bank)))
    gen_ids = [pipe.encode(t) for t in generic]
    t_ids, t_from, t_ref = [], [], []
    with torch.no_grad():
        plastic.enabled = True
        keep_ref = [F.log_softmax(plastic(ids, base)[0], -1) for ids in gen_ids]
        plastic.enabled = False                       # the teacher is the pretrained cortex
        for t in targets:
            pi, ri = pipe.encode(t[0]), pipe.encode(t[1])
            ids, k = pi + ri, max(len(pi) - 1, 0)
            if len(t) == 3:
                ci = pipe.encode(t[2])
                ref = F.log_softmax(plastic(ci + ids, base)[0], -1)[len(ci) + k:]
            else:
                ref = F.log_softmax(plastic(ids, base)[0], -1)[k:]
            t_ids.append(ids), t_from.append(k), t_ref.append(ref)
    plastic.enabled = True
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        m = rng.randrange(len(t_ids))
        lg, _ = plastic(t_ids[m], base)
        guard = F.kl_div(F.log_softmax(lg[t_from[m]:], -1), t_ref[m], log_target=True, reduction="batchmean")
        j = rng.randrange(len(gen_ids))
        g, _ = plastic(gen_ids[j], base)
        keep = F.kl_div(F.log_softmax(g, -1), keep_ref[j], log_target=True, reduction="batchmean")
        loss = kl_weight * guard + kl_weight * keep
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append((round(float(guard), 4), round(float(keep), 5)))
    for prm in plastic.parameters():
        prm.grad = None
    b._adapters_dirty = True
    b.cortex.refresh()
    return losses


def _adapter_norm(b):
    p = b.cortex.plastic
    return None if p is None or not hasattr(p, "B_k") or not p.B_k else round(float(sum(v.detach().float().norm() ** 2 for d in (p.B_k, p.B_v)
                                                  for v in d.values()) ** 0.5), 6)


def screen_night(res, night, snap, trunk, cortex, work, weekend, arms=ARMS, steps=STEPS, ledger=None, cues=None):
    from brain.eval import life
    ledger = ledger or res["ledger"]
    seed = res["seed"]
    qs = RR._qs(RR.banked_rows(res, night))
    cues = cues or {x: cues_for(res, x, ledger) for x in NAME_ITEMS}
    sim = life.SimClock()
    clock.set(sim.now)
    out = {"night": night, "arms": {}}
    h_off, c_off = life.ARMS["hippocampus-off"]
    try:
        for name, cue_item, ctx_item in arms:
            home = os.path.join(work, f"s{seed}_n{night}_{name}")
            t0 = time.time()
            sim.to(life.day_start(night + 1, weekend))
            life._reseed(seed, 500 + night)               # the voice split's B reseed
            b = RR.open_snapshot(snap, home, cortex, dict(res["brain_kw"]), trunk=trunk, with_deltas=True)
            rec = {"cue_item": cue_item, "ctx_item": ctx_item}
            try:
                b.hippocampus.FREEZE_USE = False
                b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
                if cue_item:
                    py_state = random.getstate()
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(GEN_SEED + night)
                        ctx = context_for(ledger, ctx_item) if ctx_item else None
                        tg = teacher_targets(b, cues[cue_item], ctx)
                        n0 = _adapter_norm(b)
                        losses = distill(b, tg, steps=steps, seed=GEN_SEED + 31 * night)
                    random.setstate(py_state)
                    rec.update({"targets": [list(t[:2]) for t in tg], "losses": losses,
                                "adapter_norm": [n0, _adapter_norm(b)]})
                rows = []
                for q in qs:
                    b.state = [x.clone() for x in b.base]
                    rows += life.run_battery(b, [q], ledger, night)
            finally:
                b.release()
                shutil.rmtree(home, ignore_errors=True)
            rec.update({"rows": [{k: r.get(k) for k in VS.KEEP} for r in rows], **VS.voice(rows, ledger),
                        "seconds": round(time.time() - t0, 1)})
            out["arms"][name] = rec
        return out
    finally:
        clock.set(None)


def run(res, snaps, trunk, cortex, work, weekend, nights=NIGHTS, arms=ARMS, steps=STEPS, log=print, out_path=None):
    ledger = res["ledger"]
    cues = {x: cues_for(res, x, ledger) for x in NAME_ITEMS}
    R = {"seed": res["seed"], "host": socket.gethostname(), "device": str(getattr(cortex, "device", "?")),
         "brain_kw": res["brain_kw"], "steps": steps, "cues": cues, "nights": []}
    for n in nights:
        rec = screen_night(res, n, os.path.join(snaps, f"n{n}"), trunk, cortex, work, weekend, arms=arms,
                           steps=steps, ledger=ledger, cues=cues)
        R["nights"].append(rec)
        log(res["seed"], n, {a: (v["uv_knows"], v["knows"], v.get("seconds")) for a, v in rec["arms"].items()})
        if out_path:
            json.dump(R, open(out_path + ".tmp", "w"), indent=1)
            os.replace(out_path + ".tmp", out_path)
    return R


# ---- scoring (prereg section 3) ----
def _cell_rows(out, arm, item, ledger):
    """{(seed, item, night): (uv_knows, knows)} on told rows of `item` for one arm."""
    c = {}
    for n in out["nights"]:
        k = u = 0
        for r in n["arms"][arm]["rows"]:
            if r.get("item") != item or not CS.is_told(r, ledger) or not VS.knows(r, ledger):
                continue
            k += 1
            u += CS.own2(r["brain"])
        c[(out["seed"], item, n["night"])] = (u, k)
    return c


def _rate(cells):
    k = sum(v[1] for v in cells.values())
    return (sum(v[0] for v in cells.values()) / k) if k else None


def _diff_ci(a, b, draws=VS.BOOT, seed=VS.BOOT_SEED):
    keys = sorted(set(a) & set(b))
    d0 = None if _rate(a) is None or _rate(b) is None else _rate(a) - _rate(b)
    rng = random.Random(seed)
    ds = []
    for _ in range(draws):
        pick = [rng.choice(keys) for _ in keys]
        ka = sum(a[x][1] for x in pick)
        kb = sum(b[x][1] for x in pick)
        if ka and kb:
            ds.append(sum(a[x][0] for x in pick) / ka - sum(b[x][0] for x in pick) / kb)
    ds.sort()
    ci = [round(ds[int(0.025 * len(ds))], 4), round(ds[int(0.975 * len(ds)) - 1], 4)] if ds else None
    return {"diff": None if d0 is None else round(d0, 4), "ci": ci, "n_draws": len(ds)}


def _merge(dicts):
    o = {}
    for d in dicts:
        o.update(d)
    return o


def _knows(out, arm, item, ledger):
    return sum(v[1] for v in _cell_rows(out, arm, item, ledger).values())


def _told_refusal(out, arm, ledger):
    t = [r for n in out["nights"] for r in n["arms"][arm]["rows"] if CS.is_told(r, ledger)]
    return (sum(CS.refusal(r["brain"]) or not r["brain"].strip() for r in t) / len(t)) if t else None


def _g1(outs, ledgers):
    g = {}
    for x in NAME_ITEMS:
        n_key = uv = other = dec = 0
        for o in outs:
            led = ledgers[o["seed"]]
            others = [it for it, v in led.items() if it != x and v.get("telling")]
            for n in o["nights"]:
                for _, rep in n["arms"][f"T_{x}"]["targets"]:
                    if VS.knows({"item": x, "brain": rep}, led):
                        n_key += 1
                        uv += CS.own2(rep.strip())
                    elif any(VS.knows({"item": it, "brain": rep}, led) for it in others):
                        other += 1
                    if CS.refusal(rep):
                        dec += 1
        g[x] = {"key_targets": n_key, "uv_of_key": uv, "uv_share": round(uv / n_key, 4) if n_key else None,
                "other_item_key": other, "declines": dec,
                "pass": bool(n_key >= 5 and uv / n_key <= 0.15) if n_key else False}
    return g


def table(outs, vss, ledgers):
    """The prereg's rule. outs: voice_ceil JSONs; vss: the voice split JSONs (A0 reproduction); ledgers: {seed: ledger}."""
    vs_by = {v["seed"]: v for v in vss}
    seeds, void = {}, []
    for o in outs:
        led, v = ledgers[o["seed"]], vs_by[o["seed"]]
        vn = {n["night"]: n for n in v["nights"]}
        good = []
        for n in o["nights"]:
            a0 = [r["brain"] for r in n["arms"]["A0"]["rows"] if CS.is_told(r, led)]
            ref = [r["brain"] for r in vn[n["night"]]["arms"]["hippocampus-off"]["B"]["rows"] if CS.is_told(r, led)]
            (good if a0 == ref else void).append((o["seed"], n["night"]))
        seeds[o["seed"]] = {"live_nights": [g[1] for g in good]}
    # drop void nights before scoring
    live = []
    for o in outs:
        keep = set(seeds[o["seed"]]["live_nights"])
        live.append({**o, "nights": [n for n in o["nights"] if n["night"] in keep]})
    L = lambda o: ledgers[o["seed"]]
    cell = lambda arm, item: _merge(_cell_rows(o, arm, item, L(o)) for o in live)
    a0 = {x: cell("A0", x) for x in NAME_ITEMS}
    other = {"pixel": "marguerite", "marguerite": "pixel"}
    in_T = _diff_ci(_merge(cell(f"T_{x}", x) for x in NAME_ITEMS), _merge(a0.values()))
    in_TS = _diff_ci(_merge(cell(f"T_{x}", x) for x in NAME_ITEMS), _merge(cell(f"S_{x}", x) for x in NAME_ITEMS))
    cr_T = _diff_ci(_merge(cell(f"T_{x}", other[x]) for x in NAME_ITEMS), _merge(a0.values()))
    cr_TS = _diff_ci(_merge(cell(f"T_{x}", other[x]) for x in NAME_ITEMS),
                     _merge(cell(f"S_{x}", other[x]) for x in NAME_ITEMS))
    dirs = {f"{x}->{other[x]}": _diff_ci(cell(f"T_{x}", other[x]), a0[other[x]]) for x in NAME_ITEMS}
    in_dirs = {x: _diff_ci(cell(f"T_{x}", x), a0[x]) for x in NAME_ITEMS}
    knows_ratio = {x: (sum(_knows(o, f"T_{x}", x, L(o)) for o in live) /
                       max(sum(_knows(o, "A0", x, L(o)) for o in live), 1)) for x in NAME_ITEMS}
    ref_rise = {x: round(sum((_told_refusal(o, f"T_{x}", L(o)) or 0) - (_told_refusal(o, "A0", L(o)) or 0)
                             for o in live) / max(len(live), 1), 4) for x in NAME_ITEMS}
    g1 = _g1(live, ledgers)
    lesion = {"diff_pixel": _diff_ci(cell("L_pixel", "pixel"), a0["pixel"]),
              "knows_pixel": [sum(_knows(o, a, "pixel", L(o)) for o in live) for a in ("A0", "L_pixel")]}
    report = {arm: {"summary": [{k: n["arms"][arm].get(k) for k in ("n_told", "knows", "uv_knows", "uv_told",
                                                                     "user_voice", "refusal")} for o in live
                                for n in o["nights"]],
                    "volvo": _diff_ci(cell(arm, "volvo"), cell("A0", "volvo")) if arm != "A0" else None}
              for arm, _, _ in ARMS}
    n_live = sum(len(s["live_nights"]) for s in seeds.values())
    if n_live == 0:
        verdict, why = "VOID", "no live night"
    elif not all(g1[x]["pass"] for x in NAME_ITEMS):
        verdict, why = "UNDEFINED", "G1: no usable teacher on a name item"
    else:
        ceil_in = (in_T["diff"] is not None and in_T["diff"] <= -0.15 and in_T["ci"] and in_T["ci"][1] < 0
                   and in_TS["diff"] is not None and in_TS["diff"] <= -0.10
                   and all(knows_ratio[x] >= 0.8 for x in NAME_ITEMS) and all(ref_rise[x] < 0.10 for x in NAME_ITEMS))
        transfer = (ceil_in and cr_T["diff"] is not None and cr_T["diff"] <= -0.10 and cr_T["ci"] and cr_T["ci"][1] < 0
                    and cr_TS["diff"] is not None and cr_TS["diff"] <= -0.05
                    and all(d["diff"] is not None and d["diff"] < 0 for d in dirs.values()))
        verdict = "TRANSFER" if transfer else ("ITEM-BOUND" if ceil_in else "NO-CEILING")
        why = ""
    return {"verdict": verdict, "why": why, "seeds": seeds, "void_nights": void, "G1": g1,
            "in_item": {"T-A0": in_T, "T-S": in_TS, "per_item": in_dirs},
            "cross_item": {"T-A0": cr_T, "T-S": cr_TS, "per_direction": dirs},
            "knows_ratio": knows_ratio, "told_refusal_rise": ref_rise, "lesion": lesion, "report": report}


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--nights", type=int, nargs="+", default=list(NIGHTS))
    r.add_argument("--arms", nargs="+", default=None, help="subset of arm names (smoke)")
    r.add_argument("--steps", type=int, default=STEPS)
    r.add_argument("--work", default=None)
    r.add_argument("--no-weekend", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--vs", nargs="+", required=True)
    t.add_argument("--lives", nargs="+", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex
            torch.set_num_threads(os.cpu_count())
            cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
        work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), f"cwork_s{res['seed']}")
        os.makedirs(work, exist_ok=True)
        arms = ARMS if not a.arms else tuple(x for x in ARMS if x[0] in a.arms)
        run(res, a.snaps, a.trunk, cortex, work, not a.no_weekend, nights=a.nights, arms=arms, steps=a.steps,
            out_path=a.out)
    else:
        outs = [json.load(open(p)) for p in a.outs]
        leds = {}
        for p in a.lives:
            r = json.load(open(p))
            leds[r["seed"]] = r["ledger"]
        o = table(outs, [json.load(open(p)) for p in a.vs], leds)
        json.dump(o, open(a.out, "w"), indent=1)
        print(json.dumps({k: o[k] for k in ("verdict", "why", "G1", "in_item", "cross_item", "knows_ratio",
                                            "told_refusal_rise") if k in o}, indent=1))


if __name__ == "__main__":
    main()
