# Capability: Self and identity
# Record: brain/research/2026-11-26-voice-ceiling-prereg.md (prereg); brain/experiments/results/2026-11-26-voice-split-read.md; brain/experiments/results/2026-11-17-teacher-night-1.5b.md (the killed teacher night)
# Expected-fail: yes for the cross-item transfer (missing prerequisite: self/other source tag); the in-item ceiling is expected to move
"""Voice ceiling: put the pretrained cortex's in-context answer (one item's telling read first) into the slept weights
by a KL on the reply positions of the OTHER told items' (oracle) probe questions, then ask the night's whole battery from a
fresh state per question, hippocampus off. DEV ceiling on K-P's KR night snapshots; no Brain change, no flip path.
Amendment R1: arms A0 (no training), T (trained on every rotating told item's own question, the teacher having read
that item's telling), S (sham: the same cues, teacher without context). Test: the day-1 name items (pixel, marguerite),
never trained; volvo and admin report only.
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
TEST_ITEMS = ("pixel", "marguerite")      # primary (name items); never trained (Amendment R1)
REPORT_ITEMS = ("volvo", "admin")         # the other day-1 items: report only, never trained
HELD = ("dan", "nova") + TEST_ITEMS + REPORT_ITEMS
# arm -> (trains?, teacher reads the item's telling?)
ARMS = (("A0", False, False), ("T", True, True), ("S", True, False))
NAME_ITEMS = TEST_ITEMS
STEPS, LR, KL_WEIGHT = 120, 5e-4, 3.0      # the 1.5B sleep row (brain/brain.py SLEEP_DEFAULTS)
from brain.brain import Brain as _Brain
ACK = _Brain.TEACHER_ACK   # declared scaffold (SCAFFOLDS.md), read by the teacher only
GEN_SEED = 20261127


def train_cues(res, ledger):
    """{item: [its probe questions]} for every told item outside HELD (the rotating items), over every banked night."""
    items = sorted({r.get("item") for p in res["probes"] if p["arm"] == RR.FROZEN and p.get("line") == "trunk"
                    for r in p["rows"] if r.get("item") and r["item"] not in HELD and CS.is_told(r, ledger)})
    return {x: cues_for(res, x, ledger) for x in items}


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


def teacher_targets(b, cues, ledger, with_context):
    """Per train item: (prefix, reply, context) triples with the item's own telling read first, or (prefix, reply)
    pairs for the sham, from the pretrained cortex (adapters off, birth state); greedy, 24 tokens
    (near_miss.guard_texts). Returns (targets, items) aligned."""
    from brain.near_miss import guard_texts
    out, items = [], []
    for x, qs in cues.items():
        t = guard_texts(b.raw_cortex, b.base, qs, split=True, context=context_for(ledger, x) if with_context else None)
        out += t
        items += [x] * len(t)
    return out, items


def distill(b, targets, steps=STEPS, lr=LR, kl_weight=KL_WEIGHT, seed=0):
    """KL on the targets' reply positions (student reads prefix + reply, no context) plus a keep KL on the life's own
    keep-bank texts held to TONIGHT's slept cortex (so the rest of what sleep built stays). No replay CE.
    Returns per-step (guard, keep) losses."""
    plastic = b._plastic()
    pipe = b.raw_cortex.pipe
    base = [s.float() for s in b.base]
    from brain.brain import KEEP_BANK_NIGHT, KEEP_BANK_SEED_OFFSET
    bank = b._keep_bank()
    rng = random.Random(seed)
    # the subset tonight's sleep drew (Brain.sleep: meta["sleeps"] in the snapshot is already post-sleep)
    generic = random.Random(b.sleep_seed + b.meta.get("sleeps", 1) - 1 + KEEP_BANK_SEED_OFFSET).sample(
        bank, min(KEEP_BANK_NIGHT, len(bank)))
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
    agree0 = _agreement(plastic, base, t_ids, t_from, t_ref)
    opt = torch.optim.Adam([p for p in plastic.parameters() if p.requires_grad], lr=lr)
    losses = []
    for _ in range(steps):
        m = rng.randrange(len(t_ids))
        j = rng.randrange(len(gen_ids))
        opt.zero_grad()
        # two backward passes (the same summed gradient; one graph alive at a time: memory on 23 GB CPU nodes)
        lg, _ = plastic(t_ids[m], base)
        guard = F.kl_div(F.log_softmax(lg[t_from[m]:], -1), t_ref[m], log_target=True, reduction="batchmean")
        (kl_weight * guard).backward()
        del lg
        g, _ = plastic(gen_ids[j], base)
        keep = F.kl_div(F.log_softmax(g, -1), keep_ref[j], log_target=True, reduction="batchmean")
        (kl_weight * keep).backward()
        del g
        opt.step()
        losses.append((round(guard.item(), 4), round(keep.item(), 5)))
    for prm in plastic.parameters():
        prm.grad = None
    agree1 = _agreement(plastic, base, t_ids, t_from, t_ref)
    b._adapters_dirty = True
    b.cortex.refresh()
    return losses, (agree0, agree1)


def _agreement(plastic, base, t_ids, t_from, t_ref):
    """Teacher-forced top-1 agreement of the student with the teacher on the targets' reply positions."""
    hit = tot = 0
    with torch.no_grad():
        for ids, k, ref in zip(t_ids, t_from, t_ref):
            lg, _ = plastic(ids, base)
            hit += int((lg[k:].argmax(-1) == ref.argmax(-1)).sum())
            tot += ref.shape[0]
    return round(hit / tot, 4) if tot else None


def _adapter_norm(b):
    p = b.cortex.plastic
    return None if p is None or not hasattr(p, "B_k") or not p.B_k else round(float(sum(v.detach().float().norm() ** 2 for d in (p.B_k, p.B_v)
                                                  for v in d.values()) ** 0.5), 6)


def screen_night(res, night, snap, trunk, cortex, work, weekend, arms=ARMS, steps=STEPS, ledger=None, cues=None):
    from brain.eval import life
    ledger = ledger or res["ledger"]
    seed = res["seed"]
    qs = RR._qs(RR.banked_rows(res, night))
    cues = cues or train_cues(res, ledger)
    sim = life.SimClock()
    clock.set(sim.now)
    out = {"night": night, "arms": {}}
    h_off, c_off = life.ARMS["hippocampus-off"]
    try:
        for name, trains, with_ctx in arms:
            home = os.path.join(work, f"s{seed}_n{night}_{name}")
            t0 = time.time()
            sim.to(life.day_start(night + 1, weekend))
            life._reseed(seed, 500 + night)               # the voice split's B reseed
            b = RR.open_snapshot(snap, home, cortex, dict(res["brain_kw"]), trunk=trunk, with_deltas=True)
            rec = {"trains": trains, "context": with_ctx}
            try:
                b.hippocampus.FREEZE_USE = False
                b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
                if trains:
                    py_state = random.getstate()
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(GEN_SEED + night)
                        tg, titems = teacher_targets(b, cues, ledger, with_ctx)
                        n0 = _adapter_norm(b)
                        losses, agree = distill(b, tg, steps=steps, seed=GEN_SEED + 31 * night)
                    random.setstate(py_state)
                    rec.update({"targets": [[i, t[0], t[1]] for i, t in zip(titems, tg)], "losses": losses,
                                "agreement": list(agree), "adapter_norm": [n0, _adapter_norm(b)]})
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
    cues = train_cues(res, ledger)
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


def _told_refusal(out, arm, ledger, items):
    """Refusal-or-empty share of the told rows of `items` (B5: never diluted by the other told items)."""
    t = [r for n in out["nights"] for r in n["arms"][arm]["rows"] if r.get("item") in items and CS.is_told(r, ledger)]
    return (sum(CS.refusal(r["brain"]) or not r["brain"].strip() for r in t) / len(t)) if t else None


def engaged(rec):
    """Pre-registered engagement (Amendment R1, B7): mean guard KL of the last 10 steps <= 0.3 x the first 10, and
    teacher-forced top-1 agreement after training >= 0.9."""
    L = [g for g, _ in rec.get("losses") or []]
    ag = rec.get("agreement") or [None, None]
    if len(L) < 20 or ag[1] is None:
        return False
    return sum(L[-10:]) / 10 <= 0.3 * sum(L[:10]) / 10 and ag[1] >= 0.9


def _g1(outs, ledgers):
    """Teacher quality on the T targets, deduplicated on (item, reply) (B2)."""
    seen = {}
    for o in outs:
        led = ledgers[o["seed"]]
        for n in o["nights"]:
            for x, _, rep in n["arms"]["T"]["targets"]:
                seen.setdefault((o["seed"], x, rep.strip()), led)
    distinct = {}
    for (sd, x, rep), led in seen.items():
        distinct.setdefault((x, rep), led)
    key = [(x, rep) for (x, rep), led in distinct.items() if VS.knows({"item": x, "brain": rep}, led)]
    uv = sum(CS.own2(rep) for _, rep in key)
    dec = sum(bool(CS.refusal(rep)) for _, rep in distinct)
    share = round(uv / len(key), 4) if key else None
    return {"distinct_targets": len(distinct), "key_targets": len(key), "uv_of_key": uv, "uv_share": share,
            "declines": dec, "pass": bool(len(key) >= 8 and share is not None and share <= 0.15)}


def _rate_rows(out, arm, items, ledger):
    k = u = 0
    for n in out["nights"]:
        for r in n["arms"][arm]["rows"]:
            if r.get("item") in items and CS.is_told(r, ledger) and VS.knows(r, ledger):
                k += 1
                u += CS.own2(r["brain"])
    return u, k


def table(outs, vss, ledgers):
    """Amendment R1's rule. outs: voice_ceil JSONs; vss: the voice split JSONs (A0 reproduction); ledgers: {seed: ledger}."""
    vs_by = {v["seed"]: v for v in vss}
    seeds, void, unengaged = {}, [], []
    live = []
    for o in outs:
        led, v = ledgers[o["seed"]], vs_by[o["seed"]]
        vn = {n["night"]: n for n in v["nights"]}
        keep = []
        for n in o["nights"]:
            a0 = [r["brain"] for r in n["arms"]["A0"]["rows"] if CS.is_told(r, led)]
            ref = [r["brain"] for r in vn[n["night"]]["arms"]["hippocampus-off"]["B"]["rows"] if CS.is_told(r, led)]
            if a0 != ref:
                void.append((o["seed"], n["night"]))
                continue
            if not all(engaged(n["arms"][a]) for a in ("T", "S")):
                unengaged.append((o["seed"], n["night"], {a: engaged(n["arms"][a]) for a in ("T", "S")}))
                continue
            keep.append(n)
        seeds[o["seed"]] = {"live_nights": [n["night"] for n in keep]}
        live.append({**o, "nights": keep})
    L = lambda o: ledgers[o["seed"]]
    cell = lambda arm, items: _merge(_cell_rows(o, arm, x, L(o)) for o in live for x in items)
    T_A0 = _diff_ci(cell("T", TEST_ITEMS), cell("A0", TEST_ITEMS))
    T_S = _diff_ci(cell("T", TEST_ITEMS), cell("S", TEST_ITEMS))
    per_seed = {}
    for o in live:
        if not o["nights"]:
            continue
        r = {a: _rate_rows(o, a, TEST_ITEMS, L(o)) for a in ("A0", "T", "S")}
        per_seed[o["seed"]] = {a: list(v) for a, v in r.items()}
    lower = sum(1 for v in per_seed.values() if v["T"][1] and v["A0"][1] and v["T"][0] / v["T"][1] < v["A0"][0] / v["A0"][1])
    kn = {a: sum(_knows(o, a, x, L(o)) for o in live for x in TEST_ITEMS) for a in ("A0", "T", "S")}
    kr = {a: (kn[a] / kn["A0"] if kn["A0"] else None) for a in ("T", "S")}
    ref = {a: [(_told_refusal(o, a, L(o), TEST_ITEMS) or 0) for o in live if o["nights"]] for a in ("A0", "T", "S")}
    rise = {a: (round(sum(ref[a]) / len(ref[a]) - sum(ref["A0"]) / len(ref["A0"]), 4) if ref["A0"] else None)
            for a in ("T", "S")}
    g1 = _g1(live, ledgers)
    report = {"per_item": {x: {a: _rate(cell(a, (x,))) for a in ("A0", "T", "S")} for x in TEST_ITEMS + REPORT_ITEMS},
              "report_items_T-A0": {x: _diff_ci(cell("T", (x,)), cell("A0", (x,))) for x in REPORT_ITEMS},
              "summary": {a: [{k: n["arms"][a].get(k) for k in ("n_told", "knows", "uv_knows", "uv_told", "user_voice",
                                                               "refusal", "never_false", "self_name")}
                              for o in live for n in o["nights"]] for a in ("A0", "T", "S")},
              "agreement": {a: [n["arms"][a].get("agreement") for o in live for n in o["nights"]] for a in ("T", "S")}}
    n_live = sum(len(o["nights"]) for o in live)
    s_valid = kr["S"] is not None and kr["S"] >= 0.5
    why = ""
    if n_live == 0:
        verdict, why = "VOID", "no live, engaged night"
    elif len(per_seed) < 3:
        verdict, why = "UNDEFINED", "a seed has no live, engaged night"
    elif not g1["pass"]:
        verdict, why = "UNDEFINED", "G1: no usable teacher"
    elif kn["A0"] < 20 or kn["T"] < 8:
        verdict, why = "UNDEFINED", "too few knows rows on the test items"
    elif kr["T"] < 0.8 or (rise["T"] is not None and rise["T"] >= 0.10):
        verdict = "ABSTAINS"
    else:
        moved = T_A0["diff"] is not None and T_A0["diff"] <= -0.10 and lower == len(per_seed)
        frame = (not s_valid) or (T_S["diff"] is not None and T_S["diff"] <= -0.05)
        verdict = "TRANSFER" if moved and frame else "NO-TRANSFER"
        if moved and not s_valid:
            why = "sham invalid (S knows < 0.5 x A0): T-S not applied"
    return {"verdict": verdict, "why": why, "seeds": seeds, "void_nights": void, "unengaged": unengaged, "G1": g1,
            "T-A0": T_A0, "T-S": T_S, "per_seed": per_seed, "seeds_T_lower": lower, "knows": kn, "knows_ratio": kr,
            "sham_valid": s_valid, "told_refusal_rise": rise, "report": report}


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
        print(json.dumps({k: o[k] for k in ("verdict", "why", "G1", "T-A0", "T-S", "per_seed", "knows_ratio",
                                            "told_refusal_rise", "unengaged") if k in o}, indent=1))


if __name__ == "__main__":
    main()
