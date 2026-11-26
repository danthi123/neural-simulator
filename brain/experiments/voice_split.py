# Capability: Self and identity
# Record: brain/research/2026-11-26-voice-split-prereg.md (prereg); brain/experiments/results/2026-11-25-cascade-screen-read.md; brain/research/2026-11-24-self-other-tag-prereg-DRAFT.md (BLOCKED)
# Expected-fail: no (a diagnostic, no method); UNDEFINED if the intact/hippocampus-off fresh-state rows do not reproduce the cascade screen's
"""Voice split: is the user's voice in told answers learned by sleep, or the pretrained cortex's own habit given the fact?
K-P's per-night homes (KR night snapshots) asked the banked battery in four lesion arms, each question from a fresh
state (cond B, as the cascade screen's B), plus the running battery (cond A) for consolidated-off and both-off.
Read: on told rows, P(user voice | the reply names the told fact) under slept synapses (hippocampus-off, intact) vs
pretrained synapses with the fetch (consolidated-off). Host-only scoring; no Brain change.
  python -m brain.experiments.voice_split run --life lifeKR_s<seed>.json --snaps <dir> --trunk <trunk> --out <json>
  python -m brain.experiments.voice_split table <outs...> --casc <casc_s*.json...> --lives <lifeKR...> --out <json>
"""
import argparse
import json
import os
import re
import shutil
import socket
import time

from brain import clock
from brain.experiments import cascade_screen as CS
from brain.experiments import routing_replay as RR

NIGHTS = CS.NIGHTS
PLAN = (("intact", "B"), ("hippocampus-off", "B"), ("consolidated-off", "B"), ("both-off", "B"),
        ("both-off", "ctx"), ("consolidated-off", "A"), ("both-off", "A"))
# ctx = fresh state, the item's telling heard first (the brain answers it), then the question: the pretrained
# cortex's own voice habit with the fact in context (review fix 4). Rows without a telling run as B.
KEEP = ("kind", "item", "you", "brain", "recalled", "ok", "fetched", "wrong_fetch", "right_episode", "told_first")
HEDGE = re.compile(r"\b(i think|i can |i'd|i'm glad|i hope|i remember|i'll)", re.I)
BOOT, BOOT_SEED = 2000, 20261126


def knows(row, ledger):
    """The reply names the told fact: any ledger key of the row's item as a whole word (case-insensitive)."""
    it = ledger.get(row.get("item") or "") or {}
    t = row["brain"].lower()
    return any(re.search(r"\b" + re.escape(k.lower()) + r"\b", t) for k in it.get("keys") or ())


def voice(rows, ledger):
    t = [r for r in rows if CS.is_told(r, ledger)]
    kn = [r for r in t if knows(r, ledger)]
    return {**CS.summarise(rows, ledger), "knows": len(kn), "uv_knows": sum(CS.own2(r["brain"]) for r in kn),
            "uv_told": sum(CS.own2(r["brain"]) for r in t), "hedge_told": sum(bool(HEDGE.search(r["brain"])) for r in t),
            "right_episode": sum(bool(r.get("right_episode")) for r in t),
            "self_name": [r["brain"] for r in rows if r.get("item") == "nova"]}


def screen_night(res, night, snap, trunk, cortex, work, weekend, plan=PLAN):
    from brain.eval import life
    ledger, seed = res["ledger"], res["seed"]
    qs = RR._qs(RR.banked_rows(res, night))
    sim = life.SimClock()
    clock.set(sim.now)
    out = {"night": night, "arms": {}}
    try:
        for arm, cond in plan:
            h_off, c_off = life.ARMS[arm]
            home = os.path.join(work, f"s{seed}_n{night}_{arm}_{cond}")
            t0 = time.time()
            sim.to(life.day_start(night + 1, weekend))
            life._reseed(seed, 500 + night)               # the cascade screen's A/B reseed
            b = RR.open_snapshot(snap, home, cortex, dict(res["brain_kw"]), trunk=trunk, with_deltas=True)
            try:
                b.hippocampus.FREEZE_USE = False
                b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
                if cond in ("B", "ctx"):
                    rows = []
                    for q in qs:
                        b.state = [x.clone() for x in b.base]
                        tell = (ledger.get(q.get("item") or "") or {}).get("telling") if cond == "ctx" else None
                        if tell:
                            b.hear(tell)
                        r = life.run_battery(b, [q], ledger, night)
                        r[0]["told_first"] = bool(tell)
                        rows += r
                else:
                    rows = life.run_battery(b, qs, ledger, night)
            finally:
                b.release()
                shutil.rmtree(home, ignore_errors=True)
            out["arms"].setdefault(arm, {})[cond] = {
                "rows": [{k: r.get(k) for k in KEEP} for r in rows],
                **voice(rows, ledger), "seconds": round(time.time() - t0, 1)}
        return out
    finally:
        clock.set(None)


def run(res, snaps, trunk, cortex, work, weekend, nights=NIGHTS, log=print, out_path=None):
    R = {"seed": res["seed"], "host": socket.gethostname(), "device": str(getattr(cortex, "device", "?")),
         "brain_kw": res["brain_kw"], "nights": []}
    for n in nights:
        rec = screen_night(res, n, os.path.join(snaps, f"n{n}"), trunk, cortex, work, weekend)
        R["nights"].append(rec)
        log(res["seed"], n, {f"{a}/{c}": (v["uv_knows"], v["knows"]) for a, cs in rec["arms"].items()
                             for c, v in cs.items()})
        if out_path:
            json.dump(R, open(out_path + ".tmp", "w"), indent=1)
            os.replace(out_path + ".tmp", out_path)
    return R


def reproduction(out, casc):
    """Rows of intact/hippocampus-off cond B identical to the cascade screen's B rows on the same nights."""
    cn = {n["night"]: n for n in casc["nights"]}
    same = tot = 0
    for n in out["nights"]:
        c = cn.get(n["night"])
        if c is None:
            continue
        for arm in ("intact", "hippocampus-off"):
            for x, y in zip(n["arms"][arm]["B"]["rows"], c["arms"][arm]["B"]["rows"]):
                tot += 1
                same += x["brain"] == y["brain"]
    return same, tot


def _items(o, arm, cond, ledger):
    """Per item: (uv_knows, knows, uv_told, n_told) over the seed's nights, fresh-state rows of one arm."""
    d = {}
    for n in o["nights"]:
        for r in n["arms"][arm][cond]["rows"]:
            if not CS.is_told(r, ledger):
                continue
            x = d.setdefault(r["item"], [0, 0, 0, 0])
            k, u = knows(r, ledger), CS.own2(r["brain"])
            x[0] += k and u
            x[1] += k
            x[2] += u
            x[3] += 1
    return d


def _diff(cells, a, b, f):
    """Pooled rate difference a - b; cells: list of {arm: [uv_knows, knows, uv_told, n_told]} per (seed, item)."""
    def rate(arm):
        num = sum(c[arm][0 if f == "knows" else 2] for c in cells if arm in c)
        den = sum(c[arm][1 if f == "knows" else 3] for c in cells if arm in c)
        return num / den if den else None
    ra, rb = rate(a), rate(b)
    return None if ra is None or rb is None else ra - rb


def _boot(cells, a, b, f):
    """Item-clustered bootstrap (resample (seed, item) cells), 95% percentile CI of the pooled difference."""
    import random
    rng = random.Random(BOOT_SEED)
    ds = []
    for _ in range(BOOT):
        x = _diff([cells[rng.randrange(len(cells))] for _ in cells], a, b, f)
        if x is not None:
            ds.append(x)
    ds.sort()
    return [round(ds[int(0.025 * len(ds))], 4), round(ds[int(0.975 * len(ds)) - 1], 4)] if ds else None


ARMS_B = {"hoff": ("hippocampus-off", "B"), "coff": ("consolidated-off", "B"), "ctx": ("both-off", "ctx"),
          "intact": ("intact", "B"), "both": ("both-off", "B")}


def table(outs, cascs, ledgers):
    """Prereg section 3 rule (amended before data). ledgers: {seed: ledger}."""
    cs = {c["seed"]: c for c in cascs}
    seeds, cells = {}, []
    for o in outs:
        s, led = o["seed"], ledgers[o["seed"]]
        c = cs.get(s)
        if c is not None and c.get("brain_kw") != o.get("brain_kw"):
            raise ValueError(f"seed {s}: brain_kw differs from the cascade screen's")
        same, tot = reproduction(o, c) if c is not None else (0, 0)
        per = {k: _items(o, a, cond, led) for k, (a, cond) in ARMS_B.items()}
        tot_ = {k: [sum(v[i] for v in d.values()) for i in range(4)] for k, d in per.items()}
        rec = {"repro": [same, tot], "void": not tot or same < 0.9 * tot, "totals": tot_,
               "right_episode": {k: sum(n["arms"][a][cond]["right_episode"] for n in o["nights"])
                                 for k, (a, cond) in ARMS_B.items()},
               "hedge_told": {k: sum(n["arms"][a][cond]["hedge_told"] for n in o["nights"])
                              for k, (a, cond) in ARMS_B.items()}}
        rate = lambda k, f=0: tot_[k][f] / tot_[k][f + 1] if tot_[k][f + 1] else None
        rec["rate_knows"] = {k: rate(k) for k in ARMS_B}
        rec["rate_told"] = {k: rate(k, 2) for k in ARMS_B}
        seeds[s] = rec
        if not rec["void"]:
            for item in set().union(*[set(d) for d in per.values()]):
                cells.append({k: per[k][item] for k in per if item in per[k]})
    live = [s for s in seeds if not seeds[s]["void"]]
    out = {"seeds": seeds, "live_seeds": live, "n_cells": len(cells)}
    comp = {}
    for a, b in (("hoff", "ctx"), ("hoff", "coff"), ("coff", "hoff"), ("intact", "coff"), ("hoff", "both")):
        for f in ("knows", "told"):
            comp[f"{a}-{b}/{f}"] = {"diff": None if _diff(cells, a, b, f) is None else round(_diff(cells, a, b, f), 4),
                                    "ci": _boot(cells, a, b, f)}
    out["comp"] = comp
    low = [s for s in live if any(seeds[s]["totals"][k][1] < 20 for k in ("hoff", "ctx"))]
    if len(live) < 2 or low:
        out["verdict"], out["why"] = "UNDEFINED", f"live {live}, knows<20 on {low}"
        return out
    pri, ci = comp["hoff-ctx/knows"], comp["hoff-ctx/knows"]["ci"]
    per_seed = [seeds[s]["rate_knows"]["hoff"] > seeds[s]["rate_knows"]["ctx"] for s in live]
    fp = comp["coff-hoff/knows"]["ci"]
    if ci and ci[0] > 0 and sum(per_seed) >= 2:
        out["verdict"] = "SLEEP-LEARNED"
    elif ci and -0.10 <= ci[0] and ci[1] <= 0.10:
        out["verdict"] = "NOT-SLEEP"
    elif fp and fp[0] > 0:
        out["verdict"] = "FETCH-PRIMED"
    else:
        out["verdict"] = "UNDEFINED"
    out["primary"] = pri
    return out


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--nights", type=int, nargs="+", default=list(NIGHTS))
    r.add_argument("--work", default=None)
    r.add_argument("--no-weekend", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--casc", nargs="+", required=True)
    t.add_argument("--lives", nargs="+", required=True, help="the lifeKR JSONs (ledgers)")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = json.load(open(a.life))
        if cortex is None:
            import torch
            from brain.cortex import Cortex
            torch.set_num_threads(os.cpu_count())
            cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
        work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), f"vwork_s{res['seed']}")
        os.makedirs(work, exist_ok=True)
        run(res, a.snaps, a.trunk, cortex, work, not a.no_weekend, nights=a.nights, out_path=a.out)
    else:
        outs = [json.load(open(p)) for p in a.outs]
        leds = {}
        for p in a.lives:
            r = json.load(open(p))
            leds[r["seed"]] = r["ledger"]
        o = table(outs, [json.load(open(p)) for p in a.casc], leds)
        json.dump(o, open(a.out, "w"), indent=1)
        print(json.dumps({k: o[k] for k in ("verdict", "why", "comp", "live_seeds", "n_cells") if k in o}, indent=1))


if __name__ == "__main__":
    main()
