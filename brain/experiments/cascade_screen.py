# Capability: Self and identity
# Record: brain/research/2026-11-25-cascade-screen-prereg.md (prereg); brain/research/2026-11-24-routing-cascade-research.md
# Expected-fail: partly (K5 echo guard: the just-spoken reply stays at full weight; weight-borne flip nights not reached)
"""Cascade screen: K-P's per-night homes (KR night snapshots) asked the banked battery in four conditions per lesion arm:
A as is, B each question from a fresh state (diagnostic), D50 / D25 = Brain(own_decay=0.5 / 0.25). Replies scored for
off-voice told rows (the user's voice or a refusal; probe3.py's own2 / REF verbatim).
  python -m brain.experiments.cascade_screen run --life lifeKR_s<seed>.json --snaps <dir> --trunk <trunk> --out <json>
      [--nights 2 ... 15]
  python -m brain.experiments.cascade_screen table <outs...> --out <verdict.json>
"""
import argparse
import json
import os
import re
import shutil
import socket
import time

from brain import clock
from brain.experiments import routing_replay as RR

NIGHTS = tuple(range(2, 16))
CONDS = {"A": None, "A2": None, "B": None, "D50": 0.5, "D25": 0.25}
RESEED = {"A2": 900}      # A2 = A with another sampling reseed: the regression-to-the-mean control (review fix 1)
LESIONS = ("intact", "hippocampus-off")

# verbatim from results/2026-11-24-flip-night-diagnosis/cascade/probe3.py (own2, REF)
REF = re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai", re.I)
REF2 = re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai|i think|i can |i'd|i'm glad|i hope|i remember|i'll", re.I)


def own2(t):
    t = t.strip()
    return bool(re.search(r"\b(my|mine)\b", t, re.I) or re.match(r"I(?:'m| am| | \w+)", t)) and not REF2.search(t) \
        and "my friend" not in t.lower()


def refusal(t):
    return bool(REF.search(t))


def user_q(t):
    t = t.strip()
    return t.endswith("?") or bool(re.match(r"(What|Do you|Can you)\b", t))


def is_told(row, ledger):
    if row["kind"] in ("H", "rotating"):
        return bool((ledger.get(row.get("item") or "") or {}).get("telling"))
    return row["kind"] == "neardup" and "wrong_fetch" in row


def banked_arm_rows(res, night, arm):
    p = [x for x in res["probes"] if x["arm"] == arm and x.get("line") == "trunk" and not x.get("tag")
         and x["night"] == night]
    if len(p) != 1:
        raise ValueError(f"night {night}: {len(p)} trunk {arm} probes")
    return p[0]["rows"]


def summarise(rows, ledger):
    t = [r for r in rows if is_told(r, ledger)]
    return {"n_told": len(t), "V": sum(own2(r["brain"]) or refusal(r["brain"]) for r in t),
            "user_voice": sum(own2(r["brain"]) for r in t), "refusal": sum(refusal(r["brain"]) for r in t),
            "ok": sum(bool(r.get("ok")) for r in t), "user_q": sum(user_q(r["brain"]) for r in t),
            "intrusion": sum(bool(r.get("intrusion")) for r in rows if r["kind"].startswith("open-")),
            "never_false": sum(bool(r.get("false")) for r in rows if r["kind"] == "never")}


def screen_night(res, night, snap, trunk, cortex, work, weekend):
    from brain.eval import life
    ledger, seed = res["ledger"], res["seed"]
    qs = RR._qs(RR.banked_rows(res, night))
    sim = life.SimClock()
    clock.set(sim.now)
    out = {"night": night, "arms": {}}
    try:
        for arm in LESIONS:
            h_off, c_off = life.ARMS[arm]
            bank = banked_arm_rows(res, night, arm)
            if [r["you"] for r in bank] != [q["text"] for q in qs]:
                raise ValueError(f"night {night} {arm}: banked battery order differs from the frozen probe's")
            rec = {"banked": summarise(bank, ledger)}
            for cond, beta in CONDS.items():
                home = os.path.join(work, f"s{seed}_n{night}_{arm}_{cond}")
                t0 = time.time()
                sim.to(life.day_start(night + 1, weekend))
                life._reseed(seed, RESEED.get(cond, 500) + night)
                kw = dict(res["brain_kw"], **({"own_decay": beta} if beta is not None else {}))
                b = RR.open_snapshot(snap, home, cortex, kw, trunk=trunk, with_deltas=True)
                try:
                    b.hippocampus.FREEZE_USE = False          # the banked intact arm ran with the use boost
                    b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
                    if cond == "B":
                        rows = []
                        for q in qs:
                            b.state = [x.clone() for x in b.base]
                            rows += life.run_battery(b, [q], ledger, night)
                    else:
                        rows = life.run_battery(b, qs, ledger, night)
                finally:
                    b.release()
                    shutil.rmtree(home, ignore_errors=True)
                c = {"rows": [{k: r.get(k) for k in ("kind", "item", "you", "brain", "recalled", "ok", "intrusion",
                                                      "false")} for r in rows], **summarise(rows, ledger),
                     "seconds": round(time.time() - t0, 1)}
                if cond == "A":
                    c["match_banked"] = sum(x["brain"] == y["brain"] for x, y in zip(rows, bank))
                    c["fetch_mismatch"] = [i for i, (x, y) in enumerate(zip(rows, bank))
                                           if x.get("recalled") != y.get("recalled")]
                rec[cond] = c
            for cond in CONDS:      # fetch drift against A (a state-dependent gate would show here)
                rec[cond]["fetch_vs_A"] = sum(x.get("recalled") != y.get("recalled")
                                              for x, y in zip(rec[cond]["rows"], rec["A"]["rows"]))
            rec["void"] = arm == "intact" and bool(rec["A"]["fetch_mismatch"])
            out["arms"][arm] = rec
        return out
    finally:
        clock.set(None)


def run(res, snaps, trunk, cortex, work, weekend, nights=NIGHTS, log=print, out_path=None):
    R = {"seed": res["seed"], "host": socket.gethostname(), "device": str(getattr(cortex, "device", "?")),
         "brain_kw": res["brain_kw"], "nights": []}
    for n in nights:
        rec = screen_night(res, n, os.path.join(snaps, f"n{n}"), trunk, cortex, work, weekend)
        R["nights"].append(rec)
        for arm, a in rec["arms"].items():
            log(res["seed"], n, arm, "void" if a["void"] else "", {c: (a[c]["V"], a[c]["ok"]) for c in CONDS})
        if out_path:
            json.dump(R, open(out_path + ".tmp", "w"), indent=1)
            os.replace(out_path + ".tmp", out_path)
    return R


def diagnose(a):
    if a["A"]["V"] < 4:
        return None
    return "CASCADE" if a["B"]["V"] <= 1 else "WEIGHTS" if a["B"]["V"] >= 4 else "BETWEEN"


def table(outs, k5=None):
    """The prereg's bars per beta (amended before data). outs: per-seed run() results. k5: k5_from_probe() output or
    None (UNDEFINED). K1-K4 ("core") and K5 are reported separately."""
    seeds = {}
    for o in outs:
        s = o["seed"]
        nv = sum(1 for n in o["nights"] if n["arms"]["intact"]["void"])
        seeds[s] = {"void_nights": nv, "void": nv > 3, "flip": [], "nonflip": []}
        if seeds[s]["void"]:
            continue
        for n in o["nights"]:
            for arm, a in n["arms"].items():
                if a["void"]:
                    continue
                key = "flip" if a["A"]["V"] >= 4 else "nonflip"
                seeds[s][key].append({"night": n["night"], "arm": arm, "diag": diagnose(a),
                                      **{c: {k: a[c][k] for k in ("V", "ok", "user_q", "user_voice", "refusal")}
                                         for c in CONDS}})
    out = {"seeds": seeds, "per_beta": {}}
    live = [s for s in seeds if not seeds[s]["void"]]
    fl = [x for s in live for x in seeds[s]["flip"]]
    tot = lambda xs, c, k="V": sum(x[c][k] for x in xs)
    for cond in ("D50", "D25"):
        if not live or not fl:
            out["per_beta"][cond] = {"status": "VOID" if not live else "UNDEFINED", "core": None, "k5": None,
                                     "kills": [], "flip_arm_nights": len(fl)}
            continue
        kills = []
        vA, vA2, vD = tot(fl, "A"), tot(fl, "A2"), tot(fl, cond)
        if vD > 0.5 * vA or vD > 0.5 * vA2:
            kills.append("K1")
        for s in live:
            f = seeds[s]["flip"]
            if len(f) >= 2 and not (tot(f, cond) < tot(f, "A") and tot(f, cond) < tot(f, "A2")):
                kills.append(f"K1-seed{s}")
            allr = [x for x in seeds[s]["flip"] + seeds[s]["nonflip"] if x["arm"] == "intact"]
            if tot(allr, cond, "ok") <= tot(allr, "A", "ok") - 2:
                kills.append(f"K2-seed{s}")
            nf = seeds[s]["nonflip"]
            if tot(nf, cond) > tot(nf, "A2") + 2:
                kills.append(f"K3-seed{s}")
            both = seeds[s]["flip"] + seeds[s]["nonflip"]
            if tot(both, cond, "user_q") > tot(both, "A", "user_q") + 2:
                kills.append(f"K4-seed{s}")
        core = "KILLED" if kills else "PASS"
        k5r = None if k5 is None else k5.get(cond)
        k5s = None if k5r is None else ("FAIL" if k5r["fail"] else "PASS")
        if k5s == "FAIL":
            kills.append("K5")
        status = "KILLED" if kills else ("HOLD" if k5s is None else "LEAD")
        out["per_beta"][cond] = {"status": status, "core": core, "k5": k5s, "kills": kills, "flip_arm_nights": len(fl),
                                 "V_A": vA, "V_A2": vA2, "V_D": vD,
                                 "diag": {d: sum(x["diag"] == d for x in fl) for d in ("CASCADE", "WEIGHTS", "BETWEEN")}}
    return out


def k5_from_probe(e_json: dict, b_json: dict) -> dict:
    """K5 from own_gain_probe outputs (E: the echo-onset home, B: the stored-line home), runs labelled d0.5 / d0.25.
    Per sample: echo count and free replies ending "?" (the 11-16 probe's count), on n_free free turns. Fail if E echoes
    on >= 10/14 of free turns in any sample, or >= 2 samples (E and B together) have >= 4/14 free replies ending "?"."""
    out = {}
    for lab, cond in (("d0.5", "D50"), ("d0.25", "D25")):
        er = [r for r in e_json["runs"] if r["gain"] == lab]
        br = [r for r in b_json["runs"] if r["gain"] == lab]
        if not er or not br:
            continue
        q = lambda r: sum(t["brain"].strip().endswith("?") for t in r["turns"] if not t["forced"])
        E = [r["echo"] for r in er]
        Q = [q(r) for r in er + br]
        nf = [r["n_free"] for r in er + br]
        echo_fail = any(e * 14 >= 10 * r["n_free"] for e, r in zip(E, er))
        loop_fail = sum(x * 14 >= 4 * n for x, n in zip(Q, nf)) >= 2
        out[cond] = {"E": E, "E_q": [q(r) for r in er], "B_q": [q(r) for r in br], "n_free": nf,
                     "B_onset_repeat": [r["onset_repeat"] for r in br], "fail": bool(echo_fail or loop_fail)}
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
    r.add_argument("--no-weekend", action="store_true", help="the life ran without --weekend (K-P/KR ran with it)")
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--k5", default=None, help="k5 JSON from the k5 subcommand")
    k = sub.add_parser("k5")
    k.add_argument("--e", required=True, help="own_gain_probe output on the echo-onset home (E)")
    k.add_argument("--b", required=True, help="own_gain_probe output on the stored-line home (B)")
    k.add_argument("--out", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        res = json.load(open(a.life))
        if cortex is None:
            import torch
            from brain.cortex import Cortex
            torch.set_num_threads(os.cpu_count())
            cortex = Cortex(model_path=os.path.join(os.environ["BRAIN_MODEL_DIR"], res["model"] + ".pth"), device="cpu")
        work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), f"cwork_s{res['seed']}")
        os.makedirs(work, exist_ok=True)
        run(res, a.snaps, a.trunk, cortex, work, weekend=not a.no_weekend, nights=a.nights,
            out_path=a.out)
    elif a.cmd == "k5":
        v = k5_from_probe(json.load(open(a.e)), json.load(open(a.b)))
        json.dump(v, open(a.out, "w"), indent=1)
        print(json.dumps({c: {k_: x[k_] for k_ in ("E", "E_q", "B_q", "fail")} for c, x in v.items()}, indent=1))
    else:
        v = table([json.load(open(p)) for p in a.outs], json.load(open(a.k5)) if a.k5 else None)
        json.dump(v, open(a.out, "w"), indent=1)
        print(json.dumps(v["per_beta"], indent=1))


if __name__ == "__main__":
    main()
