# Capability: Lifelong learning
# Record: brain/research/2026-11-19-life-muse-mscale-prereg.md
# Expected-fail: M-use may be inert (under 20% of wrong fetches denied) or over-gating (over 40% of right fetches denied), and M-scale may leave the speech unchanged; each is a pre-registered kill before any life (prereg section 5)
"""Offline screens P0-use and P0-scale on a banked life home (prereg 2026-11-19-life-muse-mscale-prereg.md section 5).

python -m brain.experiments.life_screen use   --home ~/brain-lane/workLIFE_s311/trunk --seed 311 --model 1.5b --work <dir> --out <json>
python -m brain.experiments.life_screen scale --home ~/brain-lane/workLIFE_s311/trunk --seed 311 --model 1.5b --work <dir> --out <json> [--norm-target 105]

--home is any life home dir: `trunk` (after the last night), or `snap_trunk` (a copy of the trunk taken at the START of the
life's last lived day, so it holds the night before: with --night N that is the home after night N = the morning battery N,
pass --night = last day - 1 for it). --night (default 7) names the morning battery (life.battery(seed, night)) and the
world clock (09:00 of day night+1). --days (default 7) only sets the length of the ledger the battery is drawn from.
The home is never opened in place: life.probe copies it under --work for every arm and deletes the copy; the scale screen
also writes one scaled copy under --work. Both screens reuse brain.eval.life's battery and probe (no re-implementation).

use: arms intact (self-boosting, as banked) and intact-frozen (Hippocampus.FREEZE_USE: no fetch changes any episode;
last_fetch logged on every fetch). Reports, over the frozen arm: the share of all fetches and of wrong fetches where the
weighted winner is not the raw winner; the share with a raw margin below MARGIN_USE (0.02); the share of wrong fetches
and of right fetches the gate would deny; the top-1 share of fetched episodes, frozen against self-boosting (and against
the banked life json if --life-json is given); the verdict by the prereg thresholds.
rank (R-raw, prereg brain/research/2026-11-20-hub-rank-raw-prereg.md section 3): four intact-frozen arms on the same
home, A = S0 weighted (as banked), D = S0 raw-ranked, B = S_all weighted (slept bar), R = S_all raw-ranked (slept bar);
per arm: fires on open/greet/never-told turns, right- and wrong-episode fetches (ledger identity), top-1 share of fetched
episodes, weighted-only winners, H right, open intrusions; writes the JSON and an .md of every arm's turns side by side.
scale: two copies of the home, unscaled and B scaled by target/norm (default target 105, g about 0.28 on s311); arms intact
and hippocampus-off, which include the open turns; writes the JSON and a side-by-side .md (--out with .md) for a
whole-transcript read. The H-off cost is recorded, not a pass condition (the scaled copy gets no re-replay).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter

from brain import clock
from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.eval import life


def _setup(a):
    home = os.path.expanduser(a.home)
    work = os.path.expanduser(a.work)
    os.makedirs(work, exist_ok=True)
    _, ledger = life.build_life(a.seed, a.days, "dev")
    qs = life.battery(a.seed, a.night, ledger)
    cortex = Cortex(model_path=resolve_model(a.model), device=a.device)
    kw = {"sleep_seed": a.seed, **json.loads(a.brain_kw)}
    return home, work, ledger, qs, cortex, kw


def _probe(home, cortex, a, arms, qs, ledger, kw, work):
    sim = life.SimClock(life.day_start(a.night + 1))
    clock.set(sim.now)
    try:
        return life.probe(home, cortex, a.night, arms, qs, ledger, sim, a.seed, kw, work, check_pending=not a.allow_pending)
    finally:
        clock.set(None)


def _share(n, d):
    return None if not d else round(n / d, 3)


def _top1(rows):
    c = Counter(r["recalled"] for r in rows if r.get("recalled"))
    return (round(c.most_common(1)[0][1] / sum(c.values()), 3), c.most_common(1)[0][0][:60], len(c)) if c else (None, None, 0)


def screen_use(a):
    home, work, ledger, qs, cortex, kw = _setup(a)
    probes = _probe(home, cortex, a, ("intact", life.FROZEN), qs, ledger, kw, work)
    by = {p["arm"]: p for p in probes}
    fz, sb = by[life.FROZEN], by["intact"]
    fetches = [r for r in fz["rows"] if r.get("fetch") and r["fetch"]["idx"] is not None]
    wrong = [r for r in fetches if r.get("wrong_fetch")]
    right = [r for r in fetches if r.get("right_episode")]
    thr = 0.02
    f = lambda rs, p: _share(sum(1 for r in rs if p(r["fetch"])), len(rs))   # noqa: E731
    out = {"screen": "use", "home": a.home, "seed": a.seed, "night": a.night, "model": a.model,
           "n_fetches": len(fetches), "n_wrong": len(wrong), "n_right": len(right),
           "weighted_not_raw_winner": {"all": f(fetches, lambda x: x["weighted_only_winner"]),
                                       "wrong": f(wrong, lambda x: x["weighted_only_winner"])},
           "raw_margin_below_0.02": {"all": f(fetches, lambda x: x["margin_raw"] < thr),
                                     "wrong": f(wrong, lambda x: x["margin_raw"] < thr)},
           "gate_would_deny": {"all": f(fetches, lambda x: not x["confirmed"]), "wrong": f(wrong, lambda x: not x["confirmed"]),
                               "right": f(right, lambda x: not x["confirmed"])},
           "top1_frozen": _top1(fz["rows"]), "top1_self_boosting_rerun": _top1(sb["rows"]),
           "summary_frozen": fz["summary"], "summary_self_boosting": sb["summary"]}
    if a.life_json:
        with open(os.path.expanduser(a.life_json)) as fh:
            res = json.load(fh)
        ip = [p for p in res["probes"] if p["line"] == "trunk" and p["night"] == a.night and p["arm"] == "intact"]
        out["top1_banked_self_boosting"] = _top1(ip[0]["rows"]) if ip else None
    wd, rd = out["gate_would_deny"]["wrong"], out["gate_would_deny"]["right"]
    out["verdict"] = ("KILL M-use as inert (under 20% of wrong fetches denied)" if wd is not None and wd < 0.20 else
                      "KILL M-use as over-gating (over 40% of right fetches denied)" if rd is not None and rd > 0.40 else
                      "no kill: continue to the dev lives" if wd is not None else "no wrong fetches in this battery: read by hand")
    out["rows_frozen"] = fz["rows"]
    return out


RANK_ARMS = {"A": {}, "D": {"rank": "raw"}, "B": {"key": "S_all", "key_bar": "slept"},
             "R": {"key": "S_all", "key_bar": "slept", "rank": "raw"}}
OPEN_KINDS = ("open-story", "open-feeling", "open-advice", "open-opinion", "open-knowledge", "open-chat", "greet", "never")


def rank_read(rows) -> dict:
    fetched = [r for r in rows if r.get("recalled")]
    opn = [r for r in rows if r["kind"] in OPEN_KINDS]
    return {"open_fires": sum(1 for r in opn if r.get("recalled")), "open_n": len(opn),
            "right_episode": sum(1 for r in rows if r.get("right_episode")),
            "wrong_episode": sum(1 for r in rows if r.get("wrong_fetch")),
            "src_n": sum(1 for r in rows if "wrong_fetch" in r),
            "top1": _top1(rows),
            "weighted_only_winners": sum(1 for r in fetched if r.get("fetch") and r["fetch"]["idx"] is not None
                                         and r["fetch"]["idx"] != r["fetch"]["raw_winner"]),
            "raw_vs_weighted_differ": sum(1 for r in fetched if (r.get("fetch") or {}).get("weighted_winner") is not None
                                          and r["fetch"]["weighted_winner"] != r["fetch"]["idx"]),
            "n_fetched": len(fetched)}


def screen_rank(a):
    home, work, ledger, qs, cortex, kw = _setup(a)
    out = {"screen": "rank", "home": a.home, "seed": a.seed, "night": a.night, "model": a.model, "arms": {}}
    for name, extra in RANK_ARMS.items():
        if "key_bar" in extra:
            extra = {**extra, "key_bar": a.sall_bar}
        p = _probe(home, cortex, a, (life.FROZEN,), qs, ledger, {**kw, **extra}, work)[0]
        out["arms"][name] = {"brain_kw": extra, "read": rank_read(p["rows"]), "summary": p["summary"], "rows": p["rows"]}
    return out


def rank_md(res: dict) -> str:
    arms = list(res["arms"])
    L = [f"# R-raw screen: {res['home']} (seed {res['seed']}, morning after night {res['night']}, {res['model']})\n",
         "A = S0 weighted (banked), D = S0 raw-ranked, B = S_all weighted (slept bar), R = S_all raw-ranked. Frozen use.\n",
         "| arm | open fires | right ep | wrong ep | top-1 | H | open intrusions |", "|---|---|---|---|---|---|---|"]
    for k in arms:
        r, m = res["arms"][k]["read"], res["arms"][k]["summary"]
        L.append(f"| {k} | {r['open_fires']}/{r['open_n']} | {r['right_episode']} | {r['wrong_episode']}/{r['src_n']} | "
                 f"{r['top1'][0]} ({(r['top1'][1] or '')[:30]}) | {m['H_right']}/{m['H_n']} | {m['open_intrusions']}/{m['open_n']} |")
    L.append("")
    rows = [res["arms"][k]["rows"] for k in arms]
    for i, r0 in enumerate(rows[0]):
        L.append(f"**[{r0['kind']}] you:** {r0['you']}\n")
        for k, rs in zip(arms, rows):
            L.append(f"- {k}: {life._line({**rs[i], 'you': '', 'kind': ''}).split('**brain:**', 1)[1].strip()}")
        L.append("")
    return "\n".join(L) + "\n"


def _scaled_copy(home, work, cortex, a, kw):
    """A copy of the home with B scaled by target/norm. Returns (path, norm_before, g, norm_after)."""
    cp = os.path.join(work, "scaled_home")
    shutil.rmtree(cp, ignore_errors=True)
    shutil.copytree(home, cp)
    sim = life.SimClock(life.day_start(a.night + 1))
    clock.set(sim.now)
    try:
        b = Brain(cp, cortex=cortex, lock=True, **{**kw, "sleep_on_restart": False})
        try:
            p = b.cortex.plastic
            if p is None:
                raise SystemExit(f"{a.home} holds no adapters: nothing to scale")
            n0 = p.adapter_norm()
            g = a.norm_target / n0
            p.downscale(g)
            b._adapters_dirty = True
            b.cortex.refresh()
            b.save()
            return cp, n0, g, p.adapter_norm(), list(p.factor_norms())
        finally:
            b.release()
    finally:
        clock.set(None)


def screen_scale(a):
    home, work, ledger, qs, cortex, kw = _setup(a)
    arms = ("intact", "hippocampus-off")
    base = _probe(home, cortex, a, arms, qs, ledger, kw, work)
    cp, n0, g, n1, fn = _scaled_copy(home, work, cortex, a, kw)
    try:
        scaled = _probe(cp, cortex, a, arms, qs, ledger, kw, work)
    finally:
        shutil.rmtree(cp, ignore_errors=True)
    return {"screen": "scale", "home": a.home, "seed": a.seed, "night": a.night, "model": a.model, "norm_target": a.norm_target,
            "norm_before": n0, "g": g, "norm_after": n1, "factor_norms_after": fn,
            "unscaled": {p["arm"]: p for p in base}, "scaled": {p["arm"]: p for p in scaled}}


def scale_md(res: dict) -> str:
    L = [f"# P0-scale side by side: {res['home']} (seed {res['seed']}, morning after night {res['night']}, {res['model']})",
         f"\nB scaled by g = {res['g']:.4f} (norm {res['norm_before']:.3f} -> {res['norm_after']:.3f}, target {res['norm_target']}). "
         "Read the whole transcript: first-person looping (\"I think autumn is my favourite season.\" repeating), off-cue identity "
         "(\"My name is Meridian.\"), collapse. H-off cost is recorded, not a pass condition.\n"]
    for arm in ("intact", "hippocampus-off"):
        u, s = res["unscaled"][arm], res["scaled"][arm]
        L.append(f"\n## arm: {arm}\n")
        for k, x in (("unscaled", u), ("scaled", s)):
            m = x["summary"]
            L.append(f"- {k}: H {m['H_right']}/{m['H_n']}; direct {m['direct_right']}/{m['direct_n']}; wrong fetch {m['wrong_fetch']}/{m['src_n']}; "
                     f"never-told false {m['never_false']}/{m['never_n']}; open intrusions {m['open_intrusions']}/{m['open_n']}; "
                     f"collapse {m['collapse']}; adapter norm {x['adapter_norm']}")
        L.append("")
        for ru, rs in zip(u["rows"], s["rows"]):
            L.append(f"**[{ru['kind']}] you:** {ru['you']}\n")
            L.append(f"- unscaled: {life._line({**ru, 'you': '', 'kind': ''}).split('**brain:**', 1)[1].strip()}")
            L.append(f"- scaled: {life._line({**rs, 'you': '', 'kind': ''}).split('**brain:**', 1)[1].strip()}\n")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("screen", choices=("use", "scale", "rank"))
    ap.add_argument("--home", required=True, help="a banked life home dir (trunk or snap_trunk)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, help="result .json (the scale screen also writes the same name with .md)")
    ap.add_argument("--work", required=True, help="scratch dir for the copies (never /tmp)")
    ap.add_argument("--night", type=int, default=7, help="the morning battery after this night (default 7)")
    ap.add_argument("--days", type=int, default=7, help="ledger length for life.build_life (default 7)")
    ap.add_argument("--norm-target", type=float, default=105.0, help="scale: the adapter norm B is scaled to (default 105)")
    ap.add_argument("--life-json", default=None, help="use: the banked life_sS.json, to compare the self-boosting probe")
    ap.add_argument("--brain-kw", default="{}", help="extra Brain kwargs as JSON (the bank used the defaults)")
    ap.add_argument("--sall-bar", default="slept", help="rank: the S_all bar of arms B/R (slept = registered; default only for a 0.1B smoke)")
    ap.add_argument("--allow-pending", action="store_true", help="accept a home with an unslept day")
    a = ap.parse_args(argv)
    res = screen_use(a) if a.screen == "use" else screen_scale(a) if a.screen == "scale" else screen_rank(a)
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    if a.screen == "rank":
        with open(os.path.splitext(out)[0] + ".md", "w") as fh:
            fh.write(rank_md(res))
        print(json.dumps({k: v["read"] for k, v in res["arms"].items()}, indent=1))
    elif a.screen == "scale":
        with open(os.path.splitext(out)[0] + ".md", "w") as fh:
            fh.write(scale_md(res))
        print(f"wrote {out} and {os.path.splitext(out)[0]}.md; g {res['g']:.4f}")
    else:
        print(json.dumps({k: v for k, v in res.items() if k not in ("rows_frozen", "summary_frozen", "summary_self_boosting")}, indent=1))
    return res


if __name__ == "__main__":
    main()
