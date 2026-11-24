# Capability: Episodic recall
# Record: brain/research/2026-11-24-routing-cascade-research.md (section 5); brain/research/2026-11-24-routing-screen-prereg.md;
#   results/2026-11-20-rank-raw-screen-1.5b.md (arm D); results/2026-11-23-m3-margin-screen.md; _scratch/route/rawwin.py
# Expected-fail: partly. The S0 key is a lexical layer-0 stand-in and near-duplicate separation is MISSING
#   (PREREQUISITES.md), so a ranking change can remove strength bias but not separate true near-duplicates; D+G may lose
#   the A-right fetches that are weighted-only winners (20/45 in the banked frozen arm). Fetch level only, never a reply claim.
"""Offline D1 routing screen on per-night snapshots of a K-P(-rerun) life. Host-side harness only; hippocampus.py is not
edited and nothing here is a brain default.

For every trunk night (default 5-15) a Brain is opened on a temporary home built from the night snapshot
(life_kw --night-snapshots: meta.json, cortex_adapters.pt, keep_bank.json, episodes.json, keys.pt), so its store is the
snapshot's episodes in episodes.json order with keys from keys.pt and strength/uses/last_used from episodes.json. The
clock is set as life.probe sets it for the morning battery (sim.to(day_start(night + 1, weekend))). Every battery cue of
the banked intact-frozen probe is read once (Hippocampus.similarities, the cue read from base), and four decisions are
made from the same similarities with FREEZE on (no fetch changes any episode):

  A   K-P as is: Hippocampus._compete on sims * tanh(effective_strength), the gated path recall() takes with FREEZE_USE.
      Must reproduce the banked fetch idx and recalled text on every row of the night, else the night is void.
  D   reference only (no bar): the existing RANK_ON_RAW switch (results/2026-11-20-rank-raw-screen-1.5b.md).
  DG  D plus fire gate: fire iff A fires (the existing weighted bar and margin, unchanged) AND the raw winner's own
      weighted score clears the same bar; the reinstated episode is the raw-similarity winner.
  SC  strength cap: A's competition with tanh(min(eff, SC_CAP)) for non-permanent episodes (permanent unchanged).

Deltas are not needed for a decision. --reinstate (optional, generation, not part of the prereg bars) loads the full
store with the deltas from the final trunk's episode files (append-only) and runs the battery through Brain.hear once
per arm, with the arm's decision as the recall.

    python -m brain.experiments.routing_replay replay --life L.json --snaps DIR --trunk T --model M --device cpu --out O.json
    python -m brain.experiments.routing_replay validate KR.json KP.json
    python -m brain.experiments.routing_replay score O_s319.json O_s320.json O_s321.json [--out verdict.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch

from brain import clock
from brain.hippocampus import Episode, Hippocampus

ARMS = ("A", "D", "DG", "SC")
BARRED = ("DG", "SC")      # the arms the prereg bars apply to; D is the R-raw reference
SC_CAP = 1.0               # = Episode.strength default = the birth strength of a told-once episode (salience 1.0)
NIGHTS = tuple(range(5, 16))
FROZEN = "intact-frozen"
SCALARS = ("text", "strength", "created", "last_used", "uses", "permanent", "source")

# Bars (brain/research/2026-11-24-routing-screen-prereg.md section 4), fixed before any KR data.
GAIN_FRAC = 0.25           # P1: pooled right-episode fetches >= ceil((1 + GAIN_FRAC) * A's) on the surviving rows
GAIN_MIN = 10              # ... and at least A's + 10
NEW_WRONG_MAX_SEED = 1     # K3: rows A fetched the right episode on that the arm turns into a wrong episode, per seed
FIRE_EXCESS_MAX_SEED = 1   # K4: fires on open/greet/never/untold-neardup cues above A's, per seed
DIRECTION_SEEDS = 2        # K2: right-episode fetches rise on >= 2 of 3 seeds, and fall by > 1 on none
VOID_NIGHTS_MAX = 2        # a seed with more than 2 of 11 nights void is void


# ------------------------------------------------------------------ the arms (a wrapper over scores / _compete)
def _frozen(hc: Hippocampus):
    hc.FREEZE_USE, hc.RANK_ON_RAW, hc.USE_GATE = True, False, None


def decide(hc: Hippocampus, cortex, base, cue: str, now: float | None = None, sims=None) -> dict:
    """The fetch decision of every arm for one cue, from one cue read. No episode is changed (FREEZE).
    Returns {arm: {"idx", "score"}, "A_fetch": A's last_fetch record, "raw_winner", "weighted_winner"}."""
    cue = cue.strip()
    question = cue.endswith("?")          # Brain.hear's own question test (the existing scaffold), not a new rule
    now = clock.now() if now is None else now
    if sims is None:
        sims = hc.similarities(cortex, base, cue)
    eff = [hc.effective_strength(e, now) for e in hc.episodes]
    w = torch.tensor([math.tanh(x) for x in eff])
    score = sims * w
    saved = (hc.FREEZE_USE, hc.RANK_ON_RAW, hc.USE_GATE, hc.last_fetch)
    out = {}
    try:
        _frozen(hc)
        _, s = hc._compete(score, question, now, sims=sims)              # = recall()'s gated path under FREEZE_USE
        fa = dict(hc.last_fetch)
        out["A"] = {"idx": fa["idx"], "score": s}
        hc.RANK_ON_RAW = True
        _, s_d = hc._compete(score, question, now, sims=sims)
        out["D"] = {"idx": hc.last_fetch["idx"], "score": s_d}
        hc.RANK_ON_RAW = False
        bar = hc.FAMILIARITY if question else hc.FAMILIARITY_SPONTANEOUS
        raw = int(torch.argmax(sims))
        fire = fa["idx"] is not None and float(score[raw]) >= bar
        out["DG"] = {"idx": raw if fire else None, "score": float(score[raw])}
        w_sc = torch.tensor([math.tanh(x if e.permanent else min(x, SC_CAP)) for x, e in zip(eff, hc.episodes)])
        _, s_sc = hc._compete(sims * w_sc, question, now, sims=sims)
        out["SC"] = {"idx": hc.last_fetch["idx"], "score": s_sc}
    finally:
        hc.FREEZE_USE, hc.RANK_ON_RAW, hc.USE_GATE, hc.last_fetch = saved
    out["A_fetch"] = fa
    out["raw_winner"] = raw
    out["weighted_winner"] = int(torch.argmax(score))
    return out


# ------------------------------------------------------------------ snapshot -> Brain
def _episode_rows(snap: str) -> list:
    with open(os.path.join(snap, "episodes.json")) as fh:
        idx = json.load(fh)
    return idx if isinstance(idx, list) else idx.get("episodes", [])


def build_home(snap: str, home: str, trunk: str | None = None, with_deltas: bool = False) -> list:
    """A temporary Brain home from a night snapshot. Without deltas the episode index is written empty (the store is
    injected by open_snapshot); with deltas the episode files are links to the final trunk's (written once)."""
    shutil.rmtree(home, ignore_errors=True)
    os.makedirs(home)
    for f in ("meta.json", "cortex_adapters.pt", "keep_bank.json"):
        if os.path.exists(os.path.join(snap, f)):
            shutil.copy2(os.path.join(snap, f), os.path.join(home, f))
    rows = _episode_rows(snap)
    if with_deltas:
        os.makedirs(os.path.join(home, "episodes"))
        for r in rows:
            src = os.path.join(trunk, "episodes", r["file"])
            if not os.path.exists(src):
                raise FileNotFoundError(f"{src}: the final trunk lacks an episode of the snapshot (not append-only?)")
            os.symlink(os.path.abspath(src), os.path.join(home, "episodes", r["file"]))
        shutil.copy2(os.path.join(snap, "episodes.json"), os.path.join(home, "episodes.json"))
    else:
        with open(os.path.join(home, "episodes.json"), "w") as fh:
            json.dump([], fh)
    return rows


def open_snapshot(snap: str, home: str, cortex, brain_kw: dict, trunk: str | None = None, with_deltas: bool = False):
    """A Brain whose store is the snapshot's: order of episodes.json, keys from keys.pt, scalars from episodes.json.
    Opened as life.probe opens a probe copy (sleep_on_restart=False, lock), FREEZE on. Without deltas, Episode.delta is
    None (a decision never reads it)."""
    from brain.brain import Brain
    rows = build_home(snap, home, trunk, with_deltas)
    keys = torch.load(os.path.join(snap, "keys.pt"), map_location="cpu", weights_only=False)
    missing = [r["file"] for r in rows if r["file"] not in keys]
    if missing:
        raise ValueError(f"{snap}/keys.pt lacks keys for {missing[:3]} ({len(missing)} episodes)")
    b = Brain(home, cortex=cortex, lock=True, **{**brain_kw, "sleep_on_restart": False})
    if with_deltas:
        for e, r in zip(b.hippocampus.episodes, rows):
            if not torch.equal(e.key, keys[r["file"]]):
                raise ValueError(f"{r['file']}: trunk episode key differs from the snapshot's keys.pt")
            for k in ("strength", "uses", "last_used"):
                assert getattr(e, k) == r[k], (r["file"], k)
    else:
        b.hippocampus.episodes = [Episode(delta=None, key=keys[r["file"]], **{k: r[k] for k in SCALARS if k in r})
                                  for r in rows]
    b.hippocampus.FREEZE_USE = True
    return b


# ------------------------------------------------------------------ population (from the banked frozen rows)
def row_class(row: dict, ledger: dict) -> str:
    """told = the 223-row population of the research note (ledger item with a telling, scored for a fetch: H, rotating,
    told near-duplicate); fire = open-*, greet, never, untold near-duplicate; identity = H items with no telling
    (name / nova: their birth episode holds the key; report only)."""
    it = row.get("item")
    if "wrong_fetch" in row:
        return "told" if it in ledger and ledger[it].get("telling") else "identity"
    return "fire"


def banked_rows(res: dict, night: int) -> list:
    p = [x for x in res["probes"] if x["arm"] == FROZEN and x.get("line") == "trunk" and not x.get("tag")
         and x["night"] == night]
    if len(p) != 1:
        raise ValueError(f"night {night}: {len(p)} trunk intact-frozen probes (need exactly 1; was --frozen-probe on?)")
    return p[0]["rows"]


def _qs(rows: list) -> list:
    """The battery as life.run_battery takes it, rebuilt from the banked rows (order kept)."""
    out = []
    for r in rows:
        q = {"kind": r["kind"], "text": r["you"], "item": r.get("item")}
        if r["kind"] == "neardup":
            q["told"] = "wrong_fetch" in r
        if r["kind"].startswith("open-") and "named_fox" in r:
            q["slot"] = True
        out.append(q)
    return out


# ------------------------------------------------------------------ one night
def replay_night(res: dict, night: int, snap: str, cortex, work: str, weekend: bool, trunk: str | None = None,
                 reinstate: bool = False) -> dict:
    from brain.eval import life
    from brain.brain import _is_telling
    LB = life.use_split(res.get("split", "dev"))
    ledger, seed = res["ledger"], res["seed"]
    rows_b = banked_rows(res, night)
    for r in rows_b:
        if _is_telling(r["you"]):
            raise ValueError(f"night {night}: battery cue {r['you']!r} is a telling (it would be stored mid-battery)")
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"s{seed}_n{night}")
    try:
        sim.to(life.day_start(night + 1, weekend))
        life._reseed(seed, 500 + night)
        b = open_snapshot(snap, home, cortex, res["brain_kw"])
        try:
            hc = b.hippocampus
            texts = [e.text for e in hc.episodes]
            out_rows, void = [], []
            for i, r in enumerate(rows_b):
                d = decide(hc, b.cortex, b.base, r["you"])
                fx = r.get("fetch") or {}
                row = {"night": night, "i": i, "item": r.get("item"), "kind": r["kind"], "cue": r["you"],
                       "cls": row_class(r, ledger), "banked_idx": fx.get("idx"), "banked_recalled": r.get("recalled"),
                       "sim_raw": float(d["A_fetch"]["sim_raw"]), "banked_sim_raw": fx.get("sim_raw"),
                       "raw_winner": d["raw_winner"], "weighted_winner": d["weighted_winner"]}
                it = LB.ALL[r["item"]] if row["cls"] in ("told", "identity") else None
                for a in ARMS:
                    k = d[a]["idx"]
                    row[a] = {"idx": k, "text": texts[k] if k is not None else None,
                              "right_episode": bool(life.episode_is_source(it, texts[k])) if it is not None and k is not None
                              else (False if it is not None else None)}
                if row["A"]["idx"] != row["banked_idx"] or row["A"]["text"] != row["banked_recalled"]:
                    void.append(i)
                if it is not None and row["A"]["right_episode"] != bool(r.get("right_episode")):
                    void.append(i)
                out_rows.append(row)
        finally:
            b.release()
        rec = {"night": night, "store": len(texts), "rows": out_rows, "void_rows": sorted(set(void)),
               "void": bool(void),
               "max_abs_sim_dev": max((abs(x["sim_raw"] - x["banked_sim_raw"]) for x in out_rows
                                       if x["banked_sim_raw"] is not None), default=None)}
        if reinstate:
            rec["replies"] = _reinstate_night(res, night, snap, cortex, work, weekend, trunk, rows_b, sim, LB)
        return rec
    finally:
        clock.set(None)
        shutil.rmtree(home, ignore_errors=True)


def _reinstate_night(res, night, snap, cortex, work, weekend, trunk, rows_b, sim, LB) -> dict:
    """Optional reply read (not barred): the battery through Brain.hear once per arm, each from a fresh copy and the
    probe's reseed, with the arm's decision as the recall. Arm A here is the banked frozen probe itself."""
    from brain.eval import life
    if trunk is None:
        raise ValueError("--reinstate needs --trunk (the episode deltas)")
    out = {}
    for arm in ARMS:
        home = os.path.join(work, f"s{res['seed']}_n{night}_{arm}")
        sim.to(life.day_start(night + 1, weekend))
        life._reseed(res["seed"], 500 + night)
        b = open_snapshot(snap, home, cortex, res["brain_kw"], trunk=trunk, with_deltas=True)
        try:
            hc = b.hippocampus

            def recall(cortex_, base, cue, speaker="User", question=True, exclude=(), hc=hc, arm=arm):
                d = decide(hc, cortex_, base, cue)
                hc.last_fetch = d["A_fetch"]
                k = d[arm]["idx"]
                return (hc.episodes[k], d[arm]["score"]) if k is not None else (None, d[arm]["score"])
            hc.recall = recall
            rows = life.run_battery(b, _qs(rows_b), res["ledger"], night)
        finally:
            b.release()
            shutil.rmtree(home, ignore_errors=True)
        out[arm] = [{"brain": r["brain"], "recalled": r["recalled"], "ok": r.get("ok")} for r in rows]
    out["A_reproduces_banked_replies"] = sum(x["brain"] == r["brain"] for x, r in zip(out["A"], rows_b))
    return out


def replay(res: dict, snaps: str, cortex, work: str, weekend: bool, nights=NIGHTS, trunk=None, reinstate=False,
           log=print) -> dict:
    out = {"seed": res["seed"], "split": res.get("split"), "brain_kw": res["brain_kw"], "weekend": weekend,
           "sc_cap": SC_CAP, "arms": list(ARMS), "nights": []}
    os.makedirs(work, exist_ok=True)
    for n in nights:
        snap = os.path.join(snaps, f"n{n}")
        if not os.path.isdir(snap):
            out["nights"].append({"night": n, "void": True, "missing_snapshot": True, "rows": []})
            continue
        rec = replay_night(res, n, snap, cortex, work, weekend, trunk, reinstate)
        log(f"s{res['seed']} n{n}: store {rec['store']}, void rows {rec['void_rows']}, sim dev {rec['max_abs_sim_dev']}")
        out["nights"].append(rec)
    return out


# ------------------------------------------------------------------ validity gate: KR life = KP life
def validate(kr: dict, kp: dict) -> list:
    """Every probe row of the KR life must equal the KP life's (reply text and recalled episode, and the frozen fetch
    idx). Returns the mismatches; any mismatch voids the seed."""
    def key(p):
        return (p.get("line"), p["night"], p["arm"], p.get("tag"))
    a, b = {key(p): p for p in kr["probes"]}, {key(p): p for p in kp["probes"]}
    bad = [{"probe": list(k), "why": "missing in KR" if k not in a else "missing in KP"} for k in set(a) ^ set(b)]
    for k in sorted(set(a) & set(b), key=str):
        ra, rb = a[k]["rows"], b[k]["rows"]
        if len(ra) != len(rb):
            bad.append({"probe": list(k), "why": f"rows {len(ra)} vs {len(rb)}"})
            continue
        for i, (x, y) in enumerate(zip(ra, rb)):
            for f in ("you", "brain", "recalled"):
                if x.get(f) != y.get(f):
                    bad.append({"probe": list(k), "row": i, "field": f})
            if (x.get("fetch") or {}).get("idx") != (y.get("fetch") or {}).get("idx"):
                bad.append({"probe": list(k), "row": i, "field": "fetch.idx"})
    return bad


# ------------------------------------------------------------------ scorer (the prereg bars)
def _count(rows, arm):
    told = [r for r in rows if r["cls"] == "told"]
    fire = [r for r in rows if r["cls"] == "fire"]
    return {"told_n": len(told), "right": sum(r[arm]["right_episode"] for r in told),
            "wrong": sum(r[arm]["idx"] is not None and not r[arm]["right_episode"] for r in told),
            "fire_n": len(fire), "fires": sum(r[arm]["idx"] is not None for r in fire),
            "new_wrong_on_A_right": sum(r["A"]["right_episode"] and r[arm]["idx"] is not None
                                        and not r[arm]["right_episode"] for r in told),
            "lost_on_A_right": sum(r["A"]["right_episode"] and r[arm]["idx"] is None for r in told),
            "gained_on_A_wrong": sum(not r["A"]["right_episode"] and r[arm]["right_episode"] for r in told)}


def score(outs: list, validity: dict | None = None) -> dict:
    """outs: replay outputs, one per seed. validity: {seed: [mismatches]} from validate(); a seed with any is void."""
    seeds, per_night = {}, []
    for o in outs:
        s = o["seed"]
        bad_life = bool((validity or {}).get(s) or (validity or {}).get(str(s)))
        nights = [n for n in o["nights"] if not n.get("void")]
        nvoid = len(o["nights"]) - len(nights)
        rows = [r for n in nights for r in n["rows"]]
        seeds[s] = {"void": bad_life or nvoid > VOID_NIGHTS_MAX, "void_life": bad_life, "void_nights": nvoid,
                    **{a: _count(rows, a) for a in ARMS}}
        for n in o["nights"]:
            per_night.append({"seed": s, "night": n["night"], "void": n.get("void", True),
                              **({a: _count(n["rows"], a) for a in ARMS} if n.get("rows") else {})})
    live = {s: v for s, v in seeds.items() if not v["void"]}
    verdict = {"seeds": seeds, "per_night": per_night, "arms": {}}
    if len(live) < 3:
        verdict["status"] = f"INCONCLUSIVE: {3 - len(live)} seed(s) void (validity gate); no bar is read"
        return verdict
    for a in BARRED:
        A_right = sum(v["A"]["right"] for v in live.values())
        right = sum(v[a]["right"] for v in live.values())
        need = max(math.ceil((1 + GAIN_FRAC) * A_right), A_right + GAIN_MIN)
        up = sum(v[a]["right"] > v["A"]["right"] for v in live.values())
        down = [s for s, v in live.items() if v[a]["right"] < v["A"]["right"] - 1]
        kills = []
        if right < need:
            kills.append(f"K1 right {right} < {need} (A {A_right})")
        if up < DIRECTION_SEEDS or down:
            kills.append(f"K2 right rises on {up}/3 seeds, falls by >1 on {down}")
        nw = {s: v[a]["new_wrong_on_A_right"] for s, v in live.items()}
        if any(x > NEW_WRONG_MAX_SEED for x in nw.values()):
            kills.append(f"K3 new wrong on A-right rows {nw} (max {NEW_WRONG_MAX_SEED}/seed)")
        fx = {s: v[a]["fires"] - v["A"]["fires"] for s, v in live.items()}
        if any(x > FIRE_EXCESS_MAX_SEED for x in fx.values()):
            kills.append(f"K4 fire excess over A {fx} (max {FIRE_EXCESS_MAX_SEED}/seed)")
        verdict["arms"][a] = {"right": right, "need": need, "A_right": A_right, "kills": kills,
                              "status": "KILLED" if kills else "PASS (fetch level only: earns a live DEV life)"}
    verdict["status"] = "READ"
    return verdict


# ------------------------------------------------------------------ CLI
def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", default=None)
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--cuda-precision", default="fp16")
    r.add_argument("--out", required=True)
    r.add_argument("--work", default=None, help="temporary homes (on disk; default <out>.work)")
    r.add_argument("--nights", default="5-15")
    r.add_argument("--no-weekend", action="store_true", help="the life ran without --weekend (K-P/KR ran with it)")
    r.add_argument("--reinstate", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("kr")
    v.add_argument("kp")
    s = sub.add_parser("score")
    s.add_argument("outs", nargs="+")
    s.add_argument("--validity", nargs="+", required=True, help="validate outputs (JSON lists), same order as outs")
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "replay":
        with open(a.life) as fh:
            res = json.load(fh)
        if cortex is None:
            from brain.cortex import Cortex
            from brain.cortex import resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision=a.cuda_precision)
        lo, _, hi = a.nights.partition("-")
        nights = range(int(lo), int(hi or lo) + 1)
        out = replay(res, a.snaps, cortex, a.work or a.out + ".work", not a.no_weekend, nights, a.trunk, a.reinstate)
        out["life"], out["snaps"], out["device"] = a.life, a.snaps, a.device
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        return out
    if a.cmd == "validate":
        with open(a.kr) as fh:
            kr = json.load(fh)
        with open(a.kp) as fh:
            kp = json.load(fh)
        bad = validate(kr, kp)
        print(json.dumps({"seed": kr.get("seed"), "mismatches": len(bad), "first": bad[:10]}, indent=1))
        return bad
    outs = []
    for f in a.outs:
        with open(f) as fh:
            outs.append(json.load(fh))
    if len(a.validity) != len(outs):
        raise SystemExit(f"score: {len(outs)} replay outputs but {len(a.validity)} --validity files (one per seed, same order)")
    validity = {}
    for o, f in zip(outs, a.validity):
        with open(f) as fh:
            x = json.load(fh)
        validity[o["seed"]] = x if isinstance(x, list) else x.get("first", []) if x.get("mismatches") else []
    for o in outs:   # tripwire (prereg review 11-24): report the worst cue-to-key similarity deviation from banked
        devs = [n.get("max_abs_sim_dev") for n in o.get("nights", []) if isinstance(n, dict) and n.get("max_abs_sim_dev") is not None]
        print(f"s{o.get('seed')}: max |sim_raw - banked| over nights = {max(devs) if devs else 'n/a'}", file=sys.stderr)
    verdict = score(outs, validity)
    text = json.dumps(verdict, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(json.dumps({k: verdict[k] for k in ("status", "arms")}, indent=1))
    print(json.dumps({s: {k: v[k] for k in ("void",) + ARMS} for s, v in verdict["seeds"].items()}, indent=1))
    return verdict


if __name__ == "__main__":
    main()
