# Capability: Episodic recall
# Record: brain/research/2026-11-27-fable-review-dg-port.md (section 3; prereg brain/research/2026-11-27-memory-ceiling-prereg.md)
# Expected-fail: partly. A measurement, not a method. If the right episode is reinstated and the slept cortex already
#   carries the fact (or speaks in the user's voice anyway), ORACLE - OFF is small: that is the NO-HEADROOM outcome, not a
#   defect. Reply-level scorer is the strict told scorer (paraphrase and voice are MISSING capabilities, PREREQUISITES.md).
"""Oracle reply ceiling on the K-P (KR) night snapshots: does a RIGHT fetch improve the reply at all?

Stage 0 (rescore, CPU seconds, report-only): the banked screen-2 diagnostic rows (s2diag_s*.json) rescored with the told-blind
bar rule of the DG prereg section 5 for arm K (S_all, slept key): the smallest question bar on the grid 0.000..1.000 step
0.005 at which K fires on the pooled fire rows no more often than A (and at half A's fires); statements use bar * 0.45/0.12;
margin 0.02 on the weighted margin. Reports right / wrong / no-fetch / fires. Rightness of a refit winner is known when K
fired at the registered bar or the weighted winner is the raw winner; otherwise it is UNKNOWN (counted separately, never
guessed). The decisions at the A-matched bar are kept for Stage 1's K-refit arm.

Stage 1 (replies, pool CPU, guarded): per (seed, night, arm) the banked battery is run through Brain.hear on a fresh copy of the
night snapshot WITH deltas (routing_replay.open_snapshot), the arm's decision replacing recall (the brain's own reinstatement
path, shipped GAIN, no host text):
  OFF           hippocampus lesion (no recall, no storing)
  A             the banked fetch (routing_replay arm A; reproduction check against the banked replies)
  ORACLE        the right episode forced on told rows (A's fetch elsewhere)
  SHAM          another told episode forced instead, matched on strength tercile and age (the no-contrast control)
  ORACLE-rekey  ORACLE with the episode's delta re-read through tonight's cortex (separates drift from content)
  K-REFIT       Stage 0's A-matched-bar decision reinstated (no change on rows it does not fire)
Non-told rows in ORACLE / SHAM / ORACLE-rekey keep A's fetch, so the running conversation differs from the banked one only
on told rows. Verdict rule fixed before data (brain/research/2026-11-27-memory-ceiling-prereg.md section 5, verdict()).

    python -m brain.experiments.memory_ceiling stage0 --diag D_s319.json D_s320.json D_s321.json --out stage0.json
    POOL_NODE=1 python -m brain.experiments.memory_ceiling run --life L.json --snaps DIR --trunk T --stage0 stage0.json \\
        --out-dir DIR [--model M] [--device cpu] [--nights 5-15] [--arms OFF,A,...]
    python -m brain.experiments.memory_ceiling score --dir DIR [--out verdict.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import time

from brain.experiments import routing_replay as RR

BAR_GRID = [round(0.005 * i, 3) for i in range(0, 201)]
MARGIN = 0.02
STMT_RATIO = 0.45 / 0.12
REG_BAR, REG_STMT = 0.165, 0.45          # the registered slept bar of arm K (screen 2)
ARMS = ("OFF", "A", "ORACLE", "SHAM", "ORACLE-rekey", "K-REFIT")
CONTROLS = ("OFF", "A", "ORACLE", "SHAM")          # a missing one makes the verdict INCONCLUSIVE
HAND_N = 40
HAND_SEED = 20261127
# Pass rule (Fable review 11-27 section 3), fixed before data.
H_ORACLE_OFF, H_NONH, H_SHAM, H_UV = 15, 6, 15, 5
N_ORACLE_OFF, N_SHAM = 8, 8
TOLD_N = 217


# ------------------------------------------------------------------ guard and provenance
def guard(env=None):
    """Heavy run only on a pool node (POOL_NODE=1) or with the explicit workstation override."""
    env = os.environ if env is None else env
    if env.get("POOL_NODE") != "1" and env.get("BRAIN_ALLOW_W3090") != "1":
        raise SystemExit("memory_ceiling run: refuses to run unless POOL_NODE=1 (pool node) or BRAIN_ALLOW_W3090=1")


def provenance(device="?") -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except OSError:
        commit = None
    if commit is None and os.path.exists(os.path.join(os.getcwd(), "COMMIT")):
        commit = open("COMMIT").read().strip()
    return {"host": socket.gethostname(), "commit": commit, "device": str(device)}


def atomic_write(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ Stage 0 (rescore of banked s2diag rows)
def fires(rec: dict, bar: float, stmt: float | None = None) -> bool:
    """The fetch rule on the recorded K_diag fields (dg_screen.apply_bar, same rule): fire iff the best weighted score clears
    the class bar (question: bar; statement: stmt or bar * 0.45/0.12) and the weighted margin is at least 0.02."""
    b = bar if rec["question"] else (stmt if stmt is not None else bar * STMT_RATIO)
    m = rec["margin_w"] if rec["margin_w"] is not None else 1.0
    return not (rec["score_best"] < b or m < MARGIN)


def fit_bar(fire_rows: list, target: int) -> float:
    """Smallest grid bar at which K fires on the pooled fire rows no more often than `target` (reads fire rows only)."""
    for b in BAR_GRID:
        if sum(fires(r["K_diag"], b) for r in fire_rows) <= target:
            return b
    return BAR_GRID[-1]


def winner_right(row: dict):
    """Is K's weighted winner the right episode? True / False when the banked fields decide it, None when they do not."""
    ww = row["K_diag"]["weighted_winner"]
    if row["K"]["idx"] is not None and row["K"]["idx"] == ww:
        return bool(row["K"]["right_episode"])
    if row["K_raw_winner"] == ww and "K_raw_right" in row:
        return bool(row["K_raw_right"])
    return None


def decide_row(row: dict, bar: float) -> dict:
    f = fires(row["K_diag"], bar)
    return {"fire": f, "idx": row["K_diag"]["weighted_winner"] if f else None, "right": winner_right(row) if f else None}


def _tally(rows: list, bar: float) -> dict:
    told = [r for r in rows if r["cls"] == "told"]
    fire = [r for r in rows if r["cls"] == "fire"]
    ident = [r for r in rows if r["cls"] == "identity"]
    d = [decide_row(r, bar) for r in told]
    return {"bar": bar, "told_n": len(told),
            "right": sum(x["fire"] and x["right"] is True for x in d),
            "wrong": sum(x["fire"] and x["right"] is False for x in d),
            "fired_unknown_right": sum(x["fire"] and x["right"] is None for x in d),
            "no_fetch": sum(not x["fire"] for x in d),
            "fires": sum(fires(r["K_diag"], bar) for r in fire),
            "fire_n": len(fire),
            "identity_fires": sum(fires(r["K_diag"], bar) for r in ident)}


def stage0(diags: list) -> dict:
    """diags: s2diag_s*.json dicts. Void nights are skipped (as the screen). Fields needed per row: cls, K_diag (score_best,
    margin_w, weighted_winner, question), K (idx, right_episode), A (idx, right_episode), K_raw_winner, K_raw_right (told)."""
    per_seed, pooled_rows = {}, []
    for d in diags:
        rows = []
        for n in d["nights"]:
            if n.get("void"):
                continue
            rows += n["rows"]
        for r in rows:
            for k in ("K_diag", "K", "A", "cls"):
                if k not in r:
                    raise ValueError(f"s{d['seed']}: row lacks {k} (s2diag fields missing)")
        per_seed[d["seed"]] = rows
        pooled_rows += rows
    a_fire_rows = [r for r in pooled_rows if r["cls"] == "fire"]
    a_fires = sum(r["A"]["idx"] is not None for r in a_fire_rows)
    reg_ok = all((r["K"]["idx"] is not None) == fires(r["K_diag"], REG_BAR, REG_STMT) for r in pooled_rows)
    targets = {"A_matched": a_fires, "half_A": a_fires // 2}
    out = {"seeds": sorted(per_seed), "A_fires_on_fire_rows": a_fires, "registered_reproduces_K": reg_ok,
           "bars": {}, "arms": {}, "per_seed": {}}
    for name, tgt in targets.items():
        b = fit_bar(a_fire_rows, tgt)
        out["bars"][name] = {"bar": b, "statement_bar": round(b * STMT_RATIO, 4), "target_fires": tgt}
        out["arms"][f"K-refit@{name}"] = _tally(pooled_rows, b)
        out["per_seed"][f"K-refit@{name}"] = {str(s): _tally(rs, b) for s, rs in per_seed.items()}
    out["arms"]["K@registered"] = _tally_registered(pooled_rows)
    out["arms"]["A"] = _tally_a(pooled_rows)
    out["per_seed"]["A"] = {str(s): _tally_a(rs) for s, rs in per_seed.items()}
    out["per_seed"]["K@registered"] = {str(s): _tally_registered(rs) for s, rs in per_seed.items()}
    dec = {}
    b = out["bars"]["A_matched"]["bar"]
    for s, rs in per_seed.items():
        for r in rs:
            x = decide_row(r, b)
            if x["fire"]:
                dec.setdefault(str(s), {}).setdefault(str(r["night"]), {})[str(r["i"])] = x["idx"]
    out["decisions_A_matched"] = dec
    return out


def _tally_registered(rows: list) -> dict:
    told = [r for r in rows if r["cls"] == "told"]
    fire = [r for r in rows if r["cls"] == "fire"]
    return {"told_n": len(told), "right": sum(r["K"]["idx"] is not None and r["K"]["right_episode"] for r in told),
            "wrong": sum(r["K"]["idx"] is not None and not r["K"]["right_episode"] for r in told),
            "no_fetch": sum(r["K"]["idx"] is None for r in told),
            "fires": sum(r["K"]["idx"] is not None for r in fire), "fire_n": len(fire)}


def _tally_a(rows: list) -> dict:
    told = [r for r in rows if r["cls"] == "told"]
    fire = [r for r in rows if r["cls"] == "fire"]
    return {"told_n": len(told), "right": sum(r["A"]["idx"] is not None and r["A"]["right_episode"] for r in told),
            "wrong": sum(r["A"]["idx"] is not None and not r["A"]["right_episode"] for r in told),
            "no_fetch": sum(r["A"]["idx"] is None for r in told),
            "fires": sum(r["A"]["idx"] is not None for r in fire), "fire_n": len(fire)}


# ------------------------------------------------------------------ Stage 1: forced decisions
def told_episode_idx(texts: list, ledger: dict, LB, is_source) -> dict:
    """{episode index: item id} for every stored episode that is some ledger item's own telling."""
    out = {}
    for i, t in enumerate(texts):
        for iid, it in ledger.items():
            if it.get("telling") and iid in LB.ALL and is_source(LB.ALL[iid], t):
                out[i] = iid
                break
    return out


def tercile_of(values: list, v: float) -> int:
    s = sorted(values)
    lo, hi = s[len(s) // 3], s[(2 * len(s)) // 3]
    return 0 if v < lo else (1 if v < hi else 2)


def pick_sham(right_idx: int, row_item: str, told_map: dict, eff: list, created: list) -> int | None:
    """A forced other told episode: not the row's own telling, same strength tercile (over the told episodes) as the right
    episode, nearest in age (created); ties by lowest index. None if there is no candidate."""
    ids = list(told_map)
    pool = [i for i in ids if told_map[i] != row_item and i != right_idx]
    if not pool:
        return None
    ref_t = tercile_of([eff[i] for i in ids], eff[right_idx])
    same = [i for i in pool if tercile_of([eff[j] for j in ids], eff[i]) == ref_t]
    cand = same or pool
    return min(cand, key=lambda i: (abs(created[i] - created[right_idx]), i))


def forced_map(arm: str, rows: list, dec_k: dict) -> dict:
    """{row index: episode idx or None} for the arm. rows: per-row decision records of the night:
    {i, cls, a_idx, right_idx, sham_idx, k_idx}. Rows absent from the map take A's fetch."""
    m = {}
    for r in rows:
        if arm in ("ORACLE", "ORACLE-rekey") and r["cls"] == "told":
            if r["right_idx"] is not None:        # no stored right episode: the row keeps A's fetch (an oracle cannot fix it)
                m[r["i"]] = r["right_idx"]
        elif arm == "SHAM" and r["cls"] == "told":
            if r["sham_idx"] is not None:
                m[r["i"]] = r["sham_idx"]
        elif arm == "K-REFIT" and r["cls"] in ("told", "fire"):     # identity rows keep A's fetch
            m[r["i"]] = r["k_idx"]
    return m


def night_decisions(res: dict, night: int, snap: str, trunk: str, cortex, work: str, weekend: bool, k_dec: dict) -> dict:
    """The cheap pass (one cue read per row, no generation): void check as routing_replay, plus right / sham / K-refit indices."""
    from brain import clock
    from brain.eval import life
    from brain.brain import _is_telling
    ledger, seed = res["ledger"], res["seed"]
    rows_b = RR.banked_rows(res, night)
    for r in rows_b:
        if _is_telling(r["you"]):
            raise ValueError(f"night {night}: battery cue {r['you']!r} is a telling")
    LB = life.use_split(res.get("split", "dev"))
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"dec_s{seed}_n{night}")
    try:
        sim.to(life.day_start(night + 1, weekend))
        life._reseed(seed, 500 + night)
        b = RR.open_snapshot(snap, home, cortex, res["brain_kw"], trunk=trunk, with_deltas=True)
        try:
            hc = b.hippocampus
            texts = [e.text for e in hc.episodes]
            now = clock.now()
            eff = [hc.effective_strength(e, now) for e in hc.episodes]
            created = [e.created for e in hc.episodes]
            told_map = told_episode_idx(texts, ledger, LB, life.episode_is_source)
            out, void = [], []
            for i, r in enumerate(rows_b):
                d = RR.decide(hc, b.cortex, b.base, r["you"], now=now)
                cls = RR.row_class(r, ledger)
                rec = {"i": i, "cls": cls, "kind": r["kind"], "item": r.get("item"), "a_idx": d["A"]["idx"],
                       "right_idx": None, "sham_idx": None, "k_idx": (k_dec or {}).get(str(i))}      # None = K-refit does not fire
                if cls == "told":
                    it = LB.ALL[r["item"]]
                    cand = [j for j, t in enumerate(texts) if life.episode_is_source(it, t)]
                    if d["A"]["idx"] in cand:
                        rec["right_idx"] = d["A"]["idx"]
                    elif cand:
                        sims = hc.similarities(b.cortex, b.base, r["you"])
                        rec["right_idx"] = max(cand, key=lambda j: (float(sims[j]), -j))
                    if rec["right_idx"] is not None:
                        rec["sham_idx"] = pick_sham(rec["right_idx"], r["item"], told_map, eff, created)
                    rec["a_right"] = bool(d["A"]["idx"] is not None and life.episode_is_source(it, texts[d["A"]["idx"]]))
                    if rec["a_right"] != bool(r.get("right_episode")):
                        void.append(i)
                fx = r.get("fetch") or {}
                if d["A"]["idx"] != fx.get("idx") or (texts[d["A"]["idx"]] if d["A"]["idx"] is not None else None) != r.get("recalled"):
                    void.append(i)
                out.append(rec)
        finally:
            b.release()
        return {"seed": seed, "night": night, "store": len(texts), "rows": out, "void_rows": sorted(set(void)), "void": bool(void)}
    finally:
        clock.set(None)
        shutil.rmtree(home, ignore_errors=True)


def run_arm(res: dict, night: int, snap: str, trunk: str, cortex, work: str, weekend: bool, arm: str, dec: dict) -> dict:
    """One (night, arm) unit: the banked battery through Brain.hear with the arm's recall."""
    from brain import clock
    from brain.eval import life
    rows_b = RR.banked_rows(res, night)
    seed = res["seed"]
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"s{seed}_n{night}_{arm}")
    t0 = time.time()
    try:
        sim.to(life.day_start(night + 1, weekend))
        life._reseed(seed, 500 + night)
        b = RR.open_snapshot(snap, home, cortex, res["brain_kw"], trunk=trunk, with_deltas=True)
        try:
            hc = b.hippocampus
            fm = forced_map(arm, dec["rows"], dec.get("k_dec") or {})
            calls = {"n": 0}
            trace, rekeyed = [], {}
            if arm == "ORACLE-rekey":      # the right episodes' deltas re-read through tonight's cortex
                def _norm(dl):
                    return None if dl is None else round(sum(float(x.float().norm()) for x in dl if x is not None), 4)
                for i in sorted({v for v in fm.values() if v is not None}):
                    ep = hc.episodes[i]
                    old = _norm(ep.delta)
                    _, st = b.cortex.read(ep.text + "\n\n", b.base)
                    d = b.cortex.matrix_delta(st, b.base)
                    ep.delta = [x.half().cpu() if x is not None else None for x in d]
                    rekeyed[str(i)] = {"old_norm": old, "new_norm": _norm(ep.delta)}
                    if not rekeyed[str(i)]["new_norm"]:
                        raise RuntimeError(f"s{seed} n{night}: rekey of episode {i} produced an empty delta")
            if arm == "OFF":
                b.set_lesions(hippocampus_off=True)
            else:
                def recall(cortex_, base, cue, speaker="User", question=True, exclude=(), hc=hc):
                    i = calls["n"]
                    calls["n"] += 1
                    if i >= len(rows_b) or cue.strip() != rows_b[i]["you"].strip():
                        raise RuntimeError(f"s{seed} n{night} {arm}: recall call {i} cue {cue!r} is not battery row {i}")
                    k = fm.get(i, dec["rows"][i]["a_idx"])
                    hc.last_fetch = {"idx": k}
                    trace.append({"i": i, "forced_idx": k, "forced": i in fm,
                                  "ep_text_ok": k is None or hc.episodes[k].text is not None})
                    return (hc.episodes[k], 0.0) if k is not None else (None, 0.0)
                hc.recall = recall
            rows = life.run_battery(b, RR._qs(rows_b), res["ledger"], night)
            if arm != "OFF" and calls["n"] != len(rows_b):
                raise RuntimeError(f"s{seed} n{night} {arm}: {calls['n']} recall calls for {len(rows_b)} battery rows")
            texts = [e.text for e in hc.episodes]
        finally:
            b.release()
            shutil.rmtree(home, ignore_errors=True)
        keep = ("kind", "item", "you", "brain", "recalled", "ok", "fetched", "wrong_fetch", "right_episode", "false",
                "invented", "intrusion", "needless")
        return {"seed": seed, "night": night, "arm": arm, "seconds": round(time.time() - t0, 1),
                "recall_calls": calls["n"] if arm != "OFF" else None, "n_rows": len(rows_b),
                "rekeyed": rekeyed, "stage0_sha": dec.get("stage0_sha"),
                "rows": [{**{k: r.get(k) for k in keep}, "cls": d["cls"], "kind_row": d.get("kind"),
                          "right_idx": d.get("right_idx"),
                          "forced_idx": t["forced_idx"] if t else None, "forced": bool(t and t["forced"]),
                          "engaged": bool(t and t["forced"] and t["forced_idx"] is not None
                                          and r.get("recalled") == texts[t["forced_idx"]])}
                         for r, d, t in zip(rows, dec["rows"], trace or [None] * len(rows))],
                "reproduces_banked": sum(x["brain"] == r["brain"] for x, r in zip(rows, rows_b)) if arm == "A" else None}
    finally:
        clock.set(None)


def run(res: dict, snaps: str, trunk: str, cortex, work: str, out_dir: str, weekend: bool, stage0_path: str,
        nights=RR.NIGHTS, arms=ARMS, log=print) -> None:
    """Per-unit atomic result files with resume: dec_s*_n*.json (the cheap pass) and u_s*_n*_<arm>.json."""
    os.makedirs(work, exist_ok=True)
    k_all = json.load(open(stage0_path)).get("decisions_A_matched", {}) if stage0_path else {}
    seed = res["seed"]
    prov = provenance(getattr(cortex, "device", "?"))
    import hashlib
    sha = hashlib.sha256(open(stage0_path, "rb").read()).hexdigest()[:16] if stage0_path else None
    for n in nights:
        snap = os.path.join(snaps, f"n{n}")
        if not os.path.isdir(snap):
            log(f"s{seed} n{n}: missing snapshot, skipped")
            continue
        dp = os.path.join(out_dir, f"dec_s{seed}_n{n}.json")
        if os.path.exists(dp):
            dec = json.load(open(dp))
            if dec.get("stage0_sha") != sha or dec.get("commit") != prov["commit"]:
                raise SystemExit(f"{dp}: written under stage0 {dec.get('stage0_sha')} / commit {dec.get('commit')}, now {sha} / "
                                 f"{prov['commit']}; use a fresh out dir (no mixing)")
        else:
            dec = night_decisions(res, n, snap, trunk, cortex, work, weekend, (k_all.get(str(seed)) or {}).get(str(n), {}))
            dec["k_dec"] = (k_all.get(str(seed)) or {}).get(str(n), {})
            dec.update(prov)
            dec["stage0_sha"] = sha
            atomic_write(dp, dec)
        if dec["void"]:
            log(f"s{seed} n{n}: VOID rows {dec['void_rows']}; no arms run")
            continue
        for arm in arms:
            up = os.path.join(out_dir, f"u_s{seed}_n{n}_{arm}.json")
            if os.path.exists(up):
                if json.load(open(up)).get("commit") != prov["commit"]:
                    raise SystemExit(f"{up}: written under another commit; use a fresh out dir (no mixing)")
                continue
            u = run_arm(res, n, snap, trunk, cortex, work, weekend, arm, dec)
            u.update(prov)
            u["stage0_sha"] = sha
            atomic_write(up, u)
            log(f"s{seed} n{n} {arm}: {u['seconds']}s")


# ------------------------------------------------------------------ measures and verdict
def _voice_fn():
    from brain.experiments import cascade_screen as CS
    return CS.own2


def _knows(row: dict, ledger_keys: dict) -> bool:
    """The banked scorer voice_split.knows itself (ledger {item: {"keys": ...}})."""
    from brain.experiments import voice_split as VS
    return VS.knows({"item": row.get("item"), "brain": row.get("brain") or ""},
                    {k: {"keys": v} for k, v in ledger_keys.items()})


def arm_measures(units: list, ledger_keys: dict, own2=None) -> dict:
    """Pooled / per seed / non-H / per (seed, item) cluster counts for one arm, from its unit files. Told rows are those
    the unit marks cls == 'told'; ok is the strict told scorer."""
    own2 = own2 or _voice_fn()
    told = [(u["seed"], r) for u in units for r in u["rows"] if r["cls"] == "told"]
    out = {"told_n": len(told), "ok": sum(bool(r["ok"]) for _, r in told),
           "ok_nonH": sum(bool(r["ok"]) for _, r in told if r["kind"] != "H"),
           "nonH_n": sum(r["kind"] != "H" for _, r in told),
           "knows": sum(_knows(r, ledger_keys) for _, r in told),
           "uv_knows": sum(_knows(r, ledger_keys) and own2(r["brain"]) for _, r in told),
           "uv_told": sum(own2(r["brain"]) for _, r in told),
           "per_seed": {}, "cluster_ok": {}}
    for s, r in told:
        ps = out["per_seed"].setdefault(str(s), {"n": 0, "ok": 0})
        ps["n"] += 1
        ps["ok"] += bool(r["ok"])
        c = out["cluster_ok"].setdefault(f"{s}:{r['item']}", {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += bool(r["ok"])
    out["units"] = sorted([u["seed"], u["night"]] for u in units)
    out["engaged"] = {"told_forced": sum(bool(r.get("forced")) for _, r in told),
                      "told_engaged": sum(bool(r.get("engaged")) for _, r in told),
                      "told_no_right_episode": sum(r.get("right_idx") is None for _, r in told)}
    fire = [r for u in units for r in u["rows"] if r["cls"] == "fire"]
    by_kind = {}
    for r in fire:
        k = by_kind.setdefault(r.get("kind") or "?", {"n": 0, "fetched": 0, "false": 0, "intrusion": 0, "invented": 0})
        k["n"] += 1
        k["fetched"] += bool(r.get("fetched") or r.get("needless"))
        k["false"] += bool(r.get("false"))
        k["intrusion"] += bool(r.get("intrusion"))
        k["invented"] += bool(r.get("invented"))
    out["fire_by_kind"] = by_kind
    out["fire_never"] = {k: v for k, v in by_kind.items() if k.startswith("never") or k.startswith("neardup")}
    out["fire"] = {"n": len(fire), "fetched": sum(bool(r.get("fetched") or r.get("needless")) for r in fire),
                   "false": sum(bool(r.get("false")) for r in fire), "intrusion": sum(bool(r.get("intrusion")) for r in fire),
                   "invented": sum(bool(r.get("invented")) for r in fire)}
    return out


def _paired(m_a: dict, m_b: dict) -> dict:
    """a - b: pooled, non-H, per seed, per cluster (sign test counts)."""
    seeds = sorted(set(m_a["per_seed"]) | set(m_b["per_seed"]))
    cl = sorted(set(m_a["cluster_ok"]) | set(m_b["cluster_ok"]))
    diffs = [m_a["cluster_ok"].get(c, {"ok": 0})["ok"] - m_b["cluster_ok"].get(c, {"ok": 0})["ok"] for c in cl]
    return {"pooled": m_a["ok"] - m_b["ok"], "nonH": m_a["ok_nonH"] - m_b["ok_nonH"],
            "per_seed": {s: m_a["per_seed"].get(s, {"ok": 0})["ok"] - m_b["per_seed"].get(s, {"ok": 0})["ok"] for s in seeds},
            "clusters": len(cl), "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs)}


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def verdict(meas: dict, n_seeds_expected: int = 3, expected_units=None) -> dict:
    """meas: {arm: arm_measures} for the arms present. The HEADROOM / NO-HEADROOM / UNDEFINED rule of Fable's review
    (section 3), verbatim thresholds; INCONCLUSIVE if any control arm (OFF, A, ORACLE, SHAM) is missing or too few seeds
    are present. All thresholds are absolute counts."""
    missing = [a for a in CONTROLS if a not in meas]
    if missing:
        return {"status": f"INCONCLUSIVE: control arm(s) missing {missing}", "checks": {}}
    seeds = sorted(meas["ORACLE"]["per_seed"])
    if len(seeds) < n_seeds_expected or any(sorted(meas[a]["per_seed"]) != seeds for a in CONTROLS):
        return {"status": f"INCONCLUSIVE: seeds present {seeds} (need {n_seeds_expected}, same in every control arm)",
                "checks": {}}
    ref = meas["ORACLE"].get("units")
    if expected_units is not None:
        ref = sorted([list(x) for x in expected_units])
    if ref is not None:
        bad = [a for a in CONTROLS if meas[a].get("units") != ref]
        if bad:
            return {"status": f"INCONCLUSIVE: (seed, night) units differ from the reference in {bad}", "checks": {}}
        if len({meas[a]["told_n"] for a in CONTROLS}) != 1:
            return {"status": "INCONCLUSIVE: told row counts differ across control arms", "checks": {}}
    o_off, o_sham = _paired(meas["ORACLE"], meas["OFF"]), _paired(meas["ORACLE"], meas["SHAM"])
    uv_gap = meas["ORACLE"]["uv_knows"] - meas["OFF"]["uv_knows"]
    checks = {"oracle_minus_off": o_off["pooled"], "oracle_minus_off_nonH": o_off["nonH"],
              "oracle_minus_off_per_seed": o_off["per_seed"],
              "cluster": {k: o_off[k] for k in ("clusters", "wins", "losses", "ties")},
              "cluster_sign_p_report_only": sign_p(o_off["wins"], o_off["losses"]),
              "oracle_minus_sham": o_sham["pooled"], "uv_knows_oracle_minus_off": uv_gap}
    head = {"H1 oracle-off >= 15": o_off["pooled"] >= H_ORACLE_OFF,
            "H2 non-H >= 6": o_off["nonH"] >= H_NONH,
            "H3 positive on every seed": all(v > 0 for v in o_off["per_seed"].values()),
            "H4 cluster wins > losses": o_off["wins"] > o_off["losses"],
            "H5 oracle-sham >= 15": o_sham["pooled"] >= H_SHAM,
            "H6 user-voice not above OFF by more than 5": uv_gap <= H_UV}
    checks["headroom_clauses"] = head
    if o_off["pooled"] < N_ORACLE_OFF or o_sham["pooled"] < N_SHAM:
        status = "NO-HEADROOM"
    elif all(head.values()):
        status = "HEADROOM"
    else:
        status = "UNDEFINED"
    return {"status": status, "checks": checks}


def handread(units_by_arm: dict, n: int = HAND_N, seed: int = HAND_SEED) -> tuple:
    """Blind hand-read sheet: the same n told (seed, night, row) keys across arms, replies shuffled under random labels.
    Returns (sheet without arm names, key {label: arm})."""
    keys = sorted({(u["seed"], u["night"], i) for us in units_by_arm.values() for u in us
                   for i, r in enumerate(u["rows"]) if r["cls"] == "told"})
    rng = random.Random(seed)
    pick = rng.sample(keys, min(n, len(keys)))
    sheet, key = [], {}
    for (s, nt, i) in pick:
        reps = []
        for arm, us in sorted(units_by_arm.items()):
            for u in us:
                if (u["seed"], u["night"]) == (s, nt):
                    reps.append((arm, u["rows"][i]["brain"]))
        rng.shuffle(reps)
        labels = {}
        for j, (arm, text) in enumerate(reps):
            labels[chr(65 + j)] = text
            key[f"{s}:{nt}:{i}:{chr(65 + j)}"] = arm
        us0 = next(u for us in units_by_arm.values() for u in us if (u["seed"], u["night"]) == (s, nt))
        sheet.append({"id": f"{s}:{nt}:{i}", "question": us0["rows"][i]["you"], "replies": labels})
    return sheet, key


def load_units(d: str) -> dict:
    by = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("u_s") and f.endswith(".json"):
            u = json.load(open(os.path.join(d, f)))
            by.setdefault(u["arm"], []).append(u)
    return by


def score(d: str, ledger_keys: dict, write_handread: bool = True) -> dict:
    by = load_units(d)
    decs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.startswith("dec_") and f.endswith(".json")]
    commits = {x.get("commit") for x in decs} | {u.get("commit") for us in by.values() for u in us}
    shas = {x.get("stage0_sha") for x in decs}
    meas = {a: arm_measures(us, ledger_keys) for a, us in by.items()}
    if len(commits) > 1 or len(shas) > 1:
        v = {"status": f"INCONCLUSIVE: mixed provenance (commits {sorted(map(str, commits))}, stage0 {sorted(map(str, shas))})",
             "checks": {}}
    else:
        expected = [(x["seed"], x["night"]) for x in decs if not x["void"] and "seed" in x]
        v = verdict(meas, expected_units=expected or None)
    v["arms"] = meas
    v["pairs"] = {f"{a}-OFF": _paired(meas[a], meas["OFF"]) for a in meas if a != "OFF" and "OFF" in meas}
    v["A_reproduces_banked"] = sum((u["reproduces_banked"] or 0) for u in by.get("A", [])), \
        sum(u["n_rows"] for u in by.get("A", []))
    if write_handread and all(a in by for a in CONTROLS):
        sheet, key = handread({a: by[a] for a in ("OFF", "A", "ORACLE", "SHAM", "K-REFIT") if a in by})
        atomic_write(os.path.join(d, "handread_sheet.json"), sheet)
        atomic_write(os.path.join(d, "handread_key.json"), key)
    v["void_nights"] = sorted(f for f in os.listdir(d) if f.startswith("dec_") and json.load(open(os.path.join(d, f)))["void"])
    return v


# ------------------------------------------------------------------ CLI
def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s0 = sub.add_parser("stage0")
    s0.add_argument("--diag", nargs="+", required=True)
    s0.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--stage0", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--cuda-precision", default="fp16")
    r.add_argument("--work", default=None)
    r.add_argument("--nights", default="5-15")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--no-weekend", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--dir", required=True)
    sc.add_argument("--lives", nargs="+", required=True, help="the KR life JSONs (ledger keys)")
    sc.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "stage0":
        diags = [json.load(open(f)) for f in a.diag]
        out = stage0(diags)
        out["provenance"] = provenance("cpu (rescore)")
        out["inputs"] = a.diag
        atomic_write(a.out, out)
        print(json.dumps({k: out[k] for k in ("A_fires_on_fire_rows", "registered_reproduces_K", "bars", "arms")}, indent=1))
        return out
    if a.cmd == "run":
        guard()
        res = json.load(open(a.life))
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device, cuda_precision=a.cuda_precision)
        lo, _, hi = a.nights.partition("-")
        arms = tuple(x for x in a.arms.split(",") if x)
        bad = [x for x in arms if x not in ARMS]
        if bad:
            raise SystemExit(f"unknown arms {bad}")
        os.makedirs(a.out_dir, exist_ok=True)
        run(res, a.snaps, a.trunk, cortex, a.work or os.path.join(a.out_dir, f"work_s{res['seed']}"), a.out_dir,
            not a.no_weekend, a.stage0, range(int(lo), int(hi or lo) + 1), arms)
        return None
    ledger_keys = {}
    for f in a.lives:
        for iid, it in json.load(open(f))["ledger"].items():
            ledger_keys[iid] = it.get("keys") or ()
    v = score(a.dir, ledger_keys)
    if a.out:
        atomic_write(a.out, v)
    print(json.dumps({"status": v["status"], "checks": v["checks"], "A_reproduces_banked": v["A_reproduces_banked"]}, indent=1))
    return v


if __name__ == "__main__":
    main()
