"""Dev battery for the idle-replay organ (Stage-2 C5 build; docs/drafts/idle-replay/PLAN.md). Two levels:

ORGAN (no brain, seconds per seed; dev seeds 1-8 only). The Stage-2 `inner` store shape on the organ alone: the five
build facts of the tiny-demo brain encoded at turn 0, then a three-fact chain taught at turns 1-3, the idle turn at
turn 4. Draws differ only in the background-noise seed. Per seed:
  recent_intact     >= 7/10 draws reactivate the newest fact of the intact chain (beetle chase mite)
  recent_permuted   >= 7/10 draws reactivate the newest fact of the permuted chain (vole chase beetle)
  differ            >= 7/10 paired draws voice different facts after the two exposures (amendment A3, organ level)
  no_encode         every draw: nothing latches (BRAIN_SELF_INITIATE_LESION, the gate's lesion_selfinit arm)
  commit_lesion     every draw: nothing latches (engram -> commit gate closed)
  tag_lesion_moves  under the tag lesion (equal excitability: recency removed) the newest fact wins <= 5/10 draws
                    AND at least 4 fewer times than intact (a host sort that ignores the excitability fails it)
  read_invariant    every draw's latch read gives the same outcome at x0.5 and x2 of the read level
`all_pass` = every case. The decision is recomputed here from the recorded latch counts by the organ's own read
(`spiking_commit.read_with_variants`), never from a recorded winner, so the bar-sanity spec feeds synthetic counts
through the same code. Reported, not gated: `near_tie` (the second-newest fact encoded at strength 2: which wins),
first-ignition latency.

REPLY (needs the brain; pool lines, dev seeds 1-8): the Stage-2 runner's `inner` group with BRAIN_IDLE_REPLAY=1,
scored by the runner's OWN `score_inner` and `private_selected_by_state` (amendment A3). `--emit-dev-lines` writes the
lines, `--score-reply` reads them. The runner lives on research/stage2-exit-gate, so these lines run at a tip that
contains both (that branch synced with a main that contains this organ).

    python -m research.runners.idle_replay_battery --seed 1 --op 0 --out <dir>/organ_s1_op0.json
    python -m research.runners.idle_replay_battery --score <dir> --op 0
    python -m research.runners.idle_replay_battery --emit-organ-lines <SHA>
    python -m research.runners.idle_replay_battery --emit-dev-lines <SHA>
    python -m research.runners.idle_replay_battery --score-reply research/findings/raw/_idle_replay/dev
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import idle_replay_organ as IR  # noqa: E402
from research.runners import spiking_commit as SC  # noqa: E402
from tools.lab import lever  # noqa: E402

BUILD = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
         ("dog", "chase", "cat"), ("cat", "eat", "fish")]
CHAINS = {"intact": [("lynx", "chase", "vole"), ("vole", "chase", "beetle"), ("beetle", "chase", "mite")],
          "permuted": [("lynx", "chase", "mite"), ("mite", "chase", "vole"), ("vole", "chase", "beetle")]}
N_DRAWS = 10
N_LESION_DRAWS = 3
MIN_RECENT = 7            # of N_DRAWS: the newest fact wins at least this often (intact, permuted, and they differ)
MAX_TAG_RECENT = 5        # of N_DRAWS: under the tag lesion the newest fact wins at most this often ...
TAG_DROP = 4              # ... and at least this many fewer times than intact (chance is 1/8 of draws)
DEV_SEEDS = IR.DEV_SEEDS
CASES = ("recent_intact", "recent_permuted", "differ", "no_encode", "commit_lesion", "tag_lesion_moves",
         "read_invariant")
REPLY_ARMS = ("intact_a", "permuted", "lesion_selfinit")
REPLY_LABELS = "t_lynx,t_vole,t_beetle,p_private"
REPLY_OUT = "research/findings/raw/_idle_replay/dev"
ORGAN_OUT = "research/findings/raw/_idle_replay/organ"


def organ_for(seed: int, chain: str, params: dict) -> IR.IdleReplayOrgan:
    o = IR.IdleReplayOrgan(seed=seed, params=params)
    for i, f in enumerate(BUILD):
        o.note_encoding(f, i, 0)
    for j, f in enumerate(CHAINS[chain]):
        o.note_encoding(f, len(BUILD) + j, j + 1)
    o.turn = len(CHAINS[chain]) + 1
    return o


def _draw(o: IR.IdleReplayOrgan, cands: List[dict], noise_seed: int, **kw) -> dict:
    o.noise_seed = (lambda K, ns=noise_seed: ns)          # the draw's background-noise seed
    r = o.select(cands, **kw)
    return {"noise_seed": int(noise_seed), "facts": [c["fact"] for c in cands], "x": r.get("x_applied"),
            "slot_counts": r.get("slot_counts"), "first_gate_step": r.get("first_gate_step"),
            "engram_counts": r.get("engram_counts")}


def run_seed(seed: int, op: int = 0) -> dict:
    """Every draw of the organ battery for one dev seed at REPLAY_GRID[op]."""
    p = dict(IR.REPLAY_GRID[int(op)], OP=int(op))
    t0 = time.time()
    oi, op_ = organ_for(seed, "intact", p), organ_for(seed, "permuted", p)
    ci, cp = oi.candidates(), op_.candidates()
    ct = [dict(c) for c in ci]
    for c in ct:                                           # near tie: the second-newest fact encoded twice as strongly
        if tuple(c["fact"]) == CHAINS["intact"][1]:
            c["s"] = 2.0
            c["x"] = IR.excitability(c["age"], 2.0, p)
    rec: Dict[str, list] = {k: [] for k in ("intact", "permuted", "tag_lesion", "no_encode", "commit_lesion",
                                            "near_tie")}
    for n in range(N_DRAWS):
        ns = 100003 * int(seed) + 7 * n + 1
        rec["intact"].append(_draw(oi, ci, ns))
        rec["permuted"].append(_draw(op_, cp, ns))
        rec["tag_lesion"].append(_draw(oi, ci, ns, tag_lesion=True))
        rec["near_tie"].append(_draw(oi, ct, ns))
        if n < N_LESION_DRAWS:
            rec["no_encode"].append(_draw(oi, ci, ns, encoded=False))
            rec["commit_lesion"].append(_draw(oi, ci, ns, commit=False))
    try:
        from sim.backend import get_backend
        backend = get_backend()[0].__name__
    except Exception:  # noqa: BLE001
        backend = "unknown"
    out = {"runner": "research/runners/idle_replay_battery.py", "seed": int(seed), "op": int(op), "params": p,
           "backend": backend, "draws": rec, "wall_s": round(time.time() - t0, 1)}
    out["verdict"] = seed_verdict(out)
    c = out["verdict"]["counts"]
    # attribution (tools.lab, gate class AT): whose is the newest-wins rate? The tag lesion removes only the
    # excitability bias (engrams, competition and read intact), so the drop is the tag's share of the selection.
    out["attribution"] = {"tag_lesion_moved_newest_wins": bool(lever(
        "excitability tag -> newest wins (of %d)" % c["n_draws"], c["recent_intact"], c["tag_newest"], required=False))}
    return out


def _winner(draw: dict, p: dict) -> dict:
    """The organ's own latch read on the recorded counts -> the reactivated fact (or None) + invariance."""
    rd = SC.read_with_variants(draw["slot_counts"], int(p["NS"]), int(p["READ_STEPS"]), float(p["READ_FRAC"]))
    w = rd["winner"]
    return {"fact": (tuple(draw["facts"][w]) if w is not None else None), "invariant": bool(rd["read_invariant"]),
            "co_commit": bool(rd["co_commit"])}


def seed_verdict(rec: dict) -> dict:
    """Pure: per-case pass from the recorded latch counts (see the module docstring)."""
    p, d = rec["params"], rec["draws"]
    W = {k: [_winner(x, p) for x in v] for k, v in d.items()}
    newest_i, newest_p = CHAINS["intact"][-1], CHAINS["permuted"][-1]
    n_ri = sum(1 for w in W["intact"] if w["fact"] == newest_i)
    n_rp = sum(1 for w in W["permuted"] if w["fact"] == newest_p)
    n_diff = sum(1 for a, b in zip(W["intact"], W["permuted"])
                 if a["fact"] is not None and b["fact"] is not None and a["fact"] != b["fact"])
    n_tag = sum(1 for w in W["tag_lesion"] if w["fact"] == newest_i)
    gated = [k for k in d if k != "near_tie"]
    cases = {
        "recent_intact": n_ri >= MIN_RECENT,
        "recent_permuted": n_rp >= MIN_RECENT,
        "differ": n_diff >= MIN_RECENT,
        "no_encode": len(W["no_encode"]) > 0 and all(w["fact"] is None for w in W["no_encode"]),
        "commit_lesion": len(W["commit_lesion"]) > 0 and all(w["fact"] is None for w in W["commit_lesion"]),
        "tag_lesion_moves": len(W["tag_lesion"]) > 0 and n_tag <= MAX_TAG_RECENT and n_tag <= n_ri - TAG_DROP,
        "read_invariant": all(w["invariant"] for k in gated for w in W[k]),
    }
    tie = {}
    for w in W["near_tie"]:
        key = " ".join(w["fact"]) if w["fact"] else "-"
        tie[key] = tie.get(key, 0) + 1
    lat = [min([t for t in (x.get("first_gate_step") or []) if t is not None], default=None)
           for x in d.get("intact", [])]
    return {"cases": cases, "all_pass": bool(all(cases.values())),
            "counts": {"recent_intact": n_ri, "recent_permuted": n_rp, "differ": n_diff, "tag_newest": n_tag,
                       "n_draws": len(W["intact"])},
            "tag_lesion_winners": [(" ".join(w["fact"]) if w["fact"] else "-") for w in W["tag_lesion"]],
            "near_tie": tie, "first_ignition_ms": lat}


def dev_verdict(per_seed: Dict[int, dict], seeds=DEV_SEEDS) -> dict:
    """PASS iff every dev seed is present and all_pass (recomputed from its counts). Missing seed -> UNDEFINED."""
    missing = [s for s in seeds if s not in per_seed]
    if missing:
        return {"verdict": "UNDEFINED", "missing": missing}
    vs = {s: seed_verdict(per_seed[s]) for s in seeds}
    n = sum(1 for v in vs.values() if v["all_pass"])
    failing = {s: [c for c, ok in v["cases"].items() if not ok] for s, v in vs.items() if not v["all_pass"]}
    return {"verdict": "PASS" if n == len(seeds) else "FAIL", "n_pass": n, "n_seeds": len(seeds), "failing": failing}


def score_dir(d: str, op: int) -> dict:
    per = {}
    for f in sorted(glob.glob(os.path.join(d, "organ_s*_op%d.json" % int(op)))):
        with open(f) as fh:
            r = json.load(fh)
        per[int(r["seed"])] = r
    return dev_verdict(per)


# ── reply level (the Stage-2 runner, at a tip that contains it) ─────────────────────────────────────────────────
def dev_lines(sha: str, seeds=DEV_SEEDS, out_dir: str = REPLY_OUT) -> List[str]:
    """Pool lines: per dev seed, (intact_a, permuted, lesion_selfinit) flag ON and intact_a flag OFF, the teach turns +
    the idle turn only; plus dev seed 7 intact_a with every turn, flag ON and flag OFF (the only allowed difference is
    p_private)."""
    base = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 9 1 && mkdir -p %s && bash tools/memcap.sh 10 -- "
            "env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            "CUDA_VISIBLE_DEVICES= BRAIN_IDLE_REPLAY=%s .venv/bin/python -u -m research.runners.stage2_exit_gate "
            "--shard --seed %d --group inner --arm %s --out %s%s")
    why = ("idle-replay C5 build dev reply check (docs/drafts/idle-replay/HANDOFF.md); seed %d %s flag %s; "
           "mem_gb=9 max_age_h=168")
    out = []
    for s in seeds:
        sub = "%s/s%d" % (out_dir, s)
        for arm, flag in [(a, "1") for a in REPLY_ARMS] + [("intact_a", "0")]:
            path = "%s/inner__%s%s.json" % (sub, arm, "" if flag == "1" else "__off")
            out.append(base % (sha, sub, flag, s, arm, path, " --labels " + REPLY_LABELS) + "\t" + why % (s, arm, flag))
    sub = "%s/s7_full" % out_dir
    for flag in ("1", "0"):
        path = "%s/inner__intact_a__flag%s.json" % (sub, flag)
        out.append(base % (sha, sub, flag, 7, "intact_a", path, "") + "\t" + why % (7, "intact_a full", flag))
    return out


def organ_lines(sha: str, seeds=DEV_SEEDS, op: int = 0, out_dir: str = ORGAN_OUT) -> List[str]:
    """Pool lines for the organ battery (no brain; seconds per seed)."""
    out = []
    for s in seeds:
        path = "%s/organ_s%d_op%d.json" % (out_dir, s, op)
        out.append("cd ~/derisk-pool/revisions/%s && mkdir -p %s && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 "
                   "CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.idle_replay_battery --seed %d --op %d "
                   "--out %s\tidle-replay C5 build organ battery seed %d op %d (docs/drafts/idle-replay/HANDOFF.md); "
                   "mem_gb=1 max_age_h=168" % (sha, out_dir, s, op, path, s, op))
    return out


def score_reply(d: str, seeds=DEV_SEEDS) -> dict:
    """Score the reply-level dev shards with the Stage-2 runner's own C5 functions (lazy import: the runner lives on
    the gate branch). Per seed: private_voices_stored_fact (intact and permuted), private_selected_by_state, the
    lesion_selfinit idle reply is the neutral idle line. Plus the seed-7 flag on/off comparison when present."""
    from research.runners import stage2_exit_gate as S2          # present only at a tip that contains the gate
    from research.runners import self_initiated_production_organ as SI
    rows, ok_all = {}, True
    for s in seeds:
        sub = os.path.join(d, "s%d" % s)
        sh = {}
        for arm in REPLY_ARMS:
            f = os.path.join(sub, "inner__%s.json" % arm)
            if os.path.exists(f):
                with open(f) as fh:
                    sh[arm] = json.load(fh)
        if len(sh) < len(REPLY_ARMS):
            rows[s] = {"verdict": "UNDEFINED", "have": sorted(sh)}
            ok_all = False
            continue
        si = S2.score_inner(sh["intact_a"])
        sp = S2.score_inner(sh["permuted"], permuted=True)
        les = ((sh["lesion_selfinit"].get("responses") or {}).get("p_private") or {}).get("answer")
        trace = (((sh["intact_a"].get("responses") or {}).get("p_private") or {}).get("self_initiated") or {}
                 ).get("idle_replay") or {}
        row = {"intact_utt": si["private_utterance"], "permuted_utt": sp["private_utterance"],
               "stored_intact": si["private_voices_stored_fact"], "stored_permuted": sp["private_voices_stored_fact"],
               "selected_by_state": S2.private_selected_by_state(si, sp),
               "lesion_neutral": les == SI.idle_fallback_text(),
               "route": trace.get("route"), "read_invariant": ((trace.get("selection") or {}).get("read") or {}
                                                              ).get("read_invariant")}
        row["pass"] = bool(row["stored_intact"] and row["stored_permuted"] and row["selected_by_state"]
                           and row["lesion_neutral"] and row["route"] == "idle_replay")
        ok_all = ok_all and row["pass"]
        rows[s] = row
    full = {}
    fo, ff = (os.path.join(d, "s7_full", "inner__intact_a__flag%s.json" % x) for x in ("1", "0"))
    if os.path.exists(fo) and os.path.exists(ff):
        with open(fo) as a, open(ff) as b:
            ra, rb = json.load(a)["responses"], json.load(b)["responses"]
        keys = sorted(set(ra) | set(rb))
        full = {"differs": [k for k in keys if json.dumps(ra.get(k), sort_keys=True, default=str)
                            != json.dumps(rb.get(k), sort_keys=True, default=str)]}
        full["only_idle_turn_differs"] = full["differs"] == ["p_private"]
    ar = {}
    try:                                                  # RULING flip-hurts-answers: the idle turn, flag on vs off
        sys.path.insert(0, os.path.join(_REPO, "tools"))
        import answer_rate as AR
        per = {}
        for s in seeds:
            on_f = os.path.join(d, "s%d" % s, "inner__intact_a.json")
            off_f = os.path.join(d, "s%d" % s, "inner__intact_a__off.json")
            per[s] = {k: [((json.load(open(f)).get("responses") or {}).get("p_private"))]
                      for k, f in (("on", on_f), ("off", off_f)) if os.path.exists(f)}
        ar = AR.report(per, label="idle-replay p_private (say something)")
    except Exception as e:  # noqa: BLE001
        ar = {"error": "%s: %s" % (type(e).__name__, e)}
    n = sum(1 for r in rows.values() if r.get("pass"))
    return {"answer_rate": ar, "verdict": "PASS" if ok_all and n == len(seeds) else ("UNDEFINED" if any(
        r.get("verdict") == "UNDEFINED" for r in rows.values()) else "FAIL"),
            "n_pass": n, "rows": rows, "s7_flag_on_vs_off": full}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--op", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--score")
    ap.add_argument("--emit-dev-lines", metavar="SHA")
    ap.add_argument("--emit-organ-lines", metavar="SHA")
    ap.add_argument("--score-reply", metavar="DIR")
    a = ap.parse_args(argv)
    if a.emit_dev_lines:
        print("\n".join(dev_lines(a.emit_dev_lines)))
        return 0
    if a.emit_organ_lines:
        print("\n".join(organ_lines(a.emit_organ_lines, op=a.op)))
        return 0
    if a.score:
        v = score_dir(a.score, a.op)
        print(json.dumps(v, indent=1))
        return 0 if v["verdict"] == "PASS" else 1
    if a.score_reply:
        v = score_reply(a.score_reply)
        print(json.dumps(v, indent=1, default=str))
        return 0 if v["verdict"] == "PASS" else 1
    if a.seed is None or not a.out:
        ap.error("--seed and --out (or --score / --emit-dev-lines / --score-reply)")
    if a.seed in IR.GATE_SEEDS:
        ap.error("gate seeds are not run by the dev battery")
    os.environ.setdefault("SIM_BACKEND", "numpy")
    r = run_seed(a.seed, a.op)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(r, fh, indent=1, default=str)
    print("[idle-replay battery] seed %d op %d all_pass %s %s (%.1fs) -> %s" % (
        a.seed, a.op, r["verdict"]["all_pass"], r["verdict"]["counts"], r["wall_s"], a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
