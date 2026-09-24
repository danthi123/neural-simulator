"""OPEN-ENDED GATED TURN -- Part A capability gate (the a3 successor on the gated turn).

Pre-registration (committed b4588775a BEFORE any run it governs):
research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md. The 2026-09-23 a3 NO-GO on the UNGATED turn
stands and is not re-scored here.

WHAT IT RUNS. One SESSION per process (a fresh production brain), reusing `_lbf_open_ended_production_turn_probe`'s
session worker UNCHANGED (same TEACH world, same ASK "what might a dog chase", same per-session OU noise-stream
installer, same draw counters), with mode `oe_gated` = BRAIN_OPEN_ENDED_GATED=1 and the a3 arms: intact
(BRAIN_SPIKING_DRAW_LESION=0), lesion (=1, the HOST weight-vector lesion), intact_rebuild (intact session 0's stream).
M = 3 sessions per arm per seed (>= 3 OU sub-streams), K = 8 asks per session.

WHAT IT ADDS (instrument only; transparent wrappers, same return values):
  * a recorder on SpikingWTASampler.draw_from_weights / _compete that keeps, per brain_chat turn, every draw's
    candidate list, every competition's firing vector fv, and the draw's winner;
  * a per-turn copy of the gated turn's decision fields (route / bg_action / reply_kind).

METRICS (PREREG Part A).
  CAT  -- the a3 categorical metric, scored by `score_seed_a3` UNCHANGED (m = 3): session value = mean w(reply)/peak.
  CONT -- per ask, the draw whose winner is the volunteered patient (the last such draw of the ask), its last
          non-silent fv: value = sum_p (fv_p / sum fv) * w_ref[p] / peak; ABSTAIN / hold -> 0. Session = mean over asks.
  GO   -- all 6 seeds DEFINED on CONT AND Delta_cont >= 0.10 on EVERY seed. CAT reported beside, with its own verdict.

HOST SHORTCUTS: those of the a3 probe (P.HOST_SHORTCUTS) plus the gated turn's declared residuals (the read->salience
transduction, the route label). The statistic and the recorder are INSTRUMENT.

Usage:
  one session (a pool job):  python -m research.runners._open_ended_gated_turn_gate --session --seed 42 --arm intact --j 0
  score:                     python -m research.runners._open_ended_gated_turn_gate --score --seeds 42,43,44,100,101,102
  job lines:                 python -m research.runners._open_ended_gated_turn_gate --jobs --root '~/derisk-pool/revisions/<sha>'
  selftest (no brain):       python -m research.runners._open_ended_gated_turn_gate --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.lab import attributable_to  # noqa: E402
from research.runners import _lbf_open_ended_production_turn_probe as P  # noqa: E402

MODE = "oe_gated"
MODE_ENV = {"BRAIN_OPEN_ENDED_GATED": "1"}
SEEDS = (42, 43, 44, 100, 101, 102)
M = 3
K = 8
FLOOR = 0.10
ARMS = ("intact", "lesion", "intact_rebuild")
OUT_DIR = "research/findings/raw/_open_ended_gated/partA"
PREREG = "research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md"
N_TEACH = len(P.TEACH)


def session_path(out_dir, seed, arm, j):
    return os.path.join(out_dir, "%s_s%s_%s_n%d.json" % (MODE, seed, arm, int(j)))


def _install_recorder():
    """Transparent wrappers (same return values) recording every draw per brain_chat turn."""
    import numpy as np
    import research.runners._followon2_spiking_wta_sampler_derisk as F2
    import webapp.server as S
    rec = {"turn": -1, "draws": [], "oeg": []}
    cur = {"draw": None}
    orig_dfw = F2.SpikingWTASampler.draw_from_weights
    orig_comp = F2.SpikingWTASampler._compete

    def _dfw(self, weights, candidates, max_retries=3):
        d = {"turn": rec["turn"], "cands": [str(c) for c in candidates], "fvs": [], "winner": None}
        cur["draw"] = d
        try:
            w = orig_dfw(self, weights, candidates, max_retries=max_retries)
        finally:
            cur["draw"] = None
        d["winner"] = (str(w) if w is not None else None)
        rec["draws"].append(d)
        return w

    def _comp(self, drive, V):
        fv = orig_comp(self, drive, V)
        if cur["draw"] is not None:
            cur["draw"]["fvs"].append([float(x) for x in np.asarray(fv).ravel().tolist()])
        return fv

    F2.SpikingWTASampler.draw_from_weights = _dfw
    F2.SpikingWTASampler._compete = _comp
    orig_bc = S.brain_chat

    def _bc(req):
        rec["turn"] += 1
        r = orig_bc(req)
        try:
            b = json.loads(r.body)
            g = b.get("open_ended_gated") or {}
            rec["oeg"].append({"turn": rec["turn"], "route": g.get("route"), "bg_action": g.get("bg_action"),
                               "reply_kind": g.get("reply_kind"), "present": bool(g)})
        except Exception as e:
            rec["oeg"].append({"turn": rec["turn"], "error": "%s: %s" % (type(e).__name__, e)})
        return r

    S.brain_chat = _bc
    return rec


def run_session(seed, arm, j, k=K, out_dir=OUT_DIR):
    """ONE session in THIS process. Idempotent: an existing valid output is kept."""
    path = session_path(out_dir, seed, arm, j)
    if os.path.exists(path) and P._load(path) is not None:
        print("[oeg gate] exists, skipping: %s" % path, flush=True)
        return 0
    env = dict(MODE_ENV)
    env.update(P.ARMS[arm])
    # the SAME env the worker sets, applied BEFORE webapp.server is imported by the recorder (several organs read
    # BRAIN_CHAT_SEED at import time); the worker re-applies it identically.
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    for kk, vv in env.items():
        os.environ[kk] = vv
    rec = _install_recorder()
    rc = P._worker(env, seed, k, path, rich=True, teach_env=None, noise_seed=P.a3_noise_seed(seed, arm, j))
    payload = P._load(path)
    if payload is None:
        return rc or 1
    ask_turns = list(range(N_TEACH, N_TEACH + int(k)))
    payload["cont_draws"] = {str(t): [d for d in rec["draws"] if d["turn"] == t] for t in ask_turns}
    payload["gated_trace"] = [o for o in rec["oeg"] if o.get("turn", -1) in ask_turns]
    payload["gate_mode"] = MODE
    payload["prereg"] = PREREG
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return rc


# ── scoring (pure) ────────────────────────────────────────────────────────────────────────────────────────────────
def ask_cont_value(draws, outcome, w_ref, peak):
    """CONT for one ask. None = UNDEFINED (a volunteered hypothesis with no recorded, non-silent competition)."""
    if outcome == "ERROR":
        return None
    if outcome == "ABSTAIN":
        return 0.0
    hits = [d for d in (draws or []) if d.get("winner") == outcome]
    if not hits or peak <= 0:
        return None
    d = hits[-1]
    fvs = [fv for fv in d.get("fvs") or [] if sum(fv) > 0]
    if not fvs:
        return None
    fv = fvs[-1]
    tot = float(sum(fv))
    cands = d.get("cands") or []
    if len(cands) != len(fv):
        return None
    return sum((fv[i] / tot) * float(w_ref.get(cands[i], 0.0) or 0.0) for i in range(len(fv))) / peak


def session_cont(payload, w_ref):
    """Per-ask CONT values and their mean; None when any ask is UNDEFINED."""
    reps = (payload or {}).get("replies") or []
    peak = max([x for x in (w_ref or {}).values() if isinstance(x, (int, float))] or [0.0])
    draws = (payload or {}).get("cont_draws") or {}
    if not reps:
        return None, []
    vals = []
    for i, r in enumerate(reps):
        o = P.outcome(r)
        vals.append(ask_cont_value(draws.get(str(N_TEACH + i)), o, w_ref or {}, peak))
    if any(v is None for v in vals):
        return None, vals
    return sum(vals) / len(vals), vals


def score_seed(intact, lesion, rebuild, m=M, floor=FLOOR):
    cat = P.score_seed_a3(intact, lesion, rebuild, m=m)
    res = {"CAT": cat, "verdict": None, "reasons": [], "m": m}
    if cat.get("verdict") == "ARM-FAILED":
        res["verdict"] = "ARM-FAILED"
        res["reasons"] = list(cat.get("reasons") or [])
        return res
    w_ref = intact[0].get("likelihood_weight") or {}
    ci = [session_cont(p, w_ref) for p in intact]
    cl = [session_cont(p, w_ref) for p in lesion]
    cr = session_cont(rebuild, w_ref)
    vi, vl = [c[0] for c in ci], [c[0] for c in cl]
    res.update({"v_intact": vi, "v_lesion": vl, "per_ask_intact_s0": ci[0][1], "per_ask_rebuild": cr[1]})
    abl_i = [p["draw_counter"]["n_ablated_calls"] for p in intact]
    abl_l = [p["draw_counter"]["n_ablated_calls"] for p in lesion]
    facts_eq = all(p.get("stored_facts") == intact[0].get("stored_facts") for p in intact + lesion + [rebuild])
    checks = [
        (any(v is None for v in vi + vl), "a session has a hypothesis ask with no recorded non-silent competition"),
        (len(set(vi)) == 1 and len(set(vl)) == 1, "degenerate null: every intact and every lesion session identical"),
        (cr[1] != ci[0][1], "the rebuild's per-ask CONT values differ from intact session 0's"),
        (any(a == 0 for a in abl_l) or any(a != 0 for a in abl_i), "the lesion was not applied (ablated counts)"),
        (not facts_eq, "stored facts differ across sessions"),
    ]
    for bad, why in checks:
        if bad:
            res["reasons"].append(why)
    if res["reasons"]:
        res["verdict"] = "UNDEFINED"
        return res
    res["verdict"] = "DEFINED"
    res["delta"] = sum(vi) / m - sum(vl) / m
    res["passes_floor"] = bool(res["delta"] >= floor)
    res["attributable_to_host_weight_drive"] = attributable_to(
        "oe_gated CONT intact vs lesion mean session value (per seed)", sum(vi) / m, sum(vl) / m)
    res["perm_p_exact_one_sided"] = P.exact_perm_p(vi, vl)
    return res


def aggregate(per_seed, floor=FLOOR, n_required=len(SEEDS)):
    seeds = sorted(per_seed, key=int)
    defined = [s for s in seeds if per_seed[s].get("verdict") == "DEFINED"]
    ds = [per_seed[s]["delta"] for s in defined]
    all_defined = len(defined) == len(seeds) == n_required
    all_pass = all_defined and all(per_seed[s]["passes_floor"] for s in defined)
    cat = P.aggregate_a3({s: per_seed[s]["CAT"] for s in seeds}, delta_floor=floor, n_required=n_required)
    return {"n_seeds": len(seeds), "n_defined": len(defined), "all_defined": all_defined,
            "per_seed_verdict": {s: per_seed[s].get("verdict") for s in seeds},
            "per_seed_reasons": {s: per_seed[s].get("reasons") for s in seeds},
            "delta_cont": {s: per_seed[s].get("delta") for s in seeds}, "floor": floor,
            "p_sign_test_cont": (P.sign_test_p(ds) if all_defined else None),
            "GO": bool(all_pass), "CAT_beside": cat}


def score(seeds, out_dir=OUT_DIR, aggregate_out=None, m=M):
    per_seed = {}
    for s in seeds:
        intact = [P._load(session_path(out_dir, s, "intact", j)) for j in range(m)]
        lesion = [P._load(session_path(out_dir, s, "lesion", j)) for j in range(m)]
        rebuild = P._load(session_path(out_dir, s, "intact_rebuild", 0))
        if any(x is None for x in intact + lesion + [rebuild]):
            per_seed[str(s)] = {"verdict": "ARM-FAILED", "reasons": ["missing session file(s)"], "CAT": {}}
            continue
        per_seed[str(s)] = score_seed(intact, lesion, rebuild, m=m)
        print("[oeg gate] seed=%s CONT %s delta=%s | CAT %s delta=%s" % (
            s, per_seed[str(s)]["verdict"], per_seed[str(s)].get("delta"),
            per_seed[str(s)]["CAT"].get("verdict"), per_seed[str(s)]["CAT"].get("delta")), flush=True)
    agg = aggregate(per_seed, n_required=len(seeds) if len(seeds) != len(SEEDS) else len(SEEDS))
    out = {"runner": "research.runners._open_ended_gated_turn_gate --score", "prereg": PREREG, "mode": MODE,
           "M": m, "K": K, "floor": FLOOR, "lesioned_edge": P.LESIONED_EDGE,
           "host_shortcuts": P.HOST_SHORTCUTS + ["gated turn: read->salience transduction; route label (host)"],
           "per_seed": per_seed, "summary": agg}
    if aggregate_out:
        os.makedirs(os.path.dirname(os.path.abspath(aggregate_out)), exist_ok=True)
        with open(aggregate_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
    print(json.dumps(agg, indent=2, default=str))
    return agg


def job_lines(root, seeds=SEEDS, mem_gb=7):
    lines = []
    for s in seeds:
        for arm in ARMS:
            for j in (range(M) if arm != "intact_rebuild" else [0]):
                lines.append("cd %s && SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m "
                             "research.runners._open_ended_gated_turn_gate --session --seed %d --arm %s --j %d"
                             "   # mem_gb=%s" % (root, s, arm, j, mem_gb))
    return lines


# ── selftest (pure; each check can fail) ────────────────────────────────────────────────────────────────────────
def selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    w = {"rabbit": 4.0, "deer": 3.0, "mouse": 2.0, "beetle": 1.0}
    cands = ["rabbit", "deer", "mouse", "beetle"]
    d_peak = {"cands": cands, "fvs": [[0, 0, 0, 0], [8, 1, 1, 0]], "winner": "rabbit"}
    v = ask_cont_value([d_peak], "rabbit", w, 4.0)
    chk("CONT: last non-silent fv, share-weighted w/peak", abs(v - (0.8 * 1 + 0.1 * 0.75 + 0.1 * 0.5)) < 1e-12)
    chk("CONT: ABSTAIN -> 0", ask_cont_value([], "ABSTAIN", w, 4.0) == 0.0)
    chk("CONT: volunteered with no recorded draw -> None", ask_cont_value([], "deer", w, 4.0) is None)
    chk("CONT: all-silent competition -> None", ask_cont_value(
        [{"cands": cands, "fvs": [[0, 0, 0, 0]], "winner": "deer"}], "deer", w, 4.0) is None)

    def sess(outs, fv_rows, abl, ns):
        reps = [{"hypothesis_svo": (["dog", "chase", o] if o != "ABSTAIN" else None)} for o in outs]
        draws = {str(N_TEACH + i): ([{"cands": cands, "fvs": [fv_rows[i]], "winner": outs[i]}]
                                    if outs[i] != "ABSTAIN" else []) for i in range(len(outs))}
        return {"replies": reps, "cont_draws": draws, "likelihood_weight": dict(w), "stored_facts": [["a", "b", "c"]],
                "draw_counter": {"n_calls": 8, "n_ablated_calls": abl}, "noise_seed": ns, "noise_stream_competes": 8}

    peaky = [9, 1, 0, 0]
    flat = [3, 3, 2, 2]
    pk = [[9, 1, 0, 0], [8, 2, 0, 0], [7, 2, 1, 0]]          # the noise stream varies the firing, session to session
    fl = [[3, 3, 2, 2], [2, 3, 3, 2], [2, 2, 3, 3]]
    I = [sess(["rabbit"] * 7 + [o], [pk[j]] * 8, 0, 42000 + j) for j, o in enumerate(["deer", "rabbit", "mouse"])]
    L = [sess(["beetle"] * 6 + [o, "deer"], [fl[j]] * 8, 8, 42500 + j)
         for j, o in enumerate(["mouse", "rabbit", "beetle"])]
    R = sess(["rabbit"] * 7 + ["deer"], [pk[0]] * 8, 0, 42000)
    s = score_seed(I, L, R)
    chk("seed: peaky intact vs flat lesion -> DEFINED, delta >= floor", s["verdict"] == "DEFINED" and s["passes_floor"])
    I2 = [sess(["rabbit"] * 8, [peaky] * 8, 0, 42000 + j) for j in range(M)]
    L2 = [sess(["rabbit"] * 8, [flat] * 8, 8, 42500 + j) for j in range(M)]
    s2 = score_seed(I2, L2, sess(["rabbit"] * 8, [peaky] * 8, 0, 42000))
    chk("seed: every session identical within both arms -> UNDEFINED (degenerate null)",
        s2["verdict"] == "UNDEFINED" and any("degenerate" in r for r in s2["reasons"]))
    s3 = score_seed(I, L, sess(["rabbit"] * 8, [flat] * 8, 0, 42000))
    chk("seed: rebuild differs -> not DEFINED", s3["verdict"] != "DEFINED")
    Lbad = [sess(["beetle"] * 8, [flat] * 8, 0, 42500 + j) for j in range(M)]
    chk("seed: lesion not applied -> not DEFINED", score_seed(I, Lbad, R)["verdict"] != "DEFINED")
    rec = lambda d, p=True: {"verdict": "DEFINED", "delta": d, "passes_floor": p, "CAT": {"verdict": "DEFINED",
                                                                                          "delta": d}}
    six = {str(x): rec(0.3) for x in SEEDS}
    chk("aggregate: 6 DEFINED all >= floor -> GO", aggregate(six)["GO"])
    chk("aggregate: one seed under the floor -> no GO", not aggregate(dict(six, **{"44": rec(0.05, False)}))["GO"])
    chk("aggregate: one UNDEFINED seed -> no GO", not aggregate(dict(six, **{"100": {"verdict": "UNDEFINED",
                                                                                     "CAT": {"verdict": "UNDEFINED"}}}))["GO"])
    chk("aggregate: 5 seeds -> no GO", not aggregate({k: v for k, v in six.items() if k != "102"})["GO"])
    chk("aggregate: CAT reported beside", "CAT_beside" in aggregate(six))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--jobs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", choices=ARMS, default="intact")
    ap.add_argument("--j", type=int, default=0)
    ap.add_argument("--k", type=int, default=K, help="asks per session (the governed value is 8; smoke only)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--aggregate-out", default=None)
    ap.add_argument("--root", default="~/derisk-pool/revisions/<sha>")
    ap.add_argument("--mem-gb", default="7")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    if a.jobs:
        print("\n".join(job_lines(a.root, mem_gb=a.mem_gb)))
        return 0
    if a.session:
        if a.seed in SEEDS and a.k != K:
            raise SystemExit("REFUSED: a gate seed must run the pre-registered K=%d" % K)
        return run_session(a.seed, a.arm, a.j, k=a.k, out_dir=a.out_dir)
    if a.score:
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        score(seeds, out_dir=a.out_dir, aggregate_out=a.aggregate_out
              or os.path.join(a.out_dir, "%s_aggregate.json" % MODE))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
