"""OPEN-ENDED GATED TURN -- dev-seed smoke of the conditioning state + the flag-off identity compare.

NOT governed by the pre-registration (research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md): seed 7
is a development seed and the identity check is an integrity check. Gate seeds 42/43/44/100/101/102 are refused here.

ARMS (one fresh production brain per process, through the REAL webapp.server.brain_chat via the regression battery's
own worker; BRAIN_CHAT_SEED = the dev seed; numpy, stub renderer, LLM disabled):
  intact_a / intact_b  BRAIN_OPEN_ENDED_GATED=1 (intact_b is the independent rebuild: the null control)
  gate_lesion          + BRAIN_OPEN_ENDED_GATE_LESION=1   (the row lesion: afferent cut into the BG race + marker WTA)
  affect_lesion        + BRAIN_AFFECT_LESION=1            (Gate-B affect_out gate closed)
  gnw_lesion           + BRAIN_GNW_2ORGAN_WS_LESION=1     (GNW workspace self-recurrence zeroed)
Turns: unknown (off-KB), emo (off-KB, strongly affective), sw_open (KB-hit recall), open (a generation prompt,
single-fact), rich_open (the same prompt on the rich path).

SCORE: per arm and turn, the gated turn's CONDITIONING STATE (route, familiarity_band, valence_sign, marker_level,
bg_action, reply_kind, spoke) and the reply; which fields change vs intact_a; whether intact_a == intact_b (clean null);
and the three LBF rows scored with `lbf_rows.open_ended_gated.score_row` (which asks tools.lab.attributable_to).

IDENTITY (flag OFF): run the regression battery worker over the FULL probe roster with the flag unset at two
revisions (this branch's head and its merge-base with main), then `--compare-identity A B` checks exact JSON equality.

Usage (pool2 check slot, memcap >= 7 GB):
  python -m research.runners._open_ended_gated_turn_smoke --arm intact_a --seed 7
  python -m research.runners._open_ended_gated_turn_smoke --score --seed 7
  python -m research.runners._open_ended_gated_turn_smoke --compare-identity A.json B.json --out C.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.lab import attributable_to  # noqa: E402

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
OUT_DIR = "research/findings/raw/_open_ended_gated/smoke"
SMOKE_TURNS = ["unknown", "emo", "sw_open", "open", "rich_open"]
BASE = {"BRAIN_OPEN_ENDED_GATED": "1"}
ARMS = {
    "intact_a": {},
    "intact_b": {},
    "gate_lesion": {"BRAIN_OPEN_ENDED_GATE_LESION": "1"},
    "affect_lesion": {"BRAIN_AFFECT_LESION": "1"},
    "gnw_lesion": {"BRAIN_GNW_2ORGAN_WS_LESION": "1"},
}
COND_FIELDS = ("route", "familiarity_band", "valence_sign", "marker_level", "bg_action", "reply_kind", "spoke")
ROW_FOR_ARM = {"gate_lesion": "open-ended-turn-faculty-drive", "affect_lesion": "open-ended-turn-affect-drive",
               "gnw_lesion": "open-ended-turn-gnw-drive"}


def arm_path(out_dir, seed, arm):
    return os.path.join(out_dir, "s%d" % int(seed), "%s.json" % arm)


def run_arm(arm, seed, out_dir=OUT_DIR, turns=None):
    if int(seed) in GATE_SEEDS:
        raise SystemExit("REFUSED: seed %s is a gate seed; this smoke is dev-seed only" % seed)
    from research.runners.onebrain_regression_battery import _collect_worker
    env = dict(BASE)
    env.update(ARMS[arm])
    env["BRAIN_CHAT_SEED"] = str(int(seed))
    return _collect_worker(json.dumps(env), list(turns or SMOKE_TURNS), arm_path(out_dir, seed, arm))


def _cond(resp):
    g = (resp or {}).get("open_ended_gated")
    if not isinstance(g, dict):
        return None
    c = {k: g.get(k) for k in COND_FIELDS}
    c["salience"] = [g.get("salience_speak"), g.get("salience_silent")]
    c["cut"] = g.get("cut")
    c["answer"] = (resp or {}).get("answer")
    c["abstained"] = (resp or {}).get("abstained")
    return c


def score(seed, out_dir=OUT_DIR):
    from research.runners.lbf_rows.open_ended_gated import score_row
    arms = {}
    for a in ARMS:
        p = arm_path(out_dir, seed, a)
        arms[a] = json.load(open(p)) if os.path.exists(p) else None
    out = {"seed": int(seed), "turns": SMOKE_TURNS, "governed": False,
           "prereg": "research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md",
           "arms_present": {a: arms[a] is not None for a in arms}, "conditioning": {}, "changes_vs_intact": {},
           "null_clean": None, "rows": {}}
    ia, ib = arms.get("intact_a") or {}, arms.get("intact_b") or {}
    for t in SMOKE_TURNS:
        out["conditioning"][t] = {a: _cond((arms[a] or {}).get(t)) for a in arms}
    # the null: intact_a vs intact_b, every conditioning field + the reply, every turn
    null_diffs = []
    for t in SMOKE_TURNS:
        ca, cb = _cond(ia.get(t)), _cond(ib.get(t))
        if ca != cb:
            null_diffs.append({"turn": t, "a": ca, "b": cb})
    out["null_clean"] = (not null_diffs) if (arms["intact_a"] and arms["intact_b"]) else None
    out["null_diffs"] = null_diffs
    for a in ("gate_lesion", "affect_lesion", "gnw_lesion"):
        if arms[a] is None:
            continue
        ch = {}
        for t in SMOKE_TURNS:
            ci, cl = _cond(ia.get(t)), _cond((arms[a] or {}).get(t))
            if ci is None and cl is None:
                continue                                   # turn not run / trace absent in both arms: no change
            if ci is None or cl is None:
                ch[t] = {"present": [ci is not None, cl is not None]}
                continue
            diff = {k: [ci.get(k), cl.get(k)] for k in COND_FIELDS + ("answer",) if ci.get(k) != cl.get(k)}
            if diff:
                ch[t] = diff
        out["changes_vs_intact"][a] = ch
        # the attribution question, asked out loud at the conditioning level: changed (turn, field) pairs under the
        # lesion (treatment) vs under the intact rebuild (control, the null)
        n_treat = sum(max(1, len([k for k in v if k != "present"])) for v in ch.values())
        n_ctrl = sum(len([k for k in COND_FIELDS + ("answer",) if (d["a"] or {}).get(k) != (d["b"] or {}).get(k)])
                     for d in null_diffs)
        out.setdefault("attribution", {})[a] = {"n_changed_treatment": n_treat, "n_changed_null": n_ctrl,
                                                "attributable_fraction": attributable_to(
                                                    "%s: changed conditioning fields vs the intact rebuild" % a,
                                                    n_treat, n_ctrl)}
        if arms["intact_a"] and arms["intact_b"]:
            out["rows"][ROW_FOR_ARM[a]] = score_row(ROW_FOR_ARM[a], ia, ib, arms[a])
    cc = out["changes_vs_intact"]
    out["success_check"] = {
        "conditioning_changes_under_affect_lesion": bool(cc.get("affect_lesion")),
        "conditioning_changes_under_gnw_lesion": bool(cc.get("gnw_lesion")),
        "clean_null": out["null_clean"],
    }
    path = os.path.join(out_dir, "s%d" % int(seed), "smoke_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps({"success_check": out["success_check"], "changes_vs_intact": cc,
                      "rows": {k: {kk: v[kk] for kk in ("load_bearing", "null_clean", "treatment_diffs")}
                               for k, v in out["rows"].items()}}, indent=2, default=str))
    return out


def _canon(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def compare_identity(path_a, path_b, out=None):
    a, b = json.load(open(path_a)), json.load(open(path_b))
    labels = sorted(set(a) | set(b))
    per = {}
    for t in labels:
        ea, eb = _canon(a.get(t)), _canon(b.get(t))
        if ea != eb:
            keys = sorted(set((a.get(t) or {}).keys()) | set((b.get(t) or {}).keys()))
            per[t] = [k for k in keys if _canon((a.get(t) or {}).get(k)) != _canon((b.get(t) or {}).get(k))]
    res = {"a": path_a, "b": path_b, "n_turns": len(labels),
           "sha256_a": hashlib.sha256(_canon(a).encode()).hexdigest(),
           "sha256_b": hashlib.sha256(_canon(b).encode()).hexdigest(),
           "identical": not per, "differing_turn_keys": per,
           "errors_a": {t: v.get("_error") for t, v in a.items() if isinstance(v, dict) and v.get("_error")},
           "errors_b": {t: v.get("_error") for t, v in b.items() if isinstance(v, dict) and v.get("_error")}}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
    print(json.dumps({k: res[k] for k in ("identical", "n_turns", "sha256_a", "sha256_b", "differing_turn_keys")},
                     indent=2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--compare-identity", nargs=2, metavar=("A", "B"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--turns", default=None, help="comma-separated probe labels (default: SMOKE_TURNS)")
    a = ap.parse_args(argv)
    if a.compare_identity:
        r = compare_identity(a.compare_identity[0], a.compare_identity[1], a.out)
        return 0 if r["identical"] else 1
    if a.score:
        score(a.seed, a.out_dir)
        return 0
    if a.arm:
        return run_arm(a.arm, a.seed, a.out_dir, turns=(a.turns.split(",") if a.turns else None))
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
