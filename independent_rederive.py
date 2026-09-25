#!/usr/bin/env python3
"""INDEPENDENT re-derivation of the open-ended-gated-turn Part A verdict.

Written from scratch against the prereg text (research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md,
Amendment 3 / A3.1) and the raw session JSON field names. Does NOT import or call the grader
(research/runners/_open_ended_gated_turn_gate.py or _lbf_open_ended_production_turn_probe.py) for any scoring logic
-- only used those files to confirm field names exist (read, not executed for scoring).
"""
import json
import glob
import itertools
import os
from math import comb

OUT_DIR = "research/findings/raw/_open_ended_gated/partA"
A3_REF_DIR = "research/findings/raw/_load_bearing/_oe_production_turn/a3/default"
SEEDS = [42, 43, 44, 100, 101, 102]
M = 3
K = 8
FLOOR = 0.10
ALPHA = 0.05
N_TEACH_GUESS = None  # derived per-file below from len(replies) vs cont_draws keys


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def outcome(reply):
    if "error" in reply:
        return "ERROR"
    h = reply.get("hypothesis_svo")
    if h and len(h) == 3:
        return str(h[2])
    return "ABSTAIN"


def session_value(payload, w_ref, peak):
    reps = (payload or {}).get("replies") or []
    if not reps or peak <= 0:
        return None
    vals = []
    for r in reps:
        o = outcome(r)
        if o == "ERROR":
            return None
        wv = w_ref.get(o, 0.0) if o != "ABSTAIN" else 0.0
        vals.append(float(wv) / peak)
    return sum(vals) / len(vals)


def exact_perm_p(a, b):
    if not a or not b or any(v is None for v in list(a) + list(b)):
        return None
    pooled = list(a) + list(b)
    n, na = len(pooled), len(a)
    obs = sum(a) / na - sum(b) / len(b)
    ge = tot = 0
    for idx in itertools.combinations(range(n), na):
        s = set(idx)
        x = [pooled[i] for i in idx]
        y = [pooled[i] for i in range(n) if i not in s]
        tot += 1
        if sum(x) / len(x) - sum(y) / len(y) >= obs - 1e-12:
            ge += 1
    return ge / float(tot)


def sign_test_p(ds):
    if not ds or any(d is None for d in ds):
        return None
    n = len(ds)
    k = sum(1 for d in ds if d > 0)
    return sum(comb(n, j) for j in range(k, n + 1)) / float(2 ** n)


def score_cat_seed(seed, m=M):
    intact = [load(os.path.join(OUT_DIR, "oe_gated_s%s_intact_n%d.json" % (seed, j))) for j in range(m)]
    lesion = [load(os.path.join(OUT_DIR, "oe_gated_s%s_lesion_n%d.json" % (seed, j))) for j in range(m)]
    rebuild = load(os.path.join(OUT_DIR, "oe_gated_s%s_intact_rebuild_n0.json" % seed))
    res = {"reasons": []}
    if rebuild is None or any(x is None for x in intact + lesion):
        res["verdict"] = "ARM-FAILED"
        res["reasons"].append("missing session payload(s)")
        return res
    outs_i = [[outcome(r) for r in p["replies"]] for p in intact]
    outs_l = [[outcome(r) for r in p["replies"]] for p in lesion]
    outs_r = [outcome(r) for r in rebuild["replies"]]
    n_errors = sum(o.count("ERROR") for o in outs_i + outs_l + [outs_r])
    if n_errors:
        res["verdict"] = "ARM-FAILED"
        res["reasons"].append("errored replies")
        return res
    deterministic = (outs_r == outs_i[0] and rebuild.get("stored_facts") == intact[0].get("stored_facts")
                      and rebuild.get("noise_seed") == intact[0].get("noise_seed"))
    if not deterministic:
        res["verdict"] = "NONDETERMINISTIC"
        res["reasons"].append("intact_rebuild did not reproduce intact session 0")
        return res
    w_ref = intact[0].get("likelihood_weight") or {}
    peak = max([x for x in w_ref.values() if isinstance(x, (int, float))] or [0.0])
    w_equal = all((p.get("likelihood_weight") or {}) == w_ref for p in intact + lesion)
    facts_equal = all(p.get("stored_facts") == intact[0].get("stored_facts") for p in intact + lesion)
    noise_seeds = [p.get("noise_seed") for p in intact] + [p.get("noise_seed") for p in lesion]
    noise_distinct = len(set(noise_seeds)) == 2 * m
    noise_engaged = all((p.get("noise_stream_competes") or 0) > 0 for p in intact + lesion)
    draws_i = [p["draw_counter"]["n_calls"] for p in intact]
    draws_l = [p["draw_counter"]["n_calls"] for p in lesion]
    abl_i = [p["draw_counter"]["n_ablated_calls"] for p in intact]
    abl_l = [p["draw_counter"]["n_ablated_calls"] for p in lesion]
    n_distinct_i = len({tuple(o) for o in outs_i})
    n_distinct_l = len({tuple(o) for o in outs_l})
    noise_live = n_distinct_i > 1 or n_distinct_l > 1
    n_never_volunteered = sum(1 for s in outs_i for o in s if o != "ABSTAIN") == 0
    vi = [session_value(p, w_ref, peak) for p in intact]
    vl = [session_value(p, w_ref, peak) for p in lesion]
    checks = [
        (not w_equal, "host weight vector differs across sessions"),
        (not facts_equal, "stored facts differ across sessions"),
        (not noise_distinct, "noise-stream seeds not distinct"),
        (not noise_engaged, "a session never ran a draw on its noise stream"),
        (any(d == 0 for d in draws_i + draws_l), "the spiking draw was never reached"),
        (any(a == 0 for a in abl_l) or any(a != 0 for a in abl_i), "the lesion did not reach the draw"),
        (n_never_volunteered, "intact never volunteered"),
        (any(v is None for v in vi + vl), "a session value is UNDEFINED"),
        (not noise_live, "noise streams never changed a reply within an arm -> degenerate null"),
    ]
    for bad, why in checks:
        if bad:
            res["reasons"].append(why)
    res.update({"host_weight_vector": w_ref, "hist_intact": {}, "v_intact": vi, "v_lesion": vl,
                "noise_live": noise_live})
    if res["reasons"]:
        res["verdict"] = "UNDEFINED"
        return res
    res["verdict"] = "DEFINED"
    res["delta"] = sum(vi) / m - sum(vl) / m
    res["perm_p"] = exact_perm_p(vi, vl)
    return res


def ablation_fraction(p):
    dc = (p or {}).get("draw_counter") or {}
    n = dc.get("n_calls") or 0
    return (float(dc.get("n_ablated_calls") or 0) / n) if n else None


def session_cont(payload, w_ref, peak, n_teach):
    reps = (payload or {}).get("replies") or []
    draws = (payload or {}).get("cont_draws") or {}
    if not reps:
        return None, []
    vals = []
    for i, r in enumerate(reps):
        o = outcome(r)
        d_list = draws.get(str(n_teach + i))
        vals.append(ask_cont_value(d_list, o, w_ref, peak))
    if any(v is None for v in vals):
        return None, vals
    return sum(vals) / len(vals), vals


def ask_cont_value(draws, out, w_ref, peak):
    if out == "ERROR":
        return None
    if out == "ABSTAIN":
        return 0.0
    hits = [d for d in (draws or []) if d.get("winner") == out]
    if not hits or peak <= 0:
        return None
    d = hits[-1]
    fvs = [fv for fv in (d.get("fvs") or []) if sum(fv) > 0]
    if not fvs:
        return None
    fv = fvs[-1]
    tot = float(sum(fv))
    cands = d.get("cands") or []
    if len(cands) != len(fv):
        return None
    return sum((fv[i] / tot) * float(w_ref.get(cands[i], 0.0) or 0.0) for i in range(len(fv))) / peak


def score_manip_seed(seed, m=M):
    intact = [load(os.path.join(OUT_DIR, "oe_gated_s%s_intact_n%d.json" % (seed, j))) for j in range(m)]
    lesion = [load(os.path.join(OUT_DIR, "oe_gated_s%s_lesion_n%d.json" % (seed, j))) for j in range(m)]
    rebuild = load(os.path.join(OUT_DIR, "oe_gated_s%s_intact_rebuild_n0.json" % seed))
    w_ref = intact[0].get("likelihood_weight") or {}
    peak = max([x for x in w_ref.values() if isinstance(x, (int, float))] or [0.0])
    # n_teach = number of turns before the K asks = len(replies_total) - K; infer from cont_draws keys min - ... just
    # use gated_trace / cont_draws keys directly: keys are str(turn); asks are the LAST K reply indices in order.
    # Determine n_teach as min(int(k)) - i for consistency: we instead index cont_draws using the same offset the
    # session file itself used when it wrote cont_draws (turn = N_TEACH + i). Recover N_TEACH from the file: it is
    # (min turn key) - 0 ... but simplest: cont_draws has exactly K keys, sorted ascending; map reply i -> ith key.
    keys_sorted = sorted(int(x) for x in (intact[0].get("cont_draws") or {}).keys())
    n_teach = keys_sorted[0] if keys_sorted else 0
    ci = [session_cont(p, w_ref, peak, n_teach) for p in intact]
    cl = [session_cont(p, w_ref, peak, n_teach) for p in lesion]
    cr = session_cont(rebuild, w_ref, peak, n_teach)
    vi, vl = [c[0] for c in ci], [c[0] for c in cl]
    res = {"reasons": [], "v_intact": vi, "v_lesion": vl}
    fr_i = [ablation_fraction(p) for p in intact]
    fr_l = [ablation_fraction(p) for p in lesion]
    facts_eq = all(p.get("stored_facts") == intact[0].get("stored_facts") for p in intact + lesion + [rebuild])
    checks = [
        (any(v is None for v in vi + vl), "a session has a hypothesis ask with no recorded non-silent competition"),
        (len(set(vi)) == 1 and len(set(vl)) == 1, "degenerate null"),
        (cr[1] != ci[0][1], "rebuild per-ask CONT differs from intact session 0"),
        (any(f is None or f != 1.0 for f in fr_l) or any(f is None or f != 0.0 for f in fr_i),
         "lesion not applied (ablation fraction)"),
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
    res["passes_floor"] = res["delta"] >= FLOOR
    return res


def main():
    per_seed = {}
    for s in SEEDS:
        cat = score_cat_seed(s)
        man = score_manip_seed(s)
        verdict = "DEFINED" if (cat["verdict"] == "DEFINED" and man["verdict"] == "DEFINED") else (
            "NONDETERMINISTIC" if cat["verdict"] == "NONDETERMINISTIC" else "UNDEFINED")
        per_seed[s] = {"CAT": cat, "MANIP": man, "verdict": verdict}
        print("seed=%s CAT=%s delta=%s | MANIP=%s delta=%s | overall=%s" % (
            s, cat["verdict"], cat.get("delta"), man["verdict"], man.get("delta"), verdict))
        if cat["verdict"] == "UNDEFINED":
            print("   CAT reasons:", cat["reasons"])
        if man["verdict"] == "UNDEFINED":
            print("   MANIP reasons:", man["reasons"])

    defined = [s for s in SEEDS if per_seed[s]["verdict"] == "DEFINED"]
    all_defined = len(defined) == len(SEEDS)
    ds = [per_seed[s]["CAT"]["delta"] for s in defined]
    p_sign = sign_test_p(ds) if all_defined else None
    mean_d = sum(ds) / len(ds) if ds else None
    every_floor = all_defined and all(per_seed[s]["CAT"]["delta"] >= FLOOR for s in defined)
    manip_every_floor = all_defined and all(per_seed[s]["MANIP"]["passes_floor"] for s in defined)
    cat_rule_go = all_defined and p_sign is not None and p_sign < ALPHA and mean_d is not None and mean_d >= FLOOR
    go = all_defined and cat_rule_go and every_floor and manip_every_floor
    if not all_defined:
        verdict = "NOT-GO (UNDEFINED)"
    else:
        verdict = "GO" if go else "NO-GO"

    print("\n=== INDEPENDENT AGGREGATE ===")
    print("seeds scored:", SEEDS, "(exactly registered set:", sorted(SEEDS) == [42, 43, 44, 100, 101, 102], ")")
    print("n_defined:", len(defined), "/", len(SEEDS))
    print("all_defined:", all_defined)
    print("per-seed CAT delta:", {s: per_seed[s]["CAT"].get("delta") for s in SEEDS})
    print("per-seed CAT verdict:", {s: per_seed[s]["CAT"]["verdict"] for s in SEEDS})
    print("per-seed MANIP delta:", {s: per_seed[s]["MANIP"].get("delta") for s in SEEDS})
    print("per-seed MANIP verdict:", {s: per_seed[s]["MANIP"]["verdict"] for s in SEEDS})
    print("sign test p (over defined, informational; gate requires all 6):", p_sign)
    print("mean delta (defined seeds):", mean_d)
    print("every_seed_at_floor (CAT):", every_floor)
    print("manip_every_seed_at_floor:", manip_every_floor)
    print("cat_rule_go:", cat_rule_go)
    print("FINAL VERDICT (independent):", verdict)
    return per_seed, verdict


if __name__ == "__main__":
    main()
