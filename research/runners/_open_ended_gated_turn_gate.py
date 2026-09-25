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

METRICS (PREREG Part A, as amended by Amendment 3 -- the review of 2026-09-24).
  REPLY (the Part A verdict) -- the a3 reply-level metric, scored by `score_seed_a3` UNCHANGED (m = 3): session value =
          mean w(reply)/peak (ABSTAIN / hold -> 0). Per seed DEFINED iff the a3 rules hold AND the manipulation check
          below is DEFINED.
  CONT (MANIPULATION CHECK, never a GO) -- per ask, the draw whose winner is the volunteered patient (the last such draw
          of the ask), its last non-silent fv: value = sum_p (fv_p / sum fv) * w_ref[p] / peak; ABSTAIN / hold -> 0.
          CONT weights firing shares by the SAME host vector the lesion flattens, so the lesion arm reads ~ mean(w)/peak
          whether or not the reply moves: it shows the lesion reached the reply-selecting competition, nothing more.
  GO   -- EXACTLY the registered seeds 42/43/44/100/101/102, all DEFINED, the a3 rule (sign test p < 0.05, mean Delta
          >= 0.10), Delta_reply >= 0.10 on EVERY seed, and CONT's Delta >= 0.10 on every seed. `summary.verdict` spells
          GO / NO-GO / NOT-GO (UNDEFINED) / NOT-GO (WRONG SEED SET).
  PASS-THROUGH (descriptive) -- the gated turn's recorded route / BG action / reply kind per ask, and the asks whose
          reply it replaced; on this ask the dev smoke read route=hypothesis + SPEAK every time (the gated routing is
          not what Part A tests). Plus a per-ask reply-identity read against the a3 `default` sessions on the same
          noise streams (A3_REF_DIR; other code revision).

HOST SHORTCUTS: those of the a3 probe (P.HOST_SHORTCUTS) plus the gated turn's declared residuals (the read->salience
transduction, the route label). The statistic and the recorder are INSTRUMENT.

Usage:
  one session (a pool job):  python -m research.runners._open_ended_gated_turn_gate --session --seed 42 --arm intact --j 0
  score:                     python -m research.runners._open_ended_gated_turn_gate --score --seeds 42,43,44,100,101,102
  job lines:                 python -m research.runners._open_ended_gated_turn_gate --jobs --root '~/derisk-pool/revisions/<sha>'
  selftest (no brain):       python -m research.runners._open_ended_gated_turn_gate --selftest
  descriptive (no brain):    --bg-curve (P(SPEAK) vs salience) / --bg-order (race-history read at s=2/3)
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


# Reply kinds with which the gated turn REPLACES the pipeline's answer (webapp/open_ended_gated_turn.decide). An ask
# whose reply_kind is one of these is an ask on which the gated routing changed the reply; any other kind is a
# pass-through of the ordinary pipeline answer.
REPLACING_KINDS = ("hold", "conditioned_generation", "ltm_fact_clause")
# PREREG Amendment 3: the a3 `default`-mode sessions (the ungated turn, amendment-3 noise streams) -- a DESCRIPTIVE
# per-ask reply-identity read against the same (seed, arm, j) stream. Their code revision differs (eefdd666a).
A3_REF_DIR = "research/findings/raw/_load_bearing/_oe_production_turn/a3/default"
A3_REF_MODE = "default"


def _ablation_fraction(p):
    dc = (p or {}).get("draw_counter") or {}
    n = dc.get("n_calls") or 0
    return (float(dc.get("n_ablated_calls") or 0) / n) if n else None


def pass_through(payloads):
    """DESCRIPTIVE (PREREG Amendment 3): how often the gated turn's own routing touched the ASK replies of these
    sessions. Counts the recorded route / bg_action / reply_kind over every ask, and the asks whose reply the gated
    turn REPLACED (REPLACING_KINDS). `exercised` is False when every ask passed the pipeline answer through."""
    routes, actions, kinds = {}, {}, {}
    n_asks = n_replaced = n_missing = 0
    for p in payloads:
        for o in (p or {}).get("gated_trace") or []:
            n_asks += 1
            if not o.get("present"):
                n_missing += 1
                continue
            for d, k in ((routes, o.get("route")), (actions, o.get("bg_action")), (kinds, o.get("reply_kind"))):
                d[str(k)] = d.get(str(k), 0) + 1
            if o.get("reply_kind") in REPLACING_KINDS:
                n_replaced += 1
    return {"n_asks": n_asks, "n_trace_missing": n_missing, "route": routes, "bg_action": actions,
            "reply_kind": kinds, "n_replaced_by_gated_turn": n_replaced, "exercised": bool(n_replaced > 0)}


def manip_check(intact, lesion, rebuild, m=M, floor=FLOOR):
    """PREREG Amendment 3: CONT intact vs lesion is a MANIPULATION CHECK, never a capability verdict. CONT weights the
    reply-selecting competition's firing shares by the SAME host weight vector the lesion replaces with ones, so a
    lesion arm whose firing follows its (uniform) drive reads ~ mean(w)/peak whatever the reply does. It asks one thing:
    did the lesion reach the competition that selects the reply? UNDEFINED rules as registered (Part A, CONT UNDEFINED
    rules), with the ablation rule implemented as the registered FRACTION (review 2026-09-24): every lesion session's
    ablated-draw fraction must be exactly 1 and every intact session's exactly 0."""
    w_ref = intact[0].get("likelihood_weight") or {}
    ci = [session_cont(p, w_ref) for p in intact]
    cl = [session_cont(p, w_ref) for p in lesion]
    cr = session_cont(rebuild, w_ref)
    vi, vl = [c[0] for c in ci], [c[0] for c in cl]
    peak = max([x for x in w_ref.values() if isinstance(x, (int, float))] or [0.0])
    fin = [x for x in w_ref.values() if isinstance(x, (int, float))]
    res = {"label": "MANIPULATION CHECK (not a capability verdict; PREREG Amendment 3)", "verdict": None,
           "reasons": [], "v_intact": vi, "v_lesion": vl, "per_ask_intact_s0": ci[0][1], "per_ask_rebuild": cr[1],
           "uniform_firing_value": ((sum(fin) / len(fin)) / peak if fin and peak > 0 else None)}
    fr_i = [_ablation_fraction(p) for p in intact]
    fr_l = [_ablation_fraction(p) for p in lesion]
    res["ablation_fraction_intact"], res["ablation_fraction_lesion"] = fr_i, fr_l
    facts_eq = all(p.get("stored_facts") == intact[0].get("stored_facts") for p in intact + lesion + [rebuild])
    checks = [
        (any(v is None for v in vi + vl), "a session has a hypothesis ask with no recorded non-silent competition"),
        (len(set(vi)) == 1 and len(set(vl)) == 1, "degenerate null: every intact and every lesion session identical"),
        (cr[1] != ci[0][1], "the rebuild's per-ask CONT values differ from intact session 0's"),
        (any(f is None or f != 1.0 for f in fr_l) or any(f is None or f != 0.0 for f in fr_i),
         "the lesion was not applied: an ablated-draw fraction below 1 in the lesion arm or above 0 in intact"),
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
        "oe_gated CONT (manipulation check) intact vs lesion mean session value (per seed)", sum(vi) / m, sum(vl) / m)
    res["perm_p_exact_one_sided"] = P.exact_perm_p(vi, vl)
    return res


def score_seed(intact, lesion, rebuild, m=M, floor=FLOOR):
    """Per-seed record (PREREG Amendment 3). The seed's Part A read is the REPLY-level metric (the a3 session value,
    `CAT`, scored by `score_seed_a3` unchanged); it is DEFINED only when CAT is DEFINED under the a3 rules AND the
    manipulation check (CONT) is DEFINED. The pass-through read is descriptive."""
    cat = P.score_seed_a3(intact, lesion, rebuild, m=m)
    res = {"CAT": cat, "MANIP": None, "verdict": None, "reasons": [], "m": m}
    if cat.get("verdict") == "ARM-FAILED":
        res["verdict"] = "ARM-FAILED"
        res["reasons"] = list(cat.get("reasons") or [])
        return res
    man = manip_check(intact, lesion, rebuild, m=m, floor=floor)
    res["MANIP"] = man
    res["pass_through"] = {"intact": pass_through(intact), "lesion": pass_through(lesion)}
    if cat.get("verdict") != "DEFINED":
        res["reasons"] += ["reply-level (CAT, a3 rules): %s" % r for r in (cat.get("reasons") or [cat.get("verdict")])]
    if man["verdict"] != "DEFINED":
        res["reasons"] += ["manipulation check: %s" % r for r in man["reasons"]]
    if res["reasons"]:
        res["verdict"] = cat.get("verdict") if cat.get("verdict") == "NONDETERMINISTIC" else "UNDEFINED"
        return res
    res["verdict"] = "DEFINED"
    res["delta"] = cat["delta"]                      # the reply-level Delta (a3 session value, intact - lesion)
    res["passes_floor"] = bool(res["delta"] >= floor)
    res["manip_passes_floor"] = bool(man["passes_floor"])
    return res


def verdict_label(seed_set_ok, all_defined, go):
    if not seed_set_ok:
        return "NOT-GO (WRONG SEED SET)"
    if not all_defined:
        return "NOT-GO (UNDEFINED)"
    return "GO" if go else "NO-GO"


def aggregate(per_seed, floor=FLOOR):
    """PREREG Amendment 3 aggregate. GO needs EXACTLY the registered seed set (review 2026-09-24: the old
    `n_required=len(seeds)` let 3 seeds, or 6 dev seeds, read GO), every seed DEFINED, the a3 reply-level rule (sign
    test p < ALPHA, mean Delta >= floor), Delta >= floor on EVERY seed, and the manipulation check >= floor on every
    seed. `verdict` spells the outcome: GO / NO-GO / NOT-GO (UNDEFINED) / NOT-GO (WRONG SEED SET)."""
    seeds = sorted(per_seed, key=int)
    seed_set_ok = sorted(int(s) for s in seeds) == sorted(SEEDS)
    defined = [s for s in seeds if per_seed[s].get("verdict") == "DEFINED"]
    ds = [per_seed[s]["delta"] for s in defined]
    all_defined = seed_set_ok and len(defined) == len(seeds) == len(SEEDS)
    cat = P.aggregate_a3({s: per_seed[s].get("CAT") or {} for s in seeds}, delta_floor=floor,
                         n_required=len(SEEDS))
    cat_rule_go = bool(seed_set_ok and cat.get("GO"))
    every_seed_floor = bool(all_defined and all(per_seed[s]["passes_floor"] for s in defined))
    manip_every_seed = bool(all_defined and all(per_seed[s]["manip_passes_floor"] for s in defined))
    go = bool(all_defined and cat_rule_go and every_seed_floor and manip_every_seed)
    man = {s: (per_seed[s].get("MANIP") or {}) for s in seeds}
    return {"n_seeds": len(seeds), "registered_seeds": list(SEEDS), "seed_set_ok": seed_set_ok,
            "n_defined": len(defined), "all_defined": all_defined,
            "per_seed_verdict": {s: per_seed[s].get("verdict") for s in seeds},
            "per_seed_reasons": {s: per_seed[s].get("reasons") for s in seeds},
            "delta_reply": {s: per_seed[s].get("delta") for s in seeds}, "floor": floor,
            "reply_a3_rule": cat, "reply_a3_rule_GO": cat_rule_go,
            "reply_every_seed_at_floor": every_seed_floor,
            "p_sign_test_reply": (P.sign_test_p(ds) if all_defined else None),
            "manipulation_check": {"label": "MANIPULATION CHECK (CONT intact vs lesion): never a GO",
                                   "per_seed_verdict": {s: man[s].get("verdict") for s in seeds},
                                   "delta_cont": {s: man[s].get("delta") for s in seeds},
                                   "uniform_firing_value": {s: man[s].get("uniform_firing_value") for s in seeds},
                                   "every_seed_at_floor": manip_every_seed},
            "gated_routing_exercised": {s: {a: ((per_seed[s].get("pass_through") or {}).get(a) or {}).get("exercised")
                                            for a in ("intact", "lesion")} for s in seeds},
            "GO": go, "verdict": verdict_label(seed_set_ok, all_defined, go)}


def a3_reference_identity(seeds, out_dir=OUT_DIR, ref_dir=A3_REF_DIR, m=M):
    """DESCRIPTIVE (PREREG Amendment 3; never gates): per (seed, arm, j), how many ask replies (categorical outcome)
    equal the a3 `default`-mode session on the same noise stream. The a3 sessions ran the UNGATED turn at another code
    revision, so a mismatch can come from code drift as well as from the gated turn."""
    out = {"ref_dir": ref_dir, "note": "a3 default sessions: ungated turn, code eefdd666a; descriptive only"}
    for s in seeds:
        row = {}
        for arm in ("intact", "lesion"):
            for j in range(m):
                a = P._load(session_path(out_dir, s, arm, j))
                b = P._load(P.a3_path(ref_dir, A3_REF_MODE, s, arm, j))
                if a is None or b is None or a.get("noise_seed") != b.get("noise_seed"):
                    row["%s_n%d" % (arm, j)] = None
                    continue
                oa = [P.outcome(r) for r in a.get("replies") or []]
                ob = [P.outcome(r) for r in b.get("replies") or []]
                row["%s_n%d" % (arm, j)] = [sum(1 for x, y in zip(oa, ob) if x == y), min(len(oa), len(ob))]
        out[str(s)] = row
    return out


def score(seeds, out_dir=OUT_DIR, aggregate_out=None, m=M, ref_dir=A3_REF_DIR, quiet=False):
    per_seed = {}
    for s in seeds:
        intact = [P._load(session_path(out_dir, s, "intact", j)) for j in range(m)]
        lesion = [P._load(session_path(out_dir, s, "lesion", j)) for j in range(m)]
        rebuild = P._load(session_path(out_dir, s, "intact_rebuild", 0))
        if any(x is None for x in intact + lesion + [rebuild]):
            per_seed[str(s)] = {"verdict": "ARM-FAILED", "reasons": ["missing session file(s)"], "CAT": {}}
            continue
        per_seed[str(s)] = score_seed(intact, lesion, rebuild, m=m)
        if not quiet:
            print("[oeg gate] seed=%s REPLY %s delta=%s | MANIP %s delta_cont=%s" % (
                s, per_seed[str(s)]["verdict"], per_seed[str(s)].get("delta"),
                (per_seed[str(s)].get("MANIP") or {}).get("verdict"), (per_seed[str(s)].get("MANIP") or {}).get("delta")),
                flush=True)
    agg = aggregate(per_seed)
    out = {"runner": "research.runners._open_ended_gated_turn_gate --score", "prereg": PREREG,
           "prereg_amendment": "Amendment 3 (the reply-level metric is the Part A verdict; CONT is a manipulation "
                               "check; exact registered seed set; explicit verdict string)",
           "mode": MODE, "M": m, "K": K, "floor": FLOOR, "lesioned_edge": P.LESIONED_EDGE,
           "host_shortcuts": P.HOST_SHORTCUTS + ["gated turn: read->salience transduction; route label (host)"],
           "per_seed": per_seed, "summary": agg,
           "a3_reference_identity_descriptive": a3_reference_identity(seeds, out_dir=out_dir, ref_dir=ref_dir, m=m)}
    if aggregate_out:
        os.makedirs(os.path.dirname(os.path.abspath(aggregate_out)), exist_ok=True)
        with open(aggregate_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
    if not quiet:
        print(json.dumps(agg, indent=2, default=str))
    return agg


BG_GRID = (0.0, 0.15, 0.3, 0.4, 0.5, 0.6, 0.67, 0.8, 1.0)
DEV_SEEDS = (7, 11, 13)


def bg_curve(seeds=DEV_SEEDS, n=8, out=None):
    """DESCRIPTIVE psychometric read of the gated turn's speak/abstain race (PREREG Amendment 1): P(SPEAK) /
    P(STAY_SILENT) / P(no commit) vs the salience split (speak=s, silent=1-s), n races per point on ONE selector per
    seed (the production usage: one warm selector, consecutive races, numpy global RNG seeded per seed). No brain."""
    import numpy as np
    from sim.backend import get_backend
    from research.runners.bg_action_selection_production_organ import BGActionSelector, ACTION_NAME
    res = {"runner": "research.runners._open_ended_gated_turn_gate --bg-curve", "prereg": PREREG,
           "backend": get_backend()[1], "n_per_point": int(n), "grid": list(BG_GRID), "seeds": [int(s) for s in seeds],
           "per_seed": {}}
    for sd in seeds:
        np.random.seed(int(sd))
        org = BGActionSelector(seed=int(sd))
        row = {}
        for s in BG_GRID:
            c = {"SPEAK": 0, "STAY_SILENT": 0, "none": 0}
            for _ in range(int(n)):
                r = org.select_once(s, 1.0 - s)
                c[ACTION_NAME[int(r["winner"])] if r["committed"] else "none"] += 1
            row[str(s)] = c
            print("[bg curve] seed=%s s=%s %s" % (sd, s, c), flush=True)
        res["per_seed"][str(sd)] = row
    res["pooled"] = {str(s): {k: sum(res["per_seed"][str(sd)][str(s)][k] for sd in seeds)
                              for k in ("SPEAK", "STAY_SILENT", "none")} for s in BG_GRID}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
    return res


def bg_order(seeds=DEV_SEEDS, n=8, s_hi=2.0 / 3.0, out=None):
    """DESCRIPTIVE race-HISTORY read (not a gate; no brain). The in-pipeline race is not an isolated draw: the selector
    persists across turns on the session's private RNG timeline, so a turn's race follows the earlier turns' races.
    Three orders at the same salience s_hi (the emo turn's intact split at tone level 2): (A) the smoke's own order on a
    fresh timeline -- race 1 at s=0 (the `unknown` turn), race 2 at s_hi (the `emo` turn); (B) n races at s_hi each
    right after an s=0 race; (C) n consecutive s_hi races on a fresh timeline. Asks whether the dev bg-curve's P(SPEAK)
    at 0.67 (an ascending grid: each point follows the 0.6 races) transfers to a race that follows an s=0 race."""
    import numpy as np
    from sim.backend import get_backend
    from research.runners.bg_action_selection_production_organ import BGActionSelector, ACTION_NAME

    def fresh(sd):
        np.random.seed(int(sd))
        return BGActionSelector(seed=int(sd))

    def race(org, s):
        r = org.select_once(s, 1.0 - s)
        return ACTION_NAME[int(r["winner"])] if r["committed"] else "none"

    res = {"runner": "research.runners._open_ended_gated_turn_gate --bg-order", "prereg": PREREG,
           "backend": get_backend()[1], "n": int(n), "s_hi": float(s_hi), "seeds": [int(s) for s in seeds],
           "per_seed": {}}
    for sd in seeds:
        org = fresh(sd)
        seq_a = [race(org, 0.0), race(org, s_hi)]
        org = fresh(sd)
        after_s0 = []
        for _ in range(int(n)):
            race(org, 0.0)
            after_s0.append(race(org, s_hi))
        org = fresh(sd)
        consecutive = [race(org, s_hi) for _ in range(int(n))]
        res["per_seed"][str(sd)] = {"smoke_order_race1_s0": seq_a[0], "smoke_order_race2_s_hi": seq_a[1],
                                    "after_s0": after_s0, "fresh_consecutive": consecutive}
        print("[bg order] seed=%s %s" % (sd, json.dumps(res["per_seed"][str(sd)])), flush=True)
    for k in ("after_s0", "fresh_consecutive"):
        res["pooled_speak_" + k] = [sum(v[k].count("SPEAK") for v in res["per_seed"].values()),
                                    sum(len(v[k]) for v in res["per_seed"].values())]
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=2)
    return res


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

    def arms_for(seed, moves=True):
        """synthetic sessions for one seed. moves=True: the reply moves (peaky intact, flat lesion replies). False: the
        firing differs exactly as above, but every lesion session answers what the matching intact session answered."""
        b = int(seed) * 1000
        i_outs = [["rabbit"] * 7 + [o] for o in ("deer", "rabbit", "mouse")]
        l_outs = ([["beetle"] * 6 + [o, "deer"] for o in ("mouse", "rabbit", "beetle")] if moves else i_outs)
        I_ = [sess(i_outs[j], [pk[j]] * 8, 0, b + j) for j in range(M)]
        L_ = [sess(l_outs[j], [fl[j]] * 8, 8, b + 500 + j) for j in range(M)]
        R_ = sess(i_outs[0], [pk[0]] * 8, 0, b)
        return I_, L_, R_

    I, L, R = arms_for(42)
    s = score_seed(I, L, R)
    chk("seed: the reply moves (peaky intact vs flat lesion) -> DEFINED, reply delta >= floor, manip passes",
        s["verdict"] == "DEFINED" and s["passes_floor"] and s["manip_passes_floor"])
    I2 = [sess(["rabbit"] * 8, [peaky] * 8, 0, 42000 + j) for j in range(M)]
    L2 = [sess(["rabbit"] * 8, [flat] * 8, 8, 42500 + j) for j in range(M)]
    s2 = score_seed(I2, L2, sess(["rabbit"] * 8, [peaky] * 8, 0, 42000))
    chk("seed: every session identical within both arms -> UNDEFINED (degenerate null)",
        s2["verdict"] == "UNDEFINED" and any("degenerate" in r.lower() for r in s2["reasons"]))
    s3 = score_seed(I, L, sess(["rabbit"] * 8, [flat] * 8, 0, 42000))
    chk("seed: rebuild differs -> not DEFINED", s3["verdict"] != "DEFINED")
    Lbad = [sess(["beetle"] * 8, [flat] * 8, 0, 42500 + j) for j in range(M)]
    chk("seed: lesion not applied -> not DEFINED", score_seed(I, Lbad, R)["verdict"] != "DEFINED")
    # review 2026-09-24 (MINOR): the registered rule is a FRACTION -- a PARTIALLY ablated lesion session (7 of 8 draws)
    # passed the old any(n_ablated == 0) check; the a3 CAT rule alone still passes it, the manipulation check must not.
    Lpart = [dict(p, draw_counter={"n_calls": 8, "n_ablated_calls": (7 if j == 0 else 8)}) for j, p in enumerate(L)]
    sp = score_seed(I, Lpart, R)
    chk("seed: a partially ablated lesion session -> not DEFINED (fraction rule)",
        sp["verdict"] != "DEFINED" and sp["MANIP"]["verdict"] == "UNDEFINED" and sp["CAT"]["verdict"] == "DEFINED")
    # review 2026-09-24 (MAJOR): CONT reads a large delta when only the FIRING moves. The reply-level metric must not.
    In, Ln, Rn = arms_for(42, moves=False)
    sn = score_seed(In, Ln, Rn)
    chk("seed: firing moves, reply never moves -> manip passes, reply delta 0 (below floor)",
        sn["verdict"] == "DEFINED" and sn["manip_passes_floor"] and not sn["passes_floor"] and sn["delta"] == 0)
    rec = lambda d, p=True, mp=True: {"verdict": "DEFINED", "delta": d, "passes_floor": p, "manip_passes_floor": mp,
                                      "CAT": {"verdict": "DEFINED", "delta": d}, "MANIP": {"verdict": "DEFINED"}}
    six = {str(x): rec(0.3) for x in SEEDS}
    a6 = aggregate(six)
    chk("aggregate: the 6 registered seeds DEFINED, all >= floor -> GO", a6["GO"] and a6["verdict"] == "GO")
    a_low = aggregate(dict(six, **{"44": rec(0.05, False)}))
    chk("aggregate: one seed under the floor -> NO-GO", not a_low["GO"] and a_low["verdict"] == "NO-GO")
    a_man = aggregate(dict(six, **{"43": rec(0.3, True, False)}))
    chk("aggregate: manipulation check under the floor on one seed -> NO-GO", not a_man["GO"] and
        a_man["verdict"] == "NO-GO")
    a_und = aggregate(dict(six, **{"100": {"verdict": "UNDEFINED", "CAT": {"verdict": "UNDEFINED"}}}))
    chk("aggregate: one UNDEFINED seed -> NOT-GO (UNDEFINED)", not a_und["GO"] and a_und["verdict"] == "NOT-GO (UNDEFINED)")
    a5 = aggregate({k: v for k, v in six.items() if k != "102"})
    chk("aggregate: 5 seeds -> NOT-GO (WRONG SEED SET)", not a5["GO"] and a5["verdict"] == "NOT-GO (WRONG SEED SET)")
    a_dev = aggregate({str(x): rec(0.3) for x in (7, 11, 13, 14, 15, 16)})
    chk("aggregate: 6 DEV seeds -> NOT-GO (WRONG SEED SET)", not a_dev["GO"] and a_dev["verdict"].endswith("SET)"))
    chk("aggregate: the a3 reply-level rule is reported", "reply_a3_rule" in a6 and "manipulation_check" in a6)
    # the SCORE PATH (the review's demonstration went through score(), not aggregate()): synthetic session files
    import tempfile

    def write(tmp, seed, moves=True):
        I_, L_, R_ = arms_for(seed, moves)
        for arm, ps in (("intact", I_), ("lesion", L_), ("intact_rebuild", [R_])):
            for j, p in enumerate(ps):
                with open(session_path(tmp, seed, arm, j), "w") as fh:
                    json.dump(p, fh)

    with tempfile.TemporaryDirectory() as tmp:
        for sd in (42, 43, 44):
            write(tmp, sd)
        g3 = score([42, 43, 44], out_dir=tmp, ref_dir=tmp, quiet=True)
        chk("score(): 3 seeds that each pass -> not GO (WRONG SEED SET)", not g3["GO"] and "SEED SET" in g3["verdict"])
    with tempfile.TemporaryDirectory() as tmp:
        dev = (7, 11, 13, 14, 15, 16)
        for sd in dev:
            write(tmp, sd)
        gd = score(list(dev), out_dir=tmp, ref_dir=tmp, quiet=True)
        chk("score(): 6 dev seeds that each pass -> not GO (WRONG SEED SET)", not gd["GO"] and "SEED SET" in gd["verdict"])
    with tempfile.TemporaryDirectory() as tmp:
        for sd in SEEDS:
            write(tmp, sd)
        g6 = score(list(SEEDS), out_dir=tmp, ref_dir=tmp, quiet=True)
        chk("score(): the 6 registered seeds, reply moves on each -> GO (the gate can pass)", g6["GO"] and
            g6["verdict"] == "GO")
    with tempfile.TemporaryDirectory() as tmp:
        for sd in SEEDS:
            write(tmp, sd, moves=False)
        gn = score(list(SEEDS), out_dir=tmp, ref_dir=tmp, quiet=True)
        chk("score(): the 6 registered seeds, firing moves but the reply never does -> NO-GO",
            not gn["GO"] and gn["verdict"] == "NO-GO" and gn["manipulation_check"]["every_seed_at_floor"])
    chk("pass_through: an ask the gated turn replaced counts as exercised",
        pass_through([{"gated_trace": [{"present": True, "route": "hypothesis", "bg_action": "STAY_SILENT",
                                        "reply_kind": "hold"}]}])["exercised"]
        and not pass_through([{"gated_trace": [{"present": True, "route": "hypothesis", "bg_action": "SPEAK",
                                                "reply_kind": "hypothesis"}]}])["exercised"])
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--jobs", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bg-curve", action="store_true", help="descriptive BG psychometric read (no brain)")
    ap.add_argument("--bg-order", action="store_true", help="descriptive race-history read at s=2/3 (no brain)")
    ap.add_argument("--curve-seeds", default=",".join(str(s) for s in DEV_SEEDS))
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
    if a.bg_curve:
        bg_curve([int(x) for x in a.curve_seeds.split(",") if x.strip()],
                 out=a.aggregate_out or os.path.join(os.path.dirname(a.out_dir), "bg_curve", "bg_curve.json"))
        return 0
    if a.bg_order:
        bg_order([int(x) for x in a.curve_seeds.split(",") if x.strip()],
                 out=a.aggregate_out or os.path.join(os.path.dirname(a.out_dir), "bg_curve", "bg_order.json"))
        return 0
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
