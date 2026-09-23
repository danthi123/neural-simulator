"""OPEN-ENDED GENERATION load-bearing on the PRODUCTION chat turn -- the distributional production-turn probe.

WHY THIS EXISTS (lane `research/open-ended-production-turn-lb`, charter D1).
  open-ended-generation sits outside the load-bearing robust core. Two prior instruments each miss the production turn:
    * the single-turn field-diff (`load_bearing_fraction`, finding 2026-09-21-open-ended-generation-single-turn-not-
      load-bearing-...) reads ONE turn's volunteered patient -- a single soft-WTA draw is OU-noise-dominated for the
      which-patient choice, so intact and lesion can land on the same patient by chance (treat=0 on seed 42);
    * the distributional ruler (`LB_OPEN_ENDED_DISTRIB_PROBE`) draws many samples, but in the SYNTHETIC _followon2
      taxonomy world -- it never builds the webapp brain nor runs a production `brain_chat` turn.
  This probe measures the draw's load-bearingness ON THE PRODUCTION TURN, DISTRIBUTIONALLY: it drives the REAL
  `webapp.server.brain_chat` (the same entry point the regression battery / load-bearing battery call), teaches a
  natural chase KB through ordinary chat turns, then asks the SAME open-ended prompt K times in the SAME session and
  records the REPLY each time (the volunteered hypothesis SVO, or an abstain). Every ask is a fresh competition on
  the draw bank (its OU membrane-noise trajectory advances between asks), so K asks = K draws from the production
  reply distribution.

ARMS (each a FRESH subprocess brain build at the identical BRAIN_CHAT_SEED, so every inter-arm difference is the
  env the arm sets -- the battery's own arm discipline):
    intact          -- production env.
    intact_rebuild  -- production env again: the DETERMINISM check (the whole K-reply sequence must be IDENTICAL,
                       an exact compare, docs/TERMS.md `byte-identical` is not claimed -- a sequence-equality claim).
    lesion          -- BRAIN_SPIKING_DRAW_LESION=1: the SPECIFIC claimed edge -- the likelihood drive into the
                       spiking soft-WTA draw bank is replaced by a uniform drive (draw_from_weights honors it). Every
                       other faculty, gate and the moat are untouched.

STATISTIC + NULL (pre-registered, see docs/plans/2026-09-23-open-ended-production-turn-lb-PREREG.md).
  T  = total-variation distance between the intact and lesion reply histograms (outcome = volunteered patient, or
       ABSTAIN). NULL DISTRIBUTION = a label-permutation null over the pooled 2K replies (B permutations, seeded);
       p = (1 + #{T_null >= T}) / (B + 1). A null DISTRIBUTION, not one shuffle.
  D  = mean graph likelihood weight of the intact volunteered patients minus the lesion's (direction check: the
       likelihood-driven draw should volunteer MORE-associated patients; lesion ~ uniform over the admissible set).
  Per-seed LOAD-BEARING iff: determinism holds AND the draw was exercised in both arms AND intact volunteered >= 1
  hypothesis (else UNDEFINED -- not exercised is never a pass) AND p < ALPHA AND D > 0.

MODES (which production reply path):
    default     -- BRAIN_OPEN_ENDED unset (today's production default turn: the rich/strict path -> chat.gate ->
                   `_generate_hypothesis` -> the spiking draw).
    oe_unfixed  -- BRAIN_OPEN_ENDED=1 (the mission's open-ended conversational reply path) WITHOUT the route fix: the
                   diagnosis arm. `webapp/open_ended_chat.answer_turn` never calls chat.gate, so the generative DRAW is
                   NEVER reached (draw count 0 in both arms) and the reply is lesion-invariant BY CONSTRUCTION.
    oe_routed   -- BRAIN_OPEN_ENDED=1 + BRAIN_OPEN_ENDED_GENERATE_ROUTE=1 (this lane's default-OFF fix): an explicit
                   generation prompt is routed to the brain's GENERATE channel instead of the free-talk FORM path.

HOST SHORTCUTS (declared): the teach KB + the ask prompt are the WORLD (conversational input). The histogram / TV /
  permutation statistic is the INSTRUMENT (measurement, not the brain). In the oe_* modes the warm-Qwen faculty is a
  STUB (`_StubFaculty`) and BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK=1 -- the mouth's FORM is not what this probe measures,
  and the ask prompt's topic is unknown to the free-talk retriever so Qwen would only have produced text that the
  post-filter replaces with the fixed abstain string anyway (see open_ended_chat.no_qwen_fallback_enabled).
  The draw itself is brain-based (a cp_firing_states read on an Izhikevich WTA bank).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ALPHA = 0.05
N_PERM = 10000
ASK = "what might a dog chase"
ASK_SESSION = "oep"
# NATURAL chase KB (the WORLD): prey chased by DIFFERENT numbers of predators, so the brain's own co-occurrence graph
# gives the (dog, chase, ?) candidates GRADED likelihoods (rabbit 4 > deer 3 > mouse 2 > beetle/minnow 1). Two
# predators also chase the dog, so the agent-action edge (dog, chase) is not a lone weight-1 edge sitting on the
# gate's median (the v2 layer-2 masking cause). One predator per (agent, chase) key -- a second patient for the same
# key would be REWRITTEN in place by the default-ON reconsolidation organ, not added.
TEACH = [
    "the wolf chase the rabbit", "the fox chase the rabbit", "the hawk chase the rabbit", "the eagle chase the rabbit",
    "the lion chase the deer", "the tiger chase the deer", "the cougar chase the deer",
    "the owl chase the mouse", "the snake chase the mouse",
    "the crow chase the beetle", "the pike chase the minnow",
    "the coyote chase the dog", "the bear chase the dog",
]
MODES = {
    "default": {},
    "oe_unfixed": {"BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_GENERATE_ROUTE": "0",
                   "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "1"},
    "oe_routed": {"BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_GENERATE_ROUTE": "1",
                  "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "1"},
    # AMENDMENT 1 (2026-09-23, pre-registered before these ran): seed 42 showed the oe_* modes ALSO bypass in-loop
    # ACQUISITION -- under BRAIN_OPEN_ENDED=1 a teach assertion goes to the free-talk path and is never learned
    # (stored_facts stays at the 5 build-time facts), so the ask has nothing novel to volunteer. The *_taught modes
    # run the TEACH turns with BRAIN_OPEN_ENDED=0 (the ordinary chat path, SAME session / SAME ChatBrain -- the cache
    # key does not include the mode) and only the ASK turns in open-ended mode, isolating the ASK path's use of the
    # draw from the (separate) teach-path bypass.
    "oe_unfixed_taught": {"BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_GENERATE_ROUTE": "0",
                          "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "1"},
    "oe_routed_taught": {"BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_GENERATE_ROUTE": "1",
                         "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "1"},
}
# env overrides applied ONLY during the TEACH phase (restored before the asks)
TEACH_ENV = {"oe_unfixed_taught": {"BRAIN_OPEN_ENDED": "0"}, "oe_routed_taught": {"BRAIN_OPEN_ENDED": "0"}}
ARMS = {
    "intact": {"BRAIN_SPIKING_DRAW_LESION": "0"},
    "intact_rebuild": {"BRAIN_SPIKING_DRAW_LESION": "0"},
    "lesion": {"BRAIN_SPIKING_DRAW_LESION": "1"},
}


# ── worker: ONE fresh production brain, teach, ask K times, dump every reply + draw provenance ─────────────────
class _StubFaculty:
    """Stand-in for the warm Qwen faculty in the oe_* modes (declared host stub: FORM is not measured here)."""

    def __getattr__(self, name):
        raise RuntimeError("stub Qwen faculty used (%s) -- the probe expects no Qwen call" % name)


def _worker(env, seed, k, out_path, rich=True, teach_env=None):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    for kk, vv in env.items():
        os.environ[kk] = vv                     # explicit values, never a pop (OFF-arm discipline)
    # INSTRUMENT: count production draws through the spiking soft-WTA (transparent wrapper -- same return value).
    import research.runners._followon2_spiking_wta_sampler_derisk as _F2
    counter = {"n_calls": 0, "n_ablated_calls": 0}
    _orig = _F2.SpikingWTASampler.draw_from_weights

    def _counted(self, weights, candidates, max_retries=3):
        counter["n_calls"] += 1
        if getattr(self, "ablate_likelihood", False):
            counter["n_ablated_calls"] += 1
        return _orig(self, weights, candidates, max_retries=max_retries)

    _F2.SpikingWTASampler.draw_from_weights = _counted
    import webapp.server as S
    if os.environ.get("BRAIN_OPEN_ENDED", "0").strip().lower() in ("1", "true", "on", "yes"):
        S._get_warm_qwen_renderer = lambda: type("R", (), {"_fac": _StubFaculty()})()
    from webapp.server import brain_chat, BrainChatRequest
    t0 = time.time()
    teach = []
    teach_env = dict(teach_env or {})
    saved = {kk: os.environ.get(kk) for kk in teach_env}
    os.environ.update(teach_env)                 # TEACH-phase-only overrides (AMENDMENT 1); {} -> no-op
    for i, msg in enumerate(TEACH):
        r = brain_chat(BrainChatRequest(session=ASK_SESSION, message=msg, brain="tiny-demo", renderer="stub",
                                        rich=False, reset=(i == 0)))
        b = json.loads(r.body)
        teach.append({"msg": msg, "answer": b.get("answer"), "recalled_svo": b.get("recalled_svo")})
    t_teach = time.time() - t0
    for kk, vv in saved.items():                 # restore the ARM env for the asks
        if vv is None:
            os.environ.pop(kk, None)
        else:
            os.environ[kk] = vv
    chat = None
    for key, c in S._BRAIN_CHATS.items():
        if key[0] == ASK_SESSION:
            chat = c
    stored = sorted(set(map(tuple, getattr(chat, "stored_facts", []) or []))) if chat is not None else []
    replies = []
    for j in range(int(k)):
        before = counter["n_calls"]
        try:
            r = brain_chat(BrainChatRequest(session=ASK_SESSION, message=ASK, brain="tiny-demo", renderer="stub",
                                            rich=bool(rich), reset=False))
            b = json.loads(r.body)
            hyp = b.get("hypothesis_svo") if b.get("hypothesis") else None
            replies.append({"i": j, "hypothesis_svo": hyp, "abstained": bool(b.get("abstained")),
                            "answer": b.get("answer"), "mode": b.get("mode"),
                            "n_draws": counter["n_calls"] - before})
        except Exception as e:
            replies.append({"i": j, "error": "%s: %s" % (type(e).__name__, e),
                            "n_draws": counter["n_calls"] - before})
    # the brain's OWN likelihood weight for each (dog, chase, p) -- read AFTER the asks (read-only; the proposer is
    # cached per fact-count so this is the SAME graph the draws used).
    weights = {}
    try:
        prop = chat._build_generation_proposer() if chat is not None else None
        if prop is not None:
            w = prop._weight_partner(("dog", "chase"), list(prop.patients))
            weights = {p: float(x) for p, x in zip(prop.patients, w)}
    except Exception as e:
        weights = {"_error": repr(e)}
    out = {"env": env, "teach_env": teach_env, "seed": int(seed), "k": int(k), "rich": bool(rich), "teach": teach,
           "stored_facts": [list(f) for f in stored], "replies": replies, "draw_counter": counter,
           "likelihood_weight": weights, "t_teach_s": round(t_teach, 1), "t_total_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND")}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print("[oep worker] seed=%s env=%s k=%s draws=%s -> %s (%.0fs)" % (seed, env, k, counter, out_path,
                                                                      time.time() - t0), flush=True)
    return 0


def _spawn(env, seed, k, out_path, rich=True, teach_env=None):
    cmd = [sys.executable, "-u", "-m", "research.runners._lbf_open_ended_production_turn_probe", "--worker",
           "--env", json.dumps(env), "--seed", str(seed), "--k", str(k), "--out", out_path]
    if teach_env:
        cmd += ["--teach-env", json.dumps(teach_env)]
    if not rich:
        cmd.append("--single-fact")
    p = subprocess.run(cmd, env=dict(os.environ))
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    return json.load(open(out_path))


# ── pure statistics (no brain build; the selftest exercises these directly) ─────────────────────────────────────
def outcome(reply):
    """A reply's categorical outcome: the volunteered PATIENT of a hypothesis, 'ABSTAIN', or 'ERROR'."""
    if "error" in reply:
        return "ERROR"
    h = reply.get("hypothesis_svo")
    if h and len(h) == 3:
        return str(h[2])
    return "ABSTAIN"


def hist(outs):
    d = {}
    for o in outs:
        d[o] = d.get(o, 0) + 1
    return d


def tv_distance(a, b):
    """Total-variation distance between the empirical distributions of two outcome lists."""
    if not a or not b:
        return None
    ha, hb = hist(a), hist(b)
    keys = set(ha) | set(hb)
    return 0.5 * sum(abs(ha.get(x, 0) / len(a) - hb.get(x, 0) / len(b)) for x in keys)


def permutation_null(a, b, n_perm=N_PERM, seed=0):
    """Label-permutation NULL DISTRIBUTION of the TV statistic over the pooled replies. Returns (T_obs, p, null)."""
    import numpy as np
    t_obs = tv_distance(a, b)
    if t_obs is None:
        return None, None, []
    pooled = list(a) + list(b)
    na = len(a)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_perm)):
        idx = rng.permutation(len(pooled))
        pa = [pooled[i] for i in idx[:na]]
        pb = [pooled[i] for i in idx[na:]]
        null.append(tv_distance(pa, pb))
    ge = sum(1 for t in null if t >= t_obs - 1e-12)
    return t_obs, (1 + ge) / (len(null) + 1), null


def mean_weight(outs, weights):
    vals = [weights[o] for o in outs if o in weights and isinstance(weights.get(o), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def score_seed(intact, rebuild, lesion, alpha=ALPHA, n_perm=N_PERM):
    """The per-seed verdict from three worker payloads. Returns a dict with verdict in
    {LOAD-BEARING, NOT-LOAD-BEARING, UNDEFINED, NONDETERMINISTIC, WRONG-DIRECTION, ARM-FAILED}."""
    res = {"verdict": None, "reasons": []}
    if intact is None or rebuild is None or lesion is None:
        res["verdict"] = "ARM-FAILED"
        return res
    oi = [outcome(r) for r in intact["replies"]]
    orb = [outcome(r) for r in rebuild["replies"]]
    ol = [outcome(r) for r in lesion["replies"]]
    res["hist_intact"], res["hist_rebuild"], res["hist_lesion"] = hist(oi), hist(orb), hist(ol)
    res["deterministic"] = (oi == orb and intact.get("stored_facts") == rebuild.get("stored_facts"))
    di = intact["draw_counter"]["n_calls"]
    dl = lesion["draw_counter"]["n_calls"]
    res["draws_intact"], res["draws_lesion"] = di, dl
    res["lesion_ablated_draws"] = lesion["draw_counter"]["n_ablated_calls"]
    res["intact_ablated_draws"] = intact["draw_counter"]["n_ablated_calls"]
    res["n_errors"] = sum(1 for o in oi + ol if o == "ERROR")
    res["n_intact_volunteered"] = sum(1 for o in oi if o not in ("ABSTAIN", "ERROR"))
    res["n_lesion_volunteered"] = sum(1 for o in ol if o not in ("ABSTAIN", "ERROR"))
    t, p, null = permutation_null(oi, ol, n_perm=n_perm, seed=int(intact.get("seed", 0)))
    res["tv"], res["p_perm"] = t, p
    if null:
        import numpy as np
        res["null_tv_q50"], res["null_tv_q95"] = float(np.percentile(null, 50)), float(np.percentile(null, 95))
        # ATTRIBUTION: how much of the observed intact-vs-lesion TV is NOT already present at chance (the median TV
        # of the label-permutation null = the TV two exchangeable samples of this size show with no lesion effect).
        from tools.lab import attributable_to
        res["attributable_fraction"] = attributable_to(
            "open-ended draw lesion TV vs permutation-null median TV", t, res["null_tv_q50"])
    w = intact.get("likelihood_weight") or {}
    mi, ml = mean_weight(oi, w), mean_weight(ol, w)
    res["mean_w_intact"], res["mean_w_lesion"] = mi, ml
    res["direction_D"] = (mi - ml) if (mi is not None and ml is not None) else None
    if res["n_errors"]:
        res["reasons"].append("errors in replies")
        res["verdict"] = "ARM-FAILED"
        return res
    if not res["deterministic"]:
        res["verdict"] = "NONDETERMINISTIC"
        return res
    if di == 0 or dl == 0:
        res["reasons"].append("the spiking draw was never reached (draw count 0) -> the reply cannot depend on it")
        res["verdict"] = "UNDEFINED"
        return res
    if res["lesion_ablated_draws"] == 0 or res["intact_ablated_draws"] != 0:
        res["reasons"].append("the lesion did not reach the draw (ablated-call count wrong)")
        res["verdict"] = "UNDEFINED"
        return res
    if res["n_intact_volunteered"] == 0:
        res["reasons"].append("intact never volunteered (all abstain) -> not exercised")
        res["verdict"] = "UNDEFINED"
        return res
    if p is not None and p < alpha:
        if res["direction_D"] is not None and res["direction_D"] > 0:
            res["verdict"] = "LOAD-BEARING"
        else:
            res["verdict"] = "WRONG-DIRECTION"
    else:
        res["verdict"] = "NOT-LOAD-BEARING"
    return res


def run_seed(mode, seed, k, out_dir, rich=True):
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for arm, aenv in ARMS.items():
        env = dict(MODES[mode])
        env.update(aenv)
        payload[arm] = _spawn(env, seed, k, os.path.join(out_dir, "%s_s%s_%s.json" % (mode, seed, arm)), rich=rich,
                              teach_env=TEACH_ENV.get(mode))
    sc = score_seed(payload["intact"], payload["intact_rebuild"], payload["lesion"])
    rep = {"runner": "research.runners._lbf_open_ended_production_turn_probe", "mode": mode, "seed": int(seed),
           "k": int(k), "rich": bool(rich), "ask": ASK, "teach": TEACH, "alpha": ALPHA, "n_perm": N_PERM,
           "score": sc}
    path = os.path.join(out_dir, "%s_s%s_verdict.json" % (mode, seed))
    json.dump(rep, open(path, "w"), indent=2, default=str)
    print("[oep] mode=%s seed=%s verdict=%s tv=%s p=%s D=%s -> %s" % (
        mode, seed, sc["verdict"], sc.get("tv"), sc.get("p_perm"), sc.get("direction_D"), path), flush=True)
    return rep


def aggregate(paths, out):
    rows = [json.load(open(p)) for p in paths]
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    summary = {}
    for mode, rs in by_mode.items():
        verdicts = {str(r["seed"]): r["score"]["verdict"] for r in rs}
        n_lb = sum(1 for v in verdicts.values() if v == "LOAD-BEARING")
        summary[mode] = {"per_seed": verdicts, "n_seeds": len(rs), "n_load_bearing": n_lb,
                         "all_six_load_bearing": (len(rs) == 6 and n_lb == 6),
                         "tv": {str(r["seed"]): r["score"].get("tv") for r in rs},
                         "p_perm": {str(r["seed"]): r["score"].get("p_perm") for r in rs},
                         "direction_D": {str(r["seed"]): r["score"].get("direction_D") for r in rs}}
    rep = {"runner": "research.runners._lbf_open_ended_production_turn_probe --aggregate", "inputs": paths,
           "summary": summary}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return rep


def selftest():
    """Pure checks -- no brain build. Each check must be able to FAIL."""
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and bool(cond)

    chk("outcome: hypothesis -> patient", outcome({"hypothesis_svo": ["dog", "chase", "rabbit"]}) == "rabbit")
    chk("outcome: none -> ABSTAIN", outcome({"hypothesis_svo": None}) == "ABSTAIN")
    chk("outcome: error -> ERROR", outcome({"error": "x"}) == "ERROR")
    chk("tv identical = 0", tv_distance(["a", "b"], ["b", "a"]) == 0.0)
    chk("tv disjoint = 1", tv_distance(["a"] * 5, ["b"] * 5) == 1.0)
    chk("tv empty = None (UNDEFINED, never 0)", tv_distance([], ["a"]) is None)
    _, p_same, _ = permutation_null(["a", "b"] * 20, ["a", "b"] * 20, n_perm=500)
    chk("perm null: identical dists -> p large", p_same > 0.5)
    _, p_diff, _ = permutation_null(["a"] * 30, ["b"] * 15 + ["c"] * 15, n_perm=500)
    chk("perm null: disjoint dists -> p small", p_diff < 0.01)
    mk = lambda outs, abl, n: {"seed": 1, "replies": [{"hypothesis_svo": (["dog", "chase", o] if o != "ABSTAIN"
                                                                            else None)} for o in outs],
                               "draw_counter": {"n_calls": n, "n_ablated_calls": abl}, "stored_facts": [],
                               "likelihood_weight": {"rabbit": 4.0, "deer": 3.0, "mouse": 2.0, "beetle": 1.0}}
    I = ["rabbit"] * 24 + ["deer"] * 6
    L = ["rabbit", "deer", "mouse", "beetle"] * 7 + ["mouse", "beetle"]
    chk("score: separated + right direction -> LOAD-BEARING",
        score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(L, 90, 90), n_perm=500)["verdict"] == "LOAD-BEARING")
    chk("score: identical intact/lesion -> NOT-LOAD-BEARING",
        score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(I, 90, 90), n_perm=500)["verdict"] == "NOT-LOAD-BEARING")
    chk("score: reversed direction -> WRONG-DIRECTION",
        score_seed(mk(["beetle"] * 30, 0, 90), mk(["beetle"] * 30, 0, 90), mk(L, 90, 90),
                   n_perm=500)["verdict"] == "WRONG-DIRECTION")
    chk("score: rebuild differs -> NONDETERMINISTIC",
        score_seed(mk(I, 0, 90), mk(L, 0, 90), mk(L, 90, 90), n_perm=500)["verdict"] == "NONDETERMINISTIC")
    chk("score: draw never reached -> UNDEFINED (the oe_unfixed bypass)",
        score_seed(mk(["ABSTAIN"] * 30, 0, 0), mk(["ABSTAIN"] * 30, 0, 0), mk(["ABSTAIN"] * 30, 0, 0),
                   n_perm=500)["verdict"] == "UNDEFINED")
    chk("score: all-abstain with draws -> UNDEFINED (not exercised)",
        score_seed(mk(["ABSTAIN"] * 30, 0, 90), mk(["ABSTAIN"] * 30, 0, 90), mk(["ABSTAIN"] * 30, 90, 90),
                   n_perm=500)["verdict"] == "UNDEFINED")
    chk("score: lesion not reaching draw -> UNDEFINED",
        score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(L, 0, 90), n_perm=500)["verdict"] == "UNDEFINED")
    chk("score: missing arm -> ARM-FAILED", score_seed(None, None, None)["verdict"] == "ARM-FAILED")
    chk("modes: oe_unfixed sets route OFF explicitly", MODES["oe_unfixed"]["BRAIN_OPEN_ENDED_GENERATE_ROUTE"] == "0")
    chk("arms: intact sets lesion OFF explicitly (never a pop)", ARMS["intact"]["BRAIN_SPIKING_DRAW_LESION"] == "0")
    chk("amendment 1: *_taught modes teach with BRAIN_OPEN_ENDED=0, ask with =1",
        all(TEACH_ENV[m]["BRAIN_OPEN_ENDED"] == "0" and MODES[m]["BRAIN_OPEN_ENDED"] == "1"
            for m in ("oe_unfixed_taught", "oe_routed_taught")))
    chk("amendment 1: original modes carry NO teach override",
        all(m not in TEACH_ENV for m in ("default", "oe_unfixed", "oe_routed")))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--teach-env", default="{}", help="env overrides applied only during the TEACH phase")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default=None, help="comma list: run each seed sequentially")
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--mode", default="default", choices=sorted(MODES))
    ap.add_argument("--single-fact", action="store_true", help="ask with rich=False (the single-fact path)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--aggregate", nargs="*", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.worker:
        return _worker(json.loads(a.env), a.seed, a.k, a.out, rich=not a.single_fact,
                       teach_env=json.loads(a.teach_env))
    if a.aggregate is not None:
        return 0 if aggregate(a.aggregate, a.out) else 1
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    out_dir = a.out_dir or "research/findings/raw/_load_bearing/_oe_production_turn/%s" % a.mode
    for s in seeds:
        run_seed(a.mode, s, a.k, out_dir, rich=not a.single_fact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
