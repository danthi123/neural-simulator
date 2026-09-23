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
  the draw bank (its OU membrane-noise trajectory advances between asks). AMENDMENT 2 (review): the K asks of ONE
  session are serially dependent and each arm is a deterministic function of the seed, so they are NOT K independent
  samples -- a seed is ONE observation. And at this operating point the bank is NOT a distributional sampler: the
  seed-42 intact arm volunteered the host-likelihood peak 39/40 times where p-proportional-to-w would give it ~3/7 --
  it behaves as a near-ARGMAX (an observation: generative diversity is lost), which contradicts the earlier premise
  that a single draw is OU-noise-dominated. The host_oracle arm measures that sharpening directly.

ARMS (each a FRESH subprocess brain build at the identical BRAIN_CHAT_SEED, so every inter-arm difference is the
  env the arm sets -- the battery's own arm discipline):
    intact          -- production env.
    intact_rebuild  -- production env again: the DETERMINISM check (the whole K-reply sequence must be IDENTICAL,
                       an exact compare, docs/TERMS.md `byte-identical` is not claimed -- a sequence-equality claim).
    lesion          -- BRAIN_SPIKING_DRAW_LESION=1: the HOST weight vector (w = _weight_partner over the HOST
                       co-occurrence matrix P) is replaced by np.ones before the host affine map into the spiking
                       soft-WTA bank's drive (see LESIONED_EDGE). It tests the host likelihood vector transmitted
                       THROUGH the spiking WTA -- not the spiking part itself. Every other gate and the moat untouched.
    host_oracle     -- (AMENDMENT 2, opt-in via --arms) BRAIN_SPIKING_DRAW=0: the SAME host w, drawn by the host
                       np.random.choice oracle. intact-vs-host_oracle = what the spiking WTA contributes.

STATISTIC (AMENDMENT 2, replaces the original within-session permutation p; see the PREREG's amendment log).
  Per seed: D_s = mean host-likelihood weight of the intact volunteered patients minus the lesion's; modal patient
  intact vs lesion; the within-session TV is kept as a DESCRIPTIVE effect size only. Per-seed verdict: DEFINED /
  UNDEFINED (draw not reached, lesion not applied, nothing volunteered -- never a pass) / NONDETERMINISTIC /
  ARM-FAILED, plus a direction label. Across seeds (the independent unit): the EXACT one-sided sign-flip
  randomization p on the D_s (n=6 -> min p = 1/64). GO for a mode iff 6 seeds, all DEFINED, sign-flip p < ALPHA.
  Reported beside it: the count of seeds whose modal reply changed, with its chance base rate.

MODES (which production reply path):
    default     -- BRAIN_OPEN_ENDED unset (today's production default turn: the rich/strict path -> chat.gate ->
                   `_generate_hypothesis` -> the spiking draw).
    oe_unfixed  -- BRAIN_OPEN_ENDED=1 (the mission's open-ended conversational reply path) WITHOUT the route fix: the
                   diagnosis arm. `webapp/open_ended_chat.answer_turn` never calls chat.gate, so the generative DRAW is
                   NEVER reached (draw count 0 in both arms) and the reply is lesion-invariant BY CONSTRUCTION.
    oe_routed   -- BRAIN_OPEN_ENDED=1 + BRAIN_OPEN_ENDED_GENERATE_ROUTE=1 (this lane's default-OFF fix): an explicit
                   generation prompt is routed to the brain's GENERATE channel instead of the free-talk FORM path.
    oe_routed_full -- (AMENDMENT 2) the TRUE open-ended configuration: teach AND ask under BRAIN_OPEN_ENDED=1 with
                   both default-OFF routes (GENERATE + ACQUIRE). Replaces oe_routed_taught, which is NOT independent
                   evidence (its teach phase ran with BRAIN_OPEN_ENDED=0; see NOT_INDEPENDENT).

HOST SHORTCUTS (declared, full list in HOST_SHORTCUTS, carried in every verdict file): the teach KB + ask prompt
  (WORLD); the statistic (INSTRUMENT); the co-occurrence matrix P + weight vector w (the LESIONED input -- host, NOT
  "the brain's own likelihood"); the affine drive map; the argmax read-out over firing counts; role induction + SVO
  template; the prompt routers; the moat. In the oe_* modes the warm-Qwen faculty is a STUB (`_StubFaculty`) with
  BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK=1 (FORM is not measured). The spiking part is the Izhikevich + OU-noise WTA bank
  whose firing decides the winner.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ALPHA = 0.05
IN_SAMPLE_SEED = 42        # amendment 2 was designed after seeing seed 42 (declared in-sample)
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
    # AMENDMENT 2 (fix round, 2026-09-23): THE TRUE OPEN-ENDED CONFIGURATION. Teach AND ask both run with
    # BRAIN_OPEN_ENDED=1 (NO teach-phase override), with both default-OFF routes on: the GENERATE route (an explicit
    # generation prompt reaches the draw) and the ACQUIRE route (a told SVO assertion reaches in-loop acquisition --
    # the teach-path bypass that made `oe_routed` UNDEFINED). With both routes on, every turn of THIS protocol leaves
    # the free-talk block and runs the ordinary pipeline, so its replies are EXPECTED to equal `default`'s: that
    # equality is checked in data (aggregate `equals_default_replies`), and the finding says so -- this mode shows the
    # generative draw is REACHABLE and load-bearing from open-ended mode; it is NOT a second, independent mechanism.
    "oe_routed_full": {"BRAIN_OPEN_ENDED": "1", "BRAIN_OPEN_ENDED_GENERATE_ROUTE": "1",
                       "BRAIN_OPEN_ENDED_ACQUIRE_ROUTE": "1", "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "1"},
}
# Modes whose result is NOT independent evidence about BRAIN_OPEN_ENDED mode (review, 2026-09-23): the teach phase
# runs with BRAIN_OPEN_ENDED=0 and the routed ask falls through to the default pipeline, so its replies equal
# `default`'s by construction. Kept only so the seed-42 row stays reproducible; never staged for more seeds.
NOT_INDEPENDENT = {"oe_routed_taught": "teach phase runs with BRAIN_OPEN_ENDED=0 and the routed ask falls through to "
                                      "the default pipeline -> replies equal `default` by construction"}
# env overrides applied ONLY during the TEACH phase (restored before the asks)
TEACH_ENV = {"oe_unfixed_taught": {"BRAIN_OPEN_ENDED": "0"}, "oe_routed_taught": {"BRAIN_OPEN_ENDED": "0"}}
ARMS = {
    "intact": {"BRAIN_SPIKING_DRAW_LESION": "0"},
    "intact_rebuild": {"BRAIN_SPIKING_DRAW_LESION": "0"},
    "lesion": {"BRAIN_SPIKING_DRAW_LESION": "1"},
    # AMENDMENT 2: the HOST-ORACLE arm -- BRAIN_SPIKING_DRAW=0 leaves the proposer on its host np.random.choice draw
    # with p proportional to the SAME host weight vector. intact-vs-host_oracle isolates what the SPIKING WTA itself
    # contributes (the lesion arm cannot: it removes the host weight vector, not the spiking part). Opt-in (--arms).
    "host_oracle": {"BRAIN_SPIKING_DRAW": "0", "BRAIN_SPIKING_DRAW_LESION": "0"},
}
DEFAULT_ARMS = ("intact", "intact_rebuild", "lesion")
# What the lesion removes, stated where every verdict file carries it (review, 2026-09-23). NOT "the brain's own
# likelihood": the weight vector is HOST code.
LESIONED_EDGE = ("BRAIN_SPIKING_DRAW_LESION=1 replaces the HOST weight vector w = _weight_partner((dog, chase), "
                 "patients) -- a sum over the HOST co-occurrence matrix P built by ChatBrain._build_generation_proposer "
                 "from the stored facts -- with np.ones(V) before it is mapped (host affine map base_pA + gain_pA*w/peak)"
                 " into drive for the spiking Izhikevich soft-WTA bank. What the lesion tests: whether that HOST "
                 "likelihood vector, transmitted THROUGH the spiking WTA, changes the reply. It does NOT test whether "
                 "the spiking part is load-bearing; the host_oracle arm (same w, host np.random.choice) measures that.")
HOST_SHORTCUTS = [
    "teach KB + ask prompt (WORLD)",
    "histogram / TV / sign-flip statistic (INSTRUMENT)",
    "co-occurrence matrix P and the weight vector w = _weight_partner (HOST likelihood; the LESIONED input)",
    "affine drive map base_pA + gain_pA*w/peak (HOST)",
    "argmax over the bank's firing counts (HOST read-out of the spiking winner)",
    "hypothesis role induction + SVO template (HOST)",
    "prompt router _parse_open_ended + _is_acquisition_candidate (HOST regex / token rules)",
    "RF-composer moat verify (HOST scaffold)",
    "warm Qwen faculty stubbed in the oe_* modes (FORM not measured)",
]


# ── worker: ONE fresh production brain, teach, ask K times, dump every reply + draw provenance ─────────────────
class _StubFaculty:
    """Stand-in for the warm Qwen faculty in the oe_* modes (declared host stub: FORM is not measured here)."""

    def __getattr__(self, name):
        raise RuntimeError("stub Qwen faculty used (%s) -- the probe expects no Qwen call" % name)


def _worker(env, seed, k, out_path, rich=True, teach_env=None):
    if not os.path.isfile(os.path.join("data", "corpus", "tinystories.txt")):
        # the untracked corpus is absent in a fresh worktree / git archive -> the one-brain XEDGE build fails and the
        # webapp silently degrades to standalone organs (not the production brain). Refuse (2026-09-23).
        raise SystemExit("REFUSED: no data/corpus in %s (symlink data -> the main checkout's data/)" % os.getcwd())
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
    # AMENDMENT 2: also count EVERY generative draw at the proposer (spiking OR host oracle), so the host_oracle arm
    # (BRAIN_SPIKING_DRAW=0, no spiking call) still proves the draw was reached. Transparent wrapper.
    import research.runners._genfrontier_b2_generative_replay_derisk as _B2
    counter["n_sample_calls"] = 0
    _orig_sw = _B2.GenerativeReplayProposer._sample_weighted

    def _counted_sw(self, candidates, weights):
        counter["n_sample_calls"] += 1
        return _orig_sw(self, candidates, weights)

    _B2.GenerativeReplayProposer._sample_weighted = _counted_sw
    # quiet the per-bank-build INFO lines (the review measured 100-230 MB logs per controller); WARNING+ still print.
    import logging
    logging.disable(logging.INFO)
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
    counter["n_sample_calls_teach"] = counter["n_sample_calls"]
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


def seed_level_signflip(ds):
    """EXACT one-sided sign-flip randomization p over SEEDS (the independent unit; review 2026-09-23).

    Each seed contributes ONE number, D_s = mean host-likelihood weight of the intact arm's volunteered patients minus
    the lesion arm's. Under H0 (the lesion does not move the reply toward or away from the likelihood) the intact /
    lesion labels are exchangeable WITHIN a seed, so each D_s is symmetric about 0. p = #{e in {+1,-1}^n :
    sum(e_i*|D_i|) >= sum(D_i)} / 2^n. With n=6 the smallest attainable p is 1/64 = 0.0156; with n=1 it is 0.5, so a
    single seed can NEVER reach significance -- which is the point: the 40 asks of one session are serially dependent
    (and each arm is deterministic given the seed), so they are one observation, not forty. None if any D_s is None."""
    if not ds or any(d is None for d in ds):
        return None
    import itertools
    obs = sum(ds)
    mags = [abs(d) for d in ds]
    n = len(ds)
    ge = 0
    for signs in itertools.product((1, -1), repeat=n):
        if sum(s * m for s, m in zip(signs, mags)) >= obs - 1e-12:
            ge += 1
    return ge / float(2 ** n)


def mean_weight(outs, weights):
    vals = [weights[o] for o in outs if o in weights and isinstance(weights.get(o), (int, float))]
    return (sum(vals) / len(vals)) if vals else None


def _modal(outs):
    vol = [o for o in outs if o not in ("ABSTAIN", "ERROR")]
    if not vol:
        return None, 0.0
    h = hist(vol)
    m = sorted(h.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return m, h[m] / float(len(outs))


def admissible_set(weights, stored_facts, observed=()):
    """APPROXIMATE admissible candidate set for the (dog, chase, ?) hypothesis: patients with positive HOST weight that
    are not already a stored (dog, chase, p) fact, plus any patient an arm actually volunteered (declared approximation:
    the plausibility gate / moat can still reject some of these)."""
    stored = {tuple(f) for f in (stored_facts or [])}
    A = {p for p, w in (weights or {}).items()
         if isinstance(w, (int, float)) and w > 0 and ("dog", "chase", p) not in stored and p != "dog"}
    A |= {o for o in observed if o not in ("ABSTAIN", "ERROR")}
    return sorted(A)


def score_seed(intact, rebuild, lesion, host_oracle=None):
    """The per-seed record from the worker payloads (AMENDMENT 2). A seed is ONE observation: no within-session p-value
    is computed (the 40 asks of one session are not exchangeable). Returns a dict with `verdict` in
    {DEFINED, UNDEFINED, NONDETERMINISTIC, ARM-FAILED} and, when DEFINED, `label` in {CHANGED-TOWARD-LIKELIHOOD,
    SHIFTED-TOWARD-LIKELIHOOD, CHANGED-AWAY, SHIFTED-AWAY, CHANGED-NO-DIRECTION, UNCHANGED} plus D_s for the
    seed-level sign-flip test (`aggregate`)."""
    res = {"verdict": None, "label": None, "reasons": [], "lesioned_edge": LESIONED_EDGE}
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
    # DESCRIPTIVE effect size only (NOT a test): TV between the two within-session reply histograms.
    res["tv_within_session_descriptive"] = tv_distance(oi, ol)
    w = intact.get("likelihood_weight") or {}
    res["host_weight_vector"] = w
    mi, ml = mean_weight(oi, w), mean_weight(ol, w)
    res["mean_w_intact"], res["mean_w_lesion"] = mi, ml
    res["direction_D"] = (mi - ml) if (mi is not None and ml is not None) else None
    res["modal_intact"], res["modal_frac_intact"] = _modal(oi)
    res["modal_lesion"], res["modal_frac_lesion"] = _modal(ol)
    res["modal_changed"] = (res["modal_intact"] != res["modal_lesion"])
    # CHANCE BASE RATE (review): if the lesion's reply were a uniformly random admissible favourite, how often would it
    # differ from the intact modal, and how often would it sit BELOW the host-likelihood peak (D_s > 0)?
    A = admissible_set(w, intact.get("stored_facts"), set(oi) | set(ol))
    res["admissible_approx"] = A
    wa = [w[p] for p in A if isinstance(w.get(p), (int, float))]
    if A and wa:
        peak = max(wa)
        res["chance_p_modal_differs"] = 1.0 - 1.0 / len(A)
        res["chance_p_below_peak"] = sum(1 for x in wa if x < peak) / float(len(A))
        res["host_argmax_prediction"] = sorted([p for p in A if w.get(p) == peak])
        tot = float(sum(wa))
        res["host_sampler_predicted_mass"] = {p: (w[p] / tot if tot > 0 else None) for p in A if p in w}
    # WHAT THE SPIKING PART CONTRIBUTES (host_oracle arm: same host w, host np.random.choice draw).
    if host_oracle is not None:
        oh = [outcome(r) for r in host_oracle["replies"]]
        res["hist_host_oracle"] = hist(oh)
        res["host_oracle_sample_calls"] = host_oracle["draw_counter"].get("n_sample_calls")
        res["host_oracle_spiking_calls"] = host_oracle["draw_counter"]["n_calls"]
        res["tv_spiking_vs_host_oracle_descriptive"] = tv_distance(oi, oh)
        res["modal_host_oracle"], res["modal_frac_host_oracle"] = _modal(oh)
        res["host_oracle_stored_facts_equal"] = (host_oracle.get("stored_facts") == intact.get("stored_facts"))
        if res["modal_intact"] is not None:
            res["sharpening_modal_frac_spiking_minus_host"] = (
                res["modal_frac_intact"] - hist(oh).get(res["modal_intact"], 0) / float(len(oh)))
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
    if res["n_intact_volunteered"] == 0 or res["direction_D"] is None:
        res["reasons"].append("intact (or lesion) never volunteered (all abstain) -> not exercised")
        res["verdict"] = "UNDEFINED"
        return res
    res["verdict"] = "DEFINED"
    D, ch = res["direction_D"], res["modal_changed"]
    if D > 0:
        res["label"] = "CHANGED-TOWARD-LIKELIHOOD" if ch else "SHIFTED-TOWARD-LIKELIHOOD"
    elif D < 0:
        res["label"] = "CHANGED-AWAY" if ch else "SHIFTED-AWAY"
    else:
        res["label"] = "CHANGED-NO-DIRECTION" if ch else "UNCHANGED"
    return res


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _worker_path(out_dir, mode, seed, arm):
    return os.path.join(out_dir, "%s_s%s_%s.json" % (mode, seed, arm))


def run_seed(mode, seed, k, out_dir, rich=True, arms=DEFAULT_ARMS, reuse=False):
    """Spawn the requested arms (or, with reuse, load an existing worker JSON), then score IF the three default arms
    are all available. Returns the verdict record, or None when only a subset of arms was produced (scored later by
    --rescore)."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {}
    for arm in arms:
        env = dict(MODES[mode])
        env.update(ARMS[arm])
        path = _worker_path(out_dir, mode, seed, arm)
        if reuse and os.path.exists(path):
            payload[arm] = _load(path)
            continue
        payload[arm] = _spawn(env, seed, k, path, rich=rich, teach_env=TEACH_ENV.get(mode))
    return score_and_write(mode, seed, k, out_dir, rich=rich)


def score_and_write(mode, seed, k, out_dir, rich=True):
    arms = {a: _load(_worker_path(out_dir, mode, seed, a)) for a in ARMS}
    if any(arms[a] is None for a in DEFAULT_ARMS):
        print("[oep] mode=%s seed=%s: arms present=%s -- not scored yet" % (
            mode, seed, sorted(a for a, v in arms.items() if v is not None)), flush=True)
        return None
    sc = score_seed(arms["intact"], arms["intact_rebuild"], arms["lesion"], host_oracle=arms.get("host_oracle"))
    rep = {"runner": "research.runners._lbf_open_ended_production_turn_probe", "scorer": "amendment-2 (seed = unit)",
           "mode": mode, "seed": int(seed), "k": int(k), "rich": bool(rich), "ask": ASK, "teach": TEACH,
           "lesioned_edge": LESIONED_EDGE, "host_shortcuts": HOST_SHORTCUTS,
           "not_independent": NOT_INDEPENDENT.get(mode), "score": sc}
    path = os.path.join(out_dir, "%s_s%s_verdict.json" % (mode, seed))
    json.dump(rep, open(path, "w"), indent=2, default=str)
    print("[oep] mode=%s seed=%s verdict=%s label=%s D=%s -> %s" % (
        mode, seed, sc["verdict"], sc.get("label"), sc.get("direction_D"), path), flush=True)
    return rep


def _answers(payload):
    return [r.get("answer") for r in (payload or {}).get("replies", [])] if payload else None


def aggregate(paths, out, compare_dir=None):
    """Seed-level aggregate (AMENDMENT 2). GO for a mode iff exactly 6 seeds, ALL DEFINED, and the exact one-sided
    sign-flip p over the 6 seeds' D_s < ALPHA. `compare_dir` (the default mode's out-dir): per seed, whether this
    mode's intact/lesion reply TEXT sequences equal default's exactly (the oe_routed_full equality check)."""
    rows = [json.load(open(p)) for p in paths]
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    summary = {}
    for mode, rs in by_mode.items():
        rs = sorted(rs, key=lambda r: int(r["seed"]))
        sc = {str(r["seed"]): r["score"] for r in rs}
        defined = [s for s, x in sc.items() if x.get("verdict") == "DEFINED"]
        ds = [sc[s].get("direction_D") for s in defined]
        p = seed_level_signflip(ds) if defined else None
        all_defined = (len(defined) == len(rs))
        go = bool(len(rs) == 6 and all_defined and p is not None and p < ALPHA)
        # seed 42 was IN-SAMPLE for amendment 2's design -> also report the held-out (43/44/100/101/102) sign-flip p
        held = [sc[s].get("direction_D") for s in defined if s != str(IN_SAMPLE_SEED)]
        p_held = seed_level_signflip(held) if held else None
        m = {"p_signflip_heldout_excl_seed42": p_held, "n_heldout_defined": len(held),
             "GO_heldout_only": bool(len(held) == 5 and all_defined and p_held is not None and p_held < ALPHA),
             "per_seed_verdict": {s: x.get("verdict") for s, x in sc.items()},
             "per_seed_label": {s: x.get("label") for s, x in sc.items()},
             "n_seeds": len(rs), "n_defined": len(defined),
             "direction_D": {s: x.get("direction_D") for s, x in sc.items()},
             "p_signflip_over_seeds": p, "alpha": ALPHA, "GO": go,
             "n_modal_changed": sum(1 for s in defined if sc[s].get("modal_changed")),
             "expected_modal_changed_if_lesion_favourite_uniform": sum(
                 sc[s].get("chance_p_modal_differs") or 0.0 for s in defined),
             "expected_D_pos_if_intact_peak_and_lesion_favourite_uniform": sum(
                 sc[s].get("chance_p_below_peak") or 0.0 for s in defined),
             "modal_intact": {s: (x.get("modal_intact"), x.get("modal_frac_intact")) for s, x in sc.items()},
             "modal_lesion": {s: (x.get("modal_lesion"), x.get("modal_frac_lesion")) for s, x in sc.items()},
             "host_argmax_prediction": {s: x.get("host_argmax_prediction") for s, x in sc.items()},
             "tv_within_session_descriptive": {s: x.get("tv_within_session_descriptive") for s, x in sc.items()},
             "not_independent": NOT_INDEPENDENT.get(mode)}
        ho = {s: x for s, x in sc.items() if "hist_host_oracle" in x}
        if ho:
            m["host_oracle"] = {s: {"modal": (x.get("modal_host_oracle"), x.get("modal_frac_host_oracle")),
                                    "hist": x.get("hist_host_oracle"),
                                    "tv_spiking_vs_host": x.get("tv_spiking_vs_host_oracle_descriptive"),
                                    "sharpening": x.get("sharpening_modal_frac_spiking_minus_host"),
                                    "predicted_mass": x.get("host_sampler_predicted_mass")} for s, x in ho.items()}
        if compare_dir:
            eq = {}
            for r in rs:
                s = r["seed"]
                cur_dir = os.path.dirname(os.path.abspath(paths[0]))
                row = {}
                for arm in ("intact", "lesion"):
                    a = _answers(_load(_worker_path(cur_dir, mode, s, arm)))
                    b = _answers(_load(_worker_path(compare_dir, "default", s, arm)))
                    row[arm] = (a is not None and b is not None and a == b) if (a and b) else None
                eq[str(s)] = row
            m["equals_default_replies"] = eq
        summary[mode] = m
    rep = {"runner": "research.runners._lbf_open_ended_production_turn_probe --aggregate", "scorer": "amendment-2",
           "inputs": paths, "lesioned_edge": LESIONED_EDGE, "host_shortcuts": HOST_SHORTCUTS, "summary": summary}
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
    chk("signflip: ONE seed can never be significant (p = 0.5)", seed_level_signflip([0.95]) == 0.5)
    chk("signflip: 6/6 positive -> 1/64", abs(seed_level_signflip([1, 0.5, 2, 1, 1, 0.3]) - 1 / 64.) < 1e-12)
    chk("signflip: 6/6 negative -> 1.0", seed_level_signflip([-1] * 6) == 1.0)
    chk("signflip: 5 pos + 1 small neg -> 2/64", abs(seed_level_signflip([1, 1, 1, 1, 1, -0.5]) - 2 / 64.) < 1e-12)
    chk("signflip: a zero D gives no evidence (all zero -> 1.0)", seed_level_signflip([0.0] * 6) == 1.0)
    chk("signflip: an UNDEFINED seed -> None", seed_level_signflip([1, None]) is None)
    mk = lambda outs, abl, n: {"seed": 1, "replies": [{"hypothesis_svo": (["dog", "chase", o] if o != "ABSTAIN"
                                                                            else None)} for o in outs],
                               "draw_counter": {"n_calls": n, "n_ablated_calls": abl, "n_sample_calls": n},
                               "stored_facts": [],
                               "likelihood_weight": {"rabbit": 4.0, "deer": 3.0, "mouse": 2.0, "beetle": 1.0}}
    I = ["rabbit"] * 39 + ["deer"]
    L = ["beetle"] * 39 + ["mouse"]
    s1 = score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(L, 90, 90))
    chk("score: modal changed toward likelihood -> DEFINED/CHANGED-TOWARD-LIKELIHOOD",
        s1["verdict"] == "DEFINED" and s1["label"] == "CHANGED-TOWARD-LIKELIHOOD")
    chk("score: no within-session p-value is reported as evidence", "p_perm" not in s1)
    chk("score: carries the lesioned-edge declaration (host weight vector)", "HOST weight vector" in s1["lesioned_edge"])
    chk("score: chance base rate computed (4 admissible -> 0.75 modal-differs)",
        abs(s1["chance_p_modal_differs"] - 0.75) < 1e-12 and s1["host_argmax_prediction"] == ["rabbit"])
    chk("score: identical intact/lesion -> UNCHANGED",
        score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(I, 90, 90))["label"] == "UNCHANGED")
    chk("score: reversed direction -> CHANGED-AWAY",
        score_seed(mk(["beetle"] * 40, 0, 90), mk(["beetle"] * 40, 0, 90), mk(I, 90, 90))["label"] == "CHANGED-AWAY")
    chk("score: rebuild differs -> NONDETERMINISTIC",
        score_seed(mk(I, 0, 90), mk(L, 0, 90), mk(L, 90, 90))["verdict"] == "NONDETERMINISTIC")
    chk("score: draw never reached -> UNDEFINED (the oe_unfixed bypass)",
        score_seed(mk(["ABSTAIN"] * 30, 0, 0), mk(["ABSTAIN"] * 30, 0, 0), mk(["ABSTAIN"] * 30, 0, 0))["verdict"]
        == "UNDEFINED")
    chk("score: all-abstain with draws -> UNDEFINED (not exercised)",
        score_seed(mk(["ABSTAIN"] * 30, 0, 90), mk(["ABSTAIN"] * 30, 0, 90), mk(["ABSTAIN"] * 30, 90, 90))["verdict"]
        == "UNDEFINED")
    chk("score: lesion not reaching draw -> UNDEFINED",
        score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(L, 0, 90))["verdict"] == "UNDEFINED")
    chk("score: missing arm -> ARM-FAILED", score_seed(None, None, None)["verdict"] == "ARM-FAILED")
    H = ["rabbit"] * 16 + ["deer"] * 12 + ["mouse"] * 8 + ["beetle"] * 4
    s2 = score_seed(mk(I, 0, 90), mk(I, 0, 90), mk(L, 90, 90), host_oracle=mk(H, 0, 0))
    chk("host_oracle: sharpening = spiking modal frac - host frac of that patient (0.975 - 0.4)",
        abs(s2["sharpening_modal_frac_spiking_minus_host"] - (39 / 40. - 16 / 40.)) < 1e-12)
    chk("modes: oe_unfixed sets route OFF explicitly", MODES["oe_unfixed"]["BRAIN_OPEN_ENDED_GENERATE_ROUTE"] == "0")
    chk("arms: intact sets lesion OFF explicitly (never a pop)", ARMS["intact"]["BRAIN_SPIKING_DRAW_LESION"] == "0")
    chk("arms: host_oracle turns the spiking draw OFF explicitly", ARMS["host_oracle"]["BRAIN_SPIKING_DRAW"] == "0")
    chk("amendment 1: *_taught modes teach with BRAIN_OPEN_ENDED=0, ask with =1",
        all(TEACH_ENV[m]["BRAIN_OPEN_ENDED"] == "0" and MODES[m]["BRAIN_OPEN_ENDED"] == "1"
            for m in ("oe_unfixed_taught", "oe_routed_taught")))
    chk("amendment 2: oe_routed_full teaches AND asks in open-ended mode (no teach override), both routes ON",
        "oe_routed_full" not in TEACH_ENV and MODES["oe_routed_full"]["BRAIN_OPEN_ENDED"] == "1"
        and MODES["oe_routed_full"]["BRAIN_OPEN_ENDED_GENERATE_ROUTE"] == "1"
        and MODES["oe_routed_full"]["BRAIN_OPEN_ENDED_ACQUIRE_ROUTE"] == "1")
    chk("amendment 2: oe_routed_taught flagged NOT independent", "oe_routed_taught" in NOT_INDEPENDENT)
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
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma list of arms to spawn (see ARMS)")
    ap.add_argument("--reuse", action="store_true", help="load an existing worker JSON instead of re-spawning it")
    ap.add_argument("--rescore", action="store_true", help="score existing worker JSONs only (no spawn)")
    ap.add_argument("--single-fact", action="store_true", help="ask with rich=False (the single-fact path)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--aggregate", nargs="*", default=None)
    ap.add_argument("--compare-dir", default=None, help="default-mode out-dir for the reply-equality check")
    ap.add_argument("--parallel", type=int, default=1, help="run up to N seeds concurrently (each spawns its workers)")
    ap.add_argument("--aggregate-out", default=None, help="after the seeds finish, aggregate this mode's verdicts here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.worker:
        return _worker(json.loads(a.env), a.seed, a.k, a.out, rich=not a.single_fact,
                       teach_env=json.loads(a.teach_env))
    if a.aggregate is not None:
        return 0 if aggregate(a.aggregate, a.out, compare_dir=a.compare_dir) else 1
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [a.seed]
    out_dir = a.out_dir or "research/findings/raw/_load_bearing/_oe_production_turn/%s" % a.mode
    arms = tuple(x for x in a.arms.split(",") if x)
    for x in arms:
        if x not in ARMS:
            ap.error("unknown arm %r" % x)
    if a.rescore:
        for s in seeds:
            score_and_write(a.mode, s, a.k, out_dir, rich=not a.single_fact)
    elif a.parallel > 1:
        # each run_seed spawns its OWN subprocess workers, so threads here only overlap independent seeds
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            list(ex.map(lambda s: run_seed(a.mode, s, a.k, out_dir, rich=not a.single_fact, arms=arms,
                                           reuse=a.reuse), seeds))
    else:
        for s in seeds:
            run_seed(a.mode, s, a.k, out_dir, rich=not a.single_fact, arms=arms, reuse=a.reuse)
    if a.aggregate_out:
        import glob
        aggregate(sorted(glob.glob(os.path.join(out_dir, "%s_s*_verdict.json" % a.mode))), a.aggregate_out,
                  compare_dir=a.compare_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
