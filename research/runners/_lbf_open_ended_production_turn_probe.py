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

AMENDMENT 3 (review 2026-09-23) SUPERSEDES the amendment-2 statistic as the GO rule. The amendment-2 D_s is
  DEGENERATE: each arm is a deterministic function of the seed, so under H0 D_s is exactly 0 and the 1/64 only relabels
  "the modal changed on 6/6 seeds". Amendment 3 (`--a3-session` / `--a3-score`, see the block above `selftest`): per
  seed, A3_SESSIONS independent fresh-process sessions per arm, each drawing on its OWN noise stream
  (`_install_noise_stream`), K = A3_K asks each; session value = mean w(reply)/peak; Delta_s = intact - lesion mean;
  the null is non-degenerate BY DESIGN and asserted IN DATA (`noise_live`); GO = 6 seeds DEFINED + exact sign test
  over seeds p < ALPHA + mean Delta_s >= A3_DELTA. Seed 42's amendment-2 rows (intact 39/40 one patient, and the
  UNIFORM-drive lesion ALSO 39/40 one patient) show the production draw noise was effectively frozen across asks, so
  the amendment-2 "near-argmax / sharpening" reading is withdrawn (a uniform drive cannot be an argmax of w).

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


def _install_noise_stream(noise_seed, F2):
    """AMENDMENT 3: give the spiking draw its OWN stochastic source for this session.

    Every `_compete` (the WTA kernel whose only randomness is the bank's OU noise, drawn from the backend's GLOBAL RNG)
    runs with the global RNG swapped to a dedicated per-session stream seeded by `noise_seed`, then swapped back. So:
      * sessions that differ only in `noise_seed` differ only in the OU-noise realization of the draw;
      * the rest of the brain sees the SAME global-RNG trajectory whatever the draw consumes (in production the
        lesion arm's ~10x more draws also shifted every later global-RNG consumer -- a confound this removes);
      * the stream is NOT reset between asks (seed 42 under the production global RNG repeated ONE winner for 39 of
        40 asks in BOTH the intact and the uniform-drive lesion arm -- the draw noise was effectively frozen per ask).
    Declared INSTRUMENT (host): it chooses which noise realization the spiking bank sees; it computes no draw.
    numpy backend only (the pool / probe backend); refuses otherwise."""
    from sim.backend import get_backend, get_random_state, set_random_state
    import numpy as np
    if get_backend()[1] != "numpy":
        raise SystemExit("REFUSED: the per-session noise stream is implemented for SIM_BACKEND=numpy only")
    stream = {"state": np.random.RandomState(int(noise_seed)).get_state(), "n_competes": 0}
    _orig = F2.SpikingWTASampler._compete

    def _compete_on_session_stream(self, drive, V):
        saved = get_random_state()
        set_random_state(stream["state"])
        try:
            return _orig(self, drive, V)
        finally:
            stream["state"] = get_random_state()
            set_random_state(saved)
            stream["n_competes"] += 1

    F2.SpikingWTASampler._compete = _compete_on_session_stream
    return stream


def _worker(env, seed, k, out_path, rich=True, teach_env=None, noise_seed=None):
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
    # the bulk is SimulationBridge._log_console (a print per bank build, ~75 MB / 10 min): drop its info level only.
    # Output-only: the method returns None and nothing reads its stdout.
    import sim.bridge as _SB
    _orig_log = _SB.SimulationBridge._log_console

    def _quiet_log(self, message, level="info"):
        if str(level).lower() != "info":
            _orig_log(self, message, level)

    _SB.SimulationBridge._log_console = _quiet_log
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
    # AMENDMENT 3: the ASK phase only runs on the session's own draw-noise stream (the teach phase is untouched, so
    # every session of a seed learns the same facts from the same world).
    stream = _install_noise_stream(noise_seed, _F2) if noise_seed is not None else None
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
           "backend": os.environ.get("SIM_BACKEND"), "noise_seed": noise_seed,
           "noise_stream_competes": (stream["n_competes"] if stream is not None else None)}
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


# ── AMENDMENT 3 (review 2026-09-23): the GO statistic with a REAL null ──────────────────────────────────────────
# The amendment-2 statistic was DEGENERATE: each arm is a deterministic function of the seed, so under H0 D_s is
# exactly 0 (not a distribution), the sign-flip p = 1/64 only relabels "the modal changed on 6/6 seeds", and the
# intact arm sat on one patient 39/40 so "toward likelihood" followed from which patient that was. Amendment 3:
#   * per seed, M INDEPENDENT fresh-process sessions per arm; each session's draw runs on its own noise stream
#     (`_install_noise_stream`, seed = `a3_noise_seed`, fixed before any run, independent of every outcome);
#   * session value v = mean over the K asks of w(reply)/max(w) (ABSTAIN -> 0; w = the host weight vector);
#   * Delta_s = mean(v | intact sessions) - mean(v | lesion sessions). Under H0 (the lesion does not change the
#     reply distribution) the 2M sessions of a seed are iid, so Delta_s is SYMMETRIC about 0 with a non-degenerate
#     distribution (the noise streams vary the reply -- asserted per seed, `noise_live`), P(Delta_s > 0) <= 1/2;
#   * across the 6 seeds (independent substrates): exact one-sided SIGN TEST on Delta_s > 0 (6/6 -> p = 1/64);
#   * GO = 6 seeds, all DEFINED, sign-test p < ALPHA, AND mean Delta_s >= A3_DELTA (pre-registered effect floor).
#   * reported beside, not gating: the per-seed EXACT permutation p over all C(2M, M) label splits, the held-out
#     (seeds != 42) sign test, per-arm reply histograms and abstain rates.
A3_SEEDS = (42, 43, 44, 100, 101, 102)
A3_SESSIONS = 4            # M sessions per arm per seed (C(8,4) = 70 label splits per seed)
A3_K = 8                   # asks per session (the session is the unit; its asks are averaged, never counted)
A3_DELTA = 0.10            # effect floor on mean Delta_s, in units of the peak host weight (see the PREREG)
A3_ARMS = ("intact", "lesion", "intact_rebuild")


def a3_noise_seed(seed, arm, j):
    """Pre-registered, outcome-independent noise-stream seed. intact and lesion streams are DISJOINT; the rebuild
    re-uses intact session 0's stream (the determinism check must reproduce it exactly)."""
    off = {"intact": 0, "intact_rebuild": 0, "lesion": 500}[arm]
    return int(seed) * 1000 + off + (0 if arm == "intact_rebuild" else int(j))


def a3_path(out_dir, mode, seed, arm, j):
    return os.path.join(out_dir, "%s_s%s_%s_n%d.json" % (mode, seed, arm, int(j)))


def session_value(payload, w_ref):
    """One session's value: mean over its asks of w(volunteered patient)/peak; ABSTAIN -> 0; a patient absent from
    w_ref -> 0. None on any errored reply or no asks (UNDEFINED, never a score of 0)."""
    reps = (payload or {}).get("replies") or []
    peak = max([x for x in (w_ref or {}).values() if isinstance(x, (int, float))] or [0.0])
    if not reps or peak <= 0:
        return None
    vals = []
    for r in reps:
        o = outcome(r)
        if o == "ERROR":
            return None
        wv = (w_ref or {}).get(o, 0.0) if o != "ABSTAIN" else 0.0
        vals.append((float(wv) if isinstance(wv, (int, float)) else 0.0) / peak)
    return sum(vals) / len(vals)


def exact_perm_p(a, b):
    """EXACT one-sided permutation p over every split of the pooled session values into |a| / |b| (no RNG):
    p = #{splits: mean(x) - mean(y) >= observed} / C(n, |a|). None if either side is empty or has a None."""
    import itertools
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
    """Exact one-sided sign test over seeds: P(Binomial(n, 1/2) >= #{D > 0}). Valid because, under H0, each seed's
    Delta is symmetric about 0 (its 2M sessions are iid), so P(D > 0) <= 1/2 independently across seeds. None if
    any seed is None (UNDEFINED is never a pass)."""
    from math import comb
    if not ds or any(d is None for d in ds):
        return None
    n = len(ds)
    k = sum(1 for d in ds if d > 0)
    return sum(comb(n, j) for j in range(k, n + 1)) / float(2 ** n)


def score_seed_a3(intact, lesion, rebuild, m=A3_SESSIONS):
    """Per-seed record (AMENDMENT 3). intact / lesion: lists of M session payloads; rebuild: one payload."""
    res = {"verdict": None, "label": None, "reasons": [], "lesioned_edge": LESIONED_EDGE, "m": m}
    if (rebuild is None or len(intact or []) != m or len(lesion or []) != m
            or any(x is None for x in list(intact) + list(lesion))):
        res["verdict"] = "ARM-FAILED"
        res["reasons"].append("missing session payload(s)")
        return res
    outs_i = [[outcome(r) for r in p["replies"]] for p in intact]
    outs_l = [[outcome(r) for r in p["replies"]] for p in lesion]
    outs_r = [outcome(r) for r in rebuild["replies"]]
    res["n_errors"] = sum(o.count("ERROR") for o in outs_i + outs_l + [outs_r])
    if res["n_errors"]:
        res["verdict"] = "ARM-FAILED"
        res["reasons"].append("errored replies")
        return res
    res["deterministic"] = (outs_r == outs_i[0] and rebuild.get("stored_facts") == intact[0].get("stored_facts")
                            and rebuild.get("noise_seed") == intact[0].get("noise_seed"))
    if not res["deterministic"]:
        res["verdict"] = "NONDETERMINISTIC"
        res["reasons"].append("intact_rebuild did not reproduce intact session 0 on the same noise stream")
        return res
    w_ref = intact[0].get("likelihood_weight") or {}
    res["host_weight_vector"] = w_ref
    res["w_equal_across_sessions"] = all((p.get("likelihood_weight") or {}) == w_ref for p in intact + lesion)
    res["stored_facts_equal_across_sessions"] = all(p.get("stored_facts") == intact[0].get("stored_facts")
                                                    for p in intact + lesion)
    res["noise_seeds"] = {"intact": [p.get("noise_seed") for p in intact],
                          "lesion": [p.get("noise_seed") for p in lesion]}
    res["noise_seeds_distinct"] = len(set(res["noise_seeds"]["intact"] + res["noise_seeds"]["lesion"])) == 2 * m
    res["noise_stream_engaged"] = all((p.get("noise_stream_competes") or 0) > 0 for p in intact + lesion)
    res["draws_intact"] = [p["draw_counter"]["n_calls"] for p in intact]
    res["draws_lesion"] = [p["draw_counter"]["n_calls"] for p in lesion]
    res["ablated_intact"] = [p["draw_counter"]["n_ablated_calls"] for p in intact]
    res["ablated_lesion"] = [p["draw_counter"]["n_ablated_calls"] for p in lesion]
    # NON-DEGENERATE NULL, asserted in data: the noise streams must actually vary the reply within an arm
    res["n_distinct_sessions_intact"] = len({tuple(o) for o in outs_i})
    res["n_distinct_sessions_lesion"] = len({tuple(o) for o in outs_l})
    res["noise_live"] = res["n_distinct_sessions_intact"] > 1 or res["n_distinct_sessions_lesion"] > 1
    res["hist_intact"] = hist([o for s in outs_i for o in s])
    res["hist_lesion"] = hist([o for s in outs_l for o in s])
    res["abstain_rate_intact"] = res["hist_intact"].get("ABSTAIN", 0) / float(sum(len(s) for s in outs_i) or 1)
    res["abstain_rate_lesion"] = res["hist_lesion"].get("ABSTAIN", 0) / float(sum(len(s) for s in outs_l) or 1)
    vi = [session_value(p, w_ref) for p in intact]
    vl = [session_value(p, w_ref) for p in lesion]
    res["v_intact"], res["v_lesion"] = vi, vl
    checks = [
        (not res["w_equal_across_sessions"], "host weight vector differs across sessions (no common scale)"),
        (not res["noise_seeds_distinct"], "noise-stream seeds not distinct across the 2M sessions"),
        (not res["noise_stream_engaged"], "a session never ran a draw on its noise stream"),
        (any(d == 0 for d in res["draws_intact"] + res["draws_lesion"]), "the spiking draw was never reached"),
        (any(a == 0 for a in res["ablated_lesion"]) or any(a != 0 for a in res["ablated_intact"]),
         "the lesion did not reach the draw (ablated-call count wrong)"),
        (sum(1 for s in outs_i for o in s if o != "ABSTAIN") == 0, "intact never volunteered (not exercised)"),
        (any(v is None for v in vi + vl), "a session value is UNDEFINED"),
        (not res["noise_live"], "noise streams never changed a reply within an arm -> the null is DEGENERATE"),
    ]
    for bad, why in checks:
        if bad:
            res["reasons"].append(why)
    if res["reasons"]:
        res["verdict"] = "UNDEFINED"
        return res
    res["verdict"] = "DEFINED"
    res["delta"] = sum(vi) / m - sum(vl) / m
    res["perm_p_exact_one_sided"] = exact_perm_p(vi, vl)
    res["label"] = "TOWARD-LIKELIHOOD" if res["delta"] > 0 else ("AWAY" if res["delta"] < 0 else "NO-DIFFERENCE")
    return res


def aggregate_a3(per_seed, delta_floor=A3_DELTA, n_required=len(A3_SEEDS)):
    """GO iff exactly n_required seeds, ALL DEFINED, sign-test p < ALPHA, and mean Delta >= delta_floor."""
    seeds = sorted(per_seed, key=int)
    defined = [s for s in seeds if (per_seed[s] or {}).get("verdict") == "DEFINED"]
    ds = [per_seed[s]["delta"] for s in defined]
    all_defined = len(defined) == len(seeds) == n_required
    p = sign_test_p(ds) if all_defined else None
    mean_d = (sum(ds) / len(ds)) if ds else None
    held = [per_seed[s]["delta"] for s in defined if int(s) != IN_SAMPLE_SEED]
    p_held = sign_test_p(held) if (all_defined and held) else None
    go = bool(all_defined and p is not None and p < ALPHA and mean_d is not None and mean_d >= delta_floor)
    return {"n_seeds": len(seeds), "n_defined": len(defined), "all_defined": all_defined,
            "per_seed_verdict": {s: (per_seed[s] or {}).get("verdict") for s in seeds},
            "per_seed_reasons": {s: (per_seed[s] or {}).get("reasons") for s in seeds},
            "delta": {s: (per_seed[s] or {}).get("delta") for s in seeds},
            "n_delta_positive": sum(1 for d in ds if d > 0),
            "p_sign_test": p, "alpha": ALPHA, "mean_delta": mean_d, "delta_floor": delta_floor,
            "p_sign_test_heldout_excl_seed42": p_held,
            "GO_heldout_only": bool(all_defined and p_held is not None and p_held < ALPHA and held
                                    and sum(held) / len(held) >= delta_floor),
            "perm_p_exact_per_seed_descriptive": {s: (per_seed[s] or {}).get("perm_p_exact_one_sided")
                                                  for s in seeds},
            "noise_live": {s: (per_seed[s] or {}).get("noise_live") for s in seeds},
            "abstain_rate": {s: ((per_seed[s] or {}).get("abstain_rate_intact"),
                                 (per_seed[s] or {}).get("abstain_rate_lesion")) for s in seeds},
            "GO": go}


def run_a3_session(mode, seed, arm, j, k, out_dir):
    """ONE amendment-3 session, in THIS process (one full brain). Idempotent: an existing output is kept."""
    path = a3_path(out_dir, mode, seed, arm, j)
    if os.path.exists(path) and _load(path) is not None:
        print("[oep a3] exists, skipping: %s" % path, flush=True)
        return 0
    env = dict(MODES[mode])
    env.update(ARMS[arm])
    return _worker(env, seed, k, path, rich=True, teach_env=TEACH_ENV.get(mode), noise_seed=a3_noise_seed(seed, arm, j))


def score_a3(mode, seeds, out_dir, m=A3_SESSIONS, aggregate_out=None, sha=None):
    per_seed = {}
    for s in seeds:
        intact = [_load(a3_path(out_dir, mode, s, "intact", j)) for j in range(m)]
        lesion = [_load(a3_path(out_dir, mode, s, "lesion", j)) for j in range(m)]
        rebuild = _load(a3_path(out_dir, mode, s, "intact_rebuild", 0))
        sc = score_seed_a3(intact, lesion, rebuild, m=m)
        per_seed[str(s)] = sc
        rep = {"runner": "research.runners._lbf_open_ended_production_turn_probe --a3-score",
               "scorer": "amendment-3 (independent noise-stream sessions; seed = replication unit)", "mode": mode,
               "seed": int(s), "m": m, "ask": ASK, "teach": TEACH, "lesioned_edge": LESIONED_EDGE,
               "host_shortcuts": HOST_SHORTCUTS, "score": sc}
        with open(os.path.join(out_dir, "%s_s%s_a3_verdict.json" % (mode, s)), "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
        print("[oep a3] mode=%s seed=%s verdict=%s delta=%s perm_p=%s reasons=%s" % (
            mode, s, sc["verdict"], sc.get("delta"), sc.get("perm_p_exact_one_sided"), sc.get("reasons")), flush=True)
    agg = aggregate_a3(per_seed)
    if aggregate_out:
        os.makedirs(os.path.dirname(os.path.abspath(aggregate_out)), exist_ok=True)
        with open(aggregate_out, "w") as fh:
            json.dump({"runner": "research.runners._lbf_open_ended_production_turn_probe --a3-score",
                       "scorer": "amendment-3", "mode": mode, "out_dir": out_dir, "code_sha_of_sessions": sha,
                       "lesioned_edge": LESIONED_EDGE, "host_shortcuts": HOST_SHORTCUTS,
                       "A3_SESSIONS": m, "A3_K": A3_K, "A3_DELTA": A3_DELTA, "summary": agg}, fh, indent=2, default=str)
    print(json.dumps(agg, indent=2, default=str))
    return agg


W_S42 = {"beetle": 1.0, "cat": 2.0, "deer": 3.0, "dog": 2.0, "fish": 0.0, "minnow": 1.0, "rabbit": 2.0}


def _mk_session(outs, abl, n, noise_seed, competes=None, w=None, facts=None):
    return {"replies": [{"hypothesis_svo": (["dog", "chase", o] if o != "ABSTAIN" else None)} for o in outs],
            "draw_counter": {"n_calls": n, "n_ablated_calls": abl}, "stored_facts": facts or [["wolf", "chase", "rabbit"]],
            "likelihood_weight": dict(w or W_S42), "noise_seed": noise_seed,
            "noise_stream_competes": n if competes is None else competes}


def _mk_seed(seed, intact_outs, lesion_outs, m=A3_SESSIONS):
    I = [_mk_session(intact_outs[j], 0, 10, a3_noise_seed(seed, "intact", j)) for j in range(m)]
    L = [_mk_session(lesion_outs[j], 10, 10, a3_noise_seed(seed, "lesion", j)) for j in range(m)]
    R = _mk_session(intact_outs[0], 0, 10, a3_noise_seed(seed, "intact_rebuild", 0))
    return I, L, R


def selftest_a3(chk):
    """AMENDMENT-3 checks: each can FAIL, and the null is shown to be REAL (an A/A simulation must not GO)."""
    import random
    ok0 = [True]

    def c(name, cond):
        chk(name, cond)
        ok0[0] = ok0[0] and bool(cond)

    c("a3: rebuild re-uses intact session 0's stream; intact/lesion streams disjoint",
      a3_noise_seed(42, "intact_rebuild", 3) == a3_noise_seed(42, "intact", 0)
      and not ({a3_noise_seed(42, "intact", j) for j in range(8)} & {a3_noise_seed(42, "lesion", j) for j in range(8)}))
    c("a3: session value = mean w/peak, ABSTAIN -> 0", abs(session_value(
        _mk_session(["deer", "ABSTAIN", "rabbit", "beetle"], 0, 4, 1), W_S42) - (1 + 0 + 2 / 3. + 1 / 3.) / 4) < 1e-12)
    c("a3: an errored reply -> session value None (UNDEFINED)", session_value({"replies": [{"error": "x"}]}, W_S42) is None)
    c("a3: exact perm p, perfectly separated 2 vs 2 -> 1/6", abs(exact_perm_p([1, 1], [0, 0]) - 1 / 6.) < 1e-12)
    c("a3: exact perm p, identical values -> 1.0", exact_perm_p([0.5] * 4, [0.5] * 4) == 1.0)
    c("a3: sign test 6/6 -> 1/64; 5/6 -> 7/64; a zero is not positive",
      abs(sign_test_p([1] * 6) - 1 / 64.) < 1e-12 and abs(sign_test_p([1] * 5 + [-1]) - 7 / 64.) < 1e-12
      and abs(sign_test_p([1] * 5 + [0]) - 7 / 64.) < 1e-12)
    c("a3: sign test with an UNDEFINED seed -> None", sign_test_p([1, None]) is None)
    good_i = [["deer"] * 6 + ["rabbit", "cat"], ["deer"] * 7 + ["rabbit"], ["deer"] * 5 + ["cat"] * 3, ["deer"] * 8]
    good_l = [["beetle", "minnow", "cat", "deer", "ABSTAIN", "rabbit", "beetle", "minnow"],
              ["minnow"] * 4 + ["cat"] * 4, ["beetle", "ABSTAIN"] * 4, ["rabbit", "minnow"] * 4]
    s1 = score_seed_a3(*_mk_seed(43, good_i, good_l))
    c("a3: likelihood-tracking intact vs uniform-ish lesion -> DEFINED, delta > 0, perm p = 1/70",
      s1["verdict"] == "DEFINED" and s1["delta"] > 0 and abs(s1["perm_p_exact_one_sided"] - 1 / 70.) < 1e-12)
    same = [["deer"] * 8] * A3_SESSIONS
    s2 = score_seed_a3(*_mk_seed(43, same, same))
    c("a3: every session identical in both arms -> UNDEFINED (degenerate null), never a pass",
      s2["verdict"] == "UNDEFINED" and any("DEGENERATE" in r for r in s2["reasons"]))
    s3 = score_seed_a3(*_mk_seed(43, same, [["beetle"] * 8] * A3_SESSIONS))
    c("a3: arms constant but different (noise inert: the old deterministic-arm case) -> UNDEFINED",
      s3["verdict"] == "UNDEFINED")
    I, L, R = _mk_seed(43, good_i, good_l)
    R = _mk_session(["rabbit"] * 8, 0, 10, a3_noise_seed(43, "intact_rebuild", 0))
    c("a3: rebuild does not reproduce intact session 0 -> NONDETERMINISTIC",
      score_seed_a3(I, L, R)["verdict"] == "NONDETERMINISTIC")
    I, L, R = _mk_seed(43, good_i, good_l)
    L[1] = _mk_session(good_l[1], 0, 10, a3_noise_seed(43, "lesion", 1))
    c("a3: a lesion session whose draw was not ablated -> UNDEFINED", score_seed_a3(I, L, R)["verdict"] == "UNDEFINED")
    I, L, R = _mk_seed(43, good_i, good_l)
    L[2] = _mk_session(good_l[2], 10, 10, a3_noise_seed(43, "lesion", 2), competes=0)
    c("a3: a session that never drew on its noise stream -> UNDEFINED",
      score_seed_a3(I, L, R)["verdict"] == "UNDEFINED")
    I, L, R = _mk_seed(43, good_i, good_l)
    L[0] = _mk_session(good_l[0], 10, 10, a3_noise_seed(43, "lesion", 0), w=dict(W_S42, deer=9.0))
    c("a3: host weight vector differs across sessions -> UNDEFINED", score_seed_a3(I, L, R)["verdict"] == "UNDEFINED")
    c("a3: missing session -> ARM-FAILED", score_seed_a3(I[:3], L, R)["verdict"] == "ARM-FAILED")
    c("a3: reversed arms -> DEFINED, AWAY", score_seed_a3(*_mk_seed(43, good_l, good_i))["label"] == "AWAY")
    rec = lambda d: {"verdict": "DEFINED", "delta": d}
    six = {str(s): rec(0.3) for s in A3_SEEDS}
    c("a3 aggregate: 6/6 positive, mean >= floor -> GO (p = 1/64)",
      aggregate_a3(six)["GO"] and abs(aggregate_a3(six)["p_sign_test"] - 1 / 64.) < 1e-12)
    c("a3 aggregate: 5 positive + 1 negative -> no GO (p = 7/64)",
      not aggregate_a3(dict(six, **{"44": rec(-0.1)}))["GO"])
    c("a3 aggregate: one UNDEFINED seed -> no GO",
      not aggregate_a3(dict(six, **{"44": {"verdict": "UNDEFINED"}}))["GO"])
    c("a3 aggregate: only 5 seeds -> no GO", not aggregate_a3({k: v for k, v in six.items() if k != "102"})["GO"])
    c("a3 aggregate: 6/6 positive but mean below the effect floor -> no GO",
      not aggregate_a3({str(s): rec(0.05) for s in A3_SEEDS})["GO"])
    # THE NULL IS REAL: an A/A world (both arms iid from the SAME session-value distribution, the noise varying the
    # value) must GO at most at the nominal rate. 4000 simulated 6-seed experiments, seeded.
    rng = random.Random(20260923)
    n_go = 0
    for _ in range(4000):
        per = {}
        for s in A3_SEEDS:
            vals = [rng.choice([1 / 3., 2 / 3., 2 / 3., 1.0, 0.0]) for _ in range(2 * A3_SESSIONS)]
            d = sum(vals[:A3_SESSIONS]) / A3_SESSIONS - sum(vals[A3_SESSIONS:]) / A3_SESSIONS
            per[str(s)] = rec(d)
        n_go += aggregate_a3(per)["GO"]
    c("a3 aggregate: A/A null world GOes at <= alpha (%d/4000)" % n_go, n_go / 4000. <= ALPHA)
    return ok0[0]


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
    ok = selftest_a3(chk) and ok
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
    # AMENDMENT 3 (the design with a real null): one session per invocation (pool job), then --a3-score locally
    ap.add_argument("--a3-session", action="store_true",
                    help="run ONE amendment-3 session in this process: --mode --seed --arm --session [--k --out-dir]")
    ap.add_argument("--arm", default=None, choices=sorted(A3_ARMS))
    ap.add_argument("--session", type=int, default=0, help="session index j (0..A3_SESSIONS-1)")
    ap.add_argument("--a3-score", action="store_true", help="score amendment-3 sessions in --out-dir for --seeds")
    ap.add_argument("--code-sha", default=None, help="commit the sessions ran at (recorded in the aggregate)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.a3_session:
        if a.arm is None or not (0 <= a.session < A3_SESSIONS) or (a.arm == "intact_rebuild" and a.session != 0):
            ap.error("--a3-session needs --arm and a --session in [0, %d) (the rebuild is session 0 only)"
                     % A3_SESSIONS)
        out_dir = a.out_dir or "research/findings/raw/_load_bearing/_oe_production_turn/a3/%s" % a.mode
        k = a.k if a.k != 40 else A3_K
        if k != A3_K:
            ap.error("amendment 3 fixes K = %d asks per session" % A3_K)
        return run_a3_session(a.mode, a.seed, a.arm, a.session, k, out_dir)
    if a.a3_score:
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else list(A3_SEEDS)
        out_dir = a.out_dir or "research/findings/raw/_load_bearing/_oe_production_turn/a3/%s" % a.mode
        score_a3(a.mode, seeds, out_dir, aggregate_out=a.aggregate_out, sha=a.code_sha)
        return 0
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
