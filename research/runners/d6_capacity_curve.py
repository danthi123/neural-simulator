"""D6 CAPACITY CURVE -- how does learn-through-use scale with the NUMBER of facts taught in conversation?

WHY (owner question, 2026-09-23: "given you mentioned small scale, I'm curious if the learning we're proving here
scales to the levels needed to go head to head with even a tiny llm?"). The D6 gate v3 GO 6/6
(research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md) showed ONE fact, taught once in chat,
stored by a LOCAL spiking Hebbian write (research/runners/d6_hebbian_store.py) on the tiny-demo brain. This instrument
asks the scaling question the gate could not: as N facts are taught, how do recall accuracy, interference
(false / crossed recall), memory per fact and encode + recall time per fact move -- for the D6 Hebbian write versus
the host pattern copy it replaces (all D6 flags 0)?

PRE-REGISTRATION: research/findings/2026-09-23-d6-capacity-curve-PREREGISTRATION.md (committed on its own, before any
run of this file). The scorer below (`score_grid`) implements that document's bands verbatim.

AMENDMENT B (filed after adversarial review of the smoke + resource-probe runs, before any grid job): the accuracy
gate (recall >= 0.90, both false rates <= 0.05) CANNOT FAIL for this store by construction -- one disjoint
trigger->readout block per fact, exact host (agent, action) DG-shard routing, a fixed cleanup codebook, and
k_max = N+16 sized from the start mean nothing that changes with N ever reaches the decode. So the labels below are
renamed to say what they actually measure: "SCALES" -> "RECALL-HOLDS" (recall holding flat is expected and
uninformative about capacity), "PARITY" -> "PARITY-BY-CONSTRUCTION" (HEBB vs COPY store weights measured complex
corr 0.99996, max |dw| 0.031, all 128 synapses saturated at W_MAX at s42 N=5 -- the instructive pathway + phase
lock + W_MAX clamp make the Hebbian write a near-copy of the host pattern, so parity is predetermined, not evidence
the learning rule scales). A NEW pre-registered CONSUMER-HARDWARE COST criterion is added (COST_ENCODE_MAX_S,
COST_READTIME_MAX_S, COST_MEM_GB below) because cost is the one quantity that DOES grow with N here (the prereg's
own prediction 3) and is the only band that can produce a real scaling verdict; `score_grid` now also reports
level_label_*_cost / curve_*_cost alongside the (renamed, non-scaling) recall labels. See AMENDMENT LOG in the
PREREGISTRATION doc for the full account and the adversarial review this responds to.

WHAT ONE JOB DOES (one (seed, N, arm) per process; numpy/CPU; `--worker`):
  1. LEXICON: a fixed per-seed synthetic vocabulary (N_AG agents, N_AC actions, N_PT patients; disjoint pools), the
     SAME for every level N -- the brain knows the words; what grows is the number of FACTS taught over them.
  2. FACTS: a per-seed MASTER list of 2000 SVO facts in blocks of 4 with CONTROLLED OVERLAP:
       (a, v1, p) (a, v2, q) (b, v3, p) (b, v4, q)
     -- every fact has a same-subject sibling and a same-object sibling inside its block; agents/patients are drawn
     with replacement across blocks, so fan per word grows with N. Every (agent, action) pair is unique. Level N
     teaches the first N facts (the levels are NESTED prefixes of one list).
  3. BRAIN: the production OneBrainComposer (the chat brain's composer class, with the chat's production arguments:
     D=128, enable_spiking_cleanup=True, vocab_headroom=128, integrated_loop off, fact-shard retrieval on), store
     sized k_max = N + 16. cfg.seed IS set (OneBrainComposer -> build_coresident_bridge sets cfg.seed = seed); the
     firing-threshold array is hashed into the output and the scorer requires it identical across the arms of a cell.
  4. TEACH: each fact is taught by the call the chat's acquisition path makes (`ChatBrain._maybe_acquire` ->
     `inner.hear("a v p", polarity=...)` inside `d6_hebbian_store.conversation_write(composer)`): the on-bridge parser
     assigns the roles by firing, then the store write runs -- the Hebbian rule (HEBB), eta=0 (FREEZE) or the host
     composite copy (COPY).
  5. PROBE (identical probe set for every arm of a (seed, N) cell; fixed by seed and N):
       taught      n_probe = min(N, 100) taught facts: query_patient(a, v) (the recall), ask_yes_no(a, v, p) (the
                   yes/no hit), the block's own substrate decode (all 3 roles, NO routing), the engram read
                   (d6_hebbian_store.engram_held) and the block's learned mean |w| (the lever).
       near-miss   one per taught probe: ask_yes_no(a, v, q) with q a stored patient != p -- preferably the patient of
                   a SAME-SUBJECT sibling (shared subject, different object). "yes" = a false accept.
                   AMENDMENT B CORRECTION: this probe does NOT measure cross-fact interference. `ask_yes_no` routes
                   on (agent, action) only (`_fact_shard_yesno_match`), which is the SAME shard as the taught fact
                   itself -- in the N=50 smoke, 37/50 near-miss shards contained only the true block. The sibling
                   (a, v2, q) can enter that shard only via a DG bucket collision, and even then is rejected because
                   its own decoded action is v2, not v. So a false accept here requires the TRUE block's patient to
                   decode as q -- that is substrate decode noise, not the sibling fact interfering. This probe is
                   kept (it is a real decode-noise measure and the near-miss shard-emptiness split is informative),
                   but its false-accept rate is no longer described as "the interference measure this instrument
                   credits to the brain". A routing-free or superposed-store arm would be needed to measure actual
                   cross-fact interference; none is run here (see AMENDMENT B).
       novel       n_probe never-taught (agent, action) pairs whose words are both stored in those roles:
                   query_patient -> any non-None answer = a false recall.
  6. RECORD: every per-probe decision, per-fact encode wall time, per-query wall time, RSS at stages and the process
     peak (ru_maxrss), n_total, store synapse count, the DG fact-shard size seen by each query (so an abstain that the
     HOST routing produced -- an empty shard -- is separable from one the substrate decode produced).

ARMS (explicit env values in BOTH directions, never a pop):
  HEBB      BRAIN_D6_HEBBIAN_STORE=1 BRAIN_D6_ENGRAM_VOCAB=1 BRAIN_D6_ENGRAM_READTIME=1 BRAIN_D6_HEBBIAN_FREEZE=0
  FREEZE    as HEBB + BRAIN_D6_HEBBIAN_FREEZE=1   (the null: same encode episode, eta=0 -> must fall to abstain)
  COPY      all four D6 flags 0                   (the production host pattern copy)
  HEBB_REP  identical to HEBB, rebuilt            (determinism null; only at N in REP_LEVELS)

DECLARED HOST SHORTCUTS / SCOPE (each named, none credited to the brain):
  * The fact-shard ROUTING (`_fact_shard_candidates`) intersects DG buckets built from the HOST kb record (the declared
    DG host-rate stand-in, research/biology/dg-ca3-sparse-index.md, and d6_hebbian_store shortcut (h)). A novel-foil
    abstain with an EMPTY shard is therefore a host-routing abstain; the output separates it (`shard_empty`).
    `block_decode_ok` is the routing-free substrate measure.
  * ENGRAM_VOCAB / ENGRAM_READTIME change only chat-level readers, which this composer-level instrument does not run;
    they are set in HEBB/FREEZE to match the D6 configuration and are inert here. The read-time view's COST at N is
    reported as a projection: N x the measured per-block engram-read time (the view re-reads every block when the store
    changes, i.e. after every teach turn).
  * No DA encoding gain (encoding_gain_fn=None, g=1): d6_hebbian_store residual (f) unmeasured here too.
  * Words are pre-known (lexicon built in); runtime word recruitment (vocab_headroom) is a separate capacity axis.
  * The chat's B3 polarity extractor / verb lemmatizer are bypassed (facts are generated as base-form "a v p").

Selftest (no brain):   .venv/bin/python -m research.runners.d6_capacity_curve --selftest
One job:               SIM_BACKEND=numpy .venv/bin/python -m research.runners.d6_capacity_curve --worker \
                           --seed 42 --n-facts 5 --arm HEBB --out <dir>/s42_N5_HEBB.json
Resource probe:        ... --resource-probe --seed 42 --n-facts 2000 --arm HEBB --probe-k 5 --out <file>
Score a grid:          .venv/bin/python -m research.runners.d6_capacity_curve --score --arm-dir <dir> \
                           --json <dir>/d6_capacity_verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

SEEDS6 = [42, 43, 44, 100, 101, 102]
LEVELS = [5, 50, 500, 2000]
REP_LEVELS = [5, 50]
N_MASTER = 2000
N_AG, N_AC, N_PT = 220, 60, 220          # lexicon pools; 220*60 = 13200 (agent, action) pairs >> 2000 + foils
N_PROBE_MAX = 100
D = 128
VOCAB_HEADROOM = 128                      # the production onebrain chat default (brain_conversational_agent)
K_MAX_PAD = 16

# ── pre-registered thresholds (research/findings/2026-09-23-d6-capacity-curve-PREREGISTRATION.md,
#    AMENDMENT B for the cost criterion) ──────────────────────────────────────────────────────────────────────────
RECALL_MIN = 0.90
FALSE_MAX = 0.05
NULL_MAX = 0.05
PARITY_TOL = 0.05
MEM_NODE_GB, MEM_AWS_GB = 13.0, 120.0     # usable RAM on a 15 GB pool node / a 128 GB AWS node (2 / 8 GB margin)

# AMENDMENT B (consumer-hardware cost criterion; the only pre-registered band that can fail for this store's
# design -- see the docstring's AMENDMENT B section and the PREREGISTRATION's own amendment log). Bar =
# "runnable as a chat turn on a single-consumer-GPU-class box" (project consumer-hardware-reference standard):
COST_ENCODE_MAX_S = 2.0          # per-fact encode (median), one taught turn
COST_READTIME_MAX_S = 10.0       # projected read-time-view cost per chat turn (engram re-read of every block)
COST_MEM_GB = 24.0               # peak RSS budget (a single consumer GPU-class box's RAM headroom, project standard)

D6_FLAGS = ("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME", "BRAIN_D6_HEBBIAN_FREEZE")
_HEBB = {"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_READTIME": "1",
         "BRAIN_D6_HEBBIAN_FREEZE": "0"}
ARMS = {
    "HEBB": dict(_HEBB),
    "FREEZE": dict(_HEBB, BRAIN_D6_HEBBIAN_FREEZE="1"),
    "COPY": {k: "0" for k in D6_FLAGS},
    "HEBB_REP": dict(_HEBB),
}
COMMON_ENV = {"SIM_BACKEND": "numpy", "BRAIN_FACT_SHARD_RETRIEVAL": "1", "BRAIN_D6_ENGRAM_PRUNE": "0"}


# ── facts + probes (pure; deterministic in seed) ────────────────────────────────────────────────────────────────────
def lexicon():
    ag = ["ag%03d" % i for i in range(N_AG)]
    ac = ["ac%02d" % i for i in range(N_AC)]
    pt = ["pt%03d" % i for i in range(N_PT)]
    return ag, ac, pt


def make_master(seed, n=N_MASTER):
    """The per-seed master fact list (blocks of 4 with a same-subject and a same-object sibling per fact)."""
    ag, ac, pt = lexicon()
    rng = np.random.default_rng(int(seed) * 7919 + 17)
    facts, pairs = [], set()
    guard = 0
    while len(facts) < n:
        guard += 1
        if guard > 100000:
            raise RuntimeError("make_master: could not place %d facts" % n)
        a, b = rng.choice(N_AG, size=2, replace=False)
        p, q = rng.choice(N_PT, size=2, replace=False)
        vs = rng.choice(N_AC, size=4, replace=False)
        blk = [(ag[a], ac[vs[0]], pt[p]), (ag[a], ac[vs[1]], pt[q]),
               (ag[b], ac[vs[2]], pt[p]), (ag[b], ac[vs[3]], pt[q])]
        if any((x, v) in pairs for (x, v, _y) in blk):
            continue
        for f in blk:
            pairs.add((f[0], f[1]))
        facts.extend(blk)
    return facts[:n]


def make_probes(facts, seed, n_facts):
    """Probe set for a (seed, N) cell: identical for every arm. Returns dict with taught / nearmiss / novel lists."""
    rng = np.random.default_rng(int(seed) * 104729 + int(n_facts))
    N = len(facts)
    n_probe = min(N, N_PROBE_MAX)
    idx = sorted(int(i) for i in rng.choice(N, size=n_probe, replace=False))
    taught_pairs = {(a, v) for (a, v, _p) in facts}
    taught_svo = set(facts)
    by_agent = {}
    for (a, v, p) in facts:
        by_agent.setdefault(a, []).append((v, p))
    stored_patients = sorted({p for (_a, _v, p) in facts})
    stored_actions = sorted({v for (_a, v, _p) in facts})
    taught, nearmiss, novel = [], [], []
    novel_seen = set()
    for i in idx:
        a, v, p = facts[i]
        taught.append({"i": i, "agent": a, "action": v, "patient": p})
        sib = sorted({q for (_v2, q) in by_agent[a] if q != p and (a, v, q) not in taught_svo})
        if sib:
            q, kind = sib[int(rng.integers(len(sib)))], "same_subject_sibling"
        else:
            cand = [q for q in stored_patients if q != p and (a, v, q) not in taught_svo]
            q, kind = (cand[int(rng.integers(len(cand)))] if cand else None), "random_stored"
        nearmiss.append({"i": i, "agent": a, "action": v, "patient": q, "kind": kind})
        cand_v = [x for x in stored_actions if (a, x) not in taught_pairs and (a, x) not in novel_seen]
        if not cand_v:                                         # agent paired with every stored action: any other agent
            pool = [(a2, x) for a2 in sorted(by_agent) for x in stored_actions
                    if (a2, x) not in taught_pairs and (a2, x) not in novel_seen]
            a2, x = pool[int(rng.integers(len(pool)))]
        else:
            a2, x = a, cand_v[int(rng.integers(len(cand_v)))]
        novel_seen.add((a2, x))
        novel.append({"agent": a2, "action": x})
    spec = {"taught": taught, "nearmiss": nearmiss, "novel": novel}
    spec["hash"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


# ── worker ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def _rss_mb():
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def _peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _set_env(arm):
    env = dict(COMMON_ENV)
    env.update(ARMS[arm])
    for k, v in env.items():
        os.environ[k] = v
    return env


def _build(seed, n_facts):
    import random
    random.seed(int(seed)); np.random.seed(int(seed))
    from research.runners.one_brain_composer import OneBrainComposer
    ag, ac, pt = lexicon()
    comp = OneBrainComposer(seed=int(seed), D=D, vocab=sorted(ag + ac + pt), k_max=int(n_facts) + K_MAX_PAD,
                            enable_spiking_cleanup=True, vocab_headroom=VOCAB_HEADROOM, integrated_loop=False)
    from sim.backend import to_host
    thr = np.asarray(to_host(comp.b.cp_neuron_firing_thresholds))
    return comp, hashlib.sha256(thr.tobytes()).hexdigest()


def _teach(comp, facts, out):
    from research.runners.d6_hebbian_store import conversation_write
    enc_t, parse_err, enc_diag = [], 0, []
    for (a, v, p) in facts:
        t0 = time.perf_counter()
        with conversation_write(comp):
            fact = comp.hear("%s %s %s" % (a, v, p), polarity="AFFIRM")
        enc_t.append(time.perf_counter() - t0)
        if (fact.get("agent"), fact.get("action"), fact.get("patient")) != (a, v, p):
            parse_err += 1
        d = getattr(comp, "_d6_last_encode", None)
        if d is not None:
            enc_diag.append((bool(d.get("frozen")), int(d.get("n_saturated", -1)), float(d.get("mean_abs_w", -1.0))))
    out["encode_s"] = [round(x, 4) for x in enc_t]
    out["parse_errors"] = parse_err
    out["kb_len_after_teach"] = len(comp.kb)
    if enc_diag:
        out["encode_diag"] = {"n": len(enc_diag), "n_frozen": sum(1 for f, _s, _m in enc_diag if f),
                              "n_saturated_min": min(s for _f, s, _m in enc_diag),
                              "mean_abs_w_mean": float(np.mean([m for _f, _s, m in enc_diag]))}
    else:
        out["encode_diag"] = None


def _block_mean_abs_w(comp, i):
    ws = comp.store_conns[i * comp.D:(i + 1) * comp.D]
    return float(np.mean([abs(complex(t[2])) for t in ws])) if ws else None


def _shard(comp, cue):
    try:
        s = comp._fact_shard_candidates(cue)
    except Exception as e:                                    # instrument only; never changes a decision
        return {"error": "%s: %s" % (type(e).__name__, e)}
    return None if s is None else len(s)


def _probe(comp, facts, spec, out):
    from research.runners.d6_hebbian_store import engram_held
    patients_of_agent = {}
    for (a, _v, p) in facts:
        patients_of_agent.setdefault(a, set()).add(p)
    stored_patients = {p for (_a, _v, p) in facts}
    writes = []
    _orig = comp._write_block

    def _counted(bi, zc, _o=_orig):
        writes.append(int(bi))
        return _o(bi, zc)
    comp._write_block = _counted
    rows_t, rows_n, rows_v = [], [], []
    for pr in spec["taught"]:
        a, v, p, i = pr["agent"], pr["action"], pr["patient"], pr["i"]
        sh = _shard(comp, {"agent": a, "action": v})
        t0 = time.perf_counter(); ans = comp.query_patient(a, v); tq = time.perf_counter() - t0
        t0 = time.perf_counter(); yn = comp.ask_yes_no(a, v, p); ty = time.perf_counter() - t0
        got = comp._read_one_block(i)
        blk_ok = (got.get("agent") == a and got.get("action") == v and got.get("patient") == p)
        t0 = time.perf_counter(); eh = engram_held(comp, i); te = time.perf_counter() - t0
        if ans == p:
            cls = "correct"
        elif ans is None:
            cls = "abstain"
        elif ans in patients_of_agent.get(a, ()):
            cls = "crossed_same_subject"
        elif ans in stored_patients:
            cls = "crossed_other_fact"
        else:
            cls = "wrong_word"
        rows_t.append({"i": i, "answer": ans, "cls": cls, "yn": yn, "block_ok": bool(blk_ok),
                       "block_decode": {k: got.get(k) for k in ("agent", "action", "patient")},
                       "held": bool(eh["held"]), "readout": round(float(eh["readout"]), 6),
                       "mean_abs_w": _block_mean_abs_w(comp, i), "shard": sh,
                       "t_query": round(tq, 4), "t_yn": round(ty, 4), "t_engram": round(te, 4)})
    for pr in spec["nearmiss"]:
        if pr["patient"] is None:
            rows_n.append({"i": pr["i"], "yn": None, "kind": pr["kind"], "void": True})
            continue
        # the shard ask_yes_no itself routes on: (agent, action) -- `_fact_shard_yesno_match` (AMENDMENT 1)
        sh = _shard(comp, {"agent": pr["agent"], "action": pr["action"]})
        t0 = time.perf_counter(); yn = comp.ask_yes_no(pr["agent"], pr["action"], pr["patient"])
        rows_n.append({"i": pr["i"], "yn": yn, "kind": pr["kind"], "shard": sh,
                       "t_yn": round(time.perf_counter() - t0, 4)})
    for pr in spec["novel"]:
        sh = _shard(comp, {"agent": pr["agent"], "action": pr["action"]})
        t0 = time.perf_counter(); ans = comp.query_patient(pr["agent"], pr["action"])
        rows_v.append({"agent": pr["agent"], "action": pr["action"], "answer": ans, "shard": sh,
                       "t_query": round(time.perf_counter() - t0, 4)})
    comp._write_block = _orig
    out["probes"] = {"taught": rows_t, "nearmiss": rows_n, "novel": rows_v}
    out["store_writes_during_probe"] = writes


def summarize(out):
    """Per-job summary metrics (also recomputed by the scorer from the raw rows)."""
    P = out.get("probes") or {}
    T, NM, NV = P.get("taught") or [], [r for r in (P.get("nearmiss") or []) if not r.get("void")], P.get("novel") or []
    f = lambda xs, pred: (sum(1 for x in xs if pred(x)) / len(xs)) if xs else None  # noqa: E731
    enc = out.get("encode_s") or []
    tq = [r["t_query"] for r in T] + [r["t_query"] for r in NV]
    s = {
        "n_taught_probes": len(T), "n_nearmiss": len(NM), "n_novel": len(NV),
        "recall": f(T, lambda r: r["cls"] == "correct"),
        "abstain": f(T, lambda r: r["cls"] == "abstain"),
        "crossed_same_subject": f(T, lambda r: r["cls"] == "crossed_same_subject"),
        "crossed_other_fact": f(T, lambda r: r["cls"] == "crossed_other_fact"),
        "wrong_word": f(T, lambda r: r["cls"] == "wrong_word"),
        "yn_hit": f(T, lambda r: r["yn"] == "yes"),
        "block_decode_ok": f(T, lambda r: r["block_ok"]),
        "engram_held": f(T, lambda r: r["held"]),
        "mean_abs_w_probed": (float(np.mean([r["mean_abs_w"] for r in T])) if T else None),
        "nearmiss_false_accept": f(NM, lambda r: r["yn"] == "yes"),
        "novel_false_recall": f(NV, lambda r: r["answer"] is not None),
        "novel_shard_empty": f(NV, lambda r: r.get("shard") == 0),
        "nearmiss_shard_empty": f(NM, lambda r: r.get("shard") == 0),
        "encode_s_median": float(np.median(enc)) if enc else None,
        "encode_s_mean": float(np.mean(enc)) if enc else None,
        "encode_s_last_decile_mean": float(np.mean(enc[-max(1, len(enc) // 10):])) if enc else None,
        "query_s_median": float(np.median(tq)) if tq else None,
        "engram_read_s_median": float(np.median([r["t_engram"] for r in T])) if T else None,
    }
    if s["engram_read_s_median"] is not None:
        s["readtime_view_s_per_turn_projected"] = s["engram_read_s_median"] * int(out.get("n_facts", 0))
    return s


def worker(seed, n_facts, arm, out_path, resource_probe=False, probe_k=5):
    env = _set_env(arm)
    t_all = time.time()
    out = {"runner": "research.runners.d6_capacity_curve", "seed": int(seed), "n_facts": int(n_facts), "arm": arm,
           "env": env, "resource_probe": bool(resource_probe), "D": D, "lexicon": [N_AG, N_AC, N_PT],
           "vocab_headroom": VOCAB_HEADROOM, "rss_mb": {}, "error": None}
    out["rss_mb"]["start"] = _rss_mb()
    try:
        facts = make_master(seed)[:int(n_facts)]
        spec = make_probes(facts, seed, n_facts)
        out["probe_spec_hash"] = spec["hash"]
        out["facts_hash"] = hashlib.sha256(json.dumps(facts).encode()).hexdigest()
        t0 = time.time()
        comp, thr_hash = _build(seed, n_facts)
        out["build_s"] = round(time.time() - t0, 2)
        out["substrate_threshold_sha256"] = thr_hash
        out["n_total"] = int(comp.n_total)
        out["k_max"] = int(comp.k_max)
        out["fact_shard_active_flag"] = bool(comp.enable_fact_shard)
        out["rss_mb"]["after_build"] = _rss_mb()
        if resource_probe:
            _teach(comp, facts[:int(probe_k)], out)
            out["rss_mb"]["after_teach"] = _rss_mb()
            # RESOURCE PROBE (fit decision only, never scored): the full N-sized bridge, probe_k taught facts, three
            # taught-fact queries -> the build RSS, per-encode and per-query wall time AT THIS n_total.
            sub = {"taught": [{"i": i, "agent": a, "action": v, "patient": p}
                              for i, (a, v, p) in enumerate(facts[:min(3, int(probe_k))])],
                   "nearmiss": [], "novel": []}
            _probe(comp, facts[:int(probe_k)], sub, out)
        else:
            _teach(comp, facts, out)
            out["rss_mb"]["after_teach"] = _rss_mb()
            _probe(comp, facts, spec, out)
        out["rss_mb"]["after_probe"] = _rss_mb()
        out["store_synapses"] = len(comp.store_conns)
        out["store_weight_bytes_complex128"] = 16 * len(comp.store_conns)
        out["summary"] = summarize(out)
    except Exception as e:                                      # recorded; the scorer VOIDs this job
        import traceback
        out["error"] = "%s: %s" % (type(e).__name__, e)
        out["traceback"] = traceback.format_exc()
    out["peak_rss_mb"] = _peak_mb()
    out["elapsed_s"] = round(time.time() - t_all, 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("[d6cap] s%s N=%s %s -> %s  (%.0fs, peak %.0f MB, err=%s)" % (seed, n_facts, arm, out_path,
          out["elapsed_s"], out["peak_rss_mb"], out["error"]), flush=True)
    return 0 if out["error"] is None else 4


# ── scoring (the pre-registered bands) ─────────────────────────────────────────────────────────────────────────────
def _job_path(arm_dir, seed, n, arm):
    return os.path.join(arm_dir, "s%d_N%d_%s.json" % (int(seed), int(n), arm))


def _load(p):
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def _decisions(job):
    P = job.get("probes") or {}
    return ([(r["answer"], r["yn"], r["block_ok"]) for r in P.get("taught") or []]
            + [r.get("yn") for r in P.get("nearmiss") or []] + [r["answer"] for r in P.get("novel") or []])


def _recall_holds(s):
    """Renamed from `_scales` under AMENDMENT B: this cannot fail for a localist, uniquely-routed store, so it is
    reported as RECALL-HOLDS, never as evidence of capacity SCALING."""
    return (s["recall"] is not None and s["recall"] >= RECALL_MIN
            and s["novel_false_recall"] is not None and s["novel_false_recall"] <= FALSE_MAX
            and s["nearmiss_false_accept"] is not None and s["nearmiss_false_accept"] <= FALSE_MAX)


def _cost_ok(s, peak_rss_mb):
    """AMENDMENT B: the pre-registered consumer-hardware cost criterion. This CAN fail (and is predicted to, at
    N=2000: prereg predicts ~12.6 s/fact encode and a ~3923 s/turn projected read-time view)."""
    if s.get("encode_s_median") is None or s.get("readtime_view_s_per_turn_projected") is None or peak_rss_mb is None:
        return None
    return (s["encode_s_median"] <= COST_ENCODE_MAX_S
            and s["readtime_view_s_per_turn_projected"] <= COST_READTIME_MAX_S
            and (peak_rss_mb / 1024.0) <= COST_MEM_GB)


def score_cell(jobs, n):
    """jobs: {arm: job|None} for one (seed, N). Returns the per-cell record."""
    from tools.lab import void_if
    rec = {"void": [], "n": int(n)}
    need = ["HEBB", "FREEZE", "COPY"] + (["HEBB_REP"] if int(n) in REP_LEVELS else [])
    for arm in need:
        j = jobs.get(arm)
        bad = (j is None or j.get("error") is not None or j.get("resource_probe")
               or j.get("kb_len_after_teach") != int(n) or not (j.get("probes") or {}).get("taught"))
        if void_if(bad, "cell N=%s arm %s missing / errored / incomplete" % (n, arm)):
            rec["void"].append(arm)
    if rec["void"]:
        rec["status"] = "UNDEFINED"
        return rec
    H, F, C = jobs["HEBB"], jobs["FREEZE"], jobs["COPY"]
    ident = {k: len({jobs[a].get(k) for a in need}) == 1
             for k in ("substrate_threshold_sha256", "probe_spec_hash", "facts_hash", "n_total")}
    rec["cross_arm_identity"] = ident
    if not all(ident.values()):
        rec["status"] = "UNDEFINED"
        rec["why"] = "arms of this cell do not share substrate / facts / probes"
        return rec
    sH, sF, sC = summarize(H), summarize(F), summarize(C)
    rec["summary"] = {"HEBB": sH, "FREEZE": sF, "COPY": sC}
    null_ok = (sF["recall"] is not None and sF["recall"] <= NULL_MAX
               and sF["novel_false_recall"] is not None and sF["novel_false_recall"] <= NULL_MAX
               and sF["nearmiss_false_accept"] is not None and sF["nearmiss_false_accept"] <= NULL_MAX
               and sF["mean_abs_w_probed"] == 0.0)
    rec["freeze_null_ok"] = bool(null_ok)
    rec["lever_moved"] = bool(sH["mean_abs_w_probed"] is not None and sH["mean_abs_w_probed"] > 0.5
                              and sF["mean_abs_w_probed"] == 0.0)
    rec["writes_during_probe_clean"] = all(not jobs[a].get("store_writes_during_probe") for a in need)
    rec["rep_identical"] = (None if "HEBB_REP" not in need else _decisions(H) == _decisions(jobs["HEBB_REP"]))
    if not (null_ok and rec["lever_moved"] and rec["writes_during_probe_clean"] and rec["rep_identical"] is not False):
        rec["status"] = "UNDEFINED"
        rec["why"] = "null / lever / probe-write / determinism check failed"
        rec["HEBB_recall_holds"] = _recall_holds(sH)            # reported, not scored (AMENDMENT B naming)
        rec["COPY_recall_holds"] = _recall_holds(sC)
        rec["HEBB_cost_ok"] = _cost_ok(sH, H.get("peak_rss_mb"))
        rec["COPY_cost_ok"] = _cost_ok(sC, C.get("peak_rss_mb"))
        return rec
    rec["status"] = "DEFINED"
    rec["HEBB_recall_holds"] = _recall_holds(sH)
    rec["COPY_recall_holds"] = _recall_holds(sC)
    rec["HEBB_cost_ok"] = _cost_ok(sH, H.get("peak_rss_mb"))
    rec["COPY_cost_ok"] = _cost_ok(sC, C.get("peak_rss_mb"))
    dr = sH["recall"] - sC["recall"]
    dfalse = max(abs(sH["novel_false_recall"] - sC["novel_false_recall"]),
                 abs(sH["nearmiss_false_accept"] - sC["nearmiss_false_accept"]))
    rec["recall_diff_HEBB_minus_COPY"] = dr
    # AMENDMENT B: PARITY (when it holds) is BY CONSTRUCTION -- see the module docstring for the s42 N=5 measurement
    # (complex corr 0.99996, all 128 synapses saturated). Not evidence the local learning rule scales.
    rec["parity"] = bool(abs(dr) <= PARITY_TOL and dfalse <= PARITY_TOL)
    return rec


def _level_label(cells, key, true_label="RECALL-HOLDS", false_label="FAILS"):
    """cells: list of 6 per-seed cell records. key e.g. 'HEBB_recall_holds' | 'COPY_recall_holds' | 'HEBB_cost_ok'.
    true_label/false_label let the SAME exhaustive band structure serve both the (by-construction, AMENDMENT B)
    recall labels and the new cost labels without conflating the two kinds of claim."""
    if len(cells) < 6:
        return "INCOMPLETE"
    if any(c.get("status") != "DEFINED" for c in cells):
        return "UNDEFINED"
    if any(c.get(key) is None for c in cells):
        return "UNDEFINED"
    k = sum(1 for c in cells if c[key])
    return true_label if k == 6 else (false_label if k == 0 else "MIXED(%d/6)" % k)


def _parity_label(cells):
    if len(cells) < 6 or any(c.get("status") != "DEFINED" for c in cells):
        return "UNDEFINED"
    if all(c["parity"] for c in cells):
        return "PARITY-BY-CONSTRUCTION"                         # AMENDMENT B: see module docstring for the measurement
    worse = sum(1 for c in cells if c["recall_diff_HEBB_minus_COPY"] < -PARITY_TOL)
    better = sum(1 for c in cells if c["recall_diff_HEBB_minus_COPY"] > PARITY_TOL)
    if worse >= 4:
        return "HEBB-WORSE"
    if better >= 4:
        return "HEBB-BETTER"
    return "PARITY-MIXED"


def curve_verdict(labels, levels, true_label="RECALL-HOLDS", false_label="FAILS"):
    """labels: {N: level label}. The exhaustive pre-registered curve bands, parameterized by which level-label
    vocabulary is in play (recall's RECALL-HOLDS/FAILS, or cost's COST-HOLDS/COST-FAILS)."""
    ls = [labels.get(n) for n in levels]
    if any(x in (None, "INCOMPLETE", "UNDEFINED") for x in ls):
        return "UNDEFINED"
    ok = [x == true_label for x in ls]
    if all(ok):
        return "%s-TO-%d" % (true_label, levels[-1])
    k = 0
    while k < len(ok) and ok[k]:
        k += 1
    if not any(ok[k:]):
        return ("%s-FROM-%d" % (false_label, levels[0]) if k == 0
                else "CEILING-BETWEEN-%d-AND-%d" % (levels[k - 1], levels[k]))
    return "NON-MONOTONE"


def score_grid(arm_dir, seeds=SEEDS6, levels=None):
    levels = list(levels or LEVELS)
    per = {}
    for n in levels:
        per[str(n)] = {}
        for s in seeds:
            jobs = {a: _load(_job_path(arm_dir, s, n, a)) for a in ARMS}
            per[str(n)][str(s)] = score_cell(jobs, n)
    labels_H = {n: _level_label(list(per[str(n)].values()), "HEBB_recall_holds") for n in levels}
    labels_C = {n: _level_label(list(per[str(n)].values()), "COPY_recall_holds") for n in levels}
    parity = {n: _parity_label(list(per[str(n)].values())) for n in levels}
    # AMENDMENT B: the cost curve is the one band this instrument can actually fail on (see module docstring).
    cost_H = {n: _level_label(list(per[str(n)].values()), "HEBB_cost_ok", "COST-HOLDS", "COST-FAILS") for n in levels}
    cost_C = {n: _level_label(list(per[str(n)].values()), "COPY_cost_ok", "COST-HOLDS", "COST-FAILS") for n in levels}
    return {"runner": "research.runners.d6_capacity_curve", "arm_dir": arm_dir, "seeds": list(seeds),
            "levels": levels, "per_level": per,
            "level_label_HEBB": {str(k): v for k, v in labels_H.items()},
            "level_label_COPY": {str(k): v for k, v in labels_C.items()},
            "parity_HEBB_vs_COPY": {str(k): v for k, v in parity.items()},
            "curve_HEBB": curve_verdict(labels_H, levels), "curve_COPY": curve_verdict(labels_C, levels),
            "level_label_HEBB_cost": {str(k): v for k, v in cost_H.items()},
            "level_label_COPY_cost": {str(k): v for k, v in cost_C.items()},
            "curve_HEBB_cost": curve_verdict(cost_H, levels, "COST-HOLDS", "COST-FAILS"),
            "curve_COPY_cost": curve_verdict(cost_C, levels, "COST-HOLDS", "COST-FAILS")}


# ── self-test: every failing direction must fail ────────────────────────────────────────────────────────────────────
def _syn_job(n, arm, recall=1.0, novel_false=0.0, nm_false=0.0, w=1.0, rep_flip=False, thr="t", err=None,
             t_engram=0.1, encode_s=0.1, peak_rss_mb=300.0):
    n_probe = min(n, N_PROBE_MAX)
    T = []
    for k in range(n_probe):
        ok = k < round(recall * n_probe)
        T.append({"i": k, "answer": ("p%d" % k) if ok else None, "cls": "correct" if ok else "abstain",
                  "yn": "yes" if ok else "unknown", "block_ok": ok, "held": ok, "readout": 1.0 if ok else 0.0,
                  "mean_abs_w": w, "shard": 1, "t_query": 0.1, "t_yn": 0.1, "t_engram": t_engram})
    if rep_flip and T:
        T[0] = dict(T[0], answer="zz", cls="wrong_word")
    NM = [{"i": k, "yn": "yes" if k < round(nm_false * n_probe) else "unknown", "kind": "same_subject_sibling",
           "shard": 1, "t_yn": 0.1} for k in range(n_probe)]
    NV = [{"agent": "a", "action": "b", "answer": "x" if k < round(novel_false * n_probe) else None, "shard": 0,
           "t_query": 0.1} for k in range(n_probe)]
    return {"arm": arm, "n_facts": n, "error": err, "resource_probe": False, "kb_len_after_teach": n,
            "substrate_threshold_sha256": thr, "probe_spec_hash": "h", "facts_hash": "f", "n_total": 1000 + n,
            "store_writes_during_probe": [], "encode_s": [encode_s] * n, "peak_rss_mb": peak_rss_mb,
            "probes": {"taught": T, "nearmiss": NM, "novel": NV}}


def _syn_cell(n, **kw):
    H = _syn_job(n, "HEBB", **kw.get("H", {}))
    F = _syn_job(n, "FREEZE", **dict({"recall": 0.0, "w": 0.0}, **kw.get("F", {})))
    C = _syn_job(n, "COPY", **kw.get("C", {}))
    jobs = {"HEBB": H, "FREEZE": F, "COPY": C}
    if n in REP_LEVELS:
        jobs["HEBB_REP"] = _syn_job(n, "HEBB_REP", **kw.get("R", kw.get("H", {})))
    return jobs


def selftest():
    import contextlib
    import io
    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)
    with contextlib.redirect_stdout(io.StringIO()):
        c = score_cell(_syn_cell(50), 50)
        check("go_cell_defined_and_recall_holds", c["status"] == "DEFINED" and c["HEBB_recall_holds"] and c["parity"])
        check("go_cell_cost_ok_at_n50", c["HEBB_cost_ok"] is True)          # 0.1s encode, 0.1*50=5s readtime, 0.3GB
        c = score_cell(_syn_cell(50, H={"recall": 0.85}), 50)
        check("recall_0.85_does_not_hold", c["status"] == "DEFINED" and not c["HEBB_recall_holds"])
        c = score_cell(_syn_cell(50, H={"novel_false": 0.1}), 50)
        check("novel_false_0.10_does_not_hold", not c["HEBB_recall_holds"])
        c = score_cell(_syn_cell(50, H={"nm_false": 0.1}), 50)
        check("nearmiss_false_0.10_does_not_hold", not c["HEBB_recall_holds"])
        # AMENDMENT B: the cost criterion CAN fail (unlike recall) -- each axis independently.
        c = score_cell(_syn_cell(50, H={"t_engram": 0.25}), 50)             # 0.25*50=12.5s > COST_READTIME_MAX_S
        check("cost_fails_on_readtime_view", c["HEBB_cost_ok"] is False)
        c = score_cell(_syn_cell(50, H={"encode_s": 3.0}), 50)              # > COST_ENCODE_MAX_S
        check("cost_fails_on_encode_time", c["HEBB_cost_ok"] is False)
        c = score_cell(_syn_cell(50, H={"peak_rss_mb": 30000.0}), 50)       # > COST_MEM_GB
        check("cost_fails_on_memory", c["HEBB_cost_ok"] is False)
        c = score_cell(_syn_cell(2000, H={"t_engram": 0.1}), 2000)          # prereg's own projection: fails at N=2000
        check("cost_fails_at_n2000_projected_readtime", c["HEBB_cost_ok"] is False)
        c = score_cell(_syn_cell(50, F={"recall": 0.5}), 50)
        check("freeze_recalling_is_UNDEFINED_not_pass", c["status"] == "UNDEFINED")
        c = score_cell(_syn_cell(50, F={"w": 0.3}), 50)
        check("freeze_lever_nonzero_is_UNDEFINED", c["status"] == "UNDEFINED")
        c = score_cell(_syn_cell(50, R={"rep_flip": True}), 50)
        check("rep_mismatch_is_UNDEFINED", c["status"] == "UNDEFINED")
        jobs = _syn_cell(50); jobs["COPY"] = None
        check("missing_arm_is_UNDEFINED", score_cell(jobs, 50)["status"] == "UNDEFINED")
        jobs = _syn_cell(50); jobs["FREEZE"]["substrate_threshold_sha256"] = "other"
        check("substrate_mismatch_is_UNDEFINED", score_cell(jobs, 50)["status"] == "UNDEFINED")
        jobs = _syn_cell(500); jobs["HEBB"]["error"] = "boom"
        check("errored_arm_is_UNDEFINED", score_cell(jobs, 500)["status"] == "UNDEFINED")
        c = score_cell(_syn_cell(500, H={"recall": 0.7}), 500)
        check("hebb_worse_not_parity", c["parity"] is False)
        # level + curve bands
        good = [score_cell(_syn_cell(50), 50) for _ in range(6)]
        check("level_RECALL_HOLDS_6of6", _level_label(good, "HEBB_recall_holds") == "RECALL-HOLDS")
        mixed = good[:5] + [score_cell(_syn_cell(50, H={"recall": 0.5}), 50)]
        check("level_MIXED_5of6", _level_label(mixed, "HEBB_recall_holds") == "MIXED(5/6)")
        check("level_INCOMPLETE_5_seeds", _level_label(good[:5], "HEBB_recall_holds") == "INCOMPLETE")
        worse = [score_cell(_syn_cell(50, H={"recall": 0.5}), 50) for _ in range(6)]
        check("parity_HEBB_WORSE", _parity_label(worse) == "HEBB-WORSE")
        check("parity_PARITY_BY_CONSTRUCTION", _parity_label(good) == "PARITY-BY-CONSTRUCTION")
        # AMENDMENT B: the cost level/curve bands, distinct vocabulary from recall's
        cost_good = [score_cell(_syn_cell(50), 50) for _ in range(6)]                    # cost holds at N=50
        check("level_COST_HOLDS_6of6",
              _level_label(cost_good, "HEBB_cost_ok", "COST-HOLDS", "COST-FAILS") == "COST-HOLDS")
        cost_bad = [score_cell(_syn_cell(2000, H={"t_engram": 0.1}), 2000) for _ in range(6)]  # cost fails at N=2000
        check("level_COST_FAILS_6of6",
              _level_label(cost_bad, "HEBB_cost_ok", "COST-HOLDS", "COST-FAILS") == "COST-FAILS")
        L = [5, 50, 500, 2000]
        check("curve_recall_holds_to_max",
              curve_verdict({5: "RECALL-HOLDS", 50: "RECALL-HOLDS", 500: "RECALL-HOLDS", 2000: "RECALL-HOLDS"}, L)
              == "RECALL-HOLDS-TO-2000")
        check("curve_ceiling", curve_verdict(
              {5: "RECALL-HOLDS", 50: "RECALL-HOLDS", 500: "FAILS", 2000: "MIXED(2/6)"}, L)
              == "CEILING-BETWEEN-50-AND-500")
        check("curve_fails_from_smallest", curve_verdict(
              {5: "FAILS", 50: "FAILS", 500: "FAILS", 2000: "FAILS"}, L) == "FAILS-FROM-5")
        check("curve_nonmonotone", curve_verdict(
              {5: "RECALL-HOLDS", 50: "FAILS", 500: "RECALL-HOLDS", 2000: "RECALL-HOLDS"}, L) == "NON-MONOTONE")
        check("curve_undefined", curve_verdict(
              {5: "RECALL-HOLDS", 50: "UNDEFINED", 500: "RECALL-HOLDS", 2000: "RECALL-HOLDS"}, L) == "UNDEFINED")
        check("curve_dropped_level_3", curve_verdict(
              {5: "RECALL-HOLDS", 50: "RECALL-HOLDS", 500: "RECALL-HOLDS"}, L[:3]) == "RECALL-HOLDS-TO-500")
        # cost curve uses its own vocabulary end to end
        check("curve_cost_holds_to_max", curve_verdict(
              {5: "COST-HOLDS", 50: "COST-HOLDS", 500: "COST-HOLDS", 2000: "COST-HOLDS"}, L, "COST-HOLDS", "COST-FAILS")
              == "COST-HOLDS-TO-2000")
        check("curve_cost_ceiling", curve_verdict(
              {5: "COST-HOLDS", 50: "COST-HOLDS", 500: "COST-FAILS", 2000: "COST-FAILS"}, L, "COST-HOLDS", "COST-FAILS")
              == "CEILING-BETWEEN-50-AND-500")
    # fact / probe generator invariants (real, not synthetic)
    m = make_master(42)
    check("master_2000_unique_pairs", len(m) == N_MASTER and len({(a, v) for a, v, _p in m}) == N_MASTER)
    check("master_deterministic", m == make_master(42) and m != make_master(43))
    for n in LEVELS:
        sp = make_probes(m[:n], 42, n)
        taught = set(m[:n]); pairs = {(a, v) for a, v, _p in m[:n]}
        check("probes_n%d_sizes" % n, len(sp["taught"]) == len(sp["novel"]) == len(sp["nearmiss"]) == min(n, 100))
        check("novel_n%d_never_taught" % n, all((r["agent"], r["action"]) not in pairs for r in sp["novel"]))
        check("nearmiss_n%d_never_taught" % n,
              all(r["patient"] is None or (r["agent"], r["action"], r["patient"]) not in taught for r in sp["nearmiss"]))
        check("probes_n%d_deterministic" % n, sp["hash"] == make_probes(m[:n], 42, n)["hash"])
    sib = [r["kind"] for r in make_probes(m[:500], 42, 500)["nearmiss"]]
    check("nearmiss_mostly_same_subject_sibling", sib.count("same_subject_sibling") >= 0.9 * len(sib))
    if fails:
        print("SELFTEST FAIL: %s" % ", ".join(fails))
        return 1
    print("SELFTEST PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--resource-probe", action="store_true")
    ap.add_argument("--probe-k", type=int, default=5)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-facts", type=int, default=5)
    ap.add_argument("--arm", choices=sorted(ARMS), default="HEBB")
    ap.add_argument("--out")
    ap.add_argument("--arm-dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS6)
    ap.add_argument("--levels", type=int, nargs="+", default=None)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.worker or a.resource_probe:
        if not a.out:
            ap.error("--out is required")
        return worker(a.seed, a.n_facts, a.arm, a.out, resource_probe=a.resource_probe, probe_k=a.probe_k)
    if a.score:
        v = score_grid(a.arm_dir, seeds=a.seeds, levels=a.levels)
        print(json.dumps({k: v[k] for k in ("level_label_HEBB", "level_label_COPY", "parity_HEBB_vs_COPY",
                                            "curve_HEBB", "curve_COPY", "level_label_HEBB_cost",
                                            "level_label_COPY_cost", "curve_HEBB_cost", "curve_COPY_cost")},
                          indent=1))
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(v, fh, indent=1, default=str)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
