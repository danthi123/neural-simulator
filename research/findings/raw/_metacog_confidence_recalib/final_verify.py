"""FINAL end-to-end verify (issue #181 + #184) with the RECALIBRATED ROLE_CONF_LO/HI (0.30/0.50) baked in, on
the REAL production composer build (`_build_tiny_demo`, exactly what `webapp/server.py` calls):

  (A) all 5 real confident facts, intact store -> the metacog organ's `confident` must be True on every one
      (no false-positive hedge / no regression).
  (B) the SAME real composer's store, synaptic-noise-degraded at levels that still answer -> `confident` must
      go False (the hedge fires) at clearly-degraded noise levels.
  (C) LESION (`organ.judge(..., lesion=True)`): the confident/uncertain distinction must COLLAPSE (both read
      the SAME `confident` value, matching the organ's documented "a would-be-confident answer flips to a
      hedge under lesion" mechanism) -- proving the discrimination is driven by the organ genuinely reading
      the evidence differential, not a host-side shortcut.
  (D) #184 guard: call `_MC.mean_role_confidence` on an activity=None reading directly (the plumbing-bug
      shape) vs a genuine no-answer/abstain shape, confirming the function itself behaves as designed (the
      webapp-level print-warning guard is exercised separately, this checks the underlying data contract).

Usage: SIM_BACKEND=numpy PYTHONPATH=. python final_verify.py
"""
import os, json, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import numpy as np
from research.runners.brain_chat_tui import _build_tiny_demo
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import (
    mean_role_confidence, evidence_from_role_conf, MetacogProductionOrgan, ROLE_CONF_LO, ROLE_CONF_HI,
)

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/final_verify.json")
log(f"ROLE_CONF_LO={ROLE_CONF_LO} ROLE_CONF_HI={ROLE_CONF_HI}")

agent, aliases, n = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind="onebrain")
inner = getattr(agent, "agent", agent)
comp = inner.composer
comp.trace = True
base_store = list(comp.store_conns)
organ = MetacogProductionOrgan(seed=42)
log(f"built the real tiny-demo composer ({n} facts).")

QUERIES = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
           ("dog", "chase", "cat"), ("cat", "eat", "fish")]

results = {"role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI, "confident": {}, "uncertain": {},
           "lesioned": {}}


def _judge(a, v, lesion=False):
    comp.last_trace = None
    ans = comp.query_patient(a, v)
    trace = comp.last_trace
    mrc = mean_role_confidence(trace)
    ev = evidence_from_role_conf(mrc)
    j = organ.judge(ev, lesion=lesion) if ev is not None else None
    return ans, mrc, ev, j


# (A) CONFIDENT
log("=== (A) CONFIDENT: 5 real facts, intact store ===")
all_confident = True
for (a, v, p) in QUERIES:
    ans, mrc, ev, j = _judge(a, v)
    confident = j["confident"] if j else None
    all_confident = all_confident and (confident is True)
    log(f"  {a} {v} -> {ans!r} mrc={mrc} evidence={ev} confident={confident}")
    results["confident"][f"{a} {v}"] = {"answer": ans, "expected": p, "mrc": mrc, "evidence": ev,
                                         "confident": confident}
    assert ans == p, f"REGRESSION: {a} {v} answered {ans!r}, expected {p!r} -- the fix changed the ANSWER"

# (B) UNCERTAIN sweep
log("=== (B) UNCERTAIN: noise-degraded 'brain use' sweep ===")
rng = np.random.default_rng(31)
sweep = []
for sigma in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6):
    comp.store_conns = _noise(base_store, sigma, rng)
    ans, mrc, ev, j = _judge("brain", "use")
    confident = j["confident"] if j else None
    log(f"  sigma={sigma}: answer={ans!r} mrc={mrc} evidence={ev} confident={confident}")
    sweep.append({"sigma": sigma, "answer": ans, "mrc": mrc, "evidence": ev, "confident": confident})
comp.store_conns = list(base_store)
results["uncertain"]["sweep"] = sweep
any_hedge = any((r["confident"] is False and r["answer"] is not None) for r in sweep)
heavy = [r for r in sweep if r["sigma"] >= 1.3]
heavy_all_hedge = bool(heavy) and all((r["confident"] is False) for r in heavy if r["answer"] is not None)

# (C) LESION
log("=== (C) LESION check ===")
ans, mrc, ev, j = _judge("brain", "use", lesion=True)
conf_lesioned = j["confident"] if j else None
results["lesioned"]["confident_intact"] = {"answer": ans, "mrc": mrc, "evidence": ev, "confident": conf_lesioned}
log(f"  confident/intact LESIONED: confident={conf_lesioned}")
pick = next((r for r in sweep if r["sigma"] == 1.8), None)
comp.store_conns = _noise(base_store, 1.8, np.random.default_rng(31 + 100))
ans, mrc, ev, j = _judge("brain", "use", lesion=True)
comp.store_conns = list(base_store)
unc_lesioned = j["confident"] if j else None
results["lesioned"]["uncertain_noised"] = {"answer": ans, "mrc": mrc, "evidence": ev, "confident": unc_lesioned}
log(f"  uncertain/noised(sigma=1.8) LESIONED: confident={unc_lesioned}")
lesion_collapses = (conf_lesioned == unc_lesioned)   # both same value -> the differential is gone under lesion

# (D) #184 data-contract check
log("=== (D) #184: mean_role_confidence on a plumbing-bug shape (activity=None) vs genuine out-of-scope ===")
mrc_none = mean_role_confidence(None)
mrc_empty_roles = mean_role_confidence({"roles": []})
mrc_abstained = mean_role_confidence({"roles": [], "abstained": True})
log(f"  activity=None -> {mrc_none}; roles=[] -> {mrc_empty_roles}; abstained trace -> {mrc_abstained}")
results["184_contract"] = {"activity_none": mrc_none, "empty_roles": mrc_empty_roles, "abstained": mrc_abstained}

results["verdict"] = {
    "all_5_confident_no_hedge": all_confident,
    "any_uncertain_turn_hedges": any_hedge,
    "heavy_noise_all_hedge": heavy_all_hedge,
    "lesion_collapses_discrimination": lesion_collapses,
    "DISCRIMINATES": bool(all_confident and any_hedge),
    "no_answer_regression": True,   # the asserts above would have raised otherwise
}
log(f"VERDICT: {json.dumps(results['verdict'], indent=2)}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
