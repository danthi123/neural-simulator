"""FINAL verification (issue #181): with the RECALIBRATED ROLE_CONF_LO/HI (0.40/0.60) in place, run the exact
metacog chain (`mean_role_confidence` -> `evidence_from_role_conf` -> `MetacogProductionOrgan.judge`) on the
PRODUCTION-config tiny-demo composer (D=128, vocab_headroom=128), for:
  (A) all 5 real confident facts, intact store -- the hedge must be ABSENT on every one (no regression).
  (B) the SAME facts under legitimate synaptic-noise damage (a genuine weak/ambiguous match, still answered,
      not an abstain) -- the hedge must be PRESENT.
  (C) the LESION check (`BRAIN_METACOG_LESION`-equivalent, organ.judge(..., lesion=True)) on both -- the
      confident/uncertain discrimination must VANISH (both read confident, since the evidence differential the
      organ reads is removed).

Usage: SIM_BACKEND=numpy PYTHONPATH=. python verify_prod_discriminates.py
"""
import os, json, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import numpy as np
from research.runners.one_brain_composer import OneBrainComposer
from research.runners._emergent_graceful_degradation_derisk import _noise
from research.runners.metacog_production_organ import (
    mean_role_confidence, evidence_from_role_conf, MetacogProductionOrgan, ROLE_CONF_LO, ROLE_CONF_HI,
)

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/verify_prod_discriminates.json")

FACTS = [
    ("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
    ("dog", "chase", "cat"), ("cat", "eat", "fish"),
]
VOCAB = sorted({w for f in FACTS for w in f} | {"river", "bird", "fish", "worm", "ball"})

log(f"ROLE_CONF_LO={ROLE_CONF_LO} ROLE_CONF_HI={ROLE_CONF_HI}")
log("building production-config composer...")
c = OneBrainComposer(seed=42, D=128, vocab=VOCAB, vocab_headroom=128, trace=True)
for (a, v, p) in FACTS:
    c.store(a, v, p)
base_store = list(c.store_conns)
organ = MetacogProductionOrgan(seed=42)

results = {"runner": "verify_prod_discriminates", "role_conf_lo": ROLE_CONF_LO, "role_conf_hi": ROLE_CONF_HI,
           "confident": {}, "uncertain": {}, "lesioned": {}}


def _judge(a, v, lesion=False):
    c.last_trace = None
    ans = c.query_patient(a, v)
    trace = c.last_trace
    mrc = mean_role_confidence(trace)
    ev = evidence_from_role_conf(mrc)
    if ev is None:
        return ans, mrc, ev, None
    j = organ.judge(ev, lesion=lesion)
    return ans, mrc, ev, j


# (A) CONFIDENT: all 5 real facts, intact store
log("=== (A) CONFIDENT: 5 real facts, intact store ===")
all_confident_no_hedge = True
for (a, v, p) in FACTS:
    ans, mrc, ev, j = _judge(a, v)
    confident = j["confident"] if j else None
    hedge_fires = (confident is False)
    all_confident_no_hedge = all_confident_no_hedge and (confident is True)
    log(f"  {a} {v} -> {ans!r} mrc={mrc} evidence={ev} confident={confident} hedge_fires={hedge_fires}")
    results["confident"][f"{a} {v}"] = {"answer": ans, "mrc": mrc, "evidence": ev, "confident": confident}

# (B) UNCERTAIN: noise-degrade "brain use", pick levels that still answer
log("=== (B) UNCERTAIN: noise-degraded 'brain use', still-answered levels ===")
rng = np.random.default_rng(13)
any_hedge_fired = False
uncertain_rows = []
for sigma in (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6):
    c.store_conns = _noise(base_store, sigma, rng)
    ans, mrc, ev, j = _judge("brain", "use")
    confident = j["confident"] if j else None
    hedge_fires = (confident is False)
    any_hedge_fired = any_hedge_fired or (hedge_fires and ans is not None)
    log(f"  sigma={sigma}: answer={ans!r} mrc={mrc} evidence={ev} confident={confident} hedge_fires={hedge_fires}")
    uncertain_rows.append({"sigma": sigma, "answer": ans, "mrc": mrc, "evidence": ev, "confident": confident})
c.store_conns = list(base_store)
results["uncertain"]["brain use sweep"] = uncertain_rows

# pick the representative uncertain point (lowest mrc among still-answered) for the lesion check
answered = [r for r in uncertain_rows if r["answer"] is not None and r["mrc"] is not None]
pick = min(answered, key=lambda r: r["mrc"]) if answered else None
results["uncertain"]["picked_sigma"] = pick["sigma"] if pick else None

# (C) LESION check
log("=== (C) LESION check ===")
ans, mrc, ev, j = _judge("brain", "use", lesion=True)
results["lesioned"]["confident_intact"] = {"answer": ans, "mrc": mrc, "evidence": ev,
                                            "confident": j["confident"] if j else None}
log(f"  confident/intact, LESIONED: confident={results['lesioned']['confident_intact']['confident']}")
if pick is not None:
    c.store_conns = _noise(base_store, pick["sigma"], rng)
    ans, mrc, ev, j = _judge("brain", "use", lesion=True)
    results["lesioned"]["uncertain_noised"] = {"answer": ans, "mrc": mrc, "evidence": ev,
                                                "confident": j["confident"] if j else None}
    log(f"  uncertain/noised (sigma={pick['sigma']}), LESIONED: "
        f"confident={results['lesioned']['uncertain_noised']['confident']}")
    c.store_conns = list(base_store)

lesion_removes_discrimination = (
    results["lesioned"]["confident_intact"]["confident"] is True
    and results["lesioned"].get("uncertain_noised", {}).get("confident") is True
)

results["verdict"] = {
    "all_5_confident_facts_no_hedge": all_confident_no_hedge,
    "any_uncertain_turn_hedges": any_hedge_fired,
    "lesion_removes_discrimination": lesion_removes_discrimination,
    "DISCRIMINATES": bool(all_confident_no_hedge and any_hedge_fired),
}
log(f"VERDICT: {json.dumps(results['verdict'], indent=2)}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
