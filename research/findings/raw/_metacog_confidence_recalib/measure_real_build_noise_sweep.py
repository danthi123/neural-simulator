"""Calibration refinement: `measure_real_confident.py` (through the REAL `/api/brain-chat` handler) found real
confident traffic's mean_role_confidence at 0.515..0.615 -- narrower AND lower than the simplified
`measure_prod_config_margins.py` composer (0.616..0.715, built without whatever the real webapp path adds -- an
always-near-zero-margin 'attribute' role chip appears on every real turn, dragging the mean down). This script
builds the composer EXACTLY the way the real webapp does (`_build_tiny_demo`, the SAME function `webapp/server.py`
calls) ONCE, then reuses that SAME real composer for a synaptic-noise sweep (bypassing the slow NL-parse-per-turn
`.hear()` for the sweep -- only the initial build pays that cost), so the CONFIDENT and UNCERTAIN comparison is
apples-to-apples on the REAL composer instance, including whatever real-traffic quirk widens its role set.

Usage: SIM_BACKEND=numpy PYTHONPATH=. python measure_real_build_noise_sweep.py
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
from research.runners.metacog_production_organ import mean_role_confidence

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/measure_real_build_noise_sweep.json")

log("building the REAL tiny-demo brain (_build_tiny_demo, composer_kind=onebrain, use_multiturn=True) ONCE...")
agent, aliases, n = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind="onebrain")
inner = getattr(agent, "agent", agent)
comp = inner.composer
comp.trace = True
log(f"built ({n} facts). composer main_roles={comp.main_roles} bind_roles={comp.bind_roles} "
    f"enable_attributed={comp.enable_attributed}")

base_store = list(comp.store_conns)
results = {"runner": "measure_real_build_noise_sweep", "main_roles": comp.main_roles,
           "bind_roles": comp.bind_roles, "confident": {}, "noise_sweep": []}

QUERIES = [("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
           ("dog", "chase", "cat"), ("cat", "eat", "fish")]

mrcs = []
for (a, v, p) in QUERIES:
    comp.last_trace = None
    ans = comp.query_patient(a, v)
    trace = comp.last_trace
    mrc = mean_role_confidence(trace)
    roles = (trace or {}).get("roles", [])
    margins = {r.get("role"): r.get("margin") for r in roles}
    log(f"CONFIDENT {a} {v} -> {ans!r} (expect {p!r}) mrc={mrc} margins={margins}")
    results["confident"][f"{a} {v}"] = {"answer": ans, "expected": p, "mrc": mrc, "margins": margins}
    if mrc is not None:
        mrcs.append(mrc)
results["confident_mrc_min"] = min(mrcs) if mrcs else None
results["confident_mrc_max"] = max(mrcs) if mrcs else None
log(f"confident mrc range on the REAL composer: min={results['confident_mrc_min']} max={results['confident_mrc_max']}")

log("noise-sweeping the SAME real composer's store_conns for 'brain use'...")
rng = np.random.default_rng(23)
for sigma in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0):
    comp.store_conns = _noise(base_store, sigma, rng)
    comp.last_trace = None
    ans = comp.query_patient("brain", "use")
    trace = comp.last_trace
    mrc = mean_role_confidence(trace)
    log(f"  sigma={sigma}: answer={ans!r} mrc={mrc}")
    results["noise_sweep"].append({"sigma": sigma, "answer": ans, "mrc": mrc})
comp.store_conns = list(base_store)

answered = [r for r in results["noise_sweep"] if r["answer"] is not None and r["mrc"] is not None]
results["uncertain_mrc_min_answered"] = min((r["mrc"] for r in answered), default=None)
results["uncertain_mrc_max_answered"] = max((r["mrc"] for r in answered), default=None)
log(f"uncertain-but-answered mrc range: {results['uncertain_mrc_min_answered']} .. "
    f"{results['uncertain_mrc_max_answered']}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
