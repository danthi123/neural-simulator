"""Fast calibration measurement: the REAL production `OneBrainComposer` config (D=128, vocab_headroom=128,
enable_spiking_cleanup=True -- exactly what `BrainConversationalAgent(composer_kind='onebrain')` builds for the
tiny-demo brain) on the tiny-demo's OWN 5 facts, using the low-level `store()`/`query_patient()` API directly
(bypassing the slow on-bridge NL-parser training `.hear()` triggers, so this runs in seconds on CPU) -- for the
issue #181 margin-band recalibration. Also sweeps synaptic noise (the same legitimate perturbation
`_emergent_graceful_degradation_derisk.py` validated) to find the natural CONFIDENT-vs-UNCERTAIN spread at
production D/vocab_headroom.

Usage: SIM_BACKEND=numpy PYTHONPATH=. python measure_prod_config_margins.py
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
from research.runners.metacog_production_organ import mean_role_confidence

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/measure_prod_config_margins.json")

# the tiny-demo's OWN facts + vocab (research/runners/brain_chat_tui.py:_build_tiny_demo)
FACTS = [
    ("brain", "use", "spikes"),
    ("brain", "learn", "words"),
    ("brain", "store", "memory"),
    ("dog", "chase", "cat"),
    ("cat", "eat", "fish"),
]
VOCAB = sorted({w for f in FACTS for w in f} | {"river", "bird", "fish", "worm", "ball"})

log(f"building PRODUCTION-config composer (D=128, vocab_headroom=128, enable_spiking_cleanup=True) "
    f"over {len(VOCAB)}-word vocab, storing {len(FACTS)} facts via store() (bypassing NL parse)...")
c = OneBrainComposer(seed=42, D=128, vocab=VOCAB, vocab_headroom=128, trace=True)
for (a, v, p) in FACTS:
    c.store(a, v, p)
base_store = list(c.store_conns)
log("built + stored.")

results = {"runner": "measure_prod_config_margins", "backend": os.environ.get("SIM_BACKEND"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "confident": {}, "noise_sweep": {}}

# (1) CONFIDENT: every real tiny-demo fact, intact store
mrcs = []
for (a, v, p) in FACTS:
    c.last_trace = None
    ans = c.query_patient(a, v)
    trace = c.last_trace
    mrc = mean_role_confidence(trace)
    roles = (trace or {}).get("roles", [])
    margins = {r.get("role"): r.get("margin") for r in roles}
    log(f"CONFIDENT {a} {v} -> {ans!r} (expect {p!r}) mrc={mrc} margins={margins}")
    results["confident"][f"{a} {v}"] = {"answer": ans, "expected": p, "mean_role_confidence": mrc,
                                         "margins_by_role": margins}
    if mrc is not None:
        mrcs.append(mrc)

results["confident_mrc_min"] = min(mrcs) if mrcs else None
results["confident_mrc_max"] = max(mrcs) if mrcs else None
log(f"confident mrc range: min={results['confident_mrc_min']} max={results['confident_mrc_max']}")

# (2) UNCERTAIN: synaptic noise sweep on ONE fact (brain/use/spikes), production D/vocab_headroom
rng = np.random.default_rng(11)
sweep = []
for sigma in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0):
    c.store_conns = _noise(base_store, sigma, rng)
    c.last_trace = None
    ans = c.query_patient("brain", "use")
    trace = c.last_trace
    mrc = mean_role_confidence(trace)
    sweep.append({"sigma": sigma, "answer": ans, "mean_role_confidence": mrc})
    log(f"  noise sigma={sigma}: answer={ans!r} mrc={mrc}")
c.store_conns = list(base_store)
results["noise_sweep"]["brain use"] = sweep

answered = [row for row in sweep if row["answer"] is not None and row["mean_role_confidence"] is not None]
results["uncertain_mrc_min_answered"] = min((r["mean_role_confidence"] for r in answered), default=None)
log(f"lowest mrc among still-ANSWERED (not abstained) noised turns: {results['uncertain_mrc_min_answered']}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
