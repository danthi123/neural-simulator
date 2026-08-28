"""Verify the FIXED metacog honesty-hedge (issue #181, #184) actually DISCRIMINATES confident vs genuinely-
uncertain turns, through the REAL production functions end-to-end:
  `OneBrainComposer._block_role_scores` (margin) -> `metacog_production_organ.mean_role_confidence` ->
  `evidence_from_role_conf` -> `MetacogProductionOrgan.judge`.

CONFIDENT case: a clean, intact store (the composer's own facts, zero perturbation) -- the same regime as real
production traffic on the tiny-demo brain.

GENUINELY-UNCERTAIN case: the SAME real composer + the SAME real query, with the stored fact's SYNAPTIC WEIGHTS
perturbed by complex Gaussian jitter (`_noise`, the exact perturbation `_emergent_graceful_degradation_derisk.py`
already validated as a legitimate biological damage model -- synaptic noise -- for this exact composer). This is
a REAL degraded read off the REAL resonate/cleanup pipeline, not a hand-built score array: a genuinely weak/
ambiguous match, picked at a noise level that still returns AN answer (not abstain) so the hedge path is actually
reached, mirroring "a partial/weak match" in the task's framing.

Also runs the LESION check (`BRAIN_METACOG_LESION=1`): the discrimination must vanish (both cases read
'confident') when the organ's evidence differential is removed, proving the discrimination is caused by the
organ genuinely reading the evidence signal.

Usage: SIM_BACKEND=numpy PYTHONPATH=. python verify_discriminates.py
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
    mean_role_confidence, evidence_from_role_conf, MetacogProductionOrgan,
)

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/verify_discriminates.json")

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]
FACTS = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south"), ("apple", "stop", "west"),
         ("river", "swim", "home"), ("home", "go", "cat"), ("north", "look", "dog"), ("west", "come", "bird")]
QUERY = ("dog", "go")   # -> "north"

log("building composer + storing facts (trace=True)...")
c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, trace=True)
for (a, v, p) in FACTS:
    c.store(a, v, p)
base_store = list(c.store_conns)

results = {"runner": "verify_discriminates (direct real-function chain, not the HTTP handler)",
           "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _query_and_trace():
    c.last_trace = None
    ans = c.query_patient(*QUERY)
    return ans, c.last_trace


def _report(label, ans, trace, lesion):
    mrc = mean_role_confidence(trace)
    ev = evidence_from_role_conf(mrc)
    roles = (trace or {}).get("roles", [])
    margins = {r.get("role"): r.get("margin") for r in roles}
    if ev is None:
        j = None
        confident = None
    else:
        organ = MetacogProductionOrgan(seed=42)
        j = organ.judge(ev, lesion=lesion)
        confident = j["confident"]
    log(f"{label}: ans={ans!r} mrc={mrc} evidence={ev} margins={margins} confident={confident} "
        f"balance={None if j is None else j['balance']} threshold={None if j is None else j['threshold']}")
    return {"answer": ans, "mean_role_confidence": mrc, "evidence": ev, "margins_by_role": margins,
            "confident": confident, "judge": j}


# (1) CONFIDENT: the intact store, zero perturbation
log("=== (1) CONFIDENT (intact store) ===")
ans, trace = _query_and_trace()
results["confident"] = _report("confident/intact", ans, trace, lesion=False)

# (2) GENUINELY-UNCERTAIN: sweep synaptic noise sigma, find a level that still answers but with a lower margin
log("=== (2) sweeping synaptic noise sigma to find a genuine weak/ambiguous (but still-answered) read ===")
rng = np.random.default_rng(7)
sweep = []
for sigma in (0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2):
    c.store_conns = _noise(base_store, sigma, rng)
    ans, trace = _query_and_trace()
    mrc = mean_role_confidence(trace)
    sweep.append({"sigma": sigma, "answer": ans, "mean_role_confidence": mrc})
    log(f"  sigma={sigma}: answer={ans!r} mrc={mrc}")
c.store_conns = list(base_store)   # restore
results["noise_sweep"] = sweep

# pick the noisiest level that STILL returned an answer (a genuinely weak match, not an abstain) for the
# discrimination check below
answered = [row for row in sweep if row["answer"] is not None and row["mean_role_confidence"] is not None]
uncertain_row = min(answered, key=lambda r: r["mean_role_confidence"]) if answered else None
results["picked_uncertain_sigma"] = uncertain_row["sigma"] if uncertain_row else None

if uncertain_row is not None:
    log(f"=== (2b) re-running the picked uncertain level (sigma={uncertain_row['sigma']}) for the full report ===")
    c.store_conns = _noise(base_store, uncertain_row["sigma"], rng)
    ans, trace = _query_and_trace()
    results["uncertain"] = _report("uncertain/noised", ans, trace, lesion=False)
    c.store_conns = list(base_store)
else:
    results["uncertain"] = None
    log("NO noise level produced an answered-but-lower-confidence turn -- reporting honestly, not faking one.")

# (3) LESION: re-run BOTH cases with BRAIN_METACOG_LESION-equivalent (organ.judge(..., lesion=True)) -- the
# evidence differential is removed, so both should collapse to the SAME side (discrimination vanishes).
log("=== (3) LESION check (organ evidence differential removed) ===")
ans, trace = _query_and_trace()   # back on the intact store
results["confident_lesioned"] = _report("confident/intact, LESIONED", ans, trace, lesion=True)
if uncertain_row is not None:
    c.store_conns = _noise(base_store, uncertain_row["sigma"], rng)
    ans, trace = _query_and_trace()
    results["uncertain_lesioned"] = _report("uncertain/noised, LESIONED", ans, trace, lesion=True)
    c.store_conns = list(base_store)
else:
    results["uncertain_lesioned"] = None

# ── verdicts ──
mrc_conf = results["confident"]["mean_role_confidence"]
mrc_unc = (results["uncertain"] or {}).get("mean_role_confidence")
discriminates_numerically = (mrc_conf is not None and mrc_unc is not None and mrc_conf > mrc_unc)
hedge_absent_confident = (results["confident"]["confident"] is True)
hedge_present_uncertain = ((results["uncertain"] or {}).get("confident") is False)
lesion_removes_discrimination = (
    results["confident_lesioned"]["confident"] is True
    and (results["uncertain_lesioned"] or {}).get("confident") is True
)
results["verdict"] = {
    "mrc_confident": mrc_conf, "mrc_uncertain": mrc_unc,
    "discriminates_numerically": discriminates_numerically,
    "hedge_absent_on_confident": hedge_absent_confident,
    "hedge_present_on_uncertain": hedge_present_uncertain,
    "lesion_removes_discrimination": lesion_removes_discrimination,
}
log(f"VERDICT: {json.dumps(results['verdict'], indent=2, default=str)}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
