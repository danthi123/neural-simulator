"""Measure the FIXED mean_role_confidence (margin-keyed) on REAL, UNPATCHED production traffic (issue #181),
through the real `/api/brain-chat` handler, on all 5 tiny-demo facts (not just the 2 the predecessor finding
checked) so the recalibration band is set from a genuine spread, not 2 points.

Usage: SIM_BACKEND=numpy python measure_real_confident.py
"""
import os, json, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
for _k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_COMPREHENSION_GATE",
           "BRAIN_PRAGMATIC", "BRAIN_EPISODIC", "BRAIN_MULTIREF", "BRAIN_SELF_INITIATE", "BRAIN_GNW_DELIBERATE",
           "BRAIN_GNW_MULTISTEP", "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_PMEM",
           "BRAIN_CURIOSITY", "BRAIN_DISCOURSE_REGISTER", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES",
           "BRAIN_DA_DRIVES", "BRAIN_GNW_STOP", "BRAIN_SELF_SCHEMA", "BRAIN_AFFECTIVE_TOM",
           "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN", "BRAIN_CONFIDENCE_FORTHCOMING"):
    os.environ[_k] = "0"
# NOTHING patched on the metacog path itself -- genuine real evidence, genuine real default.
os.environ.pop("BRAIN_METACOG", None)
os.environ.pop("BRAIN_METACOG_LESION", None)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import webapp.server as S
from research.runners.metacog_production_organ import mean_role_confidence

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/measure_real_confident.json")

QUESTIONS = [
    "what does the brain use",
    "what does the brain learn",
    "what does the brain store",
    "what does the dog chase",
    "what does the cat eat",
]

results = {"runner": "measure_real_confident (real /api/brain-chat, unpatched)",
           "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "turns": {}}

mrcs = []
for q in QUESTIONS:
    resp = S.brain_chat(S.BrainChatRequest(session=f"mcrecal_{abs(hash(q))}", message=q, brain="tiny-demo",
                                           reset=True, rich=True))
    d = json.loads(bytes(resp.body))
    activity = d.get("activity")
    mrc = mean_role_confidence(activity)
    roles = (activity or {}).get("roles", [])
    margins = {r.get("role"): r.get("margin") for r in roles}
    confs = {r.get("role"): r.get("confidence") for r in roles}
    log(f"{q!r}: answer={d.get('answer')!r} mrc={mrc} margins={margins} legacy_confidences={confs} "
        f"metacog={d.get('metacog')}")
    results["turns"][q] = {
        "answer": d.get("answer"), "n_sentences": d.get("n_sentences"),
        "mean_role_confidence": mrc, "margins_by_role": margins, "legacy_confidences_by_role": confs,
        "metacog": d.get("metacog"),
    }
    if mrc is not None:
        mrcs.append(mrc)

results["mrc_min"] = min(mrcs) if mrcs else None
results["mrc_max"] = max(mrcs) if mrcs else None
results["mrc_spread"] = (max(mrcs) - min(mrcs)) if mrcs else None
log(f"mrc range across {len(mrcs)} real confident turns: min={results['mrc_min']} max={results['mrc_max']} "
    f"spread={results['mrc_spread']}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
