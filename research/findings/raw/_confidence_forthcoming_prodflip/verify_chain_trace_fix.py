"""Verify the `RichAnswerComposer._chain_facts` trace-preservation fix (2026-08-27, board #94 honesty
follow-up), through the REAL `/api/brain-chat` handler, on REAL (UNPATCHED) production traffic:

  (1) BYTE-IDENTICAL OUTPUT: the fix only restores `composer.last_trace` after `_chain_facts` returns -- it
      never touches the `facts` list the method decides. Confirm the rendered answer/recalled_svo/n_sentences
      for two known topics are UNCHANGED from the pre-fix values recorded in this session's own probe
      (research/findings/2026-08-27-confidence-forthcomingness-production-default-GO.md's own prior probe):
        "what does the brain use"  -> "the brain uses the spikes the brain stores the memory the brain learns
                                        the words" (n_sentences=3, unrelated to this fix's own floor logic --
                                        BRAIN_CONFIDENCE_FORTHCOMING is genuinely exercised here too)
        "what does the dog chase"  -> "the dog chases the cat the cat eats the fish" (n_sentences=2)
  (2) THE FIX ITSELF: `activity` (last_trace) is no longer None on these REAL, UNPATCHED turns, and
      `mean_role_confidence` returns a real number (not None) -- the load-bearing thing this whole fix chases.
  (3) GENUINE VARIATION ON REAL TRAFFIC: does the confidence read (and therefore `confidence_forthcoming`,
      now genuinely reachable) actually DIFFER between these two topics on real evidence, i.e. is the
      default-ON flip genuinely load-bearing, not hollow? Reported honestly either way.

Usage: SIM_BACKEND=numpy python verify_chain_trace_fix.py
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
           "BRAIN_GNW_2ORGAN", "BRAIN_GNW_3ORGAN"):
    os.environ[_k] = "0"
# NOTHING patched, NOTHING forced -- genuine real evidence, genuine real default (BRAIN_CONFIDENCE_FORTHCOMING
# and BRAIN_METACOG both left UNSET -> their real shipped defaults, both ON).
os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING", None)
os.environ.pop("BRAIN_METACOG", None)
os.environ.pop("BRAIN_METACOG_LESION", None)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import webapp.server as S

_ART = os.environ.get("CF_JSON", "research/findings/raw/_confidence_forthcoming_prodflip/verify_chain_trace_fix.json")
_RESULTS = {"runner": "verify_chain_trace_fix (real /api/brain-chat, unpatched, unforced)",
            "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

EXPECTED = {
    "what does the brain use": {
        "answer": "the brain uses the spikes the brain stores the memory the brain learns the words",
        "recalled_svo": ["brain", "use", "spikes"],
        "n_sentences": 3,
    },
    "what does the dog chase": {
        "answer": "the dog chases the cat the cat eats the fish",
        "recalled_svo": ["dog", "chase", "cat"],
        "n_sentences": 2,
    },
}

turns = {}
for q in EXPECTED:
    resp = S.brain_chat(S.BrainChatRequest(session=f"chaintrace_{abs(hash(q))}", message=q, brain="tiny-demo",
                                           reset=True, rich=True))
    d = json.loads(bytes(resp.body))
    turns[q] = d
    mrc = None
    try:
        from research.runners.metacog_production_organ import mean_role_confidence
        mrc = mean_role_confidence(d.get("activity"))
    except Exception as e:
        mrc = f"ERROR: {e}"
    log(f"{q!r}: answer={d.get('answer')!r} n_sentences={d.get('n_sentences')} "
        f"activity_is_none={d.get('activity') is None} mean_role_conf={mrc} "
        f"metacog={d.get('metacog')} confidence_forthcoming={d.get('confidence_forthcoming')}")
    _RESULTS[q] = {"answer": d.get("answer"), "recalled_svo": d.get("recalled_svo"),
                   "n_sentences": d.get("n_sentences"), "activity_is_none": d.get("activity") is None,
                   "mean_role_confidence": mrc, "metacog": d.get("metacog"),
                   "confidence_forthcoming": d.get("confidence_forthcoming"), "source": d.get("source")}

# (1) byte-identical answer content vs the pre-fix probe values
identical_ok = all(
    turns[q].get("answer") == EXPECTED[q]["answer"]
    and turns[q].get("recalled_svo") == EXPECTED[q]["recalled_svo"]
    and turns[q].get("n_sentences") == EXPECTED[q]["n_sentences"]
    for q in EXPECTED
)
log(f"(1) chain-output byte-identical to pre-fix probe: {identical_ok}")

# (2) the fix itself: activity non-None + mean_role_confidence non-None on BOTH turns
mrcs = {q: _RESULTS[q]["mean_role_confidence"] for q in EXPECTED}
fix_ok = all(isinstance(v, float) for v in mrcs.values())
log(f"(2) mean_role_confidence non-None on both real turns: {fix_ok}  values={mrcs}")

# (3) genuine variation: do the two topics' confidence reads actually differ (real evidence, not forced)?
varies = False
if fix_ok:
    varies = abs(mrcs["what does the brain use"] - mrcs["what does the dog chase"]) > 1e-9
log(f"(3) genuine variation between topics on REAL evidence: {varies} (mrc_brain={mrcs.get('what does the brain use')} "
    f"mrc_dog={mrcs.get('what does the dog chase')})")

# does the confidence_forthcoming coupling ITSELF actually fire differently (n_sentences differ due to the cap)
# on these two real turns -- the actual "not hollow" proof for board #94 specifically.
cf_brain = _RESULTS["what does the brain use"]["confidence_forthcoming"]
cf_dog = _RESULTS["what does the dog chase"]["confidence_forthcoming"]
cf_fired = bool(cf_brain or cf_dog)
log(f"confidence_forthcoming key present on real traffic: brain={cf_brain is not None} dog={cf_dog is not None} "
    f"(granted anywhere: {any((c or {}).get('granted') for c in (cf_brain, cf_dog))})")

_RESULTS["chain_output_identical"] = identical_ok
_RESULTS["mean_role_confidence_nonNone_both"] = fix_ok
_RESULTS["confidence_genuinely_varies_real_traffic"] = varies
_RESULTS["confidence_forthcoming_key_present_real_traffic"] = cf_fired

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(_RESULTS, f, indent=2, default=str)
log(f"wrote {_ART}")
