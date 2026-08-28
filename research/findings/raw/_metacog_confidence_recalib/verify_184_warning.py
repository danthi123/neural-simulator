"""Verify issue #184 (the silent-regression guard) through the REAL `/api/brain-chat` handler:

  (1) NORMAL real turn (unpatched): the metacog read succeeds -> NO "METACOG WARNING" on stdout.
  (2) SIMULATED plumbing bug: wrap `OneBrainComposer.query_patient` so it produces the SAME real answer but then
      silently wipes `self.last_trace = None` before returning -- reproducing the EXACT shape of the
      TieredFactStore.__setattr__ regression (an answer WAS produced by a trace-capable composer, yet the trace
      came back empty) -- confirm the "METACOG WARNING (#184)" line DOES print.
  (3) GENUINE out-of-scope (an abstain: an unknown/unstored cue): confirm NO warning prints (nothing to read is
      not a bug).

Usage: SIM_BACKEND=numpy PYTHONPATH=. python verify_184_warning.py
"""
import os, io, json, time, contextlib
from unittest import mock

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
os.environ.pop("BRAIN_METACOG", None)
os.environ.pop("BRAIN_METACOG_LESION", None)

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import webapp.server as S
from research.runners.one_brain_composer import OneBrainComposer

_ART = os.environ.get("MC_JSON", "research/findings/raw/_metacog_confidence_recalib/verify_184_warning.json")
results = {}

# (1) NORMAL real turn -- expect NO warning
log("=== (1) normal real turn (unpatched) ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    resp = S.brain_chat(S.BrainChatRequest(session="mc184_normal", message="what does the brain use",
                                           brain="tiny-demo", reset=True, rich=True))
out = buf.getvalue()
warned = "METACOG WARNING" in out
d = json.loads(bytes(resp.body))
log(f"answer={d.get('answer')!r} metacog={d.get('metacog')} warning_printed={warned}")
results["normal_turn"] = {"answer": d.get("answer"), "metacog": d.get("metacog"), "warning_printed": warned}

# (2) SIMULATED plumbing bug -- wrap query_patient to answer normally then wipe last_trace, reproducing the
# TieredFactStore.__setattr__ shape (answer produced, trace-capable composer, trace comes back empty).
log("=== (2) simulated plumbing bug (query_patient answers, then last_trace silently wiped) ===")
_orig_query_patient = OneBrainComposer.query_patient


def _patched_query_patient(self, *a, **kw):
    ans = _orig_query_patient(self, *a, **kw)
    self.last_trace = None   # simulate the TieredFactStore-class silent wipe
    return ans


buf2 = io.StringIO()
with mock.patch.object(OneBrainComposer, "query_patient", _patched_query_patient):
    with contextlib.redirect_stdout(buf2):
        resp2 = S.brain_chat(S.BrainChatRequest(session="mc184_bug", message="what does the brain use",
                                                brain="tiny-demo", reset=True, rich=True))
out2 = buf2.getvalue()
warned2 = "METACOG WARNING" in out2
d2 = json.loads(bytes(resp2.body))
log(f"answer={d2.get('answer')!r} metacog={d2.get('metacog')} warning_printed={warned2}")
if warned2:
    line = next((ln for ln in out2.splitlines() if "METACOG WARNING" in ln), None)
    log(f"  warning line: {line}")
    results["bug_warning_line"] = line
results["bug_turn"] = {"answer": d2.get("answer"), "metacog": d2.get("metacog"), "warning_printed": warned2}

# (3) GENUINE out-of-scope: an abstain (unknown cue) -- expect NO warning
log("=== (3) genuine abstain (unknown/unstored cue) ===")
buf3 = io.StringIO()
with contextlib.redirect_stdout(buf3):
    resp3 = S.brain_chat(S.BrainChatRequest(session="mc184_abstain", message="what does the zzznope use",
                                            brain="tiny-demo", reset=True, rich=True))
out3 = buf3.getvalue()
warned3 = "METACOG WARNING" in out3
d3 = json.loads(bytes(resp3.body))
log(f"answer={d3.get('answer')!r} abstained={d3.get('abstained')} warning_printed={warned3}")
results["abstain_turn"] = {"answer": d3.get("answer"), "abstained": d3.get("abstained"), "warning_printed": warned3}

results["verdict"] = {
    "normal_turn_quiet": (results["normal_turn"]["warning_printed"] is False),
    "bug_turn_warns": (results["bug_turn"]["warning_printed"] is True),
    "abstain_turn_quiet": (results["abstain_turn"]["warning_printed"] is False),
}
results["verdict"]["GUARD_WORKS"] = all(results["verdict"].values())
log(f"VERDICT: {json.dumps(results['verdict'], indent=2)}")

os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
with open(_ART, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"wrote {_ART}")
