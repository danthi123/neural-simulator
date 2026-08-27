"""Lean re-run of JUST check (F) from verify_confidence_forthcoming_prodflip.py, after fixing a TEST-HARNESS bug
(the floor override was not applied to the UNSET arm, so it ran at a different natural floor than the explicit
ON=1 arm -- a floor mismatch masquerading as an ON-vs-OFF difference; A-E were unaffected and already passed
clean in the first run, artifact verify.json). Re-runs only the 3 sessions (F) needs instead of the full 10.
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

t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

import research.runners.metacog_production_organ as _MC
_ORIG_EV = _MC.evidence_from_role_conf
_FORCED = {"value": 0.95}
def _patched_evidence(mean_role_conf):
    if _FORCED["value"] is not None:
        return float(_FORCED["value"])
    return _ORIG_EV(mean_role_conf)
_MC.evidence_from_role_conf = _patched_evidence

import webapp.server as S

HIGH_Q = "what does the brain use"
FLOOR = "1,0"


def _clear():
    for k in ("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", "BRAIN_METACOG_LESION"):
        os.environ.pop(k, None)


def turn(session, mode):
    """mode: 'unset' | 'on1' | 'off0'."""
    _clear()
    if mode in ("unset", "on1"):
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING_FLOOR"] = FLOOR
    if mode == "on1":
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
    elif mode == "off0":
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
    else:
        os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING", None)
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=HIGH_Q, brain="tiny-demo", reset=True, rich=True))
    d = json.loads(bytes(resp.body))
    _clear()
    os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING", None)
    return d


if __name__ == "__main__":
    log("(F) DEFAULT-ON GUARD, fixed: unset==ON(1), unset!=OFF(0), all 3 arms in ONE process, same question")
    d_unset = turn("pf1v2", "unset")
    d_on1 = turn("pf2v2", "on1")
    d_off0 = turn("pf3v2", "off0")
    unset_eq_on = (d_unset.get("n_sentences") == d_on1.get("n_sentences")
                  and ("confidence_forthcoming" in d_unset) == ("confidence_forthcoming" in d_on1))
    unset_ne_off = ("confidence_forthcoming" in d_unset) and ("confidence_forthcoming" not in d_off0)
    f_ok = bool(unset_eq_on and unset_ne_off)
    log(f"(F) {'PASS' if f_ok else 'FAIL'} unset_n={d_unset.get('n_sentences')} on1_n={d_on1.get('n_sentences')} "
        f"off0_n={d_off0.get('n_sentences')} unset_has_key={'confidence_forthcoming' in d_unset} "
        f"off0_has_key={'confidence_forthcoming' in d_off0} unset_cf={d_unset.get('confidence_forthcoming')} "
        f"on1_cf={d_on1.get('confidence_forthcoming')}")
    result = {"pass": f_ok, "unset_eq_on": unset_eq_on, "unset_ne_off": unset_ne_off,
              "n_unset": d_unset.get("n_sentences"), "n_on1": d_on1.get("n_sentences"),
              "n_off0": d_off0.get("n_sentences"), "cf_unset": d_unset.get("confidence_forthcoming"),
              "cf_on1": d_on1.get("confidence_forthcoming")}
    out = "research/findings/raw/_confidence_forthcoming_prodflip/verify_part_f_fixed.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log(f"wrote {out}")
    raise SystemExit(0 if f_ok else 1)
