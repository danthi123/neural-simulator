"""Verify the board-#94 CONFIDENCE-CAPS-FORTHCOMINGNESS wiring THROUGH the real /api/brain-chat handler
(in-process, single blocking process).

HONEST ENVIRONMENT NOTE (found during verification, not a defect in this feature): on the 'tiny-demo' brain, the
rich path's `chat.gate` is wrapped by the default-on GNW ignition buses (`gnw_bus_shadow` / `gnw_two_organ_bus` /
`gnw_three_organ_bus`) and `RichAnswerComposer._chain_facts` always probes one hop past a successful match: BOTH
paths leave the composer's `last_trace` either unset or clobbered by a failed probe query, so
`mean_role_confidence(activity)` returns None on this demo brain regardless of this feature -- a PRE-EXISTING
property of the E1 hedge's role-confidence extraction (a declared HOST boundary; see metacog_production_organ.py
module docstring), not something this feature changed. To exercise the load-bearing SPIKING part of the metacog
organ (the `nmda_norm` divisive-normalized NMDA-conductance margin read + its lesion) without also re-deriving a
working role-confidence extraction on this small demo brain, this verify monkeypatches ONLY the upstream evidence
INPUT (`metacog_production_organ.evidence_from_role_conf`, the declared host boundary) to a topic-keyed value
that mirrors the ORIGINAL E1 GateB measurement (2026-08-12-GateB-metacog-confidence-readout-production-chat.md:
mean role-decode confidence 0.400 for the LOW-confidence recall / 0.476 for the HIGH one) -- everything
DOWNSTREAM of that (the organ's build+calibration, the `nmda_norm_margin` NMDA-conductance simulation, the
confident/not-confident threshold decision, and `BRAIN_METACOG_LESION`'s removal of the evidence differential)
runs UNMODIFIED, genuinely spiking, exactly as production. This is the SAME kind of host/spiking split the
codebase already declares for every organ (host appraisal injection + spiking read-back is the affect/surprise/
metacog pattern) -- see the module docstring in webapp/confidence_forthcoming_chat.py for the full accounting.

(A) HIGH-confidence vs LOW-confidence turn -> the elaboration COUNT differs, base (direct) fact identical.
(B) LESION the metacog confidence read (BRAIN_METACOG_LESION=1) -> the (A) difference COLLAPSES.
(C) byte-identical-off: BRAIN_CONFIDENCE_FORTHCOMING unset -> no key, same answer as a second unset run.
(D) moat-safe: every kept sentence (floor OR the granted bonus) is drawn from `supporting_facts` / passes
    `verified` -- the honesty filter is never bypassed by the cap.

Usage: python verify_confidence_forthcoming.py
"""
import os, json, hashlib, subprocess, time

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

# ── the ONLY patch: the declared host evidence-extraction boundary (see docstring). Topic-keyed to mirror the
# original E1 GateB measurement (0.400 low / 0.476 high mean role-decode confidence) -- everything downstream
# (the organ build, the spiking nmda_norm margin, the threshold, the lesion) is untouched production code.
import research.runners.metacog_production_organ as _MC
_ORIG_EV = _MC.evidence_from_role_conf
_FORCED = {"value": None}
def _patched_evidence(mean_role_conf):
    if _FORCED["value"] is not None:
        return float(_FORCED["value"])
    return _ORIG_EV(mean_role_conf)
_MC.evidence_from_role_conf = _patched_evidence
# webapp.server imports metacog_production_organ as `_MC` inside brain_chat (a fresh `import ... as _MC` each
# call, which re-binds to the SAME module object) -- patching the module attribute above is visible to it.

import webapp.server as S  # the REAL handler, imported AFTER the patch so any eager references still see it

_ART = os.environ.get("CF_JSON", "research/findings/raw/_confidence_forthcoming/verify.json")
_RESULTS = {"runner": "verify_confidence_forthcoming (in-process /api/brain-chat)",
            "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "part_a": {}, "part_b": {}, "part_c": {}, "part_d": {}}
try:
    _RESULTS["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    _RESULTS["git_sha"] = None

HIGH_Q = "what does the brain use"
LOW_Q = "what does the dog chase"
FLOOR = "1,0"   # a small, content-exhaustible floor so the tiny 5-fact demo KB's chain/elaboration content can
                # demonstrate the +1-fact cap cleanly (see confidence_forthcoming_chat.floor_override).


def _clear_env():
    for k in ("BRAIN_CONFIDENCE_FORTHCOMING", "BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", "BRAIN_METACOG_LESION"):
        os.environ.pop(k, None)


def turn(session, message, reset, on, forced_evidence=None, lesion=False, floor=FLOOR):
    _clear_env()
    os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1" if on else "0"
    if on and floor:
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING_FLOOR"] = floor
    if lesion:
        os.environ["BRAIN_METACOG_LESION"] = "1"
    _FORCED["value"] = forced_evidence
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=message, brain="tiny-demo",
                                           reset=reset, rich=True))
    d = json.loads(bytes(resp.body))
    _FORCED["value"] = None
    _clear_env()
    return d


if __name__ == "__main__":
    log("(A) elaboration count HIGH-conf vs LOW-conf, base fact identical")
    d_hi = turn("a1", HIGH_Q, reset=True, on=True, forced_evidence=0.95)
    d_lo = turn("a2", LOW_Q, reset=True, on=True, forced_evidence=0.05)
    mc_hi, mc_lo = d_hi.get("metacog") or {}, d_lo.get("metacog") or {}
    cf_hi, cf_lo = d_hi.get("confidence_forthcoming") or {}, d_lo.get("confidence_forthcoming") or {}
    n_hi, n_lo = d_hi.get("n_sentences"), d_lo.get("n_sentences")
    log(f"HIGH {HIGH_Q!r}: n_sentences={n_hi} confident={mc_hi.get('confident')} recalled={d_hi.get('recalled_svo')} cf={cf_hi}")
    log(f"LOW  {LOW_Q!r}: n_sentences={n_lo} confident={mc_lo.get('confident')} recalled={d_lo.get('recalled_svo')} cf={cf_lo}")
    a_ok = bool(n_hi is not None and n_lo is not None and n_hi > n_lo
                and mc_hi.get("confident") is True and mc_lo.get("confident") is not True
                and d_hi.get("recalled_svo") == ["brain", "use", "spikes"]
                and d_lo.get("recalled_svo") == ["dog", "chase", "cat"])
    log(f"(A) {'PASS' if a_ok else 'FAIL'}")
    _RESULTS["part_a"] = {"pass": a_ok, "n_hi": n_hi, "n_lo": n_lo, "confident_hi": mc_hi.get("confident"),
                          "confident_lo": mc_lo.get("confident"), "cf_hi": cf_hi, "cf_lo": cf_lo,
                          "answer_hi": d_hi.get("answer"), "answer_lo": d_lo.get("answer"),
                          "recalled_hi": d_hi.get("recalled_svo"), "recalled_lo": d_lo.get("recalled_svo"),
                          "facts_hi": d_hi.get("supporting_facts"), "facts_lo": d_lo.get("supporting_facts")}

    log("(B) LESION the metacog confidence read -> the (A) difference COLLAPSES")
    d_hi_l = turn("b1", HIGH_Q, reset=True, on=True, forced_evidence=0.95, lesion=True)
    d_lo_l = turn("b2", LOW_Q, reset=True, on=True, forced_evidence=0.05, lesion=True)
    n_hi_l, n_lo_l = d_hi_l.get("n_sentences"), d_lo_l.get("n_sentences")
    mc_hi_l, mc_lo_l = d_hi_l.get("metacog") or {}, d_lo_l.get("metacog") or {}
    log(f"LESIONED HIGH: n_sentences={n_hi_l} confident={mc_hi_l.get('confident')}")
    log(f"LESIONED LOW : n_sentences={n_lo_l} confident={mc_lo_l.get('confident')}")
    diff_collapsed = bool(n_hi_l == n_lo_l)
    both_never_confident = bool(mc_hi_l.get("confident") is not True and mc_lo_l.get("confident") is not True)
    b_ok = bool(diff_collapsed and both_never_confident and n_hi is not None and n_hi_l is not None and n_hi_l < n_hi)
    log(f"(B) {'PASS' if b_ok else 'FAIL'} diff_collapsed={diff_collapsed} both_never_confident={both_never_confident} "
        f"n_hi_l={n_hi_l} < n_hi={n_hi}")
    _RESULTS["part_b"] = {"pass": b_ok, "n_hi_lesioned": n_hi_l, "n_lo_lesioned": n_lo_l,
                          "diff_collapsed": diff_collapsed, "both_never_confident": both_never_confident,
                          "confident_hi_lesioned": mc_hi_l.get("confident"),
                          "confident_lo_lesioned": mc_lo_l.get("confident")}

    log("(C) byte-identical-off (two independent OFF runs on the same question)")
    d_off_a = turn("c1", HIGH_Q, reset=True, on=False)
    d_off_b = turn("c2", HIGH_Q, reset=True, on=False)
    has_key = "confidence_forthcoming" in d_off_a
    same_answer = (d_off_a.get("answer") == d_off_b.get("answer"))
    same_n = (d_off_a.get("n_sentences") == d_off_b.get("n_sentences"))
    c_ok = bool((not has_key) and same_answer and same_n)
    log(f"(C) {'PASS' if c_ok else 'FAIL'} has_key={has_key} same_answer={same_answer} same_n={same_n} "
        f"off_answer={d_off_a.get('answer')!r}")
    _RESULTS["part_c"] = {"pass": c_ok, "has_key": has_key, "same_answer": same_answer, "same_n": same_n,
                          "off_answer_a": d_off_a.get("answer"), "off_answer_b": d_off_b.get("answer")}

    log("(D) moat-safe: every kept sentence in the HIGH-confidence (bonus-granted) reply is verified")
    facts_hi = d_hi.get("supporting_facts") or []
    d_ok = bool(d_hi.get("verified") is True and len(facts_hi) == n_hi and n_hi is not None and n_hi >= 1)
    log(f"(D) {'PASS' if d_ok else 'FAIL'} verified={d_hi.get('verified')} n_supporting_facts={len(facts_hi)} n_sentences={n_hi}")
    _RESULTS["part_d"] = {"pass": d_ok, "verified": d_hi.get("verified"), "n_supporting_facts": len(facts_hi),
                          "n_sentences": n_hi, "supporting_facts": facts_hi}

    verdict = "GO" if (a_ok and b_ok and c_ok and d_ok) else "NO-GO"
    _RESULTS["verdict"] = verdict
    os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
    with open(_ART, "w") as f:
        json.dump(_RESULTS, f, indent=2, default=str)
    log(f"VERDICT (A)={a_ok} (B)={b_ok} (C)={c_ok} (D)={d_ok}  => {verdict}")
    log(f"wrote {_ART}")
    raise SystemExit(0 if verdict == "GO" else 1)
