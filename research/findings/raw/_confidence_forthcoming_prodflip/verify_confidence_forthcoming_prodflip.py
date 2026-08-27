"""PRODUCTION-DEFAULT verify for board #94 confidence-caps-forthcomingness (2026-08-27 flip), THROUGH the real
`/api/brain-chat` handler, in-process, on the NOW-LIVE production brain (`brain="tiny-demo"`, which since the
2026-08-26 `tiered-knowledge-ltm` default-on flip auto-attaches the shipped 15k-fact wikidata cortical LTM --
`source` comes back `"tiny-demo +LTM"` whenever the bundle dir is present on disk).

WHAT CHANGED SINCE THE ORIGINAL 2026-08-27 GO (`verify_confidence_forthcoming.py`, tiny-demo only):

  (1) A REAL BUG FOUND + FIXED while doing this production verification: `TieredFactStore` (research/runners/
      tiered_fact_store.py) had NO `__setattr__` override, so `webapp/server.py`'s per-turn activity-trace flip
      (`_composer.trace = True; _composer.last_trace = None`) silently created SHADOW instance attributes on the
      TieredFactStore WRAPPER instead of reaching `self.buffer` -- `self.buffer.trace` was NEVER actually set,
      so `last_trace` was NEVER recorded, so `activity`/`metacog.confident` read None on EVERY `tiny-demo +LTM`
      turn since the 2026-08-26 knowledge-core flip, regardless of the turn's real confidence. Fixed by adding
      `TieredFactStore.__setattr__` (forwards every attribute except `buffer`/`ltm` to `self.buffer`, matching
      the class's own documented "delegates every other attribute ... to the buffer" contract). Verified below
      (check E1): `activity` is no longer None on a real `tiny-demo +LTM` turn.

  (2) A CORRECTION to the original finding's stated honest residual. The original finding hoped "moving
      verification to a richer knowledge base" (i.e. the 2026-08-26 LTM flip) would dissolve the `mean_role_
      confidence`-returns-None residual. It does NOT: the root cause is structural and independent of KB size --
      `RichAnswerComposer._chain_facts` (max_chain_hops=3) ALWAYS issues one MORE `composer.query_patient` call
      past a successful direct match (trying to extend the chain), and `OneBrainComposer.query_patient` resets
      `self.last_trace = None` UNCONDITIONALLY at the top of every call (before checking whether this NEW query
      even matches) -- so the chain's inevitable dead-end hop clobbers the GOOD trace the direct match left
      behind, regardless of whether the direct answer came from the tiny buffer OR the 15k-fact LTM. Confirmed
      empirically below (check E2/E3): even a genuine multi-sentence BUFFER-sourced answer ("what does the brain
      use" -> 3 sentences) and a genuine LTM-sourced answer (frank_lincoln_wright's occupation) BOTH still read
      `metacog: None` on this production brain even with fix (1) applied. This is a SEPARATE, deeper residual
      (the chain-vs-trace interaction) than what fix (1) touches; NOT attempted here (a `_chain_facts`/`_trace_
      query` semantics change has a large blast radius across every OTHER consumer of `last_trace` and is out of
      scope for a production-default flip -- flagged as a follow-on).

  (3) Because of (2), checks A/B below still use the SAME declared evidence-forcing technique the original
      finding used (monkeypatching the declared host boundary `metacog_production_organ.evidence_from_role_
      conf`) -- but now run against the GENUINELY-LIVE `tiny-demo +LTM` brain (not a stripped/LTM-off build), and
      paired with a NEW check (E) proving the LTM tier is genuinely reachable + verified in the SAME test
      session, and a NEW check (F) proving the DEFAULT-ON flip's OFF-escape is genuine (guards the
      `os.environ.pop()`-as-OFF staleness pattern named in the task brief: an unset env var must now read as ON,
      not OFF, and both arms are exercised in ONE process without cross-contamination).

(A) HIGH-confidence vs LOW-confidence turn -> the elaboration COUNT differs, base (direct) fact identical.
(B) LESION the metacog confidence read (BRAIN_METACOG_LESION=1) -> the (A) difference COLLAPSES.
(C) explicit-OFF (BRAIN_CONFIDENCE_FORTHCOMING=0) is byte-identical across two independent runs, no key.
(D) moat-safe: every kept sentence in the HIGH-confidence (bonus-granted) reply is VERIFIED.
(E) LTM reachability: a genuine, UNPATCHED query against a real 7-fact wikidata LTM subject
    (frank_lincoln_wright) is answered correctly + verified through the SAME real handler + brain build used for
    A-D (source == "tiny-demo +LTM"), proving this test's environment genuinely has the live production
    knowledge base attached (not stripped) -- and confirms fix (1) (activity no longer None).
(F) DEFAULT-ON GUARD: env UNSET (after an explicit .pop(), the true "caller forgot to set it" case) behaves
    IDENTICALLY to explicit BRAIN_CONFIDENCE_FORTHCOMING=1 (both ON), and DIFFERENTLY from explicit
    BRAIN_CONFIDENCE_FORTHCOMING=0 (OFF) -- all three arms exercised in ONE process on the SAME question.

Usage: SIM_BACKEND=numpy python verify_confidence_forthcoming_prodflip.py
"""
import os, json, hashlib, subprocess, time

os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
# isolate the confidence_forthcoming signal from every OTHER default-on faculty that also touches
# rich.max_sentences/max_elaborations or the composer's trace -- the SAME isolation the original tiny-demo
# verify + the bg-action-selection flip-soak use (PART A/B pattern). The 6-seed FULL-pipeline no-regression
# soak (a separate script) is what verifies the un-isolated, all-faculties-on production default.
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

# ── the ONLY patch for checks A/B/D/F: the declared host evidence-extraction boundary (see module docstring
# point 2/3). Topic-keyed to mirror the original E1 GateB measurement (0.400 low / 0.476 high mean role-decode
# confidence) -- everything downstream (the organ build, the spiking nmda_norm margin, the threshold, the
# lesion) is untouched production code. Check E runs WITHOUT this patch (genuine, unpatched evidence).
import research.runners.metacog_production_organ as _MC
_ORIG_EV = _MC.evidence_from_role_conf
_FORCED = {"value": None}
def _patched_evidence(mean_role_conf):
    if _FORCED["value"] is not None:
        return float(_FORCED["value"])
    return _ORIG_EV(mean_role_conf)
_MC.evidence_from_role_conf = _patched_evidence

import webapp.server as S  # the REAL handler, imported AFTER the patch so any eager references still see it

_ART = os.environ.get("CF_JSON", "research/findings/raw/_confidence_forthcoming_prodflip/verify.json")
_RESULTS = {"runner": "verify_confidence_forthcoming_prodflip (in-process /api/brain-chat, tiny-demo +LTM)",
            "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "part_a": {}, "part_b": {}, "part_c": {}, "part_d": {}, "part_e": {}, "part_f": {}}
try:
    _RESULTS["git_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    _RESULTS["git_sha"] = None

HIGH_Q = "what does the brain use"
LOW_Q = "what does the dog chase"
LTM_Q = "what occupation is frank_lincoln_wright"
FLOOR = "1,0"


def _clear_cf_env():
    for k in ("BRAIN_CONFIDENCE_FORTHCOMING_FLOOR", "BRAIN_METACOG_LESION"):
        os.environ.pop(k, None)


def turn(session, message, reset, on, forced_evidence=None, lesion=False, floor=FLOOR, explicit_off=None,
        want_enabled=None):
    """on=True -> BRAIN_CONFIDENCE_FORTHCOMING=1 explicit. on=False -> explicit_off decides: True -> "0" explicit
    (the guarded OFF-escape); None/False with on=False and explicit_off not requested -> POP the var (the
    real-world "caller never set it" case, which must now read as the shipped DEFAULT -- ON).

    `want_enabled` (bug fix, 2026-08-27 -- the first run of this check false-FAILed on a TEST-HARNESS bug, not a
    product bug): whether THIS turn should behave as "the coupling is enabled" for the purpose of applying the
    FLOOR override -- defaults to `on` (old behavior) but must be passed explicitly True for the UNSET arm (which
    IS enabled, by the new default, even though `on=False` is used here only to mean "do not set the env var
    explicitly to 1"). Without this, the unset arm never got `BRAIN_CONFIDENCE_FORTHCOMING_FLOOR` applied, so it
    ran at the composer's UNRELATED natural construction-default floor while the explicit-ON=1 arm ran at the
    forced floor="1,0" -- a floor MISMATCH masquerading as an ON-vs-OFF difference. Fixed by decoupling "which
    floor to request" from "how the enable flag is spelled"."""
    _clear_cf_env()
    _enabled = on if want_enabled is None else want_enabled
    if _enabled and floor:
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING_FLOOR"] = floor
    if on:
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1"
    elif explicit_off:
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
    else:
        os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING", None)
    if lesion:
        os.environ["BRAIN_METACOG_LESION"] = "1"
    _FORCED["value"] = forced_evidence
    resp = S.brain_chat(S.BrainChatRequest(session=session, message=message, brain="tiny-demo",
                                           reset=reset, rich=True))
    d = json.loads(bytes(resp.body))
    _FORCED["value"] = None
    _clear_cf_env()
    os.environ.pop("BRAIN_CONFIDENCE_FORTHCOMING", None)
    return d


if __name__ == "__main__":
    log("(A) elaboration count HIGH-conf vs LOW-conf, base fact identical (explicit ON=1)")
    d_hi = turn("pa1", HIGH_Q, reset=True, on=True, forced_evidence=0.95)
    d_lo = turn("pa2", LOW_Q, reset=True, on=True, forced_evidence=0.05)
    mc_hi, mc_lo = d_hi.get("metacog") or {}, d_lo.get("metacog") or {}
    cf_hi, cf_lo = d_hi.get("confidence_forthcoming") or {}, d_lo.get("confidence_forthcoming") or {}
    n_hi, n_lo = d_hi.get("n_sentences"), d_lo.get("n_sentences")
    src_hi, src_lo = d_hi.get("source"), d_lo.get("source")
    log(f"HIGH {HIGH_Q!r}: n_sentences={n_hi} confident={mc_hi.get('confident')} recalled={d_hi.get('recalled_svo')} "
        f"source={src_hi} cf={cf_hi}")
    log(f"LOW  {LOW_Q!r}: n_sentences={n_lo} confident={mc_lo.get('confident')} recalled={d_lo.get('recalled_svo')} "
        f"source={src_lo} cf={cf_lo}")
    a_ok = bool(n_hi is not None and n_lo is not None and n_hi > n_lo
                and mc_hi.get("confident") is True and mc_lo.get("confident") is not True
                and d_hi.get("recalled_svo") == ["brain", "use", "spikes"]
                and d_lo.get("recalled_svo") == ["dog", "chase", "cat"]
                and src_hi == "tiny-demo +LTM" and src_lo == "tiny-demo +LTM")
    log(f"(A) {'PASS' if a_ok else 'FAIL'}")
    _RESULTS["part_a"] = {"pass": a_ok, "n_hi": n_hi, "n_lo": n_lo, "confident_hi": mc_hi.get("confident"),
                          "confident_lo": mc_lo.get("confident"), "cf_hi": cf_hi, "cf_lo": cf_lo,
                          "source_hi": src_hi, "source_lo": src_lo,
                          "answer_hi": d_hi.get("answer"), "answer_lo": d_lo.get("answer"),
                          "recalled_hi": d_hi.get("recalled_svo"), "recalled_lo": d_lo.get("recalled_svo"),
                          "facts_hi": d_hi.get("supporting_facts"), "facts_lo": d_lo.get("supporting_facts")}

    log("(B) LESION the metacog confidence read -> the (A) difference COLLAPSES")
    d_hi_l = turn("pb1", HIGH_Q, reset=True, on=True, forced_evidence=0.95, lesion=True)
    d_lo_l = turn("pb2", LOW_Q, reset=True, on=True, forced_evidence=0.05, lesion=True)
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

    log("(C) explicit-OFF byte-identical (two independent BRAIN_CONFIDENCE_FORTHCOMING=0 runs)")
    d_off_a = turn("pc1", HIGH_Q, reset=True, on=False, explicit_off=True)
    d_off_b = turn("pc2", HIGH_Q, reset=True, on=False, explicit_off=True)
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

    log("(E) LTM reachability + fix(1) check: a genuine, UNPATCHED frank_lincoln_wright query on the SAME real brain")
    _FORCED["value"] = None   # explicitly UNPATCHED -- real evidence_from_role_conf, no forcing
    d_ltm = turn("pe1", LTM_Q, reset=True, on=False, explicit_off=True)   # feature OFF here -- this checks reachability, not the cap
    ltm_answered = bool((not d_ltm.get("abstained")) and d_ltm.get("verified") is True
                        and d_ltm.get("recalled_svo") == ["frank_lincoln_wright", "occupation",
                                                           "stanford_downey_architects_inc"])
    ltm_source_ok = (d_ltm.get("source") == "tiny-demo +LTM")
    activity_populated = (d_ltm.get("activity") is not None)   # fix (1): was always None pre-fix
    e_ok = bool(ltm_answered and ltm_source_ok and activity_populated)
    log(f"(E) {'PASS' if e_ok else 'FAIL'} answered={ltm_answered} source={d_ltm.get('source')} "
        f"activity_populated={activity_populated} answer={d_ltm.get('answer')!r} metacog={d_ltm.get('metacog')}")
    _RESULTS["part_e"] = {"pass": e_ok, "ltm_answered": ltm_answered, "source": d_ltm.get("source"),
                          "activity_populated": activity_populated, "answer": d_ltm.get("answer"),
                          "recalled_svo": d_ltm.get("recalled_svo"), "metacog": d_ltm.get("metacog"),
                          "note": "metacog stays None here -- a SEPARATE, deeper residual (chain-vs-trace "
                                  "clobber, see module docstring point 2), NOT what this check tests (reachability "
                                  "+ the TieredFactStore trace passthrough fix)."}

    log("(F) DEFAULT-ON GUARD: unset==ON(1), unset!=OFF(0), all 3 arms in ONE process, same question")
    d_unset = turn("pf1", HIGH_Q, reset=True, on=False, forced_evidence=0.95, explicit_off=False, want_enabled=True)
    d_on1 = turn("pf2", HIGH_Q, reset=True, on=True, forced_evidence=0.95)
    d_off0 = turn("pf3", HIGH_Q, reset=True, on=False, forced_evidence=0.95, explicit_off=True)
    unset_eq_on = (d_unset.get("n_sentences") == d_on1.get("n_sentences")
                  and ("confidence_forthcoming" in d_unset) == ("confidence_forthcoming" in d_on1))
    unset_ne_off = ("confidence_forthcoming" in d_unset) and ("confidence_forthcoming" not in d_off0)
    f_ok = bool(unset_eq_on and unset_ne_off)
    log(f"(F) {'PASS' if f_ok else 'FAIL'} unset_n={d_unset.get('n_sentences')} on1_n={d_on1.get('n_sentences')} "
        f"off0_n={d_off0.get('n_sentences')} unset_has_key={'confidence_forthcoming' in d_unset} "
        f"off0_has_key={'confidence_forthcoming' in d_off0}")
    _RESULTS["part_f"] = {"pass": f_ok, "unset_eq_on": unset_eq_on, "unset_ne_off": unset_ne_off,
                          "n_unset": d_unset.get("n_sentences"), "n_on1": d_on1.get("n_sentences"),
                          "n_off0": d_off0.get("n_sentences")}

    from tools.verdict import Verdict
    v = Verdict("confidence-forthcomingness production-default flip (board #94)")
    v.require("A: vary (HIGH>LOW sentences, confident True/False, tiny-demo+LTM brain)", a_ok, expect=True)
    v.require("B: metacog lesion collapses the (A) vary to zero", b_ok, expect=True)
    v.require("C: explicit-OFF (BRAIN_CONFIDENCE_FORTHCOMING=0) byte-identical across two runs", c_ok, expect=True)
    v.require("D: moat-safe (every kept sentence verified, facts match)", d_ok, expect=True)
    v.require("E: genuine LTM reachability + TieredFactStore.__setattr__ trace-passthrough fix", e_ok, expect=True)
    v.require("F: DEFAULT-ON guard (env-unset==explicit-ON, unset!=explicit-OFF, one process)", f_ok, expect=True)
    decided = v.decide(go=bool(a_ok and b_ok and c_ok and d_ok and e_ok and f_ok))
    _RESULTS.update(decided)          # status/go/preconditions/undefined_reasons -> satisfies gates/verdict_preconditions
    _RESULTS["verdict"] = decided["status"]   # kept for the older bespoke readers of this artifact
    verdict = decided["status"]
    os.makedirs(os.path.dirname(os.path.abspath(_ART)), exist_ok=True)
    with open(_ART, "w") as f:
        json.dump(_RESULTS, f, indent=2, default=str)
    log(f"VERDICT (A)={a_ok} (B)={b_ok} (C)={c_ok} (D)={d_ok} (E)={e_ok} (F)={f_ok}  => {verdict}")
    log(f"wrote {_ART}")
    raise SystemExit(0 if verdict == "GO" else 1)
