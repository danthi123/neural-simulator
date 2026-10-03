"""THE host-share schema (plan step 2, docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md section 2).

One definition of "host share" for every builder that reports it. `research/runners/host_share.py` (the baseline) uses
it, and the step-3 selection-seam builder imports it for its before/after instead of keeping its own statistic. Pure
functions over plain dicts, no brain build, no project imports.

SPEC (frozen in research/findings/2026-09-29-host-share-baselines-PREREGISTRATION.md section 5)

An EVENT is one seam x one target turn x one seed. It carries:
  F   INTERVENTION (the primary causal measure). Hold the neural state at the seam fixed, set the host decision to its
      preregistered alternative. F = 1 if the reply's seam outcome equals the host alternative (and the alternative
      differs from the intact outcome), 0 if it does not, None = UNDEFINED.
  f_outcome   for F = 0: "returned_to_neural" when the reply's seam outcome equals the NEURAL-DERIVED decision's
      outcome and that differs from the host alternative; otherwise "masked" (it returned to the intact reply through a
      stage that is not the neural preference, an LTM fallback, or some other change).
  S   SUBSTITUTION. Replace the host rule with the neural-derived rule. S = 1 if a seam-downstream reply field changes,
      0 if not, None = UNDEFINED (no neural-derived alternative exists, a side effect upstream of the seam, an arm
      missing).
  D   DECODING (a screen, never evidence of ownership). 1 if the neural-derived decision equals the host decision.

Buckets, by INTERVENTION outcome (an F = 1 event is never counted as neural):
  host_decided              F = 1 and S = 1  (the host flip reaches the reply; the neural rule would decide otherwise)
  host_causal_replaceable   F = 1 and S = 0  (the host flip reaches the reply; the neural rule happens to agree)
  host_causal_s_undefined   F = 1 and S undefined
  neural_decided            F = 0 and f_outcome == "returned_to_neural"
  masked                    F = 0 otherwise
  undefined                 F undefined

Per seam: HS = mean F; NS = neural_decided share; MS = masked share (each over events with F defined); SC = mean S;
DA = mean D. Class: HOST-OWNED if HS >= 0.8 (qualified "replaceable" when SC < 0.2); NEURAL-OWNED if NS >= 0.8;
MASKED if MS >= 0.8; MIXED otherwise; UNDEFINED when the seam is live on fewer than `min_live_seeds` seeds or no F is
defined. A seam with no downstream neural path (the host decision's only route to the reply is host code or a pure
function of the host value) is a CALL-GRAPH CHECK: classified the same way, reported separately, and EXCLUDED from the
pooled panel, because its F is fixed by the call graph rather than measured.

Panel (pooled over the measured seams only): host_decision_share = #(F = 1) / #(F defined);
neural_decision_fraction = #neural_decided / #(F defined); masked_fraction likewise; replaceable_share_of_host =
#host_causal_replaceable / #(F = 1). The learned-content fraction is a separate quantity (host_share.content_fractions),
never folded into these.
"""
from __future__ import annotations

BUCKETS = ("host_decided", "host_causal_replaceable", "host_causal_s_undefined", "neural_decided", "masked",
           "undefined")
CLASSES = ("HOST-OWNED", "NEURAL-OWNED", "MASKED", "MIXED", "UNDEFINED")
HS_BAR = 0.8
NS_BAR = 0.8
MS_BAR = 0.8
SC_REPLACEABLE_BELOW = 0.2
MIN_LIVE_SEEDS_GATE = 4


def event_bucket(F, S, f_outcome=None):
    """The bucket of one event, by its INTERVENTION outcome. F = 1 is host-causal whatever S says."""
    if F is None:
        return "undefined"
    if F == 1:
        if S is None:
            return "host_causal_s_undefined"
        return "host_decided" if S == 1 else "host_causal_replaceable"
    return "neural_decided" if f_outcome == "returned_to_neural" else "masked"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (float(sum(xs)) / len(xs)) if xs else None


def seam_stats(events):
    """HS / NS / MS / SC / DA and bucket counts over one seam's LIVE events (dicts with F, S, D, f_outcome, seed)."""
    live = [e for e in events if e.get("live")]
    fdef = [e for e in live if e.get("F") is not None]
    buckets = {b: 0 for b in BUCKETS}
    for e in live:
        buckets[event_bucket(e.get("F"), e.get("S"), e.get("f_outcome"))] += 1
    nf = len(fdef)
    return {
        "n_live": len(live), "n_F": nf,
        "n_S": sum(1 for e in live if e.get("S") is not None),
        "n_D": sum(1 for e in live if e.get("D") is not None),
        "live_seeds": sorted({e.get("seed") for e in live}),
        "HS": _mean([e["F"] for e in fdef]),
        "NS": (buckets["neural_decided"] / float(nf)) if nf else None,
        "MS": (buckets["masked"] / float(nf)) if nf else None,
        "SC": _mean([e.get("S") for e in live]),
        "DA": _mean([e.get("D") for e in live]),
        "buckets": buckets,
    }


def classify_seam(stats, min_live_seeds=MIN_LIVE_SEEDS_GATE):
    """The per-seam class from `seam_stats` output. Never returns a class on too little live evidence."""
    if stats is None or len(stats.get("live_seeds") or []) < int(min_live_seeds) or stats.get("HS") is None:
        return "UNDEFINED"
    if stats["HS"] >= HS_BAR:
        sc = stats.get("SC")
        return "HOST-OWNED (replaceable)" if (sc is not None and sc < SC_REPLACEABLE_BELOW) else "HOST-OWNED"
    if stats.get("NS") is not None and stats["NS"] >= NS_BAR:
        return "NEURAL-OWNED"
    if stats.get("MS") is not None and stats["MS"] >= MS_BAR:
        return "MASKED"
    return "MIXED"


def panel(events, measured_seams):
    """Pooled host-decision share / neural-decision fraction over the MEASURED seams (downstream neural path = yes)."""
    ev = [e for e in events if e.get("live") and e.get("seam") in set(measured_seams) and e.get("F") is not None]
    n = len(ev)
    b = {k: 0 for k in BUCKETS}
    for e in ev:
        b[event_bucket(e["F"], e.get("S"), e.get("f_outcome"))] += 1
    n_host = b["host_decided"] + b["host_causal_replaceable"] + b["host_causal_s_undefined"]
    return {
        "measured_seams": list(measured_seams), "n_events_F_defined": n, "buckets": b,
        "host_decision_share": (n_host / float(n)) if n else None,
        "neural_decision_fraction": (b["neural_decided"] / float(n)) if n else None,
        "masked_fraction": (b["masked"] / float(n)) if n else None,
        "replaceable_share_of_host": (b["host_causal_replaceable"] / float(n_host)) if n_host else None,
    }


def selftest():
    checks = {
        "F=1,S=0 is host-causal (replaceable), NEVER neural": event_bucket(1, 0) == "host_causal_replaceable",
        "F=1,S=1 is host_decided": event_bucket(1, 1) == "host_decided",
        "F=1,S=None is host-causal with S undefined": event_bucket(1, None) == "host_causal_s_undefined",
        "F=0 returned to the neural preference is neural": event_bucket(0, 1, "returned_to_neural") == "neural_decided",
        "F=0 otherwise is masked, not neural": event_bucket(0, 0, "masked") == "masked",
        "F=None is undefined": event_bucket(None, 1) == "undefined",
    }
    evs = [dict(seam="a", live=True, F=1, S=0, D=1, seed=s) for s in (1, 2, 3, 4)]
    st = seam_stats(evs)
    checks["an all-(F=1,S=0) seam is HOST-OWNED (replaceable), not neural"] = (
        classify_seam(st) == "HOST-OWNED (replaceable)" and st["NS"] == 0.0)
    p = panel(evs, ["a"])
    checks["panel counts F=1,S=0 as host share 1.0 and neural 0.0"] = (
        p["host_decision_share"] == 1.0 and p["neural_decision_fraction"] == 0.0)
    checks["too few live seeds -> UNDEFINED"] = classify_seam(seam_stats(evs[:3])) == "UNDEFINED"
    nev = [dict(seam="b", live=True, F=0, S=1, f_outcome="returned_to_neural", seed=s) for s in (1, 2, 3, 4)]
    checks["all returned-to-neural -> NEURAL-OWNED"] = classify_seam(seam_stats(nev)) == "NEURAL-OWNED"
    mev = [dict(seam="c", live=True, F=0, S=0, f_outcome="masked", seed=s) for s in (1, 2, 3, 4)]
    checks["all masked -> MASKED"] = classify_seam(seam_stats(mev)) == "MASKED"
    checks["call-graph seams stay out of the panel"] = panel(evs + nev, ["b"])["host_decision_share"] == 0.0
    return all(checks.values()), checks


if __name__ == "__main__":
    ok, checks = selftest()
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("SCHEMA SELFTEST", "OK" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)
