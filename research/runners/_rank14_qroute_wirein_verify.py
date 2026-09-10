"""FOCUSED wire-in verify for the rank-14 question-route-selection spiking-WTA PRODUCTION wire-in
(`research/runners/spiking_qroute_selection_organ.py` + `brain_chat_tui.ChatBrain._extract_route` /
`_spiking_route_decision`, behind default-OFF `BRAIN_SPIKING_QROUTE`). Turns the 6/6-GO mechanism de-risk
(`_rank14_question_route_selection_derisk.py`, finding `2026-09-09-rank14-question-route-selection-wta-derisk-GO.md`,
commit d63a39ce8) from a banked de-risk into an actual wire-in.

This is a LIGHT unit verify (tiny <200-neuron nets, numpy, sub-second) -- NOT the integrated brain-chat battery (that
is DEFERRED to an AWS-CPU batch; the dev box is RAM-blocked and the owner is gaming). Two layers:

  L1 ORGAN-DIRECT: SpikingQRouteSelectorOrgan.select picks the correct route per evidence tuple incl. the ambiguous
     RELFRONT/KBREL dual-match (-> RELFRONT by drive strength, 6/6 project seeds); LESION -> GENERIC for every tuple.
  L2 _extract_route END-TO-END on a STUB ChatBrain (no agent/composer/bridge on the OFF path): flag OFF reproduces the
     host priority cascade AND never builds the organ (byte-identical-off); flag ON returns the winning route's
     candidate incl. the ambiguous case; lesion reverts a relf question to the GENERIC heuristic parse.

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._rank14_qroute_wirein_verify \
      --json research/findings/raw/_rank14_qroute_wirein/verify_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import types


SEEDS = (42, 43, 44, 100, 101, 102)


def main():
    ap = argparse.ArgumentParser(description="Rank-14 question-route-selection WTA wire-in focused verify.")
    ap.add_argument("--json", type=str, default="research/findings/raw/_rank14_qroute_wirein/verify_6seed.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", args.backend)
    # Isolate the qroute claim from orthogonal spiking passes so the stub needs no composer/substrate.
    os.environ["BRAIN_KNOWLEDGE_GROUNDING"] = "0"
    os.environ["BRAIN_NEURAL_SELFID"] = "0"

    import research.runners.spiking_qroute_selection_organ as Q
    import research.runners.brain_chat_tui as bct
    from tools.lab import attributable_to
    from tools.verdict import Verdict

    checks = []

    def rec(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)

    # ── L1 ORGAN-DIRECT ──────────────────────────────────────────────────────────────────────────────────────────
    cases = {
        (True, False, False): "RELFRONT",
        (False, True, False): "KBREL",
        (False, False, True): "DEFCOP",
        (False, False, False): "GENERIC",
        (True, True, False): "RELFRONT",       # deliberately ambiguous dual-match -> host priority via drive strength
    }
    exc_tuples = [(True, False, False), (False, True, False), (False, False, True)]   # single-exception evidence
    print("L1 ORGAN-DIRECT (seed 42, intact):", flush=True)
    org = Q.SpikingQRouteSelectorOrgan(seed=42, lesion=False)
    intact_win = {}
    for (r, k, d), exp in cases.items():
        r0 = org.select_record(r, k, d)
        intact_win[(r, k, d)] = r0["winner"]
        rec(f"select{(r, k, d)} -> {exp}", r0["winner"] == exp, f"winner={r0['winner']} margin={r0['margin']:.3f}")
    print("L1 ORGAN-DIRECT (seed 42, LESION -> all GENERIC):", flush=True)
    orgL = Q.SpikingQRouteSelectorOrgan(seed=42, lesion=True)
    lesion_win = {}
    for (r, k, d) in cases:
        r0 = orgL.select_record(r, k, d)
        lesion_win[(r, k, d)] = r0["winner"]
        rec(f"lesioned select{(r, k, d)} -> GENERIC", r0["winner"] == "GENERIC", f"winner={r0['winner']}")
    # ATTRIBUTION (gap#5 discipline): does the SUBSTRATE dispatch, not a host tie-break, OWN the routing? Fraction of
    # the single-exception tuples that route to their (non-GENERIC) exception route, intact vs full-lesion.
    intact_exc_rate = sum(1 for t, exp in cases.items() if t in exc_tuples and intact_win[t] == exp) / len(exc_tuples)
    lesion_exc_rate = sum(1 for t, exp in cases.items() if t in exc_tuples and lesion_win[t] == exp) / len(exc_tuples)
    attribution = attributable_to("exception-route dispatch on the substrate: intact vs full-lesion",
                                  intact_exc_rate, lesion_exc_rate, warn_below=0.5)
    rec("attribution(intact vs full-lesion) >= 0.5", (attribution is None or attribution >= 0.5),
        f"attribution={attribution} intact_exc_rate={intact_exc_rate} lesion_exc_rate={lesion_exc_rate}")
    print("L1 ambiguous (T,T,F)->RELFRONT across 6 seeds:", flush=True)
    for s in SEEDS:
        r0 = Q.SpikingQRouteSelectorOrgan(seed=s, lesion=False).select_record(True, True, False)
        rec(f"seed {s} ambiguous -> RELFRONT", r0["winner"] == "RELFRONT", f"margin={r0['margin']:.3f}")

    # ── L2 _extract_route END-TO-END (stub ChatBrain) ────────────────────────────────────────────────────────────
    RELF_CAND = ["country", "chelsea_fc"]
    KBREL_CAND = ["entity_x", "place_of_birth"]
    DEFO_CAND = ["canada", "isa"]
    L2 = [
        ("what country is chelsea fc from", (RELF_CAND, None, None), RELF_CAND, "RELFRONT"),
        ("where was entity_x born",         (None, KBREL_CAND, None), KBREL_CAND, "KBREL"),
        ("what is canada",                  (None, None, DEFO_CAND), DEFO_CAND, "DEFCOP"),
        ("what does the wolf hunt",         (None, None, None), ["wolf", "hunt"], "GENERIC"),
        ("what country is entity_x a citizen of", (RELF_CAND, KBREL_CAND, None), RELF_CAND, "RELFRONT"),  # ambiguous
    ]

    def make_stub():
        st = types.SimpleNamespace()
        st.agents_set = set()
        st.actions_set = set()
        st.patients_set = set()
        st.router = types.SimpleNamespace(self_aliases=set(), _resolve_self=lambda t: t)
        st.inner = types.SimpleNamespace(composer=None)
        st.agent = types.SimpleNamespace(seed=42)
        st._qroute_organ = None
        st._decision_calls = [0]
        st._cand = {"relf": None, "kbrel": None, "defcop": None}
        st._relation_fronted_route = lambda q: st._cand["relf"]
        st._kb_relation_question_route = lambda q: st._cand["kbrel"]
        st._definitional_copula_route = lambda q: st._cand["defcop"]
        st._neural_question_parse = lambda content: None

        def _dec(a, b, c):
            st._decision_calls[0] += 1
            return bct.ChatBrain._spiking_route_decision(st, a, b, c)
        st._spiking_route_decision = _dec
        return st

    print("L2 _extract_route OFF (byte-identical host priority; organ NEVER built):", flush=True)
    os.environ["BRAIN_SPIKING_QROUTE"] = "0"
    for q, (rc, kc, dc), host_out, _route in L2:
        st = make_stub()
        st._cand = {"relf": rc, "kbrel": kc, "defcop": dc}
        out = bct.ChatBrain._extract_route(st, q)
        ok = (out == host_out) and st._decision_calls[0] == 0 and st._qroute_organ is None
        rec(f"OFF {q!r} -> host priority", ok, f"out={out} decision_calls={st._decision_calls[0]}")

    print("L2 _extract_route ON (WTA dispatch returns the winning route's candidate):", flush=True)
    os.environ["BRAIN_SPIKING_QROUTE"] = "1"
    os.environ["BRAIN_SPIKING_QROUTE_LESION"] = "0"
    for q, (rc, kc, dc), host_out, route in L2:
        st = make_stub()
        st._cand = {"relf": rc, "kbrel": kc, "defcop": dc}
        out = bct.ChatBrain._extract_route(st, q)
        rec(f"ON {q!r} -> {route}", out == host_out and st._decision_calls[0] >= 1, f"out={out}")

    print("L2 _extract_route ON+LESION (relf question reverts to GENERIC heuristic):", flush=True)
    os.environ["BRAIN_SPIKING_QROUTE"] = "1"
    os.environ["BRAIN_SPIKING_QROUTE_LESION"] = "1"
    st = make_stub()
    st._cand = {"relf": RELF_CAND, "kbrel": None, "defcop": None}
    out = bct.ChatBrain._extract_route(st, "what country is chelsea fc from")
    rec("ON+LESION relf question -> generic heuristic (route retired to GENERIC)",
        out == ["country", "chelsea"] and out != RELF_CAND, f"out={out}")

    n_fail = sum(1 for c in checks if not c["ok"])
    all_pass = n_fail == 0

    # Verdict: carry every precondition into the artifact so a partial pass cannot read as GO (tools.verdict.Verdict).
    v = Verdict("question-route-selection WTA production wire-in (focused verify, default-OFF)")
    for c in checks:
        v.require(c["name"], c["ok"], expect=True)
    v.knob("BRAIN_SPIKING_QROUTE (default OFF) / BRAIN_SPIKING_QROUTE_LESION",
           requested=("off", "off"), applied=("off", "off"))
    v.disabled("homeostasis/stdp/hebbian/short-term-plasticity/structural-plasticity",
               why="frozen cross-inhibition weights; a pure feedforward-driven competitive read, no learning")
    vd = v.decide(go=all_pass, verbose=False)

    summary = {
        "runner": "_rank14_qroute_wirein_verify", "backend": os.environ.get("SIM_BACKEND"), "seeds": list(SEEDS),
        "all_pass": all_pass, "n_checks": len(checks), "n_fail": n_fail, "checks": checks,
        "attribution_intact_vs_lesion": (None if attribution is None else float(attribution)),
        "intact_exc_rate": intact_exc_rate, "lesion_exc_rate": lesion_exc_rate,
        "verdict": ("WIRED-DEFAULT-OFF (focused verify GO)" if all_pass else "FAIL"),
        "status": vd["status"], "preconditions": vd["preconditions"],
        "disabled_processes": vd["disabled_processes"], "undefined_reasons": vd["undefined_reasons"],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n{'ALL CHECKS PASS' if all_pass else f'{n_fail} FAILURE(S)'}  verdict={vd['status']}  [saved] {args.json}",
          flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
