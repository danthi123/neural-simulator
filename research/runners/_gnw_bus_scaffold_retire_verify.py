"""SCAFFOLD-RETIREMENT VERIFY for the GNW N-organ ignition bus.

The 2026-08-13 FLIP made the substrate AUTHOR the organ-combination on `webapp/server.py::brain_chat`, but its #1
declared residual was `scaffold_retired: NO` — the host `gate()` combination CODE still ran, its `if recalled == p`
result COMPUTED-then-OVERRIDDEN. This runner earns the retirement: `install_bus_gate` now wraps `chat.gate` with
`gate_via_bus`, which runs ONLY the extraction + side-effect phase (`chat.gate_extract`) and lets the substrate
consensus-ignition commit/veto the COVERED class WITHOUT EVER COMPUTING the host `if recalled == p` (no
`_substrate_recall`, no `_gate_router_combine` on a routable factual recall).

Earns it on the SAME broad real-query panel the flip verify used, PLUS a call-count RETIREMENT PROOF:
  (0) RETIREMENT PROOF (the new gate): on a covered STORED recall driven through the installed bus gate, NEITHER
      host-combination method (`_substrate_recall`, `_gate_router_combine`) is called, and the per-turn audit reads
      authored_by='bus' + host_combination_computed=False. On a self/identity turn the HOST router IS called (the
      out-of-scope residual is honestly KEPT). This is the difference between OVERRIDDEN (flip) and RETIRED (here).
  (1) BYTE-IDENTICAL: `gate_via_bus` == the original host gate() on EVERY covered panel query (retiring the redundant
      host combination must not change any answer — it was already overridden). Per class.
  (2) NO MOAT REGRESSION: unstored/inconsistent still abstain on both arms.
  (3) LESION-LOAD-BEARING: a bus lesion collapses the covered answer to abstain; the forward-recall reflex survives.
  (4) ESCAPE byte-identical: BRAIN_GNW_BUS_HOST=1 reverts to the host gate() (byte-identical through the real handler).

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_bus_scaffold_retire_verify
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path)

from tools.lab import attributable_to, void_if   # noqa: E402
# reuse-by-import the flip verify's panel + fixtures so store + retirement are tested on the SAME queries.
from research.runners._gnw_bus_default_flip_verify import (   # noqa: E402
    STORED, SELF, UNSTORED, INCONSISTENT, OPEN_ENDED, ACQUIRE_SEQ, ANAPHORA_SEQ,
    _svo_eq, _build, _stateful_seq, _handler_escape_byte_identical,
)


def _stateless_panel(chat, gbs):
    """(1)+(2): per-query host-vs-bus byte-identical on the stateless classes, driving the RETIREMENT combiner
    (`gate_via_bus` — never computes the covered-class host verdict) against the ORIGINAL host gate()."""
    rows = []
    for cls, items in (("stored", STORED), ("self", [(q, None) for q in SELF]),
                       ("unstored", [(q, None) for q in UNSTORED]),
                       ("inconsistent", [(q, None) for q in INCONSISTENT]),
                       ("open_ended", [(q, None) for q in OPEN_ENDED])):
        for q, want in items:
            host_svo = chat._gnw_orig_gate(q)                        # the HOST decision (extraction + recall + combine)
            bus_svo, info = gbs.gate_via_bus(chat, q, lesion=False)  # the RETIREMENT combiner (no covered host verdict)
            identical = _svo_eq(bus_svo, host_svo)
            rows.append({
                "cls": cls, "q": q, "want": want,
                "host_svo": (list(host_svo) if host_svo is not None else None),
                "bus_svo": (list(bus_svo) if bus_svo is not None else None),
                "authored_by": info.get("authored_by"),
                "host_combination_computed": info.get("host_combination_computed"),
                "organ_reads": info.get("organ_reads"),
                "byte_identical": identical,
                "moat_ok": (bool(host_svo is None and bus_svo is None) if cls in ("unstored", "inconsistent") else None),
                "host_correct": (host_svo == want if want is not None else None),
            })
    return rows


def _stateful_panel():
    """(1) byte-identical on the STATEFUL classes: replay each sequence through a host brain and a bus (retirement)
    brain (both FRESH) and compare every gate decision turn-by-turn."""
    rows = []
    for name, seq in (("acquisition", ACQUIRE_SEQ), ("anaphora", ANAPHORA_SEQ)):
        host = _stateful_seq(seq, escape_to_host=True)
        bus = _stateful_seq(seq, escape_to_host=False)              # chat.gate == the installed gate_via_bus wrapper
        for i, (utt, want) in enumerate(seq):
            rows.append({"cls": name, "q": utt, "want": want, "host_svo": host[i], "bus_svo": bus[i],
                         "byte_identical": _svo_eq(host[i], bus[i])})
    return rows


def _retirement_proof(chat, gbs):
    """(0) THE RETIREMENT: instrument the two host-combination methods with call counters and drive a COVERED stored
    recall + a self/identity turn through the INSTALLED bus gate (`chat.gate`). Covered -> NEITHER method is called +
    the audit reads authored_by='bus'/host_combination_computed=False (RETIRED, not overridden). Self -> the host
    router IS called (the out-of-scope residual is honestly kept)."""
    counts = {"_substrate_recall": 0, "_gate_router_combine": 0}
    orig_sr = chat._substrate_recall
    orig_rc = chat._gate_router_combine

    def _sr(q, *a, **k):
        counts["_substrate_recall"] += 1
        return orig_sr(q, *a, **k)

    def _rc(q, *a, **k):
        counts["_gate_router_combine"] += 1
        return orig_rc(q, *a, **k)

    chat._substrate_recall = _sr
    chat._gate_router_combine = _rc
    try:
        counts["_substrate_recall"] = 0
        counts["_gate_router_combine"] = 0
        covered_svo = chat.gate("what does dog chase?")             # COVERED: the bus authors -> no host combination
        covered_info = dict(getattr(chat, "_last_gnw_bus", {}) or {})
        covered_counts = dict(counts)
        counts["_substrate_recall"] = 0
        counts["_gate_router_combine"] = 0
        self_svo = chat.gate("what are you")                       # OUT OF SCOPE: the host router authors (kept)
        self_info = dict(getattr(chat, "_last_gnw_bus", {}) or {})
        self_counts = dict(counts)
    finally:
        chat._substrate_recall = orig_sr
        chat._gate_router_combine = orig_rc

    covered_retired = bool(
        covered_counts["_substrate_recall"] == 0 and covered_counts["_gate_router_combine"] == 0
        and covered_info.get("authored_by") == "bus"
        and covered_info.get("host_combination_computed") is False
        and covered_svo == ["dog", "chase", "cat"])
    out_of_scope_host_kept = bool(
        self_counts["_gate_router_combine"] >= 1
        and self_info.get("authored_by") == "host_router"
        and self_svo == ["brain", "use", "spikes"])
    return {
        "covered_counts": covered_counts, "covered_authored_by": covered_info.get("authored_by"),
        "covered_host_combination_computed": covered_info.get("host_combination_computed"),
        "covered_svo": (list(covered_svo) if covered_svo is not None else None),
        "self_counts": self_counts, "self_authored_by": self_info.get("authored_by"),
        "self_svo": (list(self_svo) if self_svo is not None else None),
        "covered_combination_retired": covered_retired, "out_of_scope_host_kept": out_of_scope_host_kept,
    }


def _lesion(chat, gbs):
    """(3): the bus lesion COLLAPSES the covered answer to abstain, the reflex survives — through the RETIREMENT
    combiner (`gate_via_bus`). Attribution 1.0 => the whole committed decision is owed to the workspace ignition."""
    q, a, v, p = "what does dog chase?", "dog", "chase", "cat"
    intact, _ = gbs.gate_via_bus(chat, q, lesion=False)
    lesioned, _ = gbs.gate_via_bus(chat, q, lesion=True)
    reflex = chat.inner.composer.query_patient(a, v)               # the workspace-INDEPENDENT recall reflex
    intact_answers = _svo_eq(intact, [a, v, p])
    lesion_collapses = (lesioned is None)
    reflex_survives = (reflex == p)
    attribution = attributable_to("covered answer owed to the workspace ignition (host combination retired)",
                                  1.0 if intact_answers else 0.0, 0.0 if lesion_collapses else 1.0)
    return {"q": q, "intact": (list(intact) if intact is not None else None),
            "lesioned": (list(lesioned) if lesioned is not None else None), "reflex": reflex,
            "intact_answers": intact_answers, "lesion_collapses": lesion_collapses,
            "reflex_survives": reflex_survives, "attribution_to_ignition": attribution}


def main():
    chat, gbs = _build()
    proof = _retirement_proof(chat, gbs)
    stateless = _stateless_panel(chat, gbs)
    stateful = _stateful_panel()
    lesion = _lesion(chat, gbs)
    handler = _handler_escape_byte_identical()

    all_rows = stateless + stateful
    per_class = {}
    for r in all_rows:
        c = r["cls"]
        d = per_class.setdefault(c, {"n": 0, "identical": 0})
        d["n"] += 1
        d["identical"] += int(bool(r["byte_identical"]))
    n_identical = sum(int(bool(r["byte_identical"])) for r in all_rows)
    n_total = len(all_rows)
    panel_void = void_if(not all_rows, "the broad panel produced ZERO rows — the retirement verdict is UNDEFINED")
    byte_identical_frac = (n_identical / n_total) if n_total else None
    moat_rows = [r for r in stateless if r["moat_ok"] is not None]
    moat_ok = bool(moat_rows) and all(r["moat_ok"] for r in moat_rows)
    stored_rows = [r for r in stateless if r["cls"] == "stored"]
    host_correct = bool(stored_rows) and all(r["host_correct"] for r in stored_rows)
    lesion_ok = bool(lesion["intact_answers"] and lesion["lesion_collapses"] and lesion["reflex_survives"])
    retired = bool(proof["covered_combination_retired"] and proof["out_of_scope_host_kept"])

    go = bool(not panel_void and n_identical == n_total and moat_ok and host_correct and lesion_ok
              and handler["ok"] and retired)

    from tools.verdict import Verdict
    v = Verdict("gnw-bus scaffold-retirement (covered-class host combination never computed)")
    v.require("COVERED-class host combination RETIRED (no _substrate_recall / _gate_router_combine; authored_by=bus)",
              proof["covered_combination_retired"], expect=True,
              note=f"counts={proof['covered_counts']} authored_by={proof['covered_authored_by']} "
                   f"host_combination_computed={proof['covered_host_combination_computed']}")
    v.require("OUT-OF-SCOPE self/identity still HOST-authored (residual honestly kept)",
              proof["out_of_scope_host_kept"], expect=True,
              note=f"self_counts={proof['self_counts']} authored_by={proof['self_authored_by']}")
    v.require("byte-identical bus-vs-host on EVERY covered panel query", (n_identical == n_total), expect=True,
              note=f"{n_identical}/{n_total} — retiring the redundant host combination changes no answer")
    v.require("no moat regression (unstored/inconsistent still abstain, host+bus)", moat_ok, expect=True)
    v.require("stored host decisions are correct (byte-identical => bus reproduces a CORRECT decision)", host_correct,
              expect=True)
    v.require("lesion collapses the covered ANSWER to abstain (ignition load-bearing)", lesion["lesion_collapses"],
              expect=True)
    v.require("forward-recall reflex survives the lesion (dissociation)", lesion["reflex_survives"], expect=True)
    v.require("intact bus answers the stored fact", lesion["intact_answers"], expect=True)
    v.require("ESCAPE flag BRAIN_GNW_BUS_HOST=1 byte-identical to bus default (real handler)", handler["ok"],
              expect=True, note="the owner can revert instantly to today's production")
    v.control("lesion dissociation (covered answer needs the workspace ignition)",
              treatment=(1.0 if lesion["intact_answers"] else 0.0),
              control=(0.0 if lesion["lesion_collapses"] else 1.0), min_separation=0.0)
    v.disabled("heavy Gate-B organs (affect/worldmodel/... = 0) in the escape byte-identical handler check",
               why="disabled ONLY for speed; they run identically on both flag arms, so the comparison is unaffected")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 108, flush=True)
    print("  GNW BUS SCAFFOLD-RETIREMENT — the substrate authors the covered class WITHOUT computing the host combination", flush=True)
    print("=" * 108, flush=True)
    print(f"  (0) RETIREMENT PROOF (covered STORED recall through the installed bus gate):", flush=True)
    print(f"        host-combination call counts = {proof['covered_counts']}  (both must be 0)", flush=True)
    print(f"        audit: authored_by={proof['covered_authored_by']!r}  "
          f"host_combination_computed={proof['covered_host_combination_computed']}  svo={proof['covered_svo']}", flush=True)
    print(f"        out-of-scope self/identity: counts={proof['self_counts']} authored_by={proof['self_authored_by']!r} "
          f"svo={proof['self_svo']}  (host router KEPT)", flush=True)
    print(f"        covered_combination_retired={proof['covered_combination_retired']}  "
          f"out_of_scope_host_kept={proof['out_of_scope_host_kept']}", flush=True)
    print(f"  (1) BYTE-IDENTICAL bus-vs-host: {n_identical}/{n_total}", flush=True)
    for c, d in per_class.items():
        print(f"        {c:14s} {d['identical']}/{d['n']}", flush=True)
    for r in all_rows:
        mark = "OK " if r["byte_identical"] else "DIVERGE"
        extra = (f" by={r.get('authored_by')}" if r.get('authored_by') else "")
        print(f"      [{mark}] {r['cls']:12s} {r['q']:28s} host={r['host_svo']} bus={r['bus_svo']}{extra}", flush=True)
    print(f"  (2) MOAT (unstored/inconsistent abstain, host+bus): {moat_ok}", flush=True)
    print(f"  (3) LESION: intact_answers={lesion['intact_answers']} lesion_collapses={lesion['lesion_collapses']} "
          f"reflex_survives={lesion['reflex_survives']} (intact={lesion['intact']} lesioned={lesion['lesioned']} "
          f"reflex={lesion['reflex']!r})", flush=True)
    print(f"  (4) ESCAPE byte-identical (real handler): {handler['ok']}", flush=True)
    for r in handler["rows"]:
        print(f"      HANDLER {r['q']:28s} identical={r['identical']} no_key={r['no_gnw_bus_key']} "
              f"answer={r['answer']!r} recalled={r['recalled_svo']}", flush=True)
    verdict = "GO — RETIRED (scoped to the covered class)" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 108, flush=True)

    out = {"runner": "_gnw_bus_scaffold_retire_verify", "go": go, "status": decided["status"],
           "retirement_proof": proof,
           "n_identical": n_identical, "n_total": n_total, "byte_identical_frac": byte_identical_frac,
           "per_class": per_class, "moat_ok": moat_ok, "host_correct": host_correct,
           "lesion": lesion, "handler_escape_byte_identical": handler,
           "stateless": stateless, "stateful": stateful,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_gnw_bus_shadow/scaffold_retire_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
