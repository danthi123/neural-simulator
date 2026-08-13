"""BROAD-PANEL FLIP VERIFY for promoting the GNW N-organ ignition bus from SHADOW to the DEFAULT organ-combination
on `webapp/server.py::brain_chat`.

The shadow (`webapp/gnw_bus_shadow.py`, commit 2cce00cca) proved 9/9 host-vs-bus agreement on a SMALL panel. Before
flipping the DEFAULT (so the SUBSTRATE ignition — consensus + WTA — AUTHORS the organ-combination instead of the host
`if recalled == p`), this runner EARNS the flip on a BROAD real-query panel through the REAL production ChatBrain
(numpy-CPU, rf recall), covering many query CLASSES:

  * STORED          — the host answers; the bus must IGNITE the same patient (multi-organ corroboration).
  * SELF/IDENTITY   — the host answers via the router; the bus routes the underlying (agent, action) binding.
  * UNSTORED        — unknown agent/action; the host abstains; the bus must abstain (the moat).
  * INCONSISTENT    — a stored agent under a WRONG action; the host abstains; the bus must abstain (the moat).
  * ACQUISITION     — teach an SVO then recall it (stateful); the bus re-authors the just-stored fact.
  * ANAPHORA        — a multi-turn "what does it eat?" (stateful); the bus re-authors the resolved recall.
  * OPEN-ENDED      — a generation prompt (a `HypothesisSVO` guess); the bus does NOT cover it -> falls back to host.

Verifies (the flip gate, per `docs/plans/2026-08-13-gnw-norgan-bus-production-wiring.md` §4):
  (1) BYTE-IDENTICAL: the bus-authored gate decision == the host gate() decision on EVERY covered panel query
      (the substrate reproduces the host decision -> flipping changes the MECHANISM, not the behaviour). Per-class.
  (2) NO MOAT REGRESSION: every UNSTORED/INCONSISTENT query still ABSTAINS (0-confab preserved by ignition).
  (3) LESION-LOAD-BEARING: a bus lesion (workspace self-recurrence 0) COLLAPSES the combined answer to abstain,
      while the forward-recall reflex (direct query_patient) survives (the substrate ignition does the work).
  (4) ESCAPE FLAG: through the REAL brain_chat handler, BRAIN_GNW_BUS_HOST=1 reverts to the host gate() path and is
      byte-identical to the bus-default response on the covered class (so the owner can revert instantly).

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_bus_default_flip_verify
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
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")   # the numpy fast-path recall (a real production path; ~s not ~180s)

from tools.lab import attributable_to, void_if   # noqa: E402


# ── the broad panel (query, class, want_or_None) ────────────────────────────────────────────────────────────
STORED = [
    ("what does dog chase?", ["dog", "chase", "cat"]),
    ("what does cat eat?", ["cat", "eat", "fish"]),
    ("what does brain use?", ["brain", "use", "spikes"]),
    ("what does brain learn?", ["brain", "learn", "words"]),
    ("what does brain store?", ["brain", "store", "memory"]),
]
SELF = ["what are you", "what do you use", "how do you learn"]         # router/self path (host decides; bus matches)
UNSTORED = ["what does fish fly?", "what does ball roll?", "what does bird sing?", "what does dragon breathe?"]
INCONSISTENT = ["what does dog eat?", "what does cat chase?", "what does brain chase?", "what does dog use?"]
OPEN_ENDED = ["what might a dog do", "tell me something new about brain"]
ACQUIRE_SEQ = [("sky hold cloud", None), ("what does sky hold?", ["sky", "hold", "cloud"])]   # teach then recall
ANAPHORA_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it eat?", ["cat", "eat", "fish"])]


def _svo_eq(x, y) -> bool:
    """Byte-identical SVO equality: None==None; else the [a,v,p] lists match (a HypothesisSVO compares as a list)."""
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


def _build():
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    chat, _src = _build_chat_brain("tiny-demo", "stub")   # the REAL production build (rf recall via BRAIN_COMPOSER_KIND)
    gbs.install_bus_gate(chat)                            # promote gate() -> the substrate authors the combination
    return chat, gbs


def _stateless_panel(chat, gbs):
    """(1)+(2): per-query host-vs-bus byte-identical on the stateless classes. host = the ORIGINAL gate() (run ONCE,
    side effects included); bus = the substrate re-authoring of that same decision (read-only)."""
    rows = []
    for cls, items in (("stored", STORED), ("self", [(q, None) for q in SELF]),
                       ("unstored", [(q, None) for q in UNSTORED]),
                       ("inconsistent", [(q, None) for q in INCONSISTENT]),
                       ("open_ended", [(q, None) for q in OPEN_ENDED])):
        for q, want in items:
            host_svo = chat._gnw_orig_gate(q)                     # the HOST decision (extraction + recall + combine)
            bus_svo, info = gbs.bus_authored_svo(chat, q, host_svo, lesion=False)
            identical = _svo_eq(bus_svo, host_svo)
            host_abstained = host_svo is None
            bus_abstained = bus_svo is None
            rows.append({
                "cls": cls, "q": q, "want": want,
                "host_svo": (list(host_svo) if host_svo is not None else None),
                "bus_svo": (list(bus_svo) if bus_svo is not None else None),
                "organ_reads": info.get("organ_reads"), "n_ignited": info.get("n_ignited"),
                "routable": info.get("routable"), "reason": info.get("reason"),
                "byte_identical": identical,
                # moat: on the abstain classes, BOTH must withhold
                "moat_ok": (bool(host_abstained and bus_abstained) if cls in ("unstored", "inconsistent") else None),
                # stored sanity: the host answered what we expect (so the bus reproduces a CORRECT decision)
                "host_correct": (host_svo == want if want is not None else None),
            })
    return rows


def _stateful_seq(seq, escape_to_host):
    """Drive a stateful SEQUENCE (teach/anaphora) through a FRESH production brain and return the per-turn gate
    decisions. escape_to_host=True -> the ORIGINAL host gate() (BRAIN_GNW_BUS_HOST path); False -> the bus-authored
    gate. Both run the SAME side effects (the wrapper calls the original gate internally), so only the COMBINATION
    verdict differs."""
    chat, gbs = _build()
    out = []
    for utt, _want in seq:
        if escape_to_host:
            svo = chat._gnw_orig_gate(utt)
        else:
            svo = chat.gate(utt)                             # the installed bus wrapper (default combination)
        out.append(list(svo) if svo is not None else None)
    return out


def _stateful_panel():
    """(1) byte-identical on the STATEFUL classes: replay each sequence through a host brain and a bus brain, compare
    the final (and every) gate decision turn-by-turn."""
    rows = []
    for name, seq in (("acquisition", ACQUIRE_SEQ), ("anaphora", ANAPHORA_SEQ)):
        host = _stateful_seq(seq, escape_to_host=True)
        bus = _stateful_seq(seq, escape_to_host=False)
        for i, (utt, want) in enumerate(seq):
            rows.append({"cls": name, "q": utt, "want": want,
                         "host_svo": host[i], "bus_svo": bus[i],
                         "byte_identical": _svo_eq(host[i], bus[i])})
    return rows


def _lesion(chat, gbs):
    """(3): the bus lesion COLLAPSES the combined answer to abstain, the reflex survives — on a stored fact the
    intact bus answers. Attribution 1.0 => the whole committed decision is owed to the workspace ignition."""
    q, a, v, p = "what does dog chase?", "dog", "chase", "cat"
    host_svo = chat._gnw_orig_gate(q)
    intact, _ = gbs.bus_authored_svo(chat, q, host_svo, lesion=False)
    lesioned, _ = gbs.bus_authored_svo(chat, q, host_svo, lesion=True)
    reflex = chat.inner.composer.query_patient(a, v)          # the workspace-INDEPENDENT recall reflex
    intact_answers = _svo_eq(intact, [a, v, p])
    lesion_collapses = (lesioned is None)                    # the ANSWER collapses to abstain under lesion
    reflex_survives = (reflex == p)
    attribution = attributable_to("combined answer owed to the workspace ignition (not a residual host read)",
                                  1.0 if intact_answers else 0.0, 0.0 if lesion_collapses else 1.0)
    return {"q": q, "intact": (list(intact) if intact is not None else None),
            "lesioned": (list(lesioned) if lesioned is not None else None), "reflex": reflex,
            "intact_answers": intact_answers, "lesion_collapses": lesion_collapses,
            "reflex_survives": reflex_survives, "attribution_to_ignition": attribution}


def _handler_escape_byte_identical():
    """(4): through the REAL brain_chat handler (single-fact path), BRAIN_GNW_BUS_HOST=1 (escape -> host, today's
    production) must be byte-identical to the bus default on the covered class. Heavy Gate-B organs disabled for
    speed (they run identically on both arms)."""
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_RICH", "BRAIN_GNW_BUS"):
        os.environ[k] = "0"
    os.environ.pop("BRAIN_GNW_BUS", None)                    # observability block OFF -> response has no gnw_bus key
    from webapp.server import brain_chat, BrainChatRequest as Req

    # stateless recall/abstain/self queries -> reuse ONE session per arm (2 warm brain builds total, not 2 per query).
    panel = ["what does dog chase?", "what does cat eat?", "what does brain use?",
             "what does fish fly?", "what does dog eat?", "what are you"]
    rows, ok = [], True
    for i, msg in enumerate(panel):
        os.environ["BRAIN_GNW_BUS_HOST"] = "1"              # ESCAPE -> the original host gate() (pre-flip production)
        esc = json.loads(brain_chat(Req(session="esc", message=msg, brain="tiny-demo", renderer="stub",
                                        rich=False)).body.decode("utf-8"))
        os.environ.pop("BRAIN_GNW_BUS_HOST", None)          # DEFAULT -> the substrate ignition bus authors
        bus = json.loads(brain_chat(Req(session="bus", message=msg, brain="tiny-demo", renderer="stub",
                                        rich=False)).body.decode("utf-8"))
        identical = (esc == bus)
        no_key = ("gnw_bus" not in esc) and ("gnw_bus" not in bus)
        ok = ok and identical and no_key
        rows.append({"q": msg, "identical": identical, "no_gnw_bus_key": no_key,
                     "answer": bus.get("answer"), "recalled_svo": bus.get("recalled_svo"),
                     "abstained": bus.get("abstained")})
    return {"ok": ok, "rows": rows}


def main():
    chat, gbs = _build()
    stateless = _stateless_panel(chat, gbs)
    stateful = _stateful_panel()
    lesion = _lesion(chat, gbs)
    handler = _handler_escape_byte_identical()

    all_rows = stateless + stateful
    # (1) BYTE-IDENTICAL across the whole panel + per class
    per_class = {}
    for r in all_rows:
        c = r["cls"]
        d = per_class.setdefault(c, {"n": 0, "identical": 0})
        d["n"] += 1
        d["identical"] += int(bool(r["byte_identical"]))
    n_identical = sum(int(bool(r["byte_identical"])) for r in all_rows)
    n_total = len(all_rows)
    panel_void = void_if(not all_rows, "the broad panel produced ZERO rows — the flip verdict is UNDEFINED, not a GO")
    byte_identical_frac = (n_identical / n_total) if n_total else None
    # (2) MOAT: every abstain-class query withholds on BOTH host + bus
    moat_rows = [r for r in stateless if r["moat_ok"] is not None]
    moat_ok = bool(moat_rows) and all(r["moat_ok"] for r in moat_rows)
    # stored sanity: the host answered correctly (so byte-identical means the bus reproduces a CORRECT decision)
    stored_rows = [r for r in stateless if r["cls"] == "stored"]
    host_correct = bool(stored_rows) and all(r["host_correct"] for r in stored_rows)

    les = lesion
    lesion_ok = bool(les["intact_answers"] and les["lesion_collapses"] and les["reflex_survives"])

    go = bool(not panel_void and n_identical == n_total and moat_ok and host_correct and lesion_ok and handler["ok"])

    # EARN the verdict — the preconditions travel with the result.
    from tools.verdict import Verdict
    v = Verdict("gnw-bus DEFAULT combination flip (broad panel)")
    v.require("byte-identical bus-vs-host on EVERY covered panel query", (n_identical == n_total), expect=True,
              note=f"{n_identical}/{n_total} — the substrate reproduces the host gate() decision on every class")
    v.require("no moat regression (unstored/inconsistent still abstain, host+bus)", moat_ok, expect=True)
    v.require("stored host decisions are correct (byte-identical => bus reproduces a CORRECT decision)", host_correct,
              expect=True)
    v.require("lesion collapses the combined ANSWER to abstain (ignition load-bearing)", les["lesion_collapses"],
              expect=True)
    v.require("forward-recall reflex survives the lesion (dissociation)", les["reflex_survives"], expect=True)
    v.require("intact bus answers the stored fact", les["intact_answers"], expect=True)
    v.require("ESCAPE flag BRAIN_GNW_BUS_HOST=1 byte-identical to bus default (real handler)", handler["ok"],
              expect=True, note="the owner can revert instantly to today's production")
    v.control("lesion dissociation (combined answer needs the workspace ignition)",
              treatment=(1.0 if les["intact_answers"] else 0.0),
              control=(0.0 if les["lesion_collapses"] else 1.0), min_separation=0.0)
    v.disabled("heavy Gate-B organs (affect/worldmodel/... = 0) in the escape byte-identical handler check",
               why="disabled ONLY for speed; they run identically on both flag arms, so the comparison is unaffected")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 108, flush=True)
    print("  GNW BUS DEFAULT-FLIP — BROAD-PANEL VERIFY (real production ChatBrain, rf recall; the substrate authors the combination)", flush=True)
    print("=" * 108, flush=True)
    print(f"  (1) BYTE-IDENTICAL bus-vs-host: {n_identical}/{n_total}", flush=True)
    for c, d in per_class.items():
        print(f"        {c:14s} {d['identical']}/{d['n']}", flush=True)
    for r in all_rows:
        mark = "OK " if r["byte_identical"] else "DIVERGE"
        print(f"      [{mark}] {r['cls']:12s} {r['q']:28s} host={r['host_svo']} bus={r['bus_svo']}"
              + (f" reads={r.get('organ_reads')}" if r.get('organ_reads') else ""), flush=True)
    print(f"  (2) MOAT (unstored/inconsistent abstain, host+bus): {moat_ok}", flush=True)
    print(f"  (3) LESION: intact_answers={les['intact_answers']} lesion_collapses={les['lesion_collapses']} "
          f"reflex_survives={les['reflex_survives']} (intact={les['intact']} lesioned={les['lesioned']} reflex={les['reflex']!r})",
          flush=True)
    print(f"  (4) ESCAPE byte-identical (real handler): {handler['ok']}", flush=True)
    for r in handler["rows"]:
        print(f"      HANDLER {r['q']:28s} identical={r['identical']} no_key={r['no_gnw_bus_key']} "
              f"answer={r['answer']!r} recalled={r['recalled_svo']}", flush=True)
    verdict = "GO — FLIP" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 108, flush=True)

    out = {"runner": "_gnw_bus_default_flip_verify", "go": go, "status": decided["status"],
           "n_identical": n_identical, "n_total": n_total, "byte_identical_frac": byte_identical_frac,
           "per_class": per_class, "moat_ok": moat_ok, "host_correct": host_correct,
           "lesion": les, "handler_escape_byte_identical": handler,
           "stateless": stateless, "stateful": stateful,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_gnw_bus_shadow/default_flip_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
