"""PRODUCTION VERIFY for the GNW N-organ ignition-bus SHADOW wired into `webapp/server.py::brain_chat`.

Asserts the four hard requirements of the safe first wiring, on the REAL production tiny-demo brain, numpy-CPU:

  (A) AGREEMENT: the substrate's committed decision (the SAME real organ reads gate() combines — spiking recall +
      VERIFY re-check + reverse-binding VERIFY — routed through the de-risked spiking ignition bus) AGREES with the
      host `gate()` decision on a panel of stored + abstain turns. This exercises the EXACT shadow code path the
      handler runs (`gate_svo = chat.gate(msg); shadow_report(chat, msg, gate_svo)`), on the production ChatBrain
      built by `webapp.server._build_chat_brain`.
  (B) LESION-LOAD-BEARING: the bus built with the workspace assembly self-recurrence zeroed COLLAPSES (abstains on a
      stored query the intact bus answers), WHILE the forward-recall reflex (direct query_patient, never routed
      through the workspace) still returns the patient — the dissociation proving the SUBSTRATE does the combining.
  (C) MOAT-SAFE: on the abstain turns the host withholds AND the bus withholds (the bus only re-derives; never
      invents a fact).
  (D) BYTE-IDENTICAL-WHEN-OFF (real handler): a turn through the REAL `brain_chat` handler with the flag OFF carries
      NO `gnw_bus` key, and with the flag ON every OTHER field is byte-identical to the off response (the host still
      authors the answer; the shadow is read-only). Heavy Gate-B organs are disabled here ONLY for speed — they run
      identically on both sides, so the comparison is unaffected.

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._gnw_bus_shadow_production_verify
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

from tools.lab import attributable_to   # noqa: E402  (attribution of the lesion dissociation; see (B) below)

# STORED turns the host answers + ABSTAIN turns the host withholds (moat). The bus must reproduce BOTH.
STORED = [
    ("what does dog chase?", ["dog", "chase", "cat"]),
    ("what does cat eat?", ["cat", "eat", "fish"]),
    ("what does brain use?", ["brain", "use", "spikes"]),
    ("what does brain learn?", ["brain", "learn", "words"]),
    ("what does brain store?", ["brain", "store", "memory"]),
]
ABSTAIN = [
    "what does dog eat?",       # dog has a CHASE fact, not EAT -> the substrate has no binding -> moat abstain
    "what does cat chase?",     # cat has an EAT fact, not CHASE -> moat abstain
    "what does fish fly?",      # unstored agent/action -> moat abstain
    "what does ball roll?",     # unknown -> moat abstain
]


def _direct_agreement_lesion():
    """(A)+(B)+(C): build the production ChatBrain and drive the EXACT shadow code path the handler runs."""
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs

    chat, _src = _build_chat_brain("tiny-demo", "stub")   # the REAL production build (rf recall via BRAIN_COMPOSER_KIND)
    composer = chat.inner.composer
    all_concepts = sorted(chat.agents_set | chat.actions_set | chat.patients_set)

    agree_n = agree_tot = 0
    stored_rows, abstain_rows = [], []
    for msg, want in STORED:
        gate_svo = chat.gate(msg)                          # the HOST decision
        blk = gbs.shadow_report(chat, msg, gate_svo)       # the SUBSTRATE decision (same code path as the handler)
        agrees = bool(blk.get("agrees"))
        agree_tot += 1
        agree_n += int(agrees)
        stored_rows.append({"q": msg, "host_recalled_svo": gate_svo, "want": want,
                            "bus_committed": blk.get("committed"), "organ_reads": blk.get("organ_reads"),
                            "agrees": agrees})
    for msg in ABSTAIN:
        gate_svo = chat.gate(msg)
        blk = gbs.shadow_report(chat, msg, gate_svo)
        host_abstained = gate_svo is None
        bus_abstained = (blk.get("bus_decision") == "abstain")
        agrees = bool(blk.get("agrees"))
        agree_tot += 1
        agree_n += int(agrees)
        abstain_rows.append({"q": msg, "host_abstained": host_abstained, "bus_abstained": bus_abstained,
                             "organ_reads": blk.get("organ_reads"), "agrees": agrees})

    # (B) LESION dissociation on a stored fact the intact bus answers.
    lam, lav, lap = "cat", "eat", "fish"
    intact = gbs.bus_combine(composer, lam, lav, all_concepts, lesion=False)
    lesioned = gbs.bus_combine(composer, lam, lav, all_concepts, lesion=True)
    reflex = composer.query_patient(lam, lav)              # the workspace-INDEPENDENT recall reflex
    intact_answers = (intact.get("committed") == lap)
    lesion_collapses = (lesioned.get("committed") is None)
    # ATTRIBUTION (tools.lab): what fraction of the bus's committed decision is OWED to the intact workspace
    # ignition (vs. present in the lesion control)? treatment = the intact bus answers correctly; control = the
    # lesioned bus answers. 1.0 => the whole decision is the substrate's ignition, not a residual host read.
    treat = 1.0 if intact_answers else 0.0
    ctrl = 0.0 if lesion_collapses else 1.0
    lesion_attribution = attributable_to("bus decision owed to the workspace ignition (not a residual host read)",
                                          treat, ctrl)
    lesion = {"fact": [lam, lav, lap], "intact_committed": intact.get("committed"),
              "lesioned_committed": lesioned.get("committed"), "reflex": reflex,
              "intact_answers": intact_answers,
              "lesion_collapses": lesion_collapses,
              "reflex_survives": (reflex == lap),
              "attribution_to_ignition": lesion_attribution}
    return {"agree_n": agree_n, "agree_tot": agree_tot, "stored": stored_rows, "abstain": abstain_rows,
            "lesion": lesion}


def _handler_byte_identical():
    """(D): a turn through the REAL `brain_chat` handler — flag OFF carries no gnw_bus key; flag ON is byte-identical
    on every other field. Heavy Gate-B organs disabled for speed (they run identically on both sides)."""
    for k in ("BRAIN_AFFECT", "BRAIN_WORLDMODEL", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_MULTIREF",
              "BRAIN_NONCONTRADICTION_GATE", "BRAIN_RECONSOLIDATION", "BRAIN_EPISODIC_STORE", "BRAIN_CURIOSITY",
              "BRAIN_RICH"):
        os.environ[k] = "0"
    from webapp.server import brain_chat, BrainChatRequest as Req

    rows, ok = [], True
    for msg in ("what does dog chase?", "what does fish fly?"):
        os.environ["BRAIN_GNW_BUS"] = "1"
        on = json.loads(brain_chat(Req(session="bi", message=msg, brain="tiny-demo", renderer="stub",
                                       rich=False)).body.decode("utf-8"))
        os.environ.pop("BRAIN_GNW_BUS", None)
        off = json.loads(brain_chat(Req(session="bi", message=msg, brain="tiny-demo", renderer="stub",
                                        rich=False)).body.decode("utf-8"))
        no_key = "gnw_bus" not in off
        has_key_on = "gnw_bus" in on
        identical = ({k: v for k, v in on.items() if k != "gnw_bus"} == off)
        ok = ok and no_key and has_key_on and identical
        rows.append({"q": msg, "off_has_no_bus_key": no_key, "on_has_bus_key": has_key_on,
                     "host_fields_identical": identical, "on_gnw_bus": on.get("gnw_bus")})
    return {"ok": ok, "rows": rows}


def main():
    direct = _direct_agreement_lesion()
    handler = _handler_byte_identical()

    agree_frac = direct["agree_n"] / direct["agree_tot"] if direct["agree_tot"] else 0.0
    les = direct["lesion"]
    go = bool(agree_frac >= 0.999 and handler["ok"] and les["intact_answers"]
              and les["lesion_collapses"] and les["reflex_survives"])

    # EARN the verdict — the preconditions travel with the result (tools.verdict.Verdict -> a `preconditions` block).
    from tools.verdict import Verdict
    v = Verdict("gnw-bus-shadow production wiring")
    v.require("bus-vs-host decision agreement == 1.0", agree_frac, expect=lambda x: x >= 0.999,
              note="the substrate reproduces the host gate() combination on every panel turn (stored + abstain)")
    v.require("byte-identical-when-off (real brain_chat handler)", handler["ok"], expect=True,
              note="flag OFF -> no gnw_bus key + every host field identical to the flag-ON response")
    v.require("intact bus answers the stored fact", les["intact_answers"], expect=True)
    v.require("lesion collapses the bus (ignition load-bearing)", les["lesion_collapses"], expect=True)
    v.require("forward-recall reflex survives the lesion (dissociation)", les["reflex_survives"], expect=True)
    v.control("lesion dissociation (bus decision needs the workspace ignition)",
              treatment=(1.0 if les["intact_answers"] else 0.0),
              control=(0.0 if les["lesion_collapses"] else 1.0), min_separation=0.0)
    v.disabled("heavy Gate-B organs (affect/worldmodel/surprise/metacog/... = 0) in the byte-identical handler check",
               why="disabled ONLY for speed; they run identically on both flag arms, so the comparison is unaffected")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 100, flush=True)
    print("  GNW BUS SHADOW — PRODUCTION VERIFY (real production ChatBrain + real brain_chat handler, rf recall)", flush=True)
    print("=" * 100, flush=True)
    print(f"  (A)+(C) AGREEMENT  bus-vs-host decision match: {direct['agree_n']}/{direct['agree_tot']} = {agree_frac:.3f}", flush=True)
    for r in direct["stored"]:
        print(f"      STORED  {r['q']:26s} host={r['host_recalled_svo']} bus_committed={r['bus_committed']!r} "
              f"reads={r['organ_reads']} agrees={r['agrees']}", flush=True)
    for r in direct["abstain"]:
        print(f"      ABSTAIN {r['q']:26s} host_abstain={r['host_abstained']} bus_abstain={r['bus_abstained']} "
              f"reads={r['organ_reads']} agrees={r['agrees']}", flush=True)
    print(f"  (B) LESION dissociation: intact_answers={les['intact_answers']} lesion_collapses={les['lesion_collapses']} "
          f"reflex_survives={les['reflex_survives']} "
          f"(intact={les['intact_committed']!r} lesioned={les['lesioned_committed']!r} reflex={les['reflex']!r})", flush=True)
    print(f"  (D) BYTE-IDENTICAL-WHEN-OFF (real handler, no gnw_bus key + host fields identical): {handler['ok']}", flush=True)
    for r in handler["rows"]:
        print(f"      HANDLER {r['q']:26s} off_no_key={r['off_has_no_bus_key']} on_key={r['on_has_bus_key']} "
              f"identical={r['host_fields_identical']}", flush=True)
    verdict = "GO" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 100, flush=True)

    out = {"runner": "_gnw_bus_shadow_production_verify", "go": go, "status": decided["status"],
           "agree_frac": agree_frac, "agree_n": direct["agree_n"], "agree_tot": direct["agree_tot"],
           "byte_identical_when_off": handler["ok"], "lesion": les, "direct": direct, "handler": handler,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_gnw_bus_shadow/production_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
