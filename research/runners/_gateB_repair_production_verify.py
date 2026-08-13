"""VERIFY the OTHER-REPAIR wiring (faculty-map T1-6) through the REAL /api/brain-chat handler, numpy-CPU.

Composes the already-wired D4 comprehension monitor: on a low-comprehension transitive that the D4 gate would
ABSTAIN on, the turn instead emits a TARGETED clarification naming the unresolved thematic ROLE (from the D4
organ's per-noun spiking sel-pool read) or the OOV token. Default-ON; BRAIN_REPAIR=0 -> the bare abstain.

Checks (all through webapp.server.brain_chat, tiny-demo, stub renderer, composer=rf):
  1. 2-inanim covered-ambiguous ('the book carries the cup') -> role=AGENT clarification (names both nouns, ?).
  2. 2-animate covered-ambiguous ('the wolf watches the owl') -> generic role-swap clarification (targeted, ?).
  3. fully-OOV ('the wug blickets the glorp')            -> OOV token naming (host-lexical scaffold).
  4. comprehensible transitive ('the wolf bites the apple') -> NO repair (no false repair).
  5. LESION (BRAIN_COMPREHENSION_LESION=1) on the 2-inanim -> repair collapses -> the BARE abstain.
  6. FLAG-OFF (BRAIN_REPAIR=0) on the 2-inanim -> the BARE abstain, NO repair key (byte-identical).

Run:  SIM_BACKEND=numpy python -m research.runners._gateB_repair_production_verify
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

OUT = "research/findings/raw/_gateB_repair_production_verify.json"


def _turn(msg, session, reset=False):
    from webapp.server import brain_chat, BrainChatRequest
    r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                    renderer="stub", reset=reset))
    return json.loads(r.body)


def main():
    import research.runners.comprehension_production_organ as CO
    bare = CO.didnt_follow_message(None)

    os.environ.pop("BRAIN_REPAIR", None)
    os.environ.pop("BRAIN_COMPREHENSION_LESION", None)

    d_role = _turn("the book carries the cup", "v1", reset=True)
    d_anim = _turn("the wolf watches the owl", "v1")
    d_oov = _turn("the wug blickets the glorp", "v1")
    d_ok = _turn("the wolf bites the apple", "v1")

    os.environ["BRAIN_COMPREHENSION_LESION"] = "1"
    d_les = _turn("the book carries the cup", "v2", reset=True)
    os.environ.pop("BRAIN_COMPREHENSION_LESION", None)

    os.environ["BRAIN_REPAIR"] = "0"
    d_off = _turn("the book carries the cup", "v3", reset=True)
    os.environ.pop("BRAIN_REPAIR", None)

    def rep(d):
        return d.get("repair") or {}

    checks = {
        "role_agent_targeted": bool(
            d_role.get("abstained") and rep(d_role).get("kind") == "role"
            and rep(d_role).get("role") == "agent" and rep(d_role).get("repaired") is True
            and rep(d_role).get("loadbearing") == "spiking_role_evidence"
            and "AGENT" in d_role.get("answer", "") and "book" in d_role.get("answer", "")
            and "cup" in d_role.get("answer", "") and d_role.get("answer", "").strip().endswith("?")),
        "role_animate_generic_targeted": bool(
            d_anim.get("abstained") and rep(d_anim).get("kind") == "role"
            and rep(d_anim).get("repaired") is True and d_anim.get("answer", "").strip().endswith("?")),
        "oov_token_named": bool(
            rep(d_oov).get("kind") == "oov" and rep(d_oov).get("repaired") is True
            and rep(d_oov).get("loadbearing") == "host_lexical"
            and any(t in d_oov.get("answer", "") for t in (rep(d_oov).get("oov_tokens") or []))),
        "no_false_repair_on_comprehensible": bool("repair" not in d_ok),
        "lesion_collapses_to_bare_abstain": bool(
            d_les.get("answer") == bare and rep(d_les).get("repaired") is False),
        "flagoff_bare_abstain_no_key": bool(d_off.get("answer") == bare and "repair" not in d_off),
    }
    all_ok = all(checks.values())
    payload = {
        "faculty": "T1-6 other-repair (targeted clarification on a low-comprehension abstain)",
        "backend": os.environ.get("SIM_BACKEND"),
        "composer": os.environ.get("BRAIN_COMPOSER_KIND"),
        "bare_abstain": bare,
        "checks": checks,
        "ALL_OK": all_ok,
        "examples": {
            "role_agent": {"in": "the book carries the cup", "answer": d_role.get("answer"),
                           "repair": rep(d_role)},
            "role_generic": {"in": "the wolf watches the owl", "answer": d_anim.get("answer"),
                             "repair": rep(d_anim)},
            "oov": {"in": "the wug blickets the glorp", "answer": d_oov.get("answer"),
                    "repair": rep(d_oov)},
            "comprehensible": {"in": "the wolf bites the apple", "answer": d_ok.get("answer"),
                               "repair": d_ok.get("repair")},
            "lesion": {"in": "the book carries the cup", "answer": d_les.get("answer"),
                       "repair": rep(d_les)},
            "flagoff": {"in": "the book carries the cup", "answer": d_off.get("answer"),
                        "repair": d_off.get("repair")},
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nALL_OK={all_ok}  wrote {OUT}")


if __name__ == "__main__":
    main()
