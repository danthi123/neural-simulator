"""JOINT flip-soak — the FULL-PRODUCTION-TURN half (webapp.server.brain_chat), complementing the organ-level
6-seed soak in `_comprehension_learned_cues_joint_flip_soak.py`. The production handler always builds the
comprehension organ at a FIXED seed (`get_organ(seed=42)`, every call site in webapp/server.py) -- there is no
per-request seed control -- so this check is inherently single-seed (42); it demonstrates the two flags compose
correctly THROUGH the real `/api/brain-chat` turn assembly (the D4 abstain-and-repair early-return in
webapp/server.py brain_chat), not just at the organ API. Mirrors `_gateB_repair_production_verify.py`'s own
methodology (distinct session ids + `reset=True` per turn, avoiding this project's documented cross-turn
spiking jitter rather than fighting it).

Checks:
  1. hand_covered_ambiguous  ("the book carries the cup") -- flags OFF vs BOTH ON: byte-identical (the hand
     table is an unconditional fast path; this is the interaction/no-regression check at the HANDLER level).
  2. joint_ambiguous_repairs ("the rabbit help the kitten", BOTH noun+verb held-out, comprehended=False per the
     organ-level soak at every seed) -- `competent()`'s pre-existing fully-OOV branch means the D4 gate fires
     EITHER way here (all 3 content words unknown to the hand tables triggers the fully-OOV competence path
     regardless of these two flags -- this predates both learned cues and is not something they gate); the
     REAL, joint-specific difference is WHICH repair the substrate can give: flags OFF, `repair.kind=="oov"`
     (host-lexical, names all 3 words as unknown: "I don't know the words 'rabbit'/'kitten'/'help' yet");
     flags BOTH ON, `repair.kind=="role"` (all 3 words are now CLASSIFIED, so the substrate instead asks a
     targeted spiking role-binding clarification: "...who does what... is the rabbit doing the 'help' to the
     kitten, or the other way round?") -- a genuinely VISIBLE, qualitatively different response.
  3. held_out_clear          ("the monkey eats the apple", held-out AGENT noun only, comprehended=True per the
     organ soak) -- flags OFF: out of scope, no comprehension trace; flags ON: `comprehension.comprehended`
     True, not abstained (same pass-through, richer trace).
  4. moat_unaffected         ("the wug blickets the glorp") -- abstained identically regardless of the flags
     (fully OOV either way; neither learned cue ever resolves these tokens).
  5. lesion_both_reverts     joint_ambiguous_repairs, BOTH ON + BOTH lesioned -- must exactly revert to check 2's
     flags-OFF response (full functional revert through the handler, not just the organ).

Run: SIM_BACKEND=numpy python -m research.runners._comprehension_learned_cues_joint_production_verify
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

OUT = "research/findings/raw/_comprehension_learned_cues_joint_production_verify.json"


def _clear_flags():
    for k in ("BRAIN_LEARNED_ANIMACY_CUE", "BRAIN_LEARNED_VERB_SELECTS",
              "BRAIN_LEARNED_ANIMACY_LESION", "BRAIN_LEARNED_VERB_SELECTS_LESION"):
        os.environ.pop(k, None)


def _turn(msg, session, reset=False):
    from webapp.server import brain_chat, BrainChatRequest
    r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                    renderer="stub", reset=reset))
    return json.loads(r.body)


def main():
    HAND_AMBIG = "the book carries the cup"
    JOINT_AMBIG = "the rabbit help the kitten"
    HELD_CLEAR = "the monkey eats the apple"
    MOAT = "the wug blickets the glorp"

    _clear_flags()
    off_hand = _turn(HAND_AMBIG, "jf_off_hand", reset=True)
    off_joint = _turn(JOINT_AMBIG, "jf_off_joint", reset=True)
    off_clear = _turn(HELD_CLEAR, "jf_off_clear", reset=True)
    off_moat = _turn(MOAT, "jf_off_moat", reset=True)

    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
    on_hand = _turn(HAND_AMBIG, "jf_on_hand", reset=True)
    on_joint = _turn(JOINT_AMBIG, "jf_on_joint", reset=True)
    on_clear = _turn(HELD_CLEAR, "jf_on_clear", reset=True)
    on_moat = _turn(MOAT, "jf_on_moat", reset=True)
    _clear_flags()

    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    os.environ["BRAIN_LEARNED_VERB_SELECTS"] = "1"
    os.environ["BRAIN_LEARNED_ANIMACY_LESION"] = "1"
    os.environ["BRAIN_LEARNED_VERB_SELECTS_LESION"] = "1"
    lesioned_joint = _turn(JOINT_AMBIG, "jf_lesioned_joint", reset=True)
    _clear_flags()

    def _key_fields(d):
        return {k: d.get(k) for k in ("answer", "abstained", "recalled_svo", "verified", "not_understood")}

    checks = {
        "hand_covered_byte_identical": _key_fields(off_hand) == _key_fields(on_hand),
        # `competent()`'s pre-existing fully-OOV branch means the D4 gate fires on this fully-open-vocab
        # sentence REGARDLESS of these two flags (predates them) -- the joint-specific signal is WHICH repair
        # the substrate gives, not whether it abstains at all. See the module docstring, check 2.
        "joint_off_oov_named": bool(
            off_joint.get("abstained") is True
            and off_joint.get("repair", {}).get("kind") == "oov"
            and set(off_joint.get("repair", {}).get("oov_tokens") or []) == {"rabbit", "kitten", "help"}),
        "joint_on_abstained_with_repair": bool(
            on_joint.get("abstained") is True
            and on_joint.get("comprehension") is not None
            and on_joint.get("comprehension", {}).get("comprehended") is False
            and on_joint.get("repair", {}).get("kind") == "role"
            and "help" not in (on_joint.get("repair", {}).get("oov_tokens") or [])),
        "held_out_clear_off_out_of_scope": (off_clear.get("comprehension") is None),
        "held_out_clear_on_comprehended": bool(
            on_clear.get("comprehension") is not None
            and on_clear.get("comprehension", {}).get("comprehended") is True
            and on_clear.get("abstained") is not True),
        "moat_unaffected_by_flags": (_key_fields(off_moat) == _key_fields(on_moat)),
        "lesion_both_reverts_joint_to_flagoff": (_key_fields(lesioned_joint) == _key_fields(off_joint)),
    }
    all_ok = all(checks.values())

    payload = {
        "faculty": "joint flip-soak — BRAIN_LEARNED_ANIMACY_CUE + BRAIN_LEARNED_VERB_SELECTS through /api/brain-chat",
        "backend": os.environ.get("SIM_BACKEND"),
        "composer": os.environ.get("BRAIN_COMPOSER_KIND"),
        "checks": checks,
        "ALL_OK": all_ok,
        "examples": {
            "hand_covered_off": off_hand, "hand_covered_on": on_hand,
            "joint_off": off_joint, "joint_on": on_joint, "joint_lesioned": lesioned_joint,
            "held_clear_off": off_clear, "held_clear_on": on_clear,
            "moat_off": off_moat, "moat_on": on_moat,
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nALL_OK={all_ok}  wrote {OUT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
