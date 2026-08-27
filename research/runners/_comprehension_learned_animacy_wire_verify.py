"""VERIFY the comprehension organ's ANIMACY-cue wire-in (`comprehension_production_organ.py`,
`BRAIN_LEARNED_ANIMACY_CUE`), numpy-CPU: (1) flag-OFF byte-identity on the organ's own outputs
(judge/repair_target), captured for an exact JSON diff against the pre-edit code; (2) LOAD-BEARING: a
held-out animate word the hand ANIMACY table lacks becomes competent+judged only with the flag ON, and
`BRAIN_LEARNED_ANIMACY_LESION=1` reverts it to an exact match of the flag-OFF case; (3) the no-confab MOAT
holds on a genuinely off-graph token.

Run: SIM_BACKEND=numpy python -m research.runners._comprehension_learned_animacy_wire_verify \
    --out research/findings/raw/_comprehension_learned_animacy_wire_verify.json
"""
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

OUT = "research/findings/raw/_comprehension_learned_animacy_wire_verify.json"


def _clear_flags():
    os.environ.pop("BRAIN_LEARNED_ANIMACY_CUE", None)
    os.environ.pop("BRAIN_LEARNED_ANIMACY_LESION", None)


def main():
    import research.runners.comprehension_production_organ as CO

    organ = CO.get_organ(seed=42)
    held_out_word = "monkey"
    text = "the monkey carries the cup"
    assert held_out_word not in CO.ANIMACY, "'monkey' must NOT be in the hand ANIMACY table for this demo"

    rows = {}
    for label, cue_on, lesion_on in (
        ("flag_off", False, False),
        ("flag_on", True, False),
        ("flag_on_lesioned", True, True),
        ("flag_off_again", False, False),
    ):
        _clear_flags()
        if cue_on:
            os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
        if lesion_on:
            os.environ["BRAIN_LEARNED_ANIMACY_LESION"] = "1"
        tr = CO.extract_transitive(text)
        rows[label] = {
            "extract_transitive": list(tr) if tr else None,
            "competent": bool(organ.competent(*tr)) if tr else None,
            "judge": organ.judge(text),
        }
    _clear_flags()

    lesioned_matches_flagoff = rows["flag_on_lesioned"] == {
        **rows["flag_off"], "judge": rows["flag_off"]["judge"]} and \
        rows["flag_on_lesioned"]["judge"] == rows["flag_off"]["judge"] == rows["flag_off_again"]["judge"]

    # ── MOAT: a genuinely off-graph token (flag ON) still abstains -- no confabulated category. ──
    os.environ["BRAIN_LEARNED_ANIMACY_CUE"] = "1"
    lex = CO._get_learned_animacy_lexicon()
    lex.set_lesion(False)
    oov_text = "the wug blickets the glorp"
    moat = {
        "classify_wug": lex.classify("wug"),
        "classify_glorp": lex.classify("glorp"),
        "judge": organ.judge(oov_text),
        "repair_target": organ.repair_target(oov_text),
    }
    _clear_flags()

    payload = {
        "held_out_word": held_out_word,
        "text": text,
        "conditions": rows,
        "lesioned_reverts_to_flag_off_exact_match": bool(lesioned_matches_flagoff),
        "moat_check_oov": moat,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    print(f"\nlesioned_reverts_to_flag_off_exact_match = {lesioned_matches_flagoff}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
