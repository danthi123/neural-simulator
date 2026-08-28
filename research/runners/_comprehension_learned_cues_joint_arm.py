"""ONE-CONDITION arm for the JOINT flip-soak of the two comprehension learned cues (Vikunja #190):
`BRAIN_LEARNED_ANIMACY_CUE` (research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md)
+ `BRAIN_LEARNED_VERB_SELECTS` (research/findings/2026-08-27-comprehension-verb-selects-wired-GO.md), BOTH
default-OFF individually. This script evaluates ONE fixed flag/seed condition, in a FRESH process (invoked by
the controller `_comprehension_learned_cues_joint_flip_soak.py` via subprocess -- the organ + the learned
lexicons are process-global singletons keyed by the FIRST seed/build, so a genuinely clean per-condition read
requires a fresh process, not a same-process sequential flag toggle), over a fixed battery:

  * HAND_COVERED  -- every hand VERB_SELECTS verb once (agent="dog", a hand-table-matching patient) + the two
    established covered-but-AMBIGUOUS sentences ("wolf watches owl" 2-animate symmetric-verb, "book carries
    cup" 2-inanimate) -- the NO-REGRESSION battery (must be byte-identical across every flag state, because
    `_animacy_of`/`_verb_selects_of` try the hand table FIRST, unconditionally, before ever consulting a flag).
  * HELD_NOUN  -- a hand-covered verb+patient, with a held-out (not in ANIMACY) agent noun -- isolates the
    ANIMACY cue. Half animate held-out nouns, half inanimate.
  * HELD_VERB  -- a hand-covered agent+patient, with a held-out (not in VERB_SELECTS) verb -- isolates the
    VERB_SELECTS cue. Half inanimate-patient held-out verbs, half animate-patient.
  * JOINT  -- BOTH the agent noun AND the verb held-out simultaneously (a genuinely fully-open-vocab sentence)
    -- the interaction stress test: does `_evs_for_organ`'s noun permute_map + verb v_eff substitution compose
    correctly when both fire on the SAME read.
  * MOAT  -- a fully-OOV sentence (verb + both nouns off every graph) -- must abstain regardless of flags.

Every held-out word was pre-verified (research/findings/raw/_comprehension_learned_cues_joint_wordcheck.json)
to (a) NOT be in the hand ANIMACY/VERB_SELECTS tables and (b) classify definitively (non-None) via the
deployment (seed=42) learned lexicon -- so a battery item abstaining is a genuine finding, not word-choice noise.

Run: SIM_BACKEND=numpy python -m research.runners._comprehension_learned_cues_joint_arm \\
    --seed 42 --animacy on --verb on --lesion-animacy off --lesion-verb off \\
    --out research/findings/raw/_comprehension_learned_cues_joint/arm_seed42_both_on.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

# ── Battery (module-level so the controller can import the SAME definitions for its own bookkeeping). ──
# All "dog"/"cat"/"apple"/... nouns and hand verbs below are HAND-TABLE-COVERED (verified against
# comprehension_production_organ.ANIMACY / VERB_SELECTS at module import in the controller's selftest).
HAND_COVERED = [
    ("hand_chase", "dog", "chase", "cat"),
    ("hand_eat", "dog", "eat", "apple"),
    ("hand_push", "dog", "push", "rock"),
    ("hand_carry", "dog", "carry", "book"),
    ("hand_bite", "dog", "bite", "bone"),
    ("hand_kick", "dog", "kick", "ball"),
    ("hand_grab", "dog", "grab", "stick"),
    ("hand_watch", "dog", "watch", "bird"),
    ("hand_ambig_2anim", "wolf", "watch", "owl"),      # 2-animate, symmetric verb -> ambiguous but COVERED
    ("hand_ambig_2inan", "book", "carry", "cup"),      # 2-inanimate-leaning -> ambiguous but COVERED
]

# Held-out (never hand-tabled) words, pre-verified against the deployment (seed=42) learned lexicons:
HELD_NOUN_ANIM = ["monkey", "rabbit", "kitten"]        # learned ANIMACY = animate
HELD_NOUN_INANIM = ["box", "table", "key"]             # learned ANIMACY = inanimate
HELD_VERB_INANIM_PATIENT = ["clean", "wash", "open"]   # learned VERB_SELECTS patient = inanimate
HELD_VERB_ANIM_PATIENT = ["help", "hug", "feed"]       # learned VERB_SELECTS patient = animate


def _held_noun_sentences():
    out = []
    for n in HELD_NOUN_ANIM:
        out.append((f"noun_anim_{n}", n, "eat", "apple"))       # held-out AGENT (animate), hand verb+patient
    for n in HELD_NOUN_INANIM:
        out.append((f"noun_inanim_{n}", "dog", "eat", n))       # hand agent+verb, held-out PATIENT (inanimate)
    return out


def _held_verb_sentences():
    out = []
    for v in HELD_VERB_INANIM_PATIENT:
        out.append((f"verb_inanimpat_{v}", "dog", v, "cup"))    # hand agent+patient, held-out inanim-patient verb
    for v in HELD_VERB_ANIM_PATIENT:
        out.append((f"verb_animpat_{v}", "dog", v, "cat"))      # hand agent+patient, held-out anim-patient verb
    return out


def _joint_sentences():
    return [
        ("joint_monkey_clean_box", "monkey", "clean", "box"),      # noun+verb+noun ALL held-out
        ("joint_rabbit_help_kitten", "rabbit", "help", "kitten"),  # noun+verb+noun ALL held-out
    ]


MOAT = [("moat_oov", "wug", "blickets", "glorp")]


def all_items():
    items = list(HAND_COVERED)
    items += [(lbl, n0, v, n1) for lbl, n0, v, n1 in _held_noun_sentences()]
    items += [(lbl, n0, v, n1) for lbl, n0, v, n1 in _held_verb_sentences()]
    items += [(lbl, n0, v, n1) for lbl, n0, v, n1 in _joint_sentences()]
    items += MOAT
    return items


def _set_flag(key: str, on: bool):
    # Always EXPLICIT ("1"/"0"), never left unset -- an arm's condition must not depend on whatever the
    # calling shell happened to have set, and "0" is the exact byte-identical escape value the eventual
    # default-flip will use (verified equal to "unset" by the controller's own dedicated safety-floor check).
    os.environ[key] = "1" if on else "0"


def _eval_battery(organ, CO, items):
    rows = {}
    for label, n0, v, n1 in items:
        text = f"the {n0} {v} the {n1}"
        tr = CO.extract_transitive(text)
        comp = bool(organ.competent(*tr)) if tr else None
        j = organ.judge(text)
        rt = organ.repair_target(text)
        rows[label] = {
            "text": text,
            "extract_transitive": list(tr) if tr else None,
            "competent": comp,
            "judge": j,
            "repair_target": rt,
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--animacy", choices=["on", "off", "unset"], default="off")
    ap.add_argument("--verb", choices=["on", "off", "unset"], default="off")
    ap.add_argument("--lesion-animacy", choices=["on", "off"], default="off")
    ap.add_argument("--lesion-verb", choices=["on", "off"], default="off")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for key, mode in (
        ("BRAIN_LEARNED_ANIMACY_CUE", args.animacy),
        ("BRAIN_LEARNED_VERB_SELECTS", args.verb),
    ):
        if mode == "unset":
            os.environ.pop(key, None)
        else:
            _set_flag(key, mode == "on")
    _set_flag("BRAIN_LEARNED_ANIMACY_LESION", args.lesion_animacy == "on")
    _set_flag("BRAIN_LEARNED_VERB_SELECTS_LESION", args.lesion_verb == "on")

    import research.runners.comprehension_production_organ as CO

    # Sanity: every HAND_COVERED word really is hand-table-covered (fails loudly if the battery drifted).
    for _, n0, v, n1 in HAND_COVERED:
        assert n0 in CO.ANIMACY and n1 in CO.ANIMACY and v in CO.VERB_SELECTS, (n0, v, n1)
    for n in HELD_NOUN_ANIM + HELD_NOUN_INANIM:
        assert n not in CO.ANIMACY, n
    for v in HELD_VERB_INANIM_PATIENT + HELD_VERB_ANIM_PATIENT:
        assert v not in CO.VERB_SELECTS, v

    organ = CO.get_organ(seed=args.seed)
    battery = _eval_battery(organ, CO, all_items())

    payload = {
        "seed": args.seed,
        "flags": {
            "animacy": args.animacy, "verb": args.verb,
            "lesion_animacy": args.lesion_animacy, "lesion_verb": args.lesion_verb,
        },
        "battery": battery,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
