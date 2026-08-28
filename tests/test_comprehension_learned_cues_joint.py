"""CI guard — the JOINT flip of the comprehension organ's two corpus-learned cues (Vikunja #190):
`BRAIN_LEARNED_ANIMACY_CUE` (research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md)
+ `BRAIN_LEARNED_VERB_SELECTS` (research/findings/2026-08-27-comprehension-verb-selects-wired-GO.md), each
already individually pinned by its own CI guard (test_comprehension_learned_animacy_cue.py /
test_comprehension_learned_verbselects_cue.py). This file pins the JOINT behaviour verified by the 6-seed
organ-level flip-soak (`research/runners/_comprehension_learned_cues_joint_flip_soak.py`, GO — byte-identical-
off, zero hand-covered regression across every flag/lesion combination, 72/72 held-out coverage-extension
flips, per-cue lesion load-bearing, joint OOV-naming coverage) + the production-turn verify
(`_comprehension_learned_cues_joint_production_verify.py`):

  (1) BOTH flags ON is byte-identical to BOTH OFF on a hand-covered (fully hand-tabled) sentence -- the hand
      table is an unconditional fast path, so turning either/both learned cues on can never move a hand-
      covered outcome (the literal "the two cues don't interfere" interaction check).
  (2) held-out NOUN coverage extends with BOTH on (isolates the animacy cue; verb+other noun hand-covered).
  (3) held-out VERB coverage extends with BOTH on (isolates the verb-selects cue; both nouns hand-covered).
  (4) a DOUBLY held-out (noun AND verb both open-vocab) sentence: OFF, `repair_target` names all 3 content
      words OOV; BOTH ON, none of them are named OOV any more (the joint-specific coverage signal -- these
      items are `competent()`=True under EITHER flag state via `competent()`'s own fully-OOV/fully-covered
      symmetry, so the coverage signal here is the OOV-naming shrinking to empty, not a competent() flip).
  (5) lesioning ONLY the animacy cue reverts the noun-held-out item's coverage while leaving the verb-held-out
      item's coverage intact (and vice versa for lesioning only verb-selects) -- per-cue load-bearing in the
      JOINT (both-flags-on) state, not just each cue tested alone.
  (6) lesioning BOTH cues, both flags on, reverts EVERYTHING (hand-covered + held-out + joint + moat) to an
      exact match of both-flags-off.
  (7) the moat holds -- a fully-OOV sentence still abstains regardless of which/how-many flags are on.

CPU/numpy.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.comprehension_production_organ as CO

HAND_TEXT = "the dog eats the apple"           # dog/eat/apple: all 3 hand-table-covered
HELD_NOUN_TEXT = "the monkey eats the apple"   # monkey (held-out animate agent), eat+apple hand-covered
HELD_VERB_TEXT = "the dog clean the cup"       # dog+cup hand-covered, clean (held-out inanim-patient verb)
JOINT_TEXT = "the monkey clean the box"        # monkey/clean/box: ALL held-out (open-vocab)
MOAT_TEXT = "the wug blickets the glorp"       # fully OOV under any flag state


def _set_flags(animacy=None, verb=None, lesion_animacy=None, lesion_verb=None):
    for key, val in (
        ("BRAIN_LEARNED_ANIMACY_CUE", animacy),
        ("BRAIN_LEARNED_VERB_SELECTS", verb),
        ("BRAIN_LEARNED_ANIMACY_LESION", lesion_animacy),
        ("BRAIN_LEARNED_VERB_SELECTS_LESION", lesion_verb),
    ):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = "1" if val else "0"


def _clear_flags():
    _set_flags(animacy=None, verb=None, lesion_animacy=None, lesion_verb=None)


def _read(text):
    organ = CO.get_organ(seed=42)
    tr = CO.extract_transitive(text)
    comp = bool(organ.competent(*tr)) if tr else None
    j = organ.judge(text)
    rt = organ.repair_target(text)
    return comp, j, rt


def _stable(text):
    """A JITTER-FREE view for cross-condition comparison: (competent, comprehended, svo, repair-kind,
    oov-tokens). Deliberately excludes the raw spiking MARGIN float -- repeated same-process `judge()` calls on
    one organ exhibit this project's own documented chaotic inter-turn spiking jitter on the margin's exact
    value (research/FAILURE_LOG.md 2026-08-25 gap5-store entry), independent of any flag; the per-cue wire-in
    findings' own byte-identical proofs used a FRESH PROCESS per condition for that reason (see the 6-seed
    flip-soak `_comprehension_learned_cues_joint_flip_soak.py`, which is the rigorous float-level byte-identical
    proof). This CI guard instead pins the DECISION -- competent/comprehended/svo/repair shape -- which the
    organ's own hard-reset makes robust to same-process call order."""
    comp, j, rt = _read(text)
    j_view = None if j is None else (bool(j["comprehended"]), tuple(j["svo"]))
    rt_kind = None if rt is None else rt.get("kind")
    rt_oov = tuple(sorted(rt.get("oov_tokens") or [])) if (rt is not None and rt.get("kind") == "oov") else None
    return comp, j_view, rt_kind, rt_oov


def test_prerequisite_word_membership():
    # sanity: the battery's held-out words really are outside the hand tables (fails loudly on drift).
    assert "monkey" not in CO.ANIMACY and "box" not in CO.ANIMACY
    assert "clean" not in CO.VERB_SELECTS
    assert "dog" in CO.ANIMACY and "apple" in CO.ANIMACY and "cup" in CO.ANIMACY
    assert "eat" in CO.VERB_SELECTS


def test_hand_covered_byte_identical_both_off_vs_both_on():
    _set_flags(animacy=False, verb=False)
    off = _stable(HAND_TEXT)
    _set_flags(animacy=True, verb=True)
    on = _stable(HAND_TEXT)
    _clear_flags()
    assert off == on


def test_hand_covered_byte_identical_across_every_flag_combo():
    """The literal 'no interaction on a hand-covered case' check: every one of the 4 on/off combinations
    (neither / animacy-only / verb-only / both) gives the SAME DECISION on a hand-covered sentence."""
    _set_flags(animacy=False, verb=False)
    baseline = _stable(HAND_TEXT)
    for a, v in ((True, False), (False, True), (True, True)):
        _set_flags(animacy=a, verb=v)
        assert _stable(HAND_TEXT) == baseline, (a, v)
    _clear_flags()


def test_held_noun_coverage_extends_with_both_on():
    _set_flags(animacy=False, verb=False)
    comp_off, j_off, _ = _read(HELD_NOUN_TEXT)
    assert comp_off is False and j_off is None
    _set_flags(animacy=True, verb=True)
    comp_on, j_on, _ = _read(HELD_NOUN_TEXT)
    _clear_flags()
    assert comp_on is True
    assert j_on is not None and j_on["in_scope"] is True


def test_held_verb_coverage_extends_with_both_on():
    _set_flags(animacy=False, verb=False)
    comp_off, j_off, _ = _read(HELD_VERB_TEXT)
    assert comp_off is False and j_off is None
    _set_flags(animacy=True, verb=True)
    comp_on, j_on, _ = _read(HELD_VERB_TEXT)
    _clear_flags()
    assert comp_on is True
    assert j_on is not None and j_on["in_scope"] is True


def test_joint_doubly_held_out_oov_naming_shrinks_to_empty():
    _set_flags(animacy=False, verb=False)
    _, _, rt_off = _read(JOINT_TEXT)
    _set_flags(animacy=True, verb=True)
    _, j_on, rt_on = _read(JOINT_TEXT)
    _clear_flags()
    assert rt_off["kind"] == "oov"
    assert set(rt_off["oov_tokens"]) == {"monkey", "clean", "box"}
    # flags ON: all 3 words are now CLASSIFIED -> the OOV branch is never entered (no "oov_tokens" key at all --
    # the ROLE branch's dict shape has no such key; asserting "kind" is the correct, non-jitter-fragile check).
    assert rt_on["kind"] == "role"
    assert "oov_tokens" not in rt_on
    assert j_on["comprehended"] is True     # monkey(animate)+clean(inanim-patient)+box(inanimate) -> well-formed


def test_lesion_animacy_only_reverts_noun_spares_verb():
    _set_flags(animacy=True, verb=True, lesion_animacy=True, lesion_verb=False)
    comp_noun, _, _ = _read(HELD_NOUN_TEXT)
    comp_verb, _, _ = _read(HELD_VERB_TEXT)
    _clear_flags()
    assert comp_noun is False    # animacy cue lesioned -> the noun-held-out item reverts to out-of-scope
    assert comp_verb is True     # verb cue untouched -> the verb-held-out item's coverage stands


def test_lesion_verb_only_reverts_verb_spares_noun():
    _set_flags(animacy=True, verb=True, lesion_animacy=False, lesion_verb=True)
    comp_noun, _, _ = _read(HELD_NOUN_TEXT)
    comp_verb, _, _ = _read(HELD_VERB_TEXT)
    _clear_flags()
    assert comp_noun is True     # animacy cue untouched -> the noun-held-out item's coverage stands
    assert comp_verb is False    # verb cue lesioned -> the verb-held-out item reverts to out-of-scope


def test_lesion_both_full_revert():
    _set_flags(animacy=False, verb=False)
    off_hand, off_noun, off_verb, off_joint, off_moat = (
        _stable(HAND_TEXT), _stable(HELD_NOUN_TEXT), _stable(HELD_VERB_TEXT), _stable(JOINT_TEXT), _stable(MOAT_TEXT))
    _set_flags(animacy=True, verb=True, lesion_animacy=True, lesion_verb=True)
    les_hand, les_noun, les_verb, les_joint, les_moat = (
        _stable(HAND_TEXT), _stable(HELD_NOUN_TEXT), _stable(HELD_VERB_TEXT), _stable(JOINT_TEXT), _stable(MOAT_TEXT))
    _clear_flags()
    assert les_hand == off_hand
    assert les_noun == off_noun
    assert les_verb == off_verb
    assert les_joint == off_joint
    assert les_moat == off_moat


def test_moat_holds_both_on():
    _set_flags(animacy=True, verb=True)
    _, j, rt = _read(MOAT_TEXT)
    _clear_flags()
    assert j is not None and j["comprehended"] is False
    assert rt["kind"] == "oov"
    assert set(rt["oov_tokens"]) >= {"wug", "glorp", "blickets"}
