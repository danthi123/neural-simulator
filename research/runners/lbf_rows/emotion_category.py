"""LBF row for the Stage-2 C4 emotion-category organ (default-ON since 2026-10-15, finding
research/findings/2026-10-15-emotion-category-default-on.md). Added 2026-10-27 on Claude's decision (option (a)) for
the BRAIN_EMOTION_CATEGORY_BOUNDED flip pipeline (RULING emo-bg-merge-1026 (2)): the faculty belongs in the
load-bearing registry, and the flip's dev-seed smoke (tools/answer_rate.py paired) reads the `_load_bearing/_shards`
layout, which only a registry row writes.

WHAT THE ROW MEASURES, AT REPLY LEVEL. The feel read-out ("how do you feel") appends one sentence naming the emotion
channel a spiking commit latched on the organ's held ladder (research/runners/emotion_category_organ.py): "The emotion
channel that latched on my affect ladder is joy." or, when nothing latched, "No single emotion channel latched on my
affect ladder." The probe is one session: the Stage-2 gate's frozen joy situation presented twice (bare, so only its
words move the mood), then the production feel query. Intact: the organ names joy. Lesioned: the organ's
readout -> commit-gate transmission is closed for the read, no latch turns on, and the same turn ends with the "no
single channel" sentence. So a reply change here is caused by the commit stage, the part that names the category.
Load-bearing = the lesion changes the REPLY (docs/TERMS.md `load-bearing`).

WHICH LESION, AND WHY NOT `affect_out`. The organ carries several transmission-gate lesions (its module docstring,
LESIONS). The Stage-2 gate's registered C4 lesion is `affect_out` (`BRAIN_AFFECT_LESION=1`), but that flag is the
PRODUCTION ladder's own readout gate too: it also moves the number and the arousal band of the sign-and-magnitude
sentence the feel read-out prints before the category clause, so an `answer` change under it would be partly the
production ladder's. This row uses the organ's own dedicated knob, `BRAIN_EMOTION_CATEGORY_COMMIT_LESION=1` (the
readout -> commit-gate cut, `emo_commit`, read in `emotion_category_organ.read_turn`): the text before the category
sentence is untouched, so the whole `answer` difference IS the organ's clause. The same pattern as the other organ-own
rows (affect-marker-spiking-wta, word-role-commit). The `affect_out` arm stays the Stage-2 runner's `lesion_affect`.

KIND. neural-lesion. Compared fields: `answer` (the reply) and `emotion_category.category` (the committed channel,
the registered trace field; absent when the organ is off, so it cannot move for a reason other than the organ).
FLIP STATUS: default-ON since the 2026-10-15 flip (webapp/server.py `_EMOTION_CATEGORY_DEFAULT_ON`), so the row carries
NO REQUIRED_ENV and joins the headline set the moment it merges. `BRAIN_EMOTION_CATEGORY=0` is the byte-identical
escape; with it the lesion changes nothing.

BOUNDED CONDUCTANCE. `BRAIN_EMOTION_CATEGORY_BOUNDED` (default OFF until its own flip) is NOT part of this row: both
arms run whatever the process env carries, so a shard launched with `--extra-env BRAIN_EMOTION_CATEGORY_BOUNDED=1`
measures the row with the guard on in the intact AND the lesion arm (the flip's dev smoke does exactly that).

INTERFACE: `EXTRA_LESIONS` / `EXTRA_TURNS` / `EXTRA_PROBES` follow the entry shapes of FACULTY_LESIONS / PROBE_TURNS /
FACULTY_PROBES; the AG-REG hook (research/runners/lbf_rows/__init__.py) merges them. EXTRA_TURNS must stay a LITERAL
list (the battery reads it with ast, never imported), so the joy situation is spelled out here and equality with
stage2_exit_gate.SITUATIONS["joy"] is pinned in selftest() and in tests/test_emotion_category_lbf_row.py.
"""
from __future__ import annotations

# Attribution discipline (tools.lab): `expectation_holds` below asks whether the lesion MOVED the reply, and whether the
# move is the organ's clause alone (the intact reply = the lesioned reply + the category sentence swapped), through
# `lever`, instead of leaving two answers one key apart for a reader to subtract.
from tools.lab import lever

# ── the lesion knob: the organ's own readout -> commit-gate cut ──────────────────────────────────────────────────────
EXTRA_LESIONS = {
    "emotion-category": dict(
        flag="BRAIN_EMOTION_CATEGORY_COMMIT_LESION", value="1", kind="neural-lesion",
        note="research/runners/emotion_category_organ.py read_turn(): BRAIN_EMOTION_CATEGORY_COMMIT_LESION=1 keeps the "
             "readout -> commit-gate transmission closed for the read window (gate emo_commit), so no commit latch "
             "turns on and the feel clause says no single channel latched. Probe = the Stage-2 frozen joy situation "
             "twice then 'how do you feel' in one fresh session ('emocat'): intact names joy, lesioned names none; the "
             "text before the clause is identical in both arms. Requires BRAIN_EMOTION_CATEGORY on (the default); "
             "=0 is the byte-identical escape and the lesion then changes nothing. Not BRAIN_AFFECT_LESION: that "
             "flag also cuts the production ladder, whose number precedes the clause."),
}

# (label, message, session, reset, percept, rich) -- the PROBE_TURNS shape, one session, the first turn resets.
# Must stay a LITERAL list (the battery reads it with ast). The joy situation is stage2_exit_gate.SITUATIONS["joy"].
EXTRA_TURNS = [
    ("emocat_s1", "you win a gift and you are happy and glad and delighted", "emocat", True, None, False),
    ("emocat_s2", "you win a gift and you are happy and glad and delighted", "emocat", False, None, False),
    ("emocat_feel", "how do you feel", "emocat", False, None, False),
]

EXTRA_PROBES = [
    # (faculty_key, turn_label, decision_field_paths, thin) -- `answer` is the primary reply field (docs/TERMS.md
    # `load-bearing`); `emotion_category.category` is the committed channel the clause is rendered from.
    ("emotion-category", "emocat_feel", ["answer", "emotion_category.category"], False),
]


def expectation_holds(intact: dict, lesion: dict) -> dict:
    """{"held", "moved", "detail"} for the intact and lesion reply dicts of the probe turn. held = the intact reply names
    joy, the lesioned reply says no single channel latched, and the text before the category sentence is identical in both
    (so the whole difference is the organ's clause). A data check, never a claim; never raises."""
    try:
        from research.runners import emotion_category_organ as E
        ia, la = str(intact.get("answer") or ""), str(lesion.get("answer") or "")
        moved = lever("emotion-category reply", ia, la, required=False)
        ci, cl = E.feel_clause("joy"), E.feel_clause(None)
        held = bool(ia.endswith(ci) and la.endswith(cl) and ia[:-len(ci)] == la[:-len(cl)])
        return {"held": held and moved, "moved": moved, "detail": "intact=%r lesion=%r" % (ia, la)}
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed as held
        return {"held": False, "moved": None, "detail": "unreadable arm: %s" % e}


def selftest() -> dict:
    """Shape checks only (no brain build)."""
    import os
    import re
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS
    from research.runners import emotion_category_organ as E
    from research.runners.affect_production_organ import is_feel_query
    labels = {t[0] for t in list(PROBE_TURNS) + list(_EXTRA_TURNS)} | set(_TURN_BY_LABEL)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        gate_src = open(os.path.join(here, "stage2_exit_gate.py"), encoding="utf8").read()
    except OSError:
        gate_src = ""
    m = re.search(r'"joy":\s*"([^"]+)"', gate_src[gate_src.find("SITUATIONS = {"):])
    joy = m.group(1) if m else None
    try:
        org_src = open(os.path.join(here, "emotion_category_organ.py"), encoding="utf8").read()
    except OSError:
        org_src = ""
    by = {t[0]: t for t in EXTRA_TURNS}
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES keys": set(EXTRA_LESIONS) == {r[0] for r in EXTRA_PROBES},
        "probe turn reachable by the battery": all(r[1] in labels for r in EXTRA_PROBES),
        "turns share one session, first resets, no other does": (
            len({t[2] for t in EXTRA_TURNS}) == 1 and EXTRA_TURNS[0][3] and not any(t[3] for t in EXTRA_TURNS[1:])),
        "the probe is the last turn and a feel query": (EXTRA_TURNS[-1][0] == EXTRA_PROBES[0][1]
                                                       and is_feel_query(by["emocat_feel"][1])),
        "the two exposures are the Stage-2 frozen joy situation, twice": (
            joy is not None and by["emocat_s1"][1] == joy and by["emocat_s2"][1] == joy),
        "the exposures are not feel queries": not any(is_feel_query(by[k][1]) for k in ("emocat_s1", "emocat_s2")),
        "the clause names one category when joy latches and none otherwise": (
            E.feel_clause("joy") != E.feel_clause(None) and "joy" in E.feel_clause("joy")),
        "lesion is the organ's own commit cut, neural-lesion, no REQUIRED_ENV": (
            EXTRA_LESIONS["emotion-category"]["flag"] == "BRAIN_EMOTION_CATEGORY_COMMIT_LESION"
            and EXTRA_LESIONS["emotion-category"]["kind"] == "neural-lesion" and "REQUIRED_ENV" not in globals()),
        "the lesion flag is read by the organ": "BRAIN_EMOTION_CATEGORY_COMMIT_LESION" in org_src,
        "reply field + registered trace field, not thin": (
            EXTRA_PROBES[0][2] == ["answer", "emotion_category.category"] and EXTRA_PROBES[0][3] is False),
    }
    return {"ok": all(checks.values()), "checks": checks}
