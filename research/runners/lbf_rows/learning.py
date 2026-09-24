"""LBF rows: d5-consolidate / sleep-replay (A2, 2026-09-24 midnight plan step S13).

Pre-registration: research/findings/2026-09-24-learning-rows-d5-consolidate-sleep-replay-lbf-PREREGISTRATION.md

Two new load-bearing-fraction rows for the shipped, default-ON offline-learning faculties:

  * D5 learn-through-use consolidation (`BRAIN_D5_CONSOLIDATE`, webapp/continuous_engine.py, default-ON since
    2026-08-21): a memory the brain RECALLED during a live turn is re-activated on the next idle tick, and the
    substrate's OWN plateau-gated BTSP (`sim/bridge.py fused_btsp_update`) strengthens its within-assembly
    weights. `BRAIN_D5_CONSOLIDATE=0` is the shipped byte-identical escape -- a MASTER-SWITCH disable
    (kind="whether-disable"; no dedicated `BRAIN_D5_CONSOLIDATE_LESION` knob exists yet), not the PRIMARY
    "neural-lesion" class.
  * Offline sleep-replay (`BRAIN_SLEEP_REPLAY`, webapp/continuous_engine.py, default-ON since 2026-08-26, 6-seed
    pool/GPU soak GO): on a genuine sleep-depth idle (>= SLEEP_IDLE_SEC), the BATCH of episodes stored since the
    last sleep is reactivated in store order through the SAME BTSP kernel. `BRAIN_SLEEP_REPLAY=0` is the shipped
    byte-identical escape -- also a master-switch disable (kind="whether-disable"), same reasoning.

CLASSIFICATION, corrected 2026-09-24 (adversarial review of this lane caught the initial commit's mislabel): both
rows are `kind="whether-disable"`, not `"neural-lesion"`. Per `load_bearing_fraction.py`'s own taxonomy, PRIMARY/
gold-standard "neural-lesion" status requires a dedicated `BRAIN_<X>_LESION` flag (value="1") that cuts the
neural read while the organ stays installed -- every one of that module's ~26 existing neural-lesion rows follows
this shape with zero exceptions. Neither `BRAIN_D5_CONSOLIDATE_LESION` nor `BRAIN_SLEEP_REPLAY_LESION` exists;
`BRAIN_D5_CONSOLIDATE=0` / `BRAIN_SLEEP_REPLAY=0` are the shipped master switches, so value="0" on a
non-`_LESION`-suffixed flag is the weaker "whether-disable" class -- matching the existing
"confidence-forthcomingness" row's precedent. See the PREREG's Amendment 2 for the full trace.

Both drive turn groups added to `research.runners.onebrain_regression_battery._EXTRA_TURNS` (label-only, kept
OUT of `PROBE_TURNS` -> the default roster, every flip-verify harness and the regression battery stay BYTE-
IDENTICAL): sessions 'd5c' (teach -> recall1 [precondition] -> the shared _WORLD_NIGHT idle tick -> recall2
[driving]) and 'slp' (3 teaches -> the same idle tick, now sleep-depth -> a recall of the middle-stored episode
[driving]).

THE GRADED DIFF FIELD. Both rows compare `episodic.graded_cue.depth_hold` (the SAME `SURFACED_GRADED_READ` the
2026-08-20 graded-apical-read finding validated: a genuine spiking apical magnitude, `max(cp_v_apical - v_hold,
0)`, not a host formula) alongside `answer` (the rendered reply, which embeds the same magnitude as "recall
strength X.X mV" only when the strength-surfacing gate is open). `episodic.in_memory` is deliberately NOT
compared: it is True in both arms on both recalls for both faculties (neither lesion touches the completion
gate, only the STRENGTH a completed recall reads), so it would only ever read `pass`.

IN-LOOP-LEARNING (not re-registered here; see the prereg's own section). Reading
`research/runners/one_brain_composer.py:604-655` confirms every live chat-teach path ends in
`self.kb.append((fact, None))` -- a plain Python list, not the spiking DG-CA3/BTSP `EpisodicDapMemory` store the
two rows above drive. The current `FACULTY_LESIONS["in-loop-learning"]` entry (kind="in-process", no env knob)
is therefore consistent with a HOST-WRITE classification; this module does not touch that existing key (AG-REG
owns literal registry edits -- see the module docstring in `research/runners/lbf_rows/__init__.py`).
"""
from __future__ import annotations

# ── EXTRA_LESIONS: same shape as research.runners.load_bearing_fraction.FACULTY_LESIONS ────────────────────────
# CORRECTED 2026-09-24 (adversarial review of this lane, A2): both rows originally shipped as kind="neural-lesion"
# in the first commit. That is a MISCLASSIFICATION per this taxonomy's own convention (load_bearing_fraction.py's
# module docstring + every one of its ~26 existing neural-lesion rows, zero exceptions): a dedicated
# `BRAIN_<X>_LESION` flag with value="1" is required for the PRIMARY/gold-standard "neural-lesion" class; neither
# `BRAIN_D5_CONSOLIDATE_LESION` nor `BRAIN_SLEEP_REPLAY_LESION` exists in webapp/continuous_engine.py (grepped,
# zero hits) -- only the two shipped default-ON MASTER SWITCHES do, and value="0" on a non-`_LESION`-suffixed flag
# is the weaker "whether-disable" class, matching the existing "confidence-forthcomingness" precedent below.
# Corrected here to kind="whether-disable"; see the PREREG's Amendment 2 for the full trace.
EXTRA_LESIONS = {
    "d5-consolidate": dict(
        flag="BRAIN_D5_CONSOLIDATE", value="0", kind="whether-disable",
        note="no BRAIN_*_LESION knob; webapp/continuous_engine.py: BRAIN_D5_CONSOLIDATE=0 removes the organ's "
             "reactivation loop wholesale (consolidate_used_memory becomes a no-op -- the tick still runs, "
             "n_sessions_ticked>0, but the substrate's plateau-gated BTSP reactivation never fires), so a later "
             "recall's graded strength (episodic.graded_cue.depth_hold) never rises and its reply never surfaces "
             "'recall strength' (a structural change, but a disable not a neural cut). Default-ON "
             "(BRAIN_D5_CONSOLIDATE unset -> True); this is the shipped byte-identical escape, not a new knob. "
             "Minimal neural lesion to add: BRAIN_D5_CONSOLIDATE_LESION cutting only the BTSP reactivation call "
             "while leaving the idle-tick scan + organ installed."),
    "sleep-replay": dict(
        flag="BRAIN_SLEEP_REPLAY", value="0", kind="whether-disable",
        note="no BRAIN_*_LESION knob; webapp/continuous_engine.py: BRAIN_SLEEP_REPLAY=0 removes the organ's "
             "batch-reactivation wholesale (consolidate_sleep_replay becomes an immediate no-op -- the deep-idle "
             "tick still runs) -- the recently-stored batch is never reactivated, so a later recall never reads a "
             "risen depth_hold and never surfaces the 'I also replayed it offline' clause + when-rank/batch-size "
             "(a structural change, but a disable not a neural cut). Default-ON (unset -> True, post 2026-08-26 "
             "6-seed soak GO); this is the shipped byte-identical escape, not a new knob. Minimal neural lesion "
             "to add: BRAIN_SLEEP_REPLAY_LESION cutting only the batch BTSP reactivation while leaving the "
             "deep-idle tick + organ installed."),
}

# ── EXTRA_PROBES: same shape as research.runners.onebrain_regression_battery.FACULTY_PROBES ────────────────────
# (faculty_key, turn_label, decision_field_paths, thin)
EXTRA_PROBES = [
    ("d5-consolidate", "d5c_recall2", ["episodic.graded_cue.depth_hold", "answer"], False),
    ("sleep-replay", "slp_recall", ["episodic.graded_cue.depth_hold", "answer"], False),
]


def _selftest_checks():
    """Static wiring checks (no brain build). Mirrors the shape load_bearing_fraction.selftest() already checks
    for every other row, so AG-REG's merge hook can fold this straight into that function's `checks` dict."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, PROBE_TURNS, _EXTRA_TURNS

    def _turn_group(label):
        target = _TURN_BY_LABEL[label]
        sess = target[2]
        grp = []
        for t in list(PROBE_TURNS) + list(_EXTRA_TURNS):
            if t[2] == sess:
                grp.append(t[0])
            if t[0] == label:
                break
        return grp

    default_labels = {t[0] for t in PROBE_TURNS}
    checks = {
        "EXTRA_LESIONS keys == EXTRA_PROBES faculty keys":
            set(EXTRA_LESIONS) == {row[0] for row in EXTRA_PROBES},
        "every EXTRA_LESIONS entry has the FACULTY_LESIONS shape (flag/value/kind/note)":
            all({"flag", "value", "kind", "note"} <= set(v) for v in EXTRA_LESIONS.values()),
        "every EXTRA_PROBES entry is a 4-tuple (key, turn, fields, thin)":
            all(len(row) == 4 and isinstance(row[2], list) and isinstance(row[3], bool) for row in EXTRA_PROBES),
        "d5c/slp turns exist in _TURN_BY_LABEL":
            all(l in _TURN_BY_LABEL for l in (
                "d5c_teach", "d5c_recall1", "d5c_tick", "d5c_recall2",
                "slp_teach1", "slp_teach2", "slp_teach3", "slp_tick", "slp_recall")),
        "d5c/slp turns are NOT in the default PROBE_TURNS roster (byte-identical harness)":
            not ({"d5c_teach", "d5c_recall1", "d5c_tick", "d5c_recall2",
                  "slp_teach1", "slp_teach2", "slp_teach3", "slp_tick", "slp_recall"} & default_labels),
        "d5c group is teach->recall1->tick->recall2, declaration order":
            _turn_group("d5c_recall2") == ["d5c_teach", "d5c_recall1", "d5c_tick", "d5c_recall2"],
        "slp group is 3 teaches->tick->recall, declaration order":
            _turn_group("slp_recall") == ["slp_teach1", "slp_teach2", "slp_teach3", "slp_tick", "slp_recall"],
        "d5-consolidate row compares the graded field, not the categorical completion gate":
            "episodic.graded_cue.depth_hold" in EXTRA_PROBES[0][2]
            and "episodic.in_memory" not in EXTRA_PROBES[0][2],
        "sleep-replay row compares the graded field, not the categorical completion gate":
            "episodic.graded_cue.depth_hold" in EXTRA_PROBES[1][2]
            and "episodic.in_memory" not in EXTRA_PROBES[1][2],
        "the graded field's leaf name is not in _NOISE_FIELDS (else compare() would always skip it)":
            "depth_hold" not in __import__(
                "research.runners.onebrain_regression_battery", fromlist=["_NOISE_FIELDS"])._NOISE_FIELDS,
        "both lesion values are the master-switch OFF value ('0'), matching the default-ON flags":
            EXTRA_LESIONS["d5-consolidate"]["value"] == "0" and EXTRA_LESIONS["sleep-replay"]["value"] == "0",
        # 2026-09-24 review fix: neither flag has a dedicated BRAIN_<X>_LESION knob (grepped, zero hits in
        # webapp/continuous_engine.py), so per FACULTY_LESIONS' own taxonomy both rows are the weaker
        # "whether-disable" class (a master-switch disable), NOT "neural-lesion" (reserved for a dedicated
        # BRAIN_<X>_LESION cut, value="1", the PRIMARY/gold-standard class -- every existing neural-lesion row
        # follows that shape with zero exceptions). Matches the "confidence-forthcomingness" precedent.
        "both rows are kind='whether-disable' (no dedicated _LESION knob exists yet for either flag)":
            EXTRA_LESIONS["d5-consolidate"]["kind"] == "whether-disable"
            and EXTRA_LESIONS["sleep-replay"]["kind"] == "whether-disable",
    }
    return checks


def selftest():
    checks = _selftest_checks()
    ok = all(checks.values())
    print("=== lbf_rows/learning.py SELF-TEST ===")
    for name, passed in checks.items():
        print("  [%s] %s" % ("PASS" if passed else "FAIL", name))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    raise SystemExit(0 if selftest() else 1)
