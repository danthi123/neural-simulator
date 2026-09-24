"""LBF rows for the OPEN-ENDED GATED TURN (BRAIN_OPEN_ENDED_GATED, default OFF; lane A3, plan step S16).

Pre-registered in research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md (Part B). Merged into
`load_bearing_fraction.FACULTY_LESIONS` / `FACULTY_PROBES` by the AG-REG import hook; the entry shapes are exactly
theirs: EXTRA_LESIONS maps faculty_key -> dict(flag, value, kind, note); EXTRA_PROBES is a list of
(faculty_key, turn_label, decision_field_paths, thin) tuples over turn labels that already exist in
`onebrain_regression_battery._TURN_BY_LABEL`.

MEASUREMENT CONDITION. The `open_ended_gated` response key exists only when BRAIN_OPEN_ENDED_GATED=1. At production
defaults (tag b2b-rows) every row below reads NOT-EXERCISED in both arms: an opt-in faculty, never "not load-bearing".
The rows are measured in tag b2b-caps with `--extra-env BRAIN_OPEN_ENDED_GATED=1` (LLM disabled, stub renderer), so
they read the brain's DECISIONS (route, BG action, reply kind, marker register), not rendered Qwen text.

PASS-BY-CONSTRUCTION AUDIT (blocking; PREREG Part B). For each row: does the lesion write a measured field, or remove
the reply template's only input? If yes the row is an integrity smoke and is excluded.
  * open-ended-turn-faculty-drive -- the lesion sets the INPUT currents of two spiking competitions to an equal
    baseline (BG channels (0.5, 0.5); marker pools the circuit's own lesion drive). The measured fields are what those
    competitions COMMIT (bg_action, marker_level) and the host label of that commit (reply_kind). The lesion writes
    none of them; the race can still commit either action. PASS.
  * open-ended-turn-affect-drive -- BRAIN_AFFECT_LESION closes the Gate-B ladder's `affect_out` transmission gate; the
    differential then feeds the salience transduction and the BG race. bg_action / reply_kind are the race's commit.
    The organ copy `valence_sign` IS set directly by this lesion, so it is a TRACE field and NOT listed here. PASS.
  * open-ended-turn-gnw-drive -- BRAIN_GNW_2ORGAN_WS_LESION zeroes the GNW workspace self-recurrence; whether the
    covered recall IGNITES is the workspace's spiking dynamics (organ A still recalls). `route` is the host label of the
    gate's committed-or-not result; bg_action / reply_kind follow through the race. `familiarity_band` is the organ copy
    of the route and is NOT listed. PASS.
Each artifact carries `open_ended_gated.cut` (the saliences actually applied, whether they equal the baseline, the
marker lesion flag, `affect_lesioned`, the 2-organ `ws_lesion`), recorded at read time.
"""
from __future__ import annotations

EXTRA_LESIONS = {
    "open-ended-turn-faculty-drive": dict(
        flag="BRAIN_OPEN_ENDED_GATE_LESION", value="1", kind="neural-lesion",
        note="OPT-IN row (needs BRAIN_OPEN_ENDED_GATED=1; not-exercised at production defaults). Cuts the afferents "
             "from the brain reads (GNW route, Gate-B affect) into the gated turn's two spiking competitions: both BG "
             "speak/abstain channels get the same baseline salience (0.5, 0.5) and the SETTLE-window marker WTA runs "
             "with its own lesion drive. Organs stay installed. Measured on the off-KB 'unknown' turn, where the intact "
             "saliences are (0, 1)."),
    "open-ended-turn-affect-drive": dict(
        flag="BRAIN_AFFECT_LESION", value="1", kind="neural-lesion",
        note="OPT-IN row (needs BRAIN_OPEN_ENDED_GATED=1). The Gate-B affect_out gate closed -> the differential that "
             "sets the gated turn's engagement salience collapses. Measured on the strongly affective off-KB 'emo' "
             "turn; decision fields are the BG commit and the reply kind (valence_sign is a trace copy, excluded)."),
    "open-ended-turn-gnw-drive": dict(
        flag="BRAIN_GNW_2ORGAN_WS_LESION", value="1", kind="neural-lesion",
        note="OPT-IN row (needs BRAIN_OPEN_ENDED_GATED=1). GNW workspace self-recurrence zeroed -> a covered recall "
             "cannot ignite. Measured on the KB-hit 'sw_open' turn ('what does the dog chase', a boot fact): intact "
             "route grounded, lesion route withheld is the expected direction; the measurement decides."),
}

EXTRA_PROBES = [
    ("open-ended-turn-faculty-drive", "unknown",
     ["open_ended_gated.bg_action", "open_ended_gated.reply_kind", "open_ended_gated.marker_level"], False),
    ("open-ended-turn-affect-drive", "emo",
     ["open_ended_gated.bg_action", "open_ended_gated.reply_kind"], False),
    ("open-ended-turn-gnw-drive", "sw_open",
     ["open_ended_gated.route", "open_ended_gated.bg_action", "open_ended_gated.reply_kind"], False),
]

# the b2b-caps condition every row above needs (the orchestrator adds it as --extra-env; never a production default)
REQUIRED_ENV = {"BRAIN_OPEN_ENDED_GATED": "1"}


def score_row(key, intact_a, intact_b, lesion):
    """Score one row the LBF way on three arm response dicts (label -> response): TREATMENT = decision-field diffs
    intact_a vs lesion, CONTROL (null) = intact_a vs intact_b (an independent rebuild). Asks the attribution question
    out loud (tools.lab.attributable_to) and returns the verdict dict the seed-7 smoke records. load_bearing requires
    a treatment change AND a clean null; not-exercised when the fields are absent in every arm (flag off)."""
    from research.runners.onebrain_regression_battery import compare
    from tools.lab import attributable_to
    row = next(r for r in EXTRA_PROBES if r[0] == key)
    treat = compare(intact_a, lesion, faculties=[row])["per_faculty"][0]
    null = compare(intact_a, intact_b, faculties=[row])["per_faculty"][0]
    n_t, n_c = len(treat["diffs"]), len(null["diffs"])
    frac = attributable_to("%s: decision-field diffs intact-vs-lesion vs intact-vs-rebuild" % key, n_t, n_c)
    exercised = treat["verdict"] != "not-exercised" or null["verdict"] != "not-exercised"
    return {"faculty": key, "turn": row[1], "fields": row[2], "treatment_diffs": treat["diffs"],
            "control_diffs": null["diffs"], "null_clean": n_c == 0, "exercised": bool(exercised),
            "attributable_fraction": frac, "load_bearing": bool(exercised and n_t > 0 and n_c == 0)}
