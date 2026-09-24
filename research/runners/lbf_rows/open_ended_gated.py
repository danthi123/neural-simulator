"""LBF rows for the OPEN-ENDED GATED TURN (BRAIN_OPEN_ENDED_GATED, default OFF; lane A3, plan step S16).

Pre-registered in research/findings/2026-09-24-open-ended-gated-turn-PREREGISTRATION.md (Part B; Amendments 2 and 3).
Merged into `load_bearing_fraction.FACULTY_LESIONS` / `FACULTY_PROBES` by the AG-REG import hook; the entry shapes are
exactly theirs: EXTRA_LESIONS maps faculty_key -> dict(flag, value, kind, note); EXTRA_PROBES is a list of
(faculty_key, turn_label, decision_field_paths, thin) tuples over turn labels that already exist in
`onebrain_regression_battery._TURN_BY_LABEL`.

MEASUREMENT CONDITION. The `open_ended_gated` response key exists only when BRAIN_OPEN_ENDED_GATED=1. At production
defaults (tag b2b-rows) every row below reads NOT-EXERCISED in both arms: an opt-in faculty, never "not load-bearing".
The rows are measured in tag b2b-caps with `--extra-env BRAIN_OPEN_ENDED_GATED=1` (LLM disabled, stub renderer), in a
DEDICATED `tools/lb_shard.py` invocation whose `--faculties` lists exactly these three keys (Amendment 3: extra-env applies
to every job of an invocation, so any other row generated with them would be measured with the gated turn ON).

PASS-BY-CONSTRUCTION AUDIT (blocking; PREREG Part B, corrected by Amendment 3 after the 2026-09-24 review). For each
row: does the lesion write a measured field, or remove the reply template's only input? If yes the field is an
integrity check, not a decision field.
  * open-ended-turn-faculty-drive -- the lesion sets the INPUT saliences of the BG speak/abstain race to an equal
    baseline (0.5, 0.5). The measured fields are what the race COMMITS (bg_action) and the host label of that commit
    (reply_kind). The lesion writes neither; the race can still commit either action. PASS for bg_action/reply_kind.
    `marker_level` is NOT a decision field (Amendment 3): the same lesion runs AffectMarkerWTA with lesion=True, whose
    own docstring says the dead-margin check then fails on (almost) every trial -> None, so on an affective turn the
    field changes BY CONSTRUCTION, and on the registered off-KB 'unknown' turn it is structurally 0 in every arm (a
    neutral mood is never sent to the circuit). It stays in the trace (`open_ended_gated.marker`).
    NON-DISCRIMINATING AS A SINGLE-SHOT ROW (Amendment 3, DESCRIPTIVE_ONLY below): on 'unknown' the intact race sits at
    (0, 1) (STAY_SILENT on 23/24 dev races) and the cut race at (0.5, 0.5) (SPEAK on 12/24): the cut ADDS speak drive,
    and one LBF build per arm is one race, so the row's verdict is a per-seed coin flip whatever the coupling. Its LBF
    verdict is reported, never counted toward a load-bearing claim; the distributional read beside it is --bg-curve.
  * open-ended-turn-affect-drive -- BRAIN_AFFECT_LESION closes the Gate-B ladder's `affect_out` transmission gate; the
    tone level then feeds the salience transduction and the BG race. bg_action / reply_kind are the race's commit.
    The organ copy `valence_sign` IS set directly by this lesion, so it is a TRACE field and NOT listed here. PASS.
    SHARED LESION (Amendment 3): the same flag drives the base registry's 'affect-coloring' row; any fraction that
    includes both counts ONE organ lesion, not two.
  * open-ended-turn-gnw-drive (probe turn 'chase', Amendment 2) -- BRAIN_GNW_2ORGAN_WS_LESION zeroes the GNW workspace
    self-recurrence; whether the covered recall IGNITES is the workspace's spiking dynamics (organ A still recalls).
    Decision field (Amendment 3): `route` ONLY -- the gated turn's host label of the gate's committed-or-not result.
    `bg_action` changes only through the host FAM table (withheld -> fam 0) and decide() ignores it on a withheld route;
    `reply_kind` 'withheld_abstain' is a direct map of route. Counting all three counted one cause three times. The
    REPLY change on this turn ('the dog chases the cat' -> "I don't know about that.") is the PIPELINE's own gate abstain
    and happens with the flag OFF too; it is not credited to the gated turn. PASS for `route`.
Each artifact carries `open_ended_gated.cut` (the saliences actually applied, whether they equal the baseline, the
marker lesion flag, `affect_lesioned`, the 2-organ `ws_lesion`), recorded at read time.
"""
from __future__ import annotations

EXTRA_LESIONS = {
    "open-ended-turn-faculty-drive": dict(
        flag="BRAIN_OPEN_ENDED_GATE_LESION", value="1", kind="neural-lesion",
        note="OPT-IN row (needs BRAIN_OPEN_ENDED_GATED=1; not-exercised at production defaults). Cuts the afferents "
             "from the brain reads (GNW route, Gate-B affect) into the gated turn's BG speak/abstain race: both "
             "channels get the same baseline salience (0.5, 0.5); organs stay installed. Measured on the off-KB "
             "'unknown' turn, where the intact saliences are (0, 1). DESCRIPTIVE ONLY (PREREG Amendment 3): a "
             "single-shot read of a race the cut moves to P(SPEAK)~1/2 is a per-seed coin flip; never counted toward "
             "a load-bearing claim. Decision fields bg_action / reply_kind (marker_level is trace: the marker WTA's "
             "own lesion mode makes it None by construction)."),
    "open-ended-turn-affect-drive": dict(
        flag="BRAIN_AFFECT_LESION", value="1", kind="neural-lesion",
        note="OPT-IN row (needs BRAIN_OPEN_ENDED_GATED=1). The Gate-B affect_out gate closed -> the tone level that "
             "sets the gated turn's engagement salience collapses. Measured on the strongly affective off-KB 'emo' "
             "turn; decision fields are the BG commit and the reply kind (valence_sign is a trace copy, excluded). "
             "Shares its lesion flag with the base 'affect-coloring' row: one organ lesion, never counted twice."),
    "open-ended-turn-gnw-drive": dict(
        flag="BRAIN_GNW_2ORGAN_WS_LESION", value="1", kind="neural-lesion",
        note="OPT-IN row (needs BRAIN_OPEN_ENDED_GATED=1). GNW workspace self-recurrence zeroed -> a covered recall "
             "cannot ignite. Measured on the default-roster KB-hit 'chase' turn ('what does the dog chase all the "
             "way', a boot fact; PREREG Amendment 2). Decision field: the gated turn's route ONLY (Amendment 3); "
             "bg_action and reply_kind follow from the route by host tables. The reply change on this turn is the "
             "pipeline's own gate abstain, present with the flag off, and is not credited to the gated turn."),
}

EXTRA_PROBES = [
    ("open-ended-turn-faculty-drive", "unknown",
     ["open_ended_gated.bg_action", "open_ended_gated.reply_kind"], False),
    ("open-ended-turn-affect-drive", "emo",
     ["open_ended_gated.bg_action", "open_ended_gated.reply_kind"], False),
    ("open-ended-turn-gnw-drive", "chase",
     ["open_ended_gated.route"], False),
]

# the b2b-caps condition every row above needs (the orchestrator adds it as --extra-env; never a production default)
REQUIRED_ENV = {"BRAIN_OPEN_ENDED_GATED": "1"}

# PREREG Amendment 3: rows whose registered single-shot verdict cannot discriminate (reported, never counted toward a
# load-bearing claim), with the reason. The LBF runner itself does not read this; the lane's scorer and finding do.
DESCRIPTIVE_ONLY = {
    "open-ended-turn-faculty-drive": "single-shot race: intact P(SPEAK) 1/24 at (0, 1) vs cut 12/24 at (0.5, 0.5) on "
                                     "the dev curve -> the verdict is a per-seed coin flip (Amendment 1 disclosed it)",
}

# PREREG Amendment 3: rows that share ONE lesion flag with another registered row (one organ lesion, counted once)
SHARED_LESION_WITH = {"open-ended-turn-affect-drive": "affect-coloring"}

# the three keys the dedicated b2b-caps invocation must be limited to (tools/lb_shard.py --faculties)
FACULTIES = tuple(r[0] for r in EXTRA_PROBES)


def score_row(key, intact_a, intact_b, lesion):
    """Score one row the LBF way on three arm response dicts (label -> response): TREATMENT = decision-field diffs
    intact_a vs lesion, CONTROL (null) = intact_a vs intact_b (an independent rebuild). Asks the attribution question
    out loud (tools.lab.attributable_to) and returns the verdict dict the seed-7 smoke records. load_bearing requires
    a treatment change AND a clean null; not-exercised when the fields are absent in every arm (flag off).
    `counts_toward_claim` is False for a DESCRIPTIVE_ONLY row whatever it reads."""
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
            "attributable_fraction": frac, "load_bearing": bool(exercised and n_t > 0 and n_c == 0),
            "counts_toward_claim": key not in DESCRIPTIVE_ONLY,
            "descriptive_only_reason": DESCRIPTIVE_ONLY.get(key),
            "shared_lesion_with": SHARED_LESION_WITH.get(key)}
