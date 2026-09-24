"""Regression pin for the 2026-09-24 record-correctness defect (review v2:2a37f2493 of commit 5d3810f2d):
`aggregate()` in research/runners/_da_tag_capture_chat_probe.py read the STORED `r["gates"]["seed_verdict"]` from
each seed*.json on disk instead of re-grading it with the current `grade_seed`. A seed*.json written before a
grading-logic fix lands can carry a stale, more-favorable verdict forever: the real confounded
`research/findings/raw/_da_tag_capture_chat/seed42.json` (written at c4c62d066, before the D1-reader-isolation fix
daa4b382d) stores `seed_verdict: "GO"`, but re-grading its own `arms` data under the new `G_isolation_gamma_consistent`
gate reads `UNDEFINED` (the lesion arm's `gamma`/`d1_a_go` do not match the intact arms', the exact confound the fix
closes). `aggregate()` must never let a stale on-disk verdict outlive the grading logic that produced it -- it must
call `grade_seed(row)` on every row before counting it toward `n_go` / the 6-seed verdict.

No brain, no SIM_BACKEND needed: `aggregate()` only reads JSON off disk and calls the pure `grade_seed`."""
import json
import os

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import _da_tag_capture_chat_probe as P  # noqa: E402

_BASE = {"recalled_svo": P.FACT, "abstained": False, "tag_capture_at_recall": {"p_max": 0.05},
         "fact_block_at_recall": None, "errors": []}


def _designed_go_arms(consistent_isolation):
    """The designed-GO arm pattern (mirrors the runner's own --selftest), with every companion-ON arm's
    gamma/d1_a_go either all equal (consistent_isolation=True) or with the lesion arm reading a different
    calibration (consistent_isolation=False, the confound the isolation fix closes)."""
    arms = {n: dict(_BASE, recall_outcome="correct") for n, _l, _e in P.ARMS}
    arms["neu_night_intact"] = dict(_BASE, recall_outcome="abstain", recalled_svo=None, abstained=True)
    arms["sal_night_lesion"] = dict(_BASE, recall_outcome="abstain", recalled_svo=None, abstained=True,
                                    tag_capture_at_recall={"p_max": 0.001})
    for name, _label, env in P.ARMS:
        rec = dict(arms[name])
        rec["env"] = dict(env)
        if env.get("BRAIN_DA_TAG_CAPTURE") == "1":
            gamma = 40.0 if (consistent_isolation or name != "sal_night_lesion") else 32.8
            d1 = 0.15 if (consistent_isolation or name != "sal_night_lesion") else 0.187
            rec["tag_capture_at_recall"] = dict(rec["tag_capture_at_recall"], gamma=gamma, d1_a_go=d1)
        arms[name] = rec
    return arms


def test_grade_seed_isolation_mismatch_is_undefined_even_though_g1_to_g6_read_designed_go():
    """Sanity on the fixture itself: the isolation-mismatch pattern reads UNDEFINED from grade_seed directly
    (the pre-existing gate this test's aggregate() check depends on)."""
    consistent = P.grade_seed({"arms": _designed_go_arms(True)})
    mismatched = P.grade_seed({"arms": _designed_go_arms(False)})
    assert consistent["seed_verdict"] == "GO"
    assert mismatched["G_isolation_gamma_consistent"] is False
    assert mismatched["seed_verdict"] == "UNDEFINED"


def test_aggregate_regrades_a_stale_stored_go_row_as_undefined(tmp_path):
    """THE FIX: a seed*.json on disk whose STORED gates say GO (as an artifact written by pre-fix code would)
    but whose arms data is the isolation-mismatch confound must be reported by aggregate() under the CURRENT
    grade_seed, i.e. UNDEFINED -- never counted as one of the n_go GO seeds on its stale grade."""
    arms = _designed_go_arms(False)
    stale_record = {"seed": 900, "arms": arms,
                    "gates": {"seed_verdict": "GO", "outcomes": {k: v["recall_outcome"] for k, v in arms.items()}}}
    (tmp_path / "seed900.json").write_text(json.dumps(stale_record))
    out = P.aggregate(str(tmp_path))
    assert out["seed_verdicts"][900] == "UNDEFINED"
    assert out["seed_verdicts"][900] != stale_record["gates"]["seed_verdict"]
    # the aggregate.json artifact itself must carry the re-graded verdict, not the stale one
    written = json.loads((tmp_path / "aggregate.json").read_text())
    assert written["seed_verdicts"]["900"] == "UNDEFINED"


def test_aggregate_regrading_is_idempotent_for_an_already_current_row(tmp_path):
    """A row already graded under the current code (stored verdict == what grade_seed(row) recomputes) must be
    unaffected by the re-grade -- this is a correction for STALE rows, not a behavior change for current ones."""
    arms = _designed_go_arms(True)
    record = {"seed": 1, "arms": arms, "gates": P.grade_seed({"arms": arms})}
    assert record["gates"]["seed_verdict"] == "GO"
    (tmp_path / "seed1.json").write_text(json.dumps(record))
    out = P.aggregate(str(tmp_path))
    assert out["seed_verdicts"][1] == "GO"
