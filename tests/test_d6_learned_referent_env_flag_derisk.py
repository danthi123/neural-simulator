"""Guards for the D6 learned-referent env-flag wiring runner (language lane E), review round 1 fixes.

No corpus/organ build needed: these exercise the pure gating/scoring logic in isolation, which is exactly what
must never crash regardless of what a real population battery measures.
"""
import json
import os

import pytest

from research.runners import _d6_learned_referent_env_flag_derisk as R
from tools.lab import LeverError, lever

_SAME_INPUT = {"corpus_env_sha256": "aa", "corpus_lexicon_sha256": "bb"}


def test_r4_gate_equal_rates_records_failed_gate_not_a_crash():
    """The two realistic failing outcomes R4 exists to catch both make intact_rate == lesion_rate.

    Before the fix, `lever(..., required=True)` raised `LeverError` here, which happened BEFORE the per-seed
    JSON was ever written -- so `score()` saw an INCOMPLETE run instead of a NO-GO. The fix must (a) never raise,
    and (b) still record the gate as failed.
    """
    # (a) broken wire: flag never reaches the organ, both arms sit at 0.0.
    lever_moved, r4_pass = R._r4_gate(intact_rate=0.0, lesion_rate=0.0, hand_rate=0.0)
    assert lever_moved is False
    assert r4_pass is False

    # (b) lesion the route ignores: both arms sit at the same nonzero rate.
    lever_moved, r4_pass = R._r4_gate(intact_rate=0.75, lesion_rate=0.75, hand_rate=0.10)
    assert lever_moved is False
    assert r4_pass is False

    # sanity: a genuinely moved lever within the ceiling still passes.
    lever_moved, r4_pass = R._r4_gate(intact_rate=0.75, lesion_rate=0.10, hand_rate=0.10)
    assert lever_moved is True
    assert r4_pass is True

    # confirm what "before the fix" looked like -- required=True DOES raise on the equal-rates case, which is
    # exactly why `_r4_gate` must use required=False internally.
    with pytest.raises(LeverError):
        lever("would-crash-the-run", 0.0, 0.0, required=True)


def test_run_seed_equal_rates_case_writes_a_complete_artifact(tmp_path):
    """End-to-end: a seed whose lesion never moves the lever still produces a JSON gate=False, not a crash.

    Builds the `out` dict the same way `run_seed` does for its R3/R4 tail (using `_r4_gate` directly, since a
    full population battery needs a real corpus/organ), then writes and re-reads it exactly as `main()` does,
    to prove the artifact-write path survives a failed lever.
    """
    out = {"seed": 42, "held_word_phrase": R.HELD_WORD_PHRASE,
           "r1_pass": True, "r2_pass": True,
           "r3_recovered_both_rate": 0.60, "r3_pass": True,
           "r4_lesion_recovered_both_rate": 0.60,
           "hand_baseline_recovered_both_rate": 0.10}
    out["r4_lever_moved"], out["r4_pass"] = R._r4_gate(
        out["r3_recovered_both_rate"], out["r4_lesion_recovered_both_rate"], out["hand_baseline_recovered_both_rate"])
    out["r5_lexicon_built"] = True

    dst = tmp_path / "s42.json"
    json.dump(out, open(dst, "w"))  # must not have raised getting here

    reloaded = json.load(open(dst))
    assert reloaded["r4_lever_moved"] is False
    assert reloaded["r4_pass"] is False


def test_score_r5_is_report_only_not_gated(tmp_path):
    """R5 is pre-registered report-only: it must never sit inside the gated `passed` computation.

    A seed set where R1-R4 all pass but R5's `r5_lexicon_built` is (hypothetically) False must still verdict
    GO, because R5 cannot independently fail the claim (R1 passing already implies a built lexicon) and the
    pre-registration explicitly excludes it from the verdict.
    """
    src = tmp_path
    for seed in R.SEEDS6:
        d = {"seed": seed, "r1_pass": True, "r2_pass": True,
             "r3_recovered_both_rate": 0.80, "r4_lesion_recovered_both_rate": 0.05,
             "r4_lever_moved": True, "hand_baseline_recovered_both_rate": 0.10,
             "r5_lexicon_built": False, **_SAME_INPUT}   # deliberately "failing" if it were (wrongly) gated
        json.dump(d, open(os.path.join(src, f"s{seed}.json"), "w"))

    out = R.score(str(src))
    assert out["verdict"] == "GO"
    assert "R5_singleton_built_all_seeds" not in out["evidence"]
    assert out["report_only"]["R5_singleton_built_all_seeds"] == (0, False)


def test_score_r4_gates_on_lever_moved_too():
    """A 6-seed set with numerically-passing rates but an unmoved lever on one seed must NOT verdict GO."""
    import tempfile
    with tempfile.TemporaryDirectory() as src:
        for seed in R.SEEDS6:
            lever_moved = seed != 44   # one seed's lesion never actually moved the lever
            d = {"seed": seed, "r1_pass": True, "r2_pass": True,
                 "r3_recovered_both_rate": 0.80, "r4_lesion_recovered_both_rate": 0.05,
                 "r4_lever_moved": lever_moved, "hand_baseline_recovered_both_rate": 0.10,
                 "r5_lexicon_built": True, **_SAME_INPUT}
            json.dump(d, open(os.path.join(src, f"s{seed}.json"), "w"))
        out = R.score(src)
        assert out["verdict"] == "NO-GO"
        assert out["evidence"]["R4_lesion_recover_mean"][1] is False


def test_score_refuses_to_pool_seeds_run_on_different_inputs(tmp_path):
    """2026-09-24 review: seed 42 ran on a 7.99 MB prefix of the corpus the other seeds would read. Six passing seeds
    on two different inputs (or with an input never recorded) must verdict MIXED-INPUT, never GO."""
    for case in ("differ", "unrecorded"):
        d_ = tmp_path / case
        d_.mkdir()
        for seed in R.SEEDS6:
            inp = dict(_SAME_INPUT)
            if seed == 42:
                inp = {"corpus_env_sha256": "zz", "corpus_lexicon_sha256": "zz"} if case == "differ" else {}
            d = {"seed": seed, "r1_pass": True, "r2_pass": True,
                 "r3_recovered_both_rate": 0.80, "r4_lesion_recovered_both_rate": 0.05,
                 "r4_lever_moved": True, "hand_baseline_recovered_both_rate": 0.10, "r5_lexicon_built": True, **inp}
            json.dump(d, open(os.path.join(d_, f"s{seed}.json"), "w"))
        out = R.score(str(d_))
        assert out["verdict"] == "MIXED-INPUT" and out["one_input"] is False, (case, out["verdict"])
